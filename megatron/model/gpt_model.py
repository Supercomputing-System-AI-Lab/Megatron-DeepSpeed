# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""GPT-2 model."""
import time 
import torch

from megatron import get_args
from megatron.core import mpu, tensor_parallel, sequence_parallel
from .module import MegatronModule, fp32_to_float16, float16_to_fp32

from .enums import AttnMaskType
from .language_model import parallel_lm_logits
from .language_model import get_language_model
from .utils import init_method_normal
from .utils import scaled_init_method_normal

from megatron.model import LayerNorm
from .language_model import EmbeddingPipe
from .transformer import ParallelTransformerLayerPipe, LMHeadPipe
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec

try:
    from apex.normalization import MixedFusedRMSNorm
except ImportError:
    MixedFusedRMSNorm = None

try:         
    from deepspeed.checkpoint import (
        VOCABULARY_PARAMETER_PATTERNS,
        PIPELINE_REPLICATED_PARAMETER_PATTERNS,
        TP_REPLICATED_PARAMETER_PATTERNS,
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS,
    )
    DS_UNIVERSAL_CHECKPOINT_INFO = True 
except ImportError:
    DS_UNIVERSAL_CHECKPOINT_INFO = False  

def _seq_chunked_cross_entropy(lm_output, labels_sb, logit_weights,
                                 parallel_output, chunk):
    """Per-token CE computed in sequence chunks so the full [s, b, vocab]
    logits (bf16 + fp32 copies) are never resident at once. CE is per-token
    independent, so chunking the sequence axis is loss-preserving; each
    chunk's logits are recomputed in backward via torch checkpointing, so
    peak memory is one chunk's forward+backward instead of the whole blob.
    TP=1-safe by construction (vocab_parallel_cross_entropy reduces over the
    TP group identically per chunk)."""
    import torch.utils.checkpoint as _torch_ckpt

    def _one_chunk(chunk_out, chunk_labels):
        logits = parallel_lm_logits(chunk_out, logit_weights, parallel_output)
        return tensor_parallel.vocab_parallel_cross_entropy(
            logits.contiguous().float(), chunk_labels)

    # Under Megatron's TP-companion --sequence-parallel, lm_output is this
    # rank's s/TP sequence shard while labels_sb stays full-length, and
    # parallel_lm_logits all-gathers EACH CHUNK over the TP group inside the
    # matmul (core/tensor_parallel/layers.py), so a chunk's logits cover the
    # non-contiguous global positions {r*s/TP + [s0,s1) : r in 0..TP-1} in
    # rank order. Labels must be taken from those positions and the chunk
    # losses placed back there; slicing labels in lock-step raises an
    # IndexError in vocab CE at TP>1 and silently permutes the loss under
    # any naive length-matching (SP_COMPANION_AUDIT.md S2.G4 + S8 cell D).
    args = get_args()
    sp_world = (mpu.get_tensor_model_parallel_world_size()
                if getattr(args, 'sequence_parallel', False) else 1)
    losses = []
    order = []
    s = lm_output.size(0)                # local shard length under SP
    if sp_world > 1:
        assert labels_sb.size(0) == s * sp_world, \
            'companion SP expects full-length labels: ' \
            f'{labels_sb.size(0)} vs {s} * {sp_world}'
    for s0 in range(0, s, chunk):
        s1 = min(s0 + chunk, s)
        if sp_world > 1:
            chunk_labels = torch.cat(
                [labels_sb[r * s + s0: r * s + s1]
                 for r in range(sp_world)], dim=0)
            order.extend(r * s + i
                         for r in range(sp_world) for i in range(s0, s1))
        else:
            chunk_labels = labels_sb[s0:s1]
        losses.append(_torch_ckpt.checkpoint(
            _one_chunk, lm_output[s0:s1], chunk_labels,
            use_reentrant=False))
    loss = torch.cat(losses, dim=0)
    if sp_world > 1:
        # Scatter to global order in one gather: concatenated row j carries
        # global position order[j], so loss_full = loss[inv] with
        # inv[order[j]] = j. The result is full-length and identical on
        # every TP rank (the in-linear gather is rank-symmetric), matching
        # the full-length loss_mask layout of companion SP.
        inv = torch.empty(len(order), dtype=torch.long, device=loss.device)
        inv[torch.as_tensor(order, device=loss.device)] = torch.arange(
            len(order), device=loss.device)
        loss = loss[inv]
    return loss


def post_language_model_processing(lm_output, labels, logit_weights,
                                   parallel_output,
                                   fp16_lm_cross_entropy):

    import os 
    rank = os.getenv ('RANK')
    # print (f'[gpt_model.py] {rank=} post_language_model_processing {output.shape=}, {labels=}, {output=}')

    if labels is None:
        # Output. Format [s b h]
        output = parallel_lm_logits(
            lm_output,
            logit_weights,
            parallel_output)
        # print (f'[gpt_model.py] {rank=} post_language_model_processing returning output.transpose(0,1).contiguous() {type(output.transpose(0,1).contiguous())=}')
        # [s b h] => [b s h]
        return output.transpose(0,1).contiguous()
    else:
        # [b s] => [s b]
        labels = labels.transpose(0,1).contiguous()
        ce_chunk = getattr(get_args(), 'seq_chunked_ce', 0) or 0
        if (ce_chunk > 0 and lm_output.size(0) > ce_chunk
                and not fp16_lm_cross_entropy
                and mpu.get_sequence_parallel_world_size() == 1):
            # Dead-store fix (128k S3.2): the full [s, b, vocab] logits used
            # to be materialised unconditionally ABOVE this branch and were
            # never read on it -- 2*vocab B/token = 24.44 GiB at s=131,072
            # (lm_head_calibration.json shipped_chunked vs fixed_chunked;
            # the measured OOM of jobs 5356145/5356146). They are now built
            # lazily on the branches that consume them; this branch's peak
            # is one chunk's logits. The DEBUG_VOCAB_PAD report needs full
            # logits and therefore does not fire on this branch (computed
            # lazily, not deleted -- see probe_lm_head.py `fixed_chunked`).
            loss = _seq_chunked_cross_entropy(lm_output, labels, logit_weights,
                                              parallel_output, ce_chunk)
            return loss.transpose(0, 1).contiguous()
        # Output. Format [s b h]
        output = parallel_lm_logits(
            lm_output,
            logit_weights,
            parallel_output)
        cross_entropy = sequence_parallel.vocab_sequence_parallel_cross_entropy if mpu.get_sequence_parallel_world_size() > 1 \
            else tensor_parallel.vocab_parallel_cross_entropy
        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = cross_entropy(output, labels)
        else:
            loss = cross_entropy(output.float(), labels)

        # [s b] => [b, s]
        loss = loss.transpose(0,1).contiguous()
        # print (f'[gpt_model.py] {rank=} post_language_model_processing returning loss')
        return loss


class GPTModel(MegatronModule):
    """GPT-2 Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 return_moe_loss=True):
        args = get_args()
        super().__init__(config=config, share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights)

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.return_moe_loss = return_moe_loss
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights
        
        print (f'[gpt_model.py] {args.num_experts=}')

        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process,
            num_experts=args.num_experts)

        if not args.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask,
                retriever_input_ids=None,
                retriever_position_ids=None,
                retriever_attn_mask=None,
                labels=None, tokentype_ids=None, inference_params=None,
                curriculum_seqlen=None):
        
        # torch.cuda.synchronize ()
        # s = time.time() 
        import os 
        rank = os.getenv ('RANK')
        # if (rank == '1'): 
        #     # assert False 
        #     print (f'{self=}')
        #     # raise (RuntimeError, 'hi')
        # print (f'[gpt_model.py] Entering {rank=}, {self.return_moe_loss=},{input_ids.requires_grad=}, {input_ids=}, {input_ids.shape=}, {position_ids.shape=}, {position_ids=}')
        
        args = get_args()
        if curriculum_seqlen is not None:
            args.curriculum_seqlen = curriculum_seqlen
            if curriculum_seqlen < input_ids.size()[1]:
                # seqlen-based curriculum learning
                # input_ids, position_ids, labels have size [batch size, seqlen]
                input_ids = input_ids[:, :curriculum_seqlen].contiguous()
                position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                if labels is not None:
                    labels = labels[:, :curriculum_seqlen].contiguous()

                # attention_mask has size [1, 1, seqlen, seqlen]
                attention_mask = attention_mask[:, :, :curriculum_seqlen, :curriculum_seqlen].contiguous()
        else:
            if args.curriculum_learning_legacy:
                # If got a None input, need to reset curriculum_seqlen on user side
                args.curriculum_seqlen = args.seq_length

        lm_output, moe_losses = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            retriever_input_ids=retriever_input_ids,
            retriever_position_ids=retriever_position_ids,
            retriever_attn_mask=retriever_attn_mask,
            inference_params=inference_params)

        if self.post_process:
            # print (f'[gpt_model.py] {rank=} if self.post_process. {self.untie_embeddings_and_output_weights=}')
            lm_output = post_language_model_processing(
                lm_output, labels,
                self.language_model.output_layer.weight if self.untie_embeddings_and_output_weights else self.shared_embedding_or_output_weight(),
                self.parallel_output,
                self.fp16_lm_cross_entropy)
        
        
        # print (f'[gpt_model.py] Returning {rank=}, {self.return_moe_loss=}, {lm_output.requires_grad=}, {lm_output.shape=}, {lm_output=}')
        
        # ****************************************************************************************************
        # Zixian: 09/17/2025: 
        # It NEEDS TO BE an explicit if-else statement. 
        # This guarantees consistent return type behavior. 
        # Otherwise, when torch.distributed.pipelining sending dummy example with device="meta"
        # to capture input/output shape for buffers, it will DEFAULT return lm_output.clone(), moe_losses
        # when the if-statement is a 1-line if.
        # ****************************************************************************************************
        if self.return_moe_loss: 
            # print (f'[gpt_model.py] return lm_output.clone(), moe_losses')
            return lm_output.clone(), moe_losses
        else: 
            # print (f'[gpt_model.py] return lm_output.clone() {lm_output.clone().shape=} {len (lm_output.clone())=}')
            return lm_output.clone() 
        # return lm_output.clone(), moe_losses if self.return_moe_loss else lm_output.clone() 

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):

        state_dict_ = {}
        language_model_state_dict = self.language_model.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars)
        # MoE states need to be handled separately by DeepSpeed engine, thus
        # moving them to the top level dictionary
        if "moe_state_dict" in language_model_state_dict:
            for key in list(language_model_state_dict["moe_state_dict"].keys()):
                state_dict_[key] = language_model_state_dict["moe_state_dict"].pop(key)
            del language_model_state_dict["moe_state_dict"]
        state_dict_[self._language_model_key] = language_model_state_dict
        # Save word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        # Gather MoE states and move under language model
        moe_state_dict = {}
        # ===== IMPLEMENTING SHARED EXPERT =====
        #   Changed: the load-side expert-vs-non-expert classifier.
        #   BEFORE:  if 'expert' in key and 'moe.gate.wg.weight' not in key:
        #   WHY:     the loose substring 'expert' ALSO matches 'shared_experts', so the
        #            shared expert was moved into moe_state_dict and then routed through the
        #            'encoder'-stripping loop in language_model.load_state_dict, which pops
        #            key parts until it finds 'encoder' -> for the (mangled) shared key it
        #            never matches and raises IndexError. Shared experts are replicated
        #            (non-expert) and must load like a normal encoder param, NOT via
        #            moe_state_dict. Match ONLY the precise routed marker.
        # ===== END SHARED EXPERT =====
        for key in list(state_dict.keys()):
            if 'deepspeed_moe.experts.deepspeed_experts.' in key:
                moe_state_dict[key] = state_dict.pop(key)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        if len(moe_state_dict) > 0:
            state_dict["moe_state_dict"] = moe_state_dict
        self.language_model.load_state_dict(state_dict, strict=strict)

    def universal_checkpoint_info(self):
        info = dict()
        if DS_UNIVERSAL_CHECKPOINT_INFO:
            # Vocabulary parameters (embeddings) that require special handling due to padding.
            info[VOCABULARY_PARAMETER_PATTERNS] = [
                r"tied_modules.embed.word_embeddings.weight"
            ]

            # Parameter slices that should be averaged not concatenated.
            info[TP_REPLICATED_PARAMETER_PATTERNS] = [
                r"tied_modules.embed.position_embeddings.weight",
                r"\d+.input_layernorm.weight",
                r"\d+.input_layernorm.bias",
                r"\d+.post_attention_layernorm.weight",
                r"\d+.post_attention_layernorm.bias",
                r"\d+.self_attention.dense.bias",
                r"\d+.mlp.dense_4h_to_h.bias",
                r"\d+.weight",
                r"\d+.bias",
            ]

            # Parameter that are sliced on the row dimension
            info[PARAMETER_WITH_ROW_PARALLELISM_PATTERNS] = [
                r"\d+.mlp.dense_4h_to_h.weight",
                r"\d+.self_attention.dense.weight",
            ]

        return info


# --pipe-moe-aux-loss / --intra-document-attention pipe support: the tail
# specs (final norm, LM head) are single-tensor modules, so a function spec
# strips the extras first. The rider is stashed stage-locally; the DeepSpeed
# pipe engine calls loss_fn(outputs, labels) inside the SAME _exec_forward_pass
# as the forward, so the stash cannot interleave across microbatches. The
# stashed tensor keeps its autograd graph, so the auxiliary loss backpropagates
# through every stage.
_PIPE_EXTRAS_STASH = []

_PIPE_DBG = {'strip': 0, 'loss': 0, 'rider': 0}

def _pipe_dbg(site, obj):
    if _PIPE_DBG[site] >= 3:
        return
    _PIPE_DBG[site] += 1
    def desc(o):
        if torch.is_tensor(o):
            return f'Tensor{tuple(o.shape)}:{o.dtype}'
        if isinstance(o, (tuple, list)):
            return type(o).__name__ + '(' + ', '.join(desc(e) for e in o) + ')'
        return type(o).__name__
    import torch.distributed as dist
    rk = dist.get_rank() if dist.is_initialized() else -1
    print(f'[pipe-dbg rank{rk}] {site} <- {desc(obj)}', flush=True)


def _strip_pipe_extras(inputs):
    _pipe_dbg('strip', inputs)
    if torch.is_tensor(inputs):
        return inputs
    if isinstance(inputs, tuple):
        if len(inputs) == 3:          # (hidden, cu_seqlens, rider)
            _PIPE_EXTRAS_STASH.append(inputs[2])
            return inputs[0]
        if len(inputs) == 2:          # (hidden, cu_seqlens)
            return inputs[0]
    return inputs


# --seq-chunked-ce pipe tail (128k S3.2): LMHeadChunkedPipe.forward publishes
# the LM-head weight here for the chunked loss functions below. Unlike
# _PIPE_EXTRAS_STASH this is NOT per-microbatch state: the weight is a
# persistent parameter, so a one-slot overwrite is idempotent across
# microbatches and immune to forward-only calls that never reach loss_fn.
_PIPE_LM_HEAD_WEIGHT = []


def _publish_lm_head_weight(weight):
    if _PIPE_LM_HEAD_WEIGHT:
        _PIPE_LM_HEAD_WEIGHT[0] = weight
    else:
        _PIPE_LM_HEAD_WEIGHT.append(weight)


def use_chunked_pipe_tail(args):
    """Whether GPTModelPipe builds the chunked cross-entropy tail (128k S3.2).

    Mirrors the non-pipe gate in post_language_model_processing: a positive
    --seq-chunked-ce and fp32 loss math (fp16_lm_cross_entropy excluded).
    Pipe-specific extra condition: untied embeddings -- the tied tail is a
    TiedLayerSpec whose forward_fn builds the logits itself and shares
    storage with the embedding, so only the untied LMHeadPipe is replaced
    (true for DeepSeek_16B: --untie-embeddings-and-output-weights). At
    seq_chunked_ce == 0 this is False and the tail, loss_fn, and spec list
    are exactly what they were before this change.
    """
    return ((getattr(args, 'seq_chunked_ce', 0) or 0) > 0
            and getattr(args, 'untie_embeddings_and_output_weights', False)
            and not getattr(args, 'fp16_lm_cross_entropy', False))


def _fold_pipe_moe_rider(loss):
    """Add the stashed --pipe-moe-aux-loss rider onto a computed LM loss.
    Shared by CrossEntropyWithMoE and ChunkedCrossEntropyWithMoE so the
    aux-loss rider composes identically with both tails."""
    if _PIPE_EXTRAS_STASH:
        rider = _PIPE_EXTRAS_STASH.pop()
        args = get_args()
        coeff = getattr(args, 'moe_loss_coeff', 0.0) or 0.0
        # Hot-path print gate (128k S3.5 hygiene): the f-string calls
        # .item() twice -- a device sync per micro-batch. First 3
        # occurrences only, unless XMOE_PIPE_MOE_VERBOSE=1.
        import os
        if _PIPE_DBG['rider'] < 3 or os.environ.get('XMOE_PIPE_MOE_VERBOSE', '0') == '1':
            _PIPE_DBG['rider'] += 1
            print(f'[pipe-moe] rider={rider.sum().item():.6e} coeff={coeff} '
                  f'lm_loss={loss.item():.6e}', flush=True)
        loss = loss + coeff * rider.sum()
    return loss


def CrossEntropyWithMoE(output, labels):
    _pipe_dbg('loss', output)
    if isinstance(output, tuple) and len(output) >= 1 and torch.is_tensor(output[0]):
        # Defensive unwrap; _pipe_dbg above records what actually arrived.
        output = output[0]
    return _fold_pipe_moe_rider(CrossEntropy(output, labels))


def CrossEntropy(output, labels):
    labels, loss_mask = labels[0], labels[1]

    args = get_args()

    # [b s] => [s b]
    labels = labels.transpose(0, 1).contiguous()
    losses = tensor_parallel.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
    # [s b] => [b, s]
    losses = losses.transpose(0, 1).contiguous()
    loss_mask = loss_mask.view(-1)
    # clamp: under --answer-loss-only a microbatch can contain zero supervised
    # tokens; identity in pretraining, where the mask is all ones.
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum().clamp(min=1)
    return loss


def ChunkedCrossEntropy(output, labels):
    """Pipe-tail loss for use_chunked_pipe_tail (128k S3.2). `output` is the
    [s, b, h] hidden states from LMHeadChunkedPipe -- NOT logits. The logits
    are built one sequence chunk at a time inside _seq_chunked_cross_entropy
    (per-chunk torch.utils.checkpoint recompute), so the full [s, b, vocab]
    tensor never exists: 1,001,001 B/token measured on the unchunked pipe
    tail (lm_head_calibration.json "pipe", ~122 GiB at s=131,072 at TP1) vs
    ~4,100 B/token chunked. parallel_output=True matches LMHeadPipe's
    ColumnParallelLinear(gather_output=False) and vocab_parallel_cross_entropy
    is TP-correct per chunk; each chunk's logits are upcast chunk-locally, so
    the numerics match CrossEntropy's output.float() bit-for-bit (probe
    "equivalence": bit_identical). The mask reduction below is byte-for-byte
    the one in CrossEntropy."""
    labels, loss_mask = labels[0], labels[1]

    args = get_args()

    if isinstance(output, tuple) and len(output) >= 1 and torch.is_tensor(output[0]):
        output = output[0]
    assert _PIPE_LM_HEAD_WEIGHT, \
        'ChunkedCrossEntropy ran before LMHeadChunkedPipe.forward published the weight'
    weight = _PIPE_LM_HEAD_WEIGHT[0]

    # [b s] => [s b]
    labels = labels.transpose(0, 1).contiguous()
    losses = _seq_chunked_cross_entropy(output, labels, weight, True,
                                        args.seq_chunked_ce)
    # [s b] => [b, s]
    losses = losses.transpose(0, 1).contiguous()
    loss_mask = loss_mask.view(-1)
    # clamp: under --answer-loss-only a microbatch can contain zero supervised
    # tokens; identity in pretraining, where the mask is all ones.
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum().clamp(min=1)
    return loss


def ChunkedCrossEntropyWithMoE(output, labels):
    _pipe_dbg('loss', output)
    if isinstance(output, tuple) and len(output) >= 1 and torch.is_tensor(output[0]):
        # Defensive unwrap; _pipe_dbg above records what actually arrived.
        output = output[0]
    return _fold_pipe_moe_rider(ChunkedCrossEntropy(output, labels))


class LMHeadChunkedPipe(LMHeadPipe):
    """use_chunked_pipe_tail head (128k S3.2): holds the SAME parameter as
    LMHeadPipe (`lm_head.weight`, at the same LayerSpec index, so checkpoint
    atoms and universal-checkpoint names are unchanged) but never builds the
    [s, b, vocab] logits as a stage output. The hidden states pass through
    unchanged and the weight is published for ChunkedCrossEntropy, which
    recomputes each chunk's logits on the fly.

    The class name must NOT match 'transformer' (case-insensitive):
    PipelineModule._find_layer_type regex-matches typename.__name__ for
    partition_method='type:transformer', and LMHeadPipe scores 0 there, so
    stage partitioning stays identical.

    Mask threading: LMHeadPipe's no-args.attn_mask tuple return
    `(logits, inputs[1])` is deliberately not reproduced. On this fork's
    pretrain driver args.attn_mask is always set except under
    --intra-document-attention, where _strip_pipe_extras has already reduced
    the tail stream to a single tensor -- so inputs here is that tensor (or
    a tuple whose [0] is, which mirrors LMHeadPipe's own unwrap).
    """

    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if isinstance(inputs, tuple):
            hidden_states = inputs[0]
        else:
            hidden_states = inputs
        _publish_lm_head_weight(self.lm_head.weight)
        return hidden_states


def _keep_dtype_for_chunked_tail(inputs):
    """Identity stand-in for float16_to_fp32 on the chunked tail. The stage
    output is the [s, b, h] hidden states, which the chunked loss consumes
    in the model dtype: each chunk's logits are upcast chunk-locally inside
    _seq_chunked_cross_entropy, so upcasting here would (a) re-create a
    full-length fp32 tensor and (b) break the matmul dtype in
    parallel_lm_logits (fp32 input x bf16 weight). A named module-level
    function -- not a lambda, not an omitted spec -- keeps the spec-list
    length and index space identical to the unchunked tail for
    custom_pipeline_partition / custom_checkpoint_partition users."""
    return inputs


class GPTModelPipe(PipelineModule,MegatronModule):
    """GPT-2 Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True):
        args = get_args()
        self.parallel_output = parallel_output

        if config.init_method is None:
            config.init_method = init_method_normal(config.init_method_std)

        if config.output_layer_init_method is None:
            config.output_layer_init_method = scaled_init_method_normal(config.init_method_std,
                                                                        config.num_layers)

        self.specs = []

        def _to_float16(inputs):
            if args.fp16:
                return fp32_to_float16(inputs, lambda v: v.half())
            elif args.bf16:
                return fp32_to_float16(inputs, lambda v: v.bfloat16())
            else:
                return inputs

        self.specs.append(_to_float16)

        # Embedding layer
        if args.untie_embeddings_and_output_weights:
            self.specs.append(LayerSpec(EmbeddingPipe,
                                        args.hidden_size,
                                        args.padded_vocab_size,
                                        args.max_position_embeddings,
                                        args.hidden_dropout,
                                        config,
                                        num_tokentypes=num_tokentypes,
                                        embedding_weights_in_fp32=args.embedding_weights_in_fp32,))
        else:
            self.specs.append(TiedLayerSpec('embed',
                                            EmbeddingPipe,
                                            args.hidden_size,
                                            args.padded_vocab_size,
                                            args.max_position_embeddings,
                                            args.hidden_dropout,
                                            config,
                                            num_tokentypes=num_tokentypes,
                                            embedding_weights_in_fp32=args.embedding_weights_in_fp32,
                                            tied_weight_attr='word_embeddings_weight'))

        for layer_idx in range(args.num_layers):
            self.specs.append(
                LayerSpec(ParallelTransformerLayerPipe,
                    config,
                    layer_number=layer_idx,
                    self_attn_mask_type=AttnMaskType.causal, 
                    
                    # Zixian: 2025-09-26: Adding num_experts=args.num_experts in PP model init
                    num_experts=args.num_experts[0], 
                    ))

        # Final layernorm after transformer layers
        if args.intra_document_attention or getattr(args, 'pipe_moe_aux_loss', False):
            # The norm/head specs take a single tensor; drop the activation
            # mask (and stash the aux-loss rider) first.
            self.specs.append(_strip_pipe_extras)
        if args.normalization == 'layernorm':
            self.specs.append(LayerSpec(LayerNorm,
                          args.hidden_size,
                          eps=args.layernorm_epsilon))
        else:
            self.specs.append(LayerSpec(MixedFusedRMSNorm, args.hidden_size, args.layernorm_epsilon))

        def _logits_helper(embedding, lm_output):
            """A wrapper to massage inputs/outputs from pipeline. """
            return parallel_lm_logits(
                lm_output,
                embedding.word_embeddings_weight,
                self.parallel_output)
        chunked_tail = use_chunked_pipe_tail(args)
        if args.untie_embeddings_and_output_weights:
            self.specs.append(
                LayerSpec(LMHeadChunkedPipe if chunked_tail else LMHeadPipe,
                          args.hidden_size, args.padded_vocab_size, config)
            )
        else:
            self.specs.append(
                TiedLayerSpec('embed',
                              EmbeddingPipe,
                              args.hidden_size,
                              args.padded_vocab_size,
                              args.max_position_embeddings,
                              args.hidden_dropout,
                              config,
                              num_tokentypes=num_tokentypes,
                              embedding_weights_in_fp32=args.embedding_weights_in_fp32,
                              forward_fn=_logits_helper,
                              tied_weight_attr='word_embeddings_weight')
            )

        # Convert to fp32 if needed (chunked tail: keep the model dtype --
        # the per-chunk loss upcasts chunk-locally)
        if args.fp16 or args.bf16:
            self.specs.append(_keep_dtype_for_chunked_tail if chunked_tail
                              else float16_to_fp32)

        if args.checkpoint_activations:
            interval = args.checkpoint_num_layers
        elif args.recompute_granularity == "full" and args.recompute_method == 'uniform':
            # deepspeed's pipeline doesn't support the block recompute method
            interval = args.recompute_num_layers
        else:
            interval = 0

        from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
        topo = PipeModelDataParallelTopology(num_pp=mpu.get_pipeline_model_parallel_world_size(),
                                             num_mp=mpu.get_tensor_model_parallel_world_size(),
                                             num_dp=mpu.get_data_parallel_world_size())
        print (f'[megatron/model/gpt_model.py] activation_checkpoint_interval={interval}, and {args.checkpoint_num_layers=}')
        
        # Read ELMoE partition configs from args (set by ELMoE_launch.py via env→args)
        custom_pp_partition = getattr(args, 'uneven_pp_partition', None)
        checkpoint_partition = getattr(args, 'dynamic_checkpoint_partition', None)
        
        print (f'[megatron/model/gpt_model.py] {custom_pp_partition=}')
        print (f'[megatron/model/gpt_model.py] {checkpoint_partition=}')
        
        if chunked_tail:
            _loss_fn = (ChunkedCrossEntropyWithMoE
                        if getattr(args, 'pipe_moe_aux_loss', False)
                        else ChunkedCrossEntropy)
        else:
            _loss_fn = (CrossEntropyWithMoE
                        if getattr(args, 'pipe_moe_aux_loss', False)
                        else CrossEntropy)
        super().__init__(layers=self.specs,
                         loss_fn=_loss_fn,
                         topology=topo,
                         activation_checkpoint_interval=interval,
                         partition_method='type:transformer',
                         custom_pipeline_partition=custom_pp_partition, 
                         custom_checkpoint_partition=checkpoint_partition,
                         )

    def universal_checkpoint_info(self):
        info = dict()
        if DS_UNIVERSAL_CHECKPOINT_INFO:
            # Vocabulary parameters (embeddings) that require special handling due to padding.
            info[VOCABULARY_PARAMETER_PATTERNS] = [
                r"tied_modules.embed.word_embeddings.weight"
            ]

            # Replicated (shared) parameters on the pipeline dimension
            info[PIPELINE_REPLICATED_PARAMETER_PATTERNS] = [
                r"tied_modules.embed.word_embeddings.weight",
                r"tied_modules.embed.position_embeddings.weight"
            ]

            # Parameter slices that should be averaged not concatenated.
            info[TP_REPLICATED_PARAMETER_PATTERNS] = [
                r"tied_modules.embed.position_embeddings.weight",
                r"\d+.input_layernorm.weight",
                r"\d+.input_layernorm.bias",
                r"\d+.post_attention_layernorm.weight",
                r"\d+.post_attention_layernorm.bias",
                r"\d+.self_attention.dense.bias",
                r"\d+.mlp.dense_4h_to_h.bias",
                r"\d+.weight",
                r"\d+.bias",
            ]

            # Parameter that are sliced on the row dimension
            info[PARAMETER_WITH_ROW_PARALLELISM_PATTERNS] = [
                r"\d+.mlp.dense_4h_to_h.weight",
                r"\d+.self_attention.dense.weight",
            ]
        return info
