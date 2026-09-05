# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""GPT-2 model."""
import re
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
from deepspeed.moe.utils import is_moe_param

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
        PARAMETER_WITH_2_SUB_PARAMS_CAT_DIM_0,
    )
    DS_UNIVERSAL_CHECKPOINT_INFO = True
except ImportError:
    DS_UNIVERSAL_CHECKPOINT_INFO = False
def _tp_pattern(param_name):
    """Regex matching `param_name`, with numeric path components left open.

    Layer and expert indices become \\d+, so one pattern covers every layer and every
    expert. For expert parallelism that is required rather than merely tidy: a rank names
    its experts 0..num_local_experts-1, while the universal checkpoint names them by global
    id, so a pattern built literally from this rank's names would miss most of them.

    Every other component is escaped. The hand-written lists this replaces used bare '.',
    which is a regex wildcard, so they matched more loosely than they read.
    """
    return r'\.'.join(r'\d+' if part.isdigit() else re.escape(part) for part in param_name.split('.'))


def _unique(patterns):
    """Order-preserving de-duplication -- collapsing indices makes many names share one pattern."""
    return list(dict.fromkeys(patterns))


def _merge_ucp_info_dicts(dicts):
    """Per-key ordered union of universal_checkpoint_info dicts gathered across the
    pipeline group. Deterministic on every rank: keys and entries appear in
    pipeline-rank order, deduplicated. Values are lists of patterns; a key present
    on any stage is present in the union."""
    merged = dict()
    for d in dicts:
        if not d:
            continue
        for key, patterns in d.items():
            merged.setdefault(key, [])
            merged[key].extend(p for p in patterns if p not in merged[key])
    return merged


# ---------------------------------------------------------------------------------------
# GPTModel <-> GPTModelPipe parameter-name correspondence.
#
# The two classes build the same network and name it differently: GPTModel by module path
# (language_model.encoder.layers.K...), GPTModelPipe by GLOBAL SPEC INDEX (pipe/module.py
# names each layer str(local_idx + _local_start)) -- spec 0 is the fp32->16 cast, 1 the
# embedding, 2..num_layers+1 the transformer layers, then the final norm, then the LM head.
# The correspondence is a bijection parameterised only by num_layers and whether the
# embedding is tied.
#
# It is used ONLY to look up atoms a counterpart model already wrote (see
# universal_checkpoint_name_aliases and DeepSpeed's ZeROOptimizer._hp_param_folder).
# Nothing is ever renamed on disk: a universal checkpoint always carries the names the
# model that wrote it used.
# ---------------------------------------------------------------------------------------
def pipe_name_candidates(name, num_layers):
    """Pipe-class names for a GPTModel parameter name (may be empty)."""
    out = []
    m = re.match(r'language_model\.encoder\.layers\.(\d+)\.(.+)', name)
    if m:
        out.append(f'{int(m.group(1)) + 2}.{m.group(2)}')
    # The embedding's pipe spelling depends on whether the model TIES embeddings:
    # untied builds LayerSpec(EmbeddingPipe) at spec 1, tied builds TiedLayerSpec('embed').
    # Offer only the spelling that matches THIS model's configuration. Offering both would
    # let a tied reader silently satisfy its single embedding from an untied checkpoint
    # while that checkpoint's separate LM-head atom goes unread -- a wrong load with no
    # error, instead of the missing-atom failure that correctly reports the mismatch.
    tied = not getattr(get_args(), 'untie_embeddings_and_output_weights', False)
    for kind in ('word_embeddings', 'position_embeddings'):
        m = re.match(r'language_model\.embedding\.' + kind + r'\.(.+)', name)
        if m:
            out.append(f'tied_modules.embed.{kind}.{m.group(1)}' if tied
                       else f'1.{kind}.{m.group(1)}')
    m = re.match(r'language_model\.encoder\.final_(?:layernorm|norm|rmsnorm)\.(.+)', name)
    if m:
        out.append(f'{num_layers + 2}.{m.group(1)}')
    m = re.match(r'language_model\.output_layer\.(.+)', name)
    if m:
        out.append(f'{num_layers + 3}.lm_head.{m.group(1)}')
    return out


def gptmodel_name_candidates(name, num_layers):
    """GPTModel names for a pipe-class parameter name (the inverse; may be empty)."""
    out = []
    m = re.match(r'tied_modules\.embed\.(word_embeddings|position_embeddings)\.(.+)', name)
    if m:
        return [f'language_model.embedding.{m.group(1)}.{m.group(2)}']
    m = re.match(r'(\d+)\.(.+)', name)
    if not m:
        return out
    idx, sub = int(m.group(1)), m.group(2)
    if idx == 1 and sub.startswith(('word_embeddings.', 'position_embeddings.')):
        out.append(f'language_model.embedding.{sub}')
    elif 2 <= idx < num_layers + 2:
        out.append(f'language_model.encoder.layers.{idx - 2}.{sub}')
    elif idx == num_layers + 2:
        # the final norm; both spellings exist across configurations
        out.append(f'language_model.encoder.final_layernorm.{sub}')
        out.append(f'language_model.encoder.final_norm.{sub}')
    elif idx == num_layers + 3 and sub.startswith('lm_head.'):
        out.append('language_model.output_layer.' + sub[len('lm_head.'):])
    return out


def _name_aliases(param_names, to_candidates):
    """{canonical_name: [alternative, ...]} for every parameter this model owns."""
    num_layers = get_args().num_layers
    aliases = {}
    for name in param_names.values():
        alts = to_candidates(name, num_layers)
        if alts:
            aliases[name] = alts
    return aliases


def _tp_partition_dim(submodule, param, param_name):
    """Dimension along which tensor parallelism splits `param`, or None if it is replicated.

    Read from the layer's TYPE first. Megatron also stamps `tensor_model_parallel` and
    `partition_dim` on each parameter, but only from inside _initialize_affine_weight_{gpu,cpu}
    (layers.py:89-116), which run under `config.perform_initialization`. A run started with
    --no-initialization therefore has no stamps at all, and a classifier that trusted them
    would call the entire model replicated -- producing a checkpoint that cannot be loaded at
    any other tensor-parallel degree, silently. The layer type is always present.

    The annotations are still consulted for layers this function does not know, so a custom
    parallel layer is classified correctly whenever it was initialised normally.
    """
    if isinstance(submodule, tensor_parallel.RowParallelLinear):
        # The weight is split along its input dimension; the bias is full-size on every rank
        # and added once after the reduction.
        return 1 if param_name == 'weight' else None
    if isinstance(submodule, (tensor_parallel.ColumnParallelLinear, tensor_parallel.VocabParallelEmbedding)):
        # Both the weight and the bias are split along the output dimension.
        return 0
    if getattr(param, 'tensor_model_parallel', False):
        dim = getattr(param, 'partition_dim', -1)
        return dim if dim in (0, 1) else None
    return None


def _classify_tp_parameters(module):
    """Group every parameter by how tensor parallelism splits it.

    Returns (replicated, row_parallel, column_split) as lists of parameter NAMES.

    Derived from the model rather than restated as hand-written regexes, so the lists stay
    true for whatever this model is configured as -- GQA, SwiGLU, RMSNorm, MoE -- and stay
    true when it changes. A stale list does not fail loudly: a row-parallel weight that
    matches no pattern falls through to the converter's default and is concatenated along
    dim 0, which gives the right numel and the wrong tensor.
    """
    args = get_args()

    # Whether experts are sharded across tensor-parallel ranks is a configuration choice.
    # It must not be read from the per-layer `is_expert_without_slicing` flag, because
    # ColumnParallelLinear sets that from the configuration (layers.py:513) while
    # RowParallelLinear sets it from the topology (`moe and world_size == 1`, layers.py:706).
    # At TP=1 with expert tensor parallelism enabled the two disagree, and every expert's
    # dense_4h_to_h.weight would be recorded replicated -- so a later TP=4 load would chunk a
    # row-parallel weight along dim 0.
    experts_are_replicated = not getattr(args, 'enable_expert_tensor_parallelism', False)

    # A sequence-parallel position embedding shards the SEQUENCE across tensor-parallel ranks
    # (layers.py:216-233) using a plain torch.nn.Embedding, so its weight carries no
    # tensor-parallel annotations while every rank holds a different slice. Calling it
    # replicated would make the converter assert the ranks' slices are equal, which they are
    # not; the correct merge is the default concatenation along dim 0, so declare nothing.
    # Matched by prefix because the weight belongs to the wrapped child, not to the wrapper.
    sequence_sharded = tuple(
        f'{name}.' for name, sub in module.named_modules()
        if isinstance(sub, tensor_parallel.layers.SequenceParallelPositionEmbedding))

    replicated, row_parallel, column_split = [], [], []
    for module_name, submodule in module.named_modules():
        if sequence_sharded and module_name.startswith(sequence_sharded):
            for param_name, _ in submodule.named_parameters(recurse=False):
                column_split.append(f'{module_name}.{param_name}')
            continue
        for param_name, param in submodule.named_parameters(recurse=False):
            name = f'{module_name}.{param_name}' if module_name else param_name
            if experts_are_replicated and is_moe_param(param):
                replicated.append(name)
                continue
            dim = _tp_partition_dim(submodule, param, param_name)
            if dim is None:
                replicated.append(name)
            elif dim == 1:
                row_parallel.append(name)
            else:
                column_split.append(name)
    return replicated, row_parallel, column_split


def _gated_mlp_names(module, column_split):
    """Names of SwiGLU's fused gate/up projection, restricted to the ones TP actually splits.

    An annotation says HOW a tensor is sharded; it cannot say WHAT the tensor contains, and
    SwiGLU's fusion is a content property. ParallelMLP projects to twice the FFN width and
    its activation does `torch.chunk(x, 2, dim=-1)`, so each tensor-parallel rank's slice is
    [gate_r ; up_r]. Concatenating the ranks therefore yields
    [gate_0, up_0, gate_1, up_1, ...] where the unsharded layout is [gate_all ; up_all] --
    the same numel and shape, a different tensor, and no error anywhere. The converter's
    two-sub-parameter rule splits each rank's slice before concatenating, which is the only
    way to recover the original order.

    Restricted to `column_split` because a parameter that TP does not split has nothing to
    reorder, and declaring a pattern that the replicated rule claims first would leave it
    unmatched under --strict.
    """
    split = set(column_split)
    names = []
    for module_name, submodule in module.named_modules():
        # ParallelMLP sets .swiglu from args.swiglu, which is also what doubles the
        # projection width (arguments.py sets gated_linear_unit from the same flag).
        if not getattr(submodule, 'swiglu', False):
            continue
        projection = getattr(submodule, 'dense_h_to_4h', None)
        if projection is None:
            continue
        for param_name, _ in projection.named_parameters(recurse=False):
            name = f'{module_name}.dense_h_to_4h.{param_name}'
            if name in split:
                names.append(name)
    return names


def _vocabulary_names(module):
    """Names of parameters whose first dimension is the padded vocabulary.

    The vocabulary is padded up to a multiple of
    make_vocab_size_divisible_by * tensor_model_parallel_size, so the padded size is a
    property of the topology rather than of the model: the same 50257-token tokenizer gives
    50304 rows at TP=1 and 50688 at TP=4. Naming these makes the converter store them
    unpadded and each run pad back to its own size on load. Without it a TP=1 checkpoint
    cannot be loaded at TP=4 at all -- the numel check in load_hp_checkpoint_state fails.

    Two modules can carry that dimension: the input embedding, and -- when the output weight
    is not tied to it -- the output projection. The output projection is an ordinary
    ColumnParallelLinear, so it cannot be recognised by type; it is recognised by having the
    padded vocabulary as its output width, which is the property that actually matters.
    """
    args = get_args()
    names = []
    for module_name, submodule in module.named_modules():
        if isinstance(submodule, tensor_parallel.VocabParallelEmbedding):
            names.append(f'{module_name}.weight')
        elif (args.untie_embeddings_and_output_weights
              and isinstance(submodule, tensor_parallel.ColumnParallelLinear)
              and getattr(submodule, 'output_size', None) == args.padded_vocab_size):
            names.append(f'{module_name}.weight')
    return names



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

    def universal_checkpoint_name_aliases(self, param_names):
        """Alternative names under which this model's parameters may already be stored in a
        universal checkpoint (see DeepSpeed ZeROOptimizer._hp_param_folder). Consulted ONLY
        when the atom directory for the parameter's own name is absent; nothing on disk is
        ever renamed. This model names parameters by module path; a checkpoint written by
        GPTModelPipe names them by global spec index.
        """
        return _name_aliases(param_names, pipe_name_candidates)

    def universal_checkpoint_info(self):
        # GPTModel is the non-pipeline model, so one rank holds every layer and the lists
        # derived here are complete. (GPTModelPipe declares its own: its parameter names are
        # stage-local, so no single stage can enumerate the model.)
        #
        # The previous lists here were GPTModelPipe's, copied verbatim -- they name
        # `tied_modules.embed.*` and stage-local `\d+.*`, neither of which exists in this
        # model's namespace, so every pattern matched nothing. At TP=1 that is invisible,
        # because with one slice every merge branch is the identity. At TP>1 it is not: the
        # embedding is stored padded to the writer's tensor-parallel degree and the load
        # fails its numel check, and row-parallel weights would be merged along the wrong
        # dimension.
        info = dict()
        if DS_UNIVERSAL_CHECKPOINT_INFO:
            replicated, row_parallel, column_split = _classify_tp_parameters(self)
            info[TP_REPLICATED_PARAMETER_PATTERNS] = _unique(map(_tp_pattern, replicated))
            info[PARAMETER_WITH_ROW_PARALLELISM_PATTERNS] = _unique(map(_tp_pattern, row_parallel))
            # column_split is deliberately not declared: concatenating along dim 0 is what
            # the converter already does for a parameter no rule claims.
            gated = _gated_mlp_names(self, column_split)
            if gated:
                info[PARAMETER_WITH_2_SUB_PARAMS_CAT_DIM_0] = _unique(map(_tp_pattern, gated))
            info[VOCABULARY_PARAMETER_PATTERNS] = _unique(map(_tp_pattern, _vocabulary_names(self)))
            # ORIGINAL_VOCAB_SIZE is not set here: megatron/checkpointing.py already records it
            # from the tokenizer before merging this dict in.

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

        # Per-layer expert counts, mirroring the non-pipe ParallelTransformer
        # (transformer.py:2106-2126): a single-entry num_experts list is replicated
        # per expert-interval slot, then the expert_interval and first-k-dense rules
        # (1-indexed) pick each layer's count. Without this the pipe model builds
        # MoE at EVERY layer and cannot load any checkpoint trained with dense lead
        # layers (DeepSeek first_k_dense_replace) -- their atoms are a dense MLP.
        _experts_per_slot = list(args.num_experts)
        if len(_experts_per_slot) == 1:
            _experts_per_slot = _experts_per_slot * (args.num_layers // args.expert_interval)
        for layer_idx in range(args.num_layers):
            layer_num = layer_idx + 1
            if layer_num % args.expert_interval == 0:
                n_e = _experts_per_slot[(layer_num - 1) // args.expert_interval]
            else:
                n_e = 1
            if layer_num <= args.first_k_dense_replace:
                n_e = 1
            self.specs.append(
                LayerSpec(ParallelTransformerLayerPipe,
                    config,
                    layer_number=layer_idx,
                    self_attn_mask_type=AttnMaskType.causal,
                    num_experts=n_e,
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

    def universal_checkpoint_name_aliases(self, param_names):
        """Alternative names under which this model's parameters may already be stored in a
        universal checkpoint (see DeepSpeed ZeROOptimizer._hp_param_folder). Consulted ONLY
        when the atom directory for the parameter's own name is absent; nothing on disk is
        ever renamed. This model names parameters by global spec index; a checkpoint written
        by GPTModel names them by module path.
        """
        return _name_aliases(param_names, gptmodel_name_candidates)

    # Derived exactly like GPTModel above, with one pipeline-specific twist: a stage
    # only holds its own layers (with --first-k-dense-replace stage 0 may hold no MoE
    # layer at all, and only the last stage holds the LM head), so no single stage's
    # module tree yields the complete lists. _tp_pattern collapses layer and expert
    # indices to \d+, which makes a stage's contribution depend only on which layer
    # FLAVORS it holds -- so an all_gather union over the pipeline group is complete
    # and identical on every rank. The record each rank saves in its mp_rank file is
    # therefore whole-model, and the converter (which reads one file) needs nothing.
    #
    # CONSTRAINT: this method performs a collective over the pipeline group. Its only
    # caller is megatron/checkpointing.py's save path, which runs on all ranks inside
    # the collective save. Do not call it from a rank subset.
    def universal_checkpoint_info(self):
        info = dict()
        if not DS_UNIVERSAL_CHECKPOINT_INFO:
            return info
        replicated, row_parallel, column_split = _classify_tp_parameters(self)
        info[TP_REPLICATED_PARAMETER_PATTERNS] = _unique(map(_tp_pattern, replicated))
        info[PARAMETER_WITH_ROW_PARALLELISM_PATTERNS] = _unique(map(_tp_pattern, row_parallel))
        # column_split is deliberately not declared: cat(dim=0) is the converter's default.
        gated = _gated_mlp_names(self, column_split)
        if gated:
            info[PARAMETER_WITH_2_SUB_PARAMS_CAT_DIM_0] = _unique(map(_tp_pattern, gated))
        info[VOCABULARY_PARAMETER_PATTERNS] = _unique(map(_tp_pattern, _vocabulary_names(self)))
        # Tied embeddings live on the first and last stage under one name; the
        # converter dedupes them by skipping pp_index > 0 for these patterns. With
        # --untie-embeddings-and-output-weights there are no tied modules and the
        # key is (correctly) absent.
        tied = [name for name, _ in self.named_parameters() if name.startswith('tied_modules.')]
        if tied:
            info[PIPELINE_REPLICATED_PARAMETER_PATTERNS] = _unique(map(_tp_pattern, tied))
        if torch.distributed.is_initialized() and mpu.get_pipeline_model_parallel_world_size() > 1:
            gathered = [None] * mpu.get_pipeline_model_parallel_world_size()
            torch.distributed.all_gather_object(gathered, info,
                                                group=mpu.get_pipeline_model_parallel_group())
            info = _merge_ucp_info_dicts(gathered)
        return info
