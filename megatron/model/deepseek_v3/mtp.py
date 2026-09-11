# coding=utf-8
"""The multi-token prediction head of DeepSeek-V3 (report p10-11, eq. 21-25; blueprint CS5), depth 1, on the pipeline path.

The last pipeline stage (the trunk's specs before the head are unchanged):

    ... layer L-1 -> final norm -> LMHeadMTPPipe -> _mtp_tail_cast -> tied 'embed' -> MTPBlockPipe -> MTPCrossEntropy
                           h ---> (logits bf16, h) -> (logits fp32, h) ---- pass, publish --- pass, publish ----> loss

    MTPCrossEntropy(outputs=(logits, h), labels=(t_next, loss_mask)), t_next[i] = t_{i+1}:
        main  = CE(logits, t_next)                         the main head, as before
        e     = Emb(t_next)                                 eq. 21: the shared table (the tied 'embed' module)
        h'    = eh_proj([enorm(e) ; hnorm(h)])              eq. 21 (the concatenation order of Megatron's implementation, decision V8)
        h_mtp = final_norm(TRM(h'))                         eq. 22: one V3 layer (latent attention + MoE)
        P     = OutHead(h_mtp)                              eq. 23: the shared head weight, published by LMHeadMTPPipe
        mtp   = CE(P, t_next shifted by one)                eq. 24: position i predicts t_{i+2}; the last position has no target
        loss  = main + lambda x mtp (+ the MoE rider)       eq. 25 with D = 1; lambda = 0.3 then 0.1 (p23)

Why the head runs at the loss: the next tokens t_{i+1} are exactly the labels the pipe engine hands to loss_fn on the last stage,
so no token ids ride the pipe. The two modules are ordinary specs of the last stage (partitioned, optimised, checkpointed like
any other) that publish themselves as they pass the stream through, the way LMHeadChunkedPipe publishes the head weight for the
chunked loss (gpt_model.py). The embedding table is shared through TiedLayerSpec('embed') (the report: "its embedding layer is
shared with the main model"), so the pipe engine all-reduces its gradient between the first and the last stage as for a tied head.

Not built here (refused by the argument validation): depth > 1, the chunked cross-entropy tail, sequence parallelism and
intra-document attention with the head.
"""
import torch

from megatron import get_args
from megatron.model.enums import AttnMaskType, LayerType
from megatron.model.language_model import Embedding, parallel_lm_logits
from megatron.model.module import MegatronModule, float16_to_fp32
from megatron.model.transformer import LMHeadPipe, ParallelTransformerLayer

from .attention import RMSNorm

# the modules of the last stage that the loss function calls, published by their own forwards in the same forward pass
_PIPE_MTP = {}
# the loss parts of the micro-batches since the last pop (the last stage only): name -> [sum, count]
_PIPE_LOSS_PARTS = {}


def _record_part(name, value):
    acc = _PIPE_LOSS_PARTS.setdefault(name, [0.0, 0])
    acc[0] += float(value.detach())
    acc[1] += 1


def pop_pipe_loss_parts():
    """The mean of each recorded loss part over the micro-batches since the last call, averaged over the data-parallel ranks of
    the stage that holds them (the last stage); {} on the other stages. training.py adds them to the printed loss dictionary."""
    if not _PIPE_LOSS_PARTS:
        return {}
    from megatron.core import mpu
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    out = {}
    for name, (total, n) in _PIPE_LOSS_PARTS.items():
        v = torch.tensor([total / max(n, 1)], dtype=torch.float32, device=device)
        if torch.distributed.is_initialized() and mpu.get_data_parallel_world_size() > 1:
            torch.distributed.all_reduce(v, group=mpu.get_data_parallel_group())
            v /= mpu.get_data_parallel_world_size()
        out[name] = v[0]
    _PIPE_LOSS_PARTS.clear()
    return out


def mtp_loss_weight(args):
    """lambda of eq. 25: --mtp-loss-scaling-factor until --mtp-loss-decay-start-iteration, then --mtp-loss-scaling-factor-decay
    (the report: 0.3 for the first 10T tokens, 0.1 for the rest)."""
    start = getattr(args, 'mtp_loss_decay_start_iteration', None)
    if start is not None and getattr(args, 'iteration', 0) >= start:
        return float(args.mtp_loss_scaling_factor_decay)
    return float(args.mtp_loss_scaling_factor)


class LMHeadMTPPipe(LMHeadPipe):
    """The untied head that also hands the trunk's normed hidden states on: (logits, hidden). Publishes the head weight for the
    prediction head's logits. The class name must not contain 'transformer' (the partition by type)."""

    def forward(self, inputs, **kwargs):
        hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
        from megatron.model.gpt_model import _publish_lm_head_weight   # gpt_model imports this module
        _publish_lm_head_weight(self.lm_head.weight)
        logits, _ = self.lm_head(hidden_states)
        return logits, hidden_states


def _mtp_tail_cast(inputs):
    """The head's logits to fp32 for the loss (what float16_to_fp32 does on the plain tail); the hidden states keep the model
    dtype for the prediction head's block. A named module-level function keeps the spec list countable."""
    logits, hidden = inputs
    return float16_to_fp32(logits), hidden


def mtp_embedding_passthrough(embedding, inputs):
    """forward_fn of the last stage's TiedLayerSpec('embed'): publishes the shared table for the loss, passes the stream on."""
    _PIPE_MTP['embedding'] = embedding
    return inputs


class MTPBlockPipe(MegatronModule):
    """eq. 21-22 around one V3 layer: h' = eh_proj([enorm(e); hnorm(h)]), TRM(h'), the module's own final norm. As a spec it
    publishes itself and passes the stream on; the loss calls forward_block with the next tokens' embeddings."""

    def __init__(self, config, layer_number, num_experts):
        super().__init__()
        args = get_args()
        h = args.hidden_size
        dtype = config.params_dtype
        eps = args.layernorm_epsilon
        self.enorm = RMSNorm(h, eps, dtype)
        self.hnorm = RMSNorm(h, eps, dtype)
        self.eh_proj = torch.nn.Linear(2 * h, h, bias=False, dtype=dtype)
        if config.perform_initialization:
            config.init_method(self.eh_proj.weight)
        self.layer = ParallelTransformerLayer(config, layer_number, layer_type=LayerType.encoder,
                                              self_attn_mask_type=AttnMaskType.causal, num_experts=num_experts,
                                              attention_kind='mla')
        self.final_norm = RMSNorm(h, eps, dtype)
        self.fp32_residual_connection = bool(args.fp32_residual_connection)

    def forward(self, inputs, **kwargs):
        _PIPE_MTP['block'] = self
        return inputs

    def forward_block(self, hidden, emb_next, attention_mask=None):
        """hidden [s, b, h] (the trunk's normed output), emb_next [s, b, h] -> (h_mtp [s, b, h], the layer's MoE balance loss)."""
        dtype = self.eh_proj.weight.dtype
        x = self.eh_proj(torch.cat((self.enorm(emb_next.to(dtype)), self.hnorm(hidden.to(dtype))), dim=-1))
        if self.fp32_residual_connection:
            x = x.float()                     # the trunk's residual stream convention at a layer's entry
        out = self.layer(x, attention_mask)
        out, moe_loss = out if isinstance(out, tuple) else (out, None)
        return self.final_norm(out.to(dtype)), moe_loss


def MTPCrossEntropy(outputs, labels):
    """The pipe's loss function with the prediction head: main + lambda x mtp (+ coefficient x the MoE rider)."""
    from megatron.model.gpt_model import CrossEntropy, _fold_pipe_moe_rider, _PIPE_LM_HEAD_WEIGHT
    args = get_args()
    logits, hidden = outputs
    main = CrossEntropy(logits, labels)
    target, loss_mask = labels                                        # [b, s]: target[:, i] = t_{i+1}
    embedding, block = _PIPE_MTP['embedding'], _PIPE_MTP['block']
    assert _PIPE_LM_HEAD_WEIGHT, 'MTPCrossEntropy ran before LMHeadMTPPipe.forward published the head weight'
    emb_next = Embedding.forward(embedding, target, None)             # [s, b, h] through the trunk's own embedding path (the tied
                                                                      # module is an EmbeddingPipe, whose own forward takes the pipe tuple)
    h_mtp, moe_loss = block.forward_block(hidden, emb_next, getattr(args, 'attn_mask', None))
    mtp_logits = parallel_lm_logits(h_mtp, _PIPE_LM_HEAD_WEIGHT[0], True)
    # eq. 24: position i predicts t_{i+2} = target[:, i + 1]; the last position has no target
    mtp_target = torch.roll(target, -1, dims=1)
    mtp_mask = torch.roll(loss_mask, -1, dims=1).clone()
    mtp_mask[:, -1] = 0
    mtp = CrossEntropy(mtp_logits, (mtp_target, mtp_mask))
    loss = main + mtp_loss_weight(args) * mtp
    if moe_loss is not None and torch.is_tensor(moe_loss) and getattr(args, 'pipe_moe_aux_loss', False):
        loss = loss + (getattr(args, 'moe_loss_coeff', 0.0) or 0.0) * moe_loss.float()
    attn = block.layer.self_attention
    if getattr(attn, 'indexer', None) is not None:                      # the block's own V3.2 indexer loss
        loss = loss + (getattr(args, 'indexer_loss_coeff', 0.0) or 0.0) * attn.pop_indexer_loss().float()
    loss = _fold_pipe_moe_rider(loss)
    if torch.is_grad_enabled():
        _record_part('lm main loss', main)
        _record_part('mtp loss', mtp)
    return loss
