# coding=utf-8
"""The lightning indexer of DeepSeek-V3.2 (the V3.2 report eq. 1-2; blueprint CS7), the released inference/model.py Indexer
without its fp8 path and without the Hadamard rotation:

    q_I = wq_b(qr)  -> [s, b, H_I, d_I], the first d_r channels rotated with the halves pairing
    k_I = k_norm(wk(x)) -> [s, b, d_I], the first d_r channels rotated the same way (one key per token, shared by the heads)
    w   = weights_proj(x) x H_I^-1/2 x d_I^-1/2   (fp32; the released softmax_scale = d_I^-1/2 and the head scale H_I^-1/2)
    I[t, u] = sum_j w[t, j] x ReLU(q_I[t, j] . k_I[u])        (eq. 1; the released fp8_index kernel: ReLU before the weight)
    keep, per query t, the index_topk visible keys u with the largest I[t, u]   (eq. 2)

The scores are built over key chunks so the transient is bounded; the [s, s] matrix itself is kept for the dense warm-up loss
(64 MB per sequence at 4,096 tokens; the streamed form for the long-context stages is the blueprint's later optimisation).
Every tensor-parallel rank scores every row (the sequence-parallel shard form of the blueprint, D8, is a later optimisation),
so the indexer's parameters carry identical gradients on every rank and are not marked for the sequence-parallel sum.
"""
import torch

from megatron.model.module import MegatronModule

from .rotary import apply_interleaved_rope


class V32Indexer(MegatronModule):

    def __init__(self, config, hidden_size, q_lora_rank, index_n_heads, index_head_dim, index_topk, qk_rope_head_dim, eps):
        super().__init__()
        assert q_lora_rank is not None, 'the indexer reads the query residual qr (the released Indexer takes q_lora_rank)'
        assert index_head_dim >= qk_rope_head_dim, 'the indexer head must hold the rotary channels'
        dtype = config.params_dtype
        self.n_heads = int(index_n_heads)
        self.head_dim = int(index_head_dim)
        self.rope_dim = int(qk_rope_head_dim)
        self.topk = int(index_topk)
        self.wq_b = torch.nn.Linear(q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=dtype)
        self.wk = torch.nn.Linear(hidden_size, self.head_dim, bias=False, dtype=dtype)
        self.k_norm = torch.nn.LayerNorm(self.head_dim, eps=eps, dtype=dtype)        # the released LayerNorm: weight and bias
        self.weights_proj = torch.nn.Linear(hidden_size, self.n_heads, bias=False, dtype=torch.float32)   # fp32 in the release
        if config.perform_initialization:
            for lin in (self.wq_b, self.wk, self.weights_proj):
                config.init_method(lin.weight)
        self.scale = self.n_heads ** -0.5 * self.head_dim ** -0.5
        self.key_chunk = 1024

    def index_scores(self, q, k, w, visible):
        """q [s, b, H_I, d_I], k [s, b, d_I], w [s, b, H_I] fp32, visible [s, s] bool (query t sees key u) or None (causal)
        -> I [s, b, s] fp32 with -inf where a key is not visible."""
        s, b = q.shape[0], q.shape[1]
        out = q.new_empty(s, b, s, dtype=torch.float32)
        qf = q.float()
        kf = k.float()
        for u0 in range(0, s, self.key_chunk):
            u1 = min(u0 + self.key_chunk, s)
            logits = torch.einsum('tbhd,ubd->tbhu', qf, kf[u0:u1])                    # [s, b, H_I, chunk]
            out[:, :, u0:u1] = (torch.relu(logits) * w[:, :, :, None]).sum(dim=2)
        if visible is None:
            visible = torch.ones(s, s, dtype=torch.bool, device=q.device).tril()
        return out.masked_fill(~visible[:, None, :], float('-inf'))

    def select(self, scores):
        """The top-k keys per query over key chunks (the running best merged chunk by chunk, exact): indices [s, b, k] int64 with
        -1 where a query has fewer visible keys than k, and the kept scores [s, b, k] (-inf at the padding)."""
        s, b, n = scores.shape
        k = min(self.topk, n)
        best_v = scores.new_full((s, b, k), float('-inf'))
        best_i = torch.full((s, b, k), -1, dtype=torch.int64, device=scores.device)
        for u0 in range(0, n, self.key_chunk):
            u1 = min(u0 + self.key_chunk, n)
            cv, ci = scores[:, :, u0:u1].topk(min(k, u1 - u0), dim=-1)
            ci = ci + u0
            merged_v = torch.cat([best_v, cv], dim=-1)
            merged_i = torch.cat([best_i, ci], dim=-1)
            best_v, order = merged_v.topk(k, dim=-1)
            best_i = merged_i.gather(-1, order)
        best_i = best_i.masked_fill(best_v == float('-inf'), -1)
        return best_i, best_v

    def forward(self, x, qr, cos, sin, visible=None):
        """x [s, b, h] and qr [s, b, q_lora_rank] (detached by the caller: the indexer learns from its own loss only), cos/sin
        [s, d_r] or [s, b, d_r] of the module's rotary table (the same angles as the attention, applied with the halves pairing)
        -> (topk_idx [s, b, k] int64, topk_scores [s, b, k] fp32, scores [s, b, s] fp32)."""
        s, b, _ = x.shape
        q = self.wq_b(qr).view(s, b, self.n_heads, self.head_dim)
        q_pe, q_nope = q.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        q_pe = apply_interleaved_rope(q_pe, cos, sin, interleaved=False)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = self.k_norm(self.wk(x))
        k_pe, k_nope = k.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        k_pe = apply_interleaved_rope(k_pe[:, :, None, :], cos, sin, interleaved=False)[:, :, 0]
        k = torch.cat([k_pe, k_nope], dim=-1)
        # the head weights in fp32 (the release keeps weights_proj in fp32; under the engine's module-wide bf16 cast the weight is
        # bf16, so the product runs in the weight's dtype and the scores are accumulated in fp32 from there)
        w = self.weights_proj(x.to(self.weights_proj.weight.dtype)).float() * self.scale
        scores = self.index_scores(q, k, w, visible)
        topk_idx, topk_scores = self.select(scores)
        return topk_idx, topk_scores, scores


def is_indexer_parameter(name):
    """A parameter of the lightning indexer: the name has an 'indexer' component (self_attention.indexer.* in the model)."""
    return 'indexer' in name.split('.')


def warmup_lr_condition(name, param):
    """Megatron's scale_lr_cond for the dense warm-up of the V3.2 report (p4, --dsa-freeze-non-indexer): True for every
    parameter outside the indexer, whose learning rate is then multiplied by lr_mult 0 (megatron/optimizer get_param_groups).
    Frozen by learning rate rather than by requires_grad: a requires_grad freeze leaves no MoE parameter group, which
    DeepSpeed's optimizer setup refuses for a model with MoE layers (run L1, 2026-09-11), and it would change the ZeRO layout;
    this way every group and the checkpoint stay as they are. The frozen parameters still receive gradients (the report's
    warm-up saves that compute; an optimisation for later)."""
    return not is_indexer_parameter(name)


def warmup_split(model):
    """(indexer parameters, all parameters) counts over a module or a list of modules, for the log line."""
    modules = model if isinstance(model, (list, tuple)) else [model]
    n_idx = total = 0
    for m in modules:
        for name, p in m.named_parameters():
            total += p.numel()
            n_idx += p.numel() if is_indexer_parameter(name) else 0
    return n_idx, total
