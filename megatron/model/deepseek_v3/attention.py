# coding=utf-8
"""DeepSeek-V3 multi-head latent attention (MLA) for this tree, beside ParallelAttention (megatron/model/transformer.py).

Same call and return as ParallelAttention (forward(hidden_states, attention_mask, inference_params=None, rotary_pos_emb=None)
-> (output, bias)), chosen per layer by the attention_kind 'mla' (megatron/arguments.py, the DeepSeek-V3 group). The mechanism is the
released model's (transformers 4.56 models/deepseek_v3/modeling_deepseek_v3.py, DeepseekV3Attention) in the [s, b, h] layout:

    hidden [s, b, h]
      q_a_proj (replicated) -> q_a_layernorm -> q_b_proj (column over heads)  -> q [s, b, H/TP, nope + rope]
      kv_a_proj_with_mqa (replicated) -> latent [s, b, kv_lora_rank] | k_pe [s, b, 1, rope]
      kv_a_layernorm -> kv_b_proj (column over heads)                          -> k_nope, v [s, b, H/TP, nope | v_head_dim]
      rotary (interleaved pairs, YaRN or plain) on q_pe and k_pe; k = [k_nope | k_pe expanded over heads]
      attention with scale = (nope + rope) ** -0.5 * mscale^2 (flash on the accelerator, the value padded to the query width; eager otherwise)
      o_proj (row, the sequence-parallel exit)                                  -> [s/TP, b, h]

Tensor parallelism slices the heads. Under Megatron's companion --sequence-parallel the module gathers the sequence once at its entry
and lets o_proj reduce-scatter at its exit; the four replicated pieces (q_a_proj, q_a_layernorm, kv_a_proj_with_mqa, kv_a_layernorm)
carry partial gradients per rank and are marked sequence_parallel so the engine's gradient sweep sums them over TP.

Parameter names are the released model's (self_attn.X -> self_attention.X), so a checkpoint importer maps them by prefix.
The inference cache follows ParallelAttention's contract (inference_params.key_value_memory_dict, sequence_len_offset,
batch_size_offset; the batched ragged decode with row_lengths and kv_extent): keys are cached rotated, values at their own width.
Not built here (open items of the blueprint): document-local positions for microbatches larger than one, the V3.2 indexer.
"""
import torch
import torch.nn.functional as F

from megatron import get_args
from megatron.core import parallel_state, tensor_parallel
from megatron.core.tensor_parallel.layers import linear_with_grad_accumulation_and_async_allreduce
from megatron.core.tensor_parallel.mappings import (copy_to_tensor_model_parallel_region,
                                                     gather_from_sequence_parallel_region)
from megatron.core.utils import divide
from megatron.model.enums import AttnMaskType
from megatron.model.module import MegatronModule

from .rotary import YarnRotaryTable, apply_interleaved_rope

REPLICATED_NAMES = ('q_a_proj', 'q_a_layernorm', 'kv_a_proj_with_mqa', 'kv_a_layernorm')


class RMSNorm(torch.nn.Module):
    """The released DeepseekV3RMSNorm: the variance in fp32, the weight applied in the input's dtype."""

    def __init__(self, hidden_size, eps=1e-6, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * h.to(input_dtype)


class ColumnParallelLinearGathered(tensor_parallel.ColumnParallelLinear):
    """A ColumnParallelLinear whose input is already the full sequence on every tensor-parallel rank.

    Under sequence parallelism the parent's forward would all-gather the input again inside the linear function; the module gathers
    once at its entry, so this forward skips that gather (and the copy). Without sequence parallelism it is the parent: the copy's
    backward all-reduces the input gradient across TP.
    """

    def forward(self, input_, weight=None):
        if weight is None:
            weight = self.weight
        bias = self.bias if not self.skip_bias_add else None
        if self.sequence_parallel or self.async_tensor_model_parallel_allreduce or self.is_expert_without_slicing:
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)
        output_parallel = linear_with_grad_accumulation_and_async_allreduce(
            input=input_parallel, weight=weight, bias=bias,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel=False)
        assert not self.gather_output, 'the latent attention keeps its heads on their rank'
        return output_parallel, (self.bias if self.skip_bias_add else None)


def yarn_from_args(args):
    """The rope_scaling dict of the arguments, or None when --yarn-factor is not set."""
    if getattr(args, 'yarn_factor', None) is None:
        return None
    return dict(factor=args.yarn_factor,
                original_max_position_embeddings=args.yarn_original_max_position_embeddings,
                beta_fast=args.yarn_beta_fast, beta_slow=args.yarn_beta_slow,
                mscale=args.yarn_mscale, mscale_all_dim=args.yarn_mscale_all_dim)


class DeepSeekV3Attention(MegatronModule):
    """See the module docstring. Built by _build_self_attention (transformer.py) for attention_kind 'mla'."""

    def __init__(self, config, layer_number, attn_mask_type=AttnMaskType.causal):
        super().__init__()
        args = get_args()
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.params_dtype = config.params_dtype
        tp = parallel_state.get_tensor_model_parallel_world_size()
        self.tp = tp
        self.sequence_parallel = bool(config.sequence_parallel and tp > 1)
        h = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.heads_per_partition = divide(config.num_attention_heads, tp)
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.attention_dropout = config.attention_dropout
        self.intra_document_attention = bool(getattr(args, 'intra_document_attention', False))
        eps = config.layernorm_epsilon
        dtype = config.params_dtype
        H = self.num_heads

        def replicated(in_f, out_f):
            lin = torch.nn.Linear(in_f, out_f, bias=False, dtype=dtype)
            if config.perform_initialization:
                config.init_method(lin.weight)
            return lin

        def column(in_f, out_f):
            return ColumnParallelLinearGathered(in_f, out_f, config=config, init_method=config.init_method,
                                                bias=False, gather_output=False)

        if self.q_lora_rank is None:
            self.q_proj = column(h, H * self.qk_head_dim)
        else:
            self.q_a_proj = replicated(h, self.q_lora_rank)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps, dtype)
            self.q_b_proj = column(self.q_lora_rank, H * self.qk_head_dim)
        self.kv_a_proj_with_mqa = replicated(h, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps, dtype)
        self.kv_b_proj = column(self.kv_lora_rank, H * (self.qk_nope_head_dim + self.v_head_dim))
        self.o_proj = tensor_parallel.RowParallelLinear(
            H * self.v_head_dim, h, config=config, init_method=config.output_layer_init_method,
            bias=False, input_is_parallel=True, skip_bias_add=True)

        self.rotary = YarnRotaryTable(self.qk_rope_head_dim, args.rotary_base, yarn_from_args(args))
        self.softmax_scale = (self.qk_head_dim ** (-0.5)) * self.rotary.softmax_scale_factor()

        from deepspeed.accelerator import get_accelerator
        self.use_flash = bool(args.use_flash_attn_v2 and not args.use_flash_attn_triton
                              and attn_mask_type == AttnMaskType.causal
                              and get_accelerator().device_name() == 'cuda')
        if self.use_flash:
            from megatron.model.transformer import FlashSelfAttention
            self.core_attention_flash = FlashSelfAttention(causal=True, softmax_scale=self.softmax_scale,
                                                           attention_dropout=config.attention_dropout)
        self._mark_replicated()
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0 and self.layer_number == 1:
            print(f'[deepseek_v3] MLA attention: heads {H} (per TP rank {self.heads_per_partition}), q_lora_rank {self.q_lora_rank}, '
                  f'kv_lora_rank {self.kv_lora_rank}, head dims nope/rope/v {self.qk_nope_head_dim}/{self.qk_rope_head_dim}/{self.v_head_dim}, '
                  f'softmax scale {self.softmax_scale:.6f}, yarn {self.rotary.yarn}, flash {self.use_flash}, '
                  f'sequence_parallel {self.sequence_parallel}', flush=True)

    # ------------------------------------------------------------------ helpers
    def _mark_replicated(self):
        """Under sequence parallelism the replicated pieces hold partial gradients per TP rank; the engine's sweep
        (_allreduce_sequence_parallel_grads) sums every parameter marked sequence_parallel."""
        if not self.sequence_parallel:
            return
        for name in REPLICATED_NAMES:
            m = getattr(self, name, None)
            if m is not None:
                for p in m.parameters():
                    setattr(p, 'sequence_parallel', True)

    def replicated_parameter_names(self):
        return [n for n in REPLICATED_NAMES if getattr(self, n, None) is not None]

    def _positions(self, s, b, attention_mask, device):
        """Document-local positions under intra-document attention (cu_seqlens ride the attention_mask slot; the pipe path pads
        them with -1), else 0..s-1."""
        if self.intra_document_attention and attention_mask is not None and attention_mask.dtype in (torch.int32, torch.int64):
            cu = attention_mask.flatten()
            cu = cu[cu >= 0]
            n = int(cu[-1].item())
            assert n == s * b and b == 1, ('document-local positions are built for micro-batch 1 under intra-document attention; '
                                           f'got {s} x {b} tokens and cu_seqlens ending at {n}')
            lengths = cu[1:] - cu[:-1]
            starts = torch.repeat_interleave(cu[:-1], lengths)
            return torch.arange(s, device=device) - starts.to(device)
        return torch.arange(s, device=device)

    def _eager_attention(self, query, key, value, query_start=0):
        """The released eager_attention_forward in the [s, b, heads, d] layout: scores in fp32, causal, softmax in fp32. The
        query rows sit at positions query_start .. query_start + s_q - 1 of the key range (the cached prefix at inference)."""
        s, b, hp, _ = query.shape
        s_k = key.shape[0]
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)
        scores = torch.matmul(q, k.transpose(-1, -2)).to(torch.float32) * self.softmax_scale
        q_pos = torch.arange(s, device=scores.device) + query_start
        causal = torch.arange(s_k, device=scores.device)[None, :] <= q_pos[:, None]
        scores = scores.masked_fill(~causal, torch.finfo(torch.float32).min)
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        if self.training and self.attention_dropout > 0.0:
            probs = F.dropout(probs, p=self.attention_dropout)
        context = torch.matmul(probs, v)
        return context.permute(2, 0, 1, 3).reshape(s, b, hp * self.v_head_dim)

    def _flash_attention(self, query, key, value, attention_mask):
        """FlashSelfAttention takes [b, s, heads, d] with one head width for q, k and v: the value is padded to the query width
        (the released flash path does the same) and the output sliced back."""
        s, b, hp, _ = query.shape
        q, k, v = [t.permute(1, 0, 2, 3).contiguous() for t in (query, key, value)]
        v = F.pad(v, [0, self.qk_head_dim - self.v_head_dim])
        kwargs = {}
        if self.intra_document_attention and attention_mask is not None:
            kwargs['cu_seqlens'] = attention_mask
        if self.sequence_parallel:
            context = self.core_attention_flash(q, k, v, **kwargs)
        else:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                context = self.core_attention_flash(q, k, v, **kwargs)
        context = context[..., : self.v_head_dim]
        return context.permute(1, 0, 2, 3).reshape(s, b, hp * self.v_head_dim)

    # ------------------------------------------------------------------ forward
    # ------------------------------------------------------------------ the inference cache (the same contract as ParallelAttention)
    def _allocate_cache(self, inference_params, like):
        """k [max_s, max_b, heads, nope + rope] (rotated) and v [max_s, max_b, heads, v_head_dim] in the activation's dtype."""
        hp = self.heads_per_partition
        n, bmax = inference_params.max_sequence_len, inference_params.max_batch_size
        k = torch.empty(n, bmax, hp, self.qk_head_dim, dtype=like.dtype, device=like.device)
        v = torch.empty(n, bmax, hp, self.v_head_dim, dtype=like.dtype, device=like.device)
        return k, v

    def _ragged_decode(self, query, key_new, value_new, k_mem, v_mem, row_lengths, inference_params):
        """One batched decode step over rows at different positions (the contract of ParallelAttention._ragged_decode_step, the
        eval adapter's batched generation): the cache slot is the token's true position; row b's new token lands at row_lengths[b]
        and attends to slots 0 .. row_lengths[b]; keys are stored rotated at their true position. fp32 eager math."""
        b = query.shape[1]
        b_idx = torch.arange(b, device=query.device)
        k_mem[row_lengths, b_idx] = key_new[0]
        v_mem[row_lengths, b_idx] = value_new[0]
        kv_hi = getattr(inference_params, 'kv_extent', None)
        if kv_hi is None:
            kv_hi = int(row_lengths.max().item()) + 1
        k = k_mem[:kv_hi, :b].float()
        v = v_mem[:kv_hi, :b].float()
        q = query[0].float()
        scores = torch.einsum('bnh,sbnh->bns', q, k) * self.softmax_scale
        dead = torch.arange(kv_hi, device=query.device)[None, :] > row_lengths[:, None]
        scores = scores.masked_fill(dead[:, None, :], float('-inf'))
        probs = torch.softmax(scores, dim=-1)
        context = torch.einsum('bns,sbnh->bnh', probs, v)
        return context.reshape(1, b, -1).to(query.dtype)

    def forward(self, hidden_states, attention_mask, encoder_output=None, inference_params=None, rotary_pos_emb=None):
        assert encoder_output is None, 'the latent attention is self-attention only'
        # rotary_pos_emb is the old table built by the model for ParallelAttention; this class owns its own table.
        if self.sequence_parallel:
            x = gather_from_sequence_parallel_region(hidden_states, tensor_parallel_output_grad=True)
        else:
            x = hidden_states
        s, b, _ = x.shape
        hp = self.heads_per_partition
        cache = None
        row_lengths = None
        query_start = 0
        if inference_params is not None:
            kv = inference_params.key_value_memory_dict
            if self.layer_number not in kv:
                kv[self.layer_number] = self._allocate_cache(inference_params, x)
            cache = kv[self.layer_number]
            row_lengths = getattr(inference_params, 'row_lengths', None)
            if row_lengths is not None and s != 1:
                row_lengths = None                       # the ragged contract is a one-token step; a prefill is uniform
            query_start = inference_params.sequence_len_offset

        if self.q_lora_rank is None:
            q, _ = self.q_proj(x)
        else:
            q, _ = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(s, b, hp, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed = self.kv_a_proj_with_mqa(x)
        latent, k_pe = torch.split(compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv, _ = self.kv_b_proj(self.kv_a_layernorm(latent))
        kv = kv.view(s, b, hp, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = k_pe.view(s, b, 1, self.qk_rope_head_dim)

        if row_lengths is not None:
            # one token per row at its own position: cos, sin [1, b, dim]
            cos, sin = self.rotary.cos_sin(row_lengths.to(x.device))
            cos, sin = cos[None], sin[None]
        elif cache is not None:
            positions = torch.arange(s, device=x.device) + query_start
            cos, sin = self.rotary.cos_sin(positions)
        else:
            positions = self._positions(s, b, attention_mask, x.device)
            cos, sin = self.rotary.cos_sin(positions)
        q_pe = apply_interleaved_rope(q_pe, cos, sin)
        k_pe = apply_interleaved_rope(k_pe, cos, sin)

        query = torch.cat((q_nope, q_pe), dim=-1)
        key = torch.cat((k_nope, k_pe.expand(s, b, hp, self.qk_rope_head_dim)), dim=-1)
        if cache is not None:
            k_mem, v_mem = cache
            if row_lengths is not None:
                context = self._ragged_decode(query, key, value, k_mem, v_mem, row_lengths, inference_params)
                output, bias = self.o_proj(context)
                return output, bias
            b0 = inference_params.batch_size_offset
            b1 = b0 + b
            seq_end = query_start + s
            assert seq_end <= k_mem.shape[0] and b1 <= k_mem.shape[1], 'the inference cache is smaller than the sequence or the batch'
            k_mem[query_start:seq_end, b0:b1] = key
            v_mem[query_start:seq_end, b0:b1] = value
            key = k_mem[:seq_end, b0:b1]
            value = v_mem[:seq_end, b0:b1]
        if self.use_flash and (cache is None or self.training is False):
            context = self._flash_attention(query, key, value, attention_mask if cache is None else None)
        else:
            context = self._eager_attention(query, key, value, query_start)
        output, bias = self.o_proj(context)
        return output, bias
