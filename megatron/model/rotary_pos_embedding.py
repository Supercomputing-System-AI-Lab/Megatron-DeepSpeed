# coding=utf-8

# The following code has been taken from https://github.com/NVIDIA/NeMo/blob/ \
# 782b4e1652aaa43c8be390d9db0dc89544afa080/nemo/collections/nlp/modules/ \
# common/megatron/rotary_pos_embedding.py

import importlib.util
import os
import torch

from torch import einsum, nn

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        # `base` (rope theta) is raised for context extension (--rotary-base);
        # default 10000 keeps every existing run bit-identical.
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        # 2026-10-17 (assistant_deep_dive/2026-10-17-rope_bf16): the July fix below keeps the POSITIONS in fp32, but the model
        # casts (DeepSpeed engine self.module.bfloat16(); the eval loader's model[0].bfloat16()) still round this buffer: 63 of
        # the 64 frequencies move (up to 0.37 % at base 5e6), and .float() in forward cannot restore them. _apply keeps the fp32
        # values: the DEFAULT since 2026-10-17; XMOE_ROPE_FP32_FREQ=0 restores the rounded frequencies (to reproduce an old run).
        self._rope_fp32_freq = os.environ.get('XMOE_ROPE_FP32_FREQ', '1') == '1'
        if self._rope_fp32_freq and os.environ.get('RANK', '0') == '0':
            print('[rope] RotaryEmbedding keeps inv_freq in fp32 through the model casts (default; XMOE_ROPE_FP32_FREQ=0 restores the bf16-rounded frequencies)', flush=True)
        if importlib.util.find_spec('einops') is None:
            raise RuntimeError("einops is required for Rotary Embedding")

    def _apply(self, fn, *args, **kwargs):
        if not getattr(self, '_rope_fp32_freq', False):
            return super()._apply(fn, *args, **kwargs)
        keep = self._buffers['inv_freq'].detach().to(torch.float32).clone()
        r = super()._apply(fn, *args, **kwargs)
        self._buffers['inv_freq'] = keep.to(self._buffers['inv_freq'].device)
        return r

    def forward(self, max_seq_len, offset=0):
        # BEFORE
        # seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        # print (f'[rotary_pos_embedding.py] at forward, {seq.dtype=}')
        # freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        # print (f'[rotary_pos_embedding.py] at forward, {freqs.dtype=}')
        
        # AFTER
        # Positions AND inv_freq MUST stay float32. Under --bf16 the model cast makes
        # self.inv_freq bf16, and `seq.type_as(self.inv_freq)` would then quantize the token
        # positions to bf16 -- which is exact only up to 256 (2^8), so positions >= 257 round
        # (257->256, 259->260, ...) and the rotary angle is corrupted, worsening with position
        # and compounding across layers. HF keeps float32 positions (cached at init), which is why
        # HF matches at all positions. Force float32 here; apply_rotary_pos_emb casts cos/sin down
        # to the tensor dtype at the end (same as HF), so no other dtype changes are needed.
        seq = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=torch.float32) + offset
        freqs = einsum('i , j -> i j', seq, self.inv_freq.float())
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        from einops import rearrange
        return rearrange(emb, 'n d -> n 1 1 d')


def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    from einops import rearrange
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * freqs.cos().to(t.dtype)) + (_rotate_half(t) * freqs.sin().to(t.dtype))
    return torch.cat((t, t_pass), dim=-1)
