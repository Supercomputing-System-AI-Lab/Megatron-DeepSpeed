# coding=utf-8
"""The rotary table of the DeepSeek-V3 attention: the plain inverse frequencies or the YaRN-scaled ones, the cos/sin scaling
and the softmax-scale multiplier of the released model, and the interleaved rotation.

Every formula is the one of the released implementation as shipped in transformers 4.56 (models/deepseek_v3/modeling_deepseek_v3.py
and modeling_rope_utils._compute_yarn_parameters): the same order of operations, so a table built here equals the reference bit for
bit in fp32 (tests/test_v3_yarn_table.py checks that). Positions and frequencies stay fp32; the cast to the tensor's dtype happens
at the apply, the rule of megatron/model/rotary_pos_embedding.py.
"""
import math

import torch


def yarn_get_mscale(scale=1.0, mscale=1.0):
    """The attention magnitude correction of YaRN (the released yarn_get_mscale)."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class YarnRotaryTable(torch.nn.Module):
    """inv_freq [dim/2] and the two scalings of the rotary channels.

    yarn is None (a plain table with base `base`) or a dict with the config.json rope_scaling keys: factor,
    original_max_position_embeddings, beta_fast (32), beta_slow (1), mscale, mscale_all_dim, truncate (True).
    """

    def __init__(self, dim, base, yarn=None):
        super().__init__()
        self.dim = int(dim)
        self.base = float(base)
        self.yarn = dict(yarn) if yarn else None
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        if not self.yarn:
            inv_freq = 1.0 / pos_freqs
            self.attention_scaling = 1.0
        else:
            factor = float(self.yarn['factor'])
            original_max = int(self.yarn['original_max_position_embeddings'])
            beta_fast = self.yarn.get('beta_fast') or 32
            beta_slow = self.yarn.get('beta_slow') or 1
            mscale = self.yarn.get('mscale')
            mscale_all_dim = self.yarn.get('mscale_all_dim')
            truncate = self.yarn.get('truncate', True)
            if mscale and mscale_all_dim:
                self.attention_scaling = float(yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim))
            else:
                self.attention_scaling = float(yarn_get_mscale(factor))

            def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
                return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

            low = find_correction_dim(beta_fast, self.dim, self.base, original_max)
            high = find_correction_dim(beta_slow, self.dim, self.base, original_max)
            if truncate:
                low = math.floor(low)
                high = math.ceil(high)
            low, high = max(low, 0), min(high, self.dim - 1)
            if low == high:
                high += 0.001
            ramp = torch.clamp((torch.arange(self.dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
            extrapolation_factor = 1 - ramp
            inv_freq_extrapolation = 1.0 / pos_freqs
            inv_freq_interpolation = 1.0 / (factor * pos_freqs)
            inv_freq = inv_freq_interpolation * (1 - extrapolation_factor) + inv_freq_extrapolation * extrapolation_factor
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def softmax_scale_factor(self):
        """The multiplier on qk_head_dim ** -0.5 (the released DeepseekV3Attention.__init__): mscale squared when
        mscale_all_dim is set, else 1."""
        if self.yarn and self.yarn.get('mscale_all_dim'):
            m = yarn_get_mscale(float(self.yarn['factor']), self.yarn['mscale_all_dim'])
            return m * m
        return 1.0

    def cos_sin(self, positions):
        """positions [s] (int64) -> cos, sin [s, dim] in fp32, scaled by attention_scaling (the released forward)."""
        inv_freq = self.inv_freq.to(positions.device)
        freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_interleaved_rope(x, cos, sin):
    """x [s, b, heads, dim]; cos, sin [s, dim] (one angle per position) or [s, b, dim] (one per position and row, the ragged
    decode) in fp32. The released apply_rotary_pos_emb_interleave: the channel pairs (2k, 2k+1) are first regrouped into two
    halves, then the halves rotation is applied; the result keeps the regrouped order, on q and k alike, so their dot product is
    the one of the interleaved pairs."""
    s, b, h, d = x.shape
    x = x.view(s, b, h, d // 2, 2).transpose(4, 3).reshape(s, b, h, d)
    if cos.dim() == 2:
        c = cos.to(x.dtype)[:, None, None, :]
        sn = sin.to(x.dtype)[:, None, None, :]
    else:
        c = cos.to(x.dtype)[:, :, None, :]
        sn = sin.to(x.dtype)[:, :, None, :]
    return (x * c) + (rotate_half(x) * sn)
