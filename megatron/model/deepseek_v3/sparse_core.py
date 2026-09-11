# coding=utf-8
"""The sparse attention core of DeepSeek-V3.2 (DSA, the report p3; blueprint CS7) in the latent (MQA) form of the released
inference/model.py decode branch: the query absorbed into the latent space, the selected latent rows gathered once for all
heads, an fp32 softmax over the selected slots, the value un-absorbed.

    q_c   = q_nope . W_UK^T        per head        (W_UK = the first d_n rows of the head's slice of kv_b_proj)
    logit = (q_c . c[u] + q_pe . k_pe[u]) x scale  over the selected keys u of the query (eq. 2 of the report)
    o     = softmax(logit) @ c[u]  then  o . W_UV  per head   (W_UV = the last d_v rows of the head's slice)

masked_mha_reference is the released prefill branch (per-head keys and values, the full scores with the index mask): the
test reference, never the training path.
"""
import torch


def build_visibility(s, cu_seqlens=None, device=None):
    """[s, s] bool: query t sees key u when u <= t and, under intra-document attention (cu_seqlens over one sequence, padded
    with -1 as on the pipe), both lie in the same document."""
    visible = torch.ones(s, s, dtype=torch.bool, device=device).tril()
    if cu_seqlens is None:
        return visible
    cu = cu_seqlens.flatten()
    cu = cu[cu >= 0].to(device)
    assert int(cu[-1].item()) == s, f'cu_seqlens end at {int(cu[-1].item())}; the sequence holds {s} tokens'
    lengths = cu[1:] - cu[:-1]
    doc = torch.repeat_interleave(torch.arange(len(lengths), device=device), lengths)
    return visible & (doc[:, None] == doc[None, :])


def sparse_latent_attention(q_nope, q_pe, c, k_pe, topk_idx, kv_b_weight, scale, q_chunk=512):
    """q_nope [s, b, H, d_n], q_pe [s, b, H, d_r], c [s, b, d_c] (the normed latent), k_pe [s, b, d_r] (rotated),
    topk_idx [s, b, k] int64 (-1 = no key), kv_b_weight [H (d_n + d_v), d_c] (this rank's kv_b_proj weight), scale
    -> (context [s, b, H, d_v], probs [s, b, H, k] fp32 (0 at the padding), lse [s, b, H] fp32)."""
    s, b, H, d_n = q_nope.shape
    d_c = c.shape[-1]
    W = kv_b_weight.view(H, -1, d_c)
    W_uk, W_uv = W[:, :d_n], W[:, d_n:]
    q_c = torch.einsum('sbhd,hdc->sbhc', q_nope, W_uk.to(q_nope.dtype))                # the query in the latent space
    b_idx = torch.arange(b, device=c.device)[None, :, None]
    contexts, probs_all, lses = [], [], []
    for t0 in range(0, s, q_chunk):
        t1 = min(t0 + q_chunk, s)
        idx = topk_idx[t0:t1]
        valid = idx >= 0
        safe = idx.clamp(min=0)
        c_sel = c[safe, b_idx]                                                            # [n, b, k, d_c]
        pe_sel = k_pe[safe, b_idx]                                                        # [n, b, k, d_r]
        logits = (torch.einsum('nbhc,nbkc->nbhk', q_c[t0:t1].float(), c_sel.float())
                  + torch.einsum('nbhr,nbkr->nbhk', q_pe[t0:t1].float(), pe_sel.float())) * scale
        logits = logits.masked_fill(~valid[:, :, None, :], float('-inf'))
        lse = torch.logsumexp(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        o_c = torch.einsum('nbhk,nbkc->nbhc', probs.to(c.dtype), c_sel)                   # [n, b, H, d_c]
        contexts.append(torch.einsum('nbhc,hdc->nbhd', o_c, W_uv.to(o_c.dtype)))         # [n, b, H, d_v]
        probs_all.append(probs)
        lses.append(lse)
    return torch.cat(contexts, 0), torch.cat(probs_all, 0), torch.cat(lses, 0)


def masked_mha_reference(q_nope, q_pe, c, k_pe, topk_idx, kv_b_weight, scale, visible):
    """The released prefill branch: kv = kv_b(c) split into per-head keys and values, the full scores, the index mask (-inf
    except at the selected keys) plus the visibility, softmax, probs @ v. Returns context [s, b, H, d_v] in fp32."""
    s, b, H, d_n = q_nope.shape
    d_c = c.shape[-1]
    kv = torch.einsum('sbc,oc->sbo', c.float(), kv_b_weight.float()).view(s, b, H, -1)
    k_nope, v = kv[..., :d_n], kv[..., d_n:]
    k = torch.cat([k_nope, k_pe.float()[:, :, None, :].expand(-1, -1, H, -1)], dim=-1)
    q = torch.cat([q_nope.float(), q_pe.float()], dim=-1)
    scores = torch.einsum('tbhd,ubhd->tbhu', q, k) * scale
    valid = topk_idx >= 0
    selected = torch.zeros(s, b, s, device=c.device)
    selected.scatter_add_(-1, topk_idx.clamp(min=0), valid.float())      # a count, so the -1 padding cannot collide with key 0
    index_mask = torch.where(selected > 0, 0.0, float('-inf')).float()
    index_mask = index_mask.masked_fill(~visible[:, None, :], float('-inf'))
    scores = scores + index_mask[:, :, None, :]
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum('tbhu,ubhd->tbhd', probs, v)
