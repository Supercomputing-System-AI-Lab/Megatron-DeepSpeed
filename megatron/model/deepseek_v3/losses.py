# coding=utf-8
"""The two forms of the indexer loss of DeepSeek-V3.2 (the report p4, eq. 3-4; blueprint CS7). The teacher is the main
attention's distribution, summed over the heads (all of them: the per-rank head sums are all-reduced over tensor parallelism),
L1-normalised over the keys, detached; the student is the softmax of the index scores; the loss is KL(teacher || student) per
query, averaged over the queries. The indexer's inputs are detached at its entry (attention.py), so the main model never
learns from this loss and the indexer learns from nothing else.

    dense warm-up (eq. 3):  the teacher over all visible keys, the student over all keys
    sparse stage (eq. 4):   both over the selected set of the query
"""
import torch


def _teacher(probs):
    """probs [s, b, H, n] (detached) -> the head-summed, tensor-parallel-summed, L1-normalised target [s, b, n] fp32."""
    p = probs.detach().float().sum(dim=2)
    if torch.distributed.is_initialized():
        from megatron.core import mpu
        if mpu.get_tensor_model_parallel_world_size() > 1:
            torch.distributed.all_reduce(p, group=mpu.get_tensor_model_parallel_group())
    return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-20)


def _kl(p, scores):
    """sum_u p (log p - log softmax(scores)) per row, the zero-probability slots and the -inf scores contributing nothing;
    the mean over the rows."""
    logq = torch.log_softmax(scores.float(), dim=-1)
    term = torch.where(p > 0, p * (torch.log(p.clamp_min(1e-20)) - logq), torch.zeros_like(p))
    return term.sum(dim=-1).mean()


def indexer_loss_dense(index_scores, dense_probs):
    """index_scores [s, b, s] fp32 (-inf where not visible), dense_probs [s, b, H, s] (the main attention's probabilities,
    0 where not visible) -> the warm-up loss (eq. 3)."""
    return _kl(_teacher(dense_probs), index_scores)


def indexer_loss_sparse(topk_scores, sparse_probs):
    """topk_scores [s, b, k] fp32 (-inf at the padding), sparse_probs [s, b, H, k] (the sparse core's probabilities, 0 at the
    padding) -> the sparse-stage loss (eq. 4)."""
    return _kl(_teacher(sparse_probs), topk_scores)


def dense_attention_probs(query, key, scale, visible, q_chunk=512):
    """The main attention's probabilities recomputed in fp32 for the warm-up teacher: query [s, b, H, d], key [s, b, H, d]
    (rotated), visible [s, s] -> [s, b, H, s] (0 where not visible), chunked over the queries."""
    s = query.shape[0]
    out = []
    for t0 in range(0, s, q_chunk):
        t1 = min(t0 + q_chunk, s)
        scores = torch.einsum('tbhd,ubhd->tbhu', query[t0:t1].float(), key.float()) * scale
        scores = scores.masked_fill(~visible[t0:t1, None, None, :], float('-inf'))
        out.append(torch.softmax(scores, dim=-1))
    return torch.cat(out, 0)
