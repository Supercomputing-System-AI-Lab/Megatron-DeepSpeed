import torch
from packaging import version

from megatron.core.parallel_state import (
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size
)
from megatron.core import tensor_parallel   # [X-MoE 2026-09-17]

class _VocabSequenceParallelCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_seq_parallel_logits, target, label_smoothing=0.0):
        # vocab_seq_parallel_logits: [S/P, B, V]
        # target: [S/P, B]
        # return: [S, B]

        # Need softmax for backward
        softmax = torch.nn.functional.softmax(vocab_seq_parallel_logits, dim=-1)
        ctx.vocab_size = vocab_seq_parallel_logits.size(2)
        loss = torch.nn.functional.nll_loss(softmax.log().view(-1, ctx.vocab_size), target.view(-1), reduction='none')
       
        ctx.seqlen = vocab_seq_parallel_logits.size(0) * get_sequence_parallel_world_size()
        batch_size = vocab_seq_parallel_logits.size(1)

        loss_all = torch.empty(ctx.seqlen, batch_size, dtype=vocab_seq_parallel_logits.dtype, device=vocab_seq_parallel_logits.device)
        if version.parse(torch.__version__) >= version.parse('1.13'):
            torch.distributed.all_gather_into_tensor(loss_all, loss, group=get_sequence_parallel_group())
        else:
            torch.distributed._all_gather_base(loss_all, loss, group=get_sequence_parallel_group())

        ctx.save_for_backward(softmax, target)

        return loss_all

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target = ctx.saved_tensors

        step_seqlen = ctx.seqlen // get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        grad_output_part = grad_output[step_seqlen*sp_rank:step_seqlen*(sp_rank + 1), :]

        grad_input = softmax
        grad_2d = grad_input.view(-1, ctx.vocab_size)
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0],
                                 device=grad_2d.device)

        grad_2d[arange_1d, target.view(-1)] -= 1
        grad_input.mul_(grad_output_part.unsqueeze(dim=-1))

        return grad_input, None, None


def vocab_sequence_parallel_cross_entropy(vocab_parallel_logits, target, label_smoothing=0.0):
    return _VocabSequenceParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)


# [X-MoE 2026-09-17, the Ulysses blueprint C7] the gathered cross entropy that is correct under tensor parallelism too. The class
# above takes a softmax over the whole last dimension, which is wrong on a vocab-parallel shard [s/P, b, V/TP]; this one composes the
# TP-correct vocab_parallel_cross_entropy (its all-reduces inside, differentiable) with an all-gather of the per-token losses over the
# P sequence ranks, in rank order = sequence order (the batch's P slice is contiguous, pretrain_gpt_deepspeed.py). The backward of the
# gather returns this rank's window of the incoming gradient. At TP 1 it equals the class above; at P 1 the gather is the identity.
class _GatherSequenceLosses(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss):
        P = get_sequence_parallel_world_size()
        if P == 1:
            return loss
        loss = loss.contiguous()
        out = torch.empty(loss.size(0) * P, *loss.shape[1:], dtype=loss.dtype, device=loss.device)
        if version.parse(torch.__version__) >= version.parse('1.13'):
            torch.distributed.all_gather_into_tensor(out, loss, group=get_sequence_parallel_group())
        else:
            torch.distributed._all_gather_base(out, loss, group=get_sequence_parallel_group())
        return out

    @staticmethod
    def backward(ctx, grad_output):
        P = get_sequence_parallel_world_size()
        if P == 1:
            return grad_output
        n = grad_output.size(0) // P
        r = get_sequence_parallel_rank()
        return grad_output[r * n:(r + 1) * n].contiguous()


def gather_sequence_losses(loss):
    """[s/P, b] per-token losses of this rank's shard -> [s, b] of the whole sequence (differentiable)."""
    return _GatherSequenceLosses.apply(loss)


def vocab_and_sequence_parallel_cross_entropy(vocab_parallel_logits, target, label_smoothing=0.0):
    """logits [s/P, b, V/TP] and targets [s/P, b] of this rank -> the per-token losses [s, b] of the whole sequence."""
    loss = tensor_parallel.vocab_parallel_cross_entropy(vocab_parallel_logits, target, label_smoothing)
    return gather_sequence_losses(loss)
