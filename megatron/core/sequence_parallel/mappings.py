# [X-MoE 2026-09-17, the Ulysses blueprint C6b] the all-gather over the sequence-parallel (Ulysses) group along the sequence axis,
# the P-group twin of tensor_parallel/mappings.py's _GatherFromSequenceParallelRegion: forward = all_gather along dim 0 (rank order =
# sequence order, the batch's P slice is contiguous), backward = reduce-scatter (every rank's head slice contributes to every token's
# latent gradient). Used by the latent attention to gather the 576-wide kv latent once instead of exchanging keys and values.
import torch
from packaging import version
from megatron.core.parallel_state import get_sequence_parallel_group, get_sequence_parallel_world_size


def _gather_along_first_dim_seq_group(input_):
    world_size = get_sequence_parallel_world_size()
    if world_size == 1:
        return input_
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size
    output = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
    if version.parse(torch.__version__) >= version.parse('1.13'):
        torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=get_sequence_parallel_group())
    else:
        torch.distributed._all_gather_base(output, input_.contiguous(), group=get_sequence_parallel_group())
    return output


def _reduce_scatter_along_first_dim_seq_group(input_):
    world_size = get_sequence_parallel_world_size()
    if world_size == 1:
        return input_
    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0, 'the first dimension must be divisible by the sequence-parallel size'
    dim_size[0] = dim_size[0] // world_size
    output = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
    if version.parse(torch.__version__) >= version.parse('1.13'):
        torch.distributed.reduce_scatter_tensor(output, input_.contiguous(), group=get_sequence_parallel_group())
    else:
        torch.distributed._reduce_scatter_base(output, input_.contiguous(), group=get_sequence_parallel_group())
    return output


class _GatherFromSequenceGroup(torch.autograd.Function):
    """[s/P, ...] of this rank -> [s, ...] of the whole sequence; the backward reduce-scatters the gradient."""
    @staticmethod
    def forward(ctx, input_):
        return _gather_along_first_dim_seq_group(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce_scatter_along_first_dim_seq_group(grad_output)


def gather_from_sequence_group(input_):
    return _GatherFromSequenceGroup.apply(input_)


# [X-MoE 2026-09-17, C6] the sequence <-> heads all-to-all of DeepSpeed-Ulysses on torch.distributed, for the [s, b, h, d] layout
# (the arithmetic of X-MoE/deepspeed/sequence/layer.py single_all_to_all with batch_dim_idx 1, without DeepSpeed's comm wrapper, so
# the model runs and is tested without the engine). scatter_idx 2, gather_idx 0: [s/P, b, h, d] -> [s, b, h/P, d] (tokens to heads);
# scatter_idx 0, gather_idx 2: the way back. The backward is the same exchange with the two indices swapped.
def _seq_all_to_all(input_, scatter_idx, gather_idx):
    P = get_sequence_parallel_world_size()
    if P == 1:
        return input_
    group = get_sequence_parallel_group()
    if scatter_idx == 2 and gather_idx == 0:
        n, b, h, d = input_.shape
        assert h % P == 0, f'{h} heads are not divisible by the sequence-parallel size {P}'
        # [n, b, P, h/P, d] -> [P, n, b, h/P, d]: block p carries the heads that rank p will own
        x = input_.reshape(n, b, P, h // P, d).permute(2, 0, 1, 3, 4).contiguous()
        out = torch.empty_like(x)
        torch.distributed.all_to_all_single(out, x, group=group)
        # block p now holds rank p's tokens of my heads: [P, n, b, h/P, d] -> [P n, b, h/P, d] (rank order = sequence order)
        return out.reshape(P * n, b, h // P, d)
    if scatter_idx == 0 and gather_idx == 2:
        s, b, hq, d = input_.shape
        assert s % P == 0
        # [P, s/P, b, hq, d]: block p carries the tokens that rank p owns
        x = input_.reshape(P, s // P, b, hq, d).contiguous()
        out = torch.empty_like(x)
        torch.distributed.all_to_all_single(out, x, group=group)
        # block p now holds rank p's heads of my tokens: [P, s/P, b, hq, d] -> [s/P, b, P hq, d]
        return out.permute(1, 2, 0, 3, 4).reshape(s // P, b, P * hq, d)
    raise ValueError(f'unsupported scatter/gather {scatter_idx}/{gather_idx}')


class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, scatter_idx, gather_idx):
        ctx.scatter_idx, ctx.gather_idx = scatter_idx, gather_idx
        return _seq_all_to_all(input_, scatter_idx, gather_idx)

    @staticmethod
    def backward(ctx, grad_output):
        return _seq_all_to_all(grad_output.contiguous(), ctx.gather_idx, ctx.scatter_idx), None, None


def seq_all_to_all(input_, scatter_idx, gather_idx):
    return _SeqAllToAll.apply(input_.contiguous(), scatter_idx, gather_idx)
