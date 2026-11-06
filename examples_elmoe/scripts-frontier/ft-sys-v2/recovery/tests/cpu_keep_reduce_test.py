#!/usr/bin/env python3
"""cpu_keep_reduce_test.py (ft-sys-v2 recovery tests, milestone 2, 2026-11-06): the "all-reduce + keep" hook on DeepSpeed's own
ZeRO stage-2 reduction code, on the login node (CPU, gloo).

DeepSpeed's real methods run (DeepSpeedZeroOptimizer.initialize_gradient_partition, average_tensor, allreduce_and_scatter,
allreduce_bucket) on an optimizer object made without __init__ and given the attributes those methods read; only the accelerator
is a stand-in (no GPU streams on the login node). Four processes, partner stride 2:
  group 0, dense: 6 parameters (sizes 7, 13, 5, 22, 9, 3), partitioned over all 4 ranks (partition 15, padded);
  group 1, experts: 4 parameters (sizes 11, 6, 17, 5), partitioned over the expert data-parallel group {0, 2} or {1, 3}
  (partition 20; parameter 2 straddles the two partitions, so in the bucket's order one owner's slices are not neighbours);
  the partner (rank + 2) is in its owner's expert group, as on the GPUs (rank + 8, the next node).
Each of 3 micro-batches: every rank fills one bucket per group (parameters in reverse order, as gradients become ready) with
integer gradients (exact sums), calls average_tensor (reduce_bucket_size 10, so the bucket is cut into several all-reduces).
Check: each rank's kept piece equals its owner's partition of the reduced gradients, element for element.
Second part, DeepSpeed's reduce path (use_multi_rank_bucket_allreduce=false, dist.reduce to each slice's owner): after
average_tensor, every slice a rank owns must hold the reduced value. reduce_bucket_size 1000 here, so each owner's slices go in
one reduction of several tensors (a reduction of one tensor reduces into a view of the gradient itself and needs no copy, which
hides the defect). Run twice: as DeepSpeed ships it (the defect found by E5c 5862600: stage_1_and_2.py:1548 checks the owner's
rank in the expert group against this rank's place in the dense group; expected wrong on dense ranks 1, 2, 3, the expert owners
whose two numbers differ) and with ZeroAdapter.fix_reduce_owner_check (expected exact everywhere).
    python tests/cpu_keep_reduce_test.py   (the venv's python; CUDA_HOME/ROCM_HOME set, for DeepSpeed's import)"""
import os
import subprocess
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.dirname(HERE)
sys.path.insert(0, RC)
DENSE = [7, 13, 5, 22, 9, 3]
EXPERT = [11, 6, 17, 5]
W, STRIDE, MB = 4, 2, 3


def grad(rank, mb, gi, j, n):
    import torch
    return (torch.arange(n, dtype=torch.float32) % 5 + 1000 * gi + 100 * j + 10 * mb + rank) * 4   # multiples of 4: /4 exact


def owned_slices(z, owner_of, i):
    """[(bucket offset, numel, flat-group offset)] of the slices of the current bucket that rank owner_of(pg) owns: DeepSpeed's
    own walk (average_tensor, stage_1_and_2.py:1075-1127), per parameter slice."""
    from deepspeed.moe.utils import is_moe_param
    out, curr = [], 0
    for gi, param, param_id in z.params_in_ipg_bucket:
        pg = z.expert_dp_process_group[param.group_name] if (z.ipg_bucket_has_moe_params and is_moe_param(param)) \
            else z.dp_process_group
        w = sorted(((p, z.grad_start_offset[gi][p][param_id]) for p in z.param_to_partition_ids[gi][param_id]),
                   key=lambda t: t[1])
        for idx, (p, off) in enumerate(w):
            n = (param.numel() - off) if idx == len(w) - 1 else (w[idx + 1][1] - off)
            if p == owner_of(pg):
                out.append((curr, n, p * z.partition_size[gi] + z.grad_partition_insertion_offset[gi][p][param_id]))
            curr += n
    return out


def worker(rank, port, out, mode='keep'):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(W), LOCAL_RANK='0')
    import contextlib
    import types
    import torch
    import torch.distributed as tdist
    import deepspeed
    deepspeed.init_distributed(dist_backend='gloo', auto_mpi_discovery=False)
    from deepspeed.runtime.zero import stage_1_and_2 as s12
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    class Acc:                                       # the accelerator's stream calls, without a GPU
        def current_stream(self):
            return None

        def stream(self, s):
            return contextlib.nullcontext()

        def resolves_data_dependency(self):
            return True
    s12.get_accelerator = lambda: Acc()

    edp = [tdist.new_group([0, 2]), tdist.new_group([1, 3])][rank % 2]     # every rank creates both, in the same order
    dense = [torch.nn.Parameter(torch.zeros(n)) for n in DENSE]
    expert = [torch.nn.Parameter(torch.zeros(n)) for n in EXPERT]
    for p in expert:
        p.allreduce, p.group_name = False, 'exp'
    groups = [dense, expert]
    pid = {id(p): k for k, p in enumerate(dense + expert)}
    z = s12.DeepSpeedZeroOptimizer.__new__(s12.DeepSpeedZeroOptimizer)
    z.get_param_id = lambda p: pid[id(p)]
    z.dp_process_group = tdist.group.WORLD
    z.expert_dp_process_group = {'exp': edp}
    z.overlap_comm, z.reduce_scatter, z.use_multi_rank_bucket_allreduce = False, True, True
    z.reduce_bucket_size, z.sequence_parallel_size, z.communication_data_type = 10, 1, torch.float32
    z.flatten, z.unflatten = _flatten_dense_tensors, _unflatten_dense_tensors
    nparts = [W, 2]
    z.partition_size = {i: -(-sum(p.numel() for p in g) // nparts[i]) for i, g in enumerate(groups)}
    z.param_to_partition_ids, z.total_grads_in_partition = {0: {}, 1: {}}, {0: {}, 1: {}}
    z.is_grad_computed = {i: {q: {} for q in range(nparts[i])} for i in range(2)}
    z.grad_partition_insertion_offset = {i: {q: {} for q in range(nparts[i])} for i in range(2)}
    z.grad_start_offset = {i: {q: {} for q in range(nparts[i])} for i in range(2)}
    for i, g in enumerate(groups):
        for q in range(nparts[i]):
            z.initialize_gradient_partition(i, g, q)
    z.optimizer = types.SimpleNamespace(param_groups=[])
    from adapters.deepspeed_zero import ZeroAdapter
    ad = ZeroAdapter(types.SimpleNamespace(optimizer=z))
    owner = (rank - STRIDE) % W
    pieces = [torch.full((z.partition_size[i],), -7.0) for i in range(2)]
    res = {'rank': rank, 'mode': mode, 'checks': []}
    if mode != 'keep':                               # DeepSpeed's reduce path, as shipped or with the fix on the object
        z.use_multi_rank_bucket_allreduce = False
        z.reduce_bucket_size = 1000
        if mode == 'reduce_fix':
            ad.fix_reduce_owner_check()
        me = lambda pg: tdist.get_rank(group=pg)
        for mb in range(MB):
            for i, g in enumerate(groups):
                order = list(reversed(range(len(g))))
                z.params_in_ipg_bucket = [(i, g[j], pid[id(g[j])]) for j in order]
                z.ipg_bucket_has_moe_params = (i == 1)
                tensor = torch.cat([grad(rank, mb, i, j, g[j].numel()) for j in order])
                mine = owned_slices(z, me, i)
                z.average_tensor(tensor)
                members = list(range(W)) if i == 0 else [r for r in range(W) if r % 2 == rank % 2]
                flat = sum(torch.cat([grad(r, mb, i, j, g[j].numel()) for j in range(len(g))]) for r in members) / W
                ok = all(torch.equal(tensor[c:c + n], flat[f:f + n]) for c, n, f in mine)
                res['checks'].append([mb, i, bool(ok), len(mine)])
        with open(f'{out}.{rank}', 'w') as fh:
            json.dump(res, fh)
        tdist.barrier()
        return
    ad.wrap_reduce_keep(owner, pieces)
    for mb in range(MB):
        for i, g in enumerate(groups):
            pieces[i].zero_()
            order = list(reversed(range(len(g))))
            z.params_in_ipg_bucket = [(i, g[j], pid[id(g[j])]) for j in order]
            z.ipg_bucket_has_moe_params = (i == 1)
            tensor = torch.cat([grad(rank, mb, i, j, g[j].numel()) for j in order])
            z.average_tensor(tensor)
            # the owner's partition of the reduced gradients, from the definition: the flat group (parameter order), summed
            # over the owner's reduction group, divided by the data-parallel world (average_tensor's own predivide)
            members = list(range(W)) if i == 0 else [r for r in range(W) if r % 2 == owner % 2]
            flat = sum(torch.cat([grad(r, mb, i, j, g[j].numel()) for j in range(len(g))]) for r in members) / W
            o = members.index(owner)
            ps = z.partition_size[i]
            want = torch.zeros(ps)
            part = flat[o * ps:(o + 1) * ps]
            want[:part.numel()] = part
            res['checks'].append([mb, i, bool(torch.equal(pieces[i], want)), float((pieces[i] - want).abs().max())])
    with open(f'{out}.{rank}', 'w') as fh:
        json.dump(res, fh)
    tdist.barrier()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
        return
    env = {k: v for k, v in os.environ.items() if 'proxy' not in k.lower()}
    fails = 0
    for n, mode in enumerate(('keep', 'reduce_shipped', 'reduce_fix')):
        out = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'cpu_keep_reduce_{os.getpid()}_{mode}')
        port = 26100 + os.getpid() % 500 + 7 * n
        ps = [subprocess.Popen([sys.executable, os.path.abspath(__file__), '--worker', str(r), str(port), out, mode], env=env)
              for r in range(W)]
        codes = [p.wait(timeout=600) for p in ps]
        bad_ranks = []
        for r in range(W):
            if not os.path.exists(f'{out}.{r}'):
                print(f'FAIL  {mode} rank {r}: no result (exit codes {codes})')
                fails += 1
                continue
            res = json.load(open(f'{out}.{r}'))
            bad = [c for c in res['checks'] if not c[2]]
            bad_ranks += [r] if bad else []
            what = (f'keeps rank {(r - STRIDE) % W}\'s reduced partition' if mode == 'keep'
                    else 'holds the reduced value in every slice it owns')
            print(f'      {mode} rank {r} {what}: {len(res["checks"]) - len(bad)} of {len(res["checks"])} (micro-batch, '
                  f'group) buckets exact' + (f'; bad (micro-batch, group): {[(c[0], c[1]) for c in bad]}' if bad else ''))
        want = [1, 2, 3] if mode == 'reduce_shipped' else []   # the defect: expert owners whose expert rank != dense rank
        good = bad_ranks == want
        print(f'{"PASS" if good else "FAIL"}  {mode}: ranks with a wrong bucket {bad_ranks}, expected {want}')
        fails += not good
    print(f'\n{fails} failed' if fails else '\nall passed')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
