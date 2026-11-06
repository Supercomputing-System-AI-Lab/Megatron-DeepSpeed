#!/usr/bin/env python3
"""cpu_commit_test.py (ft-sys-v2 recovery tests, milestone 2, 2026-11-06): the commit protocol on the login node, CPU and gloo.

Four processes (two "nodes" of two ranks, stride 2), two groups of partial sums each, a stand-in adapter that keeps the sums the way
ZeRO stage 2 does (a list of tensors per group). One step of 6 micro-batches, commits every k (p2p with k = 1 and k = 2; keep with k = 1, where the stand-in all-reduce hands
each rank its owner's piece of every micro-batch, as adapters/deepspeed_zero.py wrap_reduce_keep does on the GPUs):
  1. micro-batches 0..4 are summed and committed; then a stop arrives during micro-batch 5's epilogue on every rank: the piece just
     added is dropped (rollback to the last commit);
  2. decide_microbatch() from the survivors' reports with rank 3 lost: the resume point J;
  3. rank 3 (the joiner) receives its sums at J from rank 1's copy; every other rank takes its own snapshot at J;
  4. the sums at J are compared with the exact reference (the sum of the first J pieces), and the copies are re-seeded.
Also decide_microbatch() cases: a rank one commit ahead (allowed), two ahead (window), a lost owner's copy at another position.
    python tests/cpu_commit_test.py   (the venv's python)"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.dirname(HERE)
sys.path.insert(0, RC)
SIZES = [50, 37]
M = 6


def piece(rank, mb, gi, n):
    import torch
    return torch.arange(n, dtype=torch.float32) * 0 + float(1000 * rank + 10 * mb + gi)     # integers: exact sums


class FakeZero2Adapter:
    """What the commit hook reads from DeepSpeed's ZeRO stage 2: per group a list of tensors (here one) holding the sum."""
    def __init__(self):
        self.avg = {}
        self.keep = None                                    # keep mode: (owner, pieces), as wrap_reduce_keep receives them

    def wrap_reduce_keep(self, owner, pieces):
        self.keep = (owner, pieces)

    def zero2_groups(self):
        return list(range(len(SIZES)))

    def zero2_partition_size(self, i):
        return SIZES[i]

    def zero2_dtype(self):
        import torch
        return torch.float32

    def zero2_flat_partial(self, i):
        import torch
        return self.avg[i][0].clone() if self.avg.get(i) else torch.zeros(SIZES[i])

    def zero2_set_partial(self, i, flat):
        self.avg[i] = None if flat is None else [flat.clone()]

    def add_piece(self, rank, mb):
        for i in range(len(SIZES)):
            p = piece(rank, mb, i, SIZES[i])
            self.avg[i] = [p] if not self.avg.get(i) else [self.avg[i][0] + p]
            if self.keep is not None:                       # the all-reduce hands this rank its owner's piece as well
                self.keep[1][i].copy_(piece(self.keep[0], mb, i, SIZES[i]))


def worker(rank, port, k, out, mode='p2p'):
    import torch
    import torch.distributed as dist
    os.environ['FTV2_PARTNER_STRIDE'] = '2'
    os.environ['FTV2_COMMIT_K'] = str(k)
    os.environ['FTV2_COMMIT_MODE'] = mode
    from rc_commit import Commit, decide_microbatch
    store = dist.TCPStore('127.0.0.1', port, 4, is_master=(rank == 0))
    dist.init_process_group('gloo', store=store, rank=rank, world_size=4)
    dev = torch.device('cpu')
    ad = FakeZero2Adapter()
    c = Commit(ad, rank, 4, dev, lambda m: None)
    c.set_position(7, 0, None)
    c.seed(7, 0, None)
    for mb in range(M - 1):
        ad.add_piece(rank, mb)
        c.after_epilogue(False, False)
    ad.add_piece(rank, M - 1)                       # micro-batch 5: the stop arrives during its epilogue
    c.after_epilogue(True, False)
    reports = [None] * 4
    dist.all_gather_object(reports, c.report())
    lost = [3]
    surv = {r: reports[r] for r in range(4) if r not in lost}
    ok, J, why = decide_microbatch(surv, lost, 4, 2, 7)
    res = {'rank': rank, 'k': k, 'mode': mode, 'decide': [ok, J, why], 'report': c.report()}
    if rank == 3:                                   # the joiner: its sums at J from rank 1's copy
        flats = [torch.empty(n) for n in SIZES]
        for w in dist.batch_isend_irecv([dist.P2POp(dist.irecv, t, 1) for t in flats]):
            w.wait()
    else:
        if rank == 1:
            for w in dist.batch_isend_irecv([dist.P2POp(dist.isend, t, 3) for t in c.copy_of(7, J)]):
                w.wait()
        flats = c.target(J)
    ref = [sum(piece(rank, mb, i, SIZES[i]) for mb in range(J)) if J > 0 else torch.zeros(SIZES[i]) for i in range(len(SIZES))]
    got = flats if flats is not None else [torch.zeros(n) for n in SIZES]
    res['exact'] = all(torch.equal(a, b) for a, b in zip(got, ref))
    if rank == 1:                                   # the copy rank 1 holds of its owner (rank 3) at J, against the exact sums
        cp = c.copy_of(7, J)
        res['copy_exact'] = cp is not None and all(
            torch.equal(a, sum(piece(3, mb, i, SIZES[i]) for mb in range(J)) if J > 0 else torch.zeros(SIZES[i]))
            for i, a in enumerate(cp))
    c2 = Commit(ad, rank, 4, dev, lambda m: None)
    c2.set_position(7, J, flats)
    c2.seed(7, J, flats)
    res['reseeded_token'] = [int(x) for x in c2.tok_cur.tolist()]
    with open(f'{out}.{rank}', 'w') as fh:
        json.dump(res, fh)
    dist.destroy_process_group()


def snapshot_cases():
    """ZeroAdapter.zero2_flat_partial on a stand-in ZeRO object with torch's own flatten: the snapshot must not move when the live
    sums move (E5d 5862827: with one tensor in a group, torch's flatten returned a view of it)."""
    import types
    import torch
    from torch._utils import _flatten_dense_tensors
    from adapters.deepspeed_zero import ZeroAdapter
    fails = []
    for name, lst in (('one tensor in the group', [torch.ones(5)]), ('two tensors in the group', [torch.ones(3), torch.ones(2)])):
        z = types.SimpleNamespace(averaged_gradients={0: lst}, flatten=_flatten_dense_tensors, partition_size=[5.0],
                                  gradient_accumulation_dtype=torch.float32, device='cpu',
                                  optimizer=types.SimpleNamespace(param_groups=[]))
        ad = ZeroAdapter(types.SimpleNamespace(optimizer=z))
        snap = ad.zero2_flat_partial(0)
        for t in lst:
            t.add_(1.0)                                 # the next micro-batch adds to the live sums
        good = torch.equal(snap, torch.ones(5))
        print(f'{"PASS" if good else "FAIL"}  snapshot independent of the live sums ({name}): {snap.tolist()}')
        if not good:
            fails.append(name)
    return fails


def decide_cases():
    from rc_commit import decide_microbatch
    fails = []
    rep = lambda c, cur, prv, k=1: {'commit_step': 7, 'commit_c': c, 'commit_k': k, 'copy_cur': cur, 'copy_prev': prv}
    cases = [
        ('all at 4, copy at 4', {0: rep(4, [7, 4], [7, 3]), 1: rep(4, [7, 4], [7, 3]), 2: rep(4, [7, 4], [7, 3])}, (True, 4)),
        ('one rank one commit ahead', {0: rep(5, [7, 5], [7, 4]), 1: rep(4, [7, 4], [7, 3]), 2: rep(4, [7, 4], [7, 3])}, (True, 4)),
        ('the copy one ahead (previous holds J)', {0: rep(4, [7, 4], [7, 3]), 1: rep(4, [7, 5], [7, 4]), 2: rep(4, [7, 4], [7, 3])}, (True, 4)),
        ('two ahead: window', {0: rep(6, [7, 6], [7, 5]), 1: rep(4, [7, 4], [7, 3]), 2: rep(4, [7, 4], [7, 3])}, (False, None)),
        ('copy cut (-1): window', {0: rep(4, [7, 4], [7, 3]), 1: rep(4, [-1, -1], [7, 2]), 2: rep(4, [7, 4], [7, 3])}, (False, None)),
        ('no commit yet: step start', {0: rep(0, [7, 0], [-1, -1]), 1: rep(0, [7, 0], [-1, -1]), 2: rep(0, [7, 0], [-1, -1])}, (True, 0)),
        ('E5b: copy of J cut, all survivors at J: the previous commit', {0: rep(3, [7, 3], [7, 2]), 1: rep(3, [-1, -1], [7, 2]),
                                                                       2: rep(3, [7, 3], [7, 2])}, (True, 2)),
        ('copy of J cut, a survivor at J + k: window', {0: rep(4, [7, 4], [7, 3]), 1: rep(3, [-1, -1], [7, 2]),
                                                        2: rep(3, [7, 3], [7, 2])}, (False, None)),
        ('copy of J cut at J = 1: the step start', {0: rep(1, [7, 1], [7, 0]), 1: rep(1, [-1, -1], [7, 0]),
                                                    2: rep(1, [7, 1], [7, 0])}, (True, 0)),
    ]
    for name, reps, want in cases:
        ok, J, why = decide_microbatch(reps, [3], 4, 2, 7)
        good = (ok, J) == want
        print(f'{"PASS" if good else "FAIL"}  decide: {name}: got {(ok, J)} ({why}), want {want}')
        if not good:
            fails.append(name)
    return fails


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6])
        return
    fails = decide_cases() + snapshot_cases()
    env = {k: v for k, v in os.environ.items() if 'proxy' not in k.lower()}
    for n, (mode, k) in enumerate((('p2p', 1), ('p2p', 2), ('keep', 1))):
        out = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'cpu_commit_{os.getpid()}_{mode}{k}')
        port = 25300 + os.getpid() % 500 + 3 * n
        ps = [subprocess.Popen([sys.executable, os.path.abspath(__file__), '--worker', str(r), str(port), str(k), out, mode],
                               env=env) for r in range(4)]
        codes = [p.wait(timeout=300) for p in ps]
        rs = [json.load(open(f'{out}.{r}')) for r in range(4) if os.path.exists(f'{out}.{r}')]
        if codes != [0] * 4 or len(rs) != 4:
            print(f'FAIL  {mode} k={k}: exit codes {codes}')
            fails.append(f'{mode} k={k}')
            continue
        J = rs[0]['decide'][1]
        want_J = (M - 1) // k * k if (M - 1) // k * k < M else 0   # the last commit before the stop (micro-batches 0-4 done)
        exact = all(r['exact'] for r in rs)
        tok = {tuple(r['reseeded_token']) for r in rs}
        cex = [r.get('copy_exact') for r in rs if 'copy_exact' in r]
        good = rs[0]['decide'][0] and J == want_J and exact and tok == {(7, J)} and cex == [True]
        print(f'{"PASS" if good else "FAIL"}  {mode} k={k}: resume at micro-batch {J} (want {want_J}); sums at J exact on every '
              f'rank (the joiner from its partner\'s copy): {exact}; rank 1\'s copy of rank 3 exact: {cex}; '
              f're-seeded tokens {sorted(tok)}')
        if not good:
            fails.append(f'{mode} k={k}')
    print(f'\n{len(fails)} failed' if fails else '\nall passed')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
