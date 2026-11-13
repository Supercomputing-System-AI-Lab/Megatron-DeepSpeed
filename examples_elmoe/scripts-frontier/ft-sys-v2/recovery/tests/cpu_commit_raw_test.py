#!/usr/bin/env python3
"""cpu_commit_raw_test.py (ft-sys-v2 recovery tests, 2026-11-12): rc_commit_raw's protocol on the login node, CPU and gloo.

Four processes (stride 2: rank r's copy is on rank r + 2 mod 4), two parameters each (50 and 37 values, sent in 4 chunks). One
step of 6 micro-batches whose gradients are integers (exact sums), commits every k (1 and 2):
  1. after each k-th micro-batch the commit in flight is promoted and the next one launched; the last one stays in flight;
  2. a stop: reports; decide_microbatch_raw with rank 3 lost: J = the last promoted commit (the copy of the commit in flight
     has no token yet on rank 1);
  3. ranks 0-2 take their own snapshot at J, rank 3 (the joiner) receives rank 1's host copy; all compared with the exact sums;
  4. a new RawCommit is seeded at (7, J) and its copy checked.
Also pure cases of decide_microbatch_raw.
    python tests/cpu_commit_raw_test.py   (the venv's python)"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.dirname(HERE)
sys.path.insert(0, RC)
SIZES = [50, 37]
M = 6


def piece(rank, mb, gi, n):
    import torch
    return torch.zeros(n) + float(1000 * rank + 10 * mb + gi)


class P:
    def __init__(self, n):
        import torch
        self.data, self.grad, self.requires_grad, self.n = torch.zeros(n), None, True, n

    def numel(self):
        return self.n


class FakeAdapter:
    def __init__(self):
        self.ps = [P(n) for n in SIZES]

    def zero2_dtype(self):
        import torch
        return torch.float32

    def raw_params(self):
        return list(self.ps)

    def raw_set(self, flat, sizes):
        off = 0
        for p, n in zip(self.ps, sizes):
            p.grad = None if flat is None else flat[off:off + n].clone()
            off += n


def ref(rank, J):
    import torch
    return torch.cat([sum(piece(rank, mb, i, n) for mb in range(J)) if J else torch.zeros(n) for i, n in enumerate(SIZES)])


def worker(rank, port, k, out):
    import torch
    import torch.distributed as dist
    os.environ.update(FTV2_PARTNER_STRIDE='2', FTV2_COMMIT_K=str(k), FTV2_RAW_CHUNK_MB='0.0001')
    from rc_commit_raw import RawCommit, decide_microbatch_raw
    store = dist.TCPStore('127.0.0.1', port, 4, is_master=(rank == 0))
    dist.init_process_group('gloo', store=store, rank=rank, world_size=4)
    dev = torch.device('cpu')
    ad = FakeAdapter()
    c = RawCommit(ad, rank, 4, dev, lambda m: None)
    c.set_position(7, 0, None)
    c.seed(7, 0, None)
    c.begin_step(7)
    for mb in range(M - 1):
        for i, p in enumerate(ad.ps):
            g = piece(rank, mb, i, p.n)
            p.grad = g if p.grad is None else p.grad + g
        pos = mb + 1
        if pos % k == 0:
            assert c.promote(True, lambda: False, lambda: True) in ('none', 'done')
            c.launch(pos)
    last = (M - 1) // k * k
    rep = c.report()
    reports = [None] * 4
    dist.all_gather_object(reports, rep)
    surv = {r: reports[r] for r in range(3)}
    ok, J, why = decide_microbatch_raw(surv, [3], 4, 2, 7)
    res = {'rank': rank, 'k': k, 'decide': [ok, J, why], 'report': rep, 'want_J': last - k, 'chunks': len(c.chunks)}
    if rank == 3:
        flat = torch.empty(sum(SIZES))
        if J:
            for w in dist.batch_isend_irecv([dist.P2POp(dist.irecv, flat, 1)]):
                w.wait()
        else:
            flat = None
    else:
        if rank == 1 and J:
            cp = c.copy_of(7, J)[0]
            res['copy_exact'] = bool(torch.equal(cp, ref(3, J)))
            for w in dist.batch_isend_irecv([dist.P2POp(dist.isend, cp, 3)]):
                w.wait()
        flat = c.target(J)
    got = flat if flat is not None else torch.zeros(sum(SIZES))
    res['exact'] = bool(torch.equal(got, ref(rank, J)))
    res['own_extra_exact'] = bool(torch.equal(c.target(last), ref(rank, last))) if rank != 3 else True
    ad.raw_set(flat, c.sizes)
    c2 = RawCommit(ad, rank, 4, dev, lambda m: None)
    c2.set_position(7, J, flat)
    c2.seed(7, J, flat)
    res['seed_tok'] = c2.tok_cur_h
    res['seed_copy_exact'] = bool(torch.equal(c2.host[c2.h_cur], ref((rank - 2) % 4, J))) if J else True
    c2.restore_own(J)
    res['restore_exact'] = bool(torch.equal(torch.cat([p.grad for p in ad.ps]), ref(rank, J))) if J else all(p.grad is None for p in ad.ps)
    with open(os.path.join(out, f'r{rank}.json'), 'w') as fh:
        json.dump(res, fh)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
        return
    fails = []
    for n, k in enumerate((1, 2)):
        out = tempfile.mkdtemp()
        port = 29500 + os.getpid() % 400 + n
        ps = [subprocess.Popen([sys.executable, os.path.abspath(__file__), '--worker', str(r), str(port), str(k), out])
              for r in range(4)]
        codes = [p.wait(timeout=180) for p in ps]
        rs = [json.load(open(os.path.join(out, f'r{r}.json'))) for r in range(4)] if codes == [0] * 4 else []
        good = bool(rs) and all(r['decide'][0] and r['decide'][1] == r['want_J'] and r['exact'] and r['own_extra_exact'] and
                                r['seed_tok'] == [7, r['want_J']] and r['seed_copy_exact'] and r['restore_exact'] and
                                r.get('copy_exact', True) for r in rs)
        print(f'{"PASS" if good else "FAIL"}  raw k={k}: exit codes {codes}; ' + (f'J {rs[0]["decide"][1]} (want {rs[0]["want_J"]}); '
              f'{rs[0]["chunks"]} chunks; own sets {[r["report"]["own_set"] for r in rs]}; copies {[r["report"]["copy_cur"] for r in rs]}; '
              f'exact {[r["exact"] for r in rs]}; rank 1 copy of rank 3 exact {rs[1].get("copy_exact")}' if rs else ''))
        if not good:
            fails.append(k)
    from rc_commit_raw import decide_microbatch_raw as d
    os.environ['FTV2_PARTNER_STRIDE'] = '2'
    R = lambda own, cur, prev=(-1, -1): {'commit_step': 7, 'commit_c': own[0], 'commit_k': 1, 'raw': 1, 'own_set': list(own),
                                         'copy_cur': list(cur), 'copy_prev': list(prev)}
    cases = [
        ('all at {4,5}, copy 4', {0: R([4, 5], [7, 4]), 1: R([4, 5], [7, 4], [7, 3]), 2: R([4, 5], [7, 4])}, 4),
        ('one promoted 5 and launched 6, copy 5 and 4', {0: R([5, 6], [7, 5]), 1: R([4, 5], [7, 5], [7, 4]), 2: R([4, 5], [7, 4])}, 5),
        ('survivor past the copy: the step start', {0: R([5, 6], [7, 5]), 1: R([5, 6], [7, 4], [7, 3]), 2: R([5, 6], [7, 5])}, 0),
        ('copy of another step', {0: R([4, 5], [7, 4]), 1: R([4, 5], [6, 4]), 2: R([4, 5], [7, 4])}, 0),
    ]
    for name, reps, want in cases:
        ok, J, why = d(reps, [3], 4, 2, 7)
        g = ok and J == want
        print(f'{"PASS" if g else "FAIL"}  decide: {name}: J {J} (want {want}); {why}')
        if not g:
            fails.append(name)
    ok, J, why = d({0: R([4], [7, 4]), 2: R([4], [7, 4])}, [3], 4, 2, 7)
    print(f'{"PASS" if not ok else "FAIL"}  decide: the copy holder has no report: {why}')
    if ok:
        fails.append('holder')
    print('all passed' if not fails else f'{len(fails)} failed: {fails}')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
