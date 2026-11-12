#!/usr/bin/env python3
"""gpu_commit_overlap_test.py (ft-sys-v2 recovery tests, 2026-11-12): the commit protocol with FTV2_COMMIT_OVERLAP=1 on 4 GPUs of
one node (RCCL), before any training run uses it. The CPU test (cpu_commit_test.py) checks the same protocol on the compute
stream; here every commit's work runs on the side stream as the adapter runs it (adapters/deepspeed_zero.py install_pipe_commits,
reduce_on_side), with GPU delays that keep the side stream behind the host:

  - the "backward" of each micro-batch makes a piece on the compute stream (integers in fp32: exact sums) and adds it to the
    rank's gradient, a new tensor each time, as autograd's accumulation does;
  - at a commit: promote() of the commit in flight (the adapter's 'before'), an event on the compute stream, record_stream of the
    gradient on the side stream, then on the side stream: wait, a delay (the slow reduction), the gradient added into the sums
    (ZeRO's epilogue), keep mode: the owner's piece of the interval, and Commit.after_epilogue (the commit's own work); the host
    then drops the gradient (ZeRO's zero_grad) and runs an all-reduce of its own on the compute stream (the pipeline's traffic on
    another communicator while the side stream works);
  - clean: after the step every rank's position, snapshot and copy are exact; the host never waited for a commit at its launch;
  - stop: a stop arrives while the commit at micro-batch 4 is in flight (rank 0 sees its end first and promotes it): rank 0 at 4,
    the others at 3 (k = 1); the stop path restores each rank to its position, decide_microbatch picks J with rank 3 lost, rank 3
    takes its sums from rank 1's copy; every rank's sums at J exact;
  - control: the same clean run without record_stream must FAIL (the next backward reuses the gradient's memory before the side
    stream reads it), so that a PASS above is not luck.
    python tests/gpu_commit_overlap_test.py      (one node, 4 GPUs; the venv's python)"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.dirname(HERE)
sys.path.insert(0, RC)
SIZES = [1 << 20, 3 << 18]       # 1M and 768k values per group: the caching allocator reuses such blocks at once
M = 6
STOP_AT = 4


def val(rank, mb, gi):
    return float(1000 * rank + 10 * mb + gi)


class FakeAdapter:
    """What the commit reads from ZeRO: per group a list of tensors holding the sum (on the GPU)."""
    def __init__(self, dev):
        import torch
        self.torch, self.dev, self.avg, self.keep = torch, dev, {}, None

    def wrap_reduce_keep(self, owner, pieces):
        self.keep = (owner, pieces)

    def zero2_groups(self):
        return list(range(len(SIZES)))

    def zero2_partition_size(self, i):
        return SIZES[i]

    def zero2_dtype(self):
        return self.torch.float32

    def zero2_flat_partial(self, i):
        return self.avg[i][0].clone() if self.avg.get(i) else self.torch.zeros(SIZES[i], device=self.dev)

    def zero2_set_partial(self, i, flat):
        self.avg[i] = None if flat is None else [flat.clone()]


def worker(rank, port, k, out, mode, scenario, record):
    import torch
    import torch.distributed as dist
    os.environ['FTV2_PARTNER_STRIDE'] = '2'
    os.environ['FTV2_COMMIT_K'] = str(k)
    os.environ['FTV2_COMMIT_MODE'] = mode
    from rc_commit import Commit, decide_microbatch
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device('cuda', torch.cuda.current_device())
    store = dist.TCPStore('127.0.0.1', port, 4, is_master=(rank == 0))
    dist.init_process_group('nccl', store=store, rank=rank, world_size=4, device_id=dev)
    pg = dist.new_group(list(range(4)))          # the commit's own communicator (the controller makes one)
    pg_main = dist.new_group(list(range(4)))     # the compute stream's traffic while the side stream works
    logs = []
    ad = FakeAdapter(dev)
    c = Commit(ad, rank, 4, dev, logs.append, group=pg, spaced=True, overlap=True)
    assert c.overlap
    c.set_position(7, 0, None)
    c.seed(7, 0, None)
    side = c.side
    # GPU delays: calibrate torch.cuda._sleep (cycles) to milliseconds
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); torch.cuda._sleep(int(5e7)); b.record(); torch.cuda.synchronize()
    cyc_ms = 5e7 / a.elapsed_time(b)
    side_ms, main_ms = 60.0, 40.0               # a commit is longer than one micro-batch, shorter than two: hidden at k = 2
    w = torch.ones(1 << 18, device=dev)
    dist.all_reduce(w, group=pg_main)            # the communicators exist before the timed launches
    torch.cuda.synchronize()
    stop = {'v': False}
    stop_fn = lambda: stop['v']
    ok_fn = lambda: not stop['v']
    res = {'rank': rank, 'k': k, 'mode': mode, 'scenario': scenario, 'record': record, 'launch_ms': [], 'wait_ms': [],
           'parts_ms': [], 'hwq': os.environ.get('GPU_MAX_HW_QUEUES', 'default'), 'n_commits': 0}
    g_start = torch.cuda.Event(enable_timing=True)
    g_start.record()
    g = [None] * len(SIZES)
    last_c = 0
    for mb in range(M):
        torch.cuda._sleep(int(main_ms * cyc_ms))                         # the backward's compute
        for i, n in enumerate(SIZES):
            p = torch.full((n,), val(rank, mb, i), device=dev)
            g[i] = p if g[i] is None else g[i] + p                       # autograd's accumulation: a new tensor
        pos = mb + 1
        if pos >= M or pos % k:
            continue
        t0 = time.time()
        r = c.promote(True, stop_fn, ok_fn)                              # the adapter's 'before'
        res['wait_ms'].append(round(1000 * (time.time() - t0), 1))
        assert r in ('none', 'done'), r
        t1 = time.time()
        ev = torch.cuda.Event()
        ev.record()
        if record:
            for t in g:
                t.record_stream(side)
        ta = time.time()
        with torch.cuda.stream(side):
            side.wait_event(ev)
            st = torch.cuda.Event(enable_timing=True)
            st.record()
            torch.cuda._sleep(int(side_ms * cyc_ms))                     # the reduction
            for i in range(len(SIZES)):                                  # ZeRO's epilogue: the reduced gradient into the sums
                ad.avg[i] = [g[i].clone()] if not ad.avg.get(i) else [ad.avg[i][0] + g[i]]
            if mode == 'keep':                                           # the all-reduce hands the owner's piece of the interval
                o = ad.keep[0]
                for i in range(len(SIZES)):
                    ad.keep[1][i].fill_(sum(val(o, j, i) for j in range(pos - k, pos)))
            c.side_t0 = st
            tb = time.time()
            c.after_epilogue(False, False, pos=pos)
        tc = time.time()
        g = [None] * len(SIZES)                                          # ZeRO's zero_grad: the host lets the gradient go
        x = torch.ones(1 << 18, device=dev)
        dist.all_reduce(x, group=pg_main)                                # the pipeline's traffic on the compute stream
        td = time.time()
        res['launch_ms'].append(round(1000 * (td - t1), 1))
        res['parts_ms'].append([round(1000 * v, 1) for v in (ta - t1, tb - ta, tc - tb, td - tc)])
        res['n_commits'] += 1
        last_c = pos
        if scenario == 'stop' and pos == STOP_AT:
            if rank == 0:                                                # rank 0 sees the commit's end before the stop
                assert c.promote(True, stop_fn, ok_fn) == 'done'
            stop['v'] = True
            c.after_epilogue(True, False)                                # the stop path: back to the last promoted commit
            break
    if scenario == 'clean':
        r = c.promote(True, stop_fn, ok_fn)                              # the step's end (the adapter's 'end')
        torch.cuda.current_stream().wait_stream(side)
        g_end = torch.cuda.Event(enable_timing=True)
        g_end.record()
        torch.cuda.synchronize()
        res['gpu_total_ms'] = round(g_start.elapsed_time(g_end), 1)
        res['main_ms'], res['side_ms'] = main_ms, side_ms
        res['side_meas'] = [b for b in c.ov_ms]
        for i in range(len(SIZES)):                                      # the engine's last reduction, on the compute stream
            if g[i] is not None:
                ad.avg[i] = [ad.avg[i][0] + g[i]]
        torch.cuda.synchronize()
        exact_sum = lambda r_, upto, i: sum(val(r_, j, i) for j in range(upto))
        res['pos'] = c.c
        res['sums_exact'] = all(torch.equal(ad.avg[i][0], torch.full((n,), exact_sum(rank, M, i), device=dev))
                                for i, n in enumerate(SIZES))
        res['snap_exact'] = all(torch.equal(c.snap_cur[i], torch.full((n,), exact_sum(rank, last_c, i), device=dev))
                                for i, n in enumerate(SIZES))
        res['copy_exact'] = all(torch.equal(c.copy_cur[i], torch.full((n,), exact_sum(c.owner, last_c, i), device=dev))
                                for i, n in enumerate(SIZES))
        res['token'] = c.report()['copy_cur']
        res['want_pos'] = last_c
        res['ov_ms'] = c.ov_ms
    else:
        reports = [None] * 4
        dist.all_gather_object(reports, c.report())
        lost = [3]
        surv = {q: reports[q] for q in range(4) if q not in lost}
        ok, J, why = decide_microbatch(surv, lost, 4, 2, 7)
        res['decide'] = [ok, J, why]
        res['report'] = c.report()
        if rank == 3:                                                    # the joiner: its sums at J from rank 1's copy
            flats = [torch.empty(n, device=dev) for n in SIZES]
            for w in dist.batch_isend_irecv([dist.P2POp(dist.irecv, t, 1) for t in flats]):
                w.wait()
        else:
            if rank == 1:
                cp = c.copy_of(7, J)
                for w in dist.batch_isend_irecv([dist.P2POp(dist.isend, t, 3) for t in cp]):
                    w.wait()
            flats = c.target(J)
        torch.cuda.synchronize()
        got = flats if flats is not None else [torch.zeros(n, device=dev) for n in SIZES]
        res['exact'] = all(torch.equal(t, torch.full((n,), sum(val(rank, j, i) for j in range(J)), device=dev))
                           for i, (t, n) in enumerate(zip(got, SIZES)))
        res['dropped_logged'] = any('commit in flight dropped' in m for m in logs)
    with open(f'{out}.{rank}', 'w') as fh:
        json.dump(res, fh)
    dist.destroy_process_group()


def run(mode, k, scenario, record, n, extra=None):
    env = {kk: v for kk, v in os.environ.items() if 'proxy' not in kk.lower()}
    env.update(extra or {})
    env['NCCL_P2P_DISABLE'] = '1'      # RCCL's wrong receive inside a node (2026-11-05) is not what this test is about
    out = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'gpu_commit_ov_{os.getpid()}_{mode}{k}_{scenario}_{int(record)}')
    port = 26300 + os.getpid() % 500 + 7 * n
    ps = [subprocess.Popen([sys.executable, '-B', os.path.abspath(__file__), '--worker', str(r), str(port), str(k), out, mode,
                            scenario, str(int(record))], env=env) for r in range(4)]
    codes = [p.wait(timeout=600) for p in ps]
    rs = [json.load(open(f'{out}.{r}')) for r in range(4) if os.path.exists(f'{out}.{r}')]
    return codes, rs


def overlap_line(rs):
    """How much of the side stream's work the compute stream did NOT wait for: (serial - measured) / side work; 1 = all hidden,
    0 = the two streams ran one after the other. serial = M micro-batches of compute + every commit's side work."""
    out = []
    for r in rs:
        side = sum((a or 0) + b for a, b in r['side_meas'])
        serial = M * r['main_ms'] + side
        frac = (serial - r['gpu_total_ms']) / side if side else 0.0
        best = (serial - max(M * r['main_ms'], r['main_ms'] + side)) / side if side else 0.0   # the most the streams can hide
        out.append(f'r{r["rank"]} {r["gpu_total_ms"]:.0f} ms (serial {serial:.0f}): hidden {frac:.2f} of {best:.2f}')
    parts = [p for r in rs for p in r['parts_ms']]
    worst = max(parts, key=sum) if parts else []
    return (f'hw queues {rs[0]["hwq"]}; GPU time: ' + '; '.join(out) +
            f'; slowest launch parts (record_stream, side enqueue, commit work, compute-stream all-reduce) ms {worst}')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        a = sys.argv
        worker(int(a[2]), int(a[3]), int(a[4]), a[5], a[6], a[7], a[8] == '1')
        return
    fails, n = [], 0
    for mode, k in (('keep', 1), ('keep', 2), ('p2p', 1), ('p2p', 2)):
        n += 1
        codes, rs = run(mode, k, 'clean', True, n)
        if codes != [0] * 4 or len(rs) != 4:
            print(f'FAIL  clean {mode} k={k}: exit codes {codes}')
            fails.append(f'clean {mode} k={k}')
            continue
        good = all(r['sums_exact'] and r['snap_exact'] and r['copy_exact'] and r['pos'] == r['want_pos'] and
                   r['token'] == [7, r['want_pos']] for r in rs)
        launch = max(max(r['launch_ms']) for r in rs)
        print(f'      {overlap_line(rs)}')
        good = good and launch < 60                                   # the host never waited for the side stream at a launch
        print(f'{"PASS" if good else "FAIL"}  clean {mode} k={k}: position {rs[0]["pos"]} (want {rs[0]["want_pos"]}); sums, '
              f'snapshots, copies exact on every rank: {[(r["sums_exact"], r["snap_exact"], r["copy_exact"]) for r in rs]}; '
              f'tokens {[r["token"] for r in rs]}; longest launch {launch} ms (side work 120 ms each); host waits for the commit '
              f'in flight (ms) {rs[0]["wait_ms"]}; side ms (reduction, commit work) {rs[0]["ov_ms"][:3]}')
        if not good:
            fails.append(f'clean {mode} k={k}')
        n += 1
        codes, rs = run(mode, k, 'stop', True, n)
        if codes != [0] * 4 or len(rs) != 4:
            print(f'FAIL  stop {mode} k={k}: exit codes {codes}')
            fails.append(f'stop {mode} k={k}')
            continue
        J = rs[0]['decide'][1]
        want_J = STOP_AT - k                                          # rank 0 promoted STOP_AT, the others dropped it
        pos = [r['report']['commit_c'] for r in rs]
        good = rs[0]['decide'][0] and J == want_J and all(r['exact'] for r in rs) and pos == [STOP_AT] + [want_J] * 3 and \
            all(r['dropped_logged'] for r in rs[1:])
        print(f'{"PASS" if good else "FAIL"}  stop {mode} k={k}: positions {pos} (want {[STOP_AT] + [want_J] * 3}); resume at '
              f'{J} (want {want_J}): {rs[0]["decide"][2]}; sums at J exact on every rank (rank 3 from rank 1\'s copy): '
              f'{[r["exact"] for r in rs]}')
        if not good:
            fails.append(f'stop {mode} k={k}')
    for hwq in ('8', '16'):                                           # the GPU's hardware queues per process (ROCm default 4)
        for mode, k in (('keep', 1), ('keep', 2)):
            n += 1
            codes, rs = run(mode, k, 'clean', True, n, {'GPU_MAX_HW_QUEUES': hwq})
            if codes != [0] * 4 or len(rs) != 4:
                print(f'FAIL  clean {mode} k={k} GPU_MAX_HW_QUEUES={hwq}: exit codes {codes}')
                fails.append(f'clean {mode} k={k} hwq {hwq}')
                continue
            good = all(r['sums_exact'] and r['snap_exact'] and r['copy_exact'] and r['pos'] == r['want_pos'] for r in rs)
            print(f'{"PASS" if good else "FAIL"}  clean {mode} k={k} GPU_MAX_HW_QUEUES={hwq}: exact on every rank: {good}')
            print(f'      {overlap_line(rs)}')
            if not good:
                fails.append(f'clean {mode} k={k} hwq {hwq}')
    n += 1
    codes, rs = run('keep', 1, 'clean', False, n)                     # control: no record_stream
    caught = codes != [0] * 4 or len(rs) != 4 or not all(r['sums_exact'] and r['snap_exact'] for r in rs)
    print(f'{"PASS" if caught else "FAIL"}  control without record_stream: the test sees the reuse of the gradient\'s memory: '
          f'{caught} (sums exact per rank: {[r.get("sums_exact") for r in rs]})')
    if not caught:
        fails.append('control')
    print(f'\n{len(fails)} failed: {fails}' if fails else '\nall passed')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
