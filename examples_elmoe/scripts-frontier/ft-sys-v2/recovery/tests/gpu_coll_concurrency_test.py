#!/usr/bin/env python3
"""gpu_coll_concurrency_test.py (ft-sys-v2 recovery tests, 2026-11-12, after P4b 5912134): do two RCCL collectives of different
communicators run at the same time on one GPU?
P4b (10B, PP 2): with the commit's reduction on a high-priority side stream, the forward pass that runs beside it takes 243 ms
instead of 112 ms, about the reduction's whole time; the backward is unchanged. The forward holds the MoE all-to-all (expert
group), the reduction is an all-reduce (data-parallel group). The question: does the all-to-all wait for the all-reduce?
Eight processes on the 8 GCDs of one node, two communicators over all 8 (A: the "expert" group, B: the "data-parallel" group):
  - the compute stream: a matmul (~20 ms) then a synchronous all_to_all_single on A (64 MB per rank), repeated, as the MoE forward;
  - the side stream (high priority, as rc_commit): a synchronous all_reduce on B of a large buffer, started after the first matmul.
Measured from GPU events: each all-to-all's duration with and without the all-reduce running, and the all-reduce's interval.
Variants (environment of the 8 workers): default; TORCH_NCCL_HIGH_PRIORITY=1; GPU_MAX_HW_QUEUES=16; both; and the all-reduce on the
compute stream (the serial reference). Prints one line per variant: the all-to-all's median time alone and during the
all-reduce, the all-reduce's time, and the share of the all-reduce that ran at the same time as an all-to-all.
    python tests/gpu_coll_concurrency_test.py     (one node, 8 GCDs)"""
import json
import os
import statistics as stt
import subprocess
import sys
import time

N = 8


def worker(rank, port, out, side_mode):
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dev = torch.device('cuda', rank)
    store = dist.TCPStore('127.0.0.1', port, N, is_master=(rank == 0))
    dist.init_process_group('nccl', store=store, rank=rank, world_size=N, device_id=dev)
    A = dist.new_group(list(range(N)))
    B = dist.new_group(list(range(N)))
    side = torch.cuda.Stream(priority=-1)
    a2a_in = torch.randn(32 << 20, device=dev, dtype=torch.bfloat16)        # 64 MB
    a2a_out = torch.empty_like(a2a_in)
    big = torch.randn(1 << 30, device=dev, dtype=torch.bfloat16)             # 2 GB
    m = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    for _ in range(3):                                                       # warm both communicators and the kernels
        dist.all_to_all_single(a2a_out, a2a_in, group=A)
        dist.all_reduce(big[:1 << 20], group=B)
        torch.mm(m, m)
    torch.cuda.synchronize()
    dist.barrier()
    E = lambda: torch.cuda.Event(enable_timing=True)
    g0 = E(); g0.record()
    a2a, ar = [], None
    for i in range(14):
        torch.mm(m, m); torch.mm(m, m)
        if i == 2:
            ev = torch.cuda.Event(); ev.record()
            s = side if side_mode == 'side' else torch.cuda.current_stream()
            with torch.cuda.stream(s):
                s.wait_event(ev)
                x0 = E(); x0.record()
                dist.all_reduce(big, group=B)
                x1 = E(); x1.record()
            ar = (x0, x1)
        e0 = E(); e0.record()
        dist.all_to_all_single(a2a_out, a2a_in, group=A)
        e1 = E(); e1.record()
        a2a.append((e0, e1))
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    T = lambda e: g0.elapsed_time(e)
    A_ = [(T(a), T(b)) for a, b in a2a]
    R = (T(ar[0]), T(ar[1]))
    during = [b - a for a, b in A_ if min(b, R[1]) - max(a, R[0]) > 0.5 * (b - a)]
    alone = [b - a for a, b in A_ if min(b, R[1]) - max(a, R[0]) <= 0]
    both = sum(max(0.0, min(b, R[1]) - max(a, R[0])) for a, b in A_)
    res = {'rank': rank, 'a2a_alone': stt.median(alone) if alone else None, 'a2a_during': stt.median(during) if during else None,
           'n_during': len(during), 'ar_ms': R[1] - R[0], 'ar_with_a2a': both / max(R[1] - R[0], 1e-9),
           'hwq': os.environ.get('GPU_MAX_HW_QUEUES', 'default'), 'hi': os.environ.get('TORCH_NCCL_HIGH_PRIORITY', '0')}
    with open(f'{out}.{rank}', 'w') as f:
        json.dump(res, f)
    dist.destroy_process_group()


def run(n, extra, side_mode):
    env = {k: v for k, v in os.environ.items() if 'proxy' not in k.lower()}
    env.update(extra)
    out = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'coll_conc_{os.getpid()}_{n}')
    port = 27300 + os.getpid() % 500 + 7 * n
    ps = [subprocess.Popen([sys.executable, '-B', os.path.abspath(__file__), '--worker', str(r), str(port), out, side_mode], env=env)
          for r in range(N)]
    codes = [p.wait(timeout=600) for p in ps]
    rs = [json.load(open(f'{out}.{r}')) for r in range(N) if os.path.exists(f'{out}.{r}')]
    return codes, rs


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
        return
    variants = [('all-reduce on the compute stream (serial reference)', {}, 'main'),
                ('side stream, defaults', {}, 'side'),
                ('side stream, TORCH_NCCL_HIGH_PRIORITY=1', {'TORCH_NCCL_HIGH_PRIORITY': '1'}, 'side'),
                ('side stream, GPU_MAX_HW_QUEUES=16', {'GPU_MAX_HW_QUEUES': '16'}, 'side'),
                ('side stream, both', {'TORCH_NCCL_HIGH_PRIORITY': '1', 'GPU_MAX_HW_QUEUES': '16'}, 'side')]
    for n, (name, extra, mode) in enumerate(variants):
        codes, rs = run(n, extra, mode)
        if codes != [0] * N or len(rs) != N:
            print(f'{name}: FAILED, exit codes {codes}')
            continue
        f = lambda k: stt.median([r[k] for r in rs if r[k] is not None]) if any(r[k] is not None for r in rs) else float('nan')
        print(f'{name}: all-to-all {f("a2a_alone"):.1f} ms alone, {f("a2a_during"):.1f} ms during the all-reduce '
              f'({min(r["n_during"] for r in rs)}-{max(r["n_during"] for r in rs)} of 14 per rank); all-reduce {f("ar_ms"):.1f} ms, '
              f'share of it beside an all-to-all {f("ar_with_a2a"):.2f}')


if __name__ == '__main__':
    main()
