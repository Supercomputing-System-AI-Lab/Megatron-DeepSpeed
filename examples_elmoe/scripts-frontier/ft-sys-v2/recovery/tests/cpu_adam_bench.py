#!/usr/bin/env python3
"""cpu_adam_bench.py (ft-sys-v2 recovery tests, 2026-11-06): the two CPU Adam choices of Keep on one host, before any GPU run.
For DeepSpeed's CPU Adam kernel (called in chunks, as rc_keep does) and torch.optim.AdamW(fused=True) on the CPU:
  - the time of one update of N fp32 values in a background thread;
  - how much the main thread's pure-Python work slows down meanwhile (the Python lock: DeepSpeed's binding keeps it while a
    chunk runs, csrc/adam/cpu_adam.cpp:10; torch's operators release it).
    OMP_NUM_THREADS=6 python tests/cpu_adam_bench.py [--n 50000000] [--threads 6]"""
import argparse
import os
import threading
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=50_000_000)
ap.add_argument('--threads', type=int, default=int(os.environ.get('OMP_NUM_THREADS', '6')))
ap.add_argument('--chunk', type=int, default=8 << 20)
a = ap.parse_args()
torch.set_num_threads(a.threads)


def python_rate(seconds):
    n, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        n += 1
    return n / seconds


def run(impl):
    p, g, m, v = (torch.randn(a.n) for _ in range(4))
    v.abs_()
    if impl == 'torch':
        opt = torch.optim.AdamW([p], lr=1e-4, weight_decay=0.1, fused=True)
        opt.state[p]['step'] = torch.tensor(0.0)
        opt.state[p]['exp_avg'], opt.state[p]['exp_avg_sq'] = m, v

        def upd():
            p.grad = g
            opt.step()
    else:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        opt = DeepSpeedCPUAdam([p], lr=1e-4, weight_decay=0.1)

        def upd():
            for s in range(0, a.n, a.chunk):
                e = min(a.n, s + a.chunk)
                opt.ds_opt_adam.adam_update(opt.opt_id, 1, 1e-4, 0.9, 0.95, 1e-8, 0.1, True, p[s:e], g[s:e], m[s:e], v[s:e])
    upd()                                    # warm
    t0 = time.time(); upd(); alone = time.time() - t0
    idle = python_rate(1.0)
    th = threading.Thread(target=lambda: [upd() for _ in range(3)])
    t0 = time.time(); th.start()
    busy = python_rate(min(2.0, 2.5 * alone))
    th.join()
    print(f'{impl:6s} n={a.n:,} threads={a.threads}: one update {alone:.3f} s ({a.n / alone / 1e9:.2f} G values/s); main-thread '
          f'python rate during updates {busy / idle * 100:.0f}% of idle', flush=True)


for impl in ('ds', 'torch'):
    run(impl)
