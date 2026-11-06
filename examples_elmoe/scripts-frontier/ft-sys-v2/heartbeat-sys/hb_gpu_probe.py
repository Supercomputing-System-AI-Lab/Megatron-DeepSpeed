#!/usr/bin/env python3
"""hb_gpu_probe.py (ft-sys-v2 heartbeat system, 2026-11-06): does one GPU still compute? Run by the node's agent in a process of
its own, one per GPU, with only that GPU visible (HIP_VISIBLE_DEVICES), under the agent's timeout.

The question (V3-2L 5863006, frontier02408): two ranks stopped inside the GPU driver, could not be stack-dumped or killed, and the
heartbeat waited for evidence that never came. A stuck GPU makes a fresh process's first GPU call hang as well; a healthy GPU
answers in a second. The probe is the gate's per-GCD check made small (gate/gate_gemm.py: bf16 matmul timed on the GPU), so it
fits in seconds next to a stalled training process: a 2048^3 bf16 matmul, 5 times, synchronised, checked finite against the
same product computed twice.

Output, one line each, so the agent can tell a slow import (not a GPU fault) from a GPU that does not answer:
    imported <s>                  torch is loaded (no GPU touched yet)
    devices <n>                   the GPUs this process sees (0: the probe is unavailable here, not a fault)
    ok <ms>                       the matmuls finished and agree
    fail <reason>                 the GPU answered with an error or a wrong result
    python3 hb_gpu_probe.py       (env HIP_VISIBLE_DEVICES=<gpu>)"""
import time

t0 = time.time()
import torch                                     # noqa: E402

print(f'imported {time.time() - t0:.1f}', flush=True)
try:
    n = torch.cuda.device_count()
except Exception as e:                           # no runtime at all: unavailable, not a fault
    print('devices 0', flush=True)
    print(f'fail no device runtime: {e!r}'[:200], flush=True)
    raise SystemExit(0)
print(f'devices {n}', flush=True)
if n == 0:
    raise SystemExit(0)
try:
    t1 = time.time()
    a = torch.randn(2048, 2048, device='cuda:0', dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device='cuda:0', dtype=torch.bfloat16)
    ref = a @ b
    for _ in range(5):
        c = a @ b
    torch.cuda.synchronize()
    good = bool(torch.isfinite(c).all().item()) and bool(torch.equal(c, ref))
    print(f'ok {1000 * (time.time() - t1):.0f}' if good else 'fail the product is not finite or not repeatable', flush=True)
except Exception as e:
    print(f'fail {type(e).__name__}: {e}'[:300], flush=True)
