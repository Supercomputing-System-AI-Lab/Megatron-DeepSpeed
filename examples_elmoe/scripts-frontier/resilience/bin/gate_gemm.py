#!/usr/bin/env python3
"""gate_gemm.py: bf16 8192^3 matmul on all 8 GCDs at once, ONE PROCESS PER GCD, timed with CUDA events (GPU time).

WHAT IT MEASURES: for every GCD of the node, torch.matmul of two 8192 x 8192 bf16 matrices, 3 warm-ups and then 20 timed calls
between two CUDA events; the value printed per GCD is 2 x 8192^3 x 20 / (elapsed seconds) in TFLOP/s. The 8 children start their
timed loops together at a barrier, because the GCDs share power and memory clocks and a lone GCD would read too high.

History. v1 used 8 threads in one process: on all 18 nodes of H16 5641684 exactly one GCD per node read 71-77 TFLOP/s against
104-117 for the others (a GIL/launch artefact, not the hardware). v2 started 8 python processes, each importing torch from Lustre,
and waited for a FIXED start time 75 s later: on 36 nodes (H32 5641726) the imports took longer than the fixed budget and the test
timed out on all 288 GCDs. v3 (2026-10-09 14:30): the parent imports torch ONCE (without touching the GPU) and forks the 8 children
(one per GCD), so the import is paid once per node; the children start the timed loop together at a barrier, not at a fixed clock
time. v4 (2026-10-09 late evening): a GCD whose child is still alive at the deadline prints 'gcd<k>\ttimeout' (gate_node.sh then
writes the row with status timeout, which the judge's majority rule can act on), and 'gcd<k>\terror' only when the child died
without a result.

Output lines: 'import_s\t<s>', 'gcd<k>\t<TFLOP/s>' (or 'gcd<k>\ttimeout' / 'gcd<k>\terror' / 'gcd<k>\tmissing'), 'total_s\t<s>'.
The deadline GATE_GEMM_DEADLINE_S (seconds, set by gate_node.sh from the calibrated run time) bounds the wait for the children.

FAULT INJECTION (TEST ONLY), through XMOE_GATE_INJECT (items joined by '+'; the restart loop resolves '@<k>' to a node name):
  gemm_slow:<node>   the child of GCD 7 on that node sleeps past the deadline, so the row reads 'timeout' (the majority rule's case)
  gemm_half:<node>   the child of GCD 7 on that node sleeps one matmul time after every timed matmul, so its GPU is idle half of the
                     measured window and it reads about half speed: 'below threshold', the DEGRADED case of the two-tier exclusion."""
import multiprocessing as mp
import os
import socket
import sys
import time

NG, M, WARM, ITERS = 8, 8192, 3, 20
INJECT_GCD = 7      # the GCD the two injections act on


def injections():
    """The injection cases that name THIS node, from XMOE_GATE_INJECT ('case:node' items joined by '+' or ',')."""
    node = socket.gethostname().split('.')[0]
    out = set()
    for item in os.environ.get('XMOE_GATE_INJECT', '').replace(',', '+').split('+'):
        if ':' in item:
            c, n = item.split(':', 1)
            if n.strip() == node:
                out.add(c.strip())
    return out


def one(k, barrier, q, inj, deadline_s):
    try:
        import torch
        if k >= torch.cuda.device_count():
            q.put((k, 'missing')); barrier.abort(); return
        torch.cuda.set_device(k)
        a = torch.randn(M, M, dtype=torch.bfloat16, device=k)
        b = torch.randn(M, M, dtype=torch.bfloat16, device=k)
        for _ in range(WARM):
            torch.matmul(a, b)
        torch.cuda.synchronize(k)
        # TEST ONLY: the 'gemm_half' injection needs the time of one matmul, measured here before the barrier
        one_matmul_s = 0.0
        if k == INJECT_GCD and 'gemm_half' in inj:
            w0, w1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            w0.record(); torch.matmul(a, b); w1.record(); torch.cuda.synchronize(k)
            one_matmul_s = w0.elapsed_time(w1) / 1000.0
        try:
            barrier.wait(timeout=120)       # the 8 GCDs start the timed loop together (shared power and memory clocks)
        except Exception:
            pass
        if k == INJECT_GCD and 'gemm_slow' in inj:
            # TEST ONLY: this GCD never answers before the deadline (the parent prints 'gcd7 timeout')
            print('# gcd%d: XMOE_GATE_INJECT gemm_slow: sleeping past the deadline' % k, file=sys.stderr, flush=True)
            time.sleep(deadline_s + 120)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(ITERS):
            torch.matmul(a, b)
            if one_matmul_s > 0:
                # TEST ONLY (gemm_half): let the GPU drain, then idle for one matmul time: the event window doubles, the rate halves
                torch.cuda.synchronize(k)
                time.sleep(one_matmul_s)
        e1.record()
        torch.cuda.synchronize(k)
        dt = e0.elapsed_time(e1) / 1000.0
        q.put((k, '%.1f' % (2.0 * M ** 3 * ITERS / dt / 1e12)))
    except Exception as e:
        print('# gcd%d: %s' % (k, e), file=sys.stderr, flush=True)
        q.put((k, 'error'))


if __name__ == '__main__':
    t0 = time.time()
    import torch  # noqa: F401  (imported once, before the fork; the GPU is not touched in the parent)
    print('import_s\t%.1f' % (time.time() - t0), flush=True)
    deadline_s = float(os.environ.get('GATE_GEMM_DEADLINE_S', '240'))
    deadline = t0 + deadline_s
    inj = injections()
    if inj:
        print('# XMOE_GATE_INJECT cases for this node: %s' % ','.join(sorted(inj)), flush=True)
    ctx = mp.get_context('fork')
    barrier, q = ctx.Barrier(NG), ctx.Queue()
    procs = [ctx.Process(target=one, args=(k, barrier, q, inj, deadline_s)) for k in range(NG)]
    for p in procs:
        p.start()
    out = {}
    # collect one result per GCD until the deadline; a child that is still alive at the deadline is a 'timeout', a dead one an 'error'
    while len(out) < NG and time.time() < deadline:
        try:
            k, v = q.get(timeout=max(0.1, min(5.0, deadline - time.time())))
            out[k] = v
            print('gcd%d\t%s' % (k, v), flush=True)
        except Exception:
            if not any(p.is_alive() for p in procs) and q.empty():
                break
    for k in range(NG):
        if k not in out:
            print('gcd%d\t%s' % (k, 'timeout' if procs[k].is_alive() else 'error'), flush=True)
    for p in procs:
        if p.is_alive():
            p.kill()
    print('total_s\t%.1f' % (time.time() - t0), flush=True)
