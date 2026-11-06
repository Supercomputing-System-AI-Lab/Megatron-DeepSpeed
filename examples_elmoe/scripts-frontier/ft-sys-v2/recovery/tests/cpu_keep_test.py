#!/usr/bin/env python3
"""cpu_keep_test.py (ft-sys-v2 recovery tests, 2026-11-06): Keep, the restore and the decision on the login node, CPU and gloo.

Four processes (ranks 0-3, two "nodes" of two ranks, partner stride 2) each own two parameter groups of an inner optimizer
(torch AdamW, the role apex FusedAdam has on the GPUs), stepped the way ZeRO-1 steps it: one group at a time, through the real
adapter (adapters/deepspeed_zero.py, with a stand-in engine). Per mode of the shadow:
  1. Keep on for 4 steps: after each, every shadow is compared with its owner's fp32 values (largest difference).
  2. A recovery: rank 3 forgets its state (it plays the standby that takes its place), the survivors report their versions,
     decide() names the resume step, rank 1 (which keeps rank 3's copy) sends it, rank 3 restores it through put_state; every
     owner re-seeds its partner; 3 more steps.
  3. The final values of every rank against a reference run of 7 steps without the recovery.
Pass: mode gpu (the owner's own optimizer class on the shadow): every difference exactly 0. Modes cpu/torch and cpu/ds: the
differences are reported (CPU Adam rounds differently from the owner's kernel).
    python tests/cpu_keep_test.py [--modes gpu,cpu:torch,cpu:ds]   (with the venv's python; runs its own 4 processes)"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.dirname(HERE)
sys.path.insert(0, RC)

SIZES = [1000, 517]
STEPS, FAIL_AT = 7, 4


def grad_of(rank, step, gi, n):
    import torch
    g = torch.Generator().manual_seed(1000 * rank + 10 * step + gi)
    return torch.randn(n, generator=g) * (0.5 + 0.1 * gi)


class FakeZero:
    """The two attributes the adapter reads from DeepSpeed's ZeRO optimizer: .optimizer (the inner one) and update_lp_params()."""
    def __init__(self, inner):
        self.optimizer = inner

    def update_lp_params(self):
        pass

    def destroy(self):
        pass


class FakeEngine:
    def __init__(self, inner):
        self.optimizer = FakeZero(inner)
        self.global_steps = self.global_samples = self.micro_steps = self.skipped_steps = 0

    def destroy(self):
        pass


def build(rank):
    import torch
    ps = [torch.nn.Parameter(torch.arange(n, dtype=torch.float32) * 0.001 * (rank + 1) + gi) for gi, n in enumerate(SIZES)]
    inner = torch.optim.AdamW([{'params': [ps[0]], 'weight_decay': 0.1}, {'params': [ps[1]], 'weight_decay': 0.0}],
                              lr=1e-2, betas=(0.9, 0.95), eps=1e-8)
    return ps, inner


def zero_like_step(inner, ps, rank, step, lr):
    """ZeRO-1's step: each group's gradient set, then the inner optimizer called with that one group (stage_1_and_2.py:1839)."""
    full = inner.param_groups
    for g in full:
        g['lr'] = lr
    for gi, p in enumerate(ps):
        p.grad = grad_of(rank, step, gi, p.numel())
        inner.param_groups = [full[gi]]
        inner.step()
        inner.param_groups = full
        p.grad = None


def worker(rank, world, port, mode, out):
    import torch
    import torch.distributed as dist
    os.environ['FTV2_PARTNER_STRIDE'] = '2'
    shadow, _, impl = mode.partition(':')
    os.environ['FTV2_SHADOW'] = shadow
    os.environ['FTV2_CPU_ADAM'] = impl or 'torch'
    os.environ['FTV2_CPU_ADAM_CHUNK'] = '300'
    os.environ['FTV2_SHADOW_CHECK'] = '1'
    sys.path.insert(0, os.path.join(os.path.dirname(RC), 'heartbeat-sys'))
    import hb_monitor
    hb_monitor.install_counter()                    # as in training: every communication call goes through the counter
    from rc_keep import Keep
    from rc_controller import decide
    from adapters.deepspeed_zero import ZeroAdapter
    store = dist.TCPStore('127.0.0.1', port, world, is_master=(rank == 0))
    dist.init_process_group('gloo', store=store, rank=rank, world_size=world)
    dev = torch.device('cpu')
    log = lambda m: None
    res = {'rank': rank, 'mode': mode, 'checks': [], 'final_vs_ref': None}

    # the reference: 7 steps, no Keep
    ps_r, inner_r = build(rank)
    for s in range(STEPS):
        zero_like_step(inner_r, ps_r, rank, s, 1e-2 * (s + 1))
    ref = [p.detach().clone() for p in ps_r]

    ps, inner = build(rank)
    ad = ZeroAdapter(FakeEngine(inner))
    keep = Keep(ad, rank, world, dev, log, 0)
    keep.seed()
    keep.install()
    for s in range(FAIL_AT):
        zero_like_step(inner, ps, rank, s, 1e-2 * (s + 1))
        res['checks'] += [g['check']['max_abs'] for rec in keep.records[-1:] for g in rec['groups'].values() if 'check' in g]
    keep.drain(30)
    # every shadow against its owner now (the owner sends its fp32 values; the partner compares)
    for k, (gi, p) in enumerate(keep.slots):
        theirs = torch.empty(p.numel())
        keep._exchange([p.detach().clone()], [theirs])
        res['checks'].append(float((keep.shadow[k][0] - theirs).abs().max()))
    # the recovery: rank 3 is lost; the survivors report
    lost = [3]
    reports = [None] * world
    rep = keep.report()
    dist.all_gather_object(reports, rep)
    survivors = {r: reports[r] for r in range(world) if r not in lost}
    ok, s_resume, why, code = decide(survivors, lost, world, 2)
    res['decide'] = [ok, s_resume, why]
    if rank == 3:                                   # the standby: a fresh optimizer, its state from rank 1's shadow
        before = [p.detach().clone() for p in ps]
        ps, inner = build(rank)
        ad = ZeroAdapter(FakeEngine(inner))
        states = Keep.recv_state(ad.slots(), (rank + 2) % world, dev)
        ad.put_state(states)
        res['restored_vs_own'] = [float((p.detach() - b).abs().max()) for p, b in zip(ps, before)]
        res['restored_step'] = [st['step'] for st in states]
        old = None
    else:
        old = keep
        if (3 + 2) % world == rank:
            keep.send_shadow(3)
        kept = ad.take_state()
        ps, inner2 = build(rank)                    # a survivor rebuilds too, and restores its own kept tensors
        ad = ZeroAdapter(FakeEngine(inner2))
        ad.put_state(kept)
        inner = inner2
    keep = Keep(ad, rank, world, dev, log, s_resume, reuse=old)
    keep.seed()
    keep.install()
    for s in range(s_resume, STEPS):
        zero_like_step(inner, ps, rank, s, 1e-2 * (s + 1))
    keep.drain(30)
    res['final_vs_ref'] = [float((p.detach() - r).abs().max()) for p, r in zip(ps, ref)]
    res['steps_in_inner'] = [inner.state[p]['step'].item() for p in ps]
    dist.barrier()
    with open(f'{out}.{rank}', 'w') as fh:
        json.dump(res, fh)
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', default='gpu,cpu:torch,cpu:ds')
    ap.add_argument('--worker', nargs=5)
    a = ap.parse_args()
    if a.worker:
        rank, world, port, mode, out = a.worker
        worker(int(rank), int(world), int(port), mode, out)
        return
    fails = []
    for i, mode in enumerate(a.modes.split(',')):
        out = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'cpu_keep_{os.getpid()}_{i}')
        port = 24500 + (os.getpid() % 500) + i
        t0 = time.time()
        ps = [subprocess.Popen([sys.executable, os.path.abspath(__file__), '--worker', str(r), '4', str(port), mode, out])
              for r in range(4)]
        codes = [p.wait(timeout=600) for p in ps]
        rs = [json.load(open(f'{out}.{r}')) for r in range(4) if os.path.exists(f'{out}.{r}')]
        if codes != [0, 0, 0, 0] or len(rs) != 4:
            print(f'FAIL  {mode}: exit codes {codes}')
            fails.append(mode)
            continue
        chk = max(max(r['checks'] or [0]) for r in rs)
        fin = max(max(r['final_vs_ref']) for r in rs)
        r3 = rs[3]
        print(f'{mode:10s} decide {rs[0]["decide"]}; shadow vs owner max |diff| {chk:.3e}; restored rank 3 vs its own state '
              f'{max(r3["restored_vs_own"]):.3e} (steps {r3["restored_step"]}); final vs reference {fin:.3e}; '
              f'inner steps {rs[0]["steps_in_inner"]} ({time.time() - t0:.1f} s)')
        if mode == 'gpu' and not (chk == 0 and fin == 0 and max(r3['restored_vs_own']) == 0):
            print(f'FAIL  {mode}: not exact')
            fails.append(mode)
        if not rs[0]['decide'][0] or rs[0]['decide'][1] != FAIL_AT:
            print(f'FAIL  {mode}: decide {rs[0]["decide"]}')
            fails.append(mode)
        if rs[0]['steps_in_inner'] != [float(STEPS)] * 2:
            print(f'FAIL  {mode}: inner step count {rs[0]["steps_in_inner"]}')
            fails.append(mode)
    print(f'\n{len(fails)} failed' if fails else '\nall passed')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
