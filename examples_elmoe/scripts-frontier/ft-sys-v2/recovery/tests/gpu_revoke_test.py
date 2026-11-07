#!/usr/bin/env python3
"""gpu_revoke_test.py (ft-sys-v2 recovery tests, 2026-11-07, the coordinator's item 3 after M3C12c 5872275): does ncclCommRevoke
block on communicators that reach a lost node over the network, and in which call? Two nodes, RCCL with the OFI plugin.

16 ranks: node 0 = ranks 0-7 (survivors), node 1 = ranks 8-15 (lost). Survivor r is paired with lost rank r + 8, as stage 0
replica 1 (ranks 8-15) was with node 5 in M3C12c. The groups a survivor holds, with the lost rank in each (created in this order
everywhere), mirror rank 8's stuck groups of M3C12c (the model-parallel, pipe and slice groups across the stages, the commit group):
  A1, A2, A3   {r, r + 8}, one all-reduce each at the start, idle afterwards (the pipeline-wide groups, used at a step's end)
  C            all 16 ranks, one batched send/receive round r <-> r + 8 at the start, idle afterwards (the commit group)
  Q            {r, r + 8}, used for the in-flight receive below (a pipeline pair to the lost node)
  H            {r, r ^ 1} on node 0 (healthy), used for the in-flight healthy receive below
The work in flight when node 1 dies, by survivor:
  ranks 0, 1   nothing (idle)
  ranks 2, 3   a receive on H from the other one, which never sends (a healthy peer that waits too: the GPU stays busy)
  ranks 4, 5   a receive on Q from r + 8, which dies before sending (in flight to the lost peer, over the network)
  ranks 6, 7   both
Node 1's eight ranks exit at once (os._exit) after the setup. Each survivor then revokes A1, A2, A3, C and Q (H holds no lost rank),
one thread each (rc_stop.revoke_all, limit FTV2_REVOKE_WAIT_S, default 10 s), and at the limit, if a revoke has not returned, takes
the stacks at once (rc_stop.dump_stuck: gdb's backtrace of every thread and py-spy --native) into <out>.stacks.<rank>.revoke.txt. No
abort is started while a revoke is inside RCCL (f5759aa5): the rank then just reports and exits.
Output per survivor: each revoke's return time (None = still inside), the variant, the stack file.
    python gpu_revoke_test.py --launch          (inside a two-node allocation; runs the srun step itself)"""
import datetime
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'heartbeat-sys'))
TO = datetime.timedelta(seconds=300)
VARIANT = {0: 'idle', 1: 'idle', 2: 'healthy recv', 3: 'healthy recv', 4: 'lost-peer recv', 5: 'lost-peer recv',
           6: 'both', 7: 'both'}


def log(rank, msg):
    print(f'[revoke_test rank {rank}] {time.strftime("%T")} {msg}', flush=True)


def worker(rank, host, port, out):
    import torch
    import torch.distributed as dist
    import rc_stop
    lr = int(os.environ.get('SLURM_LOCALID', rank % 8))
    torch.cuda.set_device(lr)
    dev = torch.device('cuda', lr)
    store = dist.TCPStore(host, port, None, is_master=False, timeout=TO)
    dist.init_process_group('nccl', store=dist.PrefixStore('init', store), rank=rank, world_size=16, timeout=TO, device_id=dev)
    me = rank % 8
    groups = {}
    for name in ('A1', 'A2', 'A3'):                                  # created by every rank in the same order
        for p in range(8):
            g = dist.new_group([p, p + 8])
            if me == p:
                groups[name] = g
    groups['C'] = dist.new_group(list(range(16)))
    for p in range(8):
        g = dist.new_group([p, p + 8])
        if me == p:
            groups['Q'] = g
    for p in range(0, 8, 2):
        g = dist.new_group([p, p + 1])
        if rank in (p, p + 1):
            groups['H'] = g
    # the setup: each idle group used once (connections over the network), the commit group's batched round
    for name in ('A1', 'A2', 'A3'):
        t = torch.ones(1 << 16, device=dev)
        dist.all_reduce(t, group=groups[name])
    peer = rank + 8 if rank < 8 else rank - 8
    a, b = torch.full((1 << 18,), float(rank), device=dev), torch.empty(1 << 18, device=dev)
    for w in dist.batch_isend_irecv([dist.P2POp(dist.isend, a, peer, group=groups['C']),
                                     dist.P2POp(dist.irecv, b, peer, group=groups['C'])]):
        w.wait()
    t = torch.ones(1 << 10, device=dev)
    dist.all_reduce(t, group=groups['Q'])                             # Q's communicator made while both sides live
    if rank < 8:
        dist.all_reduce(t, group=groups['H'])
    torch.cuda.synchronize(dev)
    store.set(f'ready/{rank}', '1')
    store.wait([f'ready/{r}' for r in range(16)])
    if rank >= 8:                                                     # node 1: every rank leaves at once
        log(rank, 'leaving (node 1 lost)')
        store.set(f'gone/{rank}', '1')
        os._exit(0)
    store.wait([f'gone/{r}' for r in range(8, 16)])
    res = {'rank': rank, 'variant': VARIANT[me]}
    v = VARIANT[me]
    if v in ('healthy recv', 'both'):                                  # a receive from the healthy partner, which never sends
        x = torch.zeros(1 << 16, device=dev)
        dist.broadcast(x, src=rank ^ 1, group=groups['H'], async_op=True)
    if v in ('lost-peer recv', 'both'):                                # a receive from the lost rank, which is gone
        y = torch.zeros(1 << 16, device=dev)
        dist.broadcast(y, src=rank + 8, group=groups['Q'], async_op=True)
    time.sleep(3)
    lost = [groups[n] for n in ('A1', 'A2', 'A3', 'C', 'Q')]
    names = ['A1', 'A2', 'A3', 'C', 'Q']
    lines = []
    r = rc_stop.revoke_all(dev, lambda m: lines.append(m), limit_s=float(os.environ.get('FTV2_REVOKE_WAIT_S', '10')),
                           groups=lost, dump_path=f'{out}.stacks.{rank}.txt')
    res['revoke'] = r
    res['groups'] = names
    res['log'] = lines
    log(rank, f'{v}: revoke returned_s {r.get("returned_s")} ({names}); alive {r.get("alive")}')
    json.dump(res, open(f'{out}.{rank}', 'w'), default=str)
    os._exit(0)


def launch():
    import torch.distributed as dist
    host = subprocess.check_output(['hostname', '-i']).decode().split()[0]
    port = 27300 + os.getpid() % 500
    master = dist.TCPStore(host, port, None, is_master=True, timeout=TO, wait_for_workers=False)
    out = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'gpu_revoke_{os.environ.get("SLURM_JOB_ID", os.getpid())}')
    me = os.path.abspath(__file__)
    pre = f'LD_PRELOAD="{os.environ.get("FTV2_TEST_PRELOAD", "")}" '
    w = subprocess.Popen(['srun', '--overlap', '-N2', '-n16', '--ntasks-per-node=8', '--gpus-per-node=8', '--cpus-per-task=7',
                          'bash', '-c', f'{pre}exec {sys.executable} {me} --worker $SLURM_PROCID {host} {port} {out}'])
    code = w.wait(timeout=1000)
    for r in range(8):
        f = f'{out}.{r}'
        if not os.path.exists(f):
            print(f'rank {r}: no result')
            continue
        d = json.load(open(f))
        rv = d['revoke']
        print(f'rank {r} ({d["variant"]}): returned_s {rv.get("returned_s")} for {d["groups"]}, alive {rv.get("alive")}')
        if os.path.exists(f'{out}.stacks.{r}.revoke.txt'):
            print(f'    stacks: {out}.stacks.{r}.revoke.txt')
    print(f'srun exit code {code}')
    del master


if __name__ == '__main__':
    if sys.argv[1] == '--worker':
        worker(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5])
    elif sys.argv[1] == '--launch':
        launch()
