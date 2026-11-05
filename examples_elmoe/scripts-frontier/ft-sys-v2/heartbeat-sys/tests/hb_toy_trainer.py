#!/usr/bin/env python3
"""hb_toy_trainer.py (ft-sys-v2 heartbeat system tests, 2026-11-05): a small trainer whose communication has the shape of X-MoE's
(an all-to-all inside each node per micro-batch, an all-reduce across all ranks at the end of each step), to test the heartbeat
system end to end without the cost of starting X-MoE. GPU (backend nccl, i.e. RCCL) under srun, or CPU (gloo) on the login node.

Ranks come from RANK / WORLD_SIZE / LOCAL_RANK (or SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID); the rendezvous from MASTER_ADDR /
MASTER_PORT. --per-node sets the size of the all-to-all groups. Injections come from FTV2_INJECT (see hb_monitor)."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import hb_monitor

ap = argparse.ArgumentParser()
ap.add_argument('--backend', default='nccl', choices=['nccl', 'gloo'])
ap.add_argument('--steps', type=int, default=1000)
ap.add_argument('--micro-batches', type=int, default=8)
ap.add_argument('--per-node', type=int, default=8)
ap.add_argument('--compute-ms', type=float, default=40.0)
a = ap.parse_args()

hb_monitor.install_counter()                       # before any communication call
import torch
import torch.distributed as dist

env = os.environ
rank = int(env.get('RANK', env.get('SLURM_PROCID', 0)))
world = int(env.get('WORLD_SIZE', env.get('SLURM_NTASKS', 1)))
local = int(env.get('LOCAL_RANK', env.get('SLURM_LOCALID', 0)))
dev = torch.device('cpu')
if a.backend == 'nccl':
    torch.cuda.set_device(local)
    dev = torch.device('cuda', local)
dist.init_process_group(a.backend, rank=rank, world_size=world)
groups = [dist.new_group(list(range(s, s + a.per_node))) for s in range(0, world, a.per_node)]
node_group = groups[rank // a.per_node]
hb_monitor.start(rank, local)

x = torch.randn(a.per_node * 256, 1024, device=dev)
w = torch.randn(1024, 1024, device=dev)
for step in range(a.steps):
    t0 = time.time()
    for mb in range(a.micro_batches):
        hb_monitor.set_step(step, mb)
        hb_monitor.set_phase('forward')
        if a.backend == 'nccl':
            y = x @ w                                  # GPU work; the all-to-all below waits for it on the stream
        else:
            time.sleep(a.compute_ms / 1000.0)
            y = x
        out = torch.empty_like(y)
        dist.all_to_all_single(out, y, group=node_group)
        hb_monitor.set_phase('backward')
        dist.all_to_all_single(y, out, group=node_group)
    hb_monitor.set_phase('optimizer')
    g = y.sum(dim=0)
    dist.all_reduce(g)
    if a.backend == 'nccl':
        torch.cuda.synchronize()                       # once per step, as a trainer's logging does
    hb_monitor.set_step(step, a.micro_batches, time.time() - t0)
    if rank == 0 and step % 10 == 0:
        print(f'step {step} {time.time() - t0:.3f} s calls {hb_monitor.S["calls"]}', flush=True)
dist.destroy_process_group()
