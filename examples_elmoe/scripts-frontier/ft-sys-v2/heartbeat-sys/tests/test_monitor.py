#!/usr/bin/env python3
"""test_monitor.py (ft-sys-v2 heartbeat system tests, 2026-11-05): the call counter counts both spellings of a communication call,
torch.distributed.<name> and torch.distributed.distributed_c10d.<name> (DeepSpeed's all-gathers use the second,
deepspeed/comm/torch.py:227, 243, 251). One process, a gloo world of 1, on the login node.
    python3 -I tests/test_monitor.py   (with a python that has torch)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import hb_monitor as M

M.install_counter()
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29613')
dist.init_process_group('gloo', rank=0, world_size=1)
fails = []


def check(name, got, want):
    print(f'{"PASS" if got == want else "FAIL"}  {name}: got {got}, want {want}')
    if got != want:
        fails.append(name)


t = torch.ones(4)
c0 = M.S['calls']; dist.all_reduce(t); check('torch.distributed.all_reduce counts once', M.S['calls'] - c0, 1)
c0 = M.S['calls']; c10d.all_reduce(t); check('distributed_c10d.all_reduce counts once', M.S['calls'] - c0, 1)
c0 = M.S['calls']; c10d.broadcast(t, 0); check('distributed_c10d.broadcast counts once', M.S['calls'] - c0, 1)
out = [torch.zeros(4)]
c0 = M.S['calls']; c10d.all_gather(out, t); check('distributed_c10d.all_gather counts once', M.S['calls'] - c0, 1)
o = torch.zeros(4)
c0 = M.S['calls']
try:
    c10d.all_gather_into_tensor(o, t)
    check('distributed_c10d.all_gather_into_tensor counts (at least once)', M.S['calls'] - c0 >= 1, True)
except RuntimeError as e:                     # gloo may not have it; the count is still taken before the call
    check(f'distributed_c10d.all_gather_into_tensor counted before gloo refused it ({str(e)[:40]})', M.S['calls'] - c0 >= 1, True)
dist.destroy_process_group()
print(f'\n{len(fails)} failed' if fails else '\nall passed')
sys.exit(1 if fails else 0)
