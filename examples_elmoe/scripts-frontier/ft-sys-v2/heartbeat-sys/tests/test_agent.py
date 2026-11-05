#!/usr/bin/env python3
"""test_agent.py (ft-sys-v2 heartbeat system tests, 2026-11-05): the agent's reading of a gone rank (hb_agent.Agent.gone), with
real processes and exit records in a temporary folder, no etcd. The case HB3 5855079 found: an exception record while the process
still lives (its exit blocked in PMPI_Finalize) is a death now.
    python3 -I tests/test_agent.py"""
import os
import subprocess
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import hb_agent
from hb_beat import exit_path, write_json

fails = []


def check(name, got, want):
    ok = got == want
    print(f'{"PASS" if ok else "FAIL"}  {name}: got {got!r}, want {want!r}')
    if not ok:
        fails.append(name)


d = tempfile.mkdtemp(prefix='ftv2_test_agent_')
me = types.SimpleNamespace(dir=d)
live = os.getpid()
p = subprocess.Popen(['true'])
p.wait()
dead = p.pid                                       # ended and reaped: no /proc entry
oom = 'exception: OutOfMemoryError: HIP out of memory. Tried to allocate 1862.65 GiB'


def gone(rank, pid, record):
    path = exit_path(d, rank)
    if os.path.exists(path):
        os.remove(path)
    if record:
        write_json(path, dict(record, rank=rank))
    return hb_agent.Agent.gone(me, [{'rank': rank, 'pid': pid}])


check('alive, exception record: gone (HB3)', gone(21, live, {'pid': live, 'reason': oom}), {21: oom})
check('alive, finished record: not gone', gone(21, live, {'pid': live, 'reason': 'finished'}), {})
check('alive, no record: not gone', gone(21, live, None), {})
check('alive, exception record of an older process: not gone', gone(21, live, {'pid': dead, 'reason': oom}), {})
check('dead, no record: signal', gone(5, dead, None), {5: 'signal'})
check('dead, exception record: its reason', gone(5, dead, {'pid': dead, 'reason': oom}), {5: oom})
check('dead, finished record: not gone', gone(5, dead, {'pid': dead, 'reason': 'finished'}), {})
for f in os.listdir(d):
    os.remove(os.path.join(d, f))
os.rmdir(d)
print(f'\n{len(fails)} failed' if fails else '\nall passed')
sys.exit(1 if fails else 0)
