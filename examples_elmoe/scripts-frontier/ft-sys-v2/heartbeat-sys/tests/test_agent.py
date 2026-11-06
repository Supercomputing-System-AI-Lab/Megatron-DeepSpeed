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
# an exception that ended a non-main thread is evidence, not a death (the user, 11-05: decision 6)
from hb_beat import thread_path
os.makedirs(d, exist_ok=True)
write_json(thread_path(d, 7), {'pid': live, 'thread': 'uploader', 'reason': 'ConnectionError: timed out'})
check('a thread exception alone: not gone', hb_agent.Agent.gone(me, [{'rank': 7, 'pid': live}]), {})


# 2026-11-06, the real fault of V3-2L 5863006: a rank whose main thread is in state D (here through a vfork whose child pauses, the
# RF1 test's helper ftv2_dhang.c): the blocking py-spy dump gets no stack, the evidence reads state D, the non-blocking dump still
# reads the stack, the GPU probe is unavailable on a login node (not a fault), and the replaced node's SIGKILL ends the rank and
# its child.
import shutil as _sh
import time as _t
from hb_beat import rank_path
here = os.path.dirname(os.path.abspath(__file__))
so = os.path.join(tempfile.mkdtemp(prefix='ftv2_dhang_'), 'ftv2_dhang.so')
pyspy = _sh.which('py-spy') or os.path.expanduser('~/.local/bin/py-spy')
if subprocess.run(['gcc', '-O2', '-shared', '-fPIC', '-o', so, os.path.join(here, 'ftv2_dhang.c')]).returncode != 0 or \
        not os.path.exists(pyspy):
    print('SKIP  the D-state cases: no gcc or no py-spy here')
else:
    code = ('import ctypes,threading,time\n'
            'threading.Thread(target=lambda: [time.sleep(1) for _ in iter(int, 1)], daemon=True).start()\n'
            f'ctypes.CDLL({so!r}).ftv2_vfork_hang()\n')
    v = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _t.sleep(2)
    d2 = tempfile.mkdtemp(prefix='ftv2_test_agent_d_')
    write_json(rank_path(d2, 20), {'rank': 20, 'pid': v.pid, 'local_rank': 4, 'run': 'job.rf1', 't': _t.time()})
    logs = []
    ag = types.SimpleNamespace(dir=d2, fenced={}, node='nT', log=logs.append, a=types.SimpleNamespace(gpus=8, pyspy=pyspy))
    ag.read_lines = lambda include_fenced=False: hb_agent.Agent.read_lines(ag, include_fenced)
    t0 = _t.time()
    first = hb_agent.Agent.dump_all(ag, ag.read_lines(), pyspy)
    import hb_stacks
    check('D state: the blocking dump gets no stack', hb_stacks.main_thread_frames(first.get(20) or '') is None, True)
    os.environ['FTV2_PROBE_IMPORT_S'] = '60'
    ev = hb_agent.Agent.evidence(ag, 'g1', ag.read_lines(), pyspy)
    e = ev.get(20, {})
    check('D state: /proc reads it', str(e.get('state', '')).startswith('D'), True)
    check('D state: the non-blocking dump still reads the main thread', e.get('nonblocking') in ('outside', 'waiting'), True)
    check('no GPU here: the probe is unavailable, not a fault', e.get('probe'), 'unavailable')
    import hb_rules
    check('the coordinator would call it a hardware fault', hb_rules.evidence_verdict(ev)[0], True)
    kids = hb_agent.children(v.pid)
    hb_agent.Agent.end_my_ranks(ag, {'run': 'job.rf1', 'gen': '0001', 't': _t.time()})
    _t.sleep(1)
    v.poll()
    check('fencing: the D-state rank ended by one SIGKILL', hb_agent.alive(v.pid), False)
    check('fencing: its paused child ended too', [k for k in kids if hb_agent.alive(k)], [])
    check('fencing: nothing left behind', ag.fenced.get('job.rf1'), {})
    print('      evidence timeline:', ' | '.join(m.split('] ', 1)[-1][:110] for m in logs if 'evidence' in m or 'fenced' in m))
    # RF2's synthetic leftover: the line says test_kill_noop, so the agent does not kill it and goes on; cleanup at run_done
    v2 = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _t.sleep(2)
    write_json(rank_path(d2, 20), {'rank': 20, 'pid': v2.pid, 'local_rank': 4, 'run': 'job.rf2', 't': _t.time(),
                                   'test_kill_noop': True})
    hb_agent.Agent.end_my_ranks(ag, {'run': 'job.rf2', 'gen': '0002', 't': _t.time()})
    check('RF2: the agent does not wait for a rank that survives: left behind, fenced', list(ag.fenced.get('job.rf2', {})), [v2.pid])
    check('RF2: its lines leave the beat', [ln['rank'] for ln in ag.read_lines()], [])
    ag.etcd = types.SimpleNamespace(get=lambda k: ('{"t": 1}' if k == 'run_done/job.rf2' else None, 1))   # run_done/<FTV2_RUN>
    hb_agent.Agent.end_leftovers(ag)
    _t.sleep(1)
    v2.poll()
    check('RF2: the leftover is ended once the run is done (cleanup)', hb_agent.alive(v2.pid), False)
    for pp in (v, v2):
        try:
            pp.kill()
        except Exception:
            pass
    _sh.rmtree(d2, ignore_errors=True)
print(f'\n{len(fails)} failed' if fails else '\nall passed')
sys.exit(1 if fails else 0)
