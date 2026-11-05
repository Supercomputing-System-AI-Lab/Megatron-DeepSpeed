#!/usr/bin/env python3
"""test_coordinator.py (ft-sys-v2 heartbeat system tests, 2026-11-05): the coordinator against an in-memory store, on the
coordinator's own clock. Cases where two things happen at once (found on GPUs, job 5854211).
    python3 -I tests/test_coordinator.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from hb_coordinator import Coordinator

fails = []


class Store:
    def __init__(s):
        s.kv = {'coordinator': 'nX'}

    def range(s, p):
        return {k: v for k, v in s.kv.items() if k.startswith(p)}, 1

    def get(s, k):
        return s.kv.get(k), 1

    def put(s, k, v, lease=None):
        s.kv[k] = v
        return 1

    def create_if_absent(s, k, v, lease=None, fence=None):
        if k in s.kv or (fence and s.kv.get(fence[0]) != fence[1]):
            return False
        s.kv[k] = v
        return True


def beat(st, n, low, high, low_rank=0):
    st.kv[f'node/{n}'] = json.dumps({'node': n, 'run': 'r', 'low': low, 'low_rank': low_rank, 'high': high,
                                     'rank_ids': list(range(8)), 'dead': [], 'dead_why': {}, 'phases': ['forward'], 'role': 'train'})


def failed(st):
    return [json.loads(v) for k, v in sorted(st.kv.items()) if k.startswith('failed/')]


def check(name, ok, got):
    print(f'{"PASS" if ok else "FAIL"}  {name}: {got}')
    if not ok:
        fails.append(name)


def run(case):
    st = Store()
    for n in ('n0', 'n1', 'n2'):
        st.kv[f'members/{n}'] = json.dumps({'role': 'train'})
        beat(st, n, 100, 101)
    c = Coordinator(st, 'nX', tempfile.mkdtemp(), lambda m: None)
    c.step(0)
    for n in ('n0', 'n1', 'n2'):
        beat(st, n, 200, 201)
    c.step(5)
    del st.kv['node/n0']                                  # n0's lease ended (its agent died)
    if case == 'silent+hang':                             # at the same time rank 12 on n1 hangs: n1 is behind, n2 waits
        beat(st, 'n1', 230, 240, low_rank=12); beat(st, 'n2', 240, 240)
    elif case == 'lost, survivors differ':                # n0 really died; the survivors' counts differ a little (queued calls)
        beat(st, 'n1', 238, 240, low_rank=9); beat(st, 'n2', 240, 240)
    else:                                                 # n0 really died; the survivors wait at the same call
        beat(st, 'n1', 240, 240); beat(st, 'n2', 240, 240)
    for t in range(10, 60, 5):
        c.step(t)
        if c.incident and f'stacks/{c.incident["gen"]}/{c.incident["node"]}' not in st.kv:
            ans = 'confirmed' if case == 'silent+hang' else 'ambiguous'
            det = {'outside': [12], 'waiting': [8, 9]} if case == 'silent+hang' else {'outside': [], 'waiting': [8, 9]}
            st.kv[f'stacks/{c.incident["gen"]}/{c.incident["node"]}'] = json.dumps({'decision': ans, 'detail': det})
        if failed(st):
            return failed(st)[0], t
    return None, None


f, t = run('silent+hang')
check('agent dead on n0 + rank 12 hangs on n1: the hang, not n0', bool(f) and f['kind'] == 'hang' and f['node'] == 'n1' and f['suspect'] == 12,
      f and (f['kind'], f['node'], f['suspect'], t))
f, t = run('lost, survivors differ')
check('n0 lost, survivors differ: n0, after the stacks', bool(f) and f['kind'] == 'node lost' and f['node'] == 'n0', f and (f['kind'], f['node'], t))
f, t = run('lost, survivors equal')
check('n0 lost, survivors equal: n0, at once', bool(f) and f['kind'] == 'node lost' and f['node'] == 'n0' and t <= 20, f and (f['kind'], f['node'], t))


def run_new_coordinator():
    """The lost node held the coordinator role (login e2e, 11-05 06:06): the new coordinator never saw it beat; the ranks of the
    other node died of communication errors, so that node has no counts either."""
    st = Store()
    for n in ('n0', 'n1'):
        st.kv[f'members/{n}'] = json.dumps({'role': 'train'})
    st.kv['members/nS'] = json.dumps({'role': 'standby'})
    why = 'exception: RuntimeError: [gloo/transport/tcp/pair.cc:537] Read error: Connection reset by peer'
    st.kv['node/n1'] = json.dumps({'node': 'n1', 'run': 'r', 'low': None, 'low_rank': None, 'high': None, 'rank_ids': [0, 1],
                                   'dead': [0, 1], 'dead_why': {'0': why, '1': why}, 'phases': [], 'role': 'train'})
    st.kv['node/nS'] = json.dumps({'node': 'nS', 'run': None, 'low': None, 'high': None, 'rank_ids': [], 'dead': [],
                                   'dead_why': {}, 'phases': [], 'role': 'standby'})
    c = Coordinator(st, 'nX', tempfile.mkdtemp(), lambda m: None)
    for t in range(0, 30, 5):
        c.step(t)
        if failed(st):
            return failed(st)[0], t
    return None, None


f, t = run_new_coordinator()
check('lost node held the role; survivors died of communication errors: n0, at once',
      bool(f) and f['kind'] == 'node lost' and f['node'] == 'n0' and t == 0, f and (f['kind'], f['node'], t))


def run_death(why, reconfig=True):
    """One rank of n1 gone with this reason: the kind and the plan of the failure list (the user's rules of 11-05 11:50)."""
    import hb_coordinator as HC
    HC.OOM_RECONFIG = reconfig
    st = Store()
    for n in ('n0', 'n1'):
        st.kv[f'members/{n}'] = json.dumps({'role': 'train'})
        beat(st, n, 100, 101)
    b1 = json.loads(st.kv['node/n1']); b1.update(dead=[13], dead_why={'13': why}); st.kv['node/n1'] = json.dumps(b1)
    c = Coordinator(st, 'nX', tempfile.mkdtemp(), lambda m: None)
    c.step(0)
    HC.OOM_RECONFIG = True
    f = failed(st)
    return (f[0]['kind'], f[0]['plan']) if f else (None, None)


k, plan = run_death('exception: RuntimeError: ftv2 injected exit')
check('a program error: kind exception, stop for debugging', k == 'exception' and plan.startswith('stop the training for debugging'), (k, plan))
k, plan = run_death('exception: OutOfMemoryError: CUDA out of memory')
check('out of memory: the OOM ladder', k == 'out of memory' and 'OOM ladder' in plan and 'no node replaced' in plan, (k, plan))
k, plan = run_death('exception: OutOfMemoryError: CUDA out of memory', reconfig=False)
check('out of memory, re-configure off: stop', k == 'out of memory' and plan.startswith('stop the training: out of memory'), (k, plan))
k, plan = run_death('exception: RuntimeError: HIP error: uncorrectable ECC error encountered')
check('a hardware fault: replace the node', k == 'hardware fault' and 'no standby free' in plan, (k, plan))
k, plan = run_death('signal')
check('a death by a signal: crash, replace the node', k == 'crash' and 'no standby free' in plan, (k, plan))


def run_stuck():
    """Every rank still with equal counts past the first steps: stacks from every node, then kind stuck."""
    st = Store()
    for n in ('n0', 'n1'):
        st.kv[f'members/{n}'] = json.dumps({'role': 'train'})
    def beat5(n, c):
        st.kv[f'node/{n}'] = json.dumps({'node': n, 'run': 'r', 'low': c, 'low_rank': 0, 'high': c, 'rank_ids': list(range(8)),
                                         'dead': [], 'dead_why': {}, 'phases': ['train'], 'role': 'train', 'step': 5})
    c = Coordinator(st, 'nX', tempfile.mkdtemp(), lambda m: None)
    for n in ('n0', 'n1'):
        beat5(n, 100)
    c.step(0)
    for n in ('n0', 'n1'):
        beat5(n, 150)
    asked = None
    for t in range(5, 300, 5):
        c.step(t)
        req = [json.loads(v) for k, v in st.kv.items() if k.startswith('sample/')]
        if req and asked is None:
            asked = (t, req[0]['node'])
            for n in ('n0', 'n1'):                     # every node's agent answers
                st.kv[f'stacks/{c.incident["gen"]}/{n}'] = json.dumps({'decision': 'ambiguous', 'detail': {}})
        if failed(st):
            return asked, failed(st)[0], t
    return asked, None, None


asked, f, t = run_stuck()
check('stuck: stacks asked of every node after 180 s, then kind stuck',
      asked is not None and asked[1] == '*' and asked[0] >= 185 and f is not None and f['kind'] == 'stuck' and f['plan'].startswith('stop the training'),
      (asked, f and (f['kind'], f['plan'][:40]), t))
print(f'\n{len(fails)} failed' if fails else '\nall passed')
sys.exit(1 if fails else 0)
