#!/usr/bin/env python3
"""cpu_cut_group_test.py (ft-sys-v2 recovery tests, milestone 3 C, 2026-11-06): the single list of operations a C stop cut on groups
without a lost rank (rc_c_flow.cut_entry, decide_cut; hb_monitor's tracking). CPU and gloo on the login node.

Part 1, pure decide_cut cases:
  a  M3C1e: two ranks parked in their reduction of micro-batch 5 on the stage's data-parallel group, two parked before it
     -> C; the first two redo, the others 'none'.
  b  the coordinator's case: one member's reduction completed and was committed, its peer's did not -> the member must take its
     commit back ('redo_last'): level B in the engine (not built), C with the actions when taking a commit back is available.
  c  an all-to-all still running inside a forward -> level B.
  d  a pipeline send parked in its instruction, the receiver parked before it -> C; the sender redoes, the receiver 'none'.
Part 2, four processes (gloo), the case of b run for real: world [0..3], G1 = [2, 3], G2 = [0, 2]; rank 1 is lost.
  micro-batch 0: rank 2 all-reduces on G1 and G2, rank 3 on G1, rank 0 on G2; each commits (its sum S += the results).
  micro-batch 1: rank 3 all-reduces on G1 with rank 2 and commits; rank 2 all-reduces on G1, then on G2 (asynchronous) and
  parks in its reduction; rank 0 parks before entering G2. The tracking must show G2 running on rank 2 only; the list must
  name G1 and G2 with rank 2 'redo', rank 3 'redo_last', rank 0 'none'; the production decision must be level B. Then each
  rank acts on the list as an engine that can take a commit back (from its last commit: rank 3's commit of micro-batch 1 taken
  back), G1 and G2 get new backends in their old objects, and every sum must equal the clean run's.
    python tests/cpu_cut_group_test.py   (the venv's python)"""
import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'heartbeat-sys'))
TO = datetime.timedelta(seconds=60)


def g(n, m, p, op, s):
    return {'n': n, 'm': m, 'p': p, 'op': op, 's': s}


def part1():
    import rc_c_flow as cf
    checks = {}
    # a: M3C1e
    e = {16: {'where': 'red', 'groups': {'G': g(5, 4, True, 'all_reduce', 5)}},
         17: {'where': 'red', 'groups': {'G': g(5, 4, True, 'all_reduce', 5)}},
         24: {'where': 'p2p', 'groups': {'G': g(4, 4, False, 'all_reduce', None)}},
         25: {'where': 'boundary', 'groups': {'G': g(4, 4, False, 'all_reduce', None)}}}
    d = cf.decide_cut(e)
    checks['a: mode C, cut 5'] = d['mode'] == 'C' and d['cuts'] == {'G': 5}
    checks['a: 16, 17 redo; 24, 25 none'] = [d['actions'][str(r)]['G'] for r in (16, 17, 24, 25)] == ['redo', 'redo', 'none', 'none']
    # b: Y (rank 2) completed G1 with X, its G2 is running; X (rank 3) committed G1; Z (rank 0) parked before G2
    e = {2: {'where': 'red', 'groups': {'G1': g(2, 1, False, 'all_reduce', 2), 'G2': g(2, 1, True, 'all_reduce', 2)}},
         3: {'where': 'call', 'groups': {'G1': g(2, 2, False, 'all_reduce', None)}},
         0: {'where': 'call', 'groups': {'G2': g(1, 1, False, 'all_reduce', None)}}}
    d = cf.decide_cut(e)
    checks['b: engine falls back to level B'] = d['mode'] == 'B'
    d = cf.decide_cut(e, can_redo_last=True)
    checks['b: with taking a commit back: C, G1 and G2 cut at 2'] = d['mode'] == 'C' and d['cuts'] == {'G1': 2, 'G2': 2}
    checks['b: actions'] = d['actions'] == {'2': {'G1': 'redo', 'G2': 'redo'}, '3': {'G1': 'redo_last'}, '0': {'G2': 'none'}}
    # c: an all-to-all inside a forward
    e = {16: {'where': 'boundary', 'groups': {'E': g(40, 30, True, 'all_to_all_single', None)}},
         17: {'where': 'boundary', 'groups': {'E': g(39, 30, False, 'all_to_all_single', None)}}}
    checks['c: an all-to-all in a forward -> level B'] = cf.decide_cut(e)['mode'] == 'B'
    # d: a pipeline send parked in its instruction
    e = {0: {'where': 'p2p', 'groups': {'P': g(7, 0, True, 'broadcast', 7)}},
         16: {'where': 'boundary', 'groups': {'P': g(6, 0, False, 'broadcast', None)}}}
    d = cf.decide_cut(e)
    checks['d: C, sender redo, receiver none'] = d['mode'] == 'C' and d['actions'] == {'0': {'P': 'redo'}, '16': {'P': 'none'}}
    return checks


class StoreEtcd:
    def __init__(self, store):
        self.s = store

    def put(self, k, v):
        self.s.set(k, v)

    def get(self, k):
        return (self.s.get(k).decode(), None) if self.s.check([k]) else (None, None)

    def range(self, prefix):
        out = {}
        for r in range(4):
            k = f'{prefix}{r}'
            if self.s.check([k]):
                out[k] = self.s.get(k).decode()
        return out, None


class Ctl:
    def __init__(self, rank, hb):
        self.rank, self.hb, self.c = rank, hb, {'aborted': set()}


def contrib(rank, mb, grp):
    return float(10 * (mb + 1) + rank + (100 if grp == 'G2' else 0))


def worker(rank, port, out):
    import torch
    import torch.distributed as dist
    import hb_monitor as hb
    import rc_c_flow as cf
    import rc_swap
    hb.install_counter()
    hb.S['c_track'] = True
    store = dist.TCPStore('127.0.0.1', port, None, is_master=False, timeout=TO)
    res = {'rank': rank, 'checks': {}}
    dist.init_process_group('gloo', store=dist.PrefixStore('g0', store), rank=rank, world_size=4, timeout=TO)
    G = {'G1': dist.new_group([2, 3]), 'G2': dist.new_group([0, 2])}
    mine = [k for k, rs in (('G1', [2, 3]), ('G2', [0, 2])) if rank in rs]
    S = {k: 0.0 for k in mine}                         # this rank's sums, per group
    snap = {}                                          # sums at each commit
    dist.all_reduce(torch.ones(1))                     # the world, completed
    if rank == 1:
        store.set('lost/1', '1')
        os._exit(0)

    def reduce_mb(mb, groups, async_last=False):
        out, works = {}, []
        for i, k in enumerate(groups):
            t = torch.full((4,), contrib(rank, mb, k))
            if async_last and i == len(groups) - 1:
                works.append(dist.all_reduce(t, group=G[k], async_op=True))
            else:
                dist.all_reduce(t, group=G[k])
            out[k] = t
        return out, works

    def commit(mb, vals):
        for k, t in vals.items():
            S[k] += float(t[0])
        snap[mb] = dict(S)
        hb.S['c_mark'] = dict(hb.S['c_n'])

    hb.S['c_mark'] = dict(hb.S['c_n'])
    commit(0, reduce_mb(0, mine)[0])                    # micro-batch 0: everyone, committed
    store.wait(['lost/1'])
    if rank == 3:                                      # micro-batch 1: G1 with rank 2, committed
        commit(1, reduce_mb(1, ['G1'])[0])
        hb.S['c_where'] = 'call'
    elif rank == 2:                                    # G1 completes, G2 runs (rank 0 never enters): parked in the reduction
        vals, works = reduce_mb(1, ['G1', 'G2'], async_last=True)
        hb.S['c_where'] = 'red'
    else:                                              # rank 0 parks before entering G2
        hb.S['c_where'] = 'call'
    e = StoreEtcd(store)
    ctl = Ctl(rank, hb)
    entry = cf.cut_entry(ctl, 2.0)
    keys = rc_swap.group_keys()
    name = {keys[id(G[k])]: k for k in mine}
    res['entry'] = {name.get(k, k): v for k, v in entry['groups'].items() if k in name}
    res['checks']['G2 running on rank 2 only'] = [k for k, v in res['entry'].items() if v['p']] == (['G2'] if rank == 2 else [])
    e.put(f'rc/cg/cut/{rank}', json.dumps(entry))
    if rank == 0:                                      # the lowest survivor decides (as single_list), for both engines
        while not all(store.check([f'rc/cg/cut/{r}']) for r in (0, 2, 3)):
            pass
        ents = {r: json.loads(store.get(f'rc/cg/cut/{r}').decode()) for r in (0, 2, 3)}
        store.set('rc/cg/prod', json.dumps(cf.decide_cut(ents)))
        store.set('rc/cg/cutlist', json.dumps(cf.decide_cut(ents, can_redo_last=True)))
    prod = json.loads(store.get('rc/cg/prod').decode())
    d = cf.wait_list(e, 'cg', 60)
    res['checks']['the production engine falls back to level B'] = prod['mode'] == 'B'
    act = {name[k]: a for k, a in d['actions'].get(str(rank), {}).items()}
    res['actions'] = act
    want = {2: {'G1': 'redo', 'G2': 'redo'}, 3: {'G1': 'redo_last'}, 0: {'G2': 'none'}}[rank]
    res['checks']['my actions from the one list'] = d['mode'] == 'C' and act == want
    n = rc_swap.swap_groups(dist.PrefixStore('g1', store), lost=[], device=torch.device('cpu'), backend='gloo', timeout=TO,
                            extra=list(d['cuts']))
    res['swapped'] = n
    # act on the list: back to the last commit before the cut (rank 3 takes its commit of micro-batch 1 back), then micro-batch 1
    if rank == 3 and act.get('G1') == 'redo_last':
        S.clear(); S.update(snap[0])
    if rank == 2:
        S.clear(); S.update(snap[0])
    commit(1, reduce_mb(1, mine)[0])
    clean = {k: sum(contrib(r, mb, k) for mb in (0, 1) for r in ({'G1': [2, 3], 'G2': [0, 2]}[k])) for k in mine}
    res['sums'], res['clean'] = S, clean
    res['checks']['sums equal the clean run'] = all(abs(S[k] - clean[k]) < 1e-9 for k in mine)
    json.dump(res, open(f'{out}.{rank}', 'w'))
    os._exit(0)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
        return
    fails = 0
    for name, ok in part1().items():
        print(f'{"PASS" if ok else "FAIL"}  part 1 {name}')
        fails += not ok
    import torch.distributed as dist
    port = 27900 + os.getpid() % 500
    master = dist.TCPStore('127.0.0.1', port, None, is_master=True, timeout=TO, wait_for_workers=False)
    out = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'cut_group_{os.getpid()}')
    env = {k: v for k, v in os.environ.items() if 'proxy' not in k.lower()}
    ps = [subprocess.Popen([sys.executable, os.path.abspath(__file__), '--worker', str(r), str(port), out], env=env)
          for r in range(4)]
    codes = [p.wait(timeout=240) for p in ps]
    for r in (0, 2, 3):
        f = f'{out}.{r}'
        if not os.path.exists(f):
            print(f'FAIL  part 2 rank {r}: no result')
            fails += 1
            continue
        res = json.load(open(f))
        for name, ok in res['checks'].items():
            print(f'{"PASS" if ok else "FAIL"}  part 2 rank {r}: {name}')
            fails += not ok
        print(f'      rank {r}: entry {res["entry"]}, actions {res["actions"]}, {res["swapped"]} group(s) swapped, '
              f'sums {res["sums"]} (clean {res["clean"]})')
        os.remove(f)
    print(f'exit codes {codes}')
    print(f'\n{fails} failed' if fails or any(codes) else '\nall passed')
    del master
    sys.exit(1 if fails or any(codes) else 0)


if __name__ == '__main__':
    main()
