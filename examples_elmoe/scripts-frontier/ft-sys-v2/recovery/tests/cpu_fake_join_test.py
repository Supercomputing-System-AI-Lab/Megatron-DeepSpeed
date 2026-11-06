#!/usr/bin/env python3
"""cpu_fake_join_test.py (ft-sys-v2 recovery tests, milestone 3 C, 2026-11-06): can a joiner build its process groups without any
partner (its model and optimizer build runs collectives that the survivors, who keep training, will not join), and then be
connected to the survivors' groups, kept in place? CPU and gloo on the login node.

The joiner initializes with torch's "fake" backend (every collective completes at once, no peer needed), creates the same groups
as the survivors did at their start, then gets a real backend per group (rc_swap.swap_groups, keyed by the group's ranks and its
creation order among groups with the same ranks). The survivors swap a new backend into each of their group objects that holds the
lost rank, with the same keys; groups without the lost rank keep their old backend.
Four processes: world [0..3], A = [0, 1, 2, 3] (a second group with the world's ranks: the occurrence order matters), B = [2, 3],
C = [0, 1] (no lost rank: kept). Rank 3 is lost; the joiner takes rank 3. Checks: all-reduces on every group the rank belongs to,
the survivors through dist.all_reduce(group=<held object>), the joiner through its own group objects; C still on its old backend.
    python tests/cpu_fake_join_test.py   (the venv's python)"""
import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
TO = datetime.timedelta(seconds=20)
GROUPS = [('A', [0, 1, 2, 3]), ('B', [2, 3]), ('C', [0, 1])]


def make_groups(dist):
    held = {'W': None}
    for name, ranks in GROUPS:
        held[name] = dist.new_group(ranks)
    return held


def check_all(dist, torch, rank, held, res, tag):
    for name, ranks in [('W', [0, 1, 2, 3])] + GROUPS:
        if rank not in ranks:
            continue
        t = torch.full((4,), float(rank + 1))
        dist.all_reduce(t, group=held[name])
        want = float(sum(r + 1 for r in ranks))
        res['checks'][f'{tag} {name} sum {want}'] = bool(torch.all(t == want).item())


def worker(rank, port, role, out):
    import torch
    import torch.distributed as dist
    import rc_swap
    store = dist.TCPStore('127.0.0.1', port, None, is_master=False, timeout=TO)
    res = {'rank': rank, 'role': role, 'checks': {}}
    if role == 'first':
        dist.init_process_group('gloo', store=dist.PrefixStore('g0', store), rank=rank, world_size=4, timeout=TO)
        held = make_groups(dist)
        check_all(dist, torch, rank, held, res, 'before')
        if rank == 3:
            store.set('lost/3', '1')
            json.dump(res, open(f'{out}.{rank}.first', 'w'))
            os._exit(0)
        store.wait(['lost/3'])
        in_c = rank in dict(GROUPS)['C']
        old_c = held['C']._get_backend(torch.device('cpu')) if in_c else None
        n = rc_swap.swap_groups(dist.PrefixStore('g1', store), lost=[3], device=torch.device('cpu'), backend='gloo',
                                timeout=TO)
        res['swapped'] = n
        if in_c:
            res['checks']['C kept its old backend'] = held['C']._get_backend(torch.device('cpu')) is old_c
        check_all(dist, torch, rank, held, res, 'after')
    else:
        from torch.testing._internal.distributed.fake_pg import FakeStore
        dist.init_process_group('fake', store=FakeStore(), rank=rank, world_size=4)
        held = make_groups(dist)                       # the same creation order as at the survivors' start; no peer needed
        t = torch.ones(2)
        dist.all_reduce(t, group=held['A'])            # a "build" collective: completes alone
        n = rc_swap.swap_groups(dist.PrefixStore('g1', store), lost=[3], device=torch.device('cpu'), backend='gloo',
                                timeout=TO, joiner=True)
        res['swapped'] = n
        check_all(dist, torch, rank, held, res, 'joiner')
    json.dump(res, open(f'{out}.{rank}.{role}', 'w'))
    os._exit(0)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
        return
    import torch.distributed as dist
    port = 27400 + os.getpid() % 500
    master = dist.TCPStore('127.0.0.1', port, None, is_master=True, timeout=TO, wait_for_workers=False)
    out = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'fake_join_{os.getpid()}')
    env = {k: v for k, v in os.environ.items() if 'proxy' not in k.lower()}
    run = lambda r, role: subprocess.Popen([sys.executable, os.path.abspath(__file__), '--worker', str(r), str(port), role, out],
                                           env=env)
    ps = [run(r, 'first') for r in range(4)]
    ps[3].wait(timeout=120)
    j = run(3, 'joiner')
    codes = [p.wait(timeout=180) for p in ps] + [j.wait(timeout=180)]
    fails = 0
    for f in sorted(x for x in os.listdir(os.path.dirname(out)) if x.startswith(os.path.basename(out))):
        r = json.load(open(os.path.join(os.path.dirname(out), f)))
        for name, ok in r['checks'].items():
            print(f'{"PASS" if ok else "FAIL"}  rank {r["rank"]} ({r["role"]}): {name}')
            fails += not ok
        if 'swapped' in r:
            print(f'      rank {r["rank"]} ({r["role"]}): {r["swapped"]} group(s) got a new backend')
        os.remove(os.path.join(os.path.dirname(out), f))
    print(f'exit codes {codes}')
    print(f'\n{fails} failed' if fails or any(codes) else '\nall passed')
    del master
    sys.exit(1 if fails or any(codes) else 0)


if __name__ == '__main__':
    main()
