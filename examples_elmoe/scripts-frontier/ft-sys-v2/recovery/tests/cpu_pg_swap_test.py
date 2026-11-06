#!/usr/bin/env python3
"""cpu_pg_swap_test.py (ft-sys-v2 recovery tests, milestone 3 C0, 2026-11-06): can an existing process-group OBJECT get a new
backend in place, so that every object that holds the group (the engine, ZeRO, the MoE layers, the pipeline's send groups,
Megatron's registry) keeps working without being re-pointed? (m3_map question 6, option 2.) CPU and gloo on the login node.

Why the plain call is not enough: ProcessGroup::setBackend (torch 2.13, ProcessGroup.hpp:912-935) keeps an existing backend of
the same BackendType ("if the backendType is already set then reuse it for this device"), so _register_backend(cpu, GLOO, new)
leaves the old backend in place. The variant tested here registers the new backend under a backend type the group does not use
yet (CUSTOM, then UNDEFINED for a second swap) and makes it the group's default (_set_default_backend).

Four processes (ranks 0-3); groups W (the world), G = [0, 1, 2, 3], H = [2, 3], held by a stand-in "engine" object.
  1. all-reduces on W, G and H through the held objects;
  2. rank 3 leaves (os._exit); a joiner process takes rank 3;
  3. every survivor makes a new gloo backend per group (store prefix g1/<name>) and swaps it into the SAME group object; the joiner
     makes the same backends (it has no old objects);
  4. all-reduces on W, G and H again, the survivors through dist.all_reduce(t, group=<held object>), as DeepSpeed calls them;
  5. a second swap (rank 2 leaves, a second joiner) through another unused type, to see whether the trick repeats.
Prints PASS/FAIL per check and the backend name each group reports after the swap (code that tests for 'nccl' by name sees it).
    python tests/cpu_pg_swap_test.py   (the venv's python)"""
import datetime
import json
import os
import subprocess
import sys
import time

TO = datetime.timedelta(seconds=60)


def new_backends(store, gen, rank, names_ranks):
    import torch.distributed as dist
    out = {}
    for name, ranks in names_ranks.items():
        if rank in ranks:
            out[name] = dist.ProcessGroupGloo(dist.PrefixStore(f'{gen}/{name}', store), ranks.index(rank), len(ranks), TO)
    return out


def swap(pg, backend, btype):
    import torch
    import torch.distributed as dist
    bt = getattr(dist.ProcessGroup.BackendType, btype)
    pg._register_backend(torch.device('cpu'), bt, backend)
    pg._set_default_backend(bt)


def worker(rank, port, role, out):
    import torch
    import torch.distributed as dist
    groups = {'W': [0, 1, 2, 3], 'G': [0, 1, 2, 3], 'H': [2, 3]}
    store = dist.TCPStore('127.0.0.1', port, None, is_master=False, timeout=TO)
    res = {'rank': rank, 'role': role, 'checks': {}}

    def check(name, ok):
        res['checks'][name] = bool(ok)

    def allreduce_all(engine, joiner_backends, tag, expect):
        for name, ranks in groups.items():
            if rank not in ranks:
                continue
            t = torch.full((4,), float(rank + 1))
            if engine is not None:
                dist.all_reduce(t, group=engine[name] if name != 'W' else None)
            else:
                joiner_backends[name].allreduce([t]).wait()
            want = float(sum(r + 1 for r in ranks)) if expect is None else expect[name]
            check(f'{tag} {name} sum {want}', torch.all(t == want).item())

    if role == 'first':
        dist.init_process_group('gloo', store=dist.PrefixStore('g0/init', store), rank=rank, world_size=4, timeout=TO)
        engine = {'G': dist.new_group(groups['G']), 'H': dist.new_group(groups['H'])}
        engine['W'] = dist.distributed_c10d._get_default_group()
        allreduce_all(engine, None, 'before', None)
        if rank == 3:
            json.dump(res, open(f'{out}.{rank}.first', 'w'))
            os._exit(0)                                   # rank 3 is lost
        # survivors: swap a new backend into each held object (generation g1, type CUSTOM)
        nb = new_backends(store, 'g1', rank, groups)
        for name, b in nb.items():
            swap(engine[name], b, 'CUSTOM')
        res['names_after_swap1'] = {n: dist.get_backend(engine[n] if n != 'W' else None) for n in nb}
        allreduce_all(engine, None, 'after swap 1', None)
        if rank == 2:
            json.dump(res, open(f'{out}.{rank}.first', 'w'))
            os._exit(0)                                   # then rank 2 is lost
        nb = new_backends(store, 'g2', rank, groups)
        for name, b in nb.items():
            swap(engine[name], b, 'UNDEFINED')
        res['names_after_swap2'] = {n: dist.get_backend(engine[n] if n != 'W' else None) for n in nb}
        allreduce_all(engine, None, 'after swap 2', None)
        json.dump(res, open(f'{out}.{rank}.first', 'w'))
    else:                                                 # a joiner: its own backends of generation g1 (rank 3) or g2 (rank 2)
        gen = role
        jb = new_backends(store, gen, rank, groups)
        allreduce_all(None, jb, f'joiner {gen}', None)
        if gen == 'g1' and rank == 3:                     # the joiner of rank 3 stays for the second swap
            nb = new_backends(store, 'g2', rank, groups)
            allreduce_all(None, nb, 'joiner g1 after swap 2', None)
        json.dump(res, open(f'{out}.{rank}.{gen}', 'w'))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
        return
    import torch.distributed as dist
    port = 26100 + os.getpid() % 800
    master = dist.TCPStore('127.0.0.1', port, None, is_master=True, timeout=TO, wait_for_workers=False)
    out = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'pg_swap_{os.getpid()}')
    env = {k: v for k, v in os.environ.items() if 'proxy' not in k.lower()}
    run = lambda r, role: subprocess.Popen([sys.executable, os.path.abspath(__file__), '--worker', str(r), str(port), role, out],
                                           env=env)
    ps = [run(r, 'first') for r in range(4)]
    ps[3].wait(timeout=120)                               # rank 3 gone: its joiner starts
    j1 = run(3, 'g1')
    ps[2].wait(timeout=120)                               # rank 2 gone after swap 1: its joiner starts
    j2 = run(2, 'g2')
    codes = [p.wait(timeout=180) for p in ps] + [j1.wait(timeout=180), j2.wait(timeout=180)]
    fails = []
    for f in sorted(f for f in os.listdir(os.path.dirname(out)) if f.startswith(os.path.basename(out))):
        r = json.load(open(os.path.join(os.path.dirname(out), f)))
        for name, ok in r['checks'].items():
            print(f'{"PASS" if ok else "FAIL"}  rank {r["rank"]} ({r["role"]}): {name}')
            if not ok:
                fails.append((r['rank'], name))
        for k in ('names_after_swap1', 'names_after_swap2'):
            if k in r:
                print(f'      rank {r["rank"]}: backend names {k}: {r[k]}')
        os.remove(os.path.join(os.path.dirname(out), f))
    print(f'exit codes {codes}')
    print(f'\n{len(fails)} failed' if fails or any(codes) else '\nall passed')
    del master
    sys.exit(1 if fails or any(codes) else 0)


if __name__ == '__main__':
    main()
