#!/usr/bin/env python3
"""gpu_pg_swap_test.py (ft-sys-v2 recovery tests, milestone 3 C0 on one node, 2026-11-06): the in-place backend swap of
cpu_pg_swap_test.py with RCCL. Can a survivor keep its process-group OBJECTS after a rank is lost, by aborting the old RCCL
backend and registering a new RCCL backend (same members, the joiner at the lost rank's number) under a backend type the group
does not use yet (CUSTOM), made the default? Every object that holds the group keeps working without being re-pointed.

One node, 8 GPUs: 8 worker processes (ranks 0-7) and, after rank 7 leaves, a joiner process on GPU 7. Groups: W (the world),
G = [0..7], H = [6, 7], held by a stand-in engine. Checks:
  1. all-reduces on W, G, H; a point-to-point send 6 -> 7 on W (the pipeline's sends use the world group);
  2. rank 7 leaves (os._exit) while ranks 0-6 are inside an all-reduce on G (async, then waited in a thread): the survivors'
     old RCCL backends of W, G, H are aborted (Backend.abort(), one thread each, 30 s limit), which must end the cut all-reduce;
  3. survivors: a new ProcessGroupNCCL per group (store prefix g1/<name>) swapped into the SAME object (CUSTOM, default);
     the joiner makes the same backends;
  4. all-reduces on W, G, H and the 6 -> 7 send again, the survivors through dist.all_reduce(t, group=<held object>).
Launch (inside an allocation of one node; the store is hosted by the launcher on the batch node):
    python gpu_pg_swap_test.py --launch          (runs the srun steps itself)"""
import datetime
import json
import os
import subprocess
import sys
import threading
import time

TO = datetime.timedelta(seconds=120)
GROUPS = {'W': list(range(8)), 'G': list(range(8)), 'H': [6, 7]}


def log(rank, msg):
    print(f'[pg_swap rank {rank}] {time.strftime("%T")} {msg}', flush=True)


def new_backends(store, gen, rank, dev):
    import torch.distributed as dist
    out = {}
    for name, ranks in GROUPS.items():
        if rank in ranks:
            opts = dist.ProcessGroupNCCL.Options()
            b = dist.ProcessGroupNCCL(dist.PrefixStore(f'{gen}/{name}', store), ranks.index(rank), len(ranks), opts)
            out[name] = b
    return out


def swap(pg, backend):
    import torch
    import torch.distributed as dist
    bt = dist.ProcessGroup.BackendType.CUSTOM
    pg._register_backend(torch.device('cuda'), bt, backend)
    pg._set_default_backend(bt)


def worker(rank, host, port, role, out):
    import torch
    import torch.distributed as dist
    lr = int(os.environ.get('SLURM_LOCALID', rank))
    torch.cuda.set_device(lr)
    dev = torch.device('cuda', lr)
    store = dist.TCPStore(host, port, None, is_master=False, timeout=TO)
    res = {'rank': rank, 'role': role, 'checks': {}, 't': {}}

    def check(name, ok):
        res['checks'][name] = bool(ok)
        log(rank, f'{"PASS" if ok else "FAIL"} {name}')

    def allreduce_all(engine, jb, tag):
        for name, ranks in GROUPS.items():
            if rank not in ranks:
                continue
            t = torch.full((1 << 20,), float(rank + 1), device=dev)
            if engine is not None:
                dist.all_reduce(t, group=engine[name] if name != 'W' else None)
            else:
                jb[name].allreduce([t]).wait()
            torch.cuda.synchronize(dev)
            want = float(sum(r + 1 for r in ranks))
            check(f'{tag} {name} sum {want}', bool(torch.all(t == want).item()))
        if rank in (6, 7):                                # the pipeline's send: point to point on the world group
            t = torch.full((1 << 16,), 6.0 if rank == 6 else 0.0, device=dev)
            if engine is not None:
                (dist.send if rank == 6 else dist.recv)(t, 7 if rank == 6 else 6)
            else:
                jb['W'].recv([t], 6, 0).wait()
            torch.cuda.synchronize(dev)
            if rank == 7:
                check(f'{tag} send 6 -> 7', bool(torch.all(t == 6.0).item()))

    if role == 'first':
        dist.init_process_group('nccl', store=dist.PrefixStore('g0/init', store), rank=rank, world_size=8, timeout=TO,
                                device_id=dev)
        engine = {'G': dist.new_group(GROUPS['G']), 'H': dist.new_group(GROUPS['H'])}
        engine['W'] = dist.distributed_c10d._get_default_group()
        allreduce_all(engine, None, 'before')
        if rank == 7:
            json.dump(res, open(f'{out}.{rank}.first', 'w'))
            store.set('lost/7', '1')
            os._exit(0)
        store.wait(['lost/7'])
        t = torch.ones(1 << 20, device=dev)
        w = dist.all_reduce(t, group=engine['G'], async_op=True)   # cut: rank 7 never joins
        time.sleep(3)
        t0 = time.time()
        olds = [engine[n]._get_backend(dev) for n in GROUPS if rank in GROUPS[n]]
        ths = [threading.Thread(target=b.abort, daemon=True) for b in olds]
        [x.start() for x in ths]
        [x.join(timeout=30) for x in ths]
        stuck = sum(x.is_alive() for x in ths)
        res['t']['abort_s'] = round(time.time() - t0, 3)
        check(f'abort of {len(olds)} old backends returned ({stuck} stuck, {res["t"]["abort_s"]} s)', stuck == 0)
        waited = threading.Thread(target=lambda: w.wait(), daemon=True)
        waited.start()
        waited.join(timeout=10)
        check('the cut all-reduce is released after the abort', not waited.is_alive())
        t0 = time.time()
        nb = new_backends(store, 'g1', rank, dev)
        for name, b in nb.items():
            swap(engine[name], b)
        res['t']['swap_s'] = round(time.time() - t0, 3)
        res['names'] = {n: dist.get_backend(engine[n] if n != 'W' else None) for n in nb}
        allreduce_all(engine, None, 'after swap')
        json.dump(res, open(f'{out}.{rank}.first', 'w'))
    else:
        jb = new_backends(store, 'g1', rank, dev)
        allreduce_all(None, jb, 'joiner')
        json.dump(res, open(f'{out}.{rank}.joiner', 'w'))
    os._exit(0)


def launch():
    import torch.distributed as dist
    host = subprocess.check_output(['hostname', '-i']).decode().split()[0]
    port = 26900 + os.getpid() % 500
    master = dist.TCPStore(host, port, None, is_master=True, timeout=TO, wait_for_workers=False)
    out = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'gpu_pg_swap_{os.environ.get("SLURM_JOB_ID", os.getpid())}')
    me = os.path.abspath(__file__)
    base = ['srun', '--overlap', '-N1', '--gpus-per-node=8', '--cpus-per-task=7']
    pre = f'LD_PRELOAD="{os.environ.get("FTV2_TEST_PRELOAD", "")}" '      # the training's preload (librccl first)
    w = subprocess.Popen(base + ['-n8', '--ntasks-per-node=8', 'bash', '-c',
                                 f'{pre}exec {sys.executable} {me} --worker $SLURM_PROCID {host} {port} first {out}'])
    while not master.check(['lost/7']):
        time.sleep(0.5)
    j = subprocess.Popen(base + ['-n1', 'bash', '-c',
                                 f'SLURM_LOCALID=7 {pre}exec {sys.executable} {me} --worker 7 {host} {port} joiner {out}'])
    codes = [w.wait(timeout=600), j.wait(timeout=600)]
    fails = 0
    for f in sorted(x for x in os.listdir(os.path.dirname(out)) if x.startswith(os.path.basename(out))):
        r = json.load(open(os.path.join(os.path.dirname(out), f)))
        bad = [k for k, ok in r['checks'].items() if not ok]
        fails += len(bad)
        print(f'rank {r["rank"]} ({r["role"]}): {len(r["checks"]) - len(bad)}/{len(r["checks"])} checks pass '
              f'{bad if bad else ""} {r.get("t", {})} {r.get("names", "")}')
    print(f'srun exit codes {codes}; {"all passed" if not fails and not any(codes) else f"{fails} failed"}')
    del master


if __name__ == '__main__':
    if sys.argv[1] == '--worker':
        worker(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6])
    elif sys.argv[1] == '--launch':
        launch()
