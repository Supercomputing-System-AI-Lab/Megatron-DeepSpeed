#!/usr/bin/env python3
"""gpu_stop_batch_test.py (ft-sys-v2 recovery tests, 2026-11-07, the coordinator's validation of the 'one batch of aborts, never a
revoke' stop): two nodes, the four cases of M3R1 5872321, both batch shapes, with the real code.

16 ranks; node 1's eight ranks leave at once. Survivor r on node 0 holds three idle groups {r, r + 8}, a 16-rank group after one
batched round (the commit group's shape), a pair Q = {r, r + 8} to its lost partner and a healthy pair H = {r, r ^ 1}. In flight at
the stop: nothing (ranks 0, 1); a receive on H from the healthy partner, which never sends (2, 3); a receive on Q from the lost
partner (4, 5); both (6, 7). No revoke anywhere. Then, by FTV2_TEST_SHAPE:
  C    the single list (hb_monitor's tracking, rc_c_flow.cut_entry / single_list / decide_cut over a store) names the groups with a
       cut operation; rc_c_flow.abort_batch aborts the cut groups and the lost groups (A1, A2, A3, C, Q) in one batch;
  AB   rc_stop.abort_all aborts every backend of the registry in one batch, then torch's _abort_process_group clears the registry;
  ABF  the same batch, then the registry is cleared by hand and the objects stay referenced (rc_stop.forget_all: no destructor).
  ...K the same with the keeper the controller makes at start (rc_stop.make_keeper: a one-rank communicator, never aborted, which
       keeps the network plugin loaded when every other communicator is aborted).
Then the next communicator setup: a new 8-rank RCCL communicator on node 0 and one all-reduce (polled, 120-s limit).
Pass: every abort returns in under 1 s, no rank crashes, the new communicator's all-reduce returns with the right sum; with C, the
list names the healthy pair's group on ranks 2, 3, 6, 7 and on no other rank.
    python gpu_stop_batch_test.py --launch          (inside a two-node allocation; runs both shapes, one srun step each)"""
import datetime
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'heartbeat-sys'))
TO = datetime.timedelta(seconds=300)
CASE = {0: 'idle', 1: 'idle', 2: 'healthy recv', 3: 'healthy recv', 4: 'lost-peer recv', 5: 'lost-peer recv', 6: 'both', 7: 'both'}


def log(rank, msg):
    print(f'[stop_batch rank {rank}] {time.strftime("%T")} {msg}', flush=True)


class StoreEtcd:
    def __init__(self, store):
        self.s = store

    def put(self, k, v):
        self.s.set(k, v)

    def get(self, k):
        return (self.s.get(k).decode(), None) if self.s.check([k]) else (None, None)

    def range(self, prefix):
        out = {}
        for r in range(16):
            k = f'{prefix}{r}'
            if self.s.check([k]):
                out[k] = self.s.get(k).decode()
        return out, None


class Ctl:
    def __init__(self, rank, hb, dev, lost_ids):
        self.rank, self.hb, self.device = rank, hb, dev
        self.c = {'aborted': set(lost_ids)}
        self.lines = []

    def log(self, msg):
        self.lines.append(msg)
        log(self.rank, msg)


def worker(rank, host, port, shape, out):
    import faulthandler
    faulthandler.enable(all_threads=True)             # a crash prints every thread's Python stack
    import torch
    import torch.distributed as dist
    import hb_monitor as hb
    import rc_c_flow as cf
    import rc_stop
    import rc_swap
    lr = int(os.environ.get('SLURM_LOCALID', rank % 8))
    torch.cuda.set_device(lr)
    dev = torch.device('cuda', lr)
    hb.install_counter()
    hb.S['c_track'] = True
    store = dist.TCPStore(host, port, None, is_master=False, timeout=TO)
    if shape.endswith('K'):
        rc_stop.make_keeper(dev)                    # as the controller does at start: a one-rank communicator never aborted
    dist.init_process_group('nccl', store=dist.PrefixStore('init', store), rank=rank, world_size=16, timeout=TO, device_id=dev)
    me = rank % 8
    groups = {}
    for name in ('A1', 'A2', 'A3'):
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
    for name in ('A1', 'A2', 'A3'):
        t = torch.ones(1 << 16, device=dev)
        dist.all_reduce(t, group=groups[name])
    peer = rank + 8 if rank < 8 else rank - 8
    a, b = torch.full((1 << 18,), float(rank), device=dev), torch.empty(1 << 18, device=dev)
    for w in dist.batch_isend_irecv([dist.P2POp(dist.isend, a, peer, group=groups['C']),
                                     dist.P2POp(dist.irecv, b, peer, group=groups['C'])]):
        w.wait()
    t = torch.ones(1 << 10, device=dev)
    dist.all_reduce(t, group=groups['Q'])
    if rank < 8:
        dist.all_reduce(t, group=groups['H'])
    torch.cuda.synchronize(dev)
    hb.S['c_mark'] = dict(hb.S['c_n'])
    store.set(f'ready/{rank}', '1')
    store.wait([f'ready/{r}' for r in range(16)])
    if rank >= 8:
        log(rank, 'leaving (node 1 lost)')
        store.set(f'gone/{rank}', '1')
        os._exit(0)
    store.wait([f'gone/{r}' for r in range(8, 16)])
    case = CASE[me]
    res = {'rank': rank, 'case': case, 'shape': shape, 'checks': {}}
    hb.S['c_where'] = 'idle'
    if case in ('healthy recv', 'both'):
        hb.S['c_inst_mark'] = dict(hb.S['c_n'])                      # the instruction's start, as PipeC records it
        x = torch.zeros(1 << 16, device=dev)
        dist.broadcast(x, src=rank ^ 1, group=groups['H'], async_op=True)
        hb.S['c_where'] = 'p2p'
    if case in ('lost-peer recv', 'both'):
        y = torch.zeros(1 << 16, device=dev)
        dist.broadcast(y, src=rank + 8, group=groups['Q'], async_op=True)
        hb.S['c_where'] = 'p2p' if hb.S['c_where'] == 'p2p' else 'p2p'
    time.sleep(2)
    lost = [groups[n] for n in ('A1', 'A2', 'A3', 'C', 'Q')]
    t0 = time.time()
    if shape.startswith('C'):
        ctl = Ctl(rank, hb, dev, [id(g) for g in lost])
        d = cf.single_list(ctl, StoreEtcd(store), 'sb', list(range(8)), limit=120)
        keys = rc_swap.group_keys()
        hkey = keys[id(groups['H'])]
        res['list'] = {'mode': d['mode'], 'cuts': d['cuts'], 'mine': d['actions'].get(str(rank), {}), 'why': d['why']}
        want = case in ('healthy recv', 'both')
        res['checks']['the list names the healthy pair exactly when its receive is cut'] = (hkey in d['cuts']) == want
        cgs = [g for g in groups.values() if keys.get(id(g)) in d['cuts']]
        secs = cf.abort_batch(cgs, lost, dev, ctl.log, 30)
    else:
        r = rc_stop.abort_all(dev, lambda m: log(rank, m), limit_s=30, clear='forget' if shape.startswith('ABF') else 'torch')
        secs = r.get('returned_s', [])
        res['abort'] = {k: v for k, v in r.items() if k != 'returned_s'}
    res['abort_secs'] = secs
    res['stop_s'] = round(time.time() - t0, 3)
    res['checks']['every abort returned in under 1 s'] = bool(secs) and all(s is not None and s < 1.0 for s in secs)
    # the next communicator setup on this GPU: 8 ranks on node 0, one all-reduce, polled
    t1 = time.time()
    nb = dist.ProcessGroupNCCL(dist.PrefixStore('next', store), rank, 8, dist.ProcessGroupNCCL.Options())
    z = torch.full((1 << 12,), float(rank + 1), device=dev)
    nb.allreduce([z])
    ev = torch.cuda.Event()
    ev.record()
    ok = False
    while time.time() - t1 < 120:
        if ev.query():
            ok = True
            break
        time.sleep(0.01)
    res['next_comm_s'] = round(time.time() - t1, 3) if ok else None
    res['checks']['the next communicator setup and all-reduce return'] = ok and bool(torch.all(z == 36.0).item())
    log(rank, f'{shape} {case}: aborts {secs}; next communicator {res["next_comm_s"]} s; checks {res["checks"]}')
    json.dump(res, open(f'{out}.{shape}.{rank}', 'w'), default=str)
    os._exit(0)


def launch():
    import torch.distributed as dist
    host = subprocess.check_output(['hostname', '-i']).decode().split()[0]
    out = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'gpu_stop_batch_{os.environ.get("SLURM_JOB_ID", os.getpid())}')
    me = os.path.abspath(__file__)
    pre = f'LD_PRELOAD="{os.environ.get("FTV2_TEST_PRELOAD", "")}" '
    fails = 0
    for i, shape in enumerate(os.environ.get('FTV2_TEST_SHAPES', 'C AB').split()):
        port = 27600 + os.getpid() % 300 + 1 + i
        master = dist.TCPStore(host, port, None, is_master=True, timeout=TO, wait_for_workers=False)
        w = subprocess.Popen(['srun', '--overlap', '-N2', '-n16', '--ntasks-per-node=8', '--gpus-per-node=8', '--cpus-per-task=7',
                              '--kill-on-bad-exit=0',
                              'bash', '-c', f'{pre}exec {sys.executable} {me} --worker $SLURM_PROCID {host} {port} {shape} {out}'])
        try:
            code = w.wait(timeout=300)
        except subprocess.TimeoutExpired:
            w.kill()
            code = "step limit (300 s)"
        for r in range(8):
            f = f'{out}.{shape}.{r}'
            if not os.path.exists(f):
                print(f'{shape} rank {r}: NO RESULT (crashed or hung)')
                fails += 1
                continue
            d = json.load(open(f))
            bad = [k for k, ok in d['checks'].items() if not ok]
            fails += len(bad)
            print(f'{shape} rank {r} ({d["case"]}): {len(d["checks"]) - len(bad)}/{len(d["checks"])} checks pass {bad if bad else ""}; '
                  f'aborts {d["abort_secs"]}; stop {d["stop_s"]} s; next communicator {d["next_comm_s"]} s'
                  + (f'; list cuts {d["list"]["cuts"]} mine {d["list"]["mine"]} mode {d["list"]["mode"]}' if 'list' in d else ''))
        print(f'{shape}: srun exit code {code}')
        del master
    print('all passed' if not fails else f'{fails} failed')


if __name__ == '__main__':
    if sys.argv[1] == '--worker':
        worker(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6])
    elif sys.argv[1] == '--launch':
        launch()
