#!/usr/bin/env python3
"""gpu_cut_group_test.py (ft-sys-v2 recovery tests, milestone 3 C0c on one node, 2026-11-06): M3C1e on one node. A healthy
reduction that a stop cuts keeps a GPU busy; does the C stop (rc_c_flow: revoke the lost groups, the single list, the cut groups
aborted first and all at once, then the lost groups, the swap, the first all-reduce on the world, a state transfer) return on
every rank, and are the sums after the redo the clean run's?

8 ranks = 2 stages x 2 replicas x 2 ranks: stage 0 replica 0 = [0, 1], replica 1 = [2, 3]; stage 1 replica 0 = [4, 5],
replica 1 = [6, 7]. Groups (created in this order everywhere): D0 = [0..3], D1 = [4..7] (dense data parallel), E = [0,2], [1,3],
[4,6], [5,7] (expert data parallel), P = [0,4], [1,5], [2,6], [3,7] (pipeline pairs), R = [0,1], [2,3], [4,5], [6,7] (the
stage-replicas' all-to-all groups), C = [0..7] (the commit's group).
  micro-batch 0 on every rank: the reduction (all-reduces on its D and E groups) and a commit (sums S += results); stage 0 sends
  its activation to stage 1 on P (broadcast, as DeepSpeed's pairs do).
  the loss: ranks 2 and 3 leave (stage 0, replica 1, like node 1 in M3C1e).
  micro-batch 1, cut: ranks 0, 1 enter their reduction (D0 and E hold a lost rank); ranks 4, 5 enter theirs on D1 and E (no lost
  rank) while ranks 6, 7 never do: they wait in their receive from 2, 3 on P. So the GPUs of 4 and 5 are busy as in M3C1e.
  diagnosis: on ranks 4 and 5 a thread sets up a new 2-rank RCCL communicator while the GPU is busy; after 6 s its native stack
  is dumped (py-spy --native and gdb): the HIP call it waits in. It must end once the cut groups are aborted.
  the stop, in a thread named like the monitor's, with the real code: revoke_these (lost groups), cut_entry + single_list +
  decide_cut (one list in the store), abort_these (the cut groups, all at once, then the lost groups), rc_swap, first_collectives
  on W and C; joiners for 2 and 3 built with the fake backend and swapped in; the partners 0 -> 2 and 1 -> 3 send a 256 MB state
  in a batched round on W; then the redo: micro-batch 1's reductions on D0, D1 and E, and 2 -> 6, 3 -> 7 send again on P.
Pass: every step returns within its limit, the list names D1 and the two stage-1 E groups (4, 5 'redo'; 6, 7 'none'), the state
arrives intact, every sum equals the clean run's.
    python gpu_cut_group_test.py --launch          (inside a one-node allocation; runs the srun steps itself)"""
import datetime
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'heartbeat-sys'))
TO = datetime.timedelta(seconds=300)
ORDER = [('D0', [0, 1, 2, 3]), ('D1', [4, 5, 6, 7]), ('E0', [0, 2]), ('E1', [1, 3]), ('E2', [4, 6]), ('E3', [5, 7]),
         ('P0', [0, 4]), ('P1', [1, 5]), ('P2', [2, 6]), ('P3', [3, 7]), ('R0', [0, 1]), ('R1', [2, 3]), ('R2', [4, 5]),
         ('R3', [6, 7]), ('C', list(range(8)))]
LOST = [2, 3]
N = 1 << 22                                            # 16 MB per reduction tensor


def log(rank, msg):
    print(f'[cut_group rank {rank}] {time.strftime("%T")} {msg}', flush=True)


def contrib(rank, mb, name):
    return float(1 + rank + 10 * mb + (100 if name.startswith('E') else 0))


def red_groups(rank):
    return [n for n, rs in ORDER if rank in rs and n[0] in 'DE']


class StoreEtcd:
    def __init__(self, store):
        self.s = store

    def put(self, k, v):
        self.s.set(k, v)

    def get(self, k):
        return (self.s.get(k).decode(), None) if self.s.check([k]) else (None, None)

    def range(self, prefix):
        out = {}
        for r in range(8):
            k = f'{prefix}{r}'
            if self.s.check([k]):
                out[k] = self.s.get(k).decode()
        return out, None


class Ctl:
    def __init__(self, rank, hb, dev, lost_ids):
        self.rank, self.hb, self.device = rank, hb, dev
        self.c = {'aborted': set(lost_ids)}

    def log(self, msg):
        log(self.rank, msg)


def dump_stacks(rank, tag, out):
    pid = os.getpid()
    txt = []
    for cmd in ([os.path.expanduser('~/.local/bin/py-spy'), 'dump', '--native', '--pid', str(pid)],
                ['gdb', '-p', str(pid), '-batch', '-ex', 'thread apply all bt 25']):
        try:
            txt.append(f'$ {" ".join(cmd)}\n' + subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout)
        except Exception as e:
            txt.append(f'$ {" ".join(cmd)}: {e!r}')
    with open(f'{out}.stacks.{rank}.{tag}', 'w') as f:
        f.write('\n'.join(txt))


def worker(rank, host, port, role, out):
    import torch
    import torch.distributed as dist
    import hb_monitor as hb
    import rc_c_flow as cf
    import rc_swap
    lr = int(os.environ.get('SLURM_LOCALID', rank))
    torch.cuda.set_device(lr)
    dev = torch.device('cuda', lr)
    hb.install_counter()
    hb.S['c_track'] = True
    store = dist.TCPStore(host, port, None, is_master=False, timeout=TO)
    res = {'rank': rank, 'role': role, 'checks': {}, 't': {}}

    def check(name, ok):
        res['checks'][name] = bool(ok)
        log(rank, f'{"PASS" if ok else "FAIL"} {name}')

    def save():
        json.dump(res, open(f'{out}.{rank}.{role}', 'w'))

    def step(name, fn, limit):
        """fn in a thread named like the monitor's; a step that does not return in time fails the test (stacks dumped)."""
        box = {}

        def run():
            t0 = time.time()
            try:
                box['v'] = fn()
            except Exception as e:
                box['e'] = repr(e)[:300]
            box['s'] = round(time.time() - t0, 3)
        th = threading.Thread(target=run, name='ftv2-monitor', daemon=True)
        th.start()
        th.join(timeout=limit)
        if th.is_alive() or 'e' in box:
            check(f'step {name} returned', False)
            res['t'][name] = box.get('e', f'> {limit} s')
            if th.is_alive():
                dump_stacks(rank, name, out)
            save()
            os._exit(3)
        res['t'][name] = box['s']
        return box.get('v')

    groups = {}
    if role == 'first':
        dist.init_process_group('nccl', store=dist.PrefixStore('g0/init', store), rank=rank, world_size=8, timeout=TO,
                                device_id=dev)
    else:
        from torch.testing._internal.distributed.fake_pg import FakeStore
        dist.init_process_group('fake', store=FakeStore(), rank=rank, world_size=8)
    for name, rs in ORDER:
        groups[name] = dist.new_group(rs)
    W = dist.distributed_c10d._get_default_group()
    S, snap = {}, {}

    def reduce_mb(mb, sync=True):
        vals, works = {}, []
        for name in red_groups(rank):
            t = torch.full((N,), contrib(rank, mb, name), device=dev)
            w = dist.all_reduce(t, group=groups[name], async_op=not sync)
            if w is not None:
                works.append(w)
            vals[name] = t
        return vals, works

    def commit(mb, vals):
        torch.cuda.synchronize(dev)
        for k, t in vals.items():
            S[k] = S.get(k, 0.0) + float(t[0].item())
        snap[mb] = dict(S)
        hb.S['c_mark'] = dict(hb.S['c_n'])

    pair = {0: 'P0', 4: 'P0', 1: 'P1', 5: 'P1', 2: 'P2', 6: 'P2', 3: 'P3', 7: 'P3'}[rank]
    src = rank if rank < 4 else rank - 4

    def act(mb):
        return torch.full((1 << 16,), float(1000 * (mb + 1) + src), device=dev)

    clean = {k: sum(float(sum(contrib(r, mb, k) for r in rs)) for mb in (0, 1)) for k, rs in ORDER
             if rank in rs and k[0] in 'DE'}
    survivors = [r for r in range(8) if r not in LOST]
    e = StoreEtcd(store)
    lost_names = [n for n, rs in ORDER if set(rs) & set(LOST)]

    if role == 'first':
        hb.S['c_mark'] = dict(hb.S['c_n'])
        commit(0, reduce_mb(0)[0])
        a = act(0) if rank < 4 else torch.zeros(1 << 16, device=dev)
        dist.broadcast(a, src=src, group=groups[pair])
        torch.cuda.synchronize(dev)
        if rank >= 4:
            check('micro-batch 0 activation', bool(torch.all(a == float(1000 + src)).item()))
        if rank in LOST:
            save()
            store.set(f'lost/{rank}', '1')
            os._exit(0)
        store.wait([f'lost/{r}' for r in LOST])
        # ---- micro-batch 1, cut
        diag = {}
        if rank in (0, 1):
            vals1, works = reduce_mb(1, sync=False)            # D0 and E hold a lost rank: cut by the loss
            hb.S['c_where'] = 'red'
        elif rank in (4, 5):
            xt = torch.ones(1 << 10, device=dev)               # allocated before the GPU is busy
            vals1, works = reduce_mb(1, sync=True)             # D1 and E: 6 and 7 never enter -> the GPU stays busy
            hb.S['c_where'] = 'red'

            def diag_comm():
                t0 = time.time()
                b = dist.ProcessGroupNCCL(dist.PrefixStore('diag', store), rank - 4, 2, dist.ProcessGroupNCCL.Options())
                diag['made_s'] = round(time.time() - t0, 3)
                b.allreduce([xt]).wait()
                ev = torch.cuda.Event()
                ev.record()
                while not ev.query():
                    time.sleep(0.001)
                diag['done_s'] = round(time.time() - t0, 3)
            dth = threading.Thread(target=diag_comm, name='diag', daemon=True)
            dth.start()
            time.sleep(6)
            diag['blocked_after_6s'] = dth.is_alive()
            if dth.is_alive():
                dump_stacks(rank, 'diag', out)
        else:                                                  # 6, 7: waiting in the receive from 2, 3 (lost)
            a1 = torch.zeros(1 << 16, device=dev)
            w6 = dist.broadcast(a1, src=src, group=groups[pair], async_op=True)
            hb.S['c_where'] = 'p2p'
            hb.S['c_inst_mark'] = {i: n - 1 for i, n in hb.S['c_n'].items()}
        time.sleep(2)
        # ---- the C stop (the monitor thread's order in rc_c_flow.c_on_failure)
        ctl = Ctl(rank, hb, dev, [id(groups[n]) for n in lost_names])
        lgs = [groups[n] for n in lost_names]
        step('revoke lost groups', lambda: cf.revoke_these(lgs, dev, ctl.log, 10, 'hold a lost rank'), 30)
        d = step('single list', lambda: cf.single_list(ctl, e, 'c0c', survivors, limit=120), 180)
        keys = rc_swap.group_keys()
        allkeys = {}
        seen = {}
        for n, rs in [('W', list(range(8)))] + ORDER:     # every group's key, members or not (rc_swap.group_keys' rule)
            import hashlib
            h = hashlib.md5(','.join(map(str, sorted(rs))).encode()).hexdigest()[:12]
            allkeys[f'{h}/{seen.get(h, 0)}'] = n
            seen[h] = seen.get(h, 0) + 1
        names = allkeys
        cuts = {names[k]: v for k, v in d['cuts'].items()}
        acts = {names[k]: a for k, a in d['actions'].get(str(rank), {}).items()}
        res['list'] = {'mode': d['mode'], 'cuts': cuts, 'mine': acts, 'why': d['why']}
        check('mode C', d['mode'] == 'C')
        check('the list names D1, E2, E3', sorted(cuts) == ['D1', 'E2', 'E3'])
        want = {4: 'redo', 5: 'redo', 6: 'none', 7: 'none'}.get(rank)
        check('my actions', all(a == want for a in acts.values()) if want else not acts)
        cgs = [groups[n] for n in cuts if rank in dict(ORDER)[n]]
        ctl.c['redo'] = {id(groups[n]) for n, a in acts.items() if a == 'redo'}
        secs = step('abort cut and lost groups at once', lambda: cf.abort_batch(cgs, lgs, dev, ctl.log, 30), 60)
        res['abort_secs'] = secs
        check(f'every abort returned in under 5 s {secs}', all(x is not None and x < 5 for x in secs))
        if rank in (4, 5):
            dth.join(timeout=30)
            diag['ended_after_abort'] = not dth.is_alive()
            res['diag'] = diag
            check('the communicator setup blocked while the GPU was busy', diag['blocked_after_6s'])
            check('it ended once the cut groups were aborted', diag['ended_after_abort'])
        res['swapped'] = step('swap', lambda: rc_swap.swap_groups(dist.PrefixStore('g1', store), LOST, dev, 'nccl', TO,
                                                                  extra=list(d['cuts'])), 60)
    else:
        t = torch.ones(4, device=dev)
        dist.all_reduce(t, group=groups['D0'])          # a build-time collective completes alone
        d = cf.wait_list(e, 'c0c', 300)
        res['swapped'] = step('swap', lambda: rc_swap.swap_groups(dist.PrefixStore('g1', store), LOST, dev, 'nccl', TO,
                                                                  joiner=True), 60)
        ctl = Ctl(rank, hb, dev, [])
    step('first all-reduces (W, C)', lambda: cf.first_collectives(ctl, [W, groups['C']], limit_s=120), 180)
    # ---- the state transfer, partner -> joiner, batched on W (subset of the group: allowed after the first collective)
    big = 1 << 26                                        # 256 MB of fp32
    if rank in (0, 1, 2, 3):
        peer = rank + 2 if rank < 2 else rank - 2

        def xfer():
            if rank < 2:
                st = torch.arange(big, device=dev, dtype=torch.float32) * (rank + 1)
                s0 = torch.tensor([snap[0]['D0'], snap[0][f'E{rank}']], dtype=torch.float64, device=dev)
                ws = dist.batch_isend_irecv([dist.P2POp(dist.isend, st, peer), dist.P2POp(dist.isend, s0, peer)])
            else:
                st = torch.empty(big, device=dev, dtype=torch.float32)
                s0 = torch.empty(2, dtype=torch.float64, device=dev)
                ws = dist.batch_isend_irecv([dist.P2POp(dist.irecv, st, peer), dist.P2POp(dist.irecv, s0, peer)])
            for w in ws:
                w.wait()
            torch.cuda.synchronize(dev)
            if rank >= 2:
                ok = bool(torch.equal(st, torch.arange(big, device=dev, dtype=torch.float32) * (peer + 1)))
                v = s0.tolist()
                S.update({'D0': v[0], f'E{rank - 2}': v[1]})
                return ok
            return True
        ok = step('state transfer (256 MB)', xfer, 120)
        if rank >= 2:
            check('state arrived intact', ok)
            snap[0] = dict(S)
    # ---- the redo: micro-batch 1 from the last commit on every rank of the cut and lost groups
    S.clear()
    S.update(snap[0])
    step('redo micro-batch 1 reductions', lambda: commit(1, reduce_mb(1)[0]), 120)
    check(f'sums equal the clean run {clean}', all(abs(S[k] - clean[k]) < 1e-6 for k in clean))
    res['sums'] = S
    if rank in (2, 3, 6, 7):                             # the pipeline send cut by the loss, again
        def resend():
            a = act(1) if rank < 4 else torch.zeros(1 << 16, device=dev)
            dist.broadcast(a, src=src, group=groups[pair])
            torch.cuda.synchronize(dev)
            return bool(torch.all(a == float(2000 + src)).item())
        ok = step('pipeline send again', resend, 60)
        if rank >= 4:
            check('micro-batch 1 activation after the repair', ok)
    save()
    os._exit(0)


def launch():
    import torch.distributed as dist
    host = subprocess.check_output(['hostname', '-i']).decode().split()[0]
    port = 26900 + os.getpid() % 500
    master = dist.TCPStore(host, port, None, is_master=True, timeout=TO, wait_for_workers=False)
    out = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'gpu_cut_group_{os.environ.get("SLURM_JOB_ID", os.getpid())}')
    me = os.path.abspath(__file__)
    base = ['srun', '--overlap', '-N1', '--gpus-per-node=8', '--cpus-per-task=7']
    pre = f'LD_PRELOAD="{os.environ.get("FTV2_TEST_PRELOAD", "")}" '
    w = subprocess.Popen(base + ['-n8', '--ntasks-per-node=8', 'bash', '-c',
                                 f'{pre}exec {sys.executable} {me} --worker $SLURM_PROCID {host} {port} first {out}'])
    while not all(master.check([f'lost/{r}']) for r in LOST):
        time.sleep(0.5)
    j = subprocess.Popen(base + ['-n2', 'bash', '-c',
                                 f'R=$((SLURM_PROCID + 2)); SLURM_LOCALID=$R {pre}exec {sys.executable} {me} --worker $R {host} '
                                 f'{port} joiner {out}'])
    codes = [w.wait(timeout=1200), j.wait(timeout=1200)]
    fails = 0
    d = os.path.dirname(out)
    for f in sorted(x for x in os.listdir(d) if x.startswith(os.path.basename(out)) and '.stacks.' not in x):
        r = json.load(open(os.path.join(d, f)))
        bad = [k for k, ok in r['checks'].items() if not ok]
        fails += len(bad)
        print(f'rank {r["rank"]} ({r["role"]}): {len(r["checks"]) - len(bad)}/{len(r["checks"])} checks pass {bad if bad else ""}')
        print(f'    times {r.get("t")}; list {r.get("list")}; diag {r.get("diag")}; swapped {r.get("swapped")}; '
              f'aborts {r.get("abort_secs")}')
    for f in sorted(x for x in os.listdir(d) if x.startswith(os.path.basename(out)) and '.stacks.' in x):
        print(f'stacks: {os.path.join(d, f)}')
    print(f'srun exit codes {codes}; {"all passed" if not fails and not any(codes) else f"{fails} failed"}')
    del master


if __name__ == '__main__':
    if sys.argv[1] == '--worker':
        worker(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6])
    elif sys.argv[1] == '--launch':
        launch()
