#!/usr/bin/env python3
"""cpu_track_sync_test.py (2026-11-07, after P1b 5873196): with the C tracking on, a synchronous call is made asynchronous and
waited for (hb_monitor._sync_as_async), so the tracking holds torch's Work and makes no CUDA event. Two gloo processes on the
login node. Checks per rank: the return values are those of the synchronous API (None; recv returns the sender); the results are
right; every tracked handle is a Work (no event); send / recv are tracked through isend / irecv; an async call is left as it is;
an object collective works; the tracking off leaves the calls untouched."""
import datetime
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'heartbeat-sys'))


def worker(rank, port, q):
    import hb_monitor as hb
    hb.install_counter()
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=60))
    c = {}
    hb.S['c_track'] = False
    t = torch.ones(4) * (rank + 1)
    c['off: sync all_reduce returns None, sum right'] = dist.all_reduce(t) is None and t.tolist() == [3.0] * 4
    hb.S['c_track'] = True
    w = dist.new_group([0, 1])
    t = torch.ones(4) * (rank + 1)
    r = dist.all_reduce(t, group=w)
    grp, works, ev, name, n = hb.S['c_last'][id(w)]
    c['on: sync all_reduce returns None, sum right'] = r is None and t.tolist() == [3.0] * 4
    c['on: the handle is a Work, no event, ended'] = len(works) == 1 and ev is None and hb._ended(works, ev)
    r = dist.barrier(group=w)
    c['on: barrier returns None, tracked'] = r is None and hb.S['c_last'][id(w)][3] == 'barrier'
    a = dist.all_reduce(torch.ones(2), group=w, async_op=True)
    c['on: an async call returns its Work'] = a is not None and hasattr(a, 'wait') and hb.S['c_last'][id(w)][1] == [a]
    a.wait()
    t = torch.full((3,), 7.0 if rank == 0 else 0.0)
    if rank == 0:
        r = dist.send(t, 1, group=w)
        c['on: send returns None'] = r is None
    else:
        r = dist.recv(t, 0, group=w)
        c['on: recv returns the sender, data right'] = r == 0 and t.tolist() == [7.0] * 3
    last = hb.S['c_last'][id(w)]
    c['on: send / recv tracked through isend / irecv'] = last[3] in ('isend', 'irecv') and last[2] is None and len(last[1]) == 1
    objs = [None, None]
    dist.all_gather_object(objs, {'r': rank}, group=w)
    c['on: all_gather_object works'] = objs == [{'r': 0}, {'r': 1}]
    t = torch.zeros(2) + rank
    dist.broadcast(t, 1, group=w)
    c['on: broadcast (positional group) right'] = t.tolist() == [1.0, 1.0]
    dist.destroy_process_group()
    q.put((rank, c))


def main():
    port = 29000 + os.getpid() % 900
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, port, q)) for r in range(2)]
    [p.start() for p in ps]
    res = [q.get(timeout=120) for _ in ps]
    [p.join() for p in ps]
    fails = 0
    for rank, c in sorted(res):
        for k, v in c.items():
            print(f'{"PASS" if v else "FAIL"}  rank {rank}: {k}')
            fails += not v
    print('all passed' if not fails else f'{fails} failed')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
