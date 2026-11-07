#!/usr/bin/env python3
"""cpu_bury_test.py (2026-11-07, M3ST1 5875211): rc_stop.bury keeps the old process groups referenced until the rebuild is ready
and drops them on its own thread, so the last reference is not the main thread's; rc_stop.old_process_groups finds the process
groups Python references (a module global, an attribute, a list) and leaves the keeper's out. One gloo process on the login node."""
import datetime
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'heartbeat-sys'))
import torch.distributed as dist  # noqa: E402
import rc_stop  # noqa: E402

DIED = {}


class Obj:
    def __init__(self, name):
        self.name = name

    def __del__(self):
        DIED[self.name] = threading.current_thread().name


class Holder:
    pass


def main():
    checks = {}
    logs = []
    ev = threading.Event()
    a, b = Obj('a'), Obj('b')
    rc_stop.bury([a, b], ev, logs.append)
    del a, b                                   # the main thread's references go first
    time.sleep(0.2)
    checks['nothing released before the rebuild is ready'] = DIED == {}
    ev.set()
    for _ in range(50):
        if len(DIED) == 2:
            break
        time.sleep(0.1)
    checks['released after it, on the graveyard thread'] = DIED == {'a': 'ftv2-pg-graveyard', 'b': 'ftv2-pg-graveyard'}
    checks['the release is logged'] = any('released off the main thread' in x for x in logs)
    dist.init_process_group('gloo', init_method='tcp://127.0.0.1:%d' % (29400 + os.getpid() % 500), rank=0, world_size=1,
                            timeout=datetime.timedelta(seconds=30))
    g1 = dist.new_group([0])
    h = Holder()
    h.pg = dist.new_group([0])
    kept = dist.new_group([0])
    rc_stop._KEEPER.append((kept, None))
    found = rc_stop.old_process_groups()
    ids = {id(x) for x in found}
    checks['finds a global and an attribute'] = id(g1) in ids and id(h.pg) in ids
    checks['leaves the keeper out'] = id(kept) not in ids
    rc_stop._KEEPER.clear()
    dist.destroy_process_group()
    fails = 0
    for k, v in checks.items():
        print(f'{"PASS" if v else "FAIL"}  {k}')
        fails += not v
    print('all passed' if not fails else f'{fails} failed')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
