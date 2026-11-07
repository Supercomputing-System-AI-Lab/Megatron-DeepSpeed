#!/usr/bin/env python3
"""cpu_stop_test.py (ft-sys-v2 recovery tests, 2026-11-06, after M3C12 5872105 b_lost): the stop frees nothing while a revoke or
abort thread is still running, and says that the rank must leave. CPU and gloo on the login node, one process.

The defect it guards: B's stop went on past aborts that had not returned in their limit, cleared the registry and released the old
engine; four ranks then died in glibc ('The futex facility returned an unexpected error code'), an abort thread still inside RCCL.
Checks (rc_stop.abort_all with stand-in backends; the registry is a real gloo world of one rank):
  1. every abort returns: 'ok', the registry is cleared, no thread alive;
  2. one abort never returns: 'stuck' after the limit, the registry is NOT cleared, threads_alive() counts it;
  3. a revoke thread still running (stand-in) and every abort returning: 'stuck' as well (the rule covers revokes);
  4. rc_controller's B stop turns 'stuck' into a leave: _exit(EXIT_FAILED) is called and freed is never set.
    python tests/cpu_stop_test.py   (the venv's python)"""
import datetime
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'heartbeat-sys'))


class Backend:
    def __init__(self, block=None):
        self.block, self.aborted = block, False

    def abort(self):
        if self.block is not None:
            self.block.wait()
        self.aborted = True


def world():
    import torch.distributed as dist
    if dist.is_initialized():
        return
    store = dist.HashStore()
    dist.init_process_group('gloo', store=store, rank=0, world_size=1, timeout=datetime.timedelta(seconds=10))


def main():
    import torch
    import torch.distributed as dist
    import rc_stop
    checks = {}
    log = lambda m: print('   log:', m)
    dev = torch.device('cpu')

    # 1. every abort returns
    world()
    bs = [Backend(), Backend()]
    rc_stop._backends = lambda device: bs
    r = rc_stop.abort_all(dev, log, limit_s=1.0)
    checks['1 every abort returns: ok'] = r['abort'] in ('ok', 'partial') and r['stuck'] == 0 and r['alive'] == 0
    checks['1 the registry is cleared'] = not dist.is_initialized() or len(dist.distributed_c10d._world.pg_map) == 0

    # 2. one abort never returns
    world()
    hold = threading.Event()
    bs = [Backend(), Backend(block=hold)]
    rc_stop._backends = lambda device: bs
    t0 = time.time()
    r = rc_stop.abort_all(dev, log, limit_s=0.5)
    checks['2 one abort stuck: stuck after the limit'] = r['abort'] == 'stuck' and r['stuck'] == 1 and time.time() - t0 < 3
    checks['2 the registry is NOT cleared'] = dist.is_initialized() and len(dist.distributed_c10d._world.pg_map) == 1
    checks['2 threads_alive counts it'] = rc_stop.threads_alive() == 1
    hold.set()
    time.sleep(0.2)
    checks['2 the thread ends once RCCL returns'] = rc_stop.threads_alive() == 0

    # 3. a revoke thread still running, every abort returning
    hold3 = threading.Event()
    rt = threading.Thread(target=hold3.wait, daemon=True)
    rc_stop._THREADS.append(rt)
    rt.start()
    bs = [Backend()]
    rc_stop._backends = lambda device: bs
    r = rc_stop.abort_all(dev, log, limit_s=0.5)
    checks['3 a revoke still running: stuck'] = r['abort'] == 'stuck' and r['stuck'] == 0 and r['alive'] == 1
    hold3.set()
    time.sleep(0.2)

    # 4. the B stop turns 'stuck' into a leave
    import rc_controller

    class Exit(Exception):
        pass

    class Ctl:
        pass
    c = Ctl()
    c.device, c.log, c.world, c.rank = dev, log, 4, 1
    c.freed = threading.Event()
    c.T, c.stop_info = {}, {}
    exits = []

    def _exit(code, why):
        exits.append((code, why))
        raise Exit()
    c._exit = _exit
    c.write_times = lambda g, extra=None: None
    c.mark = lambda g, name: None
    c.etcd = lambda: None
    hold4 = threading.Event()
    bs = [Backend(block=hold4)]
    rc_stop._backends = lambda device: bs
    rc_stop.revoke_all = lambda device, log, limit_s=10.0, groups=None: {'revoked': 0}
    rc_stop.etcd_barrier = lambda *a, **k: {'arrived': 3, 'of': 3}
    import rc_c_flow
    rc_c_flow.lost_groups = lambda lost: []
    os.environ['FTV2_ABORT_WAIT_S'] = '0.5'
    try:
        rc_controller.Recovery.stop_comms(c, 'g', [3], {})      # the controller's own stop, on a stand-in object
    except Exit:
        pass
    checks['4 the rank leaves with EXIT_FAILED'] = len(exits) == 1 and exits[0][0] == rc_controller.EXIT_FAILED
    checks['4 freed is never set'] = not c.freed.is_set()
    hold4.set()

    fails = 0
    for name, ok in checks.items():
        print(f'{"PASS" if ok else "FAIL"}  {name}')
        fails += not ok
    print(f'\n{fails} failed' if fails else '\nall passed')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
