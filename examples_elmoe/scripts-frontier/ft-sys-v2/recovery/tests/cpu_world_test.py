#!/usr/bin/env python3
"""cpu_world_test.py (ft-sys-v2 recovery tests, 2026-11-06): the new world's handshake and the resume decision, on the login node.
  1. decide(): the resume step from the survivors' reports (consistent, a step half done, a lost rank and its copy both lost);
     waits_for_stop(): which exceptions of the main thread wait for the stop, and how the heartbeat classifies the others.
  2. rc_world.make_world against a real etcd (one member on local ports): 4 processes; generation 'a' hosted by rank 1, then
     the world torn down (the abort's registry clear, rc_stop.forget_all) and generation 'b' hosted by rank 2, each checked by an
     all-reduce (gloo). The address goes through etcd (world/<g>), as on the GPUs.
    python tests/cpu_world_test.py --etcd-bin <etcd>   (the venv's python)"""
import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.dirname(HERE)
sys.path.insert(0, RC)
sys.path.insert(0, os.path.join(os.path.dirname(RC), 'heartbeat-sys'))
fails = []


def check(name, got, want):
    print(f'{"PASS" if got == want else "FAIL"}  {name}: got {got}, want {want}')
    if got != want:
        fails.append(name)


def decide_cases():
    from rc_controller import decide, EXIT_WINDOW, EXIT_NEIGHBOURS
    rep = lambda own, sh: {'own': own, 'shadow': sh}
    world, stride, lost = 6, 2, [4, 5]                     # three "nodes" of two ranks; node 2 lost
    ok = {r: rep([7, 7], [7, 7]) for r in range(4)}
    check('consistent: resume at 7', decide(ok, lost, world, stride)[:2], (True, 7))
    half = dict(ok); half[1] = rep([8, 7], [7, 7])
    check('a step half done on a survivor: window', decide(half, lost, world, stride)[3], EXIT_WINDOW)
    sh = dict(ok); sh[0] = rep([7, 7], [8, 8])             # rank 0 keeps rank 4's copy, one step ahead
    check('a lost rank\'s copy ahead: window', decide(sh, lost, world, stride)[3], EXIT_WINDOW)
    after = {r: rep([8, 8], [8, 8]) for r in range(4)}
    check('every update of the step done: resume at 8', decide(after, lost, world, stride)[:2], (True, 8))
    two = {r: rep([7, 7], [7, 7]) for r in (0, 1)}
    check('neighbours lost', decide(two, [2, 3, 4, 5], world, stride)[3], EXIT_NEIGHBOURS)
    check('null recovery (nobody lost)', decide(ok, [], 4, stride)[:2], (True, 7))


def wrapper_cases():
    """Which exceptions in the main thread wait for the recovery's stop (E3 5862190: every one did, and a HIP fault became a
    "hang"). The messages are the real ones of the runs where they exist."""
    import torch
    import torch.distributed as dist
    from rc_controller import waits_for_stop
    import hb_monitor
    ill = ("CUDA error: an illegal memory access was encountered\nSearch for `hipErrorIllegalAddress' in "
           "https://rocm.docs.amd.com/projects/HIP/en/latest/index.html for more information.")
    AccErr = getattr(torch, 'AcceleratorError', RuntimeError)
    try:
        acc = AccErr(ill)
    except TypeError:
        acc = RuntimeError(ill)
    cases = [
        ('StopForRecovery (the counter, after the list)', hb_monitor.StopForRecovery('a failure list arrived'), False, True),
        ('any exception once the stop flag is up', RuntimeError('anything'), True, True),
        ('a peer gone: DistBackendError', dist.DistBackendError('NCCL error: remote process exited or there was a network '
                                                               'error'), False, True),
        ('a peer gone: RCCL text in a RuntimeError', RuntimeError('NCCL communicator was aborted on rank 3'), False, True),
        ('a HIP fault raised as AcceleratorError (E3 hip arm)', acc, False, False),
        ('out of memory', torch.OutOfMemoryError('HIP out of memory. Tried to allocate 2.00 TiB'), False, False),
        ('out of memory inside the communication library', RuntimeError('NCCL error: out of memory'), False, False),
        ('a program error', KeyError('lm loss'), False, False),
        ('the synthetic hardware fault (hwfault)', RuntimeError('ftv2 injected (synthetic) hardware fault: HW_EXCEPTION: '
                                                                'GPU lost'), False, False),
        ('SystemExit before any stop', SystemExit(3), False, False),
    ]
    for name, e, stopping, want in cases:
        check(f'waits for the stop: {name}', waits_for_stop(e, stopping), want)
    import hb_rules
    check('death_kind of the HIP fault record', hb_rules.death_kind(f'exception: {type(acc).__name__}: {ill[:200]}'), 'exception')
    check('death_kind of the hwfault record',
          hb_rules.death_kind('exception: RuntimeError: ftv2 injected (synthetic) hardware fault: HW_EXCEPTION: GPU lost'),
          'hardware fault')
    check('death_kind of a SIGABRT death (hipabort: no exit record)', hb_rules.death_kind('signal'), 'killed')


def worker(rank, endpoints):
    import torch
    import torch.distributed as dist
    import rc_world
    import rc_stop
    from hb_etcd import Etcd
    e = Etcd(endpoints.split(','))
    log = lambda m: print(f'[rank {rank}] {m}', flush=True)
    for g, host in (('a', 1), ('b', 2)):
        rc_world.make_world(e, g, rank, 4, host, 60, log, backend='gloo')
        t = torch.ones(1) * (rank + 1)
        dist.all_reduce(t)
        assert t.item() == 10, t
        dist.barrier()
        rc_stop.forget_all()
        assert not dist.is_initialized()
    print(f'[rank {rank}] both generations ok', flush=True)


def pg_timeout_cases():
    """Recovery.pg_timeouts: every group gets at least FTV2_PG_TIMEOUT_S (V3-2L 5863006: Megatron's and DeepSpeed's groups took
    torch's 10-minute default and the watchdog ended the survivors before any decision). torch's own functions are replaced by
    recorders first, so no group is made."""
    import torch.distributed as dist
    import torch.distributed.distributed_c10d as c10d
    from datetime import timedelta
    from rc_controller import Recovery
    seen = []
    keep = (dist.new_group, dist.init_process_group, c10d.new_group)
    dist.new_group = lambda *a, **k: seen.append(('new_group', a, k))
    dist.init_process_group = lambda *a, **k: seen.append(('init', a, k))
    try:
        os.environ['FTV2_PG_TIMEOUT_S'] = '1800'
        Recovery.pg_timeouts()
        dist.new_group([0, 1])
        dist.new_group([0, 1], backend='gloo')
        dist.new_group([0, 1], timedelta(seconds=60))
        dist.new_group([0, 1], timeout=timedelta(hours=2))
        dist.init_process_group('nccl', timeout=timedelta(minutes=15))
        dist.init_process_group('nccl')
        got = [(k.get('timeout') if len(a) < 2 else a[1]).total_seconds() for _, a, k in seen]
        check('group timeouts raised to the floor, a longer one kept', got, [1800.0, 1800.0, 1800.0, 7200.0, 1800.0, 1800.0])
        check('c10d.new_group is the wrapped one', c10d.new_group is dist.new_group, True)
    finally:
        dist.new_group, dist.init_process_group, c10d.new_group = keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--etcd-bin', default='/lustre/orion/world-shared/gen150/zixianw4/tools/etcd-v3.5.17/etcd-v3.5.17-linux-amd64/etcd')
    ap.add_argument('--worker', nargs=2)
    a = ap.parse_args()
    if a.worker:
        worker(int(a.worker[0]), a.worker[1])
        return
    decide_cases()
    wrapper_cases()
    pg_timeout_cases()
    tmp = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'cpu_world_{os.getpid()}')
    c, p = 24700 + os.getpid() % 200, 24900 + os.getpid() % 200
    env = {k: v for k, v in os.environ.items() if 'proxy' not in k.lower()}
    etcd = subprocess.Popen([a.etcd_bin, '--name', 'm0', '--data-dir', tmp, '--listen-client-urls', f'http://127.0.0.1:{c}',
                             '--advertise-client-urls', f'http://127.0.0.1:{c}', '--listen-peer-urls', f'http://127.0.0.1:{p}',
                             '--initial-advertise-peer-urls', f'http://127.0.0.1:{p}', '--initial-cluster', f'm0=http://127.0.0.1:{p}',
                             '--log-level', 'warn'], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        from hb_etcd import Etcd
        for _ in range(50):
            try:
                Etcd([f'127.0.0.1:{c}']).put('probe', '1')
                break
            except Exception:
                time.sleep(0.2)
        ps = [subprocess.Popen([sys.executable, os.path.abspath(__file__), '--worker', str(r), f'127.0.0.1:{c}'], env=env)
              for r in range(4)]
        codes = [q.wait(timeout=300) for q in ps]
        check('two generations of the world through etcd (gloo), 4 processes', codes, [0, 0, 0, 0])
        v, _ = Etcd([f'127.0.0.1:{c}']).get('world/b')
        check('world/b names a host:port', bool(v and ':' in v), True)
    finally:
        etcd.kill()
        etcd.wait()
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'\n{len(fails)} failed' if fails else '\nall passed')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
