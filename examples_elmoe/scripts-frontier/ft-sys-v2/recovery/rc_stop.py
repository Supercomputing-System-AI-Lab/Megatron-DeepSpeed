"""rc_stop.py (ft-sys-v2 recovery, 2026-11-06): the stop. Framework-free: torch.distributed's registry and RCCL only.

A port of the toy's stop (assistant_deep_dive/2026-10-30-standby_join/toy/recovery_core.py:203-335), tested there on RCCL 2.30.4
(job 5841333: the stop finished on 23 of 23 ranks for five failure kinds). Three parts, in this order, all from the heartbeat's
monitor thread (the main thread may be blocked inside a GPU wait):
  1. revoke    ncclCommRevoke on every communicator of every process group at once, one thread each (one after another deadlocked
               2 of 23 ranks, jobs 5839903, 5840015); the kernels waiting for the lost ranks end, the main thread's GPU wait
               returns, and its next counted call raises StopForRecovery (hb_monitor._count).
  2. barrier   in etcd: stopped/<g>/<rank>, then wait until every surviving rank has written its key (60 s at most).
  3. abort     every backend's abort() at once, one thread each, then PyTorch's registry is cleared (_abort_process_group, or by
               hand if an abort does not return). The old communicators are freed only after every survivor stopped using them.
A one-rank 'keeper' communicator made at start (outside the registry, never aborted) keeps the network plugin loaded through the
abort (on RCCL 2.22.3 aborting a process's last communicator closed it; 2.30.4 has not been tested without one).
Without ncclCommRevoke in the loaded RCCL, the stop is the abort alone (passed twice on 2.30.4: jobs 5841353, 5841367)."""
import ctypes
import json
import os
import threading
import time
from datetime import timedelta

import torch
import torch.distributed as dist

_LIB = []
_LEFT_BEHIND = []          # old groups kept referenced when an abort did not return: no destructor runs on a stuck communicator
_KEEPER = []
_THREADS = []              # every revoke and abort thread of the stops (2026-11-06, M3C12 b_lost): none may be alive when the
                           # old engine and communicators are freed


def threads_alive():
    """How many revoke or abort threads are still running (inside RCCL)."""
    return sum(t.is_alive() for t in _THREADS)


def _now():
    return round(time.time(), 3)


def rccl():
    """The RCCL already loaded in the process (LD_PRELOAD by its bare name), or None when it has no revoke."""
    if not _LIB:
        try:
            lib = ctypes.CDLL('librccl.so')
            lib.ncclCommRevoke.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.ncclCommGetAsyncError.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
            _LIB.append(lib)
        except (OSError, AttributeError):
            _LIB.append(None)
    return _LIB[0]


def make_keeper(device):
    """A one-rank communicator outside the registry (recovery_core.py:321-335): made once at start, never aborted."""
    if _KEEPER or device.type != 'cuda':
        return
    t0 = time.time()
    store = dist.HashStore()
    try:
        pg = dist.ProcessGroupNCCL(store, 0, 1, timedelta(seconds=120))
    except TypeError:
        opts = dist.ProcessGroupNCCL.Options()
        opts._timeout = timedelta(seconds=120)
        pg = dist.ProcessGroupNCCL(store, 0, 1, opts)
    one = torch.ones(1, device=device)
    pg.allreduce([one]).wait()
    torch.cuda.synchronize(device)
    _KEEPER.append((pg, store))
    return round(time.time() - t0, 3)


def _backends(device):
    out = []
    for pg in list(dist.distributed_c10d._world.pg_map.keys()):
        try:
            out.append(pg._get_backend(device))
        except Exception:
            pass
    return out


def revoke_all(device, log, limit_s=10.0, groups=None):
    """ncclCommRevoke on every communicator (or on those of `groups`: the groups that hold a lost rank, as C's stop does), one
    thread each. No device synchronization (2026-11-06: a thread waiting for the GPU while the old engine is freed is not
    needed; C's stop never waits for the GPU). Returns a summary dict."""
    t0 = time.time()
    lib = rccl()
    ptrs = []
    bs = _backends(device) if groups is None else [b for b in (_backend_of(pg, device) for pg in groups) if b is not None]
    for b in bs:
        try:
            p = b._comm_ptr()
            if p:
                ptrs.append(p)
        except Exception:
            pass                              # a group whose communicator was never made
    if lib is None or not ptrs:
        return {'revoked': 0, 'all_done': lib is not None, 'gpu_idle': None, 's': round(time.time() - t0, 3), 'no_revoke': lib is None}
    done = [False] * len(ptrs)
    ret = [None] * len(ptrs)                   # seconds until each ncclCommRevoke call returned (None: still inside it)

    def one(i):
        lib.ncclCommRevoke(ctypes.c_void_p(ptrs[i]), 0)
        ret[i] = round(time.time() - t0, 3)
        st = ctypes.c_int(-1)
        while time.time() - t0 < limit_s:
            lib.ncclCommGetAsyncError(ctypes.c_void_p(ptrs[i]), ctypes.byref(st))
            if st.value == 0:
                done[i] = True
                return
            time.sleep(0.01)
    ths = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(len(ptrs))]
    _THREADS.extend(ths)
    [t.start() for t in ths]
    [t.join(timeout=max(0.1, limit_s - (time.time() - t0))) for t in ths]
    r = {'revoked': len(ptrs), 'all_done': all(done), 'alive': sum(t.is_alive() for t in ths), 's': round(time.time() - t0, 3),
         'returned_s': ret}
    log(f'revoke: {r}')
    return r


def _backend_of(pg, device):
    try:
        return pg._get_backend(device)
    except Exception:
        return None


def etcd_barrier(etcd, g, rank, survivors, log, limit_s=60.0):
    """stopped/<g>/<rank> in etcd, then wait until every survivor wrote its key (or the limit)."""
    t0 = time.time()
    etcd.put(f'stopped/{g}/{rank}', json.dumps({'t': _now()}))
    want = set(survivors)
    have = set()
    while time.time() - t0 < limit_s:
        got, _ = etcd.range(f'stopped/{g}/')
        have = {int(k.rsplit('/', 1)[1]) for k in got}
        if want <= have:
            break
        time.sleep(0.1)
    r = {'arrived': len(want & have), 'of': len(want), 's': round(time.time() - t0, 3)}
    log(f'etcd barrier: {r}')
    return r


def forget_all():
    """PyTorch's registry cleared by hand, without aborting again (recovery_core.py:281-291)."""
    c = dist.distributed_c10d
    _LEFT_BEHIND.append(list(c._world.pg_map.keys()))
    c._update_default_pg(None)
    for name in ('pg_map', 'pg_names', 'pg_group_ranks', 'pg_backend_config', 'pg_to_tag', 'tags_to_pg', 'pg_coalesce_state'):
        getattr(c._world, name).clear()
    c._unregister_all_process_groups()
    c._world.group_count = 0


def dump_stuck(path, log):
    """The native stacks of every thread of this process (py-spy --native, when it is installed), written to path before a rank
    leaves because a revoke or abort did not return: the next run names the call it waits in."""
    import subprocess
    exe = os.path.expanduser('~/.local/bin/py-spy')
    if not os.path.exists(exe):
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = subprocess.run([exe, 'dump', '--native', '--pid', str(os.getpid())], capture_output=True, text=True, timeout=60)
        with open(path, 'w') as f:
            f.write(out.stdout + out.stderr)
        log(f'stacks of the stuck stop written to {path}')
    except Exception as e:
        log(f'stack dump failed: {e!r}'[:200])


def abort_all(device, log, limit_s=30.0, dump_path=None):
    """Every backend's abort() at once, one thread each (one batch: an abort frees device memory, so it waits behind any kernel on
    its GPU, and the abort that ends that kernel must run at the same time, M3C0c). Then every revoke and abort thread of the
    stop must have returned (rc_stop._THREADS) before the registry is cleared and the old engine freed: a thread still inside
    RCCL would touch freed memory (M3C12 5872105 b_lost: four ranks died in glibc, 'The futex facility returned an unexpected
    error code', after the recovery went on past aborts that had not returned). Returns 'ok', or 'stuck' with the registry left as
    it is: the caller must not free anything and must leave (the rank is treated as lost)."""
    t0 = time.time()
    bs = _backends(device)
    ret = [None] * len(bs)                     # seconds until each abort returned (None: still inside RCCL at the limit)

    def one(i, b):
        b.abort()
        ret[i] = round(time.time() - t0, 3)
    ths = [threading.Thread(target=one, args=(i, b), daemon=True) for i, b in enumerate(bs)]
    _THREADS.extend(ths)
    [t.start() for t in ths]
    for t in _THREADS:
        t.join(timeout=max(0.1, limit_s - (time.time() - t0)))
    stuck = sum(t.is_alive() for t in ths)
    alive = threads_alive()
    secs = round(time.time() - t0, 3)
    if alive:
        _LEFT_BEHIND.append(bs)
        if dump_path:
            dump_stuck(dump_path, log)
        log(f'abort: stuck, {len(bs)} backends, {stuck} aborts and {alive - stuck} revokes still running after {secs:.3f} s: '
            f'nothing is freed; this rank leaves; seconds per abort {ret}')
        return {'abort': 'stuck', 'backends': len(bs), 'stuck': stuck, 'alive': alive, 's': secs, 'returned_s': ret}
    res = 'partial'
    try:
        dist.distributed_c10d._abort_process_group()
        res = 'ok'
    except Exception as e:
        log(f'registry clear after abort: {e!r}'[:300])
    if res != 'ok':
        forget_all()
    log(f'abort: {res}, {len(bs)} backends, every abort returned, {time.time() - t0:.3f} s; seconds per abort {ret}')
    return {'abort': res, 'backends': len(bs), 'stuck': 0, 'alive': 0, 's': round(time.time() - t0, 3), 'returned_s': ret}
