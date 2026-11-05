"""hb_monitor.py (ft-sys-v2 heartbeat system, 2026-11-05): the only part that runs inside a training process.

Two pieces (design map 1.1, 1.5):
  install_counter()  wraps the collective and point-to-point calls of torch.distributed once, so every call adds 1 to this rank's
                     call count first. DeepSpeed (deepspeed/comm/torch.py) and Megatron reach these calls as attributes of
                     torch.distributed at call time, so one wrap counts both. Call it before the first communication call.
  start()            the monitor thread: every BEAT seconds it writes this rank's line to host memory (hb_beat.rank_path);
                     every 0.5 s it checks for the failure list the node's agent copies from etcd (hb_beat.failed_path).
The rules of map node 1.5: no GPU call for the monitor (HBM comes from the allocator's own counters, which live in CPU memory, as
in ft-sys-v1's hb.py; no .item(), no synchronize); one small file write per beat. The trainer tells the monitor its step, phase
and loss with set_step / set_phase / set_loss, which only set Python values.
Tonight's scope is the heartbeat and its decisions: on a failure list the monitor records it and calls the trainer's callback
if one was given; the stop (RCCL revoke, abort) comes with the recovery stage.
Fault injection for tests (FTV2_INJECT, comma-separated): hang:<rank>:<step>, kill:<rank>:<step>, exit:<rank>:<step>,
oom:<rank>:<step>, oomstuck:<rank>:<step> (the same, then the exit blocks), abort:<rank>:<step>,
pause:all:<step>:<seconds>, slow:<rank>:<step>:<milliseconds>. Each fires before a communication call of that step."""
import atexit
import json
import os
import signal
import sys
import threading
import time

from hb_beat import exit_path, failed_path, node_name, rank_path, read_json, shm_dir, write_json

BEAT = float(os.environ.get('FTV2_RANK_BEAT_S', '5'))
CHECK = 0.5
COLLECTIVES = ['all_reduce', 'all_gather', 'all_gather_into_tensor', '_all_gather_base', 'reduce_scatter', 'reduce_scatter_tensor',
               '_reduce_scatter_base', 'all_to_all', 'all_to_all_single', 'broadcast', 'reduce', 'barrier', 'send', 'recv',
               'isend', 'irecv', 'batch_isend_irecv', 'gather', 'scatter', 'all_gather_object', 'broadcast_object_list']

S = {'calls': 0, 'step': -1, 'micro_batch': -1, 'phase': 'start', 'loss': None, 'step_s': None, 'rank': -1, 'local_rank': -1,
     'role': 'train', 'started': False, 'installed': False, 'failure': None, 'card': None, 'tid': None, 'dir': None,
     'on_failure': None, 'inject': [], 'exit_written': False}


def _parse_inject():
    out = []
    for item in filter(None, os.environ.get('FTV2_INJECT', '').split(',')):
        f = item.split(':')
        out.append({'kind': f[0], 'rank': f[1], 'step': int(f[2]), 'arg': float(f[3]) if len(f) > 3 else 0.0, 'done': False})
    return out


def _inject():
    """Run before every counted call: fire any injection due at this rank and step."""
    for j in S['inject']:
        if j['done'] or S['step'] < j['step'] or (j['rank'] != 'all' and int(j['rank']) != S['rank']):
            continue
        k = j['kind']
        if k == 'slow':
            time.sleep(j['arg'] / 1000.0)              # every call from this step on: a slow rank, not a hang
            continue
        j['done'] = True
        _log(f'injecting {k} at step {S["step"]}, call {S["calls"]}')
        if k == 'hang':
            while True:                                # before the count: this rank stays behind, its process lives on
                time.sleep(3600)
        if k == 'kill':
            os.kill(os.getpid(), signal.SIGKILL)       # a crash by signal: no exit record
        if k == 'exit':
            raise RuntimeError('ftv2 injected exit')   # a crash by exception: the exit record says why
        if k in ('oom', 'oomstuck'):                   # a real allocation the GPU cannot hold; on the CPU, the same error
            if k == 'oomstuck':                        # then an exit that never ends, as PMPI_Finalize's barrier in HB3 5855079
                atexit.register(time.sleep, 10 ** 6)
            import torch
            if torch.cuda.is_available():
                torch.empty(int(2e12), dtype=torch.uint8, device=f'cuda:{S["local_rank"]}')
            raise torch.OutOfMemoryError('ftv2 injected: out of memory on the CPU test (no GPU)')
        if k == 'abort':
            os.abort()                                 # SIGABRT and no exit record, as a HIP memory fault ends a process
        if k == 'pause':
            time.sleep(j['arg'])                       # every rank pauses at the same call: equal counts


def _count():
    if S['inject']:
        _inject()
    S['calls'] += 1


def install_counter():
    """Wrap torch.distributed's communication calls so each one counts. Idempotent."""
    if S['installed']:
        return
    import torch.distributed as dist
    S['inject'] = _parse_inject()
    for name in COLLECTIVES:
        f = getattr(dist, name, None)
        if f is None or getattr(f, '_ftv2_counted', False):
            continue

        def wrapped(*a, _f=f, **k):
            _count()
            return _f(*a, **k)
        wrapped._ftv2_counted = True
        wrapped.__name__, wrapped.__doc__, wrapped.__wrapped__ = f.__name__, f.__doc__, f
        setattr(dist, name, wrapped)
    S['installed'] = True


def set_step(step, micro_batch=None, step_s=None):
    S['step'] = int(step)
    if micro_batch is not None:
        S['micro_batch'] = int(micro_batch)
    if step_s is not None:
        S['step_s'] = round(float(step_s), 3)


def set_phase(phase):
    S['phase'] = phase


def set_loss(loss):
    """A loss the trainer already has on the CPU (a Python float); never a GPU tensor."""
    S['loss'] = loss if isinstance(loss, (int, float)) else None


def _log(msg):
    print(f'[ftv2 monitor rank {S["rank"]}] {time.strftime("%H:%M:%S")} {msg}', file=sys.stderr, flush=True)


def _hbm():
    try:
        import torch
        return (round(torch.cuda.memory_allocated(S['local_rank']) / 1e9, 2),
                round(torch.cuda.max_memory_allocated(S['local_rank']) / 1e9, 2))
    except Exception:
        return -1.0, -1.0


def line():
    hbm, peak = _hbm()
    return {'t': round(time.time(), 3), 'rank': S['rank'], 'local_rank': S['local_rank'], 'node': node_name(), 'pid': os.getpid(),
            'tid': S['tid'], 'run': os.environ.get('FTV2_RUN') or os.environ.get('SLURM_STEP_ID', '0'), 'step': S['step'],
            'micro_batch': S['micro_batch'], 'phase': S['phase'], 'calls': S['calls'], 'step_s': S['step_s'], 'loss': S['loss'],
            'hbm_gb': hbm, 'hbm_peak_gb': peak, 'card': S['card'], 'role': S['role']}


def _write_exit(reason, code):
    if S['exit_written'] or S['dir'] is None:
        return
    S['exit_written'] = True
    try:
        write_json(exit_path(S['dir'], S['rank']), {'t': round(time.time(), 3), 'rank': S['rank'], 'pid': os.getpid(),
                                                     'reason': reason, 'code': code, 'step': S['step'], 'calls': S['calls']})
    except Exception:
        pass


def _find_card(local_rank):
    """The sysfs folder of this rank's GPU, so the agent can read its xGMI error count (ft-sys-v1 hb.py _find_card). One device
    property query at start, not per beat."""
    try:
        import glob
        import torch
        bus = getattr(torch.cuda.get_device_properties(local_rank), 'pci_bus_id', None)
        if bus is None:
            return None
        for d in glob.glob('/sys/class/drm/card*/device/uevent'):
            for ln in open(d):
                if ln.startswith('PCI_SLOT_NAME=') and int(ln.strip().split('=')[1].split(':')[1], 16) == int(bus):
                    return os.path.dirname(d)
    except Exception:
        return None
    return None


def _loop():
    path = rank_path(S['dir'], S['rank'])
    fpath = failed_path(S['dir'])
    next_beat, seen = 0.0, None
    while True:
        now = time.time()
        if now >= next_beat:
            try:
                write_json(path, line())
            except Exception as e:
                _log(f'beat write failed: {e}')
            next_beat = now + BEAT
        try:
            st = os.stat(fpath)
            if (st.st_mtime, st.st_size) != seen:
                seen = (st.st_mtime, st.st_size)
                f = read_json(fpath)
                my_run = os.environ.get('FTV2_RUN') or os.environ.get('SLURM_STEP_ID', '0')
                if f and str(f.get('run')) != str(my_run):
                    f = None                                   # a list of an earlier run on this node: not ours
                if f and f != S['failure']:
                    S['failure'] = f
                    _log(f'failure list {f.get("gen")}: {json.dumps(f)[:300]}')
                    if S['on_failure']:
                        S['on_failure'](f)
        except FileNotFoundError:
            pass
        time.sleep(CHECK)


def start(rank, local_rank, role='train', on_failure=None):
    """Start the monitor thread (once per process). on_failure(list) is called from the monitor thread when a failure list
    arrives (tonight: for tests and logging; the stop comes with the recovery stage)."""
    if S['started']:
        return
    S.update(rank=int(rank), local_rank=int(local_rank), role=role, on_failure=on_failure, dir=shm_dir(),
             tid=threading.main_thread().native_id, started=True)
    S['card'] = _find_card(S['local_rank']) if role == 'train' else None
    prev = sys.excepthook

    def hook(t, v, tb):
        _write_exit(f'exception: {t.__name__}: {str(v)[:200]}', 1)
        prev(t, v, tb)
    sys.excepthook = hook
    atexit.register(lambda: _write_exit('finished', 0))
    threading.Thread(target=_loop, name='ftv2-monitor', daemon=True).start()
    _log(f'monitor on: {rank_path(S["dir"], S["rank"])} every {BEAT:.0f} s; counter installed: {S["installed"]}')
