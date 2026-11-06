"""hb_beat.py (ft-sys-v2 heartbeat system, 2026-11-05): the shapes of the beats and where they live on a node.

A rank's line (written by its monitor thread into host memory, read by its node's agent):
  t, rank, local_rank, node, pid, tid, run, step, micro_batch, phase, calls, step_s, loss, hbm_gb, hbm_peak_gb, card, role
A node's small beat (written by the agent into etcd, read by the coordinator; design map 1.2, the fast heartbeat):
  t, node, run, step, low, low_rank, high, ranks, rank_ids, dead, phases, role
The node's files live in /dev/shm/ftv2_<job>/ (host memory, no network): rank_<r>.json, exit_<r>.json, thread_<r>.json,
failed.json.
All writes go to a temporary file first and are renamed into place, so a reader never sees half a line."""
import json
import os
import socket

RANK_FIELDS = ['t', 'rank', 'local_rank', 'node', 'pid', 'tid', 'run', 'step', 'micro_batch', 'phase', 'calls', 'step_s', 'loss',
               'hbm_gb', 'hbm_peak_gb', 'card', 'role']
NODE_FIELDS = ['t', 'node', 'run', 'step', 'low', 'low_rank', 'high', 'ranks', 'rank_ids', 'dead', 'phases', 'role']


def job_id():
    return os.environ.get('SLURM_JOB_ID', 'nojob')


def node_name():
    return os.environ.get('FTV2_NODE') or socket.gethostname().split('.')[0]


def shm_dir(job=None):
    d = os.environ.get('FTV2_SHM_DIR') or f'/dev/shm/ftv2_{job or job_id()}'
    os.makedirs(d, exist_ok=True)
    return d


def write_json(path, obj):
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, separators=(',', ':'))
    os.replace(tmp, path)


def read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def rank_path(d, rank):
    return os.path.join(d, f'rank_{rank}.json')


def exit_path(d, rank):
    return os.path.join(d, f'exit_{rank}.json')


def thread_path(d, rank):
    return os.path.join(d, f'thread_{rank}.json')


def failed_path(d):
    return os.path.join(d, 'failed.json')


def node_beat(node, lines, dead, now, role='train', dead_why=None, thread_errors=None):
    """The node's small beat from its ranks' lines: the lowest call count and the rank that has it, the highest count, the
    ranks seen, the ranks whose process is gone. Ranks that are gone do not count for low and high: their counts stop for a
    reason the agent already knows (they are reported in dead)."""
    live = [ln for ln in lines if ln.get('rank') not in dead]
    calls = [(ln.get('calls', 0), ln.get('rank')) for ln in live]
    low, low_rank = min(calls) if calls else (None, None)
    high = max(c for c, _ in calls) if calls else None
    runs = sorted({str(ln.get('run')) for ln in lines})
    return {'t': round(now, 3), 'node': node, 'run': runs[-1] if runs else None,
            'step': min((ln.get('step', -1) for ln in live), default=-1), 'low': low, 'low_rank': low_rank, 'high': high,
            'ranks': len(lines), 'rank_ids': sorted(ln.get('rank') for ln in lines), 'dead': sorted(dead), 'dead_why': dead_why or {},
            'thread_errors': thread_errors or {},
            'phases': sorted({ln.get('phase', '?') for ln in live}), 'role': role,
            'stage': min((ln.get('stage') for ln in live if ln.get('stage') is not None), default=None)}
