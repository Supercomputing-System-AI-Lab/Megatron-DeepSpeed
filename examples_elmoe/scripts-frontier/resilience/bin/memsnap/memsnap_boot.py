"""memsnap_boot.py <script.py> [args] (2026-09-14, run M1): run a training script with
(1) the CUDA peak-memory counters never reset, so every later torch.cuda.max_memory_allocated() / max_memory_reserved() (and the rank
    logs' "max allocated" / "max reserved") is the peak since the process started;
(2) on rank MEMSNAP_RANK (default 0) the caching allocator's history recorded (torch.cuda.memory._record_memory_history) and dumped at
    exit to MEMSNAP_OUT/rank<r>_snapshot_exit.pickle; bin/memsnap_peak.py reads it.
The model code is not touched: the script runs through runpy as __main__."""
import atexit, os, runpy, sys, time
import torch


def _noop(*args, **kwargs):
    return None


for _mod in (torch.cuda, torch.cuda.memory):
    for _name in ('reset_peak_memory_stats', 'reset_max_memory_allocated', 'reset_max_memory_cached'):
        if hasattr(_mod, _name):
            setattr(_mod, _name, _noop)
RANK = os.environ.get('SLURM_PROCID', '0')
OUT = os.environ.get('MEMSNAP_OUT', '.')


def _report():
    try:
        st = torch.cuda.memory_stats()
        print(f'[memsnap] rank {RANK} at exit: max allocated {torch.cuda.max_memory_allocated() / 2 ** 30:.3f} GiB, '
              f'max reserved {torch.cuda.max_memory_reserved() / 2 ** 30:.3f} GiB (never reset); allocator retries {st.get("num_alloc_retries")}, '
              f'ooms {st.get("num_ooms")}, sync-all-streams {st.get("num_sync_all_streams")}, device allocs {st.get("num_device_alloc")}, '
              f'device frees {st.get("num_device_free")}', flush=True)
    except Exception as e:
        print(f'[memsnap] rank {RANK} report failed: {type(e).__name__}: {e}', flush=True)


# [2026-10-10, resilience round 6] THE PEAK WRITER: every MEMSNAP_PEAK_EVERY_S seconds (30) a daemon thread rewrites MEMSNAP_OUT/peak_rank<r>.txt
# with this rank's never-reset max_memory_allocated and max_memory_reserved (GiB) and the time, so that the restart loop can read every
# rank's peak after an attempt that ended by os._exit (the detectors' exit path skips atexit, and srun kills the other ranks): the file is
# at most one period old, and the peak counters only grow. One small atomic write per rank per period: nothing a 512-rank job notices.
def _peak_writer():
    period = float(os.environ.get('MEMSNAP_PEAK_EVERY_S', '30'))
    try:
        os.makedirs(OUT, exist_ok=True)
    except Exception:
        return
    path = os.path.join(OUT, f'peak_rank{RANK}.txt')
    while True:
        try:
            if torch.cuda.is_available():
                body = f'{torch.cuda.max_memory_allocated() / 2 ** 30:.3f} {torch.cuda.max_memory_reserved() / 2 ** 30:.3f} {time.time():.0f}\n'
                with open(path + '.tmp', 'w') as fh:
                    fh.write(body)
                os.replace(path + '.tmp', path)
        except Exception:
            pass
        time.sleep(period)


if os.environ.get('MEMSNAP_PEAK_EVERY_S', '30') not in ('0', 'off', 'none'):
    import threading
    threading.Thread(target=_peak_writer, name='memsnap-peak', daemon=True).start()

if RANK in os.environ.get('MEMSNAP_RANK', '0').split(','):      # one rank or a comma-separated list (2026-09-15: one rank per chosen pipeline stage of the 671B); 'none' = no history on any rank (2026-10-10)
    torch.cuda.memory._record_memory_history(enabled='all', context='alloc', stacks='python',
                                             max_entries=int(os.environ.get('MEMSNAP_ENTRIES', '4000000')))

    def _dump():
        try:
            os.makedirs(OUT, exist_ok=True)
            path = os.path.join(OUT, f'rank{RANK}_snapshot_exit.pickle')
            t0 = time.time()
            torch.cuda.memory._dump_snapshot(path)
            print(f'[memsnap] rank {RANK} wrote {path} in {time.time() - t0:.1f} s', flush=True)
        except Exception as e:
            print(f'[memsnap] rank {RANK} dump failed: {type(e).__name__}: {e}', flush=True)
    atexit.register(_dump)
atexit.register(_report)          # runs before the dump (atexit is last-in, first-out)
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
runpy.run_path(sys.argv[0], run_name='__main__')
