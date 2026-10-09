# profiler_manager.py
import os
import torch
import torch.distributed as dist
import json 
import glob 
from torch.profiler import profile, record_function, ProfilerActivity, schedule

_PROF = None
_PENDING = None   # the profiler whose export is deferred to the end of training (PROFILE_EXPORT_AT_END=1, 2026-10-06)

def merge_traces(ranks=None, gzip_out=None):
    """Merge the per-rank Chrome traces (trace_rank<r>_of_<w>.json under $profile_dir) into ONE file that ui.perfetto.dev loads.
    2026-09-18: (1) a subset of ranks -- `ranks` or the env PROFILE_MERGE_RANKS: a comma list ("0,16,32,48"), "stages" (the first
    rank AND the first rank + 4 of every pipeline stage, from PROFILE_MERGE_PP_SIZE or the pipeline group; two ranks per stage for
    cross reference) or "all" (the old behaviour: 64 x 90 MB = a
    file no browser opens); default "stages". (2) gzip (PROFILE_MERGE_GZIP=1, the default): Perfetto reads .json.gz directly and the
    file is ~20x smaller. (3) the pid is 1000 + the RANK parsed from the file name (the old code numbered the files in sorted-string
    order, so rank 10 came before rank 2 and the process labels were wrong beyond 9 ranks), with a process_sort_index so the rows
    appear in rank order. Output: merged_ranks_<r>_<r>....json[.gz] next to the per-rank files; the path is returned.
    Call it only when every rank has finished exporting (an iteration boundary, or the end of training)."""
    import gzip, re
    trace_dir = os.path.abspath(os.getenv("profile_dir"))
    files = {}
    for f in glob.glob(os.path.join(trace_dir, "trace_rank*_of_*.json")) + glob.glob(os.path.join(trace_dir, "trace_rank*_of_*.json.gz")):
        m = re.search(r"trace_rank(\d+)_of_(\d+)\.json(\.gz)?$", f)
        if m: files[int(m.group(1))] = f
    if not files:
        print("No trace files found to merge."); return None
    world = int(re.search(r"_of_(\d+)\.json(\.gz)?$", next(iter(files.values()))).group(1))
    def _load(f):
        with (gzip.open(f, "rt") if f.endswith(".gz") else open(f)) as fh: return json.load(fh)
    sel = ranks if ranks is not None else os.getenv("PROFILE_MERGE_RANKS", "stages")
    if isinstance(sel, str):
        if sel == "all": ranks = sorted(files)
        elif sel == "stages":
            pp = int(os.getenv("PROFILE_MERGE_PP_SIZE", "0") or 0)
            if pp <= 0:
                try:
                    from megatron.core import mpu; pp = mpu.get_pipeline_model_parallel_world_size()
                except Exception: pp = 1
            stride = world // pp   # the first rank and the first rank + 4 of every stage (the user, 2026-09-18: two ranks per stage for cross reference)
            ranks = [r for s in range(pp) for r in (s * stride, s * stride + 4) if r < (s + 1) * stride]
        else: ranks = [int(x) for x in sel.split(",") if x.strip().isdigit()]
    ranks = [r for r in ranks if r in files]
    if gzip_out is None: gzip_out = os.getenv("PROFILE_MERGE_GZIP", "1") == "1"
    out = os.path.join(trace_dir, "merged_ranks_" + "_".join(map(str, ranks)) + (".json.gz" if gzip_out else ".json"))
    events = []; meta = None
    for r in ranks:
        data = _load(files[r])
        meta = meta or {k: v for k, v in data.items() if k != "traceEvents"}
        pid = 1000 + r
        events.append({"name": "process_name", "ph": "M", "pid": pid, "tid": 0, "args": {"name": f"rank {r}"}})
        events.append({"name": "process_sort_index", "ph": "M", "pid": pid, "tid": 0, "args": {"sort_index": r}})
        for e in data["traceEvents"]:
            if e.get("ph") == "M" and e.get("name") == "process_name": continue
            e["pid"] = pid; events.append(e)
        print(f"[profiler_manager.py] merge: rank {r} ({os.path.getsize(files[r]) // 1048576} MB) added", flush=True)
    merged = dict(meta or {}); merged["traceEvents"] = events
    opener = (lambda: gzip.open(out, "wt")) if gzip_out else (lambda: open(out, "w"))
    with opener() as f: json.dump(merged, f)
    print(f"[profiler_manager.py] merged {len(ranks)} of {len(files)} rank traces into {out} ({os.path.getsize(out) // 1048576} MB)", flush=True)
    return out


def finalize_if_pending():
    """PROFILE_EXPORT_AT_END=1 (2026-10-06): export the deferred trace on the ranks that traced, close the profiler, then rank 0 merges
    the traced ranks' files (PROFILE_MERGE_RANKS, default = PROFILE_RANKS). Every rank must call this: two barriers."""
    global _PENDING
    if os.getenv('to_profile') != 'True' or os.getenv('PROFILE_EXPORT_AT_END') != '1':
        return
    if _PENDING is not None:
        _export_trace(_PENDING)
        _PENDING = None
    finalize_profiler(merge=False)
    if dist.is_available() and dist.is_initialized(): dist.barrier()
    rank, _ = _get_rank_world()
    if rank == 0:
        sel = os.getenv('PROFILE_MERGE_RANKS') or os.getenv('PROFILE_RANKS') or 'stages'
        try: merge_traces(ranks=sel)
        except Exception as ex: print(f"[profiler_manager.py] merge of the deferred traces failed: {ex}", flush=True)
    if dist.is_available() and dist.is_initialized(): dist.barrier()


def merge_after_training():
    """The micro-batch window (PROFILE_MICROBATCHES=1) cannot merge inside the iteration (rank 0 would read files still being
    written, and a barrier inside the pipeline schedule deadlocks); call this once after the training loop: every rank has
    exported, a barrier orders the reads after the writes, rank 0 merges the selected ranks (merge_traces). 2026-09-18."""
    finalize_if_pending()   # the deferred per-iteration export (PROFILE_EXPORT_AT_END=1), a no-op otherwise
    if os.getenv("to_profile") != "True" or os.getenv("PROFILE_MICROBATCHES") != "1": return   # the per-iteration mode merges in finalize_profiler
    rank, _ = _get_rank_world()
    if dist.is_available() and dist.is_initialized(): dist.barrier()
    if rank == 0:
        try: merge_traces()
        except Exception as ex: print(f"[profiler_manager.py] merge after training failed: {ex}", flush=True)
    if dist.is_available() and dist.is_initialized(): dist.barrier()


def _get_rank_world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


# def _trace_handler(prof):
#     # Fallback to '.' if 'profile_dir' environment variable is not set
#     out_dir = os.path.abspath(os.getenv('profile_dir', '.'))
#     os.makedirs(out_dir, exist_ok=True)
#     r, w = _get_rank_world()
    
#     # # 1. Safely export chrome trace
#     # fname_json = os.path.join(out_dir, f"trace_rank{r}_of_{w}.json")
#     # try:
#     #     print(f"--- Profiler Exporting chrome trace for rank {r} ---")
#     #     prof.export_chrome_trace(fname_json)
#     # except Exception as e:
#     #     print(f"[Rank {r}] Warning: Failed to export chrome trace: {e}")

#     # # 2. Optional: print summary to stdout
#     # try:
#     #     print(f"--- Profiler Summary for Rank {r} ---")
#     #     print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
#     # except Exception:
#     #     pass

#     # # 3. Safely export memory timeline
#     # fname_html = os.path.join(out_dir, f"memory_trace_rank{r}_of_{w}.html")
#     # try:
#     #     # On ROCm, sometimes specifying the exact device (e.g., "cuda:0") helps,
#     #     # but the primary fix is preventing the empty sequence error from crashing training.
#     #     device_str = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cuda"
#     #     prof.export_memory_timeline(fname_html, device=device_str)
#     # except ValueError:
#     #     print(f"[Rank {r}] Warning: No memory events found to export timeline (ignoring empty sequence).")
#     # except Exception as e:
#     #     print(f"[Rank {r}] Warning: Failed to export memory timeline: {e}")
        
#     # Memory pickle
#     try:
#         pickle_fname = os.path.join(out_dir, f"snapshot_rank{r}_of_{w}.pickle")
#         torch.cuda.memory._dump_snapshot(pickle_fname)
#         if r == 0:
#             print(f"[training.py] Dumped memory snapshot to {pickle_fname}")
#     except Exception as e:
#         print(f"[Rank {r}] Warning: Failed to dump memory snapshot: {e}")

#     # Finally, turn off the memory tracker AFTER dumping
#     try:
#         torch.cuda.memory._record_memory_history(enabled=None)
#     except AttributeError:
#         pass


def rank_selected():
    """PROFILE_RANKS (2026-10-06): 'all' (default) or a comma list of global ranks that run the torch profiler; the other ranks do not
    trace at all (64 ranks x 50-100 MB per traced iteration is a write storm and a memory cost for nothing)."""
    sel = os.getenv('PROFILE_RANKS', 'all').strip()
    if sel.lower() == 'all':
        return True
    r, _ = _get_rank_world()
    return r in {int(x) for x in sel.split(',') if x.strip().lstrip('-').isdigit()}


def _trace_handler(prof):
    global _PENDING
    if os.getenv('PROFILE_EXPORT_AT_END') == '1':
        _PENDING = prof   # exported by finalize_if_pending after the training loop (2026-10-06)
        print(f"--- Profiler: trace of rank {_get_rank_world()[0]} kept in memory, export deferred to the end of training ---", flush=True)
        return
    _export_trace(prof)


def _export_trace(prof):
    out_dir = os.path.abspath(os.getenv('profile_dir', '.'))
    os.makedirs(out_dir, exist_ok=True)
    r, w = _get_rank_world()
    
    
    # 1. Safely export chrome trace
    # 2026-09-18 (the user): the per-rank trace is written gzipped (trace_rank<r>_of_<w>.json.gz, PROFILE_TRACE_GZIP=1, the default;
    # torch's export_chrome_trace gzips itself when the path ends with .gz): ~20x smaller, and Perfetto / the merge read .gz directly.
    want_gz = os.getenv("PROFILE_TRACE_GZIP", "1") == "1"
    fname_json = os.path.join(out_dir, f"trace_rank{r}_of_{w}.json")
    try:
        print(f"--- Profiler Exporting chrome trace for rank {r} ({'.json.gz' if want_gz else '.json'}) ---")
        prof.export_chrome_trace(fname_json)
        if want_gz:   # gzip it ourselves (torch 2.9's export_chrome_trace does not gzip on a .gz suffix), then drop the plain file
            import gzip, shutil
            with open(fname_json, "rb") as fin, gzip.open(fname_json + ".gz", "wb", compresslevel=6) as fout: shutil.copyfileobj(fin, fout, 1 << 20)
            os.remove(fname_json)
            print(f"--- rank {r}: trace gzipped to {fname_json}.gz ({os.path.getsize(fname_json + '.gz') // 1048576} MB) ---")
    except Exception as e:
        print(f"[Rank {r}] Warning: Failed to export chrome trace: {e}")
    # NOTE: memory (.pickle) is NOT dumped here. The CUDA-allocator snapshot is handled
    # standalone by dump_memory_snapshot() (called from training.py), so memory tracing
    # works even when this kineto profiler is off or crashes.


def _snapshot_rank_allowed():
    """Which ranks write a memory snapshot. `mem_snapshot_ranks` env: comma-list of global
    ranks, or 'all'. Default '0' (rank 0 only) -- avoids a write storm of large .pickles
    from every rank. Set e.g. mem_snapshot_ranks="0,8,16" for one rank per pipeline stage."""
    r, _ = _get_rank_world()
    sel = os.getenv('mem_snapshot_ranks', '0').strip()
    if sel.lower() == 'all':
        return True
    allowed = {int(x) for x in sel.split(',') if x.strip().lstrip('-').isdigit()}
    return r in allowed


def enable_memory_history(max_entries=None):
    """Start recording CUDA caching-allocator history (for the .pickle snapshot), on the
    ranks selected by mem_snapshot_ranks. Standalone -- no torch.profiler. Returns True if
    this rank will snapshot. Two environment knobs (2026-09-14): mem_snapshot_max_entries (the
    ring of recorded events, default 2,000,000) and mem_snapshot_stacks ('all' = Python and C++
    frames per event, the default; 'python' = the Python frames only). A 2,000,000-entry ring with
    both stacks took torch's _snapshot() more than 22 minutes to serialize at the dump (run
    5490252), with every other rank waiting at the next collective; 100,000 entries still hold
    several micro-batches, and the Python frames are what the per-module analysis reads."""
    if not _snapshot_rank_allowed():
        return False
    if max_entries is None:
        max_entries = int(os.getenv('mem_snapshot_max_entries', '2000000'))
    stacks = os.getenv('mem_snapshot_stacks', 'all')
    try:
        torch.cuda.memory._record_memory_history(max_entries=max_entries, stacks=stacks)
        return True
    except AttributeError:
        print("Warning: torch.cuda.memory._record_memory_history not available.")
        return False


def dump_memory_snapshot():
    """Dump the CUDA caching-allocator snapshot to <profile_dir>/snapshot_rank{r}.pickle
    and stop recording. Independent of torch.profiler; load at https://pytorch.org/memory_viz."""
    out_dir = os.path.abspath(os.getenv('profile_dir', '.'))
    os.makedirs(out_dir, exist_ok=True)
    r, w = _get_rank_world()
    try:
        pickle_fname = os.path.join(out_dir, f"snapshot_rank{r}_of_{w}.pickle")
        torch.cuda.memory._dump_snapshot(pickle_fname)
        if r == 0:
            print(f"[profiler_manager.py] Dumped memory snapshot to {pickle_fname}", flush=True)
    except Exception as e:
        print(f"[Rank {r}] Warning: Failed to dump memory snapshot: {e}", flush=True)
    try:
        torch.cuda.memory._record_memory_history(enabled=None)
    except AttributeError:
        pass


def _get_profiler(wait=0, warmup=0, active=2, repeat=1):
    global _PROF
    if _PROF is None:
        # Kineto tracks ONLY kernel/timing here. Memory (.pickle) is handled separately by
        # the standalone allocator snapshot (enable_memory_history/dump_memory_snapshot), so
        # memory tracing doesn't depend on this profiler. with_stack/with_flops dropped
        # (heavy at scale); profile_memory=False (kineto memory was a crash suspect on ROCm).
        print(f'[profiler_manager.py]: global _PROF is None, initiating a new one')
        sched = schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        _PROF = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=sched,
            record_shapes=True,
            with_stack=os.getenv('PROFILE_WITH_STACK') == '1',   # the Python call stack per op (TraceLens's nn.Module view needs it; heavy at scale, off by default)
            with_flops=False,
            with_modules=True,
            profile_memory=False,
            on_trace_ready=_trace_handler,
        )
        _PROF.__enter__()
    else:
        print(f'[profiler_manager.py]: loading pre-allocated global _PROF')
    return _PROF

def finalize_profiler(merge=True):
    # merge=False (2026-09-13, the micro-batch window of training.py): the per-rank traces stay unmerged, because this runs in the
    # middle of an iteration and rank 0 would read the other ranks' trace files while they are still being written
    global _PROF
    rank, _ = _get_rank_world()
    print(f'[profiler_manager.py] {rank=} closing profiler')
    if _PROF is not None:
        _PROF.__exit__(None, None, None)
        _PROF = None

    # the micro-batch window (PROFILE_MICROBATCHES=1) keeps its per-rank traces: the exit-path call in pretrain_gpt_deepspeed.py would
    # otherwise merge 8 x 2.2 GB into one 16 GB file, 16 minutes of the allocation that nothing reads (run Y1r, 2026-09-14)
    if rank == 0 and merge and os.getenv('PROFILE_MICROBATCHES') != '1':
        print(f'[profiler_manager.py] {rank=} merging traces', flush=True)
        # Assuming merge_traces is defined somewhere or you import it
        merge_traces()
        
        
        