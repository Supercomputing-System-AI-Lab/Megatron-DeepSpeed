# profiler_manager.py
import os
import torch
import torch.distributed as dist
import json 
import glob 
from torch.profiler import profile, record_function, ProfilerActivity, schedule

_PROF = None

def merge_traces():
    """
    Merges multiple Chrome trace files from different ranks into a single file.
    This function should be called by only one rank after all ranks have
    finished profiling and saved their individual traces.
    """
    # Directory where individual rank traces are saved
    trace_dir = os.path.abspath(os.getenv ("profile_dir"))
    
    # Path for the final merged trace file
    merged_file_path = os.path.join (trace_dir, "merged.json")

    # Use glob to find all trace files. This is more robust than hardcoding names.
    trace_files = sorted(glob.glob(os.path.join(trace_dir, "trace_rank*.json")))

    if not trace_files:
        print("No trace files found to merge.")
        return

    print(f"Found {len(trace_files)} trace files to merge.")

    all_events = []
    
    # We will remap the original PIDs to new, unique PIDs
    # Let's just use the rank number as the new PID for simplicity and clarity.
    for rank, trace_file in enumerate(trace_files):
        with open(trace_file, 'r') as f:
            data = json.load(f)
        
        # Original PIDs in this file. We need to track them to remap TIDs correctly.
        original_pids = set()

        # Add a metadata event to name this process in the Perfetto UI
        all_events.append({
            "name": "process_name", 
            "ph": "M", 
            "pid": rank,  # The new, unique PID
            "args": {"name": f"Rank {rank}"}
        })
        
        for event in data['traceEvents']:
            original_pid = event.get('pid')
            if original_pid is not None:
                original_pids.add(original_pid)

                # Overwrite the PID with our new unique rank-based PID
                event['pid'] = rank
                all_events.append(event)

    # Write the final merged file
    with open(merged_file_path, 'w') as f:
        json.dump({'traceEvents': all_events}, f)

    print(f"Successfully merged {len(trace_files)} traces into {merged_file_path}")



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


def _trace_handler(prof):
    out_dir = os.path.abspath(os.getenv('profile_dir', '.'))
    os.makedirs(out_dir, exist_ok=True)
    r, w = _get_rank_world()
    
    
    # 1. Safely export chrome trace
    fname_json = os.path.join(out_dir, f"trace_rank{r}_of_{w}.json")
    try:
        print(f"--- Profiler Exporting chrome trace for rank {r} ---")
        prof.export_chrome_trace(fname_json)
    except Exception as e:
        print(f"[Rank {r}] Warning: Failed to export chrome trace: {e}")

    
    # # pickle memory 
    # try:
    #     pickle_fname = os.path.join(out_dir, f"snapshot_rank{r}_of_{w}.pickle")
    #     torch.cuda.memory._dump_snapshot(pickle_fname)
    #     if r == 0:
    #         print(f"[profiler_manager.py] Dumped memory snapshot to {pickle_fname}")
    # except Exception as e:
    #     print(f"[Rank {r}] Warning: Failed to dump memory snapshot: {e}")

    # try:
    #     torch.cuda.memory._record_memory_history(enabled=None)
    # except AttributeError:
    #     pass

def _get_profiler(wait=0, warmup=0, active=2, repeat=1):
    global _PROF
    if _PROF is None:
        print(f'[profiler_manager.py]: global _PROF is None, initiating a new one')
        sched = schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        _PROF = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=sched,
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
            profile_memory=True,
            on_trace_ready=_trace_handler,
        )
        _PROF.__enter__() 
    else:
        print(f'[profiler_manager.py]: loading pre-allocated global _PROF')
    return _PROF

def finalize_profiler():
    global _PROF
    rank, _ = _get_rank_world()
    print(f'[profiler_manager.py] {rank=} closing profiler')
    if _PROF is not None:
        _PROF.__exit__(None, None, None)
        _PROF = None
        
    if rank == 0:
        print(f'[profiler_manager.py] {rank=} merging traces', flush=True)
        # Assuming merge_traces is defined somewhere or you import it
        merge_traces()
        
        
        