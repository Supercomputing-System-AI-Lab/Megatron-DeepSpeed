# ../utils/analyze_pipeline_memory.py

import re
import sys
import os
import argparse
from pathlib import Path

def get_job_info(input_path):
    """
    Extracts the Job ID and the base job directory from a given file path.
    """
    path = Path(input_path).resolve()
    for part in reversed(path.parts):
        match = re.search(r"job_(\d+)", part)
        if match:
            job_id = match.group(1)
            job_dir = str(path).split(part)[0] + part
            return job_id, job_dir
    if path.is_file():
        parent_dir = path.parent
        match = re.search(r"job_(\d+)", parent_dir.name)
        if match:
            return match.group(1), str(parent_dir)
    return None, None

def get_total_gpus_from_slurm_log(job_id, job_dir):
    """
    Finds the SLURM .o log file and parses the TOTAL_GPUS variable.
    """
    logs_dir = Path(job_dir)
    slurm_log_candidates = list(logs_dir.glob(f"*{job_id}.o"))
    if not slurm_log_candidates:
        print(f"Error: Could not find SLURM '.o' log file for job ID {job_id} in {logs_dir}", file=sys.stderr)
        return None
    slurm_log_path = slurm_log_candidates[0]
    print(f"Found SLURM config log: {slurm_log_path.name}")
    with open(slurm_log_path, 'r', errors='ignore') as f:
        content = f.read()
    match = re.search(r"^TOTAL_GPUS[:=]\s*(\d+)", content, re.MULTILINE)
    if match:
        total_gpus = int(match.group(1))
        print(f"Detected TOTAL_GPUS: {total_gpus}")
        return total_gpus
    else:
        print(f"Error: 'TOTAL_GPUS' variable not found in {slurm_log_path.name}", file=sys.stderr)
        return None

def find_peak_memory_in_file(file_path):
    """
    Reads a rank log file and finds the maximum 'max reserved' and 'max allocated' memory values.
    Returns a tuple (max_reserved, max_allocated).
    """
    max_reserved = None
    max_allocated = None
    
    if not os.path.exists(file_path):
        print(f"Warning: Log file not found: {file_path}", file=sys.stderr)
        return (None, None)

    try:
        with open(file_path, 'r', errors='ignore') as f:
            content = f.read()
        
        # Find all occurrences of 'max reserved: 123.456'
        reserved_matches = re.findall(r"max reserved:\s*([\d.]+)", content)
        if reserved_matches:
            max_reserved = max([float(val) for val in reserved_matches])
            
        # Find all occurrences of 'max allocated: 123.456'
        allocated_matches = re.findall(r"max allocated:\s*([\d.]+)", content)
        if allocated_matches:
            max_allocated = max([float(val) for val in allocated_matches])
            
    except Exception as e:
        print(f"Error processing file {file_path}: {e}", file=sys.stderr)
        return (None, None)
        
    return (max_reserved, max_allocated)

def main():
    parser = argparse.ArgumentParser(
        description="Analyzes peak 'max reserved' and 'max allocated' memory for each pipeline stage from rank logs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "log_file_path",
        type=str,
        help="Path to any file inside the target job directory (e.g., full_run.log or a rank_X.log)."
    )
    args = parser.parse_args()
    
    # =========================================================================
    # START: Automatically export prints to the log_file_path
    # =========================================================================
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            # Use "a" (append mode) to avoid overwriting the existing log data
            self.log = open(filename, "a", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(args.log_file_path)
    sys.stderr = sys.stdout
    # =========================================================================
    # END: Export prints
    # =========================================================================

    # 1. Determine Job ID and Directory
    job_id, job_dir = get_job_info(args.log_file_path)
    if not job_id:
        print(f"FATAL: Could not determine job ID or directory from the input path: {args.log_file_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"--- Analyzing Memory for Job ID: {job_id} ---")
    print(f"Job Directory: {job_dir}")
    print("-" * 40)

    # 2. Get Total GPUs from the SLURM log to know the number of ranks
    total_gpus = get_total_gpus_from_slurm_log(job_id, job_dir)
    if total_gpus is None:
        sys.exit(1)

    # 3. Iterate through pipeline stages (ranks 0, 8, 16, ...)
    pipeline_stage_memory = {}
    pipeline_stage_interval = 8
    
    for rank_num in range(0, total_gpus, pipeline_stage_interval):
        rank_log_file = os.path.join(job_dir, f"rank_{rank_num}.log")
        # The function now returns a tuple of (max_reserved, max_allocated)
        peak_mems = find_peak_memory_in_file(rank_log_file)
        pipeline_stage_memory[rank_num] = peak_mems
        
        
    # =========================================================================
    # START: Add this new code block here
    # =========================================================================
    # NEW: Iterate through all ranks to gather data for the full copy-pasteable lists
    all_ranks_reserved_list = []
    all_ranks_allocated_list = []
    print("\n--- Analyzing memory for all individual ranks for full lists... ---")
    for rank_num in range(total_gpus):
        rank_log_file = os.path.join(job_dir, f"rank_{rank_num}.log")
        reserved, allocated = find_peak_memory_in_file(rank_log_file)

        # Append the formatted string or "N/A" to the lists
        all_ranks_reserved_list.append(f"{reserved:.4f}" if reserved is not None else "N/A")
        all_ranks_allocated_list.append(f"{allocated:.4f}" if allocated is not None else "N/A")
    print("--- Analysis for all ranks complete. ---")
    # =========================================================================
    # END: New code block
    # =========================================================================

    # 4. Print the results
    print("\n" + "=" * 80)
    print("--- Peak Memory (GB) per Pipeline Stage ---")
    print("=" * 80)
    
    # Print header for the results table
    print(f"{'Pipeline Stage (Rank)':<25} | {'Max Reserved':<25} | {'Max Allocated':<25}")
    print("-" * 80)

    reserved_list = []
    allocated_list = []
    has_results = False

    for rank, (reserved, allocated) in pipeline_stage_memory.items():
        stage_id = int(rank / pipeline_stage_interval)
        stage_label = f"Stage {stage_id:<3} (Rank {rank:<3})"
        
        if reserved is not None and allocated is not None:
            print(f"{stage_label:<25} | {f'{reserved:.4f} GB':<25} | {f'{allocated:.4f} GB':<25}")
            reserved_list.append(f"{reserved:.4f}")
            allocated_list.append(f"{allocated:.4f}")
            has_results = True
        else:
            print(f"{stage_label:<25} | {'No data found':<25} | {'No data found':<25}")
            reserved_list.append("N/A")
            allocated_list.append("N/A")

    if not has_results:
        print("\nNo memory data could be extracted from any of the log files.")
        return

    # print("\n" + ("-" * 35) + " Copy-pasteable Lists " + ("-" * 34))
    # print("Max Reserved (GB):")
    # print(", ".join(reserved_list))
    # print("\nMax Allocated (GB):")
    # print(", ".join(allocated_list))
    # print("=" * 80)
    
    # =========================================================================
    # START: Replace the original "Copy-pasteable Lists" section with this
    # =========================================================================
    print("\n" + ("-" * 29) + " Copy-pasteable Lists (Pipeline Stages) " + ("-" * 29))
    print("Max Reserved (GB) for Pipeline Stages:")
    print(", ".join(reserved_list))
    print("\nMax Allocated (GB) for Pipeline Stages:")
    print(", ".join(allocated_list))

    print("\n" + ("-" * 31) + " Copy-pasteable Lists (All Ranks) " + ("-" * 32))
    print("Max Reserved (GB) for All Ranks:")
    print(", ".join(all_ranks_reserved_list))
    print("\nMax Allocated (GB) for All Ranks:")
    print(", ".join(all_ranks_allocated_list))
    print("=" * 80)
    # =========================================================================
    # END: Replacement block
    # =========================================================================

if __name__ == "__main__":
    main()