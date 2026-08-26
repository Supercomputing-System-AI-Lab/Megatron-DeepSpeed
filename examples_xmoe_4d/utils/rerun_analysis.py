import re
import sys
import os
import subprocess
import argparse
import glob
import tempfile

def get_job_id(file_path):
    """Extracts Job ID from filename or parent directory."""
    # 1. Try filename (e.g., xmoe-ep-4100741.o)
    match = re.search(r"(\d+)\.o$", os.path.basename(file_path))
    if match: return match.group(1)
    
    # 2. Try parent directory (e.g., logs/job_4100741/)
    match = re.search(r"job_(\d+)", file_path)
    if match: return match.group(1)
    
    return None

def find_variable(pattern, text, var_name, required=True, is_int=True, default=None):
    """Helper to find variables in the text content."""
    # Matches "VAR: value" OR "VAR=value"
    match = re.search(pattern, text, re.MULTILINE)
    if match:
        value = match.group(1).strip()
        if value.lower() in ['true', 'false']:
            return value
        return int(value) if is_int else value
    if default is not None:
        return default
    if required:
        print(f"FATAL: Could not find required variable '{var_name}' in the config file.")
        sys.exit(1)
    return None

def main():
    parser = argparse.ArgumentParser(description="Rerun analysis by merging SLURM .o and Rank logs.")
    parser.add_argument("input_file", type=str, help="Path to EITHER the .o file OR the rank log file.")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_file)
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return

    # 1. Identify Job ID
    job_id = get_job_id(input_path)
    if not job_id:
        print("FATAL: Could not determine Job ID from input path.")
        return
    print(f"--- Detected Job ID: {job_id} ---")

    # 2. Locate the two necessary files
    base_dir = os.path.dirname(os.path.dirname(input_path)) # Assuming structure logs/job_ID/ or logs/file.o
    logs_dir = os.path.join(os.getcwd(), "logs") # Default logs location
    
    # A. Find the SLURM .o file (Contains Variables & Memory)
    slurm_file = None
    input_dir = os.path.dirname(input_path) # Get directory of the file we passed in

    if input_path.endswith(".o"):
        slurm_file = input_path
    else:
        # Search for *jobid.o in the input file's directory first, then fallback to logs/
        candidates = glob.glob(os.path.join(input_dir, f"*{job_id}.o")) + \
                     glob.glob(os.path.join(logs_dir, f"*{job_id}.o"))
        if candidates:
            slurm_file = candidates[0]
    
    if not slurm_file or not os.path.exists(slurm_file):
        print(f"FATAL: Could not find the SLURM .o file for Job {job_id}. Cannot parse config.")
        print(f"Searched in: {input_dir} and {logs_dir}")
        return

    print(f"Using Config/Memory Log: {os.path.basename(slurm_file)}")

    # 3. Parse Config Variables from .o file
    with open(slurm_file, 'r') as f:
        config_content = f.read()

    try:
        # Note: Regex handles both "VAR: VAL" and "VAR=VAL"
        train_iters = find_variable(r"^TRAIN_ITERS[:=]\s*(\d+)", config_content, "TRAIN_ITERS")
        nodes = find_variable(r"^NODES[:=]\s*(\d+)", config_content, "NODES")
        total_gpus = find_variable(r"^TOTAL_GPUS[:=]\s*(\d+)", config_content, "TOTAL_GPUS")
        pp_size = find_variable(r"^PP_SIZE[:=]\s*(\d+)", config_content, "PP_SIZE")
        ep_parallel_size = find_variable(r"^EP_SIZE[:=]\s*(\d+)", config_content, "EP_PARALLEL_SIZE")
        tp_size = find_variable(r"^MP_SIZE[:=]\s*(\d+)", config_content, "MP_SIZE")
        mbs = find_variable(r"^MICRO_BATCH_SIZE[:=]\s*(\d+)", config_content, "BATCH_SIZE")
        global_batch_size = find_variable(r"^GLOBAL_BATCH_SIZE[:=]\s*(\d+)", config_content, "GLOBAL_BATCH_SIZE")
        
        model_size = find_variable(r"^Model Config Loaded for ([\w.-]+):", config_content, "MODEL_SIZE", is_int=False)
        seq_len = find_variable(r"^\s+SEQ_LEN[:=]\s*(\d+)", config_content, "SEQ_LEN")
        num_layers = find_variable(r"^\s+NUM_LAYERS[:=]\s*(\d+)", config_content, "NUM_LAYERS")
        num_experts = find_variable(r"^\s+EP_SIZE[:=]\s*(\d+)", config_content, "NUM_EXPERTS")
        expert_dim = find_variable(r"^\s+FFN_HIDDEN_SIZE[:=]\s+(\d+)", config_content, "FFN_HIDDEN_SIZE")
        topk = find_variable(r"^\s+topk[:=]\s*(\d+)", config_content, "topk")
        hidden_dim = find_variable(r"^\s+HIDDEN_SIZE[:=]\s*(\d+)", config_content, "HIDDEN_SIZE")
        moe_type = find_variable(r"^MOE_TYPE[:=]\s*([\w-]+)", config_content, "MOE_TYPE", is_int=False)
        
        activation_checkpointing = find_variable(r"^ACTIVATION_CHECKPOINT[:=]\s*(\w+)", config_content, "AC", is_int=False, default="false")
        checkpoint_interval = find_variable(r"^CHECKPOINT_NUM_LAYERS[:=]\s*(\d+)", config_content, "CKPT_INT", default=0)
        dynamic_checkpoint = find_variable(r"^DYNAMIC_CHECKPOINT[:=]\s*(\w+)", config_content, "DYN_CKPT", is_int=False, default="False")
        uneven_pp = find_variable(r"^UNEVEN_PP[:=]\s*(\w+)", config_content, "UNEVEN_PP", is_int=False, default="False")

        gpus_per_node = total_gpus // nodes
        dp_size = total_gpus // (pp_size * tp_size)
    except Exception as e:
        print(f"Error parsing variables from .o file: {e}")
        return

    # B. Find the Rank Log (Contains Loss/TFLOPs)
    # We calculate the last rank based on the parsed Total GPUs
    last_rank = total_gpus - 1
    
    # Check if the input file WAS the rank log
    rank_file = None
    if "rank_" in input_path:
        rank_file = input_path
    else:
        # Construct expected path: Look in input directory first
        rank_file = os.path.join(input_dir, f"rank_{last_rank}.log")
        if not os.path.exists(rank_file):
            # Fallback to the old strict job_ID directory structure
            rank_file = os.path.join(logs_dir, f"job_{job_id}", f"rank_{last_rank}.log")

    if not os.path.exists(rank_file):
        print(f"Warning: Rank log not found at {rank_file}. Analysis will lack Loss/TFLOPs.")
        rank_content = ""
    else:
        print(f"Using Metrics Log: {os.path.basename(rank_file)}")
        with open(rank_file, 'r', errors='ignore') as f:
            rank_content = f.read()

    # --- START OF MODIFIED SECTION ---

    # C. Find the Rank 0 Log (which contains the memory usage)
    rank0_file = os.path.join(os.path.dirname(rank_file), "rank_0.log") # Assumes rank 0 is in the same dir
    if not os.path.exists(rank0_file):
        print(f"Warning: Rank 0 log not found at {rank0_file}. Analysis will lack memory usage.")
        rank0_content = ""
    else:
        print(f"Using Memory Log: {os.path.basename(rank0_file)}")
        with open(rank0_file, 'r', errors='ignore') as f:
            rank0_content = f.read()


    # 4. Merge ALL Content into a Temp File
    # This allows analyze_log.py to see Memory, Config, and Loss metrics
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=f"_{job_id}_merged.log") as temp_file:
        # First, write the config from the .o file
        temp_file.write(config_content)
        
        # Second, write the memory logs from rank_0.log
        temp_file.write("\n\n" + "="*50 + "\nMERGED RANK 0 LOG (FOR MEMORY)\n" + "="*50 + "\n\n")
        temp_file.write(rank0_content)
        
        # Third, write the metrics from the last rank log
        temp_file.write("\n\n" + "="*50 + "\nMERGED LAST RANK LOG (FOR METRICS)\n" + "="*50 + "\n\n")
        temp_file.write(rank_content)
        
        temp_path = temp_file.name

    # --- END OF MODIFIED SECTION ---

    # 5. Run analyze_log.py on the Merged File
    analysis_script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "analyze_log.py")
    
    command = [
        "python", analysis_script_path, temp_path, # <--- Passing the merged temp file
        "--iterations", str(train_iters),
        "--nodes", str(nodes),
        "--gpus_per_node", str(gpus_per_node),
        "--pp", str(pp_size),
        "--ep", str(ep_parallel_size),
        "--dp", str(dp_size),
        "--tp", str(tp_size),
        "--gbs", str(global_batch_size),
        "--mbs", str(mbs),
        "--slurm_job_id", str(job_id),
        "--seqlen", str(seq_len),
        "--num_layers", str(num_layers),
        "--num_experts", str(num_experts),
        "--expert_dim", str(expert_dim),
        "--topk", str(topk),
        "--hidden_dim", str(hidden_dim),
        "--moe_type", moe_type,
        "--model_size", model_size,
        "--activation_checkpointing", activation_checkpointing,
        "--checkpoint_interval", str(checkpoint_interval),
        "--dynamic_checkpoint", dynamic_checkpoint,
        "--uneven_pp", uneven_pp
    ]

    print("\nRunning analysis...")
    result = subprocess.run(command, capture_output=True, text=True)

    # Cleanup temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if result.returncode != 0:
        print("--- Analysis Failed ---")
        print(result.stderr)
        print(result.stdout)
    else:
        print("--- Analysis Successful ---")
        # Append results to the SLURM .o file for persistence
        with open(slurm_file, 'a') as f:
            f.write("\n\n" + "="*40 + "\nRETROACTIVE ANALYSIS\n" + "="*40 + "\n")
            f.write(result.stdout)
        print(f"Results appended to: {os.path.basename(slurm_file)}")

if __name__ == "__main__":
    main()