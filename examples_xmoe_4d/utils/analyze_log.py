import re
import sys
import argparse
import os
import pandas as pd
from datetime import datetime
import statistics
import json
import fcntl
import shutil
import tempfile
import zipfile

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze log files and store results in XLSX.")
    # --- Existing Arguments ---
    parser.add_argument("log_file", type=str, help="Path to the log file.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations.")
    parser.add_argument("--nodes", type=int, required=True, help="Number of nodes.")
    parser.add_argument("--gpus_per_node", type=int, required=True, help="Number of GPUs per node.")
    parser.add_argument("--pp", type=int, required=True, help="PP size.")
    parser.add_argument("--ep", type=int, required=True, help="EP size.")
    parser.add_argument("--dp", type=int, required=True, help="DP size.")
    parser.add_argument("--tp", type=int, required=True, help="TP size.")
    parser.add_argument("--gbs", type=int, required=True, help="Global Batch Size.")
    parser.add_argument("--mbs", type=int, required=True, help="Micro Batch Size.")
    # --- CHANGE THIS LINE ---
    parser.add_argument("--slurm_job_id", type=str, required=False, help="SLURM Job ID. Auto-detected if not provided.")
    # ------------------------
    parser.add_argument("--seqlen", type=int, required=True, help="Sequence length.")
    parser.add_argument("--num_layers", type=int, required=True, help="Number of layers.")
    parser.add_argument("--num_experts", type=int, required=True, help="Number of experts.")
    parser.add_argument("--expert_dim", type=int, required=True, help="Expert dimension.")
    parser.add_argument("--topk", type=int, required=True, help="Top-K for MoE.")
    parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden dimension.")
    parser.add_argument("--warmup_steps", type=int, default=10, required=False, help="Warmup steps before recording TFLOPS, default 5")
    parser.add_argument("--moe_type", type=str, required=True, help="Type of the MoE implementation (e.g., X-MOE, DS-MOE).")
    parser.add_argument("--model_size", type=str, default="10B", required=False, help="Size of the model (e.g., 7b, 13b).")

    # --- NEWLY ADDED ARGUMENTS ---
    parser.add_argument("--activation_checkpointing", type=str, required=True, help="Activation checkpointing status (e.g., true/false).")
    parser.add_argument("--checkpoint_interval", type=int, required=True, help="The interval for saving model checkpoints.")
    
    # Updated to include defaults as requested
    parser.add_argument("--dynamic_checkpoint", type=str, required=False, default="False", help="Whether dynamic checkpointing activations. True if arg is str(True)")
    parser.add_argument("--uneven_pp", type=str, required=False, default="False", help="Whether use uneven pipeline layer partitioning. True if arg is str(True)")
    parser.add_argument("--zero", type=int, required=False, default=0, help="DeepSpeed ZeRO stage (0=disabled, 1, 2, or 3).")

    parser.add_argument('--profiling', action='store_true', help='This will store the gemm & comm time into a json file and redirect the output file')
    
    return parser.parse_args()

def analyze_log(log_file, num_layers, warmup_steps, args):
    """
    Analyzes the log file to extract performance metrics, and calculates
    theoretical FLOPs and hardware utilization.
    """
    iteration_pattern = re.compile(
        r"iteration\s+\d+/\s+\d+\s+\|.*?"
        r"elapsed time per iteration \(ms\):\s+([0-9.]+)\s+\|.*?"
        r"lm loss:\s+([0-9.E+-]+)"
        r"(?:\s+\|\s+moe loss:\s+([0-9.E+-]+))?"
        r".*?samples per second:\s+([0-9.]+)\s+\|"
        r"\s+TFLOPs:\s+([0-9.]+)"
    )
    time_pattern = re.compile(r"1st_a2a:\s+([0-9.]+),\s+experts:\s+([0-9.]+),\s+2nd_a2a:\s+([0-9.]+)")
    gemm_pattern = re.compile(r"qkv_gemm:\s+([0-9.]+)\s+\|\s+attn_gemm:\s+([0-9.]+)\s+\|\s+out_gemm:\s+([0-9.]+)")
    peak_mem_pattern = re.compile(r"max reserved:\s+([0-9.]+)")


    tflops_values, samples_per_sec_values, elapsed_times = [], [], []
    lm_losses, moe_losses = [], []
    first_a2a_values, experts_values, second_a2a_values = [], [], []
    qkv_gemm_values, attn_gemm_values, out_gemm_values = [], [], []
    peak_mem_values = []


    # temp_qkv, temp_attn, temp_out = [], [], []

    # with open(log_file, "r") as f:
    #     for line in f:
    #         iteration_match = iteration_pattern.search(line)
    #         gemm_match = gemm_pattern.search(line)
    #         time_match = time_pattern.search(line)
    #         peak_mem_match = peak_mem_pattern.search(line)


    #         if iteration_match:
    #             if temp_qkv:
    #                 qkv_gemm_values.append(statistics.mean(temp_qkv))
    #                 attn_gemm_values.append(statistics.mean(temp_attn))
    #                 out_gemm_values.append(statistics.mean(temp_out))
    #                 temp_qkv, temp_attn, temp_out = [], [], []

    #             elapsed_times.append(float(iteration_match.group(1)))
    #             lm_losses.append(float(iteration_match.group(2)))
    #             moe_losses.append(float(iteration_match.group(3)) if iteration_match.group(3) else 0.0)
    #             samples_per_sec_values.append(float(iteration_match.group(4)))
    #             tflops_values.append(float(iteration_match.group(5)))

    #         if time_match:
    #             first_a2a_values.append(float(time_match.group(1)))
    #             experts_values.append(float(time_match.group(2)))
    #             second_a2a_values.append(float(time_match.group(3)))

    #         if gemm_match:
    #             temp_qkv.append(float(gemm_match.group(1)))
    #             temp_attn.append(float(gemm_match.group(2)))
    #             temp_out.append(float(gemm_match.group(3)))
            
    #         if peak_mem_match:
    #             peak_mem_values.append(float(peak_mem_match.group(1)))


    # if temp_qkv:
    #     qkv_gemm_values.append(statistics.mean(temp_qkv))
    #     attn_gemm_values.append(statistics.mean(temp_attn))
    #     out_gemm_values.append(statistics.mean(temp_out))
    tflops_values, samples_per_sec_values, elapsed_times = [], [], []
    lm_losses, moe_losses = [], []
    first_a2a_values, experts_values, second_a2a_values = [], [], []
    qkv_gemm_values, attn_gemm_values, out_gemm_values = [], [], []
    peak_mem_values = []

    with open(log_file, "r") as f:
        for line in f:
            iteration_match = iteration_pattern.search(line)
            gemm_match = gemm_pattern.search(line)
            time_match = time_pattern.search(line)
            peak_mem_match = peak_mem_pattern.search(line)

            if iteration_match:
                elapsed_times.append(float(iteration_match.group(1)))
                lm_losses.append(float(iteration_match.group(2)))
                moe_losses.append(float(iteration_match.group(3)) if iteration_match.group(3) else 0.0)
                samples_per_sec_values.append(float(iteration_match.group(4)))
                tflops_values.append(float(iteration_match.group(5)))

            if time_match:
                first_a2a_values.append(float(time_match.group(1)))
                experts_values.append(float(time_match.group(2)))
                second_a2a_values.append(float(time_match.group(3)))

            if gemm_match:
                qkv_gemm_values.append(float(gemm_match.group(1)))
                attn_gemm_values.append(float(gemm_match.group(2)))
                out_gemm_values.append(float(gemm_match.group(3)))
            
            if peak_mem_match:
                peak_mem_values.append(float(peak_mem_match.group(1)))

    if not tflops_values: return None

    if len(tflops_values) > warmup_steps:
        start_index = warmup_steps
        print (f'analyze_log: warmup_steps: {warmup_steps}')
    else:
        print(f"Warning: Less than {warmup_steps} iterations found. Calculating stats over all available iterations.", file=sys.stderr)
        start_index = 0

    # tflops_for_calc = tflops_values[start_index:]
    # samples_for_calc = samples_per_sec_values[start_index:]
    # elapsed_for_calc = elapsed_times[start_index:]
    # first_a2a_for_calc = first_a2a_values[start_index:]
    # experts_for_calc = experts_values[start_index:]
    # second_a2a_for_calc = second_a2a_values[start_index:]
    # qkv_for_calc = qkv_gemm_values[start_index:]
    # attn_for_calc = attn_gemm_values[start_index:]
    # out_for_calc = out_gemm_values[start_index:]
    tflops_for_calc = tflops_values[start_index:]
    samples_for_calc = samples_per_sec_values[start_index:]
    elapsed_for_calc = elapsed_times[start_index:]
    first_a2a_for_calc = first_a2a_values[start_index:]
    experts_for_calc = experts_values[start_index:]
    second_a2a_for_calc = second_a2a_values[start_index:]

    # Dynamically align the GEMM values with iterations to drop warmup steps properly
    if len(qkv_gemm_values) > 0 and len(tflops_values) > 0:
        chunk_size = len(qkv_gemm_values) // len(tflops_values)
        gemm_start_index = start_index * chunk_size
    else:
        gemm_start_index = 0

    qkv_for_calc = qkv_gemm_values[gemm_start_index:] if len(qkv_gemm_values) > gemm_start_index else qkv_gemm_values
    attn_for_calc = attn_gemm_values[gemm_start_index:] if len(attn_gemm_values) > gemm_start_index else attn_gemm_values
    out_for_calc = out_gemm_values[gemm_start_index:] if len(out_gemm_values) > gemm_start_index else out_gemm_values

    avg_tflops = statistics.mean(tflops_for_calc) if tflops_for_calc else 0.0
    tflops_std_dev = statistics.stdev(tflops_for_calc) if len(tflops_for_calc) > 1 else 0.0
    avg_samples = statistics.mean(samples_for_calc) if samples_for_calc else 0.0
    avg_elapsed = statistics.mean(elapsed_for_calc) if elapsed_for_calc else 0.0

    avg_experts_time_pl = (statistics.mean(experts_for_calc) / num_layers) if experts_for_calc and num_layers > 0 else 0.0
    avg_qkv_gemm_time = statistics.mean(qkv_for_calc) if qkv_for_calc else 0.0
    avg_attn_gemm_time = statistics.mean(attn_for_calc) if attn_for_calc else 0.0
    avg_out_gemm_time = statistics.mean(out_for_calc) if out_for_calc else 0.0

    THEORETICAL_MAX_TFLOPS = 191.0
    b, s, h, h_ffn, topk, ep = args.mbs, args.seqlen, args.hidden_dim, args.expert_dim, args.topk, args.ep

    flops_expert = (2 * 2 * b * s * topk * h * h_ffn) / ep if ep > 0 else 0
    flops_qkv = 2 * 3 * b * s * h**2
    flops_attention = 4 * b * s**2 * h
    flops_out = 2 * b * s * h**2

    def get_tflops(flops, time_ms):
        if time_ms == 0: return 0.0
        return (flops / 1e12) / (time_ms / 1e3)

    tflops_expert = get_tflops(flops_expert, avg_experts_time_pl)
    tflops_qkv = get_tflops(flops_qkv, avg_qkv_gemm_time)
    tflops_attn = get_tflops(flops_attention, avg_attn_gemm_time)
    tflops_out = get_tflops(flops_out, avg_out_gemm_time)

    util_expert = (tflops_expert / THEORETICAL_MAX_TFLOPS) * 100 if THEORETICAL_MAX_TFLOPS > 0 else 0.0
    util_qkv = (tflops_qkv / THEORETICAL_MAX_TFLOPS) * 100 if THEORETICAL_MAX_TFLOPS > 0 else 0.0
    util_attn = (tflops_attn / THEORETICAL_MAX_TFLOPS) * 100 if THEORETICAL_MAX_TFLOPS > 0 else 0.0
    util_out = (tflops_out / THEORETICAL_MAX_TFLOPS) * 100 if THEORETICAL_MAX_TFLOPS > 0 else 0.0

    return {
        "matched_iterations": len(tflops_values),
        "avg_tflops": avg_tflops, "tflops_std_dev": tflops_std_dev,
        "avg_samples": avg_samples, "avg_elapsed": avg_elapsed,
        "final_lm_loss": lm_losses[-1] if lm_losses else 0.0,
        "final_moe_loss": moe_losses[-1] if moe_losses else 0.0,
        "lm_losses": lm_losses,
        "peak_mem_gb": max(peak_mem_values) if peak_mem_values else 0.0,
        "avg_1st_a2a_per_layer": (statistics.mean(first_a2a_for_calc) / num_layers) if first_a2a_for_calc and num_layers > 0 else 0.0,
        "avg_experts_per_layer": avg_experts_time_pl,
        "avg_2nd_a2a_per_layer": (statistics.mean(second_a2a_for_calc) / num_layers) if second_a2a_for_calc and num_layers > 0 else 0.0,
        "avg_qkv_gemm": avg_qkv_gemm_time, "avg_attn_gemm": avg_attn_gemm_time, "avg_out_gemm": avg_out_gemm_time,
        "flops_expert": flops_expert, "tflops_expert": tflops_expert, "util_expert": util_expert,
        "flops_qkv": flops_qkv, "tflops_qkv": tflops_qkv, "util_qkv": util_qkv,
        "flops_attention": flops_attention, "tflops_attn": tflops_attn, "util_attn": util_attn,
        "flops_out": flops_out, "tflops_out": tflops_out, "util_out": util_out,
    }

def _read_existing_workbook(file_path):
    """Return (main_df, loss_df), or (None, None) if the workbook is absent or unreadable.

    A workbook left corrupt by an earlier concurrent write is moved aside instead of
    raising, so one bad file cannot permanently break every subsequent run.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return None, None
    try:
        with pd.ExcelFile(file_path) as xls:
            return (pd.read_excel(xls, 'Main_Results'),
                    pd.read_excel(xls, 'Loss_per_Iteration'))
    except (zipfile.BadZipFile, ValueError, KeyError, OSError) as e:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        quarantine = f"{file_path}.corrupt-{stamp}"
        try:
            shutil.move(file_path, quarantine)
            note = f"moved it to '{quarantine}'"
        except OSError:
            note = "could not move it aside"
        print(f"Warning: '{file_path}' is unreadable ({type(e).__name__}: {e}); {note}. "
              f"Starting a fresh workbook.", file=sys.stderr)
        return None, None


def _atomic_write_workbook(file_path, main_df, loss_df):
    """Write the workbook to a temp file in the same dir, then atomically replace the target."""
    directory = os.path.dirname(file_path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-results-", suffix=".xlsx")
    os.close(fd)
    try:
        with pd.ExcelWriter(tmp_path, engine='openpyxl') as writer:
            main_df.to_excel(writer, sheet_name='Main_Results', index=False)
            loss_df.to_excel(writer, sheet_name='Loss_per_Iteration', index=False)
        os.replace(tmp_path, file_path)   # atomic within the same filesystem
        tmp_path = None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def write_to_xlsx(data, args):
    """Writes the collected data to a single XLSX file with two sheets."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    if args.profiling: 
        filename = f"{date_str}-profile.xlsx"
    else: 
        filename = f"{date_str}-results.xlsx"
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    file_path = os.path.join(results_dir, filename)

    run_id = f"{args.moe_type}-PP{args.pp}-EP{args.ep}-GBS{args.gbs}-MBS{args.mbs}-SLURM_ID-{args.slurm_job_id}"

    # --- UPDATED: Header with new columns ---
    header = [
        "Name", "MOE Type", "Model Size", "AVG TFLOPs", "TFLOPs Std Dev", 
        "Activation Checkpointing", "ckpt_interval", 
        "Dynamic Checkpoint", "Uneven PP Partition", # <--- Added here
        "Iterations", "Actual Iterations",
        "Nodes", "GPUs/node", "PP", "EP", "DP", "TP", "GBS", "MBS",
        "Final lm_loss", "peak_mem(GB)",
        "1st_a2a_PL (ms)", "experts_PL (ms)", "2nd_a2a_PL (ms)",
        "QKV GEMM PL (ms)", "Attn GEMM PL (ms)", "Out GEMM PL (ms)",
        "Expert Util (%)", "QKV Util (%)", "Attn Util (%)", "Out Util (%)",
        "SeqLen", "Num Experts", "Expert Dim", "Hidden Dim", "Top-K",
        "SLURM_JOB_ID", "Avg Iteration Time (ms)",
        "ZeRO Stage",
    ]

    # --- UPDATED: Main row dictionary with new data ---
    main_row_dict = {
        "Name": run_id, "MOE Type": args.moe_type, "Model Size": args.model_size,
        "AVG TFLOPs": f"{data['avg_tflops']:.2f}", "TFLOPs Std Dev": f"{data['tflops_std_dev']:.2f}",
        "Activation Checkpointing": args.activation_checkpointing,
        "ckpt_interval": args.checkpoint_interval,
        "Dynamic Checkpoint": args.dynamic_checkpoint,      # <--- Added value
        "Uneven PP Partition": args.uneven_pp,    # <--- Added value
        "Iterations": args.iterations, "Actual Iterations": data['matched_iterations'],
        "Nodes": args.nodes, "GPUs/node": args.gpus_per_node, "MBS": args.mbs, "GBS": args.gbs,
        "Final lm_loss": f"{data['final_lm_loss']:.6f}",
        "peak_mem(GB)": f"{data['peak_mem_gb']:.4f}",
        "PP": args.pp, "EP": args.ep, "DP": args.dp, "TP": args.tp,
        "1st_a2a_PL (ms)": f"{data['avg_1st_a2a_per_layer']:.2f}",
        "experts_PL (ms)": f"{data['avg_experts_per_layer']:.2f}",
        "2nd_a2a_PL (ms)": f"{data['avg_2nd_a2a_per_layer']:.2f}",
        "QKV GEMM PL (ms)": f"{data['avg_qkv_gemm']:.2f}",
        "Attn GEMM PL (ms)": f"{data['avg_attn_gemm']:.2f}",
        "Out GEMM PL (ms)": f"{data['avg_out_gemm']:.2f}",
        "Expert Util (%)": f"{data['util_expert']:.2f}",
        "QKV Util (%)": f"{data['util_qkv']:.2f}",
        "Attn Util (%)": f"{data['util_attn']:.2f}",
        "Out Util (%)": f"{data['util_out']:.2f}",
        "SeqLen": args.seqlen,
        "Num Experts": args.num_experts,
        "Expert Dim": args.expert_dim,
        "Hidden Dim": args.hidden_dim,
        "Top-K": args.topk,
        "SLURM_JOB_ID": args.slurm_job_id,
        "Avg Iteration Time (ms)": f"{data['avg_elapsed']:.1f}", # <<< THIS LINE WAS ADDED
        "ZeRO Stage": args.zero,
    }
    new_main_df = pd.DataFrame([main_row_dict])[header]

    loss_row_dict = {"id": run_id}
    for i, loss in enumerate(data.get('lm_losses', [])):
        loss_row_dict[f'loss_{i+1}'] = loss
    new_loss_df = pd.DataFrame([loss_row_dict])

    # ------------------------------------------------------------------------
    # Concurrency-safe append.
    #
    # main_results / loss_validate / profiling_cache each launch SEVERAL SLURM
    # jobs that finish around the same time and all append to this one daily
    # file. The previous code did an UNLOCKED read-modify-write, and
    # pd.ExcelWriter TRUNCATES the file before rewriting it — so two jobs
    # interleaving left a truncated zip ("BadZipFile: Bad magic number for
    # central directory"), after which every later job crashed too, because only
    # FileNotFoundError was caught.
    #
    # Fix: (1) an exclusive flock serialises concurrent writers; (2) the workbook
    # is written to a temp file and os.replace()d in, so a reader never observes
    # a half-written file; (3) an unreadable workbook is quarantined rather than
    # killing the run.
    # ------------------------------------------------------------------------
    # ONE hidden lock file for the whole results dir (not one per workbook), so the
    # folder does not accumulate a .lock next to every .xlsx. It must live on the
    # shared filesystem: the concurrent writers are SLURM jobs on DIFFERENT nodes,
    # so a node-local /tmp lock would give no mutual exclusion at all.
    lock_path = os.path.join(results_dir, ".xlsx-write.lock")
    with open(lock_path, "w") as lock_fh:
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
        except OSError as e:  # filesystem without flock support — atomic replace still applies
            print(f"Warning: could not lock '{lock_path}' ({e}); relying on atomic replace.",
                  file=sys.stderr)

        existing_main_df, existing_loss_df = _read_existing_workbook(file_path)
        if existing_main_df is None:
            combined_main_df, combined_loss_df = new_main_df, new_loss_df
        else:
            combined_main_df = pd.concat([existing_main_df, new_main_df], ignore_index=True)
            combined_loss_df = pd.concat([existing_loss_df, new_loss_df], ignore_index=True)

        _atomic_write_workbook(file_path, combined_main_df, combined_loss_df)

    return file_path


def write_profile_json(data, args):
    """Writes specific profiling metrics to a JSON file for the Planner."""
    from model_registry import get_filename
    
    # Calculate total GPUs (n)
    total_gpus = args.nodes * args.gpus_per_node
    
    out_dir = "planner_profiling_cache"
    os.makedirs(out_dir, exist_ok=True)
    
    # Construct filename based on Planner Cache format
    # N=nodes, n=total_gpus, d=hidden_dim, e=num_experts, f=expert_dim, k=topk, s=seqlen
    base_name = get_filename(
                    model_size=args.model_size, 
                    nodes=args.nodes, 
                    total_gpus=(args.nodes * args.gpus_per_node), 
                    mbs=args.mbs, 
                )
    
    filename = os.path.join(out_dir, base_name)

    # Extract only the specific keys required
    profile_data = {
        "avg_1st_a2a_per_layer": data['avg_1st_a2a_per_layer'],
        "avg_experts_per_layer": data['avg_experts_per_layer'],
        "avg_2nd_a2a_per_layer": data['avg_2nd_a2a_per_layer'],
        "avg_qkv_gemm": data['avg_qkv_gemm'],
        "avg_attn_gemm": data['avg_attn_gemm'],
        "avg_out_gemm": data['avg_out_gemm'], 
        "fwd_gemm_time_per_layer": data['avg_qkv_gemm'] + data['avg_attn_gemm'] + data['avg_out_gemm'] + data['avg_experts_per_layer'],
        "fwd_comm_time_per_layer": data['avg_1st_a2a_per_layer'] + data['avg_2nd_a2a_per_layer'],
    }

    try:
        with open(filename, 'w') as f:
            json.dump(profile_data, f, indent=4)
        print(f"Profiling data saved to JSON: {filename}")
    except IOError as e:
        print(f"Error writing JSON file: {e}")
        
    return filename

def extract_job_id(log_file_path):
    """
    Attempts to extract Job ID from:
    1. The filename (Old style: xmoe-type-12345.o)
    2. The parent directory (New style: logs/job_12345/rank_0.log)
    """
    filename = os.path.basename(log_file_path)
    abs_path = os.path.abspath(log_file_path)
    parent_dir = os.path.basename(os.path.dirname(abs_path))

    # 1. Try filename regex (Old Style)
    match_filename = re.search(r"xmoe-\w+-(\d+\.?\d*)\.o", filename)
    if match_filename:
        return match_filename.group(1)

    # 2. Try directory regex (New Style)
    match_dir = re.search(r"job_(\d+)", parent_dir)
    if match_dir:
        return match_dir.group(1)
    
    return None


def compare_predicted_vs_actual_memory(args):
    """Compare the planner's PREDICTED per-stage memory (the 'Stage Memory (GB): [...]' line that
    ELM_PP_launch.py prints) against the ACTUAL profiled per-rank 'max reserved' memory.

    Layout: the run samples one rank per node (stride = gpus_per_node). For a model replica
    (DP) >= 2, each pipeline stage spans nodes_per_stage = nodes // pp adjacent nodes
    (pp_stage0_dp0, pp_stage0_dp1, pp_stage1_dp0, ...). We collapse the DP replicas of each
    stage two ways -- MAX over DP and MEAN over DP -- and report MAE and MAPE of each against
    the per-stage prediction."""
    import os, re
    job_dir = os.path.dirname(os.path.abspath(args.log_file))
    pp = args.pp
    gpus_per_node = args.gpus_per_node
    total_nodes = args.nodes
    total_gpus = total_nodes * gpus_per_node

    # --- 1. predicted per-stage memory (planner output, printed on rank 0) ---
    predicted = None
    pat = re.compile(r"Stage Memory \(GB\):\s*\[([^\]]*)\]")
    for src in (args.log_file, os.path.join(job_dir, "rank_0.log")):
        try:
            with open(src, 'r', errors='ignore') as f:
                m = pat.findall(f.read())
        except OSError:
            continue
        if m:
            predicted = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", m[-1])]
            break
    if not predicted:
        print("\n[mem-compare] No planner 'Stage Memory (GB):' line found (planner disabled?); "
              "skipping predicted-vs-actual memory comparison.")
        return

    # --- 2. actual per-node 'max reserved' (one rank per node) ---
    res_re = re.compile(r"max reserved:\s*([\d.]+)")
    node_reserved = []
    for rank in range(0, total_gpus, gpus_per_node):
        val = None
        try:
            with open(os.path.join(job_dir, f"rank_{rank}.log"), 'r', errors='ignore') as f:
                hits = res_re.findall(f.read())
                if hits:
                    val = max(float(x) for x in hits)
        except OSError:
            pass
        node_reserved.append(val)

    # --- 3. group nodes -> pipeline stages (DP replicas adjacent), collapse via max & mean ---
    nps = max(1, total_nodes // pp)            # nodes (DP replicas) per pipeline stage
    actual_max, actual_avg = [], []
    for s in range(pp):
        grp = [v for v in node_reserved[s * nps:(s + 1) * nps] if v is not None]
        actual_max.append(max(grp) if grp else None)
        actual_avg.append(sum(grp) / len(grp) if grp else None)

    # --- 4. MAE / MAPE vs the per-stage prediction ---
    n = min(len(predicted), pp)
    def _errs(act):
        ae, ape = [], []
        for i in range(n):
            a = act[i] if i < len(act) else None
            if a is not None:
                d = abs(predicted[i] - a)
                ae.append(d)
                if a != 0:
                    ape.append(d / abs(a) * 100.0)
        mae = sum(ae) / len(ae) if ae else float('nan')
        mape = sum(ape) / len(ape) if ape else float('nan')
        return mae, mape, len(ae)
    mae_max, mape_max, k_max = _errs(actual_max)
    mae_avg, mape_avg, k_avg = _errs(actual_avg)

    # --- 5. report (same style as the rest) ---
    print("\n================================================================================\n"
          "--- Predicted vs Actual Per-Stage Memory (Max Reserved GB) ---\n"
          "================================================================================")
    print(f"PP stages: {pp} | DP replicas (nodes) per stage: {nps} | predicted entries: {len(predicted)}")
    print(f"{'Stage':<6} {'Predicted':<11} {'Actual(maxDP)':<14} {'Actual(avgDP)':<14} {'|err|max':<9} {'|err|avg':<9}")
    print("-" * 72)
    for i in range(n):
        p, am, av = predicted[i], actual_max[i], actual_avg[i]
        sm = f"{am:.2f}" if am is not None else "N/A"
        sa = f"{av:.2f}" if av is not None else "N/A"
        em = f"{abs(p - am):.2f}" if am is not None else "N/A"
        ev = f"{abs(p - av):.2f}" if av is not None else "N/A"
        print(f"{i:<6} {p:<11.2f} {sm:<14} {sa:<14} {em:<9} {ev:<9}")
    print("-" * 72)
    print(f"{'MAE  vs MAX-over-DP  (GB):':<30} {mae_max:.4f}   (over {k_max} stages)")
    print(f"{'MAPE vs MAX-over-DP  (%):':<30} {mape_max:.2f}")
    print(f"{'MAE  vs MEAN-over-DP (GB):':<30} {mae_avg:.4f}   (over {k_avg} stages)")
    print(f"{'MAPE vs MEAN-over-DP (%):':<30} {mape_avg:.2f}")
    print("\nPredicted per-stage (GB):")
    print(", ".join(f"{x:.2f}" for x in predicted[:n]))
    print("Actual MAX-over-DP per-stage (GB):")
    print(", ".join((f"{x:.2f}" if x is not None else "N/A") for x in actual_max[:n]))
    print("Actual MEAN-over-DP per-stage (GB):")
    print(", ".join((f"{x:.2f}" if x is not None else "N/A") for x in actual_avg[:n]))
    print("================================================================================")


def main():
    """Main function to run the analysis and save results."""
    args = parse_arguments()
    
    if not args.slurm_job_id:
        detected_id = extract_job_id(args.log_file)
        if detected_id:
            args.slurm_job_id = detected_id
            print(f"Auto-detected SLURM Job ID from path: {args.slurm_job_id}")
        else:
            print("Warning: Could not auto-detect SLURM Job ID. Defaulting to '000000'.")
            args.slurm_job_id = "000000"
    
    
    print (f'{args.profiling=}')
    analysis_results = analyze_log(args.log_file, args.num_layers, args.warmup_steps, args)

    if analysis_results:
        
        if args.profiling:
            # If profiling mode, write to JSON with the specific N_n_d... naming convention
            output_file = write_profile_json(analysis_results, args)
            
        output_file = write_to_xlsx(analysis_results, args)

        metric_names = ["Log Directory", "Actual ran iterations", "Average TFLOPs", "TFLOPs Standard Deviation",
                        "Average samples/sec", "Average elapsed time per iteration (ms)", "Average 1st a2a time per layer (ms)",
                        "Average experts time per layer (ms)", "Average 2nd a2a time per layer (ms)", "Average QKV GEMM time (ms)",
                        "Average Attn GEMM time (ms)", "Average Out GEMM time (ms)", "Final lm_loss", "Final moe_loss",
                        "Peak Memory (GB)"]
        max_width = max(len(name) for name in metric_names)
        align_width = max_width + 2

        print(f'\n\n================================================================================\n'
              f'--- Analyzing Performance from Logs ---\n'
              f'================================================================================')

        print(f"{'Log Directory:':<{align_width}} {output_file}")
        print(f"{'Actual ran iterations:':<{align_width}} {analysis_results['matched_iterations']}")
        print(f"{'Average TFLOPs:':<{align_width}} {analysis_results['avg_tflops']:.2f}")
        print(f"{'TFLOPs Standard Deviation:':<{align_width}} {analysis_results['tflops_std_dev']:.2f}")
        print(f"{'Average samples/sec:':<{align_width}} {analysis_results['avg_samples']:.3f}")
        print(f"{'Average elapsed time per iteration (ms):':<{align_width}} {analysis_results['avg_elapsed']:.1f}")
        print(f"{'Average 1st a2a time per layer (ms):':<{align_width}} {analysis_results['avg_1st_a2a_per_layer']:.2f}")
        print(f"{'Average experts time per layer (ms):':<{align_width}} {analysis_results['avg_experts_per_layer']:.2f}")
        print(f"{'Average 2nd a2a time per layer (ms):':<{align_width}} {analysis_results['avg_2nd_a2a_per_layer']:.2f}")
        print(f"{'Average QKV GEMM time (ms):':<{align_width}} {analysis_results['avg_qkv_gemm']:.2f}")
        print(f"{'Average Attn GEMM time (ms):':<{align_width}} {analysis_results['avg_attn_gemm']:.2f}")
        print(f"{'Average Out GEMM time (ms):':<{align_width}} {analysis_results['avg_out_gemm']:.2f}")
        print(f"{'Final lm_loss:':<{align_width}} {analysis_results['final_lm_loss']:.6f}")
        print(f"{'Final moe_loss:':<{align_width}} {analysis_results['final_moe_loss']:.6f}")
        print(f"{'Peak Memory (GB):':<{align_width}} {analysis_results['peak_mem_gb']:.4f}")

        print(f'================================================================================\n'
              f'--- FLOPS Utilization Analysis (Per Layer @ 191 TFLOPS (MI250 FP16) Max) ---\n'
              f'================================================================================')

        b, s, h, h_ffn, topk, ep = args.mbs, args.seqlen, args.hidden_dim, args.expert_dim, args.topk, args.ep

        print("Expert Kernels:")
        print(f"  - {'FLOPs Formula:':<20} (2 * 2 * b * s * topk * h * h_ffn) / ep")
        print(f"  - {'Calculation:':<20} (2*2*{b}*{s}*{topk}*{h}*{h_ffn})/{ep} = {analysis_results['flops_expert'] / 1e12:.4f} TFLOPs")
        print(f"  - {'Achieved TFLOPS:':<20} {analysis_results['tflops_expert']:.2f}")
        print(f"  - {'Utilization:':<20} {analysis_results['util_expert']:.2f}%\n")

        print("QKV GEMM:")
        print(f"  - {'FLOPs Formula:':<20} 2 * 3 * b * s * h^2")
        print(f"  - {'Calculation:':<20} 2*3*{b}*{s}*{h}^2 = {analysis_results['flops_qkv'] / 1e12:.4f} TFLOPs")
        print(f"  - {'Achieved TFLOPS:':<20} {analysis_results['tflops_qkv']:.2f}")
        print(f"  - {'Utilization:':<20} {analysis_results['util_qkv']:.2f}%\n")

        print("Attention GEMM (QK^T + attn*V):")
        print(f"  - {'FLOPs Formula:':<20} 4 * b * s^2 * h")
        print(f"  - {'Calculation:':<20} 4*{b}*{s}^2*{h} = {analysis_results['flops_attention'] / 1e12:.4f} TFLOPs")
        print(f"  - {'Achieved TFLOPS:':<20} {analysis_results['tflops_attn']:.2f}")
        print(f"  - {'Utilization:':<20} {analysis_results['util_attn']:.2f}%\n")

        print("Output GEMM:")
        print(f"  - {'FLOPs Formula:':<20} 2 * b * s * h^2")
        print(f"  - {'Calculation:':<20} 2*{b}*{s}*{h}^2 = {analysis_results['flops_out'] / 1e12:.4f} TFLOPs")
        print(f"  - {'Achieved TFLOPS:':<20} {analysis_results['tflops_out']:.2f}")
        print(f"  - {'Utilization:':<20} {analysis_results['util_out']:.2f}%")

        print(f'================================================================================')

        # Predicted (planner) per-stage memory vs actual profiled per-rank reserved memory.
        try:
            compare_predicted_vs_actual_memory(args)
        except Exception as _e:
            print(f"[mem-compare] skipped due to error: {_e}")
    else:
        print("No iteration stats found in log.")

if __name__ == "__main__":
    main()