import re
import sys
import argparse
import os
import pandas as pd
from datetime import datetime
import statistics
import json 

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
    parser.add_argument("--slurm_job_id", type=str, required=True, help="SLURM Job ID.")
    parser.add_argument("--seqlen", type=int, required=True, help="Sequence length.")
    parser.add_argument("--num_layers", type=int, required=True, help="Number of layers.")
    parser.add_argument("--num_experts", type=int, required=True, help="Number of experts.")
    parser.add_argument("--expert_dim", type=int, required=True, help="Expert dimension.")
    parser.add_argument("--topk", type=int, required=True, help="Top-K for MoE.")
    parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden dimension.")
    parser.add_argument("--warmup_steps", type=int, default=5, required=False, help="Warmup steps before recording TFLOPS, default 5")
    parser.add_argument("--moe_type", type=str, required=True, help="Type of the MoE implementation (e.g., X-MOE, DS-MOE).")
    parser.add_argument("--model_size", type=str, default="10B", required=False, help="Size of the model (e.g., 7b, 13b).")

    # --- NEWLY ADDED ARGUMENTS ---
    parser.add_argument("--activation_checkpointing", type=str, required=True, help="Activation checkpointing status (e.g., true/false).")
    parser.add_argument("--checkpoint_interval", type=int, required=True, help="The interval for saving model checkpoints.")
    
    # Updated to include defaults as requested
    parser.add_argument("--dynamic_checkpoint", type=str, required=False, default="False", help="Whether dynamic checkpointing activations. True if arg is str(True)")
    parser.add_argument("--uneven_pp", type=str, required=False, default="False", help="Whether use uneven pipeline layer partitioning. True if arg is str(True)")

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
    peak_mem_pattern = re.compile(r"Overall Peak Reserved:\s+([0-9.]+)\s+GB")


    tflops_values, samples_per_sec_values, elapsed_times = [], [], []
    lm_losses, moe_losses = [], []
    first_a2a_values, experts_values, second_a2a_values = [], [], []
    qkv_gemm_values, attn_gemm_values, out_gemm_values = [], [], []
    peak_mem_values = []


    temp_qkv, temp_attn, temp_out = [], [], []

    with open(log_file, "r") as f:
        for line in f:
            iteration_match = iteration_pattern.search(line)
            gemm_match = gemm_pattern.search(line)
            time_match = time_pattern.search(line)
            peak_mem_match = peak_mem_pattern.search(line)


            if iteration_match:
                if temp_qkv:
                    qkv_gemm_values.append(statistics.mean(temp_qkv))
                    attn_gemm_values.append(statistics.mean(temp_attn))
                    out_gemm_values.append(statistics.mean(temp_out))
                    temp_qkv, temp_attn, temp_out = [], [], []

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
                temp_qkv.append(float(gemm_match.group(1)))
                temp_attn.append(float(gemm_match.group(2)))
                temp_out.append(float(gemm_match.group(3)))
            
            if peak_mem_match:
                peak_mem_values.append(float(peak_mem_match.group(1)))


    if temp_qkv:
        qkv_gemm_values.append(statistics.mean(temp_qkv))
        attn_gemm_values.append(statistics.mean(temp_attn))
        out_gemm_values.append(statistics.mean(temp_out))

    if not tflops_values: return None

    if len(tflops_values) > warmup_steps:
        start_index = warmup_steps
        print (f'analyze_log: warmup_steps: {warmup_steps}')
    else:
        print(f"Warning: Less than {warmup_steps} iterations found. Calculating stats over all available iterations.", file=sys.stderr)
        start_index = 0

    tflops_for_calc = tflops_values[start_index:]
    samples_for_calc = samples_per_sec_values[start_index:]
    elapsed_for_calc = elapsed_times[start_index:]
    first_a2a_for_calc = first_a2a_values[start_index:]
    experts_for_calc = experts_values[start_index:]
    second_a2a_for_calc = second_a2a_values[start_index:]
    qkv_for_calc = qkv_gemm_values[start_index:]
    attn_for_calc = attn_gemm_values[start_index:]
    out_for_calc = out_gemm_values[start_index:]

    avg_tflops = statistics.mean(tflops_for_calc) if tflops_for_calc else 0.0
    tflops_std_dev = statistics.stdev(tflops_for_calc) if len(tflops_for_calc) > 1 else 0.0
    avg_samples = statistics.mean(samples_for_calc) if samples_for_calc else 0.0
    avg_elapsed = statistics.mean(elapsed_for_calc) if elapsed_for_calc else 0.0

    avg_experts_time_pl = (statistics.mean(experts_for_calc) / num_layers) if experts_for_calc and num_layers > 0 else 0.0
    avg_qkv_gemm_time = statistics.mean(qkv_for_calc) if qkv_for_calc else 0.0
    avg_attn_gemm_time = statistics.mean(attn_for_calc) if attn_for_calc else 0.0
    avg_out_gemm_time = statistics.mean(out_for_calc) if out_for_calc else 0.0

    THEORETICAL_MAX_TFLOPS = 181.0
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
    }
    new_main_df = pd.DataFrame([main_row_dict])[header]

    loss_row_dict = {"id": run_id}
    for i, loss in enumerate(data.get('lm_losses', [])):
        loss_row_dict[f'loss_{i+1}'] = loss
    new_loss_df = pd.DataFrame([loss_row_dict])

    try:
        with pd.ExcelFile(file_path) as xls:
            existing_main_df = pd.read_excel(xls, 'Main_Results')
            existing_loss_df = pd.read_excel(xls, 'Loss_per_Iteration')
        combined_main_df = pd.concat([existing_main_df, new_main_df], ignore_index=True)
        combined_loss_df = pd.concat([existing_loss_df, new_loss_df], ignore_index=True)
    except FileNotFoundError:
        combined_main_df = new_main_df
        combined_loss_df = new_loss_df

    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        combined_main_df.to_excel(writer, sheet_name='Main_Results', index=False)
        combined_loss_df.to_excel(writer, sheet_name='Loss_per_Iteration', index=False)

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



def main():
    """Main function to run the analysis and save results."""
    args = parse_arguments()
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
              f'--- FLOPS Utilization Analysis (Per Layer @ 181 TFLOPS (MI250 FP16) Max) ---\n'
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
    else:
        print("No iteration stats found in log.")

if __name__ == "__main__":
    main()