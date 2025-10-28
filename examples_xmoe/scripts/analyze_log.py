import re
import sys
import argparse
import os
import pandas as pd
from datetime import datetime
import statistics

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
    
    # --- New Arguments ---
    parser.add_argument("--seqlen", type=int, required=True, help="Sequence length.")
    parser.add_argument("--num_layers", type=int, required=True, help="Number of layers.")
    parser.add_argument("--num_experts", type=int, required=True, help="Number of experts.")
    parser.add_argument("--expert_dim", type=int, required=True, help="Expert dimension.")
    parser.add_argument("--topk", type=int, required=True, help="Top-K for MoE.")
    parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden dimension.")
    parser.add_argument("--warmup_steps", type=int, default=5, required=False, help="Warmup steps before recording TFLOPS, default 5")
    # --- ADDED MOE_TYPE ARGUMENT ---
    parser.add_argument("--moe_type", type=str, required=True, help="Type of the MoE implementation (e.g., X-MOE, DS-MOE).")
    parser.add_argument("--model_size", type=str, default="10B", required=False, help="Size of the model (e.g., 7b, 13b).")
    
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

    tflops_values, samples_per_sec_values, elapsed_times = [], [], []
    lm_losses, moe_losses = [], []
    first_a2a_values, experts_values, second_a2a_values = [], [], []
    qkv_gemm_values, attn_gemm_values, out_gemm_values = [], [], []

    temp_qkv, temp_attn, temp_out = [], [], []

    with open(log_file, "r") as f:
        for line in f:
            iteration_match = iteration_pattern.search(line)
            gemm_match = gemm_pattern.search(line)
            time_match = time_pattern.search(line)

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

    if temp_qkv:
        qkv_gemm_values.append(statistics.mean(temp_qkv))
        attn_gemm_values.append(statistics.mean(temp_attn))
        out_gemm_values.append(statistics.mean(temp_out))

    if not tflops_values: return None

    # Determine the starting index for calculations (post-warmup)
    if len(tflops_values) > warmup_steps:
        start_index = warmup_steps
    else:
        print(f"Warning: Less than {warmup_steps} iterations found. Calculating stats over all available iterations.", file=sys.stderr)
        start_index = 0

    # Safely slice each list, creating new lists for calculation
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
    filename = f"{date_str}-{args.seqlen}-{args.num_layers}-{args.num_experts}-{args.expert_dim}-{args.topk}-{args.hidden_dim}-results.xlsx"
    file_path = f"/work1/mzhang/zixianw4/{filename}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    run_id = f"{args.moe_type}-PP{args.pp}-EP{args.ep}-GBS{args.gbs}-MBS{args.mbs}-SLURM_ID-{args.slurm_job_id}"

    header = [
        "Name", "MOE Type", "Model Size", "AVG TFLOPs", "TFLOPs Std Dev", "Iterations", "Actual Iterations", 
        "Nodes", "GPUs/node", "PP", "EP", "DP", "TP", "GBS", "MBS",
        "Final lm_loss", 
        "1st_a2a_PL (ms)", "experts_PL (ms)", "2nd_a2a_PL (ms)",
        "QKV GEMM PL (ms)", "Attn GEMM PL (ms)", "Out GEMM PL (ms)",
        "Expert Util (%)", "QKV Util (%)", "Attn Util (%)", "Out Util (%)",
        "SLURM_JOB_ID"
    ]
    
    main_row_dict = {
        "Name": run_id, "MOE Type": args.moe_type, "Model Size": args.model_size,
        "AVG TFLOPs": f"{data['avg_tflops']:.2f}", "TFLOPs Std Dev": f"{data['tflops_std_dev']:.2f}",
        "Iterations": args.iterations, "Actual Iterations": data['matched_iterations'],
        "Nodes": args.nodes, "GPUs/node": args.gpus_per_node, "MBS": args.mbs, "GBS": args.gbs,
        "Final lm_loss": f"{data['final_lm_loss']:.6f}", 
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
        "SLURM_JOB_ID": args.slurm_job_id,
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

def main():
    """Main function to run the analysis and save results."""
    args = parse_arguments()
    analysis_results = analyze_log(args.log_file, args.num_layers, args.warmup_steps, args)

    if analysis_results:
        output_file = write_to_xlsx(analysis_results, args)
        
        metric_names = ["Log Directory", "Actual ran iterations", "Average TFLOPs", "TFLOPs Standard Deviation",
                        "Average samples/sec", "Average elapsed time per iteration (ms)", "Average 1st a2a time per layer (ms)",
                        "Average experts time per layer (ms)", "Average 2nd a2a time per layer (ms)", "Average QKV GEMM time (ms)",
                        "Average Attn GEMM time (ms)", "Average Out GEMM time (ms)", "Final lm_loss", "Final moe_loss"]
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
        
        print(f'================================================================================\n'
              f'--- FLOPS Utilization Analysis (Per Layer @ 181 TFLOPS (MI250 FP16) Max) ---\n'
              f'================================================================================')
        
        b, s, h, h_ffn, topk, ep = args.mbs, args.seqlen, args.hidden_dim, args.expert_dim, args.topk, args.ep
        
        print("Expert Kernels:")
        print(f"  - {'FLOPs Formula:':<20} (2 * 2 * b * s * topk * h * h_ffn) / ep")
        # --- UPDATED: Print in TFLOPs ---
        print(f"  - {'Calculation:':<20} (2*2*{b}*{s}*{topk}*{h}*{h_ffn})/{ep}")
        print(f"  - {'Achieved TFLOPS:':<20} {analysis_results['tflops_expert']:.2f}")
        print(f"  - {'Utilization:':<20} {analysis_results['util_expert']:.2f}%\n")

        print("QKV GEMM:")
        print(f"  - {'FLOPs Formula:':<20} 2 * 3 * b * s * h^2")
        # --- UPDATED: Print in TFLOPs ---
        print(f"  - {'Calculation:':<20} 2*3*{b}*{s}*{h}^2")
        print(f"  - {'Achieved TFLOPS:':<20} {analysis_results['tflops_qkv']:.2f}")
        print(f"  - {'Utilization:':<20} {analysis_results['util_qkv']:.2f}%\n")
        
        print("Attention GEMM (QK^T + attn*V):")
        print(f"  - {'FLOPs Formula:':<20} 4 * b * s^2 * h")
        # --- UPDATED: Print in TFLOPs ---
        print(f"  - {'Calculation:':<20} 4*{b}*{s}^2*{h}")
        print(f"  - {'Achieved TFLOPS:':<20} {analysis_results['tflops_attn']:.2f}")
        print(f"  - {'Utilization:':<20} {analysis_results['util_attn']:.2f}%\n")

        print("Output GEMM:")
        print(f"  - {'FLOPs Formula:':<20} 2 * b * s * h^2")
        # --- UPDATED: Print in TFLOPs ---
        print(f"  - {'Calculation:':<20} 2*{b}*{s}*{h}^2")
        print(f"  - {'Achieved TFLOPS:':<20} {analysis_results['tflops_out']:.2f}")
        print(f"  - {'Utilization:':<20} {analysis_results['util_out']:.2f}%")

        print(f'================================================================================')
    else:
        print("No iteration stats found in log.")

if __name__ == "__main__":
    main()