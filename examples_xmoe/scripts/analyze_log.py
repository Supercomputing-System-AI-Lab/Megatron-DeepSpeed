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

def analyze_log(log_file, num_layers, warmup_steps=5):
    """
    Analyzes the log file to extract performance metrics, including detailed
    GEMM timings, calculating averages after excluding warmup steps.
    """
    iteration_pattern = re.compile(
        r"iteration\s+\d+/\s+\d+\s+\|.*?"
        r"elapsed time per iteration \(ms\):\s+([0-9.]+)\s+\|.*?"
        r"lm loss:\s+([0-9.E+-]+)"
        r"(?:\s+\|\s+moe loss:\s+([0-9.E+-]+))?"
        r".*?samples per second:\s+([0-9.]+)\s+\|"
        r"\s+TFLOPs:\s+([0-9.]+)"
    )
    
    time_pattern = re.compile(
        r"1st_a2a:\s+([0-9.]+),\s+experts:\s+([0-9.]+),\s+2nd_a2a:\s+([0-9.]+)"
    )

    # --- NEW: Regex for attention GEMM timings ---
    gemm_pattern = re.compile(
        r"qkv_gemm:\s+([0-9.]+)\s+\|\s+attn_gemm:\s+([0-9.]+)\s+\|\s+out_gemm:\s+([0-9.]+)"
    )

    # Main lists for per-iteration values
    tflops_values, samples_per_sec_values, elapsed_times = [], [], []
    lm_losses, moe_losses = [], []
    first_a2a_values, experts_values, second_a2a_values = [], [], []
    qkv_gemm_values, attn_gemm_values, out_gemm_values = [], [], []

    # Temp lists to average GEMM timings within a single iteration
    temp_qkv, temp_attn, temp_out = [], [], []

    with open(log_file, "r") as f:
        for line in f:
            iteration_match = iteration_pattern.search(line)
            gemm_match = gemm_pattern.search(line)
            time_match = time_pattern.search(line)

            if iteration_match:
                # A new iteration line is found. First, process the GEMM values
                # collected for the *previous* iteration.
                if temp_qkv: # Check if we have values from micro-batches to process
                    qkv_gemm_values.append(statistics.mean(temp_qkv))
                    attn_gemm_values.append(statistics.mean(temp_attn))
                    out_gemm_values.append(statistics.mean(temp_out))
                    # Reset temp lists for the new iteration
                    temp_qkv, temp_attn, temp_out = [], [], []

                # Now, process the current iteration line as usual
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
                # A GEMM line was found. Add its values to the temp lists.
                temp_qkv.append(float(gemm_match.group(1)))
                temp_attn.append(float(gemm_match.group(2)))
                temp_out.append(float(gemm_match.group(3)))

    # After the loop, process the GEMM values for the very last iteration
    if temp_qkv:
        qkv_gemm_values.append(statistics.mean(temp_qkv))
        attn_gemm_values.append(statistics.mean(temp_attn))
        out_gemm_values.append(statistics.mean(temp_out))

    if not tflops_values:
        return None

    # Exclude warmup steps for all metrics
    if len(tflops_values) > warmup_steps:
        tflops_for_calc = tflops_values[warmup_steps:]
        samples_for_calc = samples_per_sec_values[warmup_steps:]
        first_a2a_for_calc = first_a2a_values[warmup_steps:]
        experts_for_calc = experts_values[warmup_steps:]
        second_a2a_for_calc = second_a2a_values[warmup_steps:]
        qkv_for_calc = qkv_gemm_values[warmup_steps:]
        attn_for_calc = attn_gemm_values[warmup_steps:]
        out_for_calc = out_gemm_values[warmup_steps:]
    else:
        print(f"Warning: Less than {warmup_steps} iterations found. Calculating stats over all available iterations.", file=sys.stderr)
        tflops_for_calc, samples_for_calc = tflops_values, samples_per_sec_values
        first_a2a_for_calc, experts_for_calc, second_a2a_for_calc = first_a2a_values, experts_values, second_a2a_values
        qkv_for_calc, attn_for_calc, out_for_calc = qkv_gemm_values, attn_gemm_values, out_gemm_values

    # Perform final calculations on post-warmup data
    avg_tflops = statistics.mean(tflops_for_calc) if tflops_for_calc else 0.0
    avg_1st_a2a = statistics.mean(first_a2a_for_calc) if first_a2a_for_calc else 0.0
    avg_experts = statistics.mean(experts_for_calc) if experts_for_calc else 0.0
    avg_2nd_a2a = statistics.mean(second_a2a_for_calc) if second_a2a_for_calc else 0.0
    
    avg_qkv_gemm = statistics.mean(qkv_for_calc) if qkv_for_calc else 0.0
    avg_attn_gemm = statistics.mean(attn_for_calc) if attn_for_calc else 0.0
    avg_out_gemm = statistics.mean(out_for_calc) if out_for_calc else 0.0

    return {
        "matched_iterations": len(tflops_values),
        "avg_tflops": avg_tflops,
        "tflops_std_dev": statistics.stdev(tflops_for_calc) if len(tflops_for_calc) > 1 else 0.0,
        "avg_samples": statistics.mean(samples_for_calc) if samples_for_calc else 0.0,
        "avg_elapsed": statistics.mean(elapsed_times[warmup_steps:]) if len(elapsed_times) > warmup_steps else statistics.mean(elapsed_times) if elapsed_times else 0.0,
        "final_lm_loss": lm_losses[-1] if lm_losses else 0.0,
        "final_moe_loss": moe_losses[-1] if moe_losses else 0.0,
        "lm_losses": lm_losses,
        "avg_1st_a2a_per_layer": (avg_1st_a2a / num_layers) if num_layers > 0 else 0.0,
        "avg_experts_per_layer": (avg_experts / num_layers) if num_layers > 0 else 0.0,
        "avg_2nd_a2a_per_layer": (avg_2nd_a2a / num_layers) if num_layers > 0 else 0.0,
        "avg_qkv_gemm": avg_qkv_gemm,
        "avg_attn_gemm": avg_attn_gemm,
        "avg_out_gemm": avg_out_gemm,
    }

def write_to_xlsx(data, args):
    """Writes the collected data to a single XLSX file with two sheets."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{date_str}-{args.seqlen}-{args.num_layers}-{args.num_experts}-{args.expert_dim}-{args.topk}-{args.hidden_dim}-results.xlsx"
    file_path = f"/work1/mzhang/zixianw4/{filename}"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    run_id = f"{args.moe_type}-PP{args.pp}-EP{args.ep}-GBS{args.gbs}-MBS{args.mbs}-SLURM_ID-{args.slurm_job_id}"

    # --- UPDATED: Header with new GEMM timing columns ---
    header = [
        "Name", "MOE Type", "Model Size", "AVG TFLOPs", "TFLOPs Std Dev", "Iterations", "Actual Iterations", 
        "Nodes", "GPUs/node", "PP", "EP", "DP", "TP", "GBS", "MBS",
        "Final lm_loss", 
        "1st_a2a_PL (ms)", "experts_PL (ms)", "2nd_a2a_PL (ms)",
        "QKV GEMM PL (ms)", "Attn GEMM PL (ms)", "Out GEMM PL (ms)",
        "SLURM_JOB_ID"
    ]
    
    # --- UPDATED: Main row dictionary with new GEMM data ---
    main_row_dict = {
        "Name": run_id, 
        "MOE Type": args.moe_type,
        "Model Size": args.model_size,
        "AVG TFLOPs": f"{data['avg_tflops']:.2f}",
        "TFLOPs Std Dev": f"{data['tflops_std_dev']:.2f}",
        "Iterations": args.iterations, 
        "Actual Iterations": data['matched_iterations'],
        "Nodes": args.nodes,
        "GPUs/node": args.gpus_per_node, 
        "MBS": args.mbs, 
        "GBS": args.gbs,
        "Final lm_loss": f"{data['final_lm_loss']:.6f}", 
        "PP": args.pp, "EP": args.ep, "DP": args.dp, "TP": args.tp, 
        "1st_a2a_PL (ms)": f"{data['avg_1st_a2a_per_layer']:.2f}",
        "experts_PL (ms)": f"{data['avg_experts_per_layer']:.2f}",
        "2nd_a2a_PL (ms)": f"{data['avg_2nd_a2a_per_layer']:.2f}",
        "QKV GEMM PL (ms)": f"{data['avg_qkv_gemm']:.2f}",
        "Attn GEMM PL (ms)": f"{data['avg_attn_gemm']:.2f}",
        "Out GEMM PL (ms)": f"{data['avg_out_gemm']:.2f}",
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
    analysis_results = analyze_log(args.log_file, args.num_layers, args.warmup_steps)

    if analysis_results:
        output_file = write_to_xlsx(analysis_results, args)
        print(f'\n\n'
              f'================================================================================\n'
              f'--- Analyzing FLOPS Utilization from Logs ---\n'
              f'Log Directory: {output_file}\n'
              f'================================================================================')
        print(f"Actual ran iterations: {analysis_results['matched_iterations']}")
        print(f"Average TFLOPs (post-warmup): {analysis_results['avg_tflops']:.2f}")
        print(f"TFLOPs Standard Deviation (post-warmup): {analysis_results['tflops_std_dev']:.2f}")
        print(f"Average samples/sec (post-warmup): {analysis_results['avg_samples']:.3f}")
        print(f"Average elapsed time per iteration (ms) (post-warmup): {analysis_results['avg_elapsed']:.1f}")
        print(f"Average 1st a2a time per layer (ms): {analysis_results['avg_1st_a2a_per_layer']:.2f}")
        print(f"Average experts time per layer (ms): {analysis_results['avg_experts_per_layer']:.2f}")
        print(f"Average 2nd a2a time per layer (ms): {analysis_results['avg_2nd_a2a_per_layer']:.2f}")
        # --- NEW: Print GEMM timing results ---
        print(f"Average QKV GEMM time (ms): {analysis_results['avg_qkv_gemm']:.2f}")
        print(f"Average Attn GEMM time (ms): {analysis_results['avg_attn_gemm']:.2f}")
        print(f"Average Out GEMM time (ms): {analysis_results['avg_out_gemm']:.2f}")
        print(f"Final lm_loss: {analysis_results['final_lm_loss']:.6f}")
        print(f"Final moe_loss: {analysis_results['final_moe_loss']:.6f}")
        print(f'================================================================================')
    else:
        print("No iteration stats found in log.")

if __name__ == "__main__":
    main()