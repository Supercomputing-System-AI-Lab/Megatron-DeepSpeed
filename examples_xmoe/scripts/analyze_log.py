import re
import sys
import argparse
import os
import pandas as pd
from datetime import datetime

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze log files and store results in XLSX.")
    # --- Existing Arguments ---
    parser.add_argument("log_file", type=str, help="Path to the log file.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations.")
    parser.add_argument("--nodes", type=int, required=True, help="Number of nodes.")
    parser.add_argument("--gpus_per_node", type=int, required=True, help="Number of GPUs per node.")
    parser.add_argument("--megatron_pp", type=int, required=True, help="Megatron PP size.")
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
    # --- ADDED MOE_TYPE ARGUMENT ---
    parser.add_argument("--moe_type", type=str, required=True, help="Type of the MoE implementation (e.g., X-MOE, DS-MOE).")
    
    return parser.parse_args()

def analyze_log(log_file):
    """Analyzes the log file to extract performance metrics and all losses."""
    iteration_pattern = re.compile(
        r"iteration\s+\d+/\s+\d+\s+\|.*?"
        r"elapsed time per iteration \(ms\):\s+([0-9.]+)\s+\|.*?"
        r"lm loss:\s+([0-9.E+-]+)"
        r"(?:\s+\|\s+moe loss:\s+([0-9.E+-]+))?"
        r".*?samples per second:\s+([0-9.]+)\s+\|"
        r"\s+TFLOPs:\s+([0-9.]+)"
    )

    tflops_values, samples_per_sec_values, elapsed_times = [], [], []
    lm_losses, moe_losses = [], []

    with open(log_file, "r") as f:
        for line in f:
            match = iteration_pattern.search(line)
            if match:
                elapsed_times.append(float(match.group(1)))
                lm_losses.append(float(match.group(2)))
                if match.group(3):
                    moe_losses.append(float(match.group(3)))
                else:
                    moe_losses.append(0.0)
                samples_per_sec_values.append(float(match.group(4)))
                tflops_values.append(float(match.group(5)))

    if not tflops_values:
        return None

    matched_iterations = len(tflops_values)
    avg_tflops = sum(tflops_values) / matched_iterations
    avg_samples = sum(samples_per_sec_values) / matched_iterations
    avg_elapsed = sum(elapsed_times) / matched_iterations

    return {
        "matched_iterations": matched_iterations,
        "avg_tflops": avg_tflops,
        "avg_samples": avg_samples,
        "avg_elapsed": avg_elapsed,
        "final_lm_loss": lm_losses[-1],
        "final_moe_loss": moe_losses[-1],
        "lm_losses": lm_losses,
        "moe_losses": moe_losses,
    }

def write_to_xlsx(data, args):
    """Writes the collected data to a single XLSX file with two sheets."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    filename = (
        f"{date_str}-{args.seqlen}-{args.num_layers}-{args.num_experts}-"
        f"{args.expert_dim}-{args.topk}-{args.hidden_dim}-results.xlsx"
    )
    file_path = f"/work1/mzhang/zixianw4/{filename}"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # ===================================================================
    # Prepare Data for Both Sheets
    # ===================================================================
    
    # --- MODIFIED: Unique ID now includes the MoE type ---
    run_id = (
        f"{args.moe_type}-PP{args.megatron_pp}-EP{args.ep}-"
        f"GBS{args.gbs}-MBS{args.mbs}-SLURM_ID-{args.slurm_job_id}"
    )

    # --- MODIFIED: Sheet 1 Header now includes MoE Type column ---
    header = [
        "Name", "MOE Type", "AVG TFLOPs", "Iterations", "Nodes", "GPUs/node",
        "Megatron PP", "EP", "DP", "TP", "GBS", "MBS",
        "Final lm_loss", "SLURM_JOB_ID"
    ]
    # --- MODIFIED: Main row dictionary now includes MoE Type ---
    main_row_dict = {
        "Name": run_id, 
        "MOE Type": args.moe_type, # <-- ADDED
        "AVG TFLOPs": f"{data['avg_tflops']:.2f}",
        "Iterations": args.iterations, 
        "Nodes": args.nodes,
        "GPUs/node": args.gpus_per_node, 
        "MBS": args.mbs, 
        "GBS": args.gbs,
        "Final lm_loss": f"{data['final_lm_loss']:.6f}", 
        "Megatron PP": args.megatron_pp,
        "EP": args.ep, 
        "DP": args.dp, 
        "TP": args.tp, 
        "SLURM_JOB_ID": args.slurm_job_id,
    }
    new_main_df = pd.DataFrame([main_row_dict])[header]

    # --- Sheet 2: Loss per Iteration (Wide Format) ---
    loss_row_dict = {"id": run_id}
    for i, loss in enumerate(data['lm_losses']):
        loss_row_dict[f'loss_{i+1}'] = loss
    new_loss_df = pd.DataFrame([loss_row_dict])

    # ===================================================================
    # Read Existing File (if any) and Append Data
    # ===================================================================
    try:
        # Load existing data from both sheets
        with pd.ExcelFile(file_path) as xls:
            existing_main_df = pd.read_excel(xls, 'Main_Results')
            existing_loss_df = pd.read_excel(xls, 'Loss_per_Iteration')
        
        # Append new data
        combined_main_df = pd.concat([existing_main_df, new_main_df], ignore_index=True)
        combined_loss_df = pd.concat([existing_loss_df, new_loss_df], ignore_index=True)

    except FileNotFoundError:
        # If file doesn't exist, the new data is the only data
        combined_main_df = new_main_df
        combined_loss_df = new_loss_df

    # ===================================================================
    # Write Both DataFrames to the Excel File
    # ===================================================================
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        combined_main_df.to_excel(writer, sheet_name='Main_Results', index=False)
        combined_loss_df.to_excel(writer, sheet_name='Loss_per_Iteration', index=False)
    
    return file_path

def main():
    """Main function to run the analysis and save results."""
    args = parse_arguments()
    analysis_results = analyze_log(args.log_file)

    if analysis_results:
        print(f"Average TFLOPs over {analysis_results['matched_iterations']} iterations: {analysis_results['avg_tflops']:.2f}")
        print(f"Average samples/sec: {analysis_results['avg_samples']:.3f}")
        print(f"Average elapsed time per iteration (ms): {analysis_results['avg_elapsed']:.1f}")
        print(f"Final lm_loss: {analysis_results['final_lm_loss']:.6f}")
        print(f"Final moe_loss: {analysis_results['final_moe_loss']:.6f}")

        # Write all data to the single Excel file
        output_file = write_to_xlsx(analysis_results, args)
        print(f"\nResults successfully saved to: {output_file}")

    else:
        print("No iteration stats found in log.")

if __name__ == "__main__":
    main()
    
    
# import re
# import sys

# def main():
#     if len(sys.argv) != 2:
#         print(f"Usage: python {sys.argv[0]} <log_file>")
#         sys.exit(1)

#     log_file = sys.argv[1]

#     tflops_values = []
#     lm_loss = None
#     moe_loss = None

#     # Regex to match iteration lines with TFLOPs, lm loss, and moe loss
#     # pattern = re.compile(
#     #     r"iteration\s+\d+/\s+\d+\s+\|.*?"
#     #     r"elapsed time per iteration \(ms\):\s+([0-9.]+)\s+\|.*?"
#     #     r"lm loss:\s+([0-9.E+-]+)\s+\|\s+moe loss:\s+([0-9.E+-]+).*?"
#     #     r"samples per second:\s+([0-9.]+)\s+\|"
#     #     r"\s+TFLOPs:\s+([0-9.]+)"
#     # )
#     pattern = re.compile(
#     r"iteration\s+\d+/\s+\d+\s+\|.*?"
#     r"elapsed time per iteration \(ms\):\s+([0-9.]+)\s+\|.*?"
#     r"lm loss:\s+([0-9.E+-]+)"
#     r"(?:\s+\|\s+moe loss:\s+([0-9.E+-]+))?"   # <- optional group
#     r".*?samples per second:\s+([0-9.]+)\s+\|"
#     r"\s+TFLOPs:\s+([0-9.]+)"
# )

#     elapsed_times = []
#     tflops_values = []
#     samples_per_sec_values = []
#     lm_loss = None
#     moe_loss = None

#     with open(log_file, "r") as f:
#         for line in f:
#             match = pattern.search(line)
#             if match:
#                 elapsed = float(match.group(1))
#                 lm_loss = float(match.group(2))
#                 moe_loss = float(match.group(3)) if match.group(3) else 0
#                 samples_per_sec = float(match.group(4))
#                 tflops = float(match.group(5))

#                 elapsed_times.append(elapsed)
#                 samples_per_sec_values.append(samples_per_sec)
#                 tflops_values.append(tflops)

#     if tflops_values:
#         avg_tflops = sum(tflops_values) / len(tflops_values)
#         avg_samples = sum(samples_per_sec_values) / len(samples_per_sec_values)
#         avg_elapsed = sum(elapsed_times) / len(elapsed_times)

#         print(f"Average TFLOPs over {len(tflops_values)} iterations: {avg_tflops:.2f}")
#         print(f"Average samples/sec: {avg_samples:.3f}")
#         print(f"Average elapsed time per iteration (ms): {avg_elapsed:.1f}")
#         print(f"Final lm_loss: {lm_loss:.6f}")
#         print(f"Final moe_loss: {moe_loss:.6f}")
#     else:
#         print("No iteration stats found in log.")


# if __name__ == "__main__":
#     main()
