# import re
# import sys
# import argparse
# import os
# import pandas as pd
# from datetime import datetime

# def parse_arguments():
#     """Parses command-line arguments."""
#     parser = argparse.ArgumentParser(description="Analyze log files and store results in XLSX.")
#     # --- Existing Arguments ---
#     parser.add_argument("log_file", type=str, help="Path to the log file.")
#     parser.add_argument("--iterations", type=int, required=True, help="Number of iterations.")
#     parser.add_argument("--nodes", type=int, required=True, help="Number of nodes.")
#     parser.add_argument("--gpus_per_node", type=int, required=True, help="Number of GPUs per node.")
#     parser.add_argument("--megatron_pp", type=int, required=True, help="Megatron PP size.")
#     parser.add_argument("--ep", type=int, required=True, help="EP size.")
#     parser.add_argument("--dp", type=int, required=True, help="DP size.")
#     parser.add_argument("--tp", type=int, required=True, help="TP size.")
#     parser.add_argument("--gbs", type=int, required=True, help="Global Batch Size.")
#     parser.add_argument("--mbs", type=int, required=True, help="Micro Batch Size.")
#     parser.add_argument("--slurm_job_id", type=str, required=True, help="SLURM Job ID.")
    
#     # --- New Arguments ---
#     parser.add_argument("--seqlen", type=int, required=True, help="Sequence length.")
#     parser.add_argument("--num_layers", type=int, required=True, help="Number of layers.")
#     parser.add_argument("--num_experts", type=int, required=True, help="Number of experts.")
#     parser.add_argument("--expert_dim", type=int, required=True, help="Expert dimension.")
#     parser.add_argument("--topk", type=int, required=True, help="Top-K for MoE.")
#     parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden dimension.")
#     # --- ADDED MOE_TYPE ARGUMENT ---
#     parser.add_argument("--moe_type", type=str, required=True, help="Type of the MoE implementation (e.g., X-MOE, DS-MOE).")
    
#     return parser.parse_args()

# def analyze_log(log_file):
#     """Analyzes the log file to extract performance metrics and all losses."""
#     iteration_pattern = re.compile(
#         r"iteration\s+\d+/\s+\d+\s+\|.*?"
#         r"elapsed time per iteration \(ms\):\s+([0-9.]+)\s+\|.*?"
#         r"lm loss:\s+([0-9.E+-]+)"
#         r"(?:\s+\|\s+moe loss:\s+([0-9.E+-]+))?"
#         r".*?samples per second:\s+([0-9.]+)\s+\|"
#         r"\s+TFLOPs:\s+([0-9.]+)"
#     )

#     tflops_values, samples_per_sec_values, elapsed_times = [], [], []
#     lm_losses, moe_losses = [], []

#     with open(log_file, "r") as f:
#         for line in f:
#             match = iteration_pattern.search(line)
#             if match:
#                 elapsed_times.append(float(match.group(1)))
#                 lm_losses.append(float(match.group(2)))
#                 if match.group(3):
#                     moe_losses.append(float(match.group(3)))
#                 else:
#                     moe_losses.append(0.0)
#                 samples_per_sec_values.append(float(match.group(4)))
#                 tflops_values.append(float(match.group(5)))

#     if not tflops_values:
#         return None

#     matched_iterations = len(tflops_values)
#     avg_tflops = sum(tflops_values) / matched_iterations
#     avg_samples = sum(samples_per_sec_values) / matched_iterations
#     avg_elapsed = sum(elapsed_times) / matched_iterations

#     return {
#         "matched_iterations": matched_iterations,
#         "avg_tflops": avg_tflops,
#         "avg_samples": avg_samples,
#         "avg_elapsed": avg_elapsed,
#         "final_lm_loss": lm_losses[-1],
#         "final_moe_loss": moe_losses[-1],
#         "lm_losses": lm_losses,
#         "moe_losses": moe_losses,
#     }

# def write_to_xlsx(data, args):
#     """Writes the collected data to a single XLSX file with two sheets."""
#     date_str = datetime.now().strftime("%Y-%m-%d")
    
#     filename = (
#         f"{date_str}-{args.seqlen}-{args.num_layers}-{args.num_experts}-"
#         f"{args.expert_dim}-{args.topk}-{args.hidden_dim}-results.xlsx"
#     )
#     file_path = f"/work1/mzhang/henryj/{filename}"

#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
#     # ===================================================================
#     # Prepare Data for Both Sheets
#     # ===================================================================
    
#     # --- MODIFIED: Unique ID now includes the MoE type ---
#     run_id = (
#         f"{args.moe_type}-PP{args.megatron_pp}-EP{args.ep}-"
#         f"GBS{args.gbs}-MBS{args.mbs}-SLURM_ID-{args.slurm_job_id}"
#     )

#     # --- MODIFIED: Sheet 1 Header now includes MoE Type column ---
#     header = [
#         "Name", "MOE Type", "AVG TFLOPs", "Iterations", "Nodes", "GPUs/node",
#         "Megatron PP", "EP", "DP", "TP", "GBS", "MBS",
#         "Final lm_loss", "SLURM_JOB_ID"
#     ]
#     # --- MODIFIED: Main row dictionary now includes MoE Type ---
#     main_row_dict = {
#         "Name": run_id, 
#         "MOE Type": args.moe_type, # <-- ADDED
#         "AVG TFLOPs": f"{data['avg_tflops']:.2f}",
#         "Iterations": args.iterations, 
#         "Nodes": args.nodes,
#         "GPUs/node": args.gpus_per_node, 
#         "MBS": args.mbs, 
#         "GBS": args.gbs,
#         "Final lm_loss": f"{data['final_lm_loss']:.6f}", 
#         "Megatron PP": args.megatron_pp,
#         "EP": args.ep, 
#         "DP": args.dp, 
#         "TP": args.tp, 
#         "SLURM_JOB_ID": args.slurm_job_id,
#     }
#     new_main_df = pd.DataFrame([main_row_dict])[header]

#     # --- Sheet 2: Loss per Iteration (Wide Format) ---
#     loss_row_dict = {"id": run_id}
#     for i, loss in enumerate(data['lm_losses']):
#         loss_row_dict[f'loss_{i+1}'] = loss
#     new_loss_df = pd.DataFrame([loss_row_dict])

#     # ===================================================================
#     # Read Existing File (if any) and Append Data
#     # ===================================================================
#     try:
#         # Load existing data from both sheets
#         with pd.ExcelFile(file_path) as xls:
#             existing_main_df = pd.read_excel(xls, 'Main_Results')
#             existing_loss_df = pd.read_excel(xls, 'Loss_per_Iteration')
        
#         # Append new data
#         combined_main_df = pd.concat([existing_main_df, new_main_df], ignore_index=True)
#         combined_loss_df = pd.concat([existing_loss_df, new_loss_df], ignore_index=True)

#     except FileNotFoundError:
#         # If file doesn't exist, the new data is the only data
#         combined_main_df = new_main_df
#         combined_loss_df = new_loss_df

#     # ===================================================================
#     # Write Both DataFrames to the Excel File
#     # ===================================================================
#     with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
#         combined_main_df.to_excel(writer, sheet_name='Main_Results', index=False)
#         combined_loss_df.to_excel(writer, sheet_name='Loss_per_Iteration', index=False)
    
#     return file_path

# def main():
#     """Main function to run the analysis and save results."""
#     args = parse_arguments()
#     analysis_results = analyze_log(args.log_file)

#     if analysis_results:
#         print(f"Average TFLOPs over {analysis_results['matched_iterations']} iterations: {analysis_results['avg_tflops']:.2f}")
#         print(f"Average samples/sec: {analysis_results['avg_samples']:.3f}")
#         print(f"Average elapsed time per iteration (ms): {analysis_results['avg_elapsed']:.1f}")
#         print(f"Final lm_loss: {analysis_results['final_lm_loss']:.6f}")
#         print(f"Final moe_loss: {analysis_results['final_moe_loss']:.6f}")

#         # Write all data to the single Excel file
#         output_file = write_to_xlsx(analysis_results, args)
#         print(f"\nResults successfully saved to: {output_file}")

#     else:
#         print("No iteration stats found in log.")

# if __name__ == "__main__":
#     main()
    
    
# # import re
# # import sys

# # def main():
# #     if len(sys.argv) != 2:
# #         print(f"Usage: python {sys.argv[0]} <log_file>")
# #         sys.exit(1)

# #     log_file = sys.argv[1]

# #     tflops_values = []
# #     lm_loss = None
# #     moe_loss = None

# #     # Regex to match iteration lines with TFLOPs, lm loss, and moe loss
# #     # pattern = re.compile(
# #     #     r"iteration\s+\d+/\s+\d+\s+\|.*?"
# #     #     r"elapsed time per iteration \(ms\):\s+([0-9.]+)\s+\|.*?"
# #     #     r"lm loss:\s+([0-9.E+-]+)\s+\|\s+moe loss:\s+([0-9.E+-]+).*?"
# #     #     r"samples per second:\s+([0-9.]+)\s+\|"
# #     #     r"\s+TFLOPs:\s+([0-9.]+)"
# #     # )
# #     pattern = re.compile(
# #     r"iteration\s+\d+/\s+\d+\s+\|.*?"
# #     r"elapsed time per iteration \(ms\):\s+([0-9.]+)\s+\|.*?"
# #     r"lm loss:\s+([0-9.E+-]+)"
# #     r"(?:\s+\|\s+moe loss:\s+([0-9.E+-]+))?"   # <- optional group
# #     r".*?samples per second:\s+([0-9.]+)\s+\|"
# #     r"\s+TFLOPs:\s+([0-9.]+)"
# # )

# #     elapsed_times = []
# #     tflops_values = []
# #     samples_per_sec_values = []
# #     lm_loss = None
# #     moe_loss = None

# #     with open(log_file, "r") as f:
# #         for line in f:
# #             match = pattern.search(line)
# #             if match:
# #                 elapsed = float(match.group(1))
# #                 lm_loss = float(match.group(2))
# #                 moe_loss = float(match.group(3)) if match.group(3) else 0
# #                 samples_per_sec = float(match.group(4))
# #                 tflops = float(match.group(5))

# #                 elapsed_times.append(elapsed)
# #                 samples_per_sec_values.append(samples_per_sec)
# #                 tflops_values.append(tflops)

# #     if tflops_values:
# #         avg_tflops = sum(tflops_values) / len(tflops_values)
# #         avg_samples = sum(samples_per_sec_values) / len(samples_per_sec_values)
# #         avg_elapsed = sum(elapsed_times) / len(elapsed_times)

# #         print(f"Average TFLOPs over {len(tflops_values)} iterations: {avg_tflops:.2f}")
# #         print(f"Average samples/sec: {avg_samples:.3f}")
# #         print(f"Average elapsed time per iteration (ms): {avg_elapsed:.1f}")
# #         print(f"Final lm_loss: {lm_loss:.6f}")
# #         print(f"Final moe_loss: {moe_loss:.6f}")
# #     else:
# #         print("No iteration stats found in log.")


# # if __name__ == "__main__":
# #     main()
# import re
# import sys
# import argparse
# import os
# import pandas as pd
# from datetime import datetime
# import statistics

# def parse_arguments():
#     """Parses command-line arguments."""
#     parser = argparse.ArgumentParser(description="Analyze log files and store results in XLSX.")
#     # --- Existing Arguments ---
#     parser.add_argument("log_file", type=str, help="Path to the log file.")
#     parser.add_argument("--iterations", type=int, required=True, help="Number of iterations.")
#     parser.add_argument("--nodes", type=int, required=True, help="Number of nodes.")
#     parser.add_argument("--gpus_per_node", type=int, required=True, help="Number of GPUs per node.")
#     parser.add_argument("--megatron_pp", type=int, required=True, help="Megatron PP size.")
#     parser.add_argument("--ep", type=int, required=True, help="EP size.")
#     parser.add_argument("--dp", type=int, required=True, help="DP size.")
#     parser.add_argument("--tp", type=int, required=True, help="TP size.")
#     parser.add_argument("--gbs", type=int, required=True, help="Global Batch Size.")
#     parser.add_argument("--mbs", type=int, required=True, help="Micro Batch Size.")
#     parser.add_argument("--slurm_job_id", type=str, required=True, help="SLURM Job ID.")
    
#     # --- New Arguments ---
#     parser.add_argument("--seqlen", type=int, required=True, help="Sequence length.")
#     parser.add_argument("--num_layers", type=int, required=True, help="Number of layers.")
#     parser.add_argument("--num_experts", type=int, required=True, help="Number of experts.")
#     parser.add_argument("--expert_dim", type=int, required=True, help="Expert dimension.")
#     parser.add_argument("--topk", type=int, required=True, help="Top-K for MoE.")
#     parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden dimension.")
#     # --- ADDED MOE_TYPE ARGUMENT ---
#     parser.add_argument("--moe_type", type=str, required=True, help="Type of the MoE implementation (e.g., X-MOE, DS-MOE).")
#     parser.add_argument("--model_size", type=str, default="10B", required=False, help="Size of the model (e.g., 7b, 13b).")
    
#     return parser.parse_args()

# def analyze_log(log_file):
#     """Analyzes the log file to extract performance metrics and all losses."""
#     iteration_pattern = re.compile(
#         r"iteration\s+\d+/\s+\d+\s+\|.*?"
#         r"elapsed time per iteration \(ms\):\s+([0-9.]+)\s+\|.*?"
#         r"lm loss:\s+([0-9.E+-]+)"
#         r"(?:\s+\|\s+moe loss:\s+([0-9.E+-]+))?"
#         r".*?samples per second:\s+([0-9.]+)\s+\|"
#         r"\s+TFLOPs:\s+([0-9.]+)"
#     )

#     tflops_values, samples_per_sec_values, elapsed_times = [], [], []
#     lm_losses, moe_losses = [], []

#     with open(log_file, "r") as f:
#         for line in f:
#             match = iteration_pattern.search(line)
#             if match:
#                 elapsed_times.append(float(match.group(1)))
#                 lm_losses.append(float(match.group(2)))
#                 if match.group(3):
#                     moe_losses.append(float(match.group(3)))
#                 else:
#                     moe_losses.append(0.0)
#                 samples_per_sec_values.append(float(match.group(4)))
#                 tflops_values.append(float(match.group(5)))

#     if not tflops_values:
#         return None

#     matched_iterations = len(tflops_values)
#     avg_tflops = sum(tflops_values) / matched_iterations
#     avg_samples = sum(samples_per_sec_values) / matched_iterations
#     avg_elapsed = sum(elapsed_times) / matched_iterations
    
#     # --- NEW: Calculate TFLOPs standard deviation ---
#     if len(tflops_values) > 1:
#         tflops_std_dev = statistics.stdev(tflops_values)
#     else:
#         tflops_std_dev = 0.0

#     return {
#         "matched_iterations": matched_iterations,
#         "avg_tflops": avg_tflops,
#         "tflops_std_dev": tflops_std_dev,  # <-- ADDED
#         "avg_samples": avg_samples,
#         "avg_elapsed": avg_elapsed,
#         "final_lm_loss": lm_losses[-1],
#         "final_moe_loss": moe_losses[-1],
#         "lm_losses": lm_losses,
#         "moe_losses": moe_losses,
#     }

# def write_to_xlsx(data, args):
#     """Writes the collected data to a single XLSX file with two sheets."""
#     date_str = datetime.now().strftime("%Y-%m-%d")
    
#     filename = (
#         f"{date_str}-{args.seqlen}-{args.num_layers}-{args.num_experts}-"
#         f"{args.expert_dim}-{args.topk}-{args.hidden_dim}-results.xlsx"
#     )
#     file_path = f"/work1/mzhang/henryj/{filename}"

#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
#     # ===================================================================
#     # Prepare Data for Both Sheets
#     # ===================================================================
    
#     run_id = (
#         f"{args.moe_type}-PP{args.megatron_pp}-EP{args.ep}-"
#         f"GBS{args.gbs}-MBS{args.mbs}-SLURM_ID-{args.slurm_job_id}"
#     )

#     # --- MODIFIED: Header updated with "Model Size" column ---
#     header = [
#         "Name", "MOE Type", "Model Size", "AVG TFLOPs", "TFLOPs Std Dev", "Iterations", "Actual Iterations", 
#         "Nodes", "GPUs/node", "Megatron PP", "EP", "DP", "TP", "GBS", "MBS",
#         "Final lm_loss", "SLURM_JOB_ID"
#     ]
    
#     # --- MODIFIED: Main row dictionary includes the new model_size data point ---
#     main_row_dict = {
#         "Name": run_id, 
#         "MOE Type": args.moe_type,
#         "Model Size": args.model_size, # <-- ADDED
#         "AVG TFLOPs": f"{data['avg_tflops']:.2f}",
#         "TFLOPs Std Dev": f"{data['tflops_std_dev']:.2f}",
#         "Iterations": args.iterations, 
#         "Actual Iterations": data['matched_iterations'],
#         "Nodes": args.nodes,
#         "GPUs/node": args.gpus_per_node, 
#         "MBS": args.mbs, 
#         "GBS": args.gbs,
#         "Final lm_loss": f"{data['final_lm_loss']:.6f}", 
#         "Megatron PP": args.megatron_pp,
#         "EP": args.ep, 
#         "DP": args.dp, 
#         "TP": args.tp, 
#         "SLURM_JOB_ID": args.slurm_job_id,
#     }
#     new_main_df = pd.DataFrame([main_row_dict])[header]

#     # --- Sheet 2: Loss per Iteration (Wide Format) ---
#     loss_row_dict = {"id": run_id}
#     for i, loss in enumerate(data['lm_losses']):
#         loss_row_dict[f'loss_{i+1}'] = loss
#     new_loss_df = pd.DataFrame([loss_row_dict])

#     # ===================================================================
#     # Read Existing File (if any) and Append Data
#     # ===================================================================
#     try:
#         # Load existing data from both sheets
#         with pd.ExcelFile(file_path) as xls:
#             existing_main_df = pd.read_excel(xls, 'Main_Results')
#             existing_loss_df = pd.read_excel(xls, 'Loss_per_Iteration')
        
#         # Append new data
#         combined_main_df = pd.concat([existing_main_df, new_main_df], ignore_index=True)
#         combined_loss_df = pd.concat([existing_loss_df, new_loss_df], ignore_index=True)

#     except FileNotFoundError:
#         # If file doesn't exist, the new data is the only data
#         combined_main_df = new_main_df
#         combined_loss_df = new_loss_df

#     # ===================================================================
#     # Write Both DataFrames to the Excel File
#     # ===================================================================
#     with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
#         combined_main_df.to_excel(writer, sheet_name='Main_Results', index=False)
#         combined_loss_df.to_excel(writer, sheet_name='Loss_per_Iteration', index=False)
    
#     return file_path

# def main():
#     """Main function to run the analysis and save results."""
#     args = parse_arguments()
#     analysis_results = analyze_log(args.log_file)

#     if analysis_results:
#         print(f"Actual ran iterations: {analysis_results['matched_iterations']}") # <-- ADDED
#         print(f"Average TFLOPs: {analysis_results['avg_tflops']:.2f}")
#         print(f"TFLOPs Standard Deviation: {analysis_results['tflops_std_dev']:.2f}") # <-- ADDED
#         print(f"Average samples/sec: {analysis_results['avg_samples']:.3f}")
#         print(f"Average elapsed time per iteration (ms): {analysis_results['avg_elapsed']:.1f}")
#         print(f"Final lm_loss: {analysis_results['final_lm_loss']:.6f}")
#         print(f"Final moe_loss: {analysis_results['final_moe_loss']:.6f}")

#         # Write all data to the single Excel file
#         output_file = write_to_xlsx(analysis_results, args)
#         print(f"\nResults successfully saved to: {output_file}")

#     else:
#         print("No iteration stats found in log.")

# if __name__ == "__main__":
#     main()
    
# =======================
# import re
# import sys
# import argparse
# import os
# import pandas as pd
# from datetime import datetime
# import statistics

# def parse_arguments():
#     """Parses command-line arguments."""
#     parser = argparse.ArgumentParser(description="Analyze log files and store results in XLSX.")
#     # --- Existing Arguments ---
#     parser.add_argument("log_file", type=str, help="Path to the log file.")
#     parser.add_argument("--iterations", type=int, required=True, help="Number of iterations.")
#     parser.add_argument("--nodes", type=int, required=True, help="Number of nodes.")
#     parser.add_argument("--gpus_per_node", type=int, required=True, help="Number of GPUs per node.")
#     parser.add_argument("--megatron_pp", type=int, required=True, help="Megatron PP size.")
#     parser.add_argument("--ep", type=int, required=True, help="EP size.")
#     parser.add_argument("--dp", type=int, required=True, help="DP size.")
#     parser.add_argument("--tp", type=int, required=True, help="TP size.")
#     parser.add_argument("--gbs", type=int, required=True, help="Global Batch Size.")
#     parser.add_argument("--mbs", type=int, required=True, help="Micro Batch Size.")
#     parser.add_argument("--slurm_job_id", type=str, required=True, help="SLURM Job ID.")
    
#     # --- New Arguments ---
#     parser.add_argument("--seqlen", type=int, required=True, help="Sequence length.")
#     parser.add_argument("--num_layers", type=int, required=True, help="Number of layers.")
#     parser.add_argument("--num_experts", type=int, required=True, help="Number of experts.")
#     parser.add_argument("--expert_dim", type=int, required=True, help="Expert dimension.")
#     parser.add_argument("--topk", type=int, required=True, help="Top-K for MoE.")
#     parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden dimension.")
#     # --- ADDED MOE_TYPE ARGUMENT ---
#     parser.add_argument("--moe_type", type=str, required=True, help="Type of the MoE implementation (e.g., X-MOE, DS-MOE).")
#     parser.add_argument("--model_size", type=str, default="10B", required=False, help="Size of the model (e.g., 7b, 13b).")
    
#     return parser.parse_args()

# def analyze_log(log_file):
#     """Analyzes the log file to extract performance metrics and all losses."""
    
#     # --- 1. 原有的 iteration pattern ---
#     iteration_pattern = re.compile(
#         r"iteration\s+\d+/\s+\d+\s+\|.*?"
#         r"elapsed time per iteration \(ms\):\s+([0-9.]+)\s+\|.*?"
#         r"lm loss:\s+([0-9.E+-]+)"
#         r"(?:\s+\|\s+moe loss:\s+([0-9.E+-]+))?"
#         r".*?samples per second:\s+([0-9.]+)\s+\|"
#         r"\s+TFLOPs:\s+([0-9.]+)"
#     )

#     # --- 2. 新增的 pattern: Fwd & Experts ---
#     # 匹配: [Rank 0] time (ms) | fwd: 259.10 (... experts: 8.19 ...)
#     fwd_experts_pattern = re.compile(r"\[Rank 0\].*?time \(ms\) \| fwd:\s+([0-9.]+).*?experts:\s+([0-9.]+)")

#     # --- 3. 新增的 pattern: Bwd ---
#     # 匹配: [Rank 0] time (ms) | ... bwd: 798.38
#     bwd_pattern = re.compile(r"\[Rank 0\].*?time \(ms\) \|.*?bwd:\s+([0-9.]+)")

#     # 存储列表
#     tflops_values, samples_per_sec_values, elapsed_times = [], [], []
#     lm_losses, moe_losses = [], []
    
#     # 新增存储列表
#     fwd_times, experts_times, bwd_times = [], [], []

#     with open(log_file, "r", encoding='utf-8', errors='ignore') as f:
#         for line in f:
#             # Check Iteration Logic
#             match = iteration_pattern.search(line)
#             if match:
#                 elapsed_times.append(float(match.group(1)))
#                 lm_losses.append(float(match.group(2)))
#                 if match.group(3):
#                     moe_losses.append(float(match.group(3)))
#                 else:
#                     moe_losses.append(0.0)
#                 samples_per_sec_values.append(float(match.group(4)))
#                 tflops_values.append(float(match.group(5)))

#             # Check Fwd & Experts Logic
#             match_fe = fwd_experts_pattern.search(line)
#             if match_fe:
#                 fwd_times.append(float(match_fe.group(1)))
#                 experts_times.append(float(match_fe.group(2)))

#             # Check Bwd Logic
#             match_bwd = bwd_pattern.search(line)
#             if match_bwd:
#                 bwd_times.append(float(match_bwd.group(1)))

#     if not tflops_values:
#         return None

#     # 计算原有指标平均值
#     matched_iterations = len(tflops_values)
#     avg_tflops = sum(tflops_values[1:]) / (matched_iterations - 1) if matched_iterations > 1 else 0.0
#     avg_samples = sum(samples_per_sec_values) / matched_iterations
#     avg_elapsed = sum(elapsed_times) / matched_iterations
    
#     if len(tflops_values) > 1:
#         tflops_std_dev = statistics.stdev(tflops_values)
#     else:
#         tflops_std_dev = 0.0

#     # --- 新增: 计算 Fwd/Bwd/Experts 平均值 ---
#     avg_fwd = sum(fwd_times) / len(fwd_times) if fwd_times else 0.0
#     avg_bwd = sum(bwd_times) / len(bwd_times) if bwd_times else 0.0
#     avg_experts = sum(experts_times) / len(experts_times) if experts_times else 0.0

#     return {
#         "matched_iterations": matched_iterations,
#         "avg_tflops": avg_tflops,
#         "tflops_std_dev": tflops_std_dev,
#         "avg_samples": avg_samples,
#         "avg_elapsed": avg_elapsed,
#         # --- Added specific times ---
#         "avg_fwd": avg_fwd,
#         "avg_bwd": avg_bwd,
#         "avg_experts": avg_experts,
#         # ----------------------------
#         "final_lm_loss": lm_losses[-1],
#         "final_moe_loss": moe_losses[-1],
#         "lm_losses": lm_losses,
#         "moe_losses": moe_losses,
#     }

# def write_to_xlsx(data, args):
#     """Writes the collected data to a single XLSX file with two sheets."""
#     date_str = datetime.now().strftime("%Y-%m-%d")
    
#     filename = (
#         f"{date_str}-{args.seqlen}-{args.num_layers}-{args.num_experts}-"
#         f"{args.expert_dim}-{args.topk}-{args.hidden_dim}-results.xlsx"
#     )
#     file_path = f"/work1/mzhang/henryj/{filename}"

#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
#     # ===================================================================
#     # Prepare Data for Both Sheets
#     # ===================================================================
    
#     run_id = (
#         f"{args.moe_type}-PP{args.megatron_pp}-EP{args.ep}-"
#         f"GBS{args.gbs}-MBS{args.mbs}-SLURM_ID-{args.slurm_job_id}"
#     )

#     # --- MODIFIED: Header updated with new Time columns ---
#     header = [
#         "Name", "MOE Type", "Model Size", "AVG TFLOPs", "TFLOPs Std Dev", 
#         "AVG Fwd(ms)", "AVG Bwd(ms)", "AVG Experts(ms)", # <-- ADDED HEADERS
#         "Iterations", "Actual Iterations", 
#         "Nodes", "GPUs/node", "Megatron PP", "EP", "DP", "TP", "GBS", "MBS",
#         "Final lm_loss", "SLURM_JOB_ID"
#     ]
    
#     # --- MODIFIED: Main row dictionary includes new metrics ---
#     main_row_dict = {
#         "Name": run_id, 
#         "MOE Type": args.moe_type,
#         "Model Size": args.model_size,
#         "AVG TFLOPs": f"{data['avg_tflops']:.2f}",
#         "TFLOPs Std Dev": f"{data['tflops_std_dev']:.2f}",
#         # --- New Values ---
#         "AVG Fwd(ms)": f"{data['avg_fwd']:.2f}",
#         "AVG Bwd(ms)": f"{data['avg_bwd']:.2f}",
#         "AVG Experts(ms)": f"{data['avg_experts']:.2f}",
#         # ------------------
#         "Iterations": args.iterations, 
#         "Actual Iterations": data['matched_iterations'],
#         "Nodes": args.nodes,
#         "GPUs/node": args.gpus_per_node, 
#         "MBS": args.mbs, 
#         "GBS": args.gbs,
#         "Final lm_loss": f"{data['final_lm_loss']:.6f}", 
#         "Megatron PP": args.megatron_pp,
#         "EP": args.ep, 
#         "DP": args.dp, 
#         "TP": args.tp, 
#         "SLURM_JOB_ID": args.slurm_job_id,
#     }
#     new_main_df = pd.DataFrame([main_row_dict])[header]

#     # --- Sheet 2: Loss per Iteration (Wide Format) ---
#     loss_row_dict = {"id": run_id}
#     for i, loss in enumerate(data['lm_losses']):
#         loss_row_dict[f'loss_{i+1}'] = loss
#     new_loss_df = pd.DataFrame([loss_row_dict])

#     # ===================================================================
#     # Read Existing File (if any) and Append Data
#     # ===================================================================
#     try:
#         # Load existing data from both sheets
#         with pd.ExcelFile(file_path) as xls:
#             existing_main_df = pd.read_excel(xls, 'Main_Results')
#             existing_loss_df = pd.read_excel(xls, 'Loss_per_Iteration')
        
#         # Append new data
#         combined_main_df = pd.concat([existing_main_df, new_main_df], ignore_index=True)
#         combined_loss_df = pd.concat([existing_loss_df, new_loss_df], ignore_index=True)

#     except FileNotFoundError:
#         # If file doesn't exist, the new data is the only data
#         combined_main_df = new_main_df
#         combined_loss_df = new_loss_df

#     # ===================================================================
#     # Write Both DataFrames to the Excel File
#     # ===================================================================
#     with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
#         combined_main_df.to_excel(writer, sheet_name='Main_Results', index=False)
#         combined_loss_df.to_excel(writer, sheet_name='Loss_per_Iteration', index=False)
    
#     return file_path

# def main():
#     """Main function to run the analysis and save results."""
#     args = parse_arguments()
#     analysis_results = analyze_log(args.log_file)

#     if analysis_results:
#         print(f"Actual ran iterations: {analysis_results['matched_iterations']}")
#         print(f"Average TFLOPs: {analysis_results['avg_tflops']:.2f}")
#         print(f"TFLOPs Standard Deviation: {analysis_results['tflops_std_dev']:.2f}")
#         print(f"Average samples/sec: {analysis_results['avg_samples']:.3f}")
#         print(f"Average elapsed time per iteration (ms): {analysis_results['avg_elapsed']:.1f}")
        
#         # --- Print new metrics ---
#         print(f"Average Fwd time (ms): {analysis_results['avg_fwd']:.2f}")
#         print(f"Average Bwd time (ms): {analysis_results['avg_bwd']:.2f}")
#         print(f"Average Experts time (ms): {analysis_results['avg_experts']:.2f}")
#         # -------------------------
        
#         print(f"Final lm_loss: {analysis_results['final_lm_loss']:.6f}")
#         print(f"Final moe_loss: {analysis_results['final_moe_loss']:.6f}")

#         # Write all data to the single Excel file
#         output_file = write_to_xlsx(analysis_results, args)
#         print(f"\nResults successfully saved to: {output_file}")

#     else:
#         print("No iteration stats found in log.")

# if __name__ == "__main__":
#     main()
#===========================

# =======================
# =======================
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
    parser.add_argument("--moe_type", type=str, required=True, help="Type of the MoE implementation.")
    parser.add_argument("--model_size", type=str, default="10B", required=False, help="Size of the model.")
    
    return parser.parse_args()

def analyze_log(log_file):
    """Analyzes the log file to extract performance metrics and all detailed losses/timings."""
    
    # --- 1. Iteration Pattern (Unchanged) ---
    iteration_pattern = re.compile(
        r"iteration\s+\d+/\s+\d+\s+\|.*?"
        r"elapsed time per iteration \(ms\):\s+([0-9.]+)\s+\|.*?"
        r"lm loss:\s+([0-9.E+-]+)"
        r"(?:\s+\|\s+moe loss:\s+([0-9.E+-]+))?"
        r".*?samples per second:\s+([0-9.]+)\s+\|"
        r"\s+TFLOPs:\s+([0-9.]+)"
    )

    # --- 2. UPDATED: Detailed Forward Pattern ---
    fwd_detailed_pattern = re.compile(
        r"\[Rank 0\].*?time \(ms\) \| fwd:\s+([0-9.]+)\s+\("
        r"fwd_moe:\s+([0-9.]+),\s+"
        r"1st_a2a:\s+([0-9.]+),\s+"
        r"experts:\s+([0-9.]+),\s+"
        r"2nd_a2a:\s+([0-9.]+),\s+"
        r"top_k:\s+([0-9.]+)"
    )

    # --- 3. UPDATED: Detailed Backward Pattern ---
    bwd_detailed_pattern = re.compile(
        r"\[Rank 0\].*?time \(ms\) \|.*?"
        r"bwd:\s+([0-9.]+)\s+\|\s+"
        r"bwd_inner:\s+([0-9.]+)\s+\|\s+"
        r"bwd_allreduce:\s+([0-9.]+)\s+\|\s+"
        r"step:\s+([0-9.]+)"
    )

    # --- 4. NEW: Memory Pattern ---
    # Matches: peak_reserved=21.5GB
    mem_pattern = re.compile(r"peak_reserved=([0-9.]+)GB")

    # Storage Lists
    tflops_values, samples_per_sec_values, elapsed_times = [], [], []
    lm_losses, moe_losses = [], []
    
    # Track Max Memory
    max_peak_reserved = 0.0
    
    # Detailed storage
    metrics = {
        "fwd": [], "fwd_moe": [], "first_a2a": [], "experts": [], "second_a2a": [], "top_k": [],
        "bwd": [], "bwd_inner": [], "bwd_allreduce": [], "step": []
    }

    with open(log_file, "r", encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Check Iteration Logic
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

            # Check Detailed Forward Logic
            match_fe = fwd_detailed_pattern.search(line)
            if match_fe:
                metrics["fwd"].append(float(match_fe.group(1)))
                metrics["fwd_moe"].append(float(match_fe.group(2)))
                metrics["first_a2a"].append(float(match_fe.group(3)))
                metrics["experts"].append(float(match_fe.group(4)))
                metrics["second_a2a"].append(float(match_fe.group(5)))
                metrics["top_k"].append(float(match_fe.group(6)))

            # Check Detailed Backward Logic
            match_bwd = bwd_detailed_pattern.search(line)
            if match_bwd:
                metrics["bwd"].append(float(match_bwd.group(1)))
                metrics["bwd_inner"].append(float(match_bwd.group(2)))
                metrics["bwd_allreduce"].append(float(match_bwd.group(3)))
                metrics["step"].append(float(match_bwd.group(4)))
                
            # Check Memory Logic (New)
            match_mem = mem_pattern.search(line)
            if match_mem:
                current_peak = float(match_mem.group(1))
                if current_peak > max_peak_reserved:
                    max_peak_reserved = current_peak

    if not tflops_values:
        return None

    # Helper for calculating averages
    def get_avg(key):
        vals = metrics[key]
        return sum(vals) / len(vals) if vals else 0.0

    # Basic stats
    matched_iterations = len(tflops_values)
    tflops_values = tflops_values[1:]  # Exclude first iteration for avg
    matched_iterations -= 1  # Adjust count accordingly
    elapsed_times = elapsed_times[1:]  # Exclude first iteration for avg
    avg_tflops = sum(tflops_values) / (matched_iterations) if matched_iterations > 1 else 0.0
    avg_samples = sum(samples_per_sec_values) / matched_iterations
    avg_elapsed = sum(elapsed_times) / matched_iterations
    tflops_std_dev = statistics.stdev(tflops_values) if len(tflops_values) > 1 else 0.0

    return {
        "matched_iterations": matched_iterations,
        "avg_tflops": avg_tflops,
        "tflops_std_dev": tflops_std_dev,
        "avg_samples": avg_samples,
        "avg_elapsed": avg_elapsed,
        # --- Detailed Averages ---
        "avg_fwd": get_avg("fwd"),
        "avg_fwd_moe": get_avg("fwd_moe"),
        "avg_1st_a2a": get_avg("first_a2a"),
        "avg_experts": get_avg("experts"),
        "avg_2nd_a2a": get_avg("second_a2a"),
        "avg_topk": get_avg("top_k"),
        "avg_bwd": get_avg("bwd"),
        "avg_bwd_inner": get_avg("bwd_inner"),
        "avg_bwd_allreduce": get_avg("bwd_allreduce"),
        "avg_step": get_avg("step"),
        # -------------------------
        "final_lm_loss": lm_losses[-1],
        "final_moe_loss": moe_losses[-1],
        "lm_losses": lm_losses,
        "max_peak_reserved": max_peak_reserved, # New field
    }

def write_to_xlsx(data, args):
    """Writes the collected data to a single XLSX file with two sheets."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    filename = (
        f"{date_str}-{args.model_size}B-{args.seqlen}-{args.num_layers}-{args.num_experts}-"
        f"{args.expert_dim}-{args.topk}-{args.hidden_dim}-results.xlsx"
    )
    file_path = f"/work1/mzhang/henryj/{filename}"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    run_id = (
        f"{args.moe_type}-PP{args.megatron_pp}-EP{args.ep}-"
        f"GBS{args.gbs}-MBS{args.mbs}-SLURM_ID-{args.slurm_job_id}"
    )

    # --- Header with detailed breakdown ---
    header = [
        "Name", "MOE Type", "Model Size", "AVG TFLOPs", "TFLOPs Std Dev", 
        "Max Peak Reserved (GB)", # New Header
        # FWD Breakdown
        "AVG Fwd", "AVG Fwd MoE", "AVG Experts", "AVG 1st A2A", "AVG 2nd A2A", "AVG TopK",
        # BWD Breakdown
        "AVG Bwd", "AVG Bwd Inner", "AVG Bwd AllReduce", "AVG Step",
        # General
        "Iterations", "Actual Iterations", 
        "Nodes", "GPUs/node", "Megatron PP", "EP", "DP", "TP", "GBS", "MBS",
        "Final lm_loss", "SLURM_JOB_ID"
    ]
    
    main_row_dict = {
        "Name": run_id, 
        "MOE Type": args.moe_type,
        "Model Size": args.model_size,
        "AVG TFLOPs": f"{data['avg_tflops']:.2f}",
        "TFLOPs Std Dev": f"{data['tflops_std_dev']:.2f}",
        "Max Peak Reserved (GB)": f"{data['max_peak_reserved']:.2f}", # New Data
        # --- FWD Metrics ---
        "AVG Fwd": f"{data['avg_fwd']:.2f}",
        "AVG Fwd MoE": f"{data['avg_fwd_moe']:.2f}",
        "AVG Experts": f"{data['avg_experts']:.2f}",
        "AVG 1st A2A": f"{data['avg_1st_a2a']:.2f}",
        "AVG 2nd A2A": f"{data['avg_2nd_a2a']:.2f}",
        "AVG TopK": f"{data['avg_topk']:.2f}",
        # --- BWD Metrics ---
        "AVG Bwd": f"{data['avg_bwd']:.2f}",
        "AVG Bwd Inner": f"{data['avg_bwd_inner']:.2f}",
        "AVG Bwd AllReduce": f"{data['avg_bwd_allreduce']:.2f}",
        "AVG Step": f"{data['avg_step']:.2f}",
        # -------------------
        "Iterations": args.iterations, 
        "Actual Iterations": data['matched_iterations'],
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

    # --- Sheet 2: Loss per Iteration ---
    loss_row_dict = {"id": run_id}
    for i, loss in enumerate(data['lm_losses']):
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
    """Main function."""
    args = parse_arguments()
    results = analyze_log(args.log_file)

    if results:
        print(f"Iterations: {results['matched_iterations']}")
        print(f"Avg TFLOPs: {results['avg_tflops']:.2f}")
        print(f"Max Peak Reserved: {results['max_peak_reserved']:.2f} GB") # Console output
        
        print("\n--- Forward Pass Details (ms) ---")
        print(f"Total Fwd: {results['avg_fwd']:.2f}")
        print(f"  > Fwd MoE: {results['avg_fwd_moe']:.2f}")
        print(f"  > Experts: {results['avg_experts']:.2f}")
        print(f"  > 1st A2A: {results['avg_1st_a2a']:.2f}")
        print(f"  > 2nd A2A: {results['avg_2nd_a2a']:.2f}")
        print(f"  > Top K:   {results['avg_topk']:.2f}")
        
        print("\n--- Backward Pass Details (ms) ---")
        print(f"Total Bwd: {results['avg_bwd']:.2f}")
        print(f"  > Inner:   {results['avg_bwd_inner']:.2f}")
        print(f"  > AllRed:  {results['avg_bwd_allreduce']:.2f}")
        print(f"Optimizer Step: {results['avg_step']:.2f}")
        
        output_file = write_to_xlsx(results, args)
        print(f"\nSaved to: {output_file}")

    else:
        print("No iteration stats found.")

if __name__ == "__main__":
    main()
# ========

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