import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal, Dict
import copy
import argparse
import sys
import json
import os

# =========================================================
# PART 1: The Provided Memory Predictor
# (Assuming this is imported as before)
# =========================================================
from memory_calculator import MemoryPredictor 

# =========================================================
# PART 2: Profile Loader & Logic
# =========================================================

class ProfileLoader:
    def __init__(self, profile_dir: str, model_config: Dict, total_gpus: int, num_nodes: int):
        self.profile_dir = profile_dir
        self.cfg = model_config
        self.total_gpus = total_gpus
        self.num_nodes = num_nodes
        self._cache = {}

    def _get_filename(self, mbs: int) -> str:
        # N{nodes}_n{gpus}_d{dim}_e{exp}_f{ffn}_k{topk}_s{seq}_b{mbs}_profile.json
        return (
            f"N{self.num_nodes}_n{self.total_gpus}_"
            f"d{self.cfg['d_model']}_e{self.cfg['num_experts']}_"
            f"f{self.cfg['expert_dim']}_k{self.cfg['topk']}_"
            f"s{self.cfg['seqlen']}_b{mbs}_profile.json"
        )

    def get_layer_times(self, mbs: int) -> Tuple[float, float]:
        """
        Returns (T_std, T_ckpt) in milliseconds.
        Strategy: 
        1. Look for exact MBS profile file.
        2. If not found, look for MBS=1 profile and scale linearly.
        """
        
        # 1. Try exact lookup
        fwd_gemm, fwd_comm = self._load_raw_values(mbs)
        
        if fwd_gemm is None:
            # 2. Fallback: Load MBS=1 and scale
            base_gemm, base_comm = self._load_raw_values(1)
            if base_gemm is None:
                raise FileNotFoundError(
                    f"Could not find profile for MBS={mbs} OR MBS=1 in {self.profile_dir}. "
                    f"Expected filename like: {self._get_filename(1)}"
                )
            # Linear scaling assumption
            fwd_gemm = base_gemm * mbs
            fwd_comm = base_comm * mbs

        # 3. Apply Theoretical Formulas
        # T_std  = 3 * Fwd_Gemm + 2 * Fwd_Comm (Fwd + Bwd)
        # T_ckpt = 4 * Fwd_Gemm + 3 * Fwd_Comm (Fwd + Bwd + Recompute)
        
        t_std = (3 * fwd_gemm) + (2 * fwd_comm)
        t_ckpt = (4 * fwd_gemm) + (3 * fwd_comm)
        
        return t_std, t_ckpt

    def _load_raw_values(self, mbs: int) -> Tuple[Optional[float], Optional[float]]:
        if mbs in self._cache:
            return self._cache[mbs]
            
        filename = self._get_filename(mbs)
        filepath = os.path.join(self.profile_dir, filename)
        
        if not os.path.exists(filepath):
            return None, None
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            gemm = data.get("fwd_gemm_time_per_layer", 0.0)
            comm = data.get("fwd_comm_time_per_layer", 0.0)
            
            self._cache[mbs] = (gemm, comm)
            return gemm, comm
        except Exception as e:
            print(f"Warning: Error reading {filepath}: {e}")
            return None, None

# =========================================================
# PART 3: The Planner Infrastructure
# =========================================================

@dataclass
class PlannerConfig:
    model_config: Dict 
    fixed_num_batches: int  
    memory_limit_gb: float
    
    # New connection to the loader
    profile_loader: ProfileLoader

@dataclass
class StagePlan:
    stage_id: int
    num_layers: int
    num_checkpoints: int
    time_ms: float
    memory_used_gb: float

@dataclass
class PartitionResult:
    micro_batch_size: int
    effective_global_batch_size: int
    num_batches: int
    bottleneck_time_ms: float
    total_step_time_ms: float
    throughput_tokens_per_sec: float
    bubble_overhead: float
    stage_plans: List[StagePlan]

# =========================================================
# PART 4: The Adapter (Modified for Loader)
# =========================================================

class PhysicalMemoryAdapter:
    def __init__(self, config: PlannerConfig):
        self.cfg = config
        self.predictor = MemoryPredictor(config.model_config)
        self.loader = config.profile_loader
        
    def get_stage_performance(self, stage_idx: int, num_layers: int, u_batch_size: int, force_strategy=None) -> Tuple[float, int, float]:
        if num_layers == 0: return 0.0, 0, 0.0
        
        # [CHANGED] Fetch times from ProfileLoader (JSON based)
        t_std, t_ckpt = self.loader.get_layer_times(u_batch_size)
        
        if force_strategy == "all": range_to_search = [num_layers]
        elif force_strategy == "none": range_to_search = [0]
        else: range_to_search = range(0, num_layers + 1)
            
        best_c, valid_mem = -1, float('inf')
        for c in range_to_search:
            mem_gb = self.predictor.query_stage_memory(stage_idx, num_layers, c, u_batch_size)
            if mem_gb <= self.cfg.memory_limit_gb:
                best_c, valid_mem = c, mem_gb
                break 
        
        if best_c == -1: return float('inf'), 0, float('inf')
        
        time_cost = (num_layers - best_c) * t_std + best_c * t_ckpt
        return time_cost, best_c, valid_mem

    def query_specific_config(self, stage_idx: int, num_layers: int, num_ckpt: int, u_batch_size: int):
        t_std, t_ckpt = self.loader.get_layer_times(u_batch_size)
        mem_gb = self.predictor.query_stage_memory(stage_idx, num_layers, num_ckpt, u_batch_size)
        time_cost = (num_layers - num_ckpt) * t_std + num_ckpt * t_ckpt
        return time_cost, mem_gb

# =========================================================
# PART 5: The Optimizer (Unchanged Logic)
# =========================================================

class DPPlanner:
    def __init__(self, config: PlannerConfig):
        self.cfg = config
        self.adapter = PhysicalMemoryAdapter(config)
        
    def solve_for_micro_batch(self, u_batch_size: int) -> Tuple[float, List[StagePlan]]:
        N, S = self.cfg.model_config['num_layers'], self.cfg.model_config['pp_stages']
        dp = [[float('inf')] * (N + 1) for _ in range(S + 1)]
        parent = [[-1] * (N + 1) for _ in range(S + 1)]
        dp[0][0] = 0
        
        for k in range(1, S + 1):
            stage_idx = k - 1
            for i in range(1, N + 1):
                for j in range(i + 1):
                    if dp[k-1][j] == float('inf'): continue
                    cost, _, _ = self.adapter.get_stage_performance(stage_idx, i - j, u_batch_size)
                    if cost == float('inf'): continue
                    bottleneck = max(dp[k-1][j], cost)
                    if bottleneck < dp[k][i]:
                        dp[k][i] = bottleneck
                        parent[k][i] = j
                        
        if dp[S][N] == float('inf'): return float('inf'), []
        plans = []
        curr_n = N
        for k in range(S, 0, -1):
            split = parent[k][curr_n]
            l = curr_n - split
            t, c, m = self.adapter.get_stage_performance(k-1, l, u_batch_size)
            plans.append(StagePlan(k-1, l, c, t, m))
            curr_n = split
        plans.reverse()
        return dp[S][N], plans

class PipelineOptimizer:
    def __init__(self, config: PlannerConfig):
        self.cfg = config
        self.planner = DPPlanner(config)
        self.adapter = self.planner.adapter 
        
    def find_optimal_config(self, candidate_micro_batches: List[int]):
        best_result = None
        max_throughput = 0.0
        
        for ub in candidate_micro_batches:
            if ub <= 0: continue
            
            try:
                bottleneck_ms, plans = self.planner.solve_for_micro_batch(ub)
            except FileNotFoundError as e:
                print(f"Skipping MBS={ub}: {e}")
                continue

            if bottleneck_ms == float('inf'): 
                continue 
            
            num_batches = self.cfg.fixed_num_batches
            dp_size = self.cfg.model_config['dp']
            batch_unit = ub * dp_size
            effective_gbs = batch_unit * num_batches
            
            pp_stages = self.cfg.model_config['pp_stages']
            total_time_ms = (num_batches + pp_stages - 1) * bottleneck_ms
            
            throughput = (effective_gbs * self.cfg.model_config['seqlen']) / (total_time_ms / 1000.0)
            ideal_time = num_batches * bottleneck_ms
            bubble = (total_time_ms - ideal_time) / total_time_ms if total_time_ms > 0 else 0.0
            
            if throughput > max_throughput:
                max_throughput = throughput
                best_result = PartitionResult(ub, effective_gbs, num_batches, bottleneck_ms, total_time_ms, throughput, bubble, plans)
        
        return best_result
    
    def evaluate_baseline(self, ub: int, strategy: Literal["all", "none"]):
        N = self.cfg.model_config['num_layers']
        S = self.cfg.model_config['pp_stages']
        dp_size = self.cfg.model_config['dp']
        num_batches = self.cfg.fixed_num_batches
        effective_gbs = (ub * dp_size) * num_batches
        
        base = N // S
        rem = N % S
        plans = []
        max_time = 0
        
        try:
            for s in range(S):
                count = base + (1 if s < rem else 0)
                t, c, m = self.adapter.get_stage_performance(s, count, ub, force_strategy=strategy)
                if t == float('inf'): return None
                max_time = max(max_time, t)
                plans.append(StagePlan(s, count, c, t, m))
        except FileNotFoundError:
            return None
            
        total_time = (num_batches + S - 1) * max_time
        throughput = (effective_gbs * self.cfg.model_config['seqlen']) / (total_time / 1000.0)
        return PartitionResult(ub, effective_gbs, num_batches, max_time, total_time, throughput, 0.0, plans)

    def print_detailed_report(self, result: PartitionResult, title: str):
        model_size_b = self.planner.adapter.predictor.calculate_model_size_in_billions()
        print("\n" + "="*80)
        print(f"|{title}")
        print("="*80)
        print(f"|Model Size:  {model_size_b:.2f} Billion Parameters")
        print(f"|Micro-Batch: {result.micro_batch_size}")
        print(f"|Num Batches: {result.num_batches} (Fixed)")
        print(f"|Global Batch:{result.effective_global_batch_size} (Calculated)")
        print(f"|Throughput:  {result.throughput_tokens_per_sec:.2f} tokens/s")
        print(f"|Step Time:   {result.total_step_time_ms:.2f} ms")
        print(f"|Bubble:      {result.bubble_overhead*100:.2f}%")
        print("-" * 80)
        print(f"|{'Stage':<6} | {'Layers':<7} | {'Ckpt':<5} | {'Mem (GB)':<10} | {'Time (ms)':<10} | {'Status'}")
        print("-" * 80)
        for p in result.stage_plans:
            status = "OK" if p.memory_used_gb <= self.cfg.memory_limit_gb else "OOM (!)"
            print(f"|{p.stage_id:<6} | {p.num_layers:<7} | {p.num_checkpoints:<5} | {p.memory_used_gb:<10.2f} | {p.time_ms:<10.2f} | {status}")
        print("-" * 80)

    def print_bash_export(self, result: PartitionResult):
        sorted_plans = sorted(result.stage_plans, key=lambda x: x.stage_id)
        layers = [str(p.num_layers) for p in sorted_plans]
        ckpts = [str(p.num_checkpoints) for p in sorted_plans]
        print("\n# --- planner.py: EXPORT FOR BASH ---")
        print(f"OPTIMAL_MBS={result.micro_batch_size}")
        print(f"OPTIMAL_PARTITION=\"{' '.join(layers)}\"")
        print(f"OPTIMAL_CKPT=\"{' '.join(ckpts)}\"")
    
    def compare_baselines(self, best_result: PartitionResult):
        print("\n\n|>>> COMPARING WITH BASELINES (Same MBS)")
        for name, strat in [("No Checkpointing (Speed)", "none"), ("Full Checkpointing (Memory)", "all")]:
            base_res = self.evaluate_baseline(best_result.micro_batch_size, strat)
            if base_res:
                diff = base_res.throughput_tokens_per_sec - best_result.throughput_tokens_per_sec
                pct = (diff / best_result.throughput_tokens_per_sec) * 100
                title = f"BASELINE: {name} | Diff: {pct:.2f}% Tput"
                self.print_detailed_report(base_res, title)
            else:
                print("\n" + "="*80)
                print(f"|BASELINE: {name}")
                print("="*80)
                print("|FAILED (OOM - Out of Memory)")
                
    def evaluate_throughput_for_mbs(self, ub: int):
        """
        Calculates the theoretical throughput for a given micro-batch size (ub)
        using a balanced layer distribution and optimal checkpointing for that distribution.
        """
        N = self.cfg.model_config['num_layers']
        S = self.cfg.model_config['pp_stages']
        dp_size = self.cfg.model_config['dp']
        num_batches = self.cfg.fixed_num_batches
        effective_gbs = (ub * dp_size) * num_batches

        # Use a simple balanced layer distribution
        base_layers = N // S
        remainder = N % S
        plans = []
        max_stage_time = 0.0

        try:
            for stage_idx in range(S):
                num_layers_in_stage = base_layers + (1 if stage_idx < remainder else 0)
                if num_layers_in_stage == 0:
                    plans.append(StagePlan(stage_idx, 0, 0, 0.0, 0.0))
                    continue

                # Find the best performance for this stage config (optimal selective checkpoints)
                # by calling get_stage_performance without a 'force_strategy'
                time_ms, ckpts, mem_gb = self.adapter.get_stage_performance(stage_idx, num_layers_in_stage, ub)

                if time_ms == float('inf'):
                    # This configuration is not feasible due to memory limits
                    print(f"Warning: MBS={ub} is infeasible (OOM) for a balanced partition.")
                    return None

                max_stage_time = max(max_stage_time, time_ms)
                plans.append(StagePlan(stage_idx, num_layers_in_stage, ckpts, time_ms, mem_gb))

        except FileNotFoundError as e:
            print(f"Skipping MBS={ub}: {e}")
            return None

        if max_stage_time == 0.0:
            return None

        # Calculate final pipeline metrics based on the bottleneck stage
        total_time_ms = (num_batches + S - 1) * max_stage_time
        throughput = (effective_gbs * self.cfg.model_config['seqlen']) / (total_time_ms / 1000.0)
        ideal_time = num_batches * max_stage_time
        bubble = (total_time_ms - ideal_time) / total_time_ms if total_time_ms > 0 else 0.0

        return PartitionResult(
            micro_batch_size=ub,
            effective_global_batch_size=effective_gbs,
            num_batches=num_batches,
            bottleneck_time_ms=max_stage_time,
            total_step_time_ms=total_time_ms,
            throughput_tokens_per_sec=throughput,
            bubble_overhead=bubble,
            stage_plans=plans
        )
        
    def evaluate_fixed_strategy(self, ub: int, strategy: Literal["all", "none"]):
        """
        Calculates throughput for a given micro-batch size (ub) using a fixed
        checkpointing strategy (all layers or no layers) and a balanced layer distribution.
        
        Returns a PartitionResult or None if the configuration results in an OOM.
        """
        N = self.cfg.model_config['num_layers']
        S = self.cfg.model_config['pp_stages']
        dp_size = self.cfg.model_config['dp']
        num_batches = self.cfg.fixed_num_batches
        effective_gbs = (ub * dp_size) * num_batches
        
        # Determine the balanced layer distribution
        base_layers = N // S
        remainder = N % S
        plans = []
        max_stage_time = 0.0
        
        try:
            for stage_idx in range(S):
                num_layers_in_stage = base_layers + (1 if stage_idx < remainder else 0)
                
                if num_layers_in_stage == 0:
                    plans.append(StagePlan(stage_idx, 0, 0, 0.0, 0.0))
                    continue

                # Force the number of checkpoints based on the strategy
                num_ckpt = num_layers_in_stage if strategy == "all" else 0
                
                # Use the adapter to query the exact performance of this specific configuration
                time_ms, mem_gb = self.adapter.query_specific_config(
                    stage_idx, num_layers_in_stage, num_ckpt, ub
                )

                # Explicitly check for OOM condition
                if mem_gb > self.cfg.memory_limit_gb:
                    print(f"OOM Detected: For MBS={ub}, strategy='{strategy}', stage {stage_idx} requires {mem_gb:.2f} GB "
                          f"(Limit: {self.cfg.memory_limit_gb:.2f} GB).")
                    return None # Signal failure
                
                max_stage_time = max(max_stage_time, time_ms)
                plans.append(StagePlan(stage_idx, num_layers_in_stage, num_ckpt, time_ms, mem_gb))

        except FileNotFoundError as e:
            print(f"Skipping MBS={ub}: {e}")
            return None
        
        if max_stage_time == 0.0: return None

        # Calculate final pipeline metrics based on the bottleneck stage
        total_time_ms = (num_batches + S - 1) * max_stage_time
        throughput = (effective_gbs * self.cfg.model_config['seqlen']) / (total_time_ms / 1000.0)
        ideal_time = num_batches * max_stage_time
        bubble = (total_time_ms - ideal_time) / total_time_ms if total_time_ms > 0 else 0.0
            
        return PartitionResult(ub, effective_gbs, num_batches, max_stage_time, total_time_ms, throughput, bubble, plans)

# =========================================================
# Execution
# =========================================================

def get_args():
    parser = argparse.ArgumentParser(description="LLM MoE Training Configuration")

    # --- Model Architecture ---
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--d_model', type=int, required=True, help='Hidden dimension size')
    model_group.add_argument('--seqlen', type=int, required=True, help='Sequence length')
    model_group.add_argument('--num_layers', type=int, required=True, help='Total number of layers')
    model_group.add_argument('--vocab_size', type=int, default=50257)
    model_group.add_argument('--attention_heads', type=int, default=16)
    model_group.add_argument('--tied_embedding', action='store_true')

    # --- Parallelism ---
    parallel_group = parser.add_argument_group('Parallelism')
    parallel_group.add_argument('--pp_stages', type=int, required=True)
    parallel_group.add_argument('--dp', type=int, required=True)
    parallel_group.add_argument('--ep', type=int, required=True)
    
    # [NEW] Hardware topology info for file matching
    parallel_group.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parallel_group.add_argument('--gpus_per_node', type=int, default=8, help='GPUs per node')

    # --- Mixture of Experts (MoE) ---
    moe_group = parser.add_argument_group('Mixture of Experts')
    moe_group.add_argument('--num_experts', type=int, required=True)
    moe_group.add_argument('--expert_dim', type=int, required=True)
    moe_group.add_argument('--topk', type=int, required=True)
    moe_group.add_argument('--moe_train_capacity_factor', type=float, default=1.25)

    # --- Training / Optimization ---
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--activation_checkpointing', type=str, default='selective',  choices=['none', 'full', 'selective'])
    train_group.add_argument('--mbs', type=int, required=True) # Input MBS (for filename check or initial guess)
    
    # [NEW] Profile Directory
    train_group.add_argument('--profile_dir', type=str, default="./planner_profiling_cache", help='Directory containing JSON profiles')

    # --- Planner Limits ---
    planner_group = parser.add_argument_group('Planner Limits')
    planner_group.add_argument('--fixed_num_batches', type=int, required=True, help='Fixed number of micro-batches (GAS)')
    planner_group.add_argument('--memory_limit_gb', type=float, default=65.5)
    
    # --- [NEW] Execution Mode ---
    exec_group = parser.add_argument_group('Execution Mode')
    exec_group.add_argument('--eval-mbs-list', type=int, nargs='+', default=None, 
                              help='A list of micro-batch sizes to evaluate instead of running the full optimizer.')
    exec_group.add_argument('--eval-strategy', type=str, choices=['none', 'all'], default=None,
                              help='Required if --eval-mbs-list is set. Forces a fixed checkpointing strategy.')
    
    return parser.parse_args()


if __name__ == "__main__":
    
    args = get_args()
    
    total_gpus = args.num_nodes * args.gpus_per_node
    gbs = args.mbs * args.dp * args.fixed_num_batches # Just for config ref, not used in math

    model_config = {
        'd_model': args.d_model, 
        'seqlen': args.seqlen, 
        'pp_stages': args.pp_stages, 
        'num_layers': args.num_layers,
        'dp': args.dp,
        'ep': args.ep,
        'num_experts': args.num_experts, 
        'expert_dim': args.expert_dim, 
        'topk': args.topk, 
        'vocab_size': args.vocab_size,
        'attention_heads': args.attention_heads,  
        'tied_embedding': args.tied_embedding, 
        'moe-train-capacity-factor': args.moe_train_capacity_factor,
        'activation_checkpointing': args.activation_checkpointing, 
        'mbs': args.mbs, 
        'gbs': gbs, 
    }
    
    # 2. Initialize Profile Loader
    profile_loader = ProfileLoader(
        profile_dir=args.profile_dir,
        model_config=model_config,
        total_gpus=total_gpus,
        num_nodes=args.num_nodes
    )
    
    # 3. Planner Config
    planner_cfg = PlannerConfig(
        model_config=model_config,
        fixed_num_batches=args.fixed_num_batches, 
        memory_limit_gb=args.memory_limit_gb,
        profile_loader=profile_loader # Inject Loader
    )
    
    optimizer = PipelineOptimizer(planner_cfg)


    if args.eval_mbs_list is None:
            
        # =====================================================
        # SCENARIO A: Run the Automatic Optimizer
        # =====================================================
        
        print("|>>> SCENARIO A: AUTOMATIC OPTIMIZATION")
        
        # We scan these MBS values. If JSON exists, we use it. If not, we scale from b1.
        best = optimizer.find_optimal_config([1, 2, 3, 4, 5, 6, 7, 8])
        
        if best:
            optimizer.print_detailed_report(best, "OPTIMAL PLAN FOUND")
            optimizer.compare_baselines(best)
            optimizer.print_bash_export(best)
        else:
            print("OPTIMAL_MBS=FAILED")
            
    else:
        if not args.eval_strategy:
            print("Error: --eval-strategy must be set to 'none' or 'all' when using --eval-mbs-list.")
            exit(1)
            
        strategy_name = "No Checkpointing" if args.eval_strategy == 'none' else "Full Checkpointing"
        print(f"|>>> SCENARIO B: EVALUATING MBS VALUES: {args.eval_mbs_list}")
        print(f"|>>> Using fixed strategy: {strategy_name}")

        for mbs in sorted(args.eval_mbs_list):
            result = optimizer.evaluate_fixed_strategy(mbs, args.eval_strategy)
            
            if result:
                title = f"STRATEGY: '{args.eval_strategy.upper()}' | MBS={mbs}"
                optimizer.print_detailed_report(result, title)
            else:
                print("\n" + "="*80)
                print(f"| FAILED: MBS={mbs} with strategy '{args.eval_strategy}' is INFEASIBLE (Out of Memory).")
                print("=" * 80)
        
        
        
        
    # print("\n\n" + "#"*80)
    # print("# FORCED BASELINE EVALUATION FOR SPECIFIC MICRO-BATCH SIZES")
    # print("#"*80)

    # # Define the list of micro-batch sizes you want to check
    # forced_mbs_list = [1, 2, 4, 6, 8]

    # for mbs in forced_mbs_list:
    #     print(f"\n\n|>>> EVALUATING BASELINES FOR A FORCED MBS = {mbs}")
        
    #     # --- Baseline 1: No Checkpointing (Speed-focused) ---
    #     # This uses a balanced partition (e.g., 8,8,8,8) with 0 checkpoints
    #     base_res_none = optimizer.evaluate_baseline(mbs, "none")
    #     if base_res_none:
    #         title = f"BASELINE: No Checkpointing (Speed) | FORCED MBS={mbs}"
    #         optimizer.print_detailed_report(base_res_none, title)
    #     else:
    #         print("\n" + "="*80)
    #         print(f"| BASELINE: No Checkpointing (Speed) | FORCED MBS={mbs}")
    #         print("="*80)
    #         print("| FAILED (Likely Out of Memory or Profile Not Found)")
    #         print("="*80)

    #     # --- Baseline 2: Full Checkpointing (Memory-focused) ---
    #     # This uses a balanced partition with every layer checkpointed
    #     base_res_all = optimizer.evaluate_baseline(mbs, "all")
    #     if base_res_all:
    #         title = f"BASELINE: Full Checkpointing (Memory) | FORCED MBS={mbs}"
    #         optimizer.print_detailed_report(base_res_all, title)
    #     else:
    #         print("\n" + "="*80)
    #         print(f"| BASELINE: Full Checkpointing (Memory) | FORCED MBS={mbs}")
    #         print("="*80)
    #         print("| FAILED (Likely Out of Memory or Profile Not Found)")
    #         print("="*80)