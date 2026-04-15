#!/usr/bin/env python
"""
X-MoE Launch Script
===================
Thin wrapper that optionally runs the pipeline planner before calling
Megatron-DeepSpeed's pretrain(). Replaces the bash-based planner logic
that was previously in the SLURM template.

Usage (from srun):
    python launch.py --run-planner true --planner-mode optimize \
        --planner-profile-dir ./planner_profiling_cache \
        [all normal megatron/deepspeed args...]
"""
import sys
import os
import argparse


# =========================================================
# 1. Argument Parsing
# =========================================================
def split_args():
    """Only consume planner-exclusive flags. Everything else passes through untouched."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--run-planner', type=str, default='false',
                        choices=['false', 'true', 'true-membal'])
    parser.add_argument('--planner-mode', type=str, default='optimize',
                        choices=['optimize', 'optimize-membal'])
    parser.add_argument('--planner-profile-dir', type=str,
                        default='./planner_profiling_cache')
    parser.add_argument('--planner-memory-limit-gb', type=float, default=62.0)
    parser.add_argument('--planner-overhead-ms', type=float, default=55.0)
    parser.add_argument('--planner-optimizer-ratio', type=float, default=0.10)
    parser.add_argument('--planner-max-mbs', type=int, default=8)

    planner_args, remaining_argv = parser.parse_known_args()
    return planner_args, remaining_argv


# =========================================================
# 2. Planner Execution
# =========================================================

def run_planner(remaining_argv, planner_args):
    """
    Parse model config from remaining_argv (without consuming them),
    run the planner, return the result.
    """
    from deepspeed.moe.planner import (
        PipelineOptimizer, PlannerConfig, ProfileLoader
    )

    # Second parse — reads shared args but remaining_argv is not modified
    # (we don't use the leftover from this parse)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--num-layers', type=int, default=None)
    parser.add_argument('--hidden-size', type=int, default=None)
    parser.add_argument('--ffn-hidden-size', type=int, default=None)
    parser.add_argument('--seq-length', type=int, default=None)
    parser.add_argument('--num-attention-heads', type=int, default=16)
    parser.add_argument('--num-experts', type=int, nargs='+', default=[1])
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--micro-batch-size', type=int, default=1)
    parser.add_argument('--global-batch-size', type=int, default=None)
    parser.add_argument('--pipeline-model-parallel-size', type=int, default=1)
    parser.add_argument('--tensor-model-parallel-size', type=int, default=1)
    parser.add_argument('--moe-expert-parallel-size', type=int, default=1)
    parser.add_argument('--checkpoint-activations', action='store_true', default=False)
    parser.add_argument('--checkpoint-num-layers', type=int, default=1)
    model_args, _ = parser.parse_known_args(remaining_argv)

    rank = int(os.environ.get('SLURM_PROCID', 0))
    total_gpus = int(os.environ.get('SLURM_NTASKS', 8))
    num_nodes = int(os.environ.get('SLURM_NNODES', 1))

    pp = model_args.pipeline_model_parallel_size
    tp = model_args.tensor_model_parallel_size
    ep = model_args.moe_expert_parallel_size
    dp = total_gpus // (pp * tp)
    num_experts = model_args.num_experts[0] if isinstance(model_args.num_experts, list) else model_args.num_experts

    mbs = model_args.micro_batch_size
    gbs = model_args.global_batch_size
    if gbs is None:
        raise ValueError("--global-batch-size must be set when using the planner")

    model_config = {
        'd_model': model_args.hidden_size,
        'seqlen': model_args.seq_length,
        'pp_stages': pp,
        'num_layers': model_args.num_layers,
        'dp': dp,
        'ep': ep,
        'num_experts': num_experts,
        'expert_dim': model_args.ffn_hidden_size,
        'topk': model_args.topk,
        'use_groupgemm': 0,
        'vocab_size': 50257,
        'attention_heads': model_args.num_attention_heads,
        'tied_embedding': False,
        'moe-train-capacity-factor': 1.25,
        'activation_checkpointing': 'full' if model_args.checkpoint_activations else 'none',
        'mbs': mbs,
        'gbs': gbs,
    }
    
    profile_total_gpu=8
    profile_num_nodes=1

    profile_loader = ProfileLoader(
        profile_dir=planner_args.planner_profile_dir,
        model_config=model_config,
        total_gpus=profile_total_gpu,
        num_nodes=profile_num_nodes
    )

    planner_cfg = PlannerConfig(
        model_config=model_config,
        target_gbs=gbs,
        memory_limit_gb=planner_args.planner_memory_limit_gb,
        profile_loader=profile_loader,
        mb_framework_overhead_ms=planner_args.planner_overhead_ms,
        optimizer_step_ratio=planner_args.planner_optimizer_ratio,
    )

    optimizer = PipelineOptimizer(planner_cfg)
    candidates = list(range(1, planner_args.planner_max_mbs + 1))

    if planner_args.planner_mode == 'optimize':
        result = optimizer.find_optimal_config(candidates)
    elif planner_args.planner_mode == 'optimize-membal':
        full_ac = model_args.checkpoint_activations
        result = optimizer.find_max_mbs_membal_config(candidates, full_ac=full_ac)
    else:
        result = None

    if result is None:
        if rank == 0:
            print("ERROR: Planner failed to find a valid configuration.", flush=True)
        sys.exit(1)

    return result

def apply_planner_result(result, remaining_argv):
    """
    Inject planner outputs into env vars and Megatron's argv.
    Called by every rank with the same deterministic result.
    """
    rank = int(os.environ.get('SLURM_PROCID', 0))
    
    sorted_plans = sorted(result.stage_plans, key=lambda x: x.stage_id)
    partition_str = " ".join(str(p.num_layers) for p in sorted_plans)
    ckpt_str = " ".join(str(p.num_checkpoints) for p in sorted_plans)
    
    # Set env vars read by Megatron's model builder
    os.environ['UNEVEN_PP_PARTITION'] = partition_str
    os.environ['DYNAMIC_CHECKPOINT_PARTITION'] = ckpt_str
    os.environ['UNEVEN_PP'] = 'True'
    os.environ['DYNAMIC_CHECKPOINT'] = 'True'
    
    # Override --micro-batch-size and --global-batch-size in argv
    remaining_argv = _override_arg(remaining_argv, '--micro-batch-size', 
                                    str(result.micro_batch_size))
    remaining_argv = _override_arg(remaining_argv, '--global-batch-size', 
                                    str(result.effective_global_batch_size))
    
    if rank == 0:
        print("=" * 70)
        print("PLANNER RESULT")
        print("=" * 70)
        print(f"  Micro-Batch Size:  {result.micro_batch_size}")
        print(f"  Num Batches:       {result.num_batches}")
        print(f"  Global Batch Size: {result.effective_global_batch_size}")
        print(f"  Layer Partition:   [{partition_str}]")
        print(f"  Ckpt Partition:    [{ckpt_str}]")
        print(f"  Throughput Est:    {result.throughput_tokens_per_sec:.0f} tokens/s")
        print("=" * 70, flush=True)
    
    return remaining_argv


def _override_arg(argv, flag, value):
    """Replace --flag X in argv list, or append if not present."""
    try:
        idx = argv.index(flag)
        argv[idx + 1] = value
    except ValueError:
        argv.extend([flag, value])
    return argv


# =========================================================
# 3. Main Entry Point
# =========================================================
def main():
    planner_args, remaining_argv = split_args()
    
    rank = int(os.environ.get('SLURM_PROCID', 0))

    if planner_args.run_planner != 'false':
        if planner_args.run_planner == 'true-membal':
            planner_args.planner_mode = 'optimize-membal'

        if rank == 0:
            print(f"[ELMoE_launch.py] Running planner (mode={planner_args.planner_mode})...")

        result = run_planner(remaining_argv, planner_args)
        remaining_argv = apply_planner_result(result, remaining_argv)

        if rank == 0:
            print(f"[ELMoE_launch.py] Planner done. Handing off to pretrain_gpt_deepspeed.py")
    else:
        if rank == 0:
            print("[ELMoE_launch.py] Planner disabled. Passing through to pretrain_gpt_deepspeed.py")

    sys.argv = [sys.argv[0]] + remaining_argv
    
    if rank == 0:
        # Debug: confirm micro-batch-size is in argv
        print(f"[ELMoE_launch.py] sys.argv has {len(sys.argv)} entries")
        mbs_check = '--micro-batch-size' in sys.argv
        print(f"[ELMoE_launch.py] --micro-batch-size in argv: {mbs_check}")

    runpy.run_module('pretrain_gpt_deepspeed', run_name='__main__')

if __name__ == '__main__':
    import runpy 
    main()