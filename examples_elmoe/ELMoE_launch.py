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
import json
import time
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
    # --- Planner memory budget (all values are BINARY GiB, matching the planner's
    # --- MemoryPredictor.GIGA = 1024**3).
    #
    # Primary knob: how much to RESERVE on each GPU. The analytic memory model only
    # accounts for tensors; the reserve covers what it does not model — the HIP/CUDA
    # context, RCCL communication buffers, hipBLASLt/Triton/aiter scratch, and the
    # caching allocator's block rounding (reserved >= allocated).
    #     limit = detect_total_memory_gib() - planner_memory_headroom_gb
    #
    # Escape hatch: --planner-memory-limit-gb pins an ABSOLUTE limit and wins if set.
    # Use it to reproduce a known budget, on heterogeneous clusters (where per-rank
    # detection could diverge), or if detection fails (then it falls back to 61.5 GiB,
    # the value that used to be hardcoded here).
    #
    # MEASURED: MI250X / MI210 report total_memory = 68702699520 B = 63.984375 GiB
    # (NOT a round 64). With the 2.5 GiB default that yields a limit of 61.484 GiB,
    # i.e. 20 MiB tighter than the 61.5 that was previously hardcoded.
    parser.add_argument('--planner-memory-headroom-gb', type=float, default=2.5)
    parser.add_argument('--planner-memory-limit-gb', type=float, default=None)
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

    rank = _get_global_rank()
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

    memory_limit_gb = _resolve_memory_limit_gib(planner_args, rank)

    planner_cfg = PlannerConfig(
        model_config=model_config,
        target_gbs=gbs,
        memory_limit_gb=memory_limit_gb,
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
    rank = _get_global_rank()
    
    sorted_plans = sorted(result.stage_plans, key=lambda x: x.stage_id)
    partition_str = " ".join(str(p.num_layers) for p in sorted_plans)
    ckpt_str = " ".join(str(p.num_checkpoints) for p in sorted_plans)
    
    # Set env vars read by Megatron's model builder
    os.environ['UNEVEN_PP_PARTITION'] = partition_str
    os.environ['DYNAMIC_CHECKPOINT_PARTITION'] = ckpt_str
    os.environ['UNEVEN_PP'] = 'True'
    os.environ['DYNAMIC_CHECKPOINT'] = 'True'
    
    print (f'[ELMoE_launch.py] {partition_str=}')
    print (f'[ELMoE_launch.py] {ckpt_str=}')
    
    # Override --micro-batch-size and --global-batch-size in argv
    remaining_argv = _override_arg(remaining_argv, '--micro-batch-size', 
                                    str(result.micro_batch_size))
    remaining_argv = _override_arg(remaining_argv, '--global-batch-size',
                                    str(result.effective_global_batch_size))

    # Keep the DeepSpeed config JSON's batch sizes consistent with the planner's
    # mbs/gbs so the engine and Megatron's data loader agree (see function docstring).
    _sync_ds_config_with_planner(remaining_argv, result)

    sorted_plans = sorted(result.stage_plans, key=lambda x: x.stage_id)
    partition_list = [str(p.num_layers) for p in sorted_plans]
    ckpt_list = [str(p.num_checkpoints) for p in sorted_plans]
    
    # Pass as CLI args (nargs='+' expects separate entries)
    remaining_argv.extend(['--uneven-pp-partition'] + partition_list)
    remaining_argv.extend(['--dynamic-checkpoint-partition'] + ckpt_list) 
    print (f'[ELMoE_launch.py] {partition_list=}')
    print (f'[ELMoE_launch.py] {ckpt_list=}')
    
    if rank == 0:
        print("=" * 70)
        print("PLANNER RESULT")
        print("=" * 70)
        print(f"  Micro-Batch Size:  {result.micro_batch_size}")
        print(f"  Num Batches:       {result.num_batches}")
        print(f"  Global Batch Size: {result.effective_global_batch_size}")
        print(f"  Layer Partition:   [{partition_str}]")
        print(f"  Ckpt Partition:    [{ckpt_str}]")
        mem_str = " ".join(f"{p.memory_used_gb:.1f}" for p in sorted_plans)
        print(f"  Stage Memory (GB): [{mem_str}]")
        print(f"  Peak Memory (GB):  {max(p.memory_used_gb for p in sorted_plans):.2f}")
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


_FALLBACK_MEMORY_LIMIT_GIB = 61.5   # the value previously hardcoded here (MI250X: 64 - 2.5)


def _resolve_memory_limit_gib(planner_args, rank):
    """Planner memory limit in BINARY GiB, resolved in priority order:

        1. --planner-memory-limit-gb   (absolute override; wins if given)
        2. detected total GPU memory - --planner-memory-headroom-gb
        3. _FALLBACK_MEMORY_LIMIT_GIB  (if detection fails)

    The planner runs on EVERY rank, so this must return the SAME value on all of
    them. Path 1 is a literal, and path 2 reads *total* memory (a static hardware
    constant) — both are rank-invariant on homogeneous nodes. Never derive this
    from free/available memory, which differs per rank and would make ranks compute
    different partitions.
    """
    if planner_args.planner_memory_limit_gb is not None:
        limit = float(planner_args.planner_memory_limit_gb)
        if rank == 0:
            print(f"[planner] memory_limit={limit:.2f} GiB (explicit --planner-memory-limit-gb)")
        return limit

    # Defensive: a failure to import/detect must never break a run that works today —
    # it degrades to the fallback limit instead of raising.
    try:
        from deepspeed.moe.planner.memory_calculator import detect_total_memory_gib
        total = detect_total_memory_gib()
    except Exception as e:
        if rank == 0:
            print(f"[planner] WARNING: memory detection unavailable "
                  f"({type(e).__name__}: {e})")
        total = None
    headroom = float(planner_args.planner_memory_headroom_gb)

    if total is None:
        if rank == 0:
            print(f"[planner] memory_limit={_FALLBACK_MEMORY_LIMIT_GIB:.2f} GiB "
                  f"(GPU detection failed; using fallback). "
                  f"Pass --planner-memory-limit-gb to set it explicitly.")
        return _FALLBACK_MEMORY_LIMIT_GIB

    limit = total - headroom
    if limit <= 0:
        raise ValueError(
            f"planner memory headroom ({headroom} GiB) >= detected GPU memory "
            f"({total:.2f} GiB); nothing left to plan with.")
    if rank == 0:
        print(f"[planner] device total={total:.2f} GiB  headroom={headroom:.2f} GiB "
              f"-> memory_limit={limit:.2f} GiB")
    return limit


def _get_global_rank():
    """Global rank, launcher-agnostic (torchrun / SLURM / MPI), not SLURM-only."""
    for var in ('RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'PMIX_RANK'):
        val = os.environ.get(var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return 0


def _shared_job_id():
    """An id shared by all ranks of this job, used to namespace the barrier marker.
    Returns None when no launcher-provided shared id exists (then we skip the
    explicit marker barrier and rely on atomic rename + the later dist-init
    collective for cross-rank visibility)."""
    for var in ('SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID', 'TORCHELASTIC_RUN_ID', 'JOB_ID'):
        val = os.environ.get(var)
        if val:
            return val
    return None


def _find_arg_value(argv, flag):
    """Return the token following `flag` in argv, or None if absent/danging."""
    try:
        idx = argv.index(flag)
        return argv[idx + 1]
    except (ValueError, IndexError):
        return None


def _patch_ds_config_file(config_path, micro_batch_size, global_batch_size):
    """Rewrite the DeepSpeed config JSON so its batch sizes match the planner's
    choice. Atomic (tmp + os.replace). Writer-only. Returns the patched dict."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    cfg['train_batch_size'] = int(global_batch_size)
    cfg['train_micro_batch_size_per_gpu'] = int(micro_batch_size)
    # Let DeepSpeed re-derive gas = train_batch_size / (mbs * dp); drop any stale value.
    cfg.pop('gradient_accumulation_steps', None)
    tmp_path = os.path.join(
        os.path.dirname(os.path.abspath(config_path)),
        f".{os.path.basename(config_path)}.tmp.{os.getpid()}",
    )
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, config_path)  # atomic on the same filesystem
    return cfg


def _sync_ds_config_with_planner(remaining_argv, result, _wait_timeout_s=120.0):
    """Make the DeepSpeed config's batch sizes consistent with the planner.

    Megatron's data loader honors the planner's --micro-batch-size, but DeepSpeed
    reads the static JSON written by the SLURM template (train_micro_batch_size_per_gpu
    baked at template time, typically 1). When the planner picks mbs>=2 the two
    disagree: the engine pulls gbs/dp micro-batches each holding mbs samples, draining
    the data index ~mbs x too fast -> StopIteration mid-run, and the reported TFLOPs
    are credited against the nominal gbs -> ~halved. We rewrite the JSON so the engine
    and the data loader agree (single writer + atomic rename; an optional filesystem
    barrier gates the readers when a shared job id is available)."""
    rank = _get_global_rank()
    config_path = _find_arg_value(remaining_argv, '--deepspeed_config')
    if config_path is None or not os.path.exists(config_path):
        if rank == 0:
            print(f"[ELMoE_launch.py] WARNING: --deepspeed_config not usable "
                  f"({config_path!r}); skipping ds-config mbs sync.", flush=True)
        return

    job_id = _shared_job_id()
    marker = f"{config_path}.ready.{job_id}" if job_id else None

    if rank == 0:
        cfg = _patch_ds_config_file(config_path, result.micro_batch_size,
                                    result.effective_global_batch_size)
        if marker is not None:
            with open(marker, 'w', encoding='utf-8') as f:
                f.write("ready\n")
        print(f"[ELMoE_launch.py] ds-config synced: "
              f"train_micro_batch_size_per_gpu={cfg['train_micro_batch_size_per_gpu']} "
              f"train_batch_size={cfg['train_batch_size']} "
              f"(gradient_accumulation_steps re-derived by DeepSpeed)", flush=True)
    elif marker is not None:
        # Barrier: block until rank 0 has atomically replaced the file. The marker is
        # namespaced by job id, so a stale marker from a previous job can't be matched.
        waited = 0.0
        while not os.path.exists(marker):
            time.sleep(0.05)
            waited += 0.05
            if waited >= _wait_timeout_s:
                raise RuntimeError(
                    f"[ELMoE_launch.py] rank {rank}: timed out after "
                    f"{_wait_timeout_s}s waiting for ds-config marker {marker}")
    # If no shared job id, the atomic rename plus the dist-init collective that runs
    # before deepspeed.initialize() reads the file provides cross-rank visibility.


# =========================================================
# 3. Main Entry Point
# =========================================================
def main():
    planner_args, remaining_argv = split_args()
    
    rank = _get_global_rank()

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