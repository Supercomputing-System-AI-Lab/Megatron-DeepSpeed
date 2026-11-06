# ft-sys-v2 recovery (milestone 1: warm recovery at the step boundary)

When the heartbeat system writes a failure list that replaces a node, the training processes stop, a standby node takes the lost
node's ranks, every rank rebuilds the same way, the state comes back from memory (the survivors' own, the lost ranks' from their
partners' copies), and training redoes the interrupted step. Off unless `XMOE_FT_V2=1` and `FTV2_RECOVERY=1`; with them off,
nothing here is imported and training is unchanged.

```
 during training   KEEP     each owner sends its final gradient of each optimizer group to its partner (rank + 8: the next node),
                            inside the inner optimizer's step; the partner updates its copy (the shadow) with CPU Adam
 a failure list    STOP     monitor thread: RCCL revoke on every communicator, etcd barrier, abort; the main thread leaves the step
                   DECIDE   every survivor reports how many updates each group and each shadow has; all agree -> resume step s
                   REBUILD  every rank, the same calls: new world (store in etcd) -> Megatron's parallel state -> the model,
                            optimizer and DeepSpeed engine (the trainer's own setup function) -> state restored -> bf16 weights
                            from the fp32 masters (ZeRO's all-gather) -> counters, scheduler, RNG -> data iterators at the samples
                   RE-SEED  every owner seeds its partner's shadow again (new pairs after the join)
                   REDO     Megatron's train() is entered again at iteration s
```

## Files

| File | What it is | Framework |
|---|---|---|
| `rc_controller.py` | the protocol: stop callback, decide, the symmetric rebuild, the standby's wait and join, the times of each recovery | none |
| `rc_keep.py` | Keep: the partner's shadow, the per-step exchange, CPU Adam (DeepSpeed's kernel in chunks, or torch fused AdamW), the GPU check mode, versions, records | none (an adapter) |
| `rc_stop.py` | revoke, etcd barrier, abort, the keeper communicator (port of the toy's recovery_core.py) | torch.distributed + RCCL |
| `rc_world.py` | the new world: TCPStore hosted by the lowest survivor, its address in etcd | torch.distributed |
| `adapters/deepspeed_zero.py` | the optimizer adapter (DeepSpeed ZeRO stage 1 or 2) | DeepSpeed |
| `adapters/megatron_train.py` | the trainer adapter (Megatron-DeepSpeed's training.py) | Megatron |
| `launch/ftv2_sbatch.sh` | the 0623 launcher's `AUTORUN_SBATCH_CMD`: +1 node, `--no-kill` | Slurm |
| `launch/ftv2_srun.sh` | sourced by `AUTORUN_PRE_CMD`: one Slurm step per arm (`FTV2_ARMS`) holding the training tasks and, after them, the standby's tasks | Slurm |
| `tests/cpu_keep_test.py`, `tests/cpu_world_test.py`, `tests/cpu_adam_bench.py` | login-node checks (gloo) | |
| `rc_report.py` | the summary of a job's recoveries and keep records | |

## Where each hook sits, and why there

| Hook | Place | Why this place |
|---|---|---|
| Keep | the INNER optimizer's `step` (ZeRO's `.optimizer`, apex FusedAdam), wrapped on the object | the first place the gradient is final (reduced, unscaled, clipped) and the last place before the state changes; every sharded optimizer (ZeRO, Megatron's distributed optimizer, FSDP) has an inner step over its own partition, so it ports |
| Stop | the heartbeat's counted `torch.distributed` wrapper (`hb_monitor._count`) and PyTorch's process-group registry | framework-free: every framework's communication goes through `torch.distributed` |
| Rebuild | `torch.distributed.init_process_group`, then the trainer's own init functions again (`initialize_model_parallel`, `setup_model_and_optimizer`, `deepspeed.initialize`) | no framework internal is re-pointed; the registries that the init functions fill once are reset first (DeepSpeed's group dicts, the RNG trackers) |
| Restore | ZeRO's partition tensors, the inner optimizer's `load_state_dict`, ZeRO's `update_lp_params` | the framework's own state interfaces: in-memory checkpoint semantics |
| Position | `pretrain()` runs `train()` inside the recovery loop; `train()` starts at `args.iteration` | one small hook in the trainer; no line of `train()` changes |

## The adapter interface (a port to another framework writes these two files only)

Optimizer adapter (`adapters/deepspeed_zero.py`, class `ZeroAdapter(engine)`):
`slots()` the inner optimizer's tensors `[(group, tensor)]`; `hyper(i)` lr, weight decay, betas, eps, bias correction, AdamW or L2,
step count; `moments(t)`; `wrap_inner_step(before, after)`; `take_state()`; `put_state(states)`; `refresh_weights()`;
`set_counters(steps, samples, micro)`; `release()`; `reset_registries()`; `gpu_shadow_optimizer(params)`; `set_step(opt, group, p, n)`.

Trainer adapter (`adapters/megatron_train.py`, class `MegatronTrainer(model_provider, model_type, data_provider, post)`):
`counters()`, `increments()`, `set_counters()`, `set_scheduler()`, `reinit_parallel()`, `build_model()`, `build_data()`,
`reset_timers()`, `rng_snapshot()`, `rng_restore()`, `standby_args()`, `standby_set_rank()`.

## A new recovery plan (the extension point)

The coordinator chooses the plan per failure kind (`hb_coordinator._plan`, `_standby`); today the layout it writes is "the lost
ranks, and the standby node that takes them" (fields `ranks`, `standby`, `replace` of the failure list). `rc_controller._join` takes
the layout's world size and this process's rank in it and calls the same build. A plan with another world size (the OOM ladder:
fewer or more GPUs, another TP or EP) adds (1) its layout fields in the list, (2) the rank map in `_join`, and (3) a restore that
re-splits the state instead of copying it (`put_state` from gathered full tensors, as the toy's `re_split_state_after_shrink`).

## Launch facts the code depends on (found on the GPUs, 2026-11-06)

- **One Slurm step.** Frontier's Slurm gives each step its own Slingshot VNI (`SwitchParameters ... job_vni=user`), so processes
  of two steps cannot reach each other over the network: a standby started as its own step failed the new world's first
  collective (P1c 5861771). The standby's tasks are tasks `FTV2_WORLD ..` of the training's own step; every process splits MPI's
  `COMM_WORLD` by role at start (`rc_controller.split_mpi_world`), because Megatron and DeepSpeed read their rank from it.
- **No MPI after a failure.** MPI's world then holds dead ranks: the rebuild reads ranks from the environment
  (`args.using_mpi = False`), and every rank of a one-step run leaves with `os._exit` after training (MPI's finalize would wait for
  the dead ranks and for the standby's tasks). An exit record is written first, or the node's agent reads a kill.
- **A non-zero exit ends the step.** In one step, a task that exits with a non-zero code makes Slurm end every task (SIGTERM);
  a rank killed by a signal does not. So the recovery's own failure exits (74-77) end the arm at once, which is what they mean.
- **No pinned allocation between batched sends** (P1b 5861716): Keep allocates every buffer before its first exchange.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `FTV2_RECOVERY` | 0 | 1 turns the recovery on (with `XMOE_FT_V2=1`) |
| `FTV2_SHADOW` | cpu | `cpu` pinned host shadow + CPU Adam; `gpu` the owner's own optimizer class on a GPU shadow (bit-identical check mode) |
| `FTV2_CPU_ADAM` | ds | `ds` DeepSpeed's CPU Adam kernel in chunks; `torch` torch.optim.AdamW(fused=True) on the CPU |
| `FTV2_CPU_ADAM_CHUNK` | 8M | values per DeepSpeed CPU Adam call (it holds the Python lock while it runs) |
| `FTV2_SHADOW_CHECK` | 0 | 1: every step the owner also sends its fp32 master; the partner records the largest difference to its shadow |
| `FTV2_PARTNER_STRIDE` | 8 | partner = rank + stride (mod world) |
| `FTV2_NULL_RECOVERY_STEP` | unset | test: at this iteration rank 0 writes a failure list that loses no rank |
| `FTV2_WORLD`, `FTV2_ROLE`, `FTV2_RUN`, `FTV2_ARM` | set by `launch/ftv2_srun.sh` | training world size, train or standby, run id, arm name |
| `FTV2_WORLD_TIMEOUT_S` | 600 | the new world's store and process-group timeout |

Exit codes of a recovery that cannot proceed: 74 a lost rank and the rank keeping its copy both lost; 75 a failure inside an
optimizer step (the versions disagree); 76 the list replaces no node; 77 the recovery itself failed. ft-sys-v1's restart takes over
from these (not wired in milestone 1).

## What milestone 1 supports, and what another model needs

Supported: one DeepSpeed engine without a pipeline (PP 1), ZeRO stage 1 (or 2), any EP inside a node, data parallelism over nodes,
TP 1; a standby node per run; a lost node whose partner node survives. The partner of rank r is r + FTV2_PARTNER_STRIDE (default
8, one node): Keep needs the owner's and the partner's partitions to have the same sizes, which holds when r and r + 8 have the
same position in their data-parallel replica (PP 1, TP 1, EP 8). Checked at every seed (a size mismatch stops the run with a message).

| example (model registry) | layout | runs with ft-sys-v2 recovery today? |
|---|---|---|
| X-MoE 10B (`10B`) | EP 8, PP 1, DP over nodes | yes (tested, this folder's experiments) |
| DeepSeek-MoE-16B (`DeepSeek_16B`) at EP 8, PP 1, 2+ nodes | EP 8, PP 1 | expected yes (same path; not run) |
| DeepSeek-V3 2 layers (`DeepSeek_V3_671B_L2`) at PP 1 | EP 8 or 16, PP 1 | expected yes at EP 8 (EP 16 spans two nodes: the partner stride must be 16) |
| X-MoE 63B (PP 4), DeepSeek-V3 8 layers and full (PP > 1) | pipeline | no: milestone 3 (a pipeline engine: its own schedule state and stage groups) |

To run a model: copy `assistant_deep_dive/2026-11-05-ftv2_heartbeat/runs/rc_common.sh`, set `AUTORUN_EP_KEY` / `AUTORUN_EP_CMD`
for the model (and `AUTORUN_PP_*` stays off), keep `XMOE_FT_V2=1 FTV2_RECOVERY=1 FTV2_STANDBY=1`, the two launch hooks and
`TORCH_NCCL_ASYNC_ERROR_HANDLING=0`, and give `FTV2_ARMS` one arm (`main:` and the settings of the table above).

## Run

See `assistant_deep_dive/2026-11-05-ftv2_heartbeat/runs/rc_common.sh` and `runs/1106_*.sh`: the 0623 launcher with
`AUTORUN_SBATCH_CMD=launch/ftv2_sbatch.sh` and `AUTORUN_PRE_CMD` sourcing `heartbeat-sys/hb_start.sh` (with `FTV2_STANDBY=1`) and
`launch/ftv2_srun.sh`; `TORCH_NCCL_ASYNC_ERROR_HANDLING=0` (the survivors must live through a peer's death).
