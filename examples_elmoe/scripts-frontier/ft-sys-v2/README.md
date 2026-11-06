# ft-sys-v2: failure detection and warm recovery for X-MoE (Megatron-DeepSpeed) on Frontier

A training job with ft-sys-v2 on finds a failure within seconds (the heartbeat), agrees on one failure list, and, when a standby
node is free and the model has no pipeline, recovers inside the same processes: the standby takes the lost node's ranks, every
rank rebuilds, the optimizer state comes back from memory, and training redoes the interrupted step. Nothing is read from disk.
Everything is off unless `XMOE_FT_V2=1` (heartbeat) and `FTV2_RECOVERY=1` (recovery); off, training runs exactly as before.

```
 a node fails        heartbeat-sys/   monitor thread in every rank -> node agent -> etcd -> coordinator writes failed/<g>
                                       (kinds: killed, hang, node lost, hardware fault; out of memory, exception, stuck -> no replacement)
 the list arrives    recovery/        stop (RCCL revoke, etcd barrier, abort) -> decide the step -> new world with the standby ->
                                       Megatron + DeepSpeed init again -> restore (own state / the partner's shadow) -> redo the step
 during training     recovery/        Keep: each rank's optimizer state is copied on the next node and updated there every step
 before training     gate/            ft-sys-v1's pre-run node checks on the new stack (optional, FTV2_GATE)
```

## Folders

| Folder | What it does | Read first |
|---|---|---|
| `heartbeat-sys/` | the detection: `hb_monitor.py` (in every training process: the call counter, the beat, the failure list's callback, test injections), `hb_agent.py` (one per node), `hb_coordinator.py` (the decisions, the failure list with the standby to use), `hb_etcd.py`, `hb_rules.py`, `hb_start.sh` (starts etcd and the agents from the batch script) | `heartbeat-sys/README.md` |
| `recovery/` | the warm recovery (milestone 1): `rc_controller.py` (the protocol), `rc_keep.py` (the partner's shadow), `rc_stop.py`, `rc_world.py`, `adapters/` (DeepSpeed ZeRO, Megatron's trainer), `launch/` (the sbatch and srun hooks of the 0623 launcher), `tests/` (login-node checks), `rc_report.py` (the tables of a run) | `recovery/README.md` |
| `gate/` | ft-sys-v1's pre-run gate on the new stack | `gate/` files' headers |
| `runs/` | one run file per example model with ft-sys-v2 on, and `ftv2_run_common.sh` | below |

Hooks outside this folder: `megatron/training.py` only (search `[ft-sys-v2`): the counter before the first communication call,
the monitor's start, and, with `FTV2_RECOVERY=1`, three lines in `pretrain()` (the standby's start without a world, Keep after
the first build, `train()` run inside the recovery's loop). DeepSpeed is not modified: the inner optimizer's `step` is wrapped on
the object at run time (`recovery/adapters/deepspeed_zero.py`).

## Run an example

```
cd examples_elmoe/scripts-frontier/ft-sys-v2/runs
RENDER_ONLY=1 bash ftv2_10B.sh      # render the job script (and the standby's +1-node copy), submit nothing
bash ftv2_10B.sh                    # submit (debug QOS, 30 min)
```

| Run file | Model and layout | ft-sys-v2 mode | Tested |
|---|---|---|---|
| `ftv2_10B.sh` | X-MoE 10B, 3 nodes, EP 8, PP 1, + 1 standby | heartbeat + warm recovery | yes (2026-11-06: RESULTS_M1_M2.md of the heartbeat scope) |
| `ftv2_DeepSeek_16B.sh` | DeepSeek-MoE-16B, 3 nodes, EP 8, PP 1, + 1 standby | heartbeat + warm recovery | render only |
| `ftv2_DeepSeek_V3_671B_L2.sh` | DeepSeek-V3 2 layers, 3 nodes, EP 8, PP 1, + 1 standby | heartbeat + warm recovery | render only |
| `ftv2_DeepSeek_V3_671B_L8D1.sh` | DeepSeek-V3 8 layers, 16 nodes, EP 32, PP 1, partner stride 32, + 1 standby | heartbeat + warm recovery | render only |
| `ftv2_63B.sh` | X-MoE 63B, 4 nodes, PP 4, EP 8 | heartbeat + v1 restart only (milestone 3 needed) | render only |
| `ftv2_DeepSeek_V3_671B.sh` | DeepSeek-V3 671B, 64 nodes, PP 32, EP 8 | heartbeat + v1 restart only (milestone 3 needed) | render only |

`ftv2_run_common.sh` holds the settings every run needs: the new stack's env, megatron from this tree, etcd, the launcher's hooks
(`AUTORUN_SBATCH_CMD=recovery/launch/ftv2_sbatch.sh` adds the standby node and `--no-kill`; `AUTORUN_PRE_CMD` starts the heartbeat
and defines the `srun` function of `recovery/launch/ftv2_srun.sh`), `TORCH_NCCL_ASYNC_ERROR_HANDLING=0` (the survivors must live
through a peer's death). Test arms and injections: `FTV2_ARMS` and `FTV2_INJECT` (`recovery/README.md`, `heartbeat-sys/hb_monitor.py`).
"Heartbeat + v1 restart only": the failure list is written with its kind and node, and the job ends `FTV2_ABORT_ON_FAILURE`
seconds later; resubmit with the resume riders (`AUTORUN_LOAD_DIR`/`AUTORUN_SAVE_DIR` with `AUTORUN_NATIVE_RESUME=1` or
`AUTORUN_UCP_RESUME=1`) to restart from the last save.

## What the recovery covers today (milestone 1)

- Failure kinds that replace a node: a rank killed (also a GPU fault that ends the process by a signal), a hang (counts +
  stacks), a node lost (lease + the others stopped), a hardware fault raised as an exception whose text names it (ECC, GPU
  lost, ...). One failure per standby node.
- Kinds that stop the run (every training rank exits with 76, the standby leaves): an exception in the program, out of memory
  (the OOM ladder is not built yet), stuck. A GPU memory access fault that the runtime reports to Python (AcceleratorError,
  "illegal memory access") is an exception by `heartbeat-sys/hb_rules.py`: most often a kernel bug, which a new node would hit
  again. The recovery's loop never holds an exception back: only a stop already under way or a communication error waits for
  the failure list (`recovery/rc_controller.py`, `waits_for_stop`).
- Layout: one DeepSpeed engine, PP 1, ZeRO stage 1 or 2, EP inside a node or spanning nodes, data parallelism over nodes, TP 1.
- The resume point is the step boundary: every rank redoes the interrupted step. A failure while the optimizer step is in
  progress, two neighbouring nodes lost, or no standby free: the recovery exits (codes 75, 74, 76) and v1's restart path applies.
- Measured on X-MoE 10B: failure list -> ready about 8-9 s, first step after it done about 23 s; Keep costs 3-6% of the step
  (`RESULTS_M1_M2.md`, M1-A and M1-C).
- Milestone 2 (resume inside a step, `FTV2_COMMIT=1`, under test): it needs ZeRO stage 2, which reduces after every
  micro-batch, and on X-MoE 10B that reduction, not the commit, is the main cost: stage 2 is 35% slower than stage 1, and the
  commits at k = 1 add 7.3% (E5 5862320). `FTV2_COMMIT_K` spaces only the commits. A k that also spaces the reductions (stage 1,
  ZeRO's own reduction called every k micro-batches; its partition sums accumulate across reductions) is the way to lower the
  35%; it is described in `RESULTS_M1_M2.md` (M2-C, design note) and not built. Two copy transports: `FTV2_COMMIT_MODE=p2p`
  (send the sums to the partner) and `keep` (the partner keeps its owner's slices from ZeRO's own all-reduce; no network).

## A new recovery plan: where it plugs in

1. **The plan** (`heartbeat-sys/hb_coordinator.py`, `_plan` and `_standby`): per failure kind, what to do. Today: replace the
   node by a free standby (`standby`, `replace` in the failure list), or nothing.
2. **The layout** (the failure list's fields): today the lost ranks and the standby node; a new plan adds its own fields (for
   example a new world size and the old rank of every new rank).
3. **The rebuild** (`recovery/rc_controller.py`, `Recovery._join(g, f, role, kept, old_keep)`): every rank, survivor or joiner,
   calls it with the list; it makes the new world, calls the trainer's init functions again and restores. A new layout changes
   the world size and this process's rank in it (read from the list) and the restore.
4. **The OOM ladder (a new world size, a different TP or EP)**: call `MegatronTrainer.set_world_size(world)`
   (`recovery/adapters/megatron_train.py`; sets `args.world_size`, `args.data_parallel_size`, rebuilds the micro-batch calculator;
   written, not yet tested) before `reinit_parallel()` and `build_model()`; and replace `ZeroAdapter.put_state(states)` (a copy
   per partition) by a re-split: gather each parameter's full fp32 master and moments from the old partitions (DeepSpeed's
   `utils/tensor_fragment.py` getters, or the universal-checkpoint conversion's merge) and set the new partitions (its setters).
   The toy's `re_split_state_after_shrink` (`assistant_deep_dive/2026-10-30-standby_join/toy/toy_ft5.py`) is the same idea.

## Tests without GPUs (login node)

```
V=/lustre/orion/world-shared/gen150/zixianw4/envs/TORCH2.13-ROCM7.14-XMOE_venv/bin/python
cd examples_elmoe/scripts-frontier/ft-sys-v2
python3 -I heartbeat-sys/tests/test_rules.py; python3 -I heartbeat-sys/tests/test_coordinator.py; python3 -I heartbeat-sys/tests/test_agent.py
$V -I heartbeat-sys/tests/test_monitor.py
python3 -I heartbeat-sys/tests/login_e2e.py --python $V --etcd-bin <etcd>           # the whole heartbeat on one node, about 8 min
$V recovery/tests/cpu_keep_test.py --modes gpu,cpu:torch                             # Keep, restore, decide (gloo, 4 processes)
$V recovery/tests/cpu_world_test.py                                                  # decide() cases, the new world through etcd
OMP_NUM_THREADS=6 $V recovery/tests/cpu_adam_bench.py                                # the two CPU Adams (DeepSpeed's needs the new stack's env)
```
