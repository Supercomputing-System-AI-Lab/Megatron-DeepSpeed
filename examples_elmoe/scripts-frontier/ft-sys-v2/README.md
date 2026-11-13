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

## Run from a fresh clone

**Members of gen150 (2026-11-13):** every outside file a run needs has a readable copy under
`/lustre/orion/world-shared/gen150/zixianw4` (the venv, ROCm 7.14.1 for gfx90a, the aws-ofi-nccl plugin, etcd, and the GPT-2
data and vocabulary the runs use since 2026-11-11). `. env/shared_gen150_env.sh` sets the variables below to those copies and to
this clone; then `bash env/ftv2_check_env.sh`. The DeepSeek tokenizer and data named in the table are no longer used or kept.
The open problems of the 63B runs and how to reproduce them: `OPEN_63B_OVERLAP.md`.

Every path a run reads is a variable of `env/ftv2_paths.sh`; a variable set in the shell wins. The defaults are the folders the
runs used on 2026-11-06. Those in `/lustre/orion/world-shared/gen150` and `/opt` are readable by every member of gen150; those
in `/lustre/orion/gen150/scratch/zixianw4` are not (the folder is `drwx------`). Another user sets the private ones to copies
of their own, checks, renders, then submits:

```
git clone -b ft-sys-v2 --recurse-submodules https://github.com/Supercomputing-System-AI-Lab/X-MoE.git
cd X-MoE/Megatron-DeepSpeed-X-MoE && git checkout ft-sys-v2     # the submodule is checked out detached at the pinned commit
FT=$PWD/examples_elmoe/scripts-frontier/ft-sys-v2
export NEWSTACK_DEEPSPEED=$(cd .. && pwd)                       # this clone's DeepSpeed (X-MoE root)
export NEWSTACK_OVERLAY=$FT/env/overlay                         # the copy of sitecustomize.py in this repository
export NEWSTACK_JIT=/lustre/orion/<project>/scratch/$USER/ftv2_jit   # any folder you can write, on Lustre
export NEWSTACK_R714=<your ROCm 7.14.1 gfx90a>/rocm NEWSTACK_PLUGIN=<your aws-ofi-nccl>/install/lib
export DATA_PATH=<prefix of a .bin/.idx pair> FTV2_VOCAB_FILE=<tokenizer.json>
export WANDB_MODE=disabled                                      # or `wandb login` once
bash $FT/env/ftv2_check_env.sh                                  # every path: OK / MISSING / NOT READABLE / NOT WRITABLE
cd $FT/runs && RENDER_ONLY=1 bash ftv2_10B.sh                   # render only, submit nothing
bash ftv2_10B.sh                                                # submit (debug QOS, 30 min, 3 + 1 nodes)
```

| Variable | What | Default (2026-11-06) | Others can read | Where to get it |
|---|---|---|---|---|
| `FTV2_TREE` | this Megatron tree (megatron/, ft-sys-v2/) | the tree of the run file | - | this clone |
| `FTV2_SF` | the launch folder: launcher, job template, `eval/submit_env.sh`, `../ds_config_gpt_TEMPLATE.json`; jobs write `logs/`, `temp_slurm/` here | `$FTV2_TREE/examples_elmoe/scripts-frontier` | - | this clone |
| `FTV2_ENV` | the new stack's env file | `env/newstack_env.sh` | - | this clone |
| `NEWSTACK_V` | Python venv: torch 2.13.0+rocm7.1, apex, flash-attn 2.8.3 (CK, gfx90a), mpi4py on Cray MPICH | `/lustre/orion/world-shared/gen150/zixianw4/envs/TORCH2.13-ROCM7.14-XMOE_venv` | yes | use as is (read only) |
| `NEWSTACK_DEEPSPEED` | the X-MoE DeepSpeed, first on PYTHONPATH (in place of `eval/submit_env.sh`'s entry) | `/lustre/orion/gen150/scratch/zixianw4/X-MoE` | no | the X-MoE clone's root |
| `NEWSTACK_R714` | ROCm 7.14.1 for gfx90a (HIP, HSA, RCCL 2.30.4), loaded by LD_PRELOAD | `/lustre/orion/gen150/scratch/zixianw4/therock-7.14.1-gfx90a/rocm` | no | `https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx90a-7.14.1.tar.gz` (2.0 GB), `tar -xzf` into `<dir>/rocm` |
| `NEWSTACK_PLUGIN` | aws-ofi-nccl 1.22.0, RCCL's network plugin | `/lustre/orion/gen150/scratch/zixianw4/aws-ofi-nccl-v1.22.0/install/lib` | no | `https://github.com/aws/aws-ofi-nccl` tag v1.22.0: `autoreconf -ivf; ./configure --prefix=<dir>/install --with-rocm=/opt/rocm-7.14.1 --with-libfabric=/opt/cray/libfabric/2.3.1 --disable-tests --enable-werror=no CC=gcc CXX=g++; make -j; make install` |
| `NEWSTACK_LIBFABRIC` | libfabric 2.3.1 | `/opt/cray/libfabric/2.3.1/lib64` | yes | site |
| `NEWSTACK_JIT` | JIT caches (torch extensions, Megatron fused kernels, Triton); written by the job | `.../2026-11-04-xmoe_newstack/jit` (user's scratch) | no | any writable folder of yours |
| `NEWSTACK_OVERLAY` | `triton_deepbind/sitecustomize.py` (Triton with RTLD_DEEPBIND on the 7.14.1 runtime) | `.../2026-11-04-xmoe_newstack/overlay` (user's scratch) | no | `env/overlay` in this repository |
| `NEWSTACK_MEGATRON_COPY` | a copy of megatron/ that the 11-04 runs used; after `FTV2_TREE` on PYTHONPATH, so not imported | `.../2026-11-04-xmoe_newstack/tree` | no | not needed |
| `FTV2_ETCD_BIN` | etcd 3.5.17 (the heartbeat's store) | `/lustre/orion/world-shared/gen150/zixianw4/tools/etcd-v3.5.17/etcd-v3.5.17-linux-amd64/etcd` | yes | or `bash heartbeat-sys/get_etcd.sh <dir>` (GitHub release, SHA-256 checked) |
| `DATA_PATH` | training data prefix; the job template reads it | template default `.../examples_elmoe/data/longctx/L8-dolma3-deepseek_text_document` (user's scratch, 6.8 GB) | no | a copy, or any Megatron .bin/.idx made with the DeepSeek tokenizer (`examples_elmoe/data/README.md`) |
| `FTV2_VOCAB_FILE` | tokenizer; empty = the template's `VOCAB_FILE` | template `.../examples_elmoe/data/deepseek_tokenizer/tokenizer.json` (user's scratch) | no | `tokenizer.json` of `deepseek-ai/deepseek-moe-16b-base` (100,015 tokens; padded to 102,400) |
| py-spy | the hang's stacks (agent: PATH, else `~/.local/bin/py-spy`) | `~/.local/bin/py-spy` 0.4.2 | own | `pip install --user py-spy` |

- `env/ftv2_check_env.sh` prints each path as OK, MISSING, NOT READABLE or NOT WRITABLE with its variable, adds "(private: ...)"
  to an OK path that other members of gen150 cannot read, shows where `deepspeed` and `megatron` will be imported from, and exits
  1 on any problem. `ftv2_run_common.sh` also stops at render time if megatron or deepspeed would not come from `FTV2_TREE` /
  `NEWSTACK_DEEPSPEED`.
- The tokenizer: the template sets `VOCAB_FILE` without reading a variable and builds `data_options` from it. With
  `FTV2_VOCAB_FILE` set, the batch script replaces the `--vocab-file` argument of `data_options` right before the training srun
  (through the launcher's `AUTORUN_PRE_CMD`) and stops if the replacement did not land. The launcher and the template are unchanged.
- The template copies the user's Triton cache (`/lustre/orion/gen150/scratch/zixianw4/.triton/cache`) to `/tmp` on every node
  before training. For another user the copy fails with "Permission denied" and nothing else happens: the new stack uses
  `NEWSTACK_JIT/triton`. `PROFILING_SKIP_CACHE_WARM=1` (read by the template) skips the copy.
- Since 2026-11-06 the launch folder is this tree's `scripts-frontier` (before: the user's shared checkout). To launch from the
  shared checkout as before: `FTV2_SF=/lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier`.
- Still private with no variable: the optional pre-run gate (`FTV2_GATE`, off by default) reads `alltoall_perf` and
  `rccl_torch_lib` from the user's `assistant_deep_dive/2026-10-09-collective_stack/tools` (`gate/gate_node.sh`).

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
- Bit-exact claims need the determinism mode: `FTV2_DETERMINISTIC=1` with `NCCL_ALGO=Ring NCCL_PROTO=Simple` (torch's
  deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, flash-attention's deterministic backward; about +4.7% step
  time). In it, two clean runs are bit-identical, and a kill recovered from the GPU shadow is bit-identical to the clean run on
  every rank (E4 5863739). In normal mode, compare against the spread of several clean runs instead.
- Milestone 2 (resume inside a step, `FTV2_COMMIT=1`, under test): it needs ZeRO stage 2, which reduces after every
  micro-batch, and on X-MoE 10B that reduction, not the commit, is the main cost: stage 2 is 35% slower than stage 1, and the
  commits at k = 1 add 7.3% (E5 5862320). `FTV2_COMMIT_K` spaces only the commits. A k that also spaces the reductions (stage 1,
  ZeRO's own reduction called every k micro-batches; its partition sums accumulate across reductions) is the way to lower the
  35%; it is described in `RESULTS_M1_M2.md` (M2-C, design note) and not built. Two copy transports: `FTV2_COMMIT_MODE=p2p`
  (send the sums to the partner; +7.3% at k = 1, nothing measurable at k = 4) and `keep` (the partner keeps its owner's slices
  from ZeRO's own all-reduce; no network, nothing measurable at k = 1; E5d 5862827). A kill in the middle of step 10 with
  `keep` resumed at micro-batch 3 with the clean step's gradient norm (M2-A).
- DeepSpeed fix applied on the object when ZeRO-2's reduce path is configured (`use_multi_rank_bucket_allreduce=false`): its
  owner check compares ranks of two different groups, so expert gradients were kept by the wrong ranks (`RESULTS_M1_M2.md`,
  DS-1; `FTV2_DS_REDUCE_FIX`).

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
