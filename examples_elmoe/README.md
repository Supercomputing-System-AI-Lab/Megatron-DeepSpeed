# ELMoE Training Examples

End-to-end launch pipeline for training ELMoE / X-MoE models on Frontier
(AMD MI250X) and on portable Triton-only deployments (incl. NVIDIA). All
submissions are driven from one script — [`scripts-frontier/autorun_frontier.sh`](scripts-frontier/autorun_frontier.sh)
— which fans out SLURM jobs from a small set of associative maps. To
reproduce a specific experiment, uncomment the matching map entry and run
the script.

## Table of contents

1. [Quick start — running experiments](#1-quick-start--running-experiments)
2. [Configuration maps](#2-configuration-maps)
3. [Execution flow under the hood](#3-execution-flow-under-the-hood)
4. [Run modes: profiling cache vs. planner-guided](#4-run-modes-profiling-cache-vs-planner-guided)
5. [Analyzing results](#5-analyzing-results)

---

## 1. Quick start — running experiments

All experiments live as (commented) entries inside
[`scripts-frontier/autorun_frontier.sh`](scripts-frontier/autorun_frontier.sh).
To reproduce one, **uncomment the relevant lines** and run:

```bash
cd Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier
bash autorun_frontier.sh
```

The script walks three maps and submits one `sbatch` per configuration.
As shipped, a single demo is active — **63B ELMoE on 64 GPUs with ELM-PP
planner + dynamic layer-wise activation checkpointing**:

```bash
PP_STRATEGY_MAP["8:64"]="4:8"
PP_BATCH_MAP["8:64"]=" 1:256:15:ELMOE-3D:63B:dynamic-ckpt:1:uneven:yes-planner "
```

All other entries are commented templates you can selectively enable.

### Prerequisites before running a planner-guided job

The ELM-PP planner (`yes-planner` / `yes-planner-membal`) is a cache
**consumer** — it reads per-layer timings from
`scripts-frontier/planner_profiling_cache/`. Before you can run a
planner-guided PP job at a given `(model, ep, mbs)` triple, that cache
entry must exist. The autorun script checks for the expected filename via
`model_registry.py get_filename` and **silently skips submission** if the
JSON is missing. Populate the cache first by enabling the matching
`PROFILE_MAP` entry and running the script (see [Section 4.1](#41-profiling-cache-collection-producer)).

### Minimal reproduction recipes

| Goal | What to uncomment | Notes |
|---|---|---|
| ELMoE 63B baseline run (the demo as shipped) | the two `8:64` lines above | Requires 63B profile cache present. |
| Populate the profile cache for a new model | one entry in `PROFILE_MAP["1:8:1"]` | Runs mbs = 1..10 on a single 8-GPU node; writes JSONs used by the planner. |
| Baseline comparison (X-MoE, DeepSpeed-MoE, Tutel, TED) | one entry in `EP_BATCH_MAP` | Flat EP, no planner, no uneven PP. |

---

## 2. Configuration maps

Three associative arrays drive submission. Each key encodes the SLURM
resource shape; each value is a colon-delimited run spec. Space-separated
run specs within a single map value submit multiple jobs against the same
node budget.

### 2.1 `PP_STRATEGY_MAP` and `PP_BATCH_MAP` — ELMoE PP-centric runs

```
PP_STRATEGY_MAP["<NODES>:<TOTAL_GPUS>"] = "<PP_SIZE>:<EP_PARALLEL_SIZE>"
PP_BATCH_MAP["<NODES>:<TOTAL_GPUS>"]    = "<mbs>:<nbs>:<iters>:<moe_type>:<model_size>:<ckpt>:<ckpt_interval>:<pp_partition>:<planner>"
```

| Token | Values |
|---|---|
| `moe_type` | `ELMOE-3D` (SeqGEMM), `ELMOE-GroupedGEMM-primus` (CK grouped-GEMM, AMD-only), `ELMOE-GroupedGEMM-triton` (Triton grouped-GEMM, portable to NVIDIA) |
| `model_size` | `10B`, `63B`, `173B`, `537B`, `1T` (see [`utils/model_registry.py`](utils/model_registry.py) for full config) |
| `ckpt` | `no-ckpt` (off), `ckpt` (every layer), `dynamic-ckpt` (planner-chosen per-layer placement) |
| `pp_partition` | `even` (uniform layers/stage), `uneven` (planner-chosen per-stage layer count) |
| `planner` | `no-planner`, `yes-planner` (ELM-PP: minimize per-stage execution time under memory budget), `yes-planner-membal` (memory-balanced partitioning only — no throughput consideration) |

DP is auto-inferred from `TOTAL_GPUS / (PP_SIZE × EP_PARALLEL_SIZE × MP_SIZE)`
and defaults to ZeRO-1.

### 2.2 `EP_BATCH_MAP` — EP-centric baselines

```
EP_BATCH_MAP["<NODES>:<EP_PARALLEL_SIZE>:<MP_SIZE>"] = "<mbs>:<nbs>:<iters>:<moe_type>:<model_size>:<ckpt>:<ckpt_interval>:<zero_stage>"
```

Used to run flat EP comparison baselines (X-MoE, DeepSpeed-MoE, Tutel,
DeepSpeed-TED). No PP, no planner. `zero_stage` is 1 or 2.

### 2.3 `PROFILE_MAP` — profiling cache collection

```
PROFILE_MAP["1:8:1"] = "<mbs>:<nbs>:<iters>:X-MOE:<model>_1L:no-ckpt:0 ..."
```

Always runs on 1 node / 8 GPUs / MP=1. Model variant is a single-layer
shim (`10B_1L`, `63B_1L`, `173B_1L`, `537B_1L`, `1T_1L`) so per-layer
timings are isolated from inter-layer scheduling noise. The submission
loop iterates `mbs = 1..10` for each entry.

---

## 3. Execution flow under the hood

```
autorun_frontier.sh              (generates one SLURM script per run spec)
        |
        v
frontier_elmoe.slurm.template    (unified template: env setup, Triton
        |                         cache warm, DeepSpeed config rendering)
        |   srun python ../ELMoE_launch.py ...
        v
ELMoE_launch.py                  (thin wrapper: optionally runs the planner,
        |                         then hands off a modified argv)
        |   runpy.run_module("pretrain_gpt_deepspeed")
        v
pretrain_gpt_deepspeed.py        (standard Megatron-DeepSpeed pretrain)
```

### 3.1 `frontier_elmoe.slurm.template`

A single unified template for both production and profiling-cache
runs. Two placeholders gate the mode:

- `{{RUN_PLANNER}}` — `true` / `true-membal` / `false`. Passed straight
  through to `ELMoE_launch.py --run-planner`.
- `{{COLLECT_PROFILING_CACHE}}` — `true` / `false`. When `true`, the
  template forces `PP_SIZE=1`, enables `WALL_CLOCK_BREAKDOWN=true`
  (needed to emit per-module timer lines), forces `RUN_PLANNER=false`
  (a producer run must not also try to consume an empty cache), skips
  the Triton cache warm, and passes `--profiling` to `analyze_log.py`.

Everything else — Megatron/DeepSpeed arg rendering, MoE-type branches,
ZeRO config, per-rank log routing — is shared between the two modes.

### 3.2 `ELMoE_launch.py`

The wrapper peels off planner-only flags (`--run-planner`,
`--planner-mode`, `--planner-profile-dir`, `--planner-memory-limit-gb`,
...) with `parser.parse_known_args()` and leaves the rest of Megatron's
argv untouched in `remaining_argv`. Dispatch:

- `--run-planner false` — pass through directly to
  `pretrain_gpt_deepspeed.py` via `runpy.run_module(...)`. No planner cost.
- `--run-planner true` — `run_planner()` runs `PipelineOptimizer.find_optimal_config(...)`:
  MiniMax search over `(mbs, uneven-PP partition, per-stage ckpt count)`
  that minimizes estimated step time under the memory budget.
- `--run-planner true-membal` — `PipelineOptimizer.find_max_mbs_membal_config(...)`:
  picks the largest micro-batch size that fits and balances residual
  memory across stages. **Prioritizes memory balance only; no throughput
  consideration.**

After the planner returns, `apply_planner_result()` writes the decision
into `remaining_argv`:

```python
remaining_argv.extend(['--uneven-pp-partition']          + partition_list)
remaining_argv.extend(['--dynamic-checkpoint-partition'] + ckpt_list)
```

Plus `--micro-batch-size` and `--global-batch-size` are overwritten in
place, and the env vars `UNEVEN_PP_PARTITION`,
`DYNAMIC_CHECKPOINT_PARTITION`, `UNEVEN_PP=True`, `DYNAMIC_CHECKPOINT=True`
are exported for Megatron's model builder. These two `extend(...)` lines
are the single boundary at which planner decisions cross into Megatron —
downstream pipeline partitioning and per-layer checkpoint placement key
off those arguments.

### 3.3 Planner internals — [`deepspeed/moe/planner/`](../../deepspeed/moe/planner)

- `memory_calculator.py` — `MemoryPredictor`: per-layer activation /
  weight / optimizer-state estimates for candidate `(mbs, pp, tp, ep, ckpt)`
  configurations.
- `ELM_planner.py` — `ProfileLoader` (reads `planner_profiling_cache/*.json`),
  `PipelineOptimizer` (search driver), and the `PlannerConfig` /
  `PartitionResult` / `StagePlan` data types.

The planner operates entirely on analytical models fed by cached profile
JSONs — no re-profiling at plan time.

---

## 4. Minimax Planner Execution

### 4.1 One-time Offline Profiling

Purpose: measure per-layer `T_gemm` and `T_comm` once per `(model, 
mbs)`, so the planner can cost candidate partitions analytically.

Pipeline:

1. Autorun renders the template with `{{COLLECT_PROFILING_CACHE}}=true`.
2. The template forces `PP_SIZE=1` and `WALL_CLOCK_BREAKDOWN=true`, so
   DeepSpeed emits per-module timer lines into the log.
3. Training runs for the configured number of iterations.
4. `utils/analyze_log.py --profiling` parses the merged log and writes
   a JSON into `planner_profiling_cache/` keyed by
   `(model_size, nodes, total_gpus, mbs)`. `RUN_PLANNER` is held at
   `false` so no consumer path runs during cache population.

### 4.2 Planner-guided training (consumer)

1. Autorun renders the template with `{{RUN_PLANNER}}=true` (or
   `true-membal`) and `{{COLLECT_PROFILING_CACHE}}=false`.
2. Before submitting, autorun computes the expected cache filename and
   **skips the job** if the JSON is missing. No crash, no partial run.
3. `ELMoE_launch.py` loads the JSON, runs the selected optimizer mode
   (`optimize` or `optimize-membal`), and injects the resulting
   `--uneven-pp-partition` and `--dynamic-checkpoint-partition` into argv.
4. `pretrain_gpt_deepspeed.py` boots with the planner-chosen layout.

---

## 5. Analyzing results

Every job writes its artifacts under a scoped directory keyed by
SLURM job ID + configuration:

```
scripts-frontier/
├── logs/
│   └── job_<SLURM_JOB_ID>_pp<PP>_ep<EP>_<MOE_TYPE>_ckpt_<CKPT>_planner_<RUN_PLANNER>/
│       ├── a-xmoe-<RUN_TYPE>-<SLURM_JOB_ID>.o   # SLURM stdout (config echo, env, memory stats)
│       ├── a-xmoe-<RUN_TYPE>-<SLURM_JOB_ID>.e   # SLURM stderr (OOM / tracebacks land here)
│       ├── rank_<i>.log                         # per-rank training log
│       └── full_run.log                         # merged .o + rank_0 + rank_last, what analyze_log.py parses
└── results/
    └── YYYY-MM-DD-results.xlsx                  # daily aggregated spreadsheet (main results + loss-per-iter)
```

### 5.1 Normal completion

When a run finishes inside its SLURM time limit, the template runs
`analyze_log.py` at the end of the script. That parses `full_run.log`,
appends a row to today's `results/YYYY-MM-DD-results.xlsx`, and — for
profiling runs only — writes the per-layer JSON into
`planner_profiling_cache/`. You typically need to do nothing.

### 5.2 Re-running analysis for timed-out or post-hoc runs

If a job wall-clocks out before its `analyze_log.py` step can execute,
the log directory is still complete — just unparsed. Point
[`utils/rerun_analysis.sh`](utils/rerun_analysis.sh) at it:

```bash
bash ../utils/rerun_analysis.sh \
    logs/job_<SLURM_JOB_ID>_pp<PP>_ep<EP>_<MOE_TYPE>_ckpt_<CKPT>_planner_<RUN_PLANNER>
```

The script re-merges `.o` + rank logs into `full_run.log`, invokes
`analyze_log.py` against it, and appends the resulting row to today's
`results/*.xlsx`. Safe to re-run — new rows append, existing ones are
not overwritten.

### 5.3 Diagnosing failures

- **OOM** — shows up in `.e` (the SLURM stderr file). Look there first
  for `CUDA out of memory`, `HIP out of memory`, or
  `RuntimeError: ... out of memory`.
- **NCCL / FI hangs** — `.e` will contain the timeout message once
  `NCCL_TIMEOUT` (1200 s in the template) elapses.
- **Planner skip** — a job listed in the autorun maps but never actually
  submitted: autorun could not find the matching profile cache JSON.
  Check the autorun stdout for `[Planner] Cache MISSING ...` and populate
  the cache via the corresponding `PROFILE_MAP` entry first.
- **Missing `full_run.log`** — the run died before post-processing; use
  [Section 5.2](#52-re-running-analysis-for-timed-out-or-post-hoc-runs)
  to re-parse manually.
