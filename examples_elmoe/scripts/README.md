# ELMoE — non-SLURM (torchrun) experiment scripts

This folder is the **non-SLURM counterpart** of `../scripts-frontier/`. Same experiments,
same config format, same analysis — but launched with `torchrun` and `bash` instead of
`sbatch`, so it runs on a plain GPU box or a non-SLURM cluster.

| file | role |
|---|---|
| `run_exp_training.sh` | **artifact-evaluation entry point** — the commands below |
| `autorun.sh` | the general-purpose, map-driven driver (mirrors `autorun_frontier.sh`) |
| `elmoe.sh.template` | the job template (`torchrun`), filled in by both of the above |
| `hostfile.example` | multi-node: one hostname per line → copy to `hostfile` |
| `env.sh.example` | site env (conda, modules, NCCL) → copy to `env.sh` |

---

## The one thing that differs from Frontier: **runs are SEQUENTIAL**

On SLURM, `main_results` (5 runs), `loss_validate` (2) and `profiling_cache` (8) are
independent **queued jobs that run in parallel**. Here they all contend for the *same*
GPUs, so they run **one after another**. Budget accordingly:

| experiment | Frontier (parallel) | here (sequential) |
|---|---|---|
| `profiling_cache` | ~8 min | ~64 min (8 × 8 min) |
| `main_results` | ~30 min | ~2.5 h (5 × 30 min) |
| `loss_validate` | ~1 h | ~2 h (2 × 1 h) |

---

## Prerequisites

```bash
cp env.sh.example env.sh        # conda activate / modules / NCCL settings  — REQUIRED
cp hostfile.example hostfile    # multi-node only: one hostname per line
```

`env.sh` is sourced **on every node**, because `ssh` does not carry your environment.
Multi-node also needs passwordless `ssh` and a **shared filesystem** (all ranks write
`rank_*.log` into one job dir).

---

## Artifact execution

Everything takes `--gpus N` (default `8`). `N` must be a multiple of `--gpus-per-node`
(default `8`).

### 0. Environment validation — confirms the build, warms the kernel caches
```bash
bash run_exp_training.sh env_validate --gpus 8
```
~30 min. *(Note: the mode is `env_validate`, not `validate`.)*

### 1. Profiling cache — required before any `yes_planner` run
```bash
bash run_exp_training.sh profiling_cache ELMoE --gpus 8
```
Sweeps micro-batch size 1→8 on a single node, ~8 min each → **~64 min sequential**.

Builds a **`63B_1L`** cache by default, because Experiment 1 trains 63B with the planner.
The cache is **hardware-specific** — a cache generated on other hardware is invalid, and
a Frontier cache will not work here.

### 2. Experiment 1 — throughput: ELMoE vs baselines (5 runs)
```bash
bash run_exp_training.sh main_results --gpus 32
```
63B, GBS=1024, 30 min per run → **~2.5 h sequential**. Runs:
ELMoE (+ELM-PP), ELMoE (no planner), X-MoE, DS-MoE, DS-Tutel.

### 3. Experiment 2 — loss curves: ELMoE vs X-MoE
```bash
bash run_exp_training.sh loss_validate --gpus 8
python3 ../utils/plot_loss_curve.py      # after BOTH runs finish
```
10B, GBS=320, 1000 steps, 1 h each → **~2 h sequential**. The plot script clips both
curves to the least common number of steps and writes `results/loss_validate_curve.png`
plus a `.csv`.

---

## Scaling down to a smaller box

**Experiment 1 is 63B, which needs ≥ 32 GPUs.** On fewer, it will not fit — the script
refuses to launch rather than let you sit through a run that OOMs:

```
ERROR: 63B needs at least 32 GPUs, but --gpus 8 was given.
       Either give it more GPUs:   --gpus 32
       or scale the model down:    --model-size 10B --gpus 8
```

To run the **same 5-way comparison on a single 8-GPU node**, drop to the 10B model:

```bash
bash run_exp_training.sh profiling_cache ELMoE --gpus 8 --model-size 10B   # 10B_1L cache
bash run_exp_training.sh main_results          --gpus 8 --model-size 10B
```

⚠️ **The absolute throughput numbers will not match the paper** — that requires 63B on
32 GPUs. What the scaled-down run *does* validate is that the whole pipeline works end to
end: the ELM-PP planner produces a partition, all five systems train, and the analyzer
emits the comparison table. Use it to check functionality, not to reproduce the figures.

Minimum GPUs per model: `10B → 8`, `63B → 32`, `173B → 256`, `537B → 480`, `1T → 960`.

**Keep the cache and the model in sync.** If you train 10B, profile 10B — the planner
looks up a cache keyed by model size and will hard-stop if it is missing.

---

## Outputs (identical to Frontier)

```
logs/job_<id>_pp<PP>_ep<EP>_<MOE>_ckpt_<..>_planner_<..>/
├── a-xmoe-<runtype>-<id>.o     driver stdout (the SLURM ".o" equivalent)
├── rank_<N>.log                per-rank logs
└── full_run.log                merged log the analyzers consume
results/<date>-results.xlsx     throughput / memory table (appended per run)
results/loss_validate_curve.png loss comparison (Experiment 2)
```

`../utils/analyze_log.py` and `analyze_memory.py` run automatically at the end of every
job — the same shared code Frontier uses, unmodified.

---

## Day-to-day use (not artifact evaluation)

`run_exp_training.sh` is a convenience wrapper. For arbitrary configurations, edit the
maps at the top of **`autorun.sh`** (`PP_STRATEGY_MAP`, `PP_BATCH_MAP`, `EP_BATCH_MAP`,
`PROFILE_MAP`) — identical format to `autorun_frontier.sh` — then:

```bash
bash autorun.sh
```
