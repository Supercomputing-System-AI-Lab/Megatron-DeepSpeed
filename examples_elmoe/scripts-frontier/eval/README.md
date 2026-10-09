# X-MoE Downstream Evaluation

Evaluates Megatron-DeepSpeed-X-MoE checkpoints on lm-evaluation-harness 0.4+ on
Frontier. Everything lives under `examples_elmoe/scripts-frontier/eval/`; nothing
here writes to checkpoints, training data, or the tokenizer.

Rewritten 2026-08-27 for the rebuilt pipeline (phase 2 of the eval rebuild; the
working documents are under `rebuild/`).

## 1. How to run anything

There is ONE entry point and one editable block:

```bash
cd .../examples_elmoe/scripts-frontier
# 1. edit the RUNS block at the top of eval/autorun_eval.sh   (nothing else)
bash eval/autorun_eval.sh --dry-run   # 2. see exactly what would be submitted
bash eval/autorun_eval.sh             # 3. submit it
```

Everything is validated before anything is queued, and every refusal names the
parameter it is about. A run already satisfied by the results registry prints
`SKIP-done`; a run whose job is still queued prints `SKIP-inflight`; neither is
resubmitted. `autorun_eval.sh` sources `submit_env.sh` itself, so running the file
is the entire procedure — there is nothing to source first.

### The chain

```
autorun_eval.sh          the RUNS block; sources submit_env.sh; execs eval_lib.py
  -> eval_lib.py         validates every declaration, plans, renders, sbatches
  -> frontier_eval.slurm.template     one rendered script per job, in temp_slurm/
  -> tasks/eval_harness/evaluate_v04.py            thin shim (kept as launch target)
  -> tasks/eval_harness/megatron_ds_eval/          the implementation:
       loader.py            checkpoint load (UCP/DS/consolidated) + Megatron setup
       lm_eval_adapter.py   the lm-eval <-> engine bridge: scoring, padding, KV decode
       cache.py             resumable per-request answer cache (section 3)
       probes.py            activation-dump tooling (deepseek-moe-verify hooks)
       main.py              CLI, per-task loop, correctness probes, registry writing
  -> render_results.py     runs at the end of EVERY job; refreshes results/tables/
```

`tasks/eval_harness/result_registry.py` is the provenance authority (code
fingerprint + registry writer), `../utils/model_registry.py` resolves model
architectures, and `../monitor_run.sh` watches each job from inside the template.

## 2. Declaring a run — every key

One line per run in the RUNS block; `DEFAULTS` in the same file fills anything you
leave out; anything DEFAULTS leaves out comes from the task table
(`task_table.py` + `config/custom_suites.py`), where each benchmark's measured
seqlen, shots, micro-batch and generation budget live. The submit-time plan prints
the values actually used, so a default is never hidden.

| key | meaning |
|---|---|
| `model=` | model_registry key, e.g. `DeepSeek_16B` (architecture + experts) |
| `moe=` | MoE implementation to build with, e.g. `X-MOE` |
| `run=` | checkpoint run dir: name under `scripts-frontier/checkpoint/`, relative or absolute path |
| `steps=` | `2280` \| `2280,2400` \| `latest` \| `all` |
| `benchmarks=` | benchmarks, suites or categories, comma-separated; misspellings are refused WITH the known names |
| `time=` | wall clock `HH:MM:SS` |
| `nodes=` | node count (8 GCDs per node on Frontier) |
| `ep=` | expert-parallel size — shards the MoE layers |
| `tp=` | tensor-parallel size — REFUSED today for tp>1 (UCP loader copies atoms whole; job 5354153) |
| `pp=` | pipeline-parallel size — REFUSED today for pp>1 (scoring would record random numbers; decode reads logits on every rank; pipe-spec names don't load; jobs 5354152/5354153). The lift rides the consolidated changeset, see section 5 |
| `mbs=` | micro-batch size |
| `shots=` | few-shot examples |
| `seqlen=` | sequence length (refused where the benchmark locks it — for wikitext and RULER the length IS the evaluation) |
| `pad=` | `left` \| `right` (see `PADDING_AND_FLASH.md`) |
| `limit=` | documents per (sub)task; omit for the whole benchmark |
| `slices=` | split one benchmark across N jobs (1 = one job) |
| `workers=` | split one benchmark across N INDEPENDENT model replicas inside ONE job (1 = one model). One node and its 8 GCDs per worker, one shared answer file, an in-job collector writes the score. Few-shot rows split at whole-sub-task granularity automatically; refused with `pp>1`, with `slices>1`, and when a few-shot row has no addressable sub-tasks. See `rebuild/WORKERS_FEATURE.md` |
| `slice_by=` | `share` (ordered share of the question list) \| `subtask` (named sub-task ranges) |
| `layout=` | `share` (contiguous) \| `round_robin` (every Nth document) |
| `docs=` | per-sub-task doc count, only to slice by share with no committed subset manifest and no limit |
| `use_ucp=` | `1` (default) universal-checkpoint path; `0` consolidated `.pt`. Part of the answer-file identity |
| `experiment=` | declare this run an EXPERIMENT — a measurement made to COMPARE, never to publish. Unset (the default) is a PRODUCTION run whose rows ARE the published numbers. An experiment gets a private answer file (so it is forced cold), tags every registry row `experiment=<tag>` so no consumer can ever select it as a published value, keys the done-check (so it never marks production done and its control arm needs no `--force`), and carries `_exp<tag>` in the job name. `answers_ns=` is the deprecated spelling. See `rebuild/ANSWER_NAMESPACE_DESIGN.md` |

### Slicing, and its two rules

Slices are disjoint by arithmetic — no locks, no claims. All slices write into the
SAME answer file (the cache keys requests by content, not by document index), so a
dead slice is resumed by resubmitting it and only unanswered questions recompute.
The score is written by a collecting run, submitted automatically with a dependency
on the slices; it runs the unsliced configuration, hits the cache on every request,
and scores the complete benchmark in minutes.

Rules: (1) slice only 0-shot benchmarks — lm-eval draws few-shot context from one
RNG stream consumed in evaluation order, so a sliced few-shot job answers
DIFFERENT prompts than the unsliced run (measured: 20/80 shared requests at
4-shot vs 80/80 at 0-shot; the submit-time refusal is phase-2 work order WO-7 in
`rebuild/WORKORDERS.md`). (2) group benchmarks whose structure lives in metadata
(`ruler8k`, `longbench`) slice via their `_s` subset rows, not `slice_by=subtask`.

## 3. The answer cache — identity and traps

`cache/responses/` holds one append-only JSONL file per
(checkpoint x code x eval-config) identity:

```
{run}__global_step{step}_m{mtime}_code{code_fp}_{ucp|ds|cons}_pad{side}_seq{len}_fs{shots}[_nokv].jsonl
```

Line 1 is a `{"_meta": ...}` provenance header; identity is enforced by the
FILENAME, never by the header. Every computed batch is flushed immediately, so a
walltime kill loses at most one batch and a chained job resumes exactly where the
last one stopped. A finished-from-cache job writes final results in minutes.

What is deliberately NOT in the identity — each one is a trap worth knowing:

- **`limit` and slices**: keys are per-request, so a `limit=20` smoke run pre-warms
  the full run instead of living in a separate file.
- **The parallel shape** (nodes/ep/tp/pp): a run at a new shape reads the default
  shape's answers as hits. A Gate-2 "does shape B match shape A?" comparison must
  therefore force a cold recompute (private answer namespace — work order WO-11),
  or it is vacuously true.
- **`mtime` IS in the identity**: it is the modification time of the step's
  weights artifact. Re-converting or even touching the checkpoint silently starts
  a new (cold) answer file; the old one is orphaned, not merged. This protects
  weight identity — do not "fix" it by renaming files.
- **`code_fp` IS in the identity**: see section 5.

`cache/` also holds the pinned dataset caches the jobs read offline (`hf_home`,
`hf_datasets_cache`, `ruler`), consolidated `.pt` checkpoints (`consolidated`),
per-document score dumps (`perdoc`), and the prebuilt kernel caches
(`torch_extensions_good`, `triton_cache`).

## 4. Results and reporting

- `results/registry/<run>.jsonl` — the per-run registry shards; one append-only row
  per (task, metric), stamped with job id, settings, and the code fingerprint.
  `results/registry.jsonl` is the frozen legacy file; readers union both.
- `results/submissions.jsonl` — the submissions ledger the launcher's in-flight
  check reads.
- `results/tables/` — rendered tables, refreshed by `render_results.py` at the end
  of every job.
- `report.py` — pivot the registry into tables; `--compare-code` puts one column
  per code fingerprint for the same measurement (the regression check).
- `significance.py`, `render_branches.py`, `render_longctx.py` — branch/long-context
  comparison tables with paired tests over `cache/perdoc`.
- `score_partial.py` / `rescore_partial.py` / `extract_cell.py` — score directly
  from the answer cache (the queue-proof delivery path); `dashboard/` builds the
  intra-doc dashboard from the registry.

## 5. Code-fingerprint lineage (why numbers are trustable)

`result_registry.py` hashes the six FORWARD_SOURCES files whose bytes can change a
forward pass (4 megatron model files + `lm_eval_adapter.py` + `loader.py`) into a
short `code_fp`. It is stamped on every registry row and baked into every answer
filename, so a code change auto-invalidates the cache instead of silently serving
stale logprobs (the RoPE-bf16 lesson). Orchestration files (`main.py`, `cache.py`,
`evaluate_v04.py`, `probes.py`) are deliberately excluded.

Consequence: when the pending consolidated FORWARD_SOURCES changeset lands (PP
score broadcast, decode termination, UCP loader TP/PP support, rotary 0-dim fix,
pipeline chunked-CE), the fingerprint rotates and all current-fp caches go cold
until re-blessed. That re-blessing is EVIDENCE-GATED and tool-run, never a hand
`mv`:

1. `rebuild/gate1_check.py` — re-runs of the 8 pinned baseline jobs must reproduce
   all 93 metrics of `rebuild/baseline/baseline.json` digit-for-digit AND every
   compared cached answer byte-for-byte; emits `rebuild/fp_migration/gate1-<ts>.json`
   with a PASS / SPLIT / FAIL verdict per method class.
2. `rebuild/migrate_fp.py` — dry-run by default; `--apply` refuses without a
   passing gate file; copies (never renames) each old-fp file to the new-fp name
   with an appended `_meta_migration` marker; writes a manifest; `--rollback`
   reverses it. Old-fp files stay on disk as the permanent record.

Lineage table (filled in by the migration when it runs):

| code_fp | period | what changed | migration evidence |
|---|---|---|---|
| 17 older fps | pre 2026-08 | historical; their caches are orphaned history, out of scope | — |
| `49b64046` | current | the rebuilt pipeline's baseline fingerprint | `rebuild/baseline/baseline.json` |
| _NEWFP (pending)_ | _after the consolidated changeset_ | _PP score broadcast + decode termination + UCP loader + rotary + chunked-CE_ | _`rebuild/fp_migration/gate1-<ts>.json` + `manifest-<ts>.json` (placeholder — not yet run)_ |

To audit any pair of generations: `python3 report.py --compare-code` shows the
same measurement side by side per fingerprint; Δ must be 0.0000 on every
measurement shared across a legitimate migration.

## 6. Layout (top level, production)

| entry | role |
|---|---|
| `autorun_eval.sh` | THE entry point (RUNS block + DEFAULTS) |
| `eval_lib.py` | validation, planning, rendering, submission, registry done-checks |
| `frontier_eval.slurm.template` | the one slurm template every job renders from |
| `submit_env.sh` | modules + env the jobs inherit (sourced by autorun) |
| `mvp_ds_config.json` | inference DeepSpeed config (ZeRO=0, bf16) |
| `task_table.py`, `config/` | per-benchmark measured settings, custom suites, subset manifests, RULER metadata |
| `custom_tasks/` | lm-eval task definitions loaded via `--include-path` |
| `check_seqlens.py`, `dataset_seqlen.py`, `prefetch_datasets.py` | dataset guards the launcher's refusals name |
| `make_subsets.py`, `make_needleppl.py`, `export_alpacaeval.py` | producers of `config/subsets` and custom-task data |
| `xmoe_consolidate.py` | consolidated-checkpoint escape hatch (`use_ucp=0`) |
| `tests/` | the test suite (section 7) — never move it; jobs import it by absolute path |
| `report.py` + section-4 tools | reporting |
| `results/`, `cache/`, `logs/`, `temp_slurm/` | registry+tables, caches+answers, per-job logs, rendered job scripts |
| `deepseek-moe-verify/` | HF-vs-us verification code driving the template's DUMP hooks |
| `rebuild/` | phase-2 working dir: acceptance/QA docs, frozen Gate-1 baseline, migration tools |
| `archived/` | everything retired; moves are reversible, nothing is deleted |
| `NOTICE_2026-08-22_CODE_CHANGES.md` | changelog of behavior-relevant code changes |
| `PADDING_AND_FLASH.md` | doc of record for left-pad eval + flash-attn interaction |

## 7. Tests

```bash
cd .../scripts-frontier/eval
python3 -m unittest discover -s tests -v        # 112 login-node tests (eval_lib & friends)
python3 rebuild/test_fp_migration.py            # 42 login-node tests (gate1_check + migrate_fp)
```

`tests/test_runtime.py` is the GPU probe suite — it runs INSIDE every eval job
(`main.py` imports it by absolute path and runs the correctness probes against the
loaded model); `tests/test_consolidate.py` needs a consolidated `.pt`. Neither is
collected by the login-node discover run above, by design.
