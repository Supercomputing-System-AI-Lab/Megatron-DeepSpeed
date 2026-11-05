# PORT: what the sandbox changes against production, and how to switch it on for the big chain

Written 2026-10-09 by the resilience agent. The list comes from a real diff, not from memory:
`diff -rq` of `/lustre/orion/gen150/scratch/zixianw4/X-MoE` (production) against
`/lustre/orion/gen150/scratch/zixianw4/X-MoE-resilience` (sandbox) over `deepspeed/`, `Megatron-DeepSpeed-X-MoE/megatron/` and
`Megatron-DeepSpeed-X-MoE/examples_elmoe/`. Logs, outputs, checkpoints, records, temp_ds_json and the results spreadsheet are
left out. The +/- counts are the `>` and `<` lines of `diff production sandbox`.

The rule for every change below: it is additive and sits behind a flag. With every flag unset, the production run behaves as it
does today. The production template `frontier_elmoe.slurm_0623_2026.template` is **identical** in the sandbox and is not touched.
The resilient template is a separate file. The launcher renders it only when `AUTORUN_RESILIENT=1`.

## 1. The files that differ (sandbox against production)

Paths are relative to `/lustre/orion/gen150/scratch/zixianw4/X-MoE{,-resilience}/`.

| path | +/- lines | what it adds | the flag that gates it |
|---|---|---|---|
| deepspeed/comm/torch.py | +6 / -0 | a timeout for the groups DeepSpeed creates (the expert-parallel group among them); before, they took torch's 10-minute default | `XMOE_PG_TIMEOUT_MIN` (unset: as before) |
| deepspeed/moe/moe_v2.py | +7 / -0 | heartbeat hooks around the MoE all-to-all (the collective count and the phase) | `XMOE_HB=1` (the hook is a no-op without the hb module) |
| deepspeed/runtime/pipe/p2p.py | +12 / -2 | heartbeat hooks around the pipeline send and receive (the 2 deleted lines are the bare `return dist.send/recv`, now inside try/finally) | `XMOE_HB=1` |
| deepspeed/sequence/layer.py | +3 / -0 | heartbeat hook around the Ulysses all-to-all | `XMOE_HB=1` |
| Megatron-DeepSpeed-X-MoE/megatron/core/tensor_parallel/data.py | +3 / -0 | heartbeat hook around the TP batch broadcast (where a TP group waits for a stopped member; H16 5641684) | `XMOE_HB=1` |
| Megatron-DeepSpeed-X-MoE/megatron/training.py | +103 / -3 | the heartbeat start; the detectors D3 (save deadline, exit 12), D4/D9 (exception hook: 11/16/17), D7 (loss rule, exit 13); the fault injection; `XMOE_PG_TIMEOUT_ALL_MIN` for Megatron's groups; **training.log**: one line per step written by rank 0 (PP 1) or the last rank, with a MAX all-reduce of one float for the HBM peak (2026-10-09 afternoon); the 3 deleted lines are the save call, now inside the save deadline | `XMOE_HB=1`, `XMOE_DETECT=1`, `XMOE_INJECT*`, `XMOE_PG_TIMEOUT_ALL_MIN`; training.log needs `XMOE_HB=1` and `RES_JOB_DIR` (set by the loop) |
| Megatron-DeepSpeed-X-MoE/examples_elmoe/ELMoE_launch.py | +66 / -0 | the dry run G2: parse the arguments with Megatron's parser, check every path, exit 0 or 2, no GPU | `XMOE_DRY_RUN=1` |
| examples_elmoe/scripts-frontier/local_autorun_frontier_0623_2026.sh | +41 / -4 | the launcher riders: render the resilient template; ask for N + s nodes; adapt the geometry guard to N + s; remove `--finetune` (a restart must be a native resume); `AUTORUN_TEMPLATE`; `AUTORUN_SBATCH_CMD=echo` renders without submitting (the 4 deleted lines are the 4 `sbatch` calls, now `${AUTORUN_SBATCH_CMD:-sbatch}`) | `AUTORUN_RESILIENT=1`, `AUTORUN_SPARES`, `AUTORUN_TEMPLATE`, `AUTORUN_SBATCH_CMD` |
| examples_elmoe/scripts-frontier/monitor_run.sh | +3 / -1 | the monitor's per-node files and tsv go to `MONITOR_DATA_DIR` (the resilient job folder: data/monitor/); a-monitor.txt stays at the top | `MONITOR_DATA_DIR` (unset: `<JOB_DIR>/monitor/`, as before) |
| examples_elmoe/utils/analyze_log.py | +23 / -1 | `rank_log_dir()`: the rank logs from `RANK_LOG_DIR`, else `<dir>/ranks/`, else the last `attempts/attempt_<k>/ranks/`, else the folder itself; the job id from any ancestor `job_<id>` folder | none needed: the flat layout resolves to the folder itself, as before |
| examples_elmoe/utils/analyze_memory.py | +22 / -2 | the same `rank_log_dir()` for the per-rank memory scan | as above |
| examples_elmoe/utils/rerun_analysis.sh | +12 / -3 | the rank logs from `RANK_LOG_DIR` or the last attempt; full_run.log into that attempt's folder | as above |
| examples_elmoe/utils/rerun_analysis.py | +18 / -2 | the same rule for the metrics log | as above |
| examples_elmoe/utils/plot_loss_curve.py | +5 / -2 | the last attempt's full_run.log / ranks/rank_0.log first, then the flat files | as above |
| examples_elmoe/scripts-frontier/frontier_elmoe.slurm_resilient.template | new file; +97 / -22 against the production 0623 template | the training srun as a function of the node list (`--no-kill --kill-on-bad-exit=1 -w <train list>`, MASTER_ADDR per attempt, the rank logs to `${RANK_LOG_DIR:-${JOB_DIR}}`); the dry-run command; `merge_full_run` split out of `run_analysis`, which now reads `RANK_LOG_DIR` / `RUN_LOG_DIR` (else `JOB_DIR`); data/darshan and data/monitor in resilient mode; the block that sources the restart loop; the 22 deleted lines are the rewritten `run_analysis` head, the Darshan path and the monitor echo, each replaced by a flag-aware line | `AUTORUN_RESILIENT=1` (off: the original launch below the block, unchanged) |
| examples_elmoe/scripts-frontier/resilience/ | new folder | bin/: restart_loop.sh 454, hb.py 495, hb_aggregator.py 367, gate_judge.py 302, failure_record.py 153, gate_node.sh 129, results.py 98, gate.sh 83, gate_gemm.py 72, fr_analyze.py 53, exit_codes.py 49, heatmap.py 48, watchdog_culprit.py 29 lines; tests/ (run_matrix.sh, test_aggregator.py, test_hb_unit.py, test_failure_record.sh, test_loop_mock.sh); README.md; failure_record.tsv (the global record); gate_baseline.tsv (the calibration file, see 3); gate_baseline.legacy_20261009.tsv (the old 2-column file, not read) | used only by the resilient template |

Not in the port: `scripts-frontier/results/2026-10-09-results.xlsx` (written by analyze_log.py, data, not code) and the
`temp_ds_json/*_job56*.json` files (the test jobs' DeepSpeed files).

Readers that were NOT changed, on purpose: `utils/replay_log_to_wandb.py` (it takes an explicit `--log` path); the eval
templates under `eval/` (their own jobs, flat layout); `temp_slurm/handoff16n/chain16k.sh` (a finished August chain that reads
flat production folders).

## 2. The exact enabling for the big chain

The exports a run file needs (on top of what the chain's run file already exports):

```bash
export AUTORUN_RESILIENT=1        # the launcher renders frontier_elmoe.slurm_resilient.template
export AUTORUN_SPARES=4           # 4 spare nodes: the job asks Slurm for N + 4 nodes (256 + 4 = 260 for the 256-node link)
export AUTORUN_GATE=full          # or quick: full adds the per-GCD GEMM and the in-node xGMI all-to-all
export XMOE_HB=1                  # the heartbeat thread, the node writer, training.log
export XMOE_DETECT=1              # the detectors D3/D4/D7/D9/D10 (their exit codes drive the restart decision)
# advised, with the numbers of the chain (not measured by this agent for the 671B):
export XMOE_T_INIT_S=<s>          # D8: seconds from the srun start to the first iteration line; the default 600 s is too short
                                  #     for a 671B load at 256 nodes: set it from the chain's measured load + first step + margin
export XMOE_SAVE_DEADLINE_FIRST_S=<s>   # D3 for the first save of each attempt (default 600 s); then 3 x the measured save, >= 120 s
export XMOE_PG_TIMEOUT_MIN=10     # the expert group's watchdog (production default 10 min); 3 in the tests
# optional:
export AUTORUN_GATE_CALIBRATE=1   # only on a job whose nodes are known good: re-measures the gate's baselines and run times
export AUTORUN_GATE_STRICT=1      # stop the job on a TEST DEFECT (default: report it, exclude nobody, continue)
export RES_MAIL_TO=<address>      # mail when the run stops for good (unset: a loud line in recovery.log and the .e)
```

The node count: the run file keeps `N` (the training nodes, e.g. 256); the launcher writes `#SBATCH -N <N + 4>` and adapts the
geometry guard to `N + 4`. The training srun always runs on exactly `N` nodes.

What the resilient template changes in the srun line, in one line: the same command, but launched by the restart loop with
`srun -N <N> -w <the N training nodes of this attempt> --no-kill --kill-on-bad-exit=1`, so a lost node does not end the job,
one failed rank ends the step at once, and the next attempt runs on the good nodes (the spares included) inside the same
allocation; the rank logs go to `attempts/attempt_<k>/ranks/` and MASTER_ADDR/MASTER_PORT are set per attempt.

## 3. The gate's calibration file

`resilience/gate_baseline.tsv`, columns `test baseline run_s date job nodes note`. A test counts (can exclude a node) only with a
line here (rule 1 of design 3.11). The sandbox file was calibrated by the smoke job L1 (see section 4); its versions (ROCm, RCCL,
plugin, libfabric) are those of the env `TORCH-ROCM6.4.1_env_groupgemm_v5`. Port it as is when production runs the same env; a
different env needs one calibration job (`AUTORUN_GATE_CALIBRATE=1`) on known-good nodes before the gate can exclude anything.

## 4. Verification

Smoke L1 = job 5644650 (4+1 nodes, DS-MoE-16B at 4k, 2026-10-09 15:44-16:05): RESULTS.md 4/4 PASS (hang 263 s to detect / 104 s to
recover, exit 8/97, oom 6/103, nan 31/98); the gate full in 140 s calibrated 12 tests (gemm 114.5 TFLOP/s, run 90.5 s; xgmi_a2a
65.9 GB/s, run 32.6 s), excluded none, no TEST DEFECT; the folder is the 3.11 tree; the template's analysis ran on
attempts/attempt_4/ranks. Details: `assistant_deep_dive/2026-10-09-resilience_impl/STATE.md` and `jobs.tsv`.

## 5. Known gaps before the port

- The resilient template reads MASTER_ADDR through ssh to the first training node at every attempt (the design's no-ssh rule is
  not met; lessons.md 2026-10-09 12:15).
- training.log's HBM column is one MAX all-reduce of a float per step, on the world group, inside the logging block where every
  rank already waits at the interval timer's barrier: cheap, but it is one more world collective per step.
- D6 (the replica hash) and the gate's NIC-pair, switch-group, run's-shape and Lustre tests are not built.

## 6. 2026-10-13: two sandbox changes for the DeepSeek-MoE-16B data-quality runs (assistant_deep_dive/2026-10-13-ds16b_data_quality)

Both are additive and flag-gated; with the flags unset the sandbox behaves as before (the two existing mocks and the rule's self-test
pass unchanged: test_oom_mock.sh ALL PASS, oom_reconfig.py --selftest 23/23, tests/test_oom_reconfig.py ALL PASS). Backups:
`*.bak_initload_20261013` next to each file. Production is NOT changed (it is identical to the sandbox in megatron/, deepspeed/, the
launcher and both templates as of 2026-10-13 02:45; only restart_loop.sh and hb.py differed before these changes).

| path | what | the flag that gates it |
|---|---|---|
| scripts-frontier/local_autorun_frontier_0623_2026.sh | rider AUTORUN_INIT_LOAD_DIR=<abs dir> (+ AUTORUN_INIT_UNIVERSAL=1): bakes `export RES_INIT_LOAD_DIR=... RES_INIT_UNIVERSAL=...` after `NODES=` in the rendered script; refuses to submit unless AUTORUN_RESILIENT=1, AUTORUN_LOAD_DIR == AUTORUN_SAVE_DIR, the path is absolute and its latest(_universal) tag names a folder; exclusive with AUTORUN_NATIVE_RESUME / AUTORUN_UCP_FINETUNE / AUTORUN_UCP_RESUME | AUTORUN_INIT_LOAD_DIR |
| resilience/bin/restart_loop.sh | res_init_load (called before the gate's dry run and before every attempt): while the save directory has no 'latest', `--load <init dir> --finetune` (+ `--universal-checkpoint` when RES_INIT_UNIVERSAL=1); once the run's own save exists, `--load <save dir>`, no --finetune, and the universal flag it added is removed unless the OOM rule asked for a universal (RES_UNIV_LOADED_STEP). One sentence per attempt in loop.log | RES_INIT_LOAD_DIR (unset: returns at once) |
| resilience/bin/oom_reconfig.py | Ctx(intra_doc=), CLI `--intra-doc`; infeasible() rejects any Ulysses split (U > 1) when the run uses intra-document attention (megatron/arguments.py:482 asserts ds_sequence_parallel_size == 1 under --intra-document-attention; :681 for the latent attention); the "no split" sentence names it; selftest case 9 | `--intra-doc` (unset: as before) |
| resilience/bin/oom_reconfig.sh | res_mg_args passes `--intra-doc` when megatron_options carries --intra-document-attention; the "no save yet" sentence of the OOM block names the initial checkpoint when RES_INIT_LOAD_DIR is set | the option string / RES_INIT_LOAD_DIR |
| resilience/tests/test_init_load_mock.sh | new mock: 8 mock nodes, the 16B at 262,144, PP1 EP8 TP8+SP, intra-doc: attempt 0 loads the init (universal, --finetune) and OOMs before a save; the rule takes EP 16 alone (no Ulysses); attempt 1 loads the init again under EP 16, saves step 3, stops on a NaN; attempt 2 resumes the save natively (no --finetune, no universal). ALL PASS 2026-10-13 02:57 | - |

Why. (1) The resilient rider removes the template's --finetune so that every restart is a native resume (M1c 5641534); a resilient run could
therefore only start from random weights or from its own save directory, never "weights of run A, iteration 0, saves into run B". The
experiment needs exactly that (lc128k-v2 step 1000 -> ds16b-256k-dq-{new,old}); resuming the counter would make the sampler skip
1000 x 144 samples of the new data. (2) For PP1 EP8 TP8 x U1 with intra-document attention the rule's first rung was PP1 EP16 TP1 x U16, a
layout every rank rejects at argument validation; with the check the ladder is EP16 TP8 (EP doubled alone), then PP2 EP16 TP8, then stop.
Verified on hardware by: the probe DQP (job 5662549, 8 nodes) for attempt 0 of (1); the OOM rungs of (2) are verified only by the mock and
the self-test (no job has run EP16 TP8 or PP2 EP16 TP8 with intra-document attention at 262,144).

## 7. The gradient norm on the single-stage path (2026-10-13, lead)
megatron/training.py train_step: the DeepSpeed non-pipeline branch set grad_norm = None, so PP=1 runs (all DeepSeek-MoE-16B runs, e.g. lc128k-v2 and dq256k-new) never printed "grad norm"; it now reads model[0].get_global_grad_norm() (the engine's cached norm from step(), the one used for clipping). Logging only; backup megatron/training.py.bak_gradnorm_20261013. Takes effect for jobs started after 2026-10-13 ~04:00 (arm OLD onward); arm NEW (5662696) runs without it.

## 8. The run console explains itself (2026-10-13, lead; model-agnostic, from logs only)
- dashboard_html.py: run facts (EP, experts, seq, GBS) re-read until the argument dump exists (was cached as defaults with PP 1: "EP 1"); _gate_details() parses gate.log DETAIL blocks (what each test checks, pass line, yardstick and its calibration job and node count, time allowed, every node not passing with value and reason).
- dashboard_template.html: the machine map wraps above 16 column groups into bands of 12 (72 replicas had zero-width columns); the gate table shows what each test checks, its pass line, the full verdict, the failing nodes, and a how-to-read note (no more 46-character truncation).
- Tested by running each page script under Node with a stub DOM on jobs 5662696 (16B 72x1), 5662549 (16B 8x1) and 5647441 (671B PP8 4x8). Backups *.bak_facts_20261013, *.bak_wrap_20261013, *.bak_gate_20261013.

## 9. The job index follows a run across its jobs (2026-10-13 15:50, sandbox)

The user: "the local host is not updated automatically". The pages were rewritten every 2 minutes, but a browser left on the console of an
ended job (5664539, stopped at step 30) kept reloading a page that never changes, while the run continued in job 5666674.
job_index_watch.py now groups the training jobs by their Slurm job name (one run = one name): run/<name>/ always opens the newest job's
console; the console of every older, ended job of that name carries a banner and moves to the newest job's console after 20 s; the ended
job's own page stays at job_<id>/record.html; the index rows say "continues in <job>" or "follow the run". Model-agnostic (Slurm names and
states only). Backup job_index_watch.py.bak_handover_20261013. Tested in a scratch output folder, then the watcher alone was restarted.

## 10. Console: alarms in their own block; device memory (VRAM) first (2026-10-13 16:08, sandbox)

The user: make the alarms "an independent and scrollable block such that it will always report the alarm", and for memory "I only want
to care about the memory usages shown on the GPU HBM, not how the model allocated ... allocated memory as a dashed line".
- Alarms: a block at the top of "Health and gates", always shown, every line of alarms.log (up to 1,000; it was the last 3 of 20),
  newest first, scrollable (240 px), raised in red, cleared in green, with a count "N raised, M cleared"; also in the script-free view.
- VRAM: dashboard_html.py reads data/monitor/monitor.tsv (monitor_run.sh: rocm-smi --showmeminfo vram, used / total per GCD, whole
  percent, every 60-180 s) incrementally: per rank the latest and the peak since the attempt started (GiB = percent x 64 / 100).
  Vitals "Device memory (VRAM)" (peak, % of 64 GiB, allocated as the note); machine-map mode "VRAM"; per-stage bars = peak VRAM, dashed
  tick = peak allocated over the attempt's heartbeats, grey tick = the memory model's predicted VRAM; node inspector and GCD table VRAM
  first; a "Memory (VRAM)" view in the training chart (VRAM solid, the per-step peak allocated from training.log dashed, 64 GiB red).
- Measured on job 5666674: VRAM 46.7 GiB peak on the fullest GCD vs 30.0 GiB allocated (PyTorch's cache, RCCL buffers, the runtime).
Tested on three run shapes (the live 16B job, the ended 16B job with 104 alarms, the 671B PP8 job 5650829) with a node harness (no script
errors); backups *.bak_vram_20261013. The ended jobs' pages were rebuilt once in the new layout.
