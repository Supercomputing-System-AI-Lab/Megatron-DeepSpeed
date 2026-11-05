# resilience/: the resilient training system (design: records/2026-10/1009_2026_RESILIENCE_DESIGN.md)

Built 2026-10-09 in the sandbox /lustre/orion/gen150/scratch/zixianw4/X-MoE-resilience. Everything is additive and behind flags;
without AUTORUN_RESILIENT / XMOE_HB / XMOE_DETECT / XMOE_INJECT* the tree runs as before.

```
the job (frontier_elmoe.slurm_resilient.template, AUTORUN_RESILIENT=1)        the trainer (megatron/training.py)
  G0 lemons (failure_record.py lemons)                                          H1 hb.py thread: 1 line / 10 s / rank -> /dev/shm
  G1 gate.sh quick|full (dry run, rocm-smi, versions, dmesg, shm, counters,       local rank 0: node writer -> <job>/hb/<node>.tsv
     GEMM, xGMI a2a) -> gate_report.tsv, failed nodes excluded                  hooks: phase + coll_seq (moe_v2 a2a, Ulysses a2a, p2p, TP batch broadcast)
  loop: train = first N good nodes; spares = the rest; duty = spares[0]|batch   D3 save deadline (exit 12)  D4 GPU OOM (11)
        aggregator (hb_aggregator.py) on the duty node                          D7 loss NaN/spike, 3 in a row (13)  D9 exception (16/17)
        srun --no-kill --kill-on-bad-exit=1 -w <train>  (resume from latest)    D10 host memory > 97 % (19)   D5 slow flag in the line
        D8 no iteration line within XMOE_T_INIT_S -> stop the step (18)         injection: XMOE_INJECT / XMOE_INJECT_SCHEDULE
        classify (exit record | watchdog | NODE_FAIL | srun task line) -> name the node (heartbeat minority/stillness, fr_analyze,
        srun, sinfo) -> failure_record.tsv line -> exclude (by class) -> retries per class (2) -> restart
        on the first iteration after a restart: the ##### RECOVERED ... ##### banner at the top of heartbeat.log and a-monitor.txt
```

## Files
- bin/exit_codes.py: the exit-code table (codes 10-21), shared by the trainer (hb.py die) and the job script (`--bash`).
- bin/hb.py: the heartbeat thread, the node writer (`--node-writer`), die() (the detectors' exit path, writes resilience/exit/a<k>_r<r>.json), the excepthook (D4/D9), SaveDeadline (D3), loss_rule (D7), the fault injection.
- bin/hb_aggregator.py: heartbeat.log (header of recoveries + one block per beat + ALARM lines), resilience/agg_last.json, alarm/ (py-spy through srun --overlap), `--compose` (the banner area of heartbeat.log and a-monitor.txt).
- bin/restart_loop.sh: the job script's loop (sourced by the template): lemons, gate, spares, duty hand-over, D8, classification, record, exclusion, retries, banner, retention of saves (the latest two), RESULTS.md.
- bin/failure_record.py: `append` and `lemons --days 30` (hard: node_lost, ecc_uncorrectable, xgmi_error; soft x2 in 30 days; injected lines ignored).
- bin/gate.sh (runs), bin/gate_node.sh (measures: value, run time, status per test), bin/gate_judge.py (judges by the five rules of
  design 3.11: calibration, one verdict per node per test, TEST DEFECT, summary first, exclusion only by a calibrated test with
  "below threshold" or "error"; --calibrate writes gate_baseline.tsv), bin/gate_gemm.py (one torch import per node, one forked
  process per GCD, barrier start, CUDA events). Budget of a test = 2 x its calibrated run time + 30 s.
- bin/watchdog_culprit.py: names the stopped rank of a collective hang from the watchdog lines (the member of the first-firing group of G ranks that did not fire). bin/fr_analyze.py: the flight-recorder reader (copied from 2026-09-10-deepseek_v3/bin); its group ids are per process, so it is trusted for naming only when one rank is absent.
- bin/heatmap.py: heatmap.png (node x local rank, colour = age of the last heartbeat line, letter = phase), drawn only on an alarm
  (the aggregator), at the end of each attempt (the loop) and on demand: `heatmap.py --job-dir <job folder>`.
- bin/results.py: RESULTS.md (one row per injected case: PASS/FAIL, detect and recover seconds, node named, exclusion, evidence).
- bin/oom_reconfig.py (the OOM rule as functions + the memory gate + `--selftest`; the bytes code copied from make_layout_budgets.py), bin/oom_reconfig.sh (the loop's side: res_mem_gate, res_train_launch, res_mem_peaks_line, res_oom_handle), bin/mem_peaks.py (the per-rank peaks of an attempt), bin/memsnap/ (the profiler shim): 2026-10-10, round 6.
- tests/: test_aggregator.py, test_hb_unit.py, test_wall_exit.py, test_gate_judge.py, test_gate_net_unit.py, test_failure_record.sh,
  test_loop_mock.sh, test_ucp_mock.sh, test_oom_mock.sh, test_oom_reconfig.py (login node; the three mocks run the restart loop with a fake srun, a fake trainer
  and a fake converter; test_oom_mock.sh runs the OOM rule through two out-of-memories: before any save, and after a save with the inline conversion);
  run_matrix.sh (the jobs: MODE=matrix|net3|gate4|wall2|ucp2|oom64|hang16|hang32).
- failure_record.tsv (global copy, appended by every resilient job), gate_baseline.tsv (the gate's calibration: test baseline run_s
  date job nodes note), PORT.md (the port list from a real diff and the enabling of the big chain).

## Flags
| flag | meaning |
|---|---|
| AUTORUN_RESILIENT=1 | launcher: render frontier_elmoe.slurm_resilient.template, ask for N + s nodes, adapt the geometry guard |
| AUTORUN_SPARES=auto\|0\|s | auto = ceil(0.015 N) at N >= 32, else 0 |
| AUTORUN_GATE=auto\|full\|quick\|0 | auto = full at N >= 32, else quick |
| XMOE_HB=1, XMOE_DETECT=1 | the heartbeat; the detectors (set by restart_loop.sh by default) |
| XMOE_PG_TIMEOUT_MIN | the timeout of the groups DeepSpeed creates (expert group); unset = torch's 10 min |
| XMOE_T_INIT_S | D8: seconds from the srun start to the first iteration line (default 600) |
| XMOE_SAVE_DEADLINE_S | D3: fixed deadline; else 3 x the measured save (>= 120 s; 60 s in M1d), 600 s for the first save |
| XMOE_INJECT=c:r:s, XMOE_INJECT_SCHEDULE=c:r:s+c:r:s | cases hang, exit, kill, slow, oom, fshang, nan, inithang; each fires once per job (marker files) |
| RES_READMIT_INJECTED=1 | TEST ONLY: when no spare is left, re-admit a node excluded for an injected fault |
| RES_DRYRUN_STRICT=1 | stop the job when the dry run fails (default: warn loudly and continue) |
| AUTORUN_GATE_CALIBRATE=1 | the gate measures every test's baseline and run time on this job's nodes and writes gate_baseline.tsv |
| AUTORUN_GATE_STRICT=1 | stop the job on a TEST DEFECT (default: say it in gate.log, exclude nobody, continue) |
| RES_MAIL_TO | mail address for the stop message (unset: printed loudly in progress.log and the .e) |
| AUTORUN_SBATCH_CMD=echo | launcher: render without submitting |
| AUTORUN_LOCAL_COPIES=0\|1\|2 | the node-local checkpoint copy (design 3.5; 2026-10-09 evening): the async engine (needs DS_ASYNC_PINNED_CHECKPOINT=1) keeps the newest k saves of each rank in /dev/shm/xmoe_ckpt/<job>_<run>/ and loads from there, per file, when the sizes match; data/local_copy.tsv + recovery.log lines; 0 = off |
| (gate full) nic_pair, rdma, switch | the cross-node tests of bin/gate_net.py (one srun step, 8 tasks per node; rows in data/gate/<node>.net.tsv): NIC-to-NIC pair transfer per NIC (host-buffer path), the same with GPU-direct requested (the detail names the path), one all-to-all over all gate nodes with the cxi NIC counters before/after; uncalibrated = informational until a calibration job writes their lines |
| GATE_SWITCH_INIT_DEADLINE_S (120), GATE_READY_WAIT_S (30) | the switch test (2026-10-10): the first all-to-all over the survivors also forms their communicator and has its own budget; the survivors' group is formed only after every survivor wrote a ready file with its torch process-group count and the lowest survivor wrote the final membership; a rank with fewer groups creates one-rank dummy groups first (torch hashes the per-rank group count into the group's name; job 5646502) |
| XMOE_WALL_AWARE_EXIT=1 (+ XMOE_WALL_MARGIN_FRAC, XMOE_WALL_MARGIN_S, XMOE_SAVE_EST_S) | the trainer's exit follows the job's Slurm end time (bin/wall_exit.py; the launcher rider AUTORUN_WALL_AWARE_EXIT=1 bakes the export); since 2026-10-10 02:30 the exit is HARD: after the save is published on every rank and one confirming all-reduce, every rank flushes its logs, stops the async engine's writer and calls os._exit(0) (W5 5646576: srun returned 0 24 s after the exit sentence; the sys.exit version hung 46 s in torch's communicator teardown, W4 5646541) |
| AUTORUN_UCP_WORKERS=<n> (0) | the UCP worker duty (2026-10-10, the user's 01:15 decision): n extra nodes at the END of the allocation (after the spares) never train and convert every complete save to the universal form inside the job (restart_loop.sh res_ucp_*: ds_to_universal with n byte-balanced shares, --fragments-inplace --optstate-from-pickle, 64/64 workers; one conversion at a time, a save completed meanwhile is skipped and caught up later; a failed share is retried once on another worker; 'latest_universal' advances only forward); the retention keeps the newest 2 ZeRO saves + the newest 8 universals + the steps of RES_KEEP_FILE (<save dir>/keep_steps.txt) and deletes an older ZeRO save only when its universal is complete; STATUS.md shows the 'conversion' line (data/ucp_state); the record is data/ucp_conversions.tsv, the logs data/ucp/; RES_UCP_BALANCED=1 adds --param-shard-balanced, RES_UCP_EXTRACT_WORKERS / RES_UCP_MERGE_WORKERS (64) |
| AUTORUN_OOM_RECONFIG=1 (0) | the OOM re-configuration rule (2026-10-10, the user's three layers of choice; bin/oom_reconfig.py + bin/oom_reconfig.sh): upon exit code 11 the loop takes ONE rung of the fixed ladder per out of memory, never a sweep: step 0 mbs > 1 -> 1 (the accumulation doubled, the same save loads); step 1 d = TP x U aligned to EP with the TP/U split of the fewest communication bytes (the layout budget code, among the splits that fit); step 2 EP and d doubled together (EP alone when the attention heads bound d: the 16B's 16 heads cap d at 16); step 3 PP doubled, EP and d kept; after the last rung the run stops with mail. Every layout change loads the UNIVERSAL checkpoint of 'latest' (the loop waits for the UCP workers' conversion, or converts on the batch node when the job has no workers; no save yet = the initial state with the new layout) by appending --universal-checkpoint and pointing --load at the save directory; the attempt's flags are re-rendered in megatron_options / deepspeed_options / the DeepSpeed json (TP, Ulysses, EP, PP, mbs; --sequence-parallel + XMOE_MEGATRON_SP=1 whenever TP > 1); the decision is one full-sentence block in recovery.log and the banner names the new layout; the record is data/oom_reconfig.tsv + data/attempt_layouts.tsv; RESULTS.md gets the section with the pass checks |
| AUTORUN_MEM_GATE=1 (on with the rule) | the memory gate before the first attempt: the memory model's peak per stage (utils/v3_memory_model.py) and the layout heatmap's colour (records/2026-10/1005_2026_LAYOUT_BUDGETS_rows.json when the cell is drawn, else the same rule at the run's own tokens per step) into gate.log; a red layout, or one above the out-of-memory line (allocated + 16 GiB + non-torch against 64), gets a WARNING and the run proceeds; within 4 GiB of the line = at the margin; never a refusal |
| AUTORUN_MEM_PROFILER=1 (0) | the memory profiler on every attempt: bin/memsnap/python (the shim of 2026-09-14 pointed at the v5 env) wraps every rank's python, so the peak-memory counters are never reset and a thread writes each rank's peak to attempts/attempt_<k>/memsnap/peak_rank<r>.txt every 30 s; after every attempt bin/mem_peaks.py folds the per-rank peaks per stage (also from the rank logs' 'max allocated' lines and the exit records) into recovery.log next to the model's prediction and data/mem_peaks_a<k>.tsv; AUTORUN_MEMSNAP_RANKS=r,r records the allocator history on those ranks (default none) |
| AUTORUN_INIT_LOAD_DIR=<abs dir> (+ AUTORUN_INIT_UNIVERSAL=1) | 2026-10-13 (PORT.md section 6): the run STARTS from another run's checkpoint, weights only (--finetune, iteration 0; the universal form loads under any layout), until its own first save exists in AUTORUN_SAVE_DIR (= AUTORUN_LOAD_DIR); then every restart resumes that save natively (restart_loop.sh res_init_load). With intra-document attention in the options the OOM rule never chooses Ulysses (oom_reconfig.py --intra-doc) |

## Where to read a job (logs/job_<id>_*/; design 3.11, 2026-10-09 afternoon)
```
README.md STATUS.md training.log recovery.log heartbeat.log gate.log heatmap.png <job>.o <job>.e a-monitor.txt RESULTS.md
attempts/attempt_<k>/{full_run.log, ranks/rank_<r>.log, nccl_trace/, exit/}      attempt 0 = the first run
data/{attempts.tsv failures.tsv gate_report.tsv failure_record.tsv results.tsv state loop.log save_times.tsv banners.txt
      heartbeat_body.log agg_last.json, hb/, alarm/, monitor/, darshan/, inject/, gate/}
```
The rank logs are written directly into the current attempt's ranks/ (RANK_LOG_DIR, set by the loop per attempt); the readers
(run_analysis, analyze_log.py, analyze_memory.py, rerun_analysis.*, plot_loss_curve.py) read RANK_LOG_DIR when set, else the flat
JOB_DIR.

## The aggregator's two rules of 2026-10-10 (hb_aggregator.py)
- THE DELIVERY LAG: when every rank of a node is equally old (within 3 s) and older than the threshold, and the node file's own
  modification time is as old as its lines, the lines were late on the path from the node to the aggregator (the 260-node job 5646348:
  195 ranks "32 s old" for one beat, fresh two seconds later); the beat block says "delivery lag N s on M nodes (Lustre), not a rank",
  those ranks are not named by the stillness or the minority rule, and no stack sample is taken; after 3 beats the line says it looks like
  a dead writer or a lost node. A lag needs at least one fresh rank somewhere: when every node is stale at once, the plain rules apply.
- THE TWO-BEAT RULE: the stillness rule names a rank only when it is older than the threshold on two consecutive beats; the first beat is
  "stillness pending" in the beat block. The save-window rule (120 s) is unchanged.

## Naming the node of a collective hang (in this order)
1. the heartbeat minority rule (the aggregator's last table): in a group of G consecutive ranks where nobody moved for 30 s (120 s
   inside a save), the ranks behind the group's collective count; 2. the watchdog-group rule (watchdog_culprit.py); 3. the flight
   recorder when it names exactly one absent rank; 4. srun's first failed task line; 5. sinfo (a DOWN/DRAIN node).

## Known limits (2026-10-09)
- D7 acts on the step's reported loss: the injection 'nan' multiplies the value the rule reads, not the gradients; a real NaN
  gradient is applied by the optimizer before the rule sees the loss (the skip would need the check between backward and step).
- D3 kills a save that is later than max(120 s, 3 x the measured save); in M1d (60-s floor) two real Lustre stalls of a single
  rank (state D) at a save cadence of 215 GB every ~20 s were classified filesystem_hang.
- D6 (replica hash) and the G3/NIC/group/Lustre gate tests are not built.
- The slow-rank rule (busy > 1.3 x stage mates) named the slowed rank first (2.2x) and three of its node mates at 1.4x.
- RES_READMIT_INJECTED=1 is for tests only: it returns a node excluded for an injected fault when no spare is left.
