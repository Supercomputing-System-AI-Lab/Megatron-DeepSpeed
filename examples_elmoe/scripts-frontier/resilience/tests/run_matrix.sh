#!/bin/bash
# run_matrix.sh: the resilience test jobs (2026-10-09). One job per call; MODE picks the configuration:
#   MODE=matrix  4+1 nodes, DeepSeek-MoE-16B (registry DeepSeek_16B), seq 4,096, a chained schedule of injected faults
#                (2026-10-09 afternoon, the layout smoke L1: hang+exit+oom+nan, gate full WITH calibration, 24 iterations, wall 40 min)
#   MODE=hang16  16+2 nodes, seq 65,536, one injected hang          MODE=hang32  32+4 nodes, seq 65,536, one injected hang
#   MODE=net3    2+1 nodes, DeepSeek_16B_4L, seq 4,096, 24 iterations, gate FULL + calibration with the three cross-node tests
#                (nic_pair, rdma, switch), the four L1 faults, and the node-local checkpoint copy (AUTORUN_LOCAL_COPIES=1); 2026-10-09 evening
#   MODE=gate4   2+2 nodes, DeepSeek_16B_4L, seq 4,096, 24 iterations, gate FULL + calibration (the new pair test calibrates itself), TWO
#                injected GATE faults through XMOE_GATE_INJECT ("pair_silent:@1": one rank of the second node never joins its all-reduce
#                in either round; "gemm_half:@2": one GCD of the third node runs the GEMM at half speed), the training faults
#                exit:9:5 (rank_died, excludes its node) + oom:2:7 + nan:0:9, the node-local copy; 2026-10-09 late evening.
#                THE QUESTION (gate4): is the silent node excluded as DEAD by the pair test's naming rule (named by two different
#                partners), the half-speed node excluded as DEGRADED (below the GEMM threshold) and kept as the spare of last resort,
#                both written into the failure record with the classes gate_pair / gate_gemm; does the spare check run on the healthy
#                spare before it replaces the dead node; does training start on the two good nodes with no healthy spare left; and
#                when the injected rank_died excludes a training node, is the degraded node put into training as the last resort
#                (with the sentence in recovery.log and the banner), so that the oom and nan recoveries follow and the job reaches
#                step 24? The majority rule (a timeout while the others passed) is covered by tests/test_gate_judge.py, not by this job.
#   MODE=wall2   2+0 nodes, DeepSeek_16B_4L, seq 4,096, ITERS 1000, wall 15 min, NO injections, AUTORUN_WALL_AWARE_EXIT=1: the wall-aware
#                exit test of 2026-10-10 (the trainer must exit by itself with code 0 before the wall, after a complete save); see the block.
#   MODE=netlaunch3  2+1 nodes, the full pre-run tests through the new network-launch retry, no faults; 2026-10-11; see the block.
#   MODE=hostoom3  2+1 nodes, DeepSeek_16B_4L, seq 4,096, a REAL host OOM injected on rank 9 at step 6 (realhostoom); 2026-10-11; see the block.
#   MODE=ucp2    2 train + 0 spares + 2 UCP WORKER nodes = 4 nodes, DeepSeek_16B_4L, seq 4,096, save every 4 steps, 24 iterations, wall 40 min,
#                NO injections, AUTORUN_UCP_WORKERS=2: the UCP worker duty test of 2026-10-10 (every complete save converted to the universal
#                form inside the job by the two worker nodes, one conversion at a time; the retention by universals); see the block.
#                LAYOUT (gate4, as net3): nodes 2 | EP 8 | PP 1 | TP 1 | DP_non_moe 16 | DP_moe 2 | mbs 1 x nbs 1 x 16 = GBS 16 | seq 4,096 |
#                ckpt on | ZeRO 1 | 4 layers | tokens per node 32,768 | + 2 spares = 4 nodes | wall 40 min, QOS normal (never debug).
# THE QUESTION (matrix): does every injected fault end in the right class, the right node named (and excluded when the class
# excludes), a restart inside the same allocation from the latest save, and a first iteration after it; with the time to detect
# (injection marker -> srun return) and to recover (srun return -> first iteration) measured. Threshold: PASS per case in RESULTS.md.
# THE QUESTION (hang16/hang32): the same for one hang at 16+2 / 32+4 nodes at 64k, with the duty on a real spare and the hang
# detected by the 3-minute expert-group timeout (XMOE_PG_TIMEOUT_MIN=3) and named by the heartbeat minority rule / the recorder.
# LAYOUTS (frontier-job-submission-principles B):
#   matrix: nodes 4 | EP 8 | PP 1 | TP 1 | DP_non_moe 32 | DP_moe 4 | mbs 1 x nbs 1 x 32 = GBS 32 | seq 4,096 | ckpt on (1 layer) |
#           ZeRO 1 | tokens per node 32,768 | save every 2 steps | random init (load = save dir, empty at the start)
#   hang16: nodes 16 | EP 8 | PP 1 | TP 8 + Megatron SP (rule f/h: long context; the 16B's 128k layout of 2026-10-01, P128s 19.8 GB at
#           128k mbs 1) | DP_non_moe 16 | DP_moe 16 | mbs 1 x nbs 1 x 16 = GBS 16 | seq 65,536 | ckpt on | chunked CE 2048 | ZeRO 1
#   hang32: the same on 32 nodes: DP_non_moe 32, DP_moe 32, GBS 32
# EXPECTED TIMELINE (matrix; estimates, not measurements): gate quick 1-3 min; each attempt's start-up (python, build, load) 2-5 min;
# a step ~2-5 s plus a save every 2 steps (~230 GB of 16B model + Adam state: 20-60 s); the hang costs the 3-min expert-group timeout,
# the save hang 3x the measured save (>= 60 s), exit/oom/nan seconds; 5 restarts x (start-up + detect) ~ 25-35 min: the six cases
# may not all fit the 45-min wall; RESULTS.md lists the cases that did not fire. D8 (no first iteration in 600 s) stops a stuck start.
# SIZE: batch partition, QOS normal (never debug: the 256-node job holds it), wall 45 min (matrix) / 40 min (hang*).
# The sandbox only: PYTHONPATH puts /lustre/orion/gen150/scratch/zixianw4/X-MoE-resilience first; the job asserts it.
set -euo pipefail
MODE=${MODE:-matrix}
# [2026-10-11] TREE=prod runs the test on the PRODUCTION tree (the port of 2026-10-11); default: the sandbox
if [ "${TREE:-sandbox}" = prod ]; then SB=/lustre/orion/gen150/scratch/zixianw4/X-MoE; else SB=/lustre/orion/gen150/scratch/zixianw4/X-MoE-resilience; fi
SF=$SB/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier
SCOPE=/lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/assistant_deep_dive/2026-10-09-resilience_impl
cd $SF
set +e +u; . ./eval/submit_env.sh; set -euo pipefail
export PATH=/lustre/orion/world-shared/gen150/zixianw4/envs/TORCH-ROCM6.4.1_env_groupgemm_v5/bin:$PATH
export CXX=$(which g++) CC=$(which gcc)
export LD_LIBRARY_PATH=/lustre/orion/gen150/scratch/zixianw4/aws-ofi-rccl-rocm6.4/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$SB:$SB/Megatron-DeepSpeed-X-MoE        # the tree under test ONLY (TREE=prod: production; else the sandbox)
python -c "import deepspeed, megatron, sys; ok = all(m.__file__.startswith('$SB/') for m in (deepspeed, megatron)); print('[submit shell] deepspeed', deepspeed.__file__, '| megatron', megatron.__file__); sys.exit(0 if ok else 1)" \
  || { echo "FATAL: deepspeed/megatron do not import from $SB"; exit 1; }

export AUTORUN_RESILIENT=1 XMOE_HB=1 XMOE_DETECT=1 XMOE_PG_TIMEOUT_MIN=${PG_MIN:-3} XMOE_T_INIT_S=${XMOE_T_INIT_S:-600}
export AUTORUN_QOS=normal AUTORUN_NO_PROFILING=1 AUTORUN_DS_CONFIG_PER_JOB=1 AUTORUN_NCCL_DEBUG=INFO
export AUTORUN_SAVE_INTERVAL=2
case $MODE in
  matrix)
    LABEL=${LABEL:-M1}
    export AUTORUN_EP_KEY="4:8:1" AUTORUN_EP_CMD=" 1:1:${ITERS:-40}:X-MOE:DeepSeek_16B:ckpt:1:1" AUTORUN_SEQ_LEN=4096
    export AUTORUN_SPARES=1 AUTORUN_GATE=${GATE:-quick} AUTORUN_WALLTIME="${WALL:-00:45:00}" AUTORUN_EXIT_MINS=${EXIT_MINS:-44}
    export AUTORUN_GATE_CALIBRATE=${CALIBRATE:-0}     # 1: the gate measures the baselines and run times on these nodes (rule 1 of 3.11)
    export XMOE_INJECT_SCHEDULE="${SCHEDULE:-hang:5:3+exit:9:6+oom:2:9+fshang:4:12+nan:0:15+slow:7:18}"
    export RES_READMIT_INJECTED=1   # TEST ONLY: 1 spare for 3 node-excluding injected faults; an injected-fault node is healthy
    ;;
  net3)
    # 2026-10-09 evening: 2 train + 1 spare = 3 nodes, DeepSeek_16B_4L (the 4-layer registry entry), seq 4,096, 24 iterations, save
    # every 2, wall 40 min, gate FULL with CALIBRATION (the three new cross-node tests nic_pair / rdma / switch calibrate themselves
    # on this run), the four injected faults of L1, and the node-local checkpoint copy (AUTORUN_LOCAL_COPIES=1 on the async engine).
    # THE QUESTION: do the three network tests measure and calibrate on 3 nodes inside the gate's budget, and does a restart on the
    # same nodes load its files from /dev/shm (time vs Lustre measured in data/local_copy.tsv) while the four cases still PASS?
    # LAYOUT: nodes 2 | EP 8 | PP 1 | TP 1 | DP_non_moe 16 | DP_moe 2 | mbs 1 x nbs 1 x 16 = GBS 16 | seq 4,096 | ckpt on | ZeRO 1 |
    #   4 layers | tokens per node 32,768 | + 1 spare = 3 nodes. Reference: the production twin P1 5645490 (14 min with gate quick).
    LABEL=${LABEL:-N1}
    export AUTORUN_EP_KEY="${N_TRAIN:-2}:8:1" AUTORUN_EP_CMD=" 1:1:${ITERS:-24}:X-MOE:${MODEL:-DeepSeek_16B_4L}:ckpt:1:1" AUTORUN_SEQ_LEN=${SEQ:-4096}
    export AUTORUN_SPARES=${SPARES:-1} AUTORUN_GATE=${GATE:-full} AUTORUN_WALLTIME="${WALL:-00:40:00}" AUTORUN_EXIT_MINS=${EXIT_MINS:-38}
    export AUTORUN_GATE_CALIBRATE=${CALIBRATE:-1}
    export XMOE_INJECT_SCHEDULE="${SCHEDULE:-hang:5:3+exit:9:5+oom:2:7+nan:0:9}"
    export RES_READMIT_INJECTED=${READMIT:-1}   # TEST ONLY: 1 spare for 2 node-excluding injected faults; an injected-fault node is healthy
    export AUTORUN_LOCAL_COPIES=${LOCAL_COPIES:-1} DS_ASYNC_PINNED_CHECKPOINT=${ASYNC_CKPT:-1}
    ;;
  gate4)
    # 2026-10-09 late evening: see the header. 2 train + 2 spares = 4 nodes; the gate's injections name the nodes by their index in the
    # allocation ('@1' = the second node, '@2' = the third; restart_loop.sh resolves them). RES_READMIT_INJECTED=0: the rank_died node
    # stays excluded, so the loop must fall back to the degraded node (AUTORUN_USE_DEGRADED=1, the default).
    LABEL=${LABEL:-G4}
    export AUTORUN_EP_KEY="${N_TRAIN:-2}:8:1" AUTORUN_EP_CMD=" 1:1:${ITERS:-24}:X-MOE:${MODEL:-DeepSeek_16B_4L}:ckpt:1:1" AUTORUN_SEQ_LEN=${SEQ:-4096}
    export AUTORUN_SPARES=${SPARES:-2} AUTORUN_GATE=${GATE:-full} AUTORUN_WALLTIME="${WALL:-00:40:00}" AUTORUN_EXIT_MINS=${EXIT_MINS:-38}
    export AUTORUN_GATE_CALIBRATE=${CALIBRATE:-1}
    export XMOE_INJECT_SCHEDULE="${SCHEDULE:-exit:9:5+oom:2:7+nan:0:9}"
    export XMOE_GATE_INJECT="${GATE_INJECT:-pair_silent:@1+gemm_half:@2}"
    export RES_READMIT_INJECTED=${READMIT:-0}
    export AUTORUN_USE_DEGRADED=${USE_DEGRADED:-1}
    export AUTORUN_LOCAL_COPIES=${LOCAL_COPIES:-1} DS_ASYNC_PINNED_CHECKPOINT=${ASYNC_CKPT:-1}
    ;;
  wall2)
    # 2026-10-10: THE WALL-AWARE EXIT TEST. 2 training nodes + 0 spares = 2 nodes (my choice: the test has no injections, so a spare
    # would only idle; the duty runs on the batch node), DeepSeek_16B_4L, seq 4,096, EP 8, save every 2 steps (the matrix default),
    # ITERS 1000 so that ONLY the wall ends the run, wall 15 min, gate quick, the asynchronous checkpoint engine, and
    # AUTORUN_WALL_AWARE_EXIT=1 (-> XMOE_WALL_AWARE_EXIT=1 in the job). AUTORUN_EXIT_MINS=14 stays as the FALLBACK end time only.
    # THE QUESTION: does the trainer exit by itself with code 0 before the wall, at the last step whose successor (step x 1.1 + the
    # measured save + 120 s) could not have finished before SLURM_JOB_END_TIME; does 'latest' name that step with the engine's
    # "complete on all 16 ranks" line; does the planned-exit sentence with the arithmetic appear in rank_0.log, training.log, loop.log;
    # does the job end COMPLETED well before its wall without the SIGUSR1 handler firing; do RESULTS.md and STATUS.md say so?
    # RECORD from the job folder: the measured step time and save time (data/save_times.tsv, data/save_async_times.tsv), the remaining
    # seconds at the decision (data/wall_exit.tsv), the srun return time against the end time (loop.log), the job's end (sacct).
    # LAYOUT: nodes 2 | EP 8 | PP 1 | TP 1 | DP_non_moe 16 | DP_moe 2 | mbs 1 x nbs 1 x 16 = GBS 16 | seq 4,096 | ckpt on | ZeRO 1 |
    #   4 layers | tokens per node 32,768 | 0 spares | wall 15 min, QOS normal.
    LABEL=${LABEL:-W2}
    export AUTORUN_EP_KEY="${N_TRAIN:-2}:8:1" AUTORUN_EP_CMD=" 1:1:${ITERS:-1000}:X-MOE:${MODEL:-DeepSeek_16B_4L}:ckpt:1:1" AUTORUN_SEQ_LEN=${SEQ:-4096}
    export AUTORUN_SPARES=${SPARES:-0} AUTORUN_GATE=${GATE:-quick} AUTORUN_WALLTIME="${WALL:-00:15:00}" AUTORUN_EXIT_MINS=${EXIT_MINS:-14}
    export AUTORUN_GATE_CALIBRATE=${CALIBRATE:-0}
    export XMOE_INJECT_SCHEDULE=""                   # no injected faults: only the wall ends this run
    export AUTORUN_WALL_AWARE_EXIT=1 XMOE_WALL_MARGIN_FRAC=${MARGIN_FRAC:-0.1} XMOE_WALL_MARGIN_S=${MARGIN_S:-240} XMOE_SAVE_EST_S=${SAVE_EST:-120}
    export DS_ASYNC_PINNED_CHECKPOINT=${ASYNC_CKPT:-1} AUTORUN_LOCAL_COPIES=${LOCAL_COPIES:-0}
    ;;
  ucp2)
    # 2026-10-10 (the user, 01:15): THE UCP WORKER DUTY TEST. 2 training nodes + 0 spares + 2 UCP worker nodes = 4 nodes (the workers are the
    # last two nodes of the allocation; they never train), DeepSeek_16B_4L, seq 4,096, EP 8, save every 4 steps, 24 iterations (saves at 4, 8,
    # 12, 16, 20, 24), wall 40 min, gate quick, the asynchronous checkpoint engine (so that 'latest' names only complete saves), no
    # injections, no wall-aware exit (the run ends by its iteration count; AUTORUN_EXIT_MINS=38 stays the fallback).
    # THE QUESTION: do the two worker nodes convert at least two complete saves to the universal form inside the job (ds_to_universal with
    # two byte-balanced shares, --fragments-inplace --optstate-from-pickle, 64/64 workers; complete = atoms > 0, zero/optimizer_state.pt,
    # no tmp folder, the finalize lock, 'latest_universal' advanced), one conversion at a time with the skipped saves said and caught up;
    # does the retention delete the oldest ZeRO save once its universal is complete (the newest 2 ZeRO saves and the newest 8 universals
    # stay); does STATUS.md carry the conversion line; and is the training unaffected (the step time of the steps without a save within
    # the 0.44-0.55 s of the earlier 16B_4L runs)? RECORD: data/ucp_conversions.tsv (the measured conversion time inside the job),
    # data/ucp/convert_s<N>_share<k>.log, loop.log, recovery.log, STATUS.md, RESULTS.md (the UCP section).
    # LAYOUT: nodes 2 | EP 8 | PP 1 | TP 1 | DP_non_moe 16 | DP_moe 2 | mbs 1 x nbs 1 x 16 = GBS 16 | seq 4,096 | ckpt on | ZeRO 1 |
    #   4 layers | tokens per node 32,768 | 0 spares + 2 UCP workers = 4 nodes | wall 40 min, QOS normal.
    LABEL=${LABEL:-U1}
    export AUTORUN_EP_KEY="${N_TRAIN:-2}:8:1" AUTORUN_EP_CMD=" 1:1:${ITERS:-24}:X-MOE:${MODEL:-DeepSeek_16B_4L}:ckpt:1:1" AUTORUN_SEQ_LEN=${SEQ:-4096}
    export AUTORUN_SPARES=${SPARES:-0} AUTORUN_GATE=${GATE:-quick} AUTORUN_WALLTIME="${WALL:-00:40:00}" AUTORUN_EXIT_MINS=${EXIT_MINS:-38}
    export AUTORUN_GATE_CALIBRATE=${CALIBRATE:-0}
    export XMOE_INJECT_SCHEDULE=""                   # no injected faults
    export AUTORUN_UCP_WORKERS=${UCP_WORKERS:-2} RES_UCP_BALANCED=${UCP_BALANCED:-1}
    export AUTORUN_SAVE_INTERVAL=${SAVE_EVERY:-4}
    export DS_ASYNC_PINNED_CHECKPOINT=${ASYNC_CKPT:-1} AUTORUN_LOCAL_COPIES=${LOCAL_COPIES:-0}
    ;;
  dash2)
    # 2026-10-10 afternoon: THE RUN CONSOLE TEST (the user: "in the sand box, please have the localhost html page rendered by default and make sure
    # that this does not interfere with the training process"; and "all the jobs show forward with no backward ... please revise if needed in the
    # sandbox"; and the training.log column-name row). 2 training nodes + 1 spare = 3 nodes (the spare holds the duty, so the heartbeat reader and
    # the page builder run on a node that never trains: the production placement), DeepSeek_16B_4L, seq 4,096, EP 8, PP 2 (the pipeline engine is
    # where backward was never marked: at PP 1 the training loop marks it already, jobs 5647555/5647611), 4 micro-batches per step, 300 iterations,
    # save every 100, gate quick, no injections, the default AUTORUN_DASHBOARD=1 (every 30 s on the duty spare).
    # THE QUESTION: during a real run, does the heartbeat reader rewrite <job>/dashboard.html every ~30 s (the file's modification time advances
    # at least 10 times; data/dashboard.err absent) while (1) the steps that overlap a page rewrite (data/dashboard_times.tsv against the step
    # times in the last rank's log) have a median step time within 5% of the other steps without a save of the same job (the paired comparison:
    # no second job), (2) the reader's beats stay on their 10-s cadence (no beat gap over 15 s in heartbeat.log), (3) the heartbeat lines carry the
    # phases forward, backward AND step (the sandbox pipe-engine marks), and (4) training.log carries the column-name row before the first step
    # line and every 25 step lines? PASS = all four plus the page's own checks (no script error, no sideways scroll; pagecheck.py on the login node).
    # LAYOUT: nodes 2 | EP 8 | PP 2 | TP 1 | DP_non_moe 8 | DP_moe 1 | mbs 1 x nbs 4 x 8 = GBS 32 | seq 4,096 | ckpt on | ZeRO 1 | 4 layers (2 per
    #   stage) | tokens per node 16,384 per micro-batch | + 1 spare = 3 nodes | wall 30 min, QOS debug (the first run of a feature: C1).
    LABEL=${LABEL:-D1}
    export AUTORUN_EP_KEY="${N_TRAIN:-2}:8:1" AUTORUN_EP_CMD=" 1:${NBS0:-4}:${ITERS:-300}:X-MOE:${MODEL:-DeepSeek_16B_4L}:ckpt:1:1" AUTORUN_SEQ_LEN=${SEQ:-4096}
    export AUTORUN_SPARES=${SPARES:-1} AUTORUN_GATE=${GATE:-quick} AUTORUN_WALLTIME="${WALL:-00:30:00}" AUTORUN_EXIT_MINS=${EXIT_MINS:-28}
    export AUTORUN_GATE_CALIBRATE=${CALIBRATE:-0}
    export AUTORUN_QOS=${QOS:-debug}
    export XMOE_INJECT_SCHEDULE=""                   # no injected faults
    export AUTORUN_PP_SIZE=${PP0:-2}
    export AUTORUN_SAVE_INTERVAL=${SAVE_EVERY:-100}
    export AUTORUN_DASHBOARD=${DASHBOARD:-1}
    export DS_ASYNC_PINNED_CHECKPOINT=${ASYNC_CKPT:-1} AUTORUN_LOCAL_COPIES=${LOCAL_COPIES:-0}
    ;;
  oom64)
    # 2026-10-10 (round 6): THE OOM RE-CONFIGURATION TEST. DeepSeek-MoE-16B (the registry's 28-layer DeepSeek_16B) at seq 131,072, chunked CE 2,048, NO
    # intra-document attention (the 16B's intra-document mask does not support Ulysses, job 5579534), token dropping on (the user's 2026-09-18 rule;
    # the memory gate takes the capacity 1.25 from the flags), 64 training nodes + 1 spare + 2 UCP workers = 67 nodes, save every step, 24 iterations or the wall (2 h, QOS
    # normal), gate full with the network tests, AUTORUN_OOM_RECONFIG=1, AUTORUN_MEM_PROFILER=1, the wall-aware exit.
    # THE LAYOUT: PP2 EP16 TP1 x U1 mbs 1 (d = 1; GBS = 1 x 1 x 256 replicas = 256 sequences = 33.5M tokens per step). WHY d = 1 AND NOT THE d = 2 I WAS
    # ASKED FOR: the published heatmap (records/2026-10/1005_2026_LAYOUT_BUDGETS_rows.json, the 16B at 64 nodes, 128k) has NO colour for the PP2 EP16
    # d2 cell (it reads 'batch': its 144 sequences per step do not split over 128 replicas) nor for d1; the memory model at this run's own tokens
    # per step puts d2 at 41 GiB allocated on the heaviest stage (green by the 16B rule: the out-of-memory line 61 of 64), so d2 would most likely
    # NOT fail and the rule would never fire; d1 is red for certain (stage 1: 77.6 GiB allocated, 95.4 VRAM, the line 97 against 64). The step-1
    # target is the same family: d aligned to EP 16 with the fewest-bytes split, which for the 16B is TP1 x U16 (123.5 GiB per step against 151.7
    # for TP2 x U8: the across-nodes bytes tie at 66.1, the total decides).
    # THE QUESTION: does gate.log carry the memory gate's WARNING for the red layout and does the run proceed; does attempt 0 end with exit code 11;
    # does recovery.log carry the rule's full-sentence block (the OOM step, stage and HBM, the rung taken, the split's bytes against the runner-up, the
    # model's prediction, the checkpoint loaded); does the next attempt run with PP2 EP16 TP1 x U16 (the re-rendered flags in attempts/attempt_1's
    # command line), loading the universal checkpoint that the 2 workers converted when a save existed, or the initial state when the OOM came before
    # the first save (the case is said in recovery.log); does it reach at least 2 steps with the profiler's peak under 64 GiB; and does RESULTS.md's
    # OOM re-configuration section show the layouts, the predicted and measured peaks and the times?
    # PASS THRESHOLD (written before the run): all five checks of RESULTS.md's "The pass checks" list PASS; the measured peak of attempt 1 under the
    # model's out-of-memory line for its layout (29.4 GiB) + 10 GiB would be a model miss to report, not a failure of the rule.
    # EXPECTED TIMELINE: gate full ~3-5 min on 64 nodes; attempt 0 reaches the first forward and OOMs within ~5-10 min of the srun start (no save,
    # so the restart loads the initial state); attempt 1 start-up ~5 min; a 33.5M-token step at ~20-30 TFLOP/s per GCD ~2-3 min plus a ~230 GB save
    # every step (~30-60 s); 24 steps may not fit the wall: the wall-aware exit ends it cleanly.
    LABEL=${LABEL:-O1}
    export AUTORUN_EP_KEY="${N_TRAIN:-64}:${EP0:-16}:${TP0:-1}" AUTORUN_EP_CMD=" ${MBS0:-1}:${NBS0:-1}:${ITERS:-24}:X-MOE:${MODEL:-DeepSeek_16B}:ckpt:1:1" AUTORUN_SEQ_LEN=${SEQ:-131072}
    export AUTORUN_PP_SIZE=${PP0:-2} AUTORUN_SEQ_CHUNKED_CE=${CCE:-2048}
    [ "${U0:-1}" -gt 1 ] && export AUTORUN_DS_SEQUENCE_PARALLEL_SIZE=${U0}
    [ "${TP0:-1}" -gt 1 ] && export AUTORUN_MEGATRON_SP=1
    export MOE_DROP_TOKEN=${DROP:-true}             # token dropping ON (the user's 2026-09-18 directive for all runs; the heatmap's recipe was dropless):
                                                  # the memory gate reads the capacity (1.25) from the rendered flags, so its numbers are the run's
    export PG_MIN=${PG_MIN:-10}                      # the expert-group timeout stays at torch's 10 min: no hang is injected, and a cold Triton compile at 128k must not trip a 3-min watchdog
    export AUTORUN_SPARES=${SPARES:-1} AUTORUN_GATE=${GATE:-full} AUTORUN_WALLTIME="${WALL:-02:00:00}" AUTORUN_EXIT_MINS=${EXIT_MINS:-115}
    export AUTORUN_GATE_CALIBRATE=${CALIBRATE:-0}
    export XMOE_INJECT_SCHEDULE="${SCHEDULE:-}"     # default: inject nothing (the out of memory must be the layout's own); O2 (2026-10-10) injects oom:<rank>:<step> AFTER saves, to test the universal reload under the new layout
    export AUTORUN_UCP_WORKERS=${UCP_WORKERS:-2} RES_UCP_BALANCED=${UCP_BALANCED:-1}
    export AUTORUN_SAVE_INTERVAL=${SAVE_EVERY:-1}
    export AUTORUN_OOM_RECONFIG=${OOM_RECONFIG:-1} AUTORUN_MEM_PROFILER=${MEM_PROFILER:-1} AUTORUN_MEM_GATE=1
    export AUTORUN_WALL_AWARE_EXIT=${WALL_AWARE:-1} XMOE_WALL_MARGIN_FRAC=${MARGIN_FRAC:-0.1} XMOE_WALL_MARGIN_S=${MARGIN_S:-300} XMOE_SAVE_EST_S=${SAVE_EST:-120}
    export XMOE_T_INIT_S=${XMOE_T_INIT_S:-1500}       # D8: the first iteration line at 128k on 64 nodes (the data index, the Triton warm-up, the first step ~3 min)
    export DS_ASYNC_PINNED_CHECKPOINT=${ASYNC_CKPT:-1} AUTORUN_LOCAL_COPIES=${LOCAL_COPIES:-0}
    ;;
  feat16)
    # 2026-10-11 (the user: port the sandbox changes to production and test them all on 16 nodes): THE PORTED FEATURES, ONE CHAIN.
    # 16 training nodes + 0 spares, DeepSeek_16B_4L, seq 4,096, EP 8, PP 2 (the pipeline engine's phase marks), 4 micro-batches, save every 2.
    #   kill:80:6     rank 80 (node 10) killed at step 6 -> rank_died (21), its node excluded -> NO SPARE LEFT -> AUTORUN_SHRINK=1: the memory
    #                 model picks the largest fitting node count (16 -> 8: the 256 sequences per step split only over a power-of-two width),
    #                 the save of step 4 is converted to universal on the batch node, attempt 1 runs on 8 nodes with --universal-checkpoint;
    #   hostmem:3:12  rank 3's heartbeat reads host memory 98 % at step 12 (no memory is allocated) -> code 19, retry 1 of 1, same 8 nodes,
    #                 and the restart loads the NATIVE save written under the 8-node layout (no --universal-checkpoint);
    #   hostmem:5:16  the second host-memory exit -> NO RESTART (host_oom seen twice, one restart allowed), mail.
    # Ride-along: the in-job dashboard.html (the duty on the batch node: no spare), the forward/backward/step phases at PP 2, the training.log header.
    # THE QUESTION: does production carry each of the three decisions exactly (recovery.log sentences, attempts' command lines and node counts,
    # data/shrink.tsv), with the loss continuing across the shrink? PASS = all of: attempt 1 on 8 nodes loading the universal of the step-4
    # save; attempt 2 on the same 8 nodes without --universal-checkpoint, resuming from a step > 4; the stop sentence after the second code 19.
    # LAYOUT: nodes 16 | EP 8 | PP 2 | TP 1 | DP_non_moe 64 | DP_moe 8 | mbs 1 x nbs 4 x 64 = GBS 256 | seq 4,096 | ckpt on | ZeRO 1 | 4 layers
    #   | after the shrink: 8 nodes, DP_non_moe 32, GA 8 | wall 45 min, QOS debug.
    LABEL=${LABEL:-PF1}
    export AUTORUN_EP_KEY="${N_TRAIN:-16}:8:1" AUTORUN_EP_CMD=" 1:${NBS0:-4}:${ITERS:-60}:X-MOE:${MODEL:-DeepSeek_16B_4L}:ckpt:1:1" AUTORUN_SEQ_LEN=${SEQ:-4096}
    export AUTORUN_PP_SIZE=${PP0:-2}
    export AUTORUN_SPARES=${SPARES:-0} AUTORUN_GATE=${GATE:-quick} AUTORUN_WALLTIME="${WALL:-00:45:00}" AUTORUN_EXIT_MINS=${EXIT_MINS:-43}
    export AUTORUN_GATE_CALIBRATE=0 AUTORUN_QOS=${QOS:-debug}
    export XMOE_INJECT_SCHEDULE="${SCHEDULE:-kill:80:6+hostmem:3:12+hostmem:5:16}"
    export RES_READMIT_INJECTED=0 AUTORUN_SHRINK=1 AUTORUN_MEM_GATE=1 AUTORUN_DASHBOARD=${DASHBOARD:-1}
    export AUTORUN_SAVE_INTERVAL=${SAVE_EVERY:-2}
    export DS_ASYNC_PINNED_CHECKPOINT=${ASYNC_CKPT:-1} AUTORUN_LOCAL_COPIES=${LOCAL_COPIES:-0}
    ;;
  tleft16)
    # 2026-10-11: THE TIME-LEFT RULE on production. 16 training nodes + 1 spare (so a restart WOULD be possible: only the time can refuse it),
    # the feat16 layout, kill:80:6, RES_RESTART_MARGIN_S=3000 (raised for the test so that the measured start-up + 3,000 s exceeds the time left
    # of a 30-minute job: the branch is forced, the measured start-up is real), AUTORUN_SHRINK=0.
    # THE QUESTION: does the loop refuse the restart with the sentence that names the time left, the measured start-up of this job and the
    # margin, and end the job without a second attempt? PASS = that sentence in recovery.log and no attempts/attempt_1.
    LABEL=${LABEL:-PT1}
    export AUTORUN_EP_KEY="${N_TRAIN:-16}:8:1" AUTORUN_EP_CMD=" 1:${NBS0:-4}:${ITERS:-60}:X-MOE:${MODEL:-DeepSeek_16B_4L}:ckpt:1:1" AUTORUN_SEQ_LEN=${SEQ:-4096}
    export AUTORUN_PP_SIZE=${PP0:-2}
    export AUTORUN_SPARES=${SPARES:-1} AUTORUN_GATE=${GATE:-quick} AUTORUN_WALLTIME="${WALL:-00:30:00}" AUTORUN_EXIT_MINS=${EXIT_MINS:-28}
    export AUTORUN_GATE_CALIBRATE=0 AUTORUN_QOS=${QOS:-normal}
    export XMOE_INJECT_SCHEDULE="${SCHEDULE:-kill:80:6}"
    export RES_READMIT_INJECTED=0 AUTORUN_SHRINK=0 RES_RESTART_MARGIN_S=${MARGIN:-3000}
    export AUTORUN_SAVE_INTERVAL=${SAVE_EVERY:-2}
    export DS_ASYNC_PINNED_CHECKPOINT=${ASYNC_CKPT:-1} AUTORUN_LOCAL_COPIES=${LOCAL_COPIES:-0}
    ;;
  perf16)
    # 2026-10-11 (the user: "make sure that the new features and the local host html will not hurt the performance at all"): ONE ARM OF A PAIR.
    # 16 training nodes + 1 spare (the duty, which renders the page, on the spare: the production placement), the feat16 layout, 800 iterations,
    # save every 400, no injections; DASHBOARD=1 (the page every 30 s) or DASHBOARD=0 (no page). LABEL PP1 (on) / PP0 (off).
    # THE QUESTION: does the in-job page cost training time? THRESHOLD (written before the runs): (a) within the ON arm, the median time of the
    # steps that overlap a page rewrite (data/dashboard_times.tsv against the step times) is within 1 % of the other steps' median; (b) the two
    # arms' mean TFLOP/s per GCD over iterations 9-800 (saves excluded) differ by less than 2 % (different node sets: the cross-job noise).
    LABEL=${LABEL:-PP1}
    export AUTORUN_EP_KEY="${N_TRAIN:-16}:8:1" AUTORUN_EP_CMD=" 1:${NBS0:-4}:${ITERS:-800}:X-MOE:${MODEL:-DeepSeek_16B_4L}:ckpt:1:1" AUTORUN_SEQ_LEN=${SEQ:-4096}
    export AUTORUN_PP_SIZE=${PP0:-2}
    export AUTORUN_SPARES=${SPARES:-1} AUTORUN_GATE=${GATE:-quick} AUTORUN_WALLTIME="${WALL:-00:35:00}" AUTORUN_EXIT_MINS=${EXIT_MINS:-33}
    export AUTORUN_GATE_CALIBRATE=0 AUTORUN_QOS=${QOS:-normal}
    export XMOE_INJECT_SCHEDULE=""
    export AUTORUN_DASHBOARD=${DASHBOARD:-1}
    export AUTORUN_SAVE_INTERVAL=${SAVE_EVERY:-400}
    export DS_ASYNC_PINNED_CHECKPOINT=${ASYNC_CKPT:-1} AUTORUN_LOCAL_COPIES=${LOCAL_COPIES:-0}
    ;;
  hang16|hang32)
    N=${MODE#hang}; LABEL=${LABEL:-H$N}
    export AUTORUN_EP_KEY="$N:8:8" AUTORUN_EP_CMD=" 1:1:20:X-MOE:DeepSeek_16B_32k:ckpt:1:1" AUTORUN_SEQ_LEN=${SEQ:-65536}
    export AUTORUN_SEQ_CHUNKED_CE=2048 AUTORUN_MEGATRON_SP=1
    export AUTORUN_SPARES=$([ $N = 16 ] && echo 2 || echo 4) AUTORUN_GATE=${GATE:-auto} AUTORUN_WALLTIME="00:40:00" AUTORUN_EXIT_MINS=39
    export XMOE_INJECT_SCHEDULE="${SCHEDULE:-hang:13:3}"
    export RES_READMIT_INJECTED=0
    export XMOE_PG_TIMEOUT_ALL_MIN=3   # at TP 8 the first collective after the stop is a TP collective (Megatron group): 3 min there too
    ;;
  hostoom3)
    # 2026-10-11 (the user: "update the sandbox code only to capture a real host oom error and test at a small scale e.g. 2+1 nodes"):
    # 2 train + 1 spare = 3 nodes, DeepSeek_16B_4L, seq 4,096, EP 8, PP 1, 24 iterations, save every 2, gate quick, debug QOS, 30 min.
    # The injected case realhostoom (sandbox hb.py) makes rank 9 (the second training node) write 4 GiB of host memory every 0.5 s from
    # step 6 until the kernel's (or the cgroup's) OOM killer ends it; XMOE_HOSTMEM_EXIT_PCT=101 keeps the 97 % guard of hb.py from exiting first.
    # THE QUESTION: does the restart loop classify a REAL host OOM (no exit record; Slurm's "task 9: Out Of Memory" / "oom-kill" lines in the
    # .e, or a killed task whose node's heartbeats read >= 95 % host memory) as host_oom (code 19) with the node and rank named, exclude no
    # node, restart once on the same 2 nodes from the save of step 6, and reach step 24 with the planned exit? PASS = all of these in
    # recovery.log and data/loop.log. Before this change the same failure was classified rank_died (21) and its node was excluded.
    # LAYOUT: nodes 2 | EP 8 | PP 1 | TP 1 | DP_non_moe 16 | DP_moe 2 | mbs 1 x nbs 1 x 16 = GBS 16 | seq 4,096 | ckpt on | ZeRO 1 |
    #   4 layers | tokens per node 32,768 | + 1 spare = 3 nodes | wall 30 min, QOS debug (the first run of a feature: C1).
    LABEL=${LABEL:-HO1}
    export AUTORUN_EP_KEY="${N_TRAIN:-2}:8:1" AUTORUN_EP_CMD=" 1:1:${ITERS:-24}:X-MOE:${MODEL:-DeepSeek_16B_4L}:ckpt:1:1" AUTORUN_SEQ_LEN=${SEQ:-4096}
    export AUTORUN_SPARES=${SPARES:-1} AUTORUN_GATE=${GATE:-quick} AUTORUN_WALLTIME="${WALL:-00:30:00}" AUTORUN_EXIT_MINS=${EXIT_MINS:-28}
    export AUTORUN_GATE_CALIBRATE=0 AUTORUN_QOS=${QOS:-debug}
    export XMOE_INJECT_SCHEDULE="${SCHEDULE:-realhostoom:9:6}" XMOE_HOSTMEM_EXIT_PCT=${HOSTMEM_PCT:-101}
    export RES_READMIT_INJECTED=0 AUTORUN_DASHBOARD=${DASHBOARD:-1}
    ;;
  netlaunch3)
    # 2026-10-11 (the user approved the launch-retry fix): 2 train + 1 spare = 3 nodes, DeepSeek_16B_4L, seq 4,096, EP 8, PP 1, 8 iterations,
    # gate FULL without calibration, no injected faults, debug QOS, 30 min.
    # THE QUESTION: with the network step's launch going through net_launch_retry.sh, does the normal path still work: the pair test
    # launches on the first try, every node gets a pair row, gate.log says "NOT LAUNCHED: none", and training starts? (The retry itself
    # fires only on Slurm's launch errors; tests/test_net_launch_retry.sh and tests/test_gate_judge.py case J cover it on the login node.)
    # LAYOUT: nodes 2 | EP 8 | PP 1 | TP 1 | DP_non_moe 16 | DP_moe 2 | mbs 1 x nbs 1 x 16 = GBS 16 | seq 4,096 | ckpt on | ZeRO 1 |
    #   4 layers | + 1 spare = 3 nodes | wall 30 min, QOS debug.
    LABEL=${LABEL:-NL1}
    export AUTORUN_EP_KEY="${N_TRAIN:-2}:8:1" AUTORUN_EP_CMD=" 1:1:${ITERS:-8}:X-MOE:${MODEL:-DeepSeek_16B_4L}:ckpt:1:1" AUTORUN_SEQ_LEN=${SEQ:-4096}
    export AUTORUN_SPARES=${SPARES:-1} AUTORUN_GATE=${GATE:-full} AUTORUN_WALLTIME="${WALL:-00:30:00}" AUTORUN_EXIT_MINS=${EXIT_MINS:-28}
    export AUTORUN_GATE_CALIBRATE=0 AUTORUN_QOS=${QOS:-debug}
    export XMOE_INJECT_SCHEDULE="" RES_READMIT_INJECTED=0 AUTORUN_DASHBOARD=${DASHBOARD:-1}
    ;;
esac
export XMOE_PG_TIMEOUT_MIN=${PG_MIN:-3}   # a mode block may raise PG_MIN (oom64: 10)
CK=checkpoint/res_${LABEL}
mkdir -p output/checkpoint; { [ -e "$CK" ] || [ -e "output/$CK" ]; } && { echo "the save dir $CK (or output/$CK) exists: remove it or pick another LABEL"; exit 1; }; mkdir -p "$CK"
export AUTORUN_SAVE_DIR=$CK AUTORUN_LOAD_DIR=$CK
export AUTORUN_PRE_CMD="find \$PWD/../../megatron/fused_kernels/build \$PWD/deepspeed -maxdepth 2 -name lock -mmin +5 -print -delete; echo \"[in-job] PYTHONPATH=\$PYTHONPATH\"; python -c \"import deepspeed, megatron, sys; ok = all(m.__file__.startswith('$SB/') for m in (deepspeed, megatron)); print('[in-job] deepspeed', deepspeed.__file__, '| megatron', megatron.__file__, '| sandbox' if ok else '| NOT THE SANDBOX'); sys.exit(0 if ok else 1)\" || { echo 'FATAL: the job does not import the sandbox' 1>&2; exit 1; }; [ \"\${AUTORUN_RESILIENT:-0}\" = 1 ] && [ \"\${XMOE_HB:-0}\" = 1 ] && [ -n \"\${XMOE_PG_TIMEOUT_MIN:-}\" ] || { echo 'FATAL: the resilience environment did not reach the job' 1>&2; exit 1; }"
LOG=$SCOPE/runs/${LABEL}.submit.$(date +%H%M%S).log; mkdir -p $SCOPE/runs
./local_autorun_frontier_0623_2026.sh 2>&1 | tee "$LOG" | grep -E "FATAL|ERROR|Submitted batch job|\[rider|template" || true
JOB=$(grep -o "Submitted batch job [0-9]*" "$LOG" | tail -1 | awk '{print $4}')
[ -z "$JOB" ] && { echo "no job id (render-only or failure); see $LOG"; exit 0; }
printf '%s\t%s\t%s\t%s\t%s\t%s\t-\tscripts-frontier/logs/job_%s_*\n' "$LABEL" "$JOB" "$(date '+%F %H:%M')" \
  "$(( ${AUTORUN_EP_KEY%%:*} + AUTORUN_SPARES + ${AUTORUN_UCP_WORKERS:-0} ))" "${AUTORUN_QOS:-normal}" "MODE=$MODE schedule=$XMOE_INJECT_SCHEDULE gate_inject=${XMOE_GATE_INJECT:-none} seq=$AUTORUN_SEQ_LEN spares=$AUTORUN_SPARES ucp_workers=${AUTORUN_UCP_WORKERS:-0} gate=$AUTORUN_GATE calibrate=${AUTORUN_GATE_CALIBRATE:-0} wall=$AUTORUN_WALLTIME iters=${ITERS:-40} save_every=${AUTORUN_SAVE_INTERVAL}${AUTORUN_OOM_RECONFIG:+ oom_reconfig=$AUTORUN_OOM_RECONFIG mem_profiler=${AUTORUN_MEM_PROFILER:-0} layout0=PP${AUTORUN_PP_SIZE:-1}_EP${AUTORUN_EP_KEY#*:}_U${U0:-1}_mbs${MBS0:-1}} (layout 3.11)${QUESTION:+ QUESTION: $QUESTION}" "$JOB" >> $SCOPE/jobs.tsv
echo "JOB $JOB ($LABEL)"
