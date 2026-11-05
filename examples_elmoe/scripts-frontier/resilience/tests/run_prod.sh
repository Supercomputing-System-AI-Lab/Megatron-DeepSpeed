#!/bin/bash
# run_prod.sh (2026-10-09 18:20): the resilience test on the PRODUCTION tree (the merged code), one job per call.
# Twin of run_matrix.sh (which runs the sandbox copy). Everything is a knob with a default; nothing here edits a shared file.
#   MODE=p2p1  2 train + 1 spare nodes, DeepSeek_16B_4L (the 4-layer 16B registry entry), seq 4,096, 24 iterations, wall 30 min,
#              the four injected faults that passed the sandbox smoke L1 (job 5644650): hang, exit, oom, nan.
# THE QUESTION: on the production tree, does every injected fault end in the right class, the right node named (and excluded when the
# class excludes), a restart inside the same allocation from the latest save, and a first iteration after it; detect/recover times
# measured. Threshold: PASS per case in RESULTS.md (results.py), as in L1.
# LAYOUT (frontier-job-submission-principles B): nodes 2 | EP 8 | PP 1 | TP 1 | DP_non_moe 16 | DP_moe 2 | mbs 1 x nbs 1 x 16 = GBS 16 |
#   seq 4,096 | ckpt on | ZeRO 1 | 4 layers | save every 2 steps | random init (load = save dir, empty at the start) | + 1 spare = 3 nodes.
# SIZE: batch partition; QOS from QOS (default normal: the debug slot is held by the running 256-node control and then by the chain).
set -euo pipefail
MODE=${MODE:-p2p1}
TREE=${TREE:-/lustre/orion/gen150/scratch/zixianw4/X-MoE}
SF=$TREE/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier
SCOPE=$TREE/Megatron-DeepSpeed-X-MoE/examples_elmoe/assistant_deep_dive/2026-10-09-resilience_impl
cd $SF
set +e +u; . ./eval/submit_env.sh; set -euo pipefail
export PATH=/lustre/orion/world-shared/gen150/zixianw4/envs/TORCH-ROCM6.4.1_env_groupgemm_v5/bin:$PATH
export CXX=$(which g++) CC=$(which gcc)
export LD_LIBRARY_PATH=/lustre/orion/gen150/scratch/zixianw4/aws-ofi-rccl-rocm6.4/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$TREE:$TREE/Megatron-DeepSpeed-X-MoE
python -c "import deepspeed, megatron, sys; ok = all(m.__file__.startswith('$TREE/') for m in (deepspeed, megatron)); print('[submit shell] deepspeed', deepspeed.__file__, '| megatron', megatron.__file__); sys.exit(0 if ok else 1)" \
  || { echo "FATAL: deepspeed/megatron do not import from the production tree $TREE"; exit 1; }

export AUTORUN_RESILIENT=1 XMOE_HB=1 XMOE_DETECT=1 XMOE_PG_TIMEOUT_MIN=${PG_MIN:-3} XMOE_T_INIT_S=${XMOE_T_INIT_S:-600}
export AUTORUN_QOS=${QOS:-normal} AUTORUN_NO_PROFILING=1 AUTORUN_DS_CONFIG_PER_JOB=1 AUTORUN_NCCL_DEBUG=INFO
export AUTORUN_SAVE_INTERVAL=${SAVE_INTERVAL:-2}
case $MODE in
  p2p1)
    LABEL=${LABEL:-P1}
    export AUTORUN_EP_KEY="${N_TRAIN:-2}:8:1" AUTORUN_EP_CMD=" 1:1:${ITERS:-24}:X-MOE:${MODEL:-DeepSeek_16B_4L}:ckpt:1:1" AUTORUN_SEQ_LEN=${SEQ:-4096}
    export AUTORUN_SPARES=${SPARES:-1} AUTORUN_GATE=${GATE:-quick} AUTORUN_WALLTIME="${WALL:-00:30:00}" AUTORUN_EXIT_MINS=${EXIT_MINS:-28}
    export AUTORUN_GATE_CALIBRATE=${CALIBRATE:-0}
    export XMOE_INJECT_SCHEDULE="${SCHEDULE:-hang:5:3+exit:9:5+oom:2:7+nan:0:9}"
    export RES_READMIT_INJECTED=${READMIT:-1}   # TEST ONLY: 1 spare for 2 node-excluding injected faults; an injected-fault node is healthy
    ;;
  p2full)
    # 2026-10-10 (after the port of rounds 2-5): the P1 twin with every new flag on. THE QUESTION: on the production tree, does the full
    # gate with the network tests run and calibrate (pair, switch: informational on this first production calibration), does the
    # node-local copy write and load, does the wall-aware exit end the job by itself with code 0 after a complete save (XMOE_WALL_AWARE_EXIT=1,
    # margin 240 s), are the sentences, alarms.log and the two-tier exclusion in place, and do the four injected faults still recover?
    # Threshold: RESULTS.md PASS per case + the wall row PASS (the exit sentence, srun 0, no USR1 handler) + a local-copy load line.
    LABEL=${LABEL:-P2}
    export AUTORUN_EP_KEY="${N_TRAIN:-2}:8:1" AUTORUN_EP_CMD=" 1:1:${ITERS:-1000}:X-MOE:${MODEL:-DeepSeek_16B_4L}:ckpt:1:1" AUTORUN_SEQ_LEN=${SEQ:-4096}
    export AUTORUN_SPARES=${SPARES:-1} AUTORUN_GATE=${GATE:-full} AUTORUN_WALLTIME="${WALL:-00:30:00}" AUTORUN_EXIT_MINS=${EXIT_MINS:-28}
    export AUTORUN_GATE_CALIBRATE=${CALIBRATE:-1}
    export XMOE_INJECT_SCHEDULE="${SCHEDULE:-hang:5:3+exit:9:5+oom:2:7+nan:0:9}"
    export RES_READMIT_INJECTED=${READMIT:-1}
    export AUTORUN_LOCAL_COPIES=${LOCAL_COPIES:-1} DS_ASYNC_PINNED_CHECKPOINT=${ASYNC_CKPT:-1}
    export AUTORUN_WALL_AWARE_EXIT=${WALL_AWARE:-1} XMOE_WALL_MARGIN_S=${MARGIN_S:-240}
    export AUTORUN_USE_DEGRADED=${USE_DEGRADED:-1}
    ;;
  *) echo "unknown MODE $MODE"; exit 1;;
esac
CK=checkpoint/res_${LABEL}
mkdir -p output/checkpoint; [ -e "output/$CK" ] && { echo "the save dir output/$CK exists: remove it or pick another LABEL"; exit 1; }
export AUTORUN_SAVE_DIR=$CK AUTORUN_LOAD_DIR=$CK
export AUTORUN_PRE_CMD="echo \"[in-job] PYTHONPATH=\$PYTHONPATH\"; python -c \"import deepspeed, megatron, sys; ok = all(m.__file__.startswith('$TREE/') for m in (deepspeed, megatron)); print('[in-job] deepspeed', deepspeed.__file__, '| megatron', megatron.__file__, '| production' if ok else '| NOT THE PRODUCTION TREE'); sys.exit(0 if ok else 1)\" || { echo 'FATAL: the job does not import the production tree' 1>&2; exit 1; }; [ \"\${AUTORUN_RESILIENT:-0}\" = 1 ] && [ \"\${XMOE_HB:-0}\" = 1 ] && [ -n \"\${XMOE_PG_TIMEOUT_MIN:-}\" ] || { echo 'FATAL: the resilience environment did not reach the job' 1>&2; exit 1; }"
LOG=$SCOPE/runs/${LABEL}.submit.$(date +%H%M%S).log; mkdir -p $SCOPE/runs
./local_autorun_frontier_0623_2026.sh 2>&1 | tee "$LOG" | grep -E "FATAL|ERROR|Submitted batch job|\[rider|template" || true
JOB=$(grep -o "Submitted batch job [0-9]*" "$LOG" | tail -1 | awk '{print $4}')
[ -z "$JOB" ] && { echo "no job id (failure); see $LOG"; exit 1; }
printf '%s\t%s\t%s\t%s\t%s\t%s\t-\tscripts-frontier/logs/job_%s_*\n' "$LABEL" "$JOB" "$(date '+%F %H:%M')" \
  "$(( ${AUTORUN_EP_KEY%%:*} + AUTORUN_SPARES ))" "$AUTORUN_QOS" "PRODUCTION tree. MODE=$MODE model=${MODEL:-DeepSeek_16B_4L} schedule=$XMOE_INJECT_SCHEDULE seq=$AUTORUN_SEQ_LEN spares=$AUTORUN_SPARES gate=$AUTORUN_GATE wall=$AUTORUN_WALLTIME iters=${ITERS:-24} local_copies=${AUTORUN_LOCAL_COPIES:-0} wall_aware=${AUTORUN_WALL_AWARE_EXIT:-0} calibrate=${AUTORUN_GATE_CALIBRATE:-0}" "$JOB" >> $SCOPE/jobs.tsv
echo "JOB $JOB ($LABEL) log $LOG"
