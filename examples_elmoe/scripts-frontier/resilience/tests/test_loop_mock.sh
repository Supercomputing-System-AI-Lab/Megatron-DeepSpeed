#!/bin/bash
# Login-node check of restart_loop.sh with a MOCK trainer (no Slurm job, no GPU): 2 training nodes + 1 spare named mockA mockB mockC.
# attempt 0: an injected hang (rank 5 on mockA); the mock writes heartbeat lines (rank 5 behind its group), a watchdog line in the .e,
#            and exits 134 -> expect class 10 collective_hang, node mockA named by the minority rule, mockA excluded, the duty moves,
#            and the SPARE CHECK (2026-10-09 late evening) runs on mockC before it enters the training set (the mock srun runs
#            gate_node.sh quick on the login node: every tool is missing or uncalibrated, so the node passes).
# attempt 1: an injected exit of rank 9 on mockC (srun's task line in the .e) -> class 21 rank_died, mockC excluded; no healthy spare is
#            left (mockD was excluded as DEGRADED by a pre-written data/gate/excluded_reasons.tsv, the test hook with the gate off), so
#            the loop puts mockD into training as the LAST RESORT (2026-10-09 late evening) after its spare check, and says so in
#            loop.log, recovery.log, training.log's header and the banner.
# attempt 2: an injected OOM on rank 2 (the exit record of D4) -> class 11, no exclusion.
# attempt 3: the planned exit (rc 0). Then RESULTS.md must show three PASS rows and heartbeat.log/a-monitor.txt three RECOVERED banners.
# 2026-10-09 (design 3.11): also the job folder tree: only the human files at the top, attempts/attempt_<k>/{ranks,exit,nccl_trace,
# full_run.log}, data/; training.log with a header per attempt and the step lines; recovery.log in words; STATUS.md; heatmap.png.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
T=${TMPD:-/tmp}/loopmock_$$; mkdir -p $T/bin $T/work/logs
cat > $T/bin/scontrol <<'S'
#!/bin/bash
if [ "$1 $2" = "show hostnames" ]; then echo mockA; echo mockB; echo mockC; echo mockD; exit 0; fi
if [ "$1 $2" = "show job" ]; then echo "JobId=1 EndTime=$(date -d '+40 min' +%Y-%m-%dT%H:%M:%S)"; exit 0; fi
S
cat > $T/bin/srun <<'S'
#!/bin/bash
while [ $# -gt 0 ]; do case $1 in -w|--jobid) shift 2;; -*) shift;; *) break;; esac; done
exec "$@"
S
cat > $T/bin/sinfo <<'S'
#!/bin/bash
exit 0
S
chmod +x $T/bin/*; export PATH=$T/bin:$PATH
cd $T/work
export SLURM_JOB_ID=999 SLURM_JOB_NODELIST=mock RUN_TYPE=ep NODES=2 TOTAL_GPUS=16 gpus_per_node=8 PP_SIZE=1 EP_PARALLEL_SIZE=8
export JOB_DIR=logs/job_999_mock AUTORUN_GATE=0 RES_PY=${RES_PY:-/lustre/orion/world-shared/gen150/zixianw4/envs/TORCH-ROCM6.4.1_env_groupgemm_v5/bin/python} RES_STILL_S=5 XMOE_T_INIT_S=60 RES_MIN_ATTEMPT_S=60
export XMOE_INJECT_SCHEDULE="hang:5:3+exit:9:5+oom:2:6"
mkdir -p $JOB_DIR/data/gate; : > $JOB_DIR/a-xmoe-ep-999.e
# [2026-10-09 late evening] mockD is DEGRADED from the start (as if the gate had excluded it for a slow GCD): the test hook of
# res_main applies this file with the gate off; the loop must use mockD as the spare of last resort when no healthy spare is left
printf 'node\ttier\ttest\tvalue\tthreshold\tsentence\nmockD\tdegraded\tgemm\t60.0\t>= 101.7\tthe gate excluded mockD as degraded: gemm: below threshold (value 60.0, threshold >= 101.7)\n' > $JOB_DIR/data/gate/excluded_reasons.tsv
export CHECKPOINT_PATH=$T/work/ckpt; mkdir -p $CHECKPOINT_PATH
xmoe_dry_run_cmd() { echo true; }
merge_full_run() { { echo "mock .o"; cat "$1/rank_0.log" "$1/rank_15.log" 2>/dev/null; } > "$2"; }
xmoe_train_srun() {   # the mock: $1 nodes csv, $2 node count; writes its rank logs into RANK_LOG_DIR directly, as the template does
  local a=$RES_ATTEMPT J=$PWD/$JOB_DIR nodes=(${1//,/ }) R=$RANK_LOG_DIR
  exec bash -c "
    sleep 2; echo ' iteration        $((a*3+1))/      40 | consumed samples' > $R/rank_15.log; echo 'rank 0 via NET/AWS Libfabric/0' > $R/rank_0.log
    cd $RES_BIN && $RES_PY -c 'import hb; hb._ctx.update(rank=0, job_dir=\"$J\", attempt=\"$a\", resumed_from=$((a*2))); hb.training_log_line($((a*3+1)), 10.5, 35.1, 2.4, 40.2, True)'
    now=\$(date +%s)
    if [ $a = 0 ]; then
      echo '{\"case\": \"hang\", \"rank\": 5, \"step_planned\": 3, \"step\": 3, \"time\": '\$now', \"time_str\": \"x\", \"node\": \"${nodes[0]}\", \"attempt\": \"0\"}' > $J/data/inject/hang_r5_s3.fired
      for k in 1 2 3 4 5 6 7 8 9; do
        t=\$(date +%s)
        for r in \$(seq 0 15); do n=${nodes[0]}; [ \$r -ge 8 ] && n=${nodes[1]}; cs=101; [ \$r = 5 ] && cs=100; [ \$r -ge 8 ] && cs=150
          printf '%s\t%s\t%s\t3\tall_to_all\t%s\t3.0\t2.0\t10\t40\t0\t0\t0\tR\t2.5\t0\t%s\t-\t%s.0\n' \$(date +%Y-%m-%dT%H:%M:%S) \$n \$r \$cs \$((1000+r)) \$t >> $J/data/hb/\$n.tsv; done
        sleep 1
      done
      sleep 12
      echo '[rank3]:[E] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=2497, OpType=ALLTOALL_BASE, Timeout(ms)=180000)' >> $J/a-xmoe-ep-999.e
      exit 134
    elif [ $a = 1 ]; then
      echo '{\"case\": \"exit\", \"rank\": 9, \"step_planned\": 5, \"step\": 5, \"time\": '\$now', \"time_str\": \"x\", \"node\": \"${nodes[1]}\", \"attempt\": \"1\"}' > $J/data/inject/exit_r9_s5.fired
      sleep 3
      echo 'srun: error: ${nodes[1]}: task 9: Exited with exit code 3' >> $J/a-xmoe-ep-999.e
      exit 3
    elif [ $a = 2 ]; then
      echo '{\"case\": \"oom\", \"rank\": 2, \"step_planned\": 6, \"step\": 6, \"time\": '\$now', \"time_str\": \"x\", \"node\": \"${nodes[0]}\", \"attempt\": \"2\"}' > $J/data/inject/oom_r2_s6.fired
      sleep 3
      echo '{\"code\": 11, \"class\": \"gpu_oom\", \"rank\": 2, \"node\": \"${nodes[0]}\", \"step\": 6, \"phase\": \"forward\", \"time\": '\$(date +%s)', \"attempt\": \"2\", \"evidence\": \"OutOfMemoryError: HIP out of memory\"}' > $RES_EXIT_DIR/r2.json
      exit 11
    else
      sleep 3; exit 0
    fi"
}
source $HERE/../bin/restart_loop.sh
RES_BIN=$HERE/../bin
# the global failure record goes to the mock folder, not the real one
sed_rec() { :; }
cp $HERE/../bin/*.py $T/bin/ 2>/dev/null
RES_BIN=$T/bin; cp $HERE/../bin/restart_loop.sh $T/bin/
res_main
echo "=== data/loop.log"; cat $JOB_DIR/data/loop.log
echo "=== RESULTS.md"; cat $JOB_DIR/RESULTS.md
echo "=== heartbeat.log head"; head -5 $JOB_DIR/heartbeat.log
echo "=== a-monitor.txt head"; head -4 $JOB_DIR/a-monitor.txt
ok=1
grep -q "| hang:5:3 | mockA |.*10 collective_hang | mockA (rank 5) | node mockA was excluded |.*\*\*PASS\*\*" $JOB_DIR/RESULTS.md || { echo "FAIL hang row"; ok=0; }
grep -q "| oom:2:6 |.*11 gpu_oom |.*\*\*PASS\*\*" $JOB_DIR/RESULTS.md || { echo "FAIL oom row"; ok=0; }
grep -q "| exit:9:5 | mockC |.*21 rank_died | mockC (rank 9) | node mockC was excluded |.*\*\*PASS\*\*" $JOB_DIR/RESULTS.md || { echo "FAIL exit row"; ok=0; }
[ $(grep -c "^##### RECOVERED" $JOB_DIR/data/banners.txt) = 3 ] || { echo "FAIL banners"; ok=0; }
grep -q "LAST RESORT: no healthy spare was left, so the degraded node mockD (gemm 60.0 against the threshold 101.7) was put into training for attempt 2" $JOB_DIR/data/loop.log || { echo "FAIL last-resort line in loop.log"; ok=0; }
grep -q "no healthy spare was left, so the degraded node mockD" $JOB_DIR/data/banners.txt || { echo "FAIL last-resort note in the banner"; ok=0; }
grep -q "spare check: before mockD joins the training set for attempt 2" $JOB_DIR/data/loop.log || { echo "FAIL spare check of the degraded node"; ok=0; }
grep -q "degraded, usable as a last resort when no healthy spare is left: mockD (gemm 60.0 against the threshold 101.7)" $JOB_DIR/STATUS.md.attempt1 2>/dev/null || true
head -2 $JOB_DIR/a-monitor.txt | grep -q "RECOVERED" || { echo "FAIL a-monitor banner"; ok=0; }
grep -q "duty hand-over" $JOB_DIR/data/loop.log || { echo "FAIL duty hand-over"; ok=0; }
echo "=== the tree (top level)"; ls -1 $JOB_DIR; echo "=== attempts"; (cd $JOB_DIR && find attempts -maxdepth 2 | sort)
echo "=== training.log"; cat $JOB_DIR/training.log; echo "=== recovery.log"; cat $JOB_DIR/recovery.log; echo "=== STATUS.md"; cat $JOB_DIR/STATUS.md
extra=$(ls -1 $JOB_DIR | grep -v -x -E "README.md|STATUS.md|training.log|recovery.log|alarms.log|alarms|heartbeat.log|gate.log|heatmap.png|a-xmoe-ep-999.[oe]|a-monitor.txt|RESULTS.md|attempts|data")
grep -q "^[0-9-]* [0-9:]* ALARM minority" $JOB_DIR/alarms.log && grep -q "became failure 1" $JOB_DIR/alarms.log || { echo "FAIL alarms.log lines"; ok=0; }
[ -n "$(ls -1 $JOB_DIR/alarms/ 2>/dev/null)" ] || { echo "FAIL alarms/ holds no stack sample"; ok=0; }
[ -z "$extra" ] || { echo "FAIL top level holds more: $extra"; ok=0; }
for k in 0 1 2 3; do for x in ranks/rank_15.log exit nccl_trace full_run.log; do [ -e $JOB_DIR/attempts/attempt_$k/$x ] || { echo "FAIL attempt_$k/$x missing"; ok=0; }; done; done
[ -f $JOB_DIR/attempts/attempt_2/exit/r2.json ] || { echo "FAIL exit record of attempt 2"; ok=0; }
[ $(grep -c "^# ===== attempt" $JOB_DIR/training.log) = 4 ] || { echo "FAIL training.log attempt headers"; ok=0; }
[ $(grep -c -E "^[0-9-]+ [0-9:]+ \|" $JOB_DIR/training.log) = 4 ] || { echo "FAIL training.log step lines"; ok=0; }
grep -q "^# node change: no healthy spare was left, so the degraded node mockD" $JOB_DIR/training.log || { echo "FAIL last-resort header in training.log"; ok=0; }
grep -q "what fired:  the NCCL watchdog" $JOB_DIR/recovery.log && grep -q "what fired:  the detector inside rank 2" $JOB_DIR/recovery.log || { echo "FAIL recovery.log prose"; ok=0; }
[ $(grep -c "^##### RECOVERED" $JOB_DIR/recovery.log) = 3 ] || { echo "FAIL recovery.log banners"; ok=0; }
grep -q "  last resort: no healthy spare was left, so the degraded node mockD" $JOB_DIR/recovery.log || { echo "FAIL last-resort line in recovery.log"; ok=0; }
grep -q '^state          the loop.s state is "done"' $JOB_DIR/STATUS.md || { echo "FAIL STATUS.md"; ok=0; }
grep -q "spare check: before mockC joins the training set for attempt 1" $JOB_DIR/data/loop.log || { echo "FAIL spare check line"; ok=0; }
grep -q "which fired at step 3" $JOB_DIR/data/banners.txt || { echo "FAIL banner step (from the injection marker)"; ok=0; }
[ -f $JOB_DIR/heatmap.png ] || { echo "FAIL heatmap.png (end of attempt)"; ok=0; }
[ -f $JOB_DIR/README.md ] || { echo "FAIL README.md"; ok=0; }
[ $ok = 1 ] && echo "MOCK LOOP: ALL PASS" || echo "MOCK LOOP: SOME FAILED"
