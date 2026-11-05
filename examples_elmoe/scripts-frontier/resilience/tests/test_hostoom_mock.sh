#!/bin/bash
# test_hostoom_mock.sh (2026-10-11, sandbox; the user: "capture a real host oom error"): the restart loop's classification of a REAL host out of
# memory (the kernel's or the cgroup's OOM killer, which the 97 % guard of hb.py did not catch), on the login node with a mock trainer.
# 2 training nodes + 1 spare (mock000..mock002, 16 ranks). VARIANT a: attempt 0 ends with exit 137 and NO exit record, and the .e carries Slurm's
# lines "srun: error: mock001: task 9: Out Of Memory" + "slurmstepd: error: Detected 1 oom-kill event(s) ...". VARIANT b: only "srun: error:
# mock001: task 9: Killed", while mock001's heartbeat file read host memory 99 %. Both must give class host_oom (code 19), node mock001 rank 9,
# NO exclusion (the class does not exclude), restart 1 on the same 2 nodes, then the planned exit.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REALPY=${RES_PY:-/lustre/orion/world-shared/gen150/zixianw4/envs/TORCH-ROCM6.4.1_env_groupgemm_v5/bin/python}
allok=1
for VARIANT in a b c d; do
T=${TMPD:-/tmp}/hostoommock_${VARIANT}_$$; mkdir -p $T/bin $T/work/logs $T/work/ckpt
cat > $T/bin/scontrol <<'S'
#!/bin/bash
if [ "$1 $2" = "show hostnames" ]; then for i in 0 1 2; do printf 'mock%03d\n' $i; done; exit 0; fi
if [ "$1 $2" = "show job" ]; then echo "JobId=1 EndTime=$(date -d '+40 min' +%Y-%m-%dT%H:%M:%S)"; exit 0; fi
S
printf '#!/bin/bash\nwhile [ $# -gt 0 ]; do case $1 in -w|--jobid) shift 2;; -*) shift;; *) break;; esac; done\nexec "$@"\n' > $T/bin/srun
printf '#!/bin/bash\nexit 0\n' > $T/bin/sinfo
chmod +x $T/bin/*; export PATH=$T/bin:$PATH
cd $T/work
export SLURM_JOB_ID=995 SLURM_JOB_NODELIST=mock RUN_TYPE=ep NODES=2 TOTAL_GPUS=16 gpus_per_node=8 PP_SIZE=1 EP_PARALLEL_SIZE=8 MP_SIZE=1 BATCH_SIZE=1 NUM_BATCHES=1
export GLOBAL_BATCH_SIZE=16 SEQ_LEN=4096 MODEL_SIZE=DeepSeek_16B_4L ACTIVATION_CHECKPOINT=true DS_SEQ_PAR_SIZE=1
export JOB_DIR=logs/job_995_mock AUTORUN_GATE=0 RES_PY=$REALPY RES_STILL_S=5 XMOE_T_INIT_S=60 RES_MIN_ATTEMPT_S=60 XMOE_INJECT_SCHEDULE="" VARIANT
mkdir -p $JOB_DIR/data/gate $JOB_DIR/data/hb; : > $JOB_DIR/a-xmoe-ep-995.e
export CHECKPOINT_PATH=$T/work/ckpt LOAD_CHECKPOINT_PATH=$T/work/ckpt
megatron_options=" --moe-expert-parallel-size 8 --micro-batch-size 1 --global-batch-size 16 --seq-length 4096 --save $CHECKPOINT_PATH --load $LOAD_CHECKPOINT_PATH"
deepspeed_options=" --deepspeed --no-pipeline-parallel"
xmoe_dry_run_cmd() { echo true; }
merge_full_run() { { echo "mock .o"; cat "$1/rank_0.log" 2>/dev/null; } > "$2"; }
xmoe_train_srun() {
  local a=$RES_ATTEMPT J=$PWD/$JOB_DIR R=$RANK_LOG_DIR E=$PWD/$JOB_DIR/a-xmoe-ep-995.e
  echo "ATTEMPT $a NODES=$2 CSV=$1" > "$RES_ATT_DIR/cmdline.txt"
  exec bash -c "
    cd $RES_BIN
    if [ $a = 0 ]; then
      sleep 2; echo ' iteration        1/      24 | consumed samples' > $R/rank_15.log; echo 'rank 0 via NET/AWS Libfabric/0' > $R/rank_0.log
      $REALPY -c 'import hb; hb._ctx.update(rank=15, job_dir=\"$J\", attempt=\"$a\", resumed_from=0); hb.training_log_line(1, 10.5, 25.1, 0.9, 9.9, True); hb.training_log_line(2, 10.4, 25.0, 0.9, 9.9, True)'
      mkdir -p $CHECKPOINT_PATH/global_step2; echo x > $CHECKPOINT_PATH/global_step2/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt; echo global_step2 > $CHECKPOINT_PATH/latest
      printf 'time\tnode\trank\tstep\tphase\tcoll_seq\tstep_time_s\tbusy_s\thbm_gb\thost_mem_pct\tecc_ce\tecc_ue\txgmi_err\tproc_state\tloss\tattempt\tpid\tflag\tepoch\n' > $J/data/hb/mock001.tsv
      printf '%s\tmock001\t9\t3\tforward\t100\t0.9\t0.8\t9.0\t%s\t0\t0\t0\tR\t10.4\t0\t123\t-\t%s\n' \"\$(date +%T)\" \$([ $VARIANT = b ] && echo 99.0 || echo 41.0) \$(date +%s) >> $J/data/hb/mock001.tsv
      sleep 1
      if [ $VARIANT = a ]; then
        echo 'slurmstepd: error: Detected 1 oom-kill event(s) in StepId=995.0. Some of your processes may have been killed by the cgroup out-of-memory handler.' >> $E
        echo 'srun: error: mock001: task 9: Out Of Memory' >> $E
      elif [ $VARIANT = b ]; then
        echo 'srun: error: mock001: task 9: Killed' >> $E
      else
        # c, d: the HO1 shape (job 5652117): the other ranks' watchdog fired first (c: an exit record of rank 12, collective_hang;
        # d: only the watchdog line in the .e), then Slurm's own wording of the OOM kill
        if [ $VARIANT = c ]; then mkdir -p $RES_ATT_DIR/exit; echo '{\"code\": 10, \"class\": \"collective_hang\", \"rank\": 12, \"node\": \"mock001\", \"step\": 6, \"phase\": \"forward\", \"time\": 1.0, \"time_str\": \"x\", \"attempt\": \"0\", \"evidence\": \"Watchdog caught collective operation timeout\"}' > $RES_ATT_DIR/exit/r12.json; fi
        echo '[rank12]:[E] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=145, OpType=ALLTOALL_BASE, NumelIn=64, NumelOut=64, Timeout(ms)=180000) ran for 180012 milliseconds before timing out.' >> $E
        echo '[2026-10-11T13:28:09.130] error: Detected 1 oom_kill event in StepId=995.0. Some of the step tasks have been OOM Killed.' >> $E
        echo 'srun: error: mock001: task 9: Out Of Memory' >> $E
      fi
      exit 137
    else
      sleep 2; echo ' iteration        3/      24 | consumed samples' > $R/rank_15.log; echo 'rank 0 via NET/AWS Libfabric/0' > $R/rank_0.log
      $REALPY -c 'import hb; hb._ctx.update(rank=15, job_dir=\"$J\", attempt=\"$a\", resumed_from=2); hb.training_log_line(3, 10.3, 24.0, 0.9, 8.8, True)'
      sleep 1; exit 0
    fi"
}
source $HERE/../bin/restart_loop.sh
cp -r $HERE/../bin/*.py $HERE/../bin/oom_reconfig.sh $T/bin/ 2>/dev/null
RES_BIN=$T/bin; cp $HERE/../bin/restart_loop.sh $T/bin/
res_main > $T/loop.out 2>&1
ok=1
grep -q "class:       host_oom (code 19)" $JOB_DIR/recovery.log || { echo "FAIL [$VARIANT] class host_oom"; ok=0; }
grep -q "named:       rank 9 on node mock001" $JOB_DIR/recovery.log || { echo "FAIL [$VARIANT] named rank 9 on mock001"; ok=0; }
grep -q "node:        no node was excluded" $JOB_DIR/recovery.log || { echo "FAIL [$VARIANT] no exclusion"; ok=0; }
[ $VARIANT = a ] && { grep -q "host OOM (Slurm): .*oom-kill" $JOB_DIR/recovery.log || { echo "FAIL [a] the Slurm evidence"; ok=0; }; }
[ $VARIANT = c ] && { grep -q "is a consequence: the other ranks waited for the killed rank" $JOB_DIR/recovery.log || { echo "FAIL [c] the consequence note"; ok=0; }; }
[ $VARIANT = d ] && { grep -q "host OOM (Slurm): .*oom_kill event" $JOB_DIR/recovery.log || { echo "FAIL [d] the Slurm evidence (oom_kill wording)"; ok=0; }; }
[ $VARIANT = b ] && { grep -q "host OOM (inferred): the node's last heartbeats read host memory 99.0 %" $JOB_DIR/recovery.log || { echo "FAIL [b] the inferred evidence"; ok=0; }; }
grep -q "NODES=2 CSV=mock000,mock001" $JOB_DIR/attempts/attempt_1/cmdline.txt || { echo "FAIL [$VARIANT] attempt 1 on the same 2 nodes: $(cat $JOB_DIR/attempts/attempt_1/cmdline.txt 2>/dev/null)"; ok=0; }
grep -q "the planned exit" $JOB_DIR/data/loop.log || { echo "FAIL [$VARIANT] the planned exit"; ok=0; }
echo "== variant $VARIANT: $(grep -m1 'class:' $JOB_DIR/recovery.log | cut -c1-120) | $(grep -m1 'evidence:' -A1 $JOB_DIR/recovery.log | tail -1 | cut -c1-160)"
[ $ok = 1 ] && echo "variant $VARIANT: PASS" || { echo "variant $VARIANT: FAIL"; allok=0; }
cd $HERE; [ "${KEEP:-0}" = 1 ] || rm -rf $T
done
[ $allok = 1 ] && echo "HOSTOOM MOCK: ALL PASS" || echo "HOSTOOM MOCK: SOME FAILED"
