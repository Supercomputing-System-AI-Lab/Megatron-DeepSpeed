#!/bin/bash
# test_oom_mock.sh (2026-10-10, round 6): the restart loop with the OOM re-configuration rule (AUTORUN_OOM_RECONFIG=1), on the login node with a
# MOCK trainer, a mock srun and a mock converter: 64 training nodes + 1 spare named mock000..mock064 (512 ranks), DeepSeek_16B at 128k,
# PP2 EP16 TP1 x U1 mbs 1, 256 sequences per step (the layout of the oom64 probe).
#   attempt 0: the trainer writes the OOM exit record of rank 300 (stage 1) at step 1 and exits 11 BEFORE any save -> the memory gate must have
#              warned (red), the rule takes step 1 (d aligned to EP 16: TP1 x U16, the fewest bytes), no universal (no save), the next attempt starts
#              from the initial state with --ds-sequence-parallel-size 16 and without --universal-checkpoint.
#   attempt 1: the trainer logs two steps, saves step 3 ('latest' = global_step3), then OOMs again (rank 5, step 4) -> the rule takes step 2 (EP
#              doubled alone, EP32, because the 16 heads bound d at 16), the loop converts the save on the batch node (no UCP workers in the mock:
#              the fake python answers ds_to_universal), 'latest_universal' names global_step3_universal, and the next attempt carries
#              --universal-checkpoint --load <the save dir> --moe-expert-parallel-size 32.
#   attempt 2: two steps and the planned exit. Then: two RECOVERED banners naming the layouts, two rule blocks in recovery.log, the profiler's peak lines
#              (the exit record and the peak files), data/oom_reconfig.tsv (2 rows), data/attempt_layouts.tsv (3 rows), the RESULTS.md section.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
T=${TMPD:-/tmp}/oommock_$$; mkdir -p $T/bin $T/work/logs $T/work/ckpt
REALPY=${RES_PY:-/lustre/orion/world-shared/gen150/zixianw4/envs/TORCH-ROCM6.4.1_env_groupgemm_v5/bin/python}
cat > $T/bin/scontrol <<'S'
#!/bin/bash
if [ "$1 $2" = "show hostnames" ]; then for i in $(seq 0 64); do printf 'mock%03d\n' $i; done; exit 0; fi
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
cat > $T/bin/fakepy <<S
#!/bin/bash
# the fake converter: python -m deepspeed.checkpoint.ds_to_universal --input_folder IN --output_folder OUT ... (no --param-shard: one share)
if [ "\$1" = -m ] && [ "\$2" = deepspeed.checkpoint.ds_to_universal ]; then
  shift 2; IN=""; OUT=""
  while [ \$# -gt 0 ]; do case \$1 in --input_folder) IN=\$2; shift 2;; --output_folder) OUT=\$2; shift 2;; *) shift;; esac; done
  step=\$(basename \$IN | sed 's/global_step//'); echo "fake converter: step \$step -> \$OUT"; sleep 1
  mkdir -p \$OUT/zero/param_0; echo x > \$OUT/zero/param_0/fp32.pt; echo x > \$OUT/zero/optimizer_state.pt; exit 0
fi
exec $REALPY "\$@"
S
chmod +x $T/bin/*; export PATH=$T/bin:$PATH
cd $T/work
export SLURM_JOB_ID=997 SLURM_JOB_NODELIST=mock RUN_TYPE=ep NODES=64 TOTAL_GPUS=512 gpus_per_node=8 PP_SIZE=2 EP_PARALLEL_SIZE=16 MP_SIZE=1 BATCH_SIZE=1 NUM_BATCHES=1
export GLOBAL_BATCH_SIZE=256 SEQ_LEN=131072 MODEL_SIZE=DeepSeek_16B ACTIVATION_CHECKPOINT=true DS_SEQ_PAR_SIZE=1
export JOB_DIR=logs/job_997_mock AUTORUN_GATE=0 RES_PY=$T/bin/fakepy RES_STILL_S=5 XMOE_T_INIT_S=60 RES_MIN_ATTEMPT_S=60
export AUTORUN_OOM_RECONFIG=1 AUTORUN_MEM_PROFILER=1 XMOE_INJECT_SCHEDULE=""
mkdir -p $JOB_DIR/data/gate; : > $JOB_DIR/a-xmoe-ep-997.e
export CHECKPOINT_PATH=$T/work/ckpt LOAD_CHECKPOINT_PATH=$T/work/ckpt
cat > $T/work/ds.json <<'J'
{ "train_batch_size": 256, "train_micro_batch_size_per_gpu": 1, "zero_optimization": {"stage": 1} }
J
config_json=$T/work/ds.json
megatron_options=" --moe-expert-parallel-size 16 --num-experts 64 --tensor-model-parallel-size 1 --micro-batch-size 1 --global-batch-size 256 --seq-length 131072 --seq-chunked-ce 2048 --checkpoint-activations --checkpoint-num-layers 1 --disable-moe-token-dropping --use-flash-attn-v2 --save $CHECKPOINT_PATH --load $LOAD_CHECKPOINT_PATH --use-rotary-position-embeddings"
deepspeed_options=" --deepspeed --deepspeed_config $config_json --pipeline-model-parallel-size 2 --deepspeed-activation-checkpointing"
xmoe_dry_run_cmd() { echo true; }
merge_full_run() { { echo "mock .o"; cat "$1/rank_0.log" "$1/rank_511.log" 2>/dev/null; } > "$2"; }
mk_save() { mkdir -p $CHECKPOINT_PATH/global_step$1; echo x > $CHECKPOINT_PATH/global_step$1/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt; echo global_step$1 > $CHECKPOINT_PATH/latest; }
xmoe_train_srun() {   # the mock: records the command line it would run, then plays its attempt's script
  local a=$RES_ATTEMPT J=$PWD/$JOB_DIR R=$RANK_LOG_DIR
  { echo "ATTEMPT $a"; echo "MEGATRON_OPTIONS=$megatron_options"; echo "DEEPSPEED_OPTIONS=$deepspeed_options"; echo "XMOE_MEGATRON_SP=${XMOE_MEGATRON_SP:-unset}"; echo "PATH_HEAD=${PATH%%:*}"; echo "MEMSNAP_OUT=${MEMSNAP_OUT:-unset}"; echo "MBS_JSON=$(grep -o '"train_micro_batch_size_per_gpu": *[0-9]*' $config_json)"; } > "$RES_ATT_DIR/cmdline.txt"
  exec bash -c "
    mkdir -p \${MEMSNAP_OUT:-$RES_ATT_DIR/memsnap}
    cd $RES_BIN
    if [ $a = 0 ]; then
      sleep 2
      echo '40.0 46.0 '\$(date +%s) > \${MEMSNAP_OUT}/peak_rank300.txt; echo '38.5 44.0 '\$(date +%s) > \${MEMSNAP_OUT}/peak_rank0.txt
      echo '{\"code\": 11, \"class\": \"gpu_oom\", \"rank\": 300, \"node\": \"mock037\", \"step\": 1, \"phase\": \"forward\", \"time\": '\$(date +%s)', \"attempt\": \"0\", \"evidence\": \"OutOfMemoryError: HIP out of memory. Tried to allocate 1024.00 MiB. GPU 4 has a total capacity of 63.98 GiB of which 812.00 MiB is free. Of the allocated memory 59.42 GiB is allocated by PyTorch | hbm_gb=63.80\"}' > $RES_EXIT_DIR/r300.json
      exit 11
    elif [ $a = 1 ]; then
      sleep 2; echo ' iteration        2/      24 | consumed samples' > $R/rank_511.log; echo 'rank 0 via NET/AWS Libfabric/0' > $R/rank_0.log
      echo '[Rank 5] (after 2 iterations) memory (GB) | allocated: 8.1 | max allocated: 9.9 | reserved: 12.0 | max reserved: 12.0' >> $R/rank_0.log
      $REALPY -c 'import hb; hb._ctx.update(rank=511, job_dir=\"$J\", attempt=\"$a\", resumed_from=0); hb.training_log_line(2, 10.5, 25.1, 120.4, 9.9, True); hb.training_log_line(3, 10.4, 25.0, 119.9, 9.9, True)'
      echo '9.9 12.0 '\$(date +%s) > \${MEMSNAP_OUT}/peak_rank5.txt
      mkdir -p $CHECKPOINT_PATH/global_step3; echo x > $CHECKPOINT_PATH/global_step3/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt; echo global_step3 > $CHECKPOINT_PATH/latest
      sleep 1
      echo '{\"code\": 11, \"class\": \"gpu_oom\", \"rank\": 5, \"node\": \"mock000\", \"step\": 4, \"phase\": \"backward\", \"time\": '\$(date +%s)', \"attempt\": \"1\", \"evidence\": \"OutOfMemoryError: HIP out of memory | hbm_gb=63.10\"}' > $RES_EXIT_DIR/r5.json
      exit 11
    else
      sleep 2; echo ' iteration        5/      24 | consumed samples' > $R/rank_511.log; echo 'rank 0 via NET/AWS Libfabric/0' > $R/rank_0.log
      $REALPY -c 'import hb; hb._ctx.update(rank=511, job_dir=\"$J\", attempt=\"$a\", resumed_from=3); hb.training_log_line(5, 10.3, 24.0, 118.0, 8.8, True); hb.training_log_line(6, 10.2, 24.1, 118.2, 8.8, True)'
      echo '8.8 11.0 '\$(date +%s) > \${MEMSNAP_OUT}/peak_rank7.txt
      sleep 1; exit 0
    fi"
}
source $HERE/../bin/restart_loop.sh
cp -r $HERE/../bin/*.py $HERE/../bin/oom_reconfig.sh $HERE/../bin/memsnap $T/bin/ 2>/dev/null
RES_BIN=$T/bin; cp $HERE/../bin/restart_loop.sh $T/bin/
res_main
echo "=== data/loop.log (grep)"; grep -E "memory gate|the OOM rule|memory profiler|restart [0-9]+ follows|dry run|latest_universal|converts the save" $JOB_DIR/data/loop.log | cut -c1-260
echo "=== recovery.log"; cat $JOB_DIR/recovery.log | cut -c1-400
echo "=== gate.log"; cat $JOB_DIR/gate.log | cut -c1-300
echo "=== cmdlines"; for k in 0 1 2; do echo "-- attempt $k"; cat $JOB_DIR/attempts/attempt_$k/cmdline.txt | cut -c1-330; done
echo "=== oom_reconfig.tsv"; cat $JOB_DIR/data/oom_reconfig.tsv | cut -c1-300
echo "=== attempt_layouts.tsv"; cat $JOB_DIR/data/attempt_layouts.tsv | cut -c1-300
echo "=== RESULTS.md (OOM section)"; sed -n '/## The OOM re-configuration/,/## The banners/p' $JOB_DIR/RESULTS.md | cut -c1-300
ok=1
grep -q "WARNING (the memory gate): the layout heatmap marks PP2 EP16 TP1 x U1 (d = 1, mbs 1) at 128k red" $JOB_DIR/gate.log || { echo "FAIL gate.log warning"; ok=0; }
grep -q "the rule:    upon the out of memory at step 1 on stage 1 (rank 300, HBM 63.80 GB allocated at the failure) in attempt 0, with 0 re-configuration(s) already taken in this job, the rule takes step 1: d aligned to EP: d 1 -> 16 (= EP 16); the split with the fewest bytes: TP1 x U16 at" $JOB_DIR/recovery.log || { echo "FAIL rule block 1"; ok=0; }
grep -q "no save exists yet (the out of memory came before the first save), so attempt 1 starts from the SAME initial state" $JOB_DIR/recovery.log || { echo "FAIL initial-state sentence"; ok=0; }
grep -q "the rule takes step 2: EP doubled alone: EP 16 -> 32" $JOB_DIR/recovery.log || { echo "FAIL rule block 2 (EP doubled alone)"; ok=0; }
grep -q "the universal checkpoint of step 3 is complete ('latest_universal' names global_step3_universal" $JOB_DIR/recovery.log || { echo "FAIL universal sentence"; ok=0; }
grep -q "memory profiler (attempt 0, layout PP2 EP16 TP1 x U1 (d = 1, mbs 1, dp_moe 16, dp_non_moe 256, GA 1)): the measured peak of this attempt is 59.4 GiB allocated on rank 300 (stage 1)" $JOB_DIR/recovery.log || { echo "FAIL profiler line of attempt 0"; ok=0; }
grep -q "memory profiler (attempt 1, layout PP2 EP16 TP1 x U16 (d = 16, mbs 1, dp_moe 16, dp_non_moe 16, GA 16)): the measured peak of this attempt is 58.8 GiB allocated on rank 5 (stage 0)" $JOB_DIR/recovery.log || { echo "FAIL profiler line of attempt 1"; ok=0; }
c1=$JOB_DIR/attempts/attempt_1/cmdline.txt; c2=$JOB_DIR/attempts/attempt_2/cmdline.txt
grep -q -- "--ds-sequence-parallel-size 16" $c1 && grep -q -- "--tensor-model-parallel-size 1 " $c1 && grep -q -- "--moe-expert-parallel-size 16 " $c1 && ! grep -q -- "--universal-checkpoint" $c1 && ! grep -q -- " --sequence-parallel" $c1 && grep -q "XMOE_MEGATRON_SP=unset" $c1 && grep -q "PATH_HEAD=$T/bin/memsnap" $c1 || { echo "FAIL attempt 1 command line"; ok=0; }
grep -q -- "--moe-expert-parallel-size 32 " $c2 && grep -q -- "--ds-sequence-parallel-size 16" $c2 && grep -q -- "--universal-checkpoint" $c2 && grep -q -- "--load $T/work/ckpt" $c2 && grep -q -- "--pipeline-model-parallel-size 2" $c2 && grep -q 'MBS_JSON="train_micro_batch_size_per_gpu": 1' $c2 || { echo "FAIL attempt 2 command line"; ok=0; }
[ "$(cat $CHECKPOINT_PATH/latest_universal)" = global_step3_universal ] && [ -f $CHECKPOINT_PATH/global_step3_universal/zero/optimizer_state.pt ] || { echo "FAIL latest_universal / the inline conversion"; ok=0; }
[ $(grep -c "^##### RECOVERED" $JOB_DIR/data/banners.txt) = 2 ] && grep -q "the OOM rule took step 1: PP2 EP16 TP1 x U1 (d = 1, mbs 1, dp_moe 16, dp_non_moe 256, GA 1) -> PP2 EP16 TP1 x U16" $JOB_DIR/data/banners.txt && grep -q "the OOM rule took step 2: .* -> PP2 EP32 TP1 x U16" $JOB_DIR/data/banners.txt || { echo "FAIL banners"; ok=0; }
[ $(grep -c -v '^#' $JOB_DIR/data/oom_reconfig.tsv) = 3 ] || { echo "FAIL oom_reconfig.tsv rows"; ok=0; }
[ $(grep -c -v '^#' $JOB_DIR/data/attempt_layouts.tsv) = 4 ] || { echo "FAIL attempt_layouts.tsv rows"; ok=0; }
grep -q "^300	1	59.4" $JOB_DIR/data/mem_peaks_a0.tsv || { echo "FAIL mem_peaks_a0 rank 300"; ok=0; }
grep -q "the first attempt ended with code 11 (gpu_oom): \*\*PASS\*\*" $JOB_DIR/RESULTS.md && grep -q "gate.log carries the memory gate's warning: \*\*PASS\*\*" $JOB_DIR/RESULTS.md && grep -q "recovery.log carries the rule's block" $JOB_DIR/RESULTS.md && grep -q "reached at least 2 steps with the profiler's peak under the 64 GiB of a GCD: \*\*PASS\*\*" $JOB_DIR/RESULTS.md || { echo "FAIL RESULTS.md checks"; ok=0; }
grep -q "the dry run of the new command line passed" $JOB_DIR/data/loop.log || { echo "FAIL dry run line"; ok=0; }
[ $(grep -c "^# ===== attempt" $JOB_DIR/training.log) = 3 ] || { echo "FAIL training.log attempt headers"; ok=0; }
[ $ok = 1 ] && echo "OOM MOCK: ALL PASS" || echo "OOM MOCK: SOME FAILED"
[ "${KEEP:-0}" = 1 ] || rm -rf $T
