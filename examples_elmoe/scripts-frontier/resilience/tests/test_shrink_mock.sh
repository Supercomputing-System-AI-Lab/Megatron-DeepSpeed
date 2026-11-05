#!/bin/bash
# test_shrink_mock.sh (2026-10-11, the user's rule "if no spare node is left, run the memory model and use a configuration with fewer GPUs
# instead of stopping"): the restart loop with AUTORUN_SHRINK=1 on the login node with a MOCK trainer, a mock srun and a mock converter.
# 64 training nodes and NO spare (mock000..mock063, 512 ranks), DeepSeek_16B at 128k, PP2 EP16 TP1 x U16 mbs 1, 256 sequences per step.
#   attempt 0: two steps, the save of step 2 ('latest' = global_step2), then a HIP error on rank 80 of mock010 (code 16, which excludes the
#              node) -> 63 healthy nodes for 64: the memory model must choose 32 nodes with the same layout (the 256 sequences per step split
#              only over a power-of-two data-parallel width), the loop converts the save on the batch node (the fake converter), and
#              attempt 1 must run on 32 nodes (256 ranks) with --universal-checkpoint --load <the save dir>.
#   attempt 1: two steps, the save of step 4 under the new layout, then host memory above 97 percent (code 19): restart on the same 32 nodes
#              from the NATIVE save of step 4 (no --universal-checkpoint), 1 retry allowed.
#   attempt 2: one step, host memory again: NO RESTART (host_oom seen twice, one restart allowed).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
T=${TMPD:-/tmp}/shrinkmock_$$; mkdir -p $T/bin $T/work/logs $T/work/ckpt
REALPY=${RES_PY:-/lustre/orion/world-shared/gen150/zixianw4/envs/TORCH-ROCM6.4.1_env_groupgemm_v5/bin/python}
cat > $T/bin/scontrol <<'S'
#!/bin/bash
if [ "$1 $2" = "show hostnames" ]; then for i in $(seq 0 63); do printf 'mock%03d\n' $i; done; exit 0; fi
if [ "$1 $2" = "show job" ]; then echo "JobId=1 EndTime=$(date -d '+40 min' +%Y-%m-%dT%H:%M:%S)"; exit 0; fi
S
cat > $T/bin/srun <<'S'
#!/bin/bash
while [ $# -gt 0 ]; do case $1 in -w|--jobid) shift 2;; -*) shift;; *) break;; esac; done
exec "$@"
S
printf '#!/bin/bash\nexit 0\n' > $T/bin/sinfo
cat > $T/bin/fakepy <<S
#!/bin/bash
if [ "\$1" = -m ] && [ "\$2" = deepspeed.checkpoint.ds_to_universal ]; then
  shift 2; IN=""; OUT=""
  while [ \$# -gt 0 ]; do case \$1 in --input_folder) IN=\$2; shift 2;; --output_folder) OUT=\$2; shift 2;; *) shift;; esac; done
  echo "fake converter: \$IN -> \$OUT"; sleep 1
  mkdir -p \$OUT/zero/param_0; echo x > \$OUT/zero/param_0/fp32.pt; echo x > \$OUT/zero/optimizer_state.pt; exit 0
fi
exec $REALPY "\$@"
S
chmod +x $T/bin/*; export PATH=$T/bin:$PATH
cd $T/work
export SLURM_JOB_ID=996 SLURM_JOB_NODELIST=mock RUN_TYPE=ep NODES=64 TOTAL_GPUS=512 gpus_per_node=8 PP_SIZE=2 EP_PARALLEL_SIZE=16 MP_SIZE=1 BATCH_SIZE=1 NUM_BATCHES=1
export GLOBAL_BATCH_SIZE=256 SEQ_LEN=131072 MODEL_SIZE=DeepSeek_16B ACTIVATION_CHECKPOINT=true DS_SEQ_PAR_SIZE=16
export JOB_DIR=logs/job_996_mock AUTORUN_GATE=0 RES_PY=$T/bin/fakepy RES_STILL_S=5 XMOE_T_INIT_S=60 RES_MIN_ATTEMPT_S=60
export AUTORUN_SHRINK=1 XMOE_INJECT_SCHEDULE=""
mkdir -p $JOB_DIR/data/gate; : > $JOB_DIR/a-xmoe-ep-996.e
export CHECKPOINT_PATH=$T/work/ckpt LOAD_CHECKPOINT_PATH=$T/work/ckpt
cat > $T/work/ds.json <<'J'
{ "train_batch_size": 256, "train_micro_batch_size_per_gpu": 1, "zero_optimization": {"stage": 1} }
J
config_json=$T/work/ds.json
megatron_options=" --moe-expert-parallel-size 16 --num-experts 64 --tensor-model-parallel-size 1 --ds-sequence-parallel-size 16 --micro-batch-size 1 --global-batch-size 256 --seq-length 131072 --seq-chunked-ce 2048 --checkpoint-activations --checkpoint-num-layers 1 --disable-moe-token-dropping --use-flash-attn-v2 --save $CHECKPOINT_PATH --load $LOAD_CHECKPOINT_PATH --use-rotary-position-embeddings"
deepspeed_options=" --deepspeed --deepspeed_config $config_json --pipeline-model-parallel-size 2 --deepspeed-activation-checkpointing"
xmoe_dry_run_cmd() { echo true; }
merge_full_run() { { echo "mock .o"; cat "$1/rank_0.log" 2>/dev/null; } > "$2"; }
xmoe_train_srun() {   # the mock: records the command line and the node count it would run with, then plays its attempt's script
  local a=$RES_ATTEMPT J=$PWD/$JOB_DIR R=$RANK_LOG_DIR last=$(( $2 * 8 - 1 ))
  { echo "ATTEMPT $a"; echo "NODES_ARG=$2"; echo "NODES_CSV_COUNT=$(echo $1 | tr ',' '\n' | wc -l)"; echo "TOTAL_GPUS=$TOTAL_GPUS"; echo "MEGATRON_OPTIONS=$megatron_options"; echo "DEEPSPEED_OPTIONS=$deepspeed_options"; } > "$RES_ATT_DIR/cmdline.txt"
  exec bash -c "
    cd $RES_BIN
    if [ $a = 0 ]; then
      sleep 2; echo ' iteration        1/      24 | consumed samples' > $R/rank_$last.log; echo 'rank 0 via NET/AWS Libfabric/0' > $R/rank_0.log
      $REALPY -c 'import hb; hb._ctx.update(rank=$last, job_dir=\"$J\", attempt=\"$a\", resumed_from=0); hb.training_log_line(1, 10.5, 25.1, 120.4, 9.9, True); hb.training_log_line(2, 10.4, 25.0, 119.9, 9.9, True)'
      mkdir -p $CHECKPOINT_PATH/global_step2; echo x > $CHECKPOINT_PATH/global_step2/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt; echo global_step2 > $CHECKPOINT_PATH/latest
      sleep 1
      echo '{\"code\": 16, \"class\": \"driver_hip_error\", \"rank\": 80, \"node\": \"mock010\", \"step\": 3, \"phase\": \"forward\", \"time\": '\$(date +%s)', \"attempt\": \"0\", \"evidence\": \"RuntimeError: HIP error: an illegal memory access was encountered\"}' > $RES_EXIT_DIR/r80.json
      exit 16
    elif [ $a = 1 ]; then
      sleep 2; echo ' iteration        3/      24 | consumed samples' > $R/rank_$last.log; echo 'rank 0 via NET/AWS Libfabric/0' > $R/rank_0.log
      $REALPY -c 'import hb; hb._ctx.update(rank=$last, job_dir=\"$J\", attempt=\"$a\", resumed_from=2); hb.training_log_line(3, 10.3, 24.0, 118.0, 8.8, True); hb.training_log_line(4, 10.2, 24.1, 118.2, 8.8, True)'
      mkdir -p $CHECKPOINT_PATH/global_step4; echo x > $CHECKPOINT_PATH/global_step4/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt; echo global_step4 > $CHECKPOINT_PATH/latest
      sleep 1
      echo '{\"code\": 19, \"class\": \"host_oom\", \"rank\": 3, \"node\": \"mock000\", \"step\": 5, \"phase\": \"forward\", \"time\": '\$(date +%s)', \"attempt\": \"1\", \"evidence\": \"host memory 98.0 percent on mock000 (an injected test reading, case hostmem)\"}' > $RES_EXIT_DIR/r3.json
      exit 19
    else
      sleep 2; echo ' iteration        5/      24 | consumed samples' > $R/rank_$last.log; echo 'rank 0 via NET/AWS Libfabric/0' > $R/rank_0.log
      $REALPY -c 'import hb; hb._ctx.update(rank=$last, job_dir=\"$J\", attempt=\"$a\", resumed_from=4); hb.training_log_line(5, 10.1, 24.0, 118.0, 8.8, True)'
      sleep 1
      echo '{\"code\": 19, \"class\": \"host_oom\", \"rank\": 3, \"node\": \"mock000\", \"step\": 6, \"phase\": \"forward\", \"time\": '\$(date +%s)', \"attempt\": \"2\", \"evidence\": \"host memory 98.0 percent on mock000 (an injected test reading, case hostmem)\"}' > $RES_EXIT_DIR/r3.json
      exit 19
    fi"
}
source $HERE/../bin/restart_loop.sh
cp -r $HERE/../bin/*.py $HERE/../bin/oom_reconfig.sh $HERE/../bin/memsnap $T/bin/ 2>/dev/null
RES_BIN=$T/bin; cp $HERE/../bin/restart_loop.sh $T/bin/
res_main
echo "=== data/loop.log (grep)"; grep -E "no spare|restart [0-9]+ follows|latest_universal|converts the save|classified" $JOB_DIR/data/loop.log | cut -c1-300
echo "=== recovery.log"; cat $JOB_DIR/recovery.log | cut -c1-500
echo "=== cmdlines"; for k in 0 1 2; do echo "-- attempt $k"; cut -c1-300 $JOB_DIR/attempts/attempt_$k/cmdline.txt; done
echo "=== shrink.tsv"; cat $JOB_DIR/data/shrink.tsv | cut -c1-300
echo "=== training.log"; cut -c1-260 $JOB_DIR/training.log
ok=1
grep -q "no spare:    no spare node is left and 63 healthy nodes remain for the 64 the run was planned on: the memory model's largest fitting choice is 32 nodes (256 GPUs) with PP2 EP16 TP1 x U16" $JOB_DIR/recovery.log || { echo "FAIL the shrink sentence"; ok=0; }
c1=$JOB_DIR/attempts/attempt_1/cmdline.txt
grep -q "NODES_ARG=32" $c1 && grep -q "NODES_CSV_COUNT=32" $c1 && grep -q "TOTAL_GPUS=256" $c1 && grep -q -- "--universal-checkpoint" $c1 && grep -q -- "--load $T/work/ckpt" $c1 && grep -q -- "--moe-expert-parallel-size 16 " $c1 && grep -q -- "--ds-sequence-parallel-size 16" $c1 || { echo "FAIL attempt 1 command line"; ok=0; }
! grep -q "mock010" <(grep -o 'NODES_CSV.*' $c1) || { echo "FAIL the excluded node is in the training set"; ok=0; }
[ "$(cat $CHECKPOINT_PATH/latest_universal 2>/dev/null)" = global_step2_universal ] || { echo "FAIL latest_universal"; ok=0; }
[ $(grep -c -v '^#' $JOB_DIR/data/shrink.tsv) = 2 ] || { echo "FAIL shrink.tsv rows"; ok=0; }
grep -q "^# no spare node was left: attempt 1 runs on 32 of the 63 healthy nodes" $JOB_DIR/training.log || { echo "FAIL training.log line"; ok=0; }
c2=$JOB_DIR/attempts/attempt_2/cmdline.txt
grep -q "NODES_ARG=32" $c2 && ! grep -q -- "--universal-checkpoint" $c2 || { echo "FAIL attempt 2 must run on the 32 nodes and load the native save of step 4 (no --universal-checkpoint)"; ok=0; }
grep -q "the next attempt loads the native save of step 4, written under the current layout, instead of the universal checkpoint of step 2" $JOB_DIR/recovery.log || { echo "FAIL the native-save sentence"; ok=0; }
grep -q "NO RESTART: the class host_oom was seen 2 time(s) and only 1 restart(s) are allowed for it" $JOB_DIR/recovery.log || { echo "FAIL the host-memory stop after the second time"; ok=0; }
[ ! -d $JOB_DIR/attempts/attempt_3 ] || { echo "FAIL a fourth attempt started"; ok=0; }
[ $ok = 1 ] && echo "SHRINK MOCK: ALL PASS" || echo "SHRINK MOCK: SOME FAILED"
[ "${KEEP:-0}" = 1 ] || rm -rf $T
