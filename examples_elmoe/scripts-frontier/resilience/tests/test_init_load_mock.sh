#!/bin/bash
# test_init_load_mock.sh (2026-10-13): the restart loop with RES_INIT_LOAD_DIR (the launcher rider AUTORUN_INIT_LOAD_DIR) and the OOM rule, on the
# login node with a MOCK trainer and srun (the harness of test_oom_mock.sh): 8 training nodes mock000..mock007, DeepSeek_16B_32k at 262,144,
# PP1 EP8 TP8 + SP x U1, mbs 1, 8 sequences per step, intra-document attention ON (the layout of the 2026-10-13 data-quality probe).
#   attempt 0: --load <init> --finetune --universal-checkpoint; OOM (rank 3, step 1) BEFORE any save -> the rule must NOT pick Ulysses (intra-doc):
#              step 2 doubles EP alone (EP 16, TP 8, U 1); no save exists, so attempt 1 starts again from the initial checkpoint.
#   attempt 1: --load <init> --finetune --universal-checkpoint --moe-expert-parallel-size 16; two steps, a save of step 3, then a NaN-loss stop (13).
#   attempt 2: a native resume of the run's own save: --load <save dir>, no --finetune, no --universal-checkpoint, EP 16 kept; planned exit.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
T=${TMPD:-/tmp}/initmock_$$; mkdir -p $T/bin $T/work/logs $T/work/ckpt $T/work/init/global_step1000_universal/zero/p
echo global_step1000_universal > $T/work/init/latest_universal; echo x > $T/work/init/global_step1000_universal/zero/optimizer_state.pt
REALPY=${RES_PY:-/lustre/orion/world-shared/gen150/zixianw4/envs/TORCH-ROCM6.4.1_env_groupgemm_v5/bin/python}
cat > $T/bin/scontrol <<'S'
#!/bin/bash
if [ "$1 $2" = "show hostnames" ]; then for i in $(seq 0 7); do printf 'mock%03d\n' $i; done; exit 0; fi
if [ "$1 $2" = "show job" ]; then echo "JobId=1 EndTime=$(date -d '+40 min' +%Y-%m-%dT%H:%M:%S)"; exit 0; fi
S
cat > $T/bin/srun <<'S'
#!/bin/bash
while [ $# -gt 0 ]; do case $1 in -w|--jobid) shift 2;; -*) shift;; *) break;; esac; done
exec "$@"
S
printf '#!/bin/bash\nexit 0\n' > $T/bin/sinfo
printf '#!/bin/bash\nexec %s "$@"\n' "$REALPY" > $T/bin/fakepy
chmod +x $T/bin/*; export PATH=$T/bin:$PATH
cd $T/work
export SLURM_JOB_ID=996 SLURM_JOB_NODELIST=mock RUN_TYPE=ep NODES=8 TOTAL_GPUS=64 gpus_per_node=8 PP_SIZE=1 EP_PARALLEL_SIZE=8 MP_SIZE=8 BATCH_SIZE=1 NUM_BATCHES=1
export GLOBAL_BATCH_SIZE=8 SEQ_LEN=262144 MODEL_SIZE=DeepSeek_16B_32k ACTIVATION_CHECKPOINT=true DS_SEQ_PAR_SIZE=1 XMOE_MEGATRON_SP=1
export JOB_DIR=logs/job_996_mock AUTORUN_GATE=0 RES_PY=$T/bin/fakepy RES_STILL_S=5 XMOE_T_INIT_S=60 RES_MIN_ATTEMPT_S=60
export AUTORUN_OOM_RECONFIG=1 AUTORUN_MEM_PROFILER=0 XMOE_INJECT_SCHEDULE=""
export RES_INIT_LOAD_DIR=$T/work/init RES_INIT_UNIVERSAL=1
mkdir -p $JOB_DIR/data/gate; : > $JOB_DIR/a-xmoe-ep-996.e
export CHECKPOINT_PATH=$T/work/ckpt LOAD_CHECKPOINT_PATH=$T/work/ckpt
printf '{ "train_batch_size": 8, "train_micro_batch_size_per_gpu": 1, "zero_optimization": {"stage": 1} }\n' > $T/work/ds.json
config_json=$T/work/ds.json
megatron_options=" --moe-expert-parallel-size 8 --num-experts 64 --tensor-model-parallel-size 8 --micro-batch-size 1 --global-batch-size 8 --seq-length 262144 --seq-chunked-ce 2048 --checkpoint-activations --checkpoint-num-layers 1 --disable-moe-token-dropping --use-flash-attn-v2 --save $CHECKPOINT_PATH --load $LOAD_CHECKPOINT_PATH --use-rotary-position-embeddings --intra-document-attention --sequence-parallel"
deepspeed_options=" --deepspeed --deepspeed_config $config_json --no-pipeline-parallel --deepspeed-activation-checkpointing"
xmoe_dry_run_cmd() { echo true; }
merge_full_run() { { echo "mock .o"; cat "$1/rank_0.log" "$1/rank_63.log" 2>/dev/null; } > "$2"; }
xmoe_train_srun() {
  local a=$RES_ATTEMPT J=$PWD/$JOB_DIR R=$RANK_LOG_DIR
  { echo "ATTEMPT $a"; echo "MEGATRON_OPTIONS=$megatron_options"; echo "DEEPSPEED_OPTIONS=$deepspeed_options"; } > "$RES_ATT_DIR/cmdline.txt"
  exec bash -c "
    cd $RES_BIN
    if [ $a = 0 ]; then
      sleep 2
      echo '{\"code\": 11, \"class\": \"gpu_oom\", \"rank\": 3, \"node\": \"mock000\", \"step\": 1, \"phase\": \"forward\", \"time\": '\$(date +%s)', \"attempt\": \"0\", \"evidence\": \"OutOfMemoryError: HIP out of memory | hbm_gb=63.80\"}' > $RES_EXIT_DIR/r3.json
      exit 11
    elif [ $a = 1 ]; then
      sleep 2; echo ' iteration        2/      24 | consumed samples' > $R/rank_63.log; echo 'rank 0 via NET/AWS Libfabric/0' > $R/rank_0.log
      mkdir -p $CHECKPOINT_PATH/global_step3; echo x > $CHECKPOINT_PATH/global_step3/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt; echo global_step3 > $CHECKPOINT_PATH/latest
      sleep 1
      echo '{\"code\": 13, \"class\": \"loss_nan\", \"rank\": 63, \"node\": \"mock007\", \"step\": 4, \"phase\": \"step\", \"time\": '\$(date +%s)', \"attempt\": \"1\", \"evidence\": \"loss nan 3 in a row\"}' > $RES_EXIT_DIR/r63.json
      exit 13
    else
      sleep 2; echo ' iteration        4/      24 | consumed samples' > $R/rank_63.log; echo 'rank 0 via NET/AWS Libfabric/0' > $R/rank_0.log
      sleep 1; exit 0
    fi"
}
source $HERE/../bin/restart_loop.sh
cp -r $HERE/../bin/*.py $HERE/../bin/oom_reconfig.sh $T/bin/ 2>/dev/null
RES_BIN=$T/bin; cp $HERE/../bin/restart_loop.sh $T/bin/
res_main
echo "=== loop.log (init lines)"; grep -E "initial checkpoint|save directory|the OOM rule|restart [0-9]+ follows" $JOB_DIR/data/loop.log | cut -c1-300
echo "=== recovery.log"; grep -E "the rule:|decision:" $JOB_DIR/recovery.log | cut -c1-500
for k in 0 1 2; do echo "-- attempt $k"; cut -c1-600 $JOB_DIR/attempts/attempt_$k/cmdline.txt; done
c0=$JOB_DIR/attempts/attempt_0/cmdline.txt; c1=$JOB_DIR/attempts/attempt_1/cmdline.txt; c2=$JOB_DIR/attempts/attempt_2/cmdline.txt
ok=1
grep -q -- "--load $T/work/init " $c0 && grep -q -- " --finetune" $c0 && grep -q -- "--universal-checkpoint" $c0 || { echo "FAIL attempt 0 loads the initial checkpoint (universal, --finetune)"; ok=0; }
grep -q -- "--load $T/work/init " $c1 && grep -q -- " --finetune" $c1 && grep -q -- "--universal-checkpoint" $c1 && grep -q -- "--moe-expert-parallel-size 16 " $c1 && grep -q -- "--tensor-model-parallel-size 8 " $c1 && grep -q -- "--ds-sequence-parallel-size 1" $c1 && grep -q -- " --sequence-parallel" $c1 || { echo "FAIL attempt 1 (EP 16 alone, TP 8, no Ulysses, the initial checkpoint again)"; ok=0; }
grep -q -- "--load $T/work/ckpt " $c2 && ! grep -q -- " --finetune" $c2 && ! grep -q -- "--universal-checkpoint" $c2 && grep -q -- "--moe-expert-parallel-size 16 " $c2 || { echo "FAIL attempt 2 (native resume of the run's own save of step 3)"; ok=0; }
[ $(grep -c -- "--load" $c2) = 1 ] && [ $(grep -o -- "--finetune" $c1 | wc -l) = 1 ] && [ $(grep -o -- "--universal-checkpoint" $c1 | wc -l) = 1 ] || { echo "FAIL duplicated flags"; ok=0; }
grep -q "EP doubled alone: EP 8 -> 16" $JOB_DIR/recovery.log && ! grep -q "U16" $JOB_DIR/recovery.log || { echo "FAIL the rule's rung (EP alone, no Ulysses)"; ok=0; }
grep -q "the initial checkpoint $T/work/init, weights only with --finetune, in its universal form" $JOB_DIR/recovery.log || { echo "FAIL the initial-state sentence of the OOM block"; ok=0; }
[ $ok = 1 ] && echo "INIT-LOAD MOCK: ALL PASS" || echo "INIT-LOAD MOCK: SOME FAILED"
[ "${KEEP:-0}" = 1 ] || rm -rf $T
