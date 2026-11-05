#!/bin/bash
# Login-node check of the UCP worker duty of restart_loop.sh (2026-10-10) with a MOCK converter and a mock srun: no Slurm job, no GPU.
# The save folder holds fake ZeRO saves (global_step4, 8 at the start; 12, 16 appear while the watcher runs; 'latest' advances with them).
# The fake python answers `-m deepspeed.checkpoint.ds_to_universal ... --param-shard k/2` by writing the universal folder structure
# (zero/<atom>/, zero/optimizer_state.pt, shards/merge_k_of_2.done; the second share to finish writes shards/finalize.lock) after a
# short sleep; share 1 of step 8 FAILS on its first try (a marker file says so) and must be retried once on the other worker.
# Expected: 4 complete conversions recorded in data/ucp_conversions.tsv (steps 4, 8 with a retry, 12, 16), latest_universal naming
# global_step16_universal, the retention deleting global_step4 and global_step8 (the newest 2 ZeRO saves 12 and 16 stay), the
# "skipped for now" sentence when a save completed while the converter was busy, the ucp_state sentence, and res_ucp_finish stopping the
# watcher. Everything the real job does except Slurm and the real converter.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
T=${TMPD:-/tmp}/ucpmock_$$; mkdir -p $T/bin $T/work/logs $T/ckpt
REALPY=${RES_PY:-/lustre/orion/world-shared/gen150/zixianw4/envs/TORCH-ROCM6.4.1_env_groupgemm_v5/bin/python}
cat > $T/bin/srun <<'S'
#!/bin/bash
while [ $# -gt 0 ]; do case $1 in -w|--jobid) shift 2;; -*) shift;; *) break;; esac; done
exec "$@"
S
cat > $T/bin/fakepy <<S
#!/bin/bash
# the fake converter: python -m deepspeed.checkpoint.ds_to_universal --input_folder IN --output_folder OUT ... --param-shard k/n [--param-shard-balanced]
if [ "\$1" = -m ] && [ "\$2" = deepspeed.checkpoint.ds_to_universal ]; then
  shift 2; IN=""; OUT=""; SH=""
  while [ \$# -gt 0 ]; do case \$1 in --input_folder) IN=\$2; shift 2;; --output_folder) OUT=\$2; shift 2;; --param-shard) SH=\$2; shift 2;; *) shift;; esac; done
  k=\${SH%/*}; n=\${SH#*/}; step=\$(basename \$IN | sed 's/global_step//')
  echo "fake converter: step \$step share \$SH -> \$OUT on \$(hostname -s)"
  if [ "\$step" = 8 ] && [ "\$k" = 1 ] && [ ! -f \$OUT/.failed_once ]; then touch \$OUT/.failed_once; echo "fake converter: share 1 of step 8 FAILS on its first try"; exit 1; fi
  sleep 1
  mkdir -p \$OUT/zero/param_\$k \$OUT/shards; echo x > \$OUT/zero/param_\$k/fp32.pt
  date > \$OUT/shards/merge_\${k}_of_\${n}.done
  missing=0; for j in \$(seq 0 \$((n-1))); do [ -f \$OUT/shards/merge_\${j}_of_\${n}.done ] || missing=1; done
  if [ \$missing = 0 ]; then mkdir \$OUT/shards/finalize.lock 2>/dev/null && { echo x > \$OUT/zero/optimizer_state.pt; echo "fake converter: finalized"; }; fi
  exit 0
fi
exec $REALPY "\$@"
S
chmod +x $T/bin/*; export PATH=$T/bin:$PATH
cd $T/work
export SLURM_JOB_ID=998 RUN_TYPE=ep NODES=2 TOTAL_GPUS=16 gpus_per_node=8 PP_SIZE=1 EP_PARALLEL_SIZE=8 JOB_DIR=logs/job_998_mock
export RES_PY=$T/bin/fakepy SAVE_INTERVAL=4 RES_UCP_BALANCED=1
mkdir -p $JOB_DIR/data $JOB_DIR/attempts $JOB_DIR/alarms
source $HERE/../bin/restart_loop.sh
RES_BIN=$HERE/../bin
# the state the functions need (what res_main sets)
export RES_JOB_DIR=$PWD/$JOB_DIR; RES_DIR=$RES_JOB_DIR/data; export RES_DATA_DIR=$RES_DIR; RES_LOOP_LOG=$RES_DIR/loop.log; export RES_LOOP_LOG
: > $RES_JOB_DIR/recovery.log
RES_CKPT_DIR=$T/ckpt; RES_UCP_N=2; RES_UCP=(mockW1 mockW2); RES_END_EPOCH=$(( $(date +%s) + 600 ))
RES_ALL=(mockA mockB); RES_EXCLUDED=(); RES_DEGRADED=(); RES_ATTEMPT=0; RES_GROUP=8; RES_TRAIN_CSV=mockA,mockB; RES_DUTY=mockA; RES_NEXT=""
mk_save() { mkdir -p $RES_CKPT_DIR/global_step$1; echo x > $RES_CKPT_DIR/global_step$1/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt; echo global_step$1 > $RES_CKPT_DIR/latest; }
# the sandbox guard of the retention looks at the save dir's real path: put the fake save dir under the sandbox tree through a symlink
SB=/lustre/orion/gen150/scratch/zixianw4/X-MoE-resilience/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier/output/checkpoint
mkdir -p $SB; rm -rf $SB/res_ucpmock_$$; mkdir -p $SB/res_ucpmock_$$; RES_CKPT_DIR=$SB/res_ucpmock_$$
mk_save 4; mk_save 8
res_ucp_start
sleep 6; mk_save 12            # completes while the converter may be busy
sleep 6; mk_save 16
# wait until step 16 is converted (bounded)
for i in $(seq 1 60); do grep -q "^16	" $RES_DIR/ucp_conversions.tsv 2>/dev/null && break; sleep 1; done
res_ucp_finish
echo "=== loop.log"; cat $RES_DIR/loop.log | cut -c1-230
echo "=== ucp_conversions.tsv"; cat $RES_DIR/ucp_conversions.tsv
echo "=== ucp_state"; cat $RES_DIR/ucp_state
echo "=== save dir"; ls $RES_CKPT_DIR; cat $RES_CKPT_DIR/latest_universal
ok=1
[ "$(grep -c -E '	complete$' $RES_DIR/ucp_conversions.tsv)" = 4 ] || { echo "FAIL: 4 complete conversions expected"; ok=0; }
grep -q "^8	.*	0,0	" $RES_DIR/ucp_conversions.tsv && grep -q "share 1/2 of the conversion of step 8 failed on mockW2 with exit code 1" $RES_DIR/loop.log || { echo "FAIL: the retry of share 1 of step 8"; ok=0; }
[ "$(cat $RES_CKPT_DIR/latest_universal)" = global_step16_universal ] || { echo "FAIL: latest_universal"; ok=0; }
[ ! -d $RES_CKPT_DIR/global_step4 ] && [ ! -d $RES_CKPT_DIR/global_step8 ] && [ -d $RES_CKPT_DIR/global_step12 ] && [ -d $RES_CKPT_DIR/global_step16 ] || { echo "FAIL: retention of the ZeRO saves (4 and 8 deleted, 12 and 16 kept)"; ok=0; }
grep -q "retention: the ZeRO save of step 4 was deleted because its universal is complete" $RES_DIR/loop.log || { echo "FAIL: retention sentence"; ok=0; }
grep -q "universal of step 4 is complete: 2 atoms" $RES_DIR/loop.log || { echo "FAIL: the completeness sentence"; ok=0; }
grep -q "^step 16 universal complete" $RES_DIR/ucp_state || { echo "FAIL: ucp_state sentence: $(cat $RES_DIR/ucp_state)"; ok=0; }
grep -q "the watcher of the UCP worker duty stopped" $RES_DIR/loop.log || { echo "FAIL: the watcher did not stop"; ok=0; }
grep -q "balanced" $RES_DIR/ucp/convert_s4_share0.log && grep -q -- "--param-shard-balanced\|share 0/2" $RES_DIR/ucp/convert_s4_share0.log || true
$REALPY -c "import sys; sys.path.insert(0, '$RES_BIN'); import hb_aggregator as a; a.status_md('$RES_JOB_DIR')" && grep -q "^conversion     step 16 universal complete" $RES_JOB_DIR/STATUS.md || { echo "FAIL: STATUS.md conversion line"; ok=0; }
rm -rf $SB/res_ucpmock_$$
[ $ok = 1 ] && echo "UCP MOCK: ALL PASS" || echo "UCP MOCK: SOME FAILED"
