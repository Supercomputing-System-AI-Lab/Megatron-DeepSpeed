#!/bin/bash
# ftv2_launch_min.sh (ft-sys-v2, 2026-11-12): the part of local_autorun_frontier_0623_2026.sh (1,279 lines) that a ft-sys-v2 run
# uses, as one short script. Not used yet: ftv2_run_common.sh still calls the 0623 launcher. To switch later, replace that call
# (section 6 of ftv2_run_common.sh) with: bash "$FT/launch-min/ftv2_launch_min.sh"
#
# What it does, in order:
#   1. checks it runs from examples_elmoe/scripts-frontier (the job script's relative paths start there)
#   2. refuses the 0623 launcher's settings it does not carry (resume, saves, UCP, the pipeline-strategy path, ...)
#   3. reads the layout: AUTORUN_EP_KEY "nodes:EP:TP" and AUTORUN_EP_CMD "mbs:nbs:iters:X-MOE:<model>:ckpt:n:<zero>",
#      plus AUTORUN_PP_SIZE (pipeline stages, default 1), and checks it (8 GPUs per node; PP, EP and TP divide the GPUs)
#   4. fills the job template (ftv2_job_min.slurm.template) into temp_slurm/ftv2min_<name>.slurm and checks no field is left
#   5. submits it with AUTORUN_SBATCH_CMD (ftv2_sbatch.sh in recovery mode: + standby nodes, --no-kill), else sbatch
#
# Reads the same settings ftv2_run_common.sh exports for the 0623 launcher:
#   AUTORUN_EP_KEY, AUTORUN_EP_CMD   the layout (required)
#   AUTORUN_PP_SIZE                  pipeline stages (default 1)
#   AUTORUN_SEQ_LEN                  overrides the model registry's sequence length
#   AUTORUN_MEGATRON_SP=1            Megatron sequence parallelism (--sequence-parallel and XMOE_MEGATRON_SP=1 together)
#   AUTORUN_QOS (debug), AUTORUN_WALLTIME (00:30:00), AUTORUN_EXIT_MINS (25), AUTORUN_NCCL_DEBUG (WARN)
#   AUTORUN_PRE_CMD                  the command run in the job before training (required: the new stack's network settings and
#                                    libraries, the heartbeat, the srun wrapper)
#   AUTORUN_SBATCH_CMD               how to submit (default sbatch)
#   AUTORUN_NO_CKPT, AUTORUN_NO_PROFILING, AUTORUN_DS_CONFIG_PER_JOB   always on in this job; 0 is refused
#   FTV2_VOCAB_FILE, DATA_PATH       the tokenizer file and the data prefix (required: the minimal job has no defaults)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TEMPLATE=${FTV2_MIN_TEMPLATE:-$HERE/ftv2_job_min.slurm.template}
die() { echo "FATAL [ftv2_launch_min]: $*"; exit 1; }

# 1. Where
for f in ../ELMoE_launch.py ../utils/model_registry.py ../ds_config_gpt_TEMPLATE.json monitor_run.sh "$TEMPLATE"; do
  [ -f "$f" ] || die "$f not found: run from examples_elmoe/scripts-frontier (now in $PWD)"
done

# 2. What the minimal job does not carry
for v in AUTORUN_PP_KEY AUTORUN_PP_STRATEGY AUTORUN_PP_CMD AUTORUN_TEMPLATE AUTORUN_RESILIENT AUTORUN_SPARES AUTORUN_GATE \
         AUTORUN_LOAD_DIR AUTORUN_SAVE_DIR AUTORUN_SAVE_INTERVAL AUTORUN_INIT_LOAD_DIR AUTORUN_INIT_UNIVERSAL AUTORUN_NATIVE_RESUME \
         AUTORUN_UCP_RESUME AUTORUN_UCP_FINETUNE AUTORUN_UCP_WORKERS AUTORUN_WALL_AWARE_EXIT AUTORUN_DATA_PATH AUTORUN_GPT2_TOKENIZER \
         AUTORUN_DEPENDENCY AUTORUN_NODELIST AUTORUN_POST_CMD AUTORUN_CAPACITY_ROUTED AUTORUN_SEQ_CHUNKED_CE AUTORUN_CKPT_IN_CPU \
         AUTORUN_PARTITION_ACT AUTORUN_DS_SEQUENCE_PARALLEL_SIZE AUTORUN_NO_FP32_RESIDUAL AUTORUN_PLANNER_MEM_GB; do
  [ -z "${!v:-}" ] || die "$v is set, and the minimal job does not carry it: use local_autorun_frontier_0623_2026.sh"
done
for v in AUTORUN_NO_CKPT AUTORUN_NO_PROFILING AUTORUN_DS_CONFIG_PER_JOB; do
  [ "${!v:-1}" = 1 ] || die "$v=${!v}: the minimal job always has it on"
done

# 3. The layout
IFS=: read -r NODES EP TP <<< "${AUTORUN_EP_KEY:?set AUTORUN_EP_KEY=nodes:EP:TP}"
cmd=$(echo "${AUTORUN_EP_CMD:?set AUTORUN_EP_CMD=mbs:nbs:iters:X-MOE:<model>:ckpt:n:<zero>}" | tr -d ' ')
IFS=: read -r BS NBS ITERS MOE MODEL CKPT CKPT_LAYERS ZERO <<< "$cmd"
[ "$MOE" = X-MOE ] || die "MoE type '$MOE': the minimal job is X-MOE only"
PP=${AUTORUN_PP_SIZE:-1}
TOTAL=$(( NODES * 8 ))
(( TOTAL % PP == 0 )) || die "PP=$PP does not divide $TOTAL GPUs"
STRIDE=$(( TOTAL / PP ))
(( EP >= 1 && STRIDE % EP == 0 )) || die "EP=$EP does not divide $STRIDE GPUs per pipeline stage"
(( TP >= 1 && STRIDE % TP == 0 )) || die "TP=$TP does not divide $STRIDE GPUs per pipeline stage"
[ "$CKPT" = ckpt ] && ACT=true || ACT=false
: "${FTV2_VOCAB_FILE:?set FTV2_VOCAB_FILE (the tokenizer file; the minimal job has no default)}"
: "${DATA_PATH:?set DATA_PATH (the data prefix, .bin/.idx; the minimal job has no default)}"
: "${AUTORUN_PRE_CMD:?set AUTORUN_PRE_CMD (ftv2_run_common.sh builds it)}"
NAME="ep_n${NODES}g${TOTAL}_pp${PP}_ep${EP}_nbs${NBS}_i${ITERS}_X-MOE"
echo "[ftv2_launch_min] $NAME: $NODES nodes x 8 GPUs, PP $PP, EP $EP, TP $TP, DP_non_moe $(( STRIDE / TP )), DP_moe $(( STRIDE / EP ))," \
     "mbs $BS x $NBS, $ITERS iterations, model $MODEL, activation checkpointing $ACT/$CKPT_LAYERS, ZeRO $ZERO"

# 4. Fill the template
mkdir -p temp_slurm logs
out="temp_slurm/ftv2min_${NAME}.slurm"
SEQ_LINE=""; [ -n "${AUTORUN_SEQ_LEN:-}" ] && SEQ_LINE="SEQ_LEN=${AUTORUN_SEQ_LEN}   # AUTORUN_SEQ_LEN: overrides the registry value"
SP_OPT=""; SP_ENV=""
if [ "${AUTORUN_MEGATRON_SP:-0}" = 1 ]; then
  SP_OPT='megatron_options="${megatron_options} --sequence-parallel"   # AUTORUN_MEGATRON_SP'
  SP_ENV='export XMOE_MEGATRON_SP=1   # AUTORUN_MEGATRON_SP: required with --sequence-parallel'
fi
# Every field is passed as an argument pair and replaced as plain text (no shell or sed quoting of the values). PRE_CMD becomes two
# lines: an echo of the command (with the job's date), then the command itself.
python3 - "$TEMPLATE" "$out" \
  NODES "$NODES" TOTAL_GPUS "$TOTAL" BATCH_SIZE "$BS" NUM_BATCHES "$NBS" PP_SIZE "$PP" EP_PARALLEL_SIZE "$EP" MP_SIZE "$TP" \
  TRAIN_ITERS "$ITERS" MODEL_SIZE "$MODEL" ACTIVATION_CHECKPOINT "$ACT" CHECKPOINT_NUM_LAYERS "$CKPT_LAYERS" ZERO "$ZERO" \
  WALLTIME "${AUTORUN_WALLTIME:-00:30:00}" QOS "${AUTORUN_QOS:-debug}" EXIT_MINS "${AUTORUN_EXIT_MINS:-25}" \
  NCCL_DEBUG "${AUTORUN_NCCL_DEBUG:-WARN}" VOCAB_FILE "$FTV2_VOCAB_FILE" DATA_PATH "$DATA_PATH" \
  SEQ_LEN_OVERRIDE "$SEQ_LINE" SP_OPTION "$SP_OPT" SP_ENV "$SP_ENV" PRE_CMD "$AUTORUN_PRE_CMD" <<'PY'
import sys
src, dst, kv = sys.argv[1], sys.argv[2], sys.argv[3:]
s = open(src).read()
bad = [l for l in s.splitlines() if l.lstrip().startswith('#') and not l.startswith('#SBATCH') and '{{' in l]
assert not bad, 'a field inside a comment (a multi-line value would leave the comment): ' + bad[0]
for k, v in zip(kv[0::2], kv[1::2]):
    if k == 'PRE_CMD':
        v = 'echo "[PRE] $(date) running: ' + v.replace('"', '\\"') + '"\n' + v + '   # [PRE]'
    assert '{{' + k + '}}' in s, 'field {{' + k + '}} is not in the template'
    s = s.replace('{{' + k + '}}', v)
assert '{{' not in s, 'a field was left: ' + s[s.index('{{'):s.index('{{') + 40]
open(dst, 'w').write(s)
PY
echo "[ftv2_launch_min] rendered $out"

# 5. Submit
${AUTORUN_SBATCH_CMD:-sbatch} "$out"
