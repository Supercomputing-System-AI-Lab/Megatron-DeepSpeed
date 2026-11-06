# ftv2_run_common.sh (ft-sys-v2, 2026-11-06): the shared part of the example run files in this folder. Sourced by ftv2_<model>.sh
# after it sets the model's layout. It submits one job through the 0623 launcher (never edited) with ft-sys-v2 on:
#   MODE=recovery   heartbeat + warm recovery (milestone 1): + FTV2_STANDBY_NODES standby node(s), one Slurm step per arm
#                   (recovery/launch/ftv2_srun.sh), Keep with the CPU shadow (torch fused AdamW). Models with PP 1 only.
#   MODE=heartbeat  heartbeat only (models with a pipeline, milestone 3 not built): the failure list is written and the job ends
#                   FTV2_ABORT_ON_FAILURE seconds later; restart by resubmitting with the resume riders (ft-sys-v1's path:
#                   AUTORUN_LOAD_DIR/AUTORUN_SAVE_DIR with AUTORUN_NATIVE_RESUME=1 or AUTORUN_UCP_RESUME=1).
# The model's run file sets: EX_LABEL, EX_EP_KEY ("nodes:EP:TP"), EX_EP_CMD (" mbs:nbs:iters:X-MOE:<model>:ckpt:1:<zero>"),
# optional EX_PP_SIZE, EX_SEQ_LEN, EX_MEGATRON_SP, EX_STRIDE (partner stride = ranks per data-parallel replica), EX_WALL, EX_QOS,
# EX_MODE. RENDER_ONLY=1 renders the job script (and the standby hook's copy) without submitting.
# Paths: FTV2_TREE (this worktree: megatron and ft-sys-v2), FTV2_SF (the scripts-frontier folder that holds the data, the DeepSpeed
# json template and the launcher; default the user's checkout, where every X-MoE run is launched from), the new stack's env file.
set -euo pipefail
FTV2_TREE=${FTV2_TREE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
FTV2_SF=${FTV2_SF:-/lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier}
FTV2_ENV=${FTV2_ENV:-/lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/assistant_deep_dive/2026-11-04-xmoe_newstack/runs/newstack_env.sh}
FT=$FTV2_TREE/examples_elmoe/scripts-frontier/ft-sys-v2
cd "$FTV2_SF"
set +e +u; . ./eval/submit_env.sh; . "$FTV2_ENV"; set -euo pipefail
export CXX=$(which g++) CC=$(which gcc)
export PYTHONPATH=$FTV2_TREE:${PYTHONPATH:-}
python -I -B -c "import sys; sys.path.insert(0, '$FTV2_TREE'); import megatron; assert megatron.__file__.startswith('$FTV2_TREE'), megatron.__file__" \
  || { echo "FATAL: megatron does not import from $FTV2_TREE"; exit 1; }
export FTV2_PY=$(command -v python) FTV2_ETCD_BIN=${FTV2_ETCD_BIN:-/lustre/orion/world-shared/gen150/zixianw4/tools/etcd-v3.5.17/etcd-v3.5.17-linux-amd64/etcd}
export AUTORUN_EP_KEY="$EX_EP_KEY" AUTORUN_EP_CMD="$EX_EP_CMD"
[ -n "${EX_PP_SIZE:-}" ] && export AUTORUN_PP_SIZE=$EX_PP_SIZE
[ -n "${EX_SEQ_LEN:-}" ] && export AUTORUN_SEQ_LEN=$EX_SEQ_LEN
[ -n "${EX_MEGATRON_SP:-}" ] && export AUTORUN_MEGATRON_SP=$EX_MEGATRON_SP
export AUTORUN_QOS=${EX_QOS:-debug} AUTORUN_WALLTIME="${EX_WALL:-00:30:00}" AUTORUN_EXIT_MINS=${EX_EXIT_MINS:-25}
export AUTORUN_NO_CKPT=${EX_NO_CKPT:-1} AUTORUN_NO_PROFILING=1 AUTORUN_NCCL_DEBUG=WARN AUTORUN_DS_CONFIG_PER_JOB=1
export XMOE_FT_V2=1
PRE="ulimit -c 0; export TORCH_NCCL_ASYNC_ERROR_HANDLING=0 NCCL_ASYNC_ERROR_HANDLING=0 PYTHONUNBUFFERED=1; $NEWSTACK_PRE_CMD; export FTV2_OUT=\$PWD/\${JOB_DIR}"
if [ "${EX_MODE:-recovery}" = recovery ]; then
  export FTV2_RECOVERY=1 FTV2_STANDBY=1 FTV2_STANDBY_NODES=${EX_STANDBY_NODES:-1} FTV2_PARTNER_STRIDE=${EX_STRIDE:-8}
  export FTV2_SHADOW=${FTV2_SHADOW:-cpu} FTV2_CPU_ADAM=${FTV2_CPU_ADAM:-torch} FTV2_ARMS=${FTV2_ARMS:-main:}
  export AUTORUN_SBATCH_CMD=$FT/recovery/launch/ftv2_sbatch.sh
  export AUTORUN_PRE_CMD="$PRE; . $FT/heartbeat-sys/hb_start.sh; . $FT/recovery/launch/ftv2_srun.sh"
else
  export FTV2_RECOVERY=0 FTV2_STANDBY_NODES=0 FTV2_ABORT_ON_FAILURE=${FTV2_ABORT_ON_FAILURE:-60} FTV2_ARMS=${FTV2_ARMS:-main:}
  export AUTORUN_PRE_CMD="$PRE; . $FT/heartbeat-sys/hb_start.sh; . $FT/recovery/launch/ftv2_srun.sh"
fi
if [ "${RENDER_ONLY:-0}" = 1 ]; then
  if [ "${EX_MODE:-recovery}" = recovery ]; then export FTV2_SBATCH_DRY=1; else export AUTORUN_SBATCH_CMD=echo; fi
fi
LOG="$FT/runs/${EX_LABEL}.submit.log"
./local_autorun_frontier_0623_2026.sh 2>&1 | tee "$LOG" | grep -E "FATAL|ERROR|Submitted batch job|ftv2_sbatch|temp_slurm.*slurm$" || true
grep -q "FATAL" "$LOG" && { echo "FATAL in $LOG"; exit 1; }
echo "[ftv2 run] $EX_LABEL: mode ${EX_MODE:-recovery}; log $LOG"
