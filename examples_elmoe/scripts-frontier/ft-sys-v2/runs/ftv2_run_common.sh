# ftv2_run_common.sh (ft-sys-v2, 2026-11-06): the shared part of every run file in this folder (ftv2_<model>.sh).
#
# What it does, in order (the numbered sections below):
#   1. finds this repository and every path a run reads (../env/ftv2_paths.sh)
#   2. loads the software environment and checks that megatron and deepspeed import from the intended folders
#   3. turns the model's layout, set by the run file, into the launcher's settings (AUTORUN_*)
#   4. turns ft-sys-v2 on and builds PRE, the command that runs inside the job just before training
#   5. picks the mode: recovery (heartbeat + warm recovery with a standby node) or heartbeat only
#   6. calls the lab's 0623 launcher (never edited), which fills in the job template and submits the job
#
# How a run file uses it (ftv2_10B.sh): it sets the layout, then sources this file:
#   EX_LABEL=ftv2_10B EX_MODE=recovery EX_EP_LAYOUT="3:8:1" EX_EP_CMD=" 1:8:40:X-MOE:10B:ckpt:1:1" EX_STRIDE=8 EX_WALL=00:30:00
#   . "$(dirname "${BASH_SOURCE[0]}")/ftv2_run_common.sh"
# RENDER_ONLY=1 bash ftv2_<model>.sh writes the job script without submitting it: read it before a real submission.
#
# What the run file sets:
#   EX_LABEL       the run's name; the launcher's full output goes to runs/<EX_LABEL>.submit.log
#   EX_EP_LAYOUT   "nodes:EP:TP": training nodes, expert-parallel size, tensor-parallel size
#   EX_EP_CMD      " mbs:nbs:iters:X-MOE:<model>:ckpt:n:<zero>", the launcher's format (local_autorun_frontier_0623_2026.sh:394):
#                  micro-batch size, micro-batches per step, iterations, MoE type, model name (a key of
#                  examples_elmoe/utils/model_registry.py), activation checkpointing, n, ZeRO stage
#   optional       EX_PP_SIZE (pipeline stages), EX_SEQ_LEN, EX_MEGATRON_SP, EX_STRIDE (ranks per data-parallel replica: rank r's
#                  optimizer copy lives on rank r + EX_STRIDE, the next node), EX_WALL (wall time), EX_QOS (default debug),
#                  EX_MODE (recovery or heartbeat), EX_STANDBY_NODES, EX_EXIT_MINS, EX_NO_CKPT
#
# The two modes:
#   recovery    heartbeat + warm recovery: the job gets FTV2_STANDBY_NODES extra node(s); when a node fails, the standby takes its
#               place inside the running job, and nothing is read from disk. Tested through this file with PP 1 (milestone 1).
#               Milestone 3 (recovery under a pipeline) ran from the heartbeat scope's own job scripts; it needs settings this file
#               does not set yet (FTV2_COMMIT=1 and FTV2_RESUME_LEVEL, recovery/rc_c_flow.py:1-3).
#   heartbeat   detection only: the failure list is written and the job ends FTV2_ABORT_ON_FAILURE seconds later; restart by
#               resubmitting with ft-sys-v1's resume settings (AUTORUN_LOAD_DIR/AUTORUN_SAVE_DIR with AUTORUN_NATIVE_RESUME=1 or
#               AUTORUN_UCP_RESUME=1).
#
# Paths: every path a run reads is a variable of ../env/ftv2_paths.sh (this tree, the launch folder, the venv, ROCm 7.14.1, the
# network plugin, etcd, the data, the tokenizer); a variable already set in your shell wins. ../env/ftv2_check_env.sh checks them.
# To launch from the user's shared checkout as before 2026-11-06: FTV2_SF=/lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier
set -euo pipefail

# ---- 1. Where things are ------------------------------------------------------------------------------------------------------
# FTV2_TREE: the root of this repository, four folders up from runs/. ftv2_paths.sh sets every other path, most of them from it.
FTV2_TREE=${FTV2_TREE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
. "$(dirname "${BASH_SOURCE[0]}")/../env/ftv2_paths.sh"
# FT: the ft-sys-v2 folder (heartbeat-sys/, recovery/, runs/, env/).
FT=$FTV2_TREE/examples_elmoe/scripts-frontier/ft-sys-v2
# The launcher, the job template and eval/submit_env.sh live in FTV2_SF (scripts-frontier); the launcher runs from there.
cd "$FTV2_SF"
mkdir -p temp_slurm logs   # the launcher writes temp_slurm/ and the job's -o/-e go to logs/; neither is in git (a fresh clone has none)

# ---- 2. The software environment, and two checks before anything is submitted ---------------------------------------------------
# eval/submit_env.sh loads the site modules (compilers, Cray MPICH, the ROCm 6.4.1 module) and the lab's older Python environment;
# FTV2_ENV (env/newstack_env.sh) then puts the new stack on top: the venv with torch 2.13, ROCm 7.14.1, the network plugin, and
# DeepSpeed from NEWSTACK_DEEPSPEED. Both are sourced with -e and -u off, because they were not written for strict mode.
set +e +u; . ./eval/submit_env.sh; . "$FTV2_ENV"; set -euo pipefail
# The compilers for the kernels that DeepSpeed and Megatron build at run time.
export CXX=$(which g++) CC=$(which gcc)
# This tree first, so "import megatron" gives this branch's code (the venv's installed megatron points at another checkout).
export PYTHONPATH=$FTV2_TREE:${PYTHONPATH:-}
# Check 1: megatron imports from this tree; if not, stop here.
python -I -B -c "import sys; sys.path.insert(0, '$FTV2_TREE'); import megatron; assert megatron.__file__.startswith('$FTV2_TREE'), megatron.__file__" \
  || { echo "FATAL: megatron does not import from $FTV2_TREE"; exit 1; }
# Check 2: deepspeed is found through PYTHONPATH in the job's order (find_spec only: nothing is imported, no sitecustomize runs)
python -I -B -c "import os, sys, importlib.util; sys.path[:0] = [p for p in os.environ['PYTHONPATH'].split(':') if p]; o = importlib.util.find_spec('deepspeed').origin; assert o.startswith('$NEWSTACK_DEEPSPEED/'), o" \
  || { echo "FATAL: deepspeed does not import from NEWSTACK_DEEPSPEED=$NEWSTACK_DEEPSPEED"; exit 1; }

# ---- 3. The model's layout, as the launcher's settings ----------------------------------------------------------------------------
# For heartbeat-sys/hb_start.sh inside the job: the Python the node agents run with, and the etcd binary (set in ftv2_paths.sh).
export FTV2_PY=$(command -v python) FTV2_ETCD_BIN
# The launcher normally submits a whole table of layouts; these two make it submit only this one.
export AUTORUN_EP_KEY="$EX_EP_LAYOUT" AUTORUN_EP_CMD="$EX_EP_CMD"
# Optional overrides, passed on only when the run file sets them: pipeline stages, sequence length, Megatron sequence parallelism.
[ -n "${EX_PP_SIZE:-}" ] && export AUTORUN_PP_SIZE=$EX_PP_SIZE
[ -n "${EX_SEQ_LEN:-}" ] && export AUTORUN_SEQ_LEN=$EX_SEQ_LEN
[ -n "${EX_MEGATRON_SP:-}" ] && export AUTORUN_MEGATRON_SP=$EX_MEGATRON_SP
# The queue (debug by default), the Slurm wall time, and the training's own limit in minutes (it exits cleanly before the wall).
export AUTORUN_QOS=${EX_QOS:-debug} AUTORUN_WALLTIME="${EX_WALL:-00:30:00}" AUTORUN_EXIT_MINS=${EX_EXIT_MINS:-25}
# No --save/--load (random init, no checkpoint written), no profiling instruments (Darshan, timers, tensorboard),
# RCCL logs at WARN instead of INFO, and a DeepSpeed json of the job's own.
export AUTORUN_NO_CKPT=${EX_NO_CKPT:-1} AUTORUN_NO_PROFILING=1 AUTORUN_NCCL_DEBUG=WARN AUTORUN_DS_CONFIG_PER_JOB=1

# ---- 4. ft-sys-v2 on, and PRE: the command that runs inside the job just before training -----------------------------------------
# XMOE_FT_V2=1: megatron/training.py wraps every torch.distributed call with a counter and starts the heartbeat's monitor thread
# in every training process (search "[ft-sys-v2" in that file).
export XMOE_FT_V2=1
# PRE, piece by piece:
#   ulimit -c 0                         no core files (a crashing process would write gigabytes to Lustre)
#   TORCH_NCCL_ASYNC_ERROR_HANDLING=0   PyTorch's watchdog no longer kills the surviving processes after a node is lost (it did,
#   NCCL_ASYNC_ERROR_HANDLING=0         about 60 s after the loss); ft-sys-v2 makes that decision instead. The second is the old name.
#   PYTHONUNBUFFERED=1                  every print reaches the log at once
#   $NEWSTACK_PRE_CMD                   the site's network settings (FI_CXI_*, NCCL_*) and the preload of ROCm 7.14.1's HSA, HIP and
#                                       RCCL libraries (through the template's DARSHAN_PRELOAD, which the training srun puts into
#                                       LD_PRELOAD); they must come after the template's own exports (env/newstack_env.sh:13-14)
#   FTV2_OUT                            the job's folder on Lustre; the heartbeat writes its records under $FTV2_OUT/ft/
PRE="ulimit -c 0; export TORCH_NCCL_ASYNC_ERROR_HANDLING=0 NCCL_ASYNC_ERROR_HANDLING=0 PYTHONUNBUFFERED=1; $NEWSTACK_PRE_CMD; export FTV2_OUT=\$PWD/\${JOB_DIR}"
# The template fixes the tokenizer (VOCAB_FILE, no variable reaches it) and builds data_options from it before this hook runs; with
# FTV2_VOCAB_FILE set, the hook replaces the --vocab-file argument in data_options (read by the training srun) and stops if it did not.
if [ -n "$FTV2_VOCAB_FILE" ]; then
  PRE="$PRE; data_options=\"\${data_options/ --vocab-file \${VOCAB_FILE} / --vocab-file $FTV2_VOCAB_FILE }\"; VOCAB_FILE=$FTV2_VOCAB_FILE; case \"\$data_options\" in *' --vocab-file $FTV2_VOCAB_FILE '*) echo '[ftv2] tokenizer FTV2_VOCAB_FILE=$FTV2_VOCAB_FILE' ;; *) echo 'FATAL [ftv2] FTV2_VOCAB_FILE did not reach data_options'; exit 1 ;; esac"
fi

# ---- 5. The mode ----------------------------------------------------------------------------------------------------------------
if [ "${EX_MODE:-recovery}" = recovery ]; then
  # FTV2_RECOVERY=1     training runs inside the recovery loop (recovery/rc_controller.py)
  # FTV2_STANDBY=1      the last node is the standby: its agent and its processes wait for a failure list naming it
  # FTV2_STANDBY_NODES  how many standby nodes the job asks for (default 1)
  # FTV2_PARTNER_STRIDE rank r's optimizer copy (Keep) lives on rank r + stride, so a node's copies sit on the next node
  export FTV2_RECOVERY=1 FTV2_STANDBY=1 FTV2_STANDBY_NODES=${EX_STANDBY_NODES:-1} FTV2_PARTNER_STRIDE=${EX_STRIDE:-8}
  # FTV2_SHADOW=cpu     the partner keeps that copy in host memory, not on its GPU
  # FTV2_CPU_ADAM=torch the copy is updated every step by torch's fused CPU AdamW (DeepSpeed's CPU Adam holds Python's lock, the
  #                     GIL, during its update)
  # FTV2_ARMS           the runs inside one job, one Slurm step each, written "name:VAR=value,VAR=value" and separated by
  #                     spaces; "main:" is one run with no extra settings (recovery/launch/ftv2_srun.sh)
  export FTV2_SHADOW=${FTV2_SHADOW:-cpu} FTV2_CPU_ADAM=${FTV2_CPU_ADAM:-torch} FTV2_ARMS=${FTV2_ARMS:-main:}
  # The two hooks into the launcher. Submit: ftv2_sbatch.sh asks Slurm for the standby node(s) and --no-kill (a lost node does not
  # end the job), then calls sbatch.
  export AUTORUN_SBATCH_CMD=$FT/recovery/launch/ftv2_sbatch.sh
  # Inside the job, before training: PRE; then hb_start.sh (etcd on the first 3 nodes, one agent on every node); then ftv2_srun.sh
  # (defines a shell function named srun, so the template's training srun starts training and standby as one Slurm step).
  export AUTORUN_PRE_CMD="$PRE; . $FT/heartbeat-sys/hb_start.sh; . $FT/recovery/launch/ftv2_srun.sh"
else
  # Heartbeat only: no standby, no recovery; once a failure list exists, the job ends FTV2_ABORT_ON_FAILURE (default 60) seconds later.
  export FTV2_RECOVERY=0 FTV2_STANDBY_NODES=0 FTV2_ABORT_ON_FAILURE=${FTV2_ABORT_ON_FAILURE:-60} FTV2_ARMS=${FTV2_ARMS:-main:}
  export AUTORUN_PRE_CMD="$PRE; . $FT/heartbeat-sys/hb_start.sh; . $FT/recovery/launch/ftv2_srun.sh"
fi
# RENDER_ONLY=1: write the job script, do not submit. Recovery: ftv2_sbatch.sh writes its copy with the standby node and stops
# before sbatch. Heartbeat: the launcher's submit command becomes echo.
if [ "${RENDER_ONLY:-0}" = 1 ]; then
  if [ "${EX_MODE:-recovery}" = recovery ]; then export FTV2_SBATCH_DRY=1; else export AUTORUN_SBATCH_CMD=echo; fi
fi

# ---- 6. Submit ----------------------------------------------------------------------------------------------------------------
# The launcher fills in the template (temp_slurm/<...>.slurm) and submits it through AUTORUN_SBATCH_CMD. Its full output goes to
# runs/<EX_LABEL>.submit.log; the screen shows only errors, the job id and the job script's path. Any FATAL line fails the run file.
LOG="$FT/runs/${EX_LABEL}.submit.log"
./local_autorun_frontier_0623_2026.sh 2>&1 | tee "$LOG" | grep -E "FATAL|ERROR|Submitted batch job|ftv2_sbatch|temp_slurm.*slurm$" || true
grep -q "FATAL" "$LOG" && { echo "FATAL in $LOG"; exit 1; }
echo "[ftv2 run] $EX_LABEL: mode ${EX_MODE:-recovery}; log $LOG"
