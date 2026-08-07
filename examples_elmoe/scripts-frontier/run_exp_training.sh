#!/bin/bash
###############################################################################
# run_exp_training.sh — one-command reviewer reproduction launcher
#
# A thin, self-contained dispatcher around frontier_elmoe.slurm.template.
# It bakes in ONE fixed topology (63B / 32 GPUs, PP4-EP8-DP1) — the same
# configuration used in autorun_frontier.sh — and lifts the model variant and
# planner choice out of the hand-edited PP_BATCH_MAP/EP_BATCH_MAP strings into
# command-line arguments.
#
# The template and autorun_frontier.sh are left untouched; the profiling
# walltime is set by sed-ing the "#SBATCH -t" line of the generated script.
#
# ---------------------------------------------------------------------------
# USAGE
#   ./run_exp_training.sh training ELMoE [VARIANT] [PLANNER]
#   ./run_exp_training.sh training <BASELINE>
#   ./run_exp_training.sh profiling_cache ELMoE
#   ./run_exp_training.sh env_validate                    # env warmup, no args
#   ./run_exp_training.sh main_results                    # launch all 5 headline runs
#   ./run_exp_training.sh loss_validate [NGPUS]           # convergence check (2 runs; default 8 GPUs)
#
#   --model-size {50B|63B}   (default 63B) model scale for training,
#       profiling_cache and main_results. May appear anywhere in the argument
#       list. Both share h=5120 / 128 experts / topk 6 / seq 4096 and differ only
#       in depth: 50B = 24 layers, 63B = 32 layers.
#           63B  paper scale; needs ~64GB HBM per GPU
#           50B  the same model 8 layers shallower — use this on 40GB-HBM GPUs
#       env_validate and loss_validate are unaffected: both are fixed at 10B.
#
#   VARIANT  (ELMoE only, default SeqGEMM):
#       SeqGEMM            -> ELMOE-3D                    (sequential GEMM)
#       PrimusGroupGEMM    -> ELMOE-GroupedGEMM-primus    (CK grouped GEMM)
#       TritonGroupGEMM    -> ELMOE-GroupedGEMM-triton    (Triton grouped GEMM)
#
#   PLANNER  (ELMoE only, default yes_planner):
#       yes_planner        -> dynamic-ckpt : uneven PP : Minimax planner
#       no_planner         ->         ckpt :   even PP : no planner
#
#   BASELINE:
#       X-MoE     -> X-MOE      DS-MoE   -> DS-MOE
#       DS-TED    -> TED-MOE    DS-Tutel -> TUTEL-MOE
#
# EXAMPLES
#   ./run_exp_training.sh env_validate                    # env warmup: X-MoE 10B, 1 node, 10 iters (~30 min)
#   ./run_exp_training.sh main_results                    # 5 runs: ELMoE (yes/no planner) + X-MoE/DS-MoE/DS-Tutel
#   ./run_exp_training.sh main_results --model-size 50B    # same 5 runs, 24-layer model (40GB HBM)
#   ./run_exp_training.sh training ELMoE --model-size 50B  # single 50B run
#   ./run_exp_training.sh loss_validate 16                # 2 runs: ELMoE-3D + X-MoE, 10B, GBS=320, 1000 steps, 32 GPUs
#   ./run_exp_training.sh profiling_cache ELMoE            # build planner cache first
#   ./run_exp_training.sh training ELMoE                   # SeqGEMM + yes_planner
#   ./run_exp_training.sh training ELMoE PrimusGroupGEMM   # CK grouped GEMM + yes_planner
#   ./run_exp_training.sh training ELMoE no_planner        # SeqGEMM + no_planner
#   ./run_exp_training.sh training ELMoE TritonGroupGEMM no_planner
#   ./run_exp_training.sh training DS-MoE
#
# BATCH SIZE: all training runs share the global batch set by GLOBAL_BATCH below
#       (1024). NBS (micro-batches per device per iteration) is derived per run as
#       ceil(GLOBAL_BATCH / EP_PARALLEL_SIZE): ELMoE (EP=8) -> 128,
#       baselines (EP=32) -> 32.
#
# NOTE: ELMoE with yes_planner requires the planner profiling cache. Run
#       `./run_exp_training.sh profiling_cache ELMoE` once before the first
#       yes_planner training run. The cache key does not encode depth, so a cache
#       built for either size is valid for both 50B and 63B.
###############################################################################

set -euo pipefail

TEMPLATE_FILE="frontier_elmoe.slurm.template"
TEMP_DIR="temp_slurm"
MP_SIZE=1
PARTITION="batch"

# Model scale for training / profiling_cache / main_results. Override with
# --model-size. 63B = 32 layers (paper scale); 50B = the same model at 24 layers,
# which is what fits on a 40GB-HBM GPU. Both resolve in ../utils/model_registry.py,
# and because the planner cache key ignores depth, ONE profiling_cache run serves
# both sizes.
MODEL_SIZE_DEFAULT="63B"
MODEL_SIZE_CHOICES="50B 63B"

# Global batch (sequences) shared by all training runs ("mbs 2048").
# Per-run NBS = ceil(GLOBAL_BATCH / EP_PARALLEL_SIZE).
GLOBAL_BATCH=1024

# Walltime per run type (profiling only needs ~10 min).
WALLTIME_TRAIN="0:30:00"
WALLTIME_PROFILE="0:10:00"
WALLTIME_VALIDATE="0:30:00"       # env warmup run — long enough to compile caches
WALLTIME_LOSS_VALIDATE="1:00:00"  # convergence check

# loss_validate: fixed global batch across every GPU budget, so the loss curves
# of ELMoE and X-MoE are directly comparable.
GBS_LOSS=320

usage() {
    sed -n '15,65p' "$0" | sed 's/^#//; s/^ //'
    exit 1
}

if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "ERROR: template '$TEMPLATE_FILE' not found (run from scripts-frontier/)." >&2
    exit 1
fi
mkdir -p "$TEMP_DIR"

# ---------------------------------------------------------------------------
# Pull --model-size out of the argument list wherever it appears, so the
# positional grammar below (MODE, MODEL, then the ELMoE option words) is
# unchanged and every existing invocation keeps working.
# ---------------------------------------------------------------------------
MODEL_SIZE_SEL=""
_ARGV=()
while [ $# -gt 0 ]; do
    case "$1" in
        --model-size)   MODEL_SIZE_SEL="${2:-}"; shift 2 ;;
        --model-size=*) MODEL_SIZE_SEL="${1#*=}"; shift ;;
        *)              _ARGV+=("$1"); shift ;;
    esac
done
set -- ${_ARGV[@]+"${_ARGV[@]}"}

MODEL_SIZE_SEL="${MODEL_SIZE_SEL:-$MODEL_SIZE_DEFAULT}"
case " ${MODEL_SIZE_CHOICES} " in
    *" ${MODEL_SIZE_SEL} "*) ;;
    *) echo "ERROR: --model-size must be one of: ${MODEL_SIZE_CHOICES} (got '${MODEL_SIZE_SEL}')." >&2; exit 1 ;;
esac

# Emitted once per invocation. main_results re-invokes this script per run, so the
# exported flag keeps the child processes from repeating it five times.
warn_model_size() {
    [ "$MODEL_SIZE_SEL" = "63B" ] || return 0
    [ "${ELMOE_SIZE_WARNED:-}" = "1" ] && return 0
    echo "WARNING: 63B model is running. For systems with 40GB HBM per GPU, please use 50B model instead to avoid OOM." >&2
    export ELMOE_SIZE_WARNED=1
}

MODE="${1:-}"
MODEL="${2:-}"
[ -z "$MODE" ] && usage
# env_validate / main_results take no MODEL arg; loss_validate takes an optional GPU budget.
case "$MODE" in
    env_validate|main_results|loss_validate) : ;;
    *) [ -n "$MODEL" ] || usage ;;
esac
shift 2 || true

# env_validate and loss_validate are fixed at 10B, so the size warning does not apply.
case "$MODE" in
    training|profiling_cache|main_results) warn_model_size ;;
esac

# ---------------------------------------------------------------------------
# render_and_submit — fill every template placeholder, set walltime, sbatch.
# All 19 placeholders are replaced in every path so no {{...}} literal leaks.
# ---------------------------------------------------------------------------
render_and_submit() {
    local job_name="$1" walltime="$2"
    local temp_script="${TEMP_DIR}/temp_slurm_${job_name}.slurm"

    echo "  - Generating job: ${job_name} (walltime ${walltime})"

    sed -e "s|^#SBATCH -t .*|#SBATCH -t ${walltime}|" \
        -e "s/{{RUN_TYPE}}/${RUN_TYPE}/g" \
        -e "s/{{NODES}}/${NODES}/g" \
        -e "s/{{TOTAL_GPUS}}/${TOTAL_GPUS}/g" \
        -e "s/{{PARTITION}}/${PARTITION}/g" \
        -e "s/{{BATCH_SIZE}}/${BS}/g" \
        -e "s/{{NUM_BATCHES}}/${NBS}/g" \
        -e "s/{{PP_SIZE}}/${PP_SIZE}/g" \
        -e "s/{{EP_PARALLEL_SIZE}}/${EP_PARALLEL_SIZE}/g" \
        -e "s/{{MP_SIZE}}/${MP_SIZE}/g" \
        -e "s/{{TRAIN_ITERS}}/${TRAIN_ITERS}/g" \
        -e "s/{{MOE_TYPE}}/${MOE_TYPE}/g" \
        -e "s/{{MODEL_SIZE}}/${MODEL_SIZE}/g" \
        -e "s/{{ACTIVATION_CHECKPOINT}}/${ACTIVATION_CHECKPOINT}/g" \
        -e "s/{{CHECKPOINT_NUM_LAYERS}}/${CHECKPOINT_NUM_LAYERS}/g" \
        -e "s/{{DYNAMIC_CHECKPOINT}}/${DYNAMIC_CHECKPOINT}/g" \
        -e "s/{{UNEVEN_PP}}/${UNEVEN_PP}/g" \
        -e "s/{{RUN_PLANNER}}/${RUN_PLANNER}/g" \
        -e "s/{{COLLECT_PROFILING_CACHE}}/${COLLECT_PROFILING_CACHE}/g" \
        -e "s/{{ZERO}}/${ZERO}/g" \
        "$TEMPLATE_FILE" > "$temp_script"

    sbatch "$temp_script"
    sleep 1
}

# ---------------------------------------------------------------------------
# derive_checkpoint_flags — map CHECKPOINT/PP_PARTITION/PLANNER_MODE strings to
# the template's boolean placeholders (same logic as autorun_frontier.sh).
# ---------------------------------------------------------------------------
derive_checkpoint_flags() {
    case "$CHECKPOINT" in
        ckpt)         ACTIVATION_CHECKPOINT="true";  DYNAMIC_CHECKPOINT="False" ;;
        dynamic-ckpt) ACTIVATION_CHECKPOINT="true";  DYNAMIC_CHECKPOINT="True"  ;;
        *)            ACTIVATION_CHECKPOINT="false"; DYNAMIC_CHECKPOINT="False" ;;
    esac

    if [[ "$PP_PARTITION" == "uneven" ]]; then UNEVEN_PP="True"; else UNEVEN_PP="False"; fi

    case "$PLANNER_MODE" in
        yes-planner)        RUN_PLANNER="true" ;;
        yes-planner-membal) RUN_PLANNER="true-membal" ;;
        *)                  RUN_PLANNER="false" ;;
    esac
}

case "$MODE" in
# ===========================================================================
# TRAINING
# ===========================================================================
training)
    if [[ "${MODEL,,}" == "elmoe" ]]; then
        # ---- ELMoE : PP path (63B, 32 GPUs, PP4-EP8-DP1) -----------------
        RUN_TYPE="pp"
        NODES=4; TOTAL_GPUS=32; PP_SIZE=4; EP_PARALLEL_SIZE=8
        # NBS = ceil(GLOBAL_BATCH / EP=8)
        BS=1; NBS=$(( (GLOBAL_BATCH + EP_PARALLEL_SIZE - 1) / EP_PARALLEL_SIZE )); TRAIN_ITERS=15; MODEL_SIZE="$MODEL_SIZE_SEL"; ZERO=1
        COLLECT_PROFILING_CACHE="false"

        # Defaults: SeqGEMM + yes_planner
        MOE_TYPE="ELMOE-3D"
        PLANNER_CHOICE="yes_planner"

        for arg in "$@"; do
            case "${arg,,}" in
                seqgemm)         MOE_TYPE="ELMOE-3D" ;;
                primusgroupgemm) MOE_TYPE="ELMOE-GroupedGEMM-primus" ;;
                tritongroupgemm) MOE_TYPE="ELMOE-GroupedGEMM-triton" ;;
                yes_planner)     PLANNER_CHOICE="yes_planner" ;;
                no_planner)      PLANNER_CHOICE="no_planner" ;;
                *) echo "ERROR: unknown ELMoE option '$arg'" >&2; usage ;;
            esac
        done

        if [[ "$PLANNER_CHOICE" == "yes_planner" ]]; then
            CHECKPOINT="dynamic-ckpt"; CHECKPOINT_NUM_LAYERS=1
            PP_PARTITION="uneven";     PLANNER_MODE="yes-planner"
        else
            CHECKPOINT="ckpt";         CHECKPOINT_NUM_LAYERS=1
            PP_PARTITION="even";       PLANNER_MODE="no-planner"
        fi
        derive_checkpoint_flags

        echo "ELMoE training: MOE_TYPE=${MOE_TYPE}, planner=${PLANNER_CHOICE}"

        # yes_planner needs the profiling cache — warn (don't launch) if missing.
        if [[ "$PLANNER_MODE" == "yes-planner" || "$PLANNER_MODE" == "yes-planner-membal" ]]; then
            EP_NODES=$(( EP_PARALLEL_SIZE / 8 ))
            CACHE_FILENAME=$(python3 ../utils/model_registry.py get_filename \
                --model_size "$MODEL_SIZE" --nodes "$EP_NODES" \
                --total_gpus "$EP_PARALLEL_SIZE" --mbs 1)
            CACHE_FILE="./planner_profiling_cache/${CACHE_FILENAME}"
            if [ -f "$CACHE_FILE" ]; then
                echo "  [Planner] Cache found: $CACHE_FILENAME"
            else
                echo "  [Planner] Cache MISSING: $CACHE_FILE" >&2
                echo "  Run: ./run_exp_training.sh profiling_cache ELMoE   (once) first." >&2
                exit 1
            fi
        fi

        JOB_NAME="${RUN_TYPE}_${MODEL_SIZE}_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}"
        render_and_submit "$JOB_NAME" "$WALLTIME_TRAIN"

    else
        # ---- Baselines : EP path (63B, 32 GPUs, EP32-DP1) ----------------
        RUN_TYPE="ep"
        NODES=4; TOTAL_GPUS=32; PP_SIZE=1; EP_PARALLEL_SIZE=32
        # NBS = ceil(GLOBAL_BATCH / EP=32)
        BS=1; NBS=$(( (GLOBAL_BATCH + EP_PARALLEL_SIZE - 1) / EP_PARALLEL_SIZE )); TRAIN_ITERS=15; MODEL_SIZE="$MODEL_SIZE_SEL"; ZERO=1
        COLLECT_PROFILING_CACHE="false"
        # PP-only knobs are unused on the EP path; set them harmlessly.
        DYNAMIC_CHECKPOINT="False"; UNEVEN_PP="False"; RUN_PLANNER="false"

        case "${MODEL,,}" in
            x-moe|xmoe)     MOE_TYPE="X-MOE";     CHECKPOINT="ckpt"; CHECKPOINT_NUM_LAYERS=1 ;;
            ds-moe|dsmoe)   MOE_TYPE="DS-MOE";    CHECKPOINT="ckpt"; CHECKPOINT_NUM_LAYERS=1 ;;
            ds-ted|dsted)   MOE_TYPE="TED-MOE";   CHECKPOINT="ckpt";    CHECKPOINT_NUM_LAYERS=1 ;;
            ds-tutel|dstutel) MOE_TYPE="TUTEL-MOE"; CHECKPOINT="ckpt";  CHECKPOINT_NUM_LAYERS=1 ;;
            *) echo "ERROR: unknown model '$MODEL'" >&2; usage ;;
        esac

        if [[ "$CHECKPOINT" == "ckpt" ]]; then ACTIVATION_CHECKPOINT="true"; else ACTIVATION_CHECKPOINT="false"; fi

        echo "Baseline training: MOE_TYPE=${MOE_TYPE}"
        JOB_NAME="${RUN_TYPE}_${MODEL_SIZE}_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}"
        render_and_submit "$JOB_NAME" "$WALLTIME_TRAIN"
    fi
    ;;

# ===========================================================================
# PROFILING CACHE  (single-node, X-MoE base, MBS 1..10, 10-min walltime)
# ===========================================================================
profiling_cache)
    if [[ "${MODEL,,}" != "elmoe" ]]; then
        echo "ERROR: profiling_cache only supports ELMoE (X-MoE-based single-layer profiling)." >&2
        usage
    fi

    RUN_TYPE="pr"
    NODES=1; TOTAL_GPUS=8; PP_SIZE=1; EP_PARALLEL_SIZE=$(( TOTAL_GPUS / MP_SIZE ))
    NBS=20; TRAIN_ITERS=30; MODEL_SIZE="${MODEL_SIZE_SEL}_1L"; MOE_TYPE="X-MOE"; ZERO=1
    CHECKPOINT_NUM_LAYERS=0
    ACTIVATION_CHECKPOINT="false"; DYNAMIC_CHECKPOINT="False"; UNEVEN_PP="False"
    RUN_PLANNER="false"; COLLECT_PROFILING_CACHE="true"

    echo "Profiling cache: MODEL_SIZE=${MODEL_SIZE}, sweeping MBS 1..8"
    for BS in {1..8}; do
        JOB_NAME="${RUN_TYPE}_${MODEL_SIZE}_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}_mbs${BS}"
        render_and_submit "$JOB_NAME" "$WALLTIME_PROFILE"
    done
    ;;

# ===========================================================================
# ENV_VALIDATE  (environment warmup — compiles JIT/Triton/aiter/kernel caches)
# ===========================================================================
env_validate)
    # Single node, X-MoE 10B, EP4 / PP2 / TP1, NBS=16, MBS=1, 10 iterations, ~30 min.
    # Purpose: warm the JIT/Triton/aiter/kernel caches on a freshly built
    # environment before the real runs. Takes no model/variant arguments.
    RUN_TYPE="va"
    NODES=1; TOTAL_GPUS=8; PP_SIZE=2; EP_PARALLEL_SIZE=4
    BS=1; NBS=16; TRAIN_ITERS=10; MODEL_SIZE="10B"; MOE_TYPE="X-MOE"; ZERO=1
    CHECKPOINT_NUM_LAYERS=0
    ACTIVATION_CHECKPOINT="false"; DYNAMIC_CHECKPOINT="False"; UNEVEN_PP="False"
    RUN_PLANNER="false"; COLLECT_PROFILING_CACHE="false"

    echo "Validate/warmup: X-MoE 10B, 1 node, EP4/PP2/TP1, NBS=16, MBS=1, ${TRAIN_ITERS} iters (walltime ${WALLTIME_VALIDATE})"
    JOB_NAME="${RUN_TYPE}_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}"
    render_and_submit "$JOB_NAME" "$WALLTIME_VALIDATE"
    ;;

# ===========================================================================
# MAIN_RESULTS  (launch the 5 headline comparison runs; each 63B / 32 GPUs)
# ===========================================================================
#   1. ELMoE SeqGEMM + yes_planner   (dynamic-ckpt, uneven PP, Minimax planner)
#   2. ELMoE SeqGEMM + no_planner    (full activation checkpointing, even PP)
#   3. X-MoE     4. DS-MoE     5. DS-Tutel
# DS-TED is intentionally NOT in this set: with TP=1 it is equivalent to DS-MoE.
# It remains available standalone via `./run_exp_training.sh training DS-TED`.
# Implemented by re-invoking THIS script per run, so each goes through the exact
# same training path. An individual failure (e.g. missing planner cache for the
# yes_planner run) is reported but does NOT stop the remaining launches.
main_results)
    echo "=================================================================="
    echo " main_results: launching 5 headline runs (each ${MODEL_SIZE_SEL} / 32 GPUs)"
    echo "=================================================================="
    declare -a RUNS=(
        "training ELMoE seqgemm yes_planner"
        "training ELMoE seqgemm no_planner"
        "training X-MoE"
        "training DS-MoE"
        "training DS-Tutel"
    )
    n=${#RUNS[@]}; ok=0; fail=0; i=0
    for run in "${RUNS[@]}"; do
        i=$((i+1))
        echo
        echo "----- [${i}/${n}] run_exp_training.sh ${run} (${MODEL_SIZE_SEL}) -----"
        if bash "$0" ${run} --model-size "$MODEL_SIZE_SEL"; then
            ok=$((ok+1))
        else
            fail=$((fail+1))
            echo "  [main_results] WARNING: '${run}' failed to submit (continuing)."
        fi
    done
    echo
    echo "main_results summary: ${ok}/${n} launched, ${fail} failed."
    ;;

# ===========================================================================
# LOSS_VALIDATE  (convergence check at a given GPU budget; 10B, GBS=320 fixed)
# ===========================================================================
#   ./run_exp_training.sh loss_validate [NGPUS]     (default 8; multiple of 8)
#
# Both frameworks hit the SAME global batch (GBS=320) so the loss curves are
# directly comparable. The template computes GBS = nbs * mbs * (GPUS/PP/TP),
# so DP = GPUS/PP and nbs = GBS / (mbs * DP).
#
#   (a) ELMoE-3D : mbs=4 always. 8 GPUs -> EP4-PP2; >8 GPUs -> EP8, PP = GPUS/8.
#   (b) X-MoE    : PP=1, EP = all GPUs. mbs = largest of {4,2,1} that keeps nbs
#                  integral  => mbs=4 at 8/16 GPUs, mbs=2 at 32 (4*32=128 does
#                  not divide 320), etc.
#
#   GPUs | ELMoE                        | X-MoE
#      8 | EP4-PP2 mbs4 dp4  nbs20 =320 | EP8-PP1  mbs4 dp8  nbs10 =320
#     16 | EP8-PP2 mbs4 dp8  nbs10 =320 | EP16-PP1 mbs4 dp16 nbs5  =320
#     32 | EP8-PP4 mbs4 dp8  nbs10 =320 | EP32-PP1 mbs2 dp32 nbs5  =320
#
# Both run with no planner (=> even PP) and NO activation checkpointing (10B fits
# without it). The in-job analyzer emits results before the walltime is hit
# (SIGUSR1 guard), so a run that does not reach TRAIN_ITERS still yields a usable
# loss curve.
loss_validate)
    # $2 (captured into MODEL) is the optional GPU budget.
    LV_GPUS="${MODEL:-8}"
    if ! [[ "$LV_GPUS" =~ ^[0-9]+$ ]] || (( LV_GPUS < 8 )) || (( LV_GPUS % 8 != 0 )); then
        echo "ERROR: loss_validate GPU budget must be a positive multiple of 8 (got '${LV_GPUS}')." >&2
        exit 1
    fi

    RUN_TYPE="lv"
    TOTAL_GPUS=$LV_GPUS
    NODES=$(( TOTAL_GPUS / 8 ))
    TRAIN_ITERS=1000; MODEL_SIZE="10B"; ZERO=1
    COLLECT_PROFILING_CACHE="false"
    # No planner (=> even PP, no dynamic ckpt) and NO activation checkpointing:
    # at 10B, activations fit without it, and skipping recompute makes the run
    # faster — so more steps land inside the walltime for the loss curve.
    ACTIVATION_CHECKPOINT="false"; CHECKPOINT_NUM_LAYERS=0
    DYNAMIC_CHECKPOINT="False"; UNEVEN_PP="False"; RUN_PLANNER="false"

    echo "=================================================================="
    echo " loss_validate: ${NODES} node(s) / ${TOTAL_GPUS} GPUs | 10B | GBS=${GBS_LOSS} | ${TRAIN_ITERS} steps | ${WALLTIME_LOSS_VALIDATE}"
    echo "=================================================================="

    # ---- (a) ELMoE-3D ------------------------------------------------------
    MOE_TYPE="ELMOE-3D"; BS=4
    if (( TOTAL_GPUS == 8 )); then
        EP_PARALLEL_SIZE=4; PP_SIZE=2
    else
        EP_PARALLEL_SIZE=8; PP_SIZE=$(( TOTAL_GPUS / 8 ))
    fi
    LV_DP=$(( TOTAL_GPUS / PP_SIZE / MP_SIZE ))
    if (( GBS_LOSS % (BS * LV_DP) != 0 )); then
        echo "ERROR: ELMoE — GBS=${GBS_LOSS} not divisible by mbs(${BS}) * DP(${LV_DP})." >&2; exit 1
    fi
    NBS=$(( GBS_LOSS / (BS * LV_DP) ))
    echo "  (a) ELMoE-3D  EP${EP_PARALLEL_SIZE}-PP${PP_SIZE}  mbs=${BS} nbs=${NBS} dp=${LV_DP}  -> GBS=$(( BS * NBS * LV_DP ))"
    JOB_NAME="${RUN_TYPE}_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_pp${PP_SIZE}_mbs${BS}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}"
    render_and_submit "$JOB_NAME" "$WALLTIME_LOSS_VALIDATE"

    # ---- (b) X-MoE ---------------------------------------------------------
    MOE_TYPE="X-MOE"; PP_SIZE=1; EP_PARALLEL_SIZE=$TOTAL_GPUS
    LV_DP=$(( TOTAL_GPUS / PP_SIZE / MP_SIZE ))
    BS=0
    for cand in 4 2 1; do
        if (( GBS_LOSS % (cand * LV_DP) == 0 )); then BS=$cand; break; fi
    done
    if (( BS == 0 )); then
        echo "ERROR: X-MoE — no mbs in {4,2,1} divides GBS=${GBS_LOSS} at DP=${LV_DP}." >&2; exit 1
    fi
    NBS=$(( GBS_LOSS / (BS * LV_DP) ))
    echo "  (b) X-MoE     EP${EP_PARALLEL_SIZE}-PP${PP_SIZE}  mbs=${BS} nbs=${NBS} dp=${LV_DP}  -> GBS=$(( BS * NBS * LV_DP ))"
    JOB_NAME="${RUN_TYPE}_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_pp${PP_SIZE}_mbs${BS}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}"
    render_and_submit "$JOB_NAME" "$WALLTIME_LOSS_VALIDATE"

    echo
    echo "When BOTH jobs finish, plot the comparison curve:"
    echo "    python3 ../utils/plot_loss_curve.py"
    ;;

*)
    echo "ERROR: unknown mode '$MODE' (expected 'training', 'profiling_cache', 'env_validate', 'main_results', or 'loss_validate')." >&2
    usage
    ;;
esac

echo
echo "All jobs submitted."
