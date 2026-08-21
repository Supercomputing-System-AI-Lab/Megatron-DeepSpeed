#!/bin/bash
###############################################################################
# run_exp_training.sh — AE (artifact-evaluation) convenience launcher, non-SLURM.
#
# This is the SAME CLI as scripts-frontier/run_exp_training.sh, but it RUNS the
# rendered script with torchrun instead of submitting it with sbatch. It is a
# convenience layer for reviewers; for day-to-day work use autorun.sh, which is
# the map-driven driver (mirrors autorun_frontier.sh).
#
# Runs are SEQUENTIAL — on a non-SLURM box every run contends for the same GPUs.
#
# ---------------------------------------------------------------------------
# USAGE
#   ./run_exp_training.sh env_validate            [--gpus N]
#   ./run_exp_training.sh loss_validate           [--gpus N]
#   ./run_exp_training.sh training ELMoE [VARIANT] [PLANNER]  [--gpus N]
#   ./run_exp_training.sh training <BASELINE>     [--gpus N]
#   ./run_exp_training.sh profiling_cache ELMoE   [--gpus N]
#
#   --gpus N    total GPU budget (default: 16; loss_validate defaults to 8 = one node).
#               Must be a multiple of --gpus-per-node.
#   --gpus-per-node G   (default: 8)
#
#   --model-size {10B|21B|25B|50B|63B}   (default 21B) scale for training /
#       profiling_cache / main_results. 50B and 63B share h=5120 / 128 experts /
#       topk 6 / seq 4096 and differ only in depth (24 vs 32 layers):
#           63B  paper scale; needs ~64GB HBM per GPU, 32 GPUs
#           50B  the same model 8 layers shallower — use this on 40GB-HBM GPUs
#           25B  same family, 12 layers
#           21B  DEFAULT. Same family (h5120 / 128 experts / topk6 / seq4096) at 10
#                layers. Fits 2x8 40GB-HBM nodes. Throughput will NOT match the paper.
#           10B  smallest escape hatch: whole pipeline on 8 GPUs, different shape
#                (h2048 / 64 experts / seq2048), so not comparable to the 63B family
#       env_validate and loss_validate are unaffected: both are fixed at 10B.
#
#   VARIANT (ELMoE, default SeqGEMM): SeqGEMM | PrimusGroupGEMM | TritonGroupGEMM
#   PLANNER (ELMoE, default yes_planner): yes_planner | no_planner
#   BASELINE: X-MoE | DS-MoE | DS-TED | DS-Tutel
#
# Multi-node needs ./hostfile and ./env.sh — see hostfile.example / env.sh.example.
#
# NOTE: yes_planner needs a planner cache generated ON THIS MACHINE (timings are
#       hardware-specific). Run `./run_exp_training.sh profiling_cache ELMoE` first.
###############################################################################

set -uo pipefail

TEMPLATE_FILE="elmoe.sh.template"
TEMP_DIR="temp_sh"
MP_SIZE=1
# Overridable so a smoke run does not need the file edited and reverted:
#     GLOBAL_BATCH=128 ./run_exp_training.sh training ELMoE no_planner
# 1024 is the AE/paper value. NBS = ceil(GLOBAL_BATCH / EP), same rule as Frontier.
GLOBAL_BATCH="${GLOBAL_BATCH:-1024}"
# loss_validate knobs. Defaults are the AE values (GBS 320 x 1000 steps); both are
# overridable so the pipeline can be smoke-tested in minutes instead of days:
#     LOSS_ITERS=10 GBS_LOSS=64 ./run_exp_training.sh loss_validate
# GBS_LOSS must stay divisible by mbs*DP for BOTH frameworks or the run aborts.
# Defaults below are tuned for 40GB-HBM GPUs (A100-40GB / p4d). The upstream AE values
# were 320 x 1000 steps at mbs=4 with no checkpointing, which needs ~64GB and OOMs here.
# Raising them back is a pure env override, e.g. for a 64GB site:
#     GBS_LOSS=320 LOSS_ITERS=1000 LOSS_MBS=4 LOSS_CKPT=false ./run_exp_training.sh loss_validate
GBS_LOSS="${GBS_LOSS:-80}"     # fixed global batch so the two curves are comparable
LOSS_ITERS="${LOSS_ITERS:-100}"
# mbs and activation checkpointing for loss_validate. The AE values (mbs=4, no ckpt)
# assume ~64GB HBM; on a 40GB card they OOM outright -- and single-node is WORSE than
# multi-node here, because halving the node count halves EP, which DOUBLES the experts
# held per GPU.
#
# Lowering these does NOT change the loss curve. Activation checkpointing recomputes
# activations rather than storing them (numerically identical), and gradient
# accumulation at a FIXED global batch is mathematically equivalent regardless of how
# the batch is split into micro-batches. NBS is recomputed from GBS_LOSS/(mbs*DP), so
# the effective global batch -- the only thing the comparison depends on -- is unchanged.
LOSS_MBS="${LOSS_MBS:-1}"
LOSS_CKPT="${LOSS_CKPT:-true}"    # "false" to disable activation checkpointing
# loss_validate runs on ONE node unless --gpus is given explicitly. Its X-MoE leg sets
# EP = TOTAL_GPUS, so on 2 nodes every expert all-to-all crosses the network: measured
# 113 s/iter over TCP vs 6 s/iter single-node, where EP stays inside NVLink. Same curve,
# ~20x the wall clock. The ELMoE leg is unaffected in kind (EP is capped per node).
LOSS_GPUS="${LOSS_GPUS:-8}"

# Model scale for training / profiling_cache / main_results (override: --model-size).
# 63B = 32 layers (paper scale); 50B = the same model at 24 layers, which is what
# fits on a 40GB-HBM GPU; 10B is the 8-GPU escape hatch. All three resolve in
# ../utils/model_registry.py. The planner cache key ignores depth, so ONE
# profiling_cache run serves both 50B and 63B.
# NOTE ON DEFAULTS: these are deliberately SMALL-BOX defaults (21B on 16 GPUs), not
# the paper topology. The paper scale is 63B on 32 GPUs -- reproduce it explicitly:
#     ./run_exp_training.sh training ELMoE --model-size 63B --gpus 32
MODEL_SIZE_DEFAULT="21B"
MODEL_SIZE_CHOICES="10B 21B 25B 50B 63B"
DEFAULT_GPUS=16        # 2x8-GPU nodes; --gpus 32 --model-size 63B for paper topology

usage() { sed -n '13,39p' "$0" | sed 's/^#//; s/^ //'; exit 1; }

[ -f "$TEMPLATE_FILE" ] || { echo "ERROR: '$TEMPLATE_FILE' not found (run from scripts/)." >&2; exit 1; }
mkdir -p "$TEMP_DIR" logs

MODE="${1:-}"; [ -z "$MODE" ] && usage
shift || true

# --- parse flags (and collect the non-flag args for the mode) ---------------
GPUS=$DEFAULT_GPUS; GPUS_PER_NODE=8; MODEL_SIZE_OPT=""; GPUS_EXPLICIT=0; ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --gpus)          GPUS="$2"; GPUS_EXPLICIT=1; shift 2 ;;
        --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
        --model-size)    MODEL_SIZE_OPT="$2"; shift 2 ;;
        --model-size=*)  MODEL_SIZE_OPT="${1#*=}"; shift ;;
        *)               ARGS+=("$1"); shift ;;
    esac
done
MODEL="${ARGS[0]:-}"

MODEL_SIZE_OPT="${MODEL_SIZE_OPT:-$MODEL_SIZE_DEFAULT}"
case " ${MODEL_SIZE_CHOICES} " in
    *" ${MODEL_SIZE_OPT} "*) ;;
    *) echo "ERROR: --model-size must be one of: ${MODEL_SIZE_CHOICES} (got '${MODEL_SIZE_OPT}')." >&2; exit 1 ;;
esac

# Emitted once per invocation. main_results re-invokes this script per run, so the
# exported flag keeps the child processes from repeating it five times.
warn_model_size() {
    [ "$MODEL_SIZE_OPT" = "63B" ] || return 0
    [ "${ELMOE_SIZE_WARNED:-}" = "1" ] && return 0
    echo "WARNING: 63B model is running. For systems with 40GB HBM per GPU, please use 50B model instead to avoid OOM." >&2
    export ELMOE_SIZE_WARNED=1
}
# env_validate and loss_validate are fixed at 10B, so the size warning does not apply.
case "$MODE" in
    training|profiling_cache|main_results) warn_model_size ;;
esac

# Mode-specific GPU default: only when the user did not say --gpus.
if [ "$MODE" = "loss_validate" ] && [ "$GPUS_EXPLICIT" -eq 0 ]; then
    GPUS="$LOSS_GPUS"
fi

if (( GPUS < 1 )) || (( GPUS % GPUS_PER_NODE != 0 )); then
    echo "ERROR: --gpus ${GPUS} must be a positive multiple of --gpus-per-node ${GPUS_PER_NODE}." >&2
    exit 1
fi
NODES=$(( GPUS / GPUS_PER_NODE ))
TOTAL_GPUS=$GPUS

# ---------------------------------------------------------------------------
# Minimum GPUs a model actually fits on. Without this guard a reviewer on an
# 8-GPU box would launch 63B and sit through a long run only to OOM.
# ---------------------------------------------------------------------------
min_gpus_for() {
    case "$1" in
        10B|10B_1L)   echo 8   ;;
        21B|21B_1L)   echo 16  ;;
        25B|25B_1L)   echo 16  ;;
        50B|50B_1L)   echo 32  ;;
        63B|63B_1L)   echo 32  ;;
        173B|173B_1L) echo 256 ;;
        537B|537B_1L) echo 480 ;;
        1T|1T_1L)     echo 960 ;;
        *)            echo 8   ;;
    esac
}

require_gpus_for_model() {
    local need; need="$(min_gpus_for "$MODEL_SIZE")"
    if (( TOTAL_GPUS < need )); then
        echo "ERROR: ${MODEL_SIZE} needs at least ${need} GPUs, but --gpus ${TOTAL_GPUS} was given." >&2
        echo "       Either give it more GPUs:   --gpus ${need}" >&2
        echo "       or scale the model down:    --model-size 10B --gpus 8" >&2
        echo "       (see scripts/README.md — 'Scaling down to a smaller box')" >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
mint_job_id() {
    local jid; jid=$(date +%s)
    while compgen -G "logs/job_${jid}_*" > /dev/null; do jid=$((jid + 1)); done
    echo "$jid"
}

render_and_run() {
    local job_name="$1"
    local temp_script="${TEMP_DIR}/${job_name}.sh"

    JOB_ID=$(mint_job_id); export JOB_ID
    local job_dir="logs/job_${JOB_ID}_pp${PP_SIZE}_ep${EP_PARALLEL_SIZE}_${MOE_TYPE}_ckpt_${ACTIVATION_CHECKPOINT}_planner_${RUN_PLANNER}"
    mkdir -p "$job_dir"
    local o_file="${job_dir}/a-xmoe-${RUN_TYPE}-${JOB_ID}.o"

    echo "  - Generating: ${job_name}  (JOB_ID=${JOB_ID})"
    sed -e "s/{{RUN_TYPE}}/${RUN_TYPE}/g" \
        -e "s/{{NODES}}/${NODES}/g" \
        -e "s/{{TOTAL_GPUS}}/${TOTAL_GPUS}/g" \
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

    bash "$temp_script" 2>&1 | tee "$o_file"
    return "${PIPESTATUS[0]}"
}

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

check_planner_cache() {
    local ep_nodes=$(( EP_PARALLEL_SIZE / GPUS_PER_NODE )); [ "$ep_nodes" -lt 1 ] && ep_nodes=1
    local f; f=$(python3 ../utils/model_registry.py get_filename \
        --model_size "$MODEL_SIZE" --nodes "$ep_nodes" --total_gpus "$EP_PARALLEL_SIZE" --mbs 1)
    if [ -f "./planner_profiling_cache/${f}" ]; then
        echo "  [Planner] Cache found: ${f}"
    else
        echo "  [Planner] Cache MISSING: ./planner_profiling_cache/${f}" >&2
        echo "            Caches are HARDWARE-SPECIFIC — a Frontier cache is not valid here." >&2
        echo "            Run: ./run_exp_training.sh profiling_cache ELMoE" >&2
        exit 1
    fi
}

case "$MODE" in
# ===========================================================================
training)
    [ -n "$MODEL" ] || usage
    COLLECT_PROFILING_CACHE="false"; ZERO=1; TRAIN_ITERS=15
    MODEL_SIZE="$MODEL_SIZE_OPT"
    require_gpus_for_model
    if [[ "${MODEL,,}" == "elmoe" ]]; then
        RUN_TYPE="pp"
        EP_PARALLEL_SIZE=$(( TOTAL_GPUS < 8 ? TOTAL_GPUS : 8 ))
        PP_SIZE=$(( TOTAL_GPUS / EP_PARALLEL_SIZE ))
        BS=1; NBS=$(( (GLOBAL_BATCH + EP_PARALLEL_SIZE - 1) / EP_PARALLEL_SIZE ))
        MOE_TYPE="ELMOE-3D"; PLANNER_CHOICE="yes_planner"
        for a in "${ARGS[@]:1}"; do
            case "${a,,}" in
                seqgemm)         MOE_TYPE="ELMOE-3D" ;;
                primusgroupgemm) MOE_TYPE="ELMOE-GroupedGEMM-primus" ;;
                tritongroupgemm) MOE_TYPE="ELMOE-GroupedGEMM-triton" ;;
                yes_planner)     PLANNER_CHOICE="yes_planner" ;;
                no_planner)      PLANNER_CHOICE="no_planner" ;;
                *) echo "ERROR: unknown ELMoE option '$a'" >&2; usage ;;
            esac
        done
        if [[ "$PLANNER_CHOICE" == "yes_planner" ]]; then
            CHECKPOINT="dynamic-ckpt"; CHECKPOINT_NUM_LAYERS=1; PP_PARTITION="uneven"; PLANNER_MODE="yes-planner"
        else
            CHECKPOINT="ckpt"; CHECKPOINT_NUM_LAYERS=1; PP_PARTITION="even"; PLANNER_MODE="no-planner"
        fi
        derive_checkpoint_flags
        echo "ELMoE training: ${MOE_TYPE}, planner=${PLANNER_CHOICE}, ${NODES}n/${TOTAL_GPUS}g EP${EP_PARALLEL_SIZE}-PP${PP_SIZE}"
        [[ "$RUN_PLANNER" == "true" || "$RUN_PLANNER" == "true-membal" ]] && check_planner_cache
        render_and_run "pp_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}"
    else
        RUN_TYPE="ep"; PP_SIZE=1; EP_PARALLEL_SIZE=$TOTAL_GPUS
        BS=1; NBS=$(( (GLOBAL_BATCH + EP_PARALLEL_SIZE - 1) / EP_PARALLEL_SIZE ))
        DYNAMIC_CHECKPOINT="False"; UNEVEN_PP="False"; RUN_PLANNER="false"
        case "${MODEL,,}" in
            x-moe|xmoe)       MOE_TYPE="X-MOE" ;;
            ds-moe|dsmoe)     MOE_TYPE="DS-MOE" ;;
            ds-ted|dsted)     MOE_TYPE="TED-MOE" ;;
            ds-tutel|dstutel) MOE_TYPE="TUTEL-MOE" ;;
            *) echo "ERROR: unknown model '$MODEL'" >&2; usage ;;
        esac
        ACTIVATION_CHECKPOINT="true"; CHECKPOINT_NUM_LAYERS=1
        echo "Baseline training: ${MOE_TYPE}, ${NODES}n/${TOTAL_GPUS}g EP${EP_PARALLEL_SIZE}"
        render_and_run "ep_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}"
    fi
    ;;

# ===========================================================================
env_validate)
    RUN_TYPE="va"; PP_SIZE=2; EP_PARALLEL_SIZE=$(( TOTAL_GPUS / PP_SIZE ))
    BS=1; NBS=16; TRAIN_ITERS=10; MODEL_SIZE="10B"; MOE_TYPE="X-MOE"; ZERO=1
    CHECKPOINT_NUM_LAYERS=0; ACTIVATION_CHECKPOINT="false"; DYNAMIC_CHECKPOINT="False"
    UNEVEN_PP="False"; RUN_PLANNER="false"; COLLECT_PROFILING_CACHE="false"
    echo "env_validate: X-MoE 10B, ${NODES}n/${TOTAL_GPUS}g, EP${EP_PARALLEL_SIZE}-PP${PP_SIZE}, ${TRAIN_ITERS} iters"
    render_and_run "va_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}"
    ;;

# ===========================================================================
loss_validate)
    RUN_TYPE="lv"; TRAIN_ITERS="${LOSS_ITERS}"; MODEL_SIZE="10B"; ZERO=1
    COLLECT_PROFILING_CACHE="false"
    if [ "$LOSS_CKPT" = "true" ]; then ACTIVATION_CHECKPOINT="true"; CHECKPOINT_NUM_LAYERS=1
    else                                ACTIVATION_CHECKPOINT="false"; CHECKPOINT_NUM_LAYERS=0; fi
    DYNAMIC_CHECKPOINT="False"; UNEVEN_PP="False"; RUN_PLANNER="false"
    echo "loss_validate: ${NODES}n/${TOTAL_GPUS}g | 10B | GBS=${GBS_LOSS} | ${TRAIN_ITERS} steps | mbs=${LOSS_MBS} ckpt=${LOSS_CKPT}"

    # (a) ELMoE-3D : mbs=4. 8 GPUs -> EP4-PP2; >8 -> EP8, PP = GPUS/8.
    MOE_TYPE="ELMOE-3D"; BS="$LOSS_MBS"
    if (( TOTAL_GPUS == 8 )); then EP_PARALLEL_SIZE=4; PP_SIZE=2
    else EP_PARALLEL_SIZE=8; PP_SIZE=$(( TOTAL_GPUS / 8 )); fi
    DP=$(( TOTAL_GPUS / PP_SIZE / MP_SIZE ))
    (( GBS_LOSS % (BS * DP) == 0 )) || { echo "ERROR: ELMoE GBS=${GBS_LOSS} not divisible by mbs*DP." >&2; exit 1; }
    NBS=$(( GBS_LOSS / (BS * DP) ))
    echo "  (a) ELMoE-3D  EP${EP_PARALLEL_SIZE}-PP${PP_SIZE} mbs=${BS} nbs=${NBS} dp=${DP} -> GBS=$(( BS*NBS*DP ))"
    render_and_run "lv_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_pp${PP_SIZE}_mbs${BS}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}"

    # (b) X-MoE : PP=1, EP=all. mbs = largest of {4,2,1} keeping nbs integral.
    MOE_TYPE="X-MOE"; PP_SIZE=1; EP_PARALLEL_SIZE=$TOTAL_GPUS
    DP=$(( TOTAL_GPUS / PP_SIZE / MP_SIZE )); BS=0
    for c in $(seq "$LOSS_MBS" -1 1); do (( GBS_LOSS % (c * DP) == 0 )) && { BS=$c; break; }; done
    (( BS != 0 )) || { echo "ERROR: X-MoE: no mbs in {4,2,1} divides GBS=${GBS_LOSS} at DP=${DP}." >&2; exit 1; }
    NBS=$(( GBS_LOSS / (BS * DP) ))
    echo "  (b) X-MoE     EP${EP_PARALLEL_SIZE}-PP${PP_SIZE} mbs=${BS} nbs=${NBS} dp=${DP} -> GBS=$(( BS*NBS*DP ))"
    render_and_run "lv_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_pp${PP_SIZE}_mbs${BS}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}"

    echo; echo "When both runs finish:  python3 ../utils/plot_loss_curve.py"
    ;;

# ===========================================================================
profiling_cache)
    # Single-layer (_1L) profiling, so it fits on ONE node regardless of the model.
    # Uses <MODEL_SIZE_OPT>_1L (so 21B_1L by default). The planner cache key encodes
    # hidden/experts/ffn/topk/seqlen but NOT depth, so 21B_1L, 50B_1L and 63B_1L all
    # resolve to the SAME cache file -- one sweep serves the whole family.
    # with yes_planner and the planner needs a cache for THAT model. Override with
    # --model-size 50B (40GB HBM) or 10B (scaled-down box).
    #
    # PINNED TO ONE NODE, like scripts-frontier. The cache filename encodes the
    # node/GPU count (N1_n8_...), and `training` always looks up the key for its
    # EP group, which is EP=8 => one node. Profiling on --gpus 32 would write
    # N4_n32_... and the training-side check would never find it.
    RUN_TYPE="pr"; PP_SIZE=1
    NODES=1; TOTAL_GPUS=$GPUS_PER_NODE; EP_PARALLEL_SIZE=$(( TOTAL_GPUS / MP_SIZE ))
    MODEL_SIZE="$MODEL_SIZE_OPT"
    case "$MODEL_SIZE" in *_1L) ;; *) MODEL_SIZE="${MODEL_SIZE}_1L" ;; esac
    NBS=20; TRAIN_ITERS=30; MOE_TYPE="X-MOE"; ZERO=1
    CHECKPOINT_NUM_LAYERS=0; ACTIVATION_CHECKPOINT="false"; DYNAMIC_CHECKPOINT="False"
    UNEVEN_PP="False"; RUN_PLANNER="false"; COLLECT_PROFILING_CACHE="true"
    echo "profiling_cache: ${MODEL_SIZE}, ${NODES}n/${TOTAL_GPUS}g, sweeping MBS 1..8 (sequential)"
    for BS in {1..8}; do
        render_and_run "pr_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}_mbs${BS}"
    done
    ;;

# ===========================================================================
# MAIN_RESULTS — the 5 headline comparison runs (Experiment 1)
# ===========================================================================
#   1. ELMoE SeqGEMM + yes_planner   (dynamic-ckpt, uneven PP, ELM-PP planner)
#   2. ELMoE SeqGEMM + no_planner    (full activation ckpt, even PP)
#   3. X-MoE     4. DS-MoE     5. DS-Tutel
#
# DS-TED is excluded: at TP=1 it is equivalent to DS-MoE. (Still runnable alone
# via `training DS-TED`.)
#
# Paper scale is 63B on 32 GPUs. Unlike SLURM — where these are 5 independent
# queued jobs that run in PARALLEL — here they contend for the same GPUs, so they
# run STRICTLY SEQUENTIALLY (~5 x 30 min).
#
# Smaller box? Scale down:  main_results --gpus 8 --model-size 10B
# (throughput numbers will NOT match the paper, but the full pipeline is exercised.)
main_results)
    MS="$MODEL_SIZE_OPT"
    MODEL_SIZE="$MS"; require_gpus_for_model

    echo "=================================================================="
    echo " main_results: 5 runs | ${MS} | ${NODES} node(s) / ${TOTAL_GPUS} GPUs | SEQUENTIAL"
    echo "=================================================================="

    declare -a RUNS=(
        "training ELMoE seqgemm yes_planner"
        "training ELMoE seqgemm no_planner"
        "training X-MoE"
        "training DS-MoE"
        "training DS-Tutel"
    )
    n=${#RUNS[@]}; okc=0; failc=0; i=0
    for run in "${RUNS[@]}"; do
        i=$((i+1))
        echo; echo "----- [${i}/${n}] ${run}  (${MS}, ${TOTAL_GPUS} GPUs) -----"
        if bash "$0" ${run} --gpus "$GPUS" --gpus-per-node "$GPUS_PER_NODE" --model-size "$MS"; then
            okc=$((okc+1))
        else
            failc=$((failc+1))
            echo "  [main_results] WARNING: '${run}' failed (continuing)."
        fi
    done
    echo; echo "main_results summary: ${okc}/${n} completed, ${failc} failed."
    ;;

*)
    echo "ERROR: unknown mode '$MODE' (expected training, env_validate, loss_validate, main_results, profiling_cache)." >&2
    usage ;;
esac

echo
echo "Done."
