#!/bin/bash
###############################################################################
# autorun.sh — non-SLURM (torchrun) counterpart of scripts-frontier/autorun_frontier.sh
#
# Same four maps, same colon-delimited config strings, same sed-fill of the
# template. The ONLY differences:
#   * TEMPLATE_FILE  -> elmoe.sh.template   (torchrun, not sbatch)
#   * no PARTITION / no walltime            (there is no scheduler)
#   * runs are executed SEQUENTIALLY with `bash` instead of queued with `sbatch`
#     — on a non-SLURM box every run contends for the SAME GPUs, so they cannot
#     overlap the way independent SLURM jobs do.
#   * a numeric JOB_ID is minted here and the driver's stdout is tee'd to
#     ${JOB_DIR}/a-xmoe-${RUN_TYPE}-${JOB_ID}.o  (the ".o equivalent").
#     utils/analyze_memory.py greps `job_(\d+)` and globs `*<job_id>.o`, and
#     utils/rerun_analysis.sh greps `job_[0-9]+`, so JOB_ID MUST be all digits.
#
# Multi-node: put one hostname per line in ./hostfile (see hostfile.example) and
# set the node count in the map key. Site env (conda/modules/RCCL) goes in
# ./env.sh (see env.sh.example) — it is sourced on every node, because ssh does
# not carry your environment.
#
# See scripts-frontier/autorun_frontier.sh for the full description of the
# config-string format; it is identical here.
###############################################################################

echo "Starting experiment launch (torchrun / non-SLURM)..."

MP_SIZE=1

# ELMoE PP-centric runs
declare -A PP_STRATEGY_MAP
declare -A PP_BATCH_MAP

# ================================== 10B ELMoE (single node, 8 GPUs) ==================================
# PP2-EP4
PP_STRATEGY_MAP["1:8"]="2:4"
PP_BATCH_MAP["1:8"]=" 4:20:15:ELMOE-3D:10B:no-ckpt:0:even:no-planner "

# ================================== 63B ELMoE (4 nodes, 32 GPUs) ==================================
# PP4-EP8
# PP_STRATEGY_MAP["4:32"]="4:8"
# PP_BATCH_MAP["4:32"]=" 1:128:15:ELMOE-3D:63B:dynamic-ckpt:1:uneven:yes-planner "
# PP_BATCH_MAP["4:32"]=" 1:128:15:ELMOE-GroupedGEMM-primus:63B:dynamic-ckpt:1:uneven:yes-planner "


# EP-centric baselines
declare -A EP_BATCH_MAP

# ================================== 10B baselines (single node, 8 GPUs) ==================================
# NODES:EP:TP
# EP_BATCH_MAP["1:8:1"]=" 4:10:15:X-MOE:10B:ckpt:1:1 "
# EP_BATCH_MAP["1:8:1"]=" 4:10:15:X-MOE:10B:ckpt:1:1  4:10:15:DS-MOE:10B:ckpt:1:1  4:10:15:TUTEL-MOE:10B:ckpt:1:1 "


# Profiling-cache generation (single node); sweeps MBS 1..8
declare -A PROFILE_MAP
# PROFILE_MAP["1:8:1"]=" 1:20:30:X-MOE:10B_1L:no-ckpt:0 "


##########################################################################
# --- SCRIPT LOGIC (no need to edit below) ---
##########################################################################
TEMPLATE_FILE="elmoe.sh.template"
TEMP_DIR="temp_sh"

if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "ERROR: The template file '$TEMPLATE_FILE' was not found." >&2
    exit 1
fi
mkdir -p "$TEMP_DIR" logs

# ---------------------------------------------------------------------------
# mint_job_id — numeric-only id, unique against existing logs/job_<id>_* dirs.
# MUST stay all-digits (see header).
# ---------------------------------------------------------------------------
mint_job_id() {
    local jid
    jid=$(date +%s)
    while compgen -G "logs/job_${jid}_*" > /dev/null; do
        jid=$((jid + 1))
    done
    echo "$jid"
}

# ---------------------------------------------------------------------------
# render_and_run — fill every placeholder, then RUN it (sequentially), teeing the
# driver's stdout to the .o that run_analysis()/analyze_memory.py expect.
# ---------------------------------------------------------------------------
render_and_run() {
    local job_name="$1"
    local temp_script="${TEMP_DIR}/${job_name}.sh"

    JOB_ID=$(mint_job_id)
    export JOB_ID

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
        "${TEMPLATE_FILE}" > "${temp_script}"

    # Sequential: these share the same GPUs, unlike independent SLURM jobs.
    bash "${temp_script}" 2>&1 | tee "${o_file}"
    return "${PIPESTATUS[0]}"
}

# ---------------------------------------------------------------------------
# derive_checkpoint_flags — identical mapping to autorun_frontier.sh
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


# --- PIPELINE PARALLEL (PP) RUNS ---
echo
echo "------------------------------------------------"
echo "--- Pipeline Parallel (PP) runs ---"
echo "------------------------------------------------"
for node_key in "${!PP_STRATEGY_MAP[@]}"; do
    IFS=':' read -r NODES TOTAL_GPUS <<< "$node_key"
    pp_strategies_string=${PP_STRATEGY_MAP[$node_key]}
    batch_configs_string=${PP_BATCH_MAP[$node_key]}

    if [ -z "$batch_configs_string" ]; then
        echo "Warning: No batch configs for PP node setup ${node_key}. Skipping."
        continue
    fi

    for pp_strategy in $pp_strategies_string; do
        IFS=':' read -r PP_SIZE EP_PARALLEL_SIZE <<< "$pp_strategy"

        for batch_config in $batch_configs_string; do
            IFS=':' read -r BS NBS TRAIN_ITERS MOE_TYPE MODEL_SIZE CHECKPOINT CHECKPOINT_NUM_LAYERS PP_PARTITION PLANNER_MODE <<< "$batch_config"

            derive_checkpoint_flags
            COLLECT_PROFILING_CACHE="false"
            ZERO=1
            RUN_TYPE="pp"

            # yes-planner requires the profiling cache for THIS machine.
            if [[ "$RUN_PLANNER" == "true" || "$RUN_PLANNER" == "true-membal" ]]; then
                EP_NODES=$(( EP_PARALLEL_SIZE / 8 ))
                [ "$EP_NODES" -lt 1 ] && EP_NODES=1
                CACHE_FILENAME=$(python3 ../utils/model_registry.py get_filename \
                    --model_size "$MODEL_SIZE" --nodes "$EP_NODES" \
                    --total_gpus "$EP_PARALLEL_SIZE" --mbs 1)
                CACHE_FILE="./planner_profiling_cache/${CACHE_FILENAME}"
                if [ -f "$CACHE_FILE" ]; then
                    echo "  [Planner] Cache found: $CACHE_FILENAME"
                else
                    echo "  [Planner] Cache MISSING: $CACHE_FILE" >&2
                    echo "            Caches are hardware-specific — generate one on THIS machine" >&2
                    echo "            via PROFILE_MAP before using yes-planner. Skipping." >&2
                    continue
                fi
            fi

            JOB_NAME="pp_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}"
            render_and_run "$JOB_NAME"
        done
    done
done


# --- EXPERT PARALLEL (EP) RUNS ---
echo
echo "-----------------------------------------------"
echo "--- Expert Parallel (EP) runs ---"
echo "-----------------------------------------------"
for node_key in "${!EP_BATCH_MAP[@]}"; do
    IFS=':' read -r NODES EP_PARALLEL_SIZE MP_SIZE <<< "$node_key"
    batch_configs_string=${EP_BATCH_MAP[$node_key]}

    PP_SIZE=1
    TOTAL_GPUS=$(( NODES * 8 ))

    for batch_config in $batch_configs_string; do
        IFS=':' read -r BS NBS TRAIN_ITERS MOE_TYPE MODEL_SIZE CHECKPOINT CHECKPOINT_NUM_LAYERS ZERO <<< "$batch_config"

        if [[ "$CHECKPOINT" == "ckpt" ]]; then ACTIVATION_CHECKPOINT="true"; else ACTIVATION_CHECKPOINT="false"; fi
        DYNAMIC_CHECKPOINT="False"; UNEVEN_PP="False"; RUN_PLANNER="false"
        COLLECT_PROFILING_CACHE="false"
        RUN_TYPE="ep"

        JOB_NAME="ep_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}"
        render_and_run "$JOB_NAME"
    done
done


# --- PROFILING-CACHE RUNS (MBS sweep 1..8) ---
echo
echo "-----------------------------------------------"
echo "--- Profiling-cache runs ---"
echo "-----------------------------------------------"
for node_key in "${!PROFILE_MAP[@]}"; do
    IFS=':' read -r NODES TOTAL_GPUS MP_SIZE <<< "$node_key"
    batch_configs_string=${PROFILE_MAP[$node_key]}

    PP_SIZE=1
    EP_PARALLEL_SIZE=$(( TOTAL_GPUS / MP_SIZE ))

    for batch_config in $batch_configs_string; do
        IFS=':' read -r _BS NBS TRAIN_ITERS MOE_TYPE MODEL_SIZE CHECKPOINT CHECKPOINT_NUM_LAYERS <<< "$batch_config"

        ACTIVATION_CHECKPOINT="false"; DYNAMIC_CHECKPOINT="False"; UNEVEN_PP="False"
        RUN_PLANNER="false"; COLLECT_PROFILING_CACHE="true"
        CHECKPOINT_NUM_LAYERS=0
        ZERO=1
        RUN_TYPE="pr"

        for BS in {1..8}; do
            JOB_NAME="pr_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}_mbs${BS}"
            render_and_run "$JOB_NAME"
        done
    done
done

echo
echo "All runs complete."
