#!/bin/bash

echo "Starting experiment submission process..."

##########################################################################
# --- 1. CONFIGURE YOUR EXPERIMENTS HERE ---
#
# We now use associative arrays (maps) to link configurations together.
# The KEY for each map is the node configuration: "NODES:TOTAL_GPUS"
# The VALUE is a space-separated string of the settings for that key.
##########################################################################

# Set the Tensor Parallelism size (usually constant)
MP_SIZE=1

# --- A) PIPELINE PARALLEL (PP) CONFIGURATIONS ---


# KEY: NODE:TOTAL_GPU
# VALUE="PP_SIZE:EP_PARALLEL_SIZE"
declare -A PP_STRATEGY_MAP
PP_STRATEGY_MAP["1:4"]="2:2" 
PP_STRATEGY_MAP["2:8"]="2:4" 
PP_STRATEGY_MAP["4:16"]="4:4" 
PP_STRATEGY_MAP["1:8"]="2:4 4:2" 
PP_STRATEGY_MAP["2:16"]="2:8 4:4 8:2" 
PP_STRATEGY_MAP["4:32"]="4:8 8:4" 


# KEY: NODE:TOTAL_GPU
# VALUE="BS:GBS:TRAIN_ITERS:MOE_TYPE"
declare -A PP_BATCH_MAP

# PP_BATCH_MAP["2:8"]="2:96:20:X-MOE   2:96:20:DS-MOE"
# PP_BATCH_MAP["2:8"]="1:96:20:X-MOE:10B   1:96:20:DS-MOE:10B"
# PP_BATCH_MAP["2:8"]="1:96:100:X-MOE:10B 1:96:100:DS-MOE:10B 2:96:100:X-MOE:10B 2:96:100:DS-MOE:10B"
# PP_BATCH_MAP["1:4"]="1:96:100:X-MOE:10B  1:96:100:X-MOE:10B  1:96:100:DS-MOE:10B 1:96:100:DS-MOE:10B  2:96:100:X-MOE:10B  2:96:100:X-MOE:10B"
# PP_BATCH_MAP["2:8"]="1:96:100:X-MOE:10B  1:96:100:X-MOE:10B  1:96:100:DS-MOE:10B 1:96:100:DS-MOE:10B   2:96:100:X-MOE:10B  2:96:100:X-MOE:10B  3:96:100:X-MOE:10B  3:96:100:X-MOE:10B "
# PP_BATCH_MAP["4:16"]="1:96:100:X-MOE:10B  1:96:100:X-MOE:10B  1:96:100:DS-MOE:10B 1:96:100:DS-MOE:10B   2:96:100:X-MOE:10B  2:96:100:X-MOE:10B  3:96:100:X-MOE:10B  3:96:100:X-MOE:10B  4:96:100:X-MOE:10B  4:96:100:X-MOE:10B "
# PP_BATCH_MAP["4:16"]="1:96:100:X-MOE:10B 1:96:100:DS-MOE:10B 2:96:100:X-MOE:10B 2:96:100:DS-MOE:10B  3:96:100:X-MOE:10B "
# PP_BATCH_MAP["4:16"]="4:96:100:X-MOE:10B "


MOE_TYPES=("X-MOE")
# Static values that will be appended to each configuration
TRAIN_ITERS="100"
MODEL_SIZE="10B"
generated_configs=""
# Loop through each BS:GBS pair
for bs_gbs in "${BS_GBS_CONFIGS[@]}"; do
  # Loop through each MOE type for the current BS:GBS pair
  for moe_type in "${MOE_TYPES[@]}"; do
    batch_config="${bs_gbs}:${TRAIN_ITERS}:${moe_type}:${MODEL_SIZE}"
    # Use a for loop to repeat the item 3 times
    for i in {1..1}; do
      generated_configs+="${batch_config} "
    done
  done
done
generated_configs=$(echo "${generated_configs}" | sed 's/ *$//')

# PP_BATCH_MAP["1:8"]="${generated_configs}"
# PP_BATCH_MAP["2:16"]="${generated_configs}"
# PP_BATCH_MAP["4:32"]="${generated_configs}"

# PP_BATCH_MAP["2:8"]="1:128:30:X-MOE:10B 2:128:30:X-MOE:10B"


declare -A EP_BATCH_MAP

EP_BATCH_MAP["2:8"]="1:64:30:X-MOE:10B_div2 2:64:30:X-MOE:10B_div2 3:96:30:X-MOE:10B_div2 4:128:30:X-MOE:10B_div2 5:320:30:X-MOE:10B_div2 6:384:30:X-MOE:10B_div2 "
EP_BATCH_MAP["2:16"]="1:64:30:X-MOE:10B_div2 2:128:30:X-MOE:10B_div2 3:192:30:X-MOE:10B_div2 4:256:30:X-MOE:10B_div2 5:320:30:X-MOE:10B_div2 6:384:30:X-MOE:10B_div2 "




##########################################################################
# --- 2. SCRIPT LOGIC (No need to edit below) ---
##########################################################################
TEMPLATE_FILE="xmoe_unified.slurm.template"

if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "ERROR: The template file '$TEMPLATE_FILE' was not found."
    exit 1
fi

# --- LAUNCH PIPELINE PARALLEL (PP) RUNS ---
echo
echo "------------------------------------------------"
echo "--- Submitting Pipeline Parallel (PP) jobs ---"
echo "------------------------------------------------"

# Iterate over the node configurations defined in the PP strategy map
for node_key in "${!PP_STRATEGY_MAP[@]}"; do
    IFS=':' read -r NODES TOTAL_GPUS <<< "$node_key"

    # Get the specific strategies and batch configs for this node setup
    pp_strategies_string=${PP_STRATEGY_MAP[$node_key]}
    batch_configs_string=${PP_BATCH_MAP[$node_key]}

    if [ -z "$batch_configs_string" ]; then
        echo "Warning: No batch configs found for PP node setup ${node_key}. Skipping."
        continue
    fi

    # --- ADDED LOGIC TO DETERMINE PARTITION ---
    GPUS_PER_NODE=$(( TOTAL_GPUS / NODES ))
    if (( GPUS_PER_NODE == 8 )); then
        PARTITION="mi2508x"
    else
        PARTITION="mi2104x"
    fi

    echo "Found PP configurations for ${NODES} nodes, ${TOTAL_GPUS} GPUs..."

    for pp_strategy in $pp_strategies_string; do
        IFS=':' read -r PP_SIZE EP_PARALLEL_SIZE <<< "$pp_strategy"

        # Sanity Check
        if (( PP_SIZE * EP_PARALLEL_SIZE * MP_SIZE != TOTAL_GPUS )); then
            echo "  - Invalid strategy: PP=${PP_SIZE}, EP=${EP_PARALLEL_SIZE} for ${TOTAL_GPUS} GPUs. Skipping."
            continue
        fi

        for batch_config in $batch_configs_string; do
            IFS=':' read -r BS GBS TRAIN_ITERS MOE_TYPE MODEL_SIZE <<< "$batch_config"

            RUN_TYPE="pp"
            JOB_NAME="${RUN_TYPE}_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_gbs${GBS}_i${TRAIN_ITERS}_${MOE_TYPE}"
            TEMP_SLURM_SCRIPT="temp_slurm_${JOB_NAME}.slurm"

            echo "  - Generating job: ${JOB_NAME}"

            sed -e "s/{{RUN_TYPE}}/${RUN_TYPE}/g" \
                -e "s/{{NODES}}/${NODES}/g" \
                -e "s/{{TOTAL_GPUS}}/${TOTAL_GPUS}/g" \
                -e "s/{{PARTITION}}/${PARTITION}/g" \
                -e "s/{{BATCH_SIZE}}/${BS}/g" \
                -e "s/{{GLOBAL_BATCH_SIZE}}/${GBS}/g" \
                -e "s/{{PP_SIZE}}/${PP_SIZE}/g" \
                -e "s/{{EP_PARALLEL_SIZE}}/${EP_PARALLEL_SIZE}/g" \
                -e "s/{{MP_SIZE}}/${MP_SIZE}/g" \
                -e "s/{{TRAIN_ITERS}}/${TRAIN_ITERS}/g" \
                -e "s/{{MOE_TYPE}}/${MOE_TYPE}/g" \
                -e "s/{{MODEL_SIZE}}/${MODEL_SIZE}/g" \
                ${TEMPLATE_FILE} > ${TEMP_SLURM_SCRIPT}

            sbatch ${TEMP_SLURM_SCRIPT}
            # rm ${TEMP_SLURM_SCRIPT}
        done
    done
done


# --- LAUNCH EXPERT PARALLEL (EP) RUNS ---
echo
echo "-----------------------------------------------"
echo "--- Submitting Expert Parallel (EP) jobs ---"
echo "-----------------------------------------------"

# Iterate over the node configurations defined in the EP batch map
for node_key in "${!EP_BATCH_MAP[@]}"; do
    IFS=':' read -r NODES TOTAL_GPUS <<< "$node_key"

    # Get the specific batch configs for this node setup
    batch_configs_string=${EP_BATCH_MAP[$node_key]}

    # --- ADDED LOGIC TO DETERMINE PARTITION ---
    GPUS_PER_NODE=$(( TOTAL_GPUS / NODES ))
    if (( GPUS_PER_NODE == 8 )); then
        PARTITION="mi2508x"
    else
        PARTITION="mi2104x"
    fi
    
    echo "Found EP configurations for ${NODES} nodes, ${TOTAL_GPUS} GPUs..."

    # For EP runs, PP=1 and EP=Total GPUs
    PP_SIZE=1
    EP_PARALLEL_SIZE=$(( TOTAL_GPUS / MP_SIZE ))

    for batch_config in $batch_configs_string; do
        IFS=':' read -r BS GBS TRAIN_ITERS MOE_TYPE MODEL_SIZE <<< "$batch_config"

        RUN_TYPE="ep"
        JOB_NAME="${RUN_TYPE}_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_gbs${GBS}_i${TRAIN_ITERS}_${MOE_TYPE}"
        TEMP_SLURM_SCRIPT="temp_slurm_${JOB_NAME}.slurm"

        echo "  - Generating job: ${JOB_NAME}"

        sed -e "s/{{RUN_TYPE}}/${RUN_TYPE}/g" \
            -e "s/{{NODES}}/${NODES}/g" \
            -e "s/{{TOTAL_GPUS}}/${TOTAL_GPUS}/g" \
            -e "s/{{PARTITION}}/${PARTITION}/g" \
            -e "s/{{BATCH_SIZE}}/${BS}/g" \
            -e "s/{{GLOBAL_BATCH_SIZE}}/${GBS}/g" \
            -e "s/{{PP_SIZE}}/${PP_SIZE}/g" \
            -e "s/{{EP_PARALLEL_SIZE}}/${EP_PARALLEL_SIZE}/g" \
            -e "s/{{MP_SIZE}}/${MP_SIZE}/g" \
            -e "s/{{TRAIN_ITERS}}/${TRAIN_ITERS}/g" \
            -e "s/{{MOE_TYPE}}/${MOE_TYPE}/g" \
            -e "s/{{MODEL_SIZE}}/${MODEL_SIZE}/g" \
            ${TEMPLATE_FILE} > ${TEMP_SLURM_SCRIPT}

        sbatch ${TEMP_SLURM_SCRIPT}
        # rm ${TEMP_SLURM_SCRIPT}
    done
done

echo
echo "All jobs submitted."