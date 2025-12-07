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
# (Example: PP_STRATEGY_MAP["4:32"]="4:8 8:4")
# (Example: PP_BATCH_MAP["4:32"]="4:256:100:X-MOE:10.1")
declare -A PP_STRATEGY_MAP
PP_STRATEGY_MAP["1:4"]="4:1" 
PP_STRATEGY_MAP["1:8"]="4:2" 
PP_STRATEGY_MAP["2:8"]="2:4" 
PP_STRATEGY_MAP["2:16"]="2:8" 
PP_STRATEGY_MAP["4:16"]="4:4" 
PP_STRATEGY_MAP["4:32"]="4:8" 
PP_STRATEGY_MAP["5:40"]="5:8" 
PP_STRATEGY_MAP["6:48"]="6:8" 
PP_STRATEGY_MAP["8:64"]="8:8" 
PP_STRATEGY_MAP["12:96"]="12:8" 
PP_STRATEGY_MAP["24:192"]="24:8" 
PP_STRATEGY_MAP["60:480"]="15:32 30:16 60:8" 


declare -A PP_BATCH_MAP
# PP_BATCH_MAP["1:8"]="1:96:100:X-MOE:10B 2:96:100:X-MOE:10B 3:96:100:X-MOE:10B 2:192:100:X-MOE:10B 3:288:100:X-MOE:10B "
# PP_BATCH_MAP["2:16"]="1:96:100:X-MOE:10B 2:96:100:X-MOE:10B 3:96:100:X-MOE:10B 2:192:100:X-MOE:10B 3:288:100:X-MOE:10B"

# PP_BATCH_MAP["1:8"]="1:96:20:X-MOE:10B"
# PP_BATCH_MAP["2:16"]="1:96:20:X-MOE:10B"
# PP_BATCH_MAP["4:32"]="1:1152:20:X-MOE:50B:ckpt 1:1152:20:X-MOE:50B:no-ckpt 2:1152:20:X-MOE:50B:ckpt 2:1152:20:X-MOE:50B:no-ckpt 3:1152:20:X-MOE:50B:ckpt 3:1152:20:X-MOE:50B:no-ckpt 4:1152:20:X-MOE:50B:ckpt 4:1152:20:X-MOE:50B:no-ckpt "
# PP_BATCH_MAP["4:32"]="5:960:20:X-MOE:50B:ckpt 6:1152:20:X-MOE:50B:ckpt 7:1344:20:X-MOE:50B:ckpt 8:1344:20:X-MOE:50B:ckpt "
# PP_BATCH_MAP["1:8"]="1:64:100:X-MOE:10B:no-ckpt:1"
# PP_BATCH_MAP["1:4"]="2:128:20:X-MOE:10B:no-ckpt:0 2:128:20:X-MOE:10B:ckpt:1 2:128:20:X-MOE:10B:ckpt:2"
# PP_BATCH_MAP["4:32"]="1:128:20:X-MOE:50B:no-ckpt:0 1:128:20:X-MOE:50B:ckpt:1 1:128:20:X-MOE:50B:ckpt:2   2:256:20:X-MOE:50B:no-ckpt:0 2:256:20:X-MOE:50B:ckpt:1 2:256:20:X-MOE:50B:ckpt:2"
# PP_BATCH_MAP["1:4"]="1:128:20:X-MOE:10B:ckpt:3   2:256:20:X-MOE:10B:ckpt:3"
# PP_BATCH_MAP["4:32"]="1:128:2:X-MOE:50B:ckpt:1 "


# PP_BATCH_MAP["60:480"]="1:1920:14:X-MOE:537B:ckpt:1 1:1920:14:X-MOE:537B:dynamic-ckpt:1  "
# PP_BATCH_MAP["4:32"]="1:96:3:X-MOE:50B:ckpt:1:even 1:96:3:X-MOE:50B:dynamic-ckpt:1:uneven 1:96:3:X-MOE:50B:dynamic-ckpt:1:even"
# PP_BATCH_MAP["1:8"]="1:960:20:X-MOE:10B:ckpt:1:uneven 1:960:20:X-MOE:10B:dynamic-ckpt:1:even 1:960:20:X-MOE:10B:dynamic-ckpt:1:uneven 1:960:20:X-MOE:10B:ckpt:1:even 1:960:20:X-MOE:10B:no-ckpt:1:even "
# PP_BATCH_MAP["4:32"]="1:960:20:X-MOE:50B:ckpt:1:uneven 1:960:20:X-MOE:50B:dynamic-ckpt:1:even 1:960:20:X-MOE:50B:dynamic-ckpt:1:uneven 1:960:20:X-MOE:50B:ckpt:1:even 1:960:20:X-MOE:50B:no-ckpt:1:even "
# PP_BATCH_MAP["1:8"]="4:96:5:X-MOE:10B:no-ckpt:1:even 4:96:5:X-MOE:10B:ckpt:1:even "
# PP_BATCH_MAP["4:32"]="1:96:5:X-MOE:50B:no-ckpt:1:even 1:96:5:X-MOE:50B:ckpt:1:even "

# check planner throughput 
# PP_BATCH_MAP["4:32"]="   4:1280:14:X-MOE:63B:ckpt:1:even   3:960:16:X-MOE:63B:ckpt:1:even     "
# PP_BATCH_MAP["4:32"]="  1:720:20:X-MOE:63B:dynamic-ckpt:1:uneven   1:720:20:X-MOE:63B:ckpt:1:even  2:1440:13:X-MOE:63B:ckpt:1:even "
# PP_BATCH_MAP["4:32"]="7:1120:20:X-MOE:50B:ckpt:1:uneven 8:1280:20:X-MOE:50B:ckpt:1:uneven 6:960:20:X-MOE:50B:ckpt:1:even"


# PP_BATCH_MAP["1:8"]="1:96:3:X-MOE:10B:dynamic-ckpt:1:uneven"
# PP_BATCH_MAP["1:4"]="1:8:2:X-MOE:10B:ckpt:1"

# PP_BATCH_MAP["4:32"]="1:1152:20:X-MOE:50B:no-ckpt"



# PP_BATCH_MAP["6:48"]="1:256:30:X-MOE:190B_6L 1:256:30:X-MOE:190B"
# PP_BATCH_MAP["8:64"]="1:256:30:X-MOE:190B_8L 1:256:30:X-MOE:190B"
# PP_BATCH_MAP["24:192"]="1:2048:30:X-MOE:190B"
# PP_BATCH_MAP["12:96"]="1:256:30:X-MOE:190B"
# PP_BATCH_MAP["12:96"]="1:256:30:X-MOE:190B"
# PP_BATCH_MAP["4:32"]="1:2048:30:X-MOE:50B "
# PP_BATCH_MAP["5:40"]="1:80:20:X-MOE:50B "
# PP_BATCH_MAP["6:48"]="1:512:20:X-MOE:50B 2:512:20:X-MOE:50B 2:1024:20:X-MOE:50B"



# PP_BATCH_MAP["4:32"]="1:1152:20:X-MOE:50B:ckpt:2 2:1152:20:X-MOE:50B:ckpt:2 4:1152:20:X-MOE:50B:ckpt:2  6:1152:20:X-MOE:50B:ckpt:2    1:1152:20:X-MOE:50B:ckpt:3 2:1152:20:X-MOE:50B:ckpt:3 4:1152:20:X-MOE:50B:ckpt:3 6:1152:20:X-MOE:50B:ckpt:3  "
# PP_BATCH_MAP["24:192"]="1:1152:15:X-MOE:190B:ckpt:1 2:1152:15:X-MOE:190B:ckpt:1 4:1152:15:X-MOE:190B:ckpt:1 "
# PP_BATCH_MAP["12:96"]="1:1152:15:X-MOE:190B:ckpt:1 2:1152:15:X-MOE:190B:ckpt:1 4:1152:15:X-MOE:190B:ckpt:1     1:1152:15:X-MOE:190B:ckpt:2 2:1152:15:X-MOE:190B:ckpt:2 4:1152:15:X-MOE:190B:ckpt:2 "
# PP_BATCH_MAP["8:64"]="1:1152:15:X-MOE:190B:ckpt:1"
# PP_BATCH_MAP["24:192"]="1:1152:15:X-MOE:190B:ckpt:1:even 1:1152:15:X-MOE:190B:dynamic-ckpt:1:even "
# PP_BATCH_MAP["12:96"]="1:1152:15:X-MOE:190B:ckpt:1:even 1:1152:15:X-MOE:190B:dynamic-ckpt:1:even "
# PP_BATCH_MAP["8:64"]="1:1152:15:X-MOE:190B:ckpt:1:even 1:1152:15:X-MOE:190B:dynamic-ckpt:1:even "



# PP_BATCH_MAP["24:192"]="1:96:5:X-MOE:10B"
# PP_BATCH_MAP["12:96"]="1:96:30:X-MOE:50B"

# PP_BATCH_MAP["1:8"]="1:16:5:X-MOE:10B:ckpt"

# --- B) EXPERT PARALLEL (EP) CONFIGURATIONS ---
# This example is adapted from your latest slurm script.
# KEY: NODE:TOTAL_GPU
# VALUE="BS:GBS:TRAIN_ITERS:MOE_TYPE:MODEL_SIZE"
declare -A EP_BATCH_MAP
# EP_BATCH_MAP["4:32"]="1:64:20:X-MOE:50B "
# EP_BATCH_MAP["8:64"]="1:512:20:X-MOE:50B 2:512:20:X-MOE:50B 2:1024:20:X-MOE:50B"
# EP_BATCH_MAP["8:64"]="1:64:15:DS-MOE:10B 2:128:15:DS-MOE:10B 3:192:15:DS-MOE:10B"
# EP_BATCH_MAP["16:128:2"]="1:2048:30:X-MOE:190B"


EP_BATCH_MAP["1:8:1"]=" 1:32:15:X-MOE:173B_1L:no-ckpt:0  "
# EP_BATCH_MAP["4:32:1"]=" 1:1024:15:X-MOE:50B:no-ckpt:0   1:1024:15:X-MOE:50B:ckpt:1 2:1024:15:X-MOE:50B:ckpt:1 4:1024:15:X-MOE:50B:ckpt:1    "
# EP_BATCH_MAP["16:128:2"]="1:1024:15:X-MOE:190B:no-ckpt:0    1:1024:15:X-MOE:190B:ckpt:1 2:1024:15:X-MOE:190B:ckpt:1 4:1024:15:X-MOE:190B:ckpt:1   "


# EP_BATCH_MAP["2:16"]="1:64:4:X-MOE:10B"
# EP_BATCH_MAP["1:8:1"]="1:24:2:X-MOE:10B:no-ckpt:0 "
# EP_BATCH_MAP["2:16"]="1:96:30:X-MOE:10B"
# EP_BATCH_MAP["1:8"]="1:96:20:X-MOE:10B "
# EP_BATCH_MAP["12:96"]="1:96:30:X-MOE:50B"
# EP_BATCH_MAP["4:32"]="1:96:30:X-MOE:190B_div_4 1:96:30:X-MOE:10B"
# Add other EP configurations here, for example:
# EP_BATCH_MAP["2:16"]="1:128:30:X-MOE:10.1 2:128:30:X-MOE:10.1"
# EP_BATCH_MAP["4:32"]="1:96:30:X-MOE:50B"


##########################################################################
# --- 2. SCRIPT LOGIC (No need to edit below) ---
##########################################################################
TEMPLATE_FILE="frontier_xmoe.slurm.template"

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

    # Set partition for the new environment
    PARTITION="batch"

    echo "Found PP configurations for ${NODES} nodes, ${TOTAL_GPUS} GPUs..."

    for pp_strategy in $pp_strategies_string; do
        IFS=':' read -r PP_SIZE EP_PARALLEL_SIZE <<< "$pp_strategy"

        # Sanity Check
        if (( PP_SIZE * EP_PARALLEL_SIZE * MP_SIZE != TOTAL_GPUS )); then
            echo "  - Invalid strategy: PP=${PP_SIZE}, EP=${EP_PARALLEL_SIZE} for ${TOTAL_GPUS} GPUs. Skipping."
            continue
        fi

        for batch_config in $batch_configs_string; do
            IFS=':' read -r BS GBS TRAIN_ITERS MOE_TYPE MODEL_SIZE CHECKPOINT CHECKPOINT_NUM_LAYERS PP_PARTITION <<< "$batch_config"

            if [[ "$CHECKPOINT" == "ckpt" ]]; then
                ACTIVATION_CHECKPOINT="true"
                DYNAMIC_CHECKPOINT="False" # Capital F
            elif [[ "$CHECKPOINT" == "dynamic-ckpt" ]]; then
                ACTIVATION_CHECKPOINT="true"
                DYNAMIC_CHECKPOINT="True" # Capital T
            else
                ACTIVATION_CHECKPOINT="false"
                DYNAMIC_CHECKPOINT="False" # Capital F
            fi

            if [[ "$PP_PARTITION" == "uneven" ]]; then
                UNEVEN_PP_PARTITION="True" # Capital T
            else
                UNEVEN_PP_PARTITION="False" # Capital F
            fi

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
                -e "s/{{ACTIVATION_CHECKPOINT}}/${ACTIVATION_CHECKPOINT}/g" \
                -e "s/{{CHECKPOINT_NUM_LAYERS}}/${CHECKPOINT_NUM_LAYERS}/g" \
                -e "s/{{DYNAMIC_CHECKPOINT}}/${DYNAMIC_CHECKPOINT}/g" \
                -e "s/{{UNEVEN_PP_PARTITION}}/${UNEVEN_PP_PARTITION}/g" \
                ${TEMPLATE_FILE} > ${TEMP_SLURM_SCRIPT}

            sbatch ${TEMP_SLURM_SCRIPT}
            # rm ${TEMP_SLURM_SCRIPT}
            sleep 1
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
    IFS=':' read -r NODES TOTAL_GPUS MP_SIZE <<< "$node_key"

    # Get the specific batch configs for this node setup
    batch_configs_string=${EP_BATCH_MAP[$node_key]}

    # Set partition for the new environment
    PARTITION="batch"
    # MP_SIZE=2
    
    echo "Found EP configurations for ${NODES} nodes, ${TOTAL_GPUS} GPUs..."

    # For EP runs, PP=1 and EP=Total GPUs
    PP_SIZE=1
    EP_PARALLEL_SIZE=$(( TOTAL_GPUS / MP_SIZE ))

    for batch_config in $batch_configs_string; do
        IFS=':' read -r BS GBS TRAIN_ITERS MOE_TYPE MODEL_SIZE CHECKPOINT CHECKPOINT_NUM_LAYERS <<< "$batch_config"

        if [[ "$CHECKPOINT" == "ckpt" ]]; then
            ACTIVATION_CHECKPOINT="true"
        else
            ACTIVATION_CHECKPOINT="false"
        fi

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
                -e "s/{{ACTIVATION_CHECKPOINT}}/${ACTIVATION_CHECKPOINT}/g" \
                -e "s/{{CHECKPOINT_NUM_LAYERS}}/${CHECKPOINT_NUM_LAYERS}/g" \
            ${TEMPLATE_FILE} > ${TEMP_SLURM_SCRIPT}

        sbatch ${TEMP_SLURM_SCRIPT}
        # rm ${TEMP_SLURM_SCRIPT}
        sleep 1
    done
done

echo
echo "All jobs submitted."