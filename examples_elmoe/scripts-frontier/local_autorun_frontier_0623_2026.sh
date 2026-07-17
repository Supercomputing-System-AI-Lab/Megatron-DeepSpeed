#!/bin/bash

echo "Starting experiment submission process..."


# ************************************ ELMoE Launch Instructions ************************************
# 1. Specify the PP-EP configurations and #nodes & #GPUs per run using PP_STRATEGY_MAP, 
#       DP (default ZeRO-1) will be auto-determined based on GPU counts 
# 2. Each run command is composed of the following structure and is seperated by colon :
#       micro_batch_size (per-device) : 
#       number_of_micro_batches (per-device-per-iteration) : 
#       number of training iterations : 
#       run model type :
#               ELMoE-3D --> SeqGEMM
#               ELMOE-GroupedGEMM-primus --> CK GroupGEMM
#               ELMOE-GroupedGEMM-primus --> Triton GroupGEMM
#       Model sizes (see ../utils/model_registry.py for more configurations):
#               10B Small
#               63B Medium 
#               173B Large
#               537B Super 
#               1T Ultra 
#       Checkpointing strategy : Checkpoint-interval :
#               no-ckpt:0:      --> no activation checkpoint 
#               ckpt:1:         --> activation checkpoint for every layer 
#               dynamic-ckpt:1: --> ELMoE's planner based layer-wise act. ckpt. partition 
#       Planner strategy : 
#               no-planner:         --> no planner 
#               yes-planner:        --> Minimax Pipeline Planner 
#               yes-planner-membal: --> Memory balanced planning strategy (no throughput consideration)
# 3. Each PP_BATCH_MAP run command will be sent to for loop below to fill in template slurm script 
#       and launch each configuration run. Multiple runs can be launched with same PP_BATCH_MAP
#       run with space sepearted. 
# **************************************************************************************************

# Set the Tensor Parallelism size (usually constant)
MP_SIZE=1
# ELMoE PP centric approach 
declare -A PP_STRATEGY_MAP

PP_STRATEGY_MAP["36:288"]="12:8" 
PP_STRATEGY_MAP["72:576"]="12:8" 
PP_STRATEGY_MAP["144:1152"]="12:8" 
PP_STRATEGY_MAP["288:2304"]="12:8" 

declare -A PP_BATCH_MAP


# ================================ EXP ======================

# DSv2-16B 
# PP_STRATEGY_MAP["8:64"]="8:8" # **
# PP_STRATEGY_MAP["4:32"]="4:8" # **

# PP_BATCH_MAP["8:64"]=" 1:32:15:ELMOE-3D:63B:ckpt:1:uneven:yes-planner "
# PP_BATCH_MAP["8:64"]=" 1:32:10:ELMOE-3D:63B_Sparse:no-ckpt:0:even:no-planner "

# PP_BATCH_MAP["4:32"]=" 1:20:2000:ELMOE-GroupedGEMM-primus:10B:dynamic-ckpt:1:uneven:yes-planner "

# PP_BATCH_MAP["4:32"]=" 1:128:15:ELMOE-GroupedGEMM-primus:63B:dynamic-ckpt:1:uneven:yes-planner "
# PP_BATCH_MAP["4:32"]=" 1:96:10:ELMOE-3D:1T_4L:no-ckpt:0:even:no-planner"
# PP_BATCH_MAP["4:32"]=" 1:96:10:ELMOE-GroupedGEMM-primus:1T_4L:no-ckpt:0:even:no-planner"

# PP_BATCH_MAP["4:32"]=" 1:96:20:ELMOE-3D:63B:dynamic-ckpt:1:uneven:yes-planner"
# PP_BATCH_MAP["8:64"]=" 3:192:540:X-MOE:FORGE_10B:no-ckpt:0:1"
# PP_BATCH_MAP["8:64"]=" 3:192:540:ELMOE-3D:FORGE_10B:no-ckpt:0:even:no-planner"
# PP_BATCH_MAP["8:64"]=" 3:192:540:ELMOE-3D:FORGE_10B:dynamic-ckpt:1:uneven:yes-planner  "
# PP_BATCH_MAP["8:64"]=" 1:256:12:ELMOE-3D:63B:dynamic-ckpt:1:uneven:yes-planner  "

# PP_BATCH_MAP["8:64"]=" 1:256:12:ELMOE-GroupedGEMM-triton:63B:dynamic-ckpt:1:uneven:yes-planner  "

PP_STRATEGY_MAP["32:256"]="8:8" # **
# PP_BATCH_MAP["32:256"]="1:256:12:ELMOE-3D:173B:dynamic-ckpt:1:uneven:yes-planner " # **

PP_STRATEGY_MAP["120:960"]="60:8" # **
# PP_BATCH_MAP["120:960"]="1:256:15:ELMOE-GroupedGEMM-primus:1T:dynamic-ckpt:1:uneven:yes-planner " # **
# PP_BATCH_MAP["120:960"]="1:256:15:ELMOE-3D:1T:dynamic-ckpt:1:uneven:yes-planner " # **
# PP_BATCH_MAP["120:960"]="1:256:15:ELMOE-GroupedGEMM-triton:1T:dynamic-ckpt:1:uneven:yes-planner " # **

# PP_BATCH_MAP["120:960"]="1:128:15:ELMOE-3D:1T:dynamic-ckpt:1:uneven:yes-planner   1:128:15:ELMOE-GroupedGEMM-triton:1T:dynamic-ckpt:1:uneven:yes-planner   1:128:15:ELMOE-GroupedGEMM-primus:1T:dynamic-ckpt:1:uneven:yes-planner " # **

# PP_STRATEGY_MAP["60:480"]="30:8" # **
PP_STRATEGY_MAP["30:240"]="30:8" # **
# PP_BATCH_MAP["30:240"]="   1:256:15:ELMOE-GroupedGEMM-triton:537B:dynamic-ckpt:1:uneven:yes-planner   1:256:15:ELMOE-GroupedGEMM-primus:537B:dynamic-ckpt:1:uneven:yes-planner    " # **
# ================================== 63B ELMoE ==================================
# ==================== 64 GPUs ===================
# PP4-EP8-DP2
# PP_STRATEGY_MAP["8:64"]="4:8" # **
# ELMoE configurations: 
# PP_BATCH_MAP["8:64"]=" 1:256:15:ELMOE-3D:63B:dynamic-ckpt:1:uneven:yes-planner  1:256:15:ELMOE-GroupedGEMM-primus:63B:dynamic-ckpt:1:uneven:yes-planner "
# PP_BATCH_MAP["8:64"]=" 1:256:15:ELMOE-3D:63B:dynamic-ckpt:1:uneven:yes-planner  "
# PP_BATCH_MAP["8:64"]=" 1:256:15:ELMOE-GroupedGEMM-primus:63B:dynamic-ckpt:1:uneven:yes-planner "
# PP_BATCH_MAP["8:64"]=" 1:256:15:ELMOE-GroupedGEMM-primus:63B:dynamic-ckpt:1:uneven:yes-planner  " # **


# launch all three together, separated by space: 
# PP_BATCH_MAP["8:64"]=" 1:256:15:ELMOE-3D:63B:dynamic-ckpt:1:uneven:yes-planner  1:256:15:ELMOE-GroupedGEMM-primus:63B:dynamic-ckpt:1:uneven:yes-planner  1:256:15:ELMOE-GroupedGEMM-triton:63B:dynamic-ckpt:1:uneven:yes-planner "

# Without planner and even partition: 
# w/ act. ckpt:
# PP_BATCH_MAP["8:64"]=" 1:256:15:ELMOE-3D:63B:no-ckpt:0:even:no-planner "
# w/o act. ckpt:
# PP_BATCH_MAP["8:64"]=" 1:256:15:ELMOE-3D:63B:ckpt:1:even:no-planner "
# ==================== 32 GPUs ===================
# PP4-EP8-DP1
# PP_STRATEGY_MAP["4:32"]="4:8" # **
# PP_BATCH_MAP["4:32"]=" 1:256:15:ELMOE-3D:63B:dynamic-ckpt:1:uneven:yes-planner "
# PP_BATCH_MAP["4:32"]=" 1:256:15:ELMOE-GroupedGEMM-primus:63B:dynamic-ckpt:1:uneven:yes-planner  " # **




# ================================== 1T ELMoE ==================================
# ==================== 960 GPUs ===================
# PP60-EP8-DP2 
# PP_STRATEGY_MAP["120:960"]="60:8" # **
# PP_BATCH_MAP["120:960"]="1:256:15:ELMOE-GroupedGEMM-primus:1T:dynamic-ckpt:1:uneven:yes-planner " # **
# PP_BATCH_MAP["120:960"]="1:256:15:ELMOE-3D:1T:dynamic-ckpt:1:uneven:yes-planner " # **
# ==================== 480 GPUs ===================
# PP60-EP8-DP1 
# PP_STRATEGY_MAP["60:480"]="60:8" # **
# PP_BATCH_MAP["60:480"]="1:256:15:ELMOE-3D:1T:dynamic-ckpt:1:uneven:yes-planner " # **



# ================================== 173B ELMoE ==================================
# ==================== 256 GPUs ===================
# PP8-EP8-DP4 
# PP_STRATEGY_MAP["32:256"]="8:8" # **
# PP_BATCH_MAP["32:256"]="1:256:15:ELMOE-3D:173B:dynamic-ckpt:1:uneven:yes-planner " # **
# PP_BATCH_MAP["32:256"]="1:256:15:ELMOE-GroupedGEMM-primus:173B:dynamic-ckpt:1:uneven:yes-planner " # **
# PP_BATCH_MAP["32:256"]="1:256:15:ELMOE-GroupedGEMM-triton:173B:dynamic-ckpt:1:uneven:yes-planner " # **




# ================================== 537B ELMoE ==================================
# ==================== 480 GPUs ===================
# PP30-EP8-DP2 
# PP_STRATEGY_MAP["60:480"]="30:8" # **
# # Activation Checkpointing 
# PP_BATCH_MAP["60:480"]="   1:256:15:ELMOE-GroupedGEMM-primus:537B:ckpt:1:even:no-planner    " # **
# # ELMoE ELM-PP with planner and layer-wise activation partition
# PP_BATCH_MAP["60:480"]="   1:256:15:ELMOE-GroupedGEMM-primus:537B:dynamic-ckpt:1:uneven:yes-planner    " # **
# # Memory balanced partition (no throughput consideration)
# PP_BATCH_MAP["60:480"]="   1:256:15:ELMOE-GroupedGEMM-primus:537B:ckpt:1:uneven:yes-planner-membal   " # **






# ************************************ Baseline Launch Instructions ************************************
# 1. Specify the NODES:EP-DEGREE:TP-DEGREE configurations using EP_BATCH_MAP["???:???:???"]
# 2. Each run command is composed of the following structure and is seperated by colon :
#       micro_batch_size (per-device) : 
#       number_of_micro_batches (per-device-per-iteration) : 
#       number of training iterations : 
#       run model type :
#               X-MoE 
#               TUTEL-MOE 
#               TED-MOE (Deepspeed TED)
#               DS-MOE (default DeepSpeed MoE)
#       Model sizes (see ../utils/model_registry.py for more configurations):
#               10B Small
#               63B Medium 
#               173B Large
#               537B Super 
#               1T Ultra 
#       Checkpointing strategy : Checkpoint-interval :
#               no-ckpt:0:      --> no activation checkpoint 
#               ckpt:1:         --> activation checkpoint for every layer 
#       ZeRO partitioning : 
#               1   --> ZeRO-1
#               2   --> ZeRO-2
# 3. Each EP_BATCH_MAP run command will be sent to for loop below to fill in template slurm script 
#       and launch each configuration run. Multiple runs can be launched with same EP_BATCH_MAP
#       run with space sepearted. 
# *******************************************************************************************************


# EP-centric baselines
declare -A EP_BATCH_MAP


# ================================== 63B Baselines ==================================
# ==================== EXP =======================
# EP_BATCH_MAP["1:8:1"]=" 1:3:50:X-MOE:1.5T_1L:no-ckpt:0:1"
# EP_BATCH_MAP["2:16:1"]=" 1:3:50:X-MOE:1.5T_1L:no-ckpt:0:1"

# EP_BATCH_MAP["1:8:1"]=" 4:50:20:X-MOE:10B:ckpt:1:1"
# Dense Llama31_8B does not belong in EP_BATCH_MAP — it has num_experts=1, so EP=1
# is required and DP scales via NODES. Use DENSE_BATCH_MAP below instead.
# EP_BATCH_MAP["1:16:1"]=" 4:15:60:X-MOE:Llama31_8B:no-ckpt:0:1"
# EP_BATCH_MAP["9:8:1"]=" 3:1:35:X-MOE:FORGE_10B:no-ckpt:0:1"
# EP_BATCH_MAP["8:8:1"]=" 3:1:35:X-MOE:FORGE_10B:no-ckpt:0:1"
# EP_BATCH_MAP["2:8:1"]=" 2:5:10:X-MOE:FORGE_10B:no-ckpt:0:1"
EP_BATCH_MAP["1:8:1"]=" 2:10:2000:X-MOE:10B:no-ckpt:0:1"
# EP_BATCH_MAP["1:8:1"]=" 2:10:2000:X-MOE:10B_k7:no-ckpt:0:1"
# EP_BATCH_MAP["1:8:1"]=" 2:10:2000:X-MOE:10B_k8:no-ckpt:0:1"

# DSv2-16B 
# EP_BATCH_MAP["8:8:1"]=" 3:24:540:X-MOE:FORGE_10B:no-ckpt:0:1"
# EP_BATCH_MAP["8:8:1"]=" 3:24:60:X-MOE:FORGE_10B:no-ckpt:0:1"
# EP_BATCH_MAP["8:8:1"]=" 4:18:540:X-MOE:FORGE_10B:no-ckpt:0:1"

# EP_BATCH_MAP["4:8:1"]=" 5:1:2000:X-MOE:10B:no-ckpt:0:1"
# PP_BATCH_MAP["4:32"]=" 1:20:2000:ELMOE-GroupedGEMM-primus:10B:dynamic-ckpt:1:uneven:yes-planner "
# ==================== 64 GPUs ===================
# X-MoE
# EP_BATCH_MAP["8:64:1"]=" 1:64:15:X-MOE:63B:no-ckpt:0:1"
# X-MoE, DeepSpeed-MoE, Tutel, DeepSpeed-TED
# EP_BATCH_MAP["8:64:1"]=" 1:64:15:X-MOE:63B:no-ckpt:0:1  1:64:15:DS-MOE:63B:no-ckpt:0:1  1:64:15:TUTEL-MOE:63B:ckpt:1:1  1:64:15:TED-MOE:63B:ckpt:1:1 " 


# ================================== 173B Baselines ==================================
# ==================== 256 GPUs ===================
# EP_BATCH_MAP["32:128:2"]=" 1:32:15:X-MOE:173B:no-ckpt:0:1 " 
# EP_BATCH_MAP["32:128:2"]=" 1:32:13:DS-MOE:173B:ckpt:1:1  1:32:13:TUTEL-MOE:173B:ckpt:1:1  1:32:13:TED-MOE:173B:ckpt:1:1 "


# ================================== 537B Baselines ==================================
# ==================== 512 GPUs ===================
# X-MoE: 
# EP_BATCH_MAP["64:128:4"]=" 1:32:15:X-MOE:537B:ckpt:1:1 "
# DeepSpeed-MoE, Tutel, DeepSpeed-TED: OOM 


# ================================== 1T Baselines ==================================
# ==================== 1024 GPUs ===================
# X-MoE:
# EP_BATCH_MAP["128:256:4"]="1:16:15:X-MOE:1T:ckpt:1:1 "
# DeepSpeed-MoE, Tutel, DeepSpeed-TED: OOM




# ************************************ Dense Launch Instructions ************************************
# Use DENSE_BATCH_MAP for dense models (num_experts=1, e.g. Llama31_8B / 65B_Dense / 63B_Dense).
# 1. Key format:  NODES:PP:TP:DP   (DP is explicit; the loop validates NODES*8 == PP*TP*DP)
# 2. Body format (colon-separated):
#       micro_batch_size : num_micro_batches : train_iters :
#       MOE_TYPE (use DENSE) :
#       Model size (see ../utils/model_registry.py — must have ep=1) :
#       Checkpointing strategy : Checkpoint-interval :
#       ZeRO partitioning (1 / 2 / 3)
# 3. EP is forced to 1 in the dense loop. Llama-style arch flags (GQA via --num-key-value-heads,
#    --swiglu, --normalization rmsnorm) are added automatically based on model_registry fields.
# ****************************************************************************************************

declare -A DENSE_BATCH_MAP

# ==================== 16 GPUs (DP=16, ZeRO-1) ===================
# 2 nodes * 8 GPUs/node = 16 GPUs ; PP=1, TP=1, DP=16
# DENSE_BATCH_MAP["2:1:1:16"]="        4:1:50:DENSE:Llama31_8B:ckpt:1:3            4:1:50:DENSE:Llama31_8B:ckpt:1:2               4:1:50:DENSE:Llama31_8B:ckpt:1:1       "
# DENSE_BATCH_MAP["2:1:1:16"]="           1:1:50:DENSE:Llama31_8B:no-ckpt:0:3    "
# DENSE_BATCH_MAP["2:1:1:16"]="           6:1:50:DENSE:Llama31_8B:ckpt:1:2    "
# DENSE_BATCH_MAP["1:1:1:8"]="             2:1:50:DENSE:Llama31_8B:ckpt:1:3 "
# DENSE_BATCH_MAP["2:1:4:4"]=" 4:16:50:DENSE:Llama31_8B:ckpt:1:1"
# DENSE_BATCH_MAP["2:1:1:16"]=" 4:16:50:DENSE:Llama31_8B:ckpt:1:1"

# ==================== template examples ===================
# 1 node, DP=8, ZeRO-1
# DENSE_BATCH_MAP["1:1:1:8"]=" 4:15:60:DENSE:Llama31_8B:no-ckpt:0:1"
# 4 nodes, DP=32, ZeRO-1
# DENSE_BATCH_MAP["4:1:1:32"]=" 4:15:60:DENSE:Llama31_8B:no-ckpt:0:1"







# ************************************ Profiling Cache Generation Instructions ************************************
# 1. Specify single node profiling with PROFILE_MAP["1:8:1"] that stands for NODES,EP,TP
# 2. Each run command is composed of the following structure and is seperated by colon :
#       micro_batch_size (per-device) : 
#       number_of_micro_batches (per-device-per-iteration) : 
#       number of training iterations : 
#       run model type (ELMoE grounded on X-MoE implementation) :
#               X-MoE 
#       Model sizes (single layer) (see ../utils/model_registry.py for more configurations):
#               10B_1L Small
#               63B_1L Medium 
#               173B_1L Large
#               537B_1L Super 
#               1T_1L Ultra 
#       Checkpointing strategy : Checkpoint-interval :
#               no-ckpt:0:      --> no activation checkpoint to acquire T_gemm & T_comm
# 3. Each PROFILE_MAP run command will be sent to for loop below to fill in template slurm script 
#       and launch each configuration run. Each configuration will run from MBS from 1 to 10. 
#       Caches dir will be created and stored by analyzing through "python ../utils/analyze_log.py " in the template
# ******************************************************************************************************************


declare -A PROFILE_MAP
# PROFILE_MAP["1:8:1"]=" 1:20:30:X-MOE:10B_1L:no-ckpt:0 1:20:30:X-MOE:63B_1L:no-ckpt:0  1:20:30:X-MOE:173B_1L:no-ckpt:0  1:20:30:X-MOE:537B_1L:no-ckpt:0   1:20:30:X-MOE:1T_1L:no-ckpt:0 "








##########################################################################
# --- 2. SCRIPT LOGIC (No need to edit below) ---
##########################################################################
TEMPLATE_FILE="frontier_elmoe.slurm_0623_2026.template"

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
        # if (( PP_SIZE * EP_PARALLEL_SIZE * MP_SIZE != TOTAL_GPUS )); then
        #     echo "  - Invalid strategy: PP=${PP_SIZE}, EP=${EP_PARALLEL_SIZE} for ${TOTAL_GPUS} GPUs. Skipping."
        #     continue
        # fi
        REQUIRED_GPUS=$(( PP_SIZE * EP_PARALLEL_SIZE * MP_SIZE ))

        # Check if the number of required GPUs is a multiple of the total available GPUs
        if (( TOTAL_GPUS % REQUIRED_GPUS == 0 )); then
            echo "Required GPUs (${REQUIRED_GPUS}) is divisible by Total GPUs (${TOTAL_GPUS})."
        else
            echo "Required GPUs (${REQUIRED_GPUS}) is NOT divisible by Total GPUs (${TOTAL_GPUS})."
        fi

        for batch_config in $batch_configs_string; do
            IFS=':' read -r BS NBS TRAIN_ITERS MOE_TYPE MODEL_SIZE CHECKPOINT CHECKPOINT_NUM_LAYERS PP_PARTITION PLANNER_MODE <<< "$batch_config"

            
            RUN_PLANNER="false"

            # 2. Check if Planner is enabled for this run
            # if [[ "$PLANNER_MODE" == "yes-planner" ]]; then
            if [[ "$PLANNER_MODE" == "yes-planner" || "$PLANNER_MODE" == "yes-planner-membal" ]]; then

                if [[ "$PLANNER_MODE" == "yes-planner" ]]; then
                    RUN_PLANNER="true"
                elif [[ "$PLANNER_MODE" == "yes-planner-membal" ]]; then
                    RUN_PLANNER="true-membal"
                else
                    RUN_PLANNER="false"
                fi

                # A. Determine Profiling Model Name (Ensure _1L)
                PROF_MODEL_NAME="${MODEL_SIZE}_1L"
                EP_NODES=$(( EP_PARALLEL_SIZE / 8 ))

                # Call to get cache file 
                CACHE_FILENAME=$(python3 ../utils/model_registry.py get_filename \
                    --model_size "$MODEL_SIZE" \
                    --nodes "$EP_NODES" \
                    --total_gpus "$EP_PARALLEL_SIZE" \
                    --mbs 1)
                
                CACHE_FILE="./planner_profiling_cache/${CACHE_FILENAME}"

                # C. Check if Cache Exists
                if [ -f "$CACHE_FILE" ]; then
                    echo "  [Planner] Cache found: $CACHE_FILENAME"
                else
                    echo "  [Planner] Cache MISSING $CACHE_FILE for $PROF_MODEL_NAME. Continue to next one"

                    continue 
                fi

            fi

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
                UNEVEN_PP="True" # Capital T
            else
                UNEVEN_PP="False" # Capital F
            fi

            RUN_TYPE="pp"
            JOB_NAME="${RUN_TYPE}_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}"
            ZERO=1
            # TEMP_SLURM_SCRIPT="temp_slurm_${JOB_NAME}.slurm"
            TEMP_DIR="temp_slurm"
            TEMP_SLURM_SCRIPT="${TEMP_DIR}/temp_slurm_${JOB_NAME}.slurm"

            echo "  - Generating job: ${JOB_NAME}"

            sed -e "s/{{RUN_TYPE}}/${RUN_TYPE}/g" \
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
                -e "s/{{ZERO}}/${ZERO}/g" \
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
    IFS=':' read -r NODES EP_PARALLEL_SIZE MP_SIZE <<< "$node_key"

    # Get the specific batch configs for this node setup
    batch_configs_string=${EP_BATCH_MAP[$node_key]}

    # Set partition for the new environment
    PARTITION="batch"
    # MP_SIZE=2


    # For EP runs, PP=1 and EP=Total GPUs
    PP_SIZE=1
    TOTAL_GPUS=$(( NODES * 8 ))
    
    echo "Found EP configurations for ${NODES} nodes, ${TOTAL_GPUS} GPUs..."


    for batch_config in $batch_configs_string; do
        IFS=':' read -r BS NBS TRAIN_ITERS MOE_TYPE MODEL_SIZE CHECKPOINT CHECKPOINT_NUM_LAYERS ZERO <<< "$batch_config"

        if [[ "$CHECKPOINT" == "ckpt" ]]; then
            ACTIVATION_CHECKPOINT="true"
        else
            ACTIVATION_CHECKPOINT="false"
        fi

        RUN_TYPE="ep"
        JOB_NAME="${RUN_TYPE}_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}"
        # TEMP_SLURM_SCRIPT="temp_slurm_${JOB_NAME}.slurm"
        TEMP_DIR="temp_slurm"
        TEMP_SLURM_SCRIPT="${TEMP_DIR}/temp_slurm_${JOB_NAME}.slurm"

        echo "  - Generating job: ${JOB_NAME}"

        sed -e "s/{{RUN_TYPE}}/${RUN_TYPE}/g" \
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
                -e "s/{{ZERO}}/${ZERO}/g" \
            ${TEMPLATE_FILE} > ${TEMP_SLURM_SCRIPT}

        sbatch ${TEMP_SLURM_SCRIPT}
        # rm ${TEMP_SLURM_SCRIPT}
        sleep 1
    done
done







# --- LAUNCH DENSE RUNS ---
echo
echo "-----------------------------------------------"
echo "--- Submitting Dense (DP-only) jobs ---"
echo "-----------------------------------------------"

for node_key in "${!DENSE_BATCH_MAP[@]}"; do
    IFS=':' read -r NODES PP_SIZE MP_SIZE DP_SIZE <<< "$node_key"

    batch_configs_string=${DENSE_BATCH_MAP[$node_key]}

    PARTITION="batch"
    TOTAL_GPUS=$(( NODES * 8 ))

    # Sanity: PP * TP * DP must equal TOTAL_GPUS
    REQUIRED_GPUS=$(( PP_SIZE * MP_SIZE * DP_SIZE ))
    if (( REQUIRED_GPUS != TOTAL_GPUS )); then
        echo "ERROR: DENSE key '${node_key}': PP*TP*DP=${REQUIRED_GPUS} != TOTAL_GPUS=${TOTAL_GPUS}. Skipping."
        continue
    fi

    # Dense path: no expert parallelism
    EP_PARALLEL_SIZE=1

    echo "Found DENSE configurations for ${NODES} nodes, ${TOTAL_GPUS} GPUs (PP=${PP_SIZE}, TP=${MP_SIZE}, DP=${DP_SIZE})..."

    for batch_config in $batch_configs_string; do
        IFS=':' read -r BS NBS TRAIN_ITERS MOE_TYPE MODEL_SIZE CHECKPOINT CHECKPOINT_NUM_LAYERS ZERO <<< "$batch_config"

        if [[ "$MOE_TYPE" != "DENSE" ]]; then
            echo "  WARN: DENSE_BATCH_MAP body has MOE_TYPE='${MOE_TYPE}'; expected 'DENSE'. Continuing anyway."
        fi

        if [[ "$CHECKPOINT" == "ckpt" ]]; then
            ACTIVATION_CHECKPOINT="true"
        else
            ACTIVATION_CHECKPOINT="false"
        fi

        RUN_TYPE="dense"
        JOB_NAME="${RUN_TYPE}_n${NODES}g${TOTAL_GPUS}_dp${DP_SIZE}_pp${PP_SIZE}_tp${MP_SIZE}_nbs${NBS}_i${TRAIN_ITERS}_${MODEL_SIZE}"
        TEMP_DIR="temp_slurm"
        TEMP_SLURM_SCRIPT="${TEMP_DIR}/temp_slurm_${JOB_NAME}.slurm"

        echo "  - Generating job: ${JOB_NAME}"

        sed -e "s/{{RUN_TYPE}}/${RUN_TYPE}/g" \
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
            -e "s/{{DYNAMIC_CHECKPOINT}}/False/g" \
            -e "s/{{UNEVEN_PP}}/False/g" \
            -e "s/{{RUN_PLANNER}}/false/g" \
            -e "s/{{COLLECT_PROFILING_CACHE}}/false/g" \
            -e "s/{{ZERO}}/${ZERO}/g" \
            ${TEMPLATE_FILE} > ${TEMP_SLURM_SCRIPT}

        sbatch ${TEMP_SLURM_SCRIPT}
        sleep 1
    done
done


# Iterate over the node configurations defined in the EP batch map
for node_key in "${!PROFILE_MAP[@]}"; do
    IFS=':' read -r NODES TOTAL_GPUS MP_SIZE <<< "$node_key"

    # Get the specific batch configs for this node setup
    batch_configs_string=${PROFILE_MAP[$node_key]}

    # Set partition for the new environment
    PARTITION="batch"
    # MP_SIZE=2
    # TEMPLATE_PLANNER_FILE="planner.slurm.template"
    TEMPLATE_PLANNER_FILE="frontier_elmoe.slurm_0623_2026.template"
    
    echo "Found EP configurations for ${NODES} nodes, ${TOTAL_GPUS} GPUs..."

    # For EP runs, PP=1 and EP=Total GPUs
    PP_SIZE=1
    EP_PARALLEL_SIZE=$(( TOTAL_GPUS / MP_SIZE ))

    for batch_config in $batch_configs_string; do
        IFS=':' read -r BS NBS TRAIN_ITERS MOE_TYPE MODEL_SIZE CHECKPOINT CHECKPOINT_NUM_LAYERS <<< "$batch_config"

        if [[ "$CHECKPOINT" == "ckpt" ]]; then
            ACTIVATION_CHECKPOINT="true"
        else
            ACTIVATION_CHECKPOINT="false"
        fi

        for BS in {1..10}; do

            RUN_TYPE="pr"
            JOB_NAME="${RUN_TYPE}_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}"
            TEMP_SLURM_SCRIPT="temp_slurm_${JOB_NAME}.slurm"

            echo "  - Generating job: ${JOB_NAME}, BS:$BS"

            ZERO=1

            sed -e "s/{{RUN_TYPE}}/${RUN_TYPE}/g" \
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
                -e "s/{{CHECKPOINT_NUM_LAYERS}}/0/g" \
                -e "s/{{DYNAMIC_CHECKPOINT}}/False/g" \
                -e "s/{{UNEVEN_PP}}/False/g" \
                -e "s/{{COLLECT_PROFILING_CACHE}}/true/g" \
                -e "s/{{RUN_PLANNER}}/false/g" \
                -e "s/{{ZERO}}/${ZERO}/g" \
                ${TEMPLATE_FILE} > ${TEMP_SLURM_SCRIPT}

            sbatch ${TEMP_SLURM_SCRIPT}
            # rm ${TEMP_SLURM_SCRIPT}
            sleep 1
        done 
    done
done



echo
echo "All jobs submitted."
