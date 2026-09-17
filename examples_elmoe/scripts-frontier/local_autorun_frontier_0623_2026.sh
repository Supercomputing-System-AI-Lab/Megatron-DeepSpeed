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
                
# ==================== 128k additive riders (2026-08-27) =====================
# All inert unless the corresponding AUTORUN_* env var is set in the submit
# shell. With none set, the maps above and every rendered job script are
# byte-identical to the pre-rider launcher (proof:
# longctx128k/baseline_render/render.diff -- QA Q3 / ACCEPTANCE 1.6).
# Purpose: launch a probe through THIS entry point without editing the maps,
# the template, or any file in the tree.
#
#   AUTORUN_EP_KEY="N:EP:TP" + AUTORUN_EP_CMD="bs:nbs:iters:MOE:MODEL:ckpt:n:zero"
#       submit ONLY this EP-loop entry; every other map is cleared for this
#       invocation only (the script file itself is untouched).
#   AUTORUN_SEQ_LEN=<tokens>     override the registry SEQ_LEN in the rendered
#                                job (inserted right after `eval "$CONFIG_VARS"`,
#                                before any consumer of SEQ_LEN).
#   AUTORUN_PP_SIZE=<n>          EP-path only: override the EP ladder's PP=1
#                                pin in the rendered job (the PP_SIZE
#                                assignment plus the pp1 tag in the SBATCH
#                                -o/-e and JOB_DIR names, which must stay
#                                consistent with each other). Everything
#                                downstream — GBS, PP_STRIDE,
#                                --pipeline-model-parallel-size vs
#                                --no-pipeline-parallel — reads $PP_SIZE at
#                                runtime inside the job. Verified or ABORT.
#   AUTORUN_NO_CKPT=1            strip --save/--load from the rendered job:
#                                random init, writes no checkpoint anywhere
#                                (probe hygiene; cannot touch any lineage dir).
#   AUTORUN_CAPACITY_ROUTED=1    remove --disable-moe-token-dropping (all
#                                occurrences): capacity routing at the factor
#                                already rendered (1.25 last-wins).
#   AUTORUN_SEQ_CHUNKED_CE=<n>   append --seq-chunked-ce <n> (non-pipe CE path).
#   AUTORUN_NCCL_DEBUG=<lvl>     replace `export NCCL_DEBUG=INFO` with <lvl>.
#   AUTORUN_NO_FP32_RESIDUAL=1   remove --fp32-residual-connection from the
#                                rendered job (checkpoint boundaries drop
#                                fp32->bf16; R1_DIAGNOSIS.md S6).
#                                PROHIBITED for submissions: the flag is
#                                owner-required (directive 2026-08-27).
#   AUTORUN_CKPT_IN_CPU=1        append --checkpoint-in-cpu (offload the
#                                activation-checkpoint boundaries to host;
#                                FLAGKEPT_PRICING.md S4/P1).
#   AUTORUN_PARTITION_ACT=1      append --partition-activations (TP-partition
#                                the checkpoint boundaries; FLAGKEPT_PRICING.md
#                                S4/P2 — requires the S8 arity fix, landed).
#   AUTORUN_DS_SEQUENCE_PARALLEL_SIZE=<P>  append --ds-sequence-parallel-size <P> (DeepSpeed-Ulysses inside the latent
#                                attention with PP, TP, EP: the blueprint of 2026-09-17; with TP > 1 set AUTORUN_MEGATRON_SP=1 too).
#   AUTORUN_MEGATRON_SP=1        companion SP (SP_COMPANION_AUDIT.md S9): append
#                                --sequence-parallel AND export XMOE_MEGATRON_SP=1
#                                — BOTH together (flag w/o env re-introduces G2
#                                double-shard; env w/o flag is a no-op). The
#                                rider verifies both landed or ABORTS the launch.
#   AUTORUN_LOAD_DIR=<dir>       point --load at <dir> (replaces the template's
#                                default LOAD_CHECKPOINT_PATH). Verified or ABORT.
#   AUTORUN_SAVE_DIR=<dir>       save checkpoints to <dir> (replaces the per-job
#                                dated CHECKPOINT_PATH). Verified or ABORT.
#   AUTORUN_UCP_FINETUNE=1       append --universal-checkpoint while KEEPING the
#                                base --finetune (the stage-transition turn-1
#                                pairing, ANALYSIS_128K.md S3.6 — deliberately
#                                NOT the UCP_ENABLE path, which strips
#                                --finetune). Verifies BOTH flags or ABORTS.
#   AUTORUN_SAVE_INTERVAL=<n>    replace the template's SAVE_INTERVAL=10 in the
#                                rendered job (checkpoint cadence; a chained
#   AUTORUN_EXIT_MINS=<n>        replace EXIT_DURATION=30000000 (--exit-duration-in-mins): the job
#                                saves and exits cleanly N minutes after launch (chain links).
#   AUTORUN_PRE_CMD=<cmd>        run <cmd> inside the allocation before the training srun (UCP I/O campaign:
#                                in-job conversion of the resume checkpoint); AUTORUN_POST_CMD=<cmd> after it.
#                                pass only makes durable progress in
#                                SAVE_INTERVAL multiples). Verified or ABORT.
#   AUTORUN_UCP_RESUME=1         PROPER universal resume: append
#                                --universal-checkpoint AND strip the base
#                                --finetune — the exact pair the template's
#                                UCP_ENABLE=1 path builds (that path is
#                                unreachable on the EP loop, which pins
#                                UCP_ENABLE=0 at render). Optimizer state,
#                                iteration counter and consumed tokens all
#                                restore (0815 ladder-certification resume
#                                semantics). Mutually exclusive with
#                                AUTORUN_UCP_FINETUNE and
#                                AUTORUN_NATIVE_RESUME. Verified or ABORT.
#   AUTORUN_NATIVE_RESUME=1      same-topology native resume for chained
#                                passes: strip the base --finetune only (NO
#                                --universal-checkpoint, so the in-job
#                                conversion block stays off) and insert a
#                                fail-fast guard that aborts the job before
#                                srun if ${LOAD_CHECKPOINT_PATH}/latest is
#                                missing — a chained pass must never fall back
#                                to random init. Mutually exclusive with the
#                                two UCP riders. Verified or ABORT.

if [ -n "${AUTORUN_EP_KEY:-}" ]; then
    PP_BATCH_MAP=()
    EP_BATCH_MAP=()
    DENSE_BATCH_MAP=()
    PROFILE_MAP=()
    EP_BATCH_MAP["${AUTORUN_EP_KEY}"]="${AUTORUN_EP_CMD:?AUTORUN_EP_KEY set but AUTORUN_EP_CMD empty}"
    echo "[rider] EP-only submission: EP_BATCH_MAP[${AUTORUN_EP_KEY}]='${AUTORUN_EP_CMD}'"
fi

# Applied to each RENDERED job script after the sed render, before sbatch.
# No-op when no AUTORUN_* rider is set.
apply_autorun_riders() {
    local f="$1"
    # Resume-mode riders are mutually exclusive: abort before any edit lands.
    if [ "${AUTORUN_UCP_RESUME:-0}" = "1" ] && { [ "${AUTORUN_UCP_FINETUNE:-0}" = "1" ] || [ "${AUTORUN_NATIVE_RESUME:-0}" = "1" ]; }; then
        echo "FATAL [rider conflict]: AUTORUN_UCP_RESUME is mutually exclusive with AUTORUN_UCP_FINETUNE / AUTORUN_NATIVE_RESUME — refusing to submit"
        exit 1
    fi
    if [ "${AUTORUN_NATIVE_RESUME:-0}" = "1" ] && [ "${AUTORUN_UCP_FINETUNE:-0}" = "1" ]; then
        echo "FATAL [rider conflict]: AUTORUN_NATIVE_RESUME is mutually exclusive with AUTORUN_UCP_FINETUNE — refusing to submit"
        exit 1
    fi
    if [ -n "${AUTORUN_SEQ_LEN:-}" ]; then
        sed -i "/^eval \"\$CONFIG_VARS\"\$/a SEQ_LEN=${AUTORUN_SEQ_LEN}   # [rider AUTORUN_SEQ_LEN] overrides the registry value" "$f"
    fi
    if [ -n "${AUTORUN_PP_SIZE:-}" ]; then
        sed -i -e "s/^PP_SIZE=1\$/PP_SIZE=${AUTORUN_PP_SIZE}   # [rider AUTORUN_PP_SIZE] overrides the EP-path PP=1 pin/" \
               -e "s/_pp1_ep/_pp${AUTORUN_PP_SIZE}_ep/g" "$f"
        if ! grep -q "^PP_SIZE=${AUTORUN_PP_SIZE} " "$f" || ! grep -q "_pp${AUTORUN_PP_SIZE}_ep" "$f"; then
            echo "FATAL [rider AUTORUN_PP_SIZE]: PP override did not land in $f — refusing to submit (EP-path renders only)"
            exit 1
        fi
    fi
    if [ "${AUTORUN_NO_CKPT:-0}" = "1" ]; then
        sed -i -e '/^[[:space:]]*--save ${CHECKPOINT_PATH} \\$/d' \
               -e '/^[[:space:]]*--load ${LOAD_CHECKPOINT_PATH} \\$/d' "$f"
    fi
    if [ "${AUTORUN_CAPACITY_ROUTED:-0}" = "1" ]; then
        sed -i -e '/^[[:space:]]*--disable-moe-token-dropping \\$/d' \
               -e 's/ --disable-moe-token-dropping//g' "$f"
    fi
    if [ -n "${AUTORUN_SEQ_CHUNKED_CE:-}" ]; then
        sed -i "/END of Megatron Setup/i megatron_options=\"\${megatron_options} --seq-chunked-ce ${AUTORUN_SEQ_CHUNKED_CE}\"   # [rider AUTORUN_SEQ_CHUNKED_CE]" "$f"
    fi
    if [ "${AUTORUN_CKPT_IN_CPU:-0}" = "1" ]; then
        sed -i "/END of Megatron Setup/i megatron_options=\"\${megatron_options} --checkpoint-in-cpu\"   # [rider AUTORUN_CKPT_IN_CPU]" "$f"
    fi
    if [ "${AUTORUN_PARTITION_ACT:-0}" = "1" ]; then
        sed -i "/END of Megatron Setup/i megatron_options=\"\${megatron_options} --partition-activations\"   # [rider AUTORUN_PARTITION_ACT]" "$f"
    fi
    if [ -n "${AUTORUN_DS_SEQUENCE_PARALLEL_SIZE:-}" ]; then
        sed -i "/END of Megatron Setup/i megatron_options=\"\${megatron_options} --ds-sequence-parallel-size ${AUTORUN_DS_SEQUENCE_PARALLEL_SIZE}\"   # [rider AUTORUN_DS_SEQUENCE_PARALLEL_SIZE]" "$f"
        if ! grep -q -- "--ds-sequence-parallel-size" "$f"; then echo "FATAL [rider AUTORUN_DS_SEQUENCE_PARALLEL_SIZE]: did not land in $f"; exit 1; fi
    fi
    if [ "${AUTORUN_MEGATRON_SP:-0}" = "1" ]; then
        sed -i "/END of Megatron Setup/i megatron_options=\"\${megatron_options} --sequence-parallel\"   # [rider AUTORUN_MEGATRON_SP]" "$f"
        sed -i "/^export NCCL_DEBUG=/a export XMOE_MEGATRON_SP=1   # [rider AUTORUN_MEGATRON_SP] REQUIRED with --sequence-parallel (SP_COMPANION_AUDIT.md G2)" "$f"
        if ! grep -q -- "--sequence-parallel" "$f" || ! grep -q "^export XMOE_MEGATRON_SP=1" "$f"; then
            echo "FATAL [rider AUTORUN_MEGATRON_SP]: flag+env pair did not both land in $f — refusing to submit (SP_COMPANION_AUDIT.md S9: both are required together)"
            exit 1
        fi
    fi
    if [ -n "${AUTORUN_LOAD_DIR:-}" ]; then
        sed -i "s|^LOAD_CHECKPOINT_PATH=checkpoint/deepseek_16b_ds\$|LOAD_CHECKPOINT_PATH=${AUTORUN_LOAD_DIR}   # [rider AUTORUN_LOAD_DIR]|" "$f"
        if ! grep -q "^LOAD_CHECKPOINT_PATH=${AUTORUN_LOAD_DIR} " "$f"; then
            echo "FATAL [rider AUTORUN_LOAD_DIR]: load-path override did not land in $f — refusing to submit"
            exit 1
        fi
    fi
    if [ -n "${AUTORUN_SAVE_DIR:-}" ]; then
        sed -i "s|^export CHECKPOINT_PATH=checkpoint/\\\$(date .*|export CHECKPOINT_PATH=${AUTORUN_SAVE_DIR}   # [rider AUTORUN_SAVE_DIR]|" "$f"
        if ! grep -q "^export CHECKPOINT_PATH=${AUTORUN_SAVE_DIR} " "$f"; then
            echo "FATAL [rider AUTORUN_SAVE_DIR]: save-path override did not land in $f — refusing to submit"
            exit 1
        fi
    fi
    if [ "${AUTORUN_UCP_FINETUNE:-0}" = "1" ]; then
        sed -i "/END of Megatron Setup/i megatron_options=\"\${megatron_options} --universal-checkpoint\"   # [rider AUTORUN_UCP_FINETUNE] paired with the base --finetune (turn-1 gotcha)" "$f"
        if ! grep -q -- "--universal-checkpoint" "$f" || ! grep -q -- "^[[:space:]]*--finetune \\\\\$" "$f"; then
            echo "FATAL [rider AUTORUN_UCP_FINETUNE]: --universal-checkpoint + --finetune pair not both present in $f — refusing to submit (ANALYSIS_128K.md S3.6 turn-1 gotcha)"
            exit 1
        fi
    fi
    if [ "${AUTORUN_NO_FP32_RESIDUAL:-0}" = "1" ]; then
        sed -i -e '/^[[:space:]]*--fp32-residual-connection \\$/d' \
               -e 's/ --fp32-residual-connection//g' "$f"
    fi
    if [ -n "${AUTORUN_NCCL_DEBUG:-}" ]; then
        sed -i "s/^export NCCL_DEBUG=INFO\$/export NCCL_DEBUG=${AUTORUN_NCCL_DEBUG}/" "$f"
    fi
    # Walltime override. The template hard-codes "#SBATCH -t 1:00:00"; a probe wants 20 minutes and a
    # production turn wants exactly the campaign's 1-hour wall. Rewriting the directive (rather than
    # passing sbatch -t) keeps the rendered script self-describing: the file you read is the job that
    # ran. Verify-or-abort, matching every other rider here.
    # QOS override. The template pins "-q debug", which is MaxSubmitPU=1 -- fine for a probe, fatal
    # for a chained campaign because turn N+1 cannot be queued while turn N is pending.
    # Dependency injected INTO the script, not applied afterwards. `sbatch` followed by
    # `scontrol update Dependency=` is a RACE: job 5380794 was started by the scheduler in the
    # gap and ran unconstrained -- a second 128-node job resuming the same checkpoint and writing
    # the same save dir as the turn still running. Putting the directive in the rendered script
    # makes it atomic with submission.
    # Bake the data blend INTO the script. LC_DATA_PATH otherwise rides only on --export=ALL, and
    # if it fails to propagate the template silently falls back to
    #   DATA_PATH="0.34 ${LC_LONG} 0.66 ${LC_REPLAY}"
    # i.e. a completely different corpus, with no error. Writing it in makes it auditable with
    # `scontrol write batch_script` after the fact -- which is the only way to prove which data a
    # finished run actually trained on.
    if [ -n "${AUTORUN_DATA_PATH:-}" ]; then
        # 2026-08-30 POSTMORTEM (job 5381973): the old rider rewrote the top-level
        # `export DATA_PATH=` line, but the template's LC_MODE block re-exports
        #   DATA_PATH="${LC_DATA_PATH:-0.34 ${LC_LONG} 0.66 ${LC_REPLAY}}"
        # LATER in execution, silently replacing the blend with the stale L8+nemotron
        # default. Turn 3 trained 2.23B tokens on 0% >=128k data while the script TEXT
        # carried the v2 blend -- the old guard checked the text, not the executed value.
        # Fix: set LC_DATA_PATH (which that line RESPECTS), and bake a runtime assert
        # right after the clobber line so the job itself dies loudly on the wrong corpus.
        # python injection: the blend is full of / and | characters; sed delimiters
        # cannot carry it safely (first attempt died on "unknown option to s").
        AUTORUN_DATA_PATH="${AUTORUN_DATA_PATH}" python3 - "$f" <<'PYINJ'
import os, re, sys
f = sys.argv[1]; blend = os.environ['AUTORUN_DATA_PATH']
marker = os.path.basename(blend.split()[1]).split('.')[0]
s = open(f).read()
anchor = 'echo "COLLECT_PROFILING_CACHE'
i = s.index(anchor)
s = (s[:i] + 'export LC_DATA_PATH="' + blend + '"   # [rider AUTORUN_DATA_PATH]\n' + s[i:])
clob = re.search(r'^(    export DATA_PATH="\$\{LC_DATA_PATH:-.*)$', s, re.M)
assert clob, 'clobber line not found'
check = ('\n    case "${DATA_PATH}" in *"' + marker + '"*) echo "blend RUNTIME-OK: contains ' + marker
         + '";; *) echo "FATAL: DATA_PATH lacks expected corpus ' + marker
         + ' -- got: ${DATA_PATH:0:200}"; exit 1;; esac')
s = s[:clob.end()] + check + s[clob.end():]
open(f, 'w').write(s)
PYINJ
        if ! grep -q "rider AUTORUN_DATA_PATH" "$f" || ! grep -q "blend RUNTIME-OK" "$f"; then
            echo "FATAL [rider AUTORUN_DATA_PATH]: injection did not land in $f -- refusing to submit"
            exit 1
        fi
    fi
    if [ -n "${AUTORUN_DEPENDENCY:-}" ]; then
        sed -i "0,/^#SBATCH -N /s//#SBATCH --dependency=${AUTORUN_DEPENDENCY}\n#SBATCH -N /" "$f"
        if ! grep -q "^#SBATCH --dependency=${AUTORUN_DEPENDENCY}$" "$f"; then
            echo "FATAL [rider AUTORUN_DEPENDENCY]: directive did not land in $f -- refusing to submit"
            exit 1
        fi
    fi
    if [ -n "${AUTORUN_QOS:-}" ]; then
        sed -i "s|^#SBATCH -q .*$|#SBATCH -q ${AUTORUN_QOS}   # [rider AUTORUN_QOS]|" "$f"
        if ! grep -q "^#SBATCH -q ${AUTORUN_QOS} " "$f"; then
            echo "FATAL [rider AUTORUN_QOS]: qos override did not land in $f -- refusing to submit"
            exit 1
        fi
    fi
    if [ -n "${AUTORUN_WALLTIME:-}" ]; then
        sed -i "s|^#SBATCH -t .*$|#SBATCH -t ${AUTORUN_WALLTIME}   # [rider AUTORUN_WALLTIME]|" "$f"
        if ! grep -q "^#SBATCH -t ${AUTORUN_WALLTIME} " "$f"; then
            echo "FATAL [rider AUTORUN_WALLTIME]: walltime override did not land in $f -- refusing to submit"
            exit 1
        fi
    fi
    # Launch-geometry guard. The bug that cost this campaign an entire stage at 27% throughput was a
    # hand-edited script whose TOTAL_GPUS disagreed with -N, so the template's
    # gpus_per_node=$(( TOTAL_GPUS / NODES )) silently produced 4 GCDs/node. A sweep of the job logs
    # found 15 runs at the wrong geometry, not 2.
    # PLACEMENT IS THE POINT: it must fire BEFORE the universal-conversion srun (template ~:730,
    # measured up to 15 min on 1 node) and before the Triton warm srun -- otherwise a 128-node
    # allocation burns those minutes doing 1-node work before anything complains. So anchor it to the
    # first echo of the config block, not to the gpus_per_node line ~180 lines further down.
    if ! grep -q '^__gpn_guard=' "$f"; then
        sed -i '/^echo "COLLECT_PROFILING_CACHE: \$COLLECT_PROFILING_CACHE"$/a __gpn_guard=$(( TOTAL_GPUS / NODES )); if [ "${__gpn_guard}" -ne 8 ]; then echo "FATAL launch geometry: gpus_per_node=${__gpn_guard} (TOTAL_GPUS=${TOTAL_GPUS} / NODES=${NODES}); expected 8. Refusing to run at partial node occupancy."; exit 1; fi; if [ -n "${SLURM_NNODES:-}" ] \&\& [ "${SLURM_NNODES}" -ne "${NODES}" ]; then echo "FATAL launch geometry: NODES=${NODES} but SLURM_NNODES=${SLURM_NNODES}."; exit 1; fi; echo "launch geometry OK: ${NODES} nodes x ${__gpn_guard} GCDs = ${TOTAL_GPUS}"' "$f"
        if ! grep -q '^__gpn_guard=' "$f"; then
            echo "FATAL [gpus_per_node guard]: guard did not land in $f -- refusing to submit"
            exit 1
        fi
    fi
    if [ -n "${AUTORUN_SAVE_INTERVAL:-}" ]; then
        sed -i "s/^SAVE_INTERVAL=10\$/SAVE_INTERVAL=${AUTORUN_SAVE_INTERVAL}   # [rider AUTORUN_SAVE_INTERVAL] overrides the template cadence/" "$f"
        if ! grep -q "^SAVE_INTERVAL=${AUTORUN_SAVE_INTERVAL} " "$f"; then
            echo "FATAL [rider AUTORUN_SAVE_INTERVAL]: override did not land in $f — refusing to submit"
            exit 1
        fi
    fi
    if [ -n "${AUTORUN_EXIT_MINS:-}" ]; then
        # [2026-09-05, UCP I/O campaign] a chain link must save and exit cleanly N minutes after
        # launch so the next link can resume; the template pins EXIT_DURATION=30000000 (never).
        sed -i "s/^EXIT_DURATION=30000000\$/EXIT_DURATION=${AUTORUN_EXIT_MINS}   # [rider AUTORUN_EXIT_MINS] --exit-duration-in-mins/" "$f"
        if ! grep -q "^EXIT_DURATION=${AUTORUN_EXIT_MINS} " "$f"; then
            echo "FATAL [rider AUTORUN_EXIT_MINS]: override did not land in $f — refusing to submit"
            exit 1
        fi
    fi
    if [ -n "${AUTORUN_PRE_CMD:-}" ]; then
        # [2026-09-05, UCP I/O campaign] a command run inside the allocation BEFORE training (e.g. the
        # Darshan-profiled conversion of the checkpoint this job resumes from); one job instead of two.
        python3 - "$f" "${AUTORUN_PRE_CMD}" <<'PY'
import sys
f, cmd = sys.argv[1], sys.argv[2]; s = open(f).read()
a = 'echo "Before launching srun python python ../pretrain_gpt_deepspeed.py"\n'
assert s.count(a) == 1, 'PRE anchor not unique'
s = s.replace(a, 'echo "[rider AUTORUN_PRE_CMD] $(date) running: ' + cmd.replace('"', '\\"') + '"\n' + cmd + '   # [rider AUTORUN_PRE_CMD]\n' + a)
open(f, 'w').write(s)
PY
        if ! grep -q "rider AUTORUN_PRE_CMD" "$f"; then echo "FATAL [rider AUTORUN_PRE_CMD]: did not land in $f"; exit 1; fi
    fi
    if [ -n "${AUTORUN_POST_CMD:-}" ]; then
        # [2026-09-05, UCP I/O campaign] a command run inside the allocation AFTER training returns
        # (e.g. the Darshan-profiled conversion of this job's newest complete save).
        python3 - "$f" "${AUTORUN_POST_CMD}" <<'PY'
import sys
f, cmd = sys.argv[1], sys.argv[2]; s = open(f).read()
a = 'wait "${TRAIN_PID}"     # SIGUSR1 near walltime interrupts this; normal finish falls through\n'
assert s.count(a) == 1, 'POST anchor not unique'
s = s.replace(a, a + 'echo "[rider AUTORUN_POST_CMD] $(date) running: ' + cmd.replace('"', '\\"') + '"\n' + cmd + '   # [rider AUTORUN_POST_CMD]\n')
open(f, 'w').write(s)
PY
        if ! grep -q "rider AUTORUN_POST_CMD" "$f"; then echo "FATAL [rider AUTORUN_POST_CMD]: did not land in $f"; exit 1; fi
    fi
    if [ "${AUTORUN_UCP_RESUME:-0}" = "1" ]; then
        sed -i '/^[[:space:]]*--finetune \\$/d' "$f"
        sed -i "/END of Megatron Setup/i megatron_options=\"\${megatron_options} --universal-checkpoint\"   # [rider AUTORUN_UCP_RESUME] proper universal resume: optimizer+iteration+consumed-tokens restore; --finetune stripped (UCP_ENABLE=1 semantics on the EP path)" "$f"
        if ! grep -q -- "--universal-checkpoint" "$f" || grep -q -- "^[[:space:]]*--finetune \\\\\$" "$f"; then
            echo "FATAL [rider AUTORUN_UCP_RESUME]: expected --universal-checkpoint present AND --finetune absent in $f — refusing to submit"
            exit 1
        fi
    fi
    if [ "${AUTORUN_NATIVE_RESUME:-0}" = "1" ]; then
        sed -i '/^[[:space:]]*--finetune \\$/d' "$f"
        sed -i '/^LOAD_CHECKPOINT_PATH=/a if [ ! -f "${LOAD_CHECKPOINT_PATH}/latest" ]; then echo "FATAL [rider AUTORUN_NATIVE_RESUME]: no latest tag under ${LOAD_CHECKPOINT_PATH} — a chained pass must not fall back to random init"; exit 1; fi   # [rider AUTORUN_NATIVE_RESUME] fail-fast resume guard' "$f"
        if grep -q -- "^[[:space:]]*--finetune \\\\\$" "$f" || ! grep -q "rider AUTORUN_NATIVE_RESUME" "$f"; then
            echo "FATAL [rider AUTORUN_NATIVE_RESUME]: expected --finetune absent AND resume guard present in $f — refusing to submit"
            exit 1
        fi
    fi
}
# =========================================================================

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
