#!/bin/bash

# ==============================================================================
# Offline Log Reconstruction and Analysis Script
# Usage: ./rerun_offline.sh <path_to_job_directory>
# Example: ./rerun_offline.sh logs/job_4230686_pp1_ep128_X-MOE_ckpt_true_planner_0
# ==============================================================================

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_job_directory>"
    exit 1
fi

JOB_DIR=$(realpath "$1")

if [ ! -d "$JOB_DIR" ]; then
    echo "Error: Directory '$JOB_DIR' does not exist."
    exit 1
fi

echo "--- Starting Log Analysis for: $JOB_DIR ---"

# 1. Locate the SLURM .o file
SLURM_LOG_FILE=$(ls "$JOB_DIR"/*.o 2>/dev/null | head -n 1)

if [ -z "$SLURM_LOG_FILE" ]; then
    echo "FATAL: Could not find any SLURM .o file in $JOB_DIR"
    exit 1
fi

echo "Found SLURM output file: $(basename "$SLURM_LOG_FILE")"

# 2. Extract Job ID
# Tries to get it from the directory name (job_XXXX), otherwise from the filename (XXXX.o)
SLURM_JOB_ID=$(echo "$JOB_DIR" | grep -o -E "job_[0-9]+" | grep -o -E "[0-9]+")
if [ -z "$SLURM_JOB_ID" ]; then
    SLURM_JOB_ID=$(basename "$SLURM_LOG_FILE" | grep -o -E "[0-9]+\.o" | grep -o -E "[0-9]+")
fi
SLURM_JOB_ID=${SLURM_JOB_ID:-"000000"}

# 3. Helper function to extract variables from the SLURM log
extract_var() {
    local name=$1
    local default=$2
    # Looks for "NAME: VAL" or "NAME=VAL", with optional spaces
    local val=$(grep -E "^[[:space:]]*${name}[:=]" "$SLURM_LOG_FILE" | head -n 1 | sed -E "s/^[[:space:]]*${name}[:=][[:space:]]*([^[:space:]]+).*/\1/")
    if [ -z "$val" ]; then
        echo "$default"
    else
        echo "$val"
    fi
}

# 4. Parse Environment Variables necessary for Analysis
TRAIN_ITERS=$(extract_var "TRAIN_ITERS" "100")
NODES=$(extract_var "NODES" "1")
TOTAL_GPUS=$(extract_var "TOTAL_GPUS" "8")
PP_SIZE=$(extract_var "PP_SIZE" "1")
EP_PARALLEL_SIZE=$(extract_var "EP_SIZE" "1")
MP_SIZE=$(extract_var "MP_SIZE" "1") # TP Size
BATCH_SIZE=$(extract_var "MICRO_BATCH_SIZE" "1")
GLOBAL_BATCH_SIZE=$(extract_var "GLOBAL_BATCH_SIZE" "1")
SEQ_LEN=$(extract_var "SEQ_LEN" "2048")
NUM_LAYERS=$(extract_var "NUM_LAYERS" "24")
FFN_HIDDEN_SIZE=$(extract_var "FFN_HIDDEN_SIZE" "4096")
TOPK=$(extract_var "topk" "2")
HIDDEN_SIZE=$(extract_var "HIDDEN_SIZE" "1024")
MOE_TYPE=$(extract_var "MOE_TYPE" "X-MOE")
ACTIVATION_CHECKPOINT=$(extract_var "ACTIVATION_CHECKPOINT" "false")
CHECKPOINT_NUM_LAYERS=$(extract_var "CHECKPOINT_NUM_LAYERS" "0")
DYNAMIC_CHECKPOINT=$(extract_var "DYNAMIC_CHECKPOINT" "False")
UNEVEN_PP=$(extract_var "UNEVEN_PP" "False")

# Extract Model Size (Special regex format)
MODEL_SIZE=$(grep -E "^Model Config Loaded for " "$SLURM_LOG_FILE" | head -n 1 | sed -E 's/^Model Config Loaded for ([^:]+):.*/\1/')
MODEL_SIZE=${MODEL_SIZE:-"10B"}

# 5. Derived Variables
gpus_per_node=$(( TOTAL_GPUS / NODES ))
LAST_RANK=$(( TOTAL_GPUS - 1 ))
DP_SIZE=$(( TOTAL_GPUS / PP_SIZE / MP_SIZE ))

# Verify rank 0 and last rank files exist
RANK_LOG_FILE_2="${JOB_DIR}/rank_0.log"
RANK_LOG_FILE="${JOB_DIR}/rank_${LAST_RANK}.log"

# 6. Reconstruct full_run.log
MERGED_LOG="${JOB_DIR}/full_run.log"
echo "Constructing merged log at: $MERGED_LOG"

cat "${SLURM_LOG_FILE}" > "${MERGED_LOG}"

echo -e "\n\n==========================================================\n" >> "${MERGED_LOG}"
echo -e "--- MERGED RANK 0 LOG (FOR MEMORY) ---\n" >> "${MERGED_LOG}"
echo -e "==========================================================\n" >> "${MERGED_LOG}"

if [ -f "${RANK_LOG_FILE_2}" ]; then
    cat "${RANK_LOG_FILE_2}" >> "${MERGED_LOG}"
else
    echo "Warning: Rank 0 log file not found at ${RANK_LOG_FILE_2}" >> "${MERGED_LOG}"
fi

echo -e "\n\n==========================================================\n" >> "${MERGED_LOG}"
echo -e "--- RANK ${LAST_RANK} METRICS (LOSS/TFLOPS) BELOW ---\n" >> "${MERGED_LOG}"
echo -e "==========================================================\n" >> "${MERGED_LOG}"

if [ -f "${RANK_LOG_FILE}" ]; then
    cat "${RANK_LOG_FILE}" >> "${MERGED_LOG}"
else
    echo "Warning: Rank log file not found at ${RANK_LOG_FILE}" >> "${MERGED_LOG}"
fi

echo "Merging Complete. Running Python Analysis..."

# 7. Run Python Analysis Scripts
# Make sure to run the script from the directory containing ../utils/ 
# or adjust the paths appropriately.
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
ANALYZE_LOG_PY="${SCRIPT_DIR}/../utils/analyze_log.py"
ANALYZE_MEM_PY="${SCRIPT_DIR}/../utils/analyze_memory.py"

python "$ANALYZE_LOG_PY" \
        "${MERGED_LOG}" \
        --iterations "$TRAIN_ITERS" \
        --nodes "$NODES" \
        --gpus_per_node "$gpus_per_node" \
        --pp "$PP_SIZE" \
        --ep "$EP_PARALLEL_SIZE" \
        --dp "$DP_SIZE" \
        --tp "$MP_SIZE" \
        --gbs "$GLOBAL_BATCH_SIZE" \
        --mbs "$BATCH_SIZE" \
        --slurm_job_id "$SLURM_JOB_ID" \
        --seqlen "$SEQ_LEN" \
        --num_layers "$NUM_LAYERS" \
        --num_experts "$EP_PARALLEL_SIZE" \
        --expert_dim "$FFN_HIDDEN_SIZE" \
        --topk "$TOPK" \
        --hidden_dim "$HIDDEN_SIZE" \
        --moe_type "$MOE_TYPE" \
        --model_size "$MODEL_SIZE" \
        --activation_checkpointing "$ACTIVATION_CHECKPOINT" \
        --checkpoint_interval "$CHECKPOINT_NUM_LAYERS" \
        --dynamic_checkpoint "$DYNAMIC_CHECKPOINT" \
        --uneven_pp "$UNEVEN_PP" >> "${MERGED_LOG}"

# Also print the output to standard out so you can see it locally while running
tail -n 45 "${MERGED_LOG}" 

# Run memory analysis if the script exists
if [ -f "$ANALYZE_MEM_PY" ]; then
    python "$ANALYZE_MEM_PY" "${MERGED_LOG}"
fi

echo "--- Retroactive Analysis Completed Successfully ---"