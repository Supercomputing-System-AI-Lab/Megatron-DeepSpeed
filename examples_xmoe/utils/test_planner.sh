#!/bin/bash

# 1. Run the Python script and capture EVERYTHING into a variable
#    This includes the tables, the logs, and the export lines.
# FULL_OUTPUT=$(python planner.py \
#     --d_model 5120 --seqlen 4096 --num_layers 32 \
#     --pp_stages 4 --dp 8 --ep 8 \
#     --num_experts 128 --expert_dim 1536 --topk 6 \
#     --activation_checkpointing none --mbs 1 \
#     --fixed_num_batches 90 --profile_dir ../scripts-frontier/planner_profiling_cache)

# # 10B
# FULL_OUTPUT=$(python planner.py \
#     --d_model 2048 --seqlen 2048 --num_layers 24 \
#     --pp_stages 2 --dp 8 --ep 8 \
#     --num_experts 64 --expert_dim 1408 --topk 6 \
#     --activation_checkpointing none --mbs 1 \
#     --fixed_num_batches 200 --profile_dir ../scripts-frontier/planner_profiling_cache)

# # 50B
# FULL_OUTPUT=$(python planner.py \
#     --d_model 5120 --seqlen 4096 --num_layers 24 \
#     --pp_stages 4 --dp 8 --ep 8 \
#     --num_experts 128 --expert_dim 1536 --topk 6 \
#     --activation_checkpointing none --mbs 1 \
#     --fixed_num_batches 200 --profile_dir ../scripts-frontier/planner_profiling_cache)

# # 63B
FULL_OUTPUT=$(python planner.py \
    --d_model 5120 --seqlen 4096 --num_layers 32 \
    --pp_stages 4 --dp 8 --ep 8 \
    --num_experts 128 --expert_dim 1536 --topk 6 \
    --activation_checkpointing none --mbs 1 \
    --fixed_num_batches 200 --profile_dir ../scripts-frontier/planner_profiling_cache)
echo "$FULL_OUTPUT"
echo "-----------------------------------------------------------------------------------------------------------------------------------------"
# # 63B
# FULL_OUTPUT=$(python planner.py \
#     --d_model 5120 --seqlen 4096 --num_layers 32 \
#     --pp_stages 6 --dp 8 --ep 8 \
#     --num_experts 128 --expert_dim 1536 --topk 6 \
#     --activation_checkpointing none --mbs 1 \
#     --fixed_num_batches 200 --profile_dir ../scripts-frontier/planner_profiling_cache)
# echo "$FULL_OUTPUT"
# echo "-----------------------------------------------------------------------------------------------------------------------------------------"
# 63B
# FULL_OUTPUT=$(python planner.py \
#     --d_model 5120 --seqlen 4096 --num_layers 32 \
#     --pp_stages 8 --dp 8 --ep 8 \
#     --num_experts 128 --expert_dim 1536 --topk 6 \
#     --activation_checkpointing none --mbs 1 \
#     --fixed_num_batches 200 --profile_dir ../scripts-frontier/planner_profiling_cache)
# echo "$FULL_OUTPUT"
# echo "-----------------------------------------------------------------------------------------------------------------------------------------"
# # 63B
# FULL_OUTPUT=$(python planner.py \
#     --d_model 5120 --seqlen 4096 --num_layers 32 \
#     --pp_stages 16 --dp 8 --ep 8 \
#     --num_experts 128 --expert_dim 1536 --topk 6 \
#     --activation_checkpointing none --mbs 1 \
#     --fixed_num_batches 200 --profile_dir ../scripts-frontier/planner_profiling_cache)
# echo "$FULL_OUTPUT"
# echo "-----------------------------------------------------------------------------------------------------------------------------------------"
# # 63B
# FULL_OUTPUT=$(python planner.py \
#     --d_model 5120 --seqlen 4096 --num_layers 32 \
#     --pp_stages 32 --dp 8 --ep 8 \
#     --num_experts 128 --expert_dim 1536 --topk 6 \
#     --activation_checkpointing none --mbs 1 \
#     --fixed_num_batches 200 --profile_dir ../scripts-frontier/planner_profiling_cache)

# # 173B
# FULL_OUTPUT=$(python planner.py \
#     --d_model 7168 --seqlen 4096 --num_layers 24 \
#     --pp_stages 10 --dp 8 --ep 8 \
#     --num_experts 256 --expert_dim 2048 --topk 8 \
#     --activation_checkpointing none --mbs 1 \
#     --fixed_num_batches 200 --profile_dir ../scripts-frontier/planner_profiling_cache)

# 537B
# FULL_OUTPUT=$(python planner.py \
#     --d_model 7168 --seqlen 4096 --num_layers 60 \
#     --pp_stages 30 --dp 8 --ep 8 \
#     --num_experts 256 --expert_dim 2560 --topk 8 \
#     --activation_checkpointing none --mbs 1 \
#     --fixed_num_batches 200 --profile_dir ../scripts-frontier/planner_profiling_cache)

# # 1T
# FULL_OUTPUT=$(python planner.py \
#     --d_model 7168 --seqlen 4096 --num_layers 60 \
#     --pp_stages 60 --dp 8 --ep 8 \
#     --num_experts 512 --expert_dim 2560 --topk 8 \
#     --activation_checkpointing none --mbs 1 \
#     --fixed_num_batches 230 --profile_dir ../scripts-frontier/planner_profiling_cache)


# FULL_OUTPUT=$(python planner.py \
#     --d_model 2048 --seqlen 2048 --num_layers 32 \
#     --pp_stages 4 --dp 8 --ep 8 \
#     --num_experts 64 --expert_dim 1408 --topk 6 \
#     --activation_checkpointing none --mbs 1 \
#     --fixed_num_batches 90 --profile_dir ../scripts-frontier/planner_profiling_cache)


# 2. Print the logs to the screen immediately so you can read them
# echo "$FULL_OUTPUT"

# # 3. Extract the variables we need
# #    We grep for lines starting with OPTIMAL_ inside the variable we already captured.
# eval "$(echo "$FULL_OUTPUT" | grep "^OPTIMAL_")"

# echo "OPTIMAL_MBS: $OPTIMAL_MBS"
# # 4. Check if optimization succeeded
# if [ "$OPTIMAL_MBS" == "FAILED" ] || [ -z "$OPTIMAL_MBS" ]; then
#     echo "Error: Planner failed to find a valid configuration."
#     exit 1
# fi

# # 5. Now you can use the variables!
# echo "---------------------------------------------------"
# echo "Bash Script is proceeding with:"
# echo "Micro-Batch Size: \"$OPTIMAL_MBS\""
# echo "Layer Partition:  \"$OPTIMAL_PARTITION\""
# echo "Checkpoints:      \"$OPTIMAL_CKPT\""
# echo "---------------------------------------------------"

# # Example: Run your training script
# # torchrun ... --mbs $OPTIMAL_MBS --partition "$OPTIMAL_PARTITION"

# export OPTIMAL_MBS=$OPTIMAL_MBS
# export OPTIMAL_PARTITION=$OPTIMAL_PARTITION
# export OPTIMAL_CKPT=$OPTIMAL_CKPT

# # python dummpy_python.py \
# #     --mbs $OPTIMAL_MBS \
# #     --partition "$OPTIMAL_PARTITION" \
# #     --ckpt "$OPTIMAL_CKPT"
# # python dummpy_python.py 



# FULL_OUTPUT=$(python planner.py \
#     --d_model 5120 --seqlen 4096 --num_layers 32 \
#     --pp_stages 8 --dp 8 --ep 8 \
#     --num_experts 128 --expert_dim 1536 --topk 6 \
#     --mbs 1 --fixed_num_batches 200 \
#     --profile_dir ../scripts-frontier/planner_profiling_cache)
# echo "$FULL_OUTPUT"

# FULL_OUTPUT=$(python planner.py \
#     --d_model 5120 --seqlen 4096 --num_layers 32 \
#     --pp_stages 8 --dp 8 --ep 8 \
#     --num_experts 128 --expert_dim 1536 --topk 6 \
#     --mbs 1 --fixed_num_batches 200 \
#     --profile_dir ../scripts-frontier/planner_profiling_cache \
#     --eval-mbs-list 1 2 4 8 \
#     --eval-strategy none)
# echo "$FULL_OUTPUT"


# FULL_OUTPUT=$(python planner.py \
#     --d_model 5120 --seqlen 4096 --num_layers 32 \
#     --pp_stages 8 --dp 8 --ep 8 \
#     --num_experts 128 --expert_dim 1536 --topk 6 \
#     --mbs 1 --fixed_num_batches 200 \
#     --profile_dir ../scripts-frontier/planner_profiling_cache \
#     --eval-mbs-list 1 2 4 8 \
#     --eval-strategy all)
# echo "$FULL_OUTPUT"




# FULL_OUTPUT=$(python planner.py \
#     --d_model 2048 --seqlen 2048 --num_layers 24 \
#     --pp_stages 1 --dp 8 --ep 8 \
#     --num_experts 64 --expert_dim 1408 --topk 6 \
#     --mbs 1 --fixed_num_batches 200 \
#     --profile_dir ../scripts-frontier/planner_profiling_cache)
# echo "$FULL_OUTPUT"

# FULL_OUTPUT=$(python planner.py \
#     --d_model 2048 --seqlen 2048 --num_layers 24 \
#     --pp_stages 1 --dp 8 --ep 8 \
#     --num_experts 64 --expert_dim 1408 --topk 6 \
#     --mbs 1 --fixed_num_batches 200 \
#     --profile_dir ../scripts-frontier/planner_profiling_cache \
#     --eval-mbs-list 1 2 4 6 8 12 16 \
#     --eval-strategy none)
# echo "$FULL_OUTPUT"


# FULL_OUTPUT=$(python planner.py \
#     --d_model 2048 --seqlen 2048 --num_layers 24 \
#     --pp_stages 1 --dp 8 --ep 8 \
#     --num_experts 64 --expert_dim 1408 --topk 6 \
#     --mbs 1 --fixed_num_batches 200 \
#     --profile_dir ../scripts-frontier/planner_profiling_cache \
#     --eval-mbs-list 1 2 4 6 8 12 16 \
#     --eval-strategy all)
# echo "$FULL_OUTPUT"