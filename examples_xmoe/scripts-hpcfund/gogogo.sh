#!/bin/bash


# # # ----------------------------------------------------------------------------------
# # # PP runs
# # # 2 node 8 proc
# # # ----------------------------------------------------------------------------------
# BATCH_SIZE=(2 2 2 2 2 3 3 3 3 3 4 4)
# GLOBAL_BATCH_SIZE=(16 32 64 96 128 24 48 96 140 192 64 128)
# # BATCH_SIZE=(3)
# # GLOBAL_BATCH_SIZE=(48)
# PP_SIZE=(2)
# EP_PARALLEL_SIZE=(8)

# # --- Safety Check: Ensure the arrays have the same length ---
# if [ ${#BATCH_SIZE[@]} -ne ${#GLOBAL_BATCH_SIZE[@]} ]; then
#     echo "Error: BATCH_SIZE and GLOBAL_BATCH_SIZE arrays must have the same number of elements."
#     exit 1
# fi


# for j in "${!PP_SIZE[@]}"; do
#     for i in "${!BATCH_SIZE[@]}"; do
#         # Get the corresponding values using the index 'i'
#         pp=${PP_SIZE[j]}
#         ep=${EP_PARALLEL_SIZE[j]}
#         bs=${BATCH_SIZE[i]}
#         gbs=${GLOBAL_BATCH_SIZE[i]}

#         echo "Submitting job with BS=${bs}, GBS=${gbs}, PP=${pp}, EP=${ep}"
#         sbatch n2-p8-Small-XMoE-pp.slurm ${bs} ${gbs} ${pp} ${ep}
#     done
# done



# # # ----------------------------------------------------------------------------------
# # # PP runs
# # # 4 node 8 proc
# # # ----------------------------------------------------------------------------------
# BATCH_SIZE=(2 2 2 2 2 3 3 3 3 3 4 4)
# GLOBAL_BATCH_SIZE=(16 32 64 96 128 24 48 96 140 192 64 128)
# # BATCH_SIZE=(3)
# # GLOBAL_BATCH_SIZE=(96)
# PP_SIZE=(4)
# EP_PARALLEL_SIZE=(8)

# # --- Safety Check: Ensure the arrays have the same length ---
# if [ ${#BATCH_SIZE[@]} -ne ${#GLOBAL_BATCH_SIZE[@]} ]; then
#     echo "Error: BATCH_SIZE and GLOBAL_BATCH_SIZE arrays must have the same number of elements."
#     exit 1
# fi


# for j in "${!PP_SIZE[@]}"; do
#     for i in "${!BATCH_SIZE[@]}"; do
#         # Get the corresponding values using the index 'i'
#         pp=${PP_SIZE[j]}
#         ep=${EP_PARALLEL_SIZE[j]}
#         bs=${BATCH_SIZE[i]}
#         gbs=${GLOBAL_BATCH_SIZE[i]}

#         echo "Submitting job with BS=${bs}, GBS=${gbs}, PP=${pp}, EP=${ep}"
#         sbatch n4-p8-Small-XMoE-pp.slurm ${bs} ${gbs} ${pp} ${ep}
#     done
# done




# # # ----------------------------------------------------------------------------------
# # # EP runs
# # # 2 node 8 proc
# # # ----------------------------------------------------------------------------------
# BATCH_SIZE=(2 2 2 2 2 3 3 3 3 3 4 4)
# GLOBAL_BATCH_SIZE=(16 32 64 96 128 24 48 96 140 192 64 128)
# # BATCH_SIZE=(3)
# # GLOBAL_BATCH_SIZE=(48)
# PP_SIZE=(2)
# EP_PARALLEL_SIZE=(8)

# # --- Safety Check: Ensure the arrays have the same length ---
# if [ ${#BATCH_SIZE[@]} -ne ${#GLOBAL_BATCH_SIZE[@]} ]; then
#     echo "Error: BATCH_SIZE and GLOBAL_BATCH_SIZE arrays must have the same number of elements."
#     exit 1
# fi


# for j in "${!PP_SIZE[@]}"; do
#     for i in "${!BATCH_SIZE[@]}"; do
#         # Get the corresponding values using the index 'i'
#         pp=${PP_SIZE[j]}
#         ep=${EP_PARALLEL_SIZE[j]}
#         bs=${BATCH_SIZE[i]}
#         gbs=${GLOBAL_BATCH_SIZE[i]}

#         echo "Submitting job with BS=${bs}, GBS=${gbs}, PP=${pp}, EP=${ep}"
#         sbatch n2-p8-Small-XMoE.slurm ${bs} ${gbs} ${pp} ${ep}
#     done
# done



# # # ----------------------------------------------------------------------------------
# # # EP runs
# # # 4 node 8 proc
# # # ----------------------------------------------------------------------------------
# BATCH_SIZE=(2 2 2 2 2 3 3 3 3 3 4 4)
# GLOBAL_BATCH_SIZE=(16 32 64 96 128 24 48 96 140 192 64 128)
# # BATCH_SIZE=(3)
# # GLOBAL_BATCH_SIZE=(96)
# PP_SIZE=(4)
# EP_PARALLEL_SIZE=(8)

# # --- Safety Check: Ensure the arrays have the same length ---
# if [ ${#BATCH_SIZE[@]} -ne ${#GLOBAL_BATCH_SIZE[@]} ]; then
#     echo "Error: BATCH_SIZE and GLOBAL_BATCH_SIZE arrays must have the same number of elements."
#     exit 1
# fi


# for j in "${!PP_SIZE[@]}"; do
#     for i in "${!BATCH_SIZE[@]}"; do
#         # Get the corresponding values using the index 'i'
#         pp=${PP_SIZE[j]}
#         ep=${EP_PARALLEL_SIZE[j]}
#         bs=${BATCH_SIZE[i]}
#         gbs=${GLOBAL_BATCH_SIZE[i]}

#         echo "Submitting job with BS=${bs}, GBS=${gbs}, PP=${pp}, EP=${ep}"
#         sbatch n4-p8-Small-XMoE.slurm ${bs} ${gbs} ${pp} ${ep}
#     done
# done



# # # ----------------------------------------------------------------------------------
# # # PP runs
# # # 4 node 4 proc
# # # ----------------------------------------------------------------------------------
# # BATCH_SIZE=(2 2 2 2 2 3 3 3 3 3)
# # GLOBAL_BATCH_SIZE=(16 32 64 96 128 24 48 96 140 192)
# BATCH_SIZE=(3)
# GLOBAL_BATCH_SIZE=(192)
# PP_SIZE=(2 4 8)
# EP_PARALLEL_SIZE=(8 4 2)

# # --- Safety Check: Ensure the arrays have the same length ---
# if [ ${#BATCH_SIZE[@]} -ne ${#GLOBAL_BATCH_SIZE[@]} ]; then
#     echo "Error: BATCH_SIZE and GLOBAL_BATCH_SIZE arrays must have the same number of elements."
#     exit 1
# fi


# for j in "${!PP_SIZE[@]}"; do
#     for i in "${!BATCH_SIZE[@]}"; do
#         # Get the corresponding values using the index 'i'
#         pp=${PP_SIZE[j]}
#         ep=${EP_PARALLEL_SIZE[j]}
#         bs=${BATCH_SIZE[i]}
#         gbs=${GLOBAL_BATCH_SIZE[i]}

#         echo "Submitting job with BS=${bs}, GBS=${gbs}, PP=${pp}, EP=${ep}"
#         sbatch n4-p4-Small-XMoE-pp.slurm ${bs} ${gbs} ${pp} ${ep}
#     done
# done



# # # ----------------------------------------------------------------------------------
# # # EP runs
# # # 4 node 4 proc
# # # ----------------------------------------------------------------------------------
# # BATCH_SIZE=(2 2 2 2 2 3 3 3 3 3)
# # GLOBAL_BATCH_SIZE=(16 32 64 96 128 24 48 96 140 192)
# # BATCH_SIZE=(3)
# # GLOBAL_BATCH_SIZE=(192)
# PP_SIZE=(1)
# EP_PARALLEL_SIZE=(16)

# # --- Safety Check: Ensure the arrays have the same length ---
# if [ ${#BATCH_SIZE[@]} -ne ${#GLOBAL_BATCH_SIZE[@]} ]; then
#     echo "Error: BATCH_SIZE and GLOBAL_BATCH_SIZE arrays must have the same number of elements."
#     exit 1
# fi


# for j in "${!PP_SIZE[@]}"; do
#     for i in "${!BATCH_SIZE[@]}"; do
#         # Get the corresponding values using the index 'i'
#         pp=${PP_SIZE[j]}
#         ep=${EP_PARALLEL_SIZE[j]}
#         bs=${BATCH_SIZE[i]}
#         gbs=${GLOBAL_BATCH_SIZE[i]}

#         echo "Submitting job with BS=${bs}, GBS=${gbs}, PP=${pp}, EP=${ep}"
#         sbatch n4-p4-Small-XMoE.slurm ${bs} ${gbs} ${pp} ${ep}
#     done
# done







# # ----------------------------------------------------------------------------------
# # PP runs
# # 2 node 4 proc
# # ----------------------------------------------------------------------------------
# # BATCH_SIZE=(2 2 2 3 3 3)
# # GLOBAL_BATCH_SIZE=(16 32 64 24 48 96)
# BATCH_SIZE=(2)
# GLOBAL_BATCH_SIZE=(192)
# PP_SIZE=(2)
# EP_PARALLEL_SIZE=(4)

# # --- Safety Check: Ensure the arrays have the same length ---
# if [ ${#BATCH_SIZE[@]} -ne ${#GLOBAL_BATCH_SIZE[@]} ]; then
#     echo "Error: BATCH_SIZE and GLOBAL_BATCH_SIZE arrays must have the same number of elements."
#     exit 1
# fi


# for j in "${!PP_SIZE[@]}"; do
#     for i in "${!BATCH_SIZE[@]}"; do
#         # Get the corresponding values using the index 'i'
#         pp=${PP_SIZE[j]}
#         ep=${EP_PARALLEL_SIZE[j]}
#         bs=${BATCH_SIZE[i]}
#         gbs=${GLOBAL_BATCH_SIZE[i]}

#         echo "Submitting job with BS=${bs}, GBS=${gbs}, PP=${pp}, EP=${ep}"
#         sbatch n2-p4-Small-XMoE-pp.slurm ${bs} ${gbs} ${pp} ${ep}
#     done
# done



# ----------------------------------------------------------------------------------
# EP runs
# 2 node 4 proc
# ----------------------------------------------------------------------------------
# BATCH_SIZE=(2 2 2 3 3 3)
# GLOBAL_BATCH_SIZE=(16 32 64 24 48 96)
BATCH_SIZE=(2)
GLOBAL_BATCH_SIZE=(192)
PP_SIZE=(1)
EP_PARALLEL_SIZE=(8)

# --- Safety Check: Ensure the arrays have the same length ---
if [ ${#BATCH_SIZE[@]} -ne ${#GLOBAL_BATCH_SIZE[@]} ]; then
    echo "Error: BATCH_SIZE and GLOBAL_BATCH_SIZE arrays must have the same number of elements."
    exit 1
fi


for j in "${!PP_SIZE[@]}"; do
    for i in "${!BATCH_SIZE[@]}"; do
        # Get the corresponding values using the index 'i'
        pp=${PP_SIZE[j]}
        ep=${EP_PARALLEL_SIZE[j]}
        bs=${BATCH_SIZE[i]}
        gbs=${GLOBAL_BATCH_SIZE[i]}

        echo "Submitting job with BS=${bs}, GBS=${gbs}, PP=${pp}, EP=${ep}"
        sbatch n2-p4-Small-XMoE.slurm ${bs} ${gbs} ${pp} ${ep}
    done
done




