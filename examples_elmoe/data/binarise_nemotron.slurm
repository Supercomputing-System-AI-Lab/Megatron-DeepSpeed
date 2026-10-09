#!/bin/bash
# ==============================================================================================
# STAGE 2+3 of the Nemotron prep, as a batch job.
#
# WHY THIS IS BATCHABLE AND STAGE 1 WAS NOT
#   STAGE 1 streams from HuggingFace, and Frontier compute nodes have no external network -- it
#   had to run on a login node. STAGE 2 only reads a local JSONL and writes a local .bin/.idx,
#   so it belongs here.
#
# WHY IT WOULD NOT HAVE BLOWN UP THE LOGIN NODE ANYWAY
#   Unlike ds_to_universal.py -- where each extract worker holds a whole ZeRO shard resident,
#   tens of GiB, which is what makes it a compute-node job -- preprocess_data.py streams the
#   JSONL line by line and each worker tokenizes one document at a time. Memory stays at a few
#   GB regardless of --workers. The login-node cost is CPU, not RAM. This is about being a good
#   neighbour, not about avoiding an OOM.
#
# SUBMIT FROM YOUR ACTIVATED CONDA ENV -- SLURM propagates PATH, which is how the training job
# picks up `python` too (there is no `conda activate` anywhere in the template).
#
#   conda activate TORCH-ROCM6.4.1_env_groupgemm_v4
#   sbatch binarise_nemotron.slurm
# ==============================================================================================
#SBATCH -A gen150
#SBATCH -J nemotron-binarise
#SBATCH -o logs/binarise_%j.out
#SBATCH -e logs/binarise_%j.err
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "host      $(hostname)"
echo "python    $(which python)"
echo "started   $(date)"
echo

# The prep script skips STAGE 1 because nemotron_deepseek.jsonl already exists, so this runs
# STAGE 2 (binarise) and STAGE 3 (verify the index describes the .bin) and nothing else.
#
# 32 workers: one node has 56 usable cores and tokenization is embarrassingly parallel.
srun -N1 -n1 -c 56 env WORKERS=32 bash prepare_nemotron_deepseek.sh

echo
echo "finished  $(date)"
