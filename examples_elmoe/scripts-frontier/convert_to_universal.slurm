#!/bin/bash
#SBATCH -A gen150
#SBATCH -J ds2universal
#SBATCH -o logs/ds2universal_%j.o
#SBATCH -e logs/ds2universal_%j.e
#SBATCH -t 0:15:00
#SBATCH -p batch
#SBATCH -N 1
#
# Convert a DeepSpeed MoE checkpoint to Universal Checkpoint format.
#
#   sbatch --export=ALL,STEP_DIR=checkpoint/2026-07-30_5120138/global_step60 convert_to_universal.slurm
#
# Optional overrides:  OUT_DIR, WORKERS_EXTRACT, WORKERS_MERGE
#
# WHY A BATCH JOB, not a login node:
#   * extract holds one whole ZeRO shard per worker -- ~23 GiB each for a 16B MoE
#   * it reads ~183 GiB and writes ~183 GiB of fragments, then merges those into ~183 GiB of atoms
#     (peak scratch ~2x the optimizer state; tmp/ is removed at the end)
#   * a login node is shared, and this is heavy on both RAM and Lustre metadata
#
# Pure CPU + IO. No GPUs are requested on purpose.
#
# WALLTIME: keep this generous. A kill during the merge phase leaves a partially populated zero/
# directory; the resume then dies on the first missing atom. 1h is ample; the job exits when done.

set -euo pipefail

: "${STEP_DIR:?set STEP_DIR to the global_step<N> folder to convert}"

# Slurm copies the batch script to its spool dir, so $0 is NOT the submit directory -- use
# SLURM_SUBMIT_DIR. (Slurm already starts the job here; this makes the script safe to run directly too.)
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

OUT_DIR="${OUT_DIR:-${STEP_DIR}_universal}"
WORKERS_EXTRACT="${WORKERS_EXTRACT:-4}"   # each worker holds ONE whole ZeRO shard (~23 GiB here)
WORKERS_MERGE="${WORKERS_MERGE:-4}"       # holds whole parameters; the largest is the embedding

echo "=== ds_to_universal ==="
echo "  cwd    : $(pwd)"
echo "  input  : ${STEP_DIR}"
echo "  output : ${OUT_DIR}"
echo "  workers: extract=${WORKERS_EXTRACT} merge=${WORKERS_MERGE}"
echo "  started: $(date '+%F %T')"
[ -d "${STEP_DIR}" ] || { echo "FATAL: ${STEP_DIR} does not exist (relative to $(pwd))"; exit 1; }
df -h . | tail -1

# Large sequential atom files: stripe so the merge phase is not served by a single OST.
mkdir -p "${OUT_DIR}"
lfs setstripe -c 8 -S 4M "${OUT_DIR}" 2>/dev/null || echo "  (lfs setstripe unavailable; continuing)"

srun -N1 -n1 -c 32 python -m deepspeed.checkpoint.ds_to_universal \
    --input_folder  "${STEP_DIR}" \
    --output_folder "${OUT_DIR}" \
    --num_extract_workers "${WORKERS_EXTRACT}" \
    --num_merge_workers   "${WORKERS_MERGE}"

echo "=== done: $(date '+%F %T') ==="
echo "--- output ---"
ls -la "${OUT_DIR}" || true
n_atoms=$(find "${OUT_DIR}/zero" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
n_expert=$(find "${OUT_DIR}/zero" -mindepth 1 -maxdepth 1 -type d -name '*deepspeed_experts*' 2>/dev/null | wc -l)
echo "atoms total   : ${n_atoms}"
echo "expert atoms  : ${n_expert}   (expect num_moe_layers * num_experts * 2)"
echo "optimizer_state.pt: $([ -f "${OUT_DIR}/zero/optimizer_state.pt" ] && echo present || echo MISSING)"
echo "tmp/ removed  : $([ -d "${OUT_DIR}/tmp" ] && echo NO--conversion did not finish cleanly || echo yes)"
echo "latest_universal -> $(cat "$(dirname "${OUT_DIR}")/latest_universal" 2>/dev/null || echo MISSING)"
