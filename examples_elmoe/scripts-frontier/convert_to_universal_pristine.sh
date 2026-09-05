#!/bin/bash
# Convert an X-MoE DeepSpeed checkpoint to Universal Checkpoint format using the
# UN-EDITED ELMoE tree.  Companion to examples_elmoe/0804_2026_UCP_ON_PRISTINE_TREE.md
#
#   # activate the same environment you train with, then:
#   bash convert_to_universal_pristine.sh
#   bash convert_to_universal_pristine.sh checkpoint/<date>_<job>/global_step<N>
#
# The one mechanism here: PYTHONPATH puts the pristine tree ahead of the installed
# deepspeed, so `deepspeed.checkpoint` resolves to ELMOE/X-MoE/deepspeed/checkpoint/.
#
# CPU + IO only, no GPU.  Expect it to stop in under a second on the login node --
# torch >= 2.6 rejects the checkpoint before the 23 GiB optimizer-shard read
# (blocker 4 in the doc).  That guard is why running it here is cheap.  If you ever
# get past it, move to a compute node: extract holds one ~23 GiB shard per worker.

set -euo pipefail

ELMOE_DIR=${ELMOE_DIR:-/lustre/orion/gen150/scratch/zixianw4/ELMOE/X-MoE}

STEP_DIR=${1:-/lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier/checkpoint/2026-08-02_5142659/global_step60}
STEP_DIR=$(readlink -f "${STEP_DIR}")
OUT_DIR=${OUT_DIR:-${ELMOE_DIR}/ucp_out/$(basename "$(dirname "${STEP_DIR}")")_$(basename "${STEP_DIR}")_universal}

python -c 'import torch, cpuinfo' 2>/dev/null || {
    echo "FATAL: this shell has no working torch + deepspeed dependencies."
    echo "       Activate the environment you train with, then re-run."
    exit 1
}

mkdir -p "${OUT_DIR}"

echo "converter : ${ELMOE_DIR}/deepspeed/checkpoint/ds_to_universal.py"
echo "input     : ${STEP_DIR}"
echo "output    : ${OUT_DIR}"
echo

PYTHONPATH="${ELMOE_DIR}:${PYTHONPATH:-}" python ${ELMOE_DIR}/deepspeed/checkpoint/ds_to_universal.py \
    --input_folder  "${STEP_DIR}" \
    --output_folder "${OUT_DIR}" \
    --num_extract_workers 4 \
    --num_merge_workers   2
