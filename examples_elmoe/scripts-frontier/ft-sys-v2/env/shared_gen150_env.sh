#!/bin/bash
# shared_gen150_env.sh (ft-sys-v2, 2026-11-13): the outside paths of a run, set to the copies every member of gen150 can read
# under /lustre/orion/world-shared/gen150/zixianw4. Source it from a fresh clone before env/ftv2_check_env.sh and the run files:
#   cd <X-MoE clone>/Megatron-DeepSpeed-X-MoE && . examples_elmoe/scripts-frontier/ft-sys-v2/env/shared_gen150_env.sh
# What is shared (read only):
#   envs/TORCH2.13-ROCM7.14-XMOE_venv       the Python venv (the default of NEWSTACK_V already)
#   ftv2/rocm-7.14.1-gfx90a                 ROCm 7.14.1 for gfx90a (HIP, HSA, RCCL 2.30.4); the same files as the runs' copy
#   ftv2/aws-ofi-nccl-1.22.0/lib            RCCL's network plugin; the same file as the runs' copy
#   tools/etcd-v3.5.17                      etcd (the default of FTV2_ETCD_BIN already)
#   ftv2/data                               GPT-2 data and vocabulary: my-gpt2_text_document.{bin,idx}, gpt2-vocab.json,
#                                           gpt2-merges.txt (what every run since 2026-11-11 uses; no DeepSeek file is needed)
# What stays yours: the two clones, a JIT folder you can write, py-spy (pip install --user py-spy).
_ft="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"                 # .../scripts-frontier/ft-sys-v2
_sf="$(cd "$_ft/.." && pwd)"                                          # .../scripts-frontier
_share=/lustre/orion/world-shared/gen150/zixianw4
export NEWSTACK_DEEPSPEED="$(cd "$_sf/../../.." && pwd)"              # the X-MoE clone's root (its deepspeed/ is imported)
export NEWSTACK_OVERLAY="$_ft/env/overlay"
export NEWSTACK_R714=$_share/ftv2/rocm-7.14.1-gfx90a
export NEWSTACK_PLUGIN=$_share/ftv2/aws-ofi-nccl-1.22.0/lib
export NEWSTACK_JIT=${NEWSTACK_JIT:-/lustre/orion/gen150/scratch/$USER/ftv2_jit}
mkdir -p "$NEWSTACK_JIT"
# The GPT-2 tokenizer rider of the launcher reads ../data/gpt2-vocab.json and ../data/gpt2-merges.txt relative to
# scripts-frontier, so the two files are linked into this clone's examples_elmoe/data (not tracked by git).
export AUTORUN_GPT2_TOKENIZER=1 DATA_PATH=$_share/ftv2/data/my-gpt2_text_document
for _f in gpt2-vocab.json gpt2-merges.txt; do
  [ -e "$_sf/../data/$_f" ] || ln -s "$_share/ftv2/data/$_f" "$_sf/../data/$_f"
done
export WANDB_MODE=${WANDB_MODE:-disabled}
unset _ft _sf _share _f
