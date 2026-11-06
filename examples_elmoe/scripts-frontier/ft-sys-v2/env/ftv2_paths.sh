# ftv2_paths.sh (ft-sys-v2, 2026-11-06): every path an ft-sys-v2 run reads, one variable each. Sourced by runs/ftv2_run_common.sh,
# env/newstack_env.sh and env/ftv2_check_env.sh; it only assigns (no export, no module, no mkdir), so sourcing it twice is harmless.
# A variable already set in the shell wins. The defaults outside the repositories are the values the runs used on 2026-11-06
# (the user's own folders), so the user's runs are unchanged. Another user sets the variables marked "private" below to copies of
# their own (README.md, "Run from a fresh clone"); env/ftv2_check_env.sh prints each path with its state.
#
# Inside the repositories (this clone):
#   FTV2_TREE           the Megatron-DeepSpeed tree: megatron/ and ft-sys-v2/ (default: the tree that holds this file)
#   FTV2_SF             the folder the job is launched from: the 0623 launcher, the job template, eval/submit_env.sh; the DeepSpeed
#                       json template is ../ds_config_gpt_TEMPLATE.json next to it. Job folders (logs/, temp_slurm/) are written here.
#   FTV2_ENV            the new stack's env file (default: newstack_env.sh next to this file)
# Outside the repositories:
#   NEWSTACK_V          the Python venv (torch 2.13.0+rocm7.1, apex, flash-attn 2.8.3 CK, mpi4py on Cray MPICH)   world-shared, readable
#   NEWSTACK_DEEPSPEED  the X-MoE DeepSpeed root put on PYTHONPATH (eval/submit_env.sh puts the user's checkout there) private
#   NEWSTACK_R714       the ROCm 7.14.1 gfx90a tree (TheRock tarball; RCCL 2.30.4 with gfx90a code)            private
#   NEWSTACK_PLUGIN     aws-ofi-nccl 1.22.0 built against ROCm 7.14.1 and libfabric 2.3.1 (its lib folder)        private
#   NEWSTACK_LIBFABRIC  libfabric 2.3.1 (site)
#   NEWSTACK_JIT        the JIT build caches (torch extensions, Megatron fused kernels, Triton); must be writable  private
#   NEWSTACK_OVERLAY    overlay/triton_deepbind (sitecustomize.py, a copy is in env/overlay) and overlay/fa_triton_amd
#                       (only with NEWSTACK_FA=triton)                                                         private
#   NEWSTACK_MEGATRON_COPY  a copy of megatron/ that the 11-04 new-stack runs put on PYTHONPATH; an ft-sys-v2 run puts FTV2_TREE
#                       before it, so megatron is imported from FTV2_TREE (runs/ftv2_run_common.sh checks it)      private, unused
#   FTV2_ETCD_BIN       the etcd 3.5.17 binary (heartbeat-sys/get_etcd.sh fetches and checks one)             world-shared, readable
#   DATA_PATH           the training data prefix (.bin + .idx); the job template reads it (empty: the template's default, private)
#   FTV2_VOCAB_FILE     the tokenizer file; empty: the template's VOCAB_FILE (private). Set, the batch script replaces the template's
#                       --vocab-file argument in data_options before the training srun (runs/ftv2_run_common.sh)
FTV2_TREE=${FTV2_TREE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
FTV2_SF=${FTV2_SF:-$FTV2_TREE/examples_elmoe/scripts-frontier}
FTV2_ENV=${FTV2_ENV:-$FTV2_TREE/examples_elmoe/scripts-frontier/ft-sys-v2/env/newstack_env.sh}
_ftv2_scope=/lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/assistant_deep_dive/2026-11-04-xmoe_newstack
NEWSTACK_V=${NEWSTACK_V:-/lustre/orion/world-shared/gen150/zixianw4/envs/TORCH2.13-ROCM7.14-XMOE_venv}
NEWSTACK_DEEPSPEED=${NEWSTACK_DEEPSPEED:-/lustre/orion/gen150/scratch/zixianw4/X-MoE}
NEWSTACK_R714=${NEWSTACK_R714:-/lustre/orion/gen150/scratch/zixianw4/therock-7.14.1-gfx90a/rocm}
NEWSTACK_PLUGIN=${NEWSTACK_PLUGIN:-/lustre/orion/gen150/scratch/zixianw4/aws-ofi-nccl-v1.22.0/install/lib}
NEWSTACK_LIBFABRIC=${NEWSTACK_LIBFABRIC:-/opt/cray/libfabric/2.3.1/lib64}
NEWSTACK_JIT=${NEWSTACK_JIT:-$_ftv2_scope/jit}
NEWSTACK_OVERLAY=${NEWSTACK_OVERLAY:-$_ftv2_scope/overlay}
NEWSTACK_MEGATRON_COPY=${NEWSTACK_MEGATRON_COPY:-$_ftv2_scope/tree}
FTV2_ETCD_BIN=${FTV2_ETCD_BIN:-/lustre/orion/world-shared/gen150/zixianw4/tools/etcd-v3.5.17/etcd-v3.5.17-linux-amd64/etcd}
FTV2_VOCAB_FILE=${FTV2_VOCAB_FILE:-}
unset _ftv2_scope
