# newstack_env.sh (ft-sys-v2 copy, 2026-11-06): assistant_deep_dive/2026-11-04-xmoe_newstack/runs/newstack_env.sh (2026-11-04) moved
# into the repository. Every path outside the repositories is now a variable of ftv2_paths.sh (next to this file) whose default is
# the original's value; with the defaults this file sets exactly what the original set. One addition: NEWSTACK_DEEPSPEED replaces
# the DeepSpeed entry that eval/submit_env.sh puts on PYTHONPATH (default: the same folder, so no change).
#
# Sourced by the run files AFTER `. ./eval/submit_env.sh`, in the submitting shell (sbatch exports it). It swaps the production
# stack (rocm/6.4.1 module, conda env v5, aws-ofi-rccl 1.4.0) for:
#   python   = the venv TORCH2.13-ROCM7.14-XMOE_venv (torch 2.13.0+rocm7.1, apex 1.16.0a0, flash-attn 2.8.3 CK gfx90a hdim 128)
#   toolchain for the run's JIT builds (Megatron fused softmax, any DeepSpeed op) = ROCm 7.1.1, the ROCm torch was built with
#   runtime  = the complete ROCm 7.14.1 for gfx90a (HSA, HIP, RCCL 2.30.4) through LD_PRELOAD by bare name (the site's 7.14.1 RCCL
#              has no gfx90a code: AMD's repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx90a-7.14.1.tar.gz unpacked),
#              aws-ofi-nccl 1.22.0, libfabric 2.3.1
# The 13 site settings and LD_PRELOAD cannot be set here: the job template re-exports FI_CXI_* and NCCL_SOCKET_IFNAME after this
# shell's values, and its srun line sets LD_PRELOAD="${DARSHAN_PRELOAD}". They are set by NEWSTACK_PRE_CMD (below), which the run
# file passes as AUTORUN_PRE_CMD: the launcher inserts it right before the training srun.
. "$(dirname "${BASH_SOURCE[0]}")/ftv2_paths.sh"
module unload rocm/6.4.1 >/dev/null 2>&1; module load rocm/7.1.1 >/dev/null 2>&1
export ROCM_HOME=/opt/rocm-7.1.1 ROCM_PATH=/opt/rocm-7.1.1 CUDA_HOME=/opt/rocm-7.1.1 HIP_PATH=/opt/rocm-7.1.1
export PATH=$NEWSTACK_V/bin:/opt/rocm-7.1.1/bin:$PATH
# the old plugin (aws-ofi-rccl-rocm6.4) must not be on the path: it needs libamdhip64.so.6
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v -e aws-ofi-rccl-rocm6.4 -e '/opt/rocm-6' | paste -sd: -)
export LD_LIBRARY_PATH=$NEWSTACK_PLUGIN:$NEWSTACK_R714/lib:$NEWSTACK_LIBFABRIC:$LD_LIBRARY_PATH
mkdir -p $NEWSTACK_JIT/torch_extensions $NEWSTACK_JIT/megatron_fused $NEWSTACK_JIT/triton
# DeepSpeed: eval/submit_env.sh sets PYTHONPATH to the user's X-MoE checkout; NEWSTACK_DEEPSPEED (a clone's root) takes its place.
export PYTHONPATH=$(echo "${PYTHONPATH:-}" | tr ':' '\n' | grep -v -x -F /lustre/orion/gen150/scratch/zixianw4/X-MoE | paste -sd: -)
export PYTHONPATH=$NEWSTACK_DEEPSPEED${PYTHONPATH:+:$PYTHONPATH}
# Triton must be loaded with RTLD_DEEPBIND on this runtime (BUILD.md step 7): the HIP 7.14.1 runtime brings libLLVM.so.23 into the
# process and `import triton` otherwise dies with a segmentation fault. overlay/triton_deepbind/sitecustomize.py does it at start-up.
export PYTHONPATH=$NEWSTACK_OVERLAY/triton_deepbind:${PYTHONPATH:-}
NEWSTACK_PRE_CMD="export NCCL_CROSS_NIC=1 NCCL_NET_GDR_LEVEL=PHB NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 HSA_FORCE_FINE_GRAIN_PCIE=1 FI_MR_CACHE_MONITOR=kdreg2 FI_CXI_DISABLE_HOST_REGISTER=1 FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_RDZV_PROTO=alt_read FI_CXI_RDZV_EAGER_SIZE=0 FI_CXI_RDZV_THRESHOLD=0 FI_CXI_RDZV_GET_MIN=0 FI_CXI_DEFAULT_TX_SIZE=2048 FI_CXI_RX_MATCH_MODE=hybrid; export DARSHAN_PRELOAD='libhsa-runtime64.so libamdhip64.so librccl.so'; export TORCH_EXTENSIONS_DIR=$NEWSTACK_JIT/torch_extensions MEGATRON_FUSED_KERNELS_BUILD_DIR=$NEWSTACK_JIT/megatron_fused TRITON_CACHE_DIR=$NEWSTACK_JIT/triton; find $NEWSTACK_JIT -maxdepth 3 -name lock -mmin +5 -print -delete; md5sum \$PWD/../../megatron/fused_kernels/*_hip.* > \${JOB_DIR}/fused_hip_md5_before.txt 2>/dev/null || true"
# after the training srun: did the run's hipify rewrite the generated *_hip.* files in the source tree? (torch's JIT on ROCm
# writes them next to megatron/fused_kernels/*.cu; production runs use the same files)
NEWSTACK_POST_CMD="md5sum \$PWD/../../megatron/fused_kernels/*_hip.* > \${JOB_DIR}/fused_hip_md5_after.txt 2>/dev/null; diff \${JOB_DIR}/fused_hip_md5_before.txt \${JOB_DIR}/fused_hip_md5_after.txt > \${JOB_DIR}/fused_hip_md5_diff.txt && echo 'hipified sources unchanged' || echo 'WARNING: hipified sources in the tree CHANGED (see fused_hip_md5_diff.txt)'"
# Flash attention: the CK build installed in the venv (default), or with NEWSTACK_FA=triton the Triton backend of the same v2.8.3
# (overlay/fa_triton_amd, no compiled code; a different kernel from production's CK build, so its TFLOP/s is not comparable with B0).
# [11-04 03:0x, the coordinator] a private copy of megatron/ (tree/megatron, rsync of the working tree's megatron/ without its
# build folders, caches and *_hip.* files) comes first on PYTHONPATH: torch 2.13's hipify writes scaled_masked_softmax_hip.h and
# scaled_upper_triang_masked_softmax_hip.h differently from torch 2.9.1's (checked offline), and it writes them next to the
# sources, so a run importing the shared tree would rewrite two files that production and the other session's jobs build from.
# (ft-sys-v2: runs/ftv2_run_common.sh puts FTV2_TREE before it, so an ft-sys-v2 run imports megatron from FTV2_TREE.)
export PYTHONPATH=$NEWSTACK_MEGATRON_COPY:${PYTHONPATH:-}
if [ "${NEWSTACK_FA:-ck}" = triton ]; then
  export PYTHONPATH=$NEWSTACK_OVERLAY/fa_triton_amd:${PYTHONPATH:-}
  export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
fi
