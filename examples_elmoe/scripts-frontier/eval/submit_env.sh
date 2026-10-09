# sourced by autorun_eval.sh before every submission (source it yourself for hand-run sbatch):  . ./submit_env.sh
# Jobs inherit the submitting shell via --export=ALL. Missing PYTHONPATH kills
# jobs at `import megatron`; missing CXX/CC makes torch regenerate build.ninja
# with the system gcc-7 and rebuild fused kernels fatally. Both happened 2026-08-20.
source /usr/share/lmod/lmod/init/bash 2>/dev/null
module load cpe/24.11 PrgEnv-gnu/8.6.0 rocm/6.4.1 cray-mpich/9.1.0 craype-accel-amd-gfx90a miniforge3/23.11.0-0 2>/dev/null
export CUDA_HOME=$ROCM_PATH PYTORCH_ROCM_ARCH=gfx90a HCC_AMDGPU_TARGET=gfx90a
export CXX=$(which g++) CC=$(which gcc)
export PATH=/lustre/orion/world-shared/gen150/zixianw4/envs/TORCH-ROCM6.4.1_env_groupgemm_v5/bin:$PATH
export PYTHONPATH=/lustre/orion/gen150/scratch/zixianw4/X-MoE
export XMOE_KV_PREFILL_CHUNK=4096   # 2026-08-31 user directive: halve 128k-gen prefill chunks again (8192 still OOMed EP-skew spikes); env-tunable, no fp rotation
