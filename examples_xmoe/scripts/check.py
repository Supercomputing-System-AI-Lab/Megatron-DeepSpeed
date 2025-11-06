# check_env.py
import os
import torch

# 打印关键的分布式环境变量
local_rank = os.environ.get("LOCAL_RANK", "N/A")
global_rank = os.environ.get("RANK", "N/A")
world_size = os.environ.get("WORLD_SIZE", "N/A")
master_addr = os.environ.get("MASTER_ADDR", "N/A")
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "N/A")
device_count = torch.cuda.device_count()

print(
    f"Global Rank: {global_rank}/{world_size}, "
    f"Local Rank: {local_rank}, "
    f"Master: {master_addr}, "
    f"CUDA_VISIBLE_DEVICES: '{cuda_visible}', "
    f"torch.cuda.device_count(): {device_count}"
)