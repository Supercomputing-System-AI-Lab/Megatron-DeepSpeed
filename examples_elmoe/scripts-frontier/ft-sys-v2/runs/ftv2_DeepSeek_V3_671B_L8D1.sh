#!/bin/bash
# DeepSeek-V3 with 8 layers (1 dense + 7 MoE) with ft-sys-v2 warm recovery at PP 1 (expected to work; not run with recovery yet).
# Layout: 16 nodes | EP 32 (an EP group spans 4 nodes) | PP 1 | TP 1 | DP_non_moe 128 | DP_moe 4 | mbs 1 x 2 x 128 = GBS 256 | seq 4096 |
# ckpt every layer | ZeRO-1 | + 1 standby. Partner stride 32: rank r's copy is on r + 32, which holds the same experts, 4 nodes on.
EX_LABEL=ftv2_DeepSeek_V3_671B_L8D1 EX_MODE=recovery EX_EP_LAYOUT="16:32:1" EX_EP_CMD=" 1:2:20:X-MOE:DeepSeek_V3_671B_L8D1:ckpt:1:1" EX_SEQ_LEN=4096 EX_STRIDE=32 EX_WALL=00:40:00 EX_EXIT_MINS=35
. "$(dirname "${BASH_SOURCE[0]}")/ftv2_run_common.sh"
