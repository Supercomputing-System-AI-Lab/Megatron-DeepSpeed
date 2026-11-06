#!/bin/bash
# DeepSeek-V3 with 2 layers (1 dense + 1 MoE) with ft-sys-v2 warm recovery (expected to work at PP 1; not run with recovery yet).
# Layout: 3 nodes | EP 8 | PP 1 | TP 1 | DP_non_moe 24 | DP_moe 3 | mbs 1 x 8 x 24 = GBS 192 | seq 4096 | ckpt every layer | ZeRO-1 | + 1 standby.
# (EP 16 over two nodes works the same with EX_STRIDE=16: a rank's partner must hold the same experts, one EP group further.)
EX_LABEL=ftv2_DeepSeek_V3_671B_L2 EX_MODE=recovery EX_EP_KEY="3:8:1" EX_EP_CMD=" 1:8:20:X-MOE:DeepSeek_V3_671B_L2:ckpt:1:1" EX_SEQ_LEN=4096 EX_STRIDE=8
. "$(dirname "${BASH_SOURCE[0]}")/ftv2_run_common.sh"
