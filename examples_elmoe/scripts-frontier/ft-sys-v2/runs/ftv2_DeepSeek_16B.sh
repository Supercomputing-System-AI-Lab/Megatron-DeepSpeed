#!/bin/bash
# DeepSeek-MoE-16B with ft-sys-v2 warm recovery (expected to work: the same path as 10B; not run with recovery yet).
# Layout: 3 nodes | EP 8 | PP 1 | TP 1 | DP_non_moe 24 | DP_moe 3 | mbs 1 x 8 x 24 = GBS 192 | seq 4096 | ckpt every layer | ZeRO-1 | + 1 standby.
EX_LABEL=ftv2_DeepSeek_16B EX_MODE=recovery EX_EP_KEY="3:8:1" EX_EP_CMD=" 1:8:40:X-MOE:DeepSeek_16B:ckpt:1:1" EX_SEQ_LEN=4096 EX_STRIDE=8
. "$(dirname "${BASH_SOURCE[0]}")/ftv2_run_common.sh"
