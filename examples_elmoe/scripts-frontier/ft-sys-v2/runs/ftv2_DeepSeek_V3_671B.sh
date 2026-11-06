#!/bin/bash
# DeepSeek-V3 671B with ft-sys-v2: heartbeat + v1 restart only (PP 32: the pipeline needs milestone 3).
# Layout (as assistant_deep_dive/2026-09-10-deepseek_v3/runs/0913_P64_v3_671b_train_n64_pp32.sh): 64 nodes | EP 8 | PP 32 | TP 1 | DP_non_moe 16 | mbs 1 x 256 x 16 = GBS 4096 | seq 4096 | ckpt.
EX_LABEL=ftv2_DeepSeek_V3_671B EX_MODE=heartbeat EX_EP_KEY="64:8:1" EX_EP_CMD=" 1:256:5:X-MOE:DeepSeek_V3_671B:ckpt:1:1" EX_PP_SIZE=32 EX_SEQ_LEN=4096 EX_QOS=normal EX_WALL=01:00:00 EX_EXIT_MINS=55
. "$(dirname "${BASH_SOURCE[0]}")/ftv2_run_common.sh"
