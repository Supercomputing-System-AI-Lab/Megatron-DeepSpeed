#!/bin/bash
# X-MoE 63B with ft-sys-v2: heartbeat + v1 restart only (PP 4: the pipeline needs milestone 3).
# Layout (B1 5853628 on the new stack): 4 nodes | PP 4 | EP 8 | TP 1 | DP_non_moe 8 | mbs 1 x 128 x 8 = GBS 1024 | seq 4096 | ckpt.
# The PP path of the launcher: AUTORUN_PP_KEY / AUTORUN_PP_STRATEGY / AUTORUN_PP_CMD.
export AUTORUN_PP_KEY="4:32" AUTORUN_PP_STRATEGY="4:8" AUTORUN_PP_CMD=" 1:128:15:X-MOE:63B:ckpt:1:even:no-planner"
EX_LABEL=ftv2_63B EX_MODE=heartbeat EX_EP_LAYOUT="" EX_EP_CMD="" EX_WALL=00:40:00 EX_EXIT_MINS=36
. "$(dirname "${BASH_SOURCE[0]}")/ftv2_run_common.sh"
