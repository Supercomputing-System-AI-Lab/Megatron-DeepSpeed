#!/bin/bash
# X-MoE 10B with ft-sys-v2 warm recovery (tested: assistant_deep_dive/2026-11-05-ftv2_heartbeat, RESULTS_M1_M2.md).
# Layout: 3 nodes | EP 8 | PP 1 | TP 1 | DP_non_moe 24 | DP_moe 3 | mbs 1 x 8 x 24 = GBS 192 | ckpt every layer | ZeRO-1 | + 1 standby.
EX_LABEL=ftv2_10B EX_MODE=recovery EX_EP_LAYOUT="3:8:1" EX_EP_CMD=" 1:8:40:X-MOE:10B:ckpt:1:1" EX_STRIDE=8 EX_WALL=00:30:00
. "$(dirname "${BASH_SOURCE[0]}")/ftv2_run_common.sh"
