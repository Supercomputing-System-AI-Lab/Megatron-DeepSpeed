#!/bin/bash
# ft-sys-v2 copy (2026-11-05) of examples_elmoe/scripts-frontier/resilience/bin/gate_net.sh (ft-sys-v1, commit 7fb97edf). Changed: only the software-stack paths, now settings with the new stack (ROCm 7.1.1, torch 2.13 venv, aws-ofi-nccl 1.22.0) as the default (GATE_PYTHON, GATE_OFI_DIR, ROCM_PATH); the tests, the rules and the judge are v1's.
# gate_net.sh <net dir> <master node> "<nodes in Slurm order>" <net-phase budget s> <gdr-phase budget s>
#
# WHAT THIS SCRIPT IS: the per-task wrapper of the gate's network step. gate.sh launches ONE srun step over the gate nodes with
# 8 tasks per node (one per GCD); every task runs this script, and this script runs gate_net.py (the pair test, then the switch
# test) once, in phase 'net'. The measurements go to <net dir>/r<rank>.net.{json,out,rc} and the RCCL log to rccl_net.<node>.<pid>.
#
# THE NETWORK PATH (the user's decision, 2026-10-09 evening): the pair test must run on the SAME path as the training run, so
# NCCL_NET_GDR_LEVEL is NOT set here: RCCL reads whatever the job exported (the template leaves it unset today, which is RCCL's
# default) and the detail of the pair row says which path the "via NET" lines show (/GDRDMA = GPU-direct, else host-buffer).
# The old separate GPU-direct phase ('gdr', NCCL_NET_GDR_LEVEL=PHB, the ring transfer of the rdma test) is kept reachable behind
# GATE_RDMA_PHASE=1 and is off by default, because it only added time once the pair test runs on the training path.
#
# THE STORE SERVER (found in job 5646299, 2026-10-09 23:21): torch's default rendezvous hosts the TCP store in RANK 0's process, so
# when rank 0 was the partner of the silent rank and left after its 30-s deadline (os._exit, by design), every other rank's next
# store operation failed with "Broken pipe" and the whole step collapsed after round 1. Now the store server runs in ITS OWN
# process on the master node (started here by the first task of that node, before gate_net.py), and every rank is only a client
# (TORCHELASTIC_USE_AGENT_STORE=True makes torch's _create_c10d_store skip the server role), so any rank may leave at any time.
# Each phase runs under `timeout` with its budget (gate.sh: 2 x the calibrated run time + 30 s per test, plus the pair test's
# 180 s for its deadlines and done-file waits; the defaults before calibration are 300 s for pair and 120 s for switch and rdma).
# MASTER_ADDR is the first node of THIS step (lessons.md 2026-10-09 trap 2), resolved by getent (the hostname resolves to the
# hsn0 address on Frontier). The plugin directory goes first on LD_LIBRARY_PATH, as gate_node.sh does for alltoall_perf.
# Exit 0 always: the rows judge, not the exit code.
NET=$1; MASTER=$2; NODES=$3; TN=${4:-420}; TG=${5:-120}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=${SLURM_PROCID:-0}; N=$(hostname -s 2>/dev/null || hostname)
PY=${GATE_PYTHON:-/lustre/orion/world-shared/gen150/zixianw4/envs/TORCH2.13-ROCM7.14-XMOE_venv/bin/python}
OFI_DIR=${GATE_OFI_DIR:-/lustre/orion/gen150/scratch/zixianw4/aws-ofi-nccl-v1.22.0/install/lib}
export LD_LIBRARY_PATH="$OFI_DIR:${LD_LIBRARY_PATH:-}"
MA=$(getent hosts "$MASTER" 2>/dev/null | awk '{print $1; exit}'); export MASTER_ADDR=${MA:-$MASTER}
# the libfabric and RCCL knobs of the training run (the template's values), only where the job did not export them already
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0} FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR:-kdreg2}
export FI_CXI_DEFAULT_CQ_SIZE=${FI_CXI_DEFAULT_CQ_SIZE:-524288} FI_CXI_DEFAULT_TX_SIZE=${FI_CXI_DEFAULT_TX_SIZE:-16384} FI_CXI_RX_MATCH_MODE=${FI_CXI_RX_MATCH_MODE:-hybrid}
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,COLL TORCH_NCCL_BLOCKING_WAIT=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=1   # COLL added 2026-10-10: each collective's line, so a stalled all-to-all is placed (job 5646502)
unset FI_CXI_RDZV_THRESHOLD      # never raised above the default for RCCL traffic (lessons.md 2026-10-09 C4)
# THE "Broken pipe" LINES (2,436 of them in job 5646336's r*.net.out): torch's NCCL watchdog polls a "should dump" flag on the TCP store
# once a second for its flight-recorder dump; once the store server was gone (killed with rank 0's task, see phase()), every poll
# wrote "Failed to check the 'should dump' flag on TCPStore ... Broken pipe" with a C++ stack (2 lines per poll, 60 per rank in the
# 30 s before the rank left; more on the ranks that lived longer). The gate's ranks never use that dump (they leave with os._exit and
# their json is the result), so the poll is switched off here; with the store alive for the whole phase the lines would not appear anyway.
export TORCH_NCCL_DUMP_ON_TIMEOUT=0
mkdir -p "$NET/done"

# the store server of a phase: one process on the master node, started by that node's first task; it lives for the budget + 60 s
store_server() {   # <phase> <port> <seconds>
  local ph=$1 port=$2 secs=$3
  "$PY" -c "
import sys, time, torch.distributed as d
s = d.TCPStore(sys.argv[1], int(sys.argv[2]), is_master=True, wait_for_workers=False)
print('store server for phase', sys.argv[4], 'up on', sys.argv[1], sys.argv[2], 'for', sys.argv[3], 's', flush=True)
time.sleep(float(sys.argv[3]))" "$MASTER_ADDR" "$port" "$secs" "$ph" > "$NET/store_$ph.log" 2>&1 &
  echo $!
}
# phase <name> <budget s> <port> <gdr level or ->: runs gate_net.py once; '-' leaves NCCL_NET_GDR_LEVEL as the job has it
phase() {
  local ph=$1 b=$2 port=$3 gdr=$4 spid=""
  echo "GATE_NET node=$N rank=$R localid=${SLURM_LOCALID:-0} phase=$ph master=$MASTER_ADDR port=$port gdr=$gdr budget=$b start=$(date '+%F %H:%M:%S') NCCL_NET_GDR_LEVEL_in_env=${NCCL_NET_GDR_LEVEL:-unset}" > "$NET/r$R.$ph.out"
  if [ "$N" = "$MASTER" ] && [ "${SLURM_LOCALID:-0}" = 0 ]; then spid=$(store_server "$ph" "$port" $((b + 60))); echo "store server pid $spid on this task" >> "$NET/r$R.$ph.out"; fi
  (
    [ "$gdr" != - ] && export NCCL_NET_GDR_LEVEL=$gdr
    export TORCHELASTIC_USE_AGENT_STORE=True       # every rank is a store CLIENT; the server is the separate process above
    MASTER_PORT=$port NCCL_DEBUG_FILE="$NET/rccl_$ph.%h.%p" \
      timeout -k 5 "$b" "$PY" "$HERE/gate_net.py" --phase "$ph" --net-dir "$NET" --nodes "$NODES" --init-timeout $(( b > 150 ? 120 : b - 20 )) >> "$NET/r$R.$ph.out" 2>&1
    echo $? > "$NET/r$R.$ph.rc"
  )
  if [ -n "$spid" ]; then
    # THE STORE SERVER OUTLIVES THIS TASK'S OWN TEST (found in job 5646336, 2026-10-09 23:44): this task (rank 0, GCD 0 of the master
    # node) was the injected silent rank's partner in round 1, timed out after 30 s and left, and the old line here killed the store
    # server at once; every other rank's next store operation then failed ("retrieving ncclUniqueId ... Broken pipe"), every rank
    # left with exit 1, the switch test never ran and the silent node was named once only. The server now stays up until every task
    # of the step has written its exit-code file (or the phase budget + 30 s passed), and the line below says what it waited for.
    local t0 n ntask; t0=$(date +%s); n=0; ntask=${SLURM_NTASKS:-1}
    while [ $(( $(date +%s) - t0 )) -lt $(( b + 30 )) ]; do
      n=$(ls "$NET"/r*."$ph".rc 2>/dev/null | wc -l); [ "$n" -ge "$ntask" ] && break; sleep 2
    done
    echo "store server for phase $ph: kept alive until $n of $ntask tasks had written their exit code ($(( $(date +%s) - t0 )) s after this task's own test ended, $(date '+%F %H:%M:%S')); stopping it now" >> "$NET/r$R.$ph.out"
    kill "$spid" 2>/dev/null; wait "$spid" 2>/dev/null
  fi
}
# the pair test and the switch test, on the training's network path
phase net "$TN" 29411 -
# the optional RDMA phase (the old ring transfer with GPU-direct requested), only on request
[ "${GATE_RDMA_PHASE:-0}" = 1 ] && phase gdr "$TG" 29412 PHB
exit 0
