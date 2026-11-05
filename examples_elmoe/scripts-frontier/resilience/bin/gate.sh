#!/bin/bash
# gate.sh: the pre-run gate (design G2/G4/G5/G6 and the five rules of 3.11, plus the rules of 2026-10-09 evening), run on the
# batch node of a Slurm job before the first training attempt.
#   bash gate.sh --mode quick|full --nodes "n1 n2 ..." --job-dir JOB_DIR [--dry-run-cmd "cmd"] [--baseline FILE] [--python PY]
#                [--calibrate] [--spares S]
#
# WHAT THIS SCRIPT DOES, IN ORDER
#   1. the dry run (G2) of the training command on the batch node, in the background (no GPU): the arguments and the paths;
#   2. the node step: ONE srun step, one task per node, running gate_node.sh, which measures the per-node tests (rocm-smi, the
#      versions, /dev/shm, dmesg; in mode full also the per-GCD GEMM and the in-node xGMI all-to-all);
#   3. the re-probe: a node that wrote no row is probed once with `srun hostname` (silence alone is never a node's fault);
#   4. the network step (mode full, 2 nodes or more): ONE srun step with 8 tasks per node running gate_net.sh -> gate_net.py: the
#      pair test (a two-rank all-reduce in two rounds of disjoint pairs, all 8 GCDs, on the training's own network path) and then
#      the switch test (one all-to-all over the survivors with the NIC counters); the optional RDMA phase only with GATE_RDMA_PHASE=1;
#      then gate_net.py --summarize turns the per-rank jsons into <node>.net.tsv rows;
#   5. the judge (gate_judge.py): one verdict per node per test, the naming rules of the pair test, the majority rule, the two
#      tiers (degraded / dead), gate.log, gate_report.tsv, failed_nodes.txt and excluded_reasons.tsv.
# Writes (design 3.11): JOB_DIR/gate.log (the summary table first, then per test the command, the budget, the worst node and
#   every node), JOB_DIR/data/gate_report.tsv (one verdict per node per test), JOB_DIR/data/gate/{<node>.raw.tsv, <node>.probe.tsv,
#   <node>.net.tsv, raw outputs, failed_nodes.txt (the nodes to exclude), excluded_reasons.tsv (one sentence per excluded node with
#   its tier), dryrun.log, srun.log, net/ (the network step's files)}.
# The budgets: each test's budget = 2 x its calibrated run time + 30 s (gate_judge.py budget_of; the pair test adds 180 s for its
#   deadlines and done-file waits); the node step's timeout = the sum of the per-node budgets + 60 s; the network step's timeout
#   = the sum of the network budgets + 60 s.
# Exit: 0; 2 when the dry run failed; 4 when a test is a TEST DEFECT (the caller stops only with AUTORUN_GATE_STRICT=1).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE=quick; NODES=""; JD=""; DRY=""; BASE="$HERE/../gate_baseline.tsv"; CAL=0; SPARES=${GATE_SPARES:-0}
PY=/lustre/orion/world-shared/gen150/zixianw4/envs/TORCH-ROCM6.4.1_env_groupgemm_v5/bin/python
while [ $# -gt 0 ]; do case $1 in
  --mode) MODE=$2; shift 2;; --nodes) NODES=$2; shift 2;; --job-dir|--out) JD=$2; shift 2;;
  --dry-run-cmd) DRY=$2; shift 2;; --baseline) BASE=$2; shift 2;; --python) PY=$2; shift 2;; --calibrate) CAL=1; shift;;
  --spares) SPARES=$2; shift 2;;
  *) echo "gate.sh: unknown argument $1" >&2; shift;; esac; done
[ -n "$JD" ] || { echo "gate.sh: --job-dir is required" >&2; exit 0; }
[ "$MODE" == full ] || MODE=quick
G=$JD/data/gate; mkdir -p "$G"; rm -f "$G"/*.raw.tsv "$G"/*.probe.tsv
LOOPLOG=${RES_LOOP_LOG:-$JD/data/loop.log}
say() { echo "$(date '+%F %H:%M:%S') gate: $*" | tee -a "$LOOPLOG"; }
T0=$(date +%s); NL=($NODES); K=${#NL[@]}; BATCH=$(hostname -s)
TESTS_Q="rocm_smi_ids ras_ue xgmi_err rocm_version rccl_md5 ofi_plugin_md5 libfabric shm_free_gb counters_after dmesg"
TESTS="$TESTS_Q"; [ $MODE == full ] && TESTS="$TESTS gemm xgmi_a2a"
# the cross-node tests of mode full (gate_net.py): ONE extra srun step over the K nodes after the per-node step, 8 tasks per node;
# their rows (<node>.net.tsv) are judged with the node rows. They need K >= 2. The rdma phase runs only with GATE_RDMA_PHASE=1.
NET_TESTS=""; RDMA=${GATE_RDMA_PHASE:-0}
[ $MODE == full ] && { NET_TESTS="pair switch"; [ "$RDMA" = 1 ] && NET_TESTS="$NET_TESTS rdma"; }
# the budgets from the calibration file (or the defaults); exported to gate_node.sh as GATE_BUDGET_<test>
BUD=$(cd "$HERE" && "$PY" -c "
import sys; sys.argv=['x']; import gate_judge as g
cal = g.read_cal('$BASE')
print(','.join(f'{t}={g.budget_of(t, cal)}' for t in g.TESTS))")
[ -n "$BUD" ] || BUD="rocm_smi_ids=30,ras_ue=30,xgmi_err=30,gemm=300,xgmi_a2a=120,answers_srun=60,pair=300,rdma=120,switch=120"
SUM=0; NSUM=0
for kv in ${BUD//,/ }; do t=${kv%%=*}; b=${kv#*=}; export GATE_BUDGET_$t=$b
  case " $TESTS " in *" $t "*) SUM=$((SUM + b));; esac
  case " $NET_TESTS " in *" $t "*) NSUM=$((NSUM + b));; esac
done
NODE_TO=$((SUM + 60)); NET_TO=$((NSUM + 60))
export GATE_PYTHON=$PY
say "the gate starts in mode $MODE on $K nodes (calibration $( [ $CAL = 1 ] && echo on || echo off); $SPARES spare node(s) in the job); the per-test budgets in seconds are $BUD; the node step's timeout is $NODE_TO s$( [ -n "$NET_TESTS" ] && echo "; the network step's timeout is $NET_TO s (tests: $NET_TESTS)" ); the calibration file is $BASE"
[ -n "${XMOE_GATE_INJECT:-}" ] && say "TEST ONLY: the gate's fault injection XMOE_GATE_INJECT=$XMOE_GATE_INJECT is set (gate_net.py pair_silent; gate_gemm.py gemm_slow, gemm_half)"

# 1. G2 the dry run, in the background on the batch node
DRYPID=""
if [ -n "$DRY" ]; then
  ( timeout -k 10 300 bash -c "$DRY" > "$G/dryrun.log" 2>&1; echo $? > "$G/dryrun.rc" ) &
  DRYPID=$!
  say "the dry run of the training command started on the batch node $BATCH (timeout 300 s); its output goes to data/gate/dryrun.log"
fi

# 2. G4/G5/G6 the node tests: one srun step, one task per node; the node script measures, gate_judge.py judges
if [ $K -gt 0 ]; then
  ts=$(date +%s)
  timeout -k 10 $NODE_TO srun -N $K -n $K --ntasks-per-node=1 -w "$(echo ${NL[*]} | tr ' ' ',')" --overlap --kill-on-bad-exit=0 \
    ${GATE_SRUN_EXTRA:-} bash "$HERE/gate_node.sh" $MODE "$G" > "$G/srun.log" 2>&1
  rc=$?; [ $rc -ne 0 ] && say "the node step ended with exit code $rc after $(( $(date +%s) - ts )) s (see data/gate/srun.log)"
fi
# 3. answers_srun: a node that wrote no row is re-probed once with srun hostname (silence alone is never a node's fault);
#    in calibration mode every node is probed, so that the probe's own run time is measured
PROBE=(); for n in "${NL[@]}"; do { [ $CAL = 1 ] || [ ! -s "$G/$n.raw.tsv" ]; } && PROBE+=("$n"); done
for n in "${NL[@]}"; do [ -s "$G/$n.raw.tsv" ] && [ $CAL = 0 ] && printf '%s\t%s\tanswers_srun\tyes\t0\tok\tthe node wrote its gate rows\n' "$(date '+%F %H:%M:%S')" "$n" > "$G/$n.probe.tsv"; done
if [ ${#PROBE[@]} -gt 0 ]; then
  pb=$GATE_BUDGET_answers_srun; ts=$(date +%s.%N)
  out=$(timeout -k 5 ${pb:-60} srun -N ${#PROBE[@]} -n ${#PROBE[@]} --ntasks-per-node=1 -w "$(echo ${PROBE[*]} | tr ' ' ',')" --overlap \
        --kill-on-bad-exit=0 hostname -s 2>&1); prc=$?
  el=$(awk -v a=$ts -v b=$(date +%s.%N) 'BEGIN{printf "%.1f", b-a}')
  for n in "${PROBE[@]}"; do
    if echo "$out" | grep -q -x "$n"; then st=ok; v=yes; d="srun hostname answered in $el s"
    elif [ $prc -ge 124 ]; then st=ok; v=no; d="srun hostname did not answer within ${pb:-60} s (re-probe after a silent gate step)"
    else st=ok; v=no; d="srun hostname failed (exit code $prc): $(echo "$out" | grep -a -m1 -i error | head -c 120)"; fi
    printf '%s\t%s\tanswers_srun\t%s\t%s\t%s\t%s\n' "$(date '+%F %H:%M:%S')" "$n" "$v" "$el" "$st" "$d" > "$G/$n.probe.tsv"
  done
  say "the re-probe with srun hostname of ${#PROBE[@]} node(s) returned exit code $prc after $el s"
fi

# 4. the network step (mode full, K >= 2): gate_net.sh per task runs gate_net.py in phase net (the pair test, then the switch test,
#    on the training's network path) and, only with GATE_RDMA_PHASE=1, in phase gdr (the rdma test); then gate_net.py --summarize
#    on this node writes <node>.net.tsv. Nodes that fail the per-node tests still take part: the judge decides. With
#    --gpus-per-node=8 every task sees all 8 GCDs and picks its own (lessons.md 2026-10-09 trap 3).
rm -f "$G"/*.net.tsv
if [ -n "$NET_TESTS" ] && [ $K -ge 2 ]; then
  mkdir -p "$G/net"; rm -rf "$G"/net/*
  # THE SCALE GUARD (2026-10-10 08:30, job 5646880 at 260 nodes): the switch test's survivor wait (a 5-s and a 20-s file wait on Lustre)
  # did not scale to 2,080 ranks (20 nodes "dead" for a missing survivors list, 220 nodes "not tested" for ranks the wait had missed) and
  # its 3-node baseline made every measured node "degraded" (0.47 GB/s against a threshold of 9.0): 66 exclusions, the run declared
  # doomed. Until the waits scale with the world size and the baseline is calibrated at the run's own scale, the switch test runs only
  # up to GATE_NET_MAX_NODES nodes (default 32); the pair test (two-rank groups, no global wait) ran cleanly at 260 nodes and stays.
  if [ $K -gt ${GATE_NET_MAX_NODES:-32} ]; then
    export GATE_SKIP_SWITCH=1
    say "the switch test is skipped at $K nodes (GATE_NET_MAX_NODES=${GATE_NET_MAX_NODES:-32}): its survivor wait and its baseline were validated up to 6 nodes only; at 260 nodes (job 5646880) the wait missed ranks and the 3-node baseline failed every node; the pair test runs"
  fi
  NB_NET=$(( ${GATE_BUDGET_pair:-300} + ${GATE_BUDGET_switch:-120} )); NB_GDR=${GATE_BUDGET_rdma:-120}
  ts=$(date +%s)
  # [2026-10-11, sandbox] the launch goes through net_launch_retry.sh: when Slurm's controller does not start the step (job 5650829:
  # "Unable to confirm allocation ... Socket timed out" after the 35-s MessageTimeout) and no rank wrote a file, it waits
  # GATE_NET_LAUNCH_WAIT_S (30) and launches again, GATE_NET_LAUNCH_RETRIES (2) times; then net/not_launched -> "NOT LAUNCHED" in the judge
  bash "$HERE/net_launch_retry.sh" "$G/net/srun.log" "$G/net" $(( ${GATE_NET_LAUNCH_RETRIES:-2} + 1 )) ${GATE_NET_LAUNCH_WAIT_S:-30} -- \
    timeout -k 10 $NET_TO srun -N $K -n $((K * 8)) --ntasks-per-node=8 --gpus-per-node=8 -c6 -w "$(echo ${NL[*]} | tr ' ' ',')" --overlap \
    --kill-on-bad-exit=0 bash "$HERE/gate_net.sh" "$G/net" "${NL[0]}" "${NL[*]}" $NB_NET $NB_GDR > "$G/net/launch_tries.txt"
  rc=$?
  while IFS= read -r l; do [ -n "$l" ] && say "the network step's launch: $l"; done < "$G/net/launch_tries.txt"
  "$PY" "$HERE/gate_net.py" --summarize --net-dir "$G/net" --nodes "${NL[*]}" --out "$G" \
    --budgets "pair=${GATE_BUDGET_pair:-300},rdma=${GATE_BUDGET_rdma:-120},switch=${GATE_BUDGET_switch:-120}" > "$G/net/summarize.out" 2>&1
  src=$?
  say "the network step ($NET_TESTS on $K nodes x 8 GCDs) ended: the srun returned exit code $rc and the summary exit code $src after $(( $(date +%s) - ts )) s (the files are in data/gate/net/ and data/gate/<node>.net.tsv)"
elif [ -n "$NET_TESTS" ]; then
  say "the network step was skipped: the cross-node tests need 2 nodes or more and $K was given"
fi

DRYRC=""
if [ -n "$DRYPID" ]; then
  wait $DRYPID; DRYRC=$(cat "$G/dryrun.rc" 2>/dev/null || echo 125)
  [ "$DRYRC" == 0 ] && say "the dry run passed (exit code 0)" || say "!!!!! THE DRY RUN FAILED with exit code $DRYRC ($( [ "$DRYRC" == 124 ] && echo 'it timed out' || echo 'an error')); read data/gate/dryrun.log !!!!!"
fi
# 5. the judge
"$PY" "$HERE/gate_judge.py" --raw "$G" --nodes "${NL[*]}" --mode $MODE --baseline "$BASE" --report "$JD/data/gate_report.tsv" \
  --failed "$G/failed_nodes.txt" --log "$JD/gate.log" --job "${SLURM_JOB_ID:-nojob}" --dry-rc "$DRYRC" --dry-log data/gate/dryrun.log \
  --t0 $T0 --budgets "$BUD" --spares "$SPARES" $( [ $CAL = 1 ] && echo --calibrate ) $( [ "$RDMA" = 1 ] && echo --rdma ) \
  $( [ "${GATE_EXCLUDE_SUSPECT_PAIR:-0}" = 1 ] && echo --exclude-suspect-pair ) > "$G/judge.out" 2>&1
JRC=$?
[ $JRC -ne 0 ] && [ $JRC -ne 4 ] && say "gate_judge.py failed with exit code $JRC: $(tail -3 "$G/judge.out" | tr '\n' ' ' | head -c 300)"
NF=$(wc -l < "$G/failed_nodes.txt" 2>/dev/null || echo 0)
NDEG=$(awk -F'\t' 'NR>1 && $2=="degraded"' "$G/excluded_reasons.tsv" 2>/dev/null | wc -l)
say "the gate ended after $(( $(date +%s) - T0 )) s on $K nodes: $NF node(s) to exclude ($( [ "$NF" -gt 0 ] && paste -sd' ' "$G/failed_nodes.txt" || echo none); $NDEG of them degraded, usable as spares of last resort); $( [ $JRC = 4 ] && echo 'there are TEST DEFECT(S), see gate.log; ' )the summary is in gate.log"
[ -n "$DRY" ] && [ "$DRYRC" != 0 ] && exit 2
[ $JRC = 4 ] && exit 4
exit 0
