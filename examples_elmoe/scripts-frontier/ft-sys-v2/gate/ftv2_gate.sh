#!/bin/bash
# ftv2_gate.sh (ft-sys-v2, 2026-11-05): the pre-run gate of ft-sys-v2, sourced by heartbeat-sys/hb_start.sh before etcd and the
# agents start, when FTV2_GATE is set (the user, 11-05 13:5x: "implement the pre-run gating checks please, I think we can reuse
# ft-sys-v1"). It runs ft-sys-v1's gate unchanged in its tests and rules (gate.sh, the copies in this folder; only the stack paths
# are settings) on every node of the allocation, then turns its verdicts into the layout to train on:
#   dead nodes      excluded; a standby node that passed takes a dead training node's place (ft-sys-v1's two tiers of exclusion)
#   degraded nodes  slow but working: kept, and listed (v1: usable as spares of last resort)
# It writes $FTV2_OUT/ft/gate/ (gate.log, data/gate_report.tsv, data/gate/failed_nodes.txt, excluded_reasons.tsv, raw rows), one
# GATE line in $FTV2_OUT/ft/record.log, and exports FTV2_GATE_EXCLUDED, FTV2_GATE_DEGRADED, FTV2_GATE_TRAIN, FTV2_GATE_STANDBY.
# hb_start.sh puts the same result in etcd (key gate/result) once etcd is up.
# Until ft-sys-v2 has its own sbatch, training runs through the 0623 launcher, whose srun uses every node of the allocation, so a
# layout without a dead node cannot be applied yet: with FTV2_GATE_STRICT=1 (the default) a dead training node ends the job before
# training starts, with the reason in the record.
# Settings: FTV2_GATE=quick|full (full adds the GEMM per GCD, the xGMI all-to-all, and the pair and switch tests across nodes);
# FTV2_GATE_CALIBRATE=1 (measure every test on these nodes and write the baseline; only on nodes known to be good);
# FTV2_GATE_BASELINE (default gate_baseline_v2.tsv here); FTV2_GATE_STRICT (1); FTV2_STANDBY=1 (the last node is the standby);
# GATE_OFI_DIR (the network plugin; default NEWSTACK_PLUGIN when the new-stack env set it). Needs FTV2_OUT, FTV2_PY.
_g_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_g_nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
_g_last=$(( ${#_g_nodes[@]} - 1 ))
_g_sb=(); [ "${FTV2_STANDBY:-0}" = 1 ] && _g_sb=("${_g_nodes[$_g_last]}")
_g_out=$FTV2_OUT/ft/gate; mkdir -p "$_g_out"
export GATE_PYTHON=${GATE_PYTHON:-$FTV2_PY}
[ -n "${NEWSTACK_PLUGIN:-}" ] && export GATE_OFI_DIR=${GATE_OFI_DIR:-$NEWSTACK_PLUGIN}
_g_base=${FTV2_GATE_BASELINE:-$_g_dir/gate_baseline_v2.tsv}
[ -f "$_g_base" ] || printf 'test\tbaseline\trun_s\tdate\tjob\tnodes\tnote\n' > "$_g_base"   # no line yet: every test is informational
echo "[ftv2 gate] $(date '+%T') mode $FTV2_GATE on ${#_g_nodes[@]} nodes (standby: ${_g_sb[*]:-none}); calibration $( [ "${FTV2_GATE_CALIBRATE:-0}" = 1 ] && echo on || echo off); baseline $_g_base"
RES_LOOP_LOG=$_g_out/gate_steps.log bash "$_g_dir/gate.sh" --mode "$FTV2_GATE" --nodes "${_g_nodes[*]}" --job-dir "$_g_out" \
  --python "$GATE_PYTHON" --baseline "$_g_base" --spares ${#_g_sb[@]} $( [ "${FTV2_GATE_CALIBRATE:-0}" = 1 ] && echo --calibrate )
_g_rc=$?
# the verdicts: dead nodes are excluded, degraded ones are kept (excluded_reasons.tsv: node, tier, reason)
_g_dead=(); _g_deg=()
if [ -f "$_g_out/data/gate/excluded_reasons.tsv" ]; then
  while IFS=$'\t' read -r n tier reason; do
    [ "$n" = node ] && continue
    [ "$tier" = degraded ] && _g_deg+=("$n") || _g_dead+=("$n")
  done < "$_g_out/data/gate/excluded_reasons.tsv"
fi
_g_isdead() { local x; for x in "${_g_dead[@]}"; do [ "$x" = "$1" ] && return 0; done; return 1; }
_g_train=(); _g_free_sb=()
for n in "${_g_sb[@]}"; do _g_isdead "$n" || _g_free_sb+=("$n"); done
_g_lost=0
for n in "${_g_nodes[@]}"; do
  case " ${_g_sb[*]} " in *" $n "*) continue;; esac
  if _g_isdead "$n"; then
    if [ ${#_g_free_sb[@]} -gt 0 ]; then _g_train+=("${_g_free_sb[0]}"); _g_free_sb=("${_g_free_sb[@]:1}"); else _g_lost=$((_g_lost + 1)); fi
  else _g_train+=("$n"); fi
done
export FTV2_GATE_EXCLUDED="${_g_dead[*]}" FTV2_GATE_DEGRADED="${_g_deg[*]}" FTV2_GATE_TRAIN="${_g_train[*]}" FTV2_GATE_STANDBY="${_g_free_sb[*]}"
_g_line="GATE: mode $FTV2_GATE, exit $_g_rc; excluded (dead): ${FTV2_GATE_EXCLUDED:-none}; degraded (kept): ${FTV2_GATE_DEGRADED:-none}; layout: train on ${FTV2_GATE_TRAIN:-none}, standby ${FTV2_GATE_STANDBY:-none}$( [ $_g_lost -gt 0 ] && echo "; $_g_lost training node(s) short: no standby left" ); see ft/gate/gate.log"
echo "$(date '+%F %T') $_g_line" >> "$FTV2_OUT/ft/record.log"
echo "[ftv2 gate] $(date '+%T') $_g_line"
python3 -I -c "import json, sys; json.dump({'rc': $_g_rc, 'excluded': '${FTV2_GATE_EXCLUDED}'.split(), 'degraded': '${FTV2_GATE_DEGRADED}'.split(),
  'train': '${FTV2_GATE_TRAIN}'.split(), 'standby': '${FTV2_GATE_STANDBY}'.split(), 'short': $_g_lost}, open('$_g_out/result.json', 'w'))"
# the 0623 launcher trains on every node of the allocation, so a dead training node cannot be left out yet: stop before training
_g_dead_train=0; for n in "${_g_dead[@]}"; do case " ${_g_sb[*]} " in *" $n "*) ;; *) _g_dead_train=1;; esac; done
if [ $_g_dead_train = 1 ] && [ "${FTV2_GATE_STRICT:-1}" = 1 ]; then
  echo "[ftv2 gate] $(date '+%T') a training node failed the gate and this launcher cannot leave it out: ending the job before training"
  echo "$(date '+%F %T') GATE: the job is ended before training (FTV2_GATE_STRICT=1): a dead training node, and the 0623 launcher trains on every node" >> "$FTV2_OUT/ft/record.log"
  scancel "$SLURM_JOB_ID"
fi
