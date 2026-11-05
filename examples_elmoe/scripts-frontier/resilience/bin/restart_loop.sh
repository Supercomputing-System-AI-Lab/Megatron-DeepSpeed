#!/bin/bash
# restart_loop.sh: the resilient job script's body (design 1009 sections 3.2, 3.6, 3.9, 3.11, 5.1), sourced by
# frontier_elmoe.slurm_resilient.template when AUTORUN_RESILIENT=1. It needs from the template: NODES (training nodes N),
# TOTAL_GPUS, gpus_per_node, JOB_DIR, SLURM_JOB_ID, RUN_TYPE, PP_SIZE, EP_PARALLEL_SIZE, and the functions
#   xmoe_train_srun <comma node list> <node count>   (the training srun; writes ${RANK_LOG_DIR}/rank_<r>.log)
#   xmoe_dry_run_cmd                                  (the G2 dry-run command line)
#   merge_full_run <rank dir> <merged log>            (the template's merge of the .o and the rank-0/last-rank logs)
#
# WHAT THE LOOP DOES, IN PLAIN WORDS
#   1. It reads the allocation (N training nodes + the spares), removes the lemons of the shared failure record, and runs the gate
#      (gate.sh) on every node. A node the gate excludes goes into the failure record with the class gate_<test> (code 30) and the
#      judge's sentence as the evidence; the record decides nothing by itself, the lemon list reads it (two soft lines in 30 days).
#      The gate's exclusions have TWO TIERS (2026-10-09 late evening): a node that only failed a PERFORMANCE test (a slow GCD, a slow
#      card) is DEGRADED and kept in RES_DEGRADED as a spare of last resort; a node that failed for an ERROR (no answer, memory or
#      xGMI errors, link errors, a version mismatch, no GCDs, a timeout under the majority rule) is DEAD and never used.
#   2. Before every attempt it chooses the N training nodes = the first N good nodes in allocation order; the rest are spares and
#      the first spare holds the duty (the heartbeat aggregator). A node that ENTERS the training set (a spare after an exclusion,
#      a re-admitted node) first passes the SPARE CHECK: the quick tests of the gate on that node alone (gate_node.sh quick through
#      srun --overlap, judged by gate_judge.py on the one node, a few seconds); a spare that fails is excluded and the next one is
#      tried. When no healthy spare is left and AUTORUN_USE_DEGRADED=1 (the default), the best degraded node (the highest value
#      against its threshold) is put into training and recovery.log and the banner say so; with AUTORUN_USE_DEGRADED=0 the loop
#      stops as before.
#   3. It runs the training srun in the background and watches it: the first iteration line (D8, the init deadline), the
#      aggregator's liveness, the injection markers. When the srun ends it classifies the failure (the detectors' exit record, the
#      NCCL watchdog, Slurm's node failure, srun's task line), names the node (the heartbeat minority or stillness rule, the
#      watchdog group, the flight recorder, srun, sinfo), appends the failure record, excludes the node when its class says so,
#      counts the retries of the class, and restarts from the latest save inside the same allocation. The RECOVERED banner is
#      written when the next attempt shows its first iteration line; its step is the step at which the failure fired (the exit
#      record's step, else the injection marker's, else the heartbeat table's, which can lag by one beat).
#   [2026-10-11, the user's rules] (a) no restart starts when the time left cannot cover this job's own measured start-up plus one step
#   (res_need_attempt_s), not a fixed 7 minutes; (b) host memory above 97 percent is retried once (exit_codes.py); (c) with AUTORUN_SHRINK=1,
#   when no spare node is left the memory model picks the largest configuration that fits on the healthy nodes (oom_reconfig.sh).
#   THE WORDING RULE (the user, 2026-10-09 evening): every line of the human-facing files (loop.log, recovery.log, training.log's
#   headers, the banner) is a full sentence that says what happened, where, when and what it means; no bare abbreviations.
#   4. THE UCP WORKER DUTY (the user, 2026-10-10 01:15; flag AUTORUN_UCP_WORKERS=<n>, default 0 = nothing changes): the allocation
#      carries n extra nodes, taken from its END (after the spares), that never train and are never swapped in; they convert every
#      complete save to the universal form so that the next link can resume under another layout without a separate conversion job.
#      A watcher (res_ucp_watch, a background loop of this script) polls 'latest' every 15 s; when a save is complete that has no
#      universal, it runs ds_to_universal on the n workers at once (--param-shard k/n, byte-balanced across the shares, --fragments-inplace
#      --optstate-from-pickle, 64 extract and 64 merge workers; the last share to finish finalizes), one conversion at a time: a save
#      that completes while a conversion runs is skipped with a sentence and caught up later (the newest unconverted save first). A failed
#      share is retried once on another worker; if it fails again the save is left unconverted and said so. A universal is complete when
#      zero/optimizer_state.pt exists, no tmp folder is left, the atom count is above zero and the finalize lock exists; then
#      'latest_universal' is advanced. The retention (res_retention_ucp) keeps the newest 2 ZeRO saves and the newest 8 universals plus
#      every step listed in the keep-file (RES_KEEP_FILE, default <save dir>/keep_steps.txt, one step per line) and deletes an older ZeRO
#      save only when its universal is complete. STATUS.md shows one line ("conversion: step N universal complete HH:MM, 2 workers, M min;
#      next: step N+k") from data/ucp_state; recovery.log and loop.log carry full sentences; data/ucp_conversions.tsv is the record.
#
# THE JOB FOLDER (design 3.11, 2026-10-09; the user's amendments of 13:55): the top level is for people, attempts/ per run, data/
# for scripts.
#   README.md  STATUS.md  training.log  recovery.log  alarms.log  heartbeat.log  gate.log  heatmap.png  <job>.o <job>.e  a-monitor.txt  RESULTS.md  dashboard.html
#   alarms/<node>_<HHMMSS>.txt (the stack samples; the user, 2026-10-09 late evening: the alarms are visible at the top level)
#   attempts/attempt_<k>/{full_run.log, ranks/rank_<r>.log, nccl_trace/, exit/}     attempt 0 = the first run; k = after the k-th failure
#   data/{attempts.tsv failures.tsv gate_report.tsv failure_record.tsv results.tsv state loop.log, hb/, alarm/ (the ALARM marker),
#         monitor/, darshan/, inject/, gate/}
# The rank logs are written DIRECTLY into the current attempt's ranks/ (RANK_LOG_DIR, exported per attempt), never moved afterwards.
RES_BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES_PY="${RES_PY:-/lustre/orion/world-shared/gen150/zixianw4/envs/TORCH-ROCM6.4.1_env_groupgemm_v5/bin/python}"

res_say() { echo "$(date '+%F %H:%M:%S') $*" | tee -a "$RES_LOOP_LOG"; }
res_loud() { local m="$*"; res_say "##### $m #####"; echo "##### $m #####" >&2; }
res_rec() { echo "$*" >> "$RES_JOB_DIR/recovery.log"; }          # recovery.log: prose, one block per failure
res_tl() { echo "$*" >> "$RES_JOB_DIR/training.log"; }           # training.log: the loop writes the attempt headers only

# [2026-10-11, the user's rule] how long a restart needs before it has trained one step: this job's own longest start-up, measured from
# the srun start to the first iteration line (which is printed after the first step) of every attempt that reached it, plus a margin
# for the classification and the spare check (RES_RESTART_MARGIN_S, default 60 s); before any attempt reached a step, the fixed
# RES_MIN_ATTEMPT_S (default 420 s). Replaces the fixed 420 s, which is too short for a 671B start-up of 4-6.5 min plus a 43-min step.
res_need_attempt_s() {
  local m; m=$(awk -F'\t' '$1 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+$/ { d = $4 - $3; if (d > m) m = d } END { print m + 0 }' "$RES_DIR/attempts.tsv" 2>/dev/null)
  if [ "${m:-0}" -gt 0 ]; then echo $(( m + ${RES_RESTART_MARGIN_S:-60} )); else echo ${RES_MIN_ATTEMPT_S:-420}; fi
}
res_need_attempt_src() {
  local m; m=$(awk -F'\t' '$1 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+$/ { d = $4 - $3; if (d > m) m = d } END { print m + 0 }' "$RES_DIR/attempts.tsv" 2>/dev/null)
  if [ "${m:-0}" -gt 0 ]; then echo "this job's longest start-up to the first step was $m s, plus ${RES_RESTART_MARGIN_S:-60} s"; else echo "no attempt has finished a step yet, so the fixed ${RES_MIN_ATTEMPT_S:-420} s applies"; fi
}

res_state() {   # rewrite the state file read by the aggregator (and STATUS.md); $1 status, $2 the loop's next action in words
  [ -n "${2:-}" ] && RES_NEXT="$2"
  cat > "$RES_DIR/state.tmp" <<EOS
attempt=${RES_ATTEMPT}
status=$1
world=${TOTAL_GPUS}
group=${RES_GROUP}
pp=${PP_SIZE}
excluded=$(echo ${RES_EXCLUDED[*]} | tr ' ' ',')
degraded=$(echo ${RES_DEGRADED[*]} | tr ' ' ',')
train_nodes=${RES_TRAIN_CSV}
all_nodes=$(echo ${RES_ALL[*]} | tr ' ' ',')
duty=${RES_DUTY}
next=${RES_NEXT:-}
ucp_workers=$(echo ${RES_UCP[*]:-} | tr ' ' ',')
EOS
  mv -f "$RES_DIR/state.tmp" "$RES_DIR/state"
  $RES_PY -c "import sys; sys.path.insert(0, '$RES_BIN'); import hb_aggregator as a; a.status_md('$RES_JOB_DIR')" 2>>"$RES_DIR/aggregator.out"
}

res_good_nodes() {   # all nodes minus the excluded, in allocation order
  local n x skip out=()
  for n in "${RES_ALL[@]}"; do
    skip=0; for x in "${RES_EXCLUDED[@]}"; do [ "$n" = "$x" ] && skip=1; done
    [ $skip = 0 ] && out+=("$n")
  done
  echo "${out[@]}"
}

res_agg_start() {   # the duty: the aggregator on the duty node (a spare) or, without a spare, in the background on the batch node
  local where=$1
  res_agg_stop
  # the run console (2026-10-10, sandbox): on by default; every 30 s on a duty spare, every 120 s on the batch node; AUTORUN_DASHBOARD=0 turns it off
  local dash=0; [ "${AUTORUN_DASHBOARD:-1}" != 0 ] && { [ "$where" = "$RES_BATCH" ] && dash=${AUTORUN_DASHBOARD_EVERY_BATCH:-120} || dash=${AUTORUN_DASHBOARD_EVERY:-30}; }
  local args="--job-dir $RES_JOB_DIR --world $TOTAL_GPUS --group $RES_GROUP --pp $PP_SIZE --interval 10 --still ${RES_STILL_S:-30} --dashboard-every $dash"
  if [ "$where" = "$RES_BATCH" ]; then
    ( cd "$RES_BIN" && exec $RES_PY hb_aggregator.py $args >> "$RES_DIR/aggregator.out" 2>&1 ) &
  else
    srun --overlap -N1 -n1 -w "$where" --cpu-bind=none --export=ALL $RES_PY "$RES_BIN/hb_aggregator.py" $args >> "$RES_DIR/aggregator.out" 2>&1 &
  fi
  RES_AGG_PID=$!; RES_AGG_ON=$where
  res_say "the duty: the heartbeat aggregator started on node $where (process id $RES_AGG_PID), which is $( [ "$where" = "$RES_BATCH" ] && echo 'the batch node, because no spare node is left to hold the duty' || echo 'a spare node, so it holds the duty')$( [ $dash != 0 ] && echo "; it rewrites the run console dashboard.html every $dash s at the lowest priority (serve it with bin/dashboard_serve.sh on a login node)")."
}
res_agg_stop() { [ -n "${RES_AGG_PID:-}" ] && kill "$RES_AGG_PID" 2>/dev/null; RES_AGG_PID=""; }

res_retention() {   # keep the save 'latest' names, the one before it, and anything newer (a save in progress); sandbox only
  local d="$RES_CKPT_DIR" lt keep_prev
  [ -n "$d" ] && [ -f "$d/latest" ] || return 0
  case "$(readlink -f "$d")" in /lustre/orion/gen150/scratch/zixianw4/X-MoE-resilience/*) ;; *) return 0;; esac
  if [ "${RES_UCP_N:-0}" -gt 0 ]; then res_retention_ucp; return 0; fi   # with UCP workers the retention follows the universals (below)
  lt=$(sed 's/global_step//' "$d/latest")
  keep_prev=$(ls -d "$d"/global_step* 2>/dev/null | sed 's/.*global_step//' | sort -n | awk -v l=$lt '$1<l' | tail -1)
  for s in $(ls -d "$d"/global_step* 2>/dev/null | sed 's/.*global_step//' | sort -n); do
    if [ "$s" -lt "$lt" ] && [ "$s" != "$keep_prev" ]; then rm -rf "$d/global_step$s"; fi
  done
}

res_latest_step() { [ -f "$RES_CKPT_DIR/latest" ] && sed 's/global_step//' "$RES_CKPT_DIR/latest" || echo 0; }

# [2026-10-13, the DeepSeek-MoE-16B data-quality runs] RES_INIT_LOAD_DIR (baked by the launcher rider AUTORUN_INIT_LOAD_DIR): a run whose
# FIRST weights come from another run's checkpoint (weights only, a fresh optimizer, iteration 0 = Megatron's --finetune), while every
# restart after the run's own first save is a native resume of that save. Before this, the resilient rider removed --finetune for good, so a
# resilient run could only start from random weights or from its own save directory. Called before the gate's dry run and before every
# attempt. While the save directory has no 'latest': --load <init dir> --finetune (+ --universal-checkpoint when RES_INIT_UNIVERSAL=1, so
# the same initial weights load under any layout the OOM rule may choose before the first save). Once 'latest' exists: --load <save dir>,
# no --finetune, and the --universal-checkpoint this function added is removed unless the OOM rule asked for the universal of a save
# (RES_UNIV_LOADED_STEP). Unset RES_INIT_LOAD_DIR: returns at once, the loop runs exactly as before.
res_init_load() {   # $1 = gate | attempt
  [ -n "${RES_INIT_LOAD_DIR:-}" ] || return 0
  local o="$megatron_options" lt; lt=$(res_latest_step)
  if [ "${lt:-0}" -gt 0 ] 2>/dev/null; then
    o=$(echo "$o" | sed -E 's/(^| )--finetune( |$)/\1\2/g; s/  +/ /g')
    if [ "${RES_INIT_UNIV_ADDED:-0}" = 1 ] && [ -z "${RES_UNIV_LOADED_STEP:-}" ]; then
      o=$(echo "$o" | sed -E 's/(^| )--universal-checkpoint( |$)/\1\2/g; s/  +/ /g')
    fi
    RES_INIT_UNIV_ADDED=0
    if echo "$o" | grep -q -E -- '--load [^ ]+'; then o=$(echo "$o" | sed -E "s#--load [^ ]+#--load $RES_CKPT_DIR#"); else o="$o --load $RES_CKPT_DIR"; fi
    megatron_options="$o"
    RES_CUR_LOADS="the run's own save of step $lt$( [ -n "${RES_UNIV_LOADED_STEP:-}" ] && echo ' (universal, the OOM rule)' || echo ' (native resume)')"
    [ "${RES_INIT_SAID:-}" = own ] || res_say "the initial checkpoint is no longer used: the save directory $RES_CKPT_DIR holds the run's own save of step $lt, so $( [ "$1" = gate ] && echo 'the first attempt' || echo "attempt $RES_ATTEMPT" ) resumes it ($(echo "$o" | grep -q -- '--universal-checkpoint' && echo 'the universal checkpoint, because the OOM rule changed the layout' || echo 'native resume: weights, optimizer, iteration and data position')); --finetune is off."
    RES_INIT_SAID=own
  else
    if echo "$o" | grep -q -E -- '--load [^ ]+'; then o=$(echo "$o" | sed -E "s#--load [^ ]+#--load $RES_INIT_LOAD_DIR#"); else o="$o --load $RES_INIT_LOAD_DIR"; fi
    echo "$o" | grep -q -E -- '(^| )--finetune( |$)' || o="$o --finetune"
    if [ "${RES_INIT_UNIVERSAL:-0}" = 1 ] && ! echo "$o" | grep -q -- '--universal-checkpoint'; then o="$o --universal-checkpoint"; RES_INIT_UNIV_ADDED=1; fi
    megatron_options="$o"
    RES_CUR_LOADS="the initial checkpoint $RES_INIT_LOAD_DIR ($( [ "${RES_INIT_UNIVERSAL:-0}" = 1 ] && echo "universal $(cat "$RES_INIT_LOAD_DIR/latest_universal" 2>/dev/null)" || echo "native $(cat "$RES_INIT_LOAD_DIR/latest" 2>/dev/null)"), weights only, iteration 0)"
    [ "${RES_INIT_SAID:-}" = "init_$1" ] && [ "$1" = gate ] || res_say "$( [ "$1" = gate ] && echo 'before the gate' || echo "attempt $RES_ATTEMPT" ): the save directory $RES_CKPT_DIR has no save of this run yet, so the trainer loads the WEIGHTS ONLY of the initial checkpoint $RES_INIT_LOAD_DIR ($( [ "${RES_INIT_UNIVERSAL:-0}" = 1 ] && echo "its universal form $(cat "$RES_INIT_LOAD_DIR/latest_universal" 2>/dev/null), which loads under any layout" || echo "its native save $(cat "$RES_INIT_LOAD_DIR/latest" 2>/dev/null)")) with --finetune: a fresh optimizer, the iteration and the consumed samples at 0; the saves go to $RES_CKPT_DIR."
    RES_INIT_SAID="init_$1"
  fi
  return 0
}

res_first_iter() {   # yes when this attempt's last-rank log shows its first iteration line
  local f="$RANK_LOG_DIR/rank_$((TOTAL_GPUS-1)).log"
  [ -f "$f" ] && grep -m1 -E "^ *iteration +[0-9]+/" "$f" >/dev/null 2>&1 && echo yes
}

res_rel() { echo "${1#$RES_JOB_DIR/}"; }   # a path relative to the job folder, for the prose files

# [2026-10-11, sandbox, after HO1 5652117] Slurm's own words for a kernel or cgroup OOM kill (Slurm 24 writes "oom_kill event ... OOM Killed")
RES_OOM_RE="oom[-_]kill|OOM Killed|out-of-memory handler|task [0-9]+: Out Of Memory|Out of memory: Killed process"
res_hostoom_slurm() {   # $1 the attempt's .e lines: sets the host_oom class with the node and rank from srun's "task N: Out Of Memory" line
  RC_CODE=19; RC_FIRED="the kernel's out-of-memory killer (Slurm reported it in the .e: a real host OOM, not the 97 % guard)"
  RC_EVID="host OOM (Slurm): $(echo "$1" | grep -a -i -E "$RES_OOM_RE" | head -2 | tr '\n' ' ' | head -c 300)"
  local ol; ol=$(echo "$1" | grep -a -m1 -E "srun: error: [A-Za-z0-9._-]+: task [0-9]+: Out Of Memory")
  if [ -n "$ol" ]; then RC_NODE=$(echo "$ol" | sed -E 's/.*srun: error: ([A-Za-z0-9._-]+): task [0-9]+:.*/\1/'); RC_RANK=$(echo "$ol" | sed -E 's/.*: task ([0-9]+):.*/\1/'); fi
}

res_classify() {   # sets RC_CODE RC_RANK RC_NODE RC_STEP RC_EVID RC_FIRED ; inputs: $1 srun rc, $2 'd8' if the init deadline fired
  local rc=$1 d8=$2 f best="" bt=""
  RC_CODE=""; RC_RANK="-"; RC_NODE="-"; RC_STEP="-"; RC_EVID=""; RC_FIRED=""; RC_STEP_SRC=""; AG_RANK=""; AG_NODE=""; AG_STEP=""; AG_WHY=""
  # 1. a detector's own exit record (the earliest of this attempt)
  # [2026-10-10, round 6] one python call finds the earliest record: a real out of memory leaves up to one record per rank (512 on 64
  # nodes), and two interpreter starts per record cost minutes before the restart
  if ls "$RES_ATT_DIR"/exit/r*.json >/dev/null 2>&1; then
    best=$($RES_PY - "$RES_ATT_DIR/exit" <<'PYB'
import glob, json, sys
b = None
for f in glob.glob(sys.argv[1] + '/r*.json'):
    try:
        t = json.load(open(f))['time']
    except Exception:
        continue
    if b is None or t < b[0]:
        b = (t, f)
print(b[1] if b else '')
PYB
)
  fi
  local etail; etail=$(tail -c +$((RES_E_OFF+1)) "$RES_E_FILE" 2>/dev/null | grep -a -E "Watchdog caught|[Tt]imeout|timed out|NODE_FAIL|node failure|not responding|srun: error|Exited with exit code|Killed|Aborted|Error|error" | head -3000)
  if [ -n "$best" ]; then
    eval "$($RES_PY - "$best" "$RES_JOB_DIR" <<'PYE'
import json, sys, shlex
r = json.load(open(sys.argv[1])); rel = sys.argv[1][len(sys.argv[2]) + 1:]
print(f"RC_CODE={r['code']}; RC_RANK={r['rank']}; RC_NODE={shlex.quote(r['node'])}; RC_STEP={r['step']}; RC_STEP_SRC='the exit record'; "
      f"RC_FIRED={shlex.quote('the detector inside rank ' + str(r['rank']) + ' (its exit record ' + rel + ')')}; "
      f"RC_EVID={shlex.quote('exit record ' + rel + ': ' + r['evidence'][:300])}")
PYE
)"
    # [2026-10-11, sandbox, HO1 5652117] a real host OOM grows for minutes; the peers' watchdog can fire first and write the earliest record
    # (collective_hang). When Slurm reports an OOM kill in this attempt, it is the cause and those classes are its consequences.
    if echo "$etail" | grep -q -a -i -E "$RES_OOM_RE" && [[ " 10 16 17 21 " == *" $RC_CODE "* ]]; then
      local prev="class $RC_CODE (${RC_EVID:0:160})"
      res_hostoom_slurm "$etail"; RC_STEP_SRC="the exit record of the rank that waited"
      RC_EVID="$RC_EVID; the earlier exit record, $prev, is a consequence: the other ranks waited for the killed rank"
    fi
    return
  fi
  if echo "$etail" | grep -q -a -i -E "$RES_OOM_RE"; then
    # [2026-10-11, sandbox] a REAL host out of memory: the kernel's or the cgroup's OOM killer ended a rank (Slurm says so in the .e), which
    # the 97 % guard of hb.py (exit 19) did not catch in time; the class is host_oom like the guard's, so no node is excluded; it is checked
    # before the watchdog lines and the init deadline, which a killed rank causes in the others (HO1 5652117)
    res_hostoom_slurm "$etail"
  elif [ "$d8" = d8 ]; then
    RC_CODE=18; RC_FIRED="the init deadline (D8) of the job script"; RC_EVID="the init deadline (D8): no iteration line appeared within ${RES_T_INIT_S} s of the srun start"
  elif echo "$etail" | grep -q -a -E "Watchdog caught collective operation timeout|ProcessGroupNCCL.*[Tt]imeout|NCCL.*timed out"; then
    RC_CODE=10; RC_FIRED="the NCCL watchdog (the collective timeout), seen in the .e file"
    RC_EVID="watchdog: $(echo "$etail" | grep -a -m1 -o -E "Watchdog caught collective operation timeout: WorkNCCL\([^)]*\)" | head -c 250)"
  elif echo "$etail" | grep -q -a -E "Unable to confirm allocation|Socket timed out on send/recv|Job step creation temporarily disabled|Unable to create step"; then
    RC_CODE=21; RC_FIRED="srun (Slurm could not launch the step; no rank started)"
    RC_EVID="srun could not launch the step (Slurm, not a rank): $(echo "$etail" | grep -a -m1 -E "Unable to confirm allocation|Socket timed out|step creation|Unable to create step" | head -c 200)"
  elif echo "$etail" | grep -q -a -i -E "NODE_FAIL|node failure|Node .* not responding"; then
    RC_CODE=20; RC_FIRED="Slurm (a node failure reported to srun)"; RC_EVID="srun: $(echo "$etail" | grep -a -m1 -i -E "NODE_FAIL|node failure|not responding" | head -c 200)"
  else
    RC_CODE=21; RC_FIRED="srun --kill-on-bad-exit (a rank exited without an exit record)"
    RC_EVID="srun: $(echo "$etail" | grep -a -m1 -E "srun: error: [A-Za-z0-9._-]+: task [0-9]+: " | head -c 200)"
    # [2026-10-11, sandbox] the fallback for a real host OOM that Slurm did not name: the rank was killed by a signal on a node whose last
    # heartbeats read host memory at or above RES_HOSTOOM_INFER_PCT (90 %; HO1 5652117: 5 points per 10-s beat, the last beat 94.7 % 14 s before the kill): the kernel's OOM killer is the likely cause, so the class is host_oom
    local kl kn hm; kl=$(echo "$etail" | grep -a -m1 -E "srun: error: [A-Za-z0-9._-]+: task [0-9]+: (Killed|Exited with exit code 137)")
    if [ -n "$kl" ]; then
      kn=$(echo "$kl" | sed -E 's/.*srun: error: ([A-Za-z0-9._-]+): task [0-9]+:.*/\1/')
      hm=$($RES_PY - "$RES_DIR/hb/$kn.tsv" <<'PYH' 2>/dev/null
import sys
best = -1.0
try:
    lines = open(sys.argv[1], errors='replace').read().splitlines()[-200:]
except OSError:
    lines = []
h = None
for l in lines:
    v = l.split('\t')
    if l.startswith('time'):
        h = v; continue
    if l.startswith('#') or h is None or len(v) != len(h):
        continue
    try:
        best = max(best, float(v[h.index('host_mem_pct')]))
    except (ValueError, IndexError):
        pass
print(f'{best:.1f}')
PYH
)
      if [ -n "$hm" ] && awk -v a="$hm" -v b="${RES_HOSTOOM_INFER_PCT:-90}" 'BEGIN{exit !(a >= b)}'; then
        RC_CODE=19; RC_FIRED="the kernel's out-of-memory killer (inferred: a rank was killed by a signal while its node's host memory read $hm %)"
        RC_NODE=$kn; RC_RANK=$(echo "$kl" | sed -E 's/.*: task ([0-9]+):.*/\1/')
        RC_EVID="host OOM (inferred): the node's last heartbeats read host memory $hm % (>= ${RES_HOSTOOM_INFER_PCT:-90}); $RC_EVID"
      fi
    fi
  fi
  # 2. name the node: the aggregator's last table (minority, then stillness), then the recorder dumps, then srun's first task line
  local agg="$RES_DIR/agg_last.json"
  if [ -f "$agg" ]; then
    eval "$($RES_PY - "$agg" "$RES_ATTEMPT" <<'PYA'
import json, sys, shlex
t = json.load(open(sys.argv[1]))
if str(t.get('attempt')) == sys.argv[2]:
    for key in ('minority', 'still'):
        if t.get(key):
            r = t[key][0]; d = t['ranks'].get(str(r), {})
            print(f"AG_RANK={r}; AG_NODE={shlex.quote(d.get('node', '-'))}; AG_STEP={d.get('step', '-')}; AG_WHY={key}")
            break
    else:
        print(f"AG_STEP={t.get('step', '-')}")
PYA
)"
  fi
  if [ -n "${AG_NODE:-}" ] && [ "$RC_NODE" = "-" ] && [ "$RC_CODE" != 21 -o -z "$(echo "$etail" | grep -a -m1 -E "srun: error: [A-Za-z0-9._-]+: task [0-9]+: (Exited|Killed)")" ]; then
    RC_RANK=$AG_RANK; RC_NODE=$AG_NODE; RC_EVID="$RC_EVID; heartbeat ${AG_WHY} rule: rank $AG_RANK on $AG_NODE (heartbeat.log, data/agg_last.json)"
  fi
  # the step: the injection marker (the step at which the fault fired) is preferred over the heartbeat table, whose step is the
  # last COMPLETED step of the last line and so lags by one beat (the user, 2026-10-09 evening)
  local ms; ms=$(res_injected_step)
  if [ -n "$ms" ]; then RC_STEP=$ms; RC_STEP_SRC="the injection marker"
  else RC_STEP=${AG_STEP:-$RC_STEP}; RC_STEP_SRC="the heartbeat table (the last completed step of the last line, which can lag by one beat)"; fi
  if [ "$RC_CODE" = 10 ]; then
    # the watchdog lines: the member of the first-firing group of G ranks that did not fire never entered the collective
    local wd; wd=$($RES_PY "$RES_BIN/watchdog_culprit.py" "$RES_E_FILE" "$RES_E_OFF" "$RES_GROUP" 2>/dev/null)
    if [ -n "$wd" ]; then
      RC_EVID="$RC_EVID; watchdog lines: $wd"
      if [ "$RC_NODE" = "-" ]; then
        local wr; wr=$(echo "$wd" | awk '{print $2}'); RC_RANK=$wr; RC_NODE=${RES_TRAIN[$((wr / gpus_per_node))]}
        RC_EVID="$RC_EVID (named by the watchdog-group rule)"
      fi
    fi
  fi
  if [ "$RC_CODE" = 10 ] && ls "$RES_ATT_DIR"/nccl_trace/nccl_trace_rank_* >/dev/null 2>&1; then
    # fr_analyze keys collectives by a per-process group id: trusted for naming only when ONE rank is absent (see watchdog_culprit.py)
    local fr; fr=$(timeout 60 $RES_PY "$RES_BIN/fr_analyze.py" "$RES_ATT_DIR/nccl_trace" 2>&1 | tee "$RES_ATT_DIR/nccl_trace/fr_analyze.txt" | grep -m1 "ABSENT")
    [ -n "$fr" ] && RC_EVID="$RC_EVID; recorder ($(res_rel "$RES_ATT_DIR")/nccl_trace/fr_analyze.txt): $(echo $fr | head -c 160)"
    if [ "$RC_NODE" = "-" ]; then
      local ar; ar=$(echo "$fr" | grep -o -E "ABSENT on ranks? \[[0-9]+\]" | grep -o -E "[0-9]+")
      [ -n "$ar" ] && { RC_RANK=$ar; RC_NODE=${RES_TRAIN[$((ar / gpus_per_node))]}; RC_EVID="$RC_EVID (named by the recorder)"; }
    fi
  fi
  if [ "$RC_NODE" = "-" ]; then
    local tl; tl=$(echo "$etail" | grep -a -m1 -o -E "srun: error: [A-Za-z0-9._-]+: task [0-9]+: (Exited with exit code [0-9]+|Killed|Aborted|[A-Za-z ]+)")
    if [ -n "$tl" ]; then
      RC_NODE=$(echo "$tl" | sed -E 's/^srun: error: ([A-Za-z0-9._-]+): task.*/\1/'); RC_RANK=$(echo "$tl" | sed -E 's/.*task ([0-9]+).*/\1/')
      RC_EVID="$RC_EVID; first failed task: $tl"
    fi
  fi
  if [ "$RC_CODE" = 20 ] || [ "$RC_NODE" = "-" ]; then   # Slurm's view of the training nodes
    local down; down=$(sinfo -h -n "$RES_TRAIN_CSV" -o "%N %T" 2>/dev/null | grep -i -E "down|drain|fail" | head -1)
    [ -n "$down" ] && { RC_CODE=20; RC_NODE=$(echo $down | awk '{print $1}'); RC_EVID="$RC_EVID; sinfo: $down"; }
  fi
}

res_injected_info() {   # prints "yes <case>" if an injection of this attempt fired (not 'slow'), else "no -"
  local f c
  for f in "$RES_DIR"/inject/*.fired; do
    [ -f "$f" ] || continue
    c=$($RES_PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['case'] if str(d['attempt'])==sys.argv[2] else '')" "$f" "$RES_ATTEMPT")
    [ -n "$c" ] && [ "$c" != slow ] && { echo "yes $c"; return; }
  done
  echo "no -"
}

res_injected_step() {   # prints the step at which an injection of this attempt fired (the marker's 'step'), else nothing
  local f s
  for f in "$RES_DIR"/inject/*.fired; do
    [ -f "$f" ] || continue
    s=$($RES_PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['step'] if str(d['attempt'])==sys.argv[2] and d['case']!='slow' else '')" "$f" "$RES_ATTEMPT")
    [ -n "$s" ] && { echo "$s"; return; }
  done
}

res_injected_desc() {   # one sentence for an injection marker file (loop.log)
  $RES_PY -c "
import json,sys
d=json.load(open(sys.argv[1]))
print(f\"an injected fault fired: case '{d['case']}' on rank {d['rank']} (node {d['node']}) at step {d['step']} (planned for step {d['step_planned']}), in attempt {d['attempt']}, at {d.get('time_str','?')}\")" "$1"
}

res_banner() {   # the user's LOUD banner: first lines of heartbeat.log and of a-monitor.txt, and the recovery block
  local b="$*"
  echo "$b" >> "$RES_DIR/banners.txt"
  $RES_PY "$RES_BIN/hb_aggregator.py" --job-dir "$RES_JOB_DIR" --compose 2>>"$RES_DIR/aggregator.out"
  res_rec "$b"; res_rec ""
  res_say "$b"
  echo "$b" >&2
}

res_mail() {   # mail when the run stops for good; no address -> print loudly instead
  local subj="$1" body="$2"
  res_rec "===== $(date '+%F %H:%M:%S')  THE RUN STOPS: $subj ====="; res_rec "$body"; res_rec ""
  if [ -n "${RES_MAIL_TO:-}" ] && command -v mail >/dev/null 2>&1; then
    echo "$body" | mail -s "$subj" "$RES_MAIL_TO" && res_say "a mail was sent to $RES_MAIL_TO with the subject: $subj" && return
  fi
  res_loud "THE RUN STOPS (no mail address is configured): $subj -- $body"
}

res_heatmap() {   # heatmap.png at the end of an attempt (the aggregator draws it on an alarm); never once a minute
  [ -f "$RES_DIR/agg_last.json" ] || return 0
  timeout 90 $RES_PY "$RES_BIN/heatmap.py" --job-dir "$RES_JOB_DIR" --why "$1" 2>>"$RES_DIR/heatmap.err" \
    && res_say "heatmap.png was drawn ($1)." || res_say "heatmap.png was not drawn ($1); the error is in data/heatmap.err."
}

res_readme() {
  cat > "$RES_JOB_DIR/README.md" <<EOF
# Job $SLURM_JOB_ID: the map of this folder (design 3.11 of records/2026-10/1009_2026_RESILIENCE_DESIGN.md)

Read in this order:

| file | what it answers | written by |
|---|---|---|
| STATUS.md | is the job alive, and where is it: state, attempt, step, loss, TFLOP/s, seconds per step, last save, nodes (training / spare / degraded / excluded), the open alarm, the loop's next action | the aggregator, every 10 s |
| training.log | is it training well: one line per step across attempts (time, attempt, step, loss, TFLOP/s per GCD, seconds per step, HBM max GB, note); each attempt opens with a header (versions, GPU-direct, NCCL warnings so far) | the trainer (rank 0 at PP 1, else the last rank); the headers by the job script |
| dashboard.html | the run console in a browser: the curves, the checkpoints and their scores, the machine map by stage and replica, the node inspector, the fleet, the timeline, the pre-run gate (serve it with bin/dashboard_serve.sh on a login node) | the heartbeat aggregator, every 30 s on the duty node or 120 s on the batch node, at the lowest priority; AUTORUN_DASHBOARD=0 turns it off |
| recovery.log | did anything go wrong, and what was done: one block per failure in words (when, what fired, the evidence file, node and rank, class, excluded or not, restart k, resumed from step s) and the RECOVERED banners | the job script |
| alarms.log | which ranks were named by the stillness or minority rule, when, in which phase, whether in a save window, where the stack sample is (alarms/), and whether the alarm cleared or became a failure: one full-sentence line per event | the aggregator (the "became failure" lines by the job script) |
| heartbeat.log | the RECOVERED banners on top, then one block per 10-s beat (ranks per phase, oldest line, slow ranks, counters) and the ALARM lines | the aggregator |
| gate.log | was the machine healthy at the start: the summary table first (test, calibration, budget, passed, failed, verdict), then per test the command, the budget, the worst node and every node | the gate |
| heatmap.png | the rank grid of the last alarm or of the end of the last attempt | the aggregator (on an alarm), the job script (end of an attempt) |
| a-monitor.txt | the 20-line banner area, then the existing monitor's samples (memory, utilisation, stacks) | the monitor; the banner area by the aggregator |
| RESULTS.md | test jobs only: the injected-case table (PASS / FAIL, detect s, recover s) and the gate's injected faults | results.py at the end |
| a-xmoe-*.o / .e | the batch script's own output (Slurm) | Slurm |

Folders:
- alarms/ : the stack samples (py-spy) of the ranks named by an alarm, one file per node and alarm time (listed in alarms.log).
- attempts/attempt_<k>/ : one run of the trainer (attempt 0 = the first run; attempt k = the run after the k-th failure):
  full_run.log (the .o + rank 0 + the last rank, merged), ranks/rank_<r>.log (written there directly), nccl_trace/ (the flight-recorder
  dumps when the watchdog fired, and fr_analyze.txt), exit/ (the detectors' exit records of this attempt).
- data/ : for scripts: attempts.tsv, failures.tsv, gate_report.tsv, failure_record.tsv, results.tsv, state, loop.log (every
  milestone of the job script), save_times.tsv, banners.txt, heartbeat_body.log, agg_last.json; hb/<node>.tsv (the heartbeat lines),
  alarm/ALARM (the marker the aggregator appends to on every new alarm), monitor/ (the monitor's per-node files), darshan/, inject/, gate/ (the raw gate outputs,
  excluded_reasons.tsv, and spare_check_*/ for the spare checks).
EOF
}

# [2026-10-09 evening] the node-local checkpoint copies (design 3.5, AUTORUN_LOCAL_COPIES >= 1; the engine writes them to
# /dev/shm/xmoe_ckpt/<job>_<run>/ on every training node). The loop only cleans and reports: stale folders of other jobs at the
# start, this job's folders at the end (/dev/shm survives the job on a node), and one prose line per attempt in recovery.log.
res_shm_clean() {   # $1 = start | end
  [ "${AUTORUN_LOCAL_COPIES:-0}" != 0 ] || return 0
  local base=${XMOE_LOCAL_CKPT_DIR:-/dev/shm/xmoe_ckpt} nl="$(res_good_nodes)" k; k=$(echo $nl | wc -w)
  [ "$k" -gt 0 ] || return 0
  if [ "$1" = start ]; then
    timeout -k 5 60 srun --overlap -N $k -n $k --ntasks-per-node=1 -w "$(echo $nl | tr ' ' ',')" --kill-on-bad-exit=0 \
      bash -c "[ -d $base ] && find $base -mindepth 1 -maxdepth 1 ! -name '${SLURM_JOB_ID}_*' -exec rm -rf {} + 2>/dev/null; echo \"\$(hostname -s) /dev/shm free \$(df -Pk /dev/shm | awk 'NR==2{printf \"%.1f\", \$4/1048576}') GB\"" \
      2>&1 | sort | tr '\n' ';' | sed 's/;$//' | { read -r l; res_say "the node-local copy area $base was cleaned of other jobs' folders; the free space per node: $l."; }
  else
    timeout -k 5 60 srun --overlap -N $k -n $k --ntasks-per-node=1 -w "$(echo $nl | tr ' ' ',')" --kill-on-bad-exit=0 \
      bash -c "u=\$(du -sk $base/${SLURM_JOB_ID}_* 2>/dev/null | awk '{s+=\$1} END{printf \"%.2f\", s/1048576}'); rm -rf $base/${SLURM_JOB_ID}_* 2>/dev/null; echo \"\$(hostname -s) held \${u:-0} GB of local copies at the end (removed)\"" \
      2>&1 | sort | tr '\n' ';' | sed 's/;$//' | { read -r l; res_say "the node-local copies of this job were removed at the end: $l."; }
  fi
}
res_local_copy_lines() {   # $1 = summary <attempt> | saves ; one prose line (or a few) into recovery.log and loop.log
  [ "${AUTORUN_LOCAL_COPIES:-0}" != 0 ] || return 0
  local out
  if [ "$1" = summary ]; then out=$($RES_PY "$RES_BIN/local_copy.py" summary --data-dir "$RES_DIR" --attempt "$2" 2>&1)
  else out=$($RES_PY "$RES_BIN/local_copy.py" saves --data-dir "$RES_DIR" 2>&1); fi
  echo "$out" | while IFS= read -r l; do [ -n "$l" ] && { res_rec "  $l"; res_say "$l"; }; done
}

res_versions() {   # the gate's environment check, first node, in one line (for the training.log headers)
  local r="$RES_DIR/gate_report.tsv"
  [ -f "$r" ] || { echo "the gate did not run, so the versions were not checked"; return; }
  awk -F'\t' 'NR>1 && ($2=="rocm_version"||$2=="rccl_md5"||$2=="ofi_plugin_md5"||$2=="libfabric"||$2=="gpu_direct_env") && !s[$2]++ {printf "%s%s=%s", (n++?"; ":""), $2, $3}' "$r"
}

res_nccl_warn() {   # NCCL WARN lines so far: in the .e file and in the rank logs of the earlier attempts
  local e r=0 d
  e=$(grep -a -c "NCCL WARN" "$RES_E_FILE" 2>/dev/null); e=${e:-0}
  for d in "$RES_JOB_DIR"/attempts/attempt_*/ranks; do
    [ -d "$d" ] && [ "$d" != "$RANK_LOG_DIR" ] && r=$(( r + $(timeout 60 cat "$d"/rank_*.log 2>/dev/null | grep -a -c "NCCL WARN") ))
  done
  echo "$e NCCL WARN lines in the .e file and $r in the rank logs of the earlier attempts"
}

res_gdr() {   # the GPU-direct state RCCL chose, from rank 0's log of the given attempt folder
  local f="$1/ranks/rank_0.log" n g
  [ -f "$f" ] || { echo "rank 0 wrote no log yet"; return; }
  n=$(grep -a -c " via NET/" "$f"); g=$(grep -a -c " via NET/.*/GDRDMA" "$f")
  echo "rank 0 opened $n network channels, $g of them with /GDRDMA (GPU-direct RDMA)$( [ "$n" -gt 0 ] && [ "$g" = 0 ] && echo ', so GPU-direct appears to be OFF (see 2026-10-09-collective_stack C2: no measured cost)')"
}

# ---- [2026-10-09 late evening] the gate's fault injection (TEST ONLY): XMOE_GATE_INJECT="pair_silent:@1+gemm_half:@2"; '@<k>'
#      means the k-th node of the allocation (0 = the first), resolved here to a node name before the gate runs
res_gate_inject_resolve() {
  [ -n "${XMOE_GATE_INJECT:-}" ] || return 0
  local resolved; resolved=$($RES_PY - "$XMOE_GATE_INJECT" "${RES_ALL[@]}" <<'PYR'
import re, sys
spec, nodes = sys.argv[1], sys.argv[2:]
def rep(m):
    k = int(m.group(1))
    return nodes[k] if k < len(nodes) else m.group(0)
print(re.sub(r'@(\d+)', rep, spec))
PYR
)
  export XMOE_GATE_INJECT="$resolved"
  echo "$resolved" > "$RES_DIR/gate_inject.txt"
  res_say "TEST ONLY: the gate's fault injection is set: XMOE_GATE_INJECT=$resolved (pair_silent: one rank of that node never joins its all-reduce in either round; gemm_slow: one GCD of that node sleeps past the GEMM deadline; gemm_half: one GCD of that node runs the GEMM at half speed)."
}
res_gate_injected() {   # yes when the node is named in XMOE_GATE_INJECT, else no
  case "+${XMOE_GATE_INJECT:-}+" in *":$1+"*|*":$1,"*) echo yes;; *) echo no;; esac
}

# ---- the gate's exclusions into the failure record and the two tiers (RES_EXCLUDED, RES_DEGRADED)
res_apply_gate_exclusions() {   # reads data/gate/excluded_reasons.tsv (node tier test value threshold sentence)
  local f="$RES_DIR/gate/excluded_reasons.tsv" line node tier test value thr sent inj
  [ -f "$f" ] || return 0
  while IFS=$'\t' read -r node tier test value thr sent; do
    [ "$node" = node ] && continue; [ -n "$node" ] || continue
    RES_EXCLUDED+=("$node")
    inj=$(res_gate_injected "$node")
    if [ "$tier" = degraded ]; then
      RES_DEGRADED+=("$node:$test:$value:${thr#>= }")
      res_say "the gate excluded $node as DEGRADED: $sent; the node is kept as a spare of last resort (AUTORUN_USE_DEGRADED=${AUTORUN_USE_DEGRADED:-1})."
    else
      res_say "the gate excluded $node as DEAD: $sent; the node is never used in this job."
    fi
    $RES_PY "$RES_BIN/failure_record.py" append --record "$RES_DIR/failure_record.tsv" --record "$RES_BIN/../failure_record.tsv" \
      --job "$SLURM_JOB_ID" --cls "gate_$test" --code 30 --node "$node" --rank "-" --step "-" --injected "$inj" \
      --evidence "the gate, before attempt 0 ($tier): $sent" >> "$RES_LOOP_LOG" 2>&1
  done < "$f"
}

# ---- the spare check: the quick tests of the gate on ONE node, before it enters the training set (a few seconds)
res_spare_check() {   # $1 node; returns 0 when the node passes, 1 when it is excluded (RES_EXCLUDED updated, record appended)
  local n=$1
  local d="$RES_DIR/gate/spare_check_a${RES_ATTEMPT}_$n" base="$RES_BIN/../gate_baseline.tsv" t0 bud to rc reasons   # (two statements: one local expands $n before it is set)
  mkdir -p "$d"
  t0=$(date +%s.%N)
  bud=$(cd "$RES_BIN" && $RES_PY -c "
import sys; sys.argv=['x']; import gate_judge as g
cal = g.read_cal('$base'); q = [t for t in g.TESTS if t not in g.FULL_ONLY]
print(','.join(f'{t}={g.budget_of(t, cal)}' for t in q)); print(sum(g.budget_of(t, cal) for t in q), file=sys.stderr)" 2>"$d/budget_sum")
  to=$(( $(cat "$d/budget_sum" 2>/dev/null || echo 300) + 60 ))
  local kv; for kv in ${bud//,/ }; do export GATE_BUDGET_${kv%%=*}=${kv#*=}; done
  GATE_PYTHON=$RES_PY timeout -k 10 $to srun --overlap -N1 -n1 -w "$n" --kill-on-bad-exit=0 ${GATE_SRUN_EXTRA:-} bash "$RES_BIN/gate_node.sh" quick "$d" > "$d/srun.log" 2>&1
  rc=$?
  if [ -s "$d/$n.raw.tsv" ]; then printf '%s\t%s\tanswers_srun\tyes\t0\tok\tthe node wrote its gate rows in the spare check\n' "$(date '+%F %H:%M:%S')" "$n" > "$d/$n.probe.tsv"
  else printf '%s\t%s\tanswers_srun\tno\t0\tok\tthe node wrote no gate row in the spare check (srun exit code %s)\n' "$(date '+%F %H:%M:%S')" "$n" "$rc" > "$d/$n.probe.tsv"; fi
  $RES_PY "$RES_BIN/gate_judge.py" --raw "$d" --nodes "$n" --mode quick --baseline "$base" --report "$d/report.tsv" --failed "$d/failed.txt" \
    --log "$d/gate.log" --job "$SLURM_JOB_ID" --t0 "${t0%.*}" --budgets "$bud" > "$d/judge.out" 2>&1
  local el; el=$(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f", b-a}')
  if [ -s "$d/failed.txt" ]; then
    reasons=$(awk -F'\t' 'NR>1 {print $6}' "$d/excluded_reasons.tsv" 2>/dev/null | head -c 400)
    RES_EXCLUDED+=("$n")
    res_say "spare check: before $n could join the training set for attempt $RES_ATTEMPT, the quick tests of the gate ran on it alone ($el s): it FAILED, so it is excluded and the next spare is tried. The judge's sentence: ${reasons:-see $(res_rel "$d")/gate.log}."
    res_rec "  spare check: $n failed the quick tests of the gate before joining attempt $RES_ATTEMPT ($el s) and was excluded: ${reasons:-see $(res_rel "$d")/gate.log}"
    $RES_PY "$RES_BIN/failure_record.py" append --record "$RES_DIR/failure_record.tsv" --record "$RES_BIN/../failure_record.tsv" \
      --job "$SLURM_JOB_ID" --cls "gate_$(awk -F'\t' 'NR==2 {print $3}' "$d/excluded_reasons.tsv" 2>/dev/null || echo spare_check)" --code 30 --node "$n" --rank "-" --step "-" \
      --injected "$(res_gate_injected "$n")" --evidence "the spare check before attempt $RES_ATTEMPT: ${reasons:-failed}" >> "$RES_LOOP_LOG" 2>&1
    return 1
  fi
  local nt; nt=$(awk -F'\t' 'NR>1 && $6=="yes"' "$d/report.tsv" 2>/dev/null | wc -l)
  res_say "spare check: before $n joins the training set for attempt $RES_ATTEMPT, the quick tests of the gate ran on it alone ($nt tests passed in $el s): it passed, so it joins (the rows are in $(res_rel "$d")/)."
  res_rec "  spare check: $n passed the quick tests of the gate ($nt tests, $el s) before joining attempt $RES_ATTEMPT"
  return 0
}

# ---- the last resort: the best degraded node (the highest value against its threshold) when no healthy spare is left
res_pick_degraded() {   # prints "node test value threshold" of the best degraded node, or nothing
  [ ${#RES_DEGRADED[@]} -gt 0 ] || return 0
  printf '%s\n' "${RES_DEGRADED[@]}" | awk -F: '{r=($4+0>0)?$2*0+$3/$4:0; print r, $1, $2, $3, $4}' | sort -k1,1gr | head -1 | awk '{print $2, $3, $4, $5}'
}
res_use_degraded() {   # moves the best degraded node back into the good set; sets RES_SWAP_NOTE; returns 1 when there is none
  local pick; pick=($(res_pick_degraded))
  [ ${#pick[@]} -ge 4 ] || return 1
  local n=${pick[0]} t=${pick[1]} v=${pick[2]} thr=${pick[3]} tmp=() x
  for x in "${RES_EXCLUDED[@]}"; do [ "$x" != "$n" ] && tmp+=("$x"); done; RES_EXCLUDED=("${tmp[@]}")
  tmp=(); for x in "${RES_DEGRADED[@]}"; do [ "${x%%:*}" != "$n" ] && tmp+=("$x"); done; RES_DEGRADED=("${tmp[@]}")
  RES_SWAP_NOTE="no healthy spare was left, so the degraded node $n ($t $v against the threshold $thr) was put into training for attempt $RES_ATTEMPT; expect the step to be slower"
  res_loud "LAST RESORT: $RES_SWAP_NOTE"
  res_rec "  last resort: $RES_SWAP_NOTE"
  return 0
}

# ================================================== THE UCP WORKER DUTY (AUTORUN_UCP_WORKERS, 2026-10-10) ==================================================
res_ucp_say() { res_say "ucp: $*"; }
res_ucp_rec() { echo "  ucp: $*" >> "$RES_JOB_DIR/recovery.log"; }
res_ucp_state() {   # one full sentence into data/ucp_state (STATUS.md shows it as the 'conversion' line)
  printf '%s\n' "$*" > "$RES_DIR/ucp_state.tmp" && mv -f "$RES_DIR/ucp_state.tmp" "$RES_DIR/ucp_state"
}
res_ucp_complete() {   # $1 step: 0 when the universal of that step is complete (the four checks of v3x_convert.sh), else 1
  local out="$RES_CKPT_DIR/global_step$1_universal" n_atoms n_tmp
  [ -d "$out/zero" ] || return 1
  [ -f "$out/zero/optimizer_state.pt" ] || return 1
  n_tmp=$(ls -d "$out"/tmp* 2>/dev/null | wc -l); [ "$n_tmp" -eq 0 ] || return 1
  n_atoms=$(find "$out/zero" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l); [ "$n_atoms" -gt 0 ] || return 1
  if [ -d "$out/shards" ]; then [ -d "$out/shards/finalize.lock" ] || return 1; fi
  return 0
}
res_ucp_atoms() { find "$RES_CKPT_DIR/global_step$1_universal/zero" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l; }
res_ucp_save_complete() {   # $1 step: 0 when 'latest' names this step or a later one AND the folder exists (the engine writes 'latest' only when every rank's files are complete)
  local lt; lt=$(res_latest_step)
  [ -d "$RES_CKPT_DIR/global_step$1" ] && [ "$1" -le "$lt" ]
}
res_ucp_steps_unconverted() {   # the steps of the complete saves that have no complete universal and did not fail twice, newest first
  local lt s; lt=$(res_latest_step)
  for s in $(ls -d "$RES_CKPT_DIR"/global_step* 2>/dev/null | sed 's/.*global_step//' | grep -E '^[0-9]+$' | sort -rn); do
    [ "$s" -le "$lt" ] || continue
    res_ucp_complete "$s" && continue
    grep -q -x "$s" "$RES_DIR/ucp/failed_steps" 2>/dev/null && continue
    echo "$s"
  done
}
res_ucp_run_share() {   # $1 step, $2 share k, $3 worker node, $4 log file: one ds_to_universal share on one worker, foreground; returns its rc
  local N=$1 k=$2 w=$3 log=$4 in="$RES_CKPT_DIR/global_step$1" out="$RES_CKPT_DIR/global_step$1_universal"
  mkdir -p "$out"; lfs setstripe -c 8 -S 4M "$out" 2>/dev/null || true
  srun --overlap -N1 -n1 -w "$w" --cpu-bind=none --kill-on-bad-exit=0 --export=ALL bash -c "cd '$PWD' && echo \"[ucp share $k/$RES_UCP_N of step $N on \$(hostname -s), start \$(date '+%F %T')]\" && \
    $RES_PY -m deepspeed.checkpoint.ds_to_universal --input_folder '$in' --output_folder '$out' --num_extract_workers ${RES_UCP_EXTRACT_WORKERS:-64} --num_merge_workers ${RES_UCP_MERGE_WORKERS:-64} \
    --fragments-inplace --optstate-from-pickle --param-shard $k/$RES_UCP_N ${RES_UCP_BALANCED:+--param-shard-balanced}; rc=\$?; echo \"[ucp share $k/$RES_UCP_N of step $N: exit code \$rc, end \$(date '+%F %T')]\"; exit \$rc" > "$log" 2>&1 &
  local pid=$!; echo $pid >> "$RES_DIR/ucp/srun_pids"
  wait $pid; local rc=$?
  sed -i "/^$pid\$/d" "$RES_DIR/ucp/srun_pids" 2>/dev/null
  return $rc
}
res_ucp_convert() {   # $1 step: the conversion of one complete save on every worker at once (one share per worker), the retry, the checks, the record
  local N=$1 t0 t1 k rc pids=() rcs=() w other logs=() tries=() n_atoms result="" skipped
  t0=$(date +%s)
  res_ucp_say "the conversion of the save of step $N to the universal form starts on the $RES_UCP_N worker node(s) ${RES_UCP[*]} ($RES_UCP_N shares, one per worker, byte-balanced across the shares: ds_to_universal --param-shard k/$RES_UCP_N --fragments-inplace --optstate-from-pickle, ${RES_UCP_EXTRACT_WORKERS:-64} extract and ${RES_UCP_MERGE_WORKERS:-64} merge workers); its logs are data/ucp/convert_s${N}_share<k>.log."
  res_ucp_state "the conversion of step $N to the universal form is running since $(date -d @$t0 '+%H:%M:%S') on the $RES_UCP_N worker nodes ${RES_UCP[*]} (one share per worker); the last complete universal is $( [ -f "$RES_CKPT_DIR/latest_universal" ] && echo "$(cat "$RES_CKPT_DIR/latest_universal" | sed 's/global_step//; s/_universal//') (latest_universal)" || echo none )."
  for k in $(seq 0 $((RES_UCP_N - 1))); do
    logs[$k]="$RES_DIR/ucp/convert_s${N}_share$k.log"; tries[$k]=1
    res_ucp_run_share "$N" "$k" "${RES_UCP[$k]}" "${logs[$k]}" & pids[$k]=$!
  done
  for k in $(seq 0 $((RES_UCP_N - 1))); do wait ${pids[$k]}; rcs[$k]=$?; done
  # a failed share is retried ONCE on another worker (the share's markers make it resumable); the finalize happens in whichever share ends last
  for k in $(seq 0 $((RES_UCP_N - 1))); do
    if [ "${rcs[$k]}" != 0 ]; then
      other=${RES_UCP[$(( (k + 1) % RES_UCP_N ))]}; tries[$k]=2
      res_ucp_say "share $k/$RES_UCP_N of the conversion of step $N failed on ${RES_UCP[$k]} with exit code ${rcs[$k]} (the last lines of its log: $(tail -2 "${logs[$k]}" | tr '\n' ' ' | head -c 240)); it is retried once on the other worker $other."
      res_ucp_run_share "$N" "$k" "$other" "${logs[$k]}.retry"; rcs[$k]=$?
      if [ "${rcs[$k]}" != 0 ]; then
        res_ucp_say "share $k/$RES_UCP_N of the conversion of step $N failed again on $other (exit code ${rcs[$k]}): the save of step $N is left unconverted (its ZeRO folder is kept by the retention) and is not tried again in this job."
        res_ucp_rec "the save of step $N could not be converted: share $k failed on ${RES_UCP[$k]} and then on $other (data/ucp/convert_s${N}_share$k.log and .retry)"
        echo "$N" >> "$RES_DIR/ucp/failed_steps"
      fi
    fi
  done
  t1=$(date +%s)
  if res_ucp_complete "$N"; then
    n_atoms=$(res_ucp_atoms "$N"); result=complete
    # 'latest_universal' only ADVANCES and the LOOP OWNS IT (2026-10-10, job 5646628): ds_to_universal itself writes 'latest_universal'
    # with the step it converted, so a catch-up of an older save moved the tag backwards (it named step 4 after steps 24, 20, 16, 12
    # and 4 had been converted in that order); after every conversion the tag is set to the NEWEST complete universal
    local cur_lu newest_lu s_
    cur_lu=$(sed 's/global_step//; s/_universal//' "$RES_CKPT_DIR/latest_universal" 2>/dev/null); cur_lu=${cur_lu:-0}
    newest_lu=0
    for s_ in $(ls -d "$RES_CKPT_DIR"/global_step*_universal 2>/dev/null | sed 's/.*global_step//; s/_universal//' | sort -n); do
      res_ucp_complete "$s_" && newest_lu=$s_
    done
    [ "$N" -gt "$newest_lu" ] && newest_lu=$N
    echo "global_step${newest_lu}_universal" > "$RES_CKPT_DIR/latest_universal.tmp" && mv -f "$RES_CKPT_DIR/latest_universal.tmp" "$RES_CKPT_DIR/latest_universal"
    if [ "$newest_lu" = "$N" ]; then
      lu_txt="'latest_universal' now names global_step${N}_universal"
    else
      lu_txt="'latest_universal' names global_step${newest_lu}_universal, the newest complete universal (this was a catch-up of an older save; the converter had written the tag as global_step${cur_lu}_universal and the loop set it back)"
    fi
    res_ucp_say "the universal of step $N is complete: $n_atoms atoms, zero/optimizer_state.pt present, no tmp folder, the finalize lock exists; $lu_txt; the conversion took $(( (t1 - t0 + 30) / 60 )) min ($((t1 - t0)) s) on $RES_UCP_N workers (tries per share: $(echo ${tries[*]} | tr ' ' ',' ))."
    res_ucp_rec "step $N converted to the universal form in $((t1 - t0)) s on ${RES_UCP[*]} ($n_atoms atoms); $lu_txt"
  else
    n_atoms=$(res_ucp_atoms "$N"); result=incomplete
    grep -q -x "$N" "$RES_DIR/ucp/failed_steps" 2>/dev/null || echo "$N" >> "$RES_DIR/ucp/failed_steps"
    res_ucp_say "the universal of step $N is NOT complete after $((t1 - t0)) s (atoms $n_atoms, optimizer_state.pt $( [ -f "$RES_CKPT_DIR/global_step${N}_universal/zero/optimizer_state.pt" ] && echo present || echo missing ), tmp folders $(ls -d "$RES_CKPT_DIR/global_step${N}_universal"/tmp* 2>/dev/null | wc -l), finalize lock $( [ -d "$RES_CKPT_DIR/global_step${N}_universal/shards/finalize.lock" ] && echo present || echo missing )); 'latest_universal' is not advanced and the save of step $N is kept."
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$N" "$(date -d @$t0 '+%F %H:%M:%S')" "$(date -d @$t1 '+%F %H:%M:%S')" "$((t1 - t0))" "$RES_UCP_N" "$(echo ${RES_UCP[*]} | tr ' ' ',')" "$(echo ${rcs[*]} | tr ' ' ',')" "$n_atoms" "$result" >> "$RES_DIR/ucp_conversions.tsv"
  res_retention_ucp
  res_ucp_state "$(res_ucp_state_sentence "$N" "$result" "$t1" "$t0")"
}
res_ucp_state_sentence() {   # $1 the step just converted, $2 its result, $3 its end epoch, $4 its start epoch: the STATUS.md line, from the NEWEST complete universal
  local N=$1 result=$2 t1=$3 t0=$4 newest row nt ns skipped nxt
  newest=$(awk -F'\t' '!/^#/ && $1 ~ /^[0-9]+$/ && $9=="complete" {if ($1+0 > m) m=$1+0} END{print m+0}' "$RES_DIR/ucp_conversions.tsv" 2>/dev/null)
  if [ "${newest:-0}" -gt 0 ]; then
    row=$(awk -F'\t' -v n=$newest '!/^#/ && $1==n && $9=="complete" {s=$0} END{print s}' "$RES_DIR/ucp_conversions.tsv")
    nt=$(echo "$row" | cut -f3 | cut -c12-16); ns=$(echo "$row" | cut -f4)
    nxt=$(( newest + ${SAVE_INTERVAL:-0} ))
    echo -n "step $newest universal complete $nt, $RES_UCP_N workers, $(( (ns + 30) / 60 )) min; next: $( [ "${SAVE_INTERVAL:-0}" -gt 0 ] && echo "step $nxt (when its save is complete)" || echo 'the next complete save' )"
    [ "$N" != "$newest" ] && echo -n "; the older save of step $N was $( [ "$result" = complete ] && echo "caught up at $(date -d @$t1 '+%H:%M') ($(( (t1 - t0 + 30) / 60 )) min)" || echo "tried at $(date -d @$t1 '+%H:%M') and is NOT complete" )"
  else
    echo -n "no complete universal yet; the conversion of step $N ended $(date -d @$t1 '+%H:%M') as $result ($(( (t1 - t0 + 30) / 60 )) min, $RES_UCP_N workers)"
  fi
  skipped=$(res_ucp_steps_unconverted | awk -v n=${newest:-0} '$1<n' | tr '\n' ' ')
  [ -n "$skipped" ] && echo -n "; older saves still to catch up: ${skipped% }"
  echo
}
res_ucp_watch() {   # the background loop: every 15 s, the newest complete save without a universal is converted (one at a time); stops on data/ucp/stop
  local N last_seen="" lt skipped
  res_ucp_state "no conversion yet: the watcher waits for the first complete save ('latest' names step $(res_latest_step)); the $RES_UCP_N worker nodes are ${RES_UCP[*]}."
  while [ ! -f "$RES_DIR/ucp/stop" ]; do
    lt=$(res_latest_step)
    if [ -n "$last_seen" ] && [ "$lt" != "$last_seen" ] && [ "$lt" -gt 0 ]; then
      skipped=$(res_ucp_steps_unconverted | awk -v n=$lt '$1<n' | tr '\n' ' ')
      [ -n "$skipped" ] && res_ucp_say "the save of step $lt completed while the converter was busy or idle with older saves; the older unconverted saves (steps ${skipped% }) are skipped for now and caught up after the newest."
    fi
    last_seen=$lt
    N=$(res_ucp_steps_unconverted | head -1)       # the newest complete save without a complete universal
    if [ -n "$N" ]; then
      if [ -f "$RES_DIR/ucp/finish" ] && [ $(( ${RES_END_EPOCH:-0} - $(date +%s) )) -lt ${RES_UCP_MIN_LEFT_S:-240} ]; then
        res_ucp_say "the conversion of step $N is not started: only $(( ${RES_END_EPOCH:-0} - $(date +%s) )) s of the wall are left and a conversion needs at least ${RES_UCP_MIN_LEFT_S:-240} s; the save of step $N stays unconverted."
        break
      fi
      res_ucp_convert "$N"
      continue                                     # look again at once: a newer save may have completed meanwhile
    fi
    [ -f "$RES_DIR/ucp/finish" ] && break          # the training is over and nothing is left to convert
    sleep 15
  done
  res_ucp_say "the watcher of the UCP worker duty stopped ($( [ -f "$RES_DIR/ucp/stop" ] && echo 'asked to stop by the loop' || echo 'nothing left to convert after the training' )); $(grep -c -E '^[0-9]' "$RES_DIR/ucp_conversions.tsv" 2>/dev/null || echo 0) conversion(s) were recorded in data/ucp_conversions.tsv."
}
res_ucp_start() {
  mkdir -p "$RES_DIR/ucp"; : > "$RES_DIR/ucp/srun_pids"
  [ -f "$RES_DIR/ucp_conversions.tsv" ] || printf '# ucp_conversions.tsv: one line per conversion by the UCP workers. Columns: step, start, end, seconds, workers, worker_nodes, share_exit_codes, atoms, result\nstep\tstart\tend\tseconds\tworkers\tworker_nodes\tshare_rcs\tatoms\tresult\n' > "$RES_DIR/ucp_conversions.tsv"
  res_ucp_watch &
  RES_UCP_PID=$!
  res_ucp_say "the UCP worker duty started (watcher process id $RES_UCP_PID): the $RES_UCP_N worker node(s) ${RES_UCP[*]} were taken from the end of the allocation, never train and are never swapped in; every complete save is converted to the universal form, one conversion at a time; the retention keeps the newest 2 ZeRO saves and the newest 8 universals plus the steps in ${RES_KEEP_FILE:-$RES_CKPT_DIR/keep_steps.txt} and deletes an older ZeRO save only when its universal is complete."
  res_ucp_rec "the UCP worker duty: $RES_UCP_N worker node(s) ${RES_UCP[*]}, converting every complete save (data/ucp_conversions.tsv, data/ucp/)"
}
res_ucp_finish() {   # the training is over: let the watcher convert what is left while the wall allows, then stop it
  [ -n "${RES_UCP_PID:-}" ] || return 0
  touch "$RES_DIR/ucp/finish"
  local left=$(( ${RES_END_EPOCH:-0} - $(date +%s) )) budget t0 pid
  budget=$(( left - 90 )); [ $budget -lt 0 ] && budget=0
  res_ucp_say "the training is over; the watcher may finish the running conversion and catch up the unconverted saves for up to $budget s (the $left s left before the end time minus 90 s), then it is stopped."
  t0=$(date +%s)
  while kill -0 $RES_UCP_PID 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt $budget ]; do sleep 5; done
  if kill -0 $RES_UCP_PID 2>/dev/null; then
    res_ucp_say "the wall is close: the watcher is stopped now; a conversion still running is killed and its save stays unconverted (a partial universal folder may be left; the completeness checks reject it)."
    touch "$RES_DIR/ucp/stop"
    for pid in $(cat "$RES_DIR/ucp/srun_pids" 2>/dev/null); do kill $pid 2>/dev/null; done
    sleep 3; kill $RES_UCP_PID 2>/dev/null
  fi
  wait $RES_UCP_PID 2>/dev/null
  res_retention_ucp
  # the aggregator (which writes STATUS.md every 10 s) has stopped with the training, so the conversion line it last wrote is stale:
  # rewrite that one line from data/ucp_state (job 5646628: RESULTS.md quoted "the conversion of step 8 ... is running" after 6 had finished)
  if [ -f "$RES_JOB_DIR/STATUS.md" ] && [ -f "$RES_DIR/ucp_state" ]; then
    python3 - "$RES_JOB_DIR/STATUS.md" "$RES_DIR/ucp_state" <<'PY' 2>/dev/null || true
import sys, re
p, q = sys.argv[1], sys.argv[2]
txt = open(p).read(); line = open(q).read().strip().splitlines()[-1] if open(q).read().strip() else ''
if line:
    new = 'conversion     ' + line
    txt2, n = re.subn(r'^conversion     .*$', new.replace('\\', '\\\\'), txt, count=1, flags=re.M)
    open(p, 'w').write(txt2 if n else txt.rstrip('\n') + '\n' + new + '\n')
PY
  fi
  res_ucp_rec "at the end: $(grep -c -E '\tcomplete$' "$RES_DIR/ucp_conversions.tsv" 2>/dev/null || echo 0) complete universal(s); latest_universal names $(cat "$RES_CKPT_DIR/latest_universal" 2>/dev/null || echo nothing); the ZeRO saves left: $(ls -d "$RES_CKPT_DIR"/global_step*[0-9] 2>/dev/null | sed 's/.*global_step//' | sort -n | tr '\n' ' ')"
}
res_retention_ucp() {   # keep the newest 2 ZeRO saves, the newest 8 universals, the keep-file's steps; delete an older ZeRO save only when its universal is complete
  local d="$RES_CKPT_DIR" lt keep=() s zs us k kf="${RES_KEEP_FILE:-$RES_CKPT_DIR/keep_steps.txt}"
  [ -n "$d" ] && [ -f "$d/latest" ] || return 0
  case "$(readlink -f "$d")" in /lustre/orion/gen150/scratch/zixianw4/X-MoE-resilience/*) ;; *) return 0;; esac
  lt=$(res_latest_step)
  [ -f "$kf" ] && keep=($(grep -E '^[0-9]+$' "$kf"))
  zs=$(ls -d "$d"/global_step*[0-9] 2>/dev/null | sed 's/.*global_step//' | grep -E '^[0-9]+$' | sort -n)
  us=$(ls -d "$d"/global_step*_universal 2>/dev/null | sed 's/.*global_step//; s/_universal//' | grep -E '^[0-9]+$' | sort -n)
  local newest2; newest2=$(echo "$zs" | awk -v l=$lt '$1<=l' | tail -2 | tr '\n' ' ')
  for s in $zs; do
    [ "$s" -lt "$lt" ] || continue                               # 'latest' and anything newer (a save in progress) stay
    case " $newest2 ${keep[*]} " in *" $s "*) continue;; esac     # the newest 2 and the keep-file's steps stay
    if res_ucp_complete "$s"; then
      rm -rf "$d/global_step$s" && res_ucp_say "retention: the ZeRO save of step $s was deleted because its universal is complete and newer saves exist (the newest 2 kept: ${newest2% }; the keep-file steps: ${keep[*]:-none})."
    else
      grep -q -x "zero_kept_$s" "$RES_DIR/ucp/retention_said" 2>/dev/null || { echo "zero_kept_$s" >> "$RES_DIR/ucp/retention_said"; res_ucp_say "retention: the ZeRO save of step $s is older than the newest 2 but is KEPT because its universal is not complete yet."; }
    fi
  done
  k=$(echo "$us" | grep -c .)
  for s in $us; do
    [ $k -gt 8 ] || break
    case " ${keep[*]} " in *" $s "*) continue;; esac
    rm -rf "$d/global_step${s}_universal" && res_ucp_say "retention: the universal of step $s was deleted (more than 8 universals; the keep-file steps stay: ${keep[*]:-none})."
    k=$((k - 1))
  done
}
# ================================================== end of the UCP worker duty ==================================================

# [2026-10-10, round 6] the OOM re-configuration rule, the memory gate and the memory profiler live in oom_reconfig.sh (the rule itself
# is oom_reconfig.py); every function there returns at once unless AUTORUN_OOM_RECONFIG / AUTORUN_MEM_GATE / AUTORUN_MEM_PROFILER is 1,
# so a job without the flags runs exactly as before.
source "$RES_BIN/oom_reconfig.sh"

res_main() {
  export RES_JOB_DIR="$PWD/$JOB_DIR"; RES_DIR="$RES_JOB_DIR/data"; export RES_DATA_DIR="$RES_DIR"; RES_LOOP_LOG="$RES_DIR/loop.log"
  export RES_LOOP_LOG
  mkdir -p "$RES_DIR/inject" "$RES_DIR/hb" "$RES_DIR/alarm" "$RES_DIR/gate" "$RES_JOB_DIR/attempts" "$RES_JOB_DIR/alarms"
  export RES_BIN XMOE_HB=${XMOE_HB:-1} XMOE_DETECT=${XMOE_DETECT:-1}
  eval "$($RES_PY "$RES_BIN/exit_codes.py" --bash)"
  RES_GROUP=${EP_PARALLEL_SIZE:-8}; RES_T_INIT_S=${XMOE_T_INIT_S:-600}; RES_BATCH=$(hostname -s)
  RES_ALL=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
  RES_E_FILE="$RES_JOB_DIR/a-xmoe-${RUN_TYPE}-${SLURM_JOB_ID}.e"
  RES_CKPT_DIR="${CHECKPOINT_PATH:-}"; [ -n "$RES_CKPT_DIR" ] && RES_CKPT_DIR=$(readlink -f "$RES_CKPT_DIR" 2>/dev/null || echo "$RES_CKPT_DIR")
  # [2026-10-10] the UCP worker nodes: the LAST n nodes of the allocation (after the spares); they leave RES_ALL, so the gate, the
  # training set and the spares never see them
  RES_UCP_N=${AUTORUN_UCP_WORKERS:-0}; RES_UCP=(); RES_UCP_PID=""
  if [ "$RES_UCP_N" -gt 0 ]; then
    if [ $(( ${#RES_ALL[@]} - NODES - RES_UCP_N )) -lt 0 ]; then
      res_loud "AUTORUN_UCP_WORKERS=$RES_UCP_N but the allocation has only ${#RES_ALL[@]} nodes for $NODES training nodes: no worker node is taken, the duty is off"
      RES_UCP_N=0
    else
      RES_UCP=("${RES_ALL[@]: -$RES_UCP_N}")
      RES_ALL=("${RES_ALL[@]:0:$(( ${#RES_ALL[@]} - RES_UCP_N ))}")
    fi
  fi
  local N=$NODES nall=${#RES_ALL[@]}
  RES_EXCLUDED=(); RES_DEGRADED=(); RES_READMIT=(); RES_ATTEMPT=0; RES_TRAIN=(); RES_TRAIN_CSV=""; RES_DUTY=""; RES_AGG_ON=""; RES_NEXT=""; RES_E_OFF=0; RES_SWAP_NOTE=""
  export RANK_LOG_DIR="$RES_JOB_DIR/attempts/attempt_0/ranks" RUN_LOG_DIR="$RES_JOB_DIR/attempts/attempt_0"
  # [2026-10-10] the job's end time, from the same source as the trainer's wall-aware exit and the timeout handler: SLURM_JOB_END_TIME
  # (the epoch Slurm exports into the job), else `scontrol show job` EndTime once. RES_MIN_ATTEMPT_S (no restart when too little
  # wall is left) is measured against it.
  if [ -n "${SLURM_JOB_END_TIME:-}" ]; then RES_END_EPOCH=$SLURM_JOB_END_TIME; RES_END_SRC="SLURM_JOB_END_TIME"
  else RES_END_EPOCH=$(date -d "$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -o -E 'EndTime=[0-9T:-]+' | head -1 | cut -d= -f2)" +%s 2>/dev/null || echo 0); RES_END_SRC="scontrol show job EndTime (SLURM_JOB_END_TIME was not in the environment)"; fi
  RES_WALL_NOTE=""
  printf 'attempt\tnodes\tstart\tfirst_iter\tend\trc\tcode\tclass\tnode\trank\tstep\tresumed_from\n' > "$RES_DIR/attempts.tsv"
  printf 'attempt\tcase\tinjected\tinject_time\texit_time\tfirst_iter_after\tcode\tclass\tnode\trank\tstep\texcluded\tevidence\n' > "$RES_DIR/failures.tsv"
  res_readme
  { echo "# training.log of job $SLURM_JOB_ID: one line per step across attempts (design 3.11). The columns of a step line are:"
    echo "# time | attempt (0 = the first run; k = the run after the k-th failure) | step | loss (the logged lm loss) | TFLOP/s per GCD (from"
    echo "# the throughput calculator) | seconds per step | HBM max GB (the maximum over all ranks of torch.cuda.max_memory_allocated) | note."
    echo "# Each attempt opens with a header block of lines starting with '#', written by the job script. A column-name row (time | att |"
    echo "# step | loss | TFLOP/s/GCD | step s | HBM max GB | note) precedes the first step line of every attempt and repeats every 25 step lines."; } > "$RES_JOB_DIR/training.log"
  [ -f "$RES_JOB_DIR/alarms.log" ] || { echo "# alarms.log of job $SLURM_JOB_ID: one line per alarm event, written by the heartbeat aggregator; the time, the rule (stillness or"
    echo "# minority), the rank and node named, its phase and collective count, how old its line was, whether it is in a save window, and"
    echo "# the stack sample taken (the folder alarms/); a second line when the alarm clears, or when it became a failure (recovery.log)."; } > "$RES_JOB_DIR/alarms.log"
  { echo "# recovery.log of job $SLURM_JOB_ID: one block per failure, in words; the RECOVERED banner follows the block when the next"
    echo "# attempt shows its first iteration line. Attempt k = attempts/attempt_k (attempt 0 = the first run)."; echo; } > "$RES_JOB_DIR/recovery.log"
  res_say "resilient job $SLURM_JOB_ID started: Slurm allocated $(( nall + RES_UCP_N )) nodes, of which $N train and $((nall - N)) are spares (${RES_ALL[*]})$( [ "$RES_UCP_N" -gt 0 ] && echo " and $RES_UCP_N are UCP worker nodes taken from the end of the allocation (${RES_UCP[*]}), which never train and are not swapped in" ); the batch script runs on $RES_BATCH."
  if [ ${RES_END_EPOCH:-0} -gt 0 ]; then
    res_say "the job's end time is $(date -d @$RES_END_EPOCH '+%F %H:%M:%S') ($(( RES_END_EPOCH - $(date +%s) )) s from now; source: $RES_END_SRC); the trainer's wall-aware exit is $( [ "${XMOE_WALL_AWARE_EXIT:-0}" = 1 ] && echo "ON (XMOE_WALL_AWARE_EXIT=1: the trainer exits by itself when one more step and its save would not fit before this end time; --exit-duration-in-mins is only the fallback)" || echo "off (XMOE_WALL_AWARE_EXIT unset: the trainer exits by --exit-duration-in-mins as before)"); no restart is made when the time left cannot cover a start-up and one step (this job's own longest start-up to the first step plus ${RES_RESTART_MARGIN_S:-60} s; ${RES_MIN_ATTEMPT_S:-420} s until an attempt has finished a step)."
  else
    res_say "the job's end time is NOT known (neither SLURM_JOB_END_TIME nor scontrol gave one): the loop cannot refuse a late restart, and the trainer's wall-aware exit (if on) falls back to --exit-duration-in-mins."
  fi
  echo "${XMOE_INJECT_SCHEDULE:-}" > "$RES_DIR/schedule.txt"
  res_say "the flags of this job are: AUTORUN_UCP_WORKERS=${AUTORUN_UCP_WORKERS:-0} AUTORUN_SPARES=${AUTORUN_SPARES:-auto} AUTORUN_GATE=${AUTORUN_GATE:-auto} AUTORUN_GATE_CALIBRATE=${AUTORUN_GATE_CALIBRATE:-0} AUTORUN_GATE_STRICT=${AUTORUN_GATE_STRICT:-0} AUTORUN_USE_DEGRADED=${AUTORUN_USE_DEGRADED:-1} GATE_EXCLUDE_SUSPECT_PAIR=${GATE_EXCLUDE_SUSPECT_PAIR:-0} GATE_RDMA_PHASE=${GATE_RDMA_PHASE:-0} XMOE_PG_TIMEOUT_MIN=${XMOE_PG_TIMEOUT_MIN:-unset} XMOE_T_INIT_S=$RES_T_INIT_S XMOE_INJECT_SCHEDULE=${XMOE_INJECT_SCHEDULE:-none} XMOE_GATE_INJECT=${XMOE_GATE_INJECT:-none} RES_READMIT_INJECTED=${RES_READMIT_INJECTED:-0} AUTORUN_LOCAL_COPIES=${AUTORUN_LOCAL_COPIES:-0} DS_ASYNC_PINNED_CHECKPOINT=${DS_ASYNC_PINNED_CHECKPOINT:-unset} XMOE_WALL_AWARE_EXIT=${XMOE_WALL_AWARE_EXIT:-0} (margins: XMOE_WALL_MARGIN_FRAC=${XMOE_WALL_MARGIN_FRAC:-0.1} XMOE_WALL_MARGIN_S=${XMOE_WALL_MARGIN_S:-120} XMOE_SAVE_EST_S=${XMOE_SAVE_EST_S:-120})."
  $RES_PY -c "import deepspeed, megatron; print('[resilience] deepspeed is imported from', deepspeed.__file__); print('[resilience] megatron is imported from', megatron.__file__)" 2>/dev/null | grep -a "^\[resilience\]" | tee -a "$RES_LOOP_LOG"
  if [ "${AUTORUN_LOCAL_COPIES:-0}" != 0 ] && [ "${DS_ASYNC_PINNED_CHECKPOINT:-0}" != 1 ]; then
    res_loud "AUTORUN_LOCAL_COPIES=${AUTORUN_LOCAL_COPIES} needs the asynchronous checkpoint engine (DS_ASYNC_PINNED_CHECKPOINT=1): no local copy will be made"
  fi
  res_gate_inject_resolve
  res_shm_clean start
  $RES_PY "$RES_BIN/hb_aggregator.py" --job-dir "$RES_JOB_DIR" --compose
  # G0 the lemons of the global record
  local lem; lem=$($RES_PY "$RES_BIN/failure_record.py" lemons --days 30 --record "$RES_BIN/../failure_record.tsv" --nodes-only 2>/dev/null)
  for x in $lem; do for n in "${RES_ALL[@]}"; do [ "$n" = "$x" ] && RES_EXCLUDED+=("$x") && res_say "the lemon list of the shared failure record names $x (a hard failure, or two soft ones, in the last 30 days): it is excluded before the start."; done; done
  res_init_load gate   # [2026-10-13] AUTORUN_INIT_LOAD_DIR: the first attempt's --load/--finetune before the gate's dry run (no-op when unset)
  # G1 the gate (gate.log at the top; its raw files in data/gate/)
  local gm=${AUTORUN_GATE:-auto}; [ "$gm" = auto ] && { [ $N -ge 32 ] && gm=full || gm=quick; }
  if [ "$gm" != 0 ]; then
    res_state gate "the gate ($gm) is running on $(echo $(res_good_nodes) | wc -w) nodes; its summary goes to gate.log"
    GATE_SRUN_EXTRA="${GATE_SRUN_EXTRA:---gpus-per-node=8}" GATE_SPARES=$(( nall - N )) bash "$RES_BIN/gate.sh" --mode $gm --nodes "$(res_good_nodes)" --job-dir "$RES_JOB_DIR" \
      --dry-run-cmd "$(xmoe_dry_run_cmd)" --python "$RES_PY" --spares $(( nall - N )) $( [ "${AUTORUN_GATE_CALIBRATE:-0}" = 1 ] && echo --calibrate )
    local grc=$?
    if [ $grc = 2 ]; then
      if [ "${RES_DRYRUN_STRICT:-0}" = 1 ]; then res_mail "X-MoE job $SLURM_JOB_ID stopped at the gate" "the dry run failed: see data/gate/dryrun.log"; res_state done "none: the job stopped at the gate because the dry run failed (RES_DRYRUN_STRICT=1)"; return 2; fi
      res_loud "THE GATE'S DRY RUN FAILED (data/gate/dryrun.log: $(grep -a -m1 'dry-run\] FAIL' "$RES_DIR/gate/dryrun.log" | head -c 300)); RES_DRYRUN_STRICT=0, so the run continues (set RES_DRYRUN_STRICT=1 to stop here)"
    fi
    if [ $grc = 4 ]; then
      if [ "${AUTORUN_GATE_STRICT:-0}" = 1 ]; then res_mail "X-MoE job $SLURM_JOB_ID stopped at the gate" "a TEST DEFECT (gate.log) and AUTORUN_GATE_STRICT=1"; res_state done "none: the job stopped at the gate because of a TEST DEFECT (AUTORUN_GATE_STRICT=1)"; return 4; fi
      res_loud "THE GATE FOUND TEST DEFECT(S) OR TESTS NOT LAUNCHED: $(grep -a -m1 '^TEST DEFECTS:' "$RES_JOB_DIR/gate.log" | head -c 300); $(grep -a -m1 '^NOT LAUNCHED:' "$RES_JOB_DIR/gate.log" | head -c 300); no node is excluded for them; the job continues (AUTORUN_GATE_STRICT=0)"
    fi
    local gf; gf=$(cat "$RES_DIR/gate/failed_nodes.txt" 2>/dev/null)
    local ngf; ngf=$(echo $gf | wc -w)
    if [ "$ngf" -gt 0 ] && [ "$ngf" = "$(echo $(res_good_nodes) | wc -w)" ]; then
      res_loud "THE GATE WOULD EXCLUDE EVERY NODE: that is a test or environment problem, not a node problem; no node is excluded (read gate.log)"
    else
      res_apply_gate_exclusions
    fi
  elif [ -f "$RES_DIR/gate/excluded_reasons.tsv" ]; then
    # TEST HOOK (the mock loop test): with the gate off, a pre-written excluded_reasons.tsv is applied as if the gate had written it
    res_say "the gate is off (AUTORUN_GATE=0) but data/gate/excluded_reasons.tsv exists: its exclusions are applied as the gate's (a test hook)."
    res_apply_gate_exclusions
  fi
  local good=($(res_good_nodes))
  if [ ${#good[@]} -lt $N ] && [ "${AUTORUN_USE_DEGRADED:-1}" = 1 ]; then
    while [ ${#good[@]} -lt $N ] && res_use_degraded; do good=($(res_good_nodes)); done
  fi
  if [ ${#good[@]} -lt $N ]; then res_mail "X-MoE job $SLURM_JOB_ID: doomed run" "fewer than $N good nodes are left after the gate: ${good[*]}"; res_state done "none: the run is doomed (fewer good nodes than $N after the gate)"; return 3; fi
  res_mem_gate   # [round 6] the memory model's peak per stage and the heatmap colour into gate.log; a red layout gets a WARNING and the run proceeds
  declare -A RETRIES; local rc pending_banner="" t_exit=0 resumed nfail=0
  [ "$RES_UCP_N" -gt 0 ] && [ -n "$RES_CKPT_DIR" ] && res_ucp_start
  RES_ATTEMPT=-1
  local prev_train=("${RES_ALL[@]:0:$N}")     # the nodes that would train with no exclusion: a node outside this set ENTERS the training set
  while true; do
    RES_ATTEMPT=$((RES_ATTEMPT + 1)); export RES_ATTEMPT
    RES_SWAP_NOTE=""
    good=($(res_good_nodes))
    if [ ${#good[@]} -lt $N ] && [ "${RES_READMIT_INJECTED:-0}" = 1 ]; then   # TEST MODE: a node excluded for an injected fault is healthy
      while [ ${#good[@]} -lt $N ] && [ ${#RES_READMIT[@]} -gt 0 ]; do
        local back=${RES_READMIT[0]}; RES_READMIT=("${RES_READMIT[@]:1}")
        local tmp=(); for x in "${RES_EXCLUDED[@]}"; do [ "$x" != "$back" ] && tmp+=("$x"); done; RES_EXCLUDED=("${tmp[@]}")
        res_loud "TEST MODE (RES_READMIT_INJECTED=1): $back is re-admitted because its exclusion came from an INJECTED fault and the node is healthy; no spare was left"
        res_rec "  test mode: $back was re-admitted for attempt $RES_ATTEMPT (its exclusion came from an injected fault; no spare was left)"
        good=($(res_good_nodes))
      done
    fi
    # the spare check on every node that enters the training set; a node that fails is excluded and the next one is tried; when
    # no healthy spare is left, the best degraded node is the last resort (AUTORUN_USE_DEGRADED=1)
    while true; do
      good=($(res_good_nodes))
      if [ ${#good[@]} -lt $N ]; then
        [ "${AUTORUN_USE_DEGRADED:-1}" = 1 ] && res_use_degraded && continue
        break
      fi
      local cand=("${good[@]:0:$N}") entering=() failed_one=0 c p seen
      for c in "${cand[@]}"; do seen=0; for p in "${prev_train[@]}"; do [ "$c" = "$p" ] && seen=1; done; [ $seen = 0 ] && entering+=("$c"); done
      for c in "${entering[@]}"; do
        if ! res_spare_check "$c"; then failed_one=1; break; fi
      done
      [ $failed_one = 0 ] && break
    done
    if [ ${#good[@]} -lt $N ]; then
      # [2026-10-11, the user's rule] no spare node is left: with AUTORUN_SHRINK=1 the memory model chooses the largest configuration that
      # fits on the healthy nodes (oom_reconfig.sh res_shrink_handle lowers N, NODES and TOTAL_GPUS and rewrites the layout); else stop as before
      if res_shrink_on && [ ${#good[@]} -gt 0 ] && res_shrink_handle "${#good[@]}"; then
        RES_SWAP_NOTE="no spare node was left, so attempt $RES_ATTEMPT runs on $N of the ${#good[@]} healthy nodes with $(res_layout_text)"
      else
        res_shrink_on || res_mail "X-MoE job $SLURM_JOB_ID stopped" "no spare is left: ${#good[@]} good nodes for $N; excluded ${RES_EXCLUDED[*]}"
        break
      fi
    fi
    RES_TRAIN=("${good[@]:0:$N}"); local spares=("${good[@]:$N}")
    prev_train=("${RES_TRAIN[@]}")
    RES_TRAIN_CSV=$(echo ${RES_TRAIN[*]} | tr ' ' ',')
    local duty=${spares[0]:-$RES_BATCH}
    # this attempt's folder: the rank logs, the recorder dumps and the exit records are written here directly
    RES_ATT_DIR="$RES_JOB_DIR/attempts/attempt_$RES_ATTEMPT"
    mkdir -p "$RES_ATT_DIR/ranks" "$RES_ATT_DIR/nccl_trace" "$RES_ATT_DIR/exit"
    export RANK_LOG_DIR="$RES_ATT_DIR/ranks" RUN_LOG_DIR="$RES_ATT_DIR" RES_EXIT_DIR="$RES_ATT_DIR/exit"
    export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$RES_ATT_DIR/nccl_trace/nccl_trace_rank_"
    if [ "$duty" != "${RES_AGG_ON:-}" ]; then
      [ -n "${RES_AGG_ON:-}" ] && res_say "duty hand-over: the duty holder ${RES_AGG_ON} is $( for x in "${RES_TRAIN[@]}"; do [ "$x" = "$RES_AGG_ON" ] && echo 'now a training node'; done ), so the duty (the heartbeat aggregator) moves to $duty."
      RES_DUTY=$duty; res_state running; res_agg_start "$duty"
    fi
    RES_DUTY=$duty
    resumed=$(res_latest_step)
    RES_E_OFF=$(stat -c %s "$RES_E_FILE" 2>/dev/null || echo 0)
    res_state running "the loop watches attempt $RES_ATTEMPT; on a failure it classifies it, names the node, excludes it when the class says so, and restarts on the good nodes (spares left: ${#spares[@]}, degraded kept as a last resort: ${#RES_DEGRADED[@]})"
    local t0; t0=$(date +%s)
    res_tl "#"
    res_tl "# ===== attempt $RES_ATTEMPT started at $(date '+%F %H:%M:%S'): $( [ $RES_ATTEMPT = 0 ] && echo 'the first run' || echo "restart $RES_ATTEMPT after failure $RES_ATTEMPT (its block is in recovery.log)") ====="
    res_tl "# this attempt trains on $N nodes (${RES_TRAIN[*]}); the spares are ${spares[*]:-none}; the duty node is $duty; the excluded nodes are ${RES_EXCLUDED[*]:-none}$( [ ${#RES_DEGRADED[@]} -gt 0 ] && echo " (degraded, kept as a last resort: $(echo ${RES_DEGRADED[*]} | tr ' ' ',' | sed 's/:[^,]*//g'))"); it resumes from the save of step $resumed."
    [ -n "$RES_SWAP_NOTE" ] && res_tl "# node change: $RES_SWAP_NOTE."
    res_tl "# the environment (the gate's check, first node): $(res_versions)."
    res_tl "# NCCL warnings so far: $(res_nccl_warn)."
    [ $RES_ATTEMPT -gt 0 ] && res_tl "# GPU-direct in the previous attempt: $(res_gdr "$RES_JOB_DIR/attempts/attempt_$((RES_ATTEMPT-1))")."
    res_tl "# the rank logs of this attempt are in $(res_rel "$RES_ATT_DIR")/ranks/."
    res_say "attempt $RES_ATTEMPT started: srun on $N training nodes (${RES_TRAIN[*]}); the spares are ${spares[*]:-none}; the duty node is $duty; it resumes from the save of step $resumed; the rank logs go to $(res_rel "$RANK_LOG_DIR")."
    res_init_load attempt   # [2026-10-13] AUTORUN_INIT_LOAD_DIR: the initial weights until the run's own first save, then the save (no-op when unset)
    res_layout_record   # [round 6] the layout of this attempt into data/attempt_layouts.tsv (with the memory gate on)
    res_train_launch    # [round 6] xmoe_train_srun in the background (TRAIN_PID); through the memsnap shim when AUTORUN_MEM_PROFILER=1
    local d8="" first="" fi_t="" last_ret=0
    while kill -0 $TRAIN_PID 2>/dev/null; do
      sleep 5
      if [ -z "$first" ] && [ -n "$(res_first_iter)" ]; then
        first=1; fi_t=$(date +%s)
        res_say "attempt $RES_ATTEMPT showed its first iteration line $((fi_t - t0)) s after the srun start ($(grep -a -m1 -E '^ *iteration +[0-9]+/' "$RANK_LOG_DIR/rank_$((TOTAL_GPUS-1)).log" | cut -c1-60 | tr -s ' ' | sed 's/^ //'))."
        res_tl "# attempt $RES_ATTEMPT showed its first iteration line $((fi_t - t0)) s after the srun start; GPU-direct: $(res_gdr "$RES_ATT_DIR")."
        if [ -n "$pending_banner" ]; then
          res_banner "$(echo "$pending_banner" | sed "s/@TIME@/$(date '+%F %H:%M:%S')/; s/@REC@/$((fi_t - t_exit))/; s/@SWAP@/$( [ -n "$RES_SWAP_NOTE" ] && echo " $RES_SWAP_NOTE;" | sed 's/[&/\]/\\&/g')/")"
          sed -i "\$ s/\tPENDING\t/\t$fi_t\t/" "$RES_DIR/failures.tsv"
          pending_banner=""
        fi
        [ $RES_ATTEMPT -gt 0 ] && res_local_copy_lines summary "$RES_ATTEMPT"
      fi
      if [ -z "$first" ] && [ $(( $(date +%s) - t0 )) -gt $RES_T_INIT_S ]; then
        d8=d8; res_say "the init deadline (D8) fired: no iteration line appeared within ${RES_T_INIT_S} s of the srun start, so the step is stopped (the loop counts it as exit code 18, class init_hang)."
        kill -TERM $TRAIN_PID 2>/dev/null; sleep 20; kill -KILL $TRAIN_PID 2>/dev/null; break
      fi
      if [ $(( $(date +%s) - last_ret )) -ge 60 ]; then
        last_ret=$(date +%s); res_retention
        local alive="$RES_DIR/aggregator.alive"
        if [ -f "$alive" ] && [ $(( $(date +%s) - $(stat -c %Y "$alive") )) -gt 120 ]; then
          res_say "the duty: the aggregator's liveness file is $(( $(date +%s) - $(stat -c %Y "$alive") )) s old, so the aggregator is restarted on $RES_DUTY."; res_agg_start "$RES_DUTY"
        elif [ -n "${RES_AGG_PID:-}" ] && ! kill -0 "$RES_AGG_PID" 2>/dev/null; then
          # the duty's process (or its srun) is gone, e.g. the srun failed to start (H16b 5641968: "Unable to confirm allocation")
          res_say "the duty: the aggregator process $RES_AGG_PID is gone ($(tail -1 "$RES_DIR/aggregator.out" 2>/dev/null | head -c 160)), so it is restarted on $RES_DUTY."; res_agg_start "$RES_DUTY"
        fi
        local inj; for inj in "$RES_DIR"/inject/*.fired; do
          [ -f "$inj" ] && [ ! -f "$inj.seen" ] && { touch "$inj.seen"; res_say "$(res_injected_desc "$inj")."; }
        done
      fi
    done
    wait $TRAIN_PID; rc=$?
    [ -n "$d8" ] && rc=18
    t_exit=$(date +%s); res_state restarting "attempt $RES_ATTEMPT ended (srun returned exit code $rc): the loop classifies the failure, then decides the restart"
    [ "$(type -t merge_full_run)" = function ] && merge_full_run "$RANK_LOG_DIR" "$RES_ATT_DIR/full_run.log" >/dev/null 2>&1
    res_heatmap "at the end of attempt $RES_ATTEMPT, whose srun returned exit code $rc"
    res_mem_peaks_line "$RES_ATTEMPT"   # [round 6] the per-rank peaks of the attempt next to the model's prediction (AUTORUN_MEM_PROFILER=1)
    for inj in "$RES_DIR"/inject/*.fired; do [ -f "$inj" ] && [ ! -f "$inj.seen" ] && { touch "$inj.seen"; res_say "$(res_injected_desc "$inj")."; }; done
    if [ $rc = 0 ] && [ -z "$d8" ]; then
      # [2026-10-10] the wall-aware exit (XMOE_WALL_AWARE_EXIT=1) leaves its sentence in data/wall_exit.tsv (rank 0, the last column);
      # when the line belongs to this attempt, the planned exit is named as the wall-aware one, with its step and its arithmetic
      local wl=""; wl=$(awk -F'\t' -v a="$RES_ATTEMPT" '!/^#/ && $2==a {s=$0} END{print s}' "$RES_DIR/wall_exit.tsv" 2>/dev/null)
      if [ -n "$wl" ]; then
        local wstep wsent; wstep=$(echo "$wl" | cut -f3); wsent=$(echo "$wl" | cut -f15)
        RES_WALL_NOTE="the planned exit (wall-aware) at step $wstep: $wsent ($(( RES_END_EPOCH - t_exit )) s were left before the job's end time when the srun returned)"
        res_say "attempt $RES_ATTEMPT ended: srun returned exit code 0 after $((t_exit - t0)) s, which is $RES_WALL_NOTE; the run is done."
        res_tl "# attempt $RES_ATTEMPT ended with the planned exit (wall-aware) at step $wstep at $(date '+%F %H:%M:%S'): $wsent"
        res_state done "none: $RES_WALL_NOTE; RESULTS.md and the analysis follow"
      else
        res_say "attempt $RES_ATTEMPT ended: srun returned exit code 0 after $((t_exit - t0)) s, which is the planned exit; the run is done."
        res_tl "# attempt $RES_ATTEMPT ended with the planned exit (srun returned exit code 0) at $(date '+%F %H:%M:%S')."
      fi
      printf '%s\t%s\t%s\t%s\t%s\t0\t0\tplanned_exit\t-\t-\t-\t%s\n' $RES_ATTEMPT "$RES_TRAIN_CSV" $t0 "${fi_t:--}" $t_exit "$resumed" >> "$RES_DIR/attempts.tsv"
      [ -n "$pending_banner" ] && res_rec "  (attempt $RES_ATTEMPT ended before its first iteration was seen, so no RECOVERED banner was written for the previous failure)"
      break
    fi
    sleep 15   # the aggregator's last beat and the .e tail settle
    res_classify $rc "$d8"
    local cls=${RES_CLASS[$RC_CODE]:-unknown}
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $RES_ATTEMPT "$RES_TRAIN_CSV" $t0 "${fi_t:--}" $t_exit $rc $RC_CODE $cls "$RC_NODE" "$RC_RANK" "$RC_STEP" "$resumed" >> "$RES_DIR/attempts.tsv"
    local ii; ii=($(res_injected_info)); local injected=${ii[0]} icase=${ii[1]}
    local itime="-"; [ "$injected" = yes ] && itime=$($RES_PY -c "
import json,glob,sys
for f in glob.glob(sys.argv[1]+'/inject/*.fired'):
    d=json.load(open(f))
    if str(d['attempt'])==sys.argv[2] and d['case']==sys.argv[3]: print(round(d['time'])); break" "$RES_DIR" "$RES_ATTEMPT" "$icase")
    res_say "attempt $RES_ATTEMPT ended: srun returned exit code $rc after $((t_exit - t0)) s; the failure is classified as $cls (code $RC_CODE); the named node is $RC_NODE (rank $RC_RANK), at step $RC_STEP (from $RC_STEP_SRC); $( [ "$injected" = yes ] && echo "it was an injected fault (case '$icase')$( [ "$itime" != - ] && echo ", detected $((t_exit - itime)) s after the injection fired")" || echo "no injected fault fired in this attempt, so this failure is real or unexplained")."
    res_say "the evidence: $RC_EVID"
    $RES_PY "$RES_BIN/failure_record.py" append --record "$RES_DIR/failure_record.tsv" --record "$RES_BIN/../failure_record.tsv" \
      --job "$SLURM_JOB_ID" --cls "$cls" --code "$RC_CODE" --node "$RC_NODE" --rank "$RC_RANK" --step "$RC_STEP" --injected "$injected" \
      --evidence "attempt $RES_ATTEMPT; $( [ $injected = yes ] && echo "injected $icase; ")$RC_EVID" >> "$RES_LOOP_LOG" 2>&1
    local excl_txt="no node was excluded (the class $cls does not exclude a node)"
    [ "${RES_EXCLUDE[$RC_CODE]:-0}" = 1 ] && [ "$RC_NODE" = "-" ] && excl_txt="no node was excluded (no node could be named)"
    if [ "${RES_EXCLUDE[$RC_CODE]:-0}" = 1 ] && [ "$RC_NODE" != "-" ]; then
      RES_EXCLUDED+=("$RC_NODE"); excl_txt="node $RC_NODE was excluded"
      [ "$injected" = yes ] && RES_READMIT+=("$RC_NODE")
    fi
    RETRIES[$RC_CODE]=$(( ${RETRIES[$RC_CODE]:-0} + 1 ))
    nfail=$((nfail + 1))
    printf '%s\t%s\t%s\t%s\t%s\tPENDING\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $RES_ATTEMPT "$icase" "$injected" "$itime" $t_exit $RC_CODE $cls "$RC_NODE" "$RC_RANK" "$RC_STEP" "$excl_txt" "$(echo "$RC_EVID" | tr '\t' ' ' | head -c 400)" >> "$RES_DIR/failures.tsv"
    # alarms.log: the open alarms of this attempt became this failure
    if [ -f "$RES_DIR/agg_last.json" ] && [ "$($RES_PY -c "import json,sys; t=json.load(open(sys.argv[1])); print(1 if str(t.get('attempt'))==sys.argv[2] and t.get('alarms') else 0)" "$RES_DIR/agg_last.json" "$RES_ATTEMPT" 2>/dev/null)" = 1 ]; then
      echo "$(date '+%F %H:%M:%S') became failure $nfail at $(date -d @$t_exit '+%H:%M:%S'): the open alarm(s) of attempt $RES_ATTEMPT ended in the failure classified as $cls (code $RC_CODE) on node $RC_NODE (rank $RC_RANK); the block is in recovery.log." >> "$RES_JOB_DIR/alarms.log"
    fi
    # recovery.log: the failure in words
    local evf; evf=$(echo "$RC_EVID" | grep -o -E "(exit record |recorder \()[^:)]+" | head -1 | sed -E 's/^(exit record |recorder \()//')
    res_rec "===== failure $nfail at $(date -d @$t_exit '+%F %H:%M:%S'): attempt $RES_ATTEMPT ended after $((t_exit - t0)) s (srun returned exit code $rc) ====="
    res_rec "  what fired:  ${RC_FIRED:-unknown}"
    res_rec "  class:       $cls (code $RC_CODE), which means: ${RES_MEANING[$RC_CODE]:-}"
    res_rec "  named:       rank $RC_RANK on node $RC_NODE; the failure fired at step $RC_STEP (the step is taken from $RC_STEP_SRC)"
    res_rec "  evidence:    the file ${evf:-$(res_rel "$RES_ATT_DIR")/ (the .e file from byte $RES_E_OFF; heartbeat.log; data/agg_last.json)}"
    res_rec "               $(echo "$RC_EVID" | head -c 600)"
    [ "$injected" = yes ] && res_rec "  injected:    yes, the case '$icase' of the schedule fired in this attempt (its marker is in data/inject/); the failure was detected $((t_exit - itime)) s after the injection fired"
    res_rec "  node:        $excl_txt"
    RES_OOM_BANNER=""
    if [ "$RC_CODE" = 11 ] && res_oom_on; then
      # [2026-10-10, round 6] with AUTORUN_OOM_RECONFIG=1 the OOM rule governs the restart after a gpu_oom: one rung of the fixed ladder per
      # out of memory (the variables for xmoe_train_srun are rewritten, the universal checkpoint loaded when the parallelism changed), and
      # the stop with mail when the ladder is exhausted; the class's retry count is not consulted
      res_oom_handle || break
    elif [ ${RETRIES[$RC_CODE]} -gt ${RES_RETRIES[$RC_CODE]:-0} ]; then
      res_rec "  decision:    NO RESTART: the class $cls was seen ${RETRIES[$RC_CODE]} time(s) and only ${RES_RETRIES[$RC_CODE]:-0} restart(s) are allowed for it"; res_rec ""
      res_mail "X-MoE job $SLURM_JOB_ID stopped: $cls repeats" "the class $cls ($RC_CODE) was seen ${RETRIES[$RC_CODE]} times and ${RES_RETRIES[$RC_CODE]:-0} retries are allowed; the last one was on $RC_NODE rank $RC_RANK"
      break
    fi
    local left=$(( ${RES_END_EPOCH:-0} - $(date +%s) )) need; need=$(res_need_attempt_s)
    if [ ${RES_END_EPOCH:-0} -gt 0 ] && [ $left -lt $need ]; then
      res_rec "  decision:    NO RESTART: only $left s of the wall are left, and a restart needs $need s to start and finish one step ($(res_need_attempt_src))"; res_rec ""
      res_loud "not restarting: only $left s of the wall are left, and a restart needs $need s to start and finish one step ($(res_need_attempt_src))"; break
    fi
    resumed=$(res_latest_step)
    # [2026-10-11] after a layout change (the OOM rule or the no-spare shrink) the attempts load the universal checkpoint of the step
    # they changed at; once a save exists under the NEW layout (latest is newer than that step), a restart that keeps the layout loads
    # that native save instead, so no step is replayed
    if [ -n "${RES_UNIV_LOADED_STEP:-}" ] && [ -z "${RES_OOM_BANNER:-}" ] && echo "$megatron_options" | grep -q -- '--universal-checkpoint' && [ "${resumed:-0}" -gt "$RES_UNIV_LOADED_STEP" ]; then
      megatron_options=$(echo "$megatron_options" | sed -E 's/(^| )--universal-checkpoint( |$)/\1\2/g; s/  +/ /g')
      res_rec "  checkpoint:  the next attempt loads the native save of step $resumed, written under the current layout, instead of the universal checkpoint of step $RES_UNIV_LOADED_STEP"
      res_say "the next attempt loads the native save of step $resumed (written under the current layout) instead of the universal checkpoint of step $RES_UNIV_LOADED_STEP."
      unset RES_UNIV_LOADED_STEP
    fi
    if [ -n "${RES_OOM_BANNER:-}" ]; then
      pending_banner="$RES_OOM_BANNER"   # [round 6] the OOM rule wrote its own decision block; its banner names the new layout
    else
    res_rec "  decision:    restart $((RES_ATTEMPT + 1)) inside the same allocation (attempts/attempt_$((RES_ATTEMPT + 1))), resuming from the save of step $resumed; this class has been retried ${RETRIES[$RC_CODE]} time(s) of the ${RES_RETRIES[$RC_CODE]:-0} allowed"
    pending_banner="##### RECOVERED at @TIME@: the failure was a $cls on node $RC_NODE (rank $RC_RANK), which fired at step $RC_STEP; restart $((RES_ATTEMPT + 1)) ran inside the same allocation; $excl_txt;@SWAP@ training continues from the save of step $resumed; the recovery took @REC@ s from the srun exit to the first iteration line #####"
    res_say "restart $((RES_ATTEMPT + 1)) follows: the $cls was handled ($excl_txt; this class has been retried ${RETRIES[$RC_CODE]} time(s) of the ${RES_RETRIES[$RC_CODE]:-0} allowed); the next attempt resumes from the save of step $resumed."
    fi
  done
  res_state done "none: the loop finished after $((RES_ATTEMPT + 1)) attempt(s) with $nfail failure(s)$( [ -n "$RES_WALL_NOTE" ] && echo "; the last attempt ended with $RES_WALL_NOTE" )$( [ "$RES_UCP_N" -gt 0 ] && echo '; the UCP workers finish the conversions that are left' ); RESULTS.md and the analysis follow"
  [ "$RES_UCP_N" -gt 0 ] && res_ucp_finish
  res_agg_stop
  if [ "${AUTORUN_LOCAL_COPIES:-0}" != 0 ]; then
    res_rec "===== the node-local checkpoint copies (AUTORUN_LOCAL_COPIES=${AUTORUN_LOCAL_COPIES}; data/local_copy.tsv) ====="
    res_local_copy_lines saves; res_shm_clean end; res_rec ""
  fi
  $RES_PY "$RES_BIN/results.py" --job-dir "$RES_JOB_DIR" > "$RES_DIR/results.out" 2>&1
  # RESULTS.md is for test jobs only: an injection schedule, a gate injection, or (2026-10-10) a wall-aware exit test (data/wall_exit.tsv)
  [ -s "$RES_DIR/schedule.txt" ] || [ -s "$RES_DIR/gate_inject.txt" ] || [ -s "$RES_DIR/wall_exit.tsv" ] || [ "$RES_UCP_N" -gt 0 ] || [ -s "$RES_DIR/oom_reconfig.tsv" ] || [ -s "$RES_DIR/mem_gate.txt" ] || rm -f "$RES_JOB_DIR/RESULTS.md"
  res_say "the resilient loop finished after $((RES_ATTEMPT + 1)) attempt(s); $( [ -f "$RES_JOB_DIR/RESULTS.md" ] && echo "RESULTS.md shows $(grep -c '\*\*PASS\*\*' "$RES_JOB_DIR/RESULTS.md") PASS and $(grep -c -E '\*\*(FAIL|DETECTED, NOT RECOVERED)\*\*' "$RES_JOB_DIR/RESULTS.md") not PASS" || echo 'no RESULTS.md was written (this is not a test job)')."
}
