#!/bin/bash
# oom_reconfig.sh: the restart loop's side of my OOM re-configuration rule (2026-10-10 01:45; the rule itself is bin/oom_reconfig.py),
# sourced by restart_loop.sh right after its own functions. Nothing here runs unless its flag is on:
#   AUTORUN_OOM_RECONFIG=1   upon exit code 11 (gpu_oom) the loop takes one rung of the fixed ladder (micro-batch 1; d aligned to EP with the
#                            fewest-bytes split; EP and d doubled; PP doubled; then stop with mail), re-renders the attempt's layout variables
#                            for xmoe_train_srun, loads the universal checkpoint of 'latest' when the parallelism changed, and writes the
#                            decision as one full-sentence block into recovery.log; default 0 = today's restart-only behaviour.
#   AUTORUN_MEM_GATE=1       the memory gate before the first attempt (on by itself when AUTORUN_OOM_RECONFIG=1): the memory model's peak per
#                            stage and the layout heatmap's colour into gate.log; a red layout gets a WARNING and the run proceeds.
#   AUTORUN_SHRINK=1         [2026-10-11, the user's rule] when no spare node is left (fewer healthy nodes than the run needs), the loop runs
#                            the memory model (oom_reconfig.py shrink) for the largest node count among the healthy nodes at which the
#                            current configuration, or a configuration of the out-of-memory fallback order, can be launched and fits;
#                            it restarts there from the universal checkpoint of 'latest' (the tokens per step unchanged) and stops with
#                            mail only when nothing fits; default 0 = the loop stops with mail as before.
#   AUTORUN_MEM_PROFILER=1   the memory profiler (bin/memsnap, the shim of 2026-09-14 with a peak writer) on every attempt: the never-reset
#                            per-rank peaks are folded per stage after each attempt (bin/mem_peaks.py) and written into recovery.log next to
#                            the model's prediction; AUTORUN_MEMSNAP_RANKS=<r,r,...> also records the allocator history on those ranks.
# HOW THE LAYOUT REACHES THE TRAINER (the template, read 2026-10-10): xmoe_train_srun expands ${megatron_options} ${deepspeed_options} at
# every call, so the loop rewrites the four flags in those strings (--tensor-model-parallel-size, --ds-sequence-parallel-size,
# --moe-expert-parallel-size, --micro-batch-size; --pipeline-model-parallel-size / --no-pipeline-parallel in deepspeed_options),
# the DeepSpeed json's train_micro_batch_size_per_gpu (the engine derives the accumulation from train_batch_size, which never changes,
# and the new data-parallel width), adds --sequence-parallel + XMOE_MEGATRON_SP=1 whenever TP > 1 (the S9 pair), and sets the shell
# variables the loop and the aggregator read (PP_SIZE, EP_PARALLEL_SIZE, MP_SIZE, DS_SEQ_PAR_SIZE, BATCH_SIZE, RES_GROUP). A layout change
# loads the universal checkpoint: --finetune stays absent (the resilient rider removed it), --universal-checkpoint is appended once, and
# --load is pointed at the save directory, whose 'latest_universal' the UCP workers wrote (megatron/training.py turns the flag into the
# DeepSpeed config's checkpoint.load_universal, and the engine reads 'latest_universal' of the load directory).
# THE RECORD: data/oom_reconfig.tsv (one line per decision), data/attempt_layouts.tsv (the layout of every attempt), data/mem_gate.txt,
# data/mem_peaks_a<k>.tsv (the profiler's per-rank peaks), data/oom_block_a<k>.txt (the rule's block as written into recovery.log).

res_oom_on() { [ "${AUTORUN_OOM_RECONFIG:-0}" = 1 ]; }
res_mg_on() { res_oom_on || [ "${AUTORUN_MEM_GATE:-0}" = 1 ]; }
res_prof_on() { [ "${AUTORUN_MEM_PROFILER:-0}" = 1 ]; }
res_shrink_on() { [ "${AUTORUN_SHRINK:-0}" = 1 ]; }

res_layout_text() {   # the current layout in words (the same vocabulary as oom_reconfig.py)
  local u=${DS_SEQ_PAR_SIZE:-1}
  echo "PP${PP_SIZE} EP${EP_PARALLEL_SIZE} TP${MP_SIZE} x U${u} (d = $(( MP_SIZE * u )), mbs ${BATCH_SIZE}, dp_moe $(( TOTAL_GPUS / PP_SIZE / EP_PARALLEL_SIZE )), dp_non_moe $(( TOTAL_GPUS / PP_SIZE / MP_SIZE / u )), GA $(( GLOBAL_BATCH_SIZE / BATCH_SIZE / (TOTAL_GPUS / PP_SIZE / MP_SIZE / u) )))"
}

res_mg_capacity() {   # the routed-rows bound the memory model uses: dropless = 1.0; token dropping = the capacity factor the trainer sees (the LAST --moe-train-capacity-factor wins)
  if echo "$megatron_options" | grep -q -- '--disable-moe-token-dropping'; then echo 1.0
  else echo "$megatron_options" | grep -o -E -- '--moe-train-capacity-factor [0-9.]+' | tail -1 | awk '{print $2}' | grep . || echo 1.25; fi
}

res_mg_args() {   # the CURRENT layout and recipe as oom_reconfig.py arguments
  local cce ckpt
  cce=$(echo "$megatron_options" | grep -o -E -- '--seq-chunked-ce [0-9]+' | awk '{print $2}')
  if echo "$megatron_options" | grep -q -- '--checkpoint-activations'; then ckpt=full; else ckpt=none; fi
  # [2026-10-13] --intra-doc: the run uses intra-document attention, so the rule never chooses a Ulysses split (megatron/arguments.py asserts it)
  echo "--model $MODEL_SIZE --nodes $NODES --seq $SEQ_LEN --gbs $GLOBAL_BATCH_SIZE --pp $PP_SIZE --ep $EP_PARALLEL_SIZE --tp $MP_SIZE --ulysses ${DS_SEQ_PAR_SIZE:-1} --mbs $BATCH_SIZE ${cce:+--chunked-ce $cce} --capacity $(res_mg_capacity) --ckpt $ckpt$(echo "$megatron_options" | grep -q -- '--intra-document-attention' && echo ' --intra-doc')"
}

res_layout_record() {   # one line per attempt into data/attempt_layouts.tsv (the layout the attempt runs with, and the model's prediction when known)
  res_mg_on || return 0
  [ -f "$RES_DIR/attempt_layouts.tsv" ] || printf '# attempt_layouts.tsv: the parallel layout of every attempt (written at its start). pred = the memory model per stage as stage:allocated/VRAM/line GiB\nattempt\tpp\tep\ttp\tulysses\td\tmbs\tgbs\tga\ttext\tpred\tloads\n' > "$RES_DIR/attempt_layouts.tsv"
  local u=${DS_SEQ_PAR_SIZE:-1}
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$RES_ATTEMPT" "$PP_SIZE" "$EP_PARALLEL_SIZE" "$MP_SIZE" "$u" $(( MP_SIZE * u )) "$BATCH_SIZE" "$GLOBAL_BATCH_SIZE" \
    $(( GLOBAL_BATCH_SIZE / BATCH_SIZE / (TOTAL_GPUS / PP_SIZE / MP_SIZE / u) )) "$(res_layout_text)" "${RES_CUR_PRED:--}" "${RES_CUR_LOADS:-the save named by latest (native resume), or the load path}" >> "$RES_DIR/attempt_layouts.tsv"
}

# ---------------------------------------------------------------- the memory gate (before the first attempt)
res_mem_gate() {
  res_mg_on || return 0
  local out="$RES_DIR/mem_gate.txt" rc
  RES_OOM_INIT_ARGS="$(res_mg_args)"; RES_OOM_TAKEN=0; RES_CUR_PRED=""; RES_MG_WARNING=""
  $RES_PY "$RES_BIN/oom_reconfig.py" gate $RES_OOM_INIT_ARGS --shell > "$out" 2> "$RES_DIR/mem_gate.err"; rc=$?
  { echo; echo "## The memory gate ($(date '+%F %H:%M:%S'); AUTORUN_OOM_RECONFIG=${AUTORUN_OOM_RECONFIG:-0}, AUTORUN_MEM_GATE=${AUTORUN_MEM_GATE:-0}; the memory model utils/v3_memory_model.py and the layout heatmap; a red layout gets a WARNING and the run proceeds)"
    if [ $rc = 0 ]; then grep -v -E '^MG_[A-Z_]+=' "$out"; else echo "the memory gate could not predict the layout $(res_layout_text): $(tail -3 "$RES_DIR/mem_gate.err" | tr '\n' ' ' | head -c 400)"; fi; } >> "$RES_JOB_DIR/gate.log"
  if [ $rc != 0 ]; then res_say "the memory gate could not predict the layout $(res_layout_text) (data/mem_gate.err: $(tail -1 "$RES_DIR/mem_gate.err" | head -c 200)); the run proceeds without a verdict."; return 0; fi
  eval "$(grep -E '^MG_[A-Z_]+=' "$out")"
  RES_CUR_PRED="$MG_PRED"; RES_MG_WARNING="$MG_WARNING"; RES_CUR_LOADS="the load path (attempt 0)"
  res_say "the memory gate: the layout $(res_layout_text) at $(( SEQ_LEN / 1024 ))k is $MG_COLOUR by the heatmap rule; the model's peak on the heaviest stage $MG_STAGE is $MG_ALLOC GiB allocated / $MG_VRAM VRAM, the out-of-memory line $MG_LINE GiB against 64; per stage (allocated/VRAM/line): $MG_PRED; the bytes per step $MG_BYTES GiB ($MG_INTER across nodes); the block is in gate.log."
  res_rec "  memory gate: the layout $(res_layout_text) at $(( SEQ_LEN / 1024 ))k is $MG_COLOUR by the heatmap rule (heaviest stage $MG_STAGE: $MG_ALLOC GiB allocated / $MG_VRAM VRAM; the out-of-memory line $MG_LINE GiB against 64); per stage: $MG_PRED$( [ -n "$MG_WARNING" ] && echo "; $MG_WARNING" )"
  res_rec ""
  if [ -n "$MG_WARNING" ]; then res_loud "$MG_WARNING"; fi
}

# ---------------------------------------------------------------- the memory profiler on every attempt
res_train_launch() {   # starts the training srun in the background (TRAIN_PID), with the memsnap shim first on PATH when the profiler is on
  if res_prof_on; then
    ( export PATH="$RES_BIN/memsnap:$PATH" MEMSNAP_OUT="$RES_ATT_DIR/memsnap" MEMSNAP_RANK="${AUTORUN_MEMSNAP_RANKS:-none}" MEMSNAP_PEAK_EVERY_S="${MEMSNAP_PEAK_EVERY_S:-30}" RES_PY="$RES_PY"
      xmoe_train_srun "$RES_TRAIN_CSV" "$N" ) &
    TRAIN_PID=$!
    res_say "the memory profiler is on for attempt $RES_ATTEMPT: the memsnap shim ($RES_BIN/memsnap/python) wraps every rank's python, so the peak-memory counters are never reset and every rank writes its peak to $(res_rel "$RES_ATT_DIR")/memsnap/peak_rank<r>.txt every ${MEMSNAP_PEAK_EVERY_S:-30} s (allocator history on ranks: ${AUTORUN_MEMSNAP_RANKS:-none})."
  else
    xmoe_train_srun "$RES_TRAIN_CSV" "$N" &
    TRAIN_PID=$!
  fi
}

res_mem_peaks_line() {   # $1 attempt: the profiler's per-rank peaks of the attempt, per stage, next to the model's prediction; sets RES_MEAS_*
  RES_MEAS_PEAK="-"; RES_MEAS_RANK="-"; RES_MEAS_STAGE="-"; RES_MEAS_STAGES=""
  res_prof_on || return 0
  local out sent m
  out=$($RES_PY "$RES_BIN/mem_peaks.py" --attempt-dir "$RES_ATT_DIR" --world "$TOTAL_GPUS" --pp "$PP_SIZE" --out "$RES_DIR/mem_peaks_a$1.tsv" --pred "${RES_CUR_PRED:-}" 2>&1)
  sent=$(echo "$out" | grep -v '^PEAK_GIB=' | tail -1); m=$(echo "$out" | grep '^PEAK_GIB=' | tail -1)
  eval "$(echo "$m" | tr ';' '\n' | sed -E 's/^(PEAK_GIB|PEAK_RANK|PEAK_STAGE|STAGES|N_RANKS)=(.*)$/RES_MEAS_\1=\x27\2\x27/' | grep '^RES_MEAS_')" 2>/dev/null
  RES_MEAS_PEAK=${RES_MEAS_PEAK_GIB:--}; RES_MEAS_RANK=${RES_MEAS_PEAK_RANK:--}; RES_MEAS_STAGE=${RES_MEAS_PEAK_STAGE:--}; RES_MEAS_STAGES=${RES_MEAS_STAGES:-}
  res_rec "  memory profiler (attempt $1, layout $(res_layout_text)): $sent"
  res_say "the memory profiler of attempt $1: $sent"
}

# ---------------------------------------------------------------- the decision upon exit code 11
res_stage_of_rank() { [ "$1" = "-" ] && echo "-" || echo $(( $1 / (TOTAL_GPUS / PP_SIZE) )); }

res_oom_hbm() {   # the allocated memory at the failure, from the OOM exit record of this attempt (GB as the record has it), else '?'
  $RES_PY - "$RES_ATT_DIR/exit" <<'PYH'
import glob, json, re, sys
best = None
for f in glob.glob(sys.argv[1] + '/r*.json'):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    if r.get('code') != 11:
        continue
    if best is None or r['time'] < best['time']:
        best = r
if best is None:
    print('?'); sys.exit(0)
ev = best.get('evidence', '')
m = re.search(r'hbm_gb=([0-9.]+)', ev); m2 = re.search(r'([0-9.]+) GiB is allocated by PyTorch', ev)
print(m.group(1) if m else (f"{float(m2.group(1)) * 1.0737:.1f}" if m2 else '?'))
PYH
}

res_oom_apply_layout() {   # NEW_PP NEW_EP NEW_TP NEW_U NEW_MBS from the rule -> the template's strings and variables for the next attempt
  local o="$megatron_options" d="$deepspeed_options"
  o=$(echo "$o" | sed -E "s/--tensor-model-parallel-size [0-9]+/--tensor-model-parallel-size $NEW_TP/; s/--moe-expert-parallel-size [0-9]+/--moe-expert-parallel-size $NEW_EP/; s/--micro-batch-size [0-9]+/--micro-batch-size $NEW_MBS/")
  if echo "$o" | grep -q -- '--ds-sequence-parallel-size'; then o=$(echo "$o" | sed -E "s/--ds-sequence-parallel-size [0-9]+/--ds-sequence-parallel-size $NEW_U/")
  else o="$o --ds-sequence-parallel-size $NEW_U"; fi
  o=$(echo "$o" | sed -E 's/(^| )--sequence-parallel( |$)/\1\2/g; s/  +/ /g')
  if [ "$NEW_TP" -gt 1 ]; then o="$o --sequence-parallel"; export XMOE_MEGATRON_SP=1; else unset XMOE_MEGATRON_SP; fi
  if [ "$NEW_PP" -gt 1 ]; then
    if echo "$d" | grep -q -- '--pipeline-model-parallel-size'; then d=$(echo "$d" | sed -E "s/--pipeline-model-parallel-size [0-9]+/--pipeline-model-parallel-size $NEW_PP/")
    else d=$(echo "$d" | sed -E "s/--no-pipeline-parallel/--pipeline-model-parallel-size $NEW_PP/"); fi
  else
    d=$(echo "$d" | sed -E "s/--pipeline-model-parallel-size [0-9]+/--no-pipeline-parallel/")
  fi
  megatron_options="$o"; deepspeed_options="$d"
  if [ -n "${config_json:-}" ] && [ -f "$config_json" ]; then
    sed -i -E "s/\"train_micro_batch_size_per_gpu\": *[0-9]+/\"train_micro_batch_size_per_gpu\": $NEW_MBS/" "$config_json"
  fi
  PP_SIZE=$NEW_PP; EP_PARALLEL_SIZE=$NEW_EP; MP_SIZE=$NEW_TP; DS_SEQ_PAR_SIZE=$NEW_U; BATCH_SIZE=$NEW_MBS; RES_GROUP=$NEW_EP
  export PP_SIZE EP_PARALLEL_SIZE MP_SIZE DS_SEQ_PAR_SIZE BATCH_SIZE
}

res_oom_set_universal_load() {   # the next attempt loads the universal checkpoint of the save directory (its 'latest_universal')
  local o="$megatron_options"
  o=$(echo "$o" | sed -E 's/(^| )--finetune( |$)/\1\2/g; s/  +/ /g')
  echo "$o" | grep -q -- '--universal-checkpoint' || o="$o --universal-checkpoint"
  if echo "$o" | grep -q -E -- '--load [^ ]+'; then o=$(echo "$o" | sed -E "s#--load [^ ]+#--load $RES_CKPT_DIR#"); else o="$o --load $RES_CKPT_DIR"; fi
  megatron_options="$o"
}

res_oom_universal_ready() {   # $1 the step N of 'latest': returns 0 once global_stepN_universal is complete and 'latest_universal' names it; sets RES_OOM_WAIT_S
  local N=$1 t0 deadline said=0 lu rc
  t0=$(date +%s); deadline=$(( ${RES_END_EPOCH:-0} - $(res_need_attempt_s) - 60 )); RES_OOM_WAIT_S=0
  if ! res_ucp_complete "$N"; then
    if [ "${RES_UCP_N:-0}" -gt 0 ]; then
      res_say "the OOM rule needs the universal checkpoint of step $N and waits for the UCP workers' conversion (the watcher converts the newest complete save first; the state: $(cat "$RES_DIR/ucp_state" 2>/dev/null | head -c 300))."
      while ! res_ucp_complete "$N"; do
        if [ ${RES_END_EPOCH:-0} -gt 0 ] && [ $(date +%s) -ge $deadline ]; then res_say "the universal of step $N is still not complete and only $(( RES_END_EPOCH - $(date +%s) )) s of the wall are left, which is not enough for a restart after it."; return 1; fi
        grep -q -x "$N" "$RES_DIR/ucp/failed_steps" 2>/dev/null && { res_say "the conversion of step $N failed twice on the workers (data/ucp/convert_s${N}_share*.log): no universal to load."; return 1; }
        [ -f "$RES_DIR/ucp/stop" ] && { res_say "the UCP watcher was stopped: no universal of step $N will come."; return 1; }
        sleep 10; said=$((said + 10))
        [ $((said % 60)) = 0 ] && res_say "still waiting for the universal of step $N ($(( $(date +%s) - t0 )) s so far; $(cat "$RES_DIR/ucp_state" 2>/dev/null | head -c 200))."
      done
    else
      local in="$RES_CKPT_DIR/global_step$N" out="$RES_CKPT_DIR/global_step${N}_universal" log="$RES_DIR/oom_convert_s$N.log"
      res_say "no UCP worker node is in this job, so the OOM rule converts the save of step $N to the universal form on the batch node now (ds_to_universal --fragments-inplace --optstate-from-pickle, 32 extract and 32 merge workers; the log is data/oom_convert_s$N.log)."
      mkdir -p "$out"; lfs setstripe -c 8 -S 4M "$out" 2>/dev/null || true
      ( cd "$PWD" && $RES_PY -m deepspeed.checkpoint.ds_to_universal --input_folder "$in" --output_folder "$out" --num_extract_workers ${RES_UCP_EXTRACT_WORKERS:-32} --num_merge_workers ${RES_UCP_MERGE_WORKERS:-32} --fragments-inplace --optstate-from-pickle ) > "$log" 2>&1; rc=$?
      if [ $rc != 0 ] || ! res_ucp_complete "$N"; then res_say "the conversion of step $N on the batch node ended with exit code $rc and the universal is $( res_ucp_complete "$N" && echo complete || echo 'NOT complete' ) (data/oom_convert_s$N.log): no universal to load."; return 1; fi
    fi
  fi
  lu=$(cat "$RES_CKPT_DIR/latest_universal" 2>/dev/null)
  if [ "$lu" != "global_step${N}_universal" ]; then
    echo "global_step${N}_universal" > "$RES_CKPT_DIR/latest_universal.tmp" && mv -f "$RES_CKPT_DIR/latest_universal.tmp" "$RES_CKPT_DIR/latest_universal"
    res_say "'latest_universal' named ${lu:-nothing}; it now names global_step${N}_universal, the save the next attempt resumes from (the loader reads this tag)."
  fi
  RES_OOM_WAIT_S=$(( $(date +%s) - t0 ))
  return 0
}

res_oom_dry_run() {   # the new command line through the dry run on the batch node (informational: the arguments and the divisibility checks)
  local log="$RES_DIR/oom_dryrun_a$((RES_ATTEMPT + 1)).log" rc
  [ "$(type -t xmoe_dry_run_cmd)" = function ] || return 0
  timeout -k 10 300 bash -c "$(xmoe_dry_run_cmd)" > "$log" 2>&1; rc=$?
  if [ $rc = 0 ]; then res_say "the dry run of the new command line passed ($(grep -a -m1 -o 'parse + validate OK.*' "$log" | head -c 200))."
  else res_say "!!!!! the dry run of the new command line FAILED with exit code $rc ($(grep -a -m1 -E 'FAIL|Error|error' "$log" | head -c 300)); the attempt is launched anyway (RES_DRYRUN_STRICT is not applied here) and will show the same error if it is real !!!!!"; fi
}

res_oom_handle() {   # the branch of res_main upon code 11 with AUTORUN_OOM_RECONFIG=1; 0 = restart with the new layout (variables rewritten, RES_OOM_BANNER set), 1 = stop (mail sent)
  local taken=${RES_OOM_TAKEN:-0} blk="$RES_DIR/oom_block_a${RES_ATTEMPT}.txt" sh="$RES_DIR/oom_next_a${RES_ATTEMPT}.sh" t0 t1 N stage hbm load_txt old_pred="${RES_CUR_PRED:--}" wait_s=0 univ="-"
  t0=$(date +%s); stage=$(res_stage_of_rank "$RC_RANK"); hbm=$(res_oom_hbm)
  [ -n "${RES_OOM_INIT_ARGS:-}" ] || RES_OOM_INIT_ARGS="$(res_mg_args)"
  if ! $RES_PY "$RES_BIN/oom_reconfig.py" next $RES_OOM_INIT_ARGS --taken "$taken" --oom-step "$RC_STEP" --oom-stage "$stage" --oom-rank "$RC_RANK" --oom-hbm "$hbm" --oom-attempt "$RES_ATTEMPT" --out-block "$blk" --shell > "$sh" 2> "$sh.err"; then
    res_rec "  decision:    NO RESTART: the OOM rule could not be evaluated ($(tail -2 "$sh.err" | tr '\n' ' ' | head -c 300))"; res_rec ""
    res_mail "X-MoE job $SLURM_JOB_ID stopped: the OOM rule could not be evaluated" "see data/oom_next_a${RES_ATTEMPT}.sh.err"; return 1
  fi
  eval "$(cat "$sh")"
  res_rec "  the rule:    $(cat "$blk")"
  res_rec "  measured:    the memory profiler of the failed attempt: peak ${RES_MEAS_PEAK:--} GiB allocated on rank ${RES_MEAS_RANK:--} (stage ${RES_MEAS_STAGE:--}); per stage ${RES_MEAS_STAGES:--} (data/mem_peaks_a${RES_ATTEMPT}.tsv); the model had predicted (allocated/VRAM/line GiB per stage) $old_pred"
  if [ "$OOM_NEXT" = 0 ]; then
    res_rec "  decision:    NO RESTART: $OOM_TEXT"; res_rec ""
    res_mail "X-MoE job $SLURM_JOB_ID stopped: the OOM ladder is exhausted" "$OOM_TEXT"; return 1
  fi
  N=$(res_latest_step)
  if [ "$OOM_NEEDS_UNIVERSAL" = 1 ] && [ "$N" -gt 0 ]; then
    if ! res_oom_universal_ready "$N"; then
      res_rec "  decision:    NO RESTART: the layout change needs the universal checkpoint of step $N and it could not be produced in time (loop.log says why); the save of step $N is kept"; res_rec ""
      res_mail "X-MoE job $SLURM_JOB_ID stopped: no universal checkpoint for the OOM re-configuration" "step $N; read loop.log and data/ucp/"; return 1
    fi
    wait_s=$RES_OOM_WAIT_S; univ=$N; RES_UNIV_LOADED_STEP=$N
    res_oom_set_universal_load
    load_txt="the universal checkpoint of step $N is complete ('latest_universal' names global_step${N}_universal; the rule waited $wait_s s for it) and attempt $((RES_ATTEMPT + 1)) loads it with --universal-checkpoint from $RES_CKPT_DIR"
    RES_CUR_LOADS="the universal checkpoint of step $N"
  elif [ "$N" -gt 0 ]; then
    load_txt="the same save of step $N loads natively (step 0 changes no parallelism)"; RES_CUR_LOADS="the same save of step $N (native)"
  else
    load_txt="no save exists yet (the out of memory came before the first save), so attempt $((RES_ATTEMPT + 1)) starts from the SAME initial state $( [ -n "${RES_INIT_LOAD_DIR:-}" ] && echo "(the initial checkpoint ${RES_INIT_LOAD_DIR}, weights only with --finetune$( [ "${RES_INIT_UNIVERSAL:-0}" = 1 ] && echo ', in its universal form, which loads under the new layout'))" || echo "(the load path ${LOAD_CHECKPOINT_PATH:-unset}; random init when it holds no save)") with the new layout"
    RES_CUR_LOADS="the initial state (no save existed)"
  fi
  res_oom_apply_layout
  res_oom_dry_run
  RES_OOM_TAKEN=$((taken + 1)); RES_CUR_PRED="$NEW_PRED"
  t1=$(date +%s)
  res_rec "  decision:    attempt $((RES_ATTEMPT + 1)) starts with the new layout $NEW_LAYOUT (the rule's step $OOM_STEP, re-configuration $RES_OOM_TAKEN of this job); $load_txt; the model predicts (allocated/VRAM/line GiB per stage) $NEW_PRED, $NEW_COLOUR by the heatmap rule; the bytes per step $NEW_BYTES GiB ($NEW_INTER across nodes); the decision took $((t1 - t0)) s"
  [ -f "$RES_DIR/oom_reconfig.tsv" ] || printf '# oom_reconfig.tsv: one line per decision of the OOM rule (AUTORUN_OOM_RECONFIG=1). pred = stage:allocated/VRAM/line GiB; measured = the profiler peak of the failed attempt\nattempt\toom_step\toom_stage\toom_rank\toom_hbm_gb\ttaken\trule_step\told_layout\tnew_layout\tbytes_gib\tinter_gib\told_pred\tmeasured_peak_gib\tmeasured_stages\tnew_pred\tuniversal_step\tuniversal_wait_s\tdecided_at\tdecision_s\n' > "$RES_DIR/oom_reconfig.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$RES_ATTEMPT" "$RC_STEP" "$stage" "$RC_RANK" "$hbm" "$taken" "$OOM_STEP" "$OLD_LAYOUT" "$NEW_LAYOUT" "$NEW_BYTES" "$NEW_INTER" "$old_pred" "${RES_MEAS_PEAK:--}" "${RES_MEAS_STAGES:--}" "$NEW_PRED" "$univ" "$wait_s" "$(date -d @$t1 '+%F %H:%M:%S')" "$((t1 - t0))" >> "$RES_DIR/oom_reconfig.tsv"
  RES_OOM_BANNER="##### RECOVERED at @TIME@: the failure was a gpu_oom on node $RC_NODE (rank $RC_RANK, stage $stage), which fired at step $RC_STEP; the OOM rule took step $OOM_STEP: $OLD_LAYOUT -> $NEW_LAYOUT; restart $((RES_ATTEMPT + 1)) ran inside the same allocation with the new layout; $load_txt; the recovery took @REC@ s from the srun exit to the first iteration line #####"
  res_say "restart $((RES_ATTEMPT + 1)) follows with the new layout $NEW_LAYOUT (the OOM rule's step $OOM_STEP); $load_txt."
  # the aggregator groups the ranks by EP and PP: restart it with the new layout
  [ -n "${RES_DUTY:-}" ] && res_agg_start "$RES_DUTY"
  return 0
}

# ---------------------------------------------------------------- [2026-10-11] no spare node left: a smaller configuration chosen by the memory model
res_shrink_handle() {   # $1 the healthy nodes left; 0 = N, NODES, TOTAL_GPUS and the layout rewritten for the next attempt; 1 = stop (mail sent)
  local good=$1 sh="$RES_DIR/shrink_a${RES_ATTEMPT}.sh" t0 t1 S load_txt univ="-" wait_s=0
  t0=$(date +%s)
  if ! $RES_PY "$RES_BIN/oom_reconfig.py" shrink $(res_mg_args) --good "$good" --shell > "$sh" 2> "$sh.err"; then
    res_rec "  decision:    NO RESTART: no spare node is left and the memory model could not be evaluated ($(tail -2 "$sh.err" | tr '\n' ' ' | head -c 300))"; res_rec ""
    res_mail "X-MoE job $SLURM_JOB_ID stopped: no spare node is left" "the memory model could not be evaluated; see data/shrink_a${RES_ATTEMPT}.sh.err"; return 1
  fi
  eval "$(cat "$sh")"
  res_rec "  no spare:    $SHRINK_TEXT"
  if [ "$SHRINK_NEXT" != 1 ]; then
    res_rec "  decision:    NO RESTART: no configuration on the $good healthy nodes fits the memory model"; res_rec ""
    res_mail "X-MoE job $SLURM_JOB_ID stopped: no spare node is left and nothing smaller fits" "$SHRINK_TEXT"; return 1
  fi
  S=$(res_latest_step)
  if [ "$S" -gt 0 ]; then
    if ! res_oom_universal_ready "$S"; then
      res_rec "  decision:    NO RESTART: the smaller configuration needs the universal checkpoint of step $S and it could not be produced in time (loop.log says why)"; res_rec ""
      res_mail "X-MoE job $SLURM_JOB_ID stopped: no universal checkpoint for the smaller configuration" "step $S; read loop.log and data/ucp/"; return 1
    fi
    wait_s=$RES_OOM_WAIT_S; univ=$S; RES_UNIV_LOADED_STEP=$S
    res_oom_set_universal_load
    load_txt="the universal checkpoint of step $S loads (the rule waited $wait_s s for it)"; RES_CUR_LOADS="the universal checkpoint of step $S"
  else
    load_txt="no save exists yet, so the attempt starts from the same initial state with the smaller configuration"; RES_CUR_LOADS="the initial state (no save existed)"
  fi
  res_oom_apply_layout
  NODES=$SHRINK_NODES; TOTAL_GPUS=$(( SHRINK_NODES * ${gpus_per_node:-8} )); N=$SHRINK_NODES
  export NODES TOTAL_GPUS
  RES_OOM_INIT_ARGS="$(res_mg_args)"; RES_OOM_TAKEN=0; RES_CUR_PRED="$NEW_PRED"
  res_oom_dry_run
  t1=$(date +%s)
  res_rec "  decision:    attempt $RES_ATTEMPT starts on $SHRINK_NODES nodes with $NEW_LAYOUT; $load_txt; the decision took $((t1 - t0)) s"
  res_tl "# no spare node was left: attempt $RES_ATTEMPT runs on $SHRINK_NODES of the $good healthy nodes with $NEW_LAYOUT; $load_txt."
  [ -f "$RES_DIR/shrink.tsv" ] || printf '# shrink.tsv: one line per decision taken because no spare node was left (AUTORUN_SHRINK=1)\nattempt\tgood\tnodes\told_layout\tnew_layout\tnew_pred\tuniversal_step\tuniversal_wait_s\tdecided_at\tdecision_s\n' > "$RES_DIR/shrink.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$RES_ATTEMPT" "$good" "$SHRINK_NODES" "$OLD_LAYOUT" "$NEW_LAYOUT" "$NEW_PRED" "$univ" "$wait_s" "$(date -d @$t1 '+%F %H:%M:%S')" "$((t1 - t0))" >> "$RES_DIR/shrink.tsv"
  res_say "no spare node is left: the next attempt runs on $SHRINK_NODES of the $good healthy nodes with $NEW_LAYOUT; $load_txt."
  return 0
}
