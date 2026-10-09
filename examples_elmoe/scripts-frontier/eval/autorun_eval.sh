#!/bin/bash
# =============================================================================
# autorun_eval.sh -- the entry point for every evaluation. There is no other.
#
#   1. edit the RUNS block below            (nothing else, in no other file)
#   2. bash eval/autorun_eval.sh --dry-run  (see exactly what would be submitted)
#   3. bash eval/autorun_eval.sh            (submit it)
#
#   autorun_eval.sh  ->  frontier_eval.slurm.template  ->  evaluate_v04.py
#
# Everything is validated BEFORE anything is queued, and every complaint names
# the parameter it is about. A run that is already in the results registry is
# skipped; a run whose job is still in the queue is not duplicated.
#
# -----------------------------------------------------------------------------
# EVERY PARAMETER, in one place. Anything you leave out takes the value from
# DEFAULTS below; anything DEFAULTS leaves out takes it from the task table
# (eval/task_table.py + eval/config/custom_suites.py), which is where each
# benchmark's measured sequence length, shots, micro-batch and generation budget
# live. The plan printed at submit time states the value actually used, so a
# default is never hidden.
#
#   model=       model_registry key, e.g. DeepSeek_16B  (architecture + experts)
#   moe=         MoE implementation the job builds with, e.g. X-MOE
#   run=         checkpoint run dir: a name under scripts-frontier/checkpoint/,
#                a path relative to scripts-frontier/, or an absolute path
#   steps=       2280 | 2280,2400 | latest | all
#   benchmarks=  one or more benchmarks, suites or categories, comma-separated,
#                e.g. sciq | sciq,piqa | core | longctx-short
#                (a misspelling is refused at submit time WITH the known names)
#   time=        wall clock, HH:MM:SS
#   nodes=       node count; Frontier gives 8 GCDs per node
#   ep=          expert-parallel size   -- shards the MoE layers
#   tp=          tensor-parallel size   -- shards the dense layers only.
#                tp>1 is REFUSED TODAY. The universal-checkpoint loader copies
#                each atom whole into the local parameter and never slices it,
#                so under TP every tensor-split parameter mismatches: job
#                5354153 (4 nodes, EP8/TP4/PP1) died at loader.py:166 with "0
#                params without atoms, 114 shape mismatches". Lifted when
#                _load_from_universal_atoms shards by the checkpoint's own
#                partition metadata.
#   pp=          pipeline-parallel size -- shards both, across stages.
#                pp>1 is REFUSED TODAY, for EVERY benchmark, for three reasons:
#                (a) a cold-cache SCORING run would record a random number --
#                    the adapter returns np.random.rand() on every rank that is
#                    not the last pipeline stage (lm_eval_adapter.py:737-740),
#                    nothing broadcasts the real values back, and world rank 0
#                    (stage 0) writes the results anyway;
#                (b) a GENERATING run reads logits on every rank in the decode
#                    step and only the last stage has them;
#                (c) the checkpoint does not load at all -- at pp>1 DeepSpeed
#                    builds a PipelineModule whose parameter names are pipe-spec
#                    names, while the checkpoint's atoms carry Megatron names:
#                    job 5354152 (EP4/TP1/PP2), "514 params without atoms".
#                The arithmetic itself is unrestricted -- nodes x 8 must equal
#                pp x (GPUs per stage), and ep and tp must each divide one stage.
#                There is NO "pp divides the layer count" rule: DeepSpeed
#                partitions the pipeline spec, not the transformer-layer count
#                (job 5354152 put 16 spec entries on stage 0 of a 28-layer model).
#   mbs=         micro-batch size
#   shots=       examples shown before the question (few-shot)
#   seqlen=      sequence length (refused when the benchmark locks it -- for
#                wikitext and RULER the length IS the evaluation)
#   pad=         left | right
#   limit=       documents per (sub)task; omit for the whole benchmark
#   slices=      split this benchmark across N jobs (1 = one job, as today)
#   workers=     split this benchmark across N INDEPENDENT MODEL REPLICAS inside
#                ONE job (1 = one model, as today). Each worker gets one node and
#                its 8 GCDs, builds the whole model there, answers its own share
#                of the documents into the SHARED answer file, and the job then
#                runs a COLLECTING step -- the unsliced configuration, one node,
#                every request a cache hit -- which is what writes the score.
#                Requires nodes>=N. Refused with pp>1 (a pipeline spreads ONE
#                model over nodes; workers gives each node its own) and with
#                slices>1 (both split the same documents). ep and tp must fit in
#                one node's 8 GCDs, because that is all one replica owns.
#                A FEW-SHOT benchmark is split at whole-SUB-TASK granularity,
#                automatically: each lm-eval sub-task owns its own few-shot rng,
#                so whole sub-tasks keep every prompt identical, while splitting
#                one sub-task's documents changes them (4-shot slices shared 20
#                of 80 requests with the unsliced run). A few-shot benchmark with
#                no addressable sub-tasks is refused, naming workers=.
#                Zero-shot benchmarks split by document share. Either way the
#                shares are balanced by DOCUMENT COUNT, not by sub-task count.
#                  benchmarks=ruler16k_s nodes=8  workers=8  (0-shot, doc split)
#                  benchmarks=mmlu_5shot  nodes=10 workers=10 (5-shot, subtask)
#                Read rebuild/WORKERS_FEATURE.md before changing the shape.
#   split=       WHAT workers= cuts. document (default) | request.
#                  document -- each replica is handed its own DOCUMENTS (or whole
#                    sub-tasks) and REBUILDS the prompts for them. This is what
#                    workers= has always done, and it is why the few-shot rule above
#                    exists: lm-eval draws a document's few-shot context from one rng
#                    stream consumed in build order, so a replica given a subset of
#                    the documents writes different prompts (gsm8k 8-shot: 1 of 20
#                    requests in common with the unsliced run). Rows that cannot be
#                    cut safely are refused, naming workers=.
#                  request  -- no document is cut. Every replica builds the WHOLE
#                    benchmark, exactly as the collector does, and computes only its
#                    own blocks of the resulting request list (blocks of 4 x mbs,
#                    replica k takes blocks k, k+N, k+2N, ...). The prompts are
#                    therefore bit-identical to an unsliced run by construction --
#                    measured 40 of 40 on the benchmark the document split gets 1 of
#                    20 on -- so THE FEW-SHOT REFUSAL DOES NOT APPLY and the six rows
#                    that are stuck at workers=1 today (drop, mbpp, gsm8k, race,
#                    triviaqa, triviaqa_s) become splittable across N nodes.
#                    Every replica's rows are stamped partial='shard k/N'; the
#                    collector, which carries no shard flags, writes the score.
#                Everything else is unchanged: same one job, same shared answer file,
#                same in-job collector, same refusals for pp>1 / slices>1 / a replica
#                that does not fit one node. split= is inert at workers=1.
#                  benchmarks=triviaqa nodes=8 workers=8 split=request
#                Read rebuild/DISPATCH_ENGINE_DESIGN.md (Stage 1) and
#                rebuild/WORKERS_FEATURE.md section 8.
#   slice_by=    share   -- each job takes a share of the ordered question list
#                subtask -- each job takes a named range of sub-tasks
#   layout=      share (contiguous quarters) | round_robin (every Nth document)
#   docs=        documents per sub-task, only needed to slice by share when the
#                benchmark has no committed subset manifest and no limit=
#   use_ucp=     1 (default) loads the checkpoint through the universal-checkpoint
#                path; 0 loads the consolidated .pt instead. It is part of the answer
#                file's identity, so changing it starts a new answer file.
#   experiment=  declare this run an EXPERIMENT: a measurement made to COMPARE, never
#                to publish. Unset (the default) is a PRODUCTION run, and production
#                rows are the published numbers.
#                An experiment is a separate identity end to end:
#                  * its answers go to a private answer file, so it is forced COLD and
#                    cannot read -- or extend -- a production one (the parallel shape is
#                    NOT in the answer-file identity, so a 4-node run otherwise reads
#                    the 1-node answers as hits and the comparison is vacuous);
#                  * every registry row it writes carries experiment=<tag>, and NO
#                    consumer will ever select such a row as a published value -- not
#                    report.py, not the dashboard, not the tables, not the baseline;
#                  * the done-check keys on it, so an experiment never marks production
#                    work done, and the control arm of an A/B is its own question that
#                    needs no --force to ask;
#                  * the job name carries _exp<tag>, so an experiment in flight never
#                    makes the production job of the same benchmark read as 'inflight'.
#                Use letters, digits, dot, dash or underscore.
#                  benchmarks=openbookqa experiment=wq_equiv   (the arm)
#                  benchmarks=openbookqa experiment=wq_ref     (the control)
#                answers_ns= is the DEPRECATED spelling of this key; it still works and
#                means exactly the same thing. Read rebuild/ANSWER_NAMESPACE_DESIGN.md.
#
# SLICING, in one paragraph. Slices are disjoint by arithmetic -- no locks, no
# claims, no coordination. They all write into the SAME answer file (the cache
# is keyed by request content, not by document index), so a slice that dies is
# resumed by resubmitting it: only the unanswered questions are recomputed. The
# score is written by a COLLECTING run, submitted automatically with a
# dependency on the slices, which runs the unsliced configuration and therefore
# scores the complete benchmark -- every request a cache hit, minutes not hours.
#
# SLICES vs WORKERS, in one paragraph. They are the same split, spent differently.
# slices=N is N separate one-node JOBS plus a collecting job: N queue entries, N
# waits, N model loads, and each slice independently resumable. workers=N is ONE
# job holding N nodes, each running its own complete model on its own share, with
# the collector as a final step inside the same job: one queue entry, one wait,
# still N model loads but paid in parallel. Use workers= when the queue is what
# hurts (measured: 1-node median wait 80 min against a 16.5 min median runtime)
# and slices= when a partial result must survive the job dying.
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="${HERE}"
[[ -f "${LIB_DIR}/eval_lib.py" ]] || LIB_DIR="${HERE}/rebuild"          # pre-cutover
EVAL_DIR="${HERE}"
[[ -f "${EVAL_DIR}/submit_env.sh" ]] || EVAL_DIR="$(dirname "${HERE}")" # pre-cutover

# ============================ DEFAULTS =======================================
# Applied to every run below; any run may override any of them.
DEFAULTS="model=DeepSeek_16B moe=X-MOE time=02:00:00 nodes=1 ep=8 tp=1 pp=1 \
pad=right slices=1 slice_by=share layout=share workers=1 split=document"

# =========================== DECLARE RUNS HERE ===============================
# One line per run. Comment a line out with a leading '#'.
RUNS=(
  # 2026-10-07 18:30: the block was left holding a 128k-chain row by two concurrent chain scripts (bin/v3lc_eval.sh of steps 12 and 16,
  # since serialized with flock); restored to the last row seen here, the 671B 64k cross-check of 2026-09-17.
  "run=deepseek_v3_base_671b_universal steps=0 use_ucp=1 model=DeepSeek_V3_671B benchmarks=ruler64k_v3_base time=02:00:00 nodes=8 ep=64 tp=8 pp=1 mbs=1 rotary_base=10000.0 dp_split=1 experiment=xcheck64k"
)
# ======================= END OF THE EDITABLE BLOCK ===========================

# Jobs inherit this shell through sbatch --export=ALL: without PYTHONPATH they
# die at `import megatron`, and without CXX/CC torch rebuilds the fused kernels
# with the system gcc. Sourced here so that running this file IS the whole
# procedure -- nothing to remember, nothing to source first.
# shellcheck disable=SC1091
if [[ -f "${EVAL_DIR}/submit_env.sh" ]]; then
    . "${EVAL_DIR}/submit_env.sh"
else
    echo "[warn] ${EVAL_DIR}/submit_env.sh not found; jobs may fail at 'import megatron'"
fi
# Dataset environment, matched to rebuild/capture_baseline.sh so a rebuilt run reads
# the SAME cached dataset revisions the baseline read. Without HF_DATASETS_OFFLINE a
# job may re-fetch through the proxy and silently score a different document set.
export LM_EVAL_RULER_CACHE="${EVAL_DIR}/cache/ruler"
export HF_HOME="${EVAL_DIR}/cache/hf_home"
export HF_DATASETS_CACHE="${EVAL_DIR}/cache/hf_datasets_cache"
export HF_DATASETS_OFFLINE=1

RUNS_FILE="$(mktemp "${TMPDIR:-/tmp}/autorun_eval.XXXXXX")"
trap 'rm -f "${RUNS_FILE}"' EXIT
for entry in "${RUNS[@]}"; do
    trimmed="${entry#"${entry%%[![:space:]]*}"}"          # strip leading blanks
    [[ -z "${trimmed}" || "${trimmed}" == \#* ]] && continue
    printf '%s %s\n' "${DEFAULTS}" "${trimmed}" >> "${RUNS_FILE}"
done
if [[ ! -s "${RUNS_FILE}" ]]; then
    echo "no run declared -- uncomment a line in the RUNS block of $0"
    exit 1
fi

python3 "${LIB_DIR}/eval_lib.py" --runs-file "${RUNS_FILE}" "$@"
