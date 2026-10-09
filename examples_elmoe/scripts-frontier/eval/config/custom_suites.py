import os
# custom_suites.py — user-owned additions/overrides, merged over task_table.py defaults.
# `Row`, `replace`, and the default `TASKS`/`SUITES` dicts are in scope.
#
# Add a row:                CUSTOM_TASKS = {'mmlu_2shot': replace(TASKS['mmlu_0shot'],
#                                                                fewshot=2, seqlen=2048,
#                                                                seqlen_src='assumed')}
# Tweak a default row:      CUSTOM_TASKS = {'sciq': {'mbs': 4}}
# Define a suite:           CUSTOM_SUITES = {'quickcheck': ['sciq', 'piqa']}
#
# Everything here survives updates to task_table.py.

# ---------------------------------------------------------------------------------
# think-budget rows (added 2026-08-16). The Nemotron branch checkpoint
# 2026-08-09_5213097 answers with a `<think>` reasoning monologue: measured in the
# response caches, 80/80 humaneval_chat and 80/80 mbpp_chat replies open `<think>`
# and NONE close it within max_gen_toks=512, so those scores measured the budget
# rather than the model. These rows re-measure the same committed 80-document
# subsets with a 4x budget (and the seqlen headroom the longer reply needs) so the
# reasoning can finish and the grader sees a real answer.
#
# Same subset manifest, same seed, same grader -- only the generation budget differs,
# and the difference is stated wherever the numbers are reported.
# ---------------------------------------------------------------------------------
CUSTOM_TASKS = {
    'humaneval_chat_think': replace(TASKS['humaneval_chat_s'], gen_toks=2048,
                                    seqlen=4096, mbs=4, est_minutes=140,
                                    seqlen_src='assumed:think-budget re-measure'),
    'mbpp_chat_think': replace(TASKS['mbpp_chat_s'], gen_toks=2048,
                               seqlen=4096, mbs=4, est_minutes=180,
                               seqlen_src='assumed:think-budget re-measure'),
    'ifeval_think': replace(TASKS['ifeval_s'], gen_toks=3072,
                            seqlen=4096, mbs=4, est_minutes=150,
                            seqlen_src='assumed:think-budget re-measure'),
}
CUSTOM_SUITES = {
    'think-budget': ['humaneval_chat_think', 'mbpp_chat_think', 'ifeval_think'],
}

# ---------------------------------------------------------------------------------
# 2026-09-04: reproduction of arXiv:2406.02500 (Towards Efficient MoE: A Holistic
# Study of Compression Techniques), Sec. 8 table: 0-shot ARC-C, BoolQ, HellaSwag,
# MMLU, OBQA, PIQA, RTE, WinoGrande on DeepSeek-MoE-16B base and after Alpaca-GPT4
# SFT. Seven rows already exist in task_table.py (mmlu_0shot is the 0-shot MMLU row).
# RTE is the one benchmark the table lacked: lm_eval 0.4.13 task `rte` (GLUE RTE,
# 277 validation pairs, loglikelihood over two labels); dataset nyu-mll/glue
# prefetched into the cache on 2026-09-04. seqlen 512 covers every document:
# measured with the DeepSeek tokenizer on the 277 validation rows in lm-eval's
# prompt format, max 274 tokens, p99 214.
# ---------------------------------------------------------------------------------
CUSTOM_TASKS.update({
    'rte': Row('rte', 'commonsense', 'loglik', 512, est_minutes=5,
               seqlen_src='measured:2026-09-04 max=274 p99=214 (277 docs)',
               needs=('dataset:glue',)),
})
CUSTOM_SUITES.update({
    # the paper's eight benchmarks, all 0-shot, in the paper's column order
    'repro8': ['arc_challenge', 'boolq', 'hellaswag', 'mmlu_0shot',
               'openbookqa', 'piqa', 'rte', 'winogrande'],
    # the seven short ones (mmlu_0shot is scheduled separately with workers)
    'repro7': ['arc_challenge', 'boolq', 'hellaswag',
               'openbookqa', 'piqa', 'rte', 'winogrande'],
})

# ---------------------------------------------------------------------------------
# MEASURED generation rates (2026-08-16, live fleet, 1 node, committed subsets):
#   ruler4k + longbench share one 4k cache: 21.9 docs/min -> 1,750 docs in ~80 min
#   ruler8k:  6.0 docs/min -> 1,300 docs in ~217 min
#   ruler16k: 1.66 docs/min -> 1,300 docs in ~785 min  (13 hours per checkpoint)
# The table's original 360-minute estimate for ruler16k was less than half the real
# cost, so its chains were built with 5 links where ~10 are needed: the chain would
# run out and stop with the evaluation unfinished (the drivers re-submit, but each
# restart pays another model load). Sizing the estimates from measurement instead.
# ---------------------------------------------------------------------------------
CUSTOM_TASKS.update({
    'ruler8k_s':  {'est_minutes': 220, 'seqlen_src': 'measured:2026-08-16 6.0 docs/min'},
    'ruler16k_s': {'est_minutes': 790, 'seqlen_src': 'measured:2026-08-16 1.66 docs/min'},
})

# ---------------------------------------------------------------------------------
# 2026-08-16: ruler16k is PAUSED. At 1.66 docs/min it needs ~13 h per checkpoint and
# was crowding out the 4k/8k/LongBench cells and the instruct suite, which together
# cost less than one 16k checkpoint. `longctx-short` therefore excludes it for now;
# the queued 16k links are held/cancelled and every generation already computed stays
# in the response cache, so resuming costs only the model load.
#
# TO RESUME: delete this block (or set CUSTOM_SUITES['longctx-short'] back to include
# 'ruler16k_s'), then `scontrol release <jobids>` for anything still held.
# ---------------------------------------------------------------------------------
CUSTOM_SUITES['longctx-short'] = ['ruler4k_s', 'ruler8k_s', 'longbench_s']

# 2026-08-17 03:20: RULER-16k RESUMED, but only as its own suite so the drivers do not
# re-add it to every ladder cell. At ~13 h per checkpoint it is affordable for the four
# ENDPOINT models (base + the final measured step of each run), which is what the 16k
# column needs to answer -- does long-context training extend usable context -- without
# spending 13 h on every intermediate step.
CUSTOM_SUITES['ruler16k-endpoints'] = ['ruler16k_s']

# 2026-08-16: the 12k run's 8k evaluation burned four chains (jobs 5283120, 5283226-8,
# 5285093-5, 5287871-3) computing ZERO batches -- ranks alive inside the MoE forward at
# 89-99% HBM, i.e. near-OOM thrash at micro-batch 2 with an 8192-token context. Other
# arms cleared the same row, so this is a memory cliff rather than a code fault. Halving
# the micro-batch trades throughput for actually finishing; est_minutes doubles with it.
CUSTOM_TASKS.update({
    'ruler8k_s': {'mbs': 1, 'est_minutes': 440,
                  'seqlen_src': 'measured:2026-08-16 6.0 docs/min at mbs=2; mbs=1 for headroom'},
})

# 2026-08-17: FULL-SET humaneval-chat (all 164 problems, no subset manifest) on the
# endpoint checkpoints. The headline coding claim rests on an 80-problem subset; at
# 164 problems it rests on the whole benchmark, which removes the "n is too small"
# objection for that row. ~1 h per checkpoint.
_hc = TASKS['humaneval_chat_s']
_inc = [w for w in _hc.extra_eval_flags.split() if w]          # keep --include-path <dir>
_inc = ' '.join(_inc[_inc.index('--include-path'):][:2]) if '--include-path' in _inc else ''
CUSTOM_TASKS['humaneval_chat_full'] = replace(_hc, limit=0, est_minutes=70,
                                              extra_eval_flags=_inc)
CUSTOM_SUITES['humaneval-full'] = ['humaneval_chat_full']

# 2026-08-17: ifeval at n=250 (of 541 total) on the endpoint checkpoints. The headline
# instruction-following claim is +0.175 with McNemar p=0.065 at n=40 -- suggestive but
# not resolvable at that size. The n=250 draw is a strict SUPERSET of the committed 40,
# so the original comparison survives unchanged inside it.
_if = TASKS['ifeval_s']
CUSTOM_TASKS['ifeval_s250'] = replace(_if, limit=250, est_minutes=260,
    extra_eval_flags=_if.extra_eval_flags.replace('ifeval_s.json', 'ifeval_s250.json'))
CUSTOM_SUITES['ifeval-wide'] = ['ifeval_s250']

# 2026-08-17 04:40: the 8k and 12k runs are far further along than our ladder captured
# (8k at step 150, 12k at 80 -- the other session's chain kept training them). Their
# LATEST checkpoints are the most valuable remaining measurement: they show where the
# RULER-8k climb actually lands rather than where it was at step 40.
# ruler8k_s is pinned to mbs=1 for the 12k run's memory cliff; the 8k run cleared mbs=2
# repeatedly, and at mbs=2 it finishes in ~4 h instead of ~7 h, which is the difference
# between landing before the deadline and not.
CUSTOM_TASKS['ruler8k_mbs2'] = replace(TASKS['ruler8k_s'], mbs=2, est_minutes=220)
CUSTOM_SUITES['ruler8k-fast'] = ['ruler8k_mbs2']

# 2026-08-17 11:10: a 390-document nested subset of ruler4k_s (30 of each subtask's
# committed 100) so the 4k/8k/12k step-80 head-to-head can land inside a ~1 h window.
# Shorter walltime also backfills sooner, which is the actual binding constraint --
# the full-size jobs were sitting in the queue on scheduler priority, not running slowly.
_r4 = TASKS['ruler4k_s']
CUSTOM_TASKS['ruler4k_s390'] = replace(_r4, limit=390, est_minutes=35,
    extra_eval_flags=_r4.extra_eval_flags.replace('ruler4k_s.json', 'ruler4k_s390.json'))
CUSTOM_SUITES['ruler4k-quick'] = ['ruler4k_s390']

# 2026-08-20: base-model-only RULER sweep at 8k / 12k / 16k. RULER takes arbitrary
# context lengths -- max_seq_lengths is metadata (niah_utils.py:24), 4096 is only the
# default -- so 12k needed nothing but its own metadata file. Cost scales with the
# square of the length: measured 6.0 docs/min at 8k and 1.66 at 16k.
_r8 = TASKS['ruler8k_s']
CUSTOM_TASKS['ruler12k_s'] = replace(_r8, seqlen=12288, mbs=1, est_minutes=560,
    seqlen_src='assumed:2026-08-20 interpolated between measured 8k and 16k rates',
    extra_eval_flags=_r8.extra_eval_flags.replace('ruler8k_metadata.json',
                                                  'ruler12k_metadata.json'))
CUSTOM_SUITES['ruler-base-sweep'] = ['ruler8k_s', 'ruler12k_s', 'ruler16k_s']

# 2026-08-20 (doc-sharding for parallel single-link cells): round-robin disjoint
# shards of the committed manifests, one independent 1-node job per shard, merged
# afterwards with rescore_partial under the parent manifest. Distinct row ids per
# shard keep job names (and the submit-time dedup key) unique. Sizing: est*1.3
# must stay under the 105-min link budget so every shard is a SINGLE link.
#   ruler4k 2-way (debug validation of the shard mechanism): 650 docs @ 21.9/min ~ 30 min
#   ruler12k 8-way: 163-169 docs @ 2.61/min ~ 65 min
# NOTE: shard limits are deliberately DISTINCT (650/651, and 169..172/156..159 below):
# the launcher's inflight/done dedup key is (run, step, task, fs, seqlen, padding, limit)
# WITHOUT the row id, so equal-limit shards skip each other. lm-eval ignores limit
# entirely when a samples manifest is given (api/task.py doc_iterator passes limit=None),
# so the value is pure bookkeeping and can exceed the shard's doc count safely.
_r4s = TASKS['ruler4k_s']
for _i in (1, 2):
    CUSTOM_TASKS[f'ruler4k_shd{_i}'] = replace(_r4s, limit=649 + _i, est_minutes=40,
        extra_eval_flags=_r4s.extra_eval_flags.replace(
            'subsets/ruler4k_s.json', f'subsets/ruler4k_shd{_i}of2.json'))
    CUSTOM_SUITES[f'ruler4k-shd{_i}'] = [f'ruler4k_shd{_i}']
_r12 = CUSTOM_TASKS['ruler12k_s']
# ruler8k 5-way and longbench 3-way shards (2026-08-20, same pattern): parallel
# cache-fillers for the step-20 wave -- the unsharded chains then finish from the
# shared response cache. Distinct limits (dedup key), single-link est sizing.
_r8s = TASKS['ruler8k_s']
for _i in range(1, 6):
    CUSTOM_TASKS[f'ruler8k_shd{_i}'] = replace(_r8s, limit=259 + _i, est_minutes=60,
        extra_eval_flags=_r8s.extra_eval_flags.replace(
            'subsets/ruler8k_s.json', f'subsets/ruler8k_shd{_i}of5.json'))
    CUSTOM_SUITES[f'ruler8k-shd{_i}'] = [f'ruler8k_shd{_i}']
_lbs = TASKS['longbench_s']
for _i in range(1, 4):
    CUSTOM_TASKS[f'longbench_shd{_i}'] = replace(_lbs, limit=152 + _i, est_minutes=55,
        extra_eval_flags=_lbs.extra_eval_flags.replace(
            'subsets/longbench_s.json', f'subsets/longbench_shd{_i}of3.json'))
    CUSTOM_SUITES[f'longbench-shd{_i}'] = [f'longbench_shd{_i}']

# FINE shards (2026-08-20 evening, 'one checkpoint in 1-2h' widening): 8k x8 and
# 12k x16, every shard a single short link. Limits distinct (dedup key); offset
# ranges chosen to never collide with the coarse shard rows above.
_r8s2 = TASKS['ruler8k_s']
for _i in range(1, 9):
    CUSTOM_TASKS[f'ruler8k_f{_i}'] = replace(_r8s2, limit=300 + _i, est_minutes=50,
        extra_eval_flags=_r8s2.extra_eval_flags.replace(
            'subsets/ruler8k_s.json', f'subsets/ruler8k_f{_i}of8.json'))
    CUSTOM_SUITES[f'ruler8k-f{_i}'] = [f'ruler8k_f{_i}']
_r12f = CUSTOM_TASKS['ruler12k_s']
for _i in range(1, 17):
    CUSTOM_TASKS[f'ruler12k_f{_i}'] = replace(_r12f, limit=400 + _i, est_minutes=55,
        extra_eval_flags=_r12f.extra_eval_flags.replace(
            'subsets/ruler8k_s.json', f'subsets/ruler12k_f{_i}of16.json'))
    CUSTOM_SUITES[f'ruler12k-f{_i}'] = [f'ruler12k_f{_i}']

# 2026-08-21 v2 gen budgets (user directive: official recommended gen tokens).
# RULER rows stay gen_toks=128 -- that IS the official budget for niah (8/13 subtasks)
# and cwe's 120; never below official anywhere. LongBench drops 512 -> 128, the
# official max across our 9 leaves (dataset2maxlen: 32-128) -- the real waste cut.
_lbv = TASKS['longbench_s']
CUSTOM_TASKS['longbench_v2'] = replace(_lbv, gen_toks=128, est_minutes=60)
CUSTOM_SUITES['longbench-v2'] = ['longbench_v2']
# ruler16k 16-way shards for the 16k-model morning suite; official gen 128 already.
_r16 = TASKS['ruler16k_s']
_l16 = {i: 400+i for i in range(1,17)}  # distinct limits, dedup-key safe, > any shard's docs? shards are 78-82; limit is bookkeeping only (samples pin docs)
for _i in range(1, 17):
    CUSTOM_TASKS[f'ruler16k_f{_i}'] = replace(_r16, limit=_l16[_i], est_minutes=45,
        extra_eval_flags=_r16.extra_eval_flags.replace(
            'subsets/ruler16k_s.json', f'subsets/ruler16k_f{_i}of16.json'))
    CUSTOM_SUITES[f'ruler16k-f{_i}'] = [f'ruler16k_f{_i}']

# base-model 4k tail completion (2026-08-21): the 5 non-niah subtasks at full 100
# docs (old cache holds only 30 each from the s390 era).
_r4t = TASKS['ruler4k_s']
CUSTOM_TASKS['ruler4k_tails'] = replace(_r4t, limit=500, est_minutes=45,
    extra_eval_flags=_r4t.extra_eval_flags.replace(
        'subsets/ruler4k_s.json', 'subsets/ruler4k_tails.json'))
CUSTOM_SUITES['ruler4k-tails'] = ['ruler4k_tails']

_r16t = TASKS['ruler16k_s']
CUSTOM_TASKS['ruler16k_qatails'] = replace(_r16t, limit=200, est_minutes=80,
    extra_eval_flags=_r16t.extra_eval_flags.replace(
        'subsets/ruler16k_s.json', 'subsets/ruler16k_qatails.json'))
CUSTOM_SUITES['ruler16k-qatails'] = ['ruler16k_qatails']

# kv-per-row variants of the 8k fine shards (2026-08-21 evening): the ragged batched
# KV decode hangs at step 1 on the lc16k-idoc s40 @ 8k cells (the known unlanded
# fp32-KV-transient defect). --kv-per-row is a RUNTIME flag -- no source change,
# no code_fp fork -- routing decode through the per-row path. Slower, immune.
for _i in range(1, 9):
    _src = CUSTOM_TASKS[f'ruler8k_f{_i}']
    CUSTOM_TASKS[f'ruler8k_kvr{_i}'] = replace(_src, limit=320 + _i, est_minutes=75,
        extra_eval_flags=(_src.extra_eval_flags + ' --kv-per-row'))
    CUSTOM_SUITES[f'ruler8k-kvr{_i}'] = [f'ruler8k_kvr{_i}']

# LongBench v2 3-way shards (2026-08-22 night): the single longbench_v2 job clips
# samsum's tail (last in doc order). Shards spread the tail; official 128 budget kept.
_lbv2 = CUSTOM_TASKS['longbench_v2']
for _i in range(1, 4):
    CUSTOM_TASKS[f'longbench_v2s{_i}'] = replace(_lbv2, limit=145 + _i, est_minutes=35,
        extra_eval_flags=_lbv2.extra_eval_flags.replace(
            'subsets/longbench_s.json', f'subsets/longbench_shd{_i}of3.json'))
    CUSTOM_SUITES[f'longbench-v2s{_i}'] = [f'longbench_v2s{_i}']

# Gap-closure rows (2026-08-22 midday): the tails/qa completions hit the ragged-
# decode hang on wide same-length batches (three sightings now). kv-per-row for
# the 4k/8k tails; ests sized to measured per-row pace.
_r4t2 = CUSTOM_TASKS['ruler4k_tails']
CUSTOM_TASKS['ruler4k_tails_kvr'] = replace(_r4t2, limit=502, est_minutes=100,
    extra_eval_flags=(_r4t2.extra_eval_flags + ' --kv-per-row'))
CUSTOM_SUITES['ruler4k-tails-kvr'] = ['ruler4k_tails_kvr']
_r8t = TASKS['ruler8k_s']
CUSTOM_TASKS['ruler8k_qatails'] = replace(_r8t, limit=201, est_minutes=110,
    extra_eval_flags=(_r8t.extra_eval_flags.replace('subsets/ruler8k_s.json',
        'subsets/ruler8k_qatails.json') + ' --kv-per-row'))
CUSTOM_SUITES['ruler8k-qatails'] = ['ruler8k_qatails']

_shard_limits = {1: 169, 2: 170, 3: 171, 4: 172, 5: 156, 6: 157, 7: 158, 8: 159}
for _i in range(1, 9):
    CUSTOM_TASKS[f'ruler12k_shd{_i}'] = replace(_r12, limit=_shard_limits[_i], est_minutes=75,
        extra_eval_flags=_r12.extra_eval_flags.replace(
            'subsets/ruler8k_s.json', f'subsets/ruler12k_shd{_i}of8.json'))
    CUSTOM_SUITES[f'ruler12k-shd{_i}'] = [f'ruler12k_shd{_i}']

# Per-subtask closure rows (2026-08-22 evening): lm-eval's doc order starves the
# later tails inside a shared manifest -- every timed-out kvr attempt re-ate vt
# first. One row per subtask makes the five starve-proof and parallel. Distinct
# limits keep launcher dedup keys apart; samples manifests pin the actual docs.
_t4k = CUSTOM_TASKS['ruler4k_tails_kvr']
for _n, _l in {'vt': 511, 'cwe': 512, 'fwe': 513, 'qa1': 514, 'qa2': 515}.items():
    CUSTOM_TASKS[f'ruler4k_{_n}_kvr'] = replace(_t4k, limit=_l, est_minutes=75,
        extra_eval_flags=_t4k.extra_eval_flags.replace(
            'subsets/ruler4k_tails.json', f'subsets/ruler4k_1{_n}.json'))
    CUSTOM_SUITES[f'ruler4k-{_n}-kvr'] = [f'ruler4k_{_n}_kvr']
_t8q = CUSTOM_TASKS['ruler8k_qatails']
CUSTOM_TASKS['ruler8k_qa2_kvr'] = replace(_t8q, limit=205, est_minutes=65,
    extra_eval_flags=_t8q.extra_eval_flags.replace(
        'subsets/ruler8k_qatails.json', 'subsets/ruler8k_1qa2.json'))
CUSTOM_SUITES['ruler8k-qa2-kvr'] = ['ruler8k_qa2_kvr']
# v1-budget (gen 512) LongBench 3-way shards: the lc8k-phase cells were measured
# at gen 512, so v2 (gen 128) responses can never satisfy them -- these shards
# stay on the longbench_s row budget.
_lb1 = TASKS['longbench_s']
for _i in range(1, 4):
    CUSTOM_TASKS[f'longbench_s_shd{_i}'] = replace(_lb1, limit=150 + _i, est_minutes=80,
        extra_eval_flags=_lb1.extra_eval_flags.replace(
            'subsets/longbench_s.json', f'subsets/longbench_shd{_i}of3.json'))
    CUSTOM_SUITES[f'longbench-s-shd{_i}'] = [f'longbench_s_shd{_i}']

# ---------------------------------------------------------------------------------
# 32k stage support (2026-08-22, rigor sprint). 25 docs/subtask (325 total): at the
# measured ~2x-per-length-doubling slowdown (16k = 1.66 docs/min) a 100-doc set is
# ~13h unsharded. est_minutes assumes ~0.7 docs/min; shard at submit time like the
# 16k f-shards, 90-min walls (74-min walls killed the s360 16k shards on 08-22).
# PREREQUISITE: the chunked fp32-KV patch must be landed before any 32k decode job.
# ---------------------------------------------------------------------------------
CUSTOM_TASKS.update({
    'ruler32k_s': replace(TASKS['ruler16k_s'], seqlen=32768, est_minutes=480,
                          limit=25,
                          extra_eval_flags='--task-metadata-json-file /lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier/eval/config/ruler32k_metadata.json --task-samples-json-file /lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier/eval/config/subsets/ruler32k_s.json',
                          seqlen_src='measured:2026-08-28 max=32894 p99=32743'),
})
CUSTOM_SUITES.update({
    # trimmed per-checkpoint cadence grid (audit 2026-08-22): 16k-stage ckpts
    'lc-cadence':    ['ruler8k_s', 'ruler16k_s', 'longbench_s', 'hellaswag_s'],
    # once the 32k stage runs:
    'lc-cadence32':  ['ruler8k_s', 'ruler16k_s', 'ruler32k_s', 'longbench_s', 'hellaswag_s'],
})

# ---------------------------------------------------------------------------------
# needle-PPL monitor (2026-08-22): scoring-only logprob of a 7-digit needle answer
# at controlled depths in synthetic noise haystacks. Prefill-only -> immune to the
# ragged-decode hang; minutes per checkpoint. Rationale + evidence: Hunyuan LCCP
# (arXiv:2604.02650) needle-PPL predicts post-SFT better than NIAH (-0.821 vs .749).
# Datasets are committed under custom_tasks/needleppl/data (seed 20260814, sha in
# manifest.json). needleppl = 4k+8k+16k group at seqlen 16384; the 32k task rides
# in 32k-capable jobs only.
# ---------------------------------------------------------------------------------
CUSTOM_TASKS.update({
    'needleppl': Row('needleppl', 'ppl', 'loglik', 16384, mbs=1, est_minutes=45,
                     seqlen_src='measured:2026-08-30 max=15626 p100 over all 150 docs (manifest.json tok_range, DeepSeek tokenizer, seed 20260814); 758 tokens headroom under 16384',
                     extra_eval_flags='--include-path /lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier/eval/custom_tasks'),
    'needleppl32': Row('needleppl_32k', 'ppl', 'loglik', 32768, mbs=1, est_minutes=30,
                     seqlen_src='measured:2026-08-30 max=31286 (manifest.json tok_range); 1482 tokens headroom under 32768',
                     extra_eval_flags='--include-path /lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier/eval/custom_tasks'),
    # 64k, added 2026-08-30. Prefill-only loglik of the needle answer, so it does NOT hit the
    # ragged-decode hang that killed generation at long context. NOTE: this is the first eval row
    # above 32,768 -- the one 64k job ever attempted (probe 5374225, EP8/TP1/PP1, 1 node) OOM'd in
    # the MoE all-to-all regather at deepspeed/moe/moe_v2.py:366, needing 6.00 GiB with 4.20 free.
    # It is worth retrying because the campaign topology is different in the way that matters:
    # under TP4 + companion SP the sequence is sharded 4 ways, so per-rank load is 16,384 tokens,
    # not 65,536. Treat the first run as a probe, not as a number.
    'needleppl16k200': Row('needleppl_16k_n200', 'ppl', 'loglik', 16384, mbs=1, est_minutes=40,
                     seqlen_src='measured:2026-08-30 n200 set, same generator/margin as the 16k row (max<15700)',
                     extra_eval_flags='--include-path /lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier/eval/custom_tasks'),
    'needleppl128': Row('needleppl_128k', 'ppl', 'loglik', 131072, mbs=1, est_minutes=60,
                     seqlen_src='measured:2026-08-30 max=125270 (manifest.json tok_range); 5802 tokens headroom under 131072',
                     extra_eval_flags='--include-path /lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier/eval/custom_tasks'),
    'needleppl64': Row('needleppl_64k', 'ppl', 'loglik', 65536, mbs=1, est_minutes=60,
                     seqlen_src='measured:2026-08-30 max=62627 (manifest.json tok_range); 2909 tokens headroom under 65536',
                     extra_eval_flags='--include-path /lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier/eval/custom_tasks'),
})
CUSTOM_SUITES['lc-cadence'] = CUSTOM_SUITES['lc-cadence'] + ['needleppl']
CUSTOM_SUITES['lc-cadence32'] = CUSTOM_SUITES['lc-cadence32'] + ['needleppl', 'needleppl32']

# 2026-08-23 KV-patch score A/B (user-ordered): ONE subtask (niah_single_1, its
# committed 100 docs) scored end-to-end twice -- 'kv' = patched ragged decode
# (production path; its generations already sit in the f1-shard cache), 'nokv' =
# legacy full-reforward oracle (submit with NO_KV_DECODE=1; cache file gets the
# _nokv suffix so the arms never share responses). Distinct limits = dedup key,
# same house pattern as the shard rows above.
_r8ab = TASKS['ruler8k_s']
CUSTOM_TASKS['niah1_kv'] = replace(_r8ab, limit=100, est_minutes=35,
    extra_eval_flags=_r8ab.extra_eval_flags.replace(
        'subsets/ruler8k_s.json', 'subsets/niah1_kvab.json'))
CUSTOM_TASKS['niah1_nokv'] = replace(_r8ab, limit=99, est_minutes=85,
    extra_eval_flags=_r8ab.extra_eval_flags.replace(
        'subsets/ruler8k_s.json', 'subsets/niah1_kvab.json'))
CUSTOM_SUITES['niah1-kv'] = ['niah1_kv']
CUSTOM_SUITES['niah1-nokv'] = ['niah1_nokv']

# 2026-08-23 SFT 0.1B-token eval cadence (user spec: instruction following, QA,
# long-context retention). One lean trio per SFT checkpoint; the full
# instruct-short suite runs once on the final SFT checkpoint.
CUSTOM_SUITES['sft-cadence'] = ['ifeval_s', 'triviaqa_s', 'needleppl']

# 2026-08-24 preliminary 32k sense-check (user spec): ONE niah subtask
# (niah_single_1, 100 committed docs) at 16k and 32k context.
CUSTOM_TASKS['niah1_16k'] = replace(TASKS['ruler16k_s'], limit=100, est_minutes=240,  # measured: no-KV gen ~2-4 min/doc at 16k
    extra_eval_flags=TASKS['ruler16k_s'].extra_eval_flags.replace(
        'subsets/ruler16k_s.json', 'subsets/niah1_16k.json'))
CUSTOM_TASKS['niah1_32k'] = replace(CUSTOM_TASKS['ruler32k_s'], limit=100, est_minutes=600,
    extra_eval_flags=CUSTOM_TASKS['ruler32k_s'].extra_eval_flags.replace(
        'subsets/ruler32k_s.json', 'subsets/niah1_32k.json'))
CUSTOM_SUITES['niah1-16k'] = ['niah1_16k']
CUSTOM_SUITES['niah1-32k'] = ['niah1_32k']

# 2026-08-24 ground-truth pass (user): FULL-size runs of the light instruct +
# commonsense benchmarks on BOTH the final SFT checkpoint and the untouched
# DeepSeek-MoE-16B base, so every number is like-for-like. Heavies (bbh, math,
# ifeval) deliberately excluded - they cost ~504 one-hour links alone.
CUSTOM_SUITES['ground-truth'] = ['gsm8k_chat', 'humaneval', 'mbpp', 'triviaqa',
                                 'hellaswag', 'lambada', 'boolq', 'piqa',
                                 'winogrande', 'openbookqa']

# 2026-08-25: base humaneval pass@1=0.0000 (jobs 5280476-8) is implausible for a
# 16B base model; rerun standalone to confirm or replace it.
CUSTOM_SUITES['humaneval-only'] = ['humaneval']

# 2026-08-25 CORRECTED-CONFIG ground truth: bbh sharded 9-way by its native
# subtasks and minerva_math 7-way by subject, so each shard is ~1.4 node-h and
# the whole benchmark finishes in one evening instead of 12 sequential hours.
# gen_toks/seqlen inherit the verified 'bbh'/'math' rows (256 gen, 4096/2560 seq).
_BBH = TASKS['bbh']
_MATH = TASKS['math']
CUSTOM_TASKS['bbh_sh1_0'] = replace(_BBH, task='bbh_cot_fewshot_boolean_expressions', est_minutes=40)
CUSTOM_TASKS['bbh_sh1_1'] = replace(_BBH, task='bbh_cot_fewshot_causal_judgement', est_minutes=40)
CUSTOM_TASKS['bbh_sh1_2'] = replace(_BBH, task='bbh_cot_fewshot_date_understanding', est_minutes=40)
CUSTOM_SUITES['bbh-sh1'] = ['bbh_sh1_0', 'bbh_sh1_1', 'bbh_sh1_2']
CUSTOM_TASKS['bbh_sh2_0'] = replace(_BBH, task='bbh_cot_fewshot_disambiguation_qa', est_minutes=40)
CUSTOM_TASKS['bbh_sh2_1'] = replace(_BBH, task='bbh_cot_fewshot_dyck_languages', est_minutes=40)
CUSTOM_TASKS['bbh_sh2_2'] = replace(_BBH, task='bbh_cot_fewshot_formal_fallacies', est_minutes=40)
CUSTOM_SUITES['bbh-sh2'] = ['bbh_sh2_0', 'bbh_sh2_1', 'bbh_sh2_2']
CUSTOM_TASKS['bbh_sh3_0'] = replace(_BBH, task='bbh_cot_fewshot_geometric_shapes', est_minutes=40)
CUSTOM_TASKS['bbh_sh3_1'] = replace(_BBH, task='bbh_cot_fewshot_hyperbaton', est_minutes=40)
CUSTOM_TASKS['bbh_sh3_2'] = replace(_BBH, task='bbh_cot_fewshot_logical_deduction_five_objects', est_minutes=40)
CUSTOM_SUITES['bbh-sh3'] = ['bbh_sh3_0', 'bbh_sh3_1', 'bbh_sh3_2']
CUSTOM_TASKS['bbh_sh4_0'] = replace(_BBH, task='bbh_cot_fewshot_logical_deduction_seven_objects', est_minutes=40)
CUSTOM_TASKS['bbh_sh4_1'] = replace(_BBH, task='bbh_cot_fewshot_logical_deduction_three_objects', est_minutes=40)
CUSTOM_TASKS['bbh_sh4_2'] = replace(_BBH, task='bbh_cot_fewshot_movie_recommendation', est_minutes=40)
CUSTOM_SUITES['bbh-sh4'] = ['bbh_sh4_0', 'bbh_sh4_1', 'bbh_sh4_2']
CUSTOM_TASKS['bbh_sh5_0'] = replace(_BBH, task='bbh_cot_fewshot_multistep_arithmetic_two', est_minutes=40)
CUSTOM_TASKS['bbh_sh5_1'] = replace(_BBH, task='bbh_cot_fewshot_navigate', est_minutes=40)
CUSTOM_TASKS['bbh_sh5_2'] = replace(_BBH, task='bbh_cot_fewshot_object_counting', est_minutes=40)
CUSTOM_SUITES['bbh-sh5'] = ['bbh_sh5_0', 'bbh_sh5_1', 'bbh_sh5_2']
CUSTOM_TASKS['bbh_sh6_0'] = replace(_BBH, task='bbh_cot_fewshot_penguins_in_a_table', est_minutes=40)
CUSTOM_TASKS['bbh_sh6_1'] = replace(_BBH, task='bbh_cot_fewshot_reasoning_about_colored_objects', est_minutes=40)
CUSTOM_TASKS['bbh_sh6_2'] = replace(_BBH, task='bbh_cot_fewshot_ruin_names', est_minutes=40)
CUSTOM_SUITES['bbh-sh6'] = ['bbh_sh6_0', 'bbh_sh6_1', 'bbh_sh6_2']
CUSTOM_TASKS['bbh_sh7_0'] = replace(_BBH, task='bbh_cot_fewshot_salient_translation_error_detection', est_minutes=40)
CUSTOM_TASKS['bbh_sh7_1'] = replace(_BBH, task='bbh_cot_fewshot_snarks', est_minutes=40)
CUSTOM_TASKS['bbh_sh7_2'] = replace(_BBH, task='bbh_cot_fewshot_sports_understanding', est_minutes=40)
CUSTOM_SUITES['bbh-sh7'] = ['bbh_sh7_0', 'bbh_sh7_1', 'bbh_sh7_2']
CUSTOM_TASKS['bbh_sh8_0'] = replace(_BBH, task='bbh_cot_fewshot_temporal_sequences', est_minutes=40)
CUSTOM_TASKS['bbh_sh8_1'] = replace(_BBH, task='bbh_cot_fewshot_tracking_shuffled_objects_five_objects', est_minutes=40)
CUSTOM_TASKS['bbh_sh8_2'] = replace(_BBH, task='bbh_cot_fewshot_tracking_shuffled_objects_seven_objects', est_minutes=40)
CUSTOM_SUITES['bbh-sh8'] = ['bbh_sh8_0', 'bbh_sh8_1', 'bbh_sh8_2']
CUSTOM_TASKS['bbh_sh9_0'] = replace(_BBH, task='bbh_cot_fewshot_tracking_shuffled_objects_three_objects', est_minutes=40)
CUSTOM_TASKS['bbh_sh9_1'] = replace(_BBH, task='bbh_cot_fewshot_web_of_lies', est_minutes=40)
CUSTOM_TASKS['bbh_sh9_2'] = replace(_BBH, task='bbh_cot_fewshot_word_sorting', est_minutes=40)
CUSTOM_SUITES['bbh-sh9'] = ['bbh_sh9_0', 'bbh_sh9_1', 'bbh_sh9_2']
CUSTOM_TASKS['math_sh1'] = replace(_MATH, task='minerva_math_algebra', est_minutes=60)
CUSTOM_SUITES['math-sh1'] = ['math_sh1']
CUSTOM_TASKS['math_sh2'] = replace(_MATH, task='minerva_math_counting_and_prob', est_minutes=60)
CUSTOM_SUITES['math-sh2'] = ['math_sh2']
CUSTOM_TASKS['math_sh3'] = replace(_MATH, task='minerva_math_geometry', est_minutes=60)
CUSTOM_SUITES['math-sh3'] = ['math_sh3']
CUSTOM_TASKS['math_sh4'] = replace(_MATH, task='minerva_math_intermediate_algebra', est_minutes=60)
CUSTOM_SUITES['math-sh4'] = ['math_sh4']
CUSTOM_TASKS['math_sh5'] = replace(_MATH, task='minerva_math_num_theory', est_minutes=60)
CUSTOM_SUITES['math-sh5'] = ['math_sh5']
CUSTOM_TASKS['math_sh6'] = replace(_MATH, task='minerva_math_prealgebra', est_minutes=60)
CUSTOM_SUITES['math-sh6'] = ['math_sh6']
CUSTOM_TASKS['math_sh7'] = replace(_MATH, task='minerva_math_precalc', est_minutes=60)
CUSTOM_SUITES['math-sh7'] = ['math_sh7']
CUSTOM_SUITES['humaneval-only'] = ['humaneval']
CUSTOM_SUITES['mbpp-only'] = ['mbpp']
CUSTOM_SUITES['gsm8k-only'] = ['gsm8k']
CUSTOM_SUITES['triviaqa-only'] = ['triviaqa']
CUSTOM_SUITES['ifeval-only'] = ['ifeval']

# 2026-08-25 MMLU 5-shot on the SFT model, sharded 20 ways over the 57 native
# subjects. User measured ~13 node-h for the whole benchmark single-node, so
# each shard is ~0.7 node-h; est_minutes set from that, not the stale table value.
_MMLU5 = TASKS['mmlu_5shot']
CUSTOM_TASKS['mmlu5_sh1_0'] = replace(_MMLU5, task='mmlu_abstract_algebra', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh1_1'] = replace(_MMLU5, task='mmlu_high_school_computer_science', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh1_2'] = replace(_MMLU5, task='mmlu_medical_genetics', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh1'] = ['mmlu5_sh1_0', 'mmlu5_sh1_1', 'mmlu5_sh1_2']
CUSTOM_TASKS['mmlu5_sh2_0'] = replace(_MMLU5, task='mmlu_anatomy', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh2_1'] = replace(_MMLU5, task='mmlu_high_school_european_history', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh2_2'] = replace(_MMLU5, task='mmlu_miscellaneous', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh2'] = ['mmlu5_sh2_0', 'mmlu5_sh2_1', 'mmlu5_sh2_2']
CUSTOM_TASKS['mmlu5_sh3_0'] = replace(_MMLU5, task='mmlu_astronomy', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh3_1'] = replace(_MMLU5, task='mmlu_high_school_geography', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh3_2'] = replace(_MMLU5, task='mmlu_moral_disputes', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh3'] = ['mmlu5_sh3_0', 'mmlu5_sh3_1', 'mmlu5_sh3_2']
CUSTOM_TASKS['mmlu5_sh4_0'] = replace(_MMLU5, task='mmlu_business_ethics', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh4_1'] = replace(_MMLU5, task='mmlu_high_school_government_and_politics', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh4_2'] = replace(_MMLU5, task='mmlu_moral_scenarios', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh4'] = ['mmlu5_sh4_0', 'mmlu5_sh4_1', 'mmlu5_sh4_2']
CUSTOM_TASKS['mmlu5_sh5_0'] = replace(_MMLU5, task='mmlu_clinical_knowledge', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh5_1'] = replace(_MMLU5, task='mmlu_high_school_macroeconomics', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh5_2'] = replace(_MMLU5, task='mmlu_nutrition', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh5'] = ['mmlu5_sh5_0', 'mmlu5_sh5_1', 'mmlu5_sh5_2']
CUSTOM_TASKS['mmlu5_sh6_0'] = replace(_MMLU5, task='mmlu_college_biology', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh6_1'] = replace(_MMLU5, task='mmlu_high_school_mathematics', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh6_2'] = replace(_MMLU5, task='mmlu_philosophy', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh6'] = ['mmlu5_sh6_0', 'mmlu5_sh6_1', 'mmlu5_sh6_2']
CUSTOM_TASKS['mmlu5_sh7_0'] = replace(_MMLU5, task='mmlu_college_chemistry', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh7_1'] = replace(_MMLU5, task='mmlu_high_school_microeconomics', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh7_2'] = replace(_MMLU5, task='mmlu_prehistory', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh7'] = ['mmlu5_sh7_0', 'mmlu5_sh7_1', 'mmlu5_sh7_2']
CUSTOM_TASKS['mmlu5_sh8_0'] = replace(_MMLU5, task='mmlu_college_computer_science', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh8_1'] = replace(_MMLU5, task='mmlu_high_school_physics', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh8_2'] = replace(_MMLU5, task='mmlu_professional_accounting', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh8'] = ['mmlu5_sh8_0', 'mmlu5_sh8_1', 'mmlu5_sh8_2']
CUSTOM_TASKS['mmlu5_sh9_0'] = replace(_MMLU5, task='mmlu_college_mathematics', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh9_1'] = replace(_MMLU5, task='mmlu_high_school_psychology', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh9_2'] = replace(_MMLU5, task='mmlu_professional_law', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh9'] = ['mmlu5_sh9_0', 'mmlu5_sh9_1', 'mmlu5_sh9_2']
CUSTOM_TASKS['mmlu5_sh10_0'] = replace(_MMLU5, task='mmlu_college_medicine', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh10_1'] = replace(_MMLU5, task='mmlu_high_school_statistics', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh10_2'] = replace(_MMLU5, task='mmlu_professional_medicine', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh10'] = ['mmlu5_sh10_0', 'mmlu5_sh10_1', 'mmlu5_sh10_2']
CUSTOM_TASKS['mmlu5_sh11_0'] = replace(_MMLU5, task='mmlu_college_physics', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh11_1'] = replace(_MMLU5, task='mmlu_high_school_us_history', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh11_2'] = replace(_MMLU5, task='mmlu_professional_psychology', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh11'] = ['mmlu5_sh11_0', 'mmlu5_sh11_1', 'mmlu5_sh11_2']
CUSTOM_TASKS['mmlu5_sh12_0'] = replace(_MMLU5, task='mmlu_computer_security', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh12_1'] = replace(_MMLU5, task='mmlu_high_school_world_history', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh12_2'] = replace(_MMLU5, task='mmlu_public_relations', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh12'] = ['mmlu5_sh12_0', 'mmlu5_sh12_1', 'mmlu5_sh12_2']
CUSTOM_TASKS['mmlu5_sh13_0'] = replace(_MMLU5, task='mmlu_conceptual_physics', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh13_1'] = replace(_MMLU5, task='mmlu_human_aging', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh13_2'] = replace(_MMLU5, task='mmlu_security_studies', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh13'] = ['mmlu5_sh13_0', 'mmlu5_sh13_1', 'mmlu5_sh13_2']
CUSTOM_TASKS['mmlu5_sh14_0'] = replace(_MMLU5, task='mmlu_econometrics', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh14_1'] = replace(_MMLU5, task='mmlu_human_sexuality', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh14_2'] = replace(_MMLU5, task='mmlu_sociology', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh14'] = ['mmlu5_sh14_0', 'mmlu5_sh14_1', 'mmlu5_sh14_2']
CUSTOM_TASKS['mmlu5_sh15_0'] = replace(_MMLU5, task='mmlu_electrical_engineering', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh15_1'] = replace(_MMLU5, task='mmlu_international_law', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh15_2'] = replace(_MMLU5, task='mmlu_us_foreign_policy', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh15'] = ['mmlu5_sh15_0', 'mmlu5_sh15_1', 'mmlu5_sh15_2']
CUSTOM_TASKS['mmlu5_sh16_0'] = replace(_MMLU5, task='mmlu_elementary_mathematics', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh16_1'] = replace(_MMLU5, task='mmlu_jurisprudence', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh16_2'] = replace(_MMLU5, task='mmlu_virology', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh16'] = ['mmlu5_sh16_0', 'mmlu5_sh16_1', 'mmlu5_sh16_2']
CUSTOM_TASKS['mmlu5_sh17_0'] = replace(_MMLU5, task='mmlu_formal_logic', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh17_1'] = replace(_MMLU5, task='mmlu_logical_fallacies', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh17_2'] = replace(_MMLU5, task='mmlu_world_religions', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh17'] = ['mmlu5_sh17_0', 'mmlu5_sh17_1', 'mmlu5_sh17_2']
CUSTOM_TASKS['mmlu5_sh18_0'] = replace(_MMLU5, task='mmlu_global_facts', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh18_1'] = replace(_MMLU5, task='mmlu_machine_learning', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh18'] = ['mmlu5_sh18_0', 'mmlu5_sh18_1']
CUSTOM_TASKS['mmlu5_sh19_0'] = replace(_MMLU5, task='mmlu_high_school_biology', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh19_1'] = replace(_MMLU5, task='mmlu_management', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh19'] = ['mmlu5_sh19_0', 'mmlu5_sh19_1']
CUSTOM_TASKS['mmlu5_sh20_0'] = replace(_MMLU5, task='mmlu_high_school_chemistry', est_minutes=45)
CUSTOM_TASKS['mmlu5_sh20_1'] = replace(_MMLU5, task='mmlu_marketing', est_minutes=45)
CUSTOM_SUITES['mmlu5-sh20'] = ['mmlu5_sh20_0', 'mmlu5_sh20_1']
CUSTOM_SUITES['lambada-only'] = ['lambada']
CUSTOM_SUITES['winogrande-only'] = ['winogrande']
CUSTOM_SUITES['openbookqa-only'] = ['openbookqa']
CUSTOM_SUITES['hellaswag-only'] = ['hellaswag']
CUSTOM_SUITES['boolq-only'] = ['boolq']
CUSTOM_SUITES['piqa-only'] = ['piqa']
CUSTOM_SUITES['ruler32k'] = ['ruler32k_s']

# ---------------------------------------------------------------------------------
# 2026-08-29 -- the lc32k-idoc campaign. lc32k-idoc is the FIRST checkpoint trained
# at 32,768, so it is the first model for which a 32k RULER score can be anything
# but zero. The evidence that this is a model-length effect and not a pipeline one:
# every model on record collapses to ~0 exactly when RULER exceeds ITS training
# length and scores well at or below it -- the stock 4k-context deepseek_16b_ds
# reads 0.7267 at 4k and 0.0005 at 8k, in the same pipeline where the 8k-trained
# lc8k-idoc reads 0.7271 at 8k. The 32k zeros on record belong to three models
# trained at 16k or less. `causal_mask_sanity` also PASSED at 32k (job 5347825).
#
# ruler64k_s is deliberately 10 docs/sub-task: at 65,536 it asks lc32k-idoc for 2x
# its training length, so by that same rule it is expected to read ~0. It is priced
# as a cheap boundary probe, not as a headline number.
# ---------------------------------------------------------------------------------
CUSTOM_TASKS['ruler32k_f'] = replace(
    CUSTOM_TASKS['ruler32k_s'], limit=1300, est_minutes=1920,
    extra_eval_flags=CUSTOM_TASKS['ruler32k_s'].extra_eval_flags.replace(
        'subsets/ruler32k_s.json', 'subsets/ruler32k_f.json'))

CUSTOM_TASKS['ruler64k_s'] = replace(
    CUSTOM_TASKS['ruler32k_s'], seqlen=65536, limit=130, est_minutes=650,
    seqlen_src='assumed:2026-08-29 first 64k run -- no measurement exists yet',
    extra_eval_flags=CUSTOM_TASKS['ruler32k_s'].extra_eval_flags
        .replace('ruler32k_metadata.json', 'ruler64k_metadata.json')
        .replace('subsets/ruler32k_s.json', 'subsets/ruler64k_s.json'))

# 2026-08-30 ~21:20 (owner): RULER-128k on the newest checkpoints -- decision instrument.
# No pre-built subset exists at 131072; RULER doc data auto-generates into LM_EVAL_RULER_CACHE
# on first run (hash-keyed by task+seqlen), so this row runs with a plain --limit instead of a
# samples-json. GENERATION GEOMETRY REQUIRED: tp=1 (SSMB seq%TP assert), and the gen path was
# memory-fixed for long seq the same day (lean flash mask + last-position logits).
CUSTOM_TASKS['ruler128k_p'] = replace(
    CUSTOM_TASKS['ruler64k_s'], seqlen=131072, limit=65, est_minutes=650,
    seqlen_src='assumed:2026-08-30 first 128k ruler run -- probe tier',
    extra_eval_flags=CUSTOM_TASKS['ruler64k_s'].extra_eval_flags
        .replace('ruler64k_metadata.json', 'ruler128k_metadata.json')
        .replace(' --task-samples-json-file /lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier/eval/config/subsets/ruler64k_s.json', ''))
CUSTOM_SUITES['ruler-128k-probe'] = ['ruler128k_p']

# 2026-10-04 (user): RULER-128k at 100 docs per sub-task on the 128k-stage checkpoints (lc128k-v2 steps 1000, 800, 600, 400, 200):
# the 128k counterpart of ruler32k_f. Same manifest-less pattern as ruler128k_p (the RULER doc pool at 131,072 is generated once into
# LM_EVAL_RULER_CACHE, hash-keyed by task+seqlen; --limit takes the FIRST 100 docs of every sub-task, a fixed set across checkpoints).
# 13 sub-tasks x 100 = 1,300 docs at 131,072 tokens; est_minutes from ruler128k_p's 2.1 min per doc per worker (16 docs/sub-task,
# 34 min on 16 one-node workers, jobs 5387239/5388644/5388883).
CUSTOM_TASKS['ruler128k_f'] = replace(
    CUSTOM_TASKS['ruler128k_p'], limit=100, est_minutes=2730,
    seqlen_src='assumed:2026-10-04 100-doc tier at 128k, from the 16-doc probe runs of 2026-08-30')
CUSTOM_SUITES['ruler-128k-full'] = ['ruler128k_f']

# 2026-08-31 (owner): 25 docs/sub-task tier for the 64k boundary on the pause checkpoint.
# Same manifest-less pattern as ruler128k_p: run-line limit passes per-task to lm-eval.
CUSTOM_TASKS['ruler64k_s25'] = replace(
    CUSTOM_TASKS['ruler64k_s'], limit=25, est_minutes=120,
    extra_eval_flags=CUSTOM_TASKS['ruler64k_s'].extra_eval_flags
        .replace(' --task-samples-json-file /lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier/eval/config/subsets/ruler64k_s.json', ''))

# 2026-08-30: niah_single_1-ONLY 64k probe row (owner: one subtask, not the suite).
# Only 10 docs/task exist at 64k; 100-doc coverage at 64k+ needs a data-generation pass.
CUSTOM_TASKS['niah_64k'] = replace(CUSTOM_TASKS['ruler64k_s'], limit=10, est_minutes=90,
    seqlen_src=CUSTOM_TASKS['ruler64k_s'].seqlen_src.replace('assumed:2026-08-29 first 64k run -- no measurement exists yet',
        'assumed:2026-08-30 niah-only; bounded by ruler64k tok budget'),
    extra_eval_flags=CUSTOM_TASKS['ruler64k_s'].extra_eval_flags.replace(
        'subsets/ruler64k_s.json', 'subsets/niah_64k.json'))
CUSTOM_SUITES['niah-64k'] = ['niah_64k']


# ---------------------------------------------------------------------------------
# 2026-09-04: FULL-coverage manifest rows (suffix _f) for the reproduction. A row that
# pins a --task-samples-json-file manifest can be split across worker nodes; these
# manifests (config/subsets/<task>.json, built by sft_repro/make_full_manifests.py)
# list EVERY document of the task in dataset order, so the measurement is the whole
# benchmark, identical to the plain row, just splittable to fit a 20-minute debug job.
# ---------------------------------------------------------------------------------
import os as _os
_CFG = _os.path.dirname(_os.path.abspath(__file__)) if '__file__' in globals() else _os.path.join(_os.getcwd(), 'config')
_FULL = lambda base: replace(TASKS[base], extra_eval_flags='--task-samples-json-file ' +
                             _os.path.join(_CFG, 'subsets', base + '.json'),
                             seqlen_src=TASKS[base].seqlen_src + '; full manifest 2026-09-04')
CUSTOM_TASKS.update({
    'arc_challenge_f': _FULL('arc_challenge'),
    'boolq_f':         _FULL('boolq'),
    'hellaswag_f':     _FULL('hellaswag'),
    'openbookqa_f':    _FULL('openbookqa'),
    'piqa_f':          _FULL('piqa'),
    'winogrande_f':    _FULL('winogrande'),
})
CUSTOM_TASKS['rte_f'] = replace(CUSTOM_TASKS['rte'], extra_eval_flags='--task-samples-json-file ' +
                                _os.path.join(_CFG, 'subsets', 'rte.json'))
CUSTOM_SUITES.update({
    'repro7f': ['arc_challenge_f', 'boolq_f', 'hellaswag_f', 'openbookqa_f', 'piqa_f', 'rte_f', 'winogrande_f'],
})

# ---------------------------------------------------------------------------------
# Moonlight-16B-A3B rows (2026-09-12): the released DeepSeek-V3-architecture model on the
# X-MoE path (universal atoms from the HF import, registry Moonlight_16B_A3B), scored with
# the benchmarks of its model card that our suite already has. The template's tokenizer
# is the DeepSeek one; these rows append Moonlight's own tokenizer.json (converted from its
# tiktoken model, verified token-for-token on four texts; re-saved in the merges-as-strings form the
# campaign environment's tokenizers 0.15 reads, as tokenizer.old.json) and its EOS through the extra
# flags, which come last on the launch line. Shots follow the Moonlight report: MMLU
# 5-shot, TriviaQA 5-shot, GSM8K 4-shot, HumanEval pass@1.
# ---------------------------------------------------------------------------------
_MOON_FLAGS = "--vocab-file /lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/assistant_deep_dive/2026-09-10-deepseek_v3/tools/hf_v3/moonlight_tokenizer/tokenizer.old.json --eod-token '[EOS]'"
CUSTOM_TASKS.update({
    'mmlu_5shot_moon': replace(TASKS['mmlu_5shot'], extra_eval_flags=_MOON_FLAGS),
    'triviaqa_moon':   replace(TASKS['triviaqa'], extra_eval_flags=_MOON_FLAGS),
    'gsm8k_4shot_moon': replace(TASKS['gsm8k'], fewshot=4, est_minutes=120,
                                seqlen_src='measured:2026-09-12 max=1715 p99=1508', extra_eval_flags=_MOON_FLAGS),
    'humaneval_moon':  replace(TASKS['humaneval'], extra_eval_flags='--allow-code-eval ' + _MOON_FLAGS),
})
CUSTOM_SUITES = dict(globals().get('CUSTOM_SUITES', {}), **{'moonlight': ['mmlu_5shot_moon', 'triviaqa_moon', 'gsm8k_4shot_moon', 'humaneval_moon']})

# ---------------------------------------------------------------------------------
# DeepSeek-V3 671B rows (2026-09-12): the released model on the X-MoE path (universal atoms of run C671, registry
# DeepSeek_V3_671B, 8 nodes with TP 8 and the experts over all 64 GCDs; the eval refuses PP). Its own tokenizer.json
# (readable by the campaign's tokenizers) and its end token through the extra flags. The card's settings: HumanEval pass@1,
# GSM8K 8-shot (the card reports 65.2 and 89.3).
# ---------------------------------------------------------------------------------
_V3_FLAGS = "--vocab-file /lustre/orion/gen150/scratch/zixianw4/hf_cache/hub/models--deepseek-ai--DeepSeek-V3/snapshots/e815299b0bcbac849fa540c768ef21845365c9eb/tokenizer.json --eod-token '<｜end▁of▁sentence｜>'"
CUSTOM_TASKS.update({
    # --no-correctness-tests: the pre-eval probes took 29 minutes of the 671B probe job 5476348 (each probe a forward of
    # the whole model over 8 nodes) and flagged five of fifteen at magnitudes inside the 671B's own kernel-to-kernel floor
    # (logits max|diff| 0.66-1.6 where two HF kernels differ by 3.3 on a 10-token prompt, run H4); the benchmark is the test.
    # --extra-until '\x60\x60\x60' (the fence ``` as python escapes: a literal backtick on the launch line is a command
    # substitution for the shell, job 5476496 died at its syntax check): the 671B closes its completion with a markdown fence and a prose explanation (probe 5476348: four
    # of eight completions fenced, each a SyntaxError once appended to the test); lm-eval's humaneval stops only at
    # "\nclass" / "\ndef" / "\n#" / "\nif" / "\nprint", the bigcode harness stops at the fence too.
    'humaneval_v3': replace(TASKS['humaneval'], extra_eval_flags="--allow-code-eval --no-optimizer --no-correctness-tests --extra-until '\\x60\\x60\\x60' " + _V3_FLAGS),
    'gsm8k_8shot_v3': replace(TASKS['gsm8k'], extra_eval_flags='--no-optimizer --no-correctness-tests ' + _V3_FLAGS),
})
CUSTOM_SUITES = dict(globals().get('CUSTOM_SUITES', {}), **{'v3_671b': ['humaneval_v3', 'gsm8k_8shot_v3']})

# ---- 2026-09-13: the released DeepSeek-V3-BASE (deepseek-ai/DeepSeek-V3-Base, the checkpoint the card's base table was
# measured on), the same two rows: the apple-to-apple comparison against HumanEval 65.2 (0-shot pass@1) and GSM8K 89.3
# (8-shot EM). The rows above (humaneval_v3, gsm8k_8shot_v3) ran the post-trained chat model and compare to nothing in the card.
_V3_BASE_FLAGS = "--vocab-file /lustre/orion/gen150/scratch/zixianw4/hf_cache/hub/models--deepseek-ai--DeepSeek-V3-Base/snapshots/afb92e1fa402c2be2a9eb085312bb02e0384d6c7/tokenizer.json --eod-token '<｜end▁of▁sentence｜>'"
CUSTOM_TASKS.update({
    'humaneval_v3_base': replace(TASKS['humaneval'], extra_eval_flags="--allow-code-eval --no-optimizer --no-correctness-tests --extra-until '\\x60\\x60\\x60' " + _V3_BASE_FLAGS),
    'gsm8k_8shot_v3_base': replace(TASKS['gsm8k'], extra_eval_flags='--no-optimizer --no-correctness-tests ' + _V3_BASE_FLAGS),
})
CUSTOM_SUITES = dict(globals().get('CUSTOM_SUITES', {}), **{'v3_671b_base': ['humaneval_v3_base', 'gsm8k_8shot_v3_base']})

# ---- 2026-09-13 11:40: the GSM8K prompt-format probe. The base scored 92.7 with lm-eval's gsm8k (8 exemplars from the train
# split in the dataset's own "#### N" style with <<calculator>> annotations) against the card's 89.3, +3.4 with a 3-point
# threshold. The card's harness is internal (the V3 report 4.4.1: "our internal evaluation framework integrated in our HAI-LLM
# framework"; the DeepSeek LLM paper: 8-shot, greedy). The other common 8-shot format is the chain-of-thought exemplars of
# Wei et al. (2022): lm-eval's gsm8k_cot ("Q: ... A: ... The answer is N."). Same model, same 1,319 test problems, the exemplar
# format the only change: if the score moves to the card's, the gap is the prompt, not the framework.
CUSTOM_TASKS.update({
    'gsm8k_8shot_cot_v3_base': replace(TASKS['gsm8k'], task='gsm8k_cot', extra_eval_flags='--no-optimizer --no-correctness-tests ' + _V3_BASE_FLAGS),
    # the card's own format (DeepSeek LLM, arXiv 2401.02954, Appendix A.6, Table 29: "Q: ... A: Let's think step by step. ... The
    # answer is N." with its eight fixed exemplars), transcribed into custom_tasks/gsm8k_deepseek: the definitive apple-to-apple row.
    'gsm8k_8shot_dsfmt_v3_base': replace(TASKS['gsm8k'], task='gsm8k_deepseek', extra_eval_flags='--include-path /lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier/eval/custom_tasks --no-optimizer --no-correctness-tests ' + _V3_BASE_FLAGS),
})


# ---------------------------------------------------------------------------------
# 2026-09-14 (the eval session): the long-context suite for the 256k-1M evaluations of the 671B base, and the Moonlight
# chunked-prefill check of the latent cache. RULER at 256k, 512k and 1M is generated on the login node with the V3 tokenizer
# (assistant_deep_dive/2026-09-15-eval_framework/bin/ruler_synth.py: 20 documents per task instead of 500; the essay haystack
# caps five tasks near 700k tokens at 1M) into LM_EVAL_RULER_CACHE under the key the job computes; the rows take the first 16.
# OpenAI MRCR (custom_tasks/mrcr) is the released data rendered as a User:/Assistant: transcript for a base model, with the
# rows that fit the run's length (the metadata's max_seq_lengths) and the card's SequenceMatcher metric per token bin.
# ---------------------------------------------------------------------------------
_EF_CFG = '/lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier/eval/config'
_EF_INC = '--include-path /lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier/eval/custom_tasks'
_V3B_LC = '--no-optimizer --no-correctness-tests ' + _V3_BASE_FLAGS
CUSTOM_TASKS.update({
    # Moonlight, one RULER single-needle task at 8k (8 documents, generated with Moonlight's tokenizer): every prompt is
    # longer than the 4,096-token prefill chunk of submit_env.sh, so the prefill runs in two chunks against the cache.
    'niah1_8k_moon': replace(TASKS['ruler8k'], task='niah_single_1', mbs=1, limit=8, est_minutes=40,
                             seqlen_src='constructed:2026-09-14 8 docs, the Moonlight tokenizer',
                             extra_eval_flags=f'--task-metadata-json-file {_EF_CFG}/ruler8k_moon_metadata.json ' + _MOON_FLAGS),
    # 2026-09-17 (the user): the cross-comparison with vLLM: RULER at 32k and 64k on the 671B base, the first 100 documents per
    # sub-task, generated with the V3 tokenizer by bin/ruler_synth.py --n 100 (the same cache the vLLM runs read)
    'ruler32k_v3_base': replace(TASKS['ruler8k'], seqlen=32768, mbs=1, limit=100, est_minutes=240,
                                seqlen_src='constructed:2026-09-17 the V3 tokenizer, 100 docs per task on the login node',
                                extra_eval_flags=f'--task-metadata-json-file {_EF_CFG}/ruler32k_v3_metadata.json ' + _V3B_LC),
    'ruler64k_v3_base': replace(TASKS['ruler8k'], seqlen=65536, mbs=1, limit=100, est_minutes=360,
                                seqlen_src='constructed:2026-09-17 the V3 tokenizer, 100 docs per task on the login node',
                                extra_eval_flags=f'--task-metadata-json-file {_EF_CFG}/ruler64k_v3_metadata.json ' + _V3B_LC),
    'ruler256k_v3_base': replace(TASKS['ruler8k'], seqlen=262144, mbs=1, limit=16, est_minutes=600,
                                 seqlen_src='constructed:2026-09-14 the V3 tokenizer, 20 docs per task on the login node',
                                 extra_eval_flags=f'--task-metadata-json-file {_EF_CFG}/ruler256k_v3_metadata.json ' + _V3B_LC),
    'ruler512k_v3_base': replace(TASKS['ruler8k'], seqlen=524288, mbs=1, limit=16, est_minutes=1200,
                                 seqlen_src='constructed:2026-09-14 the V3 tokenizer, 20 docs per task on the login node',
                                 extra_eval_flags=f'--task-metadata-json-file {_EF_CFG}/ruler512k_v3_metadata.json ' + _V3B_LC),
    'ruler1m_v3_base': replace(TASKS['ruler8k'], seqlen=1048576, mbs=1, limit=16, est_minutes=2400,
                               seqlen_src='constructed:2026-09-14 the V3 tokenizer, 20 docs per task; the essay tasks cap near 700k',
                               extra_eval_flags=f'--task-metadata-json-file {_EF_CFG}/ruler1m_v3_metadata.json ' + _V3B_LC),
    'mrcr8_128k_v3_base': replace(TASKS['ruler8k'], task='mrcr_8needle', seqlen=131072, mbs=1, gen_toks=1200, limit=0, est_minutes=600,
                                  needs=('harness:task-metadata-json',), seqlen_src='constructed:2026-09-14 the rows whose prompt + 1,200 fit',
                                  extra_eval_flags=f'{_EF_INC} --task-metadata-json-file {_EF_CFG}/ruler128k_v3_metadata.json ' + _V3B_LC),
    'mrcr8_256k_v3_base': replace(TASKS['ruler8k'], task='mrcr_8needle', seqlen=262144, mbs=1, gen_toks=1200, limit=0, est_minutes=1200,
                                  needs=('harness:task-metadata-json',), seqlen_src='constructed:2026-09-14 the rows whose prompt + 1,200 fit',
                                  extra_eval_flags=f'{_EF_INC} --task-metadata-json-file {_EF_CFG}/ruler256k_v3_metadata.json ' + _V3B_LC),
})
CUSTOM_SUITES = dict(globals().get('CUSTOM_SUITES', {}), **{'lc_v3_base': ['ruler256k_v3_base', 'mrcr8_128k_v3_base']})

# LongBench v2 (Bai et al., 2024-12, arXiv 2412.15204) as lm-eval's longbench2 group (503 multiple-choice questions, the paper's
# prompt, the loglikelihood of A/B/C/D); the contexts (8k to 2M words) are left-truncated to the row's length by the adapter, which
# keeps the question at the end; rows longer than XMOE_SCORE_KV_FROM go through the chunked prefill (lm_eval_adapter._score_long).
CUSTOM_TASKS.update({
    'longbench2_128k_v3_base': replace(TASKS['mmlu_0shot'], task='longbench2', seqlen=131072, mbs=1, est_minutes=600, needs=(),
                                       seqlen_src='constructed:2026-09-14 the row length; the documents are longer',
                                       extra_eval_flags=_V3B_LC),
    'longbench2_256k_v3_base': replace(TASKS['mmlu_0shot'], task='longbench2', seqlen=262144, mbs=1, est_minutes=1200, needs=(),
                                       seqlen_src='constructed:2026-09-14 the row length; the documents are longer',
                                       extra_eval_flags=_V3B_LC),
})

# the Moonlight check of the long-context scoring path (2026-09-14): LongBench v2 at 8k, the chunked-prefill scoring forced onto the
# documents longer than XMOE_SCORE_KV_FROM=2048 in one arm, the one-piece batched forward in the other; the same documents, the same
# answers expected up to bf16 noise.
CUSTOM_TASKS.update({
    'longbench2_8k_moon': replace(TASKS['mmlu_0shot'], task='longbench2', seqlen=8192, mbs=1, est_minutes=40, needs=(),
                                  seqlen_src='constructed:2026-09-14 the row length; the documents are longer',
                                  extra_eval_flags=_MOON_FLAGS),
})

# 2026-09-16: the rest of the long-context suite rows on the 671B base: the needle grid (custom_tasks/needlegrid, 50 docs per length, ten
# depths, 16 generated tokens) and the 2- and 4-needle MRCR sets. Every row takes the run's length from the RULER metadata file of that
# length (max_seq_lengths), the tokenizer and end token from _V3_BASE_FLAGS, and runs one row per forward (mbs 1) through the chunked prefill.
def _lc_row(task, seqlen, meta, gen_toks, est):
    return replace(TASKS['ruler8k'], task=task, seqlen=seqlen, mbs=1, gen_toks=gen_toks, limit=0, est_minutes=est, needs=('harness:task-metadata-json',),
                   seqlen_src='constructed:2026-09-16 the row length; the data generated for it',
                   extra_eval_flags=f'{_EF_INC} --task-metadata-json-file {_EF_CFG}/{meta} ' + _V3B_LC)
CUSTOM_TASKS.update({
    'needlegrid_128k_v3_base': _lc_row('needlegrid', 131072, 'ruler128k_v3_metadata.json', 16, 120),
    'needlegrid_256k_v3_base': _lc_row('needlegrid', 262144, 'ruler256k_v3_metadata.json', 16, 240),
    'needlegrid_512k_v3_base': _lc_row('needlegrid', 524288, 'ruler512k_v3_metadata.json', 16, 480),
    'needlegrid_1m_v3_base': _lc_row('needlegrid', 1048576, 'ruler1m_v3_metadata.json', 16, 960),
    'mrcr2_128k_v3_base': _lc_row('mrcr_2needle', 131072, 'ruler128k_v3_metadata.json', 1200, 600),
    'mrcr4_128k_v3_base': _lc_row('mrcr_4needle', 131072, 'ruler128k_v3_metadata.json', 1200, 600),
    'mrcr2_256k_v3_base': _lc_row('mrcr_2needle', 262144, 'ruler256k_v3_metadata.json', 1200, 1200),
    'mrcr4_256k_v3_base': _lc_row('mrcr_4needle', 262144, 'ruler256k_v3_metadata.json', 1200, 1200),
})
CUSTOM_SUITES = dict(globals().get('CUSTOM_SUITES', {}), **{'lc256k_v3_base': ['ruler256k_v3_base', 'needlegrid_256k_v3_base', 'mrcr8_256k_v3_base', 'longbench2_256k_v3_base']})

# 2026-10-06 (user): the 671B 128k long-context extension chain scores every 4th step on RULER-128k niah_single 1-3 only, 100 docs per
# sub-task (300 docs per checkpoint): three leaf rows cloned from ruler128k_f (the 128k metadata flag, no samples manifest, --limit
# takes the FIRST 100 docs of the leaf, the same docs for every checkpoint), one suite naming the three. est_minutes from the 128k
# probe tier's 2.1 min per doc per worker (the 16B on one node); the 671B on 8 nodes is measured by the first run.
# 2026-10-07 18:30 correction: the rows now derive from the 671B base-model long-context rows (_lc_row: the V3 tokenizer's 128k
# metadata, --no-optimizer, --no-correctness-tests, the V3 vocab flags). The ruler128k_f lineage (the 16B probe tier) built the full
# DeepSpeed optimizer at deepspeed.initialize and every 8-node job died of HIP out of memory in BF16_Optimizer (jobs 5631439-41,
# 5631466-68, 5633119-24, 5633304-06, 5633453-55, 7 min each); the September 671B RULER runs carried --no-optimizer through _V3B_LC.
for _k in (1, 2, 3):
    CUSTOM_TASKS[f'niah{_k}_128k'] = replace(_lc_row(f'niah_single_{_k}', 131072, 'ruler128k_v3_metadata.json', 128, 210), limit=100,
                                             seqlen_src='constructed:2026-10-07 niah_single leaf at 128k, the first 100 docs of the V3-tokenizer RULER cache')
CUSTOM_SUITES['ruler128k_niah3'] = ['niah1_128k', 'niah2_128k', 'niah3_128k']

# 2026-10-08 (the user): the 671B 256k extension (YaRN scale 64) scores every 2-step checkpoint on four RULER-256k sub-tasks with
# 50 documents each (the factor-64 zero-shot base of 2026-09-21: niah_single_3 0.800, niah_multiquery 0.887, niah_multivalue 0.925 on
# 20 documents; multikey_3 unmeasured), 16-node jobs (EP 128 x TP 8). Rows from _lc_row (the V3 256k metadata, --no-optimizer, the V3
# vocab flags); est_minutes from ~30 min per document per attention group at 256k (16 groups, 50 docs -> ~3.2 rounds) + a 15-min load.
for _k, _t in (('niah3_256k', 'niah_single_3'), ('mk3_256k', 'niah_multikey_3'), ('mq_256k', 'niah_multiquery'), ('mv_256k', 'niah_multivalue')):
    CUSTOM_TASKS[_k] = replace(_lc_row(_t, 262144, 'ruler256k_v3_metadata.json', 128, 115), limit=50,
                               seqlen_src='constructed:2026-10-08 the V3-tokenizer RULER cache at 262,144, the first 50 docs')
CUSTOM_SUITES['ruler256k_x4'] = ['niah3_256k', 'mk3_256k', 'mq_256k', 'mv_256k']
