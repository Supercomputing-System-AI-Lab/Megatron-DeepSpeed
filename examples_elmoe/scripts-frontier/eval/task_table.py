#!/usr/bin/env python3
"""task_table.py — single source of truth for eval task metadata and suites.

Every number the old EVAL_MAP entries hardcoded per line (seqlen, shots, micro-batch,
walltime) lives HERE, once per task, with provenance. The launcher (eval_lib.py), the
seqlen checker (check_seqlens.py) and the dataset prefetcher all read this table; nothing
else defines task metadata.

User-tunable surface: eval/config/custom_suites.py (see load_custom()) may add or override
rows and suites without touching this file — "you edit eval/config/, everything else is
machinery".

CLI (pure stdlib, login-node python3):
    python3 task_table.py suites
    python3 task_table.py expand --suite core [--tsv]
"""
import os
import sys
from dataclasses import dataclass, replace

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(EVAL_DIR, 'config')


@dataclass(frozen=True)
class Row:
    task: str            # lm-eval task name (groups like `mmlu` allowed)
    category: str        # commonsense | science | knowledge | ppl | instruct | longctx
    output_type: str     # loglik | rolling | generate
    seqlen: int          # padded seq length for the job (cost multiplier; see seqlen_src)
    fewshot: int = 0
    mbs: int = 8
    est_minutes: int = 60    # SCORING time estimate at this mbs on 1 node; planning hint
                             # only, never a contract — the cache makes overruns resumable
    seqlen_src: str = 'assumed'   # 'measured:<date> max=<N>' | 'prod' | 'assumed'
    seqlen_locked: bool = False   # True: the seqlen IS the eval semantics (wikitext, ruler)
    gen_toks: int = 0             # generate rows: max_gen_toks (0 for scoring rows)
    padding: str = 'right'
    needs: tuple = ()             # preflight tags: pip:<mod> dataset:<dir> harness:<flag>
    extra_eval_flags: str = ''    # appended verbatim to eval_options (e.g. ruler metadata)
    limit: int = 0                # 0 = full run. N = FIXED first-N-docs subset (per
                                  # SUBTASK for group tasks -- stratified by construction).
                                  # Same docs every checkpoint => paired deltas; registry
                                  # rows carry the limit, DONE checks are limit-scoped,
                                  # and the cache (limit-agnostic keys) pre-warms the
                                  # eventual full run. See 0814_2026_INSTRUCT_SMALL_DESIGN.md.


# RULER metadata travels as a FILE, never inline JSON: extra_eval_flags crosses two shell
# parse layers (the template's double-quoted assignment, then the srun `bash -c` re-parse
# with brace expansion), which the review fleet showed destroys inline JSON. The files live
# in config/ (user-tunable) and main.py reads them via --task-metadata-json-file.
_RULER_META = '--task-metadata-json-file ' + os.path.join(CONFIG_DIR, 'ruler%s_metadata.json')

# ----------------------------------------------------------------------------------------
# Curated rows. seqlen_src='prod' = ran full at this value in production with no truncation
# (values from the registry's history); 'measured:' = check_seqlens.py output; 'assumed'
# rows are REFUSED for full runs by submit_eval's preflight until measured.
# est_minutes calibration anchors (2026-08-13/14, MI250X 1 node, mbs 8):
#   sciq  4000 req @2048: 28.7 min (2.32 it/s)      piqa 3676 req @2048: 28 min (2.16 it/s)
#   mmlu 5-shot @4096: ~11 h total                  hellaswag @512: ~75 min
# ----------------------------------------------------------------------------------------
TASKS = {
    # ---- perplexity ----
    'wikitext':       Row('wikitext', 'ppl', 'rolling', 4096, est_minutes=20,
                          seqlen_locked=True, seqlen_src='prod'),
    # ---- commonsense ----
    'hellaswag':      Row('hellaswag', 'commonsense', 'loglik', 512, est_minutes=75,
                          seqlen_src='prod'),
    'piqa':           Row('piqa', 'commonsense', 'loglik', 2048, est_minutes=30,
                          seqlen_src='prod'),
    'lambada':        Row('lambada_openai', 'commonsense', 'loglik', 2048, est_minutes=45,
                          seqlen_src='prod'),
    'winogrande':     Row('winogrande', 'commonsense', 'loglik', 2048, est_minutes=20,
                          seqlen_src='prod'),
    'openbookqa':     Row('openbookqa', 'commonsense', 'loglik', 2048, est_minutes=15,
                          seqlen_src='prod'),
    'boolq':          Row('boolq', 'commonsense', 'loglik', 2048, est_minutes=45,
                          seqlen_src='measured:2026-08-14 max=1295 p99=394',
                          needs=('dataset:super_glue',)),
    # ---- science ----
    'sciq':           Row('sciq', 'science', 'loglik', 2048, est_minutes=30,
                          seqlen_src='prod'),
    'arc_easy':       Row('arc_easy', 'science', 'loglik', 2048, est_minutes=90,
                          seqlen_src='prod'),
    'arc_challenge':  Row('arc_challenge', 'science', 'loglik', 2048, est_minutes=45,
                          seqlen_src='prod'),
    # ---- knowledge ----
    'truthfulqa_mc1': Row('truthfulqa_mc1', 'knowledge', 'loglik', 2048, est_minutes=20,
                          seqlen_src='prod'),
    'truthfulqa_mc2': Row('truthfulqa_mc2', 'knowledge', 'loglik', 2048, est_minutes=35,
                          seqlen_src='prod'),
    'mmlu_0shot':     Row('mmlu', 'knowledge', 'loglik', 1536, est_minutes=300,
                          seqlen_src='prod'),
    'mmlu_5shot':     Row('mmlu', 'knowledge', 'loglik', 4096, fewshot=5, est_minutes=660,
                          seqlen_src='prod'),
    'triviaqa':       Row('triviaqa', 'knowledge', 'generate', 1024, fewshot=5,
                          est_minutes=120, gen_toks=32, seqlen_src='prod'),
    # ---- instruction-following / reasoning (generation; expensive until Phase 5 KV) ----
    # 8-shot prompts measure 2765 tokens max -- 2048 would TRUNCATE (caught by
    # check_seqlens on 2026-08-14). est doubled with the seqlen (per-token forward is [B,S]).
    'gsm8k':          Row('gsm8k', 'instruct', 'generate', 4096, fewshot=8,
                          est_minutes=180, gen_toks=256,
                          seqlen_src='measured:2026-08-14 max=2765 p99=2318',
                          needs=('dataset:gsm8k',)),
    'ifeval':         Row('ifeval', 'instruct', 'generate', 4096, est_minutes=300,
                          gen_toks=1280,
                          seqlen_src='measured:2026-08-14 max=1636 p99=1386',
                          needs=('pip:langdetect', 'pip:immutabledict', 'dataset:if_eval')),
    # 3-shot prompts measure 3576 max, p99 3485 -- at the assumed 2048 nearly every prompt
    # would have been truncated (caught by check_seqlens 2026-08-14). est doubled with seqlen.
    'bbh':            Row('bbh', 'instruct', 'generate', 4096, fewshot=3,
                          est_minutes=4800, gen_toks=256,
                          seqlen_src='measured:2026-08-14 max=3576 p99=3485',
                          needs=('dataset:bbh',)),
    # ---- DeepSeekMoE-chat-protocol rows (paper Table 7; survey 2026-08-14). The chat
    # protocol drops several tasks to 0-shot ("SFT should internalize the task"); target
    # scores for our SFT ckpts are in 0814_2026_INSTRUCT_EVAL_TARGETS.md. All 'assumed'
    # rows are preflight-gated until measured with check_seqlens.py. ----
    'gsm8k_chat':     Row('gsm8k', 'instruct', 'generate', 1024, fewshot=0,
                          est_minutes=400, gen_toks=256,
                          seqlen_src='measured:2026-08-14 max=397 p99=387',
                          needs=('dataset:gsm8k',)),
    # 4-shot prompts measure 2193 max (p99 1607) -- the assumed 2048 would truncate the
    # longest ~1% (third truncation catch by check_seqlens). 2560 covers max with margin
    # at 1.25x the 2048 cost instead of 2x for a 4096 bucket.
    'math':           Row('minerva_math', 'instruct', 'generate', 2560, fewshot=4,
                          est_minutes=2400, gen_toks=256,
                          seqlen_src='measured:2026-08-14 max=2193 p99=1607',
                          needs=('dataset:hendrycks', 'pip:sympy', 'pip:math_verify')),
    'drop':           Row('drop', 'knowledge', 'generate', 2048, fewshot=1,
                          est_minutes=1200, gen_toks=256,
                          seqlen_src='measured:2026-08-14 max=1628 p99=1313',
                          needs=('dataset:drop',)),
    'nq_open':        Row('nq_open', 'knowledge', 'generate', 512, fewshot=0,
                          est_minutes=250, gen_toks=256,
                          seqlen_src='measured:2026-08-14 max=64 p99=61',
                          needs=('dataset:nq_open',)),
    'race':           Row('race', 'knowledge', 'loglik', 4096, fewshot=5,
                          est_minutes=120,
                          seqlen_src='measured:2026-08-14 max=3875 p99=3671',
                          needs=('dataset:race',)),
    # Code rows (UNGATED 2026-08-14, user decision): pass@1 EXECUTES the generated
    # Python via lm-eval's standard harness -- --allow-code-eval (per-row, below) sets
    # HF_ALLOW_CODE_EVAL=1 + confirm_run_unsafe_code=True. Runs as the job user on
    # compute nodes with no outbound network. Code is 17.2% of the SFT mix -- the
    # largest trained skill; DeepSeekMoE-chat targets: humaneval 45.7, mbpp 46.2.
    'humaneval':      Row('humaneval', 'instruct', 'generate', 2048, est_minutes=120,
                          gen_toks=1024, extra_eval_flags='--allow-code-eval',
                          seqlen_src='measured:2026-08-14 max=937 p99=858',
                          needs=('dataset:openai_humaneval', 'harness:allow-code-eval')),
    # mbpp 3-shot has outlier prompts (long test lists) up to 4867 tok -- beyond the
    # model's 4096 context entirely; locked at 4096, the <1% tail truncates by design.
    'mbpp':           Row('mbpp', 'instruct', 'generate', 4096, fewshot=3,
                          est_minutes=240, gen_toks=1024, seqlen_locked=True,
                          seqlen_src='measured:2026-08-14 max=4867 p99=1547',
                          extra_eval_flags='--allow-code-eval',
                          needs=('dataset:mbpp', 'harness:allow-code-eval')),
    # Chat-quality transport: the model answers AlpacaEval 2's 805 official
    # instructions in the TRAINED wrapper (User:/Assistant:, sft_chat_format.py); the
    # LC win-rate judging is inherently an API-judge step, done offline later from
    # export_alpacaeval.py's file. AlpacaEval 2 is the most-adopted chat eval
    # (OLMo 2/3, Tulu 3, DeepSeek-V2, V3, R1).
    'alpaca_gen':     Row('alpaca_eval_gen', 'instruct', 'generate', 2048,
                          est_minutes=700, gen_toks=768,
                          seqlen_src='measured:2026-08-14 max=1284 p99=1037',
                          extra_eval_flags='--include-path ' +
                                           os.path.join(EVAL_DIR, 'custom_tasks'),
                          needs=('dataset:alpaca_eval',)),
    # ---- instruct-short: fixed-subset fast-cadence variants (design doc:
    # 0814_2026_INSTRUCT_SMALL_DESIGN.md). Same FIRST-N docs every checkpoint =>
    # paired deltas with no doc-sampling noise; group tasks (bbh 27 subtasks,
    # minerva_math 7 subjects) stratify because lm-eval applies limit PER SUBTASK.
    # est_minutes sized for ~1 h jobs at the clean KV rate; recalibrate from the
    # limit-8 gate measurements. Never report these as final numbers -- full rows at
    # milestone checkpoints are the reported ones (the shared cache pre-warms them).
    'gsm8k_chat_s':  Row('gsm8k', 'short', 'generate', 1024, fewshot=0,
                          gen_toks=256, est_minutes=45, limit=100,
                          seqlen_src='measured:2026-08-14 max=397 p99=387',
                          needs=('dataset:gsm8k',),
                          extra_eval_flags='--task-samples-json-file ' + os.path.join(CONFIG_DIR, 'subsets', 'short_q512_fs0.json')),
    'hellaswag_s':   Row('hellaswag', 'short', 'loglik', 512, est_minutes=2,
                          limit=100, seqlen_src='prod',
                          extra_eval_flags='--task-samples-json-file ' + os.path.join(CONFIG_DIR, 'subsets', 'short_q512_fs0.json')),
    'math_s':        Row('minerva_math', 'short', 'generate', 2560, fewshot=4,
                          gen_toks=256, est_minutes=45, limit=49,   # 7 x 7 subjects = 49
                          seqlen_src='measured:2026-08-14 max=2193 p99=1607',
                          needs=('dataset:hendrycks', 'pip:sympy', 'pip:math_verify'),
                          extra_eval_flags='--task-samples-json-file ' + os.path.join(CONFIG_DIR, 'subsets', 'math_s.json')),
    'ifeval_s':      Row('ifeval', 'short', 'generate', 4096, gen_toks=1280,
                          est_minutes=300, limit=40,
                          seqlen_src='measured:2026-08-14 max=1636 p99=1386',
                          needs=('pip:langdetect', 'pip:immutabledict', 'dataset:if_eval'),
                          extra_eval_flags='--task-samples-json-file ' + os.path.join(CONFIG_DIR, 'subsets', 'ifeval_s.json')),
    'bbh_s':         Row('bbh', 'short', 'generate', 4096, fewshot=3,
                          # measured 2026-08-15: 130/216 docs in 74 min -> ~125 min for the row
                          gen_toks=256, est_minutes=90, limit=216,  # 8 x 27 subtasks = 216
                          seqlen_src='measured:2026-08-14 max=3576 p99=3485',
                          needs=('dataset:bbh',),
                          extra_eval_flags='--task-samples-json-file ' + os.path.join(CONFIG_DIR, 'subsets', 'bbh_s.json')),
    'humaneval_s':   Row('humaneval', 'short', 'generate', 2048, gen_toks=1024,
                          est_minutes=120, limit=80,
                          extra_eval_flags='--allow-code-eval '
                                           + '--task-samples-json-file ' + os.path.join(CONFIG_DIR, 'subsets', 'humaneval_s.json'),
                          seqlen_src='measured:2026-08-14 max=937 p99=858',
                          needs=('dataset:openai_humaneval', 'harness:allow-code-eval')),
    'humaneval_chat_s': Row('humaneval_chat', 'short', 'generate', 2048, gen_toks=1024,
                          est_minutes=35, limit=80,
                          seqlen_src='measured:2026-08-15 derived humaneval(max937)+wrapper<1024',
                          extra_eval_flags='--include-path ' + os.path.join(EVAL_DIR, 'custom_tasks')
                                           + ' --task-samples-json-file '
                                           + os.path.join(CONFIG_DIR, 'subsets', 'humaneval_chat_s.json'),
                          needs=('dataset:openai_humaneval',)),
    # MBPP through the trained chat wrapper (custom_tasks/mbpp_chat), cloned from
    # humaneval_chat_s: same grader design (subprocess-exec of test_list, NOT HF
    # code_eval, so no allow-code-eval flag needed). 0-shot chat prompt = text +
    # test_list[0] hint; dataset config 'full' (sanitized not cached). NOT in any
    # suite yet (user decision pending). p99=698 but one outlier doc (huge test
    # string) measures 2287 with gen -- 2560 covers max with margin at 1.25x the
    # 2048 cost (same reasoning as the math row).
    'mbpp_chat_s':   Row('mbpp_chat', 'short', 'generate', 2560, gen_toks=1024,
                          est_minutes=60, limit=80,
                          seqlen_src='measured:2026-08-15 max=2287 p99=698',
                          extra_eval_flags='--include-path ' + os.path.join(EVAL_DIR, 'custom_tasks')
                                           + ' --task-samples-json-file '
                                           + os.path.join(CONFIG_DIR, 'subsets', 'mbpp_chat_s.json'),
                          needs=('dataset:mbpp',)),
    'triviaqa_s':    Row('triviaqa', 'short', 'generate', 1024, fewshot=5,
                          # measured 2026-08-15: 309/500 docs in 67 min -> ~110 min for the row
                          gen_toks=32, est_minutes=120, limit=500, seqlen_src='prod',
                          extra_eval_flags='--task-samples-json-file ' + os.path.join(CONFIG_DIR, 'subsets', 'triviaqa_s.json')),
    # ---- long context (RULER; per-checkpoint-type — 8k/16k only for context-extended ckpts) ----
    # RULER sizes every prompt to max_seq_lengths with the target tokenizer, so the
    # seqlen is correct BY CONSTRUCTION; full 13-task group build verified 2026-08-14
    # (hotpot from the HF-converted disk cache, squad disk-cached, essays not needed at 4k).
    'ruler4k':        Row('ruler', 'longctx', 'generate', 4096, est_minutes=600,
                          gen_toks=128, seqlen_locked=True, mbs=4,
                          seqlen_src='constructed:2026-08-14 group-build-verified',
                          needs=('pip:html2text', 'pip:wonderwords', 'harness:task-metadata-json'),
                          extra_eval_flags=_RULER_META % '4k'),
    # LongBench English, length-bucketed (_e): 13 tasks x {0-4k, 4-8k, 8k+} buckets.
    # At the 4k base model only the 0-4k bucket is fully in-context (our adapter
    # tail-truncates longer docs); the full curve unlocks with the context-extended ckpt.
    # Docs measure up to 24k tokens (p99 12.8k) -- truncation at the model context is
    # LongBench's intended regime for short-context models; score the _e 0-4k bucket at
    # the base model. A longbench8k row becomes meaningful with the context-extended ckpt.
    'longbench':      Row('longbench_e', 'longctx', 'generate', 4096, est_minutes=900,
                          gen_toks=512, seqlen_locked=True, mbs=4,
                          seqlen_src='measured:2026-08-14 max=24038 truncation-by-design',
                          needs=('dataset:long_bench', 'pip:jieba', 'pip:fuzzywuzzy', 'pip:rouge')),
    'ruler8k':        Row('ruler', 'longctx', 'generate', 8192, est_minutes=900,
                          gen_toks=128, seqlen_locked=True, mbs=2,
                          seqlen_src='measured:2026-08-15 max=8311 p99=8230 over-length-truncates-by-design',
                          needs=('pip:html2text', 'pip:wonderwords', 'harness:task-metadata-json'),
                          extra_eval_flags=_RULER_META % '8k'),
    # Same 13-task group at 16k for the context-extended ckpt; mbs halves again with the
    # seqlen. est is ~2x the 8k row (per-doc prefill roughly doubles; the 4k->8k ests
    # grew only 1.5x, but at 16k the quadratic attention term dominates) -- planning
    # hint only, chains absorb the error.
    'ruler16k':       Row('ruler', 'longctx', 'generate', 16384, est_minutes=1800,
                          gen_toks=128, seqlen_locked=True, mbs=1,
                          seqlen_src='measured:2026-08-15 max=16477 p99=16413 over-length-truncates-by-design',
                          needs=('pip:html2text', 'pip:wonderwords', 'harness:task-metadata-json'),
                          extra_eval_flags=_RULER_META % '16k'),
    # ---- longctx-short: fixed-subset fast-cadence variants of the RULER/LongBench
    # rows, same design as instruct-short (seed 20260814, committed manifests from
    # make_subsets.py, --verify drift check). RULER builds DIFFERENT docs per length,
    # so each length carries its OWN manifest (identical index draws, per-length
    # doc_sha); the metadata file still rides along -- doc construction needs it.
    # 100 x 13 ruler subtasks = 1300/length; 50 x 9 longbench_e subtasks = 450.
    # est_minutes = full-row rate scaled to the subset (~1300/6500 resp. 450/~4.7k docs).
    'ruler4k_s':      Row('ruler', 'longctx', 'generate', 4096, est_minutes=120,
                          gen_toks=128, seqlen_locked=True, mbs=4, limit=1300,
                          seqlen_src='constructed:2026-08-14 group-build-verified',
                          needs=('pip:html2text', 'pip:wonderwords', 'harness:task-metadata-json'),
                          extra_eval_flags=_RULER_META % '4k'
                                           + ' --task-samples-json-file '
                                           + os.path.join(CONFIG_DIR, 'subsets', 'ruler4k_s.json')),
    'ruler8k_s':      Row('ruler', 'longctx', 'generate', 8192, est_minutes=180,
                          gen_toks=128, seqlen_locked=True, mbs=2, limit=1300,
                          seqlen_src='measured:2026-08-15 max=8311 p99=8230 over-length-truncates-by-design',
                          needs=('pip:html2text', 'pip:wonderwords', 'harness:task-metadata-json'),
                          extra_eval_flags=_RULER_META % '8k'
                                           + ' --task-samples-json-file '
                                           + os.path.join(CONFIG_DIR, 'subsets', 'ruler8k_s.json')),
    'ruler16k_s':     Row('ruler', 'longctx', 'generate', 16384, est_minutes=360,
                          gen_toks=128, seqlen_locked=True, mbs=1, limit=1300,
                          seqlen_src='measured:2026-08-15 max=16477 p99=16413 over-length-truncates-by-design',
                          needs=('pip:html2text', 'pip:wonderwords', 'harness:task-metadata-json'),
                          extra_eval_flags=_RULER_META % '16k'
                                           + ' --task-samples-json-file '
                                           + os.path.join(CONFIG_DIR, 'subsets', 'ruler16k_s.json')),
    'longbench_s':    Row('longbench_e', 'longctx', 'generate', 4096, est_minutes=150,
                          gen_toks=512, seqlen_locked=True, mbs=4, limit=450,
                          seqlen_src='measured:2026-08-14 max=24038 truncation-by-design',
                          needs=('dataset:long_bench', 'pip:jieba', 'pip:fuzzywuzzy', 'pip:rouge'),
                          extra_eval_flags='--task-samples-json-file ' + os.path.join(CONFIG_DIR, 'subsets', 'longbench_s.json')),
}

# Derived category -> [row ids] (declaration order preserved)
CATEGORIES = {}
for _id, _row in TASKS.items():
    CATEGORIES.setdefault(_row.category, []).append(_id)

# Suites are lists whose elements are row ids OR category names. Per-checkpoint-type
# profiles (user decision): base/mid-training runs get `core`; the instruction-tuned
# checkpoint gets `instruct`; the context-extended checkpoint gets `longctx`.
SUITES = {
    # second execution-graded coding bench, run standalone so instruct-short's
    # committed 6-job shape stays unchanged
    'mbpp-chat': ['mbpp_chat_s'],
    'core':      ['commonsense', 'science', 'truthfulqa_mc1', 'truthfulqa_mc2',
                  'wikitext', 'mmlu_0shot'],
    'knowledge': ['mmlu_5shot', 'triviaqa'],
    # instruct: one benchmark per trained/standard skill area (landscape survey,
    # 0814_2026_INSTRUCT_BENCHMARK_LANDSCAPE.md): IF=ifeval, math=gsm8k_chat+math,
    # reasoning=bbh, code=humaneval+mbpp (ungated 2026-08-14), knowledge
    # recall=triviaqa, plus the commonsense regression guard. MMLU deliberately lives
    # in its own suite (below): it is loglik scoring the KV work cannot speed up
    # (~5 h at 0-shot) and would dominate every instruct pass. Chat quality
    # (alpaca_gen, ~805 long generations) likewise runs as its own suite at chosen
    # checkpoints, not on every pass.
    'instruct':  ['gsm8k_chat', 'math', 'ifeval', 'bbh', 'humaneval', 'mbpp',
                  'triviaqa', 'commonsense'],
    # standalone heavies -- schedule deliberately, not per-checkpoint by default
    'mmlu':      ['mmlu_0shot'],
    'chat':      ['alpaca_gen'],
    # instruct-short: the per-checkpoint CADENCE suite -- six ~1 h jobs, all parallel
    # (no chains). One fixed subset per skill area + a collapse canary; ~1,185 docs
    # total. Full `instruct` at milestone checkpoints remains the reported number;
    # these track movement between them. Design + statistical defense:
    # 0814_2026_INSTRUCT_SMALL_DESIGN.md.
    'instruct-short': ['gsm8k_chat_s', 'hellaswag_s', 'math_s', 'ifeval_s', 'bbh_s',
                       'humaneval_chat_s', 'triviaqa_s'],
    # NOTE on humaneval_s (still out of the cadence suite, but the stated reason was
    # WRONG): the "raw-completion prompt collapses on chat-formatted models" diagnosis
    # was a misreading of a framework bug. The adapter computed
    # maxctx = seq_length - max_gen_toks = 1024 - 1024 = 0 and clamped it to 1, so every
    # HumanEval prompt was cut to its final token; the untrained BASE model collapsed
    # identically, which no prompt-format story can explain. Fixed 2026-08-16 (adapter
    # raises instead of clamping; the table's gen_toks is pinned via --max-gen-toks).
    # Re-run raw humaneval and, if it lands near the paper's 26.8% base score, this row
    # can rejoin the suite as a second execution-graded coding measure.
    # paper_chat: the feasible replica of DeepSeekMoE's chat Table 7 (omits: HumanEval/
    # MBPP until the code-exec sandbox decision; CLUEWSC/CHID no lm-eval task; ceval/
    # cmmlu available ad-hoc). Targets in 0814_2026_INSTRUCT_EVAL_TARGETS.md.
    'paper_chat': ['hellaswag', 'piqa', 'arc_easy', 'arc_challenge', 'bbh', 'race',
                   'drop', 'gsm8k_chat', 'math', 'triviaqa', 'nq_open', 'mmlu_0shot',
                   'winogrande'],
    'longctx':   ['ruler4k', 'ruler8k', 'ruler16k', 'longbench', 'commonsense'],
    # longctx-short: fixed-subset cadence pass over the long-context rows (manifests
    # in config/subsets/; full `longctx` at milestone checkpoints stays the reported
    # number). No commonsense guard here -- instruct-short already carries one.
    'longctx-short': ['ruler4k_s', 'ruler8k_s', 'ruler16k_s', 'longbench_s'],
    'milestone': ['knowledge', 'instruct', 'longctx'],
    'smoke':     ['hellaswag', 'piqa', 'wikitext'],
}


def load_custom():
    """Merge eval/config/custom_suites.py (user-owned) over the curated defaults.

    The file may define:
        CUSTOM_TASKS  = {'my_row': Row(...)} or {'sciq': {'mbs': 4}}   (Row or override dict)
        CUSTOM_SUITES = {'my_suite': ['sciq', 'commonsense', ...]}
    Returns (tasks, suites) — merged copies; the module-level dicts stay pristine.
    """
    tasks, suites = dict(TASKS), {k: list(v) for k, v in SUITES.items()}
    path = os.path.join(CONFIG_DIR, 'custom_suites.py')
    if not os.path.isfile(path):
        return tasks, suites
    ns = {'Row': Row, 'replace': replace, 'TASKS': tasks, 'SUITES': suites}
    with open(path) as f:
        exec(compile(f.read(), path, 'exec'), ns)
    for rid, spec in (ns.get('CUSTOM_TASKS') or {}).items():
        if isinstance(spec, Row):
            tasks[rid] = spec
        elif isinstance(spec, dict):
            if rid not in tasks:
                raise KeyError(f'custom override for unknown row id {rid!r}')
            tasks[rid] = replace(tasks[rid], **spec)
        else:
            raise TypeError(f'CUSTOM_TASKS[{rid!r}] must be a Row or an override dict')
    for name, ids in (ns.get('CUSTOM_SUITES') or {}).items():
        suites[name] = list(ids)
    return tasks, suites


def expand(suite_expr, tasks=None, suites=None):
    """'core' or 'core+longctx' -> ordered, de-duplicated [(row_id, Row)].

    Suite elements may be row ids, category names, or other suite names (one level of
    nesting is resolved recursively with cycle protection).
    """
    if tasks is None or suites is None:
        tasks, suites = load_custom()
    cats = {}
    for rid, row in tasks.items():
        cats.setdefault(row.category, []).append(rid)

    out, seen = [], set()

    def _add(rid):
        if rid not in seen:
            seen.add(rid)
            out.append((rid, tasks[rid]))

    def _walk(name, stack):
        # Resolution precedence: suite > row id > category. A suite may share a name with
        # a category (knowledge/instruct/longctx do); the SUITE is the user-facing meaning.
        # `stack` breaks cycles, which also lets a suite reference the same-named category
        # implicitly: once the suite is on the stack, the name re-resolves to the category.
        if name in suites and name not in stack:
            for element in suites[name]:
                _walk(element, stack | {name})
        elif name in tasks:
            _add(name)
        elif name in cats:
            for rid in cats[name]:
                _add(rid)
        else:
            raise KeyError(f'unknown suite/category/row id: {name!r}')

    for part in suite_expr.split('+'):
        part = part.strip()
        if part:
            _walk(part, set())
    return out


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('suites', help='list suites and their expansion')
    e = sub.add_parser('expand', help='expand one suite expression')
    e.add_argument('--suite', required=True)
    e.add_argument('--tsv', action='store_true', help='machine-readable TSV')
    args = ap.parse_args(argv)

    tasks, suites = load_custom()
    if args.cmd == 'suites':
        for name in suites:
            ids = [rid for rid, _ in expand(name, tasks, suites)]
            print(f'{name:<12} ({len(ids):>2}) {" ".join(ids)}')
        return 0
    rows = expand(args.suite, tasks, suites)
    if args.tsv:
        print('id\ttask\tseqlen\tfewshot\tmbs\test_min\tout\tneeds')
        for rid, r in rows:
            print(f'{rid}\t{r.task}\t{r.seqlen}\t{r.fewshot}\t{r.mbs}\t{r.est_minutes}'
                  f'\t{r.output_type}\t{",".join(r.needs)}')
    else:
        total = sum(r.est_minutes for _, r in rows)
        for rid, r in rows:
            lock = ' [locked]' if r.seqlen_locked else ''
            print(f'  {rid:<16} {r.task:<16} seq={r.seqlen}{lock} fs={r.fewshot} '
                  f'mbs={r.mbs} ~{r.est_minutes}min ({r.seqlen_src})')
        print(f'  total scoring estimate: {total} min ({total/60:.1f} h) before packing')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
