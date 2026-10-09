#!/usr/bin/env python3
"""significance.py — PAIRED per-claim statistical testing for the branch showcase.

WHY PAIRED, AND WHY IT MATTERS
  Every branch is scored on the IDENTICAL committed document subset (config/subsets/*.json:
  one seeded draw, the same doc indices, forever). Two checkpoints therefore answer the SAME
  questions, so each document is a matched pair and the only thing that varies within a pair
  is the model. That is a paired design, and comparing two independent binomial error bars
  throws its power away: at n=40 an unpaired comparison of 0.375 vs 0.275 has overlapping
  +/-1 SE bars and says nothing, while the paired form asks the much sharper question "on how
  many documents did exactly one of the two succeed, and did one side win those discordant
  documents more often than a coin would?". McNemar's test is exactly that question, and it
  is the correct test for this design.

  Concretely: shared documents cancel. If both checkpoints get the same 25 documents right and
  the same 10 wrong, those 35 documents carry no information about which is better -- only the
  5 discordant ones do. The unpaired test still pays variance for all 40.

HOW PER-DOCUMENT CORRECTNESS IS RECOVERED (no GPU, no model)
  The response cache (megatron_ds_eval/cache.py) stores every per-request generation keyed by
  a content hash of (method, request args). So we rebuild the task through lm-eval EXACTLY as
  the job did -- same subset manifest, same num_fewshot, same include_path, same max_gen_toks
  -- serve every request from the cache via a CacheOnlyLM stub, and let lm-eval's own filters
  and `process_results` grade each document. `log_samples=True` then hands back the per-document
  metric values. Nothing here re-implements a grader: the numbers come from the task's own
  scoring code, which is why the recomputed mean is checked against the value the registry
  published and a mismatch is printed in the row rather than swallowed.

  request_key() hashes gen_kwargs, and jobs from 2026-08-16 pin max_gen_toks from the task
  table (main.py: "the task table's gen_toks is AUTHORITATIVE"). A cache written before that
  change therefore has DIFFERENT keys from one written after, for the same task and model. We
  try both key families and record which one matched, so an old cache is recovered rather than
  reported as missing.

  THE MEMO IS KEYED ON THE GRADER, NOT JUST THE CACHE. Re-scoring is expensive (humaneval_chat
  and mbpp_chat exec 80 programs each), so per-document vectors are memoised -- but a memo
  keyed only on (response cache, task) is a trap: when custom_tasks/*/utils.py was fixed on
  2026-08-17 (the `_extract_code` leading-space defect that read the whole full-loss coding
  column as 0.000), a cache-keyed memo would have kept serving the WRONG per-document
  correctness for exactly the claim under test. The key therefore includes a fingerprint of
  the task's grader sources and the lm-eval version, so editing a grader invalidates the memo
  automatically instead of relying on someone remembering to clear it.

SUBSET VARIANTS (n=40 today, n=250 when the bigger job lands)
  A task may have more than one committed manifest -- ifeval_s (n=40) and ifeval_s250 (n=250,
  a strict superset built from the same seed). Manifests are DISCOVERED from config/subsets/,
  not hardcoded, and every variant for which BOTH sides have a provenance-matching registry row
  is tested and labelled with its own n. So when the n=250 rows land this file re-runs and
  reports them with no edit. Variants that select identical documents (gsm8k appears in both
  gsm8k_chat_s and short_q512_fs0) are deduplicated by the index content, not by their name.

WHAT IS TESTED
  McNemar's EXACT test (binomial on the discordant pairs, not the chi-square approximation --
  with 3 or 5 discordant documents the asymptotic form is not usable), a bootstrap 95% CI over
  documents for the paired difference, and Cohen's g / the discordance odds ratio for effect
  size, plus Holm-Bonferroni across the family. A comparison touching a
  render_branches.GEN_ARTIFACT cell is EXCLUDED, not tested: those cells measure the generation
  budget, not the model, so a p-value on them would be a significance claim about max_gen_toks.

  Rows that cannot be computed are PRINTED AS ROWS with the specific reason -- missing registry
  row, absent cache file, key-family mismatch, or artifact flag. A missing test must be
  visible; silently dropping it is how a table comes to look stronger than the evidence.

    ROCM_PATH=/opt/rocm-default $ENV_PY significance.py [--boot 10000] [--only SUBSTR]
"""
import argparse
import hashlib
import json
import math
import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(EVAL_DIR)))
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'tasks', 'eval_harness'))

os.environ.setdefault('HF_DATASETS_CACHE', os.path.join(EVAL_DIR, 'cache', 'hf_datasets_cache'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_ALLOW_CODE_EVAL', '1')

# The assignment of record for which (run, step) belongs to which branch, which document
# subset each task committed to, and which cells measure the budget instead of the model.
# Imported, never restated -- a branch re-attribution must move this file with it.
import render_branches as rb                                   # noqa: E402
import result_registry                                         # noqa: E402
import task_table                                              # noqa: E402
from megatron_ds_eval.cache import request_key                 # noqa: E402

CACHE_DIR = os.path.join(EVAL_DIR, 'cache', 'responses')
SUBSET_DIR = os.path.join(EVAL_DIR, 'config', 'subsets')
MEMO_DIR = os.path.join(EVAL_DIR, 'cache', 'perdoc')
BOOT_SEED = 20260817        # fixed: a CI that moves between runs is not a CI


# ----------------------------------------------------------------- cache-served LM
def load_cache(path):
    """Response cache -> {key: value}. Torn final line skipped, duplicate keys last-wins,
    exactly as ResponseCache._load_and_sync does (a walltime kill leaves a half line)."""
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if 'k' in rec:
                data[rec['k']] = rec['v']
    return data


from lm_eval.api.model import LM as _LM       # noqa: E402  (needs the eval env python)


# A cache miss is answered with a SENTINEL rather than a plausible dummy, so that the
# logged sample record itself says whether the document was really evaluated.
#
# This replaced tracking coverage by request.doc_id, which is a trap: `request.doc_id` is
# POSITIONAL within the selected sample list (0..39) while the `doc_id` lm-eval writes into
# log_samples is the ORIGINAL dataset index (14, 24, 35, ...). Cross-referencing the two
# silently matched the wrong documents -- on the committed ifeval subset it turned a 0.375
# measurement into 1.000. The sentinel is carried by the response, so no id arithmetic is
# needed and group tasks (whose leaves each restart at 0) need no special case either.
MISS_TEXT = '\x00XMOE_CACHE_MISS\x00'
MISS_LOGPROB = -1e300


def _has_miss(resps):
    """True if any response behind this sample record was a cache miss."""
    if isinstance(resps, str):
        return resps == MISS_TEXT
    if isinstance(resps, float):
        return resps == MISS_LOGPROB
    if isinstance(resps, (list, tuple)):
        return any(_has_miss(x) for x in resps)
    return False


class CacheOnlyLM(_LM):
    """lm-eval LM that answers only from a response-cache dict (see score_partial.py).

    Every request is looked up by the same content hash the harness wrote it under; a hit
    replays the stored generation, a miss returns a sentinel (see above) so the document
    can be dropped after grading.
    """

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.hits = 0
        self.misses = 0

    def _serve(self, method, requests, miss):
        out = []
        for req in requests:
            key = request_key(method, tuple(req.args))
            if key in self.data:
                self.hits += 1
                out.append(self.data[key])
            else:
                self.misses += 1
                out.append(miss)
        return out

    def loglikelihood(self, requests):
        return [tuple(v) for v in
                self._serve('loglikelihood', requests, [MISS_LOGPROB, False])]

    def loglikelihood_rolling(self, requests):
        return [float(v) for v in
                self._serve('loglikelihood_rolling', requests, MISS_LOGPROB)]

    def generate_until(self, requests):
        return [str(v) for v in self._serve('generate_until', requests, MISS_TEXT)]


# ------------------------------------------------------- task configuration (derived)
def short_row(task):
    """The task_table row that DEFINES the committed measurement for this task.

    Derived, not hardcoded: the fast-cadence subsets are exactly the category=='short'
    rows, and within that category the lm-eval task name is unique (humaneval_s carries
    task 'humaneval', humaneval_chat_s carries 'humaneval_chat' -- different tasks).
    """
    hits = [r for r in task_table.TASKS.values()
            if r.category == 'short' and r.task == task]
    if len(hits) != 1:
        raise KeyError(f'{task}: expected exactly 1 short row, found {len(hits)}')
    return hits[0]


def eval_config(task):
    """(include_path, allow_code_eval, gen_toks, fewshot), parsed out of the SAME
    extra_eval_flags string submit_eval hands the harness."""
    row = short_row(task)
    toks = row.extra_eval_flags.split()
    include = None
    allow_code = False
    for i, t in enumerate(toks):
        if t in ('--include-path', '--include_path'):
            include = toks[i + 1]
        elif t in ('--allow-code-eval', '--allow_code_eval'):
            allow_code = True
    return include, allow_code, row.gen_toks, row.fewshot


def _own_keys(manifest, task):
    """The manifest keys belonging to `task` -- itself, or its leaf subtasks.

    Same rule main.py uses to decide a task's own share of a shared manifest, homed in
    result_registry.own_manifest_keys(): bbh's leaves are `bbh_cot_fewshot_*`,
    minerva_math's are `minerva_math_*`, short_q512_fs0.json holds gsm8k and hellaswag
    side by side -- and RULER's leaves span two families (niah_*, ruler_*), which the
    bare prefix rule counted as 5 of 13. Taking the whole file would build the other
    task too and mis-state n.
    """
    return {k: manifest[k]
            for k in result_registry.own_manifest_keys(task, manifest)}


def subset_variants(task):
    """Every distinct document set in config/subsets/ that scores this task.

    DISCOVERED, not hardcoded, so a new manifest (ifeval_s250) becomes testable the moment
    its evaluation rows land -- no edit here. Deduplicated by the doc indices themselves:
    gsm8k lives in both gsm8k_chat_s.json and short_q512_fs0.json with the same 100 indices,
    which is one measurement under two filenames, not two variants.
    """
    _, _, _, fewshot = eval_config(task)
    committed_n = (rb.COMMITTED.get(task) or (None, None))[0]
    seen, out = {}, []
    for fn in sorted(os.listdir(SUBSET_DIR)):
        if not fn.endswith('.json'):
            continue
        d = json.load(open(os.path.join(SUBSET_DIR, fn)))
        meta = d.pop('_meta', {})
        own = _own_keys(d, task)
        if not own:
            continue
        n = sum(len(v) for v in own.values())
        sha = hashlib.sha256(json.dumps({k: sorted(v) for k, v in own.items()},
                                        sort_keys=True).encode()).hexdigest()[:16]
        if sha in seen:
            seen[sha]['ids'].append(fn[:-5])
            continue
        v = {'ids': [fn[:-5]], 'n': n, 'samples': own, 'sha': sha,
             'fewshot': meta.get('num_fewshot', fewshot),
             'committed': n == committed_n}
        seen[sha] = v
        out.append(v)
    # Committed subset first (it is what the published tables report), then by size.
    out.sort(key=lambda v: (not v['committed'], -v['n']))
    return out


def grader_fp(task):
    """Fingerprint of everything that decides how a document is SCORED.

    lm-eval's version covers the built-in graders and filters; the custom chat tasks are
    local files that change under us -- the 2026-08-17 `_extract_code` fix rewrote the whole
    full-loss coding column. Including this in the memo key makes a grader edit invalidate
    the cached per-document vectors automatically.
    """
    import lm_eval
    h = hashlib.sha256(getattr(lm_eval, '__version__', '?').encode())
    include, _, _, _ = eval_config(task)
    tdir = os.path.join(include, task) if include else None
    if tdir and os.path.isdir(tdir):
        for fn in sorted(os.listdir(tdir)):
            if fn.endswith(('.py', '.yaml')):
                h.update(fn.encode())
                h.update(open(os.path.join(tdir, fn), 'rb').read())
    return h.hexdigest()[:12]


# ----------------------------------------------------- provenance: which row, which cache
def winning_row(rows, run, step, task, metric, want):
    """The registry row that owns this (checkpoint, task, metric) at subset size `want`.

    Mirrors render_branches.cell's policy -- rank (complete-before-partial, timestamp), and
    reject any row whose ACTUALLY SCORED document count and shot count are not `want` --
    because the significance test must be computed on the measurement the table publishes,
    not on some other run wearing the same task name. `want` is parameterised rather than
    read from rb.COMMITTED so a larger committed subset (ifeval n=250) is a value, not a
    code change.

    Returns (row, [rejection reasons]) so a missing test can explain itself.
    """
    step = rb.STEP_ALIAS.get(run, {}).get(step, step)
    best, why = None, []
    for r in rows:
        if (r.get('ckpt_run') != run or str(r.get('step')) != step
                or r.get('task') != task or str(r.get('metric', '')) != metric):
            continue
        partial = bool(r.get('partial'))
        if not partial and want is not None:
            # The sample_len row must come from THE SAME WRITE (job_id + ts); matching only
            # (run, step, task) borrows a document count from an unrelated job.
            n_scored = next(
                (x.get('value') for x in rows
                 if x.get('ckpt_run') == run and str(x.get('step')) == step
                 and x.get('task') == task and x.get('metric') == 'sample_len'
                 and bool(x.get('partial')) == partial
                 and x.get('job_id') == r.get('job_id') and x.get('ts') == r.get('ts')),
                None)
            got = (int(n_scored) if n_scored is not None else r.get('limit'),
                   r.get('num_fewshot'))
            if got != want:
                why.append(f'(docs,fewshot)={got}')
                continue
        key = (not partial, r.get('ts') or '')
        if best is None or key > best[0]:
            best = (key, r)
    return (best[1] if best else None), sorted(set(why))


# --------------------------------------------------------------- per-document recovery
def per_doc(cache_name, task, variant, refresh=False, log=print):
    """Recover per-document metric values for one (cached checkpoint, task, subset).

    Returns {'ok', 'why', 'gen_variant', 'n', 'n_total', 'docs'} where docs is
    {metric_name: {unit_id: float}}, metric_name uses the registry's spelling
    ('exact_match/flexible-extract') and unit_id is 'leaf_task#doc_id'.

    Two key families are tried, because request_key hashes gen_kwargs and the task table's
    max_gen_toks pin (main.py, 2026-08-16) changed the keys mid-campaign: 'yaml' = the task
    YAML's own generation_kwargs, 'pinned' = max_gen_toks forced to the table's gen_toks.
    Whichever covers more documents wins and is reported. Where the YAML already sets the
    same value (humaneval_chat, mbpp_chat: 512 in both places) the families coincide and
    either answer is the same answer.
    """
    fp = grader_fp(task)
    memo = os.path.join(MEMO_DIR, f'{task}__{variant["sha"]}__{fp}__'
                                  f'{hashlib.sha256(cache_name.encode()).hexdigest()[:16]}.json')
    if not refresh and os.path.isfile(memo):
        try:
            return json.load(open(memo))
        except ValueError:
            pass

    path = os.path.join(CACHE_DIR, cache_name)
    if not os.path.isfile(path):
        return {'ok': False, 'why': f'response cache file absent on disk (`{cache_name}`)',
                'gen_variant': '', 'n': 0, 'n_total': 0, 'docs': {}}
    data = load_cache(path)

    include, allow_code, gen_toks, _ = eval_config(task)
    from lm_eval import simple_evaluate

    def run(gen_kwargs):
        lm = CacheOnlyLM(data)
        kw = {'samples': variant['samples']}
        if gen_kwargs:
            kw['gen_kwargs'] = gen_kwargs
        if include:
            from lm_eval.tasks import TaskManager as _TM
            kw['task_manager'] = _TM(include_path=include)
        if allow_code:
            kw['confirm_run_unsafe_code'] = True
        res = simple_evaluate(model=lm, tasks=[task], num_fewshot=variant['fewshot'],
                              bootstrap_iters=0, verbosity='ERROR',
                              log_samples=True, **kw)
        # Coverage read off the LOGGED RECORDS, in their own doc_id space -- never by
        # cross-referencing request.doc_id, which counts positions, not documents.
        covered, total = set(), set()
        for leaf, recs in (res.get('samples') or {}).items():
            for rec in recs:
                unit = f'{leaf}#{rec["doc_id"]}'
                total.add(unit)
                if not _has_miss(rec.get('resps')):
                    covered.add(unit)
        return res, covered, total

    variants = [('yaml', None)]
    if gen_toks:
        variants.append(('pinned', f'max_gen_toks={int(gen_toks)}'))

    best = None
    for name, gk in variants:
        try:
            res, covered, total = run(gk)
        except Exception as e:                       # a task that will not build at all
            log(f'[{task}] key family {name} failed to build: {type(e).__name__}: {e}')
            continue
        log(f'[{task}/{variant["ids"][0]}] key family {name}: '
            f'{len(covered)}/{len(total)} documents covered')
        if best is None or len(covered) > len(best[1]):
            best = (name, covered, total, res)
        if covered and len(covered) == len(total):
            break

    if best is None:
        return {'ok': False, 'why': 'task could not be rebuilt under any key family',
                'gen_variant': '', 'n': 0, 'n_total': 0, 'docs': {}}
    name, covered, total, res = best
    if not covered:
        return {'ok': False,
                'why': (f'KEY-FAMILY MISMATCH — 0/{len(total)} documents recoverable from '
                        f'`{cache_name}` under either gen_kwargs family'),
                'gen_variant': name, 'n': 0, 'n_total': len(total), 'docs': {}}

    # PARTIAL COVERAGE simply drops the uncovered documents. Every document was graded in
    # the pass above; the ones whose generation was a sentinel are not evidence and are not
    # reported. Re-running with samples= restricted to the covered ids was tried and is
    # WRONG -- re-entering simple_evaluate renumbers the selection, so the ids no longer
    # denote the same documents.
    docs = {}
    for leaf, recs in (res.get('samples') or {}).items():
        for rec in recs:
            filt = rec.get('filter', 'none')
            suffix = '' if filt in ('', 'none') else f'/{filt}'
            unit = f'{leaf}#{rec["doc_id"]}'
            if unit not in covered:
                continue                    # answered from a sentinel; not evidence
            for k, v in rec.items():
                if k in ('doc', 'doc_hash', 'prompt_hash', 'target_hash', 'target',
                         'arguments', 'resps', 'filtered_resps', 'metrics', 'filter',
                         'doc_id'):
                    continue
                if isinstance(v, bool):
                    v = float(v)
                elif isinstance(v, (int, float)):
                    v = float(v)
                else:
                    continue                    # list-valued metrics (ifeval inst_level)
                docs.setdefault(k + suffix, {})[unit] = v

    out = {'ok': True, 'why': '', 'gen_variant': name, 'n': len(covered),
           'n_total': len(total), 'docs': docs}
    os.makedirs(MEMO_DIR, exist_ok=True)
    json.dump(out, open(memo, 'w'))
    return out


# ------------------------------------------------------------------------- statistics
def mcnemar_exact_p(b, c):
    """Two-sided exact McNemar: binomial(b+c, 0.5) on the discordant pairs.

    Exact rather than the chi-square approximation on purpose. With b+c in the single
    digits -- which is what n=40 with two similar checkpoints produces -- the continuity-
    corrected chi-square is not trustworthy, and the exact form costs one loop.
    Verified identical to scipy.stats.binomtest(b, b+c, 0.5) over all b,c < 40.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2 ** n)
    return min(1.0, 2.0 * tail)


def bootstrap_ci(a, b, iters, seed=BOOT_SEED):
    """Percentile 95% CI for mean(a)-mean(b), resampling DOCUMENTS (paired: the same
    resampled index is taken from both vectors, which is what preserves the pairing)."""
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    if n == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm_bonferroni(ps):
    """Holm-Bonferroni step-down adjusted p-values, in the input order.

    This table runs a FAMILY of tests (every task x every branch pair), and at ~30 tests
    one p<0.05 is roughly what noise alone produces. Holm controls the family-wise error
    rate, is uniformly more powerful than plain Bonferroni, and needs no independence
    assumption -- which matters here, because the same base checkpoint appears in several
    rows and the tests are therefore correlated.

    Reported ALONGSIDE the raw p, never instead of it: the raw p is the right number for a
    single claim stated in advance (the within-branch learning claims are exactly that),
    the adjusted p is the right number for "which of these 30 cells lit up".
    """
    m = len(ps)
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda i: ps[i])):
        running = max(running, (m - rank) * ps[i])      # step-down monotonicity
        adj[i] = min(1.0, running)
    return adj


def effect_size(b, c):
    """(Cohen's g, odds ratio) on the discordant pairs.

    Cohen's g = |P - 0.5| where P is A's share of the discordant documents: a pure
    "how lopsided are the documents that actually distinguish the two models" number.
    The odds ratio b/c uses the Haldane-Anscombe +0.5 correction so a clean sweep
    (c == 0) reports a large finite value instead of an infinity.
    """
    n = b + c
    g = abs(b / n - 0.5) if n else float('nan')
    return g, (b + 0.5) / (c + 0.5)


# ------------------------------------------------------------------------ comparisons
def branch_named(substr):
    hits = [k for k in rb.BRANCHES if substr.lower() in k.lower()]
    if len(hits) != 1:
        raise KeyError(f'branch match {substr!r} -> {hits}')
    return hits[0]


def point(branch, step):
    for run, s, _ in rb.BRANCHES[branch]:
        if s == step:
            return run, s
    raise KeyError(f'{branch} has no step {step}')


def final_point(branch):
    run, s, _ = rb.BRANCHES[branch][-1]
    return run, s


def build_comparisons():
    """The showcase claims as (title, A-label, A-point, B-label, B-point, tasks-or-None).

    A minus B is the direction the table reports, so A is always the side the claim is
    ABOUT ("full-loss beats masked-SFT", "training helped", "it learned to code").
    Every (run, step) comes from render_branches.BRANCHES -- including the Nemotron branch,
    whose display step 60 STEP_ALIAS maps to the stored checkpoint 120.
    """
    base_b, masked, full = branch_named('Base'), branch_named('Masked-SFT'), branch_named('Full-loss')
    base_pt = final_point(base_b)
    comps = []

    # 1. Head-to-head: the two instruction branches at their final COMMON step, all tasks.
    a, b = final_point(full), final_point(masked)
    assert a[1] == b[1] == '130', f'final common step moved: {a} {b}'
    comps.append(('Full-loss vs Masked-SFT @ final step 130',
                  f'full-loss@{a[1]}', a, f'masked-SFT@{b[1]}', b, None))

    # 2. "Did this training help at all" -- every branch's final checkpoint against the
    #    untrained base, all tasks. The base is the only checkpoint every branch shares.
    for name in rb.BRANCHES:
        if name == base_b:
            continue
        a = final_point(name)
        comps.append((f'{name} final (step {a[1]}) vs BASE (step 0)',
                      f'{name.split(" (")[0]}@{a[1]}', a, 'base@0', base_pt, None))

    # 3. Within-branch endpoints: did the branch LEARN the skill between step 20 and 130?
    #    Restricted to the task each claim is about -- a within-branch endpoint test on an
    #    unrelated task is not the claim and would only inflate the multiple-testing family.
    for bn, short in ((masked, 'masked-SFT'), (full, 'full-loss')):
        for task, label in (('humaneval_chat', 'learns to code (humaneval-chat'),
                            ('mbpp_chat', 'learns to code (mbpp-chat')):
            a, b = point(bn, '130'), point(bn, '20')
            comps.append((f'CLAIM: {short} {label}, step 20 -> 130)',
                          f'{short}@130', a, f'{short}@20', b, [task]))
    a, b = point(full, '130'), point(full, '20')
    comps.append(('CLAIM: full-loss learns instruction-following (ifeval, step 20 -> 130)',
                  'full-loss@130', a, 'full-loss@20', b, ['ifeval']))
    return comps


# ------------------------------------------------------------------------------ main
def verdict(p, n_disc):
    if n_disc == 0:
        return 'no discordant documents — identical on this subset'
    if p < 0.001:
        return 'SIGNIFICANT at p<0.001'
    if p < 0.01:
        return 'SIGNIFICANT at p<0.01'
    if p < 0.05:
        return 'SIGNIFICANT at p<0.05'
    return f'not distinguishable at this n ({n_disc} discordant)'


NA = '| — | — | — | — | — | — | — | — | — | — |'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--boot', type=int, default=10000, help='bootstrap resamples')
    ap.add_argument('--only', default=None, help='substring filter on the comparison title')
    ap.add_argument('--refresh', action='store_true', help='ignore the per-doc memo cache')
    args = ap.parse_args()

    rows = rb.load_rows()
    comps = build_comparisons()
    if args.only:
        comps = [c for c in comps if args.only.lower() in c[0].lower()]

    out = [
        '# Paired significance tests — X-MoE branch showcase',
        '',
        'Every branch is scored on the SAME committed document subset (`config/subsets/*.json`:',
        'one seeded draw, identical doc indices for every checkpoint), so each document is a',
        'matched pair and these comparisons are **paired**, not two independent means. That is',
        'not a stylistic choice: shared documents carry no information about which model is',
        'better, so only the *discordant* documents — the ones exactly one side gets right —',
        'are evidence. McNemar\'s exact test asks whether one side wins those discordant',
        'documents more often than a coin would, which is strictly more powerful than',
        'comparing two ±1 binomial SE bars at n=40.',
        '',
        'Per-document correctness is recovered with each task\'s OWN grader: the task is rebuilt',
        'through lm-eval exactly as the job built it (same subset manifest, shot count,',
        '`--include-path`, `max_gen_toks`), every request is served from that checkpoint\'s',
        'response cache, and lm-eval\'s filters and `process_results` score each document.',
        'No grader is re-implemented here. The recomputed mean is checked against the value the',
        'registry published, and a mismatch is printed in the row.',
        '',
        '**Columns.** `subset` = which committed manifest, with its n. `n` = paired documents',
        'actually recoverable from BOTH caches. `A`/`B` = each side\'s score on exactly those n',
        'documents. `b`/`c` = documents only A / only B got right. `p` = two-sided exact McNemar',
        '(binomial on b+c; exact, not chi-square — b+c is usually single-digit). `95% CI` =',
        f'percentile bootstrap over documents, {args.boot} resamples, seed {BOOT_SEED},',
        'resampling the pair so the pairing survives. `g` = Cohen\'s g on the discordant split,',
        '`OR` = discordance odds ratio (Haldane +0.5).',
        '',
        '**When the CI and the p-value disagree, the exact test governs.** A percentile bootstrap',
        'CI can exclude 0 while McNemar\'s exact p sits above 0.05 (it happens below on the ifeval',
        'claim). The bootstrap CI is an *approximation* whose coverage is unreliable at n=40 with',
        'a small discordant count, while the exact binomial test is exact by construction.',
        'Reading the CI instead of the p because the CI is the friendlier number is exactly the',
        'cherry-pick a reviewer is looking for, so the verdict column is driven by the exact test',
        'and the CI is shown for effect magnitude.',
        '',
        'Cells flagged in `render_branches.GEN_ARTIFACT` measure the GENERATION BUDGET, not the',
        'model, so those comparisons are **EXCLUDED** rather than tested — a p-value there would',
        'be a significance claim about `max_gen_toks`. Comparisons that cannot be computed are',
        'printed as rows with the specific reason: a missing test is kept visible.',
        '',
    ]

    computable = attempted = 0
    summary, notcomp, pending = [], [], set()

    for title, alab, apt, blab, bpt, only_tasks in comps:
        print(f'\n=== {title}', flush=True)
        out += [f'## {title}', '',
                f'A = `{alab}` ({apt[0]} @ step {apt[1]}) · '
                f'B = `{blab}` ({bpt[0]} @ step {bpt[1]})', '',
                '| task | subset | n | A | B | A−B | b (A only) | c (B only) | McNemar p '
                '| bootstrap 95% CI | g | OR | verdict |',
                '|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|']

        for task, metric, disp in rb.METRICS:
            if only_tasks and task not in only_tasks:
                continue

            # --- generation-budget artifacts are excluded, not tested (subset-independent)
            flagged = [(lab, pt) for lab, pt in ((alab, apt), (blab, bpt))
                       if (pt[0], pt[1], task) in rb.GEN_ARTIFACT]
            if flagged:
                attempted += 1
                why = '; '.join(f'{lab}: {rb.GEN_ARTIFACT[(pt[0], pt[1], task)]}'
                                for lab, pt in flagged)
                out.append(f'| {disp} | — {NA} **EXCLUDED** — generation-budget artifact '
                           f'({why}) |')
                notcomp.append((title, disp, '—', f'EXCLUDED (generation-budget artifact) — {why}'))
                print(f'  {disp}: EXCLUDED (gen artifact)', flush=True)
                continue

            for var in subset_variants(task):
                attempted += 1
                sid = f'{var["ids"][0]} (n={var["n"]})'
                want = (var['n'], var['fewshot'])

                # --- both sides must have a provenance-matching registry row
                miss, side = [], {}
                for lab, pt in ((alab, apt), (blab, bpt)):
                    r, rej = winning_row(rows, pt[0], pt[1], task, metric, want)
                    if r is None:
                        miss.append(f'{lab}: no registry row at (docs,fewshot)={want}'
                                    + (f' — rows present but rejected: {", ".join(rej)}'
                                       if rej else ' — no row for this task/metric at all'))
                        continue
                    if not r.get('response_cache'):
                        miss.append(f'{lab}: registry row records no response cache')
                        continue
                    side[lab] = r
                if miss:
                    if var['committed'] or len(side) == 2:
                        out.append(f'| {disp} | {sid} {NA} **NOT COMPUTABLE** — '
                                   f'{"; ".join(miss)} |')
                        notcomp.append((title, disp, sid, '; '.join(miss)))
                        print(f'  {disp} [{sid}]: NOT COMPUTABLE — {miss}', flush=True)
                    else:
                        # A non-committed subset nobody has evaluated yet (ifeval_s250
                        # before its jobs land) is PENDING, not a gap in the evidence.
                        # Recorded so the table shows the bigger test is coming rather
                        # than pretending the manifest does not exist.
                        attempted -= 1
                        pending.add((task, sid, title))
                    continue

                # --- recover per-document correctness from each side's own cache
                vecs, notes, fail = {}, [], []
                for lab, pt in ((alab, apt), (blab, bpt)):
                    r = side[lab]
                    pd = per_doc(r['response_cache'], task, var, refresh=args.refresh,
                                 log=lambda m: print(f'    {m}', flush=True))
                    if not pd['ok']:
                        fail.append(f'{lab}: {pd["why"]}')
                        continue
                    mvals = pd['docs'].get(metric)
                    if not mvals:
                        fail.append(f'{lab}: metric `{metric}` absent from the recovered '
                                    f'per-document scores (have: {sorted(pd["docs"])[:6]})')
                        continue
                    vecs[lab] = mvals
                    notes.append(f'{lab} keys={pd["gen_variant"]}')
                    recomputed = sum(mvals.values()) / len(mvals)
                    if pd['n'] == pd['n_total'] and abs(recomputed - r['value']) > 5e-4:
                        notes.append(f'**{lab} MEAN MISMATCH: recomputed {recomputed:.4f} '
                                     f'vs registry {r["value"]:.4f}**')
                    if pd['n'] < pd['n_total']:
                        notes.append(f'{lab} partial cache {pd["n"]}/{pd["n_total"]}')
                if fail:
                    out.append(f'| {disp} | {sid} {NA} **NOT COMPUTABLE** — '
                               f'{"; ".join(fail)} |')
                    notcomp.append((title, disp, sid, '; '.join(fail)))
                    print(f'  {disp} [{sid}]: NOT COMPUTABLE — {fail}', flush=True)
                    continue

                va, vb = vecs[alab], vecs[blab]
                units = sorted(set(va) & set(vb))
                if not units:
                    out.append(f'| {disp} | {sid} {NA} **NOT COMPUTABLE** — the two '
                               f'recovered document sets do not overlap |')
                    notcomp.append((title, disp, sid, 'recovered document sets disjoint'))
                    continue
                a = [va[u] for u in units]
                b = [vb[u] for u in units]
                n = len(units)
                sa, sb = sum(a) / n, sum(b) / n
                lo, hi = bootstrap_ci(a, b, args.boot)

                if all(x in (0.0, 1.0) for x in a + b):
                    nb = sum(1 for x, y in zip(a, b) if x == 1.0 and y == 0.0)
                    nc = sum(1 for x, y in zip(a, b) if x == 0.0 and y == 1.0)
                    nboth = sum(1 for x, y in zip(a, b) if x == 1.0 and y == 1.0)
                    p = mcnemar_exact_p(nb, nc)
                    g, orr = effect_size(nb, nc)
                    vd = verdict(p, nb + nc)
                    pcol, gcol, ocol = f'{p:.4f}', f'{g:.3f}', f'{orr:.2f}'
                    bcol, ccol = str(nb), str(nc)
                    extra = f' · both right {nboth}, both wrong {n - nb - nc - nboth}'
                else:
                    p = float('nan')
                    nb = nc = 0
                    vd = ('non-binary metric — McNemar n/a; read the bootstrap CI '
                          '(significant if it excludes 0)')
                    pcol = gcol = ocol = bcol = ccol = 'n/a'
                    extra = ''

                note_txt = '; '.join(notes) + extra
                out.append(f'| {disp} | {sid} | {n} | {sa:.3f} | {sb:.3f} | {sa - sb:+.3f} '
                           f'| {bcol} | {ccol} | {pcol} | [{lo:+.3f}, {hi:+.3f}] | {gcol} '
                           f'| {ocol} | {vd}{" · " + note_txt if note_txt else ""} |')
                print(f'  {disp} [{sid}]: n={n} A={sa:.3f} B={sb:.3f} d={sa-sb:+.3f} '
                      f'b={nb} c={nc} p={pcol} CI=[{lo:+.3f},{hi:+.3f}] -> {vd}', flush=True)
                computable += 1
                summary.append([title, disp, sid, sa - sb, p, vd])
        out.append('')

    # ------------------------------------------------------------------ headline verdict
    testable = [s for s in summary if s[4] == s[4]]          # p is not NaN
    adj = holm_bonferroni([s[4] for s in testable])
    for s, a in zip(testable, adj):
        s.append(a)

    out += ['## Which claims survive', '',
            f'{computable} of {attempted} attempted comparisons were computable '
            f'(the rest appear above as EXCLUDED or NOT COMPUTABLE, never dropped).',
            '',
            f'`p_holm` is the Holm-Bonferroni adjusted p over the {len(testable)} tests in this',
            'table. It is the number to quote when scanning the table for whatever lit up; the',
            'raw `p` is the number to quote for a claim stated in advance, which is what the',
            'within-branch learning claims are.',
            '',
            '| comparison | task | subset | A−B | p | p_holm | verdict |',
            '|---|---|---|---:|---:|---:|---|']
    for s in summary:
        title, disp, sid, d, p = s[:5]
        pa = s[6] if len(s) > 6 else None
        out.append(f'| {title} | {disp} | {sid} | {d:+.3f} | '
                   f'{"n/a" if p != p else f"{p:.4f}"} | '
                   f'{"n/a" if pa is None else f"{pa:.3f}"} | {s[5]} |')

    sig = [s for s in testable if s[4] < 0.05]
    sig_h = [s for s in testable if s[6] < 0.05]
    out += ['',
            f'**{len(sig)} of {len(summary)} computable comparisons reach p<0.05 raw; '
            f'{len(sig_h)} survive Holm-Bonferroni across the family.**',
            '']
    if sig:
        out += ['Significant at p<0.05 (raw):', '']
        for s in sorted(sig, key=lambda s: s[4]):
            out.append(f'  - **{s[1]}** · {s[0]} · {s[2]} · Δ={s[3]:+.3f}, '
                       f'p={s[4]:.4f}, p_holm={s[6]:.3f}')
        out.append('')
    out += ['Everything else is a difference this subset size cannot resolve — which is a',
            'result about the evidence, not a claim that the effect is zero. The honest',
            'reading of a non-significant row is "n=40 (or n=80) cannot tell these apart",',
            'and the fix is more documents, not a smaller p. Concretely: at n=40 a paired',
            'comparison needs about a 9-vs-1 split of the discordant documents before the',
            'exact test clears 0.05 (p=0.021), and 9-vs-2 does not (p=0.065) — so a',
            'single-task instruct claim on the n=40 ifeval subset is underpowered by',
            'construction, however the numbers land. That is why ifeval was re-run at n=250;',
            'this file picks the larger subset up automatically once its rows are in the',
            'registry.',
            '']

    if notcomp:
        out += ['## Comparisons that could not be computed', '',
                'Listed so that a missing test stays visible instead of quietly improving the',
                'table. Each row names the specific blocker.', '',
                '| comparison | task | subset | reason |', '|---|---|---|---|']
        for t, d, sid, why in notcomp:
            out.append(f'| {t} | {d} | {sid} | {why} |')
        out.append('')

    if pending:
        out += ['## Larger subsets discovered but not yet evaluated', '',
                'These manifests exist in `config/subsets/` and would give a more powerful',
                'version of the comparison, but no checkpoint has been scored on them yet.',
                'Nothing needs editing here: `subset_variants()` discovers manifests from',
                'disk, so re-running this script once the jobs land tests them automatically',
                'and labels each row with its own n.', '',
                '| subset | task | comparison waiting on it |', '|---|---|---|']
        for task, sid, title in sorted(pending):
            out.append(f'| {sid} | {task} | {title} |')
        out.append('')

    dest = os.path.join(EVAL_DIR, 'results', 'tables', 'SIGNIFICANCE.md')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    open(dest, 'w').write('\n'.join(out) + '\n')
    print(f'\nwrote {dest}  ({computable}/{attempted} computable, {len(sig)} raw p<0.05, '
          f'{len(sig_h)} Holm-significant, {len(notcomp)} not computable)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
