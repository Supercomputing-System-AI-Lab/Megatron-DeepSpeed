#!/usr/bin/env python3
"""rescore_partial.py — score one (run, step, task) cell from its response cache ALONE.

WHY THIS EXISTS
  A generate row whose job keeps hitting the walltime can sit one document short of its
  committed subset forever (bbh 215/216, triviaqa 499/500). The response cache holds every
  generation that WAS produced, but no registry row is ever written, so render_branches.py
  leaves the cell empty and a whole column of the branch table is blank because of one
  missing document out of 216.

  score_partial.py cannot rescue those cells. It probes coverage with one simple_evaluate
  call and then re-evaluates with `samples={task: covered}` — four things break:

    1. GROUP tasks. bbh is 27 subtasks; doc_ids are per-SUBTASK, so `probe.doc_hits` mixes
       27 different documents under doc_id 0 and `samples={'bbh': [...]}` names a group,
       which lm-eval's per-leaf `samples` lookup never matches.
    2. The committed subset. The job ran `--task-samples-json-file config/subsets/<row>.json`;
       rebuilding the FULL task produces different documents at different indices.
    3. The pinned generation budget. main.py passes the task table's gen_toks as
       `gen_kwargs=max_gen_toks=N`, which lands inside the request's gen_kwargs dict and
       therefore inside the cache key. Rebuild without it and every lookup misses.
    4. Second-pass drift. `ContextSampler.rnd` is ONE Random advanced document by document
       (api/samplers.py:26,58), so a few-shot context depends on how many documents were
       built before it. Re-building with a REDUCED document set changes every later context
       and the cache misses even on documents that are present. That is why this script
       builds the requests exactly ONCE, with the committed manifest, and then drops the
       uncovered documents from the already-built instance list instead of rebuilding.

HOW IT WORKS (single build, cache-only serve)
  * Seeds random=0 / numpy=1234 / torch=1234 and the few-shot seed 1234 BEFORE any task is
    loaded — the same values simple_evaluate uses. Proven necessary: without them the
    rebuilt documents differ and every cache lookup misses.
  * Loads the task through TaskManager exactly as main.py does (include_path for the custom
    chat tasks, gen_kwargs pin, num_fewshot override, set_fewshot_seed).
  * Calls build_all_requests(samples=<committed manifest for that leaf>) per LEAF task, so
    the contexts are byte-identical to the ones the job generated from.
  * Marks each instance hit/miss against the cache by recomputing megatron_ds_eval.cache
    .request_key, keeps only documents whose requests ALL hit, and drops the rest from
    task._instances.
  * Hands the pruned tasks to lm_eval.evaluator.evaluate() with a cache-only LM. lm-eval
    itself then runs the task's own filters, process_results, per-leaf aggregation, group
    aggregation (bbh's unweighted mean over its 27 subtasks) and sample_len — no metric
    logic is reimplemented here, which is what makes the recovered number comparable with
    the published ones.
  * --record appends a `partial`-marked row per metric to results/registry/<run>.jsonl with
    limit = the committed subset size and sample_len = the documents actually scored.

USAGE (login node, eval env python; no GPU is touched)

    PATH=<env>/bin:$PATH ROCM_PATH=/opt/rocm-default \\
    HF_DATASETS_CACHE=$PWD/cache/hf_datasets_cache HF_HUB_OFFLINE=1 \\
      python rescore_partial.py --run 2026-08-15_5276906 --step 100 --task bbh --record

    # sanity gate: recompute a cell that already has a COMPLETE registry row and compare
      python rescore_partial.py --run 2026-08-14_5266788 --step 80 --task bbh --verify
"""
import argparse
import glob
import json
import os
import random
import re
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(EVAL_DIR)))
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'tasks', 'eval_harness'))

import task_table

# Rows defined in config/custom_suites.py (ifeval_s250, humaneval_chat_full, ...) live
# only in the merged view; the raw TASKS dict does not know them.
_ALL_TASKS, _ALL_SUITES = task_table.load_custom()                                 # noqa: E402
import result_registry                            # noqa: E402
from megatron_ds_eval.cache import request_key    # noqa: E402  (hash only, no torch model)

CACHE_DIR = os.path.join(EVAL_DIR, 'cache', 'responses')

# lm-eval task name -> the task_table row that defines the COMMITTED measurement for it.
# Everything else (few-shot count, gen_toks, seqlen, subset manifest, include_path) is read
# out of that row, so this file never restates a number task_table.py already owns.
ROW_FOR_TASK = {
    'bbh': 'bbh_s',
    'triviaqa': 'triviaqa_s',
    'minerva_math': 'math_s',
    'mbpp_chat': 'mbpp_chat_s',
    'humaneval_chat': 'humaneval_chat_s',
    'humaneval': 'humaneval_s',
    'ifeval': 'ifeval_s',
    'gsm8k': 'gsm8k_chat_s',
    'hellaswag': 'hellaswag_s',
}


# --------------------------------------------------------------------------- row plumbing
def row_for(task, row_id=None):
    """The task_table Row that pins this measurement, plus its parsed extra flags."""
    rid = row_id or ROW_FOR_TASK.get(task)
    if rid is None or rid not in _ALL_TASKS:
        raise SystemExit(f'no task_table row for task {task!r}; pass --row explicitly')
    row = _ALL_TASKS[rid]
    if row.task != task:
        raise SystemExit(f'row {rid} is task {row.task!r}, not {task!r}')
    flags = row.extra_eval_flags or ''
    def _opt(name):
        m = re.search(re.escape(name) + r'\s+(\S+)', flags)
        return m.group(1) if m else None
    return rid, row, {
        'samples_json': _opt('--task-samples-json-file'),
        'include_path': _opt('--include-path'),
        'metadata_json_file': _opt('--task-metadata-json-file'),
        'allow_code_eval': '--allow-code-eval' in flags,
    }


def load_manifest(path):
    with open(path) as f:
        samples = json.load(f)
    samples.pop('_meta', None)
    return samples


def committed_size(samples, task):
    """This task's OWN share of the manifest — result_registry.own_manifest_keys's rule,
    the ONE home of the own-keys rule (main.py and significance.py read the same one).

    A manifest may serve several tasks in one job (short_q512_fs0.json holds gsm8k and
    hellaswag), and the registry's `limit` is the subset identity for THIS task only.
    Group manifests key on LEAF names, which carry the group prefix for bbh
    (bbh_cot_fewshot_*) and minerva_math (minerva_math_*) — and span TWO families for
    RULER (niah_*, ruler_*), which is the special case the shared rule carries.
    """
    own = sum(len(samples[k]) for k in result_registry.own_manifest_keys(task, samples))
    return own or sum(len(v) for v in samples.values())


# --------------------------------------------------------------------------- cache lookup
def load_cache(path):
    data = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:          # torn final line from a walltime kill
                continue
            if 'k' in rec:
                data[rec['k']] = rec['v']       # duplicate keys: last wins
    return data


CACHE_RE = re.compile(r'(?P<run>.+)__global_step(?P<step>\d+)_m(?P<mtime>\d+)_'
                      r'code(?P<code_fp>[0-9a-f]+)_(?P<ws>\w+)_pad(?P<pad>\w+)_'
                      r'seq(?P<seq>\d+)_fs(?P<fs>\d+)(?P<mode>_nokv)?\.jsonl$')


def find_caches(run, step, seq, fs, code_fp=None):
    """Every cache file matching this cell's identity, richest first.

    A cell can have one file per code fingerprint (the fingerprint is part of the cache
    identity so a forward-code change cannot serve stale generations). We never merge
    across fingerprints — that would publish a number produced by two different forwards —
    so ONE file is chosen and the choice is recorded in the registry row.

    Size is only the first guess; the caller picks by COVERAGE. A file can be full of
    responses none of which this cell can use: the caches written before the task table's
    gen_toks pin (main.py:412) carry the YAML's max_gen_toks in their request keys, so
    every lookup misses even though the file holds 499 generations.
    """
    pat = os.path.join(CACHE_DIR, f'{run}__global_step{step}_*_seq{seq}_fs{fs}.jsonl')
    out = []
    for p in sorted(glob.glob(pat)):
        m = CACHE_RE.search(os.path.basename(p))
        if not m or m.group('mode'):
            continue
        ident = m.groupdict()
        if code_fp and ident['code_fp'] != code_fp:
            continue
        out.append((p, ident, load_cache(p)))
    out.sort(key=lambda t: (-len(t[2]), -os.path.getmtime(t[0])))
    return out


# ------------------------------------------------------------------------------ cache LM
from lm_eval.api.model import LM as _LM   # noqa: E402  (needs the eval env's lm_eval)


class CacheOnlyLM(_LM):
    """Serves lm-eval requests exclusively from a response-cache dict.

    Every request reaching it has already been proven present (uncovered documents are
    dropped before evaluate() runs), so a miss here is a bug, not a partial run — it
    raises rather than silently scoring a dummy.
    """

    def __init__(self, data):
        super().__init__()
        self.data = data

    def _serve(self, method, requests):
        out = []
        for req in requests:
            key = request_key(method, tuple(req.args))
            if key not in self.data:
                raise KeyError(f'MISS after pruning: {method} doc {req.doc_id} '
                               f'task {req.task_name}')
            out.append(self.data[key])
        return out

    def loglikelihood(self, requests):
        return [tuple(v) for v in self._serve('loglikelihood', requests)]

    def loglikelihood_rolling(self, requests):
        return [float(v) for v in self._serve('loglikelihood_rolling', requests)]

    def generate_until(self, requests):
        return [str(v) for v in self._serve('generate_until', requests)]


# ------------------------------------------------------------------------------- scoring
def build_tasks(task, row, opts, samples):
    """Load the task and build its committed requests EXACTLY as the job did.

    Returns (loaded_task_dict, leaves). Done once per process: the build is the expensive
    step, and every candidate cache is then probed against the same instances.
    """
    import numpy as np
    from lm_eval.defaults import DEFAULT_OTHER_SEED, DEFAULT_RANDOM_SEED
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import set_torch_seed, simple_parse_args_string

    # SEEDS FIRST. simple_evaluate seeds before it loads tasks; RULER's synthetic document
    # generation and every few-shot draw depend on this. Rebuilt documents differ without
    # it and every single cache lookup misses.
    random.seed(DEFAULT_RANDOM_SEED)                 # 0
    np.random.seed(DEFAULT_OTHER_SEED)               # 1234
    set_torch_seed(DEFAULT_OTHER_SEED)               # 1234

    metadata = None
    if opts.get('metadata_json_file'):
        with open(opts['metadata_json_file']) as f:
            metadata = json.load(f)
    tm = TaskManager(include_path=opts.get('include_path'), metadata=metadata)
    loaded = tm.load([task])
    leaves = loaded['tasks']

    # Config overrides, in simple_evaluate's order and with its semantics.
    gen_kwargs = None
    if row.gen_toks:
        # main.py:412-416 — the task table's gen_toks is authoritative over the YAML, and
        # it travels as the same `max_gen_toks=N` string, so the parsed dict (and hence the
        # request key) is identical to the job's.
        gen_kwargs = simple_parse_args_string(f'max_gen_toks={int(row.gen_toks)}')
    for name, obj in leaves.items():
        if obj.get_config('output_type') == 'generate_until' and gen_kwargs:
            obj.set_config(key='generation_kwargs', value=gen_kwargs, update=True)
        if row.fewshot is not None:
            if obj.get_config('num_fewshot') != 0:
                obj.set_config(key='num_fewshot', value=row.fewshot)
        obj.set_fewshot_seed(seed=DEFAULT_OTHER_SEED)

    for name, obj in leaves.items():
        idx = samples.get(name)
        if idx is None:
            raise SystemExit(f'manifest has no entry for leaf task {name!r}; the cell was '
                             f'measured with a different subset file')
        obj.build_all_requests(limit=None, samples=idx, rank=0, world_size=1,
                               cache_requests=False, rewrite_requests_cache=False,
                               system_instruction=None, apply_chat_template=False,
                               fewshot_as_multiturn=False, chat_template=None,
                               tokenizer_name='')
    return loaded, leaves


def probe(leaves, data):
    """leaf -> (covered doc-id set, docs built). A doc counts only if ALL its requests hit."""
    out = {}
    for name, obj in leaves.items():
        hits_by_doc = {}
        for inst in obj.instances:
            ok = request_key(inst.request_type, tuple(inst.args)) in data
            hits_by_doc[inst.doc_id] = hits_by_doc.get(inst.doc_id, True) and ok
        out[name] = ({d for d, ok in hits_by_doc.items() if ok}, len(hits_by_doc))
    return out


def score_covered(loaded, leaves, samples, data, cov, opts):
    """Prune every leaf to its covered documents and let lm-eval do all the metric work."""
    from lm_eval.evaluator import evaluate

    for name, obj in leaves.items():
        covered = cov[name][0]
        if not covered:
            raise SystemExit(f'{name}: 0 of {cov[name][1]} documents are cached — '
                             f'nothing to recover for this cell')
        obj._instances = [i for i in obj.instances if i.doc_id in covered]
        # evaluate() would rebuild the requests (with the reduced set, changing every
        # later few-shot context). Freeze what we built, and narrow the scoring iterator
        # to the covered documents while KEEPING their original local ids so
        # instances_by_doc_id still lines up.
        obj.build_all_requests = lambda **kw: None
        obj.doc_iterator = _covered_iterator(type(obj).doc_iterator.__get__(obj), covered)

    results = evaluate(lm=CacheOnlyLM(data), task_dict=loaded, limit=None,
                       samples={n: samples[n] for n in leaves},
                       bootstrap_iters=100000, log_samples=False,
                       confirm_run_unsafe_code=bool(opts.get('allow_code_eval')))
    return results['results']


def _covered_iterator(orig, covered):
    def it(*, rank=0, limit=None, world_size=1, samples=None):
        for doc_id, doc in orig(rank=rank, limit=limit, world_size=world_size,
                                samples=samples):
            if doc_id in covered:
                yield doc_id, doc
    return it


# ------------------------------------------------------------------------------ registry
def existing_meta(run, step, task):
    """Provenance fields copied from any earlier PRODUCTION row for the same checkpoint.

    Reusing the job's own ckpt_step_dir / model / moe_type keeps a recovered row
    indistinguishable from a measured one everywhere except `partial`. Experiment rows
    are skipped (result_registry.experiment_of): copying one would carry its
    `experiment` and `response_cache` fields onto a row meant for publication.
    """
    path = result_registry.shard_path(REPO_ROOT, run)
    best = None
    for r in result_registry.load_rows(path):
        if not result_registry.is_production(r):
            continue
        if r.get('ckpt_run') == run and str(r.get('step')) == str(step):
            if r.get('task') == task:
                return r
            best = best or r
    return best or {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', required=True, help='ckpt_run, e.g. 2026-08-15_5276906')
    ap.add_argument('--step', required=True, help='STORED checkpoint step')
    ap.add_argument('--task', required=True, help='lm-eval task name, e.g. bbh')
    ap.add_argument('--row', default=None, help='task_table row id (default: from --task)')
    ap.add_argument('--cache', default=None, help='explicit response-cache path')
    ap.add_argument('--code-fp', default=None, help='restrict cache choice to this code_fp')
    ap.add_argument('--record', action='store_true',
                    help='append partial-marked rows to results/registry/<run>.jsonl')
    ap.add_argument('--verify', action='store_true',
                    help='compare against the published COMPLETE row instead of recording')
    args = ap.parse_args()

    rid, row, opts = row_for(args.task, args.row)
    if not opts['samples_json']:
        raise SystemExit(f'row {rid} has no --task-samples-json-file; this script only '
                         f'recovers committed-subset cells')
    samples = load_manifest(opts['samples_json'])
    n_committed = committed_size(samples, args.task)

    print(f'[rescore] {rid} {args.task} {args.run}@{args.step}')
    if args.cache:
        cands = [(args.cache, CACHE_RE.search(os.path.basename(args.cache)).groupdict(),
                  load_cache(args.cache))]
    else:
        cands = find_caches(args.run, args.step, row.seqlen, row.fewshot, args.code_fp)
    cands = [c for c in cands if c[2]]
    if not cands:
        raise SystemExit(f'no non-empty response cache for {args.run}@{args.step} '
                         f'seq{row.seqlen} fs{row.fewshot} — nothing to recover')

    loaded, leaves = build_tasks(args.task, row, opts, samples)
    best = None
    for path, ident, data in cands:
        cov = probe(leaves, data)
        n_cov = sum(len(c[0]) for c in cov.values())
        print(f'[rescore] cache code{ident["code_fp"]}: {len(data)} responses, '
              f'{n_cov}/{n_committed} documents usable')
        if best is None or n_cov > best[0]:
            best = (n_cov, path, ident, data, cov)
    n_cov, cache_path, ident, data, cov = best
    if n_cov == 0:
        raise SystemExit('[rescore] no cached response matches this cell\'s requests '
                         '(wrong shot count, seqlen, or a cache written before the '
                         'gen_toks pin) — nothing to recover')
    print(f'[rescore] using {os.path.basename(cache_path)} '
          f'(caches are never merged across code fingerprints)')
    for name, (covered, built) in sorted(cov.items()):
        if len(covered) != built:
            print(f'  [coverage] {name}: {len(covered)}/{built}')

    results = score_covered(loaded, leaves, samples, data, cov, opts)
    task_res = results.get(args.task, {})
    n_scored = int(task_res.get('sample_len') or 0)
    shown = {k: round(v, 6) for k, v in task_res.items()
             if isinstance(v, (int, float)) and not isinstance(v, bool)}
    print(f'[rescore] {args.task}: {n_scored}/{n_committed} docs scored -> {shown}')

    if args.verify:
        # The PUBLISHED row, so experiment rows are not candidates: verifying a
        # recovered cell against an experiment's number would compare against a value
        # nothing publishes (result_registry.experiment_of, the shared rule).
        prev = [r for r in result_registry.load_all_rows(REPO_ROOT)
                if r.get('ckpt_run') == args.run and str(r.get('step')) == str(args.step)
                and r.get('task') == args.task and not r.get('partial')
                and result_registry.is_production(r)
                and r.get('limit') == n_committed]
        if not prev:
            print('[verify] no complete published row for this cell')
            return 2
        ok = True
        for r in prev:
            # registry metric names drop a 'none' filter and spell others with '/'
            # (result_registry._flatten_task_metrics); bookkeeping metrics like
            # sample_len carry no filter at all.
            key = str(r.get('metric', '')).replace('/', ',')
            got = task_res.get(key, task_res.get(f'{key},none'))
            delta = None if got is None else abs(got - r['value'])
            flag = 'MATCH' if (delta is not None and delta < 1e-9) else 'DIFFER'
            ok = ok and flag == 'MATCH'
            print(f'[verify] {r["metric"]}: published {r["value"]!r} vs recomputed '
                  f'{got!r} -> {flag}')
        return 0 if ok else 1

    if args.record:
        base = existing_meta(args.run, args.step, args.task)
        meta = {
            'job_id': None,
            'model': base.get('model') or os.environ.get('MODEL_SIZE'),
            'moe_type': base.get('moe_type') or os.environ.get('MOE_TYPE'),
            'step': str(args.step),
            'num_fewshot': row.fewshot,
            'seq_len': row.seqlen,
            'padding_side': ident.get('pad'),
            'limit': n_committed,
            'consolidated_load': base.get('consolidated_load'),
            'ckpt_step_dir': base.get('ckpt_step_dir'),
            'ckpt_run': args.run,
            'ckpt_mtime': int(ident['mtime']) if ident.get('mtime') else None,
            'weight_source': ident.get('ws'),
            'response_cache': os.path.basename(cache_path),
            'results_path': None,
            'code_fp': ident.get('code_fp'),
            'git_sha': base.get('git_sha'),
            'git_dirty': base.get('git_dirty'),
            # `partial` is what render_branches keys the `*[n=N]` tag on, and what keeps a
            # recovered cell from ever outranking a finished measurement.
            'partial': True,
            'partial_source': 'rescore_partial.py',
            'partial_scored': n_scored,
            'partial_committed': n_committed,
        }
        path = result_registry.shard_path(REPO_ROOT, args.run)
        n = result_registry.record(path, meta, {args.task: task_res})
        print(f'[rescore] recorded {n} partial rows -> {path}')
    else:
        print('[rescore] dry run (pass --record to append registry rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
