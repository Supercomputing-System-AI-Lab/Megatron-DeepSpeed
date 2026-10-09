#!/usr/bin/env python3
"""score_partial.py — provisional scores from a response cache, no GPU, no model.

The Phase-2 cache stores every per-request result, so a login-node process can answer
"how is that 11-hour task doing?" mid-chain: build the task's requests through lm_eval,
find the documents whose requests are ALL cached, and run the metric pipeline over exactly
those docs with a cache-only LM stub. Output: a provisional score + coverage, optionally
appended to the registry marked `partial` (never satisfies submit_eval's DONE check; the
tables render it as `value*` until a full run supersedes it).

Run under the EVAL ENV python (lm_eval import needs it; no ROCm device is touched):

    ROCM_PATH=/opt/rocm-default $ENV_PY score_partial.py \
        --cache cache/responses/<file>.jsonl --task sciq [--fewshot 0] [--record]

Two passes, both model-free: pass 1 probes coverage per doc_id; pass 2 re-evaluates with
`samples={task: covered_docs}` so lm-eval's own aggregation produces the score. Group tasks
(mmlu, ruler) are refused in v1 — score their subtasks individually.
"""
import argparse
import json
import os
import re
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(EVAL_DIR)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'tasks', 'eval_harness'))

from megatron_ds_eval.cache import request_key  # noqa: E402  (hash only; no model deps used)
import result_registry                          # noqa: E402


def load_cache(path):
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


from lm_eval.api.model import LM as _LM   # after sys.path setup; needs the eval env


class CacheOnlyLM(_LM):
    """lm_eval LM implementation that answers exclusively from a response-cache dict.

    probe mode: records (doc_id, hit) and returns dummies for misses (metrics discarded).
    strict mode: raises on any miss (used for the scoring pass over covered docs only).
    Subclasses the real lm_eval LM (simple_evaluate type-checks it); the base __init__
    provides rank=0/world_size=1/cache_hook.
    """

    def __init__(self, data, probe):
        super().__init__()
        self.data = data
        self.probe = probe
        self.doc_hits = {}          # doc_id -> all-requests-hit so far

    def _serve(self, method, requests, dummy):
        out = []
        for req in requests:
            key = request_key(method, tuple(req.args))
            hit = key in self.data
            doc = getattr(req, 'doc_id', None)
            self.doc_hits[doc] = self.doc_hits.get(doc, True) and hit
            if hit:
                out.append(self.data[key])
            elif self.probe:
                out.append(dummy)
            else:
                raise KeyError(f'MISS in strict mode: doc {doc}, {method}')
        return out

    def loglikelihood(self, requests):
        return [tuple(v) for v in self._serve('loglikelihood', requests, [0.0, False])]

    def loglikelihood_rolling(self, requests):
        return [float(v) for v in self._serve('loglikelihood_rolling', requests, 0.0)]

    def generate_until(self, requests):
        return [str(v) for v in self._serve('generate_until', requests, '')]


def parse_cache_name(path):
    """<run>__global_step<N>_m..._<ws>_pad<side>_seq<S>_fs<F>.jsonl -> identity dict."""
    base = os.path.basename(path)
    m = re.match(r'(?P<run>.+)__global_step(?P<step>\d+)_m\d+_code[0-9a-f]+_'
                 r'(?P<ws>\w+)_pad(?P<pad>\w+)_seq(?P<seq>\d+)_fs(?P<fs>\d+)'
                 r'(?P<mode>_nokv)?\.jsonl', base)
    return m.groupdict() if m else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cache', required=True, help='response-cache .jsonl path')
    ap.add_argument('--task', required=True, help='one PLAIN lm-eval task (no groups in v1)')
    ap.add_argument('--fewshot', type=int, default=None,
                    help='default: fs from the cache filename')
    ap.add_argument('--record', action='store_true',
                    help='append a `partial`-marked registry row')
    ap.add_argument('--model', default='DeepSeek_16B')
    ap.add_argument('--moe', default='X-MOE')
    args = ap.parse_args()

    ident = parse_cache_name(args.cache)
    fewshot = args.fewshot if args.fewshot is not None else int(ident.get('fs', 0))
    data = load_cache(args.cache)
    print(f'[score-partial] {len(data)} cached responses from {os.path.basename(args.cache)}')

    from lm_eval import simple_evaluate

    # Pass 1 — coverage probe (dummy answers; metrics discarded).
    probe = CacheOnlyLM(data, probe=True)
    simple_evaluate(model=probe, tasks=[args.task], num_fewshot=fewshot,
                    bootstrap_iters=0, verbosity='ERROR')
    covered = sorted(d for d, ok in probe.doc_hits.items() if ok and d is not None)
    total = len(probe.doc_hits)
    if not covered:
        print(f'[score-partial] 0/{total} docs fully covered — nothing to score yet')
        return 1
    coverage = len(covered) / max(total, 1)
    print(f'[score-partial] coverage: {len(covered)}/{total} docs ({coverage:.1%})')

    # Pass 2 — real aggregation over covered docs only.
    strict = CacheOnlyLM(data, probe=False)
    res = simple_evaluate(model=strict, tasks=[args.task], num_fewshot=fewshot,
                          samples={args.task: covered}, bootstrap_iters=0,
                          verbosity='ERROR')
    task_res = res['results'].get(args.task, {})
    shown = {k: v for k, v in task_res.items() if isinstance(v, (int, float))}
    print(f'[score-partial] PROVISIONAL {args.task}: {shown}')

    if args.record:
        meta = {'job_id': None, 'model': args.model, 'moe_type': args.moe,
                'step': ident.get('step'), 'num_fewshot': fewshot,
                'seq_len': int(ident['seq']) if ident.get('seq') else None,
                'padding_side': ident.get('pad'), 'limit': None,
                'ckpt_run': ident.get('run'), 'weight_source': ident.get('ws'),
                'partial': round(coverage, 4), 'response_cache': os.path.basename(args.cache),
                'results_path': None, 'code_fp': None, 'git_sha': None, 'git_dirty': None}
        path = result_registry.shard_path(REPO_ROOT, ident.get('run'))
        n = result_registry.record(path, meta, {args.task: task_res})
        print(f'[score-partial] recorded {n} partial rows -> {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
