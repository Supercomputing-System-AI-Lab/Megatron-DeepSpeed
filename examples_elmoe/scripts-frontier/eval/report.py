#!/usr/bin/env python
# coding=utf-8
"""Pivot the eval results registry into a readable table. Pure stdlib; runs anywhere.

The registry (eval/results/registry.jsonl) is append-only: every eval run adds one row per
(task, metric), stamped with the code fingerprint that produced it. This tool turns that stream
into answers.

EXAMPLES
    # everything, latest value per (task, metric, model, step)
    python eval/report.py

    # one model, markdown (paste into a writeup)
    python eval/report.py --model DeepSeek_16B --format md

    # did my code change move the numbers?  <-- the regression check
    python eval/report.py --task mmlu --compare-code

    # full history for one task (every run, not just the latest)
    python eval/report.py --task hellaswag --all

NOTE ON "latest": rows are appended chronologically, so the last row for a given key wins. Use
--all to see superseded runs, or --compare-code to see them side by side by code version.

NOTE ON EXPERIMENTS: a row written by a run declared `experiment=<tag>` is a measurement made to
compare, not to publish, and is EXCLUDED from every pivot here -- otherwise, being the latest row
for its key, it would simply become the number (openbookqa/sft-16k15b, 2026-08-28: 0.336 -> 0.352
-> 0.356 from two shape experiments). `--experiment <tag>` pivots that experiment's rows instead
of production's; `--all` shows both, experiment rows marked. The rule is
result_registry.experiment_of() -- shared with the done-check and the dashboard.
"""

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, os.pardir, 'tasks', 'eval_harness'))
try:
    import result_registry
except ImportError:
    sys.exit("could not import result_registry (expected at tasks/eval_harness/result_registry.py)")

DEFAULT_REGISTRY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'results', 'registry.jsonl')

# Metrics worth showing first when a task reports several.
PREFERRED = ('acc_norm', 'acc', 'word_perplexity', 'byte_perplexity', 'bits_per_byte',
             'exact_match', 'f1')


def fmt(v):
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.4f}' if abs(v) < 1000 else f'{v:.1f}'
    return str(v)


def render_table(headers, rows, style='table'):
    if not rows:
        return '(no rows)'
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    if style == 'csv':
        return '\n'.join([','.join(map(str, headers))] + [','.join(map(str, r)) for r in rows])
    sep = ' | '
    out = [sep.join(str(h).ljust(widths[i]) for i, h in enumerate(headers))]
    if style == 'md':
        out[0] = '| ' + out[0] + ' |'
        out.append('| ' + sep.join('-' * widths[i] for i in range(len(headers))) + ' |')
        for r in rows:
            out.append('| ' + sep.join(str(c).ljust(widths[i]) for i, c in enumerate(r)) + ' |')
    else:
        out.append('-+-'.join('-' * w for w in widths))
        for r in rows:
            out.append(sep.join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
    return '\n'.join(out)


def _report_exclusions(excluded, wanted_exp):
    """Say which rows this pivot refused to consider, and how to see them.

    Silence is what made the openbookqa overwrite invisible for four days: the number
    changed and nothing said a different KIND of run had produced it.
    """
    tags = sorted(t for t in excluded if t != 'production')
    if tags:
        n = sum(excluded[t] for t in tags)
        print(f'({n} experiment row(s) excluded -- {", ".join(f"{t}:{excluded[t]}" for t in tags)}'
              f'; they can never define a published value. --experiment <tag> pivots one.)')
    if wanted_exp and excluded.get('production'):
        print(f'({excluded["production"]} production row(s) excluded: showing experiment '
              f'{wanted_exp!r} only.)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--registry', default=DEFAULT_REGISTRY)
    ap.add_argument('--model', default=None, help='filter by model (substring match)')
    ap.add_argument('--task', default=None, help='filter by task (substring match)')
    ap.add_argument('--metric', default=None, help='filter by metric name (exact)')
    ap.add_argument('--all', action='store_true', help='show every run, not just the latest per key')
    ap.add_argument('--experiment', default=None, metavar='TAG',
                    help='pivot the rows of experiment TAG instead of the production '
                         'rows (default: production only -- experiment rows can never '
                         'define a published value)')
    ap.add_argument('--compare-code', action='store_true',
                    help='pivot the same (model,step,task,metric) across code fingerprints')
    ap.add_argument('--format', default='table', choices=['table', 'md', 'csv'])
    args = ap.parse_args()

    # Union view: the legacy file plus the per-run shards (results/registry/*.jsonl).
    # --registry still narrows to one explicit file when given a non-default path.
    if args.registry == DEFAULT_REGISTRY:
        repo_root = result_registry.infer_repo_root(result_registry.__file__)
        rows = result_registry.load_all_rows(repo_root)
    else:
        rows = result_registry.load_rows(args.registry)
    if not rows:
        sys.exit(f'no rows in {args.registry} (has any eval run completed since Step 2 landed?)')

    def keep(r):
        if args.model and args.model.lower() not in str(r.get('model', '')).lower():
            return False
        if args.task and args.task.lower() not in str(r.get('task', '')).lower():
            return False
        if args.metric and args.metric != r.get('metric'):
            return False
        return True

    rows = [r for r in rows if keep(r)]
    if not rows:
        sys.exit('no rows matched the filters')

    # THE experiment rule, applied once for every pivot below. A row from a run declared
    # `experiment=<tag>` is not a candidate for the published number; --experiment asks
    # for one experiment's rows instead, and --all shows the lot with the tag visible.
    wanted_exp = args.experiment or None
    excluded = collections.Counter()
    if not args.all:
        kept = []
        for r in rows:
            exp = result_registry.experiment_of(r)
            if exp == wanted_exp:
                kept.append(r)
            else:
                excluded[exp or 'production'] += 1
        rows = kept
        if not rows:
            sys.exit(f'no rows for experiment {wanted_exp!r}' if wanted_exp else
                     'no production rows matched the filters (every matching row came '
                     'from an experiment; use --experiment <tag> or --all)')
    show_exp = args.all or wanted_exp is not None

    # Partial rows -- rescore_partial.py's recovered cells and slice jobs' rows, both
    # marked with the `partial` field -- are NOT the benchmark's number: a slice scored
    # part of the document set at the full row's identity, so letting it win the
    # "latest per key" pivot prints a partial score as the benchmark (2026-08-27 slice
    # wave). Excluded from both pivots; --all shows them, asterisked.
    n_partial = sum(1 for r in rows if r.get('partial'))

    # ---- compare-code: same measurement, one column per code fingerprint -------------------
    if args.compare_code:
        fps, cells = [], {}
        for r in rows:
            if r.get('partial'):
                continue
            fp = r.get('code_fp') or '?'
            if fp not in fps:
                fps.append(fp)
            key = (r.get('model'), r.get('step'), r.get('task'), r.get('metric'),
                   r.get('num_fewshot'))
            cells[(key, fp)] = r.get('value')          # later row wins
        headers = ['model', 'step', 'task', 'metric', 'fs'] + [f'code:{f}' for f in fps]
        out = []
        for key in sorted({k for k, _ in cells}, key=lambda k: tuple(str(x) for x in k)):
            vals = [cells.get((key, f)) for f in fps]
            row = [fmt(x) for x in key] + [fmt(v) for v in vals]
            known = [v for v in vals if v is not None]
            if len(known) >= 2:                        # flag movement between code versions
                delta = max(known) - min(known)
                row[-1] = f'{row[-1]}  (Δ={delta:+.4f})'
            out.append(row)
        print(render_table(headers, out, args.format))
        print(f'\n{len(fps)} code fingerprint(s): {", ".join(fps)}')
        print('Δ = spread across code versions for that measurement; non-zero means a code change '
              'moved the number.')
        _report_exclusions(excluded, wanted_exp)
        return

    # ---- default: latest value per (model, step, task, metric, fewshot) --------------------
    latest = {}
    for r in rows:
        if r.get('partial') and not args.all:
            continue                                   # a partial row never defines the number
        key = (r.get('model'), r.get('step'), r.get('task'), r.get('metric'), r.get('num_fewshot'))
        if args.all:
            latest[(key, r.get('ts'), r.get('job_id'))] = r
        else:
            latest[key] = r                            # append-only => last wins

    ordered = sorted(latest.values(), key=lambda r: (
        str(r.get('model')), str(r.get('step')), str(r.get('task')),
        PREFERRED.index(r.get('metric')) if r.get('metric') in PREFERRED else 99,
        str(r.get('metric'))))

    headers = ['model', 'step', 'task', 'metric', 'value', 'stderr', 'fs', 'seq', 'code_fp', 'job']
    table = [[fmt(r.get('model')), fmt(r.get('step')), fmt(r.get('task')), fmt(r.get('metric')),
              fmt(r.get('value')) + ('*' if r.get('partial') else ''), fmt(r.get('stderr')),
              fmt(r.get('num_fewshot')), fmt(r.get('seq_len')), fmt(r.get('code_fp')),
              fmt(r.get('job_id'))]
             for r in ordered]
    if show_exp:
        # Only when experiment rows can actually be in the table: a column that always
        # reads '-' teaches nobody anything.
        headers.append('experiment')
        for cells, r in zip(table, ordered):
            cells.append(result_registry.experiment_of(r) or '-')
    print(render_table(headers, table, args.format))
    print(f'\n{len(table)} measurement(s) from {args.registry}')
    if not args.all:
        print('(latest per model/step/task/metric/fewshot; use --all for full history)')
        if n_partial:
            print(f'({n_partial} partial row(s) -- slice jobs / rescored cells -- excluded; '
                  f'--all shows them as value*)')
        _report_exclusions(excluded, wanted_exp)
    elif n_partial:
        print('(* = partial row: a slice or rescored cell, not the full benchmark)')


if __name__ == '__main__':
    main()
