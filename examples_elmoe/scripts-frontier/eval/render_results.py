#!/usr/bin/env python3
"""render_results.py — regenerate the human-readable result tables from the registry.

Turns the append-only registry (legacy file + per-run shards) into per-training-run pivots:

    eval/results/tables/<run_tag>.md      task rows x step columns (grouped by category)
    eval/results/tables/<run_tag>.csv     the same pivot, Excel/screenshot-friendly

Latest-row-wins per (task, metric, step, seq_len, fewshot): a re-run (e.g. after a bug fix,
like the bogus sciq=0.2 row from the random-weights job) is superseded by its newer row.
Rows carrying `partial` (provisional scores from score_partial.py) render as `value*` and
never shadow a full row.

Pure stdlib; safe from the template epilogue, the watcher, or by hand:
    python3 render_results.py [--run-tag 2026-08-09_5213105] [--quiet]
"""
import argparse
import json
import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(EVAL_DIR)))
TABLES_DIR = os.path.join(EVAL_DIR, 'results', 'tables')
sys.path.insert(0, os.path.join(REPO_ROOT, 'tasks', 'eval_harness'))
import result_registry  # noqa: E402

# Primary metric per task name; tasks not listed pick the first available metric from
# METRIC_PRIORITY (generation tasks emit exact_match/... variants, never 'acc').
PRIMARY_METRIC = {'wikitext': 'word_perplexity', 'lambada_openai': 'acc'}
METRIC_PRIORITY = ('acc', 'exact_match/strict-match', 'exact_match',
                   'prompt_level_strict_acc', 'exact_match/flexible-extract', 'f1')
# Higher is better except perplexities.
LOWER_IS_BETTER = {'word_perplexity', 'byte_perplexity', 'bits_per_byte', 'ppl'}


def run_tag_of(row):
    """Training-run bucket for a row. ckpt_run (Phase 2+) or a legacy fallback."""
    if row.get('ckpt_run'):
        return row['ckpt_run']
    # Legacy rows: infer from the consolidated filename or bucket by model.
    cons = row.get('consolidated_load') or ''
    base = os.path.basename(cons)
    if base.startswith('forge_10b'):
        return 'legacy-FORGE_10B'
    if base.startswith('deepseek_16b'):
        return 'legacy-DeepSeek_16B'
    return f'legacy-{row.get("model") or "unknown"}'


def _sanctioned_limits():
    """(task, fewshot, seqlen, limit) combos declared by table rows with a fixed-subset
    limit (instruct-short). Only these limited rows render -- as their own '[n=...]'
    series -- so ad-hoc --limit smoke rows stay out of the tables while the cadence
    suite is visible. A full row and its subset series never share a table line."""
    try:
        import task_table
        tasks, _ = task_table.load_custom()
        return {(r.task, r.fewshot, r.seqlen, r.limit)
                for r in dict(tasks).values() if getattr(r, 'limit', 0)}
    except Exception:
        return set()


_SANCTIONED = _sanctioned_limits()


def collect(rows):
    """-> {run_tag: {(task, metric): {step:int: (value, stderr, partial, ts)}}} latest-wins."""
    grid = {}
    for r in rows:
        task, metric = r.get('task'), r.get('metric')
        if not task or not metric:
            continue
        lim = r.get('limit')
        if lim is not None:
            key = (task, int(r.get('num_fewshot') or 0), r.get('seq_len'), lim)
            if key not in _SANCTIONED:
                continue                      # ad-hoc smoke row, not a cadence subset
            task = f'{task} [n={lim}]'
        if metric.endswith('_stderr'):
            continue
        # RULER reports one metric per context length and fills every OTHER length with
        # a -1.0 sentinel, keyed by lm-eval's hardcoded DEFAULT_SEQ_LENGTHS (["4096"],
        # common_utils.py:15-17,65). The grid key below has no seq_len, so an 8k job's
        # sentinel row (ruler/"4096" = -1.0) collided with and overwrote the 4k job's
        # real score at the same step -- every ruler cell in results/tables read
        # -1.0000 while the registry held 0.6982 (audit 2026-08-16). Keep only the
        # metric that names this row's own context length.
        if metric.isdigit() and r.get('seq_len') and int(metric) != int(r['seq_len']):
            continue
        try:
            step = int(str(r.get('step')))
        except (TypeError, ValueError):
            continue
        tag = run_tag_of(r)
        cell = grid.setdefault(tag, {}).setdefault((task, metric), {})
        old = cell.get(step)
        partial = bool(r.get('partial'))
        ts = r.get('ts') or ''
        # full rows beat partial rows; among same kind, latest ts wins
        if old is None or (old[2] and not partial) or (old[2] == partial and ts >= old[3]):
            cell[step] = (r.get('value'), r.get('stderr'), partial, ts)

    # Keep ONE metric per task: the declared primary, else the highest-priority metric
    # that actually has rows, else the alphabetically-first (so nothing vanishes from the
    # table just because its metric name is exotic).
    for tag, cells in grid.items():
        by_task = {}
        for (task, metric) in cells:
            by_task.setdefault(task, []).append(metric)
        keep = {}
        for task, metrics in by_task.items():
            want = PRIMARY_METRIC.get(task)
            if want is None or want not in metrics:
                want = next((m for m in METRIC_PRIORITY if m in metrics),
                            sorted(metrics)[0])
            keep[(task, want)] = cells[(task, want)]
        grid[tag] = keep
    return grid


def fmt(value, partial):
    if value is None:
        return ''
    s = f'{value:.4f}' if isinstance(value, float) else str(value)
    return s + ('*' if partial else '')


def render_run(tag, cells, quiet=False):
    steps = sorted({s for m in cells.values() for s in m})
    if not steps:
        return
    os.makedirs(TABLES_DIR, exist_ok=True)
    tasks_sorted = sorted(cells.keys())

    md = [f'# {tag} — eval results', '',
          '`value*` = provisional (partial coverage via score_partial.py); '
          'latest full run wins per cell. Regenerated by render_results.py — do not edit.',
          '', '| task (metric) | ' + ' | '.join(f'step {s}' for s in steps) + ' | Δ |',
          '|---' * (len(steps) + 2) + '|']
    csv = ['task,metric,' + ','.join(f'step_{s}' for s in steps) + ',delta']
    for (task, metric) in tasks_sorted:
        row = cells[(task, metric)]
        vals = [row.get(s) for s in steps]
        present = [(s, v) for s, v in zip(steps, vals) if v is not None and v[0] is not None]
        delta = ''
        if len(present) >= 2:
            d = present[-1][1][0] - present[0][1][0]
            if metric in LOWER_IS_BETTER:
                d = -d
            delta = f'{d:+.4f}'
        md.append(f'| {task} ({metric}) | '
                  + ' | '.join(fmt(v[0], v[2]) if v else '' for v in vals)
                  + f' | {delta} |')
        csv.append(f'{task},{metric},'
                   + ','.join(fmt(v[0], v[2]) if v else '' for v in vals)
                   + f',{delta}')

    for ext, content in (('md', '\n'.join(md) + '\n'), ('csv', '\n'.join(csv) + '\n')):
        path = os.path.join(TABLES_DIR, f'{tag}.{ext}')
        with open(path, 'w') as f:
            f.write(content)
        if not quiet:
            print(f'  wrote {path}')


def render_all_runs(grid, quiet=False):
    """The centralized record: every run, one file.

    tables/ALL_RUNS.md   one section per run (each identical to its per-run table)
    tables/ALL_RUNS.csv  long format -- run,task,metric,step,value,partial -- one row per
                         cell, so Excel/pandas can pivot any way (the per-run CSVs stay
                         wide/screenshot-shaped).
    """
    os.makedirs(TABLES_DIR, exist_ok=True)
    md = ['# ALL RUNS — centralized eval record', '',
          'Regenerated by render_results.py after every eval job; latest full row wins '
          'per cell; `value*` = provisional. One section per training run.', '']
    csv = ['run,task,metric,step,value,partial']
    for tag in sorted(grid):
        cells = grid[tag]
        steps = sorted({s for m in cells.values() for s in m})
        if not steps:
            continue
        md += [f'## {tag}', '',
               '| task (metric) | ' + ' | '.join(f'step {s}' for s in steps) + ' |',
               '|---' * (len(steps) + 1) + '|']
        for (task, metric) in sorted(cells):
            row = cells[(task, metric)]
            md.append(f'| {task} ({metric}) | '
                      + ' | '.join(fmt(row[s][0], row[s][2]) if s in row else ''
                                   for s in steps) + ' |')
            for s in steps:
                if s in row and row[s][0] is not None:
                    v = row[s]
                    csv.append(f'{tag},{task},{metric},{s},'
                               f'{v[0]:.6f},{1 if v[2] else 0}')
        md.append('')
    for ext, content in (('md', '\n'.join(md) + '\n'), ('csv', '\n'.join(csv) + '\n')):
        path = os.path.join(TABLES_DIR, f'ALL_RUNS.{ext}')
        with open(path, 'w') as f:
            f.write(content)
        if not quiet:
            print(f'  wrote {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run-tag', default=None, help='render only this run (default: all)')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    # Production rows only: a row from a run declared `experiment=<tag>` is a
    # measurement made to compare, never to publish, and this file's tables ARE the
    # published ones (result_registry.experiment_of -- the shared rule).
    grid = collect([r for r in result_registry.load_all_rows(REPO_ROOT)
                    if result_registry.is_production(r)])
    if args.run_tag:
        grid = {k: v for k, v in grid.items() if k == args.run_tag}
        if not grid:
            print(f'  no rows for run tag {args.run_tag!r}')
            return 1
    for tag, cells in sorted(grid.items()):
        render_run(tag, cells, args.quiet)
    if not args.run_tag:
        render_all_runs(grid, args.quiet)
    return 0


if __name__ == '__main__':
    sys.exit(main())
