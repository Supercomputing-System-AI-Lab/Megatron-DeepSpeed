#!/usr/bin/env python
# coding=utf-8
"""Warm the HF datasets cache for lm-eval tasks, from a LOGIN NODE, before submitting jobs.

WHY
---
Compute-node jobs download their datasets on first use. When many eval jobs start at once (running
in parallel, or after `-d singleton` is disabled) they hammer the HF Hub unauthenticated and get
HTTP 429 Too Many Requests -- which killed the mmlu / hellaswag / ceval / truthfulqa jobs in the
2026-07-24 sweep. Downloading once, serially, up front removes that entire failure mode: the jobs
then hit a warm local cache and never touch the network.

WHY VIA lm-eval RATHER THAN load_dataset() DIRECTLY
---------------------------------------------------
A task name does not map trivially to a (repo, config) pair. `ceval-valid` expands to 52 per-subject
configs of `ceval/ceval-exam`; `mmlu` expands to 57. Hand-writing those is error-prone (a plain
`load_dataset('ceval/ceval-exam')` fails with "Config name is missing"). Asking lm-eval to build the
task dict makes IT resolve every dataset/config exactly as the real eval will, so what we cache is
by construction what the job needs.

USAGE (login node, where the proxy works)
    export HF_DATASETS_CACHE=$PWD/cache/hf_datasets_cache
    export http_proxy=http://proxy.ccs.ornl.gov:3128 https_proxy=$http_proxy
    export HF_TOKEN=...            # optional; raises the Hub rate limit substantially
    python prefetch_datasets.py ceval-valid truthfulqa_mc1 truthfulqa_mc2
    python prefetch_datasets.py --from-eval-map      # every task named in autorun_eval.sh

Exit code is 0 only if every requested task cached successfully.
"""

import argparse
import os
import re
import sys
import time

DEFAULT_EVAL_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autorun_eval.sh')
DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'cache', 'hf_datasets_cache')


def tasks_from_eval_map(path):
    """Extract task names from the UNCOMMENTED EVAL_MAP entries of autorun_eval.sh.

    Value layout: <TASK_LIST>:<LIMIT>:<MODEL>:<MOE>:<MBS>:<TIME>[:pad[:seq[:fewshot]]]
    so the task list is field 1 of the quoted value.
    """
    tasks = []
    try:
        with open(path) as f:
            for line in f:
                s = line.strip()
                if s.startswith('#') or not s.startswith('EVAL_MAP['):
                    continue
                m = re.search(r'=\s*"([^"]+)"', s)
                if not m:
                    continue
                for t in m.group(1).split(':')[0].split(','):
                    t = t.strip()
                    if t and t not in tasks:
                        tasks.append(t)
    except OSError as e:
        print(f'could not read {path}: {e}', file=sys.stderr)
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tasks', nargs='*', help='lm-eval task names')
    ap.add_argument('--from-table', metavar='SUITE', default=None,
                    help='prefetch every task of a task_table suite expression '
                         '(e.g. core, instruct+longctx)')
    ap.add_argument('--from-eval-map', action='store_true',
                    help=f'take every task named in {DEFAULT_EVAL_MAP}')
    ap.add_argument('--eval-map', default=DEFAULT_EVAL_MAP)
    ap.add_argument('--sleep', type=float, default=5.0,
                    help='seconds between tasks; spacing avoids re-triggering the 429 (default 5)')
    args = ap.parse_args()
    if args.from_table:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import task_table
        args.tasks += [r.task for _, r in task_table.expand(args.from_table)]

    tasks = list(args.tasks)
    if args.from_eval_map:
        tasks += [t for t in tasks_from_eval_map(args.eval_map) if t not in tasks]
    if not tasks:
        ap.error('no tasks given (pass names, or --from-eval-map)')

    # Default the cache to the eval tree so compute jobs (which export the same var) find it.
    os.environ.setdefault('HF_DATASETS_CACHE', DEFAULT_CACHE)
    print(f'HF_DATASETS_CACHE = {os.environ["HF_DATASETS_CACHE"]}')
    print(f'HF_TOKEN set      = {bool(os.environ.get("HF_TOKEN"))}'
          f'{"" if os.environ.get("HF_TOKEN") else "   (unauthenticated -> lower rate limit)"}')
    print(f'tasks ({len(tasks)}): {", ".join(tasks)}\n')

    from lm_eval.tasks import TaskManager
    tm = TaskManager()

    def _load(task_name):
        """Resolve+download one task. Prefers TaskManager.load(); get_task_dict is deprecated in
        lm_eval 0.4.13 but kept as a fallback so this tool also works on older installs."""
        if hasattr(tm, 'load'):
            return tm.load([task_name])
        from lm_eval.tasks import get_task_dict
        return get_task_dict([task_name], tm)

    ok, failed = [], []
    for i, task in enumerate(tasks, 1):
        print(f'[{i}/{len(tasks)}] {task} ... ', end='', flush=True)
        t0 = time.time()
        try:
            # Building the task object(s) makes lm-eval resolve AND download every underlying
            # dataset/config for the task (including all subtasks of a group like mmlu/ceval).
            td = _load(task)
            n_sub = len(td) if isinstance(td, dict) else 1
            print(f'OK  ({n_sub} task obj(s), {time.time() - t0:.1f}s)')
            ok.append(task)
        except Exception as e:
            msg = str(e).replace('\n', ' ')[:200]
            print(f'FAILED  ({type(e).__name__}: {msg})')
            failed.append(task)
        if i < len(tasks) and args.sleep:
            time.sleep(args.sleep)

    print(f'\ncached OK ({len(ok)}): {", ".join(ok) if ok else "-"}')
    if failed:
        print(f'FAILED    ({len(failed)}): {", ".join(failed)}')
        print('\nA 429/rate-limit failure is usually transient: set HF_TOKEN, or wait and re-run '
              '(cached tasks are skipped quickly). A "Dataset scripts are no longer supported" '
              'failure is NOT transient -- that dataset needs a parquet mirror or an older '
              '`datasets` version.')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
