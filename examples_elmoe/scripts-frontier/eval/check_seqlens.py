#!/usr/bin/env python3
"""check_seqlens.py — measure task_table seqlens against real tokenized request lengths.

Wraps dataset_seqlen.py (the existing measurement engine) per table row, honoring the row's
fewshot and gen_toks, then compares:

    FAIL  measured_max > row.seqlen                (truncation — fix the table)
    WARN  row.seqlen  > 2 * measured_max           (waste; suppressed for seqlen_locked rows,
                                                    where the seqlen IS the eval semantics)

Prints the provenance string to paste into the row's seqlen_src. Run under the EVAL ENV
python (needs lm_eval + tokenizers; datasets must be prefetched):

    ROCM_PATH=/opt/rocm-default $ENV_PY check_seqlens.py --rows gsm8k,ifeval [--limit 500]
    ROCM_PATH=/opt/rocm-default $ENV_PY check_seqlens.py --suite core

Re-run when: adding a row, changing a row's fewshot/gen_toks, or upgrading lm-eval.
submit_eval's preflight refuses full runs of rows still marked seqlen_src='assumed'.
"""
import argparse
import datetime
import os
import re
import subprocess
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)
import task_table  # noqa: E402


def measure(row, limit):
    """Run dataset_seqlen.py for one row; return (max_len, p99) or None on failure."""
    cmd = [sys.executable, os.path.join(EVAL_DIR, 'dataset_seqlen.py'), row.task,
           '--num-fewshot', str(row.fewshot)]
    if row.gen_toks:
        cmd += ['--max-gen-toks', str(row.gen_toks)]
    # custom-task rows carry '--include-path <dir>' in extra_eval_flags; forward it
    m = re.search(r'--include-path\s+(\S+)', row.extra_eval_flags or '')
    if m:
        cmd += ['--include-path', m.group(1)]
    # ruler rows carry the tokenizer/length metadata file; forward it the same way
    m = re.search(r'--task-metadata-json-file\s+(\S+)', row.extra_eval_flags or '')
    if m:
        cmd += ['--task-metadata-json-file', m.group(1)]
    if limit:
        cmd += ['--limit', str(limit)]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = out.stdout.decode()
    # dataset_seqlen prints per-task lines including "max=<N>" and "p99=<N>"
    mx = re.findall(r'max[= ]+(\d+)', text)
    p99 = re.findall(r'p99[= ]+(\d+)', text)
    if out.returncode != 0 or not mx:
        sys.stderr.write(text[-2000:])
        return None
    return max(int(x) for x in mx), (max(int(x) for x in p99) if p99 else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--rows', help='comma-separated row ids')
    g.add_argument('--suite', help='measure every row of a suite expression')
    ap.add_argument('--limit', type=int, default=None,
                    help='sample docs per task (full dataset when omitted)')
    args = ap.parse_args()

    tasks, suites = task_table.load_custom()
    if args.rows:
        rows = [(rid.strip(), tasks[rid.strip()]) for rid in args.rows.split(',') if rid.strip()]
    else:
        rows = task_table.expand(args.suite, tasks, suites)

    today = datetime.date.today().isoformat()
    failures = 0
    for rid, row in rows:
        res = measure(row, args.limit)
        if res is None:
            print(f'  {rid:<16} MEASUREMENT FAILED (see stderr above)')
            failures += 1
            continue
        mx, p99 = res
        prov = f'measured:{today} max={mx}' + (f' p99={p99}' if p99 else '')
        if mx > row.seqlen and row.seqlen_locked:
            # Locked rows pin seqlen to the eval's semantics (model context, rolling
            # window): docs exceeding it are truncated BY DESIGN (e.g. LongBench on a
            # short-context model). Informational, not a failure.
            print(f'  {rid:<16} INFO  table={row.seqlen} [locked] measured_max={mx}: '
                  f'over-length docs truncate by design; seqlen_src="{prov}"')
        elif mx > row.seqlen:
            suggested = max(512, 1 << (mx - 1).bit_length())
            print(f'  {rid:<16} FAIL  table={row.seqlen} measured_max={mx} -> raise to '
                  f'{suggested}; set seqlen_src="{prov}"')
            failures += 1
        elif not row.seqlen_locked and row.seqlen > 2 * mx:
            print(f'  {rid:<16} WARN  table={row.seqlen} measured_max={mx} '
                  f'(>2x headroom); consider lowering; seqlen_src="{prov}"')
        else:
            lock = ' [locked]' if row.seqlen_locked else ''
            print(f'  {rid:<16} OK    table={row.seqlen}{lock} measured_max={mx}; '
                  f'seqlen_src="{prov}"')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
