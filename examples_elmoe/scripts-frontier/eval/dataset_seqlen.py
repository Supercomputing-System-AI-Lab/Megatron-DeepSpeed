#!/usr/bin/env python
# coding=utf-8
"""Measure the tokenized sequence-length distribution of any lm-eval task(s).

WHY
---
Our eval pads every request to `seq_length` (adaptive_seq_len is off), so seq_length drives both
correctness (too small => truncation => wrong scores) and cost (too large => wasted compute).
This tool reports, per task, the max / percentiles of the ACTUAL forwarded length so you can pick
the smallest safe seq_length. It respects the real prompt template AND few-shot count, so
`--num-fewshot 5` reflects the true 5-shot length (the reason MMLU-5shot is huge).

WHAT IT MEASURES
----------------
It builds the exact lm-eval requests for the task, then tokenizes with the DeepSeek tokenizer (the
one the model uses). Length reported per request:
  loglikelihood (MC/cloze): len(tok(context)) + len(tok(continuation))   <- what the model forwards
  loglikelihood_rolling   : len(tok(the document))                       <- rolling handles windows itself
  generate_until          : len(tok(context)) + max_gen_toks (prompt + room to generate)
The MAX over requests is the seq_length below which some request would be TRUNCATED.

USAGE (login node; needs `datasets` -> use the eval env python)
  <v6 python> dataset_seqlen.py mmlu --num-fewshot 5 \
      --tokenizer ../data/deepseek_tokenizer/tokenizer.json
  <v6 python> dataset_seqlen.py mmlu --num-fewshot 0
  <v6 python> dataset_seqlen.py arc_easy hellaswag ceval-valid   # any tasks

Offline-safe (datasets must be pre-cached; see prefetch_datasets.py).
"""

import argparse
import os
import sys

# Default tokenizer = the model's (DeepSeek). This file lives in .../examples_elmoe/scripts-frontier/eval,
# and the tokenizer is at .../examples_elmoe/data/deepseek_tokenizer/tokenizer.json -> two dirs up.
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOK = os.path.normpath(os.path.join(_HERE, os.pardir, os.pardir, 'data',
                                            'deepseek_tokenizer', 'tokenizer.json'))
os.environ.setdefault('HF_DATASETS_CACHE', os.path.join(_HERE, 'cache', 'hf_datasets_cache'))
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')


def flatten_tasks(task_dict, out):
    """get_task_dict returns {name-or-ConfigurableGroup: Task | nested-dict}. Recurse to leaf Tasks."""
    for _, v in task_dict.items():
        if isinstance(v, dict):
            flatten_tasks(v, out)
        else:
            out.append(v)          # a Task/ConfigurableTask
    return out


def pctile(sorted_vals, p):
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tasks', nargs='+', help='lm-eval task names (groups like mmlu expand to subtasks)')
    ap.add_argument('--num-fewshot', type=int, default=0)
    ap.add_argument('--tokenizer', default=DEFAULT_TOK, help='tokenizer.json (default: DeepSeek)')
    ap.add_argument('--limit', type=int, default=None,
                    help='cap docs per subtask (faster estimate; default: all)')
    ap.add_argument('--max-gen-toks', type=int, default=256,
                    help='room added to prompt length for generate_until tasks')
    ap.add_argument('--thresholds', default='512,1024,2048,4096',
                    help='report %% of requests exceeding each of these seq lengths')
    ap.add_argument('--include-path', default=None,
                    help='custom lm-eval task dir (TaskManager include_path), e.g. '
                         'custom_tasks/ for alpaca_eval_gen')
    ap.add_argument('--task-metadata-json-file', default=None,
                    help='JSON dict passed to TaskManager(metadata=...); RULER reads '
                         '{"tokenizer": <dir>, "max_seq_lengths": [N]} from it '
                         '(same plumbing as megatron_ds_eval/main.py)')
    args = ap.parse_args()

    # Raw `tokenizers` lib (not transformers' PreTrainedTokenizerFast) -- version-independent and
    # avoids the transformers-5.x fast-tokenizer instantiation quirk. Only .encode length is needed.
    from tokenizers import Tokenizer
    from lm_eval.tasks import TaskManager, get_task_dict
    tok = Tokenizer.from_file(args.tokenizer)
    n = lambda s: len(tok.encode(s).ids) if s else 0
    thresholds = [int(x) for x in args.thresholds.split(',')]

    tm_kwargs = {}
    if args.include_path:
        tm_kwargs['include_path'] = args.include_path
    if args.task_metadata_json_file:
        import json
        tm_kwargs['metadata'] = json.load(open(args.task_metadata_json_file))
    tm = TaskManager(**tm_kwargs)
    print(f"tokenizer: {args.tokenizer}")
    print(f"num_fewshot: {args.num_fewshot}   limit/subtask: {args.limit or 'all'}\n")

    grand = []
    for top in args.tasks:
        td = get_task_dict([top], tm)
        leaves = flatten_tasks(td, [])
        lens = []
        rtypes = set()
        for task in leaves:
            try:
                task.set_config(key='num_fewshot', value=args.num_fewshot)
            except Exception:
                pass
            task.build_all_requests(limit=args.limit, rank=0, world_size=1)
            for inst in task.instances:
                rt = inst.request_type
                rtypes.add(rt)
                a = inst.arguments
                if rt == 'loglikelihood':                 # (context, continuation)
                    L = n(a[0]) + n(a[1])
                elif rt == 'loglikelihood_rolling':        # (document,)
                    L = n(a[0])
                else:                                      # generate_until: (context, gen_kwargs)
                    L = n(a[0]) + args.max_gen_toks
                lens.append(L)
        if not lens:
            print(f"{top:22s} no requests built"); continue
        lens.sort()
        grand += lens
        over = {t: sum(1 for x in lens if x > t) for t in thresholds}
        print(f"== {top}  ({len(leaves)} subtask(s), {len(lens)} requests, type={'/'.join(sorted(rtypes))})")
        print(f"   max={lens[-1]}  p99={pctile(lens,99)}  p90={pctile(lens,90)}  "
              f"p50={pctile(lens,50)}  min={lens[0]}")
        print("   truncated if seq_length = " + "  ".join(
            f"{t}:{over[t]}({100*over[t]/len(lens):.1f}%)" for t in thresholds))
        # smallest power-of-two seq_length with zero truncation
        safe = 1
        while safe < lens[-1]:
            safe *= 2
        print(f"   -> smallest power-of-2 seq_length with NO truncation: {safe}\n")

    if len(args.tasks) > 1 and grand:
        grand.sort()
        print(f"== ALL TASKS COMBINED  ({len(grand)} requests)")
        print(f"   max={grand[-1]}  p99={pctile(grand,99)}  p90={pctile(grand,90)}  p50={pctile(grand,50)}")


if __name__ == '__main__':
    main()
