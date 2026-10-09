#!/usr/bin/env python3
"""export_alpacaeval.py — response cache -> AlpacaEval 2 judge-input file. Login node only.

The `chat` suite (alpaca_gen row) generates answers to the 805 official AlpacaEval 2
instructions; this tool maps each cached generation back to its instruction and writes
the outputs JSON the official judge consumes. Judging needs an API judge model, so it
runs wherever you have credentials (laptop):

    pip install alpaca-eval
    export OPENAI_API_KEY=...
    alpaca_eval --model_outputs outputs_<gen>.json     # reports LC win-rate vs GPT-4 baseline

Usage (eval env python; same interpreter as score_partial):
    ROCM_PATH=/opt/rocm-default $ENV_PY export_alpacaeval.py \
        --cache cache/responses/<run>__<step>_..._fs0.jsonl \
        [--generator mymodel_step160] [--out outputs.json]
"""
import argparse
import json
import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EVAL_DIR, '..', '..', '..',
                                                'tasks', 'eval_harness')))
from megatron_ds_eval.cache import request_key  # noqa: E402

ALPACA_JSON = os.path.join(EVAL_DIR, 'cache', 'hf_datasets_cache',
                           'alpaca_eval', 'alpaca_eval.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cache', required=True, help='response-cache .jsonl of the chat run')
    ap.add_argument('--generator', default=None,
                    help='model label in the output file (default: cache basename stem)')
    ap.add_argument('--out', default=None,
                    help='output path (default: alpacaeval_outputs_<generator>.json '
                         'next to the cache)')
    args = ap.parse_args()

    data = {}
    with open(args.cache) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                      # torn tail line
            if 'k' in rec:
                data[rec['k']] = rec['v']

    from lm_eval.tasks import TaskManager
    tm = TaskManager(include_path=os.path.join(EVAL_DIR, 'custom_tasks'))
    task = tm.load_task_or_group('alpaca_eval_gen')['alpaca_eval_gen']
    task.build_all_requests(limit=None, rank=0, world_size=1)

    docs = list(task.test_docs())
    gen = args.generator or os.path.basename(args.cache).split('_m')[0]
    rows, missing = [], 0
    for inst in task.instances:
        key = request_key('generate_until', tuple(inst.args))
        out = data.get(key)
        if out is None:
            missing += 1
            continue
        doc = docs[inst.doc_id]
        rows.append({'instruction': doc['instruction'],
                     'output': out.strip(),
                     'generator': gen,
                     'dataset': doc.get('dataset', 'alpaca_eval')})

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.cache)), f'alpacaeval_outputs_{gen}.json')
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print(f'[export-alpacaeval] {len(rows)}/805 answers -> {out_path}'
          + (f'  ({missing} not yet cached — partial export, rerun after the chain '
             f'finishes)' if missing else ''))
    return 0 if rows else 1


if __name__ == '__main__':
    sys.exit(main())
