"""The needle grid as a generation task: data from assistant_deep_dive/2026-09-15-eval_framework/bin/needlegrid_build.py
(custom_tasks/needlegrid/data/needlegrid_<len>.jsonl); the length comes from the metadata's max_seq_lengths (the RULER metadata
files); the metric is the exact match of the 7-digit number in the first line of the reply, reported overall and per depth."""
import json
import os
import re

import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
DEPTHS = ['0.05', '0.15', '0.25', '0.35', '0.45', '0.55', '0.65', '0.75', '0.85', '0.95']


def load(**kwargs):
    L = max(kwargs.get('max_seq_lengths', [131072]))
    fp = os.path.join(HERE, 'data', f'needlegrid_{L}.jsonl')
    rows = [json.loads(line) for line in open(fp)]
    return {'test': datasets.Dataset.from_list(rows, split=datasets.Split.TEST)}


def process_results(doc, results):
    m = re.search(r'\d{7}', results[0])
    hit = 1.0 if m and m.group(0) == doc['answer'] else 0.0
    out = {'exact': hit}
    for d in DEPTHS:
        out[f'exact_d{d}'] = hit if f"{doc['depth']:.2f}" == d else -1.0
    return out


def aggregate_depth(values):
    v = [x for x in values if x >= 0]
    return sum(v) / len(v) if v else -1.0
