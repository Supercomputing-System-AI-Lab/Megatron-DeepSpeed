"""OpenAI MRCR (multi-round co-reference resolution; huggingface openai/mrcr, 2026-04) as an lm-eval task for this harness.

The data: custom_tasks/mrcr/data/mrcr_{2,4,8}needle.jsonl, written by assistant_deep_dive/2026-09-15-eval_framework/bin/mrcr_build.py
(the conversation rendered through the model's own chat template, the V3 token count, the card's bin). The task keeps the
rows that fit the run's sequence length: metadata max_seq_lengths=[L] (the same metadata file the RULER rows use), a row is kept when
n_tokens + the generation budget <= L. The metric is the card's: 0 unless the reply starts with the random prefix, else the
difflib SequenceMatcher ratio of the stripped reply against the stripped answer; reported overall and per bin (a bin with no rows
in the run reports -1 and is left out of the mean, RULER's convention).
"""
import json
import os
from difflib import SequenceMatcher

import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
GEN_BUDGET = 1200
BINS = ['4k-8k', '8k-16k', '16k-32k', '32k-64k', '64k-128k', '128k-256k', '256k-512k', '512k-1m']


def _load(k, **kwargs):
    seqs = list(kwargs.get('max_seq_lengths', [1048576]))
    limit = max(seqs)
    fp = os.path.join(HERE, 'data', f'mrcr_{k}needle.jsonl')
    rows = [json.loads(line) for line in open(fp)]
    keep = [r for r in rows if r['n_tokens'] + GEN_BUDGET <= limit]
    return {'test': datasets.Dataset.from_list(keep, split=datasets.Split.TEST)}


def load_2needle(**kwargs):
    return _load(2, **kwargs)


def load_4needle(**kwargs):
    return _load(4, **kwargs)


def load_8needle(**kwargs):
    return _load(8, **kwargs)


def grade(response, answer, prefix):
    response = response.strip()
    if not response.startswith(prefix):
        return 0.0
    response = response[len(prefix):].strip()
    answer = answer[len(prefix):].strip() if answer.startswith(prefix) else answer.strip()
    return float(SequenceMatcher(None, response, answer).ratio())


def process_results(doc, results):
    r = grade(results[0], doc['answer'], doc['prefix'])
    out = {'ratio': r}
    for b in BINS:
        out[f'ratio_{b}'] = r if doc['bin'] == b else -1.0
    return out


def aggregate_bin(values):
    """the mean over the rows of the bin; -1 when the run had none (RULER's convention for a length it did not run)"""
    v = [x for x in values if x >= 0]
    return sum(v) / len(v) if v else -1.0
