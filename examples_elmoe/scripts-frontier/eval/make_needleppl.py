#!/usr/bin/env python
"""Generate the committed needle-PPL monitor datasets (scoring-only, no decode).

Rationale (0822_2026_RIGOR_PLAN.html §3): Hunyuan LCCP (arXiv:2604.02650) found
base-model needle-PPL predicts post-SFT long-context performance better than NIAH
accuracy (-0.821 vs 0.749), and NIAH saturates early (HELMET, MiniMax-01).
This is a loglikelihood task: logprob of a 7-digit needle answer whose needle is
embedded at a controlled depth in a synthetic noise haystack. Prefill-only ->
immune to the ragged-decode hang, minutes per checkpoint.

Fully synthetic (RULER-noise-style), zero training-data contamination.
Committed artifact: data/needleppl_{4k,8k,16k,32k}.jsonl + manifest (seed, sha).
"""
import hashlib, json, os, random

SEED = 20260814
N_PER_LEN = 50           # 10 depths x 5 keys, stratified
LENGTHS = {'4k': 4096, '8k': 8192, '16k': 16384, '32k': 32768, '64k': 65536, '128k': 131072}   # 64k 2026-08-30; 128k 2026-08-30 PM (trained length)
MARGIN = 32              # tokens reserved under the job seqlen
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'custom_tasks', 'needleppl', 'data')
TOKJSON = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                       'data', 'deepseek_tokenizer', 'tokenizer.json')

NOISE = ["The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.",
         "The birds sing in the morning. The river flows to the sea. The wind moves the trees.",
         "A quiet road runs past the old mill. The clock on the wall keeps ticking. Days pass slowly here.",
         "The market opens at dawn. People come and go. The bell above the door rings again and again."]
ADJ = ['bright', 'silent', 'ancient', 'gentle', 'rapid', 'hollow', 'amber', 'crimson', 'velvet', 'iron']
NOUN = ['harbor', 'lantern', 'meadow', 'compass', 'anvil', 'orchard', 'glacier', 'quarry', 'beacon', 'archive']

from tokenizers import Tokenizer
tok = Tokenizer.from_file(TOKJSON)
def nt(s): return len(tok.encode(s, add_special_tokens=False).ids)

def build(L, rng):
    docs = []
    unit_toks = [nt(s + ' ') for s in NOISE]
    for i in range(N_PER_LEN):
        depth = (i % 10) / 10 + 0.05
        key = f"{rng.choice(ADJ)}-{rng.choice(NOUN)}"
        number = str(rng.randint(1000000, 9999999))
        needle = f" The special magic number for {key} is {number}."
        question = f"\nQuestion: What is the special magic number for {key}?\nThe special magic number for {key} is"
        budget = L - MARGIN - nt(needle) - nt(question) - 8
        units, toks, k = [], 0, i % len(NOISE)
        while toks + unit_toks[(k + len(units)) % len(NOISE)] <= budget:
            units.append(NOISE[(k + len(units)) % len(NOISE)])
            toks += unit_toks[(k + len(units) - 1) % len(NOISE)]
        pos = int(depth * len(units))
        context = ' '.join(units[:pos]) + needle + ' ' + ' '.join(units[pos:]) + question
        total = nt(context) + nt(' ' + number)
        assert total <= L - 8, (L, total)
        docs.append({'context': context, 'answer': ' ' + number,
                     'depth': round(depth, 2), 'key': key, 'n_tokens': total})
    return docs

os.makedirs(OUT, exist_ok=True)
manifest = {'seed': SEED, 'n_per_len': N_PER_LEN, 'margin': MARGIN, 'files': {}}
for name, L in LENGTHS.items():
    rng = random.Random(SEED + L)
    docs = build(L, rng)
    p = os.path.join(OUT, f'needleppl_{name}.jsonl')
    body = '\n'.join(json.dumps(d, ensure_ascii=False) for d in docs) + '\n'
    open(p, 'w').write(body)
    manifest['files'][name] = {'n': len(docs), 'sha256': hashlib.sha256(body.encode()).hexdigest()[:16],
                               'tok_range': [min(d['n_tokens'] for d in docs), max(d['n_tokens'] for d in docs)]}
    print(name, manifest['files'][name])
json.dump(manifest, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=2)
print('manifest written')
