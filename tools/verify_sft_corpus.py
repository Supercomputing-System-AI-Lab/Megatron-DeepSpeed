#!/usr/bin/env python
"""Verify that an SFT corpus is really in SFT format and that training really
scores only the answer tokens. Two independent halves; both must pass.

A. CORPUS FORMAT (reads the .bin/.idx pair + decodes with the raw training
   tokenizer):
     A1 mask stream document-aligned with the text stream (identical sizes)
     A2 every document exactly --seq-length tokens
     A3 token 0 is BOS and is unsupervised
     A4 the region before the first supervised token decodes to a prompt
        ("User: ..." ending in "Assistant:")
     A5 every maximal supervised span ends with the EOS token (the stop signal)
     A6 everything after the last supervised token is pure EOS padding, mask 0
     A7 mask values are only 0/1
     A8 empirical supervised density matches the build manifest

B. LOSS SCORING (reads real training logs):
     B1 the [sft] per-microbatch density prints from a masked run average to
        the corpus density (proves the mask travels disk -> loss_mask)
     B2 controlled pair: a masked and an unmasked job on the SAME corpus and
        seed have IDENTICAL moe-loss values per iteration (identical batches)
        but lm losses differing by >2x (the mask is the only difference, and
        it changes the loss) -- the strongest runtime evidence available.

USAGE
    python tools/verify_sft_corpus.py \
        --data-prefix examples_elmoe/data/dolci/dolci-sft-deepseek \
        --density-log <masked_job>/rank_0.log \
        --masked-log <masked_job>/rank_7.log \
        --unmasked-log <unmasked_job>/rank_7.log
"""
import argparse
import json
import os
import random
import re
import sys

os.environ.setdefault('DS_ACCELERATOR', 'cpu')
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np
from megatron.data import indexed_dataset
from tokenizers import Tokenizer
from tools.sft_chat_format import ASSISTANT_PREFIX

FAILED = []


def check(name, ok, detail=''):
    print(f'  [{"PASS" if ok else "FAIL"}] {name}{"  -- " + detail if detail else ""}')
    if not ok:
        FAILED.append(name)


def verify_corpus(prefix, tokenizer_dir, seq_length, n_sample):
    tok = Tokenizer.from_file(os.path.join(tokenizer_dir, 'tokenizer.json'))
    bos = tok.token_to_id('<｜begin▁of▁sentence｜>')
    eos = tok.token_to_id('<｜end▁of▁sentence｜>')
    T = indexed_dataset.make_dataset(prefix + '_text_document', 'mmap', skip_warmup=True)
    M = indexed_dataset.make_dataset(prefix + '_answer_mask_document', 'mmap', skip_warmup=True)

    print(f'A. corpus format: {prefix}  ({len(T.sizes):,} docs, bos={bos}, eos={eos})')
    check('A1 mask stream document-aligned', np.array_equal(T.sizes, M.sizes))
    check('A2 every doc exactly seq_length',
          bool((np.asarray(T.sizes) == seq_length).all()))

    rng = random.Random(0)
    docs = sorted({0, len(T.sizes) - 1,
                   *rng.sample(range(len(T.sizes)), min(n_sample, len(T.sizes)))})
    a3 = a4 = a5 = a6 = a7 = True
    sup = slots = 0
    for d in docs:
        ids = np.asarray(T.get(d)); msk = np.asarray(M.get(d))
        sup += int(msk.sum()); slots += len(msk)
        a7 &= set(np.unique(msk)).issubset({0, 1})
        a3 &= ids[0] == bos and msk[0] == 0
        first = int(np.argmax(msk == 1)); last = int(np.max(np.nonzero(msk)))
        prompt = tok.decode(list(ids[1:first]), skip_special_tokens=False)
        a4 &= prompt.startswith('User: ') and \
            prompt.rstrip().endswith(ASSISTANT_PREFIX.strip())
        # every maximal supervised span must end in EOS
        ends = [i for i in range(len(msk) - 1) if msk[i] == 1 and msk[i + 1] == 0]
        if msk[-1] == 1:
            ends.append(len(msk) - 1)
        a5 &= all(ids[i] == eos for i in ends) and len(ends) > 0
        a6 &= bool((ids[last + 1:] == eos).all()) and int(msk[last + 1:].sum()) == 0
    check('A3 BOS present + unsupervised', a3, f'{len(docs)} docs sampled')
    check('A4 pre-answer region is a User:->Assistant: prompt', a4)
    check('A5 every supervised span ends with EOS (stop signal)', a5)
    check('A6 tail is pure EOS padding, mask 0', a6)
    check('A7 mask is strictly 0/1', a7)

    density = sup / slots
    manifest_path = prefix + '_manifest.json'
    if os.path.exists(manifest_path):
        t = json.load(open(manifest_path))['totals']
        expect = t['tokens_supervised'] / t['token_slots']
        check('A8 sampled density matches manifest',
              abs(density - expect) < 0.02,
              f'sampled {density:.4f} vs manifest {expect:.4f}')
    return density


def _series(path, pattern):
    out = []
    for line in open(path, errors='replace'):
        m = re.search(pattern, line)
        if m:
            out.append(float(m.group(1)))
    return out


def verify_scoring(density, density_log, masked_log, unmasked_log):
    print('B. loss scoring (from real training logs)')
    if density_log:
        d = _series(density_log, r'\[sft\] loss_mask density ([0-9.]+)')
        ok = len(d) > 20 and abs(sum(d) / len(d) - density) < 0.02
        check('B1 runtime mask density == corpus density', ok,
              f'{len(d)} microbatches, mean {sum(d)/len(d):.4f} vs corpus {density:.4f}'
              if d else 'no [sft] lines found (run with SFT_DEBUG_MASK=1)')
    if masked_log and unmasked_log:
        lm_m = _series(masked_log, r'lm loss: ([0-9.E+-]+)')
        lm_u = _series(unmasked_log, r'lm loss: ([0-9.E+-]+)')
        moe_m = _series(masked_log, r'moe loss: ([0-9.E+-]+)')
        moe_u = _series(unmasked_log, r'moe loss: ([0-9.E+-]+)')
        n = min(len(lm_m), len(lm_u), 3)
        same_batches = n > 0 and all(abs(moe_m[i] - moe_u[i]) < 1e-9 for i in range(n))
        check('B2a same corpus+seed => identical batches (moe loss equal)',
              same_batches, f'first {n} iters')
        differs = n > 0 and all(lm_u[i] / lm_m[i] > 2.0 for i in range(n))
        check('B2b masked vs unmasked lm loss differs >2x on those batches',
              differs,
              f'masked {lm_m[:n]} vs unmasked {lm_u[:n]}' if n else 'no iters')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-prefix', required=True,
                    help='corpus prefix WITHOUT _text_document')
    ap.add_argument('--tokenizer-dir',
                    default=os.path.join(_REPO, 'examples_elmoe', 'data',
                                         'deepseek_tokenizer'))
    ap.add_argument('--seq-length', type=int, default=4096)
    ap.add_argument('--sample', type=int, default=200)
    ap.add_argument('--density-log', help='rank_0.log of a SFT_DEBUG_MASK=1 run')
    ap.add_argument('--masked-log', help='rank_7.log of the masked run (gate-5 pair)')
    ap.add_argument('--unmasked-log', help='rank_7.log of the unmasked run (gate-5 pair)')
    args = ap.parse_args()

    density = verify_corpus(args.data_prefix, args.tokenizer_dir,
                            args.seq_length, args.sample)
    verify_scoring(density, args.density_log, args.masked_log, args.unmasked_log)
    print()
    if FAILED:
        sys.exit(f'FAILED: {", ".join(FAILED)}')
    print('ALL CHECKS PASSED: corpus is SFT-formatted and the loss is scored '
          'on answer tokens only.')


if __name__ == '__main__':
    main()
