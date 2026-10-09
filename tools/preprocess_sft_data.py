#!/usr/bin/env python
"""Render an SFT conversation corpus into TWO aligned Megatron indexed streams.

    <prefix>_text_document.{bin,idx}          int32   the token ids
    <prefix>_answer_mask_document.{bin,idx}   uint8   1 = supervised position

Written directly with MMapIndexedDatasetBuilder -- the same writer
tools/preprocess_data.py uses -- so the on-disk format is byte-identical to
every other corpus in this tree. tools/preprocess_data.py itself is NOT used:
the mask is a product of chat rendering, which that tool knows nothing about
(and its multi-key path writes only the last key's .idx -- the :177 finalize
bug).

Every document is emitted at EXACTLY --seq-length tokens (pad included).
GPTDataset packs by concatenating documents and cutting every seq_length+1
tokens, so equal-length documents are what makes one sample == one
conversation. Conversations longer than the limit are DROPPED, never truncated
(a severed answer supervises stopping mid-thought). Under --answer-loss-only
the trainer asserts the two streams' `sizes` arrays are identical -- that
invariant is what this script guarantees and its verify step re-checks.

USAGE (login node; rendering is CPU-only)
    python tools/preprocess_sft_data.py \
        --input <snapshot_dir_with_parquet> \
        --output-prefix examples_elmoe/data/dolci/dolci-sft-deepseek \
        --drop-sources <comma_list_from_the_measurement_pass> \
        --workers 16

Needs: tokenizers, pyarrow (both in the training envs). Tokenization uses the
RAW Rust tokenizer (tokenizers.Tokenizer) -- the same code path as training's
_HFTokenizer -- NOT transformers' AutoTokenizer, whose behavior on this
tokenizer dir is version-dependent (5.8.1 silently encodes non-ASCII to zero
tokens; 4.36.2 matches the Rust core).
"""

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

# Data prep is CPU-only; without this, importing megatron on a login node dies
# in DeepSpeed's CUDA/ROCm op-builder scan. setdefault: never overrides a shell
# that already chose an accelerator.
os.environ.setdefault('DS_ACCELERATOR', 'cpu')

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from megatron.data import indexed_dataset
from tools.sft_chat_format import render, pad_to, normalize_messages

DEFAULT_TOKENIZER_DIR = os.path.join(
    _REPO, 'examples_elmoe', 'data', 'deepseek_tokenizer')

# ---------------------------------------------------------------- workers ----

_tok = _bos = _eos = None


class _RustTokenizer:
    """The RAW Rust tokenizer -- the same code path training uses
    (_HFTokenizer wraps tokenizers.Tokenizer.from_file). transformers'
    AutoTokenizer wrapper is version-dependent on this tokenizer dir:
    verified 2026-08-13 that 4.36.2 matches the Rust core exactly while
    5.8.1 silently encodes non-ASCII text to ZERO tokens. Using the Rust
    core directly makes the corpus match training by construction,
    independent of the active env."""

    def __init__(self, tokenizer_dir):
        from tokenizers import Tokenizer
        self._t = Tokenizer.from_file(
            os.path.join(tokenizer_dir, 'tokenizer.json'))
        cfg = json.load(open(os.path.join(tokenizer_dir,
                                          'tokenizer_config.json')))

        def _tok_id(field):
            v = cfg[field]
            content = v['content'] if isinstance(v, dict) else v
            i = self._t.token_to_id(content)
            assert i is not None, f'{field} {content!r} not in vocab'
            return i

        self.bos_token_id = _tok_id('bos_token')
        self.eos_token_id = _tok_id('eos_token')
        self.vocab_size = self._t.get_vocab_size()

    def encode(self, text, add_special_tokens=False):
        return self._t.encode(text,
                              add_special_tokens=add_special_tokens).ids


def _init_worker(tokenizer_dir):
    global _tok, _bos, _eos
    _tok = _RustTokenizer(tokenizer_dir)
    _bos, _eos = _tok.bos_token_id, _tok.eos_token_id


def _render_row(row):
    """(messages, source) -> (ids, mask, n_supervised, source) or a drop tag."""
    msgs = normalize_messages(row['messages'])
    try:
        ids, mask = render(msgs, _tok, _bos, _eos)
    except Exception:
        return ('bad', row['source'], None, None, 0)
    if not any(mask):                    # no assistant turn -> nothing to learn
        return ('bad', row['source'], None, None, 0)
    return ('ok', row['source'], ids, mask, sum(mask))


# ------------------------------------------------------------------ input ----

def iter_rows(parquet_files, keep, drop, limit, counts):
    """Yield {'messages','source'} dicts, applying the source filter. Any shard
    that fails to read is FATAL -- no silent partial corpora."""
    import pyarrow.parquet as pq
    seen = 0
    for path in parquet_files:
        pf = pq.ParquetFile(path)          # raises on a broken shard: good
        # schema_arrow: top-level column names (plain .schema gives parquet
        # LEAF names like 'messages.list.element.role').
        # Dolci names its provenance column source_dataset; tulu-style uses source.
        cols = [c for c in ('messages', 'source_dataset', 'source')
                if c in pf.schema_arrow.names]
        assert 'messages' in cols, f'{path}: no messages column'
        for batch in pf.iter_batches(batch_size=1024, columns=cols):
            for row in batch.to_pylist():
                if limit and seen >= limit:
                    return
                seen += 1
                src = row.get('source_dataset') or row.get('source') or 'unknown'
                if (keep and src not in keep) or (src in drop):
                    counts[src]['dropped_source'] += 1
                    continue
                yield {'messages': row['messages'], 'source': src}


# ------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True,
                    help='directory containing the corpus parquet files (searched recursively)')
    ap.add_argument('--output-prefix', required=True)
    ap.add_argument('--tokenizer-dir', default=DEFAULT_TOKENIZER_DIR)
    ap.add_argument('--seq-length', type=int, default=4096)
    ap.add_argument('--keep-sources', default='',
                    help='comma list; if set, keep ONLY these `source` values')
    ap.add_argument('--drop-sources', default='',
                    help='comma list of `source` values to drop')
    ap.add_argument('--drop-longer-than', type=int, default=0,
                    help='drop conversations over this many tokens (default: --seq-length)')
    ap.add_argument('--dedup-prompts', action='store_true',
                    help='keep only the first conversation per distinct prompt prefix')
    ap.add_argument('--packed-masked', action='store_true',
                    help='PACKED + MASKED variant (2026-08-23): variable-length '
                    'documents (GPTDataset packs them by concatenation) WITH an '
                    'aligned answer-mask stream. The training-side mask sampler '
                    '(gpt_dataset.py:337) applies the identical doc/offset window '
                    'to both streams, so masks survive packing byte-for-byte. '
                    'Train WITH --answer-loss-only. Grounding: DeepSeek-V3 sample-'
                    'masked packing (arXiv:2412.19437 5.1); OLMo 3 document '
                    'packing (arXiv:2512.13961 A.6.1).')
    ap.add_argument('--no-pad', action='store_true',
                    help='FULL-LOSS ablation variant: same rows, same renderer, but '
                    'variable-length documents (GPTDataset packs them) and NO mask '
                    'stream. Train WITHOUT --answer-loss-only (SFT_MODE=full). The '
                    'padded corpus must never be used for full loss: 85%% of its '
                    'slots are EOS padding and would be supervised.')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0,
                    help='stop after this many input rows (0 = all); for smoke runs')
    ap.add_argument('--log-interval', type=int, default=20000)
    args = ap.parse_args()

    drop_over = args.drop_longer_than or args.seq_length
    assert drop_over <= args.seq_length, \
        'documents are padded to --seq-length; cannot keep longer ones'
    keep = set(s for s in args.keep_sources.split(',') if s)
    drop = set(s for s in args.drop_sources.split(',') if s)

    parquet_files = sorted(
        glob.glob(os.path.join(args.input, '**', '*.parquet'), recursive=True))
    assert parquet_files, f'no parquet under {args.input}'

    # Config fingerprint: a config change MUST NOT silently reuse old output.
    config = {k: v for k, v in sorted(vars(args).items())
              if k not in ('workers', 'log_interval')}
    config['n_input_files'] = len(parquet_files)
    config['script_version'] = 4   # bump on behavior changes: the guard keys on
                                   # config, and a code change must not silently
                                   # pass as "already built"
    fingerprint = hashlib.sha1(
        json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]

    out_dir = os.path.dirname(os.path.abspath(args.output_prefix))
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = args.output_prefix + '_manifest.json'
    if os.path.exists(manifest_path):
        old = json.load(open(manifest_path))
        assert old.get('fingerprint') == fingerprint, (
            f'{manifest_path} was built with a different config '
            f'({old.get("fingerprint")} != {fingerprint}); '
            'delete the old output or change --output-prefix')
        print('identical config already built -- nothing to do')
        return

    _init_worker(args.tokenizer_dir)      # main process copy, for ids/vocab
    vocab_size = _tok.vocab_size
    print(f'tokenizer  : {args.tokenizer_dir} (vocab {vocab_size}, '
          f'bos {_bos}, eos {_eos})')
    print(f'input      : {len(parquet_files)} parquet files under {args.input}')
    print(f'fingerprint: {fingerprint}')

    text_prefix = args.output_prefix + '_text_document'
    mask_prefix = args.output_prefix + '_answer_mask_document'
    text_builder = indexed_dataset.make_builder(
        text_prefix + '.bin', impl='mmap', vocab_size=vocab_size)
    mask_builder = None if (args.no_pad and not args.packed_masked) else \
        indexed_dataset.MMapIndexedDatasetBuilder(
            mask_prefix + '.bin', dtype=np.uint8)

    from collections import defaultdict
    counts = defaultdict(lambda: defaultdict(int))
    seen_prompts = set()
    kept = tokens_real = tokens_sup = 0
    t0 = time.time()

    rows = iter_rows(parquet_files, keep, drop, args.limit, counts)
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.tokenizer_dir,)) as pool:
        for status, src, ids, mask, n_sup in pool.imap(
                _render_row, rows, chunksize=64):
            c = counts[src]
            if status == 'bad':
                c['dropped_bad'] += 1
                continue
            if len(ids) > drop_over:
                c['dropped_long'] += 1
                continue
            if args.dedup_prompts:
                # prompt = everything before the first supervised position
                first_sup = mask.index(1)
                key = hashlib.blake2b(
                    np.asarray(ids[:first_sup], dtype=np.int64).tobytes(),
                    digest_size=16).digest()
                if key in seen_prompts:
                    c['dropped_dup'] += 1
                    continue
                seen_prompts.add(key)
            n_real = len(ids)
            if args.packed_masked:
                text_builder.add_doc(
                    torch.tensor(ids, dtype=torch.int32), [n_real])
                mask_builder.add_doc(
                    torch.tensor(mask, dtype=torch.uint8), [n_real])
            elif args.no_pad:
                text_builder.add_doc(
                    torch.tensor(ids, dtype=torch.int32), [n_real])
            else:
                ids, mask = pad_to(ids, mask, args.seq_length, pad_id=_eos)
                text_builder.add_doc(
                    torch.tensor(ids, dtype=torch.int32), [args.seq_length])
                mask_builder.add_doc(
                    torch.tensor(mask, dtype=torch.uint8), [args.seq_length])
            c['kept'] += 1
            kept += 1
            tokens_real += n_real
            tokens_sup += n_sup
            if kept % args.log_interval == 0:
                r = kept / (time.time() - t0)
                print(f'  {kept:,} docs kept ({r:,.0f} docs/s)', flush=True)

    assert kept > 0, 'no documents survived the filters'
    text_builder.finalize(text_prefix + '.idx')
    if mask_builder is not None:
        mask_builder.finalize(mask_prefix + '.idx')

    # Verify by reading back with the real reader -- never a hand-rolled parse.
    tds = indexed_dataset.make_dataset(text_prefix, 'mmap', skip_warmup=True)
    assert len(tds.sizes) == kept
    if mask_builder is not None:
        mds = indexed_dataset.make_dataset(mask_prefix, 'mmap', skip_warmup=True)
        assert np.array_equal(tds.sizes, mds.sizes), 'ALIGNMENT BROKEN: sizes differ'
        if not args.packed_masked:
            assert int(tds.sizes[0]) == args.seq_length

    slots = tokens_real if (args.no_pad or args.packed_masked) else kept * args.seq_length
    manifest = {
        'fingerprint': fingerprint,
        'config': config,
        'per_source': {k: dict(v) for k, v in sorted(counts.items())},
        'totals': {
            'docs_kept': kept,
            'tokens_real': tokens_real,
            'tokens_supervised': tokens_sup,
            'supervised_density_of_real': round(tokens_sup / tokens_real, 4),
            'token_slots': slots,
            'padding_waste': round(1 - tokens_real / slots, 4),
        },
        'outputs': [text_prefix + e for e in ('.bin', '.idx')]
                 + ([] if (args.no_pad and not args.packed_masked) else
                    [mask_prefix + e for e in ('.bin', '.idx')]),
    }
    with open(manifest_path, 'w') as fh:
        json.dump(manifest, fh, indent=2)

    print(json.dumps(manifest['totals'], indent=2))
    if args.no_pad:
        print(f'\nVERIFIED: {kept:,} variable-length documents (full-loss ablation corpus)')
        print('\ntraining: SFT_MODE=full SFT_DATA_PATH=' + text_prefix)
        print('  (full loss, NO --answer-loss-only; never mix up with the padded corpus)')
    else:
        print(f'\nVERIFIED: {kept:,} ' + ('variable-length PACKED+MASKED documents' if args.packed_masked else f'aligned documents of exactly {args.seq_length} tokens'))
        print('\ntraining flags:')
        print(f'  --data-path {text_prefix}')
        print('  --answer-loss-only        # the trainer derives the mask stream path')


if __name__ == '__main__':
    main()
