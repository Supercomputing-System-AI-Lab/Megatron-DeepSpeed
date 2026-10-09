#!/usr/bin/env python
"""Length-filter a raw-text long-document corpus for context-extension CPT.

Two modes:

  filter (default)  parquet snapshot -> {"text": ...} JSONL of documents whose
                    length UNDER THE DEEPSEEK TOKENIZER is >= --min-tokens.
                    Published bucket boundaries are someone else's tokenizer
                    counts (Olmo's, for dolma3_longmino) and differ 5-15% in an
                    unpredictable direction -- the bucket decides what to
                    DOWNLOAD, this filter decides what to KEEP.

  --verify-prefix   after tools/preprocess_data.py has binarised the JSONL,
                    re-open the .bin/.idx with the real reader and check the
                    per-document token histogram against --min-tokens.

This is raw-text continued pretraining: full loss, no mask, single stream.
The binarisation itself is tools/preprocess_data.py, UNCHANGED (and without
--append-eod, which corrupts the .idx); the EOD is appended here as the
literal token string instead, exactly as the nemotron pipeline did.

USAGE (login node)
    python tools/preprocess_longctx_data.py \
        --input <snapshot>/data --min-tokens 8192 --target-tokens 1.7e9 \
        --output-jsonl examples_elmoe/data/longctx/L8.jsonl
"""

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault('DS_ACCELERATOR', 'cpu')   # CPU-only; login-node safe

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

DEFAULT_TOKENIZER_JSON = os.path.join(
    _REPO, 'examples_elmoe', 'data', 'deepseek_tokenizer', 'tokenizer.json')
EOD_TEXT = '<｜end▁of▁sentence｜>'   # <｜end▁of▁sentence｜>

_tok = None
_min = _max = 0


def _init_worker(tokenizer_json, min_tokens, max_tokens):
    global _tok, _min, _max
    from tokenizers import Tokenizer
    _tok = Tokenizer.from_file(tokenizer_json)
    _min, _max = min_tokens, max_tokens


def _filter_doc(text):
    """Worker: count under the DeepSeek tokenizer, decide keep/drop.
    Returns (n_tokens, text) when kept, (n_tokens, None) when dropped -- the
    text only crosses the process boundary back when it survived."""
    n = len(_tok.encode(text, add_special_tokens=False).ids)
    if n < _min or (_max and n > _max):
        return n, None
    return n, text


def do_filter(args):
    # dolma3_longmino_pool ships .jsonl.zst; parquet kept for other corpora
    files = sorted(glob.glob(os.path.join(args.input, '**', '*.parquet'),
                             recursive=True)
                   + glob.glob(os.path.join(args.input, '**', '*.jsonl.zst'),
                               recursive=True))
    assert files, f'no parquet or jsonl.zst under {args.input}'

    config = {'min_tokens': args.min_tokens, 'max_tokens': args.max_tokens,
              'target_tokens': args.target_tokens, 'n_input_files': len(files),
              'input': os.path.abspath(args.input), 'append_eod': not args.no_append_eod}
    fingerprint = hashlib.sha1(
        json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    manifest_path = args.output_jsonl + '.manifest.json'
    if os.path.exists(manifest_path):
        old = json.load(open(manifest_path))
        assert old.get('fingerprint') == fingerprint, (
            f'{manifest_path} was built with a different config; '
            'delete the old output or change --output-jsonl')
        print('identical config already built -- nothing to do')
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)

    def texts_from(path):
        """Yield the text field of every document in one shard. A broken
        shard raises -- FATAL by design, no silent partial corpora."""
        if path.endswith('.parquet'):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            assert 'text' in pf.schema_arrow.names, f'{path}: no text column'
            for batch in pf.iter_batches(batch_size=64, columns=['text']):
                yield from batch.column('text').to_pylist()
        else:                                   # .jsonl.zst
            import io
            import zstandard
            with open(path, 'rb') as fh:
                reader = zstandard.ZstdDecompressor(
                    max_window_size=2**31).stream_reader(fh)
                for line in io.TextIOWrapper(reader, encoding='utf-8'):
                    if line.strip():
                        yield json.loads(line)['text']

    def batches(batch_docs=512):
        """Yield lists of raw texts; bounded memory, deterministic order."""
        buf = []
        for path in files:
            for text in texts_from(path):
                buf.append(text)
                if len(buf) == batch_docs:
                    yield buf
                    buf = []
        if buf:
            yield buf

    kept = seen = 0
    kept_tokens = 0
    hist = {}                                   # 1k-bucket histogram of KEPT docs
    t0 = time.time()
    done = False
    tmp = args.output_jsonl + '.partial'
    with open(tmp, 'w') as out, \
         mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.tokenizer_json, args.min_tokens,
                           args.max_tokens)) as pool:
        for batch in batches():
            # map over one bounded batch at a time: pool.imap would drain the
            # whole corpus into its (unbounded) task queue
            for n, text in pool.map(_filter_doc, batch, chunksize=8):
                seen += 1
                if text is None:
                    continue
                doc = text + ('' if args.no_append_eod else EOD_TEXT)
                out.write(json.dumps({'text': doc}) + '\n')
                kept += 1
                kept_tokens += n
                hist[n // 1024] = hist.get(n // 1024, 0) + 1
                if kept % 2000 == 0:
                    print(f'  kept {kept:,}/{seen:,} docs, '
                          f'{kept_tokens/1e9:.2f}B tokens '
                          f'({seen/(time.time()-t0):,.0f} docs/s seen)',
                          flush=True)
                if args.target_tokens and kept_tokens >= args.target_tokens:
                    print('  target reached')
                    done = True
                    break
            if done:
                break
    assert kept > 0, 'no documents passed the length filter'
    os.replace(tmp, args.output_jsonl)

    manifest = {
        'fingerprint': fingerprint, 'config': config,
        'totals': {'docs_seen': seen, 'docs_kept': kept,
                   'tokens_kept': kept_tokens,
                   'mean_tokens_per_doc': round(kept_tokens / kept)},
        'kept_length_histogram_1k': {f'{k}k-{k+1}k': v
                                     for k, v in sorted(hist.items())},
    }
    with open(manifest_path, 'w') as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest['totals'], indent=2))
    print(f'wrote {args.output_jsonl}')


def do_verify(args):
    import numpy as np
    from megatron.data import indexed_dataset
    ds = indexed_dataset.make_dataset(args.verify_prefix, 'mmap',
                                      skip_warmup=True)
    sizes = np.asarray(ds.sizes)
    under = int((sizes < args.min_tokens).sum())
    print(f'documents           {len(sizes):,}')
    print(f'tokens              {int(sizes.sum()):,}')
    print(f'min/median/max      {sizes.min()} / {int(np.median(sizes))} / {sizes.max()}')
    print(f'under --min-tokens  {under}  ({100*under/len(sizes):.2f}%)')
    assert under == 0, 'length filter did not hold in the binarised corpus'
    print('VERIFIED: every document >= min-tokens')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', help='snapshot dir with parquet (filter mode)')
    ap.add_argument('--output-jsonl', help='filtered JSONL to write (filter mode)')
    ap.add_argument('--tokenizer-json', default=DEFAULT_TOKENIZER_JSON)
    ap.add_argument('--min-tokens', type=int, required=True,
                    help='keep only docs with >= this many DeepSeek tokens '
                    '(= the training seq_length, so every sequence carries '
                    'genuine long-range dependency)')
    ap.add_argument('--max-tokens', type=int, default=0, help='0 = no cap')
    ap.add_argument('--target-tokens', type=float, default=0,
                    help='stop once this many tokens are kept (0 = all)')
    ap.add_argument('--no-append-eod', action='store_true',
                    help='do not append the EOD token string to each document')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--verify-prefix',
                    help='verify a built .bin/.idx prefix instead of filtering')
    args = ap.parse_args()

    if args.verify_prefix:
        do_verify(args)
    else:
        assert args.input and args.output_jsonl, \
            'filter mode needs --input and --output-jsonl'
        do_filter(args)


if __name__ == '__main__':
    main()
