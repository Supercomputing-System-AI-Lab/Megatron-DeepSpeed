#!/usr/bin/env python
"""Re-filter existing mix16k corpora to >=32,768-token documents.

    python filter_32k.py --sources code,books

Reads  books/books.jsonl and code/code.jsonl (docs already carry the EOD as
text), keeps every doc whose deepseek-tokenizer count INCLUDING that EOD is
>= 32768, writes books32/books32.jsonl + code32/code32.jsonl (+ manifests),
then binarises (prefix <src>32-deepseek) and verifies via the stages of
build_mix16k_corpus.py.  House rules inherited from that script: EOD stays
as text (never --append-eod), one jsonl line == one true document (kept
lines are copied verbatim), binarise serialises on mix16k/.binarise.lock.

Safety cross-check: the first N kept-or-dropped docs are compared against the
existing <src>-deepseek_text_document.idx sizes (idx size == text tokens incl
EOD; proven exact on the 16k build: bin = manifest + 1 token/doc).
"""
import argparse, hashlib, json, os, struct, sys, time
import multiprocessing as mp

os.environ.setdefault('DS_ACCELERATOR', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_mix16k_corpus as B

THRESH = 32768
STATUS = os.path.join(HERE, 'filter32.status.json')

_tok = None
def _init(tokjson):
    global _tok
    from tokenizers import Tokenizer
    _tok = Tokenizer.from_file(tokjson)

def _count_line(line):
    """Token count of one jsonl line's text (text already includes EOD)."""
    try:
        text = json.loads(line)['text']
    except Exception:
        return -1
    return len(_tok.encode(text, add_special_tokens=False).ids)

def idx_sizes(idx_path):
    """Per-document token counts from an mmap .idx (one sentence per doc)."""
    import numpy as np
    with open(idx_path, 'rb') as f:
        assert f.read(9) == b'MMIDIDX\x00\x00', 'bad idx magic'
        f.read(8); f.read(1)                       # version, dtype code
        n_len = struct.unpack('<q', f.read(8))[0]
        f.read(8)                                  # doc_count
        return np.frombuffer(f.read(4 * n_len), dtype=np.int32)

def upd_status(**kw):
    st = json.load(open(STATUS)) if os.path.exists(STATUS) else {}
    st.update(kw); st['updated'] = time.strftime('%F %T')
    json.dump(st, open(STATUS + '.tmp', 'w'), indent=2)
    os.replace(STATUS + '.tmp', STATUS)

def filter_jsonl(src, workers, xcheck_n=200):
    src_jsonl = os.path.join(HERE, src, f'{src}.jsonl')
    out_dir = os.path.join(HERE, src + '32')
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, f'{src}32.jsonl')
    man_path = out_jsonl + '.manifest.json'
    if os.path.exists(out_jsonl) and os.path.exists(man_path):
        man = json.load(open(man_path))
        print(f'[{src}32 jsonl] exists, skip'); return out_jsonl, man
    ref = idx_sizes(os.path.join(HERE, src, f'{src}-deepseek_text_document.idx'))
    seen = kept = tok_kept = mism = 0
    hist = {}
    t0 = time.time()
    pool = mp.Pool(workers, initializer=_init, initargs=(B.TOKJSON,))
    import itertools
    with open(src_jsonl, encoding='utf-8') as inp, \
         open(out_jsonl + '.tmp', 'w', encoding='utf-8') as out:
        while True:
            batch = list(itertools.islice(inp, 256))
            if not batch:
                break
            counts = pool.map(_count_line, batch, chunksize=16)
            for line, n in zip(batch, counts):
                if seen < xcheck_n and seen < len(ref) and n != int(ref[seen]):
                    mism += 1
                    print(f'  XCHECK MISMATCH doc {seen}: tokenizer {n} idx {int(ref[seen])}')
                seen += 1
                if n >= THRESH:
                    out.write(line if line.endswith('\n') else line + '\n')
                    kept += 1; tok_kept += n
                    hist[f'{n//1024}k'] = hist.get(f'{n//1024}k', 0) + 1
            if seen % 5120 == 0:
                el = time.time() - t0
                print(f'  [{src}] seen {seen:,} kept {kept:,} '
                      f'tokens {tok_kept/1e9:.3f}B ({seen/el:.0f} docs/s)')
                sys.stdout.flush()
                upd_status(**{src + '32_filter': dict(seen=seen, kept=kept,
                              tokens=tok_kept, docs_per_s=round(seen/el, 1))})
    pool.terminate()
    assert mism == 0, f'{mism} tokenizer-vs-idx mismatches -- alignment broken, aborting'
    os.replace(out_jsonl + '.tmp', out_jsonl)
    cfg = dict(source=src + '32', parent=src_jsonl, min_tokens=THRESH,
               eod='as-text (already present in parent, none added)',
               count_includes_eod=True)
    man = dict(fingerprint=hashlib.sha1(json.dumps(cfg, sort_keys=True)
                                        .encode()).hexdigest()[:12],
               config=cfg,
               totals=dict(docs_seen=seen, docs_kept=kept, tokens_kept=tok_kept,
                           mean_tokens_per_doc=int(tok_kept / max(kept, 1))),
               idx_crosscheck=dict(docs_checked=min(xcheck_n, seen), mismatches=mism),
               kept_length_histogram_1k=dict(sorted(hist.items(),
                   key=lambda kv: int(kv[0][:-1]))))
    json.dump(man, open(man_path, 'w'), indent=2)
    print(f'[{src}32 jsonl] kept {kept:,}/{seen:,} docs, {tok_kept/1e9:.3f}B tokens')
    return out_jsonl, man

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', default='code,books')
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    results = {}
    for src in a.sources.split(','):
        s32 = src + '32'
        print(f'===== {s32}: jsonl ====='); sys.stdout.flush()
        out_jsonl, man = filter_jsonl(src, a.workers)
        tok = man['totals']['tokens_kept']
        results[s32] = man['totals']
        upd_status(**{s32 + '_jsonl': man['totals']})
        prefix = os.path.join(HERE, s32, f'{s32}-deepseek')
        idx = prefix + '_text_document.idx'
        if os.path.exists(idx) and os.path.getmtime(idx) > os.path.getmtime(out_jsonl):
            print(f'===== {s32}: binarise (exists, skip) =====')
        else:
            print(f'===== {s32}: binarise ====='); sys.stdout.flush()
            upd_status(**{s32 + '_stage': 'binarise'})
            B.do_binarise(s32, out_jsonl, prefix, max(a.workers, 16))
        print(f'===== {s32}: verify ====='); sys.stdout.flush()
        upd_status(**{s32 + '_stage': 'verify'})
        B.do_verify(s32, dict(min_tokens=THRESH), prefix)
        upd_status(**{s32 + '_stage': 'done'})
        if src == 'code':
            # 16k code build consumed 22 github-code-clean shards
            n_shards = json.load(open(os.path.join(HERE, 'code',
                'code.jsonl.manifest.json')))['config']['n_input_files']
            per_shard = tok / n_shards
            note = dict(tokens=tok, target=0.6e9,
                        tokens32_per_shard=int(per_shard))
            if tok < 0.6e9:
                import math
                note['SHORTFALL'] = True
                note['extra_shards_needed'] = math.ceil((0.6e9 - tok) / per_shard)
            else:
                note['SHORTFALL'] = False
            upd_status(code32_budget=note)
            print('[code32 budget]', json.dumps(note))
    upd_status(all_done=True, results=results)
    print('ALL DONE:', json.dumps(results))

if __name__ == '__main__':
    main()
