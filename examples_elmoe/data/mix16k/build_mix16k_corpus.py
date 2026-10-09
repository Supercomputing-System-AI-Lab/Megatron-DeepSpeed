#!/usr/bin/env python
"""Build one corpus of the 16k continual-training mix (mix16k).

One script, six sources, four stages:

    python build_mix16k_corpus.py --source wiki --target-tokens 1.1e9 --stage all

Stages: download -> jsonl -> binarise -> verify   (--stage all runs the chain;
each stage is idempotent/resumable and can be re-run alone).

House rules this script enforces (see docs/PLAN_INTRA_DOC_ATTENTION.md):
  * the EOD is appended AS TEXT per document (never --append-eod, it corrupts
    the .idx) -- identical to the nemotron and longctx pipelines;
  * binarisation is the stock tools/preprocess_data.py with HFTokenizer and
    the deepseek tokenizer.json; one jsonl line == one true document, so
    intra-document attention sees real document boundaries (a concatenated
    repo IS one document, by design);
  * binarisation serialises across concurrent builders via flock on
    mix16k/.binarise.lock -- login-node courtesy;
  * a manifest json records config + per-corpus token histogram.

Scaling: rerun with a bigger --target-tokens; downloads resume (curl -C -),
jsonl/bin are rebuilt from scratch for the new target.
"""
import argparse, glob, gzip, hashlib, json, math, os, shutil, subprocess, sys, time
import multiprocessing as mp
import urllib.request

os.environ.setdefault('DS_ACCELERATOR', 'cpu')

HERE   = os.path.dirname(os.path.abspath(__file__))          # .../data/mix16k
DATA   = os.path.dirname(HERE)                               # .../data
REPO   = os.path.dirname(os.path.dirname(DATA))              # Megatron-DeepSpeed-X-MoE
TOKJSON = os.path.join(DATA, 'deepseek_tokenizer', 'tokenizer.json')
EOD_TEXT = '<｜end▁of▁sentence｜>'
LOCK = os.path.join(HERE, '.binarise.lock')

# channels: list of (hf_repo, subpath) walked round-robin, one file at a time.
# field: column holding text. group_by: repo-concat column (code only).
# dl_factor: bytes to download per target token (heuristic incl. filter loss).
SOURCES = {
  'wiki': dict(channels=[('wikimedia/wikipedia', '20231101.en')],
               field='text', min_tokens=300, group_by=None, dl_factor=2.8),
  'openwebmath': dict(channels=[('open-web-math/open-web-math', 'data')],
               field='text', min_tokens=64, group_by=None, dl_factor=2.8),
  'pes2o': dict(channels=[('allenai/peS2o', 'data/v2')],
               field='text', min_tokens=64, group_by=None, dl_factor=2.4),
  'books': dict(channels=[('emozilla/pg19', 'data')],
               field='text', min_tokens=16384, group_by=None, dl_factor=3.8),
  # pdf32 (2026-08-22): dolma3 long PDFs for the 32k stage. Bucket names
  # decode as 2^N Olmo-token lower bounds, so -2e15 = 32768..65536; all 23
  # topic buckets are walked round-robin for topic diversity (jsonl.zst,
  # column 'text'). The real keep/drop decision is min_tokens=32768 under the
  # DeepSeek tokenizer. Measured on the on-disk -2e13 snapshot: 3.8 text
  # bytes/deepseek-token, zstd ~3x -> ~1.3 compressed B/token; dl_factor 3.5
  # covers up to ~60% loss from Olmo-vs-DeepSeek boundary re-tokenisation.
  'pdf32': dict(channels=[('allenai/dolma3_longmino_pool',
                           f'data/olmocr_science_pdfs-{t}-2e15')
                          for t in (
                'adult_content', 'art_and_design', 'crime_and_law',
                'education_and_jobs', 'electronics_and_hardware',
                'entertainment', 'fashion_and_beauty', 'finance_and_business',
                'food_and_dining', 'games', 'health', 'history_and_geography',
                'home_and_hobbies', 'industrial', 'literature', 'politics',
                'religion', 'science_math_and_technology', 'software',
                'software_development', 'sports_and_fitness',
                'transportation', 'travel_and_tourism')],
               field='text', min_tokens=32768, group_by=None, dl_factor=3.5),
  # NOTE 2026-08-21: bigcode/starcoderdata is gated and the available HF token
  # has canReadGatedRepos=false (tree listing works, resolve 403s). Substituted
  # with ungated codeparrot/github-code-clean: same semantics (file rows with a
  # repo column, concatenated per repo within a shard), but shards mix all
  # languages, so lang_filter keeps the six intended ones. Measured on shard 0:
  # ~118M kept tokens / 355MB shard = 3.0 B/kept-token -> dl_factor 4.0.
  'code': dict(channels=[('codeparrot/github-code-clean', 'data')],
               field='code', min_tokens=16384, group_by='repo_name',
               lang_filter=('language', ('Python', 'C++', 'Java',
                                         'JavaScript', 'C', 'GO')),
               dl_factor=4.0),
  'stackexchange': dict(channels=[('togethercomputer/RedPajama-Data-1T', 'urls')],
               field='text', min_tokens=64, group_by=None, dl_factor=5.5,
               special='rp_urls'),
}

def hf_token():
    t = os.environ.get('HF_TOKEN')
    if not t:
        p = os.path.expanduser('~/.cache/huggingface/token')
        t = open(p).read().strip() if os.path.exists(p) else ''
    return t

def hf_api(url):
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {hf_token()}'})
    return json.load(urllib.request.urlopen(req, timeout=60))

def list_channel(repo, sub):
    """All data files under repo/sub, recursive, sorted."""
    out, url = [], f'https://huggingface.co/api/datasets/{repo}/tree/main/{sub}?recursive=true'
    try:
        for f in hf_api(url):
            if f.get('type') == 'file' and f['path'].split('.')[-1] in (
                    'parquet', 'gz', 'zst', 'jsonl', 'json'):
                out.append((f['path'], f.get('size', 0)))
    except Exception as e:
        print(f'  [list_channel] {repo}/{sub}: {e}')
    return sorted(out)

def curl(url, dest, auth=True):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    cmd = ['curl', '-sSL', '-C', '-', '--retry', '5', '--retry-delay', '10',
           '-o', dest, url]
    if auth:
        cmd[1:1] = ['-H', f'Authorization: Bearer {hf_token()}']
    return subprocess.run(cmd).returncode == 0

def do_download(src, cfg, raw, target_tokens):
    budget = int(target_tokens * cfg['dl_factor'])
    have = sum(os.path.getsize(f) for f in glob.glob(raw + '/**/*', recursive=True)
               if os.path.isfile(f))
    print(f'[download] budget {budget/2**30:.1f} GiB, have {have/2**30:.1f} GiB')
    if have >= budget:
        return
    if cfg.get('special') == 'rp_urls':
        return dl_rp_stackexchange(raw, budget)
    chans = [(repo, sub, list_channel(repo, sub)) for repo, sub in cfg['channels']]
    chans = [c for c in chans if c[2]]
    assert chans, 'no listable channels'
    i = 0
    while have < budget and any(c[2] for c in chans):
        repo, sub, files = chans[i % len(chans)]
        i += 1
        if not files:
            continue
        path, size = files.pop(0)
        dest = os.path.join(raw, repo.replace('/', '__'), path.replace('/', '__'))
        if os.path.exists(dest) and os.path.getsize(dest) >= size > 0:
            have += 0; continue
        url = f'https://huggingface.co/datasets/{repo}/resolve/main/{path}'
        print(f'  fetch {path} ({size/2**20:.0f} MB)'); sys.stdout.flush()
        assert curl(url, dest), f'download failed: {url}'
        have += os.path.getsize(dest)
    print(f'[download] done, {have/2**30:.1f} GiB on disk')

def dl_rp_stackexchange(raw, budget):
    lst = os.path.join(raw, 'stackexchange.txt')
    if not os.path.exists(lst):
        ok = curl('https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T/resolve/main/urls/stackexchange.txt', lst)
        assert ok, 'could not fetch RP stackexchange url list from HF'
    urls = [u.strip() for u in open(lst) if u.strip().startswith('http')]
    assert urls, 'RP url list empty/HTML -- agent: switch to fallback SE source'
    have = 0
    for u in urls:
        dest = os.path.join(raw, 'files', u.rsplit('/', 1)[-1])
        need = budget - have
        if not (os.path.exists(dest) and os.path.getsize(dest) >= need):
            # the RP SE shard is a single 80 GB jsonl; fetch only the budget
            # prefix (jsonl stage skips the torn final line as malformed json)
            print(f'  fetch prefix {need/2**30:.1f} GiB of {u}'); sys.stdout.flush()
            fetch_prefix(u, dest, need)
        have += os.path.getsize(dest)
        if have >= budget:
            break
    print(f'[download] SE done, {have/2**30:.1f} GiB')

def fetch_prefix(url, dest, nbytes):
    """Resumable byte-range download of the first nbytes of url."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    stall = 0
    while stall < 6:
        cur = os.path.getsize(dest) if os.path.exists(dest) else 0
        if cur >= nbytes:
            return
        part = dest + '.part'
        if os.path.exists(part):
            os.remove(part)
        subprocess.run(['curl', '-sSL', '--retry', '5', '--retry-delay', '10',
                        '-r', f'{cur}-{nbytes-1}', '-o', part, url])
        got = os.path.getsize(part) if os.path.exists(part) else 0
        stall = 0 if got else stall + 1
        if got:
            with open(dest, 'ab') as o, open(part, 'rb') as p:
                shutil.copyfileobj(p, o)
        if os.path.exists(part):
            os.remove(part)
    assert False, f'range fetch stalled: {url}'

_tok = None
def _init_tok(tokenizer_json):
    global _tok
    from tokenizers import Tokenizer
    _tok = Tokenizer.from_file(tokenizer_json)

def _count(text):
    return len(_tok.encode(text, add_special_tokens=False).ids)

def iter_texts(files, cfg):
    """Yield document texts from raw files (parquet / jsonl / json.gz / zst)."""
    import pyarrow.parquet as pq
    field, group = cfg['field'], cfg['group_by']
    for f in files:
        if f.endswith('.parquet'):
            if group:                       # code: repo-concat within one shard
                import pandas as pd
                lf = cfg.get('lang_filter')           # (column, allowed) or None
                cols = [field, group] + ([lf[0]] if lf else [])
                df = pq.read_table(f, columns=cols).to_pandas()
                if lf:
                    df = df[df[lf[0]].isin(lf[1])]
                for _, g in df.groupby(group, sort=False):
                    yield '\n\n'.join(g[field].tolist())
                del df
            else:
                pf = pq.ParquetFile(f)
                for batch in pf.iter_batches(batch_size=512, columns=[field]):
                    for t in batch.column(0).to_pylist():
                        if t:
                            yield t
        elif f.endswith(('.jsonl', '.json')):
            for line in open(f, encoding='utf-8', errors='ignore'):
                try:
                    t = json.loads(line).get(field) or json.loads(line).get('text')
                except Exception:
                    continue
                if t:
                    yield t
        elif f.endswith('.gz'):
            with gzip.open(f, 'rt', encoding='utf-8', errors='ignore') as fh:
                for line in fh:
                    try:
                        t = json.loads(line).get(field) or json.loads(line).get('text')
                    except Exception:
                        continue
                    if t:
                        yield t
        elif f.endswith('.zst'):
            import io, zstandard
            with open(f, 'rb') as bh:
                sh = zstandard.ZstdDecompressor().stream_reader(bh)
                for line in io.TextIOWrapper(sh, encoding='utf-8', errors='ignore'):
                    try:
                        t = json.loads(line).get(field) or json.loads(line).get('text')
                    except Exception:
                        continue
                    if t:
                        yield t

def do_jsonl(src, cfg, raw, out_jsonl, target_tokens, workers):
    files = sorted(f for f in glob.glob(raw + '/**/*', recursive=True)
                   if os.path.isfile(f) and not f.endswith('.txt'))
    assert files, f'no raw files under {raw}'
    cfgdump = dict(source=src, min_tokens=cfg['min_tokens'],
                   target_tokens=target_tokens, n_input_files=len(files),
                   group_by=cfg['group_by'], eod='as-text')
    kept = seen = tok_kept = 0
    hist = {}
    t0 = time.time()
    pool = mp.Pool(workers, initializer=_init_tok, initargs=(TOKJSON,))
    with open(out_jsonl + '.tmp', 'w', encoding='utf-8') as out:
        it = iter_texts(files, cfg)
        for text, n in zip_texts_counts(pool, it):
            seen += 1
            if n >= cfg['min_tokens']:
                out.write(json.dumps({'text': text + EOD_TEXT},
                                     ensure_ascii=False) + '\n')
                kept += 1; tok_kept += n
                hist[f'{n//1024}k'] = hist.get(f'{n//1024}k', 0) + 1
            if seen % 20000 == 0:
                r = tok_kept / max(time.time()-t0, 1)
                print(f'  seen {seen:,} kept {kept:,} tokens {tok_kept/1e9:.3f}B '
                      f'({r/1e6:.2f}M tok/s)'); sys.stdout.flush()
            if tok_kept >= target_tokens:
                break
    pool.terminate()
    os.replace(out_jsonl + '.tmp', out_jsonl)
    man = dict(fingerprint=hashlib.sha1(json.dumps(cfgdump, sort_keys=True)
                                        .encode()).hexdigest()[:12],
               config=cfgdump,
               totals=dict(docs_seen=seen, docs_kept=kept, tokens_kept=tok_kept,
                           mean_tokens_per_doc=int(tok_kept/max(kept, 1))),
               kept_length_histogram_1k=dict(sorted(hist.items(),
                   key=lambda kv: int(kv[0][:-1]))))
    json.dump(man, open(out_jsonl + '.manifest.json', 'w'), indent=2)
    print(f'[jsonl] kept {kept:,}/{seen:,} docs, {tok_kept/1e9:.3f}B tokens')
    assert tok_kept >= 0.85 * target_tokens, \
        f'SHORTFALL: only {tok_kept/1e9:.2f}B of {target_tokens/1e9:.2f}B -- download more shards and rerun'

def zip_texts_counts(pool, texts_iter):
    """Stream (text, n_tokens) pairs; token counting in the worker pool."""
    import itertools
    BATCH = 256
    while True:
        batch = list(itertools.islice(texts_iter, BATCH))
        if not batch:
            return
        for text, n in zip(batch, pool.map(_count, batch, chunksize=16)):
            yield text, n

def do_binarise(src, out_jsonl, prefix, workers):
    assert os.path.exists(out_jsonl), f'missing {out_jsonl} -- run --stage jsonl'
    cmd = ['flock', '-w', '28800', LOCK, sys.executable,
           os.path.join(REPO, 'tools', 'preprocess_data.py'),
           '--input', out_jsonl, '--output-prefix', prefix,
           '--tokenizer-type', 'HFTokenizer', '--vocab-file', TOKJSON,
           '--eod-token', EOD_TEXT, '--dataset-impl', 'mmap',
           '--workers', str(workers)]
    print('[binarise]', ' '.join(cmd)); sys.stdout.flush()
    assert subprocess.run(cmd).returncode == 0, 'binarise failed'

def do_verify(src, cfg, prefix):
    cmd = [sys.executable, os.path.join(REPO, 'tools', 'preprocess_longctx_data.py'),
           '--verify-prefix', prefix + '_text_document',
           '--min-tokens', str(cfg['min_tokens'])]
    print('[verify]', ' '.join(cmd)); sys.stdout.flush()
    assert subprocess.run(cmd).returncode == 0, 'verify failed'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, choices=sorted(SOURCES))
    ap.add_argument('--target-tokens', type=float, required=True)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--stage', default='all',
                    choices=['download', 'jsonl', 'binarise', 'verify', 'all'])
    a = ap.parse_args()
    cfg = SOURCES[a.source]
    out_dir = os.path.join(HERE, a.source)
    raw = os.path.join(out_dir, 'raw')
    os.makedirs(raw, exist_ok=True)
    out_jsonl = os.path.join(out_dir, f'{a.source}.jsonl')
    prefix = os.path.join(out_dir, f'{a.source}-deepseek')
    stages = ['download', 'jsonl', 'binarise', 'verify'] if a.stage == 'all' else [a.stage]
    for s in stages:
        print(f'===== {a.source}: {s} ====='); sys.stdout.flush()
        if s == 'download': do_download(a.source, cfg, raw, a.target_tokens)
        elif s == 'jsonl':  do_jsonl(a.source, cfg, raw, out_jsonl, a.target_tokens, a.workers)
        elif s == 'binarise': do_binarise(a.source, out_jsonl, prefix, max(a.workers, 16))
        elif s == 'verify': do_verify(a.source, cfg, prefix)
    print(f'DONE {a.source}: {prefix}_text_document')

if __name__ == '__main__':
    main()
