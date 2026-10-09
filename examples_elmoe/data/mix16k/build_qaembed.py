#!/usr/bin/env python
"""Build the 'qaembed' corpus: contamination-safe long-document QA (mix16k).

Each synthetic document embeds the evidence paragraphs of 3-8 real QA items
(SQuAD v2 TRAIN + HotpotQA TRAIN/fullwiki) among random Wikipedia distractor
paragraphs, padded to a target length drawn uniformly from [8192, 32768]
tokens (approximated as chars/3.7 during assembly), followed by a QA section
('Question: ...\nAnswer: ...' per item). One synthetic document == one jsonl
line; the EOD string is appended AS TEXT (never --append-eod).

Evidence basis (see manifest): Nemotron Nano 2 (20% long-document-QA in its
long-context phase), MiniMax-01 (10% long-QA in the last 20% of each stage),
OLMo 3 (synthetic QA-in-doc, CWE/REX), and arXiv:2502.15592 (long-context
instructions must require in-context knowledge and include distractors).

Contamination safety: TRAIN splits only. RULER's qa_1/qa_2 evals draw from
the SQuAD dev and HotpotQA dev/validation splits of these same datasets, so
train-split usage cannot leak eval items.

Usage:
    build_qaembed.py --target-tokens 0.75e9 --stage all
Stages: download -> jsonl -> binarise -> verify (reuses build_mix16k_corpus
helpers: same flock-serialised stock preprocess_data.py binarisation).
"""
import argparse, hashlib, json, os, random, sys, time
import multiprocessing as mp

os.environ.setdefault('DS_ACCELERATOR', 'cpu')

HERE = os.path.dirname(os.path.abspath(__file__))            # .../data/mix16k
sys.path.insert(0, HERE)
import build_mix16k_corpus as B                              # shared helpers

EOD_TEXT = B.EOD_TEXT
CHARS_PER_TOKEN = 3.7
WIKI_JSONL = os.path.join(HERE, 'wiki', 'wiki.jsonl')

DL_FILES = [  # (hf_repo, path_in_repo)
    ('rajpurkar/squad_v2', 'squad_v2/train-00000-of-00001.parquet'),
    ('hotpotqa/hotpot_qa', 'fullwiki/train-00000-of-00002.parquet'),
    ('hotpotqa/hotpot_qa', 'fullwiki/train-00001-of-00002.parquet'),
]


def do_download(raw):
    for repo, path in DL_FILES:
        dest = os.path.join(raw, repo.replace('/', '__'),
                            path.replace('/', '__'))
        url = f'https://huggingface.co/datasets/{repo}/resolve/main/{path}'
        if os.path.exists(dest) and os.path.getsize(dest) > 1 << 20:
            print(f'  have {path}')
            continue
        print(f'  fetch {path}'); sys.stdout.flush()
        assert B.curl(url, dest), f'download failed: {url}'
    print('[download] done')


def load_squad(raw):
    """-> list of (question, answer, (evidence_para,)) ; SQuAD v2 TRAIN."""
    import pyarrow.parquet as pq
    f = os.path.join(raw, 'rajpurkar__squad_v2',
                     'squad_v2__train-00000-of-00001.parquet')
    items, n_unans = [], 0
    tbl = pq.read_table(f, columns=['title', 'context', 'question', 'answers'])
    for title, ctx, q, ans in zip(tbl.column('title').to_pylist(),
                                  tbl.column('context').to_pylist(),
                                  tbl.column('question').to_pylist(),
                                  tbl.column('answers').to_pylist()):
        texts = ans.get('text') or []
        if texts:
            a = texts[0]
        else:                      # SQuAD v2 unanswerable: keep, they force
            a = 'unanswerable'     # the model to check the context (distractor
            n_unans += 1           # -robustness per 2502.15592)
        para = f'{title}\n{ctx}' if title else ctx
        items.append((q.strip(), a.strip(), (para,)))
    print(f'[load] squad_v2 train: {len(items):,} items '
          f'({n_unans:,} unanswerable)')
    return items, n_unans


def load_hotpot(raw):
    """-> list of (question, answer, (evidence_para, ...)) ; HotpotQA TRAIN.

    Evidence = the supporting-fact paragraphs only; the other retrieved
    context paragraphs are dropped (our distractors come from wiki.jsonl).
    """
    import pyarrow.parquet as pq
    items = []
    for shard in ('fullwiki__train-00000-of-00002.parquet',
                  'fullwiki__train-00001-of-00002.parquet'):
        f = os.path.join(raw, 'hotpotqa__hotpot_qa', shard)
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=256,
                                     columns=['question', 'answer',
                                              'supporting_facts', 'context']):
            d = batch.to_pydict()
            for q, a, sf, ctx in zip(d['question'], d['answer'],
                                     d['supporting_facts'], d['context']):
                sup = set(sf['title'])
                paras = tuple(
                    f'{t}\n{" ".join(sents)}'
                    for t, sents in zip(ctx['title'], ctx['sentences'])
                    if t in sup)
                if paras and q and a:
                    items.append((q.strip(), a.strip(), paras))
    print(f'[load] hotpot_qa fullwiki train: {len(items):,} items')
    return items


def wiki_paragraphs(rng, min_chars=200, max_chars=3000, buf_n=50000):
    """Endless locally-shuffled stream of distractor paragraphs from the
    existing wiki.jsonl (trailing EOD string stripped!). Yields
    (paragraph, epoch)."""
    epoch = 0
    while True:
        epoch += 1
        buf = []
        with open(WIKI_JSONL, encoding='utf-8') as fh:
            for line in fh:
                try:
                    text = json.loads(line)['text']
                except Exception:
                    continue
                if text.endswith(EOD_TEXT):
                    text = text[:-len(EOD_TEXT)]
                for p in text.split('\n\n'):
                    p = p.strip()
                    if min_chars <= len(p) <= max_chars:
                        buf.append(p)
                if len(buf) >= buf_n:
                    rng.shuffle(buf)
                    for p in buf:
                        yield p, epoch
                    buf = []
        rng.shuffle(buf)
        for p in buf:
            yield p, epoch


class QAPool:
    """Draw QA items evenly (shuffled, epoch-reshuffled) from both sources."""
    def __init__(self, squad, hotpot, rng):
        self.pools = {'squad_v2': squad, 'hotpot_qa': hotpot}
        self.order = [('squad_v2', i) for i in range(len(squad))] + \
                     [('hotpot_qa', i) for i in range(len(hotpot))]
        self.rng = rng
        self.rng.shuffle(self.order)
        self.pos, self.epoch = 0, 1
        self.used = {'squad_v2': 0, 'hotpot_qa': 0}

    def draw(self, k):
        out = []
        for _ in range(k):
            if self.pos >= len(self.order):
                self.rng.shuffle(self.order)
                self.pos, self.epoch = 0, self.epoch + 1
            src, i = self.order[self.pos]; self.pos += 1
            self.used[src] += 1
            out.append(self.pools[src][i])
        return out


def gen_docs(pool, distractors, rng, stats):
    """Yield assembled synthetic documents (EOD appended as text)."""
    while True:
        target_chars = rng.uniform(8192, 32768) * CHARS_PER_TOKEN
        items = pool.draw(rng.randint(3, 8))
        evid, seen = [], set()
        for _, _, paras in items:
            for p in paras:
                if p not in seen:
                    seen.add(p); evid.append(p)
        qa = '\n\n'.join(f'Question: {q}\nAnswer: {a}' for q, a, _ in items)
        used = sum(len(p) + 2 for p in evid) + len(qa) + 2
        paras = list(evid)
        while used < target_chars:
            p, ep = next(distractors)
            stats['wiki_epoch'] = ep
            paras.append(p); used += len(p) + 2
            stats['distractor_paras'] += 1
        rng.shuffle(paras)
        stats['qa_items'] += len(items)
        yield '\n\n'.join(paras) + '\n\n' + qa + EOD_TEXT


def do_jsonl(raw, out_jsonl, target_tokens, workers, seed):
    rng = random.Random(seed)
    squad, n_unans = load_squad(raw)
    hotpot = load_hotpot(raw)
    pool = QAPool(squad, hotpot, rng)
    stats = dict(qa_items=0, distractor_paras=0, wiki_epoch=0)
    distractors = wiki_paragraphs(random.Random(seed + 1))
    docs = tok_kept = 0
    hist = {}
    t0 = time.time()
    wpool = mp.Pool(workers, initializer=B._init_tok, initargs=(B.TOKJSON,))
    with open(out_jsonl + '.tmp', 'w', encoding='utf-8') as out:
        for text, n in B.zip_texts_counts(wpool, gen_docs(pool, distractors,
                                                          rng, stats)):
            out.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
            docs += 1; tok_kept += n
            hist[f'{n//4096*4}k'] = hist.get(f'{n//4096*4}k', 0) + 1
            if docs % 1000 == 0:
                r = tok_kept / max(time.time() - t0, 1)
                eta = (target_tokens - tok_kept) / max(r, 1)
                print(f'  docs {docs:,} tokens {tok_kept/1e9:.3f}B '
                      f'({r/1e6:.2f}M tok/s, eta {eta/60:.0f} min)')
                sys.stdout.flush()
            if tok_kept >= target_tokens:
                break
    wpool.terminate()
    os.replace(out_jsonl + '.tmp', out_jsonl)
    write_manifest(out_jsonl, target_tokens, seed, docs, tok_kept, hist,
                   pool, stats, n_unans, len(squad), len(hotpot))
    print(f'[jsonl] {docs:,} docs, {tok_kept/1e9:.3f}B tokens')


def write_manifest(out_jsonl, target_tokens, seed, docs, tok_kept, hist,
                   pool, stats, n_unans, n_squad, n_hotpot):
    cfg = dict(source='qaembed', target_tokens=target_tokens, seed=seed,
               doc_len_tokens=[8192, 32768], qa_per_doc=[3, 8],
               chars_per_token=CHARS_PER_TOKEN, eod='as-text')
    man = dict(
        fingerprint=hashlib.sha1(json.dumps(cfg, sort_keys=True)
                                 .encode()).hexdigest()[:12],
        config=cfg,
        construction=(
            'Synthetic long-document QA ("QA embedded in document"). Each '
            'document: 3-8 QA items sampled evenly from SQuAD v2 TRAIN and '
            'HotpotQA TRAIN (fullwiki); their evidence paragraphs (SQuAD '
            'context paragraph; HotpotQA supporting-fact paragraphs) are '
            'interleaved in random order with distractor paragraphs sampled '
            'from the existing mix16k wiki.jsonl (trailing EOD stripped on '
            'reuse) up to a target length ~U[8192,32768] tokens '
            '(chars/3.7 approximation during assembly); a QA section '
            '("Question: ...\\nAnswer: ..." per item) is appended at the end. '
            'SQuAD v2 unanswerable items are kept with the literal answer '
            '"unanswerable" (their context paragraph is still embedded). '
            'One synthetic document = one jsonl line, EOD appended as text.'),
        contamination_statement=(
            'TRAIN splits only. RULER qa_1 (SQuAD) and qa_2 (HotpotQA) build '
            'their eval items from the dev/validation splits of these same '
            'datasets; using only the train splits therefore cannot leak '
            'RULER eval items into training. Distractors come from the '
            'wikimedia/wikipedia 20231101.en dump already used for the '
            'mix16k wiki corpus.'),
        evidence_basis=[
            'Nemotron Nano 2: 20% long-document-QA data in its long-context '
            'phase',
            'MiniMax-01: mixes 10% long-QA in the last 20% of each training '
            'stage',
            'OLMo 3: synthetic QA-in-doc long-context data (CWE/REX)',
            'arXiv:2502.15592: long-context instructions must require '
            'in-context knowledge to answer, and include distractors'],
        per_source_counts=dict(
            squad_v2_train_pool=n_squad,
            squad_v2_unanswerable_in_pool=n_unans,
            hotpot_qa_fullwiki_train_pool=n_hotpot,
            qa_items_used=stats['qa_items'],
            qa_items_used_by_source=dict(pool.used),
            qa_pool_epochs=pool.epoch,
            wiki_distractor_paragraphs=stats['distractor_paras'],
            wiki_stream_epochs=stats['wiki_epoch']),
        totals=dict(docs=docs, tokens=tok_kept,
                    mean_tokens_per_doc=int(tok_kept / max(docs, 1))),
        doc_length_histogram_4k=dict(
            sorted(hist.items(), key=lambda kv: int(kv[0][:-1]))))
    json.dump(man, open(out_jsonl + '.manifest.json', 'w'), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target-tokens', type=float, required=True)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--seed', type=int, default=20260822)
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'qaembed'))
    ap.add_argument('--stage', default='all',
                    choices=['download', 'jsonl', 'binarise', 'verify', 'all'])
    a = ap.parse_args()
    raw = os.path.join(a.out_dir, 'raw')
    os.makedirs(raw, exist_ok=True)
    out_jsonl = os.path.join(a.out_dir, 'qaembed.jsonl')
    prefix = os.path.join(a.out_dir, 'qaembed-deepseek')
    stages = (['download', 'jsonl', 'binarise', 'verify']
              if a.stage == 'all' else [a.stage])
    for s in stages:
        print(f'===== qaembed: {s} ====='); sys.stdout.flush()
        if s == 'download':
            do_download(raw)
        elif s == 'jsonl':
            do_jsonl(raw, out_jsonl, a.target_tokens, a.workers, a.seed)
        elif s == 'binarise':
            B.do_binarise('qaembed', out_jsonl, prefix, max(a.workers, 16))
        elif s == 'verify':
            B.do_verify('qaembed', dict(min_tokens=4096), prefix)
    print(f'DONE qaembed: {prefix}_text_document')


if __name__ == '__main__':
    main()
