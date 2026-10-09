#!/usr/bin/env python3
"""make_subsets.py — generate the FIXED, SEEDED, STRATIFIED doc-index manifests for
fast-cadence suites (instruct-short, longctx-short). Login node, eval env python.

Answers the "your subset is not representative" critique by construction: indices are
drawn by seeded simple random sampling WITHOUT replacement, independently per subtask
(groups like bbh/minerva_math are therefore stratified by their own structure), from
the full doc range of each task. The manifests are committed under config/subsets/ so
the exact evaluation set is auditable and identical for every checkpoint forever.
Regenerating with the same seed reproduces them bit-for-bit; changing n or seed is a
VISIBLE diff in the repo.

    ROCM_PATH=/opt/rocm-default $ENV_PY make_subsets.py [--seed 20260814]

Output: config/subsets/<row_id>.json  = {"_meta": {...}, "<task_or_subtask>": [ids]}
"""
import argparse
import json
import os
import hashlib
import random
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(EVAL_DIR, 'config', 'subsets')

# row_id -> (top-level lm-eval task, docs per (sub)task[, TaskManager metadata file
#            in config/ -- RULER constructs its docs FROM the metadata (tokenizer +
#            max_seq_lengths), so the file is part of the manifest's identity])
PLAN = {
    'gsm8k_chat_s': ('gsm8k', 100),
    'hellaswag_s': ('hellaswag', 100),
    'math_s': ('minerva_math', 7),
    'ifeval_s': ('ifeval', 40),
    'bbh_s': ('bbh', 8),
    'humaneval_s': ('humaneval', 80),
    'mbpp_chat_s': ('mbpp_chat', 80),   # custom task (custom_tasks/mbpp_chat); fresh
                                        # seeded draw -- no mbpp_s to mirror, unlike
                                        # humaneval_chat_s which reuses humaneval_s indices
    'triviaqa_s': ('triviaqa', 500),
    # longctx-short: RULER builds DIFFERENT docs per length, so each length gets its
    # OWN manifest, built under that length's metadata. The 13 subtask names (the
    # keys lm-eval matches samples= against) are identical at every length, so the
    # seeded per-name draw picks the SAME index list per length -- paired subsets
    # across lengths by construction; the per-length doc_sha pins what those indices
    # mean at that length. 13 x 100 = 1300 docs per length (NUM_SAMPLES=500 upstream).
    'ruler4k_s':  ('ruler', 100, 'ruler4k_metadata.json'),
    'ruler8k_s':  ('ruler', 100, 'ruler8k_metadata.json'),
    'ruler16k_s': ('ruler', 100, 'ruler16k_metadata.json'),
    # 32k cadence set (2026-08-22): 25 docs/subtask -- at the ~2x-per-doubling
    # slowdown (16k measured 1.66 docs/min) 100 docs would be ~13h/shard-free run.
    'ruler32k_s': ('ruler', 25, 'ruler32k_metadata.json'),
    # 2026-08-29: the 32k stage evaluated at FULL RULER size, and a first 64k point.
    # Same seeded per-name draw as every other length, so 32k_f's 100/sub-task is a
    # strict superset of 32k_s's 25 and the two are comparable by construction.
    'ruler32k_f': ('ruler', 100, 'ruler32k_metadata.json'),
    'ruler64k_s': ('ruler', 10,  'ruler64k_metadata.json'),
    # LongBench English length-bucketed group: 9 leaf subtasks, 50 each (all if fewer)
    'longbench_s': ('longbench_e', 50),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seed', type=int, default=20260814)
    ap.add_argument('--only', metavar='IDS', default=None,
                    help='comma-separated PLAN row ids to build/verify (default: all). '
                         'The ruler rows rebuild their docs each run -- ruler16k_s '
                         'alone takes several minutes.')
    ap.add_argument('--verify', action='store_true',
                    help='Do not write anything: recompute each committed manifest '
                         '(same seed) AND the content hash of the docs its indices '
                         'select, and compare both against the files on disk. Run '
                         'after lm-eval/datasets upgrades: index manifests are '
                         'positional, so a silent dataset reorder would change WHICH '
                         'docs the indices mean — this catches it.')
    args = ap.parse_args()

    os.environ.setdefault('HF_DATASETS_CACHE',
                          os.path.join(EVAL_DIR, 'cache', 'hf_datasets_cache'))
    os.environ.setdefault('HF_ALLOW_CODE_EVAL', '1')   # humaneval task build requires it
    # RULER builds tokenize with the model tokenizer (a local dir -- never the hub) and
    # read the squad/hotpot corpora from our qa_utils disk cache; same env as the job.
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('LM_EVAL_RULER_CACHE', os.path.join(EVAL_DIR, 'cache', 'ruler'))
    import numpy as np
    from lm_eval.defaults import DEFAULT_OTHER_SEED, DEFAULT_RANDOM_SEED
    from lm_eval.tasks import TaskManager
    tm = TaskManager(include_path=os.path.join(EVAL_DIR, 'custom_tasks'))

    plan = PLAN
    if args.only:
        ids = [x for x in args.only.split(',') if x]
        unknown = sorted(set(ids) - set(PLAN))
        if unknown:
            ap.error(f'unknown row id(s) {unknown}; known: {list(PLAN)}')
        plan = {rid: PLAN[rid] for rid in ids}

    os.makedirs(OUT_DIR, exist_ok=True)
    fails = []
    for rid, spec in plan.items():
        top, k = spec[0], spec[1]
        meta_file = spec[2] if len(spec) > 2 else None
        # Rebuild each row under the EXACT global-RNG state its eval job will have:
        # simple_evaluate() seeds python/numpy randomness before constructing tasks,
        # and RULER's generators draw doc content from those global streams. One
        # re-seed per row == one simple_evaluate call per job (each ruler length IS
        # its own job). Instruct rows read docs from datasets, so this is a no-op
        # for them -- the committed manifests are unchanged.
        random.seed(DEFAULT_RANDOM_SEED)
        np.random.seed(DEFAULT_OTHER_SEED)
        tm_row = tm
        if meta_file:
            meta = json.load(open(os.path.join(EVAL_DIR, 'config', meta_file)))
            tm_row = TaskManager(include_path=os.path.join(EVAL_DIR, 'custom_tasks'),
                                 metadata=meta)
        loaded = tm_row.load_task_or_group(top)
        # flatten: a group loads {name: task-or-subdict}
        flat = {}
        def _walk(d):
            for name, t in d.items():
                if isinstance(t, dict):
                    _walk(t)
                else:
                    flat[name] = t
        _walk(loaded)

        manifest, hashes = {}, {}
        for name in sorted(flat):
            t = flat[name]
            docs = list(t.test_docs() if t.has_test_docs() else t.validation_docs())
            n_total = len(docs)
            # independent, deterministic stream per (seed, task-name): adding or
            # reordering PLAN entries never changes another row's draw
            rng = random.Random(f'{args.seed}:{name}')
            take = min(k, n_total)
            idx = sorted(rng.sample(range(n_total), take))
            manifest[name] = idx
            # content fingerprint of the SELECTED docs: indices are positional, so
            # this pins what they point at; any dataset reorder/upgrade flips it
            h = hashlib.sha256()
            for i in idx:
                h.update(repr(docs[i]).encode())
            hashes[name] = h.hexdigest()[:16]
        total = sum(len(v) for v in manifest.values())
        out = {'_meta': {'row_id': rid, 'task': top, 'per_subtask': k,
                         'total_docs': total, 'seed': args.seed,
                         'doc_sha': hashes,
                         'sampling': 'seeded SRS w/o replacement, per subtask '
                                     '(stratified for groups)'}}
        if meta_file:
            out['_meta']['task_metadata'] = meta_file
        out.update(manifest)
        path = os.path.join(OUT_DIR, f'{rid}.json')
        if args.verify:
            old = json.load(open(path))
            ok_idx = all(old.get(nm) == manifest[nm] for nm in manifest)
            ok_sha = old.get('_meta', {}).get('doc_sha') == hashes
            status = 'OK' if (ok_idx and ok_sha) else \
                ('INDEX DRIFT' if not ok_idx else 'DOC-CONTENT DRIFT')
            print(f'{rid:<16} {status}')
            if status != 'OK':
                fails.append(rid)
        else:
            with open(path, 'w') as f:
                json.dump(out, f, indent=1)
            print(f'{rid:<16} {total:>4} docs over {len(manifest)} subtask(s) -> {path}')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
