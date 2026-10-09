#!/usr/bin/env python
# coding=utf-8
"""Durable, provenance-stamped evaluation result registry.

WHY THIS EXISTS
---------------
Before this module, every eval run dropped a `results.json` inside its own job log dir. Two problems:

  1. No aggregate view. Comparing checkpoints x tasks x settings meant opening many files by hand.
  2. No provenance. A number on disk did not say WHICH code produced it. When the RoPE bf16 bug was
     fixed, pre-fix and post-fix results were indistinguishable, and the lm-eval response cache --
     keyed only on checkpoint mtime -- happily served stale pre-fix logprobs until someone remembered
     to `rm` the .db by hand.

The fix for both is one idea: a CODE FINGERPRINT. Hash the sources that can change a forward pass,
then (a) fold it into the response-cache key so a code change auto-invalidates the cache, and (b)
stamp it on every recorded result so any number is attributable to an exact forward.

DESIGN CONSTRAINTS (so this stays a foundation, not a one-off)
-------------------------------------------------------------
  * Pure stdlib. No torch, no Megatron, no lm-eval imports -- so training code, a future generation
    path, or an offline analysis script can all emit into the same registry.
  * Append-only JSONL. Concurrent jobs never rewrite shared state, so nothing can be corrupted or
    lost; history is preserved rather than overwritten.
  * Every function degrades gracefully (missing files, no git, unreadable registry) and NEVER raises
    into the caller's eval loop. Losing a bookkeeping row must not lose a multi-hour eval.

Consumed by: tasks/eval_harness/evaluate_v04.py (writer) and eval/report.py (reader).
"""

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone

# Sources whose contents can change the numeric result of a forward pass. Keep this list SMALL and
# meaningful: adding unrelated files causes spurious cache invalidation; omitting a real one lets a
# stale cache survive a semantic change. Paths are repo-root-relative; missing files are skipped and
# reported so the fingerprint stays computable on a partial checkout.
FORWARD_SOURCES = (
    'megatron/model/rotary_pos_embedding.py',   # RoPE (the bf16 position bug lived here)
    'megatron/model/transformer.py',            # attention + MoE wiring
    'megatron/model/language_model.py',         # rotary sizing, embedding, output layer
    'megatron/model/gpt_model.py',              # forward entry / curriculum plumbing
    'megatron/model/deepseek_v3/attention.py',  # the latent attention and its inference cache (2026-09-14: the latent form)
    'megatron/model/deepseek_v3/rotary.py',     # YaRN rotary of the latent attention
    # Forward-math-relevant harness code only. main.py / probes.py / the evaluate_v04.py shim are
    # deliberately EXCLUDED: orchestration changes must not invalidate response caches.
    'tasks/eval_harness/megatron_ds_eval/lm_eval_adapter.py',  # scoring, padding, decode
    'tasks/eval_harness/megatron_ds_eval/loader.py',           # decides which bytes become weights
)

# RULER's leaf tasks live in TWO name families; its lm-eval group task is 'ruler'.
# Any own-keys rule of the shape `k == task or k.startswith(task + '_')` therefore
# counts only the ruler_* five of RULER's thirteen leaves (500 docs of 1300), so a
# finished RULER could never satisfy a done-check and all four 2026-08 cells had to be
# retired by hand. The constant lives HERE -- not in dispatch/catalog.py, which is
# scheduled for archive -- because main.py, eval_lib.py and the reporting scripts all
# import this module already, and this module is deliberately outside FORWARD_SOURCES
# (no fingerprint cost). Every own-keys rule in the tree points at own_manifest_keys()
# below; do not clone the rule again.
RULER_PREFIXES = ('niah_', 'ruler_')


# --------------------------------------------------------------------------------------------
# EXPERIMENT IDENTITY -- the one rule for "may this row be published?"
# --------------------------------------------------------------------------------------------
# A run declared `experiment=<tag>` is a run whose purpose is to COMPARE, not to publish: a
# shape A/B, a throughput probe, a gate re-run. Before 2026-08-28 the only thing such a run
# changed was its answer FILENAME (`answers_ns=<tag>` -> a `_ns<tag>` suffix); the registry row
# it wrote was an ordinary production row. So the isolation was cosmetic, and five experiment
# jobs silently redefined published cells -- openbookqa on sft-16k15b went 0.336 (job 5340728,
# the published value) -> 0.352 (job 5370275, experiment `wq_equiv`) -> 0.356 (job 5370346,
# experiment `wq_ref`). See eval/rebuild/ANSWER_NAMESPACE_DESIGN.md.
#
# The fix is that the EXPERIMENT IS PART OF THE ROW'S IDENTITY, and one function decides it for
# every consumer -- the done-check, the report builder, the dashboard, the rescore/significance
# tooling. There is no per-consumer variant of this rule; adding one is how the five rows got in.
EXPERIMENT_FIELD = 'experiment'

# Answer files of experiment runs are named `..._ns<tag>.jsonl` (megatron_ds_eval/cache.py). That
# suffix is a NAME, not the identity -- but it is the only trace on rows written before the field
# existed, so it is read as a fallback. The fallback can only ever ADD quarantine, never remove
# it, which is what makes it safe to leave in place permanently.
_LEGACY_NS_RE = re.compile(r'_ns(?P<tag>[A-Za-z0-9][A-Za-z0-9._-]*)\.jsonl$')


def experiment_of(row):
    """The experiment tag this row was produced under, or None when it is a production row.

    THE rule. `None` means "this row may define the published value of its cell"; any string
    means "this row is a measurement inside experiment <tag> and must never be selected as the
    published value, at any consumer".

    Two sources, in order:
      1. the row's own `experiment` field -- the identity, written by main.py since 2026-08-28;
      2. failing that, an `_ns<tag>` suffix on the row's `response_cache` filename -- the only
         evidence rows written before the field carry. Backfilled rows carry both.
    """
    row = row or {}
    tag = row.get(EXPERIMENT_FIELD)
    if tag not in (None, ''):
        return str(tag)
    m = _LEGACY_NS_RE.search(str(row.get('response_cache') or ''))
    return m.group('tag') if m else None


def is_production(row):
    """True when this row is eligible to be the published value of its cell."""
    return experiment_of(row) is None


def own_manifest_keys(task, keys):
    """The keys of a (possibly shared) subset manifest that belong to `task`.

    The general rule: a task owns its own name and its prefixed leaves (bbh ->
    bbh_cot_fewshot_*, minerva_math -> minerva_math_*), which is what splits a manifest
    genuinely shared across a multi-task job (short_q512_fs0.json: gsm8k + hellaswag).
    RULER is the one special case: its group task is 'ruler' and its leaves span BOTH
    families in RULER_PREFIXES -- the prefix rule alone sees ruler_vt but never
    niah_single_1. '_meta' is the manifest's provenance header, never a leaf.
    Returns the matching keys in their input order.
    """
    def _owns(k):
        if k == task or k.startswith(task + '_'):
            return True
        return task == 'ruler' and k.startswith(RULER_PREFIXES)
    return [k for k in keys if k != '_meta' and _owns(k)]


REGISTRY_RELPATH = os.path.join('examples_elmoe', 'scripts-frontier', 'eval',
                                'results', 'registry.jsonl')

# Per-training-run shards (2026-08 redesign). New rows go to registry/<ckpt_run>.jsonl when
# the run is known; the legacy single file above stays frozen read-only so history is never
# rewritten. Readers union everything via load_all_rows().
REGISTRY_DIR_RELPATH = os.path.join('examples_elmoe', 'scripts-frontier', 'eval',
                                    'results', 'registry')


def shard_path(repo_root, ckpt_run):
    """Where a new row belongs: the run's shard, or the legacy file when run is unknown."""
    if ckpt_run:
        safe = str(ckpt_run).replace(os.sep, '_')
        return os.path.join(repo_root, REGISTRY_DIR_RELPATH, f'{safe}.jsonl')
    return os.path.join(repo_root, REGISTRY_RELPATH)


def load_all_rows(repo_root, include_partial=False):
    """Union view over the legacy registry and every shard. Same tolerance as load_rows.

    PARTIAL ROWS ARE DROPPED BY DEFAULT (E1, 2026-08-30). Every worker of a sliced or
    dispatch-sharded job writes its own row stamped `partial` ('shard 11/16', 'slice 3/4' --
    main.py:603,:617) scored against the FULL document denominator. Averaging those in with the
    collector's row is not a small error: a healthy RULER-32k cell of 0.6728 reads 0.0796 when the
    sixteen shard rows are included, and that already misled one analysis into reporting a working
    model as a catastrophic zero.

    The filter lives HERE, at the single aggregation chokepoint, rather than at each call site,
    because the failure mode is a caller forgetting -- which is exactly what happened. Callers that
    genuinely want the shard rows (rescore_partial's recovery accounting) pass include_partial=True.
    load_rows() is untouched: it stays the raw reader.
    """
    import glob as _glob
    paths = [os.path.join(repo_root, REGISTRY_RELPATH)]
    paths += sorted(_glob.glob(os.path.join(repo_root, REGISTRY_DIR_RELPATH, '*.jsonl')))
    rows = []
    for p in paths:
        rows.extend(load_rows(p))
    if not include_partial:
        rows = [r for r in rows if not r.get('partial')]
    return rows


# --------------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------------
def infer_repo_root(start_file):
    """Repo root from a file inside tasks/eval_harness/ (3 levels up). Falls back to cwd."""
    try:
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(start_file))))
    except Exception:
        return os.getcwd()


def code_fingerprint(repo_root, sources=FORWARD_SOURCES, length=8):
    """sha1 over the forward-relevant sources -> short hex string.

    Returns (fingerprint, detail) where detail lists which files were hashed vs missing, so a
    surprising fingerprint change can always be explained. Deterministic: files are hashed in the
    declared order, each prefixed by its path so a rename alone changes the hash.
    """
    h = hashlib.sha1()
    hashed, missing = [], []
    for rel in sources:
        path = os.path.join(repo_root, rel)
        try:
            with open(path, 'rb') as f:
                data = f.read()
        except OSError:
            missing.append(rel)
            continue
        h.update(rel.encode('utf-8'))
        h.update(data)
        hashed.append(rel)
    return h.hexdigest()[:length], {'hashed': hashed, 'missing': missing}


def git_info(repo_root):
    """{'sha': short-sha or None, 'dirty': bool}. Never raises; git may be absent on a compute node."""
    def _run(cmd):
        try:
            out = subprocess.run(cmd, cwd=repo_root, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, timeout=10)
            return out.stdout.decode('utf-8', 'replace').strip() if out.returncode == 0 else None
        except Exception:
            return None

    sha = _run(['git', 'rev-parse', '--short', 'HEAD'])
    status = _run(['git', 'status', '--porcelain'])
    return {'sha': sha, 'dirty': bool(status) if status is not None else None}


def provenance(repo_root):
    """One call for everything identity-related about this code state."""
    fp, detail = code_fingerprint(repo_root)
    g = git_info(repo_root)
    return {'code_fp': fp, 'code_fp_detail': detail, 'git_sha': g['sha'], 'git_dirty': g['dirty']}


# --------------------------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------------------------
def _flatten_task_metrics(task_results):
    """lm-eval 0.4 emits keys like 'acc,none' / 'acc_stderr,none' (and a string 'alias').

    Returns [(metric_name, value, stderr_or_None), ...] with stderr paired to its base metric.
    """
    metrics, stderrs = {}, {}
    for key, val in (task_results or {}).items():
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue                      # skip 'alias' strings, nested dicts, etc.
        base, _, filt = str(key).partition(',')
        # Preserve non-trivial filter names: gsm8k emits BOTH 'exact_match,strict-match'
        # and 'exact_match,flexible-extract' -- collapsing to 'exact_match' silently keeps
        # whichever came last. 'none' (the common case) stays suffix-free: 'acc,none'->'acc'.
        suffix = '' if filt in ('', 'none') else f'/{filt}'
        if base.endswith('_stderr'):
            stderrs[base[:-len('_stderr')] + suffix] = float(val)
        else:
            metrics[base + suffix] = float(val)
    return [(m, v, stderrs.get(m)) for m, v in sorted(metrics.items())]


def build_rows(run_meta, results_by_task):
    """Cross-product run metadata with (task, metric) -> list of flat JSON-able rows."""
    ts = datetime.now(timezone.utc).isoformat(timespec='seconds')
    rows = []
    for task, task_results in sorted((results_by_task or {}).items()):
        for metric, value, stderr in _flatten_task_metrics(task_results):
            row = {'ts': ts, 'task': task, 'metric': metric, 'value': value, 'stderr': stderr}
            row.update(run_meta)
            rows.append(row)
    return rows


def append_rows(registry_path, rows):
    """Append rows as JSONL. Single write call to minimise interleaving under concurrent jobs.

    Returns the number of rows written (0 on any failure -- bookkeeping must never break an eval).
    """
    if not rows:
        return 0
    try:
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        payload = ''.join(json.dumps(r, default=str) + '\n' for r in rows)
        with open(registry_path, 'a') as f:
            f.write(payload)
        return len(rows)
    except Exception:
        return 0


def record(registry_path, run_meta, results_by_task):
    """Convenience: build + append. Returns rows written."""
    return append_rows(registry_path, build_rows(run_meta, results_by_task))


# --------------------------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------------------------
def load_rows(registry_path):
    """Read JSONL, skipping any malformed line (a partially-written row must not break reporting)."""
    rows = []
    try:
        with open(registry_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows
