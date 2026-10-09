#!/usr/bin/env python3
"""eval_lib.py -- everything autorun_eval.sh needs to turn a declaration into jobs.

    autorun_eval.sh  ->  frontier_eval.slurm.template  ->  evaluate_v04.py

autorun_eval.sh owns the declaration block and nothing else; this module owns the
logic. Pure stdlib, login-node python3 -- no torch, no lm_eval, no GPU -- so every
function here is unit-testable on a login node (tests/test_eval_lib.py).

WHAT IT REUSES RATHER THAN REINVENTS
  task_table.py + config/custom_suites.py   the task settings table (seqlen, shots,
        micro-batch, generation budget, extra harness flags). Composed via
        task_table.load_custom(); neither file stands alone and neither is edited here.
  result_registry.py                        row identity, the code fingerprint, and
        shard_path/load_all_rows for judging what is already done.
  megatron_ds_eval/cache.py                 the answer file. answer_file_name() below
        mirrors ResponseCache.build()'s filename rule exactly; that rule IS the cache's
        identity, and slicing depends on every slice landing in the same file.
  frontier_eval.slurm.template              rendered by substitution, with the
        unfilled-placeholder guard that submit_eval.py introduced.

SLICING (ACCEPTANCE.md section 4)
  A slice is a job that answers part of a benchmark into the SHARED answer file.
  * disjoint by construction -- share_ranges()/round_robin_positions() partition the
    ordered question list arithmetically, so no job coordinates with another.
  * question-granular resume -- the answer file is keyed by request CONTENT, not by
    document index, so re-submitting an interrupted slice recomputes only what is
    missing. leftover_manifest() states that arithmetic explicitly, for the tests and
    for any caller that wants the remainder as data; the resume itself needs no
    bookkeeping because the cache already knows what it holds.
  * the final score comes from a COLLECTING run: one more job, dependent on the slices,
    running the unsliced configuration. Every request is a cache hit, so it scores the
    complete set in minutes. That is what keeps Gate 1 satisfiable.
  * slices=1 emits exactly one job with no manifest and no collector -- byte-identical
    to today's single job.

EXPERIMENTS (`experiment=<tag>`, rebuild/ANSWER_NAMESPACE_DESIGN.md)
  A run declared `experiment=<tag>` is a measurement made to COMPARE, not to publish.
  The tag is part of the row's identity everywhere -- the answer file it writes, the
  registry rows it records, the done-check that decides whether to submit it, and every
  reader that selects a published number. So an experiment can neither overwrite a
  published cell nor mark production's work done, and the control arm of an A/B is a
  separate identity that needs no --force. `answers_ns=` is the deprecated spelling.
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Optional

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(EVAL_DIR) == 'rebuild':          # pre-cutover staging location
    EVAL_DIR = os.path.dirname(EVAL_DIR)
SCRIPTS_DIR = os.path.dirname(EVAL_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPTS_DIR))

sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'tasks', 'eval_harness'))
sys.path.insert(0, os.path.join(SCRIPTS_DIR, '..', 'utils'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'examples_elmoe', 'utils'))

import task_table                                    # noqa: E402
import result_registry                               # noqa: E402
import model_registry                                # noqa: E402

TEMPLATE = os.path.join(EVAL_DIR, 'frontier_eval.slurm.template')
TEMP_SLURM = os.path.join(EVAL_DIR, 'temp_slurm')
LEDGER = os.path.join(EVAL_DIR, 'results', 'submissions.jsonl')
SUBSETS_DIR = os.path.join(EVAL_DIR, 'config', 'subsets')
SLICE_DIR = os.path.join(SUBSETS_DIR, 'slices')      # generated slice manifests
HF_DS_CACHE = os.path.join(EVAL_DIR, 'cache', 'hf_datasets_cache')
HARNESS_MAIN = os.path.join(REPO_ROOT, 'tasks', 'eval_harness', 'megatron_ds_eval',
                            'main.py')
CHECKPOINT_DIR = os.path.join(SCRIPTS_DIR, 'checkpoint')
GPUS_PER_NODE = 8                                    # Frontier: 8 GCDs per node

_ANY = object()                                      # "do not compare this field"


class BadParameter(ValueError):
    """One declared parameter is wrong. The message always names it."""

    def __init__(self, parameter, message):
        self.parameter = parameter
        super().__init__(f'{parameter}: {message}')


# =====================================================================================
# Slice arithmetic -- pure, no I/O
# =====================================================================================
def share_ranges(n_docs, n_slices):
    """Split range(n_docs) into n_slices contiguous, non-empty, balanced shares.

    Returns [(lo, hi), ...] with hi exclusive. Disjoint and covering by construction.
    """
    n_docs, n_slices = int(n_docs), int(n_slices)
    if n_slices < 1:
        raise BadParameter('slices', f'must be >= 1, got {n_slices}')
    if n_docs < 1:
        raise BadParameter('slices', f'nothing to slice: {n_docs} documents')
    if n_slices > n_docs:
        raise BadParameter('slices', f'{n_slices} slices over {n_docs} documents would '
                                     f'leave empty jobs; use slices <= {n_docs}')
    # Front-loaded: the first `rem` shares get one extra document, so no later share
    # is ever larger than an earlier one and share 1 of 4 is the first quarter.
    size, rem = divmod(n_docs, n_slices)
    out, lo = [], 0
    for k in range(n_slices):
        hi = lo + size + (1 if k < rem else 0)
        out.append((lo, hi))
        lo = hi
    return out


def round_robin_positions(n_docs, k, n_slices):
    """Positions of slice k (1-based) under the round-robin layout: k-1, k-1+S, ...

    This is the layout the committed production shard manifests
    (config/subsets/ruler8k_f<k>of8.json) already use; it spreads each slice evenly
    over the document order instead of giving one slice the whole tail.
    """
    _check_slice(k, n_slices)
    if n_slices > n_docs:
        raise BadParameter('slices', f'{n_slices} slices over {n_docs} documents would '
                                     f'leave empty jobs; use slices <= {n_docs}')
    return list(range(k - 1, n_docs, n_slices))


def _check_slice(k, n_slices):
    """Guard shared by every sliced accessor: k must name a real slice of n_slices."""
    if n_slices < 1:
        raise BadParameter('slices', f'must be >= 1, got {n_slices}')
    if not 1 <= k <= n_slices:
        raise BadParameter('slices', f'slice {k} is outside 1..{n_slices}')


def limit_manifest(subtasks, n_docs):
    """The ordered question list of a --limit N run: docs 0..N-1 of every sub-task.

    lm-eval's limit takes the FIRST N documents per task (api/task.py doc_iterator ->
    create_iterator -> islice), so this manifest selects exactly the same documents
    that --limit N would.
    """
    if n_docs is None or int(n_docs) < 1:
        raise BadParameter('limit', f'need a positive document count, got {n_docs!r}')
    return {t: list(range(int(n_docs))) for t in subtasks}


def slice_manifest(parent, k, n_slices, layout='share'):
    """Slice k (1-based) of a per-sub-task manifest, sliced INSIDE each sub-task.

    Slicing within each sub-task rather than across them keeps every slice stratified
    the same way the parent is, so a slice that never runs costs coverage of every
    sub-task equally instead of wiping one out.
    """
    _check_slice(k, n_slices)
    out = {}
    for task in sorted(parent):
        idx = list(parent[task])
        if layout == 'round_robin':
            take = [idx[p] for p in round_robin_positions(len(idx), k, n_slices)]
        elif layout in ('share', 'contiguous'):    # 'contiguous': undocumented alias
            lo, hi = share_ranges(len(idx), n_slices)[k - 1]
            take = idx[lo:hi]
        else:
            raise BadParameter('layout', f'{layout!r} is not a share layout; use '
                                         f'"share" or "round_robin"')
        if take:
            out[task] = take
    return out


def lpt_bins(names, sizes, n_bins):
    """Longest-Processing-Time bin-pack: `names` into `n_bins` lists, balanced by size.

    Equal-COUNT bundles of sub-tasks are badly imbalanced when the sub-tasks are not
    equal-sized: measured on the 57 one-subtask MMLU jobs, max/mean job duration was
    3.01 (base TP1) and 3.68 (lc16k TP8) purely because the natural subtask ranges
    100..1534 documents (rebuild/DDP_EVAL_DISTRIBUTION.md section 2.3). Document count
    predicts duration with R2 = 0.999 on the tp8 fleet (ibid.), so a static LPT pack on
    document count captures essentially all of what a dynamic scheduler could.

    Deterministic: names are visited in descending size with the name as tie-breaker,
    and each goes to the currently-lightest bin (index as tie-breaker). Disjoint and
    covering by construction -- every name is placed exactly once.
    """
    n_bins = int(n_bins)
    names = list(names)
    if n_bins < 1:
        raise BadParameter('workers', f'must be >= 1, got {n_bins}')
    if n_bins > len(names):
        raise BadParameter('workers', f'{n_bins} bins over {len(names)} sub-task(s) '
                                      f'would leave empty workers')
    bins = [[] for _ in range(n_bins)]
    load = [0] * n_bins
    for name in sorted(names, key=lambda t: (-int(sizes.get(t, 1)), t)):
        j = min(range(n_bins), key=lambda i: (load[i], i))
        bins[j].append(name)
        load[j] += int(sizes.get(name, 1))
    return [sorted(b) for b in bins]


def slice_subtasks(subtasks, k, n_slices, sizes=None):
    """Slice k (1-based) of an ordered sub-task NAME list.

    `sizes` absent (the slices= path, unchanged): a contiguous named range, split by
    NAME COUNT. `sizes` present ({sub-task: documents}, the workers= path): the LPT
    bin-pack above, split by DOCUMENT COUNT. Both are disjoint and covering; only the
    assignment differs, so this stays the single splitter.
    """
    _check_slice(k, n_slices)
    names = list(subtasks)
    if n_slices > len(names):
        raise BadParameter('slices', f'{n_slices} slices over {len(names)} sub-task(s); '
                                     f'use slices <= {len(names)} or slice_by=share')
    if sizes is not None:
        return lpt_bins(names, sizes, n_slices)[k - 1]
    lo, hi = share_ranges(len(names), n_slices)[k - 1]
    return names[lo:hi]


# =====================================================================================
# Leftover detection -- pure
# =====================================================================================
def leftover_manifest(planned, answered):
    """What a rerun still has to do: planned minus answered, at DOCUMENT granularity.

    `planned` is {sub-task: [doc index]}, `answered` is {sub-task: set(doc index)}.
    Sub-tasks that are complete disappear; answers outside the plan are ignored.
    Empty result means the run is finished.
    """
    out = {}
    for task in sorted(planned):
        done = set(answered.get(task, ()))
        rest = [i for i in planned[task] if i not in done]
        if rest:
            out[task] = rest
    return out


def leftover_slices(n_slices, completed):
    """Slice numbers (1-based) that still need a job."""
    return [k for k in range(1, int(n_slices) + 1) if k not in set(completed)]


# =====================================================================================
# Manifests
# =====================================================================================
def read_manifest(path):
    """A samples manifest as {sub-task: [doc index]} -- the _meta header dropped."""
    with open(path) as fh:
        data = json.load(fh)
    data.pop('_meta', None)
    return {k: list(v) for k, v in data.items()}


def write_manifest(path, manifest, meta):
    """Commit a samples manifest, with a _meta header recording where it came from.

    Written to a temp file and renamed, because a queued job may read this path at any
    moment and a half-written manifest is a job that scores the wrong document set.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {'_meta': dict(meta, total_docs=sum(len(v) for v in manifest.values()))}
    out.update({k: manifest[k] for k in sorted(manifest)})
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, path)
    return path


# Both spellings are the same flag (megatron_ds_eval/main.py:161 declares the pair),
# and rows in the task table are written either way, so every reader must accept both.
# Index 0 is the spelling this module WRITES.
_MANIFEST_FLAGS = ('--task-samples-json-file', '--task_samples_json_file')


def manifest_path_in(flags):
    """The --task-samples-json-file path inside a row's extra flags, or None."""
    toks = flags.split()
    for i, tok in enumerate(toks):
        if tok in _MANIFEST_FLAGS and i + 1 < len(toks):
            return toks[i + 1]
    return None


def replace_manifest_in(flags, path):
    """Point a row's extra flags at a different samples manifest (adding it if absent)."""
    toks = flags.split()
    out, i, replaced = [], 0, False
    while i < len(toks):
        if toks[i] in _MANIFEST_FLAGS:
            out += [_MANIFEST_FLAGS[0], path]
            replaced = True
            i += 2
            continue
        out.append(toks[i])
        i += 1
    if not replaced:
        out += [_MANIFEST_FLAGS[0], path]
    return ' '.join(out)


def parent_manifest(row):
    """A row's committed subset manifest as (path, {sub-task: [doc index]}), or (None, None).

    'Parent' because slicing carves this manifest up: the slice manifests are subsets
    of it, and a row that pins its documents this way expresses its size through the
    manifest rather than through --limit (lm-eval refuses both at once).
    """
    path = manifest_path_in(row.extra_eval_flags)
    if path and os.path.isfile(path):
        return path, read_manifest(path)
    return path, None


# =====================================================================================
# Answer-file identity -- mirrors megatron_ds_eval/cache.py ResponseCache.build()
# =====================================================================================
def answer_file_name(run_tag, step, mtime, code_fp, weight_source, padding, seqlen,
                     shots, no_kv_decode=False, limit=None, slice_index=None,
                     slice_count=None, experiment=None, answers_ns=None):
    """The answer file for one (checkpoint x code x eval-config) identity.

    limit / slice_index / slice_count are accepted and DELIBERATELY IGNORED: the cache
    keys requests by content, so a slice, a --limit smoke run and the full run all
    belong in the same file. That is what makes a collecting run cheap and what makes
    a smoke run pre-warm the full one.

    THE PARALLEL SHAPE IS NOT IN THIS NAME, and that is a trap worth stating. This rule
    mirrors cache.py's ResponseCache.build() exactly -- the cache is the authority and a
    name invented here would simply be wrong -- and cache.py does not know the shape. So
    a run at a new shape reads the DEFAULT shape's answers as hits and can score
    identically by construction, while the registry row it writes now claims nodes/ep/
    tp/pp. To measure a new shape against a known one (ACCEPTANCE Gate 2), give it a
    private answer file so it must recompute; a warm 4-node run proves the launcher
    resolved to the same file, nothing more.

    `experiment=<tag>` IS that declaration (deprecated spelling: `answers_ns=`). It
    suffixes `_ns<tag>` here and in cache.py's build(), so such a job is forced cold
    without touching a production file. The SUFFIX SPELLING IS NOT THE IDENTITY -- the
    identity is the `experiment` field the job stamps on every registry row
    (result_registry.experiment_of); `_ns` is kept only so the experiment answer files
    already on disk keep resolving. Unset (the default) keeps every existing name byte
    for byte -- the 429 answer files on disk resolve exactly as before.
    """
    mode = '_nokv' if no_kv_decode else ''
    ns = f'_ns{experiment or answers_ns}' if (experiment or answers_ns) else ''
    return (f'{run_tag}__global_step{step}_m{int(mtime)}_code{code_fp}_{weight_source}'
            f'_pad{padding}_seq{int(seqlen)}_fs{int(shots or 0)}{mode}{ns}.jsonl')


def answer_writer_rank(world_size, pipeline_size):
    """Which world rank appends to the answer file: first rank of the LAST pipeline stage.

    Mirrors megatron_ds_eval/cache.py writer_rank(). The rule is duplicated here only so
    the plan can SAY which rank will write without importing torch on a login node; a
    test loads both and requires them to agree over a grid of topologies, so the two can
    never drift apart silently.
    """
    world_size, pipeline_size = int(world_size or 1), int(pipeline_size or 1)
    if pipeline_size <= 1 or pipeline_size > world_size or world_size % pipeline_size:
        return 0
    return world_size - world_size // pipeline_size


def code_fingerprint():
    """The short code fingerprint that names every answer file, from result_registry.
    An exported REPO_ROOT (a code snapshot the job runs from, 2026-09-17) is fingerprinted
    instead of the live tree, so the plan names the answer file the job will actually use."""
    return result_registry.code_fingerprint(os.environ.get('REPO_ROOT', REPO_ROOT))[0]


def weights_mtime(step_dir):
    """Modification time of the step's weights -- part of the answer-file identity.

    Re-saving a checkpoint at the same step therefore starts a new answer file instead
    of silently mixing answers from two different sets of weights. 0 when the file is
    not there, which only happens on a path validate() has already refused.
    """
    try:
        return int(os.path.getmtime(os.path.join(step_dir, 'mp_rank_00_model_states.pt')))
    except OSError:
        return 0


# =====================================================================================
# Registry read
# =====================================================================================
def default_shape():
    """The parallel shape a declaration gets when it asks for none: 1 node, EP8/TP1/PP1.

    Read off RunSpec's own defaults so the two cannot drift. It is also the shape every
    row now in the registry was produced at, which is what lets a row that carries no
    parallelism fields be read as this shape rather than as unknown.
    """
    d = {f.name: f.default for f in dataclass_fields(RunSpec)}
    return {k: int(d[k]) for k in ('nodes', 'ep', 'tp', 'pp')}


def row_shape(row):
    """The parallel shape a registry row was produced at.

    A row written before the shape was recorded carries none of these fields and is read
    as default_shape(), which is what all of them actually ran at: 3974 historical eval
    jobs at pp=1, and the only tp>1 jobs (5337015, 5336680) produced no rows at all. If
    that assumption is ever wrong the cost is a resubmission -- the answer file turns it
    into a resume -- never a 4-node number credited to a 1-node run.
    """
    d = default_shape()
    return {k: int(row.get(k) or d[k]) for k in ('nodes', 'ep', 'tp', 'pp')}


def rows_cover(rows, run_tag, step, seq_len, tasks, shots, padding, limit=_ANY,
               shape=None, experiment=None):
    """True when a FULL row exists for every task, at this run/step/context length.

    ACCEPTANCE section 3 names run, step, context length and task. Shots and padding
    side are matched too: a 5-shot row is not a 0-shot result and a left-padded row is
    not a right-padded one, so treating them as interchangeable would report a
    different measurement as done. `limit` is compared only when supplied -- the
    planner supplies it to tell a SLICE's rows apart from a collecting run's.

    `experiment` completes the identity. None asks "is PRODUCTION done?", a tag asks
    "is experiment <tag> done?", and the two never answer for each other: an experiment
    cannot mark a production cell done (which would silence real work), and a control
    arm re-measuring what an experiment already ran is a different question that needs
    no --force to ask. The rule itself lives in result_registry.experiment_of -- one
    home, so the done-check, the report builder and the dashboard cannot drift.
    """
    want = set(tasks)
    for row in rows:
        if row.get('partial'):
            continue
        if result_registry.experiment_of(row) != (experiment or None):
            continue
        if str(row.get('ckpt_run')) != str(run_tag):
            continue
        if str(row.get('step')) != str(step):
            continue
        if int(row.get('seq_len') or 0) != int(seq_len):
            continue
        if int(row.get('num_fewshot') or 0) != int(shots or 0):
            continue
        if (row.get('padding_side') or 'right') != padding:
            continue
        if limit is not _ANY and (row.get('limit') or None) != (limit or None):
            continue
        # A result is only comparable within the shape that produced it: a different
        # reduction order and a different batch composition are a different measurement
        # (ACCEPTANCE Gate 2). Compared only when the caller supplies a shape.
        if shape is not None and row_shape(row) != dict(shape):
            continue
        want.discard(row.get('task'))
        if not want:
            return True
    return not want


_LEAF_CACHE = {}


def group_leaves(task, subsets_dir=None):
    """Leaf sub-task names of a group task, learned from the committed manifests.

    RULER writes one row per SUBTASK and never a row named 'ruler' (a benchmark that
    finished therefore looked 0%-scored and was resubmitted forever, 2026-08-27). The
    committed subset manifests already carry that mapping -- their _meta.task names the
    group and their keys are its leaves -- so the names come from data we already have
    rather than from importing lm_eval, which the login node cannot do.
    """
    subsets_dir = subsets_dir or SUBSETS_DIR
    if subsets_dir not in _LEAF_CACHE:
        table = {}
        for path in sorted(glob.glob(os.path.join(subsets_dir, '*.json'))):
            try:
                with open(path) as fh:
                    data = json.load(fh)
            except (OSError, ValueError):
                continue
            top = (data.get('_meta') or {}).get('task')
            if not top:
                continue
            leaves = [k for k in data if k != '_meta']
            if len(leaves) > 1 or (leaves and leaves[0] != top):
                table.setdefault(top, set()).update(leaves)
        _LEAF_CACHE[subsets_dir] = {k: sorted(v) for k, v in table.items()}
    return _LEAF_CACHE[subsets_dir].get(task)


def table_leaves(task):
    """Leaf sub-task names of a group task, learned from the TASK TABLE's own rows.

    The second source of leaf names, used only when no committed manifest names them.
    MMLU is the case that matters: `mmlu_5shot` pins no subset manifest, but the table
    carries one row per subject (mmlu5_sh1_0 -> mmlu_abstract_algebra, ...), so the 57
    leaves are addressable by NAME even though their document indices are not.

    A name-only split is enough for the workers= subtask rule -- each worker runs whole
    sub-tasks via --task_list and never touches a sub-task's document order, which is
    exactly the condition the few-shot rng needs (section C1 of
    rebuild/DDP_EVAL_DISTRIBUTION.md). It is NOT enough for slice_by=share, which needs
    indices; that path keeps its existing refusal.

    Prefix-derived, and deliberately so: if the table were to miss a leaf, the leaf is
    simply not pre-computed by any worker and the collector computes it -- slower, never
    wrong. Returns [] when the task has no leaf rows (RULER's leaves are niah_*/ruler_*,
    not ruler_*-prefixed uniformly, and RULER has manifests anyway).
    """
    tasks, _suites = task_table.load_custom()
    prefix = f'{task}_'
    return sorted({r.task for r in tasks.values()
                   if r.task.startswith(prefix) and r.task != task})


_DOC_COUNT_CACHE = {}


def registry_doc_counts(names):
    """{sub-task: documents} for balancing, measured from the registry's own sample_len.

    BALANCE ONLY. These numbers decide which worker gets which sub-task; they never
    decide which documents are evaluated (that is the manifest's job, or the whole
    sub-task's). A stale or missing count therefore costs balance, never correctness --
    which is why an approximate, self-measured source is the right one here.

    `sample_len` is the per-task document count every finished eval already records
    (main.py writes it beside acc/acc_norm). The MAX over the registry's history is the
    full-benchmark size: a limit=20 smoke run of the same task records 20 as well.
    """
    names = list(names)
    missing = [n for n in names if n not in _DOC_COUNT_CACHE]
    if missing:
        for name in missing:
            _DOC_COUNT_CACHE.setdefault(name, None)
        try:
            rows = registry_rows()
        except Exception:                                # registry unreadable: no balance data
            rows = []
        for row in rows:
            if row.get('metric') != 'sample_len':
                continue
            task = row.get('task')
            if task in _DOC_COUNT_CACHE:
                try:
                    val = int(float(row.get('value')))
                except (TypeError, ValueError):
                    continue
                prev = _DOC_COUNT_CACHE[task]
                _DOC_COUNT_CACHE[task] = val if prev is None else max(prev, val)
    return {n: _DOC_COUNT_CACHE[n] for n in names if _DOC_COUNT_CACHE.get(n)}


def expected_row_tasks(task, manifest=None):
    """Which registry task names a finished benchmark writes.

    A group task writes one row per LEAF (RULER writes niah_*/ruler_*, never 'ruler'),
    and a manifest's keys ARE those leaf names. Without a manifest, fall back to the
    leaves the committed manifests know about, then to the task itself.
    """
    if manifest:
        return sorted(manifest)
    return group_leaves(task) or [task]


# =====================================================================================
# Template rendering
# =====================================================================================
_PLACEHOLDER_RE = re.compile(r'{{([A-Z_]+)}}')


def template_placeholders(path=None):
    """Every slot the template defines. {{PLACEHOLDER}} appears in a comment, not a slot."""
    with open(path or TEMPLATE) as fh:
        return set(_PLACEHOLDER_RE.findall(fh.read())) - {'PLACEHOLDER'}


def render_template(path, subs):
    """Substitute every placeholder. An unfilled or unknown slot is a hard error.

    A silent blank is how a job loses its subset manifest or its generation budget and
    still exits zero, so both directions are checked before a single byte is written.
    Plain string replacement, not sed: a value containing '|' or '&' is safe.
    """
    with open(path) as fh:
        text = fh.read()
    slots = set(_PLACEHOLDER_RE.findall(text)) - {'PLACEHOLDER'}
    unknown = sorted(set(subs) - slots)
    if unknown:
        raise BadParameter(unknown[0], f'no such placeholder in {os.path.basename(path)} '
                                       f'(unknown: {unknown})')
    missing = sorted(slots - set(subs))
    if missing:
        raise BadParameter(missing[0], f'template placeholder not filled '
                                       f'(missing: {missing})')
    for key in sorted(subs):
        text = text.replace('{{%s}}' % key, str(subs[key]))
    left = sorted(set(_PLACEHOLDER_RE.findall(text)) - {'PLACEHOLDER'})
    if left:
        raise BadParameter(left[0], f'placeholder survived rendering: {left}')
    return text


# =====================================================================================
# The declaration
# =====================================================================================
# =====================================================================================
# Rotary base -- REQUIRED, never inferred (2026-08-29)
# =====================================================================================
# The template passes --use-rotary-position-embeddings and used to pass no base, so
# megatron's default of 10,000 (arguments.py:689) applied unless loader.py's
# override_args() happened to copy the checkpoint's own value onto the runtime args.
# That inheritance is IMPLICIT and it demonstrably fails: a scan of 1,964 job logs
# found 65 lc16k-idoc jobs and 5 sft-16k15b jobs that ran at 10,000 against
# checkpoints trained at 100,000 -- all of them on 2026-08-28, interleaved with
# correct-base jobs the same day. Intermittent, not historical.
#
# So the base is now declared, passed explicitly, and the template refuses to start
# without it. Values below are read from each checkpoint's OWN stored
# args.rotary_base (mp_rank_00_model_states.pt), 2026-08-29 -- not from a template,
# a doc, or a recipe.
ROTARY_BASE_BY_RUN = {
    'lc8k-base':       100000.0,
    'lc8k-idoc':       100000.0,
    'lc16k-base':      100000.0,
    'lc16k-idoc':      100000.0,
    'sft-16k15b':      100000.0,
    'lc32k-idoc':     1000000.0,   # the 32k rung retuned the base; 1e5 here costs points
    'deepseek_16b_ds':  10000.0,   # stock HF DeepSeek-MoE-16B, native base, no stored args
    # 2026-08-30 campaign runs, read from each checkpoint's OWN stored args
    # (torch.load(.../mp_rank_00_model_states.pt)['args'].rotary_base), not from the recipe:
    #   lc32k-v2  step 80 -> 1000000.0 at seq 32768
    #   lc128k-v1 step 70 -> 5000000.0 at seq 131072
    'lc32k-v2':       1000000.0,
    'lc128k-v1':      5000000.0,
}


def resolve_rotary_base(spec):
    """The rotary base this run MUST be evaluated at.

    Explicit `rotary_base=` on the declaration wins. Otherwise the run must appear in
    ROTARY_BASE_BY_RUN. An unknown run is refused AT SUBMIT rather than defaulted,
    because the failure mode it prevents is silent: a wrong base does not crash, it
    just scores lower (a matched pair cost 5.44 points of HellaSwag).
    """
    if getattr(spec, 'rotary_base', None):
        return float(spec.rotary_base)
    if spec.run in ROTARY_BASE_BY_RUN:
        return ROTARY_BASE_BY_RUN[spec.run]
    raise BadParameter('rotary_base',
                       f'run {spec.run!r} is not in ROTARY_BASE_BY_RUN and no '
                       f'rotary_base= was declared. Read it from the checkpoint '
                       f'(torch.load(.../mp_rank_00_model_states.pt)["args"].rotary_base) '
                       f'and either declare rotary_base=<value> or add the run to the '
                       f'table in eval_lib.py. It is NOT safe to default: megatron '
                       f'falls back to 10,000 and scores silently lower.')


REQUIRED_KEYS = ('model', 'run', 'steps', 'benchmarks', 'time')
INT_KEYS = ('nodes', 'ep', 'tp', 'pp', 'mbs', 'shots', 'seqlen', 'limit', 'slices',
            'docs', 'workers', 'dp_split')
FLOAT_KEYS = ('rotary_base',)


@dataclass
class RunSpec:
    """One line of the RUNS block: every parameter a single run can state.

    A field left at None is not a missing value, it is a deferral: the task table
    supplies the benchmark's own measured setting. The plan printed at submit time
    always shows the value that was actually used, so a deferral is never hidden.
    """
    model: str
    run: str
    steps: str
    benchmarks: str
    time: str = '02:00:00'
    nodes: int = 1
    ep: int = 8
    tp: int = 1
    pp: int = 1
    mbs: Optional[int] = None       # None -> the task table's value for the benchmark
    shots: Optional[int] = None     # None -> the task table's value
    seqlen: Optional[int] = None    # None -> the task table's value
    pad: Optional[str] = None       # None -> the task table's value
    limit: Optional[int] = None     # None -> full benchmark
    slices: int = 1
    slice_by: str = 'share'         # share of the questions | named range of sub-tasks
    layout: str = 'share'           # contiguous shares | round_robin (every Nth doc)
    workers: int = 1                # independent model replicas inside ONE slurm job
    split: str = 'document'         # WHAT workers= cuts. 'document' is the split this
                                    # launcher has always done -- each replica is given
                                    # its own documents (or sub-tasks) and REBUILDS the
                                    # prompts for them, which is why few-shot rows with
                                    # no addressable sub-tasks are refused. 'request'
                                    # cuts one step later: every replica builds the
                                    # IDENTICAL full request list and computes only its
                                    # own slice of it, so the prompts are bit-identical
                                    # to an unsliced run by construction and the
                                    # few-shot restriction does not apply.
    dispatch: str = 'static'        # HOW the request blocks are handed out at
                                    # split=request. 'static' (the default, and byte for
                                    # byte what renders today) gives replica k blocks
                                    # k, k+N, k+2N, ... decided at submit time. 'engine'
                                    # starts a coordinator in the BATCH SHELL -- a plain
                                    # CPU python process, no torch, no model -- and each
                                    # replica pulls the next block when it is free. The
                                    # static rule balances a COLD cache correctly and a
                                    # warm one not at all: job 5369888 balanced documents
                                    # to max/mean 1.06 and its cache MISSES, which is the
                                    # actual work, to 1.54 -- 35% of the compute phase
                                    # idle. One job in three already resumes warm.
                                    # Inert at workers=1, so it is safe in DEFAULTS.
    dp_split: int = 0               # 1: the request list is split across the DATA-PARALLEL
                                    # ATTENTION GROUPS of one model copy (the TP groups at
                                    # pp=1: 8 nodes of TP 8 = 8 groups for the 671B), each
                                    # group taking interleaved blocks; the groups agree on
                                    # the number of forwards per round and an idle group runs
                                    # an empty forward. Ends the replication of the same
                                    # prompts on every group (2026-09-14). Orthogonal to
                                    # workers= (which replicates whole copies).
    docs: Optional[int] = None      # documents per sub-task, when there is no manifest
    moe: str = 'X-MOE'
    use_ucp: int = 1                # 0 -> load the consolidated .pt instead of UCP
    qos: Optional[str] = None       # sbatch -q. 'debug' schedules sooner but caps
                                    # walltime at 2h and allows ONE submitted debug
                                    # job per user -- see sbatch_command().
    experiment: Optional[str] = None  # a run made to COMPARE, never to publish: its
                                    # answers go to a private file and its registry
                                    # rows carry experiment=<tag>, which no consumer
                                    # will ever select as a published value.
    rotary_base: Optional[float] = None  # RoPE theta. Omit to take the run's
                                    # value from ROTARY_BASE_BY_RUN; the
                                    # template refuses to run without one.
    answers_ns: Optional[str] = None  # DEPRECATED spelling of experiment=

    def __post_init__(self):
        """Fold the deprecated `answers_ns=` into `experiment=`.

        Both keys stay declarable -- an old declaration line, an old shell wrapper and
        an old rendered .slurm all keep working -- but exactly ONE value reaches the
        rest of the launcher, so the answer file and the registry row can never end up
        under different tags. Declaring both to different values is a contradiction and
        is refused here rather than resolved silently.
        """
        if (self.answers_ns and self.experiment
                and str(self.answers_ns) != str(self.experiment)):
            raise BadParameter('experiment',
                               f'declared twice and differently: experiment='
                               f'{self.experiment!r} and answers_ns={self.answers_ns!r} '
                               f'(answers_ns= is the deprecated spelling of the same '
                               f'key). Declare experiment= only.')
        if self.answers_ns and not self.experiment:
            self.experiment = self.answers_ns

    def run_dir(self):
        """The checkpoint run directory: absolute, or a name/path resolved against
        scripts-frontier/ then scripts-frontier/checkpoint/. Falls back to the
        checkpoint/ candidate so validate() can name a concrete path when it is wrong."""
        r = self.run
        if os.path.isabs(r):
            return os.path.normpath(r)
        for base in (SCRIPTS_DIR, CHECKPOINT_DIR):
            cand = os.path.normpath(os.path.join(base, r))
            if os.path.isdir(cand):
                return cand
        return os.path.normpath(os.path.join(CHECKPOINT_DIR, r))

    def run_tag(self):
        return os.path.basename(os.path.normpath(self.run_dir()))

    def step_list(self):
        """steps= as a list of ints. 'latest' and 'all' read the run dir."""
        sel = str(self.steps).strip()
        if sel in ('all', 'latest'):
            found = sorted(int(m.group(1)) for m in
                           (re.fullmatch(r'global_step(\d+)', os.path.basename(d))
                            for d in _glob_steps(self.run_dir())) if m)
            return found[-1:] if sel == 'latest' else found
        try:
            return [int(x) for x in sel.split(',') if x.strip()]
        except ValueError:
            raise BadParameter('steps', f'{sel!r} is not a step number, a comma list, '
                                        f'"latest" or "all"')

    def shape(self):
        """The parallel shape ONE MODEL runs at -- part of a result's identity.

        At workers>1 that is not the allocation: the job holds `workers` INDEPENDENT
        replicas, each on one node, so the shape that produced every answer and every
        registry row is a ONE-NODE shape. Reporting the allocation instead would make
        every workers>1 result look like a different shape from the 1-node runs it is
        numerically identical to, and the done-check would never be satisfiable (the
        job's own rows record nodes=1, taken from the per-worker srun step).
        """
        nodes = 1 if int(self.workers) > 1 else int(self.nodes)
        return {'nodes': nodes, 'ep': int(self.ep),
                'tp': int(self.tp), 'pp': int(self.pp)}

    def replica_gpus(self):
        """GPUs one model replica gets: one node at workers>1, the allocation otherwise."""
        return GPUS_PER_NODE if int(self.workers) > 1 else self.total_gpus()

    def shape_tag(self):
        """'' for the default shape, else n{N}ep{E}tp{T}pp{P} for the job name.

        Every registry row and every temp_slurm reference on disk was produced at the
        default shape, and Gate 1 compares the rendered script byte for byte, so the
        default keeps its bare name. Any other shape says so in the job name, in the
        ledger record and therefore in the in-flight guard: job 5354152 (EP4/TP1/PP2)
        was skipped as 'inflight' against a 1-node job of the same benchmark for no
        reason other than that the name did not carry the shape.
        """
        shape = self.shape()
        tag = ('' if shape == default_shape()
               else 'n{nodes}ep{ep}tp{tp}pp{pp}'.format(**shape))
        # workers= does not change the shape one model runs at, so it is not in shape();
        # it does change what the JOB is, and the in-flight guard keys on the job name.
        # Without this a workers=8 job and a plain 1-node job of the same benchmark are
        # one name -- the collision that made job 5354152 read as 'inflight'.
        if int(self.workers) > 1:
            tag = f'{tag}_w{int(self.workers)}' if tag else f'w{int(self.workers)}'
        return tag

    def total_gpus(self):
        return int(self.nodes) * GPUS_PER_NODE

    def gpus_per_stage(self):
        return self.total_gpus() // int(self.pp)


def _glob_steps(run_dir):
    """Step directories of a run. global_step*_universal is a UCP conversion of a step,
    not a step of its own, and counting it would offer the same step twice."""
    return [d for d in glob.glob(os.path.join(run_dir, 'global_step[0-9]*'))
            if not d.endswith('_universal') and os.path.isdir(d)]


def parse_run(line, defaults=None):
    """One `key=value key=value ...` declaration line -> RunSpec.

    Unknown keys and missing required keys are refused here, naming the key, before
    anything touches the filesystem or the scheduler.
    """
    fields = {f for f in RunSpec.__dataclass_fields__}
    kw = dict(defaults or {})
    for tok in line.split():
        if tok.startswith('#'):
            break
        if '=' not in tok:
            raise BadParameter(tok, 'not a key=value pair')
        key, _, value = tok.partition('=')
        if key not in fields:
            raise BadParameter(key, f'unknown parameter; known: {sorted(fields)}')
        kw[key] = value
    for key in REQUIRED_KEYS:
        if not kw.get(key):
            raise BadParameter(key, 'is required and was not declared')
    for key in INT_KEYS:
        if key in kw and kw[key] not in (None, '', 'default'):
            try:
                kw[key] = int(kw[key])
            except (TypeError, ValueError):
                raise BadParameter(key, f'{kw[key]!r} is not an integer')
        elif key in kw:
            kw[key] = None
    for key in FLOAT_KEYS:
        if key in kw and kw[key] not in (None, '', 'default'):
            try:
                kw[key] = float(kw[key])
            except (TypeError, ValueError):
                raise BadParameter(key, f'{kw[key]!r} is not a number')
        elif key in kw:
            kw[key] = None
    return RunSpec(**kw)


def known_benchmarks():
    """Every name a declaration may use: (benchmarks, suites, categories)."""
    tasks, suites = task_table.load_custom()
    cats = sorted({r.category for r in tasks.values()})
    return sorted(tasks), sorted(suites), cats


def expand_benchmarks(expr):
    """'sciq', 'sciq,piqa', 'core+longctx' -> ordered, de-duplicated [(row_id, Row)].

    Comma and '+' are both accepted: task_table.expand() speaks '+', a human writing a
    list of benchmarks writes commas.
    """
    tasks, suites = task_table.load_custom()
    try:
        return task_table.expand(str(expr).replace(',', '+'), tasks, suites)
    except KeyError as exc:
        rows, sui, cats = known_benchmarks()
        raise BadParameter('benchmarks', f'{exc.args[0]}\n'
                                         f'  known benchmarks: {" ".join(rows)}\n'
                                         f'  known suites    : {" ".join(sui)}\n'
                                         f'  known categories: {" ".join(cats)}')


def job_python(path=None):
    """The interpreter the JOB will use, read out of the template's PYTHON_BIN.

    Preflight must probe THAT python, not whichever one happens to be running the
    launcher. submit_eval.py probed sys.executable, which is why submit_run_suite.sh
    had to prepend the eval env to PATH before calling it -- launch it with the login
    python3 instead and every `pip:` need reports a false failure. Reading the value
    from the template keeps one source of truth: if the template's interpreter moves,
    the check moves with it.
    """
    try:
        with open(path or TEMPLATE) as fh:
            found = [l.split('=', 1)[1].strip().strip('"\'')
                     for l in fh if l.startswith('PYTHON_BIN=')]
    except OSError:
        found = []
    # last uncommented assignment wins, matching how the shell would evaluate the file
    for cand in reversed(found):
        if cand and os.path.isfile(cand):
            return cand
    return sys.executable


def preflight(rows):
    """Every reason these benchmarks would fail late, found before anything is queued.

    Same three `needs:` tags task_table already carries -- pip:<module>,
    dataset:<dir fragment>, harness:<flag> -- checked against the interpreter the JOB
    will use and the eval dataset cache. Returns a list rather than raising: every
    problem is reported at once so one submit fixes them all.
    """
    problems = []
    try:
        with open(HARNESS_MAIN) as fh:
            harness_src = fh.read()
    except OSError:
        harness_src = ''
    python = job_python()
    for rid, row in rows:
        for need in row.needs:
            kind, _, what = need.partition(':')
            if kind == 'pip':
                probe = subprocess.run([python, '-c', f'import {what}'],
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
                if probe.returncode != 0:
                    problems.append(f'{rid}: python package {what!r} missing from '
                                    f'{python} (the interpreter the job runs, '
                                    f'PYTHON_BIN in the template)')
            elif kind == 'dataset':
                if not glob.glob(os.path.join(HF_DS_CACHE, f'*{what}*')):
                    problems.append(f'{rid}: dataset {what!r} not in {HF_DS_CACHE} '
                                    f'(run prefetch_datasets.py from a login node)')
            elif kind == 'harness' and harness_src and f'--{what}' not in harness_src:
                problems.append(f'{rid}: harness flag --{what} is not implemented')
        if row.seqlen_src.startswith('assumed'):
            problems.append(f'{rid}: seqlen={row.seqlen} is {row.seqlen_src!r} -- '
                            f'unmeasured. Run check_seqlens.py --rows {rid} first, or '
                            f'smoke it with limit=N.')
    return problems


def validate(spec, validate_paths=True, skip_preflight=False):
    """Every parameter checked at submit time. Each failure names its parameter.

    Cheapest first: a typo in `time=` is refused before the task table is read, and
    nothing touches the filesystem until the declaration itself is sound. The groups
    below raise on the first problem they find; preflight is the exception and
    collects all of them.
    """
    if spec.model not in model_registry.MODEL_SPECS:
        raise BadParameter('model', f'{spec.model!r} is not in model_registry; known: '
                                    f'{sorted(model_registry.MODEL_SPECS)}')
    mspec = model_registry.MODEL_SPECS[spec.model]
    _validate_scalars(spec)
    _validate_topology(spec, mspec)
    _validate_workers(spec)
    _validate_weight_loading(spec)
    rows = _validate_benchmarks(spec)
    # A limit=N run is the smoke test preflight itself recommends when a row's sequence
    # length is unmeasured, so preflight must not be what stops it from being submitted.
    if not (skip_preflight or spec.limit is not None):
        problems = preflight(rows)
        if problems:
            raise BadParameter('benchmarks', 'preflight failed:\n    '
                               + '\n    '.join(problems))
    if validate_paths:
        _validate_checkpoint(spec)
    return spec


def _validate_scalars(spec):
    """The parameters that are wrong on their own, with no reference to anything else."""
    if not re.fullmatch(r'\d{1,3}:[0-5]\d:[0-5]\d', str(spec.time)):
        raise BadParameter('time', f'{spec.time!r} is not HH:MM:SS')

    for key in ('nodes', 'ep', 'tp', 'pp'):
        if int(getattr(spec, key)) < 1:
            raise BadParameter(key, f'must be >= 1, got {getattr(spec, key)}')
    if spec.mbs is not None and int(spec.mbs) < 1:
        raise BadParameter('mbs', f'must be >= 1, got {spec.mbs}')
    if spec.shots is not None and int(spec.shots) < 0:
        raise BadParameter('shots', f'must be >= 0, got {spec.shots}')
    if spec.limit is not None and int(spec.limit) < 1:
        raise BadParameter('limit', f'must be >= 1 (omit for the full benchmark), '
                                    f'got {spec.limit}')
    if spec.pad is not None and spec.pad not in ('left', 'right'):
        raise BadParameter('pad', f'{spec.pad!r} must be "left" or "right"')
    if int(spec.slices) < 1:
        raise BadParameter('slices', f'must be >= 1, got {spec.slices}')
    if int(spec.workers) < 1:
        raise BadParameter('workers', f'must be >= 1, got {spec.workers}')
    if spec.split not in ('document', 'request'):
        raise BadParameter('split', f'{spec.split!r} must be "document" (each replica '
                                    f'gets its own documents and rebuilds their prompts '
                                    f'-- what workers= has always done) or "request" '
                                    f'(every replica builds the whole benchmark and '
                                    f'computes its own slice of the request list)')
    if spec.dispatch not in ('static', 'engine'):
        raise BadParameter('dispatch', f'{spec.dispatch!r} must be "static" (each '
                                       f'replica takes blocks k, k+N, k+2N, ... decided '
                                       f'at submit time -- the default and what renders '
                                       f'today) or "engine" (a coordinator in the batch '
                                       f'shell hands out the next block to whichever '
                                       f'replica is free)')
    if int(getattr(spec, 'dp_split', 0) or 0) not in (0, 1):
        raise BadParameter('dp_split', f'must be 0 or 1, got {spec.dp_split}')
    if int(getattr(spec, 'dp_split', 0) or 0) and int(spec.pp) > 1:
        raise BadParameter('dp_split', 'the data-parallel prompt split cuts the request list across the '
                                       'tensor-parallel groups of one model copy; with pp>1 those groups are '
                                       'not the data-parallel groups')
    if int(getattr(spec, 'dp_split', 0) or 0) and str(spec.dispatch) == 'engine':
        raise BadParameter('dp_split', 'the data-parallel prompt split hands out blocks by the static rule '
                                       'inside one model copy; dispatch=engine is not wired for it')
    if str(spec.dispatch) == 'engine' and str(spec.split) != 'request':
        raise BadParameter('dispatch', f'dispatch=engine hands out REQUEST BLOCKS, so it '
                                       f'needs split=request; this line declares '
                                       f'split={spec.split!r}. A document split has no '
                                       f'blocks to hand out -- its unit is decided at '
                                       f'submit time by the manifest each replica is '
                                       f'given. Declare `workers=N split=request '
                                       f'dispatch=engine`.')
    if spec.slice_by not in ('share', 'subtask'):
        raise BadParameter('slice_by', f'{spec.slice_by!r} must be "share" or "subtask"')
    if spec.layout not in ('share', 'contiguous', 'round_robin'):
        raise BadParameter('layout', f'{spec.layout!r} must be "share" or "round_robin"')
    # The experiment tag becomes a filename suffix, a job-name suffix AND a field on
    # every registry row this run writes, so it must be a plain token. `answers_ns` is
    # checked first so the refusal names the key the declaration actually used.
    for key in ('answers_ns', 'experiment'):
        val = getattr(spec, key, None)
        if val is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', str(val)):
            raise BadParameter(key, f'{val!r} becomes part of the answer filename and '
                                    f'of every registry row this run writes; use '
                                    f'letters, digits, dot, dash or underscore')


def _validate_topology(spec, mspec):
    """nodes/ep/tp/pp against each other and against the model.

    TP shards the dense layers, EP shards the experts, PP shards both across stages.
    32 GPUs = PP4 stages x 8 GPUs, with EP8 and TP4 inside a stage.
    """
    total, pp, tp, ep = spec.total_gpus(), int(spec.pp), int(spec.tp), int(spec.ep)
    if total % pp:
        raise BadParameter('pp', f'{pp} pipeline stages do not divide {total} GPUs '
                                 f'({spec.nodes} node(s) x {GPUS_PER_NODE})')
    stage = total // pp
    if stage % tp:
        raise BadParameter('tp', f'{tp} does not divide the {stage} GPUs of a pipeline '
                                 f'stage')
    if stage % ep:
        raise BadParameter('ep', f'{ep} does not divide the {stage} GPUs of a pipeline '
                                 f'stage')
    if ep > int(mspec['ep']):
        raise BadParameter('ep', f'{ep} exceeds the {mspec["ep"]} experts of '
                                 f'{spec.model}')
    if int(mspec['ep']) % ep:
        raise BadParameter('ep', f'{ep} does not divide the {mspec["ep"]} experts of '
                                 f'{spec.model}')
    # There is deliberately NO "layers % pp == 0" rule here. DeepSpeed partitions the
    # pipeline SPEC -- the embedding, every transformer layer, the final norm and the LM
    # head -- and balances that list; the transformer-layer count does not have to divide
    # the stage count. Measured: job 5354152 ran DeepSeek_16B (28 layers) at pp=2 and
    # logged '[engine.py:165] RANK=0 STAGE=0 LAYERS=16 [0, 16)'. A divisibility check
    # neither guarantees a legal partition nor rejects an illegal one, and it fired
    # BEFORE the refusal that actually matters, hiding it behind a wrong reason.


def _validate_workers(spec):
    """workers=N against the allocation and against the other parallelism keys.

    A worker IS a node: one complete model per node, `workers` of them, all inside one
    Slurm job, all answering into one shared answer file. That makes workers= the
    orthogonal axis to ep/tp/pp -- those SHARD one model, workers REPLICATES it -- and
    the two shapes are mutually exclusive at the node boundary:

      * pp>1 spreads ONE model over several nodes. A pipeline and an independent replica
        cannot both own the same node, so pp>1 with workers>1 is refused rather than
        silently reinterpreted.
      * tp and ep shard inside one replica, so at workers>1 they must fit the 8 GCDs of
        ONE node -- not the nodes x 8 of the allocation, which is what _validate_topology
        checks and what would otherwise wave through a tp=16 that cannot be built.
      * slices= splits a benchmark across JOBS; workers= splits it across replicas INSIDE
        one job. Declaring both splits the same document pool twice by two rules.
    """
    w, nodes = int(spec.workers), int(spec.nodes)
    if w == 1:
        return
    if w > nodes:
        raise BadParameter('workers', f'workers={w} needs {w} node(s) -- one complete '
                                      f'model per node -- but nodes={nodes}. Declare '
                                      f'nodes={w}.')
    if int(spec.pp) > 1:
        raise BadParameter('workers', f'workers={w} with pp={spec.pp}. workers= gives '
                                      f'each node its OWN complete model; pp= spreads '
                                      f'ONE model across nodes. They are mutually '
                                      f'exclusive shapes -- run pp>1 at workers=1, or '
                                      f'workers>1 at pp=1.')
    if int(spec.slices) > 1:
        raise BadParameter('workers', f'workers={w} with slices={spec.slices}. Both '
                                      f'split the same documents -- slices= across '
                                      f'JOBS, workers= across replicas inside ONE job. '
                                      f'Declare one of them.')
    replica = spec.replica_gpus()                    # 8: one node
    for key in ('tp', 'ep'):
        val = int(getattr(spec, key))
        if val > replica:
            raise BadParameter(key, f'{key}={val} but at workers={w} one replica owns a '
                                    f'single node ({replica} GCDs). {key} shards inside '
                                    f'one replica, so it must fit in {replica}.')
        if replica % val:
            raise BadParameter(key, f'{key}={val} does not divide the {replica} GCDs of '
                                    f'one replica (workers={w} gives each worker one '
                                    f'node).')


def _validate_weight_loading(spec):
    """tp>1 cannot load this checkpoint. Refuse at submit time, not at minute six.

    Measured, not inferred. Job 5354153 (4 nodes, EP8/TP4/PP1, piqa) reached
    loader.py:166 and raised: "universal atom load FAILED: 0 params without atoms,
    114 shape mismatches", the first three being
    ('language_model.embedding.word_embeddings.weight', atom (100015, 2048) vs param
    (25088, 2048)), ('...layers.0.self_attention.query_key_value.weight', (6144, 2048)
    vs (1536, 2048)) and ('...layers.0.self_attention.dense.weight', (2048, 2048) vs
    (2048, 512)) -- the full tensor against one of four shards, on the correct axis
    each time. _load_from_universal_atoms copies each atom WHOLE into the local
    parameter; it consults the checkpoint's VOCAB_TENSOR flag but never its
    row/column-parallel partition metadata and has no tensor_model_parallel_rank
    slicing at all.

    There is no TP precedent to appeal to. sacct on the two jobs that were cited as
    one: 5337015 CANCELLED after 00:00:06 with no rank logs, 5336680 TIMEOUT at
    02:00:28 with all 32 ranks frozen; zero registry rows between them.

    Lift this only when _load_from_universal_atoms slices each atom by the
    checkpoint's own partition metadata. Note loader.py is in
    result_registry.FORWARD_SOURCES, so that edit moves the code fingerprint and
    orphans every answer file -- it belongs in one batched migration, not on its own.
    """
    # LIFTED 2026-08-27 (LANDING_ORDER refusal-lift R1): _load_from_universal_atoms
    # now shards each atom by the checkpoint's own partition metadata
    # (wo16_loader_tp_pp, landed in the consolidated changeset, fp 49b64046 ->
    # 710b0047 -> 68496c20; loader tests 16/16). The topology arithmetic above
    # still validates the shape; tp>1 is no longer refused here. Evidence trail:
    # rebuild/LANDING_REPORT.md + rebuild/GATE1_INTERIM.md (GATE1_FINAL).
    pass


def _validate_benchmarks(spec):
    """The declaration against what each benchmark actually is. Returns the rows."""
    pp = int(spec.pp)
    rows = expand_benchmarks(spec.benchmarks)
    if not rows:
        raise BadParameter('benchmarks', f'{spec.benchmarks!r} expands to nothing')
    for rid, row in rows:
        # A generating benchmark cannot run under pipeline parallelism yet. The decode
        # loop reads logits on EVERY rank to pick the next token, and under PP only the
        # last stage has them (lm_eval_adapter._use_kv_decode is gated on
        # --no-pipeline-parallel, and the fallback full-reforward loop has the same
        # requirement). Running it anyway produces a number, which is the one outcome
        # that must not happen. Whether a benchmark generates comes from the task
        # table's own fields, never from a list kept here.
        if pp > 1 and (row.gen_toks or row.output_type == 'generate'):
            raise BadParameter('pp', f'{rid} generates (task table: gen_toks='
                                     f'{row.gen_toks}, output_type={row.output_type!r}) '
                                     f'and pp={pp}. The decode step reads logits on every '
                                     f'rank and only the last pipeline stage has them, so '
                                     f'the job would return answers produced from nothing. '
                                     f'Run it at pp=1.')
        # A SCORING benchmark cannot run under pipeline parallelism either, for a
        # quieter and more dangerous reason. lm_eval_adapter.py:737-740 replaces the
        # result with np.random.rand() on every rank that is not the last pipeline
        # stage, and nothing broadcasts the real values back -- there is no
        # broadcast/all_gather/all_reduce anywhere in the adapter or in main.py --
        # while main.py:286-290 makes world rank 0, which at pp>1 sits on stage 0, the
        # sole writer of results.json and of the registry rows. Cache HITS are answered
        # earlier (loglikelihood():154-158) and are real, so a WARM-cache pp>1 run
        # scores correctly and a COLD one records noise that is indistinguishable from a
        # result. That is the one outcome this rebuild exists to make impossible.
        # LIFTED 2026-08-27 (LANDING_ORDER refusal-lift R3, together with R1 per
        # WORKORDERS 2.4 step 4): wo1_score_broadcast landed -- the last pipeline
        # stage now broadcasts real scores to every rank, so world rank 0 writes
        # real numbers; wo16 landed -- pipe-spec parameter names map to the
        # checkpoint's atoms. A scoring benchmark may run at pp>1. The generating
        # refusal ABOVE stays: _use_kv_decode's PP-aware gate is the one remaining
        # R2 work item (adapter :227-232). Evidence: rebuild/LANDING_REPORT.md,
        # staged tests wo1 7/7 + wo16 16/16, GATE1_FINAL.
        if spec.seqlen is not None and row.seqlen_locked and int(spec.seqlen) != row.seqlen:
            raise BadParameter('seqlen', f'{rid} locks its sequence length at '
                                         f'{row.seqlen} (the length IS the evaluation); '
                                         f'{spec.seqlen} would change what the number '
                                         f'means')
        if int(spec.slices) > 1:
            # A sliced FEW-SHOT run does not answer the same questions as the unsliced
            # one. lm-eval draws every document's few-shot context from ONE rng stream
            # consumed in evaluation order (lm_eval/api/task.py:561 seeds; :478,:487
            # sample), so a document's prompt depends on its position in the evaluated
            # subset: 4-shot slices shared only 20 of 80 requests with the unsliced run
            # (jobs 5355076-79 vs 5354888), while the 0-shot control shared 80 of 80
            # (jobs 5355418-21). Seeding per document instead would change the UNSLICED
            # few-shot prompts too -- every existing few-shot answer cache goes stale
            # and Gate-1's mmlu/gsm8k numbers move -- so this is refused, not fixed
            # (coordinator decision, 2026-08-27).
            shots = int(spec.shots if spec.shots is not None else row.fewshot)
            if shots:
                src = (f'shots={spec.shots}' if spec.shots is not None
                       else f'the task table\'s {row.fewshot}-shot default for {rid}')
                raise BadParameter('slices', f'{rid} would run few-shot ({src}) with '
                                             f'slices={spec.slices}. A slice\'s few-shot '
                                             f'prompts depend on the documents evaluated '
                                             f'beside it (one rng stream, consumed in '
                                             f'evaluation order), so sliced few-shot '
                                             f'answers DIFFERENT questions than the '
                                             f'unsliced run: 4-shot slices shared 20 of '
                                             f'80 requests with it (jobs 5355076-79 vs '
                                             f'5354888); 0-shot shared 80 of 80. Run '
                                             f'few-shot benchmarks at slices=1, or '
                                             f'slice at shots=0.')
            _slice_units(spec, rid, row)              # raises if it cannot be sliced
        if int(spec.workers) > 1 and str(spec.split) != 'request':
            # split=document has to find something it may legally cut -- sub-tasks for a
            # few-shot row, documents for a zero-shot one -- and refuses when it cannot.
            # split=request cuts the materialized request list instead, which every row
            # has and which needs no manifest, no sub-tasks and no document arithmetic,
            # so there is nothing here to resolve and nothing to refuse. The refusal is
            # not relaxed; it is simply not the question being asked.
            _worker_units(spec, rid, row)             # raises if it cannot be split
    return rows


def weights_dir(spec, run_dir, step):
    """The directory whose mp_rank_00_model_states.pt identifies the step's weights: the ZeRO
    save global_step<N>, or, when that save was deleted after its universal conversion (the
    128k chain's retention rule, 2026-10-07) and the run loads universal atoms, the complete
    global_step<N>_universal, which holds copies of the model-state files with their original
    modification time (ds_to_universal copies them with shutil.copy2), so the answer-file
    identity does not change when the ZeRO save goes."""
    sd = os.path.join(run_dir, f'global_step{step}')
    if (not os.path.isdir(sd) and int(getattr(spec, 'use_ucp', 1) or 0)
            and os.path.isfile(os.path.join(sd + '_universal', 'zero', 'optimizer_state.pt'))):
        return sd + '_universal'
    return sd


def _validate_checkpoint(spec):
    """The weights this run needs are on disk and are not a half-written save."""
    run_dir = spec.run_dir()
    if not os.path.isdir(run_dir):
        raise BadParameter('run', f'checkpoint run dir not found: {run_dir}')
    for step in spec.step_list():
        sd = weights_dir(spec, run_dir, step)
        if not os.path.isdir(sd):
            raise BadParameter('steps', f'checkpoint dir missing: {sd}')
        if not os.path.isfile(os.path.join(sd, 'mp_rank_00_model_states.pt')):
            raise BadParameter('steps', f'{sd} has no mp_rank_00_model_states.pt '
                                        f'(half-written save?)')


def _slice_units(spec, rid, row):
    """The ordered thing this benchmark is sliced over. Raises BadParameter if unknown.

    Returns ('manifest', {sub-task: [idx]}) or ('subtasks', [name, ...]).
    """
    _, parent = parent_manifest(row)
    if spec.slice_by == 'subtask':
        names = sorted(parent) if parent else [row.task]
        if parent is None and int(spec.slices) > 1:
            # A group benchmark without a committed subset manifest (ruler8k,
            # longbench) DOES have leaves, but they live in lm-eval group expansion /
            # metadata JSON, which this launcher cannot address as an ordered unit
            # list -- so slice_by=subtask has nothing to cut. Refused rather than
            # half-built (coordinator decision, 2026-08-27): the _s manifest rows
            # exist precisely to make the leaves addressable (QA verified their
            # slices disjoint and complete), and slice_by=share stays universal.
            leaves = group_leaves(row.task)
            if leaves:
                tasks, _suites = task_table.load_custom()
                alt = (f'benchmarks={rid}_s' if f'{rid}_s' in tasks
                       else 'the benchmark\'s _s manifest row')
                raise BadParameter('slice_by', f'{rid} is a group benchmark '
                                               f'({len(leaves)} leaves) but pins no '
                                               f'committed sub-task manifest -- its '
                                               f'structure lives in lm-eval group '
                                               f'expansion / metadata JSON, which this '
                                               f'launcher cannot address, so '
                                               f'slice_by=subtask cannot cut it. Use '
                                               f'{alt} (committed manifest, every leaf '
                                               f'addressable), or slice_by=share.')
        if int(spec.slices) > len(names):
            raise BadParameter('slices', f'{rid} has {len(names)} sliceable sub-task(s) '
                                         f'({", ".join(names[:4])}...); '
                                         f'slices={spec.slices} would leave empty jobs. '
                                         f'Use slice_by=share, or slices<={len(names)}.')
        return 'subtasks', names
    # share
    if parent is not None:
        return 'manifest', parent
    n_docs = spec.docs if spec.docs is not None else spec.limit
    if n_docs is None:
        raise BadParameter('slices', f'{rid} has no committed subset manifest, so the '
                                     f'ordered question list is unknown. Declare '
                                     f'limit=N (slice the first N documents) or docs=N '
                                     f'(documents per sub-task), or use '
                                     f'slice_by=subtask.')
    return 'manifest', limit_manifest([row.task], n_docs)


# =====================================================================================
# workers= -- N independent model replicas inside ONE job
# =====================================================================================
def worker_split_by(spec, shots):
    """How workers= must cut this row: 'request', else 'subtask' at shots>0, else slice_by.

    split=request short-circuits everything below. THE SAFETY RULE the rest of this
    docstring states is a property of splitting DOCUMENTS, and at split=request no
    document is split: every replica builds the identical full request list (the few-shot
    rng is consumed in canonical order on every worker) and then takes a share of that
    already-materialized list. Measured on the benchmark this rule refuses -- gsm8k at 8
    shots -- a document split shares 1 of 20 requests with the unsliced run while a
    request-level partition of the same build shares 40 of 40, disjoint and complete
    (rebuild/DISPATCH_ENGINE_DESIGN.md section 1). So the refusal stays exactly as it is
    for split=document, where it is correct, and does not apply at split=request.

    THE SAFETY RULE, and it is measured, not argued (rebuild/DDP_EVAL_DISTRIBUTION.md
    section C1 and section 2.5b):

      * A few-shot prompt is drawn from ONE rng stream per task, consumed in evaluation
        order (lm_eval/api/task.py:562-565 seeds `fewshot_rnd`; :487 samples it once per
        document from the build loop at :331-345, which carries upstream's own
        `#TODO: need to offset doc_id by rank now!`). Split a task's DOCUMENTS and every
        surviving document gets a different prompt: 4-shot slices shared 20 of 80
        requests with the unsliced run (jobs 5355076-79 vs 5354888) where the 0-shot
        control shared 80 of 80.
      * Split whole SUB-TASKS and nothing moves, because each lm-eval sub-task owns its
        own `fewshot_rnd`. Measured end to end: all 5,768 answer keys the whole-group
        mmlu job 5364783 produced are a strict subset of the 56,060 keys the 57
        one-subtask jobs wrote.

    So a few-shot benchmark is split at whole-subtask granularity or not at all, and a
    zero-shot benchmark may be split by document share.
    """
    if str(getattr(spec, 'split', 'document')) == 'request':
        return 'request'
    return 'subtask' if int(shots) else spec.slice_by


def _worker_sizes(names, parent):
    """{sub-task: documents} for the LPT pack: exact from the manifest, else measured."""
    if parent:
        return {t: len(parent[t]) for t in names}
    return registry_doc_counts(names)


def _worker_units(spec, rid, row, shots=None):
    """What workers= cuts this row into: (kind, units, sizes). Raises naming `workers`.

    kind 'subtasks' -> units is a name list and each worker runs whole sub-tasks;
    kind 'manifest' -> units is {sub-task: [doc index]} and each worker takes a share.

    This is the split=document resolver and it is the only one that can refuse: cutting
    documents is what needs something addressable to cut. split=request never reaches
    here -- it cuts the materialized request list, which every row has (_request_shards).
    """
    shots = int(spec.shots if spec.shots is not None else row.fewshot) \
        if shots is None else int(shots)
    by = worker_split_by(spec, shots)
    _, parent = parent_manifest(row)
    if by == 'subtask':
        names = sorted(parent) if parent else (group_leaves(row.task)
                                               or table_leaves(row.task)
                                               or [row.task])
        if len(names) < 2:
            if shots:
                raise BadParameter(
                    'workers',
                    f'{rid} runs few-shot (shots={shots}) and has no addressable '
                    f'sub-tasks, so workers={spec.workers} has nothing it may legally '
                    f'cut. A few-shot benchmark may only be split at whole-sub-task '
                    f'granularity -- each lm-eval sub-task owns its own few-shot rng, '
                    f'while splitting one sub-task\'s DOCUMENTS changes every prompt '
                    f'(4-shot slices shared 20 of 80 requests with the unsliced run, '
                    f'jobs 5355076-79 vs 5354888; 0-shot shared 80 of 80). Addressable '
                    f'means a committed subset manifest in config/subsets/ or one task-'
                    f'table row per leaf. Run {rid} at workers=1.')
            raise BadParameter(
                'workers',
                f'{rid} has {len(names)} addressable sub-task(s), so '
                f'slice_by=subtask has nothing to cut for workers={spec.workers}. '
                f'Use the default slice_by=share.')
        if int(spec.workers) > len(names):
            raise BadParameter('workers', f'{rid} has {len(names)} sub-task(s) '
                                          f'({", ".join(names[:4])}...) and workers='
                                          f'{spec.workers} would leave empty workers. '
                                          f'Use workers<={len(names)}.')
        return 'subtasks', names, _worker_sizes(names, parent)
    # share: reuse the slices= resolver verbatim, and relabel its refusal so the message
    # names the parameter that was actually declared.
    try:
        kind, units = _slice_units(spec, rid, row)
    except BadParameter as exc:
        if exc.parameter == 'slices':
            raise BadParameter('workers', str(exc).split(': ', 1)[-1]
                               .replace('slices', 'workers'))
        raise
    n_docs = min(len(v) for v in units.values()) if units else 0
    if int(spec.workers) > n_docs:
        raise BadParameter('workers', f'{rid} has a sub-task with only {n_docs} '
                                      f'document(s); workers={spec.workers} would leave '
                                      f'empty workers. Use workers<={n_docs}.')
    return kind, units, {}


# =====================================================================================
# Planning
# =====================================================================================
@dataclass
class Job:
    """One sbatch submission: a rendered template plus what is needed to judge it done.

    A job knows which answer file it will write into and which registry rows a finished
    run of it leaves behind, which is what lets the next pass skip it instead of
    running it twice.
    """
    name: str
    row_id: str
    step: int
    subs: dict
    answer_identity: dict
    expected_tasks: list
    slice_index: int = None
    slice_count: int = None
    depends_on: str = None                       # None | 'slices'
    manifest: dict = None
    manifest_path: str = None
    workers: int = 1                             # replicas inside THIS job
    worker_shards: list = None                   # what each replica was given
    registry_limit: object = None                # limit value this job's rows carry
    meta: dict = field(default_factory=dict)


def template_supports_parallelism(spec, path=None):
    """Refuse tp>1 / pp>1 unless the template can actually carry them.

    The template pins PP_SIZE=1 and passes --no-pipeline-parallel unconditionally, and
    takes TP from the EVAL_MP_SIZE environment variable rather than a placeholder. A
    job declared at pp=4 would therefore RUN at pp=1 and report a number that looks
    fine -- the failure this whole rebuild exists to make impossible. Until the
    template carries {{TP_SIZE}} and {{PP_SIZE}} (rebuild/template_parallelism.patch),
    say so at submit time.
    """
    path = path or TEMPLATE
    slots = template_placeholders(path)
    if int(spec.tp) > 1 and 'TP_SIZE' not in slots:
        raise BadParameter('tp', f'tp={spec.tp} but {os.path.basename(path)} has no '
                                 f'{{{{TP_SIZE}}}} slot: it takes TP from $EVAL_MP_SIZE, '
                                 f'so the job would silently run at tp=1. Apply '
                                 f'rebuild/template_parallelism.patch first.')
    if int(spec.workers) > 1 and 'LAUNCH' not in slots:
        raise BadParameter('workers', f'workers={spec.workers} but '
                                      f'{os.path.basename(path)} has no {{{{LAUNCH}}}} '
                                      f'slot: it hard-codes a single '
                                      f'`srun -N ${{NODES}}`, so the job would build ONE '
                                      f'model across {spec.nodes} nodes instead of '
                                      f'{spec.workers} independent replicas and would '
                                      f'answer the whole benchmark {spec.workers} times '
                                      f'over. Use the template that declares '
                                      f'{{{{LAUNCH}}}}.')
    if int(spec.pp) > 1 and 'PP_SIZE' not in slots:
        raise BadParameter('pp', f'pp={spec.pp} but {os.path.basename(path)} pins '
                                 f'PP_SIZE=1 and always passes --no-pipeline-parallel, '
                                 f'so the job would silently run at pp=1. Apply '
                                 f'rebuild/template_parallelism.patch first, and read '
                                 f'its header: pp>1 also turns KV decode OFF and stops '
                                 f'the answer cache being written.')


def plan(spec, validate_paths=True, manifest_dir=None, skip_preflight=True):
    """Declaration -> jobs. Validates first; nothing is rendered until it passes."""
    validate(spec, validate_paths=validate_paths, skip_preflight=skip_preflight)
    template_supports_parallelism(spec)
    rows = expand_benchmarks(spec.benchmarks)
    run_dir, run_tag = spec.run_dir(), spec.run_tag()
    fp = code_fingerprint()
    jobs = []
    for step in spec.step_list():
        step_dir = os.path.join(run_dir, f'global_step{step}')
        mtime = weights_mtime(weights_dir(spec, run_dir, step))   # the universal copy's model-state file when the ZeRO save is gone
        for rid, row in rows:
            jobs += _plan_row(spec, rid, row, step, step_dir, run_tag, fp, mtime,
                              manifest_dir)
    return jobs


@dataclass
class _RowPlan:
    """One benchmark row with the declaration's overrides already applied.

    The task table carries each benchmark's measured sequence length, shots,
    micro-batch and generation budget; the declaration may override any of them. This
    is the single place the two meet, so no job below re-decides a value: the collector
    and every slice of the same row read the SAME resolved settings, which is why they
    all land in one answer file.
    """
    spec: 'RunSpec'
    rid: str                        # task-table row id, e.g. 'ruler8k_s'
    row: object                     # the task_table Row itself
    step: int
    step_dir: str
    run_tag: str
    seqlen: int
    shots: int
    mbs: int
    pad: str
    base_flags: str                 # the row's extra harness flags, generation budget included
    limit: object                   # documents per sub-task, or None for the whole benchmark
    parent_path: str                # the row's committed subset manifest, if it pins one
    parent: dict
    identity: dict                  # answer-file identity, shared by every job of this row
    expected: list                  # registry task names a finished run of this row writes
    slots: set                      # placeholders the template actually declares
    base_name: str

    @property
    def limit_flag(self):
        """--limit for the WHOLE benchmark. A slice never carries it; a manifest replaces it."""
        return f'--limit {self.limit}' if self.limit else ''

    def subs(self, name, task_list, flags, limit_flag):
        """Template substitutions for one job of this row."""
        spec = self.spec
        out = {
            'JOB_NAME': name,
            'STEP': str(self.step),
            'NODES': str(spec.nodes),
            'TOTAL_GPUS': str(spec.total_gpus()),
            'MODEL_SIZE': spec.model,
            'MOE_TYPE': spec.moe,
            'TASK_LIST': task_list,
            'MICRO_BATCH_SIZE': str(self.mbs),
            'EP_PARALLEL_SIZE': str(spec.ep),
            'LIMIT_FLAG': limit_flag,
            'CKPT_STEP_DIR': self.step_dir,
            'CONSOLIDATED_LOAD': os.path.join(EVAL_DIR, 'cache', 'consolidated',
                                              f'{spec.model.lower()}_step{self.step}.pt'),
            'PADDING_SIDE': self.pad,
            'LEFT_PAD_FLAG': '--use-left-pad-eval' if self.pad == 'left' else '',
            'SEQ_LEN_OVERRIDE': str(self.seqlen),
            'NUM_FEWSHOT': str(self.shots) if self.shots else '',
            'TIME': spec.time,
            'EVAL_DIR': EVAL_DIR,
            'EXTRA_EVAL_FLAGS': flags.strip(),
            'SUITE': spec.benchmarks,
        }
        # TP_SIZE / PP_SIZE exist only once template_parallelism.patch is applied.
        # Filling a slot the template does not declare is an error, and so is leaving
        # one it does declare empty, so follow the template rather than assume.
        if 'ROTARY_BASE' in self.slots:
            out['ROTARY_BASE'] = repr(resolve_rotary_base(spec))
        if 'TP_SIZE' in self.slots:
            out['TP_SIZE'] = str(spec.tp)
        if 'PP_SIZE' in self.slots:
            out['PP_SIZE'] = str(spec.pp)
        # LAUNCH carries the srun(s). At workers=1 it is the single srun the template
        # used to spell out, byte for byte; _plan_row overwrites it for workers>1.
        if 'LAUNCH' in self.slots:
            out['LAUNCH'] = _SINGLE_LAUNCH
        return out

    def meta(self):
        """The identity fields this row's jobs record in the submission ledger."""
        return dict({'run_dir': self.spec.run_dir(), 'run_tag': self.run_tag,
                     'benchmark': self.rid, 'seq_len': self.seqlen,
                     'num_fewshot': self.shots, 'mbs': self.mbs, 'padding': self.pad,
                     'model': self.spec.model, 'moe': self.spec.moe,
                     'suite': self.spec.benchmarks,
                     # None on a production run. Carried into the submission ledger too,
                     # so "what is in flight" and "what is done" agree about which
                     # experiment a job belongs to.
                     'experiment': self.spec.experiment or None},
                    **self.spec.shape())


def _resolve_row(spec, rid, row, step, step_dir, run_tag, fp, mtime):
    """Task-table settings overridden by the declaration -> a _RowPlan."""
    pad = spec.pad or row.padding
    seqlen = int(spec.seqlen if spec.seqlen is not None else row.seqlen)
    shots = int(spec.shots if spec.shots is not None else row.fewshot)
    base_flags = row.extra_eval_flags
    if row.gen_toks:
        base_flags = (base_flags + f' --max-gen-toks {int(row.gen_toks)}').strip()
    if spec.experiment:
        # The tag must reach the JOB, or the plan below would PRINT a private answer
        # file while the job read the shared one -- the vacuous-hit trap this exists to
        # close -- and, worse, the job's registry rows would carry no experiment and
        # would be published. One flag drives both: cache.py names the file from it and
        # main.py stamps it on every row.
        base_flags = (base_flags + f' --experiment {spec.experiment}').strip()
    if int(getattr(spec, 'dp_split', 0) or 0):
        base_flags = (base_flags + ' --dp-split').strip()
    parent_path, parent = parent_manifest(row)
    limit = spec.limit if spec.limit is not None else (row.limit or None)
    if parent is not None:
        limit = None
    return _RowPlan(
        spec=spec, rid=rid, row=row, step=step, step_dir=step_dir, run_tag=run_tag,
        seqlen=seqlen, shots=shots,
        mbs=int(spec.mbs if spec.mbs is not None else row.mbs), pad=pad,
        base_flags=base_flags, limit=limit,
        parent_path=parent_path, parent=parent,
        identity=dict(run_tag=run_tag, step=step, mtime=mtime, code_fp=fp,
                      weight_source='ucp' if int(spec.use_ucp) else 'cons',
                      padding=pad, seqlen=seqlen, shots=shots,
                      experiment=(spec.experiment or None)),
        expected=expected_row_tasks(row.task, parent),
        slots=template_placeholders(TEMPLATE),
        # The experiment is in the JOB NAME as well, because the in-flight guard keys
        # on the name: without it an experiment job queued for benchmark X would make
        # the production job for X read as 'inflight' and never be submitted -- the
        # same collision the shape tag exists to prevent (job 5354152).
        base_name=(f'{run_tag}_s{step}_q{seqlen}_fs{shots}_{rid[:40]}'
                   + (f'_{spec.shape_tag()}' if spec.shape_tag() else '')
                   + (f'_exp{spec.experiment}' if spec.experiment else '')))


def _plan_row(spec, rid, row, step, step_dir, run_tag, fp, mtime, manifest_dir):
    """The jobs one benchmark row becomes at one step.

    slices=1 is one job and nothing else -- no manifest, no collector, byte-identical
    to today. Above 1 it is N slices plus the collecting run that scores the complete
    benchmark from what they answered.
    """
    rp = _resolve_row(spec, rid, row, step, step_dir, run_tag, fp, mtime)
    collector = _collecting_job(rp)
    if int(spec.workers) > 1:
        # ONE job. The replicas and the collector are steps inside it, so there is no
        # second Job and no sbatch dependency -- only a different {{LAUNCH}}.
        shards = _worker_shards(rp, manifest_dir)
        collector.subs['LAUNCH'] = worker_launch_block(shards)
        collector.workers = len(shards)
        collector.worker_shards = shards
        collector.meta = dict(collector.meta, workers=len(shards))
        return [collector]
    if int(spec.slices) == 1:
        return [collector]
    collector.depends_on = 'slices'
    return _slice_jobs(rp, manifest_dir) + [collector]


def _collecting_job(rp):
    """The unsliced configuration: scores the whole benchmark and writes the result.

    It is the only job when slices=1, and the collector when there are slices -- the
    same job either way, which is precisely why a sliced run and an unsliced run score
    the same set.
    """
    return Job(
        name=rp.base_name, row_id=rp.rid, step=rp.step,
        subs=rp.subs(rp.base_name, rp.row.task, rp.base_flags, rp.limit_flag),
        answer_identity=rp.identity, expected_tasks=rp.expected,
        manifest=rp.parent, manifest_path=rp.parent_path,
        registry_limit=(sum(len(v) for v in rp.parent.values()) if rp.parent
                        else rp.limit),
        meta=rp.meta())


def _slice_manifest_name(rp, k, n, by, unit='sl'):
    """The slice manifest's filename carries the PARENT declaration's full identity.

    rp.base_name is {run_tag}_s{step}_q{seqlen}_fs{shots}_{rid}[_{shape}] -- the same
    identity spelling the job name and answer_file_name() already use -- plus docs and
    limit, the two fields base_name lacks and the one QA proved collides: two sliced
    declarations differing only in limit wrote the SAME {rid}_sl{k}of{n}_by_*.json path
    with different contents, jobs read the manifest at RUN time, and the document sets
    swapped under a single launcher invocation (QA_ANSWERS defect 2, jobs 5354994 vs
    5355041 both recording at one identity).
    """
    ident = rp.base_name
    if rp.spec.docs is not None:
        ident += f'_d{int(rp.spec.docs)}'
    if rp.spec.limit is not None:
        ident += f'_l{int(rp.spec.limit)}'
    # `unit` keeps the two splitters' manifests apart: slices= packs sub-tasks by NAME
    # COUNT and workers= packs them by DOCUMENT COUNT, so slice 2 of 4 and worker 2 of 4
    # of one declaration are different document sets that must never share a path (the
    # QA defect-2 collision class -- a queued job reads its manifest at RUN time).
    return f'{ident}_{unit}{k}of{n}_by_{by}.json'


def _slice_jobs(rp, manifest_dir):
    """The N slice jobs of one row: each answers its own share into the shared file.

    By sub-task each job scores a named range of leaves completely; by share each job
    takes part of every sub-task's question list. Either way the slices are disjoint by
    arithmetic and none of them writes a score -- the collector does that.
    """
    spec, rid, row, parent = rp.spec, rp.rid, rp.row, rp.parent
    n = int(spec.slices)
    kind, units = _slice_units(spec, rid, row)
    out = []
    for k in range(1, n + 1):
        name = f'{rp.base_name}_sl{k}of{n}'
        if kind == 'subtasks':
            names = slice_subtasks(units, k, n)
            task_list = ','.join(names)
            man = {t: parent[t] for t in names} if parent else None
            flags = rp.base_flags
            lim = rp.limit_flag
            if man is not None:
                path = os.path.join(manifest_dir or SLICE_DIR,
                                    _slice_manifest_name(rp, k, n, 'subtask'))
                flags = replace_manifest_in(rp.base_flags, path)
                lim = ''
            else:
                path = None
            exp = names
        else:
            man = slice_manifest(units, k, n, spec.layout)
            path = os.path.join(manifest_dir or SLICE_DIR,
                                _slice_manifest_name(rp, k, n, spec.layout))
            flags = replace_manifest_in(rp.base_flags, path)
            task_list = row.task
            lim = ''
            exp = rp.expected
        # A dry run plans the same jobs but writes nothing, so the manifest is only
        # committed to disk when a directory to commit it to was supplied.
        if manifest_dir and man is not None:
            write_manifest(path, man, {'row_id': rid, 'task': row.task,
                                       'parent': os.path.basename(rp.parent_path)
                                       if rp.parent_path else f'limit:{spec.limit}',
                                       'slice': [k, n], 'slice_by': spec.slice_by,
                                       'layout': spec.layout})
        out.append(Job(name=name, row_id=rid, step=rp.step,
                       subs=rp.subs(name, task_list, flags, lim),
                       answer_identity=rp.identity, expected_tasks=exp,
                       slice_index=k, slice_count=n, manifest=man, manifest_path=path,
                       registry_limit=(sum(len(v) for v in man.values())
                                       if man else None),
                       meta=dict(rp.meta(), slice=[k, n])))
    return out


def dispatch_block_size(mbs, seqlen=0):
    """Requests per dispatch block: 4 x the micro-batch this row runs at -- except at
    long sequence lengths, where the block IS the generation batch and the batched-KV
    buffer scales with it (lm_eval_adapter:477 cap = max(batch_size, B)). At 65536 a
    4-row block allocates ~53 GB of KV alone and every worker of jobs 5384536/37 and
    5384602/03 OOM'd on a 1.49 GiB ask with ~1.2 GiB free, looping dead for the whole
    wall while banking nothing. Block=1 caps KV at one row (~13 GB @64k).

    Four micro-batches is small enough that a worker cannot be left holding a long tail
    of work and large enough that the adapter's own sort-and-chunk (which happens inside
    a block's worth of requests) still sees a useful spread of lengths. It is passed to
    the job EXPLICITLY rather than derived there, so the block size is a property of the
    declaration -- recorded on the registry row, quotable, and identical for every worker
    of the run whatever the job's own defaults are.
    """
    if seqlen >= 49152:
        return 1 * int(mbs)
    return 4 * int(mbs)


def _request_shards(rp, n):
    """The N request-level shards of one row: integers, not documents.

    A shard gets NO manifest and NO task list. That is the whole point: it runs the same
    ${eval_options} the collector runs, builds the same benchmark the collector builds --
    identical prompts, because lm-eval materializes them from one deterministically
    seeded rng in canonical order on every worker -- and then computes only blocks
    k-1, k-1+n, k-1+2n, ... of the resulting request list. Nothing about the DOCUMENTS is
    cut, so nothing about the few-shot prompts can move.

    Its rows are stamped partial by main.py off --shard-index/--shard-count, exactly as a
    slice manifest's _meta['slice'] stamps a document shard, so a request shard can no
    more satisfy a done-check than a document shard can. The collector writes the row.
    """
    block = dispatch_block_size(rp.mbs, getattr(rp, "seqlen", 0) or 0)
    docs = (sum(len(v) for v in rp.parent.values()) if rp.parent
            else (int(rp.limit) if rp.limit else None))
    return [{'index': k, 'count': n, 'split_by': 'request', 'names': None,
             'task_list': None, 'manifest': None, 'manifest_path': None,
             'block': block, 'docs': docs if docs is not None else 'all',
             'dispatch': str(getattr(rp.spec, 'dispatch', 'static')),
             'tag': ((f'request blocks pulled on demand from the coordinator '
                      f'(block={block} requests); builds the WHOLE benchmark and '
                      f'computes whatever it is handed')
                     if str(getattr(rp.spec, 'dispatch', 'static')) == 'engine' else
                     (f'request blocks {k - 1}, {k - 1 + n}, {k - 1 + 2 * n}, ... '
                      f'(block={block} requests); builds the WHOLE benchmark and '
                      f'computes ~1/{n} of its request list'))}
            for k in range(1, n + 1)]


def _worker_shards(rp, manifest_dir):
    """The W disjoint shares the replicas of one job get. Same machinery as a slice.

    A worker IS a slice that happens to live in the same job as its siblings, so this
    calls the same proven splitters -- slice_manifest()/round_robin_positions() for a
    document share, slice_subtasks() for whole sub-tasks -- and writes the same kind of
    manifest, with the same _meta['slice'] header that makes main.py stamp the row
    partial (WO-4). The only thing workers= adds is the LPT pack: sub-tasks go into bins
    by DOCUMENT COUNT rather than by name count, because the natural MMLU sub-tasks range
    100..1534 documents and equal-count bundles measured max/mean 3.01 at TP1
    (rebuild/DDP_EVAL_DISTRIBUTION.md section 2.3).
    """
    spec, rid, row, parent = rp.spec, rp.rid, rp.row, rp.parent
    n = int(spec.workers)
    by = worker_split_by(spec, rp.shots)
    if by == 'request':
        return _request_shards(rp, n)
    kind, units, sizes = _worker_units(spec, rid, row, shots=rp.shots)
    out = []
    for k in range(1, n + 1):
        if kind == 'subtasks':
            names = slice_subtasks(units, k, n, sizes=(sizes or None))
            task_list = ','.join(names)
            man = {t: parent[t] for t in names} if parent else None
            docs = (sum(len(v) for v in man.values()) if man
                    else sum(int(sizes.get(t, 0)) for t in names) or '?')
            tag = f'{len(names)} sub-task(s): {task_list}'
        else:
            names = None
            man = slice_manifest(units, k, n, spec.layout)
            task_list = None                     # the row's own task list is unchanged
            docs = sum(len(v) for v in man.values())
            tag = (f'{docs} document(s) over {len(man)} sub-task(s), '
                   f'layout={spec.layout}')
        path = None
        if man is not None:
            path = os.path.join(manifest_dir or SLICE_DIR,
                                _slice_manifest_name(rp, k, n, by, unit='w'))
            # A dry run plans the same shards but writes nothing.
            if manifest_dir:
                write_manifest(path, man, {'row_id': rid, 'task': row.task,
                                           'parent': os.path.basename(rp.parent_path)
                                           if rp.parent_path else f'limit:{spec.limit}',
                                           'slice': [k, n], 'slice_by': by,
                                           'layout': spec.layout, 'workers': n})
        out.append({'index': k, 'count': n, 'split_by': by, 'names': names,
                    'task_list': task_list, 'manifest': man, 'manifest_path': path,
                    'docs': docs, 'tag': tag})
    return out

# =====================================================================================
# The launch block -- {{LAUNCH}} in frontier_eval.slurm.template
#
# The template used to hold ONE `srun -N ${NODES} -n ${TOTAL_GPUS}` at this point. It now
# holds a slot, and this module fills it. The reason is the regression guarantee: at
# workers=1 the slot is filled with the srun that was there, BYTE FOR BYTE, so every
# script this launcher has ever rendered still renders identically -- pinned by
# tests/test_eval_lib.py against rebuild/baseline/frontier_eval.slurm.template.pre_workers
# and against the six frozen baseline-capture scripts. Everything workers= adds lives in
# the workers>1 value and cannot reach a workers=1 job.
# =====================================================================================
_SINGLE_LAUNCH = r'''srun -N ${NODES} -n ${TOTAL_GPUS} -c 6 \
     --ntasks-per-node=${GPUS_PER_NODE} \
     --gpus-per-node=${GPUS_PER_NODE} \
     --kill-on-bad-exit=1 \
     --export=ALL,MASTER_ADDR,MASTER_PORT,FI_MR_CACHE_MONITOR,FI_CXI_DEFAULT_CQ_SIZE,FI_CXI_RX_MATCH_MODE,NCCL_DEBUG,NCCL_SOCKET_IFNAME,NCCL_TIMEOUT,TRITON_CACHE_DIR,TORCH_EXTENSIONS_DIR,MEGATRON_FUSED_KERNELS_BUILD_DIR,HF_HOME,HF_DATASETS_CACHE,http_proxy,https_proxy,no_proxy,USE_TRITON_GROUPGEMM,USE_AITER_GROUPGEMM \
     bash -c "
        export LOCAL_RANK=\${SLURM_LOCALID}
        export RANK=\${SLURM_PROCID}
        export WORLD_SIZE=\${SLURM_NTASKS}
        cd ${REPO_ROOT}
        ${PYTHON_BIN} tasks/eval_harness/evaluate_v04.py \
            ${megatron_options} \
            ${data_options} \
            ${deepspeed_options} \
            ${eval_options} \
            > ${JOB_DIR}/rank_\${SLURM_PROCID}.log 2>&1
     "'''

# The srun flags every replica and the collector share: ONE node, its eight GCDs,
# --exclusive so two replicas can never land on the same node whatever Slurm does with
# the allocation, and the template's own --export list unchanged.
_WORKER_SRUN = r'''@IND@srun --exclusive -N1 -n${GPUS_PER_NODE} -c 6 \
@IND@     --ntasks-per-node=${GPUS_PER_NODE} \
@IND@     --gpus-per-node=${GPUS_PER_NODE} \
@IND@     --kill-on-bad-exit=1 \
@IND@     -w "@HOST@" \
@IND@     --export=ALL,MASTER_ADDR,MASTER_PORT,@DEXP@FI_MR_CACHE_MONITOR,FI_CXI_DEFAULT_CQ_SIZE,FI_CXI_RX_MATCH_MODE,NCCL_DEBUG,NCCL_SOCKET_IFNAME,NCCL_TIMEOUT,TRITON_CACHE_DIR,TORCH_EXTENSIONS_DIR,MEGATRON_FUSED_KERNELS_BUILD_DIR,HF_HOME,HF_DATASETS_CACHE,http_proxy,https_proxy,no_proxy,USE_TRITON_GROUPGEMM,USE_AITER_GROUPGEMM \
@IND@     bash -c "
@IND@        export LOCAL_RANK=\${SLURM_LOCALID}
@IND@        export RANK=\${SLURM_PROCID}
@IND@        export WORLD_SIZE=\${SLURM_NTASKS}
@IND@        export SLURM_NNODES=1
@ENV@@IND@        cd ${REPO_ROOT}
@IND@        ${PYTHON_BIN} tasks/eval_harness/evaluate_v04.py \
@IND@            ${megatron_options} \
@IND@            ${data_options} \
@IND@            ${deepspeed_options} \
@IND@            ${eval_options} \
@OVR@@IND@            > @LOGDIR@/rank_\${SLURM_PROCID}.log 2>&1
@IND@     "'''

# Per-worker JIT and kernel-cache dirs. The pattern is the training template's
# (../frontier_elmoe.slurm.template:448-457): a node-local /tmp dir per job, seeded from
# a persistent Lustre copy, so nothing queues on a shared build lock. workers= extends it
# by one level -- /tmp/<job>/w<i>/ instead of /tmp/<job>/ -- because N replicas start at
# the same instant and torch's file_baton and the fused-kernels ninja build both
# serialise on one directory. TRITON_CACHE_DIR is the one that was NOT node-local (it
# lives on Lustre and is shared by every job), so each worker gets its own copy of it,
# seeded no-clobber so a warm cache stays warm across jobs.
_WORKER_JIT = r'''@IND@        export TORCH_EXTENSIONS_DIR=@TE@
@IND@        export MEGATRON_FUSED_KERNELS_BUILD_DIR=@FK@
@IND@        export TRITON_CACHE_DIR=@TC@
@IND@        mkdir -p @TE@ @FK@
@IND@        flock @TE@.seedlock -c 'test -e @TE@/.seeded || { cp -a ${EVAL_DIR}/cache/torch_extensions_good/. @TE@/ 2>/dev/null; touch @TE@/.seeded; }' || true
'''

_WORKER_HEAD = r'''# ============================================================================
# workers=@N@ -- @N@ INDEPENDENT model replicas, one per node, inside this one job.
#
# This is not data parallelism and nothing is collective across workers. Each replica
# builds the whole model on its own node (@N@ x ${GPUS_PER_NODE} GCDs), answers its own
# disjoint share of the documents, and appends to the SAME content-keyed answer file
# (megatron_ds_eval/cache.py: sha256 request key + flock'd append, measured clean at a
# 57-way concurrent cross-node append -- 0 torn lines, 0 duplicate rows, 1 header).
#
# Measured motivation (rebuild/EVAL_WORKLOAD_ACCOUNTING.md section 8 and
# rebuild/DDP_EVAL_DISTRIBUTION.md section 2.3): spreading ONE model over 14 nodes with
# PP14 bought 1.61x throughput for 14x the nodes, while 16 independent one-node shards
# of RULER-16k landed the whole 1,300-document row in a 52-minute span against 13.1 h
# unsharded -- 15x wall at ~0.9x the node-hours.
# ============================================================================
WORKERS=@N@
readarray -t w_hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST:-${SLURM_NODELIST}}")
if [[ ${#w_hosts[@]} -lt ${WORKERS} ]]; then
    echo "FATAL: workers=${WORKERS} but the allocation holds ${#w_hosts[@]} node(s)"
    exit 1
fi
echo "============================================================"
echo "  workers           : ${WORKERS} independent replicas, 1 node x ${GPUS_PER_NODE} GCDs each"
echo "  worker hosts      : ${w_hosts[@]:0:${WORKERS}}"
echo "  NOTE                the Parallelism line above describes the ALLOCATION. Each worker"
echo "                      runs its OWN WORLD_SIZE=${GPUS_PER_NODE} group at EP=${EP_PARALLEL_SIZE}, TP=${MP_SIZE}, PP=1, DP=1,"
echo "                      with its own MASTER_ADDR/MASTER_PORT. No collective crosses a worker."
echo "  worker logs       : ${JOB_DIR}/worker_<i>/rank_<r>.log"
echo "============================================================"
w_failed=""
LAUNCH_RC=0
'''

_WORKER_STANZA = r'''
# ---- worker @K@/@N@ -- @TAG@
mkdir -p "@WDIR@" "@TC@"
cp -an ${TRITON_CACHE_DIR}/. "@TC@/" 2>/dev/null || true
w_addr_@I@=$(ssh "@HOST@" "ip -4 addr show ${HIGH_SPEED_INTERFACE} | grep 'inet' | awk '{print \$2}' | cut -d/ -f1 | head -n1" 2>/dev/null)
[[ -n "${w_addr_@I@}" ]] || w_addr_@I@="@HOST@"
echo "[worker @K@/@N@] host=@HOST@ addr=${w_addr_@I@} port=@PORT@ docs=@DOCS@"
echo "[worker @K@/@N@] @TAG@"
(
  export MASTER_ADDR="${w_addr_@I@}"
  export MASTER_PORT=@PORT@
@SRUN@
) &
w_pid_@I@=$!
'''

_WORKER_WAIT = r'''wait ${w_pid_@I@}; w_rc_@I@=$?
if [[ ${w_rc_@I@} -ne 0 ]]; then
    echo "[workers] SHARD @K@/@N@ FAILED rc=${w_rc_@I@} host=@HOST@ -- @TAG@"
    w_failed="${w_failed} @K@/@N@"
    LAUNCH_RC=${w_rc_@I@}
else
    echo "[workers] shard @K@/@N@ ok   host=@HOST@"
fi
'''

# =====================================================================================
# dispatch=engine -- the coordinator, and where it lives
# =====================================================================================
# IN THE BATCH SHELL. Not a 9th rank (EP=8 fills all 8 GCDs of a node, and a rank inside
# a model group cannot poll a socket while its own blocking collectives are running), not
# rank 0 of worker 0 doing double duty (same reason -- a dispatcher that stops answering
# for the duration of a batch is a dispatcher that cannot heartbeat), and not another
# node (it would burn a whole allocation to run a queue of integers). The batch shell is
# already a live CPU process for the whole job, it already backgrounds a helper
# (monitor_run.sh, template :612-618), and it survives every srun step. It costs nothing
# and consumes no allocation.
#
# The transport is a plain TCP socket, not a torch process group. NCCL cannot receive from
# ANY_SOURCE at all ("ProcessGroupNCCL does not support recvAnysource", a literal string
# in libtorch_hip.so) and would need one global group across every worker, undoing the
# isolation workers= exists to provide. gloo can receive from any source and must not be
# used: measured, one dead peer's timeout closed the connections to the HEALTHY peers and
# killed a worker that was fine. Raw TCP detects a death by EOF at 0.0 s, in isolation,
# and its heartbeat is independent of the payload -- so the silence timeout is sized
# against silence rather than against the longest legitimate block.
# The port the coordinator binds. Workers already claim 29500+i (@PORT@ below), so at the
# workers<=16 shapes this launcher permits there is no collision. Asserted in the tests
# rather than assumed.
DISPATCH_PORT = 29600

# Declare a worker dead after this much SILENCE. It is NOT sized against how long a block
# takes -- the leader's heartbeat thread ticks every 30 s independently of the block --
# which is the whole reason the control plane is a socket rather than a process group.
DISPATCH_HEARTBEAT_TIMEOUT = 300

# Assignments per block before it is abandoned, logged by name and journalled. A block
# that kills whatever worker touches it must not be able to eat the job one worker at a
# time. An abandoned block is not lost: it is simply uncomputed, so it stays a miss in the
# answer file and the next pass recomputes it.
DISPATCH_MAX_ATTEMPTS = 3


_DISPATCH_HEAD = r'''
# ---- dispatch coordinator (rebuild/DISPATCH_ENGINE_DESIGN.md, Stages 2 and 3) --------
# A queue of BLOCK INDICES and a socket. No torch, no model, no dataset, no GPU: every
# worker already builds the identical request list, so the only thing that crosses the
# wire is a small integer saying which block to compute.
DISPATCH_PORT=@DPORT@
DISPATCH_ADDR=$(ip -4 addr show ${HIGH_SPEED_INTERFACE} 2>/dev/null | grep 'inet' | awk '{print $2}' | cut -d/ -f1 | head -n1)
[[ -n "${DISPATCH_ADDR}" ]] || DISPATCH_ADDR=$(hostname)
export DISPATCH_ADDR DISPATCH_PORT
DISPATCH_JOURNAL="@DJOURNAL@"
mkdir -p "$(dirname "${DISPATCH_JOURNAL}")"
${PYTHON_BIN} ${REPO_ROOT}/tasks/eval_harness/megatron_ds_eval/dispatch_server.py \
    --host 0.0.0.0 \
    --port ${DISPATCH_PORT} \
    --journal "${DISPATCH_JOURNAL}" \
    --port-file "${JOB_DIR}/dispatch.port" \
    --heartbeat-timeout @DHB@ \
    --max-attempts @DTRIES@ \
    > ${JOB_DIR}/dispatch.log 2>&1 &
DISPATCH_PID=$!
# The monitor already owns an EXIT trap (template :618); replace it with one that kills
# both, so neither helper outlives the batch shell.
trap 'kill ${DISPATCH_PID} ${MONITOR_PID:-} 2>/dev/null' EXIT
d_ready=0
for d_try in $(seq 1 60); do
    if (exec 3<>/dev/tcp/127.0.0.1/${DISPATCH_PORT}) 2>/dev/null; then d_ready=1; break; fi
    sleep 1
done
if [[ -s "${JOB_DIR}/dispatch.port" ]]; then
    DISPATCH_PORT=$(head -n1 "${JOB_DIR}/dispatch.port")
    export DISPATCH_PORT
fi
if [[ ${d_ready} -eq 1 ]]; then
    echo "[dispatch] coordinator pid ${DISPATCH_PID} listening on ${DISPATCH_ADDR}:${DISPATCH_PORT}"
    echo "[dispatch]   journal ${DISPATCH_JOURNAL} (write-only audit trail: nothing ever reads"
    echo "[dispatch]   it back. The answer cache is the only authority on what is done.)"
    echo "[dispatch]   log     ${JOB_DIR}/dispatch.log"
else
    echo "[dispatch] WARNING: coordinator did not accept on :${DISPATCH_PORT} within 60s."
    echo "[dispatch] The workers will fall back to the STATIC k, k+N, k+2N split, which is"
    echo "[dispatch] exactly what dispatch=static does. The job still completes."
fi
'''

_DISPATCH_TAIL = r'''
# ---- dispatch coordinator shutdown ---------------------------------------------------
# Stop it BEFORE the collector: the collector is unsliced, holds no shard flags and never
# talks to the coordinator, and the final report is what the run is read from.
if [[ -n "${DISPATCH_PID:-}" ]]; then
    kill -TERM ${DISPATCH_PID} 2>/dev/null
    wait ${DISPATCH_PID} 2>/dev/null
    trap 'kill ${MONITOR_PID:-} 2>/dev/null' EXIT
    echo "[dispatch] ---- coordinator final report ----"
    tail -n 60 "${JOB_DIR}/dispatch.log"
fi
'''

_COLLECTOR = r'''
if [[ -n "${w_failed}" ]]; then
    echo "[workers] FAILED SHARD(S):${w_failed}"
    echo "[workers] Whatever those shards computed before dying is still in the answer"
    echo "[workers] file, so the collector below still runs -- but this job will NOT exit 0."
fi

# ---- collector: the UNSLICED configuration, scoring the WHOLE benchmark ---------------
# One more single-node step of this same job, every request now a cache hit. It is the
# only step that writes results.json and the production registry row: every worker above
# carries a slice manifest, and main.py stamps a manifest whose _meta says 'slice' as
# partial='slice k/n' (WO-4), which rows_cover() skips and report.py excludes -- so a
# half-finished workers run can never satisfy a done-check or be read as the benchmark.
echo "[collect] scoring the whole benchmark from the shared answer cache on ${w_hosts[0]}"
@SRUN@
COLLECT_RC=$?
echo "[collect] exit ${COLLECT_RC}"
if [[ ${COLLECT_RC} -ne 0 ]]; then LAUNCH_RC=${COLLECT_RC}; fi
( exit ${LAUNCH_RC} )'''


def _srun_text(host, logdir, indent='', env='', overrides=(), export_extra=''):
    """One `srun --exclusive -N1 -n8` step: a replica's, or the collector's.

    `export_extra` is the ONLY thing dispatch=engine adds to this text, and it renders
    empty everywhere else -- which is what keeps every declaration that does not say
    dispatch=engine byte-for-byte identical to what it renders today (TestRenderIdentity
    compares whole rendered {{LAUNCH}} blocks against rebuild/baseline/launch/).
    """
    ovr = ''.join(f'{indent}            {o} \\\n' for o in overrides)
    return (_WORKER_SRUN.replace('@ENV@', env).replace('@OVR@', ovr)
            .replace('@DEXP@', export_extra)
            .replace('@HOST@', host).replace('@LOGDIR@', logdir)
            .replace('@IND@', indent))


def _worker_overrides(shard, wdir):
    """The flags a worker appends AFTER ${eval_options} to claim its own share.

    APPENDED, never spliced. Every one is an ordinary argparse option, so the LAST
    occurrence wins (main.py:42 --task_list, :45 --results_path, :161 the
    --task-samples-json-file / --task_samples_json_file pair sharing one dest). Appending
    is what lets a worker reuse the template's single ${eval_options} recipe rather than
    re-deriving it, so a worker can never drift from the collector it feeds.
    """
    out = []
    if shard.get('task_list'):
        out.append(f'--task_list {shard["task_list"]}')
    if shard.get('manifest_path'):
        out.append(f'{_MANIFEST_FLAGS[0]} {shard["manifest_path"]}')
    if shard.get('split_by') == 'request':
        # A request shard carries neither a task list nor a manifest -- it runs the
        # collector's own recipe and cuts the materialized request list instead. The
        # index is 0-based on the wire (main.py --shard-index) and 1-based in the plan,
        # the ledger and the registry row, matching how slices are already numbered.
        out.append(f'--shard-index {int(shard["index"]) - 1}')
        out.append(f'--shard-count {int(shard["count"])}')
        out.append(f'--dispatch-block {int(shard["block"])}')
        if shard.get('dispatch') == 'engine':
            # The coordinator's address, resolved by the BATCH SHELL for its own host and
            # expanded here (unescaped, so it expands at launch time in the outer shell,
            # exactly like ${JOB_DIR} on the next line). --shard-index/--shard-count stay:
            # under the engine they no longer choose the blocks, but they still name the
            # worker on its registry row and they are the STATIC rule this worker falls
            # back to if the coordinator is unreachable.
            out.append('--dispatch-addr ${DISPATCH_ADDR}:${DISPATCH_PORT}')
            out.append(f'--dispatch-id {int(shard["index"]) - 1}')
    out.append(f'--results_path {wdir}/results.json')
    return out


def worker_launch_block(shards):
    """{{LAUNCH}} for workers>1: N replica sruns, a wait, then the in-job collector.

    Ordering, and why it is a `wait` and not a Slurm dependency: this is ONE job, so the
    collector is one more step inside it. It runs even when a worker failed, for exactly
    the reason the sliced path submits its collector with --dependency=afterany -- a
    worker that died still wrote every answer it computed, and the collector reads the
    answer FILE, not the exit status. The failure is named and carried into the job's
    exit code instead of being swallowed.
    """
    n = len(shards)
    out = [_WORKER_HEAD.replace('@N@', str(n))]
    # dispatch=engine adds a coordinator to the batch shell and one flag to each worker.
    # Everything else -- the head, the stanzas, the waits, the collector -- is unchanged,
    # and at dispatch=static this whole feature renders zero bytes.
    engine = any(sh.get('dispatch') == 'engine' for sh in shards)
    dexp = 'DISPATCH_ADDR,DISPATCH_PORT,' if engine else ''
    if engine:
        out.append(_DISPATCH_HEAD
                   .replace('@DPORT@', str(DISPATCH_PORT))
                   .replace('@DJOURNAL@', '${JOB_DIR}/dispatch.journal.jsonl')
                   .replace('@DHB@', str(DISPATCH_HEARTBEAT_TIMEOUT))
                   .replace('@DTRIES@', str(DISPATCH_MAX_ATTEMPTS)))
    for i, shard in enumerate(shards):
        wdir = '${JOB_DIR}/worker_%d' % i
        host = '${w_hosts[%d]}' % i
        te = '/tmp/${SLURM_JOB_ID}/w%d/torch_extensions' % i
        fk = '/tmp/${SLURM_JOB_ID}/w%d/fused_kernels_build' % i
        tc = '${EVAL_DIR}/cache/triton_cache_w%d' % i
        env = (_WORKER_JIT.replace('@TE@', te).replace('@FK@', fk)
               .replace('@TC@', tc).replace('@IND@', '  '))
        srun = _srun_text(host, wdir, indent='  ', env=env,
                          overrides=_worker_overrides(shard, wdir),
                          export_extra=dexp)
        out.append(_WORKER_STANZA
                   .replace('@SRUN@', srun).replace('@TAG@', shard['tag'])
                   .replace('@DOCS@', str(shard.get('docs', '?')))
                   .replace('@WDIR@', wdir).replace('@TC@', tc)
                   .replace('@HOST@', host).replace('@PORT@', str(29500 + i))
                   .replace('@K@', str(i + 1)).replace('@N@', str(n))
                   .replace('@I@', str(i)))
    out.append('\necho "[workers] all ${WORKERS} replicas launched; waiting"\n')
    for i, shard in enumerate(shards):
        out.append(_WORKER_WAIT.replace('@TAG@', shard['tag'])
                   .replace('@HOST@', '${w_hosts[%d]}' % i)
                   .replace('@K@', str(i + 1)).replace('@N@', str(n))
                   .replace('@I@', str(i)))
    if engine:
        out.append(_DISPATCH_TAIL)
    out.append(_COLLECTOR.replace(
        '@SRUN@', _srun_text('${w_hosts[0]}', '${JOB_DIR}')))
    return ''.join(out)


# =====================================================================================
# Done / in-flight / submission
# =====================================================================================
def registry_rows():
    """Every result row the registry holds, across all run shards."""
    return result_registry.load_all_rows(REPO_ROOT)


def job_is_done(job, rows):
    """True when this exact job's rows are already in the registry.

    'This exact job' includes the parallel shape it will run at -- a row produced on one
    node does not answer for a four-node run of the same benchmark, and crediting it
    would make Gate 2 unanswerable -- and the EXPERIMENT it belongs to: an experiment's
    rows never mark production done, and production's rows never mark an experiment
    done. Before that was true the control arm of an A/B had to be forced past a SKIP.
    """
    d = default_shape()
    shape = {k: int(job.meta.get(k, d[k])) for k in ('nodes', 'ep', 'tp', 'pp')}
    return rows_cover(rows, job.meta['run_tag'], job.step, job.meta['seq_len'],
                      job.expected_tasks, job.meta['num_fewshot'],
                      job.meta['padding'], limit=job.registry_limit, shape=shape,
                      experiment=job.meta.get('experiment'))


def squeue_job_ids():
    """This user's live job ids, or None when squeue cannot be consulted.

    None and the empty set mean different things: empty is "the queue is empty, submit
    everything", None is "the queue is unknown". The caller treats None as no
    in-flight work, which can duplicate a job -- the answer file makes that a resume
    rather than a repeat, so an unreachable scheduler slows a submit down instead of
    corrupting it.
    """
    try:
        out = subprocess.run(['squeue', '-h', '-u', os.environ.get('USER', ''),
                              '-o', '%A'], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return {l.strip() for l in out.stdout.decode().splitlines() if l.strip()}


def inflight_job_names(qids=None):
    """Job names covered by a ledger record whose job id is still in the queue.

    A dead job's record goes inert by itself: the work is then neither done nor
    in flight, the next pass resubmits it, and the answer file turns that into a
    resume. No state transitions, no cleanup.
    """
    if qids is None:
        qids = squeue_job_ids()
    names = set()
    if qids is None or not os.path.isfile(LEDGER):
        return names
    with open(LEDGER) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:                       # a half-written line is not a claim
                continue
            if str(rec.get('job_id')) in qids and rec.get('job_name'):
                names.add(rec['job_name'])
    return names


def write_slurm(job, out_dir=TEMP_SLURM):
    """Render this job's batch script. temp_slurm/ holds the script of every queued job."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'eval_{job.name}.slurm')
    with open(path, 'w') as fh:
        fh.write(render_template(TEMPLATE, job.subs))
    return path


def sbatch_command(path, walltime, dependency=None, qos=None):
    """afterany, not afterok: a slice that times out has still written its answers, and
    the collector reads the answer file rather than the slice's exit status.

    qos: passed through as -q. Frontier's `debug` QOS schedules far sooner (a pending
    debug job reports reason None rather than Priority) but caps walltime at 2h and
    allows only ONE SUBMITTED debug job per user at a time -- submit a second and
    sbatch refuses the whole batch. The owner's standing rule is that debug/validation
    runs go 20 minutes on debug; production waves stay on the default QOS so they can
    be submitted in bulk."""
    cmd = ['sbatch', '-t', walltime]
    if qos:
        cmd.extend(['-q', qos])
    if dependency:
        cmd.append(f'--dependency=afterany:{dependency}')
    cmd.append(path)
    return cmd


def record_submission(job, job_id, path):
    """Append this submission to the ledger, which is what makes a job "in flight".

    Locked because two people may run autorun_eval.sh at the same time and an
    interleaved append would lose a record -- and a lost record is a duplicate job.
    """
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    rec = dict(job.meta)
    rec.update({'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), 'ts_epoch': time.time(),
                'job_id': job_id, 'job_name': job.name, 'step': job.step,
                'slurm_file': path, 'limit': job.registry_limit,
                'tool': 'autorun_eval/1'})
    with open(LEDGER, 'a') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps(rec, default=str) + '\n')
        fh.flush()
    return rec


# =====================================================================================
# CLI -- driven by autorun_eval.sh
# =====================================================================================
def _print_plan(spec, jobs, skipped):
    """The plan, in the words of the declaration: what will run, what will not, and why.

    Every value printed is the value the job will actually use, so a default taken from
    the task table is never hidden behind a blank.
    """
    print(f'[{spec.run_tag()} :: {spec.benchmarks} :: steps {spec.steps}]')
    print(f'  model={spec.model} moe={spec.moe} time={spec.time} nodes={spec.nodes} '
          f'EP={spec.ep} TP={spec.tp} PP={spec.pp} '
          f'({spec.total_gpus()} GPUs, {spec.gpus_per_stage()} per stage)')
    if int(spec.pp) > 1:
        print(f'  answer-file writer: rank '
              f'{answer_writer_rank(spec.total_gpus(), spec.pp)} '
              f'(first rank of the last pipeline stage; rank 0 is on stage 0 and never '
              f'reaches put_batch)')
    # What this run will and will not be allowed to record. Printed on EVERY plan,
    # because "is this the published number?" is the question a declaration most often
    # gets wrong, and a silent default is what let five published cells be overwritten.
    if spec.experiment:
        print(f'  recording: EXPERIMENT {spec.experiment!r} -- every registry row is '
              f'tagged experiment={spec.experiment} and can NEVER become the published '
              f'value of a cell; answers go to a private file. Nothing published moves.')
    else:
        print('  recording: PRODUCTION -- these rows define the published value of each '
              'cell. Declare experiment=<tag> for a run that must not move published '
              'numbers.')
    if spec.answers_ns:
        print(f'  note: answers_ns={spec.answers_ns} is the DEPRECATED spelling of '
              f'experiment={spec.answers_ns}; it still means exactly this. Rename it.')
    for why, job in skipped:
        print(f'  SKIP-{why:<9} {job.name}')
    # Slice resume, at slice granularity: which slices of a sliced benchmark are
    # already in the registry. (Inside a slice, resume is finer than this -- the
    # answer file is keyed by request content, so only unanswered questions are
    # recomputed.)
    for total in sorted({j.slice_count for _, j in skipped if j.slice_count}):
        done = {j.slice_index for _, j in skipped if j.slice_count == total}
        print(f'  resuming slice(s) {leftover_slices(total, done)} of {total}; '
              f'{sorted(done)} already in the registry')
    if int(spec.workers) > 1:
        print(f'  workers={spec.workers}: ONE job, {spec.workers} independent replicas '
              f'(1 node x {GPUS_PER_NODE} GCDs each), one shared answer file, '
              f'in-job collector')
        if str(spec.split) == 'request':
            print(f'  split=request: no document is cut. Every replica BUILDS the whole '
                  f'benchmark -- so every few-shot prompt is bit-identical to an '
                  f'unsliced run -- and computes only its own blocks of the '
                  f'materialized request list. Worker rows are stamped partial; the '
                  f'collector scores.')
        if str(spec.dispatch) == 'engine':
            print(f'  dispatch=engine: a coordinator runs in the BATCH SHELL (plain CPU '
                  f'python on :{DISPATCH_PORT}, no torch and no model) and hands the '
                  f'next block to whichever replica is free, instead of each replica '
                  f'taking a fixed set decided now. This is the resume case: a static '
                  f'split balances documents (max/mean 1.06 measured) while the actual '
                  f'work is the cache misses (1.54, i.e. 35% idle). Every assignment is '
                  f'written to the audit trail ${{JOB_DIR}}/dispatch.journal.jsonl -- '
                  f'write-only, never read back, because the answer cache is the only '
                  f'authority on what is done; a lost coordinator degrades the job to '
                  f'dispatch=static, never to failure.')
        if int(spec.workers) < int(spec.nodes):
            print(f'  WARNING: nodes={spec.nodes} but workers={spec.workers} -- '
                  f'{int(spec.nodes) - int(spec.workers)} node(s) would sit idle')
    for job in jobs:
        tag = (f'slice {job.slice_index}/{job.slice_count}' if job.slice_index
               else ('collect' if job.depends_on
                     else (f'{job.workers} workers' if job.workers > 1 else 'single')))
        print(f'  SUBMIT {tag:<12} {job.name}')
        for shard in (job.worker_shards or []):
            print(f'         worker {shard["index"]}/{shard["count"]} '
                  f'({shard["split_by"]}, {shard["docs"]} docs) {shard["tag"][:120]}')
        print(f'         tasks={job.subs["TASK_LIST"]} mbs={job.subs["MICRO_BATCH_SIZE"]} '
              f'seq={job.subs["SEQ_LEN_OVERRIDE"]} fs={job.subs["NUM_FEWSHOT"] or 0} '
              f'{job.subs["LIMIT_FLAG"] or "(full)"}')
        if job.subs['EXTRA_EVAL_FLAGS']:
            print(f'         extra={job.subs["EXTRA_EVAL_FLAGS"]}')
        print(f'         answers -> {answer_file_name(**job.answer_identity)}')


def _read_declarations(path):
    """The declaration file autorun_eval.sh wrote -> validated-shape RunSpecs.

    Every line is parsed before any of them is planned, so a typo on the last line of
    the RUNS block is reported before the first job of the first line is rendered.
    """
    lines = []
    with open(path) as fh:
        for raw in fh:
            raw = raw.split('#', 1)[0].strip()
            if raw:
                lines.append(raw)
    if not lines:
        raise SystemExit(f'no run declared in {path}')
    return [parse_run(line) for line in lines]


def _split_done(jobs, rows, live, force):
    """Planned jobs -> (still to submit, [(why skipped, job)]).

    Done is judged from the registry and in-flight from the ledger; a job that is
    neither is submitted, and the answer file makes that a resume when the same work
    was interrupted earlier.
    """
    keep, skipped = [], []
    for job in jobs:
        if not force and job_is_done(job, rows):
            skipped.append(('done', job))
        elif job.name in live:
            skipped.append(('inflight', job))
        else:
            keep.append(job)
    return keep, skipped


def _submit(spec, keep, dry_run):
    """Render and queue every kept job, in order. Returns how many were submitted.

    Slices are submitted before the collector that depends on them, so their job ids
    are known by the time the dependency is built.
    """
    n_sub, slice_ids = 0, []
    for job in keep:
        dep = ':'.join(slice_ids) if job.depends_on == 'slices' and slice_ids else None
        if dry_run:
            # A dry run writes NOTHING -- not the .slurm file, not a manifest.
            # temp_slurm/ holds the scripts of jobs that are still queued.
            path = os.path.join(TEMP_SLURM, f'eval_{job.name}.slurm')
            render_template(TEMPLATE, job.subs)      # prove it renders
            if job.depends_on == 'slices':
                # No job ids exist yet, so show the shape the dependency will take.
                n = job.slice_count or len(keep) - 1
                dep = ':'.join(f'<slice{k}>' for k in range(1, n + 1))
            print('  DRY-RUN:', ' '.join(sbatch_command(path, spec.time, dep, spec.qos)))
            continue
        path = write_slurm(job)
        cmd = sbatch_command(path, spec.time, dep, spec.qos)
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        text = out.stdout.decode().strip()
        m = re.search(r'Submitted batch job (\d+)', text)
        if out.returncode != 0 or not m:
            print(f'  SBATCH FAILED for {path}: {text}')
            continue
        record_submission(job, m.group(1), path)
        print(f'  submitted {m.group(1)}  {job.name}')
        n_sub += 1
        if job.slice_index:
            slice_ids.append(m.group(1))
    return n_sub


def main(argv=None):
    """Plan and submit every declaration in --runs-file. Refusals exit 2, not 1."""
    ap = argparse.ArgumentParser(description='plan and submit evaluations from a '
                                             'declaration file')
    ap.add_argument('--runs-file', required=True,
                    help='one key=value declaration per line (written by autorun_eval.sh)')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the plan and the sbatch commands; submit nothing')
    ap.add_argument('--force', action='store_true',
                    help='submit work the registry already shows as done')
    ap.add_argument('--skip-preflight', action='store_true',
                    help='submit without checking datasets, packages and measured '
                         'sequence lengths')
    args = ap.parse_args(argv)

    specs = _read_declarations(args.runs_file)
    for spec in specs:                     # every declaration is validated ...
        validate(spec, skip_preflight=args.skip_preflight)
    rows = registry_rows()                 # ... before anything is planned or rendered
    qids = None if args.dry_run else squeue_job_ids()
    live = inflight_job_names(qids) if not args.dry_run else set()

    n_sub = 0
    for spec in specs:
        jobs = plan(spec, manifest_dir=None if args.dry_run else SLICE_DIR,
                    skip_preflight=True)   # preflight already ran, above
        keep, skipped = _split_done(jobs, rows, live, args.force)
        _print_plan(spec, keep, skipped)
        n_sub += _submit(spec, keep, args.dry_run)
    print(f'== {n_sub} job(s) submitted{" (dry-run)" if args.dry_run else ""} ==')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except BadParameter as exc:
        print(f'\nREFUSED AT SUBMIT TIME -- {exc}\n', file=sys.stderr)
        sys.exit(2)
