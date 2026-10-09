"""Resumable per-request response cache for the eval harness.

WHY THIS EXISTS (and why lm-eval's ``use_cache``/CachingLM was dropped)
  CachingLM commits its sqlite only AFTER the full underlying method call returns, and the
  0.4 evaluator sends all of a run's requests as one call -- an 11 h MMLU job killed at hour
  10 caches nothing. This cache lives inside the adapter and flushes after EVERY computed
  batch, so a walltime kill loses at most one batch. That is what lets 2-hour chained SLURM
  jobs pick up exactly where the previous one stopped, and lets a finished-from-cache job
  write the final results in minutes.

STORAGE: append-only JSONL, one file per (checkpoint x code x eval-config) identity.
  sqlite is the wrong tool on Lustre (WAL needs shared memory across clients; rollback
  journals lean on byte-range locks). Append-only complete lines with torn-line-tolerant
  reads is the same pattern result_registry has used all along. A torn final line from a
  kill is skipped on load; duplicate keys are last-wins. Line 1 is a ``{"_meta": ...}``
  provenance header; identity itself is enforced by the FILENAME, never by the header.

RANK PROTOCOL (lockstep is correctness, not an optimization)
  Every rank must arrive at the SAME hit/miss partition or the MoE all-to-all deadlocks
  with no error message. Rank 0 reads the file once and broadcasts the dict, so another
  job's writer appending between two ranks' reads cannot desynchronize them. During the
  run all ranks update their in-memory dict (they compute identical values by design);
  ONLY ONE rank appends to disk. No rank ever re-reads the file.

WHICH RANK WRITES (this is a pipeline-parallelism question, not a preference)
  The adapter calls put_batch() only where mpu.is_pipeline_last_stage() is true -- on a
  non-last stage the logits do not exist and the text would be garbage. The writer must
  therefore BE a last-stage rank. At PP=1 every rank is simultaneously the first and the
  last stage, so world rank 0 qualifies and always did. At PP>1 world rank 0 sits on
  stage 0, never reaches put_batch, and the answer file would receive its _meta header
  and nothing else -- an eval that looks like it ran and caches nothing.

  writer = the FIRST rank of the LAST pipeline stage. Megatron builds pipeline groups as
  `range(i, world_size, world_size // pp)` (megatron/core/parallel_state.py:264-266), so
  stage p owns the contiguous rank block [p*G, (p+1)*G) with G = world_size // pp, and
  the last stage starts at world_size - G. At pp=1 that expression is 0, i.e. bitwise
  today's behaviour -- see writer_rank()'s tests.

  The READER stays rank 0 and the broadcast source stays 0: reading is unconditional and
  rank 0 always exists, so there is no reason to move it and moving it would change the
  broadcast topology for no gain.
"""
import fcntl
import hashlib
import json
import os

import torch

# get() sentinel: distinguishes "not cached" from any legitimately cached value (which may
# be 0.0, "", or false-y).
MISS = object()


def writer_rank(world_size, pipeline_size):
    """World rank that appends to the answer file: first rank of the LAST pipeline stage.

    Pure arithmetic so it can be reasoned about and tested without a job. Returns 0 for
    any degenerate input (pp <= 1, pp not dividing world_size, pp > world_size), which is
    both the historical behaviour and the only safe answer when the topology is not the
    one Megatron builds.
    """
    world_size = int(world_size or 1)
    pipeline_size = int(pipeline_size or 1)
    if pipeline_size <= 1 or pipeline_size > world_size or world_size % pipeline_size:
        return 0
    return world_size - world_size // pipeline_size


def _pipeline_size(args=None):
    """Pipeline degree, from mpu if it is initialized, else from args, else 1.

    mpu is authoritative: it reports what the process groups were ACTUALLY built with,
    which is what is_pipeline_last_stage() answers against. args is the fallback for a
    caller that has not initialized megatron (the unit tests, and any future offline use).
    """
    try:
        from megatron.core import mpu
        size = int(mpu.get_pipeline_model_parallel_world_size())
        if size:
            return size
    except Exception:
        pass
    return int(getattr(args, 'pipeline_model_parallel_size', 1) or 1)


def request_key(method, req_args):
    """Stable content key for one lm-eval request.

    Canonical JSON with sort_keys so dict-valued args (generate_until's gen_kwargs) hash
    identically regardless of insertion order. The method name is inside the hashed
    payload AND in the readable prefix.
    """
    canon = json.dumps([method, list(req_args)], sort_keys=True, ensure_ascii=False,
                       separators=(',', ':'), default=str)
    return f'{method}:{hashlib.sha256(canon.encode("utf-8")).hexdigest()[:32]}'


class ResponseCache:
    """See module docstring. Construct via ``ResponseCache.build(args, ...)``."""

    def __init__(self, path, meta, is_writer):
        self.path = path
        self.meta = meta
        self.is_writer = is_writer
        self._data = {}
        self._fh = None
        self.hits = 0
        self.misses = 0
        self._load_and_sync()

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, args, code_fp, cache_dir, print_fn=print):
        """Return a ResponseCache, or None when the flag is off / no identity derivable.

        The filename is the cache's whole identity:
          WHICH WEIGHTS  ckpt_id (run dir + step) + mtime of the actual weights artifact
          WHICH CODE     code_fp over the forward-relevant sources (result_registry)
          WHICH CONFIG   weight source, padding side, seq_length, num_fewshot
        --limit is deliberately NOT in the key: keys are per-request, so a --limit smoke
        run pre-warms the full run instead of living in a separate file.
        """
        if not getattr(args, 'lm_eval_response_cache', False):
            return None

        step_dir = getattr(args, 'ckpt_step_dir', None)
        cons = getattr(args, 'consolidated_load', None)
        if step_dir:
            step_dir = os.path.normpath(step_dir)
            ckpt_id = (f'{os.path.basename(os.path.dirname(step_dir))}'
                       f'__{os.path.basename(step_dir)}')
        elif cons:
            ckpt_id = os.path.basename(cons).replace('.pt', '')
            print_fn('[response-cache] WARN: --ckpt-step-dir not passed; keying on the '
                     'consolidated filename (works, but pass it for run-qualified identity).')
        elif getattr(args, 'load', None):
            ckpt_id = os.path.basename(os.path.normpath(args.load))
            print_fn('[response-cache] WARN: --ckpt-step-dir not passed; falling back to '
                     'basename(--load). Under UCP that is the per-job selector dir -- its '
                     'NAME is stable but carries no training-run identity, so two runs with '
                     'the same step number would collide. Pass --ckpt-step-dir.')
        else:
            print_fn('[response-cache] WARN: no checkpoint identity available; '
                     'cache disabled for this run.')
            return None

        # Weight source, and the artifact whose mtime means "the bytes changed". For the
        # consolidated path that is the .pt itself (a re-consolidation from the same step
        # dir must invalidate); for UCP/DS it is the step dir's model-states file (the
        # UCP conversion output is derived from it, and the per-job .eval_ucp selector
        # dir is rebuilt every job -- keying on THAT was the bug that made the old cache
        # never hit across runs).
        if cons:
            ws, weights_artifact = 'cons', cons
        else:
            ws = 'ucp' if getattr(args, 'universal_checkpoint', False) else 'ds'
            weights_artifact = (os.path.join(step_dir, 'mp_rank_00_model_states.pt')
                                if step_dir else None)
        try:
            mtime = int(os.path.getmtime(weights_artifact)) if weights_artifact else 0
        except OSError:
            mtime = 0
        if mtime == 0:
            print_fn(f'[response-cache] WARN: cannot stat {weights_artifact}; using mtime=0 '
                     '(a re-converted checkpoint would NOT invalidate this cache).')

        fs = getattr(args, 'num_fewshot', 0) or 0
        # Decode-mode qualifier: the KV path (default) keeps the legacy unqualified
        # name; --no-kv-decode oracle runs get their own file. Both modes share one
        # code_fp now that they are a runtime flag, so without this an A/B pair
        # collides -- the second job reads the first mode's generations as hits and
        # computes nothing. Loglikelihood rows are decode-mode-independent; forking
        # them in the rare nokv run costs recompute, never correctness.
        mode = '_nokv' if getattr(args, 'no_kv_decode', False) else ''
        # EXPERIMENT fork (--experiment <tag>, deprecated spelling --answers-ns): the
        # parallel shape is deliberately NOT in this name, so a 4-node run reads the
        # 1-node answers as hits and a Gate-2 "does the new shape match?" comparison is
        # vacuously true. An experiment forks the file so such a run recomputes into its
        # own; unset (the default) keeps every existing filename byte for byte.
        #
        # The SUFFIX stays `_ns<tag>` even though the declaration key is now
        # `experiment=`: the five experiment answer files already on disk must keep
        # resolving, and this string is a filename, not an identity -- the identity is
        # the `experiment` field on the registry row (result_registry.experiment_of).
        ns_tag = (getattr(args, 'experiment', None)
                  or getattr(args, 'answers_ns', None))
        ns = f'_ns{ns_tag}' if ns_tag else ''
        fname = (f'{ckpt_id}_m{mtime}_code{code_fp}_{ws}'
                 f'_pad{args.padding_side}_seq{args.seq_length}_fs{fs}{mode}{ns}.jsonl')
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, fname)

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            w_rank = writer_rank(torch.distributed.get_world_size(), _pipeline_size(args))
        else:
            rank, w_rank = 0, 0
        meta = {
            'ckpt_step_dir': step_dir,
            'weights_artifact': weights_artifact,
            'mtime': mtime,
            'code_fp': code_fp,
            'weight_source': ws,
            'padding_side': getattr(args, 'padding_side', None),
            'seq_len': getattr(args, 'seq_length', None),
            'num_fewshot': fs,
            'decode_mode': 'nokv' if mode else 'kv',
            # `experiment` is the current name; `answers_ns` is kept as its deprecated
            # mirror so an answer file written before 2026-08-28 and one written after
            # are read by the same code.
            'experiment': ns_tag,
            'answers_ns': ns_tag,
            'created_by_job': os.environ.get('SLURM_JOB_ID'),
            'writer_rank': w_rank,
        }
        if getattr(args, 'dp_split', False):
            # The data-parallel prompt split: each attention group answers its own blocks, so each group's first
            # tensor-parallel rank appends to the file (several writers under the file lock, as several workers' jobs
            # already are; put_batch skips a key already present).
            from megatron.core import mpu as _mpu
            if _mpu.get_data_parallel_world_size() > 1:
                w_rank = int(_mpu.get_tensor_model_parallel_src_rank())
        cache = cls(path, meta, is_writer=(rank == w_rank))
        print_fn(f'[response-cache] {len(cache._data)} cached responses loaded from {path}')
        if w_rank != 0:
            print_fn(f'[response-cache] writer is rank {w_rank} (first rank of the last '
                     f'pipeline stage); rank 0 is on stage 0 and never reaches put_batch')
        return cache

    # ------------------------------------------------------------- load/sync
    def _load_and_sync(self):
        state = None
        dist_ok = torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if dist_ok else 0
        if rank == 0:
            state = {}
            if os.path.isfile(self.path):
                with open(self.path, 'r', encoding='utf-8') as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except ValueError:          # torn final line from a walltime kill
                            continue
                        if 'k' in rec:
                            state[rec['k']] = rec['v']   # duplicate keys: last wins
        if dist_ok:
            # One broadcast so every rank holds byte-identical state even if another
            # job's writer appended between two ranks' reads. Divergent dicts mean
            # divergent miss sets mean a hung all-to-all -- this is a correctness step,
            # not an optimization.
            box = [state]
            torch.distributed.broadcast_object_list(box, src=0)
            state = box[0]
        self._data = state or {}
        if self.is_writer:
            self._fh = open(self.path, 'a', encoding='utf-8')
            # Create-plus-header is a critical section, and the "is it new?" check is
            # re-done on the OPEN handle INSIDE it. Two jobs creating the same file
            # both used to stat "new" before either had written the header, and the
            # file got two _meta lines (the observed race). flock, not byte-range
            # locks: the scope is the whole file, and whole-file flock is what Lustre
            # honours across clients.
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            try:
                if os.fstat(self._fh.fileno()).st_size == 0:
                    self._fh.write(json.dumps({'_meta': self.meta},
                                              ensure_ascii=False) + '\n')
                    self._fh.flush()
                    os.fsync(self._fh.fileno())
            finally:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)

    # -------------------------------------------------------------- get/put
    def get(self, method, req_args):
        """-> (key, value | MISS). Every rank calls this with identical arguments."""
        key = request_key(method, req_args)
        if key in self._data:
            self.hits += 1
            return key, self._data[key]
        self.misses += 1
        return key, MISS

    def put_batch(self, pairs):
        """pairs: [(key, json-serializable value)]. All ranks update memory; only the
        writer appends + fsyncs. Call after EVERY computed batch -- per-batch flushing IS
        the resumability guarantee. None keys (uncacheable, e.g. rolling windows) skipped.

        The append is ONE locked write per batch. Writing line by line into the buffered
        handle used to flush at arbitrary buffer boundaries (not line boundaries), and
        kernel append atomicity is only promised up to PIPE_BUF -- less than one long
        generation line -- so two jobs' writers could interleave a torn line. The lock is
        per-batch, never per-run, so concurrent jobs still interleave at batch
        granularity.
        """
        lines = []
        for key, value in pairs:
            if key is None or key in self._data:
                continue
            self._data[key] = value
            if self._fh is not None:
                lines.append(json.dumps({'k': key, 'v': value},
                                        ensure_ascii=False) + '\n')
        if self._fh is not None and lines:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            try:
                self._fh.write(''.join(lines))
                self._fh.flush()
                os.fsync(self._fh.fileno())
            finally:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)

    def __len__(self):
        return len(self._data)
