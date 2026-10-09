# test_runtime_pp.py -- pipeline-parallel awareness of the GPU probe suite.
#
# Evidence: job 5363010 (first PP4xTP4xEP8 eval, 4 nodes / 32 ranks) built the
# model, then every non-last-stage rank crashed in tests/test_runtime.py at
# `S = logits.shape[1]` (`AttributeError: 'NoneType' object has no attribute
# 'shape'`, _seq_chunks <- _t_logits_no_nan_no_inf <- run_all) -- under PP only
# the last pipeline stage materializes logits (lm_eval_adapter._model_call
# returns None elsewhere), and the probe suite dereferenced them on every rank.
# The surviving last-stage ranks then wedged on the orphaned forward collective
# until walltime. The 1-node twin 5363011 ran probes 14/14: the suite is fine
# at PP=1.
#
# These tests drive the REAL run_all through two mock ranks of the 5363010
# topology (world=32, pp=4): a stage-0-shaped rank whose _model_call returns
# None, and a last-stage-shaped rank that holds logits. The verdict broadcast
# is a single simulated collective (_Wire). The contract under test:
#   * no rank crashes;
#   * every rank makes EXACTLY the same model calls (the forward's collectives
#     require full participation -- a rank that skips one deadlocks the rest);
#   * run_all returns identical (ok, failed_names) on every rank, judged where
#     the logits exist and broadcast from there (WO-1's pattern, verdict-sized);
#   * a probe no rank can judge at PP>1 surfaces as an explicit
#     'skipped-under-pp:<name>' entry -- never a silent pass.
#
# FAIL-BEFORE: on the pre-PP-aware test_runtime.py, the stage-0-shaped rank
# reproduces the 5363010 AttributeError, so this file is red until the fix
# lands (test_stage0_shaped_rank_runs_run_all_without_crashing).
#
# SECOND FINDING (job 5363247, the second PP4 attempt, post PP-awareness;
# FIXED by LANDING 3): the judge itself died — `IndexError: index 9 is out of
# bounds for dimension 1 with size 8` in _t_argmax_diversity_across_prompts.
# The adapter's pipeline-engine _model_call branch USED to return
# eval_batch's per-micro-batch outputs in Megatron's seq-major [s, b, v]
# layout (GPTModelPipe emits [s, b, v]; CrossEntropy transposes labels to
# match, gpt_model.py:771; only non-pipe GPTModel transposes to [b, s, v],
# gpt_model.py:431) cat'd and sliced along dim 0 — the SEQ dim — so the
# judge received [rows, mbs, vocab], not [rows, seq, vocab]. LANDING 3
# transposes each micro-batch to [b, s, v] before the cat
# (lm_eval_adapter.py:1085-1093) and removes the interim
# _skip_if_pipe_engine gates from test_runtime.py.
#
# The mock therefore carries BOTH code states:
#   * pipe_engine_shape=True — FAITHFUL model of the LANDED pipe branch:
#     pad rows to a micro_batch_size multiple, per-micro-batch [S, mbs, V]
#     from eval_batch, transpose each back to [mbs, S, V], cat dim 0, slice
#     [:rows]. The full run_all suite must pass on it: canary + all 13
#     formerly-gated logits/score probes judging real values.
#   * pre_landing3_shape=True — the .pre code path (snapshot in
#     rebuild/fp_migration/landing3_snapshots/): cat WITHOUT the transpose,
#     the [rows, mbs, vocab] garble of job 5363247. The fail-before lives
#     here: the canary must catch it with the root-cause diagnosis, and a
#     full run_all must fail FAST (the _TR fail-fast rule) after that
#     diagnosis is printed — never a silent pass.

import io
import os
import sys
import unittest
import zlib
from contextlib import redirect_stdout

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import test_runtime  # noqa: E402  (the probe suite under test)

# The 5363010 topology.
WORLD, PP = 32, 4
LAST_STAGE_FIRST_RANK = WORLD - WORLD // PP  # 24: cache.writer_rank(32, 4)

# Mock model dimensions. V must clear the probe's `vocab >= 50_000` assertion;
# S small enough that _multiseq_probes_fit() keeps every probe enabled.
V, S, K = 50304, 128, 97
# The sciq task-table micro-batch size in job 5363247; the garbled pipe return
# is [rows, MBS, V] and rank 24's IndexError was index 9 against this dim.
MBS = 8


def _make_base():
    """[K, V] token-local logit table. Deterministic (fixed generator); sigma=1
    keeps softmax entropy ~log(V)-0.5 (inside the probe's (0.5, 11.0) band) and
    the eot-logprob margin well above -5; the per-row +6 spike at a distinct
    column gives distinct argmaxes across rows for the diversity probe."""
    g = torch.Generator().manual_seed(536_3010)
    base = torch.randn(K, V, generator=g)
    for k in range(K):
        base[k, (k * 131 + 17) % V] += 6.0
    return base


class _FakeArgs:
    def __init__(self, use_left_pad_eval=False):
        self.use_left_pad_eval = use_left_pad_eval
        self.reset_attention_mask = False
        self.reset_position_ids = False
        self.eod_mask_loss = False
        self.adaptive_seq_len = False


class _FakeLM:
    """One rank of a PP>1 eval, shaped like MegatronDeepSpeedLM where the probe
    suite touches it. The two PP-relevant contracts are copied from the
    adapter, not invented:
      * _model_call materializes logits ONLY on the last pipeline stage and
        returns None elsewhere (lm_eval_adapter.py:1090-1092);
      * _loglikelihood_tokens returns REAL scores on every rank -- WO-1
        broadcasts them from the last stage (lm_eval_adapter.py:901-904).
    Logits are token-local (row = base[token % K]), so causal masking, padding
    invariance, batch invariance and adapter/manual scoring alignment all hold
    exactly and every probe genuinely PASSES on the judging rank. Every model
    call is appended to self.calls so tests can assert cross-rank lockstep.
    """

    def __init__(self, base, is_last_stage, pipe=True, use_left_pad_eval=False,
                 pipe_engine_shape=False, pre_landing3_shape=False, mbs=MBS):
        self.base = base
        self.args = _FakeArgs(use_left_pad_eval)
        self.model = object()
        self.max_length = S
        self.device = 'cpu'
        self.vocab_size = V
        self.EOT_TOKEN_ID = 0
        self._padding_side = 'right'
        self.is_pipe_parallel = pipe
        self.is_last_stage = is_last_stage
        # pipe_engine_shape=True models the LANDED (LANDING 3) pipe-engine
        # _model_call faithfully: pad/chunk to micro-batches, seq-major
        # [S, mbs, V] from eval_batch, transpose back, cat, slice. With
        # pre_landing3_shape=True it instead reproduces the .pre code path
        # (job 5363247): the same micro-batch outputs cat'd WITHOUT the
        # transpose -> [rows, mbs, vocab]. pipe_engine_shape=False returns
        # [B, S, V] directly (PP=1, or shape-mechanics not under test).
        self.pipe_engine_shape = pipe_engine_shape
        self.pre_landing3_shape = pre_landing3_shape
        self.mbs = mbs
        self.batch_size = mbs   # the adapter exposes this; the canary prints it
        self.calls = []

    def tok_encode(self, text, add_special_tokens=False):
        # Stable across processes (crc32, not hash()); ids clear of the 0 pad.
        return [zlib.crc32(w.encode()) % 4000 + 3 for w in text.split()]

    def _row(self, token_id):
        return self.base[int(token_id) % K]

    def _build_padded_batch(self, real_tokens_list, padding_length, side='right'):
        assert side == 'right', 'mock models the house right-pad path only'
        tokens = torch.zeros((len(real_tokens_list), padding_length),
                             dtype=torch.long)
        inplens = []
        for b, t in enumerate(real_tokens_list):
            tokens[b, :t.shape[0]] = t
            inplens.append(int(t.shape[0]))
        return tokens, None, None, inplens

    def _model_call(self, inps, position_ids=None, attention_mask=None):
        self.calls.append(('model_call', tuple(inps.shape),
                           position_ids is not None,
                           attention_mask is not None))
        if self.is_pipe_parallel and not self.is_last_stage:
            return None  # the adapter's non-last-stage contract
        if not (self.is_pipe_parallel and self.pipe_engine_shape):
            return self.base[inps % K]  # [B, S] -> [B, S, V], token-local
        # Faithful pipe-engine mechanics (lm_eval_adapter.py:1071-1093):
        # rows padded to a multiple of micro_batch_size; eval_batch hands
        # back each micro-batch's last-stage output in Megatron's seq-major
        # [s, b, v] layout (gpt_model.py:771 vs :431).
        n = int(inps.shape[0])
        new_size = ((n + self.mbs - 1) // self.mbs) * self.mbs
        if new_size > n:
            pad = torch.zeros((new_size - n, inps.shape[1]), dtype=inps.dtype)
            inps = torch.cat([inps, pad], dim=0)
        full = self.base[inps % K]                      # [new_size, S, V]
        output = [full[i:i + self.mbs].transpose(0, 1)  # eval_batch: [S, mbs, V]
                  for i in range(0, new_size, self.mbs)]
        if self.pre_landing3_shape:
            # The .pre code path (job 5363247; landing3_snapshots/*.pre):
            # cat along dim 0 — the SEQ dim — then slice it to the row
            # count -> [rows, mbs, vocab], the garble that killed rank 24.
            return torch.cat(output, 0)[:n]
        # The LANDED pipe branch: transpose each micro-batch back to
        # [b, s, v] so the cat runs along the true batch dim and [:n]
        # strips exactly the pad rows -> [n, S, V].
        return torch.cat([o.transpose(0, 1) for o in output], 0)[:n]

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        self.calls.append(('ll_tokens',
                           tuple((len(r[1]), len(r[2])) for r in requests)))
        out = []
        for _key, ctx, cont in requests:
            inp = list(ctx) + list(cont)
            logp, greedy = 0.0, True
            for i in range(len(ctx), len(inp)):
                row = F.log_softmax(self._row(inp[i - 1]), dim=-1)
                logp += float(row[inp[i]].item())
                greedy = greedy and int(row.argmax().item()) == inp[i]
            out.append((logp, greedy))
        return out


class _Wire:
    """The verdict broadcast as one simulated collective. The judge rank runs
    first: its broadcast_object_list call records the payload. The stage-0
    rank runs second: the same call hands the recorded payload back."""

    def __init__(self, world=WORLD, pp=PP):
        self.world, self.pp = world, pp
        self.sent = None
        self.srcs = []

    def writer_rank(self, world, pp):
        assert (world, pp) == (self.world, self.pp), 'topology must pass through'
        return world - world // pp  # cache.writer_rank for a well-formed topology

    def broadcast_object_list(self, payload, src=None):
        self.srcs.append(src)
        if payload[0] is not None:
            self.sent = payload[0]
        else:
            assert self.sent is not None, 'stage-0 ran before the judge (test bug)'
            payload[0] = self.sent

    def channel_for(self, _lm):
        return (self.world, self.pp, self.writer_rank,
                self.broadcast_object_list)


# All labels run_all registers at S=128 (multiseq probes fit).
ALL_PROBES = [
    'pipe_logits_shape_contract',
    'forward_returns_logits', 'logits_no_nan_no_inf', 'logits_finite_range',
    'logits_distribution_entropy', 'causal_mask_sanity',
    'left_pad_batch_size_invariance', 'eot_continuation_logprob_sane',
    'argmax_diversity_across_prompts', 'left_right_pad_equivalence',
    'padding_token_invariance', 'loglikelihood_rolling_self_consistency',
    'loglikelihood_scoring_alignment', 'padding_length_invariance',
    'content_length_scoring_alignment',
]

# Every probe that reads logits or scores through the adapter — gated by
# _skip_if_pipe_engine while the pipe branch was broken (job 5363247),
# deliberately UN-GATED by LANDING 3: all 13 must now judge real values on
# the last-stage rank and PASS. In run_all order.
LOGITS_SCORE_PROBES = [
    'forward_returns_logits', 'logits_no_nan_no_inf', 'logits_finite_range',
    'logits_distribution_entropy', 'causal_mask_sanity',
    'left_pad_batch_size_invariance', 'eot_continuation_logprob_sane',
    'argmax_diversity_across_prompts',
    'padding_token_invariance', 'loglikelihood_rolling_self_consistency',
    'loglikelihood_scoring_alignment', 'padding_length_invariance',
    'content_length_scoring_alignment',
]


class TestRunAllUnderPipelineParallelism(unittest.TestCase):

    def _run_pair(self, use_lpe=False, pipe_shape=True):
        """run_all on a last-stage-shaped rank then a stage-0-shaped rank of
        the same world, sharing one logit table and one verdict wire.
        pipe_shape=True (default) is the FAITHFUL model of the LANDED
        adapter: the judge's _model_call runs the full pad/chunk/seq-major/
        transpose-back/cat/slice mechanics and returns well-formed
        [B, S, vocab]. pipe_shape=False bypasses the mechanics (plain
        [B, S, vocab] return). The .pre garble is reached separately via
        pre_landing3_shape=True in the fail-before tests."""
        base = _make_base()
        judge = _FakeLM(base, is_last_stage=True, use_left_pad_eval=use_lpe,
                        pipe_engine_shape=pipe_shape)
        stage0 = _FakeLM(base, is_last_stage=False, use_left_pad_eval=use_lpe,
                         pipe_engine_shape=pipe_shape)
        wire = _Wire()
        orig = test_runtime._verdict_channel
        test_runtime._verdict_channel = wire.channel_for
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                verdict_judge = test_runtime.run_all(judge)
                verdict_stage0 = test_runtime.run_all(stage0)
        finally:
            test_runtime._verdict_channel = orig
        return verdict_judge, verdict_stage0, judge, stage0, wire, buf.getvalue()

    def test_stage0_shaped_rank_runs_run_all_without_crashing(self):
        """THE 5363010 regression (fail-before): on the pre-PP-aware suite the
        stage-0 shape dies with AttributeError NoneType .shape in _seq_chunks
        at the second probe -- exactly the crash in rank_0.log -- and the
        peers it abandoned wedge on the orphaned forward. Passing means every
        probe on a logits-less rank completes without dereferencing None."""
        (_, _), (_, _), judge, stage0, _, _ = self._run_pair()
        # The design invariant that prevents the drain pattern: both ranks
        # made EXACTLY the same model calls, in the same order.
        self.assertEqual(
            [c for c in judge.calls],
            [c for c in stage0.calls],
            'ranks diverged in their model-call sequence -- under a real PP '
            'forward the missing collective deadlocks the peers')

    def test_verdict_is_identical_on_every_rank_and_judged_where_logits_exist(self):
        """run_all must return the same verdict on stage 0 as on the last
        stage -- main.py stamps the registry `probes` field from world rank 0
        (stage 0), which holds no logits -- and the broadcast source must be
        the first rank of the last stage (cache.writer_rank's answer, 24 in
        this world), the only stage that judged. On the FAITHFUL model of the
        LANDED adapter the verdict is: every probe PASSES -- the canary
        against the fixed [B, S, vocab] return, and all 13 formerly-gated
        logits/score probes judging real values through the full
        pad/chunk/transpose/cat/slice mechanics. ok=True, no failures, no
        skipped-under-pp."""
        vj, v0, _, _, wire, out = self._run_pair()
        self.assertEqual(vj, v0, 'ranks returned different probe verdicts')
        ok, flagged = vj
        self.assertTrue(ok, 'the landed pipe branch must pass the full '
                            'suite:\n' + out)
        self.assertEqual(flagged, [],
                         'no probe may fail or skip on the landed adapter')
        self.assertEqual(wire.srcs, [LAST_STAGE_FIRST_RANK] * 2,
                         'verdict must be broadcast from the judging stage')
        passed, failed, skipped = wire.sent
        self.assertEqual([label for label, _ in passed], ALL_PROBES)
        self.assertEqual(failed, [])
        self.assertEqual(skipped, [])

    def test_canary_catches_pre_landing3_shape_with_root_cause(self):
        """Fail-before, pointed at the .pre code path: against the
        pre-LANDING 3 return ([rows, mbs, vocab], the landing3_snapshots/
        .pre cat-without-transpose) the canary fails as an ASSERTION -- a
        verdict, not rank 24's bare IndexError -- and its message carries
        the diagnosis: the observed garbled shape, the regressed pipe-branch
        cat/slice, and the warning that SCORING shares the return. That
        message is what the job log hands the operator on any future
        regression."""
        judge = _FakeLM(_make_base(), is_last_stage=True,
                        pipe_engine_shape=True, pre_landing3_shape=True)
        with self.assertRaises(AssertionError) as cm:
            test_runtime._t_pipe_logits_shape_contract(judge)
        msg = str(cm.exception)
        self.assertIn(f'({1}, {MBS}, {V})', msg.replace('torch.Size', ''),
                      'message must show the observed [rows, mbs, vocab] shape')
        self.assertIn('lm_eval_adapter.py:1085-1093', msg)
        self.assertIn('SCORING', msg)

    def test_skipped_under_pp_is_explicit_never_a_silent_pass(self):
        """A skip is not a failure and never a silent pass. With the 13
        logits/score probes un-gated by LANDING 3, the one structural skip
        left at PP>1 is left_right_pad_equivalence under --use-left-pad-eval
        (the 7a finding: the eval_batch path derives its own masks and drops
        the crafted left-pad kwargs). It surfaces as
        'skipped-under-pp:<name>' on every rank, ok stays True, every other
        probe passes, and no skipped label leaks into the pass/fail lists."""
        vj, v0, _, _, wire, _ = self._run_pair(use_lpe=True)
        self.assertEqual(vj, v0)
        ok, flagged = vj
        self.assertTrue(ok, 'a skip must not read as a failure')
        self.assertEqual(flagged,
                         ['skipped-under-pp:left_right_pad_equivalence'])
        passed, failed, skipped = wire.sent
        self.assertEqual([label for label, _ in skipped],
                         ['left_right_pad_equivalence'])
        self.assertEqual(failed, [])
        self.assertEqual([label for label, _ in passed],
                         [n for n in ALL_PROBES
                          if n != 'left_right_pad_equivalence'])

    def test_landed_adapter_ungates_probes_under_pp(self):
        """Policy pin, FLIPPED by LANDING 3 exactly as its predecessor
        demanded ('when pipe_logits_shape_contract passes under PP for real,
        remove the gate and flip these expectations'): the interim
        _skip_if_pipe_engine gate is gone from test_runtime, and against the
        landed adapter shape all 13 formerly-gated logits/score probes judge
        real values on the last-stage rank and land in the PASSED list --
        none skipped-under-pp, none failed."""
        self.assertFalse(hasattr(test_runtime, '_skip_if_pipe_engine'),
                         'the interim PP gate has come back -- it was '
                         'deliberately removed with the LANDING 3 adapter fix')
        vj, v0, _, _, wire, _ = self._run_pair()
        self.assertEqual(vj, v0)
        ok, flagged = vj
        self.assertTrue(ok)
        self.assertEqual(flagged, [])
        passed, failed, skipped = wire.sent
        passed_labels = [label for label, _ in passed]
        self.assertIn('pipe_logits_shape_contract', passed_labels,
                      'the landed [B, S, vocab] shape must satisfy the canary')
        for name in LOGITS_SCORE_PROBES:
            self.assertIn(name, passed_labels,
                          f'{name} must judge real values under PP, '
                          f'not skip or fail')
        self.assertEqual(failed, [])
        self.assertEqual(skipped, [])

    def test_regressed_shape_fails_fast_only_after_canary_diagnosis(self):
        """What a future regression to the .pre shape looks like with the
        gates deliberately gone: run_all on the pre-LANDING 3 judge fails
        FAST with rank 24's IndexError class (the _TR fail-fast rule --
        non-assertion errors propagate rather than poisoning Megatron's
        timers), but only AFTER the canary, which runs first, has printed
        its FAIL with the full root-cause diagnosis. The job log therefore
        always carries the diagnosis, never just the bare IndexError."""
        judge = _FakeLM(_make_base(), is_last_stage=True,
                        pipe_engine_shape=True, pre_landing3_shape=True)
        wire = _Wire()
        orig = test_runtime._verdict_channel
        test_runtime._verdict_channel = wire.channel_for
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                with self.assertRaises(IndexError):
                    test_runtime.run_all(judge)
        finally:
            test_runtime._verdict_channel = orig
        out = buf.getvalue()
        self.assertIn('[FAIL] pipe_logits_shape_contract', out,
                      'the canary must have failed BEFORE the crash')
        self.assertIn('SCORING', out,
                      'the printed diagnosis must carry the scoring warning')

    def test_pre_landing3_mock_reproduces_rank24_indexing_crash(self):
        """Mock fidelity: the .pre garbled return really does reproduce the
        crash mechanism from rank_24.log. A 1-row call comes back [1, mbs, V]
        (job 5363247's forward probe saw 'got 8, expected 2048'); indexing
        the last content position of a prompt longer than mbs tokens -- the
        unguarded expression the argmax probe used pre-fix -- raises the
        IndexError that killed rank 24 (index 9, size 8 there; index 8,
        size 8 here with the whitespace tokenizer)."""
        judge = _FakeLM(_make_base(), is_last_stage=True,
                        pipe_engine_shape=True, pre_landing3_shape=True)
        toks = torch.tensor(
            judge.tok_encode("The cell is the basic structural unit of all"),
            dtype=torch.long)  # 9 whitespace tokens -> last_idx 8 >= mbs 8
        last_idx = int(toks.shape[0]) - 1
        self.assertGreaterEqual(last_idx, MBS)
        tokens, _, _, _ = judge._build_padded_batch([toks], S)
        logits = judge._model_call(tokens)
        self.assertEqual(tuple(logits.shape), (1, MBS, V),
                         'the pipe-engine return is [rows, mbs, vocab]')
        with self.assertRaises(IndexError):
            logits[0, last_idx, :]  # rank 24's crash expression

    def test_mock_modes_match_the_real_adapter_code_states(self):
        """Ties the two mock shapes to the two REAL code states so neither is
        a strawman: the LIVE adapter's pipe branch carries LANDING 3's
        per-micro-batch transpose-before-cat, and the .pre snapshot
        (rebuild/fp_migration/landing3_snapshots/, the landing's audit
        substrate) carries the un-transposed cat/slice that job 5363247 died
        on. If the live pipe branch ever loses the transpose, this fails
        alongside the canary."""
        fixed_cat = ('torch.cat([o.transpose(0, 1) for o in output], 0)'
                     '[:len(inps)]')
        garbled_cat = ('torch.cat(output, 0)[:len(inps)]\n'
                       '                      if output is not None else None')
        adapter = os.path.normpath(os.path.join(
            HERE, '..', '..', '..', '..', 'tasks', 'eval_harness',
            'megatron_ds_eval', 'lm_eval_adapter.py'))
        with open(adapter) as fh:
            live = fh.read()
        self.assertIn(fixed_cat, live,
                      "live pipe branch lost LANDING 3's transpose-before-cat")
        self.assertNotIn(garbled_cat, live,
                         'live pipe branch has regressed to the .pre cat/slice')
        pre = os.path.normpath(os.path.join(
            HERE, '..', 'rebuild', 'fp_migration', 'landing3_snapshots',
            'lm_eval_adapter.py.pre'))
        if not os.path.exists(pre):
            self.skipTest('landing3 .pre snapshot not on disk (archived?)')
        with open(pre) as fh:
            pre_src = fh.read()
        self.assertIn(garbled_cat, pre_src,
                      'the .pre snapshot must carry the un-transposed '
                      'cat/slice the pre_landing3 mock models')
        self.assertNotIn(fixed_cat, pre_src)

    def test_landed_layout_feeds_scoring_slice_arithmetic(self):
        """SCORING was the exposure (LIFT_RECORD 7b): _loglikelihood_tokens
        iterates dim 0 of the SAME return as rows and slices
        logits_i[inplen-contlen:inplen] along each row's SEQ dim
        (lm_eval_adapter.py:857-875). Run the exact landed expression --
        transpose each seq-major micro-batch, cat dim 0, slice [:rows] --
        on identity-encoded values (value = row*1000 + seq position) and
        verify by VALUE, across a micro-batch boundary and with pad rows in
        play, that row r is global row r and the scoring slice selects true
        seq positions. The .pre expression on the same inputs yields the
        [rows, mbs, vocab] garble the canary's shape check rejects."""
        rows, mbs, S_, V_ = 11, 4, 16, 6  # rows deliberately not a mbs multiple
        new_size = ((rows + mbs - 1) // mbs) * mbs
        ident = (torch.arange(new_size).view(-1, 1, 1) * 1000
                 + torch.arange(S_).view(1, -1, 1)
                 ).expand(new_size, S_, V_).float()          # [b, s, v] truth
        # eval_batch's per-micro-batch return: seq-major [s, mbs, v].
        output = [ident[i:i + mbs].transpose(0, 1)
                  for i in range(0, new_size, mbs)]
        inps = list(range(rows))  # only len(inps) matters to the expression
        logits = torch.cat([o.transpose(0, 1) for o in output], 0)[:len(inps)]
        self.assertEqual(tuple(logits.shape), (rows, S_, V_))
        for r in (0, mbs - 1, mbs, rows - 1):  # spans a micro-batch boundary
            self.assertTrue(torch.equal(logits[r], ident[r]),
                            f'row {r} of the landed return is not global '
                            f'row {r}')
        inplen, contlen, r = 10, 3, mbs + 1
        sl = logits[r][inplen - contlen:inplen]  # the adapter's scoring slice
        self.assertEqual([int(x) for x in sl[:, 0]],
                         [r * 1000 + s for s in range(inplen - contlen, inplen)],
                         'scoring slice must select true seq positions of '
                         'the true row')
        garbled = torch.cat(output, 0)[:len(inps)]  # the .pre expression
        self.assertEqual(tuple(garbled.shape), (rows, mbs, V_),
                         'the .pre expression yields the canary-rejected '
                         '[rows, mbs, vocab] shape')

    def test_broadcast_verdict_pure_wiring(self):
        """_broadcast_verdict_from_judge is WO-1's score-broadcast pattern at
        verdict size: source = writer_rank(world, pp); non-judges adopt the
        judge's lists; an empty payload raises instead of stamping an
        unjudged status."""
        seen = {}

        def writer_rank(world, pp):
            seen['args'] = (world, pp)
            return 24

        verdict = ([('p', 0.1)], [('f', 'msg')], [('s', 'why')])

        def bcast_judge(payload, src=None):
            seen['src'] = src
            seen['payload'] = payload[0]

        got = test_runtime._broadcast_verdict_from_judge(
            verdict, True, WORLD, PP, writer_rank, bcast_judge)
        self.assertEqual(seen['args'], (WORLD, PP))
        self.assertEqual(seen['src'], 24)
        self.assertEqual(tuple(got), tuple(seen['payload']))

        def bcast_stage0(payload, src=None):
            payload[0] = seen['payload']

        got0 = test_runtime._broadcast_verdict_from_judge(
            ([], [], []), False, WORLD, PP, writer_rank, bcast_stage0)
        self.assertEqual(tuple(got0), tuple(got), 'non-judge must adopt the '
                                                  "judge's verdict")

        def bcast_empty(payload, src=None):
            payload[0] = None

        with self.assertRaises(RuntimeError):
            test_runtime._broadcast_verdict_from_judge(
                ([], [], []), False, WORLD, PP, writer_rank, bcast_empty)

    def test_pp1_contract_is_untouched(self):
        """At PP=1 (is_pipe_parallel False) every rank judges locally: no
        verdict channel is opened and run_all returns (True, []) on the same
        mock that passes under PP -- the 5363011 twin's 14/14."""
        def _forbidden(_lm):
            raise AssertionError('PP=1 must not open a verdict channel')

        lm = _FakeLM(_make_base(), is_last_stage=True, pipe=False)
        orig = test_runtime._verdict_channel
        test_runtime._verdict_channel = _forbidden
        try:
            with redirect_stdout(io.StringIO()):
                ok, flagged = test_runtime.run_all(lm)
        finally:
            test_runtime._verdict_channel = orig
        self.assertTrue(ok)
        self.assertEqual(flagged, [])

    def test_main_py_stamps_skips_distinguishably(self):
        """WO-9 truthfulness at PP>1: main.py must partition run_all's flagged
        names on the 'skipped-under-pp:' marker so a skipped probe lands in
        the registry row's `probes` field (after ';') instead of vanishing
        into 'passed' or masquerading as a failure."""
        main_path = os.path.normpath(os.path.join(
            HERE, '..', '..', '..', '..', 'tasks', 'eval_harness',
            'megatron_ds_eval', 'main.py'))
        with open(main_path) as fh:
            src = fh.read()
        block = src[src.index('import test_runtime'):][:3000]
        self.assertIn("skipped-under-pp:", block,
                      'main.py no longer separates skipped-under-pp entries '
                      'when stamping the probes field')


if __name__ == '__main__':
    unittest.main()
