"""MegatronDeepSpeedLM -- the lm-eval <-> Megatron-DeepSpeed engine bridge.

lm-evaluation-harness (0.4+) defines an abstract `LM` interface; this class implements
it by driving the engine's collective forward pass: batching, padding layout, logprob
extraction, greedy decoding. It is engine-level, not model-specific -- any model served
by this Megatron-DeepSpeed fork evaluates through it.
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval import utils as lm_utils

from megatron import get_args, print_rank_0
from megatron.core import mpu
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.utils import get_ltor_masks_and_position_ids

from deepspeed.accelerator import get_accelerator

from .cache import MISS


def _broadcast_scores_from_last_stage(res, is_last_stage, world_size, pipeline_size,
                                      writer_rank_fn, broadcast_object_list_fn):
    """WO-1 core: hand every rank the LAST pipeline stage's real results.

    At PP>1 only the last stage holds logits; every other stage used to return
    np.random.rand() dummies while world rank 0 -- stage 0 -- wrote results.json
    and the registry rows (main.py:287-290). This function is the fix: ONE
    broadcast over the default (world) process group whose source is the writer
    rank -- the FIRST rank of the LAST pipeline stage, the exact rank
    cache.writer_rank() computes (world - world//pp; Megatron builds stage p as
    the contiguous block [p*G, (p+1)*G), G = world//pp, so the last stage starts
    at world - G). After it, every rank -- including the results writer on
    stage 0 -- holds byte-identical real scores.

    Pure wiring by design: the topology decision (who is source, what is sent)
    takes plain values and injected collaborators so the login-node test can
    drive a simulated world through the REAL shipped code with no torch.
    Collaborators:
      writer_rank_fn            cache.writer_rank (the single authority)
      broadcast_object_list_fn  torch.distributed.broadcast_object_list
    Raises instead of ever returning a dummy: if the payload comes back empty
    the topology is wrong and random numbers must not reach the registry.
    """
    src = writer_rank_fn(world_size, pipeline_size)
    payload = [list(res) if is_last_stage else None]
    broadcast_object_list_fn(payload, src=src)
    if payload[0] is None:
        raise RuntimeError(
            f'PP score broadcast returned no payload: source rank {src} '
            f'(world={world_size}, pp={pipeline_size}) did not hold last-stage '
            f'results. Refusing to fabricate scores.')
    return payload[0]


def with_extra_until(gen_kwargs, extra):
    """The request's gen_kwargs with the adapter-wide extra stop strings appended to its `until` list (a copy; the
    strings already present are not repeated). Applied before the cache lookup, so answers generated with a different
    stop set never share a key. Why: lm-eval's humaneval stops at "\\nclass", "\\ndef", "\\n#", "\\nif", "\\nprint" and
    not at a markdown fence; the released DeepSeek-V3 closes its completion with ``` and a prose explanation, which
    the create_test filter appends to the program (a SyntaxError, so a correct function scores 0: job 5476348, 3 of
    8 with four fenced completions). The bigcode harness stops at the fence as well."""
    if not extra:
        return gen_kwargs
    u = gen_kwargs.get('until', [])
    u = [u] if isinstance(u, str) else list(u or [])
    return dict(gen_kwargs, until=u + [e for e in extra if e not in u])


class MegatronDeepSpeedLM(LM):
    """0.4+ LM subclass that routes lm-eval requests through a Megatron-DeepSpeed
    distributed model.

    Implements all three request types: `loglikelihood` / `loglikelihood_rolling`
    (batched right-pad scoring through `_model_call`) and `generate_until` (batched
    greedy decoding -- currently a full re-forward per generated token; see its
    docstring for why and for the KV-cache replacement criteria).
    """

    def __init__(self, model, tokenizer, response_cache=None, extra_until=None):
        self.extra_until = list(extra_until or [])   # --extra-until: stop strings appended to every generation request
        super().__init__()
        args = get_args()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        # Resumable per-request cache (megatron_ds_eval.cache.ResponseCache) or None.
        # INVARIANT: with None, every code path below is byte-equivalent to the
        # pre-cache harness -- that is the bisection guarantee.
        self._cache = response_cache
        self.vocab_size = tokenizer.vocab_size
        self.EOT_TOKEN_ID = tokenizer.eod

        self._max_length = args.seq_length
        self._batch_size = args.micro_batch_size

        # Padding side: 'left' (opt-in, requires args.use_left_pad_eval AND
        # the matching Megatron-side patches in rotary_pos_embedding.py +
        # FlashSelfAttention) or 'right' (default; matches the proven
        # evaluate.py behavior bitwise: zeros at the END of each row,
        # Megatron's standard causal mask, sequential position_ids).
        side = getattr(args, 'padding_side', 'right')
        if side not in ('left', 'right'):
            raise ValueError(
                f"--padding-side must be 'left' or 'right', got '{side}'")
        if side == 'left' and not getattr(args, 'use_left_pad_eval', False):
            raise ValueError(
                "--padding-side=left also requires --use-left-pad-eval so "
                "the Megatron-side rotary/FlashSelfAttention patches activate.")
        self._padding_side = side

        self.is_main = args.rank == 0
        self._device = get_accelerator().current_device_name()
        self.is_pipe_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
        self.is_data_parallel = mpu.get_data_parallel_world_size() > 1
        # The data-parallel prompt split (main.py --dp-split, 2026-09-14): the attention groups of this model copy hold
        # DIFFERENT requests. The lockstep primitives below then work inside the group, and the groups agree over the whole
        # job only on the NUMBER of forwards of a round (_world_max), an idle group filling in with _empty_forward.
        self.dp_split = False
        self.adaptive_seq_len = args.adaptive_seq_len

        if self.is_data_parallel and args.moe_expert_parallel_size == 1:
            raise NotImplementedError(
                "Data parallelism is not supported at eval time for "
                "non-MoE models."
            )

        self.is_last_stage = (
            True if not self.is_pipe_parallel
            else mpu.is_pipeline_last_stage()
        )

    # -- 0.4+ required properties / metadata ----------------------------
    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def eot_token_id(self):
        return self.EOT_TOKEN_ID

    # Lie to lm-evaluation-harness about our parallelism layout.
    #
    # lm-eval 0.4+ is built for data-parallel inference: when world_size > 1,
    # it partitions requests across ranks and uses all_gather to coordinate.
    # That works for HF-style replicated models but NOT for expert-parallel:
    # our N physical ranks each hold 1/N of the experts, so EVERY request
    # needs ALL ranks to participate in a collective forward through the
    # DeepSpeed engine. We can't split requests by rank.
    #
    # Reporting rank=0, world_size=1 makes lm-eval treat us as a single
    # logical model. All N physical ranks then execute the same eval loop
    # in lockstep; each rank's _model_call triggers the collective forward
    # (which uses all N ranks under the hood); each rank arrives at the same
    # logprob result. We then gate the JSON write to world-rank 0 in main().
    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    # -- Tokenization ---------------------------------------------------
    def tok_encode(self, text, add_special_tokens=False):
        """Tokenize text via the same HFTokenizer used at training time."""
        from megatron.tokenizer.gpt2_tokenization import GPT2Tokenizer
        if isinstance(self.tokenizer.tokenizer, GPT2Tokenizer):
            return self.tokenizer.tokenizer.encode(text)
        # HF tokenizers library: .encode(text).ids
        enc = self.tokenizer.tokenizer.encode(text)
        if hasattr(enc, 'ids'):
            return enc.ids
        return enc

    def tok_decode(self, tokens):
        return self.tokenizer.detokenize(tokens)

    # -- 0.4+ public API: loglikelihood ---------------------------------
    def loglikelihood(self, requests):
        """Score (context, continuation) pairs.

        Each Instance carries .args = (context_str, continuation_str).
        Returns list[(logprob, is_greedy)].

        With a response cache attached, cached requests are answered from memory and only
        misses are computed; each computed batch is flushed inside _loglikelihood_tokens,
        so a walltime kill loses at most one batch. All ranks hold identical cache state,
        so the hit/miss partition is identical everywhere (EP lockstep preserved).
        """
        results = [None] * len(requests)
        pending = []                              # (original_index, cache_key or None)
        for i, req in enumerate(requests):
            context, continuation = req.args
            if self._cache is not None:
                key, val = self._cache.get('loglikelihood', (context, continuation))
                if val is not MISS:
                    results[i] = (float(val[0]), bool(val[1]))
                    continue
            else:
                key = None
            pending.append((i, key))
        if self._cache is not None:
            print_rank_0(f">>> [response-cache] loglikelihood: "
                         f"{len(requests) - len(pending)} hits / {len(pending)} to compute "
                         f"(of {len(requests)})")

        new_reqs = []
        for i, key in pending:
            context, continuation = requests[i].args
            if context == "":
                context_enc = [self.EOT_TOKEN_ID]
            else:
                context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation)
            new_reqs.append((key, context_enc, continuation_enc))

        computed = self._loglikelihood_tokens(new_reqs)
        for (i, _key), r in zip(pending, computed):
            results[i] = r
        return results

    def loglikelihood_rolling(self, requests):
        """Rolling-window scoring (used by wikitext for perplexity).
        Not wired for the data-parallel prompt split (its forwards are not counted per round).

        Each Instance carries .args = (string,). Returns list[float] of
        total NLLs.
        """
        loglikelihoods = []
        with torch.no_grad():
            for req in tqdm(requests, disable=not self.is_main):
                string, = req.args
                # Cached at DOCUMENT granularity: one request = one document = several
                # windows, and the windows themselves stay uncacheable (key None below).
                if self._cache is not None:
                    key, val = self._cache.get('loglikelihood_rolling', (string,))
                    if val is not MISS:
                        loglikelihoods.append(float(val))
                        continue
                else:
                    key = None
                windows = list(map(
                    lm_utils.make_disjoint_window,
                    lm_utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.EOT_TOKEN_ID,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                ))
                # _loglikelihood_tokens expects (cache_key, ctx_enc, cont_enc)
                windows = [(None,) + w for w in windows]
                string_nll = self._loglikelihood_tokens(windows, disable_tqdm=True)
                total_nll = sum(x[0] for x in string_nll)
                # Guarded to the last PP stage: non-last stages return dummy values that
                # must never enter the cache. (At PP>1 the writer, world rank 0, sits on
                # the FIRST stage, so the cache is simply inert there -- safe.)
                if self._cache is not None and mpu.is_pipeline_last_stage():
                    self._cache.put_batch([(key, float(total_nll))])
                loglikelihoods.append(total_nll)
        return loglikelihoods

    @property
    def max_gen_toks(self):
        # lm-eval default cap when a task's gen_kwargs doesn't set max_gen_toks.
        return 256

    def _use_kv_decode(self):
        """KV decode is the default on the no-pipeline path; --no-kv-decode forces the
        legacy full-reforward loop (the string-exact equivalence oracle)."""
        args = get_args()
        return (bool(getattr(args, 'no_pipeline_parallel', False))
                and not getattr(args, 'no_kv_decode', False))

    # -- WO-2: per-step decode lockstep (the generation-hang fix) ------------
    #
    # THE HANG (directive 3.2): every rank used to decide LOCALLY when a
    # row/batch was done. MoE routing is bf16-ULP nondeterministic across
    # ranks, so on a near-tie two ranks argmax different tokens, hit stop
    # conditions at different steps, and leave the loop at different times --
    # the next MoE all-to-all then blocks forever. Census: 7 of 8 stale logs
    # at KV-decode step 3, 1 inside PREFILL (job 5355830) -- the prefill case
    # is the same divergence surfacing at a row/batch boundary (one rank has
    # moved on to the next row's prefill while a peer still decodes), so a fix
    # keyed on decode steps alone is insufficient. These primitives make EVERY
    # per-step decision collective: the sampled token is broadcast from the
    # decode authority (the writer rank: first rank of the last pipeline
    # stage under PP; world rank 0 under pure EP -- cache.writer_rank() is the
    # single source of that arithmetic), and the frozen/done mask is
    # all-reduced (MAX == elementwise OR, see _merge_done_flags) so every
    # rank advances rows, enters prefill, and exits loops in lockstep.
    #
    # Recorded outputs are PRESERVED where no hang occurred: at PP=1 the
    # authority is rank 0, which was already the rank whose text the cache
    # writer recorded -- broadcasting rank 0's token changes nothing rank 0
    # ever wrote. Behaviour changes exactly in the divergence/hang cases,
    # which by definition never completed and never cached (see the patch
    # header and FP_MIGRATION_PLAN Gate-1b's per-method split verdict).
    # Single process (world size 1): all primitives are exact no-ops.

    def _decode_lockstep_enabled(self):
        """Lockstep is on whenever more than one rank runs the collective."""
        return (torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1)

    def _decode_src(self):
        """Decode authority = the writer rank (cache.writer_rank arithmetic):
        first rank of the last pipeline stage under PP, world rank 0 at pp=1."""
        import torch.distributed as dist
        from .cache import writer_rank
        return writer_rank(dist.get_world_size(),
                           mpu.get_pipeline_model_parallel_world_size())

    def _agree_next_tokens(self, nxt, batch_size):
        """Broadcast the greedily chosen token(s) from the decode authority.

        nxt: list[int] on ranks that hold logits; None on ranks that do not
        (non-last PP stages). Returns the agreed list[int] on every rank, or
        the input unchanged when lockstep is off (single process). All stop
        decisions downstream are pure functions of the agreed tokens plus
        state that is itself derived from agreed tokens, so agreement here
        makes every rank's control flow identical by construction.
        """
        if not self._decode_lockstep_enabled():
            return nxt
        import torch.distributed as dist
        t = torch.tensor(nxt if nxt is not None else [0] * batch_size,
                         dtype=torch.long, device=self.device)
        if self.dp_split:
            dist.broadcast(t, src=mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())
        else:
            dist.broadcast(t, src=self._decode_src())
        return [int(x) for x in t.tolist()]

    def _agree_done_flags(self, flags):
        """All-reduce(MAX) of the done/frozen mask == elementwise OR across
        ranks (_merge_done_flags is the pure spec, tested on the login node):
        a row is done EVERYWHERE as soon as it is done ANYWHERE, so no rank
        can keep forwarding a batch a peer has abandoned. Executed once per
        step by every rank, so the collective itself is always matched.
        """
        if not self._decode_lockstep_enabled():
            return flags
        import torch.distributed as dist
        t = torch.tensor([1 if f else 0 for f in flags], dtype=torch.long,
                         device=self.device)
        if self.dp_split:
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        else:
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return [bool(int(x)) for x in t.tolist()]

    # -- the data-parallel prompt split (2026-09-14) -------------------------
    #
    # Drawings 1-3 and 6 of records/2026-09/0915_2026_EVAL_FRAMEWORK_STUDY.html: the
    # attention groups (the TP groups at pp=1) are the rooms, the experts over the
    # whole job the hall every forward must enter together. A room may hold any
    # prompts; the only thing the rooms share is the count of forwards. Each round
    # they agree on the max of the prefill chunks and, step by step, on whether any
    # room is still decoding; a room with nothing to do runs one dummy-token forward
    # (vLLM's execute_dummy_batch, Megatron's dummy_forward, SGLang's idle batch).

    def _world_max(self, n):
        """The max of an integer over the whole job (all groups); the integer itself when the split is off."""
        if not (self.dp_split and self._decode_lockstep_enabled()):
            return int(n)
        import torch.distributed as dist
        t = torch.tensor([int(n)], dtype=torch.long, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return int(t.item())

    def _empty_forward(self, causal=None):
        """One forward of one dummy token, no cache: this group enters every collective of a forward it has no work for."""
        tok = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        with torch.no_grad():
            self.model(tok, tok, None if causal is None else causal[:, :, :1, :1])

    def _idle_generation_round(self, causal=None):
        """The forwards of one generation round for a group with no rows in it: the agreed prefill chunks, then one empty
        forward per decode step while any group still decodes (the same two agreements _kv_generate_rows_batched makes)."""
        for _ in range(self._world_max(0)):
            self._empty_forward(causal)
        while self._world_max(0):
            self._empty_forward(causal)

    def _idle_scoring_round(self):
        """One forward of a dummy batch of one row, the twin of a scoring chunk's forward on the busy groups."""
        tok = torch.zeros((1, self.max_length), dtype=torch.long, device=self.device)
        with torch.no_grad():
            self._model_call(tok, content_slices=[(0, 1)])

    def _kv_generate_rows(self, ctx_ids, until, max_gen, on_row=None):
        """KV-cached greedy generation via the model's own InferenceParams machinery.

        ONE ROW AT A TIME (B=1), deliberately: FlashSelfAttention's decode branch takes
        (q, k, v) only -- no mask -- so a BATCHED cache with variable-length prompts would
        let short rows attend to their own padding's K/V. Per-row there is no padding
        anywhere: prefill is exactly [1, L], each step exactly [1, 1] at its true offset,
        and the rotary table is sized to this row's total length (language_model.py sizes
        it from inference_params.max_sequence_len -- which is also why NO pad-to-seq_length
        happens here at all). Cost per generated token: one [1, 1] forward instead of the
        legacy [B, seq_length] re-forward.

        Semantics are a token-for-token mirror of the legacy loop (the oracle): greedy
        argmax; EOT stops WITHOUT being emitted; stop strings are checked on the decoded
        continuation after every append (a stop can straddle a token boundary) and
        truncated again at the end. EP lockstep holds: every rank runs the identical
        per-row stream and argmaxes identical logits.
        """
        from megatron.text_generation.forward_step import InferenceParams

        self.model.eval()
        self.model.micro_batches = 1
        texts = []
        # ALLOCATE ONCE, REUSE FOREVER. The KV buffers, the InferenceParams handle, and
        # the causal mask are process-lifetime objects sized to max_length; rows reuse
        # them by resetting sequence_len_offset to 0 (transformer.py derives
        # is_first_step from the offset, not from buffer allocation, precisely to allow
        # this). Nothing inference-owned is freed or reallocated between rows -- the
        # earlier free-and-realloc versions (per-row sizes, then constant sizes with
        # per-row release) both ballooned the caching allocator to 99% HBM and cut step
        # rate ~4-8x (jobs 5266669, 5267301/15/19): long-lived KV blocks interleaving
        # with variable-length per-row activations re-fragmented the pool every row.
        # Pinning KV once leaves only short-lived activation churn, which the allocator
        # absorbs. Stale cache beyond each row's sequence_end is never read.
        total_len = self.max_length
        if getattr(self, '_kv_ip', None) is None:
            self._kv_ip = InferenceParams(1, total_len)
            # [1, 1, S, S] bool causal (True = masked). Flash ignores it; the non-flash
            # core path needs the sliced rows, same pattern as
            # megatron/text_generation/generation.py:183-185.
            # The dense [1,1,S,S] bool mask is 4.3 GB at 64k / 17 GB at 131k, and under
            # flash it is NEVER READ (the comment above already says so). It was a top
            # memory hog in the 64k gen OOM (job 5385510: MoE a2a asked 1.49 GiB with
            # 44 GB held). Allocate it only for the non-flash core path.
            if bool(getattr(get_args(), 'use_flash_attn_v2', False) or getattr(get_args(), 'use_flash_attn_v1', False)):
                self._kv_causal = None
            else:
                causal = torch.triu(torch.ones(total_len, total_len, dtype=torch.bool,
                                               device=self.device), diagonal=1)
                self._kv_causal = causal.unsqueeze(0).unsqueeze(0)
        ip = self._kv_ip
        assert ip.max_sequence_len == total_len, \
            'max_length changed across generate_until chunks; persistent KV buffers ' \
            'are sized once and cannot shrink/grow'
        causal = self._kv_causal
        with torch.no_grad():
            for ids in ctx_ids:
                L = len(ids)
                # Reuse the persistent buffers: offset 0 makes this forward a prefill
                # (full-prefix rotary) that overwrites cache rows [0:L] in place.
                ip.sequence_len_offset = 0

                toks = torch.tensor([ids], dtype=torch.long, device=self.device)
                pos = torch.arange(L, device=self.device).unsqueeze(0)
                # Job 5264961 hung symmetrically somewhere in this row loop with no output;
                # these breadcrumbs localize prefill vs first-step vs later-step hangs.
                print_rank_0(f">>> [kv] row {len(texts) + 1}: prefill [1,{L}] ...")
                out = self.model(toks, pos, None if causal is None else causal[:, :, :L, :L], inference_params=ip)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                logits = gather_from_tensor_model_parallel_region(logits)[..., :self.vocab_size]
                torch.cuda.synchronize()
                print_rank_0(f">>> [kv] row {len(texts) + 1}: prefill done")
                ip.sequence_len_offset = L
                # WO-2: agree the first generated token before any stop
                # decision -- per-rank argmax on near-tie logits is the hang.
                nxt = self._agree_next_tokens(
                    [int(logits[0, -1].argmax(-1).item())], 1)[0]

                gen, cur = [], L
                for _ in range(max_gen):
                    if nxt == self.EOT_TOKEN_ID:
                        break                    # stop signal, never emitted
                    gen.append(nxt)
                    if until and any(u in self.tok_decode(gen) for u in until):
                        break
                    if len(gen) >= max_gen or cur + 1 >= total_len:
                        break                    # budget exhausted; skip the unused forward
                    st = torch.tensor([[nxt]], dtype=torch.long, device=self.device)
                    sp = torch.tensor([[cur]], dtype=torch.long, device=self.device)
                    if len(gen) <= 3:
                        print_rank_0(f">>> [kv] row {len(texts) + 1}: step {len(gen)} "
                                     f"(pos {cur}) ...")
                    out = self.model(st, sp, None if causal is None else causal[:, :, cur:cur + 1, :cur + 1],
                                     inference_params=ip)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                    logits = gather_from_tensor_model_parallel_region(logits)[..., :self.vocab_size]
                    ip.sequence_len_offset += 1
                    cur += 1
                    # WO-2: agreed token -> identical EOT/stop-string/budget
                    # decisions -> every rank breaks this loop, and enters the
                    # NEXT row's prefill, on the same iteration (the prefill
                    # hang of job 5355830 is a row-boundary divergence).
                    nxt = self._agree_next_tokens(
                        [int(logits[0, -1].argmax(-1).item())], 1)[0]

                text = self.tok_decode(gen)
                for u in until:
                    j = text.find(u)
                    if j != -1:
                        text = text[:j]
                texts.append(text)
                if on_row is not None:
                    # Per-ROW cache flush: at ~1 min/row a per-chunk flush loses the whole
                    # chunk to a walltime kill (jobs 5264961/5265418 proved it). Every
                    # finished generation must hit disk before the next row starts.
                    on_row(len(texts) - 1, text)
        return texts

    def _kv_generate_rows_batched(self, ctx_ids, until, max_gen, on_row=None):
        """Phase 5b: the whole chunk decodes together on one shared KV cache.

        Measured motivation: a [1,1] decode step costs ~1.3 s regardless of batch --
        28 layers of kernel launches + an EP all-to-all per forward -- so per-row
        decode pays it per ROW-token while this path pays it per BATCH-token (~B x
        throughput). Positions are per-row (ragged): ONE right-padded [B, Lmax]
        prefill writes every row's K/V at its true positions (right-pad => slot ==
        true position; the causal mask keeps real tokens from attending pad, exactly
        like the scoring forward); decode steps are [B, 1] with per-row positions in
        InferenceParams.row_lengths, which transformer.py's ragged branch uses for
        per-row rotary angles, true-position cache scatter, and per-row attention
        extents (eager masked attention -- flash decode takes no mask). Semantics
        mirror the per-row path exactly: greedy, EOT stops without emitting, stop
        strings on the decoded continuation, per-row budget, final truncation, and a
        per-row cache flush the moment a row finishes. Finished rows are FROZEN
        (keep stepping on their last token, output ignored, write position clamped)
        so tensor shapes and EP collectives stay uniform until the batch drains.
        """
        from megatron.text_generation.forward_step import InferenceParams

        self.model.eval()
        self.model.micro_batches = 1
        B = len(ctx_ids)
        total_len = self.max_length
        if getattr(self, '_kv_ip_b', None) is None:
            cap = max(self.batch_size, B)
            self._kv_ip_b = InferenceParams(cap, total_len)
            if bool(getattr(get_args(), 'use_flash_attn_v2', False) or getattr(get_args(), 'use_flash_attn_v1', False)):
                self._kv_causal_b = None   # flash never reads it; see _kv_causal note
            else:
                causal = torch.triu(torch.ones(total_len, total_len, dtype=torch.bool,
                                               device=self.device), diagonal=1)
                self._kv_causal_b = causal.unsqueeze(0).unsqueeze(0)
        ip = self._kv_ip_b
        assert B <= ip.max_batch_size and ip.max_sequence_len == total_len, \
            'chunk exceeds the persistent batched-KV capacity'
        causal = self._kv_causal_b

        L = [len(x) for x in ctx_ids]
        Lmax = max(L)
        with torch.no_grad():
            toks = torch.zeros((B, Lmax), dtype=torch.long, device=self.device)
            for b, ids in enumerate(ctx_ids):
                toks[b, :len(ids)] = torch.tensor(ids, dtype=torch.long,
                                                  device=self.device)
            pos = torch.arange(Lmax, device=self.device).unsqueeze(0).expand(B, -1)
            ip.row_lengths = None            # uniform branch handles the prefill
            # CHUNKED PREFILL (2026-08-31): a monolithic 131072-token prefill cannot fit
            # at TP1 -- the LM head materializes [1, L, vocab] (25.6 GiB bf16) on top of
            # weights (13) + KV (26.8). Chunking bounds the logits to one chunk; the KV
            # accumulates in place exactly as decode steps do (a decode step IS a
            # 1-token chunk: offset-derived rotary + KV append, transformer.py:840-960).
            # Chunks engage only for single-row prefills longer than KV_PREFILL_CHUNK,
            # so every previously-working length is byte-identical.
            # 8192 default since 2026-08-31: at 16384, 131k prefills sat ~60GiB/GCD and
            # EP-rank routing skew OOM'd half the workers (jobs 5386833/5386857). Halving
            # the chunk halves per-chunk activation + expert all-to-all buffers.
            import os as _os_env
            KV_PREFILL_CHUNK = int(_os_env.getenv('XMOE_KV_PREFILL_CHUNK', '8192'))
            _chunk_off = 0
            _n_chunks = (Lmax + KV_PREFILL_CHUNK - 1) // KV_PREFILL_CHUNK if (B == 1 and Lmax > KV_PREFILL_CHUNK) else 1
            _n_chunks_max = self._world_max(_n_chunks)       # the data-parallel split: the round's prefill forwards
            if B == 1 and Lmax > KV_PREFILL_CHUNK:
                print_rank_0(f">>> [kv-b] CHUNKED prefill [1,{Lmax}] in {KV_PREFILL_CHUNK}-token chunks ...")
                out = None
                for _k in range(0, Lmax, KV_PREFILL_CHUNK):
                    _e = min(_k + KV_PREFILL_CHUNK, Lmax)
                    ip.sequence_len_offset = _k
                    out = self.model(toks[:, _k:_e], pos[:, _k:_e],
                                     None if causal is None else causal[:, :, _k:_e, :_e],
                                     inference_params=ip)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                logits = gather_from_tensor_model_parallel_region(logits)[..., :self.vocab_size]
                _chunk_off = Lmax - logits.shape[1]
            else:
                ip.sequence_len_offset = 0       # first step: full-prefix rotary
                print_rank_0(f">>> [kv-b] prefill [{B},{Lmax}] ...")
                out = self.model(toks, pos, None if causal is None else causal[:, :, :Lmax, :Lmax],
                                 inference_params=ip)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                logits = gather_from_tensor_model_parallel_region(logits)[..., :self.vocab_size]
            torch.cuda.synchronize()
            print_rank_0(f">>> [kv-b] prefill done")
            for _ in range(_n_chunks_max - _n_chunks):
                self._empty_forward(causal)                  # another group's longer prompt: enter its chunks' collectives
            # nonzero scalar offset = "not a first step" for every decode forward;
            # actual positions travel per-row in ip.row_lengths from here on
            ip.sequence_len_offset = Lmax
            # WO-2: agree the whole first-token row before any freeze decision.
            nxt = self._agree_next_tokens(
                [int(logits[b, L[b] - 1 - _chunk_off].argmax(-1).item()) for b in range(B)], B)
            # The top-2 margin probe (2026-09-16, XMOE_GAP_PROBE=1): read-only on the same logits, after the choice.
            _probe = _os_env.getenv('XMOE_GAP_PROBE', '0') == '1'
            gaps = [[self._top2_margin(logits[b, L[b] - 1 - _chunk_off])] for b in range(B)] if _probe else None

            gens = [[] for _ in range(B)]
            frozen = [False] * B
            texts = [None] * B
            # host-side per-row positions: pos_h[b] = slot of the token being fed
            # this step. Frozen rows STOP advancing (they re-scatter at their last
            # slot, output ignored) so a long row that finishes early cannot drag
            # the batch attention extent upward forever; and the extent (kv_extent)
            # is derived here on the host so the model layers never pay a
            # .max().item() GPU sync (28x/token otherwise).
            pos_h = [min(l, total_len - 1) for l in L]

            def _finalize(b):
                if texts[b] is not None:
                    return
                text = self.tok_decode(gens[b])
                for u in until:
                    j = text.find(u)
                    if j != -1:
                        text = text[:j]
                texts[b] = text
                if on_row is not None:
                    on_row(b, text)

            step = 0
            while True:
                if step < max_gen:
                    for b in range(B):
                        if frozen[b]:
                            continue
                        if nxt[b] == self.EOT_TOKEN_ID:
                            frozen[b] = True         # stop signal, never emitted
                            _finalize(b)
                            continue
                        gens[b].append(nxt[b])
                        stop = bool(until) and any(
                            u in self.tok_decode(gens[b]) for u in until)
                        if stop or len(gens[b]) >= max_gen \
                                or L[b] + len(gens[b]) >= total_len:
                            frozen[b] = True
                            _finalize(b)
                    # WO-2: agree the frozen mask (all-reduce MAX == OR) so every
                    # rank drains the batch -- and leaves for the next chunk's
                    # prefill -- on the same step. One collective per step, always
                    # matched because it is unconditional (inside the group under
                    # the data-parallel split).
                    frozen = self._agree_done_flags(frozen)
                # The data-parallel split: the groups agree, once per step, whether ANY of
                # them still decodes; a group whose rows are all frozen keeps entering the
                # step's collectives with an empty forward until none does. Without the
                # split _world_max returns its input and this is the old loop exactly.
                running = step < max_gen and not all(frozen)
                if not self._world_max(running):
                    break
                if not running:
                    self._empty_forward(causal)
                    step += 1
                    continue
                step += 1
                for b in range(B):
                    if not frozen[b]:
                        pos_h[b] = L[b] + len(gens[b]) - 1   # slot of the fed token
                st = torch.tensor([[t] for t in nxt], dtype=torch.long,
                                  device=self.device)
                rl = torch.tensor(pos_h, dtype=torch.long, device=self.device)
                ip.row_lengths = rl
                ip.kv_extent = max(pos_h) + 1
                if step <= 2:
                    print_rank_0(f">>> [kv-b] step {step + 1} "
                                 f"(pos {min(pos_h)}..{max(pos_h)}) ...")
                out = self.model(st, rl.unsqueeze(1), None if causal is None else causal[:, :, :1, :1],
                                 inference_params=ip)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                logits = gather_from_tensor_model_parallel_region(logits)[..., :self.vocab_size]
                # WO-2: agreed tokens keep gens/frozen/pos_h identical everywhere.
                nxt = self._agree_next_tokens(
                    [int(logits[b, -1].argmax(-1).item()) for b in range(B)], B)
                if gaps is not None:
                    for b in range(B):
                        if not frozen[b]:
                            gaps[b].append(self._top2_margin(logits[b, -1]))
            ip.row_lengths = None                # never leak into the next prefill
            ip.kv_extent = None
            for b in range(B):
                _finalize(b)
        self._last_gaps = gaps                   # the probe's record for this batch (None when off)
        return texts

    @staticmethod
    def _top2_margin(row, k=8):
        """(chosen id, runner-up id, margin, top-k ids, top-k 32-bit values) of one logits row: the probe's unit of evidence."""
        v, i = torch.topk(row.float(), k)
        return (int(i[0]), int(i[1]), round(float(v[0] - v[1]), 5), [int(x) for x in i.tolist()], [round(float(x), 4) for x in v.tolist()])

    def _write_gaps(self, pairs, ctxs=None):
        """The writer rank appends one line per request to <answer file>.gaps.jsonl: the chosen tokens, the runner-ups and the
        margins, in the order the tokens were generated (the first entry is the prefill's decision). Same lock discipline as
        put_batch; never read by the harness."""
        import fcntl
        import json as _json
        if self._cache is None or not getattr(self._cache, 'is_writer', False):
            return
        lines = []
        for j, (key, rec) in enumerate(pairs):
            if key is None or rec is None:
                continue
            ctx = ctxs[j] if ctxs is not None else None
            lines.append(_json.dumps({'k': key, 'top1': [r[0] for r in rec], 'top2': [r[1] for r in rec],
                                      'gap': [r[2] for r in rec], 'topk_id': [r[3] for r in rec], 'topk_val': [r[4] for r in rec],
                                      'ctx': list(ctx) if ctx is not None else None}) + '\n')
        if not lines:
            return
        with open(self._cache.path + '.gaps.jsonl', 'a', encoding='utf-8') as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.write(''.join(lines)); fh.flush()
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def generate_until(self, requests):
        """Batched GREEDY generation via naive re-forward (no KV cache).

        DESIGN (see deepseek-moe-verify/DOWNSTREAM_EVAL_SUPPORT.md, Step 3):
        * Each decode step re-runs the FULL forward on the batch, right-padded to seq_length -- the
          SAME regime scoring uses. This is required, not lazy: the rotary table is sized to
          seq_length, so a shorter forward hits the freqs-vs-query length mismatch that
          `--adaptive_seq_len` was removed to avoid. Cost is ~max_gen forwards of [B, seq_length].
        * GREEDY only (argmax). Eval tasks are do_sample=False; sampling is intentionally unsupported.
        * EP lockstep is automatic: with no_pipeline_parallel + TP=1 every rank gets identical
          logits, so argmax picks the same token on every rank -> the MoE all-to-all stays in sync
          with zero cross-rank broadcast. Do NOT add per-rank randomness here.
        * Reuses the verified right-pad `_model_call`; no new forward code path.
        Correctness-first; a KV-cache path may replace this later, validated against it as an oracle.
        """
        import collections
        res_by_idx = {}
        keys_by_idx = {}

        # Group by gen_kwargs so a batch shares the same stop strings / length cap.
        # Cache hits are answered here and never enter a group; the whole generation is
        # ONE cache value, so a resumed job skips finished generations entirely. Caching
        # is safe despite gen_kwargs possibly carrying do_sample: this adapter is
        # greedy-only (sampling params are ignored below), so the result is deterministic.
        groups = collections.defaultdict(list)
        n_hits = 0
        for i, req in enumerate(requests):
            a = req.args
            ctx = a[0]
            gen_kwargs = a[1] if len(a) > 1 and isinstance(a[1], dict) else {}
            gen_kwargs = with_extra_until(gen_kwargs, self.extra_until)
            if self._cache is not None:
                key, val = self._cache.get('generate_until', (ctx, gen_kwargs))
                keys_by_idx[i] = key
                if val is not MISS:
                    res_by_idx[i] = str(val)
                    n_hits += 1
                    continue
            until = gen_kwargs.get('until', [])
            if isinstance(until, str):
                until = [until]
            if gen_kwargs.get('do_sample', False):
                print_rank_0(">>> [generate_until] WARNING: do_sample requested but this adapter is "
                             "greedy-only; ignoring sampling params (eval expects greedy).")
            max_gen = int(gen_kwargs.get('max_gen_toks', self.max_gen_toks))
            groups[(tuple(until), max_gen)].append((i, ctx))

        total = len(requests)
        n_pending = total - n_hits
        if self._cache is not None:
            print_rank_0(f">>> [response-cache] generate_until: {n_hits} hits / "
                         f"{n_pending} to compute (of {total})")
        seen = 0
        # The data-parallel split: every group runs the same number of rounds; a group short of rounds runs idle ones
        # (main.py calls this method on every group, with its own requests, possibly none).
        n_rounds_here = sum((len(v) + self.batch_size - 1) // self.batch_size for v in groups.values())
        n_rounds = self._world_max(n_rounds_here)
        if self.dp_split:
            assert self._use_kv_decode() and not getattr(get_args(), 'kv_per_row', False), \
                'the data-parallel prompt split runs the batched KV decode path only'
        for (until, max_gen), items in groups.items():
            until = [u for u in until if u]
            # Room to generate within seq_length. This MUST fail loudly rather than
            # clamp: with max_gen == max_length the old max(1, ...) silently sliced
            # every prompt down to its final token, and HumanEval (whose prompts all
            # end in '"""\n') became 164 identical requests for "continue a newline",
            # scoring a hard 0.0 on every checkpoint including the untrained base
            # (root-caused 2026-08-16). Upstream lm-eval asserts here
            # (huggingface.py:1647); match that contract.
            maxctx = self.max_length - max_gen
            if maxctx < 1:
                raise ValueError(
                    f'seq_length ({self.max_length}) must exceed max_gen_toks '
                    f'({max_gen}); with no room left for the prompt every request '
                    f'would be truncated to its last token. Raise the row\'s seqlen '
                    f'or lower its gen_toks in task_table.py.')
            for start in range(0, len(items), self.batch_size):
                chunk = items[start:start + self.batch_size]
                full_ids = [self.tok_encode(c) for _, c in chunk]
                n_cut = sum(1 for t in full_ids if len(t) > maxctx)
                if n_cut:
                    worst = max(len(t) - maxctx for t in full_ids)
                    print_rank_0(f'>>> [truncation] {n_cut}/{len(chunk)} prompts exceed '
                                 f'maxctx={maxctx} (seq_length {self.max_length} - '
                                 f'max_gen_toks {max_gen}); dropping up to {worst} '
                                 f'leading tokens. Scores from this row are truncated.')
                ctx_ids = [t[-maxctx:] for t in full_ids]

                if self._use_kv_decode():
                    # KV path. Batched ragged decode (Phase 5b) is the default: one
                    # [B,1] forward serves the whole chunk per token. --kv-per-row
                    # falls back to the v1 B=1 loop (the batching-free reference).
                    _flush = None
                    if self._cache is not None and mpu.is_pipeline_last_stage():
                        _flush = (lambda b, text: self._cache.put_batch(
                            [(keys_by_idx.get(chunk[b][0]), text)]))
                    _args = get_args()
                    _per_row = getattr(_args, 'kv_per_row', False)
                    if not _per_row and getattr(_args, 'moe_token_dropping', False):
                        # HARD GUARD: with dropping on, a [B,1] step's capacity is
                        # floor-bound (~4) and survivors are picked by a GLOBAL sort
                        # over all B*k assignments -- one row's tokens (even a frozen
                        # row's garbage) can evict another row's expert assignment,
                        # silently changing live logits. Per-row [1,1] can never
                        # drop. The eval template passes --disable-moe-token-dropping
                        # by default; if a config removes it, fall back rather than
                        # produce cross-row-coupled generations.
                        print_rank_0('>>> [kv-b] WARNING: moe_token_dropping is '
                                     'enabled -- batched ragged decode would let '
                                     'rows evict each other\'s expert assignments; '
                                     'falling back to per-row KV decode.')
                        _per_row = True
                    if _per_row:
                        kv_texts = self._kv_generate_rows(ctx_ids, until, max_gen,
                                                          on_row=_flush)
                    else:
                        kv_texts = self._kv_generate_rows_batched(
                            ctx_ids, until, max_gen, on_row=_flush)
                        if getattr(self, '_last_gaps', None) is not None:
                            self._write_gaps([(keys_by_idx.get(i), self._last_gaps[b]) for b, (i, _c) in enumerate(chunk)], ctxs=[list(t) for t in ctx_ids])
                    for b, (i, _c) in enumerate(chunk):
                        res_by_idx[i] = kv_texts[b]
                    if self._cache is not None and mpu.is_pipeline_last_stage():
                        self._cache.put_batch([(keys_by_idx.get(i), res_by_idx[i])
                                               for i, _c in chunk])
                    seen += len(chunk)
                    print_rank_0(f">>> [generate_until] {seen}/{n_pending} computed "
                                 f"(KV decode, batch of {len(chunk)}, max_gen={max_gen})")
                    continue

                seqs = [list(t) for t in ctx_ids]         # content + generated (grows in place)
                gstart = [len(t) for t in ctx_ids]        # where generation begins per row
                done = [False] * len(chunk)

                for _step in range(max_gen):
                    if all(done):
                        break
                    # Right-pad the whole batch to seq_length (pad id 0), forward once.
                    tokens = torch.zeros((len(chunk), self.max_length),
                                         dtype=torch.long, device=self.device)
                    for b, s in enumerate(seqs):
                        tokens[b, :len(s)] = torch.tensor(s, dtype=torch.long, device=self.device)
                    # MUST be no_grad: without it every decode step retains the autograd graph /
                    # backward activations (28 layers x experts) and memory grows until OOM. The
                    # scoring path wraps its forwards the same way. Inference-only, so this is free.
                    with torch.no_grad():
                        logits = self._model_call(tokens)  # [B, seq_length, vocab] (last stage; None elsewhere)
                    # WO-2: the old `if logits is None: break` bailed a PP>1
                    # non-last stage out of the loop WHILE ITS PEERS KEPT
                    # FORWARDING -- exactly the strand this fix exists to end.
                    # Every rank now takes the agreed tokens (non-last stages
                    # contribute None and receive the last stage's choices) and
                    # runs the identical update, so all ranks step and exit
                    # together. A single process with no logits has nothing to
                    # decode and stops for all rows at once.
                    nxt_all = None
                    if logits is not None:
                        nxt_all = [int(logits[b, len(seqs[b]) - 1].argmax(-1).item())
                                   for b in range(len(chunk))]
                    nxt_all = self._agree_next_tokens(nxt_all, len(chunk))
                    if nxt_all is None:
                        break
                    for b in range(len(chunk)):
                        if done[b]:
                            continue
                        nxt = nxt_all[b]
                        if nxt == self.EOT_TOKEN_ID:
                            done[b] = True          # EOT is a STOP signal, not output: don't append
                            continue                # (else its special-token string leaks into the answer)
                        seqs[b].append(nxt)
                        # stop at the earliest 'until' string, checked on the decoded continuation
                        # (a stop string can straddle a token boundary, so decode-then-search).
                        if until:
                            text = self.tok_decode(seqs[b][gstart[b]:])
                            if any(u in text for u in until):
                                done[b] = True
                    # WO-2: agreed done mask -> every rank leaves this loop, and
                    # reaches the next chunk, on the same step.
                    done = self._agree_done_flags(done)

                for b, (i, _) in enumerate(chunk):
                    text = self.tok_decode(seqs[b][gstart[b]:])
                    for u in until:
                        j = text.find(u)
                        if j != -1:
                            text = text[:j]
                    res_by_idx[i] = text
                # Flush this batch of finished generations before starting the next --
                # per-batch flushing is the resumability guarantee. Last-stage guard as in
                # the scoring path (non-last PP stages would emit garbage text).
                if self._cache is not None and mpu.is_pipeline_last_stage():
                    self._cache.put_batch([(keys_by_idx.get(i), res_by_idx[i])
                                           for i, _c in chunk])
                seen += len(chunk)
                print_rank_0(f">>> [generate_until] {seen}/{n_pending} computed "
                             f"(batch of {len(chunk)}, max_gen={max_gen})")

        for _ in range(n_rounds - n_rounds_here):
            self._idle_generation_round(getattr(self, '_kv_causal_b', None))

        out = [res_by_idx[i] for i in range(total)]
        # WO-1 (same treatment as scoring): under PP the non-last stages assembled
        # their texts without logits. Sync the final list from the writer rank so
        # every rank -- including the results writer on stage 0 -- returns the
        # last stage's real generations. No-op at PP=1 (bisection guarantee).
        if self.is_pipe_parallel:
            import torch.distributed as dist
            from .cache import writer_rank
            out = _broadcast_scores_from_last_stage(
                out, bool(mpu.is_pipeline_last_stage()), dist.get_world_size(),
                mpu.get_pipeline_model_parallel_world_size(),
                writer_rank, dist.broadcast_object_list)
        return out

    # -- Internal: shared scoring loop, copied from evaluate.py --------
    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        disable_tqdm = disable_tqdm if self.is_main else True
        res = []
        self.model.eval()

        # Long-context scoring (2026-09-14): a request whose tokens exceed XMOE_SCORE_KV_FROM is scored through the chunked
        # prefill of the KV cache (one row at a time, _score_long) instead of the one-piece padded forward below, whose MoE
        # dispatch buffers do not fit a GCD at 128k and beyond. The long rows are taken out here and scored after the batched
        # loop; the results are re-merged in the original order.
        import os as _os_env
        long_from = int(_os_env.getenv('XMOE_SCORE_KV_FROM', '32768'))
        # the length that is actually forwarded (the batched path truncates to max_length + 1 tokens), so a window shorter
        # than the threshold never routes a row here whatever the document's raw length
        long_idx = [i for i, (_, c, k) in enumerate(requests)
                    if min(len(c) + len(k), self.max_length + 1) > long_from and self._use_kv_decode()]
        long_set = set(long_idx)
        all_requests = requests
        requests = [r for i, r in enumerate(requests) if i not in long_set]

        with torch.no_grad():
            def _collate(x):
                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            # Reorder so longest-first → better padding efficiency.
            # 0.4+ moved Reorderer; tolerate both locations / a manual fallback.
            reord = self._make_reorderer(requests, _collate)
            iterable = reord['iter']
            # The data-parallel split: one forward per chunk on every group; the groups agree on the count and a group
            # short of chunks runs dummy forwards after its own (below).
            n_here = (len(requests) + self.batch_size - 1) // self.batch_size
            n_rounds = self._world_max(n_here)

            for chunk in self._chunks(tqdm(iterable, disable=disable_tqdm),
                                      self.batch_size):
                # 1) Materialize unpadded content tensors and per-row metadata.
                real_tokens_list = []
                contlens = []
                padding_length = None
                for _, context_enc, continuation_enc in chunk:
                    real = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1):][:-1],
                        dtype=torch.long,
                    ).to(self.device)
                    real_tokens_list.append(real)
                    contlens.append(continuation_enc)
                    if padding_length is None:
                        padding_length = int(real.shape[0])
                    else:
                        padding_length = max(padding_length, int(real.shape[0]))

                # All eval inputs pad to seq_length (= args.seq_length). The
                # `adaptive_seq_len` knob is forced off — Megatron's rotary
                # apply doesn't slice freqs to variable input length, so
                # everything runs at the training-time seq_len.
                padding_length = self.max_length

                # 2) Build padded batch. Two paths, selected by --padding-side:
                #
                # LEFT-PAD (new opt-in path, requires --use-left-pad-eval):
                #   [PAD, PAD, ..., PAD,  c_0, c_1, ..., k_0, ..., k_{|cont|-2}]
                #   position_ids = [0, 0, ..., 0, 0, 1, 2, ..., inplen-1]
                #   attention_mask masks padding KEY columns; flows to FlashSelfAttention
                #   which does varlen strip/restitch via flash_attn varlen.
                #
                # RIGHT-PAD (default; matches evaluate.py):
                #   [c_0, c_1, ..., k_{|cont|-2}, PAD, PAD, ..., PAD]
                #   position_ids = [0, 1, ..., inplen-1, inplen, ..., S-1]
                #   attention_mask = None — Megatron derives standard causal mask
                #   internally; Flash applies causal=True over full S which is
                #   correct because future positions (the pads) are masked by
                #   causality and content at [0..N-1] gets training-correct
                #   rotary phases.
                if self._padding_side == 'left':
                    tokens, position_ids, attention_mask, inplens = \
                        self._build_padded_batch(real_tokens_list, padding_length,
                                                 side='left')
                    S = tokens.shape[1]
                    slices = [(S - len(ct), S) for ct in contlens]
                    logits = self._model_call(
                        tokens,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        content_slices=slices,
                    )
                else:
                    tokens, position_ids, attention_mask, inplens = \
                        self._build_padded_batch(real_tokens_list, padding_length,
                                                 side='right')
                    # Right-pad: let Megatron derive its own standard mask &
                    # position_ids inside the chunked _model_call path.
                    # Right-pad path: gather only the scored windows (see
                    # _model_call.content_slices). Same math as the old full
                    # softmax-then-slice -- softmax is per-position -- with the
                    # [S, vocab] materialization gone.
                    slices = [(inplen - len(ct), inplen)
                              for inplen, ct in zip(inplens, contlens)]
                    logits = self._model_call(tokens, content_slices=slices)

                if logits is not None:
                    # logits is PACKED [sum(contlen), vocab]; split per row.
                    row_lens = [len(ct) for ct in contlens]
                    rows = torch.split(logits, row_lens, dim=0)
                    chunk_put = []
                    for (cache_key, _, _), row, inplen, cont_toks in zip(
                            chunk, rows, inplens, contlens):
                        contlen = len(cont_toks)
                        logits_i = F.log_softmax(row, dim=-1).cpu().unsqueeze(0)
                        greedy_tokens = logits_i.argmax(dim=-1)
                        cont_toks_t = torch.tensor(cont_toks,
                                                   dtype=torch.long).unsqueeze(0)
                        max_equal = (greedy_tokens == cont_toks_t).all()
                        logits_i = torch.gather(
                            logits_i, 2, cont_toks_t.unsqueeze(-1)
                        ).squeeze(-1)
                        ans = (float(logits_i.sum()), bool(max_equal))
                        res.append(ans)
                        if cache_key is not None:
                            chunk_put.append((cache_key, [ans[0], ans[1]]))
                    # Flush THIS batch before computing the next one -- per-batch flushing
                    # is the resumability guarantee. `logits is not None` already implies
                    # the last PP stage; the mpu check is cheap insurance against the
                    # dummy-results path below ever feeding a put.
                    if (self._cache is not None and chunk_put
                            and mpu.is_pipeline_last_stage()):
                        self._cache.put_batch(chunk_put)
            for _ in range(n_rounds - n_here):
                self._idle_scoring_round()

            # the long rows: the count agreed over the job (the data-parallel split), each row's chunks agreed inside _score_long
            n_long = self._world_max(len(long_idx))
            long_res = []
            if long_idx:
                print_rank_0(f'>>> [score-long] {len(long_idx)} request(s) longer than {long_from} tokens scored through the '
                             f'chunked prefill (XMOE_KV_PREFILL_CHUNK={_os_env.getenv("XMOE_KV_PREFILL_CHUNK", "8192")})')
            for j in range(n_long):
                if j < len(long_idx):
                    cache_key, context_enc, continuation_enc = all_requests[long_idx[j]]
                    ans = self._score_long(context_enc, continuation_enc)
                    long_res.append(ans)
                    if (self._cache is not None and cache_key is not None and mpu.is_pipeline_last_stage()):
                        self._cache.put_batch([(cache_key, [ans[0], ans[1]])])
                else:
                    for _ in range(self._world_max(0)):
                        self._empty_forward(getattr(self, '_kv_causal_b', None))

        # WO-1: at PP>1 only the last stage computed real scores. Broadcast them
        # from the writer rank (first rank of the last stage, cache.writer_rank())
        # so EVERY rank -- including world rank 0 on stage 0, the sole writer of
        # results.json and the registry rows (main.py:287-290) -- returns real
        # values. The np.random.rand() dummy path that stood here is deleted
        # outright, never reachable again: a cold-cache pp>1 scoring run used to
        # RECORD RANDOM NUMBERS (directive 3.1). The undo permutation below is
        # identical on all ranks (same requests, same deterministic sort), so
        # broadcasting the reordered list is sound. No-op at PP=1, where the
        # dummy path was dead anyway (every rank is the last stage) -- the PP=1
        # scoring bytes this fingerprint gate re-verifies are untouched.
        if self.is_pipe_parallel:
            import torch.distributed as dist
            from .cache import writer_rank
            res = _broadcast_scores_from_last_stage(
                res, bool(mpu.is_pipeline_last_stage()), dist.get_world_size(),
                mpu.get_pipeline_model_parallel_world_size(),
                writer_rank, dist.broadcast_object_list)

        res = reord['undo'](res)
        if long_idx:
            merged, it_short, it_long = [], iter(res), iter(long_res)
            for i in range(len(all_requests)):
                merged.append(next(it_long) if i in long_set else next(it_short))
            res = merged
        return res

    def _score_long(self, context_enc, continuation_enc):
        """One (context, continuation) scored through the chunked prefill: the input is (context + continuation)[-(max+1):][:-1]
        as in the batched path; the chunks write the KV cache in place; the continuation's logits are the last len(continuation)
        rows of the sequence, taken from the last chunk(s). Returns (sum log p, is_greedy). The prefill of an identical input
        (a multiple-choice document's other choices) is not repeated: its last logits are kept."""
        from megatron.text_generation.forward_step import InferenceParams
        import os as _os_env
        real = list((context_enc + continuation_enc)[-(self.max_length + 1):][:-1])
        cont = list(continuation_enc)
        L = len(real)
        chunk = int(_os_env.getenv('XMOE_KV_PREFILL_CHUNK', '8192'))
        n_chunks = (L + chunk - 1) // chunk
        key = tuple(real)
        hit = getattr(self, '_score_long_last', None)
        if hit is not None and hit[0] == key:
            n_max = self._world_max(0)                       # this group's forwards this row: none (a memo hit)
            for _ in range(n_max):
                self._empty_forward(getattr(self, '_kv_causal_b', None))
            tail = hit[1]
        else:
            n_max = self._world_max(n_chunks)
            if getattr(self, '_kv_ip_long', None) is None or self._kv_ip_long.max_sequence_len != self.max_length:
                self._kv_ip_long = InferenceParams(1, self.max_length)
            ip = self._kv_ip_long
            ip.key_value_memory_dict.clear()                 # a fresh row: the layers re-allocate (zeros) on first use
            ip.row_lengths = None
            ip.kv_extent = None
            toks = torch.tensor(real, dtype=torch.long, device=self.device)[None]
            pos = torch.arange(L, device=self.device)[None]
            causal = getattr(self, '_kv_causal_b', None)
            tail_rows = []                                   # the logits of the last chunks, newest last
            need = len(cont)
            with torch.no_grad():
                for k in range(0, L, chunk):
                    e = min(k + chunk, L)
                    ip.sequence_len_offset = k
                    out = self.model(toks[:, k:e], pos[:, k:e], None if causal is None else causal[:, :, k:e, :e],
                                     inference_params=ip)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                    if L - e < need:                         # this chunk holds scored positions
                        logits = gather_from_tensor_model_parallel_region(logits)[..., :self.vocab_size]
                        tail_rows.append(logits[0].float())
                for _ in range(n_max - n_chunks):
                    self._empty_forward(causal)
                tail = torch.cat(tail_rows, 0)[-need:] if tail_rows else None
            ip.key_value_memory_dict.clear()
            self._score_long_last = (key, tail)
        logp = F.log_softmax(tail, dim=-1).cpu()
        cont_t = torch.tensor(cont, dtype=torch.long)
        greedy = bool((logp.argmax(-1) == cont_t).all())
        return (float(torch.gather(logp, 1, cont_t[:, None]).sum()), greedy)

    @staticmethod
    def _make_reorderer(items, key):
        """Return {'iter': sorted-iterable, 'undo': fn[list]->list}.

        Tries Reorderer from a few possible locations (the class moved between
        lm-eval versions), falling back to a manual sort+invert if absent.
        """
        Reorderer = getattr(lm_utils, 'Reorderer', None)
        if Reorderer is None:
            try:
                from lm_eval.utils import Reorderer  # noqa: F401
            except Exception:
                Reorderer = None
        if Reorderer is not None:
            r = Reorderer(items, key)
            return {'iter': r.get_reordered(),
                    'undo': lambda res: r.get_original(res)}
        # Manual fallback
        indexed = list(enumerate(items))
        indexed.sort(key=lambda p: key(p[1]))
        order = [i for i, _ in indexed]
        sorted_items = [x for _, x in indexed]
        inv = [0] * len(order)
        for new_i, old_i in enumerate(order):
            inv[old_i] = new_i
        return {'iter': sorted_items,
                'undo': lambda res: [res[inv[i]] for i in range(len(res))]}

    @staticmethod
    def _chunks(iterable, n):
        out = []
        for x in iterable:
            out.append(x)
            if len(out) == n:
                yield out
                out = []
        if out:
            yield out

    def _build_padded_batch(self, real_tokens_list, padding_length, side='left'):
        """Build a padded batch in either left-pad or right-pad layout.

        LEFT (opt-in, requires --use-left-pad-eval AND Megatron-side rotary +
        FlashSelfAttention patches): zeros prefix each row, content sits at
        the right.

            [PAD, ..., PAD, c_0, c_1, ..., c_{N-1}]
            position_ids = [0, ..., 0, 0, 1, ..., N-1]
            attention_mask: [B, 1, S, S] bool, True=mask, with the leading
              S-N key columns blocked AND the upper-triangular causal pattern.

        RIGHT (default; matches the proven evaluate.py): content at the left,
        zeros at the right. Standard Megatron mask + sequential position_ids
        cover correctness because future positions (the pads) are causal-masked
        anyway and content occupies training-identical absolute slots.

            [c_0, c_1, ..., c_{N-1}, PAD, ..., PAD]
            position_ids = sequential [0..S-1] (Megatron derives this itself)
            attention_mask = None (Megatron's standard causal applies)

        Returns:
            tokens         : [B, S] long
            position_ids   : [B, S] long or None (right-pad: None)
            attention_mask : [B, 1, S, S] bool, True=mask, or None (right-pad)
            inplens        : list[int] of unpadded content lengths
        """
        B = len(real_tokens_list)
        S = padding_length
        device = real_tokens_list[0].device

        tokens = torch.zeros((B, S), dtype=torch.long, device=device)
        inplens = []

        if side == 'right':
            # Content at [0..N-1], zeros at [N..S-1]. No custom mask /
            # position_ids — Megatron derives the right thing internally and
            # Flash applies causal=True over the full S, which is correct
            # because pads are "future" tokens that causal masking blocks.
            for b, t in enumerate(real_tokens_list):
                n = int(t.shape[0])
                inplens.append(n)
                tokens[b, :n] = t
            return tokens, None, None, inplens

        # side == 'left'
        position_ids = torch.zeros((B, S), dtype=torch.long, device=device)
        causal = torch.triu(torch.ones((S, S), dtype=torch.bool, device=device),
                            diagonal=1)
        attention_mask = causal.unsqueeze(0).expand(B, -1, -1).clone()
        for b, t in enumerate(real_tokens_list):
            n = int(t.shape[0])
            inplens.append(n)
            start = S - n
            tokens[b, start:] = t
            # Content tokens get position_ids 0..n-1 (matches training); the
            # leading slots stay 0 — masked out anyway.
            position_ids[b, start:] = torch.arange(n, device=device)
            # Mask leading padding KEY columns for all queries on this row.
            attention_mask[b, :, :start] = True
        attention_mask = attention_mask.unsqueeze(1)  # [B, 1, S, S]
        return tokens, position_ids, attention_mask, inplens

    # Back-compat alias kept so any external caller of the old name still works.
    def _build_left_pad_batch(self, real_tokens_list, padding_length):
        return self._build_padded_batch(real_tokens_list, padding_length, side='left')

    # -- Distributed forward pass --------------------------------------
    def create_model_inputs(self, tokens):
        args = get_args()
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens, self.EOT_TOKEN_ID,
            args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss,
        )
        return (tokens, position_ids, attention_mask), (tokens, loss_mask)

    def _model_call(self, inps, position_ids=None, attention_mask=None,
                    content_slices=None):
        """Run the distributed forward pass.

        If `position_ids` and `attention_mask` are provided (the left-padding
        path from _loglikelihood_tokens), they're passed through directly to
        the model — no per-micro-batch re-derivation via
        get_ltor_masks_and_position_ids. This is required for left-padding to
        work correctly: the caller built position_ids that put 0 at the first
        content token (so rotary phases match training) and an attention_mask
        that blocks padding KEYS (so content queries don't attend to pads).

        If those kwargs are None, fall back to the original behavior: derive
        causal mask + sequential position_ids per micro-batch chunk.
        """
        args = get_args()

        if args.no_pipeline_parallel:
            # When caller supplied left-padding masks, use them directly
            # without the inner pad-to-micro-batch-multiple chunking
            # (that chunking would invalidate the per-row attention_mask
            # alignment). One forward on the whole batch.
            if position_ids is not None and attention_mask is not None:
                self.model.micro_batches = 1
                a_output, *_ = self.model(inps, position_ids, attention_mask)
                output = a_output
                if not mpu.is_pipeline_last_stage():
                    return None
                if content_slices is not None:   # same packed contract as the tail
                    packed = torch.cat(
                        [output[i, s0:s1] for i, (s0, s1) in enumerate(content_slices)], 0)
                    return gather_from_tensor_model_parallel_region(packed)[..., :self.vocab_size]
                return gather_from_tensor_model_parallel_region(output)[..., :self.vocab_size]

            new_size = ((len(inps) + args.micro_batch_size - 1)
                        // args.micro_batch_size) * args.micro_batch_size
            padded = F.pad(inps, (0, 0, 0, new_size - len(inps)), value=0)
            data_iterator = list(
                torch.stack(inp) for inp in self._chunks(padded, args.micro_batch_size)
            )
            self.model.micro_batches = len(data_iterator)

            output = []
            use_flash = bool(getattr(args, 'use_flash_attn_v2', False)
                             or getattr(args, 'use_flash_attn_v1', False))
            for tokens in data_iterator:
                if use_flash:
                    # Flash attention never reads the dense causal mask (it runs
                    # causal=True internally; the mask slot is only consumed by the
                    # non-flash fallback). get_ltor materializes it as float ones
                    # [1,1,S,S] on the tokens' device -- 68.7 GB at S=131072, built
                    # on HOST here because the batch is pre-engine -- which kernel-
                    # OOM-killed every 128k job (5383151, 5383410: memwatch shows
                    # RSS 2.3->58 GB in 10 s with GPU flat). With reset flags off,
                    # get_ltor's position_ids are a plain arange -- built here
                    # directly, mask omitted.
                    position_ids_d = torch.arange(
                        tokens.shape[1], dtype=torch.long,
                        device=tokens.device).unsqueeze(0).expand_as(tokens)
                    attention_mask_d = None
                else:
                    attention_mask_d, _, position_ids_d = get_ltor_masks_and_position_ids(
                        tokens, self.EOT_TOKEN_ID,
                        args.reset_position_ids, args.reset_attention_mask,
                        args.eod_mask_loss,
                    )
                a_output, *_ = self.model(
                    tokens, position_ids_d, attention_mask_d,
                )
                output.append(a_output)
            output = torch.cat(output, 0)[:len(inps)] if output else None
        else:
            self.model.set_batch_fn(self.create_model_inputs)
            new_size = ((len(inps) + args.micro_batch_size - 1)
                        // args.micro_batch_size) * args.micro_batch_size
            padded = F.pad(inps, (0, 0, 0, new_size - len(inps)), value=0)
            data_iterator = list(
                torch.stack(inp) for inp in self._chunks(padded, args.micro_batch_size)
            )
            self.model.micro_batches = len(data_iterator)
            output = self.model.eval_batch(
                iter(data_iterator),
                compute_loss=False,
                reduce_output=None,
            )
            # eval_batch returns one output per micro-batch in GPTModelPipe's
            # seq-major [s, b, v] layout (only the non-pipe GPTModel transposes
            # its output to [b, s, v], megatron/model/gpt_model.py:431; the
            # pipe path instead transposes labels to match [s, b, v],
            # gpt_model.py:771). Transpose each micro-batch to [b, s, v] first
            # so the cat runs along the true batch dim and [:len(inps)] strips
            # exactly the pad rows.
            output = (torch.cat([o.transpose(0, 1) for o in output], 0)[:len(inps)]
                      if output is not None else None)
            if args.adaptive_seq_len:
                self.model.total_loss = None

        if mpu.is_pipeline_last_stage():
            # content_slices (loglik scoring only): gather ONLY the scored rows.
            # The full [b, S, vocab] gather is seq x vocab x 2B PER RANK -- 12.25 GiB
            # at 64k / 24.5 GiB at 128k -- and OOM'd every 64k+ scoring job in every
            # geometry tried (5382813-18 TP8, 5382869 TP8+expandable, 5382995 TP1 at
            # the softmax of the same tensor). Scored continuations are ~10-30
            # positions/row: packing them first shrinks the gather ~2000x. Exact:
            # gather is elementwise along vocab; slicing positions commutes with it.
            if content_slices is not None:
                packed = torch.cat(
                    [output[i, s0:s1] for i, (s0, s1) in enumerate(content_slices)], 0)
                return gather_from_tensor_model_parallel_region(packed)[..., :self.vocab_size]
            return gather_from_tensor_model_parallel_region(output)[..., :self.vocab_size]
        return None


# Pre-refactor name, kept for any external caller.
XMoELM = MegatronDeepSpeedLM


def _merge_done_flags(per_rank_flags):
    """WO-2 pure spec of _agree_done_flags' distributed MAX-reduce.

    per_rank_flags: one list[bool] of equal length per rank. Returns the merged
    mask every rank must proceed with: elementwise OR -- a row is done
    everywhere as soon as it is done anywhere. OR (not AND) is the safe
    direction: freezing a row a peer still thought live only stops stepping it
    (its text was already finalized from agreed tokens), while AND would keep
    dead rows stepping and let per-rank texts diverge again. Kept as a module
    function so the login-node test pins these semantics without torch.
    """
    merged = [any(col) for col in zip(*per_rank_flags)]
    assert all(len(f) == len(merged) for f in per_rank_flags), \
        'ranks disagree on batch size -- lockstep already lost upstream'
    return merged
