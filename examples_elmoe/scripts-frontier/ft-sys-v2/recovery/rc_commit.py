"""rc_commit.py (ft-sys-v2 recovery, milestone 2, 2026-11-06): commits inside the step. Framework-free: it sees the optimizer
adapter's per-group partial gradient sums only. Tested on GPUs (X-MoE 10B, ZeRO-2): E5 5862320, E5b 5862517, E5c 5862600,
E5d 5862827, E6 5863343 (RESULTS_M1_M2.md, M2-Z, M2-C, M2-A/B, M2-K).

The question of milestone 2 (the user, 11-06): resume from the last committed micro-batch instead of the step start; the reduced
partial sums are copied to the partner every k micro-batches (FTV2_COMMIT_K, default 1); the copy is asynchronous so that it hides
behind the next micro-batch's compute.

Where the partial sums are: ZeRO stage 2 reduces every micro-batch (DeepSpeed's own stage-2 path: each owner keeps its partition's
reduced sum in `averaged_gradients`, stage_1_and_2.py:767-811), so milestone 2 runs at ZeRO stage 2 (the EP_CMD's last field).
The hook (adapters/deepspeed_zero.py wrap_epilogue) runs after ZeRO's per-micro-batch epilogue:
  - if a stop began while the epilogue ran, the piece just added may come from a revoked reduction: the sums go back to the last
    commit and the main thread leaves;
  - at a commit (micro-batches done a multiple of k, and not the last one: the step's own Keep covers the boundary), the sums are
    copied flat (the commit's snapshot S_c, kept with the one before, S_{c-k}), sent to the partner (rank + stride), followed by a
    token [step, c] in a second round on the same communicator (a token that arrived means its sums arrived); the owner's S and token
    are received into the copy buffers (current and previous).
Positions after a stop: own = my last commit c (micro-batches), copy = my owner's commit from its last arrived token. Every rank
computes the same resume point J = the lowest own position; a survivor at c = J + k goes back to S_{c-k}; a lost owner's copy must
hold J (current or previous); otherwise the window (exit 75). J = 0 is the step start (milestone 1).
Redo: the iterator rebuilt at the step's start skips J micro-batches; Megatron's schedule starts at J
(megatron/core/pipeline_parallel/schedules.py, FTV2_START_MICROBATCH); the new engine's sums are S_J; every pair re-seeds its copy.

Two transports for the copy (FTV2_COMMIT_MODE; the user's comparison, the lead 11-06 05:4x):
  p2p   (default) the owner sends its sums to the partner after each commit (above). With ZeRO-2's all-reduce of every
        micro-batch this adds the partition (bf16) to the inter-node traffic of every micro-batch: E5 5862320, +7.3% at k = 1.
  keep  "all-reduce + keep": with use_multi_rank_bucket_allreduce, ZeRO-2 already all-reduces every bucket over the
        data-parallel group, so the partner receives its owner's reduced slices and drops them. The adapter's
        wrap_reduce_keep copies them into `piece` (the owner's partition layout) instead; after each micro-batch's epilogue the
        partner adds the piece to its copy, as the owner adds it to its sums (the same bf16 additions, in the same order). No
        network; k = 1 only (every micro-batch is folded in). After a rebuild the copy is seeded once by p2p, as in p2p mode.

Overlap (FTV2_COMMIT_OVERLAP=1, under the pipeline only; 2026-11-12, the user's verdict on V1 of the overnight campaign: "please do
it asynchronously"). Without it a commit's reduction runs on the compute stream between the stage's backward pass and its gradient
send, and the stages' reductions follow one another (63B PP 4: 0.45 s per commit, the step 186.6 -> 415.6 s at k = 1). With it the
adapter enqueues the reduction, ZeRO's epilogue and this commit's work on self.side, after an event recorded on the compute stream at
the backward's end; the host goes on to the gradient send and the next forward at once. What changes here:
  - the work is written into spare buffers and kept in self.pend (keep: the piece itself holds the copy in flight and the snapshot
    is taken at promote(), into the snapshot that drops out, so the buffers are the synchronous commit's; p2p: copy_spare and a
    fresh snapshot, which is also what is sent); the token goes to tok_spare;
    the rank's position (c, snapshots, copies, tokens) moves only in promote(), when the host sees the work's end event complete
    and the stop flag still clear and no reduction communicator in error: the check FTV2_COMMIT_VERIFY made by waiting on the
    stream after every commit, made when the commit's end is seen instead. Until then the last two promoted commits stay intact,
    so decide_microbatch's rules (J or J + k; the copy at J or J - k) hold unchanged;
  - one commit in flight: the adapter promotes the previous one before it enqueues the next reduction (the host waits there only
    when the side stream is a whole commit interval behind); that also bounds the memory to one reduction's gradients;
  - settle() before any use of the sums outside the side stream (a stop, the recovery's report, restore, target, copy, seed): a
    commit still in flight is dropped (conservative: it may hold a cut reduction) and the current stream waits for the side one.
Not built for level C (its cut checks wait on the stream at every commit): the controller keeps the reductions on the compute
stream there."""
import os
import time

import torch
import torch.distributed as dist


class Commit:
    def __init__(self, adapter, rank, world, device, log, group=None, spaced=False, overlap=False):
        self.ad, self.rank, self.world, self.device, self.log = adapter, rank, world, device, log
        self.spaced = spaced                                 # milestone 3: one reduction (and one epilogue) every k micro-batches
        self.stride = int(os.environ.get('FTV2_PARTNER_STRIDE', '8'))
        import rc_pairs                                      # under a pipeline: the same stage's next replica (milestone 3)
        self.partner, self.owner = rc_pairs.partner(rank, world, self.stride), rc_pairs.owner(rank, world, self.stride)
        self.k = int(os.environ.get('FTV2_COMMIT_K', '1'))
        self.asyn = os.environ.get('FTV2_COMMIT_ASYNC', '1') == '1'
        self.pg = group
        self.cuda = device.type == 'cuda'
        # overlap: the side stream at high priority (FTV2_COMMIT_SIDE_PRIORITY=high, the default; 'normal' for the test's control).
        # torch.cuda.Stream() hands out one of 32 shared pool streams, and ROCm maps streams onto GPU_MAX_HW_QUEUES hardware queues
        # (default 4) per priority: a normal-priority side stream can share a hardware queue with the compute stream, and the GPU
        # then runs the two one after the other (tests/gpu_commit_overlap_test.py, T1c 5911701: 0.40 of the side work concurrent
        # with 8 queues, 0.00 with 1). High-priority streams take queues of their own; the synchronous collectives ZeRO issues
        # (async_op=False) run on the current stream, so the reduction's RCCL kernels run there too.
        self.side_prio = -1 if (overlap and spaced and self.cuda and
                                os.environ.get('FTV2_COMMIT_SIDE_PRIORITY', 'high') == 'high') else 0
        self.side = torch.cuda.Stream(device=device, priority=self.side_prio) if self.cuda else None
        self.groups = adapter.zero2_groups()
        self.step, self.added, self.c = None, 0, 0           # step; micro-batches added; last commit (micro-batches)
        self.snap_cur, self.snap_prev = None, None           # my S_c and S_{c-k} (flat per group), None = zeros
        self.copy_cur = [torch.zeros(adapter.zero2_partition_size(i), dtype=adapter.zero2_dtype(), device=device)
                         for i in self.groups]
        self.copy_prev = [torch.zeros_like(t) for t in self.copy_cur]
        self.tok_cur = torch.full((2,), -1.0, dtype=torch.float64, device=device)
        self.tok_prev = torch.full((2,), -1.0, dtype=torch.float64, device=device)
        self.tok_send = torch.zeros(2, dtype=torch.float64, device=device)
        # host mirrors of the two tokens (None = only the GPU knows: a token in flight from the owner, p2p mode). report() and
        # copy_of() read them without a device wait: under the pipeline a healthy reduction can keep the GPU busy while the
        # monitor thread reports (M3C1b 5868846: rank 16's report hung in tok_cur.tolist())
        self.tok_cur_h, self.tok_prev_h = [-1, -1], [-1, -1]
        self.tok_pin = [torch.empty(2, dtype=torch.float64, pin_memory=True) for _ in range(2)] if self.cuda else None
        self.done_ev, self.inflight = None, None
        self.ms = []                                         # host time of each commit's launch
        # overlap (FTV2_COMMIT_OVERLAP=1, the controller decides): the commit in flight (self.pend) writes into the spare buffers
        self.overlap = bool(overlap and spaced and self.cuda)
        self.pend, self.side_t0 = None, None
        self.ov_ms = []                                      # per promoted commit: (reduction ms, commit work ms) on the side stream
        self.copy_spare = None
        self.snap_pool = []                                  # overlap, keep: snapshot buffers free for the next promote (at most 2)
        self.snap_owned = []                                 # overlap, keep: the snapshot buffers _snap_take made (only these are reused)
        if self.overlap:
            self.tok_spare = torch.full((2,), -1.0, dtype=torch.float64, device=device)
            self.tok_stream = torch.cuda.Stream(device=device, priority=self.side_prio)   # token reads never behind a reduction
            self.tok_pin_p = torch.empty(2, dtype=torch.float64, pin_memory=True)
        # the transport (FTV2_COMMIT_MODE): under the pipeline (spaced reductions) the default is "keep" (all-reduce + keep; the
        # user, 2026-11-06: "Alright for 2", m3_map question 4), when ZeRO's all-reduce path is configured; otherwise p2p
        # (milestone 2's default, and the only transport on the reduce path)
        allreduce_path = bool(getattr(getattr(adapter, 'zero', None), 'use_multi_rank_bucket_allreduce', False))
        self.mode = os.environ.get('FTV2_COMMIT_MODE') or ('keep' if spaced and allreduce_path else 'p2p')
        # diagnosis (FTV2_COMMIT_DEBUG=1): the norm of every group's sums at each epilogue of the steps in
        # [FTV2_COMMIT_DEBUG_FROM, FTV2_COMMIT_DEBUG_TO], one line per rank (a device sync each: diagnosis runs only)
        self.debug = os.environ.get('FTV2_COMMIT_DEBUG') == '1'
        self.debug_from = int(os.environ.get('FTV2_COMMIT_DEBUG_FROM', '0'))
        self.debug_to = int(os.environ.get('FTV2_COMMIT_DEBUG_TO', '1000000'))
        self.piece = None
        if self.mode == 'keep':
            if self.k != 1 and not self.spaced:
                raise ValueError('FTV2_COMMIT_MODE=keep folds in every micro-batch: FTV2_COMMIT_K must be 1')
            self.piece = [torch.zeros_like(t) for t in self.copy_cur]
            adapter.wrap_reduce_keep(self.owner, self.piece)  # the hook reads self.piece[i] at each call: rotate in place only
        # overlap memory (63B, 2026-11-12: keep_k2_ovnt 5912999 and keep_k4_ovnt 5912588 ran stage 0 out of HBM in step 2):
        # keep mode folds the commit in flight into the piece itself and takes the snapshot at promote(), so it holds the same
        # buffers as a synchronous keep commit (two copies, the piece, two snapshots); p2p mode receives into a spare copy.
        if self.overlap and self.mode != 'keep':
            self.copy_spare = [torch.zeros_like(t) for t in self.copy_cur]

    def _snap_release(self):
        """overlap, keep: the two snapshots' buffers go to the pool (reused by the next promotes) instead of the allocator."""
        if self.overlap and self.mode == 'keep':
            for s in (self.snap_cur, self.snap_prev):
                if s is not None and any(s is o for o in self.snap_owned) and len(self.snap_pool) < 2:
                    self.snap_pool.append(s)
        self.snap_cur, self.snap_prev = None, None

    def _snap_take(self):
        """overlap, keep, in promote(): my sums at the commit just promoted, copied on the side stream (after the reduction that
        made them, before the next one) into the buffer of the snapshot that drops out (snap_prev), or one from the pool. Only
        buffers made here are ever written again (snap_owned): a snapshot handed in by set_position is never reused."""
        buf = None
        if self.snap_prev is not None and any(self.snap_prev is o for o in self.snap_owned):
            buf = self.snap_prev
        elif self.snap_pool:
            buf = self.snap_pool.pop()
        with torch.cuda.stream(self.side):
            out = [self.ad.zero2_flat_partial_into(i, None if buf is None else buf[n]) for n, i in enumerate(self.groups)]
        if buf is None or any(a is not b for a, b in zip(out, buf)):
            self.snap_owned = [o for o in self.snap_owned if o is not buf][-3:] + [out]
            return out
        return buf

    def begin_step(self, step):
        """The top of an iteration: no micro-batch of this step is summed or committed yet."""
        self.settle()                                        # overlap: the buffers below are zeroed on this stream
        self._snap_release()
        self.step, self.added, self.c, self.snap_cur, self.snap_prev = int(step), 0, 0, None, None
        if self.mode == 'keep':                              # the owner's sums at micro-batch 0 of this step are zeros
            for t in self.copy_cur + self.copy_prev + self.piece:
                t.zero_()
            self.tok_cur[0].fill_(float(step)); self.tok_cur[1].fill_(0.0)
            self.tok_prev.fill_(-1.0)
            self.tok_cur_h, self.tok_prev_h = [int(step), 0], [-1, -1]

    def set_position(self, step, c, snap):
        """After a restore at (step, c): the sums are S_c."""
        self.step, self.added, self.c, self.snap_cur, self.snap_prev = int(step), int(c), int(c), snap, None

    # ------------------------------------------------------------------ after ZeRO's per-micro-batch epilogue
    def debug_norms(self, what, flats=None):
        """FTV2_COMMIT_DEBUG=1: log the norm of each group's sums (the current sums, or `flats`)."""
        if not self.debug or self.step is None or not (self.debug_from <= self.step <= self.debug_to):
            return
        fl = flats if flats is not None else [self.ad.zero2_flat_partial(i) for i in self.groups]
        ns = [round(float(f.float().norm()), 6) if f is not None else 0.0 for f in fl]
        self.log(f'commit debug step {self.step} {what}: group norms {ns}')

    def after_epilogue(self, stopping, last, pos=None):
        """After ZeRO's epilogue. pos (milestone 3, spaced reductions) = the micro-batches of this step summed after it; without
        it every epilogue is one micro-batch (ZeRO-2)."""
        if not stopping:
            self.debug_norms(f'after micro-batch {self.added + 1}{" (last)" if last else ""}')
        if stopping:
            self.restore_own(self.c)                         # drop whatever came after the last commit
            if self.piece is not None:
                for t in self.piece:
                    t.zero_()                                # a piece of a cut micro-batch is never folded in
            return
        self.added = int(pos) if pos is not None else self.added + 1
        if last or self.added % self.k != 0:
            if self.piece is not None:
                for t in self.piece:
                    t.zero_()
            return
        if self.overlap:
            self._launch_overlap()
            return
        if self.mode == 'keep':
            self._keep_commit()
            return
        t0 = time.time()
        flats = [self.ad.zero2_flat_partial(i) for i in self.groups]         # fresh copies: the snapshot S_added
        if self.cuda and self.done_ev is not None:
            torch.cuda.current_stream().wait_event(self.done_ev)             # my previous round is complete
        # the old current copy becomes the previous one, buffers and tokens together; the new current token is invalid until a
        # complete round overwrites it (a round cut by a stop leaves -1, never an old token next to new data)
        self.copy_prev, self.copy_cur = self.copy_cur, self.copy_prev
        self.tok_prev, self.tok_cur = self.tok_cur, self.tok_prev
        self.tok_cur.fill_(-1.0)
        self.tok_prev_h, self.tok_cur_h = self.tok_cur_h, None
        self.tok_send[0].fill_(float(self.step)); self.tok_send[1].fill_(float(self.added))     # on the GPU: no host wait
        works = dist.batch_isend_irecv([dist.P2POp(dist.isend, f, self.partner, group=self.pg) for f in flats] +
                                       [dist.P2POp(dist.irecv, c, self.owner, group=self.pg) for c in self.copy_cur])
        works += dist.batch_isend_irecv([dist.P2POp(dist.isend, self.tok_send, self.partner, group=self.pg),
                                         dist.P2POp(dist.irecv, self.tok_cur, self.owner, group=self.pg)])
        if self.asyn and self.cuda:
            with torch.cuda.stream(self.side):
                for w in works:
                    w.wait()
                ev = torch.cuda.Event()
                ev.record(self.side)
            for f in flats:
                f.record_stream(self.side)
            self.done_ev, self.inflight = ev, works
        else:
            for w in works:
                w.wait()
            if self.cuda:                                    # the round's end on the current stream, for _tok_host's query
                ev = torch.cuda.Event()
                ev.record()
                self.done_ev = ev
        self.snap_prev, self.snap_cur, self.c = self.snap_cur, flats, self.added
        self.ms.append(round(1000 * (time.time() - t0), 2))

    def _tok_host(self, i, t, mirror, limit_s=2.0):
        """A token's value without waiting on the GPU (the monitor thread may ask while a cut healthy reduction keeps it busy:
        M3C1b, M3C1e): the host copy; else (p2p mode) once the round has ended (done_ev, an event query), a copy on the side
        stream into a pinned buffer made at the start, polled by an event; [-1, -1] while unknown."""
        if mirror is not None:
            return list(mirror)
        if not self.cuda:
            return [int(x) for x in t.tolist()]
        if self.done_ev is not None and not self.done_ev.query():
            return [-1, -1]
        h = self.tok_pin[i]
        s = self.tok_stream if self.overlap else self.side         # overlap: never behind a reduction on the side stream
        with torch.cuda.stream(s):
            if self.overlap and self.done_ev is not None:
                s.wait_event(self.done_ev)
            h.copy_(t, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(s)
        t0 = time.time()
        while not ev.query():
            if time.time() - t0 > limit_s:
                return [-1, -1]
            time.sleep(0.001)
        return [int(x) for x in h.tolist()]

    def _keep_commit(self):
        """keep mode, after micro-batch `added`: my own snapshot (as in p2p mode), and my owner's sums folded in from the piece
        this rank kept during the micro-batch's all-reduces. The previous copy and token move to prev first."""
        t0 = time.time()
        flats = [self.ad.zero2_flat_partial(i) for i in self.groups]
        self.copy_prev, self.copy_cur = self.copy_cur, self.copy_prev
        self.tok_prev, self.tok_cur = self.tok_cur, self.tok_prev
        for cur, prv, pc in zip(self.copy_cur, self.copy_prev, self.piece):
            if self.added == self.k:
                cur.copy_(pc)                                # the owner's first commit of the step: its sums ARE the piece
            else:
                torch.add(prv, pc, out=cur)                  # the owner's accumulated_grad.add_(new) (stage_1_and_2.py:802)
            pc.zero_()
        self.tok_cur[0].fill_(float(self.step)); self.tok_cur[1].fill_(float(self.added))
        self.tok_prev_h, self.tok_cur_h = self.tok_cur_h, [int(self.step), int(self.added)]
        self.snap_prev, self.snap_cur, self.c = self.snap_cur, flats, self.added
        self.ms.append(round(1000 * (time.time() - t0), 2))

    # ------------------------------------------------------------------ overlap (FTV2_COMMIT_OVERLAP=1)
    def _launch_overlap(self):
        """This commit's work after micro-batch `added`, enqueued on the caller's current stream (the side stream: the adapter runs
        the reduction and ZeRO's epilogue there), into the spare buffers. The work is the same as the synchronous commit's (the
        snapshot; keep: the owner's piece folded into its copy; p2p: the sums and the token sent to the partner, the owner's
        received); only where it is written differs. The position moves in promote()."""
        assert self.pend is None, 'a commit is in flight: the adapter promotes or drops it before the next reduction'
        t0 = time.time()
        step, pos = int(self.step), int(self.added)
        mid = torch.cuda.Event(enable_timing=True)
        mid.record()                                         # the reduction and ZeRO's epilogue end here on the side stream
        tok = self.tok_spare
        if self.mode == 'keep':
            # the copy in flight is folded into the piece itself (no fourth buffer): piece := the owner's previous commit +
            # this interval's piece, the same bf16 sum as the synchronous torch.add(prv, pc, out=cur) (a + b == b + a); at
            # pos == k the piece IS the owner's first commit of the step. The snapshot S_pos is taken at promote().
            flats = None
            for base, pc in zip(self.copy_cur, self.piece):
                if pos != self.k:
                    pc.add_(base)
            new = list(self.piece)
            tok[0].fill_(float(step)); tok[1].fill_(float(pos))
            tok_h = [step, pos]
        else:
            flats = [self.ad.zero2_flat_partial(i) for i in self.groups]     # fresh copies: the snapshot S_pos, sent as is
            new = self.copy_spare
            tok.fill_(-1.0)
            self.tok_send[0].fill_(float(step)); self.tok_send[1].fill_(float(pos))
            works = dist.batch_isend_irecv([dist.P2POp(dist.isend, f, self.partner, group=self.pg) for f in flats] +
                                           [dist.P2POp(dist.irecv, c, self.owner, group=self.pg) for c in new])
            works += dist.batch_isend_irecv([dist.P2POp(dist.isend, self.tok_send, self.partner, group=self.pg),
                                             dist.P2POp(dist.irecv, tok, self.owner, group=self.pg)])
            for w in works:
                w.wait()                                     # the side stream waits for the transfers (not the host)
            tok_h = None                                     # read once the round is seen complete (promote)
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.pend = {'c': pos, 'flats': flats, 'copy': new, 'tok': tok, 'tok_h': tok_h, 'ev': ev, 'mid': mid,
                     't0': self.side_t0}
        self.side_t0 = None
        self.ms.append(round(1000 * (time.time() - t0), 2))

    def promote(self, block, stop_fn, ok_fn):
        """The commit in flight becomes this rank's position once the host sees its end event complete and ok_fn() holds (no stop
        begun, no reduction communicator in error): a reduction that ended because of a stop finds the stop flag set (it is set
        before the revoke), as the synchronous check found it. block: poll until the end, giving up when stop_fn() turns true.
        Returns 'none' (nothing in flight), 'done' (promoted), 'pending' (not ended, block False) or 'refused'."""
        p = self.pend
        if p is None:
            return 'none'
        while not p['ev'].query():
            if not block:
                return 'pending'
            if stop_fn():
                return 'refused'
            time.sleep(0.0002)
        if not ok_fn():
            return 'refused'
        if p['tok_h'] is None:                               # p2p: the owner's token arrived with its sums
            p['tok_h'] = self._tok_read(p['tok'], p['ev'])
        try:
            red = p['t0'].elapsed_time(p['mid']) if p['t0'] is not None else None
            self.ov_ms.append((round(red, 1) if red is not None else None, round(p['mid'].elapsed_time(p['ev']), 1)))
        except Exception:
            pass
        if self.mode == 'keep':
            # the copy dropping out (the owner's commit c - k) becomes the next piece, zeroed on the side stream before the
            # next reduction writes into it; my snapshot of the commit promoted is copied now, into the snapshot dropping out
            old = self.copy_prev
            self.copy_prev, self.copy_cur = self.copy_cur, p['copy']
            self.piece[:] = old
            with torch.cuda.stream(self.side):
                for t in self.piece:
                    t.zero_()
            snap = self._snap_take()
        else:
            self.copy_prev, self.copy_cur, self.copy_spare = self.copy_cur, p['copy'], self.copy_prev
            snap = p['flats']
        self.tok_prev, self.tok_cur, self.tok_spare = self.tok_cur, p['tok'], self.tok_prev
        self.tok_prev_h, self.tok_cur_h = self.tok_cur_h, p['tok_h']
        self.snap_prev, self.snap_cur, self.c = self.snap_cur, snap, p['c']
        self.done_ev, self.pend = p['ev'], None
        return 'done'

    def _tok_read(self, t, ev, limit_s=2.0):
        """A token's value once its round has ended (ev complete), copied on a stream of its own; None if not read in time."""
        with torch.cuda.stream(self.tok_stream):
            self.tok_stream.wait_event(ev)
            self.tok_pin_p.copy_(t, non_blocking=True)
            e2 = torch.cuda.Event()
            e2.record(self.tok_stream)
        t0 = time.time()
        while not e2.query():
            if time.time() - t0 > limit_s:
                return None
            time.sleep(0.0002)
        return [int(x) for x in self.tok_pin_p.tolist()]

    def rollback(self):
        """Drop the commit in flight (a stop came before its end was seen): the position stays at the last promoted commit, whose
        buffers it never touched; the spare buffers are spare again."""
        if self.pend is not None:
            self.log(f'commit in flight dropped: step {self.step} micro-batch {self.pend["c"]}; the position stays at {self.c}')
            self.pend = None

    def settle(self):
        """Before the sums or the copies are used outside the side stream: the commit in flight is dropped and the current stream
        waits for the side stream's work (a wait on the GPU; the host goes on)."""
        if self.overlap:
            self.rollback()
            torch.cuda.current_stream().wait_stream(self.side)

    def restore_own(self, c):
        """Set my partial sums to my snapshot at commit position c (c = 0: zeros)."""
        self.settle()
        snap = None if c == 0 else (self.snap_cur if c == self.c else self.snap_prev if c == self.c - self.k else 'missing')
        if snap == 'missing':
            raise RuntimeError(f'no snapshot of commit {c} (mine {self.c}, k {self.k})')
        for n, i in enumerate(self.groups):
            self.ad.zero2_set_partial(i, None if snap is None else snap[n])
        return snap

    def target(self, J):
        """My own sums at commit position J (flat per group; None = zeros)."""
        self.settle()
        if J == 0:
            return None
        if J == self.c:
            return self.snap_cur
        if J == self.c - self.k:
            return self.snap_prev
        raise RuntimeError(f'no snapshot of commit {J} (mine {self.c}, k {self.k})')

    def seed(self, step, J, flats):
        """After a rebuild at (step, J): every rank sends its sums S_J (flats, None = zeros) and the token to its partner, and
        receives its owner's (collective: new pairs after a join)."""
        self.settle()
        send = flats if flats is not None else [torch.zeros_like(c) for c in self.copy_cur]
        self.tok_send[0].fill_(float(step)); self.tok_send[1].fill_(float(J))
        for w in dist.batch_isend_irecv([dist.P2POp(dist.isend, f, self.partner, group=self.pg) for f in send] +
                                        [dist.P2POp(dist.irecv, c, self.owner, group=self.pg) for c in self.copy_cur]):
            w.wait()
        for w in dist.batch_isend_irecv([dist.P2POp(dist.isend, self.tok_send, self.partner, group=self.pg),
                                         dist.P2POp(dist.irecv, self.tok_cur, self.owner, group=self.pg)]):
            w.wait()
        self.tok_prev.fill_(-1.0)
        if self.cuda:
            torch.cuda.synchronize(self.device)
        self.tok_cur_h, self.tok_prev_h = [int(x) for x in self.tok_cur.tolist()], [-1, -1]

    # ------------------------------------------------------------------ recovery
    def report(self):
        """After a stop (GPU idle): my last commit, and my owner's last two commits as their tokens say."""
        self.settle()
        cur, prv = self._tok_host(0, self.tok_cur, self.tok_cur_h), self._tok_host(1, self.tok_prev, self.tok_prev_h)
        return {'commit_step': self.step, 'commit_c': self.c, 'commit_k': self.k,
                'copy_cur': [int(cur[0]), int(cur[1])], 'copy_prev': [int(prv[0]), int(prv[1])]}

    def copy_of(self, step, J):
        """The owner's sums at (step, J) from my copies, or None."""
        self.settle()
        cur, prv = self._tok_host(0, self.tok_cur, self.tok_cur_h), self._tok_host(1, self.tok_prev, self.tok_prev_h)
        if (int(cur[0]), int(cur[1])) == (step, J):
            return self.copy_cur
        if (int(prv[0]), int(prv[1])) == (step, J):
            return self.copy_prev
        return None


def decide_microbatch(reports, lost, world, stride, step):
    """(ok, J, why): the resume micro-batch of step `step` from the survivors' commit reports (pure; every rank computes it).
    J = the lowest last commit among the survivors; every survivor is at J or J + k; every lost owner's copy holds (step, J).
    If a lost owner's copy of J was cut (the owner died while its copy was in flight: E5b 5862517, the kill fired at the first
    call after the commit's asynchronous send) and every survivor is at J itself, J - k is the next candidate: each survivor
    keeps its own snapshot of J - k (snap_prev) and the copy keeps the owner's previous commit (copy_prev)."""
    cs = [r['commit_c'] if r.get('commit_step') == step else 0 for r in reports.values()]
    if not cs:
        return True, 0, 'no survivor report'
    k = max(r.get('commit_k', 1) for r in reports.values())
    J = min(cs)
    if any(c not in (J, J + k) for c in cs):
        return False, None, f'commit positions {sorted(set(cs))}: a rank more than one commit ahead'

    def copies_hold(j):
        for o in lost:
            import rc_pairs
            h = reports.get(rc_pairs.partner(o, world, stride))
            if h is None:
                return f'no report from the rank that keeps the copy of rank {o}'
            if [step, j] not in (h['copy_cur'], h['copy_prev']):
                return f'the copy of rank {o} holds {h["copy_cur"]} / {h["copy_prev"]}, not [{step}, {j}]'
        return None

    if J == 0:
        return True, 0, f'resume step {step} at its start (no commit common to every rank)'
    why = copies_hold(J)
    if why is None:
        return True, J, f'resume step {step} at micro-batch {J}'
    if all(c == J for c in cs) and J - k >= 0:
        if J - k == 0:
            return True, 0, f'{why}; resume step {step} at its start'
        why2 = copies_hold(J - k)
        if why2 is None:
            return True, J - k, f'{why}; resume step {step} at micro-batch {J - k} (the previous commit)'
        return False, None, f'{why}; and {why2}'
    return False, None, why


def decide_microbatch_pipe(reports, lost, world, stride, step):
    """(ok, J, J_by_stage, why): milestone 3, under a pipeline (pure; every rank computes it). Each stage commits on its own
    clock: its data-parallel group reduces after the stage's own k-th backward, and in one-forward-one-backward the last stage
    finishes a micro-batch's backward first, so the stages' positions differ by design. decide_microbatch runs per stage, on
    that stage's survivors and lost ranks (a lost rank's copy is on the same stage's next replica, rc_pairs): J_s. Every stage
    restarts the pipeline at J = min over the stages of J_s (the data and the schedule start there), and stage s drops the
    gradients of micro-batches J..J_s-1, which its sums hold already (m3_map question 5, B)."""
    import rc_pairs
    by = {}
    for r, rep in reports.items():
        by.setdefault(rc_pairs.stage_of(int(r), world), {})[int(r)] = rep
    Js, whys = {}, []
    for st, reps in sorted(by.items()):
        lost_s = [o for o in lost if rc_pairs.stage_of(o, world) == st]
        if any(r.get('raw') for r in reps.values()):      # FTV2_COMMIT_MODE=raw: own snapshots and host copies (rc_commit_raw)
            from rc_commit_raw import decide_microbatch_raw
            ok, J, why = decide_microbatch_raw(reps, lost_s, world, stride, step)
        else:
            ok, J, why = decide_microbatch(reps, lost_s, world, stride, step)
        whys.append(f'stage {st}: {why}')
        if not ok:
            return False, None, None, '; '.join(whys)
        Js[st] = J
    missing = sorted({rc_pairs.stage_of(o, world) for o in lost} - set(Js))
    if missing:
        return False, None, None, f'no survivor report from stage(s) {missing}'
    J = min(Js.values()) if Js else 0
    return True, J, Js, '; '.join(whys) + f'; every stage restarts at micro-batch {J}'
