"""rc_commit.py (ft-sys-v2 recovery, milestone 2, 2026-11-06): commits inside the step. Framework-free: it sees the optimizer
adapter's per-group partial gradient sums only. NOT YET RUN ON GPUs (written while the queue was full; see RESULTS_M1_M2.md M2).

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
        network; k = 1 only (every micro-batch is folded in). After a rebuild the copy is seeded once by p2p, as in p2p mode."""
import os
import time

import torch
import torch.distributed as dist


class Commit:
    def __init__(self, adapter, rank, world, device, log, group=None):
        self.ad, self.rank, self.world, self.device, self.log = adapter, rank, world, device, log
        self.stride = int(os.environ.get('FTV2_PARTNER_STRIDE', '8'))
        self.partner, self.owner = (rank + self.stride) % world, (rank - self.stride) % world
        self.k = int(os.environ.get('FTV2_COMMIT_K', '1'))
        self.asyn = os.environ.get('FTV2_COMMIT_ASYNC', '1') == '1'
        self.pg = group
        self.cuda = device.type == 'cuda'
        self.side = torch.cuda.Stream(device=device) if self.cuda else None
        self.groups = adapter.zero2_groups()
        self.step, self.added, self.c = None, 0, 0           # step; micro-batches added; last commit (micro-batches)
        self.snap_cur, self.snap_prev = None, None           # my S_c and S_{c-k} (flat per group), None = zeros
        self.copy_cur = [torch.zeros(adapter.zero2_partition_size(i), dtype=adapter.zero2_dtype(), device=device)
                         for i in self.groups]
        self.copy_prev = [torch.zeros_like(t) for t in self.copy_cur]
        self.tok_cur = torch.full((2,), -1.0, dtype=torch.float64, device=device)
        self.tok_prev = torch.full((2,), -1.0, dtype=torch.float64, device=device)
        self.tok_send = torch.zeros(2, dtype=torch.float64, device=device)
        self.done_ev, self.inflight = None, None
        self.ms = []                                         # host time of each commit's launch
        self.mode = os.environ.get('FTV2_COMMIT_MODE', 'p2p')
        self.piece = None
        if self.mode == 'keep':
            if self.k != 1:
                raise ValueError('FTV2_COMMIT_MODE=keep folds in every micro-batch: FTV2_COMMIT_K must be 1')
            self.piece = [torch.zeros_like(t) for t in self.copy_cur]
            adapter.wrap_reduce_keep(self.owner, self.piece)

    def begin_step(self, step):
        """The top of an iteration: no micro-batch of this step is summed or committed yet."""
        self.step, self.added, self.c, self.snap_cur, self.snap_prev = int(step), 0, 0, None, None
        if self.mode == 'keep':                              # the owner's sums at micro-batch 0 of this step are zeros
            for t in self.copy_cur + self.copy_prev + self.piece:
                t.zero_()
            self.tok_cur[0].fill_(float(step)); self.tok_cur[1].fill_(0.0)
            self.tok_prev.fill_(-1.0)

    def set_position(self, step, c, snap):
        """After a restore at (step, c): the sums are S_c."""
        self.step, self.added, self.c, self.snap_cur, self.snap_prev = int(step), int(c), int(c), snap, None

    # ------------------------------------------------------------------ after ZeRO's per-micro-batch epilogue
    def after_epilogue(self, stopping, last):
        if stopping:
            self.restore_own(self.c)                         # drop whatever came after the last commit
            if self.piece is not None:
                for t in self.piece:
                    t.zero_()                                # a piece of a cut micro-batch is never folded in
            return
        self.added += 1
        if last or self.added % self.k != 0:
            if self.piece is not None:
                for t in self.piece:
                    t.zero_()
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
        self.snap_prev, self.snap_cur, self.c = self.snap_cur, flats, self.added
        self.ms.append(round(1000 * (time.time() - t0), 2))

    def _keep_commit(self):
        """keep mode, after micro-batch `added`: my own snapshot (as in p2p mode), and my owner's sums folded in from the piece
        this rank kept during the micro-batch's all-reduces. The previous copy and token move to prev first."""
        t0 = time.time()
        flats = [self.ad.zero2_flat_partial(i) for i in self.groups]
        self.copy_prev, self.copy_cur = self.copy_cur, self.copy_prev
        self.tok_prev, self.tok_cur = self.tok_cur, self.tok_prev
        for cur, prv, pc in zip(self.copy_cur, self.copy_prev, self.piece):
            if self.added == 1:
                cur.copy_(pc)                                # the owner's first micro-batch: its sums ARE the piece
            else:
                torch.add(prv, pc, out=cur)                  # the owner's accumulated_grad.add_(new) (stage_1_and_2.py:802)
            pc.zero_()
        self.tok_cur[0].fill_(float(self.step)); self.tok_cur[1].fill_(float(self.added))
        self.snap_prev, self.snap_cur, self.c = self.snap_cur, flats, self.added
        self.ms.append(round(1000 * (time.time() - t0), 2))

    def restore_own(self, c):
        """Set my partial sums to my snapshot at commit position c (c = 0: zeros)."""
        snap = None if c == 0 else (self.snap_cur if c == self.c else self.snap_prev if c == self.c - self.k else 'missing')
        if snap == 'missing':
            raise RuntimeError(f'no snapshot of commit {c} (mine {self.c}, k {self.k})')
        for n, i in enumerate(self.groups):
            self.ad.zero2_set_partial(i, None if snap is None else snap[n])
        return snap

    def target(self, J):
        """My own sums at commit position J (flat per group; None = zeros)."""
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

    # ------------------------------------------------------------------ recovery
    def report(self):
        """After a stop (GPU idle): my last commit, and my owner's last two commits as their tokens say."""
        cur, prv = self.tok_cur.tolist(), self.tok_prev.tolist()
        return {'commit_step': self.step, 'commit_c': self.c, 'commit_k': self.k,
                'copy_cur': [int(cur[0]), int(cur[1])], 'copy_prev': [int(prv[0]), int(prv[1])]}

    def copy_of(self, step, J):
        """The owner's sums at (step, J) from my copies, or None."""
        if (int(self.tok_cur[0]), int(self.tok_cur[1])) == (step, J):
            return self.copy_cur
        if (int(self.tok_prev[0]), int(self.tok_prev[1])) == (step, J):
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
            h = reports.get((o + stride) % world)
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
