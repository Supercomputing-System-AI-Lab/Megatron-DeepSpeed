"""rc_commit_raw.py (ft-sys-v2 recovery, 2026-11-12): commits by a copy of the RAW gradient, FTV2_COMMIT_MODE=raw (pipeline only).

The question (the user and a colleague, 11-12): why reduce the gradient every k micro-batches at all? Send the stage's raw
accumulated gradient to the partner instead, and leave DeepSpeed's one reduction at the step's end as it is.

What a commit is here. After micro-batch c (a multiple of k, below m) of a step, every parameter's .grad holds the sum of the
micro-batches 1..c of this rank (ZeRO-1 under the pipeline reduces nothing inside the step). The commit:
  1. copies every .grad into one flat buffer on the GPU (the snapshot S_c; on the side stream, after the backward's event; the
     next BACKWARD waits for this copy on the GPU, the next forward does not: wait_snapshot);
  2. sends the buffer to the partner in chunks, and receives the owner's the same way: each chunk lands in one of two small GPU
     buffers and is copied from there into pinned HOST memory on a third stream, so the partner's copies cost no GPU memory
     beyond the two chunk buffers;
  3. a token [step, c, numel] in a last round (a token that arrived means its gradient arrived).
Nothing of DeepSpeed runs on the side stream and nothing is allocated there: every buffer is made in __init__. (The reduce
modes run ZeRO's reduction on the side stream, which allocates and frees temporaries there; the 63B overlap runs of 11-12
turned NaN from step 2 on, on the fullest stage.)
The step's end is DeepSpeed's own, untouched: ReduceTiedGrads and ReduceGrads see every micro-batch's gradient in .grad, so the
sums are those of a run without commits, bit for bit (same additions in the same order).

Buffers per rank (N = the rank's trainable parameters, in ZeRO's accumulation dtype, bf16):
  GPU   own[0], own[1]   N each    my snapshot of the last promoted commit, and of the one in flight (they alternate)
        rbuf[0], rbuf[1] one chunk the owner's chunks on their way to the host
  host  host[0..2]       N each    pinned: the owner's last two commits (cur, prev) and the one in flight
Positions after a stop. own_set = [c promoted] + [the commit in flight, if its snapshot ended on the GPU]: my own snapshot is
valid without the network. copy_cur / copy_prev = the owner's last two tokens. decide_microbatch_raw: J_s = the highest
position that every survivor of the stage holds and every lost owner's copy holds; 0 (the step's start) always qualifies.
Restore: a survivor sets .grad from its snapshot at J_s; the joiner receives the lost rank's from the partner's host copy.

Cost per commit: N bf16 each way over the network (63B PP 4 TP 2: 5.1 GB per GPU), against the reduce modes' reduction
(about 2.2 GB) plus slice copy (2.1 GB)."""
import os
import time

import torch
import torch.distributed as dist


class RawCommit:
    mode = 'raw'
    spaced = False

    def __init__(self, adapter, rank, world, device, log, group=None):
        self.ad, self.rank, self.world, self.device, self.log = adapter, rank, world, device, log
        self.stride = int(os.environ.get('FTV2_PARTNER_STRIDE', '8'))
        import rc_pairs
        self.partner, self.owner = rc_pairs.partner(rank, world, self.stride), rc_pairs.owner(rank, world, self.stride)
        self.k = int(os.environ.get('FTV2_COMMIT_K', '1'))
        self.asyn, self.overlap = True, True                # always on the side stream
        self.pg = group
        self.cuda = device.type == 'cuda'
        prio = -1 if (self.cuda and os.environ.get('FTV2_COMMIT_SIDE_PRIORITY', 'high') == 'high') else 0
        self.side = torch.cuda.Stream(device=device, priority=prio) if self.cuda else None
        self.h_stream = torch.cuda.Stream(device=device, priority=prio) if self.cuda else None   # GPU -> host copies
        self.tok_stream = torch.cuda.Stream(device=device, priority=prio) if self.cuda else None
        self.dtype = adapter.zero2_dtype()
        self.sizes = [int(p.numel()) for p in adapter.raw_params()]
        self.N = sum(self.sizes)
        self.own = [torch.zeros(self.N, dtype=self.dtype, device=device) for _ in range(2)]
        per = max(1, int(float(os.environ.get('FTV2_RAW_CHUNK_MB', '512')) * 2 ** 20) // self.own[0].element_size())
        self.chunks = [(a, min(a + per, self.N)) for a in range(0, self.N, per)]
        self.rbuf = [torch.empty(min(per, self.N), dtype=self.dtype, device=device) for _ in range(2)]
        self.host = [torch.zeros(self.N, dtype=self.dtype, pin_memory=self.cuda) for _ in range(3)]
        self.h_cur, self.h_prev, self.h_spare = 0, 1, 2
        self.tok_cur_h, self.tok_prev_h = [-1, -1], [-1, -1]
        self.tok_send = torch.zeros(3, dtype=torch.float64, device=device)
        self.tok_recv = torch.full((3,), -1.0, dtype=torch.float64, device=device)
        self.tok_pin = torch.empty(3, dtype=torch.float64, pin_memory=self.cuda)
        self.step, self.added, self.c = None, 0, 0
        self.own_idx = None                                  # own[own_idx] = S_c (None: c = 0, zeros)
        self.pend, self.extra, self.side_t0 = None, None, None
        self.ov_ms, self.ms = [], []                         # per promoted commit: (snapshot ms, exchange ms); host launch ms

    # ------------------------------------------------------------------ the step
    def begin_step(self, step):
        self.settle()
        self.step, self.added, self.c, self.own_idx, self.extra = int(step), 0, 0, None, None

    def set_position(self, step, c, snap):
        """After a restore at (step, c): snap (a flat tensor, or None = zeros) is S_c."""
        self.step, self.added, self.c, self.extra = int(step), int(c), int(c), None
        self.own_idx = None
        if int(c) > 0 and snap is not None:
            self.own[0].copy_(snap)
            self.own_idx = 0

    def after_epilogue(self, stopping, last, pos=None):
        """ZeRO's epilogue runs only at the step's end in this mode (no spaced reductions): nothing to commit or to take back."""
        if not stopping and pos is not None:
            self.added = int(pos)

    def wait_snapshot(self):
        """Before the next backward (which adds into .grad in place): the current stream waits for the snapshot in flight."""
        p = self.pend
        if p is not None and self.cuda:
            torch.cuda.current_stream().wait_event(p['snap_ev'])

    def _event(self, timing=False):
        if not self.cuda:
            return None
        e = torch.cuda.Event(enable_timing=timing)
        e.record()
        return e

    def _exchange(self, buf, hostdst, step, pos, bulk=True):
        """On the current stream: my flat buffer to the partner and the owner's into hostdst (pinned), chunk by chunk, then the
        tokens. The host does not wait: the stream does."""
        cur = torch.cuda.current_stream() if self.cuda else None
        d2h = [None, None]
        if bulk:
            for j, (a, b) in enumerate(self.chunks):
                rb = self.rbuf[j % 2][:b - a]
                if d2h[j % 2] is not None:
                    cur.wait_event(d2h[j % 2])               # this chunk buffer's previous content is on the host
                for w in dist.batch_isend_irecv([dist.P2POp(dist.isend, buf[a:b], self.partner, group=self.pg),
                                                 dist.P2POp(dist.irecv, rb, self.owner, group=self.pg)]):
                    w.wait()
                if self.cuda:
                    e = self._event()
                    with torch.cuda.stream(self.h_stream):
                        self.h_stream.wait_event(e)
                        hostdst[a:b].copy_(rb, non_blocking=True)
                        d2h[j % 2] = self._event()
                else:
                    hostdst[a:b].copy_(rb)
        self.tok_send[0].fill_(float(step)); self.tok_send[1].fill_(float(pos)); self.tok_send[2].fill_(float(self.N))
        self.tok_recv.fill_(-1.0)
        for w in dist.batch_isend_irecv([dist.P2POp(dist.isend, self.tok_send, self.partner, group=self.pg),
                                         dist.P2POp(dist.irecv, self.tok_recv, self.owner, group=self.pg)]):
            w.wait()
        for e in d2h:
            if e is not None:
                cur.wait_event(e)

    def launch(self, pos):
        """After the backward of micro-batch pos (the host's current stream is the compute stream): enqueue the snapshot and
        the exchange on the side stream. The position moves in promote()."""
        assert self.pend is None, 'a commit is in flight: the adapter promotes or drops it before the next launch'
        t0 = time.time()
        self.added = int(pos)
        b = 0 if self.own_idx is None else 1 - self.own_idx
        buf = self.own[b]
        params = self.ad.raw_params()
        ev0 = self._event()
        if self.cuda:
            for p in params:
                if p.grad is not None:
                    p.grad.record_stream(self.side)
            ctx = torch.cuda.stream(self.side)
        else:
            import contextlib
            ctx = contextlib.nullcontext()
        with ctx, torch.no_grad():
            if self.cuda:
                self.side.wait_event(ev0)
            ta = self._event(True)
            off = 0
            for p, n in zip(params, self.sizes):
                g = p.grad
                if g is None:
                    buf[off:off + n].zero_()
                else:
                    buf[off:off + n].copy_(g.reshape(-1))
                off += n
            snap_ev = self._event(True)
            self._exchange(buf, self.host[self.h_spare], int(self.step), int(pos))
            ev = self._event(True)
        self.pend = {'c': int(pos), 'buf': b, 'snap_ev': snap_ev, 'ev': ev, 't0': ta}
        self.ms.append(round(1000 * (time.time() - t0), 2))

    def _tok_read(self, ev, limit_s=2.0):
        if not self.cuda:
            return [int(x) for x in self.tok_recv.tolist()]
        with torch.cuda.stream(self.tok_stream):
            self.tok_stream.wait_event(ev)
            self.tok_pin.copy_(self.tok_recv, non_blocking=True)
            e2 = self._event()
        t0 = time.time()
        while not e2.query():
            if time.time() - t0 > limit_s:
                return None
            time.sleep(0.0002)
        return [int(x) for x in self.tok_pin.tolist()]

    def promote(self, block, stop_fn, ok_fn):
        """As Commit.promote: 'none', 'done', 'pending' or 'refused'."""
        p = self.pend
        if p is None:
            return 'none'
        if self.cuda:
            while not p['ev'].query():
                if not block:
                    return 'pending'
                if stop_fn():
                    return 'refused'
                time.sleep(0.0002)
        if not ok_fn():
            return 'refused'
        tok = self._tok_read(p['ev'])
        if tok is not None and tok[:2] == [int(self.step), p['c']]:
            self.h_prev, self.h_cur, self.h_spare = self.h_cur, self.h_spare, self.h_prev
            self.tok_prev_h, self.tok_cur_h = self.tok_cur_h, tok[:2]
        else:
            self.log(f'raw commit {p["c"]} of step {self.step}: the owner\'s token is {tok}; its copy stays at {self.tok_cur_h}')
        if self.cuda:
            try:
                self.ov_ms.append((round(p['t0'].elapsed_time(p['snap_ev']), 1), round(p['snap_ev'].elapsed_time(p['ev']), 1)))
            except Exception:
                pass
        self.c, self.own_idx, self.pend = p['c'], p['buf'], None
        return 'done'

    def rollback(self):
        """A stop came before the commit in flight was promoted: its exchange does not count, its snapshot does if it ended."""
        if self.pend is not None:
            self.log(f'commit in flight dropped: step {self.step} micro-batch {self.pend["c"]}; the position stays at {self.c}')
            self.extra, self.pend = self.pend, None

    def settle(self):
        self.rollback()
        if self.cuda:
            torch.cuda.current_stream().wait_stream(self.side)

    def _extra_ok(self):
        e = self.extra
        return e is not None and (not self.cuda or e['snap_ev'].query())

    # ------------------------------------------------------------------ recovery
    def own_set(self):
        return [int(self.c)] + ([int(self.extra['c'])] if self._extra_ok() else [])

    def target(self, J):
        """My raw gradient sums at commit position J (a flat tensor; None = zeros)."""
        self.settle()
        if J == 0:
            return None
        if J == self.c and self.own_idx is not None:
            return self.own[self.own_idx]
        if self._extra_ok() and J == self.extra['c']:
            return self.own[self.extra['buf']]
        raise RuntimeError(f'no snapshot of commit {J} (mine {self.own_set()}, k {self.k})')

    def restore_own(self, c):
        flat = self.target(c)
        self.ad.raw_set(flat, self.sizes)
        return flat

    def copy_of(self, step, J):
        """The owner's sums at (step, J) from my host copies (a list of one pinned tensor), or None."""
        self.settle()
        if self.tok_cur_h == [step, J]:
            return [self.host[self.h_cur]]
        if self.tok_prev_h == [step, J]:
            return [self.host[self.h_prev]]
        return None

    def seed(self, step, J, flats):
        """After a (re)build at (step, J): every rank sends S_J (when J > 0) and the token to its partner (collective)."""
        self.settle()
        self._exchange(self.own[self.own_idx] if (J > 0 and self.own_idx is not None) else self.own[0], self.host[self.h_cur],
                       int(step), int(J), bulk=J > 0)
        if self.cuda:
            torch.cuda.synchronize(self.device)
        tok = [int(x) for x in self.tok_recv.tolist()]
        if tok[2] != self.N:
            raise ValueError(f'raw commits need equal sizes on a rank and its owner: mine {self.N}, rank {self.owner} has {tok[2]}')
        self.tok_cur_h, self.tok_prev_h = tok[:2], [-1, -1]

    def report(self):
        self.settle()
        return {'commit_step': self.step, 'commit_c': self.c, 'commit_k': self.k, 'raw': 1, 'own_set': self.own_set(),
                'copy_cur': list(self.tok_cur_h), 'copy_prev': list(self.tok_prev_h)}


def decide_microbatch_raw(reports, lost, world, stride, step):
    """(ok, J, why) for one stage in raw mode (pure). J = the highest position held by every survivor's own snapshots and by the
    copy of every lost rank (on its partner); the step's start (0) always qualifies."""
    import rc_pairs
    if not reports:
        return True, 0, 'no survivor report'
    common = None
    for r in reports.values():
        s = set(r.get('own_set', [r.get('commit_c', 0)])) if r.get('commit_step') == step else set()
        s.add(0)
        common = s if common is None else common & s
    for o in lost:
        h = reports.get(rc_pairs.partner(o, world, stride))
        if h is None:
            return False, None, f'no report from the rank that keeps the copy of rank {o}'
        common &= {0} | {int(t[1]) for t in (h['copy_cur'], h['copy_prev']) if int(t[0]) == step}
    J = max(common)
    return True, J, (f'resume step {step} at micro-batch {J}' if J else f'resume step {step} at its start (no commit common '
                                                                          f'to every rank and copy)')
