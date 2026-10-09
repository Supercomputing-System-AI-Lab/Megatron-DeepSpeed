#!/usr/bin/env python3
"""The dispatch coordinator -- Stage 2 + Stage 3 of rebuild/DISPATCH_ENGINE_DESIGN.md.

WHAT IT IS
  A queue of BLOCK INDICES and a socket. Nothing else. It has no torch, no lm_eval, no
  tokenizer, no dataset and no model: every worker already builds the identical request
  list (Stage 1's guarantee -- simple_evaluate seeds deterministically and the adapter
  reports rank=0/world_size=1 on purpose), so the only thing that has to cross the wire
  is a small integer saying WHICH block to compute. That is why this file is stdlib and
  why it runs in the batch shell rather than on a GPU.

WHY IT EXISTS -- the measured number
  workers=N with a STATIC split balances the wrong quantity on a resume. Job 5369888
  (BBH, workers=7) balanced DOCUMENTS to max/mean 1.06 and its actual work -- cache
  misses -- to max/mean 1.54: 90/2/82/98/105/3/98 misses, i.e. 35.0% of the compute
  phase idle, with two workers finishing essentially instantly. One job in three already
  runs against a warm cache (95 of 281 jobs measured), so this is the common case, not
  the corner. Pulling blocks on demand removes that idle by construction.

WHY RAW TCP AND NOT A PROCESS GROUP  (both measured, DISPATCH_ENGINE_DESIGN.md section 2)
  * NCCL cannot receive from ANY_SOURCE at all: `ProcessGroupNCCL does not support
    recvAnysource` is a literal string in libtorch_hip.so. It also would need one global
    process group across all workers, undoing the isolation workers= exists to give.
  * gloo CAN, and must not: when one peer died, the gloo timeout closed the pairs to the
    HEALTHY peers too (`pair.cc:547 Connection closed by peer`) and killed a worker that
    was fine. A dispatcher whose whole purpose is surviving a worker death cannot use a
    transport where one death destroys the control plane.
  * Raw TCP: receive-from-any is native (`selectors.select()` over the connection set),
    a dead peer is an immediate `recv() == b''` with no timer at all, survivors are
    untouched, and the heartbeat is INDEPENDENT of the payload -- so the silence timeout
    is sized against silence (300 s) rather than against the longest legitimate block (a
    32-request generate_until block at gen_toks=1024 is minutes).
  NCCL is untouched and still does what it is good at: the model's own collectives and
  the leader's broadcast_object_list fan-out to its 7 peers.

WHAT THE COORDINATOR IS NOT
  It is NOT the definition of done. Completion is defined by the shared content-keyed
  answer file (cache.py:87-96, :280-306), which every worker appends to under flock and
  every worker reads. A block that is claimed and never finished simply has no answers,
  and the next pass recomputes it. The coordinator's contribution is DISPATCH LATENCY,
  not correctness -- saying otherwise would oversell it. Every failure path here is
  therefore allowed to be approximate; none of them can lose an answer.

STAGE 3 -- THE JOURNAL (AUDIT TRAIL ONLY; NOTHING EVER READS IT BACK)
  Every open / assign / complete / requeue / abandon / refuse is appended to a
  line-buffered, fsync'd file next to the answer file. It is a RECORD, NEVER A
  PERMISSION: nothing waits on it, no worker consults it, it grants no exclusivity, and
  -- since 2026-08-29 -- no code path in this program opens it for reading. It exists so
  an operator can reconstruct what the coordinator decided, and for nothing else.

WHY THE REPLAY WAS DELETED (2026-08-29)
  A restart used to replay this file to skip blocks a previous coordinator had recorded
  complete. That made the journal a SECOND SOURCE OF TRUTH about which work is finished.
  The answer cache is the first and only real one, two authorities can disagree, and when
  they do the replay skips work the cache does not actually have -- the same "a record
  asserting completion, trusted without verification" failure that was rejected for
  filesystem lock-claims.

  Its measured value was nothing. A block whose answers are already cached completes in
  0.00 s (job 5373261: 192 of 204 blocks at 0.00 s, 12 at ~190 s), so all a replay could
  avoid is re-offering blocks that cost a round trip rather than compute. It was also
  unreachable in production: the journal path is ${JOB_DIR}/dispatch.journal.jsonl and
  JOB_DIR carries the Slurm job id, so every job read its own empty file, and nothing
  restarts a dead coordinator. Dead code that looks finished is how someone wires it up
  later without ever meeting the authority question -- hence deleted, not disabled.

  THE STANDING INVARIANT: THE ANSWER CACHE IS THE ONLY AUTHORITY ON WHAT IS DONE. This
  coordinator's completion state derives ONLY from live worker reports on its own socket,
  never from a file.

PROTOCOL  (length-framed JSON, 8-byte big-endian prefix; JSON not pickle because the
           payload is small integers and an unpickler on a socket is a code-execution
           surface this needs nothing from)
  worker -> server   {"op":"READY",  "worker":w, "phase":p, "nblocks":n, "digest":d}
                     {"op":"DONE",   "worker":w, "phase":p, "block":b, "n":k, "secs":s}
                     {"op":"FAILED", "worker":w, "phase":p, "block":b, "error":e}
                     {"op":"BEAT",   "worker":w}
  server -> worker   {"op":"BLOCK",  "phase":p, "block":b}
                     {"op":"WAIT",   "secs":x}      nothing assignable YET; ask again
                     {"op":"STOP",   "phase":p}     phase complete -- leave the loop
                     {"op":"REFUSE", "reason":r}    digest/shape disagreement: fall back

PHASES
  One `_run(method, requests)` call on the worker is one phase. Its name is
  "<method>:<call-index>", which is identical on every worker because every worker runs
  the identical task recipe in the identical order -- and its DIGEST is a hash of the
  request list. The name is what makes two workers meet; the digest is what proves they
  are talking about the same list. Disagreeing digests are REFUSED loudly rather than
  dispatched, which converts lm-eval's silent-wrongness DP failure mode
  (lm_eval/api/model.py:189-205 are base-class no-ops) into a visible refusal.
"""
import argparse
import errno
import json
import os
import selectors
import signal
import socket
import struct
import sys
import time
from collections import deque


# --- wire ---------------------------------------------------------------------------
# 8-byte big-endian length prefix, then UTF-8 JSON. Both halves of the protocol import
# these from HERE so there is exactly one definition of the frame; dispatch.py loads this
# module by path when its package cannot be imported (login-node tests).
_LEN = struct.Struct('!Q')

# A frame is a control message about integers. Anything larger is a bug or an attacker,
# and either way the connection should die rather than allocate.
MAX_FRAME = 1 << 20


class ProtocolError(Exception):
    """A frame that cannot be trusted. Always fatal to the CONNECTION, never to the job."""


def encode(msg):
    body = json.dumps(msg, separators=(',', ':')).encode('utf-8')
    return _LEN.pack(len(body)) + body


def send_msg(sock, msg):
    sock.sendall(encode(msg))


def take_frame(buf):
    """(message, rest) if `buf` holds a whole frame, else (None, buf). Never blocks."""
    if len(buf) < _LEN.size:
        return None, buf
    (n,) = _LEN.unpack_from(buf, 0)
    if n > MAX_FRAME:
        raise ProtocolError(f'frame of {n} bytes exceeds the {MAX_FRAME}-byte cap')
    end = _LEN.size + n
    if len(buf) < end:
        return None, buf
    try:
        msg = json.loads(buf[_LEN.size:end].decode('utf-8'))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProtocolError(f'undecodable frame: {exc}')
    if not isinstance(msg, dict):
        raise ProtocolError(f'frame is {type(msg).__name__}, not an object')
    return msg, buf[end:]


def recv_msg(sock, timeout=None):
    """One whole frame from a BLOCKING socket, or None on a clean EOF.

    Used by the client, which is one thread against one server and has no reason to
    multiplex. The server does its own buffered reads under selectors.
    """
    if timeout is not None:
        sock.settimeout(timeout)
    buf = b''
    while True:
        msg, buf = take_frame(buf)
        if msg is not None:
            return msg
        chunk = sock.recv(65536)
        if not chunk:
            if buf:
                raise ProtocolError('EOF in the middle of a frame')
            return None
        buf += chunk


# --- defaults -----------------------------------------------------------------------
# The port workers already claim is 29500+i (eval_lib.py:1854), so 29600 is clear for
# every workers<=16 shape. Asserted in the tests rather than assumed.
DEFAULT_PORT = 29600

# Sized against SILENCE, not against a block. This is the whole point of not using a
# process group: dist.recv would have to use ONE number that is simultaneously longer
# than the longest legitimate block and short enough to notice a death.
HEARTBEAT_INTERVAL = 30.0
HEARTBEAT_TIMEOUT = 300.0

# How long a worker is told to sleep when the phase is not finished but nothing is
# assignable (its siblings hold the last blocks). Short, because the only cost of being
# wrong is one round trip, and the only cost of being slow is the tail of the phase.
WAIT_HINT = 2.0

# A poison block -- one that kills whatever worker touches it -- must not be able to eat
# the whole job one worker at a time. After this many assignments it is ABANDONED: logged
# by name, journalled, excluded from the phase's completion test, and left to the answer
# cache (it will simply be a miss, which is exactly what it would have been without any
# of this machinery).
MAX_ATTEMPTS = 3

# Stop handing out blocks this long before SLURM_JOB_END_TIME, so the job keeps time to
# re-read the answer file and score. Without it a job at the wall loses its score even
# though every answer is already on disk.
DISPATCH_MARGIN = 600.0


def _now():
    return time.time()


# --- Stage 3: the journal -------------------------------------------------------------
class Journal:
    """Append-only AUDIT TRAIL of every dispatch decision. Write-only, by design.

    Line-buffered and fsync'd per record, so a job killed at the wall leaves a readable
    file rather than a truncated one -- the same discipline the answer file already uses
    (cache.py:280-306). Cost is one small write per block, i.e. per multiple seconds of
    GPU time, against a Lustre file nothing else contends for.

    There is deliberately no reader. Nothing in this program ever BLOCKS on the journal,
    no worker consults it, and no coordinator -- this one or a later one -- may use it to
    decide that a block is done: that is the answer cache's job and only the answer
    cache's job. If the file cannot be opened, the coordinator says so once and runs
    without it: losing the record must never be able to stop the dispatch.

    The FORMAT is load-bearing (operators read these files today): one JSON object per
    line, keys in insertion order with `event` and `ts` appended last, separators
    (',', ':'), non-serialisable values coerced with str.
    """

    def __init__(self, path, fsync=True, log=None):
        self.path = path
        self.fsync = bool(fsync)
        self._log = log or (lambda m: None)
        self._fh = None
        self.n_written = 0
        if not path:
            return
        try:
            d = os.path.dirname(os.path.abspath(path))
            if d:
                os.makedirs(d, exist_ok=True)
            self._fh = open(path, 'a', encoding='utf-8', buffering=1)
        except OSError as exc:
            self._log(f'[dispatch] journal disabled ({path}): {exc}')
            self._fh = None

    def write(self, event, **fields):
        if self._fh is None:
            return
        rec = dict(fields)
        rec['event'] = event
        rec['ts'] = round(_now(), 3)
        try:
            self._fh.write(json.dumps(rec, separators=(',', ':'), default=str) + '\n')
            if self.fsync:
                os.fsync(self._fh.fileno())
            self.n_written += 1
        except OSError as exc:                  # a full or unwritable filesystem
            self._log(f'[dispatch] journal write failed, continuing without it: {exc}')
            self.close()

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None


# --- the queue ------------------------------------------------------------------------
class Phase:
    """One `_run(method, requests)` call's worth of blocks, and their state machine.

    THE INVARIANT, and it is the whole of the failure story: a block leaves `pending`
    only into `inflight`, and every exit from `inflight` is either into `completed` or
    back into `pending` (or, after MAX_ATTEMPTS, into `abandoned`, which is recorded and
    named). There is no other transition, so no block can be silently dropped.
    """

    def __init__(self, name, nblocks, digest, max_attempts=MAX_ATTEMPTS):
        self.name = name
        self.nblocks = int(nblocks)
        self.digest = digest
        self.max_attempts = int(max_attempts)
        self.pending = deque(range(self.nblocks))
        self.inflight = {}                      # block -> (conn_key, assigned_ts)
        self.completed = set()
        self.abandoned = set()
        self.attempts = {}                      # block -> times it has been assigned
        self.opened = _now()

    # -- transitions -------------------------------------------------------------
    # There is no way to mark a block complete except `complete()`, which is only ever
    # called from a live worker's DONE on the socket. Nothing pre-populates `completed`
    # from a file: the answer cache is the only authority on what is done.
    def take(self, conn_key):
        """The next block for this worker, or None when nothing is assignable NOW."""
        while self.pending:
            b = self.pending.popleft()
            if b in self.completed or b in self.abandoned:
                continue
            tries = self.attempts.get(b, 0)
            if tries >= self.max_attempts:
                self.abandoned.add(b)
                return ('abandon', b)
            self.attempts[b] = tries + 1
            self.inflight[b] = (conn_key, _now())
            return ('block', b)
        return None

    def complete(self, block, conn_key=None):
        """Idempotent on the block. A zombie's late DONE is absorbed, never double-counted.

        There is nothing to double-count: results are content-keyed writes and
        put_batch skips a key already present (cache.py:294), so two workers computing
        the same immutable block produce the same key with the same value. Duplicate work
        costs GPU seconds, never correctness.
        """
        block = int(block)
        fresh = block not in self.completed
        self.completed.add(block)
        self.inflight.pop(block, None)
        return fresh

    def release(self, conn_key):
        """Requeue whatever this connection was holding. Returns the blocks requeued.

        Head of the queue, not the tail: it is the OLDEST work in the phase and the
        thing most likely to be holding up the tail.
        """
        out = []
        for b, (owner, _ts) in list(self.inflight.items()):
            if owner != conn_key:
                continue
            del self.inflight[b]
            if b in self.completed:
                continue
            if self.attempts.get(b, 0) >= self.max_attempts:
                self.abandoned.add(b)
                out.append((b, 'abandon'))
            else:
                self.pending.appendleft(b)
                out.append((b, 'requeue'))
        return out

    # -- predicates --------------------------------------------------------------
    @property
    def resolved(self):
        return len(self.completed) + len(self.abandoned)

    @property
    def done(self):
        return self.resolved >= self.nblocks

    def summary(self):
        return (f'{self.name}: {len(self.completed)}/{self.nblocks} complete, '
                f'{len(self.abandoned)} abandoned, {len(self.inflight)} in flight, '
                f'{len(self.pending)} pending')


class _Conn:
    """One worker's socket and the bookkeeping the final report is made of."""

    def __init__(self, sock, addr, key):
        self.sock = sock
        self.addr = addr
        self.key = key
        self.buf = b''
        self.worker = None
        self.last_seen = _now()
        self.connected = _now()
        self.phase = None
        self.blocks_done = 0
        self.requests_done = 0
        self.compute_secs = 0.0
        self.wait_secs = 0.0                    # time spent being told WAIT
        self.waits = 0
        self._wait_since = None
        self.gone_reason = None

    def label(self):
        return f'worker {self.worker}' if self.worker is not None else f'conn {self.key}'


class DispatchServer:
    """The coordinator. One thread, one selector, one queue per phase.

    Single-threaded on purpose: every message it handles is bounded work over small
    integers, so there is no reason for a lock and no way for one worker's message to
    delay another's by more than microseconds. The only timer in the loop is the
    heartbeat reaper.
    """

    def __init__(self, host='0.0.0.0', port=DEFAULT_PORT, journal_path=None,
                 heartbeat_timeout=HEARTBEAT_TIMEOUT, max_attempts=MAX_ATTEMPTS,
                 deadline=None, log=None, journal_fsync=True, wait_hint=WAIT_HINT):
        self.log = log or (lambda m: print(m, flush=True))
        self.heartbeat_timeout = float(heartbeat_timeout)
        self.max_attempts = int(max_attempts)
        self.deadline = deadline
        self.wait_hint = float(wait_hint)

        self.phases = {}                        # name -> Phase
        self.conns = {}                         # key -> _Conn
        # Connections that have gone away. The final report must name every worker that
        # ever ran -- a worker that died holding work is exactly the one an operator is
        # looking for, and dropping it from the report would hide it.
        self.gone = []
        self._next_key = 0
        self.stopping = False                   # set by SIGTERM/SIGINT
        self.started = _now()
        self.n_seen_workers = 0
        self.n_requeued = 0
        self.n_reaped = 0
        self.n_refused = 0

        # Write-only. A journal already on disk is an operator's audit trail, not an
        # input: this coordinator starts every phase with an empty `completed` set no
        # matter what the file says.
        self.journal = Journal(journal_path, fsync=journal_fsync, log=self.log)

        self.sel = selectors.DefaultSelector()
        self.lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.lsock.bind((host, int(port)))
        self.lsock.listen(64)
        self.lsock.setblocking(False)
        self.port = self.lsock.getsockname()[1]
        self.sel.register(self.lsock, selectors.EVENT_READ, None)
        self.log(f'[dispatch] listening on {host}:{self.port} '
                 f'(heartbeat timeout {self.heartbeat_timeout:.0f}s, '
                 f'max {self.max_attempts} attempt(s) per block)')

    # -- lifecycle ---------------------------------------------------------------
    def close(self):
        for conn in list(self.conns.values()):
            self._drop(conn, 'shutdown', requeue=False)
        try:
            self.sel.unregister(self.lsock)
        except (KeyError, ValueError):
            pass
        self.lsock.close()
        self.sel.close()
        self.journal.close()

    def request_stop(self, *_a):
        self.stopping = True

    def serve_forever(self, poll=1.0, until=None):
        """Loop until `until()` says so, or a signal arrives. `until` defaults to never."""
        while not self.stopping:
            if until is not None and until(self):
                break
            self.poll(poll)
        return self

    def poll(self, timeout=1.0):
        """One turn: accept, read whatever is readable, then reap the silent."""
        for key, _events in self.sel.select(timeout=timeout):
            if key.data is None:
                self._accept()
            else:
                self._readable(key.data)
        self._reap()

    # -- connections -------------------------------------------------------------
    def _accept(self):
        try:
            sock, addr = self.lsock.accept()
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return
            raise
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError:
            pass
        # BLOCKING with a short timeout, not non-blocking: reads only happen when
        # selectors already said the socket is readable, and replies are <200 bytes, so
        # neither call can actually wait. The timeout is the belt for a half-open peer.
        sock.settimeout(30.0)
        self._next_key += 1
        conn = _Conn(sock, addr, self._next_key)
        self.conns[conn.key] = conn
        self.sel.register(sock, selectors.EVENT_READ, conn)
        self.n_seen_workers += 1

    def _drop(self, conn, reason, requeue=True):
        """Remove a connection and requeue whatever it held. The crash path.

        Detected by recv() returning b'' -- immediately, with no timer. This is the
        mechanism the whole transport choice was made for, and it is the reason a worker
        dying costs one block instead of its whole static share.
        """
        if conn.key not in self.conns:
            return
        del self.conns[conn.key]
        self._end_wait(conn)
        conn.gone_reason = reason
        self.gone.append(conn)
        try:
            self.sel.unregister(conn.sock)
        except (KeyError, ValueError):
            pass
        try:
            conn.sock.close()
        except OSError:
            pass
        if not requeue:
            return
        for phase in self.phases.values():
            for block, what in phase.release(conn.key):
                if what == 'requeue':
                    self.n_requeued += 1
                    self.log(f'[dispatch] REQUEUE phase={phase.name} block={block} '
                             f'from {conn.label()} ({reason})')
                    self.journal.write('requeue', phase=phase.name, block=block,
                                       worker=conn.worker, reason=reason)
                else:
                    self._log_abandon(phase, block, f'{reason}, attempts exhausted')

    def _reap(self):
        """The silent-but-alive case: a worker whose socket is open and whose process is
        wedged. It cannot be seen by EOF, so it is seen by the absence of heartbeats."""
        if self.heartbeat_timeout <= 0:
            return
        cutoff = _now() - self.heartbeat_timeout
        for conn in list(self.conns.values()):
            if conn.last_seen >= cutoff:
                continue
            self.n_reaped += 1
            silent = _now() - conn.last_seen
            self.log(f'[dispatch] REAP {conn.label()}: silent for {silent:.0f}s '
                     f'(> {self.heartbeat_timeout:.0f}s heartbeat timeout)')
            self._drop(conn, 'heartbeat timeout')

    def _readable(self, conn):
        try:
            chunk = conn.sock.recv(65536)
        except socket.timeout:
            return
        except OSError as exc:
            self._drop(conn, f'recv error: {exc}')
            return
        if not chunk:                           # TCP EOF: the process is gone
            self._drop(conn, 'EOF')
            return
        conn.last_seen = _now()
        conn.buf += chunk
        while True:
            try:
                msg, conn.buf = take_frame(conn.buf)
            except ProtocolError as exc:
                self.log(f'[dispatch] protocol error from {conn.label()}: {exc}')
                self._drop(conn, 'protocol error')
                return
            if msg is None:
                return
            try:
                self._handle(conn, msg)
            except OSError as exc:              # the reply could not be sent
                self._drop(conn, f'send error: {exc}')
                return

    # -- protocol ----------------------------------------------------------------
    def _handle(self, conn, msg):
        op = str(msg.get('op', '')).upper()
        if msg.get('worker') is not None:
            conn.worker = msg['worker']
        if op == 'BEAT':
            return                              # last_seen already refreshed
        if op == 'READY':
            return self._on_ready(conn, msg)
        if op == 'DONE':
            return self._on_done(conn, msg)
        if op == 'FAILED':
            return self._on_failed(conn, msg)
        send_msg(conn.sock, {'op': 'REFUSE', 'reason': f'unknown op {op!r}'})

    def _phase_for(self, conn, msg):
        """Find or open the phase this worker is asking about, or refuse it loudly.

        THE DIGEST GATE. Two workers whose request lists disagree are not two workers on
        one benchmark, they are two different benchmarks -- and dispatching blocks
        between them would scatter answers for list A into positions of list B. lm-eval's
        own data-parallel path fails this silently (its rank/world_size hooks are
        base-class no-ops); here it is a refusal with both digests printed.
        """
        name = str(msg.get('phase'))
        nblocks = int(msg.get('nblocks', 0) or 0)
        digest = msg.get('digest')
        phase = self.phases.get(name)
        if phase is None:
            phase = Phase(name, nblocks, digest, max_attempts=self.max_attempts)
            self.phases[name] = phase
            self.journal.write('open', phase=name, nblocks=nblocks, digest=digest)
            self.log(f'[dispatch] phase {name} opened by {conn.label()}: {nblocks} '
                     f'block(s), digest {str(digest)[:12]}')
            return phase
        if phase.digest != digest or phase.nblocks != nblocks:
            self.n_refused += 1
            self.log(f'[dispatch] REFUSED {conn.label()} on phase {name}: it built '
                     f'{nblocks} block(s) at digest {str(digest)[:12]}, the phase is '
                     f'{phase.nblocks} block(s) at digest {str(phase.digest)[:12]}. '
                     f'The workers are not evaluating the same request list; this '
                     f'worker will fall back to computing locally.')
            self.journal.write('refuse', phase=name, worker=conn.worker,
                               digest=digest, nblocks=nblocks)
            return None
        return phase

    def _on_ready(self, conn, msg):
        phase = self._phase_for(conn, msg)
        if phase is None:
            send_msg(conn.sock, {'op': 'REFUSE', 'reason': 'request-list digest '
                                                           'disagrees with the phase'})
            return
        conn.phase = phase.name

        if self.deadline is not None and _now() >= self.deadline:
            # Out of walltime. Stop handing out blocks so the job keeps enough time to
            # re-read the answer file and score; the answers already written are safe.
            self.log(f'[dispatch] walltime margin reached; telling {conn.label()} to '
                     f'stop asking for new blocks on {phase.name}')
            send_msg(conn.sock, {'op': 'STOP', 'phase': phase.name, 'reason': 'walltime'})
            return

        taken = phase.take(conn.key)
        while taken is not None and taken[0] == 'abandon':
            self._log_abandon(phase, taken[1], 'retry bound reached at assign time')
            taken = phase.take(conn.key)

        if taken is not None:
            block = taken[1]
            self._end_wait(conn)
            send_msg(conn.sock, {'op': 'BLOCK', 'phase': phase.name, 'block': block})
            self.journal.write('assign', phase=phase.name, block=block,
                               worker=conn.worker,
                               attempt=phase.attempts.get(block, 1))
            self.log(f'[dispatch] assign phase={phase.name} block={block} -> '
                     f'{conn.label()} (attempt {phase.attempts.get(block, 1)})')
            return

        if phase.done:
            self._end_wait(conn)
            send_msg(conn.sock, {'op': 'STOP', 'phase': phase.name})
            self.log(f'[dispatch] phase {phase.name} complete for {conn.label()}: '
                     f'{phase.summary()}')
            return

        # Nothing pending but the phase is not finished: siblings hold the last blocks.
        # WAIT rather than STOP, because one of those blocks may come back -- that is the
        # whole crash story -- and this worker is the one that would pick it up.
        self._begin_wait(conn)
        send_msg(conn.sock, {'op': 'WAIT', 'secs': self.wait_hint,
                             'inflight': len(phase.inflight)})

    def _on_done(self, conn, msg):
        name = str(msg.get('phase'))
        phase = self.phases.get(name)
        block = int(msg.get('block', -1))
        if phase is None or block < 0:
            return
        fresh = phase.complete(block, conn.key)
        conn.blocks_done += 1
        conn.requests_done += int(msg.get('n', 0) or 0)
        conn.compute_secs += float(msg.get('secs', 0.0) or 0.0)
        if fresh:
            self.journal.write('complete', phase=name, block=block, worker=conn.worker,
                               n=msg.get('n'), secs=msg.get('secs'))
            self.log(f'[dispatch] complete phase={name} block={block} by '
                     f'{conn.label()} n={msg.get("n")} '
                     f'secs={float(msg.get("secs", 0) or 0):.1f} '
                     f'({phase.resolved}/{phase.nblocks})')
        else:
            # A block that was reissued after a false death and then finished twice. Free
            # by construction -- same content key, same value -- and logged so it is
            # never a mystery.
            self.log(f'[dispatch] duplicate complete phase={name} block={block} by '
                     f'{conn.label()} (already done; harmless -- content-keyed answers)')

    def _on_failed(self, conn, msg):
        """The worker survived but the block did not. Requeue under the same retry bound
        as a crash -- a block that fails everywhere is a poison block, not bad luck."""
        name = str(msg.get('phase'))
        phase = self.phases.get(name)
        block = int(msg.get('block', -1))
        if phase is None or block < 0:
            return
        phase.inflight.pop(block, None)
        if block in phase.completed:
            return
        if phase.attempts.get(block, 0) >= phase.max_attempts:
            phase.abandoned.add(block)
            self._log_abandon(phase, block, f'reported failure: {msg.get("error")}')
            return
        phase.pending.appendleft(block)
        self.n_requeued += 1
        self.log(f'[dispatch] REQUEUE phase={name} block={block} from {conn.label()} '
                 f'(worker reported failure: {msg.get("error")})')
        self.journal.write('requeue', phase=name, block=block, worker=conn.worker,
                           reason='failed', error=str(msg.get('error'))[:400])

    def _log_abandon(self, phase, block, why):
        self.log(f'[dispatch] ABANDON phase={phase.name} block={block} after '
                 f'{phase.attempts.get(block, 0)} attempt(s) -- {why}. It is NOT lost: '
                 f'it is simply uncomputed, so it stays a miss in the answer file and '
                 f'the next pass recomputes it.')
        self.journal.write('abandon', phase=phase.name, block=block,
                           attempts=phase.attempts.get(block, 0), reason=str(why)[:400])

    def _begin_wait(self, conn):
        conn.waits += 1
        if conn._wait_since is None:
            conn._wait_since = _now()

    def _end_wait(self, conn):
        if conn._wait_since is not None:
            conn.wait_secs += _now() - conn._wait_since
            conn._wait_since = None

    # -- reporting ---------------------------------------------------------------
    @property
    def all_done(self):
        return bool(self.phases) and all(p.done for p in self.phases.values())

    def report(self):
        """The lines that prove the coordinator dispatched and that no worker idled."""
        lines = ['[dispatch] ==== final report ====']
        total = done = aband = 0
        for name in sorted(self.phases):
            p = self.phases[name]
            total += p.nblocks
            done += len(p.completed)
            aband += len(p.abandoned)
            lines.append(f'[dispatch]   phase {p.summary()}')
            if p.abandoned:
                lines.append(f'[dispatch]     ABANDONED blocks: '
                             f'{sorted(p.abandoned)} -- uncomputed, still misses in the '
                             f'answer file')
        lines.append(f'[dispatch]   blocks: {done}/{total} complete, {aband} abandoned, '
                     f'{self.n_requeued} requeued, {self.n_reaped} worker(s) reaped, '
                     f'{self.n_refused} refused')
        every = list(self.conns.values()) + list(self.gone)
        for conn in sorted(every, key=lambda c: (c.worker is None, c.worker, c.key)):
            lines.append(f'[dispatch]   {conn.label()}: {conn.blocks_done} block(s), '
                         f'{conn.requests_done} request(s), '
                         f'{conn.compute_secs:.1f}s computing, '
                         f'{conn.wait_secs:.1f}s idle waiting for work '
                         f'({conn.waits} wait reply/replies)'
                         + (f' [left: {conn.gone_reason}]'
                            if getattr(conn, 'gone_reason', None) else ''))
        lines.append(f'[dispatch]   wall {_now() - self.started:.1f}s, '
                     f'{self.journal.n_written} journal record(s)')
        lines.append('[dispatch] Completion is defined by the shared answer file, not by '
                     'this queue: anything above that is not complete is simply a cache '
                     'miss for the next pass.')
        return '\n'.join(lines)


# --- batch-shell entry point -----------------------------------------------------------
def slurm_deadline(margin=DISPATCH_MARGIN, env=None):
    """SLURM_JOB_END_TIME minus a margin, or None when Slurm did not say.

    The margin reserves time for the final cache-resync-and-score pass. Every worker's
    answers are already on disk by then; what the margin protects is the SCORE.
    """
    env = os.environ if env is None else env
    raw = env.get('SLURM_JOB_END_TIME')
    try:
        end = float(raw)
    except (TypeError, ValueError):
        return None
    if end <= 0:
        return None
    return end - float(margin)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Block-index dispatch coordinator for eval workers. Runs in the '
                    'batch shell: plain CPU python, no torch, no model, no dataset.')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=DEFAULT_PORT)
    ap.add_argument('--journal', default=None,
                    help='where to WRITE the audit log: an append-only record of every '
                         'open/assign/complete/requeue/abandon/refuse. Write-only -- '
                         'nothing ever reads it back, and a journal already at this path '
                         'does not change what gets dispatched. The answer cache is the '
                         'only authority on what is done.')
    ap.add_argument('--heartbeat-timeout', type=float, default=HEARTBEAT_TIMEOUT,
                    help='declare a worker dead after this many seconds of SILENCE '
                         '(not of block duration -- the heartbeat is independent of the '
                         'payload, which is the reason this is not a process group)')
    ap.add_argument('--max-attempts', type=int, default=MAX_ATTEMPTS,
                    help='assignments per block before it is abandoned, so a poison '
                         'block cannot eat the job one worker at a time')
    ap.add_argument('--margin', type=float, default=DISPATCH_MARGIN,
                    help='stop dispatching this long before SLURM_JOB_END_TIME')
    ap.add_argument('--port-file', default=None,
                    help='write the bound port here once listening, so the batch shell '
                         'can wait for readiness instead of sleeping')
    ap.add_argument('--exit-when-idle', type=float, default=0.0,
                    help='exit this many seconds after the last worker disconnects '
                         '(0 = run until signalled, which is what the batch shell does)')
    args = ap.parse_args(argv)

    srv = DispatchServer(host=args.host, port=args.port, journal_path=args.journal,
                         heartbeat_timeout=args.heartbeat_timeout,
                         max_attempts=args.max_attempts,
                         deadline=slurm_deadline(args.margin))
    if args.port_file:
        try:
            with open(args.port_file, 'w') as fh:
                fh.write(str(srv.port) + '\n')
        except OSError as exc:
            print(f'[dispatch] could not write {args.port_file}: {exc}', flush=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, srv.request_stop)

    last_had_workers = [False]
    idle_since = [None]

    def until(s):
        if not args.exit_when_idle:
            return False
        if s.conns:
            last_had_workers[0] = True
            idle_since[0] = None
            return False
        if not last_had_workers[0]:
            return False
        if idle_since[0] is None:
            idle_since[0] = _now()
        return (_now() - idle_since[0]) >= args.exit_when_idle

    try:
        srv.serve_forever(poll=1.0, until=until)
    finally:
        print(srv.report(), flush=True)
        srv.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
