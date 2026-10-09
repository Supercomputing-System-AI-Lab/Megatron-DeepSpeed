"""Request-level work splitting -- Stage 1 of rebuild/DISPATCH_ENGINE_DESIGN.md.

WHAT THIS SOLVES
  workers=N splits a benchmark by DOCUMENT, and lm-eval bakes each document's few-shot
  prompt into its Instance at BUILD time: `Task.build_all_requests` walks the documents
  once (lm_eval/api/task.py:332), draws the few-shot context for each from ONE rng stream
  consumed in build order (:337 -> :487), and freezes the resulting string into the
  Instance (:350). Hand a worker a SUBSET of the documents and it rebuilds every prompt
  differently: measured on gsm8k at 8 shots, a document-split rebuild shares 1 of 20
  requests with the unsliced run. That is why eval_lib refuses workers>1 on gsm8k, mbpp,
  triviaqa, drop and race -- six task-table rows, 1,980 estimated minutes, stuck at one
  node.

  The prompts are already correct one step later. The model is called ONCE, with the whole
  list, at lm_eval/evaluator.py:596 (`resps = getattr(lm, reqtype)(cloned_reqs)`). Every
  worker builds the SAME list there -- simple_evaluate seeds deterministically
  (evaluator.py:196-214,342) and the adapter reports rank=0/world_size=1 on purpose
  (lm_eval_adapter.py:135-155), so build_all_requests runs unsharded on every worker. So a
  worker that builds the full list and computes only ITS OWN SLICE OF THAT LIST computes
  prompts that are bit-identical to an unsliced run. Measured: request-level partition of
  the same build gives 40 of 40, disjoint and complete.

WHY A WRAPPER, AND WHY IT LIVES HERE
  The obvious seam is inside MegatronDeepSpeedLM. It is also fatal: lm_eval_adapter.py is
  FORWARD_SOURCES[4] (result_registry.py:48), so editing it rotates code_fp away from
  4b56868f and silences 23 answer files / 222,361 cached answers. DispatchLM WRAPS the
  adapter instead and is constructed in main.py, which result_registry.py:47-48
  deliberately keeps OUT of FORWARD_SOURCES ("orchestration changes must not invalidate
  response caches"). No FORWARD_SOURCES file is opened by this feature. tests/test_dispatch
  asserts the fingerprint.

THE UNIT, AND WHY IT IS A BLOCK RATHER THAN A STRIDE
  Requests are cut into fixed blocks of B consecutive entries and shard k of K takes blocks
  k, k+K, k+2K, ... B is fixed by the declaration (4 x micro_batch_size, passed as
  --dispatch-block), NOT derived from K, so block b holds the same requests whatever K is.
  That is the reproducibility contract: dispatching a pre-composed immutable block is
  reproducible, COMPOSING a batch dynamically is not (batch composition moves logliks by
  0.66 nats mean, B5_VERDICT.md section 1). B rides on the registry row next to
  nodes/ep/tp/pp, never in the answer filename -- same rule the parallel shape follows
  (cache.py:176-182); an A/B that changes it needs experiment=<tag>.

EP LOCKSTEP
  All 8 ranks of a worker are one srun step and read the SAME --shard-index/--shard-count,
  so `shard_indices()` returns the identical integers on every rank and the adapter is
  called with the identical sublist everywhere. The hit/miss partition is therefore
  identical too, which is the property the answer cache already treats as a correctness
  requirement rather than an optimization (cache.py:244-247): divergent miss sets mean a
  hung MoE all-to-all with no error message. Nothing here is inside a rank conditional.

WHAT A WORKER RETURNS
  A FULL-LENGTH result list, so lm-eval's own bookkeeping (one result per request, in
  order) stays intact and simple_evaluate does not have to be told anything. Positions
  this shard computed carry real values; the rest are answered from the shared answer file
  when they are already there, and otherwise get a PLACEHOLDER. A placeholder is not a
  measurement and the row that would contain it says so: main.py stamps every dispatched
  worker's rows partial='shard k/N', which rows_cover() skips and report.py excludes, so a
  shard can never satisfy a done-check or be printed as the benchmark. The COLLECTOR -- the
  unsliced step at the end of the same job, carrying no shard flags -- is what scores, off
  the shared content-keyed answer file, exactly as it does for a document split today.

STAGE 2 -- PULLING BLOCKS INSTEAD OF COMPUTING A FIXED SET  (--dispatch-addr)
  The static rule above is balanced on a COLD cache and wrong on a warm one. Job 5369888
  balanced documents to max/mean 1.06 and its actual work -- the cache misses -- to
  max/mean 1.54: 35.0% of the compute phase idle, two of seven workers finishing
  instantly. One job in three already resumes against a warm cache. So when the launcher
  renders `dispatch=engine`, each worker is given the coordinator's address and asks for
  blocks one at a time instead of taking blocks k, k+K, k+2K.

  Static mode is unchanged and remains the default. With no --dispatch-addr this file
  executes exactly the code it executed before Stage 2 existed.

  EP LOCKSTEP, RESTATED FOR THE PULL LOOP. Only the leader (RANK 0 of the worker's 8 EP
  ranks) holds the socket. Every iteration, ALL EIGHT ranks call broadcast_object_list on
  the assignment, so all eight enter the same adapter call with the same sublist, or all
  eight wait, or all eight leave the loop. There is no rank-conditional collective
  anywhere in the loop -- that is a hang with no error message (cache.py:244-247 states
  the same rule for the miss set), and the waiting case is broadcast as a WAIT tick
  rather than by parking the peers inside one long broadcast, so no collective is ever
  held open for longer than one poll interval.

  IF THE ENGINE FAILS, THE JOB DEGRADES TO STAGE 1. A lost coordinator, a refused digest
  or an I/O timeout puts the worker in LOCAL mode: it finishes the phase on the static
  k, k+K, k+2K rule, skipping the blocks it already computed. The engine can only make
  the job faster; it can never make it fail.
"""
import hashlib
import math
import os
import threading
import time

# lm-eval requires the object passed to simple_evaluate to be an LM subclass
# (evaluator.py:266), and the answer cache's miss sentinel is a module-level object in
# cache.py, which imports torch. Both are guarded ONLY against ImportError so that this
# file -- whose arithmetic is the thing most worth testing -- imports on a login-node
# python3 with neither torch nor lm_eval, the way tests/ already runs. Inside a job both
# imports succeed and the real objects are used; a test that fakes a cache uses
# dispatch.MISS, so the fake and the real path can never disagree about what a miss is.
try:                                            # the job's interpreter
    from lm_eval.api.model import LM
except ImportError:                             # login-node tests: stdlib only
    LM = object

try:                                            # the job's interpreter
    from .cache import MISS
except ImportError:                             # login-node tests: stdlib only
    MISS = object()

# The wire format lives in dispatch_server.py so there is exactly ONE definition of a
# frame -- a client and a server that each own half a protocol is how a protocol drifts.
# dispatch_server.py is pure stdlib and imports nothing from this package, so the relative
# import works inside the job; the by-path fallback is for the login-node tests, which
# load THIS file by path (no package) precisely to avoid torch.
try:                                            # the job's interpreter
    from . import dispatch_server as _wire
except ImportError:                             # login-node tests: no package context
    import importlib.util as _ilu
    _wire_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'dispatch_server.py')
    _wire_spec = _ilu.spec_from_file_location('_dispatch_wire', _wire_path)
    _wire = _ilu.module_from_spec(_wire_spec)
    _wire_spec.loader.exec_module(_wire)


# The three model-facing methods lm-eval calls. evaluator.py dispatches on the request
# type name, and these are the only three MegatronDeepSpeedLM implements.
METHODS = ('loglikelihood', 'loglikelihood_rolling', 'generate_until')

# A generated string that is unmistakably not a generation. It is what a shard returns for
# a request another shard owns and the answer file does not hold yet.
PLACEHOLDER_TEXT = '<dispatch-shard: not computed by this worker>'

# Default block size when the launcher does not pass one. 4 x the task table's default
# micro_batch_size of 8 (task_table.py:32). The launcher normally passes B explicitly so
# that B is a property of the DECLARATION rather than of whatever the job resolved.
DEFAULT_BLOCK = 32


def placeholder(method):
    """The not-a-measurement value for one method's result slot.

    NaN for the two scoring methods because no legitimate logprob is NaN, so it cannot be
    mistaken for one wherever it does propagate (a rolling-window perplexity built from
    one is NaN, not a number that looks plausible).
    """
    if method == 'generate_until':
        return PLACEHOLDER_TEXT
    if method == 'loglikelihood':
        return (float('nan'), False)
    if method == 'loglikelihood_rolling':
        return float('nan')
    raise ValueError(f'unknown method {method!r}; known: {METHODS}')


def block_count(n_requests, block=DEFAULT_BLOCK):
    """How many blocks a request list of this length is cut into (the last may be short)."""
    if int(block) < 1:
        raise ValueError(f'block must be >= 1, got {block}')
    return int(math.ceil(max(0, int(n_requests)) / float(int(block))))


def shard_blocks(n_requests, shard_index, shard_count, block=DEFAULT_BLOCK):
    """The [lo, hi) ranges shard `shard_index` of `shard_count` owns. Ascending.

    Block b belongs to shard b % shard_count -- a round robin over blocks, so the shards
    interleave and each gets an even mix of the (length-sorted-by-the-adapter-later) list
    rather than one contiguous region. Membership of a block does not depend on
    shard_count, which is what makes a block a stable, quotable unit.
    """
    n = max(0, int(n_requests))
    k, K, B = int(shard_index), int(shard_count), int(block)
    if K < 1:
        raise ValueError(f'shard_count must be >= 1, got {shard_count}')
    if not 0 <= k < K:
        raise ValueError(f'shard_index {k} is not in [0, {K})')
    return [(b * B, min((b + 1) * B, n))
            for b in range(k, block_count(n, B), K)]


def shard_indices(n_requests, shard_index, shard_count, block=DEFAULT_BLOCK):
    """The request positions this shard computes, ascending.

    Over shard_index = 0..shard_count-1 these lists are disjoint and their union is
    range(n_requests), for every n, every shard_count and every block -- including
    shard_count > n (the tail shards get nothing and must not fail) and a ragged final
    block. tests/test_dispatch.py pins that for a wide sweep of shapes.
    """
    out = []
    for lo, hi in shard_blocks(n_requests, shard_index, shard_count, block):
        out.extend(range(lo, hi))
    return out


def cache_lookup(cache, method, args):
    """The answer file's value for one request, or None when it is not there yet.

    Keys must be formed EXACTLY as the adapter forms them or a shard would look up
    something the adapter never wrote: loglikelihood on (context, continuation)
    (lm_eval_adapter.py:186), loglikelihood_rolling on (string,) (:227), generate_until on
    (context, gen_kwargs) with a non-dict second argument normalised to {} (:608-612).
    """
    if cache is None:
        return None
    args = tuple(args)
    if method == 'generate_until':
        gen_kwargs = args[1] if len(args) > 1 and isinstance(args[1], dict) else {}
        _, val = cache.get('generate_until', (args[0], gen_kwargs))
        return None if val is MISS else str(val)
    _, val = cache.get(method, args)
    if val is MISS:
        return None
    if method == 'loglikelihood':
        return (float(val[0]), bool(val[1]))
    return float(val)


# =====================================================================================
# Stage 2 client -- the worker half of the pull protocol
# =====================================================================================
# How long the client waits for one reply before it decides the coordinator is not
# answering. Bounded and SMALL, because the reply is a single integer computed by a
# process with no GPU work to do: if it takes longer than this, the coordinator is wedged
# and the correct move is to stop asking it. A timeout costs a fallback to Stage 1's
# static rule; it never costs an answer.
IO_TIMEOUT = 60.0
CONNECT_TIMEOUT = 30.0

# The tick the leader's daemon thread sends while a block is being computed. It is what
# makes the coordinator's silence timeout independent of how long a block takes -- the
# property no torch process group can give, and the reason a 20-minute generate_until
# block does not force a 20-minute death detector.
HEARTBEAT_INTERVAL = 30.0


def request_digest(method, requests):
    """A hash of the request LIST, so two workers can prove they built the same one.

    The engine ships block INDICES, which is only legal because every worker's list is
    identical -- and that is a claim, so it gets checked rather than assumed. Digests are
    announced at every phase open and a disagreement is REFUSED by the coordinator with
    both digests printed. lm-eval's own data-parallel path has exactly this hazard and
    fails it silently (lm_eval/api/model.py:189-205 are base-class no-ops); here it is a
    visible refusal and a fallback to computing locally.
    """
    h = hashlib.sha256()
    h.update(str(method).encode('utf-8'))
    h.update(b'\x00')
    for r in requests:
        h.update(repr(getattr(r, 'args', r)).encode('utf-8', 'replace'))
        h.update(b'\x00')
    return h.hexdigest()


class DispatchClient:
    """The leader's socket to the coordinator. One thread of control plus a heartbeat.

    Every method returns rather than raises: a control-plane failure must degrade the job
    to Stage 1's static split, never end it. `self.lost` latches on the first failure and
    every later call short-circuits, so a dead coordinator costs one timeout for the whole
    run rather than one per block.
    """

    def __init__(self, addr, worker_id=0, io_timeout=IO_TIMEOUT,
                 connect_timeout=CONNECT_TIMEOUT, heartbeat=HEARTBEAT_INTERVAL,
                 log=print, sleep_fn=time.sleep):
        self.host, self.port = parse_addr(addr)
        self.worker_id = worker_id
        self.io_timeout = float(io_timeout)
        self.connect_timeout = float(connect_timeout)
        self.heartbeat = float(heartbeat)
        self._log = log
        self._sleep = sleep_fn
        self.sock = None
        self.lost = None                    # the reason, once it is lost
        self._send_lock = threading.Lock()
        self._beat = None
        self._beat_stop = threading.Event()

    # -- lifecycle ---------------------------------------------------------------
    def connect(self):
        import socket as _socket
        try:
            self.sock = _socket.create_connection((self.host, self.port),
                                                  timeout=self.connect_timeout)
            self.sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
            try:
                self.sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE, 1)
            except OSError:
                pass
            self.sock.settimeout(self.io_timeout)
        except OSError as exc:
            return self._lose(f'could not connect to {self.host}:{self.port}: {exc}')
        self._start_heartbeat()
        self._log(f'>>> [dispatch] connected to the coordinator at '
                  f'{self.host}:{self.port} as worker {self.worker_id}')
        return True

    def _lose(self, why):
        if self.lost is None:
            self.lost = why
            self._log(f'>>> [dispatch] control plane lost ({why}). Falling back to the '
                      f'STATIC split for the rest of this phase -- the engine can only '
                      f'make the job faster, never make it fail.')
        self.close()
        return False

    def close(self):
        self._beat_stop.set()
        sock, self.sock = self.sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    # -- heartbeat ---------------------------------------------------------------
    def _start_heartbeat(self):
        if self.heartbeat <= 0 or self._beat is not None:
            return

        def _tick():
            while not self._beat_stop.wait(self.heartbeat):
                if self.sock is None:
                    return
                with self._send_lock:
                    sock = self.sock
                    if sock is None:
                        return
                    try:
                        _wire.send_msg(sock, {'op': 'BEAT', 'worker': self.worker_id})
                    except OSError:
                        return          # the main thread will see it and fall back

        self._beat = threading.Thread(target=_tick, name='dispatch-heartbeat',
                                      daemon=True)
        self._beat.start()

    # -- messages ----------------------------------------------------------------
    def _send(self, msg):
        if self.sock is None:
            return False
        with self._send_lock:
            try:
                _wire.send_msg(self.sock, msg)
            except OSError as exc:
                return self._lose(f'send failed: {exc}')
        return True

    def _recv(self):
        if self.sock is None:
            return None
        try:
            msg = _wire.recv_msg(self.sock, timeout=self.io_timeout)
        except (OSError, _wire.ProtocolError) as exc:
            self._lose(f'receive failed: {exc}')
            return None
        if msg is None:
            self._lose('coordinator closed the connection')
            return None
        return msg

    def ready(self, phase, nblocks, digest):
        """Ask for work. -> ('block', b) | ('wait', secs) | ('stop', None) | ('local', why)

        'local' is any control-plane failure AND a refused digest: both mean this worker
        stops taking instructions and finishes on the static rule.
        """
        if self.lost is not None or self.sock is None:
            return ('local', self.lost or 'not connected')
        if not self._send({'op': 'READY', 'worker': self.worker_id, 'phase': phase,
                           'nblocks': int(nblocks), 'digest': digest}):
            return ('local', self.lost)
        msg = self._recv()
        if msg is None:
            return ('local', self.lost)
        op = str(msg.get('op', '')).upper()
        if op == 'BLOCK':
            return ('block', int(msg['block']))
        if op == 'WAIT':
            return ('wait', float(msg.get('secs', 1.0) or 1.0))
        if op == 'STOP':
            return ('stop', None)
        if op == 'REFUSE':
            self._lose(f'coordinator refused this worker: {msg.get("reason")}')
            return ('local', self.lost)
        self._lose(f'unknown reply {op!r}')
        return ('local', self.lost)

    def done(self, phase, block, n, secs):
        return self._send({'op': 'DONE', 'worker': self.worker_id, 'phase': phase,
                           'block': int(block), 'n': int(n), 'secs': round(secs, 3)})

    def failed(self, phase, block, error):
        """Report a block this worker could not compute but survived.

        DispatchLM does not call this: an exception inside a collective forward leaves the
        EP group in an undefined state, so the honest response is to die and let the
        coordinator see the EOF. It exists because the coordinator's retry bound must be
        exercised from both directions, and because a future non-collective failure has
        somewhere to go.
        """
        return self._send({'op': 'FAILED', 'worker': self.worker_id, 'phase': phase,
                           'block': int(block), 'error': str(error)[:400]})


def parse_addr(addr, default_port=None):
    """'host:port' -> (host, int(port)). Accepts a bare host with a default."""
    if default_port is None:
        default_port = _wire.DEFAULT_PORT
    text = str(addr).strip()
    if not text:
        raise ValueError('empty dispatch address')
    if ':' in text:
        host, _, port = text.rpartition(':')
        return (host or '127.0.0.1'), int(port)
    return text, int(default_port)


class DispatchLM(LM):
    """One shard of the canonical request list, wrapped around the real adapter.

    Constructed in main.py when --shard-count > 1. Everything it does not implement is
    delegated to the adapter, so lm-eval sees the model it would have seen.
    """

    def __init__(self, inner, shard_index, shard_count, block=DEFAULT_BLOCK,
                 print_fn=print, client=None, is_leader=True, broadcast_fn=None, gather_fn=None):
        super().__init__()
        # The data-parallel prompt split (2026-09-14): called with (indices, values) of this shard's blocks and returning
        # every shard's, gathered over the job; None (the default) leaves the other shards' positions to the answer file.
        self._gather_fn = gather_fn
        if int(shard_count) < 1:
            raise ValueError(f'shard_count must be >= 1, got {shard_count}')
        if not 0 <= int(shard_index) < int(shard_count):
            raise ValueError(f'shard_index {shard_index} is not in [0, {shard_count})')
        if int(block) < 1:
            raise ValueError(f'dispatch block must be >= 1, got {block}')
        self.inner = inner
        self.shard_index = int(shard_index)
        self.shard_count = int(shard_count)
        self.block = int(block)
        self._print = print_fn
        # Reported at the end of the run and stamped on the registry row: how much of the
        # full list this shard could not fill in from the shared answer file. It is the
        # honest measure of how partial this worker's own results.json is.
        self.n_computed = 0
        self.n_from_cache = 0
        self.n_placeholder = 0

        # -- Stage 2. `client` None (the default) is the static rule and the code path
        # that runs today; nothing below this line is reachable without --dispatch-addr.
        self.client = client
        # RANK 0 of the worker's 8 EP ranks. It is also cache.writer_rank(8, 1), so the
        # rank that receives a block is the rank that persists its answers.
        self.is_leader = bool(is_leader)
        # Called on EVERY rank with the leader's object (None elsewhere) and returning
        # the leader's object everywhere. main.py supplies torch's broadcast_object_list;
        # the default makes a single-rank worker (and every login-node test) work with no
        # torch at all. There is no rank-conditional collective anywhere in the loop.
        self._bcast_fn = broadcast_fn
        self._phase_seq = {}          # method -> how many times it has been called here
        self.n_blocks_engine = 0
        self.n_wait_ticks = 0
        self.engine_local_reason = None

    # -- delegation ----------------------------------------------------------------
    def __getattr__(self, name):
        """Anything not defined here is the adapter's (max_length, tok_encode, ...).

        Only reached when normal lookup fails, so the overrides below still win. Read
        `inner` out of __dict__ (never self.inner) so a lookup made before __init__ has
        finished raises AttributeError rather than recursing or raising KeyError.
        """
        try:
            inner = self.__dict__['inner']
        except KeyError:
            raise AttributeError(name)
        return getattr(inner, name)

    @property
    def rank(self):
        return self.inner.rank

    @property
    def world_size(self):
        return self.inner.world_size

    @property
    def device(self):
        return self.inner.device

    # -- the three model-facing methods --------------------------------------------
    def loglikelihood(self, requests):
        return self._run('loglikelihood', requests)

    def loglikelihood_rolling(self, requests):
        return self._run('loglikelihood_rolling', requests)

    def generate_until(self, requests):
        return self._run('generate_until', requests)

    # -- the split ------------------------------------------------------------------
    def _run(self, method, requests):
        """Compute this shard's slice of `requests`; return a FULL-LENGTH result list."""
        requests = list(requests)
        n = len(requests)
        if self.client is not None:
            return self._run_engine(method, requests)
        mine = shard_indices(n, self.shard_index, self.shard_count, self.block)
        blocks = shard_blocks(n, self.shard_index, self.shard_count, self.block)
        self._print(f'>>> [dispatch] {method}: shard {self.shard_index + 1}/'
                    f'{self.shard_count} takes {len(mine)} of {n} request(s) in '
                    f'{len(blocks)} block(s) of {self.block}')

        # Under the data-parallel split the adapter is called even with nothing of its own: a group without a block still
        # runs the round's empty forwards (the expert all-to-all needs every rank).
        computed = getattr(self.inner, method)([requests[i] for i in mine]) if (mine or self._gather_fn is not None) else []
        computed = list(computed)
        if len(computed) != len(mine):
            # The adapter's contract is one result per request in order. If that ever
            # breaks, the scatter below would silently mis-align answers with prompts.
            raise RuntimeError(
                f'{method}: adapter returned {len(computed)} result(s) for '
                f'{len(mine)} request(s); results would be mis-aligned')

        results = [placeholder(method)] * n
        for i, value in zip(mine, computed):
            results[i] = value
        mine_set = set(mine)
        if self._gather_fn is not None:
            n_gathered = 0
            for item in self._gather_fn((list(mine), computed)):
                if not item:
                    continue
                for i, value in zip(item[0], item[1]):
                    if i not in mine_set:
                        results[i] = value
                        mine_set.add(i)
                        n_gathered += 1
            self._print(f'>>> [dispatch] {method}: {n_gathered} answer(s) gathered from the other data-parallel groups')

        cache = getattr(self.inner, '_cache', None)
        n_cached = 0
        for i in range(n):
            if i in mine_set:
                continue
            value = cache_lookup(cache, method, requests[i].args)
            if value is not None:
                results[i] = value
                n_cached += 1
        n_missing = n - len(mine_set) - n_cached
        self.n_computed += len(mine)
        self.n_from_cache += n_cached
        self.n_placeholder += n_missing
        self._print(f'>>> [dispatch] {method}: {len(mine)} computed here, {len(mine_set) - len(mine)} gathered from the '
                    f'other data-parallel groups, {n_cached} read from the shared answer file, {n_missing} left as '
                    f'placeholders (other shards own them; the collector scores)')
        return results

    # -- Stage 2: the pull loop -----------------------------------------------------
    def _broadcast(self, obj):
        """The leader's assignment, on every rank. Called UNCONDITIONALLY by all 8.

        A rank-conditional collective is a hang with no error message, which is why this
        is one call on one path rather than an `if self.is_leader` around a send. The
        default (no broadcast_fn) is the identity, which is what a single-rank worker and
        every login-node test want.
        """
        if self._bcast_fn is None:
            return obj
        return self._bcast_fn(obj)

    def _run_engine(self, method, requests):
        """Pull blocks from the coordinator until it says stop. Full-length result list.

        The phase name is `<method>:<call index>` and it is identical on every worker,
        because every worker runs the identical task recipe in the identical order. The
        DIGEST is what proves that -- it is announced on every READY and a disagreement is
        refused by the coordinator rather than dispatched, which is the difference between
        this and lm-eval's own data-parallel path (whose rank/world_size hooks are
        base-class no-ops, so a mismatch there is silent).
        """
        n = len(requests)
        B = self.block
        nb = block_count(n, B)
        seq = self._phase_seq.get(method, 0)
        self._phase_seq[method] = seq + 1
        phase = f'{method}:{seq}'
        # Computed on EVERY rank from the same list, so it is identical everywhere and
        # costs no collective. Cheap: a sha256 over the reprs of already-built args.
        digest = request_digest(method, requests)

        self._print(f'>>> [dispatch] {method}: ENGINE mode, phase {phase}, {n} request(s) '
                    f'= {nb} block(s) of {B}; digest {digest[:12]}')

        results = [placeholder(method)] * n
        mine = set()
        blocks = []
        local_reason = None

        while True:
            assignment = None
            if self.is_leader:
                kind, payload = self.client.ready(phase, nb, digest)
                assignment = [kind, payload]
            assignment = self._broadcast(assignment)
            if not assignment:                      # a broadcast that returned nothing
                local_reason = 'no assignment broadcast'
                break
            kind, payload = assignment[0], assignment[1]

            if kind == 'stop':
                break
            if kind == 'local':
                local_reason = payload or 'control plane lost'
                break
            if kind == 'wait':
                # ALL ranks sleep, so the collective is re-entered promptly instead of
                # one long broadcast being held open across another worker's block.
                self.n_wait_ticks += 1
                time.sleep(min(max(float(payload or 1.0), 0.01), 30.0))
                continue

            b = int(payload)
            lo, hi = b * B, min((b + 1) * B, n)
            t0 = time.time()
            chunk = list(getattr(self.inner, method)(requests[lo:hi]))
            if len(chunk) != hi - lo:
                raise RuntimeError(
                    f'{method}: adapter returned {len(chunk)} result(s) for '
                    f'{hi - lo} request(s); results would be mis-aligned')
            for off, value in enumerate(chunk):
                results[lo + off] = value
            mine.update(range(lo, hi))
            blocks.append(b)
            self.n_blocks_engine += 1
            if self.is_leader:
                self.client.done(phase, b, hi - lo, time.time() - t0)

        if local_reason is not None:
            # Degrade to Stage 1 rather than to nothing. This worker finishes the phase on
            # the static k, k+K, k+2K rule, skipping what it already did. Duplicate work
            # against the siblings is free: answers are content-keyed and put_batch skips
            # a key already present (cache.py:294).
            self.engine_local_reason = local_reason
            todo = [b for b in range(self.shard_index, nb, self.shard_count)
                    if b not in set(blocks)]
            self._print(f'>>> [dispatch] {method}: LOCAL fallback ({local_reason}); '
                        f'finishing {len(todo)} block(s) on the static rule')
            for b in todo:
                lo, hi = b * B, min((b + 1) * B, n)
                chunk = list(getattr(self.inner, method)(requests[lo:hi]))
                if len(chunk) != hi - lo:
                    raise RuntimeError(
                        f'{method}: adapter returned {len(chunk)} result(s) for '
                        f'{hi - lo} request(s); results would be mis-aligned')
                for off, value in enumerate(chunk):
                    results[lo + off] = value
                mine.update(range(lo, hi))
                blocks.append(b)

        n_cached = self._fill_from_cache(method, requests, results, mine)
        n_missing = n - len(mine) - n_cached
        self.n_computed += len(mine)
        self.n_from_cache += n_cached
        self.n_placeholder += n_missing
        self._print(f'>>> [dispatch] {method}: phase {phase} done -- {len(blocks)} '
                    f'block(s) {sorted(blocks)[:12]}{"..." if len(blocks) > 12 else ""} '
                    f'= {len(mine)} computed here, {n_cached} read from the shared '
                    f'answer file, {n_missing} left as placeholders')
        return results

    def _fill_from_cache(self, method, requests, results, mine):
        """Answer the positions other workers own from the shared answer file.

        Identical in intent to the static path's tail: the engine changes WHICH positions
        this worker computes, never what it does with the rest. Anything still missing
        stays a placeholder, the row is stamped partial, and the collector scores.
        """
        cache = getattr(self.inner, '_cache', None)
        n_cached = 0
        for i in range(len(requests)):
            if i in mine:
                continue
            value = cache_lookup(cache, method, requests[i].args)
            if value is not None:
                results[i] = value
                n_cached += 1
        return n_cached

    def summary(self):
        """One line for the job log, and the numbers main.py puts on the registry row."""
        mode = 'engine' if self.client is not None else 'static'
        extra = ''
        if self.client is not None:
            extra = (f', {self.n_blocks_engine} block(s) pulled, '
                     f'{self.n_wait_ticks} wait tick(s)')
            if self.engine_local_reason:
                extra += f', LOCAL fallback: {self.engine_local_reason}'
        return (f'shard {self.shard_index + 1}/{self.shard_count} block={self.block} '
                f'mode={mode}: {self.n_computed} computed, {self.n_from_cache} from the '
                f'answer file, {self.n_placeholder} placeholder(s){extra}')
