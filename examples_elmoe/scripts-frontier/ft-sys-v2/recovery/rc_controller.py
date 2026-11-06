"""rc_controller.py (ft-sys-v2 recovery, 2026-11-06): the protocol of a warm recovery at the step boundary (milestone 1).
Framework-free: it drives a trainer adapter (adapters/megatron_train.py) and an optimizer adapter (adapters/deepspeed_zero.py).

  KEEP     during training: rc_keep.Keep (each owner's optimizer state copied on its partner, updated every step)
  STOP     the heartbeat's failure list -> the monitor thread: revoke, etcd barrier, abort (rc_stop); the main thread leaves the
           step at its next communication call (StopForRecovery) and comes here (Recovery.train)
  DECIDE   every survivor reports its versions (rc/<g>/report/<rank> in etcd); every rank computes the same resume step s, or a
           window (an optimizer step half done) -> exit for the restart path
  REBUILD  the same calls on every rank, survivors and the standby: a new world (rc_world), the trainer's own init functions
           again (parallel state, model, optimizer, engine), the state restored (own kept tensors, or the partner's shadow),
           the bf16 weights from the fp32 masters, counters and scheduler, the data iterators at the consumed samples
  RE-SEED  every owner seeds its partner's shadow again (new pairs after a join)
  REDO     Megatron's train() is entered again at iteration s
Extension point (a new recovery plan, for example the OOM ladder with another world size): the coordinator's plan per failure
kind gives a layout (today: the lost ranks and the standby node that takes them); _join() takes the layout's world size and the
rank of this process in it, and calls the same build. A plan that changes the partitions needs a restore that re-splits the
state (put_state with a different layout) instead of copying it.
Times of each recovery: $FTV2_OUT/ft/recovery/<g>/rank_<r>.json (rc_report.py summarises them).
Exit codes: 74 a lost rank and its partner both lost; 75 a failure inside an optimizer step (versions disagree); 76 the list
has no standby (a plan that replaces nothing); 77 the recovery itself failed."""
import gc
import hashlib
import json
import os
import pickle
import sys
import threading
import time

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import rc_pairs
import rc_stop
import rc_world
from rc_keep import Keep
from rc_commit import Commit, decide_microbatch, decide_microbatch_pipe
from adapters.deepspeed_zero import ZeroAdapter

EXIT_NEIGHBOURS, EXIT_WINDOW, EXIT_NO_PLAN, EXIT_FAILED = 74, 75, 76, 77


def enabled():
    return os.environ.get('XMOE_FT_V2') == '1' and os.environ.get('FTV2_RECOVERY') == '1'


def decide(reports, lost, world, stride):
    """(ok, s, why) from the survivors' reports {rank: {'own': [...], 'shadow': [...]}}: every survivor's own versions and the
    shadows of every lost rank must name one step s. Pure: every rank computes it from the same keys."""
    vals = set()
    for r, rep in reports.items():
        vals |= set(rep['own'])
    for o in sorted(lost):
        holder = rc_pairs.partner(o, world, stride)    # under a pipeline: the same stage's next replica (milestone 3)
        if holder in lost:
            return False, None, f'rank {o} and the rank that keeps its copy ({holder}) are both lost', EXIT_NEIGHBOURS
        if holder not in reports:
            return False, None, f'no report from rank {holder}, which keeps the copy of rank {o}', EXIT_FAILED
        vals |= set(reports[holder]['shadow'])
    if len(vals) != 1:
        return False, None, f'an optimizer step was in progress: versions {sorted(vals)}', EXIT_WINDOW
    return True, vals.pop(), 'consistent', 0


def waits_for_stop(exc, stopping):
    """Does an exception in the main thread wait for the recovery's stop (True), or go up at once (False)? It waits only when
    the stop has begun (StopForRecovery, or the flag is up) or when the exception is a communication error, which a peer's
    failure can raise before the failure list arrives. Every other exception (out of memory, a GPU fault raised as an
    exception, an error in the program) goes up at once, so that the heartbeat's exit record and hb_rules.death_kind classify it
    by the user's rules (out of memory -> the OOM plan; a hardware fault -> replace the node; any other exception -> stop for
    debugging). Found in E3 5862190: the wrapper waited 90 s on every exception, the hang detector fired first (37 s), and a
    HIP fault raised as AcceleratorError became a "hang" with a node replaced. The test of "communication error" is the
    heartbeat's own (hb_rules.death_cause: COMM_ERROR, out of memory excluded)."""
    if stopping or type(exc).__name__ == 'StopForRecovery':
        return True
    if isinstance(exc, (SystemExit, KeyboardInterrupt)):
        return False
    import hb_rules
    return hb_rules.death_cause(f'exception: {type(exc).__name__}: {str(exc)[:400]}') == 'consequence'


class Recovery:
    def __init__(self, hb, trainer):
        self.hb, self.tr = hb, trainer
        self.role = os.environ.get('FTV2_ROLE') or self._role_from_task()
        self.base_run = os.environ.get('FTV2_RUN') or os.environ.get('SLURM_JOB_ID', 'run')
        self.out = os.environ.get('FTV2_ARM_OUT') or os.environ.get('FTV2_OUT')
        self.local = int(os.environ.get('SLURM_LOCALID', os.environ.get('LOCAL_RANK', '0')))
        self.device = torch.device('cuda', self.local) if torch.cuda.is_available() else torch.device('cpu')
        self.rank, self.world = None, None
        self.keep, self.adapter = None, None
        self.failure, self.freed = None, threading.Event()
        self.stop_info = {}
        self.T = {}
        self.recovered = 0
        self.step_snap = {}
        self.first_step_wait = None
        self.standby_list = None
        self.standby_event = threading.Event()
        self._etcd = None
        self.commit = None
        self.commit_on = os.environ.get('FTV2_COMMIT') == '1'   # milestone 2: commits inside the step (ZeRO stage 2)
        if os.environ.get('FTV2_DETERMINISTIC') == '1':
            self.deterministic_mode()
        self.pg_timeouts()

    @staticmethod
    def pg_timeouts():
        """Every process group of a run with the recovery gets a timeout of at least FTV2_PG_TIMEOUT_S (default 1800 s), so that
        PyTorch's watchdog cannot end the survivors before the heartbeat's decision and the recovery have run (the stuck rule
        alone takes 3 minutes plus the node checks). Megatron's model-parallel groups and DeepSpeed's groups are made with
        torch.distributed.new_group(ranks) and no timeout, which took torch's 10-minute default: V3-2L 5863006's survivors were
        ended by the watchdog 600 s + 480 s after a stall the heartbeat had seen in 15 s. Wrapped on the module, before
        Megatron's initialization; an explicit longer timeout is kept."""
        import torch.distributed as dist
        from datetime import timedelta
        floor = timedelta(seconds=float(os.environ.get('FTV2_PG_TIMEOUT_S', '1800')))
        if getattr(dist.new_group, '_ftv2_wrapped', False):
            return

        def raised(t):
            return floor if t is None or t < floor else t
        orig_new, orig_init = dist.new_group, dist.init_process_group

        def new_group(*a, **k):
            if len(a) >= 2:
                a = (a[0], raised(a[1])) + tuple(a[2:])
            else:
                k['timeout'] = raised(k.get('timeout'))
            return orig_new(*a, **k)

        def init_process_group(*a, **k):
            k['timeout'] = raised(k.get('timeout'))
            return orig_init(*a, **k)
        new_group._ftv2_wrapped = init_process_group._ftv2_wrapped = True
        dist.new_group = new_group
        dist.init_process_group = init_process_group
        import torch.distributed.distributed_c10d as c10d
        c10d.new_group = new_group

    @staticmethod
    def deterministic_mode():
        """Test M1-D (FTV2_DETERMINISTIC=1): PyTorch's deterministic algorithms where they exist (warn elsewhere), and the
        flash-attention backward in its deterministic mode (its dq accumulation uses atomics otherwise); the RCCL algorithm and
        protocol are fixed by the arm's environment (NCCL_ALGO, NCCL_PROTO). Called before the model is built."""
        import functools
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            import megatron.model.transformer as T
            f = getattr(T, 'flash_attn_varlen_func', None)
            if f is not None and not isinstance(f, functools.partial):
                T.flash_attn_varlen_func = functools.partial(f, deterministic=True)
        except Exception as e:
            print(f'[ftv2 recovery] deterministic flash attention not set: {e!r}'[:200], file=sys.stderr, flush=True)

    # ------------------------------------------------------------------ small helpers
    @staticmethod
    def _role_from_task():
        """One Slurm step holds the training tasks 0 .. FTV2_WORLD-1 and the standby tasks after them (launch/ftv2_srun.sh)."""
        w, t = os.environ.get('FTV2_WORLD'), os.environ.get('SLURM_PROCID')
        return 'standby' if (w and t and int(t) >= int(w)) else 'train'

    def split_mpi_world(self):
        """With the standby in the training's Slurm step (FTV2_ONE_STEP=1), MPI's COMM_WORLD holds both roles; Megatron and
        DeepSpeed read their rank and world size from MPI.COMM_WORLD at call time (megatron/arguments.py:67,
        initialize.py:217, deepspeed/comm/comm.py:724), so the module attribute is replaced by the communicator of this
        process's role. Collective over every task of the step; called once, before any of them reads it."""
        if os.environ.get('FTV2_ONE_STEP') != '1' or getattr(self, '_mpi_split', False):
            return
        from mpi4py import MPI
        world = MPI.COMM_WORLD
        sub = world.Split(color=1 if self.role == 'standby' else 0, key=world.Get_rank())
        MPI.COMM_WORLD = sub
        self._mpi_split = True
        print(f'[ftv2 recovery] MPI world split: task {world.Get_rank()} of {world.Get_size()} -> {self.role} rank '
              f'{sub.Get_rank()} of {sub.Get_size()}', file=sys.stderr, flush=True)

    def log(self, msg):
        print(f'[ftv2 recovery rank {self.rank}] {time.strftime("%H:%M:%S")} {msg}', file=sys.stderr, flush=True)

    def etcd(self):
        if self._etcd is None:
            from hb_etcd import Etcd
            self._etcd = Etcd(os.environ['FTV2_ETCD'].split(','))
        return self._etcd

    def mark(self, g, name):
        self.T.setdefault(g, {})[name] = round(time.time(), 3)

    def write_times(self, g, extra=None):
        if not self.out:
            return
        d = os.path.join(self.out, 'ft', 'recovery', str(g))
        os.makedirs(d, exist_ok=True)
        self.extra = getattr(self, 'extra', {})
        self.extra.setdefault(g, {}).update(extra or {})     # every later write keeps what earlier writes recorded
        rec = {'rank': self.rank, 'role': self.role, 'gen': g, 'times': self.T.get(g, {}), 'stop': self.stop_info.get(g, {})}
        rec.update(self.extra[g])
        with open(os.path.join(d, f'rank_{self.rank}.json'), 'w') as fh:
            json.dump(rec, fh, indent=1)

    # ------------------------------------------------------------------ start
    def on_start(self, rank, world):
        """A training rank, after the first world exists: the keeper communicator; the step hook."""
        self.rank, self.world = int(rank), int(world)
        s = rc_stop.make_keeper(self.device)
        self.hb.S['on_step'] = self._on_step
        self.log(f'recovery on: world {world}, keeper {s} s, shadow {os.environ.get("FTV2_SHADOW", "cpu")}, '
                 f'cpu adam {os.environ.get("FTV2_CPU_ADAM", "ds")}, partner stride {os.environ.get("FTV2_PARTNER_STRIDE", "8")}')

    def standby_init(self, extra_args_provider, args_defaults):
        """A standby process at start, instead of initialize_megatron: the arguments with world size FTV2_WORLD, the device,
        the fused kernels, the keeper communicator and the heartbeat monitor (role standby). No world yet."""
        world = int(os.environ['FTV2_WORLD'])
        stride = int(os.environ.get('FTV2_PARTNER_STRIDE', '8'))
        t0 = time.time()
        self.tr.standby_args(extra_args_provider, args_defaults, world - stride + self.local, world, self.local)
        self.world = world
        self.hb.S['inject'] = []              # test injections are for the training ranks of the first world only
        self.hb.start(1000 + self.local, self.local, role='standby', on_failure=self.on_failure)
        s = rc_stop.make_keeper(self.device)
        torch.ones(1 << 20, device=self.device).sum().item()
        self.log(f'standby ready in {time.time() - t0:.1f} s (keeper {s} s): world {world} when it joins')

    def _ds_fixes(self, adapter):
        """DeepSpeed defects fixed on the object for this run (FTV2_DS_REDUCE_FIX, default 1): the owner check of ZeRO-2's
        reduce path (adapters/deepspeed_zero.py fix_reduce_owner_check; E5c 5862600), when that path is configured. ZeRO-1 reduces
        through the same average_tensor -> allreduce_and_copy path at its boundary (and, under the pipeline with commits, every k
        micro-batches), so the fix applies to stage 1 too (milestone 3)."""
        z = adapter.zero
        if os.environ.get('FTV2_DS_REDUCE_FIX', '1') == '1' and getattr(z, 'reduce_scatter', True) and \
                getattr(z, 'use_multi_rank_bucket_allreduce', True) is False:
            adapter.fix_reduce_owner_check()
            self.log('DeepSpeed fix on the object: ZeRO reduce path, the owner check in the bucket\'s own group '
                     '(stage_1_and_2.py:1548)')

    def after_build(self, model, optimizer, sched, version):
        """The first build: install Keep and seed every partner's shadow (collective)."""
        self.adapter = ZeroAdapter(model[0])
        self._ds_fixes(self.adapter)
        if os.environ.get('FTV2_KEEP', '1') == '0':        # test arms that measure training without Keep
            self.log('Keep off (FTV2_KEEP=0): no shadows; a lost rank cannot be restored')
            return
        self.keep = Keep(self.adapter, self.rank, self.world, self.device, self.log, int(version),
                         out_dir=os.path.join(self.out, 'ft') if self.out else None)
        self.keep.seed()
        self.keep.install()
        self._start_commit(self.adapter, int(version), 0, None)
        stride = int(os.environ.get('FTV2_PARTNER_STRIDE', '8'))
        if self.rank < stride and rc_pairs.pp() == 1:   # the sizes a standby needs to pre-allocate its shadow while it waits
            # (under a pipeline the stages' slices differ and the standby does not know its stage before the list: no pre-allocation)
            try:
                self.etcd().put(f'keep/sizes/{self.rank}', json.dumps([t.numel() for _, t in self.keep.slots]))
            except Exception as e:
                self.log(f'keep sizes not published: {e!r}'[:200])

    def _start_commit(self, adapter, step, J, flats):
        """Milestone 2 on a (re)built engine: its own communicator (collective), the epilogue hook, the copies seeded at (step, J)."""
        if not self.commit_on:
            return
        pipe = adapter.is_pipe()
        if not getattr(adapter.zero, 'partition_gradients', False) and not pipe:
            self.log('FTV2_COMMIT=1 needs ZeRO stage 2 (per-micro-batch reduction) or the pipeline engine: commits off')
            self.commit_on = False
            return
        if pipe and adapter.pipe_tied_groups():
            self.log('FTV2_COMMIT=1 under the pipeline: weights tied across stages are summed after the last backward '
                     '(ReduceTiedGrads), after a spaced reduction would have cleared them: commits off')
            self.commit_on = False
            return
        pg = dist.new_group(list(range(self.world)))
        c = Commit(adapter, self.rank, self.world, self.device, self.log, group=pg, spaced=pipe)
        c.set_position(step, J, flats)
        c.seed(step, J, flats)
        if pipe:                                             # milestone 3: ZeRO-1 reduces every k micro-batches (spaced)
            self.pipe_st = adapter.install_pipe_commits(c.k, self._pipe_start)
            adapter.wrap_epilogue(self._commit_after, last_fn=lambda: not self.pipe_st['spaced'])
        else:
            self.pipe_st = None
            adapter.wrap_epilogue(self._commit_after)
        self.commit = c
        self.log(f'commits on: every {c.k} micro-batch(es), {c.mode}, async {c.asyn}, position step {step} micro-batch {J}'
                 f'{" (pipeline: spaced ZeRO-1 reductions)" if pipe else ""}')

    def _pipe_start(self):
        """The pipeline engine's train_batch asks where its step starts: (J, drop_until) once after a recovery, else (0, 0)."""
        r, self.pipe_resume = getattr(self, 'pipe_resume', None) or (0, 0), None
        return r

    def _commit_after(self, last):
        stopping = bool(self.hb.S.get('stop'))
        st = getattr(self, 'pipe_st', None)
        self.commit.after_epilogue(stopping, last, pos=st['pos'] if st is not None else None)
        self.hb.S['micro_batch'] = self.commit.added           # the heartbeat line's micro-batch (and the tests' injection point)
        if stopping:
            raise self.hb.StopForRecovery('a stop began during a micro-batch epilogue')

    def _on_step(self, iteration):
        """The top of every training iteration (hb_monitor.set_step): counters and RNG as they are at this step's start."""
        if self.commit is not None and self.commit.step != iteration:
            self.commit.begin_step(iteration)
            self.hb.S['micro_batch'] = 0
        try:
            self.step_snap = {'iteration': iteration, 'counters': self.tr.counters(), 'inc': self.tr.increments(),
                              'rng': self.tr.rng_snapshot()}
        except Exception as e:
            self.log(f'step snapshot failed: {e!r}'[:200])
        null_at = os.environ.get('FTV2_NULL_RECOVERY_STEP')
        if null_at and int(null_at) == iteration and not self.recovered and not getattr(self, '_null_sent', 0):
            # test M1-N at a fixed point: every rank waits here, at the top of the iteration (outside any communication call),
            # until its stop is done; its next communication call then raises StopForRecovery
            self._null_sent = 1
            if self.rank == 0:
                self._write_null_list(iteration)
            self.freed.wait(timeout=180)
        if self.first_step_wait is not None and iteration > self.first_step_wait[1]:
            g = self.first_step_wait[0]
            self.mark(g, 'first_step_done')
            self.first_step_wait = None
            self.write_times(g)

    def _write_null_list(self, iteration):
        """Test M1-N: a failure list that loses no rank, so every rank runs stop, rebuild and restore with the same ranks. It goes
        the way a real list goes (etcd failed/<g> -> each node's agent -> failed.json -> each rank's monitor)."""
        g = f'n{iteration:03d}-{os.environ.get("FTV2_ARM", "main")}'
        f = {'gen': g, 'run': os.environ.get('FTV2_RUN', self.base_run), 'kind': 'null recovery (test)', 'node': None,
             'suspect': None, 'ranks': [], 'why': f'FTV2_NULL_RECOVERY_STEP={iteration}', 'plan': 'rebuild with the same ranks',
             't': round(time.time(), 3), 'by': 'rank 0 (test)', 'standby': None, 'replace': True}
        ok = self.etcd().create_if_absent(f'failed/{g}', json.dumps(f))
        self.log(f'null recovery list {g} written: {ok}')

    # ------------------------------------------------------------------ STOP (the monitor thread)
    def on_failure(self, f):
        try:
            self._on_failure(f)
        except BaseException as e:
            self.log(f'stop failed: {e!r}'[:300])
            self.stop_info.setdefault(f.get('gen'), {})['error'] = repr(e)[:300]
            self.freed.set()

    def _on_failure(self, f):
        g = f.get('gen')
        if self.role == 'standby':            # the standby has no world to stop: it only learns the list
            self.standby_list = f
            self.standby_event.set()
            return
        self.freed.clear()
        self.failure = f
        self.mark(g, 'list_written')
        self.T[g]['list_written'] = f.get('t')
        self.mark(g, 'list_seen')
        self.hb.S['stop'] = True
        lost = set(int(r) for r in f.get('ranks', []))
        info = self.stop_info.setdefault(g, {})
        if not f.get('replace'):
            # the plan stops the run (an exception, out of memory, stuck, no standby free): every training rank ends with 76, and
            # the run is marked done so that the standby stops waiting (E3b: without it the standby waited for ever)
            why = f'the plan replaces no node: {f.get("plan")}'
            self.log(f'list {g}: {why}; exiting')
            self.write_times(g, {'exit': why})
            try:
                self.etcd().put(f'run_done/{self.base_run}', json.dumps({'t': round(time.time(), 3), 'stopped': why}))
            except Exception:
                pass
            self._exit(EXIT_NO_PLAN, f'recovery: {why}')
        if self.rank in lost and self.hb.S.get('test_kill_noop'):
            # the RF2 test: this stuck process ignores the list (synthetic: a process inside a stuck driver call cannot act on it)
            self.log(f'list {g}: my node is replaced; test_kill_noop: this process stays (synthetic leftover)')
            return
        if self.rank in lost:                 # before any revoke: this process leaves (its GPU may be the faulty one)
            why = 'my node is replaced'
            self.log(f'list {g}: {why}; exiting')
            self.write_times(g, {'exit': why})
            self._exit(0, f'recovery: {why}')
        if self.device.type == 'cuda':
            # HIP's current device is per thread; _comm_ptr() looks up the communicator of the CURRENT device, and this monitor
            # thread's device was 0, so only GPU 0 of each node found its communicators (E2 5861942: 6 on ranks 0, 8, 16, none on
            # the other survivors, whose stop was the abort alone)
            torch.cuda.set_device(self.device)
        # the waits (seconds): FTV2_REVOKE_WAIT_S (10) and FTV2_ABORT_WAIT_S (30). Under a pipeline the ranks in a point-to-point
        # exchange with the lost node neither finish the revoke nor return from the abort (M3P1 5867020: ranks 24-31 waited
        # 13 s + 30.9 s, the network plugin could not close its receive endpoints, 'Device or resource busy'); the recovery goes on
        # after the limit (the registry is cleared by hand, rc_stop.forget_all) and the new world worked, so the limit is the cost.
        info['revoke'] = rc_stop.revoke_all(self.device, self.log, limit_s=float(os.environ.get('FTV2_REVOKE_WAIT_S', '10')))
        self.mark(g, 'revoke_done')
        survivors = [r for r in range(self.world) if r not in lost]
        info['barrier'] = rc_stop.etcd_barrier(self.etcd(), g, self.rank, survivors, self.log)
        self.mark(g, 'barrier_done')
        info['abort'] = rc_stop.abort_all(self.device, self.log, limit_s=float(os.environ.get('FTV2_ABORT_WAIT_S', '30')))
        self.mark(g, 'abort_done')
        self.freed.set()

    def wait_stop(self, limit_s):
        """After an exception in the main thread: is a stop under way? Waits up to limit_s for a failure list (an RCCL error can
        reach the main thread before the list does), then for the stop to finish."""
        t0 = time.time()
        while not self.hb.S.get('stop') and time.time() - t0 < limit_s:
            time.sleep(0.2)
        if not self.hb.S.get('stop'):
            return False
        self.freed.wait(timeout=120)
        return True

    # ------------------------------------------------------------------ the training loop wrapper
    def train(self, train_fn, box, forward_step_func, process_non_loss_data_func):
        while True:
            try:
                return train_fn(forward_step_func, box[0], box[1], box[2], box[3], box[4], process_non_loss_data_func)
            except BaseException as e:
                name, text = type(e).__name__, str(e)[:240]
                if not waits_for_stop(e, bool(self.hb.S.get('stop'))):
                    if isinstance(e, (SystemExit, KeyboardInterrupt)):
                        raise
                    # the exit record the heartbeat's excepthook would write, then this process waits for the failure list,
                    # which ends it (0 when its node is replaced, 76 when the plan stops the run). It does not exit by itself:
                    # a normal exit hangs in MPI's finalize (collective over the step's first world; the standby tasks never
                    # call it), and a non-zero exit ends every task of the Slurm step on Frontier, the survivors included.
                    import traceback
                    traceback.print_exc()
                    self.hb._write_exit(f'exception: {name}: {str(e)[:200]}', 1)
                    self.log(f'the main thread raised {name}: not a stop and not a communication error: exit record written; '
                             f'waiting for the failure list (the heartbeat classifies it): {text}')
                    limit = float(os.environ.get('FTV2_EXCEPTION_WAIT_S', '600'))
                    t0 = time.time()
                    while time.time() - t0 < limit:
                        time.sleep(1.0)
                    self._exit(1, f'exception: {name}: {str(e)[:200]} (no failure list after {limit:.0f} s)')
                if not self.wait_stop(90):
                    raise
                self.log(f'the main thread left the step: {name}: {text}')
            self.recover(box)

    # ------------------------------------------------------------------ a survivor's recovery
    def recover(self, box):
        f = self.failure
        g = f['gen']
        self.mark(g, 'main_caught')
        lost = sorted(int(r) for r in f['ranks'])
        if self.rank in lost:
            while True:                 # the monitor ends this process
                time.sleep(60)
        self.hb.S['stop'] = False       # the old world is gone; the rebuild's own calls must pass the counter
        if self.keep is None:
            self.log('no Keep in this run: nothing to restore a lost rank from')
            self._exit(EXIT_FAILED, 'exception: recovery impossible: Keep was off')
        ok, drain_s = self.keep.drain(60)
        rep = dict(self.keep.report(), base=self.step_snap.get('counters'), inc=self.step_snap.get('inc'),
                   base_iteration=self.step_snap.get('iteration'), drained=ok,
                   rng_md5=hashlib.md5(pickle.dumps(self.step_snap.get('rng'))).hexdigest() if self.step_snap.get('rng') else None)
        if self.commit is not None:
            rep.update(self.commit.report())
        self.etcd().put(f'rc/{g}/report/{self.rank}', json.dumps(rep))
        self.mark(g, 'reported')
        kept = self.adapter.take_state()
        self.old_commit, self.commit = self.commit, None
        old_keep = self.keep
        old_keep.gpu_recv = [None] * len(old_keep.gpu_recv)
        self.adapter.release()
        self.adapter = self.keep = None
        for k in range(len(box)):
            box[k] = None
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        self.mark(g, 'released')
        self.log(f'released the old engine: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB still allocated '
                 f'(kept state {sum(4 * s["fp32"].numel() * 3 for s in kept) / 1e9:.2f} GB)')
        new = self._join(g, f, 'survivor', kept=kept, old_keep=old_keep)
        box[:] = new

    # ------------------------------------------------------------------ the standby
    def standby_wait(self):
        """A standby process: wait for a failure list that names this node as the standby, or for the end of the run."""
        me = os.environ.get('FTV2_NODE') or __import__('socket').gethostname().split('.')[0]
        self.log(f'standby on {me}: waiting for a failure list that names this node')
        self.prealloc = None
        if os.environ.get('FTV2_STANDBY_WARM') == '1':
            try:
                self.warm_standby(me)
            except Exception as ex:
                self.log(f'standby warm-up failed (the standby still waits, cold): {ex!r}'[:300])
                import traceback
                traceback.print_exc()
        if os.environ.get('FTV2_SHADOW', 'cpu') == 'cpu' and os.environ.get('FTV2_CPU_ADAM', 'ds') == 'ds':
            from rc_keep import _ds_cpu_adam
            t0 = time.time()
            _ds_cpu_adam(True)                 # DeepSpeed's CPU Adam op loaded while waiting (P2c: 4-5 s of the joiner's re-seed)
            self.log(f'standby: DeepSpeed CPU Adam loaded while waiting ({time.time() - t0:.1f} s)')
        while True:
            if self.prealloc is None and os.environ.get('FTV2_SHADOW', 'cpu') == 'cpu':
                self._prealloc_shadow()
            if self.standby_event.wait(timeout=2.0):
                f = self.standby_list
                self.standby_event.clear()
                if f and f.get('standby') == me:
                    return f
                if f and not f.get('replace'):
                    self.log(f'list {f.get("gen")} replaces no node ({f.get("plan")}): the run stops; exiting')
                    self._exit(0, 'finished: the run stopped')
                self.log(f'list {f.get("gen") if f else None} names standby {f.get("standby") if f else None}, not me')
            try:
                v, _ = self.etcd().get(f'run_done/{self.base_run}')
            except Exception:
                v = None
            if v:
                self.log('the training finished without needing this standby: exiting')
                self._exit(0, 'finished')

    def warm_standby(self, node):
        """The lead's option (11-06): the standby's first step after a join was 15 s against 6-7 s (first use of the kernels in a
        fresh process). While it waits, the standby builds the model in a world of its own GPUs (EP 8 fits one node), runs one
        forward and backward of synthetic tokens, frees everything and goes back to waiting; the kernels stay loaded."""
        t0 = time.time()
        n = int(os.environ.get('FTV2_GPUS_PER_NODE', '8'))
        run = os.environ.get('FTV2_RUN', self.base_run)
        rc_world.make_world(self.etcd(), f'sb-{node}', self.local, n, 0, 300, self.log, key=f'standby_world/{run}/{node}',
                            prefix=f'ftv2/standby/{run}/{node}/')
        rc_world.world_barrier(self.device)
        old = self.tr.set_world_args(n, self.local)
        try:
            self.tr.reinit_parallel(ZeroAdapter)
            model, optimizer, sched = self.tr.build_model()
            t1 = time.time()
            self.tr.warm_step(model)
            t2 = time.time()
            ZeroAdapter(model[0]).release()
            del model, optimizer, sched
        finally:
            self.tr.restore_world_args(old)
        gc.collect()
        torch.cuda.empty_cache()
        rc_world.world_barrier(self.device)
        dist.destroy_process_group()
        if dist.is_initialized():
            rc_stop.forget_all()
        self.warm_s = round(time.time() - t0, 1)
        self.log(f'standby warmed in {self.warm_s} s (build {t1 - t0:.1f} s, forward+backward {t2 - t1:.1f} s); '
                 f'{torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB still allocated')

    def _prealloc_shadow(self):
        """The standby's pinned host buffers for the shadow it will keep, made while it waits (the lead, 11-06: the first
        seed's 19 s were mostly pinned allocation); sizes from keep/sizes/<GPU position>, written by node 0 after its seed."""
        try:
            v, _ = self.etcd().get(f'keep/sizes/{self.local}')
        except Exception:
            return
        if not v:
            return
        t0 = time.time()

        class _Buffers:                        # what Keep._alloc_shadow takes over from an earlier Keep ('reuse')
            mode = 'cpu'
        b = _Buffers()
        b.shadow, b.host_grad = [], []
        for n in json.loads(v):
            b.shadow.append(tuple(torch.zeros(n, dtype=torch.float32, pin_memory=True) for _ in range(3)))
            b.host_grad.append(torch.zeros(n, dtype=torch.float32, pin_memory=True))
        self.prealloc = b
        self.prealloc_s = round(time.time() - t0, 3)
        self.log(f'standby: shadow buffers pre-allocated while waiting ({sum(4 * 4 * n for n in json.loads(v)) / 1e9:.2f} GB '
                 f'pinned, {self.prealloc_s} s)')

    def standby_join(self, f):
        g = f['gen']
        self.mark(g, 'list_seen')
        self.T[g]['list_written'] = f.get('t')
        lost = sorted(int(r) for r in f['ranks'])
        self.rank = lost[self.local]
        self.tr.standby_set_rank(self.rank)
        self.hb.S['rank'] = self.rank
        return self._join(g, f, 'joiner', kept=None, old_keep=None)

    # ------------------------------------------------------------------ DECIDE + REBUILD + RE-SEED (every rank, same calls)
    def _join(self, g, f, role, kept, old_keep):
        lost = sorted(int(r) for r in f['ranks'])
        world = int(os.environ.get('FTV2_WORLD', self.world or 0)) or self.world
        self.world = world
        survivors = [r for r in range(world) if r not in lost]
        host = min(survivors)
        stride = int(os.environ.get('FTV2_PARTNER_STRIDE', '8'))
        e = self.etcd()
        t0 = time.time()
        reports = {}
        while time.time() - t0 < 120:
            got, _ = e.range(f'rc/{g}/report/')
            reports = {int(k.rsplit('/', 1)[1]): json.loads(v) for k, v in got.items()}
            if set(survivors) <= set(reports):
                break
            time.sleep(0.1)
        self.mark(g, 'reports_in')
        ok, s, why, code = decide(reports, lost, world, stride) if set(survivors) <= set(reports) else \
            (False, None, f'reports from {sorted(reports)} only', EXIT_FAILED)
        self.log(f'decision for list {g}: {why}; resume at step {s}')
        if not ok:
            self.write_times(g, {'exit': why, 'code': code})
            self._exit(code, f'exception: recovery stopped ({code}): {why}')
        hrep = reports[host]
        base, inc = hrep['base'], hrep['inc']
        if s == hrep['base_iteration']:
            cnt = dict(base)
        elif s == hrep['base_iteration'] + 1:
            cnt = {'iteration': base['iteration'] + 1, 'samples': base['samples'] + inc['samples'],
                   'tokens': base['tokens'] + inc['tokens']}
        else:
            self.write_times(g, {'exit': f'step {s} far from the host report {hrep["base_iteration"]}', 'code': EXIT_FAILED})
            self._exit(EXIT_FAILED, 'exception: recovery failed: resume step far from the report')
        J = 0
        Jown = 0                                             # this rank's stage's commit position (J without a pipeline)
        if self.commit_on:
            if rc_pairs.pp() > 1:                            # milestone 3: each stage commits on its own clock
                okc, J, Js, whyc = decide_microbatch_pipe(reports, lost, world, stride, s)
                Jown = Js[rc_pairs.stage_of(self.rank, world)] if okc else 0
            else:
                okc, J, whyc = decide_microbatch(reports, lost, world, stride, s)
                Jown = J
            self.log(f'commit decision: {whyc}')
            if not okc:
                self.write_times(g, {'exit': whyc, 'code': EXIT_WINDOW})
                self._exit(EXIT_WINDOW, f'exception: recovery stopped (75): {whyc}')
        rng_same = len({r.get('rng_md5') for r in reports.values()}) == 1
        timeout_s = float(os.environ.get('FTV2_WORLD_TIMEOUT_S', '600'))
        done = threading.Event()
        threading.Thread(target=self._join_watch, args=(g, done), daemon=True).start()
        try:
            rc_world.make_world(e, g, self.rank, world, host, timeout_s, self.log)
            rc_world.world_barrier(self.device)
            self.mark(g, 'world_up')
            self.tr.reinit_parallel(ZeroAdapter)
            self.mark(g, 'parallel_state_done')
            model, optimizer, sched = self.tr.build_model()
            self.mark(g, 'model_built')
            adapter = ZeroAdapter(model[0])
            self._ds_fixes(adapter)
            # restore: own kept tensors, or the shadow from the rank that keeps it (point to point)
            sums = None
            if role == 'joiner':
                holder = rc_pairs.partner(self.rank, world, stride)
                states = Keep.recv_state(adapter.slots(), holder, self.device)
                src = f'shadow of rank {holder}'
                sums = {'received': [st['checksum'] for st in states]}
            else:
                for o in lost:
                    if rc_pairs.partner(o, world, stride) == self.rank:
                        sums = {'sent_to': o, 'sent': old_keep.send_shadow(o)}
                states, src = kept, 'own'
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            self.mark(g, 'state_moved')
            adapter.put_state(states)
            flats = None
            if self.commit_on and Jown > 0:                  # milestone 2: the partial sums of the step at micro-batch J
                old_c = getattr(self, 'old_commit', None)
                if role == 'joiner':
                    holder = rc_pairs.partner(self.rank, world, stride)
                    flats = [torch.empty(adapter.zero2_partition_size(i), dtype=adapter.zero2_dtype(), device=self.device)
                             for i in adapter.zero2_groups()]
                    for w in dist.batch_isend_irecv([dist.P2POp(dist.irecv, t, holder) for t in flats]):
                        w.wait()
                else:
                    for o in lost:
                        if rc_pairs.partner(o, world, stride) == self.rank:
                            cp = old_c.copy_of(s, Jown)
                            for w in dist.batch_isend_irecv([dist.P2POp(dist.isend, t, o) for t in cp]):
                                w.wait()
                    flats = old_c.target(Jown)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                for n, i in enumerate(adapter.zero2_groups()):
                    adapter.zero2_set_partial(i, None if flats is None else flats[n])
                if os.environ.get('FTV2_COMMIT_DEBUG') == '1':
                    ns = [round(float(adapter.zero2_flat_partial(i).float().norm()), 6) for i in adapter.zero2_groups()]
                    self.log(f'commit debug step {s} restored at micro-batch {Jown} ({role}): group norms {ns}')
            if os.environ.get('FTV2_STEP_SUMS_FROM') and self.out:
                from rc_keep import state_fingerprint
                sums_, samples_ = state_fingerprint(adapter)        # M1-B: the restored state, to compare with the step-end one
                dd = os.path.join(self.out, 'ft', 'sums')
                os.makedirs(dd, exist_ok=True)
                torch.save({'rank': self.rank, 'tag': f'restored_{g}_v{s}', 'sums': sums_, 'samples': samples_, 'from': src},
                           os.path.join(dd, f'rank_{self.rank}_restored_{g}.pt'))
            adapter.refresh_weights()
            del states, kept
            self.tr.set_counters(cnt['iteration'], cnt['samples'], cnt['tokens'])
            self.tr.set_scheduler(sched, cnt['samples'], max(0, cnt['tokens'] - inc['tokens']))
            adapter.set_counters(cnt['iteration'], cnt['samples'], cnt['iteration'] * inc.get('micro', 1))
            # RNG: a survivor keeps its own state of the step start; a joiner takes the state of the survivor with its GPU position
            # (its partner, rank + stride: the same expert-parallel rank, whose tracker is seeded the same way), not rank 0's
            mine = self.step_snap.get('rng') if (role == 'survivor' and self.step_snap.get('iteration') == s) else None
            snaps = [None] * world
            dist.all_gather_object(snaps, mine)
            if role == 'survivor' and mine is not None:
                self.tr.rng_restore(mine)
                rng_src = 'own'
            else:
                src_rank = rc_pairs.partner(self.rank, world, stride)
                if snaps[src_rank] is None:
                    src_rank = next((r for r in survivors if r % stride == self.rank % stride and snaps[r] is not None
                                     and rc_pairs.stage_of(r, world) == rc_pairs.stage_of(self.rank, world)), host)
                if snaps[src_rank] is not None:
                    self.tr.rng_restore(snaps[src_rank])
                rng_src = f'rank {src_rank}'
            self.mark(g, 'restore_done')
            train_it, valid_it, test_it = self.tr.build_data()
            if J > 0 and train_it is not None:               # milestone 2: the step resumes at micro-batch J
                for _ in range(J):
                    next(train_it)
            if J > 0 and rc_pairs.pp() == 1:
                import megatron.core.pipeline_parallel.schedules as sch
                sch.FTV2_START_MICROBATCH = J
            if rc_pairs.pp() > 1 and (J > 0 or Jown > 0):  # milestone 3: every stage restarts at J, drops J..Jown-1
                self.pipe_resume = (J, Jown)
            self.mark(g, 'data_built')
            keep = Keep(adapter, self.rank, world, self.device, self.log, s,
                        out_dir=os.path.join(self.out, 'ft') if self.out else None,
                        reuse=old_keep if role == 'survivor' else getattr(self, 'prealloc', None))
            reseed_s = keep.seed()
            keep.install()
            self._start_commit(adapter, s, Jown, flats if self.commit_on else None)
            self.mark(g, 'reseed_done')
            rc_world.world_barrier(self.device)
            self.mark(g, 'ready')
        except SystemExit:
            raise
        except BaseException as ex:
            self.log(f'recovery {g} failed: {ex!r}'[:400])
            try:
                e.put(f'rc/{g}/failed/{self.rank}', json.dumps({'t': round(time.time(), 3), 'err': repr(ex)[:300]}))
            except Exception:
                pass
            import traceback
            traceback.print_exc()
            self.write_times(g, {'exit': repr(ex)[:300], 'code': EXIT_FAILED})
            self._exit(EXIT_FAILED, f'exception: recovery failed: {ex!r}'[:200])
        done.set()
        self.adapter, self.keep = adapter, keep
        self.hb.S['on_step'] = self._on_step
        self.recovered += 1
        self.tr.reset_timers()
        self.hb.begin_generation(f'{self.base_run}.g{g}', rank=self.rank, role='train')
        self.role = 'train'
        self.first_step_wait = (g, s)
        mem = torch.cuda.memory_allocated(self.device) / 1e9 if self.device.type == 'cuda' else 0
        self.log(f'recovery {g} ready: resume at step {s} (samples {cnt["samples"]}), state from {src}, '
                 f'rng from {rng_src} (survivors\' snapshots all equal: {rng_same}), re-seed {reseed_s} s, {mem:.2f} GB allocated')
        self.write_times(g, {'resume_step': s, 'resume_microbatch': J, 'state_from': src, 'rng_same': rng_same, 'rng_from': rng_src,
                             'reseed_s': reseed_s, 'checksums': sums,
                             'counters': cnt, 'lost': lost, 'kind': f.get('kind'), 'node': f.get('node'),
                             'standby': f.get('standby')})
        return [model, optimizer, sched, train_it, valid_it]

    def _join_watch(self, g, done):
        """While this rank rebuilds: if another rank's rebuild failed (rc/<g>/failed/), this one is blocked in a collective with
        it for good (async error handling is off); end it, so the arm ends instead of waiting for the wall (P2 5861808)."""
        while not done.wait(2.0):
            try:
                got, _ = self.etcd().range(f'rc/{g}/failed/')
            except Exception:
                continue
            if got:
                self.log(f'recovery {g}: rank(s) {sorted(k.rsplit("/", 1)[1] for k in got)} failed their rebuild; ending')
                self.write_times(g, {'exit': 'another rank failed its rebuild', 'code': EXIT_FAILED})
                self._exit(EXIT_FAILED, 'exception: recovery failed on another rank')

    # ------------------------------------------------------------------ the end
    def final_fingerprint(self):
        """M1-B: each rank's fp32 masters at the end: an md5 of each group's partition, its sum and sum of squares (fp64), and
        every 409th value (about 1M values per rank), in <arm>/ft/final/rank_<r>.pt, so two arms compare exactly (md5) and by
        their largest difference on the sample."""
        if self.adapter is None or not self.out:
            return
        out = []
        for gi, t in self.adapter.slots():
            h = t.detach().float().cpu()
            out.append({'group': gi, 'numel': h.numel(), 'md5': hashlib.md5(h.numpy().tobytes()).hexdigest(),
                        'sum': float(h.double().sum()), 'sumsq': float((h.double() ** 2).sum()), 'sample': h[::409].clone()})
        d = os.path.join(self.out, 'ft', 'final')
        os.makedirs(d, exist_ok=True)
        torch.save({'rank': self.rank, 'iteration': self.tr.counters()['iteration'], 'groups': out},
                   os.path.join(d, f'rank_{self.rank}.pt'))

    def finish(self):
        try:
            self.final_fingerprint()
        except Exception as e:
            self.log(f'final fingerprint failed: {e!r}'[:200])
        if self.rank == 0:
            try:
                self.etcd().put(f'run_done/{self.base_run}', json.dumps({'t': round(time.time(), 3)}))
            except Exception:
                pass
        if self.recovered or os.environ.get('FTV2_ONE_STEP') == '1':
            # MPI's finalize is collective over the step's first world: dead ranks after a recovery, and the standby tasks
            # (which leave with os._exit) in every one-step run, would hold it forever (HB3 5855079 saw such a barrier)
            self.log('training done: exiting without MPI finalize' + (' (after a recovery)' if self.recovered else ''))
            self._exit(0, 'finished')

    def _exit(self, code, reason):
        """os._exit skips atexit, where the heartbeat writes the exit record; without one the node's agent reads a death by a
        signal (P1c 5861771: a normal end after a recovery became a 'killed' failure list). Write it first."""
        try:
            self.hb._write_exit(reason, code)
        except Exception:
            pass
        sys.stderr.flush(); sys.stdout.flush()
        os._exit(code)
