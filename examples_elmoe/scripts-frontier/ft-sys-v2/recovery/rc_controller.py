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
import rc_stop
import rc_world
from rc_keep import Keep
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
        holder = (o + stride) % world
        if holder in lost:
            return False, None, f'rank {o} and the rank that keeps its copy ({holder}) are both lost', EXIT_NEIGHBOURS
        if holder not in reports:
            return False, None, f'no report from rank {holder}, which keeps the copy of rank {o}', EXIT_FAILED
        vals |= set(reports[holder]['shadow'])
    if len(vals) != 1:
        return False, None, f'an optimizer step was in progress: versions {sorted(vals)}', EXIT_WINDOW
    return True, vals.pop(), 'consistent', 0


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

    def after_build(self, model, optimizer, sched, version):
        """The first build: install Keep and seed every partner's shadow (collective)."""
        self.adapter = ZeroAdapter(model[0])
        if os.environ.get('FTV2_KEEP', '1') == '0':        # test arms that measure training without Keep
            self.log('Keep off (FTV2_KEEP=0): no shadows; a lost rank cannot be restored')
            return
        self.keep = Keep(self.adapter, self.rank, self.world, self.device, self.log, int(version),
                         out_dir=os.path.join(self.out, 'ft') if self.out else None)
        self.keep.seed()
        self.keep.install()
        stride = int(os.environ.get('FTV2_PARTNER_STRIDE', '8'))
        if self.rank < stride:                 # the sizes a standby needs to pre-allocate its shadow while it waits
            try:
                self.etcd().put(f'keep/sizes/{self.rank}', json.dumps([t.numel() for _, t in self.keep.slots]))
            except Exception as e:
                self.log(f'keep sizes not published: {e!r}'[:200])

    def _on_step(self, iteration):
        """The top of every training iteration (hb_monitor.set_step): counters and RNG as they are at this step's start."""
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
        info['revoke'] = rc_stop.revoke_all(self.device, self.log)
        self.mark(g, 'revoke_done')
        if self.rank in lost or not f.get('replace'):
            why = 'my node is replaced' if self.rank in lost else f'the plan replaces no node: {f.get("plan")}'
            self.log(f'list {g}: {why}; exiting')
            self.write_times(g, {'exit': why})
            self._exit(0 if self.rank in lost else EXIT_NO_PLAN, f'recovery: {why}' if self.rank in lost else 'finished')
        survivors = [r for r in range(self.world) if r not in lost]
        info['barrier'] = rc_stop.etcd_barrier(self.etcd(), g, self.rank, survivors, self.log)
        self.mark(g, 'barrier_done')
        info['abort'] = rc_stop.abort_all(self.device, self.log)
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
                if isinstance(e, (SystemExit, KeyboardInterrupt)) and not self.hb.S.get('stop'):
                    raise
                name, text = type(e).__name__, str(e)[:240]
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
        self.etcd().put(f'rc/{g}/report/{self.rank}', json.dumps(rep))
        self.mark(g, 'reported')
        kept = self.adapter.take_state()
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
                self.log(f'list {f.get("gen") if f else None} names standby {f.get("standby") if f else None}, not me')
            try:
                v, _ = self.etcd().get(f'run_done/{self.base_run}')
            except Exception:
                v = None
            if v:
                self.log('the training finished without needing this standby: exiting')
                self._exit(0, 'finished')

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
            # restore: own kept tensors, or the shadow from the rank that keeps it (point to point)
            sums = None
            if role == 'joiner':
                holder = (self.rank + stride) % world
                states = Keep.recv_state(adapter.slots(), holder, self.device)
                src = f'shadow of rank {holder}'
                sums = {'received': [st['checksum'] for st in states]}
            else:
                for o in lost:
                    if (o + stride) % world == self.rank:
                        sums = {'sent_to': o, 'sent': old_keep.send_shadow(o)}
                states, src = kept, 'own'
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            self.mark(g, 'state_moved')
            adapter.put_state(states)
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
            snap = [self.step_snap.get('rng') if (role == 'survivor' and self.step_snap.get('iteration') == s) else None]
            dist.broadcast_object_list(snap, src=host)
            if role == 'survivor' and self.step_snap.get('iteration') == s and self.step_snap.get('rng'):
                self.tr.rng_restore(self.step_snap['rng'])
            elif snap[0] is not None:
                self.tr.rng_restore(snap[0])
            self.mark(g, 'restore_done')
            train_it, valid_it, test_it = self.tr.build_data()
            self.mark(g, 'data_built')
            keep = Keep(adapter, self.rank, world, self.device, self.log, s,
                        out_dir=os.path.join(self.out, 'ft') if self.out else None,
                        reuse=old_keep if role == 'survivor' else getattr(self, 'prealloc', None))
            reseed_s = keep.seed()
            keep.install()
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
                 f'rng snapshots equal on every survivor: {rng_same}, re-seed {reseed_s} s, {mem:.2f} GB allocated')
        self.write_times(g, {'resume_step': s, 'state_from': src, 'rng_same': rng_same, 'reseed_s': reseed_s, 'checksums': sums,
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
