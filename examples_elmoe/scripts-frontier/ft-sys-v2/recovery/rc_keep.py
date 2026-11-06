"""rc_keep.py (ft-sys-v2 recovery, 2026-11-06): Keep, the partner's copy of each owner's optimizer state. Framework-free: it sees
only the inner optimizer's parameters, gradients, moments and hyperparameters, through the adapter (adapters/*.py).

Pairs. Rank r keeps the copy (the "shadow") of owner r - S (mod W) and its own copy lives on r + S, S = FTV2_PARTNER_STRIDE
(default 8, the GPUs of one node), so a node's state is always also on the next node.
Every step. Right before the inner optimizer updates a group (its gradient is final there: reduced, unscaled, clipped), the owner
sends that gradient and the update's hyperparameters to its partner and receives its own owner's (one batched send/receive);
then the real update runs. The partner applies the same Adam update to its shadow:
  FTV2_SHADOW=cpu (default)  shadow in pinned host memory; the gradient goes to the host on a side stream; a worker thread runs
                             CPU Adam, FTV2_CPU_ADAM=ds (DeepSpeed's CPU Adam kernel, called in chunks of FTV2_CPU_ADAM_CHUNK
                             elements because it holds the Python lock while it runs) or torch (torch.optim.AdamW(fused=True) on
                             the CPU, which releases the lock).
  FTV2_SHADOW=gpu            shadow on the GPU, updated by an optimizer of the owner's own class (apex FusedAdam): the same
                             kernel on the same inputs, so the shadow equals the owner bit for bit (the check mode).
Seeding. After every build (the start, and every recovery) each owner sends its full state (fp32 master, both moments, step) to
its partner: one code path for the start and for new pairs after a join, and it removes any CPU-Adam rounding drift.
Versions. own[i] and shadow[i] count the updates applied to group i (absolute: the step count). After a stop, every survivor
reports them; the recovery resumes only when they agree (rc_controller.decide).
Counters per step (one JSON line per step in $FTV2_OUT/ft/keep/rank_<r>.jsonl): send/receive ms, the owner's inner step ms,
device-to-host ms, the shadow update ms (CPU wall or GPU), the main thread's wait for the worker ms.
FTV2_SHADOW_CHECK=1: every step the owner also sends its fp32 master (as it was before this update) and the partner records the
largest difference to its shadow of the same version (M1-S)."""
import json
import os
import queue
import threading
import time

import torch
import torch.distributed as dist

MB = 1 << 20
_DS_CPU_ADAM = {}


def _ds_cpu_adam(adamw):
    """DeepSpeed's CPU Adam kernel object, made once per process and mode (its construction loads the op, about 1.4 s)."""
    if adamw not in _DS_CPU_ADAM:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        _DS_CPU_ADAM[adamw] = DeepSpeedCPUAdam([torch.zeros(1)], adamw_mode=adamw)
    return _DS_CPU_ADAM[adamw]


def _env(name, default):
    return os.environ.get(name, default)


class Keep:
    def __init__(self, adapter, rank, world, device, log, version, out_dir=None, reuse=None):
        self.ad, self.rank, self.world, self.device, self.log = adapter, rank, world, device, log
        self.stride = int(_env('FTV2_PARTNER_STRIDE', '8'))
        self.partner = (rank + self.stride) % world        # keeps my copy
        self.owner = (rank - self.stride) % world          # I keep its copy
        self.mode = _env('FTV2_SHADOW', 'cpu')
        self.cpu_impl = _env('FTV2_CPU_ADAM', 'ds')
        self.chunk = int(_env('FTV2_CPU_ADAM_CHUNK', str(8 * MB)))
        self.check = _env('FTV2_SHADOW_CHECK', '0') == '1'
        self.slots = adapter.slots()                        # [(group index, tensor)]
        self.index = {id(t): k for k, (_, t) in enumerate(self.slots)}
        n = len(self.slots)
        self.own = [version] * n
        self.shadow_v = [version] * n
        self.shadow_step = [0] * n         # the shadow optimizer's step count (the owner's inner-optimizer count it mirrors)
        self.shadow = [None] * n           # (p, m, v) on the host (pinned) or the GPU
        self.host_grad = [None] * n
        self.gpu_recv = [None] * n
        self.hyper_r = [None] * n          # the owner's hyperparameters of the update, received with its gradient
        self.host_hyper = [None] * n
        self.hy_host = [None] * n          # pinned, the hyperparameters this rank sends (no pinned allocation during a step)
        self.hy_gpu = [None] * n
        self.d2h_done = [None] * n
        self.idle = [threading.Event() for _ in range(n)]
        [e.set() for e in self.idle]
        self.cpu_opt = [None] * n
        self.gpu_opt = [None] * n
        self.pending = {}                  # step record being filled: events to resolve later
        self.records = []
        self.out = None
        if out_dir:
            os.makedirs(os.path.join(out_dir, 'keep'), exist_ok=True)
            self.out = os.path.join(out_dir, 'keep', f'rank_{rank}.jsonl')
        self.q = queue.Queue()
        self.worker = None
        self.installed = False
        self.side = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
        self._reuse = reuse
        self.cur_step = None

    # ------------------------------------------------------------------ point-to-point helpers
    def _exchange(self, send, recv, send_to=None, recv_from=None):
        """One batched send/receive round: send[k] -> send_to, recv[k] <- recv_from (lists may be empty)."""
        ops = [dist.P2POp(dist.isend, t, self.partner if send_to is None else send_to) for t in send]
        ops += [dist.P2POp(dist.irecv, t, self.owner if recv_from is None else recv_from) for t in recv]
        if not ops:
            return
        for w in dist.batch_isend_irecv(ops):
            w.wait()

    # ------------------------------------------------------------------ seeding (after every build)
    def _alloc_shadow(self, k, numel):
        if self.mode == 'gpu':
            if self.shadow[k] is None or self.shadow[k][0].numel() != numel:
                self.shadow[k] = tuple(torch.zeros(numel, dtype=torch.float32, device=self.device) for _ in range(3))
            return
        old = self._reuse.shadow[k] if (self._reuse is not None and self._reuse.mode == 'cpu' and k < len(self._reuse.shadow)) else None
        if old is not None and old[0].numel() == numel:
            self.shadow[k] = old
            self.host_grad[k] = self._reuse.host_grad[k]
        else:
            pin = self.device.type == 'cuda'
            self.shadow[k] = tuple(torch.zeros(numel, dtype=torch.float32, pin_memory=pin) for _ in range(3))
            self.host_grad[k] = torch.zeros(numel, dtype=torch.float32, pin_memory=pin)
        self.host_hyper[k] = torch.zeros(8, dtype=torch.float64, pin_memory=self.device.type == 'cuda')

    def seed(self):
        """Every owner sends its full state to its partner; every partner stores its owner's as the shadow. Collective over the
        world (every rank sends once and receives once). Returns seconds."""
        t0 = time.time()
        sizes = torch.tensor([t.numel() for _, t in self.slots], dtype=torch.int64, device=self.device)
        theirs = torch.empty_like(sizes)
        self._exchange([sizes], [theirs])
        if not torch.equal(sizes, theirs):
            raise RuntimeError(f'Keep: my partition sizes {sizes.tolist()} differ from my owner {self.owner}: {theirs.tolist()}')
        # every buffer first (pinned host memory, GPU receive buffers), with no send or receive in flight: a pinned allocation
        # between batched sends hung half of the rings in P1b (5861716, ranks blocked in the copy after their exchange)
        t_alloc = time.time()
        for k, (gi, p) in enumerate(self.slots):
            self._alloc_shadow(k, p.numel())
            self.gpu_recv[k] = torch.empty(p.numel(), dtype=torch.float32, device=self.device)
            self.hyper_r[k] = torch.zeros(8, dtype=torch.float64, device=self.device)
            self.hy_host[k] = torch.zeros(8, dtype=torch.float64, pin_memory=self.device.type == 'cuda')
            self.hy_gpu[k] = torch.zeros(8, dtype=torch.float64, device=self.device)
        self._sync()
        t_alloc = time.time() - t_alloc
        for k, (gi, p) in enumerate(self.slots):
            m, v = self.ad.moments(p)
            m = m if m is not None else torch.zeros_like(p, dtype=torch.float32)
            v = v if v is not None else torch.zeros_like(p, dtype=torch.float32)
            step = torch.tensor([float(self.ad.hyper(gi)['step']), float(self.own[k])], dtype=torch.float64, device=self.device)
            rp, rm, rv = (torch.empty(p.numel(), dtype=torch.float32, device=self.device) for _ in range(3))
            rstep = torch.empty(2, dtype=torch.float64, device=self.device)
            self._exchange([p.detach().float(), m, v, step], [rp, rm, rv, rstep])
            self._sync()
            for dst, src in zip(self.shadow[k], (rp, rm, rv)):
                dst.copy_(src, non_blocking=False)
            st = rstep.tolist()
            self.shadow_step[k], self.shadow_v[k] = int(st[0]), int(st[1])
            del rp, rm, rv
        self._sync()
        self._reuse = None
        self._make_optimizers()
        if self.mode == 'cpu' and self.worker is None:
            self.worker = threading.Thread(target=self._work, name='ftv2-keep', daemon=True)
            self.worker.start()
        s = round(time.time() - t0, 3)
        nbytes = sum(3 * 4 * t.numel() for _, t in self.slots)
        self.log(f'keep seeded: {len(self.slots)} groups, {nbytes / 1e9:.2f} GB per rank to rank {self.partner}, '
                 f'shadow of rank {self.owner} ({self.mode}, {self.cpu_impl if self.mode == "cpu" else "owner class"}), {s} s '
                 f'(buffers {t_alloc:.2f} s)')
        return s

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _make_optimizers(self):
        for k, (gi, p) in enumerate(self.slots):
            sp, sm, sv = self.shadow[k]
            h = self.ad.hyper(gi)
            if self.mode == 'gpu':
                opt = self.ad.gpu_shadow_optimizer([sp])
                opt.state[sp]['exp_avg'] = sm
                opt.state[sp]['exp_avg_sq'] = sv
                self.gpu_opt[k] = opt
            elif self.cpu_impl == 'torch':
                opt = torch.optim.AdamW([sp], lr=h['lr'], betas=(h['beta1'], h['beta2']), eps=h['eps'],
                                        weight_decay=h['weight_decay'], fused=True)
                opt.state[sp]['step'] = torch.tensor(0.0)
                opt.state[sp]['exp_avg'] = sm
                opt.state[sp]['exp_avg_sq'] = sv
                self.cpu_opt[k] = opt
            else:
                self.cpu_opt[k] = _ds_cpu_adam(h['adamw'])     # one DeepSpeed CPU Adam per process: every call passes its own
                                                               # step and hyperparameters (P1c: one per group cost 14 s)

    # ------------------------------------------------------------------ the inner step wrapper (installed by the adapter)
    def install(self):
        if not self.installed:
            self.ad.wrap_inner_step(self._before, self._after)
            self.installed = True

    def _ev(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def _before(self, params):
        """Called with the tensors the inner optimizer is about to update (ZeRO-1/2: one group per call)."""
        cuda = self.device.type == 'cuda'
        for p in params:
            k = self.index.get(id(p))
            if k is None or p.grad is None:
                continue
            gi = self.slots[k][0]
            h = self.ad.hyper(gi)
            v_next = self.own[k] + 1
            rec = self.pending.setdefault(v_next, {'step': v_next, 'groups': {}})
            g = rec['groups'].setdefault(k, {})
            t_wait = time.time()
            if self.mode == 'cpu':
                self.idle[k].wait()                 # the worker finished my previous update of this group (host buffers free)
            g['wait_ms'] = round(1000 * (time.time() - t_wait), 2)
            if cuda and self.d2h_done[k] is not None:
                torch.cuda.current_stream().wait_event(self.d2h_done[k])    # the previous copy out of gpu_recv[k] is done
            self.hy_host[k].copy_(torch.tensor([h['lr'], h['weight_decay'], h['beta1'], h['beta2'], h['eps'],
                                                float(h['bias_correction']), float(h['step'] + 1), float(v_next)], dtype=torch.float64))
            hy = self.hy_gpu[k]
            hy.copy_(self.hy_host[k], non_blocking=cuda)          # pinned buffer made at seeding: no host wait, no allocation
            send, recv = [p.grad.detach(), hy], [self.gpu_recv[k], self.hyper_r[k]]
            chk = None
            if self.check:
                chk = torch.empty(p.numel(), dtype=torch.float32, device=self.device)
                send.append(p.detach())
                recv.append(chk)
            e0 = self._ev() if cuda else None
            self._exchange(send, recv)
            e1 = self._ev() if cuda else None
            g['ev_p2p'] = (e0, e1)
            if chk is not None:
                g['check'] = self._compare(k, chk)
            if self.mode == 'gpu':
                e2 = self._ev()
                self._gpu_update(k, h)
                g['ev_shadow'] = (e2, self._ev())
                self.shadow_v[k], self.shadow_step[k] = v_next, h['step'] + 1
            else:
                self.idle[k].clear()
                if cuda:
                    main = torch.cuda.current_stream()          # the stream that waits for the receive (w.wait() above)
                    with torch.cuda.stream(self.side):
                        self.side.wait_stream(main)
                        d0 = torch.cuda.Event(enable_timing=True); d0.record(self.side)
                        self.host_grad[k].copy_(self.gpu_recv[k], non_blocking=True)
                        self.host_hyper[k].copy_(self.hyper_r[k], non_blocking=True)
                        d1 = torch.cuda.Event(enable_timing=True); d1.record(self.side)
                    self.d2h_done[k] = d1
                    g['ev_d2h'] = (d0, d1)
                else:
                    self.host_grad[k].copy_(self.gpu_recv[k])
                    self.host_hyper[k].copy_(self.hyper_r[k])
                self.q.put((k, v_next, self.d2h_done[k] if cuda else None, g))
            g['e_own0'] = self._ev() if cuda else time.time()

    def _after(self, params):
        cuda = self.device.type == 'cuda'
        for p in params:
            k = self.index.get(id(p))
            if k is None:
                continue
            self.own[k] += 1
            rec = self.pending.get(self.own[k])
            if rec is not None and k in rec['groups']:
                rec['groups'][k]['e_own1'] = self._ev() if cuda else time.time()
        self._flush_records()

    def _gpu_update(self, k, h):
        opt = self.gpu_opt[k]
        sp = self.shadow[k][0]
        sp.grad = self.gpu_recv[k]
        grp = opt.param_groups[0]
        grp['lr'], grp['weight_decay'], grp['eps'] = h['lr'], h['weight_decay'], h['eps']
        grp['betas'] = (h['beta1'], h['beta2'])
        self.ad.set_step(opt, grp, sp, h['step'])        # the step count before this update; the optimizer adds one
        opt.step()
        sp.grad = None

    def _compare(self, k, owner_fp32):
        """Largest |shadow - owner| for this group, both at the version before this update (check mode only: syncs)."""
        if self.mode == 'cpu':
            self.idle[k].wait()
            sp = self.shadow[k][0].to(self.device, non_blocking=False)
        else:
            sp = self.shadow[k][0]
        d = (sp - owner_fp32).abs()
        return {'max_abs': float(d.max()), 'max_owner': float(owner_fp32.abs().max()),
                'n_diff': int((d > 0).sum()), 'numel': int(d.numel()), 'version': self.shadow_v[k]}

    # ------------------------------------------------------------------ the CPU worker
    def _work(self):
        while True:
            k, v_next, ev, g = self.q.get()
            try:
                t0 = time.time()
                if ev is not None:
                    ev.synchronize()
                t1 = time.time()
                hy = self.host_hyper[k].tolist()
                lr, wd, b1, b2, eps, bc, step, ver = hy
                sp, sm, sv = self.shadow[k]
                grad = self.host_grad[k]
                if self.cpu_impl == 'torch':
                    opt = self.cpu_opt[k]
                    grp = opt.param_groups[0]
                    grp['lr'], grp['weight_decay'], grp['eps'], grp['betas'] = lr, wd, eps, (b1, b2)
                    opt.state[sp]['step'].fill_(step - 1)
                    sp.grad = grad
                    opt.step()
                    sp.grad = None
                else:
                    o = self.cpu_opt[k]
                    n = sp.numel()
                    for a in range(0, n, self.chunk):
                        b = min(n, a + self.chunk)
                        o.ds_opt_adam.adam_update(o.opt_id, int(step), lr, b1, b2, eps, wd, bool(bc),
                                                  sp[a:b], grad[a:b], sm[a:b], sv[a:b])
                self.shadow_v[k], self.shadow_step[k] = int(ver), int(step)
                g['d2h_wait_ms'] = round(1000 * (t1 - t0), 2)
                g['cpu_adam_ms'] = round(1000 * (time.time() - t1), 2)
            except Exception as e:
                self.log(f'keep worker error on group {k}: {e!r}'[:300])
                g['error'] = repr(e)[:200]
            finally:
                self.idle[k].set()

    def drain(self, limit_s=60.0):
        """Wait until the worker has applied every update it was given (after a stop: before the versions are reported)."""
        t0 = time.time()
        for e in self.idle:
            e.wait(timeout=max(0.0, limit_s - (time.time() - t0)))
        return all(e.is_set() for e in self.idle), round(time.time() - t0, 3)

    # ------------------------------------------------------------------ restore helpers (a joiner's state from its shadow)
    def send_shadow(self, to_rank):
        """Send the shadow I keep (of my owner) to `to_rank`, group by group: fp32, exp_avg, exp_avg_sq, [step, version]."""
        for k, (gi, p) in enumerate(self.slots):
            parts = [t.to(self.device, non_blocking=False) if t.device != self.device else t for t in self.shadow[k]]
            meta = torch.tensor([float(self.shadow_step[k]), float(self.shadow_v[k])], dtype=torch.float64, device=self.device)
            self._exchange(parts + [meta], [], send_to=to_rank)
            self._sync()
            del parts

    @staticmethod
    def recv_state(slots, from_rank, device):
        """A joiner receives its state from the rank that keeps its shadow: per group (fp32, exp_avg, exp_avg_sq, step)."""
        out = []
        for gi, p in slots:
            parts = [torch.empty(p.numel(), dtype=torch.float32, device=device) for _ in range(3)]
            meta = torch.empty(2, dtype=torch.float64, device=device)
            ops = [dist.P2POp(dist.irecv, t, from_rank) for t in parts + [meta]]
            for w in dist.batch_isend_irecv(ops):
                w.wait()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            st, ver = meta.tolist()
            out.append({'fp32': parts[0], 'exp_avg': parts[1], 'exp_avg_sq': parts[2], 'step': int(st), 'version': int(ver)})
        return out

    # ------------------------------------------------------------------ records
    def _flush_records(self):
        """Resolve the timing events of finished steps (they are long done when the next step's first update arrives)."""
        done = []
        for v, rec in sorted(self.pending.items()):
            gs = rec['groups']
            if len(gs) < len(self.slots) or any('e_own1' not in g for g in gs.values()):
                continue
            if self.mode == 'cpu' and any('cpu_adam_ms' not in g and 'error' not in g for g in gs.values()):
                continue
            try:
                out = {'t': round(time.time(), 3), 'rank': self.rank, 'step': v, 'mode': self.mode,
                       'impl': self.cpu_impl if self.mode == 'cpu' else 'owner-class', 'groups': {}}
                for k, g in gs.items():
                    o = {}
                    for name in ('p2p', 'd2h', 'shadow'):
                        ev = g.get(f'ev_{name}')
                        if ev is not None and ev[0] is not None:
                            if not ev[1].query():
                                raise StopIteration
                            o[f'{name}_ms'] = round(ev[0].elapsed_time(ev[1]), 3)
                    e0, e1 = g.get('e_own0'), g.get('e_own1')
                    if isinstance(e0, torch.cuda.Event):
                        if not e1.query():
                            raise StopIteration
                        o['own_adam_ms'] = round(e0.elapsed_time(e1), 3)
                    for key in ('wait_ms', 'cpu_adam_ms', 'd2h_wait_ms', 'check', 'error'):
                        if key in g:
                            o[key] = g[key]
                    o['numel'] = self.slots[k][1].numel()
                    out['groups'][str(k)] = o
                out['sum'] = {key: round(sum(o.get(key, 0) for o in out['groups'].values()), 3)
                              for key in ('p2p_ms', 'own_adam_ms', 'shadow_ms', 'd2h_ms', 'cpu_adam_ms', 'wait_ms')}
                if self.check:
                    out['check_max_abs'] = max(o['check']['max_abs'] for o in out['groups'].values() if 'check' in o)
                self.records.append(out)
                if self.out:
                    with open(self.out, 'a') as fh:
                        fh.write(json.dumps(out, separators=(',', ':')) + '\n')
                done.append(v)
            except StopIteration:
                break
        for v in done:
            self.pending.pop(v, None)

    def report(self):
        return {'own': list(self.own), 'shadow': list(self.shadow_v), 'owner': self.owner, 'partner': self.partner}
