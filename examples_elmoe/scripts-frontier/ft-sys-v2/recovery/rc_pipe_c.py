"""rc_pipe_c.py (ft-sys-v2 recovery, milestone 3 C, 2026-11-06): Figure 8 on DeepSpeed's PipelineEngine. Only the lost stage redoes;
the working nodes keep their engines, their autograd graphs in flight and their place in the schedule.

The build order (m3_map, M3-C, decided 11-06 18:35):
  1 · the working nodes keep their group objects: rc_swap gives every group that holds a lost rank a new RCCL backend in place
  2 · a survivor never leaves train_batch: it pauses at its next communication call (hb_monitor, C mode), and an instruction whose
      communication was cut by the loss is found after a stream wait (its group was aborted after it began) and run again
  3 · a cut reduction can run again: the stage's .grad is copied before each reduction (decision 4.1, reduce out of place); ZeRO's
      bucket state is reset and the sums go back to the last commit
  4 · stage s + 1 keeps the input gradients it sent, stage s - 1 the activations it sent (the last KEEP of each, clones)
  5 · the joiner builds its engine with torch's fake backend (no partner needed), swaps real backends in, takes its state from its
      partner (Keep's shadow, the bf16 weights, the sums at its stage's commit), then replays its stage's schedule from micro-batch
      `start`: instructions the lost rank had finished as seen by its neighbours are skipped, the rest run
  6 · the neighbours re-send what the joiner must receive again (s - 1: activations of micro-batches start..; s + 1: gradients of
      start..) on a repair communicator of their own, so the regular pair communicator keeps the schedule's order
Pipeline sends run on DeepSpeed's per-pair groups in C mode (p2p.can_send_recv returns False, set before the engine is built), so
that only the pairs holding a lost rank are aborted and swapped.

v1 limits: one lost node (one stage of one replica), two data-parallel replicas (the partner is also the owner), commits every
micro-batch (B, k = 1), a failure outside the optimizer step, static activation shapes."""
import collections
import time

import torch
import torch.distributed as dist

KEEP = 6                      # sent activations / gradients kept per stage (micro-batches in flight are at most the pipeline depth)
KINDS = {'LoadMicroBatch': 'load', 'ForwardPass': 'fwd', 'SendActivation': 'sact', 'RecvActivation': 'ract',
         'SendGrad': 'sgrad', 'RecvGrad': 'rgrad', 'BackwardPass': 'bwd', 'ReduceTiedGrads': 'tied', 'ReduceGrads': 'red',
         'OptimizerStep': 'opt'}


class CutReduction(Exception):
    """A reduction whose communicator was aborted while it ran (raised by the commit hook in C mode)."""


def use_pair_groups():
    """DeepSpeed's per-pair groups for the pipeline's sends (p2p.py:21-34, 60-63): call before the engine is built."""
    from deepspeed.runtime.pipe import p2p
    p2p.can_send_recv = lambda: False


class PipeC:
    def __init__(self, adapter, st, ctl, log):
        """adapter: the ZeroAdapter of this engine; st: install_pipe_commits' state; ctl: the controller's C hooks (gen(),
        cut(group, t0), wait_resumed(), commit(), lost())."""
        from deepspeed.runtime.pipe import p2p
        self.p2p = p2p
        self.ad, self.st, self.ctl, self.log = adapter, st, ctl, log
        self.eng, self.z = adapter.engine, adapter.zero
        self.cnt = dict.fromkeys(('load', 'fwd', 'sact', 'ract', 'sgrad', 'rgrad', 'bwd'), 0)
        self.kept_act, self.kept_grad = collections.OrderedDict(), collections.OrderedDict()
        self.replay = None             # the joiner's plan for its first step
        self.repair = {}               # 'act' / 'grad' -> (backend, my rank in the pair)
        self.gbak = None
        self.resends = []

    # ------------------------------------------------------------------ helpers
    def _pair(self, a, b):
        try:
            return self.p2p._get_send_recv_group(a, b)
        except Exception:
            return None

    def _dp_groups(self):
        z = self.z
        gs = list(getattr(z, 'real_dp_process_group', []) or [])
        try:
            gs += [g for _, g in self.eng.module.get_tied_weights_and_groups()]
        except Exception:
            pass
        return gs

    def _sync(self):
        self.ctl.c_sync()                             # parks for the repair if a stop begins while it waits

    def _backup(self):
        self.gbak = [(p, p.grad.detach().clone()) for g in self.z.bit16_groups for p in g if p.grad is not None]

    def _restore(self):
        """.grad back to the copy taken before the reduction; ZeRO's bucket state empty again (stage_1_and_2.py:942-979,
        1367-1423, 679-683)."""
        z = self.z
        z.zero_grad(set_to_none=True)
        for p, g in (self.gbak or []):
            p.grad = g.clone()
        z.ipg_buffer = None
        z.elements_in_ipg_bucket = 0
        z.grads_in_ipg_bucket, z.params_in_ipg_bucket = [], []
        z.ipg_bucket_has_moe_params = False
        z.extra_large_param_to_reduce = None
        for i in range(len(z.params_already_reduced)):
            z.params_already_reduced[i] = False
        z.grads_in_partition, z.grads_in_partition_offset = None, 0

    def reset_step(self):
        for k in self.cnt:
            self.cnt[k] = 0
        self.kept_act.clear()
        self.kept_grad.clear()

    # ------------------------------------------------------------------ the instruction wrappers
    def install(self):
        from deepspeed.runtime.pipe import schedule as S
        eng = self.eng
        imap = dict(eng._INSTRUCTION_MAP)            # B's map (BackwardPass, ReduceTiedGrads already wrapped)
        for cls in list(imap):
            kind = KINDS.get(cls.__name__)
            if kind:
                imap[cls] = self._wrap(kind, imap[cls], cls)
        eng._INSTRUCTION_MAP = imap
        orig_tb = eng.train_batch

        def train_batch(data_iter=None):
            self.reset_step()
            if self.replay is not None:
                self._replay_begin()
            try:
                return orig_tb(data_iter=data_iter)
            finally:
                self.replay = None
                for w in self.resends:
                    try:
                        w.wait()
                    except Exception:
                        pass
                self.resends = []
        eng.train_batch = train_batch

    def _skip(self, kind, mb):
        """The joiner's replay: True for an instruction the lost rank had finished, as its neighbours saw it."""
        p = self.replay
        if p is None:
            return False
        if kind in ('load', 'fwd', 'ract', 'rgrad', 'bwd'):
            return mb < p['start']
        if kind == 'sact':
            return p['f_up'] is not None and mb < p['f_up']
        if kind == 'sgrad':
            return p['g'] is not None and mb < p['g']
        return False

    def _wrap(self, kind, fn, cls):
        me = self

        def run(eng_, *a, **kw):
            hb = me.ctl.hb
            hb.inst_begin()                          # after a stop: run on to the stage-replica's agreed instruction, then park
            try:
                return body(eng_, *a, **kw)
            finally:
                hb.inst_end()

        def body(eng_, *a, **kw):
            mb = me.cnt.get(kind)
            if mb is not None and me._skip(kind, mb):
                me.cnt[kind] += 1
                return None
            if kind == 'bwd':
                return me._run_bwd(fn, eng_, a, kw)
            if kind in ('tied', 'red'):
                return me._run_reduction(fn, eng_, a, kw, kind)
            if kind == 'ract' and me.replay is not None and mb < me.replay.get('act_end', 0):
                me._recv_resent_act(*a, **kw)
                me.cnt[kind] += 1
                return None
            if kind == 'rgrad' and me.replay is not None and mb < me.replay.get('grad_end', 0):
                me._recv_resent_grad(*a, **kw)
                me.cnt[kind] += 1
                return None
            if kind == 'sact':
                me._keep('act', mb, eng_.pipe_buffers['outputs'][kw.get('buffer_id', a[0] if a else None)])
            saved_inputs = None
            if kind == 'sgrad':
                bid = kw.get('buffer_id', a[0] if a else None)
                ins = eng_.pipe_buffers['inputs'][bid]
                saved_inputs = (bid, ins)            # SendGrad frees the input buffer after sending (pipe/engine.py:1124)
                ts = [ins] if torch.is_tensor(ins) else list(ins)
                me._keep('grad', mb, [t.grad for t in ts if t.is_floating_point() and t.grad is not None])
            pair = {'sact': (eng_.stage_id, eng_.stage_id + 1), 'ract': (eng_.stage_id - 1, eng_.stage_id),
                    'sgrad': (eng_.stage_id, eng_.stage_id - 1), 'rgrad': (eng_.stage_id + 1, eng_.stage_id)}.get(kind)
            while True:
                t0 = time.time()
                out = fn(eng_, *a, **kw)
                if pair is None:
                    break
                me.ctl.hb.S['c_where'] = 'p2p'
                me._sync()
                g = me._pair(*pair)
                if not me.ctl.cut(g, t0):
                    break
                me.log(f'C: {kind} of micro-batch {mb} was cut by the loss: runs again after the repair')
                me.ctl.wait_resumed()
                if saved_inputs is not None:
                    eng_.pipe_buffers['inputs'][saved_inputs[0]] = saved_inputs[1]
            if mb is not None:
                me.cnt[kind] += 1
            return out
        run._ftv2_c = True
        return run

    def _keep(self, which, mb, ts):
        d = self.kept_act if which == 'act' else self.kept_grad
        ts = [ts] if torch.is_tensor(ts) else list(ts or [])
        d[mb] = [t.detach().clone() for t in ts if torch.is_tensor(t)]
        while len(d) > KEEP:
            d.popitem(last=False)

    def _run_bwd(self, fn, eng_, a, kw):
        """B's BackwardPass wrapper split in two: the backward once, then the commit, which runs again if its reduction was
        cut (the .grad copy restored first)."""
        st = self.st
        st['c_bwd_only'] = True                      # B's wrapper: compute and position only (adapter install_pipe_commits)
        try:
            fn(eng_, *a, **kw)
        finally:
            st['c_bwd_only'] = False
        self._backup()
        while True:
            try:
                self.ctl.hb.S['c_red_mark'] = dict(self.ctl.hb.S['c_n'])   # each group's index where this reduction begins
                st['after_bwd']()
                break
            except CutReduction:
                self.log(f'C: the reduction after micro-batch {self.cnt["bwd"]} was cut: .grad restored, runs again after the '
                         f'repair')
                self._restore()
                self.ctl.wait_resumed()
        self.gbak = None
        self.cnt['bwd'] += 1

    def _run_reduction(self, fn, eng_, a, kw, kind):
        self._backup()
        while True:
            t0 = time.time()
            try:
                self.ctl.hb.S['c_red_mark'] = dict(self.ctl.hb.S['c_n'])
                fn(eng_, *a, **kw)
                self.ctl.hb.S['c_where'] = 'red'
                self._sync()
                if not any(self.ctl.cut(g, t0) for g in self._dp_groups()):
                    break
            except CutReduction:
                pass
            self.log(f'C: {kind} was cut: sums back to the last commit, .grad restored, runs again after the repair')
            self.ctl.restore_commit()
            self._restore()
            self.ctl.wait_resumed()
        self.gbak = None

    # ------------------------------------------------------------------ the neighbours' re-sends (survivors)
    def resend(self, role, joiner, lo, hi, store, key):
        """role 'act' (stage s - 1) or 'grad' (stage s + 1): send the kept tensors of micro-batches lo..hi-1 to the joiner on a
        repair communicator (pair rank 0 = me), without waiting (the regular pair keeps the schedule's order)."""
        b = dist.ProcessGroupNCCL(dist.PrefixStore(key, store), 0, 2, _opts())
        d = self.kept_act if role == 'act' else self.kept_grad
        missing = [mb for mb in range(lo, hi) if mb not in d]
        if missing:
            raise RuntimeError(f'C: {role} of micro-batches {missing} are no longer kept (kept {list(d)})')
        for mb in range(lo, hi):
            for t in d[mb]:
                self.resends.append(b.send([t], 1, 0))
        self.repair[role] = b
        self.log(f'C: re-sent {role} of micro-batches {lo}..{hi - 1} to rank {joiner}')

    # ------------------------------------------------------------------ the joiner
    def set_replay(self, plan, store):
        """plan: start, J (stage commit), g (s-1's received gradients), f_up (s+1's received activations), act_end, grad_end,
        the repair keys."""
        self.replay = dict(plan)
        if plan.get('act_end', 0) > plan['start']:
            self.repair['act'] = dist.ProcessGroupNCCL(dist.PrefixStore(plan['act_key'], store), 1, 2, _opts())
        if plan.get('grad_end', 0) > plan['start']:
            self.repair['grad'] = dist.ProcessGroupNCCL(dist.PrefixStore(plan['grad_key'], store), 1, 2, _opts())

    def _replay_begin(self):
        p, st, eng = self.replay, self.st, self.eng
        st['c_replay_pos'] = (int(p['start']), int(p['J']))      # read by install_pipe_commits' train_batch
        if not eng.is_first_stage():
            eng.pipe_recv_buf = eng._recv_tensor_meta(eng.prev_stage)       # the first message on the new pair (s - 1 sent it)
        eng.first_output_send = False                                       # stage s + 1 already knows the shapes
        self.log(f'C: replay from micro-batch {p["start"]} (commit {p["J"]}; s-1 has gradients up to {p["g"]}, s+1 activations '
                 f'up to {p["f_up"]}; re-sent activations to {p.get("act_end")}, gradients to {p.get("grad_end")})')

    def _recv_resent_act(self, buffer_id):
        eng = self.eng
        b = self.repair['act']
        buf = eng.pipe_recv_buf
        if torch.is_tensor(buf):
            b.recv([buf], 0, 0).wait()
            self._sync()
            recvd = buf.clone().detach()
            recvd.requires_grad = recvd.is_floating_point()
        else:
            out = []
            for t in buf:
                b.recv([t], 0, 0).wait()
                self._sync()
                out.append(t.clone().detach())
            if eng.has_attention_mask or eng.has_bool_tensors:
                out[-1] = out[-1].bool()
            recvd = tuple(out)
            for t in recvd:
                t.requires_grad = t.is_floating_point()
        eng.pipe_buffers['inputs'][buffer_id] = recvd

    def _recv_resent_grad(self, buffer_id):
        eng = self.eng
        outputs = eng.pipe_buffers['outputs'][buffer_id]
        if eng.grad_layer is None:
            if torch.is_tensor(outputs):
                eng.grad_layer = eng._allocate_or_extend_buffers(0, list(outputs.size()), outputs.dtype)
            else:
                eng.grad_layer = [eng._allocate_or_extend_buffers(i, list(t.size()), t.dtype)
                                  for i, t in enumerate([t for t in outputs if t.is_floating_point()])]
        b = self.repair['grad']
        for t in ([eng.grad_layer] if torch.is_tensor(eng.grad_layer) else eng.grad_layer):
            b.recv([t], 0, 0).wait()
        self._sync()

    # ------------------------------------------------------------------ reports
    def report(self):
        return {'cnt': dict(self.cnt), 'kept_act': list(self.kept_act), 'kept_grad': list(self.kept_grad),
                'stage': int(self.eng.stage_id), 'stages': int(self.eng.num_stages)}


def _opts():
    from datetime import timedelta
    o = dist.ProcessGroupNCCL.Options()
    o._timeout = timedelta(seconds=1800)
    return o


def plan_for(lost_rank, reports, world, stages, J):
    """The joiner's replay plan for lost rank L (pure): its neighbours in the same pipeline are L - R (stage s - 1) and L + R
    (stage s + 1), R = world / stages ranks per stage. start = the first micro-batch whose backward the joiner must run:
    min(J, what s - 1 has received), so that every micro-batch s - 1 still waits for is computed again and every micro-batch the
    stage's sums lack is added; the backward of micro-batches start..J-1 only feeds stage s - 1 (their parameter gradients are in
    the sums, drop_until = J)."""
    R = world // stages
    s = lost_rank // R
    prev = lost_rank - R if s > 0 else None
    nxt = lost_rank + R if s < stages - 1 else None
    g = reports[prev]['c']['cnt']['rgrad'] if prev is not None else None
    f_up = reports[nxt]['c']['cnt']['ract'] if nxt is not None else None
    start = min(J, g) if g is not None else J
    act_end = reports[prev]['c']['cnt']['sact'] if prev is not None else 0
    grad_end = reports[nxt]['c']['cnt']['sgrad'] if nxt is not None else 0
    return {'rank': lost_rank, 'stage': s, 'prev': prev, 'next': nxt, 'start': int(start), 'J': int(J), 'g': g, 'f_up': f_up,
            'act_end': int(max(act_end, start)), 'grad_end': int(max(grad_end, start)),
            'act_key': f'repair/act/{prev}-{lost_rank}/', 'grad_key': f'repair/grad/{nxt}-{lost_rank}/'}
