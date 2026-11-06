"""adapters/deepspeed_zero.py (ft-sys-v2 recovery, 2026-11-06): the optimizer adapter for a DeepSpeed engine with ZeRO stage 1 or 2.

The recovery's core (rc_keep, rc_controller) never touches DeepSpeed; it talks to this object only. A port to another sharded
optimizer (Megatron's distributed optimizer, FSDP) replaces this file and nothing else.

Interface, and where each hook sits (the reason in one line):
  slots()                    the inner optimizer's parameters, [(group index, tensor)]: ZeRO hands each group's fp32 partition to
                             its inner optimizer (stage_1_and_2.py:403-412); that partition IS the rank's optimizer state.
  hyper(i)                   lr, weight decay, betas, eps, bias correction, AdamW or L2, the step count of group i: read from the
                             inner optimizer's own param_groups, which the LR scheduler writes before every step.
  moments(t)                 (exp_avg, exp_avg_sq) of t from the inner optimizer's state (None before its first step).
  wrap_inner_step(before, after)
                             the Keep hook: the inner optimizer's step, wrapped on the object (no DeepSpeed source changed). It is
                             the first place where the gradient is final (reduced, unscaled and clipped, stage_1_and_2.py:1934-1937)
                             and the last place before the state changes; every sharded optimizer has such an inner step over its
                             own partition. ZeRO calls it once per group (stage_1_and_2.py:1839-1851).
  take_state()               a survivor's own state, kept by reference across the rebuild: fp32 partitions, moments, step counts.
  put_state(states)          fill a freshly built engine: fp32 partitions copied in, moments and step counts through the inner
                             optimizer's load_state_dict (torch's own interface; same device and dtype, so no copy).
  refresh_weights()          the bf16 weights from the fp32 masters: ZeRO's own update_lp_params (own partition, then its all-gather).
  set_counters(steps, samples, micro)  the engine's step counters (they drive its logging and the flops profiler's step).
  release()                  engine.destroy() (removes ZeRO's gradient hooks), so the old engine can be freed.
  reset_registries()         (static) DeepSpeed's group dictionaries and the activation-checkpointing RNG tracker, so that
                             deepspeed.initialize run again builds new groups instead of reusing the aborted ones
                             (deepspeed/utils/groups.py:35-52; the expert-group builder skips names already present).
  gpu_shadow_optimizer(ps)   an optimizer of the inner optimizer's own class over the shadow tensors (FTV2_SHADOW=gpu).
  set_step(opt, group, p, n) set an optimizer's step count before an update (apex keeps it in the group, torch in the state)."""
import inspect

import torch


class ZeroAdapter:
    def __init__(self, engine):
        self.engine = engine
        self.zero = engine.optimizer
        self.inner = self.zero.optimizer
        self._groups = list(self.inner.param_groups)      # the full list (ZeRO swaps in one group while it steps)

    # ------------------------------------------------------------------ what the Keep reads
    def slots(self):
        out = []
        for i, g in enumerate(self._groups):
            for p in g['params']:
                out.append((i, p))
        return out

    def hyper(self, i):
        g = self._groups[i]
        b1, b2 = g.get('betas', (0.9, 0.999))
        step = g.get('step', None)
        if step is None:
            st = self.inner.state.get(g['params'][0], {})
            step = st.get('step', 0)
        if torch.is_tensor(step):
            step = int(step.item())
        adamw = bool(getattr(self.inner, 'adam_w_mode', True)) if hasattr(self.inner, 'adam_w_mode') else \
            ('AdamW' in type(self.inner).__name__ or g.get('decoupled_weight_decay', False))
        return {'lr': float(g['lr']), 'weight_decay': float(g.get('weight_decay', 0.0)), 'beta1': float(b1), 'beta2': float(b2),
                'eps': float(g.get('eps', 1e-8)), 'bias_correction': bool(g.get('bias_correction', True)), 'step': int(step),
                'adamw': adamw}

    def moments(self, p):
        st = self.inner.state.get(p, {})
        return st.get('exp_avg'), st.get('exp_avg_sq')

    def wrap_inner_step(self, before, after):
        inner = self.inner
        orig = inner.step

        def step(*a, **k):
            ps = [p for g in inner.param_groups for p in g['params']]
            before(ps)
            out = orig(*a, **k)
            after(ps)
            return out
        step._ftv2_wrapped = True
        inner.step = step

    # ------------------------------------------------------------------ restore
    def take_state(self):
        out = []
        for i, p in self.slots():
            m, v = self.moments(p)
            out.append({'fp32': p.data, 'exp_avg': m, 'exp_avg_sq': v, 'step': self.hyper(i)['step']})
        return out

    @torch.no_grad()
    def put_state(self, states):
        slots = self.slots()
        assert len(states) == len(slots), f'{len(states)} states for {len(slots)} groups'
        sd = {'state': {}, 'param_groups': []}
        for k, ((i, p), s) in enumerate(zip(slots, states)):
            assert s['fp32'].numel() == p.numel(), f'group {i}: {s["fp32"].numel()} != {p.numel()}'
            p.data.copy_(s['fp32'].view_as(p))
            m = s['exp_avg'] if s['exp_avg'] is not None else torch.zeros_like(p, dtype=torch.float32)
            v = s['exp_avg_sq'] if s['exp_avg_sq'] is not None else torch.zeros_like(p, dtype=torch.float32)
            sd['state'][k] = {'exp_avg': m.view_as(p), 'exp_avg_sq': v.view_as(p)}
        k = 0
        for i, g in enumerate(self._groups):
            ng = {key: val for key, val in g.items() if key != 'params'}
            ng['params'] = list(range(k, k + len(g['params'])))
            steps = [states[j]['step'] for j in range(k, k + len(g['params']))]
            if 'step' in g or type(self.inner).__module__.startswith('apex'):
                ng['step'] = int(steps[0])                     # apex FusedAdam keeps the count in the group
            k += len(g['params'])
            sd['param_groups'].append(ng)
        if not type(self.inner).__module__.startswith('apex'):
            for j, s in enumerate(states):
                sd['state'][j]['step'] = torch.tensor(float(s['step']))
        self.inner.load_state_dict(sd)
        self._groups = list(self.inner.param_groups)
        # the hp-param links to the optimizer state are made lazily after the next step (stage_1_and_2.py:609-614)

    def refresh_weights(self):
        self.zero.update_lp_params()

    def set_counters(self, steps, samples, micro):
        e = self.engine
        e.global_steps, e.global_samples, e.micro_steps, e.skipped_steps = int(steps), int(samples), int(micro), 0

    def release(self):
        try:
            self.engine.destroy()
        except Exception:
            pass
        self.engine = self.zero = self.inner = None
        self._groups = []

    @staticmethod
    def reset_registries():
        from deepspeed.utils import groups
        for name in ('_EXPERT_PARALLEL_GROUP', '_EXPERT_DATA_PARALLEL_GROUP', '_ALL_TO_ALL_GROUP', '_RBD_LOCAL_GROUP'):
            d = getattr(groups, name, None)
            if isinstance(d, dict):
                d.clear()                      # in place: other modules hold the dict objects themselves
        for name in ('_WORLD_GROUP', '_ZERO_PARAM_INTRA_PARALLEL_GROUP', '_DATA_PARALLEL_GROUP'):
            if hasattr(groups, name):
                setattr(groups, name, None)
        try:
            from deepspeed.runtime.activation_checkpointing import checkpointing as ckpt
            ckpt.get_cuda_rng_tracker().reset()
        except Exception:
            pass

    # ------------------------------------------------------------------ the GPU shadow (check mode)
    def gpu_shadow_optimizer(self, params):
        cls = type(self.inner)
        kw = dict(self.inner.defaults)
        if hasattr(self.inner, 'adam_w_mode'):
            kw['adam_w_mode'] = bool(self.inner.adam_w_mode)
        sig = inspect.signature(cls.__init__).parameters
        kw = {k: v for k, v in kw.items() if k in sig}
        return cls(params, **kw)

    @staticmethod
    def set_step(opt, group, p, n):
        if type(opt).__module__.startswith('apex'):
            if n > 0:
                group['step'] = int(n)
            else:
                group.pop('step', None)
        else:
            st = opt.state[p]
            st['step'] = torch.tensor(float(n))
