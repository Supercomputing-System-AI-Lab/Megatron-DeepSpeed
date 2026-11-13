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
import time

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

    def release(self, log=None):
        """Release the engine: DeepSpeed's destroy (removes ZeRO's gradient-accumulation hooks, clears its debug registries); the
        hooks are removed again here if destroy raised (2026-11-07, M3MA2 5874600: a parameter of the old model stayed alive,
        held from C++, which is what a hook on its AccumulateGrad node does), and destroy's error is reported, not swallowed."""
        try:
            self.engine.destroy()
        except Exception as ex:
            if log:
                log(f'release: engine.destroy raised {ex!r}')
        for h in list(getattr(self.zero, '_grad_acc_hooks', None) or []):
            try:
                h.remove()
            except Exception:
                pass
        if self.zero is not None and hasattr(self.zero, '_grad_acc_hooks'):
            self.zero._grad_acc_hooks = []
        if self.zero is not None and hasattr(self.zero, 'grad_accs'):
            self.zero.grad_accs = []
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

    # ------------------------------------------------------------------ milestone 2: the partial sums of ZeRO stage 2
    # (each owner's reduced partition summed over the micro-batches so far: zero.averaged_gradients[i], a list of tensors that
    # get_flat_partition made, stage_1_and_2.py:785-803)
    def zero2_groups(self):
        return list(range(len(self.zero.bit16_groups)))

    def zero2_partition_size(self, i):
        return int(self.zero.partition_size[i])

    def zero2_dtype(self):
        return self.zero.gradient_accumulation_dtype

    def zero2_flat_partial(self, i):
        """A flat COPY of group i's partial sum (zeros before the first micro-batch). torch's _flatten_dense_tensors returns a
        view of the tensor itself when the list holds one tensor, so a partition made of one parameter slice gave a "snapshot"
        that kept accumulating with the live sums (E5d 5862827 z2_kill: the resume one commit back, J = 2, took S_3 for those
        groups; the redone step's grad norm was 6.848 against 6.713-6.715). Always a fresh tensor now."""
        z = self.zero
        lst = z.averaged_gradients.get(i) if isinstance(z.averaged_gradients, dict) else None
        if not lst:
            return torch.zeros(self.zero2_partition_size(i), dtype=self.zero2_dtype(), device=z.device)
        flat = z.flatten(lst)
        if any(flat.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in lst):
            flat = flat.clone()
        return flat

    @torch.no_grad()
    def zero2_flat_partial_into(self, i, out):
        """zero2_flat_partial written into `out` (a flat buffer of the partition's size and dtype, from an earlier snapshot), so
        a snapshot needs no new memory (ft-sys-v2 overlap, keep mode). The same values in the same order as
        _flatten_dense_tensors(lst) (torch.cat of the flattened tensors). out None, or a size or dtype that does not match: a
        fresh tensor, as zero2_flat_partial."""
        if out is None or out.numel() != self.zero2_partition_size(i) or out.dtype != self.zero2_dtype():
            return self.zero2_flat_partial(i)
        z = self.zero
        lst = z.averaged_gradients.get(i) if isinstance(z.averaged_gradients, dict) else None
        if not lst:
            out.zero_()
            return out
        if sum(t.numel() for t in lst) != out.numel():
            return self.zero2_flat_partial(i)
        torch.cat([t.reshape(-1) for t in lst], out=out)
        return out

    @torch.no_grad()
    def zero2_set_partial(self, i, flat):
        """Group i's partial sum := flat (None: no sum yet, so the next micro-batch starts it, as at a step's start)."""
        z = self.zero
        if flat is None:
            z.averaged_gradients[i] = None
            return
        lst = z.averaged_gradients.get(i)
        if not lst:
            lst = z.get_flat_partition(z.params_in_partition[i], z.first_offset[i], z.partition_size[i],
                                       dtype=self.zero2_dtype(), device=z.device, return_tensor_list=True)
            lst = [t.clone() for t in lst]                 # own storage, never a view of a parameter's gradient
        off = 0
        for t in lst:
            n = t.numel()
            t.copy_(flat[off:off + n].view_as(t))
            off += n
        assert off == flat.numel(), f'group {i}: {off} != {flat.numel()}'
        z.averaged_gradients[i] = lst

    def wrap_epilogue(self, after, last_fn=None):
        """Milestone 2's hook: after ZeRO's per-micro-batch epilogue (stage 2: the micro-batch's reduced piece was just added to
        the owner's sums), call after(last), last = the gradient-accumulation boundary. Wrapped on the object. Milestone 3
        (ZeRO-1 under the pipeline, install_pipe_commits) passes last_fn: a spaced reduction sets the boundary flag too."""
        z = self.zero
        orig = z.overlapping_partition_gradients_reduce_epilogue

        def epilogue(*a, **k):
            out = orig(*a, **k)
            after(bool(last_fn()) if last_fn is not None else bool(getattr(z, 'is_gradient_accumulation_boundary', False)))
            return out
        epilogue._ftv2_wrapped = True
        z.overlapping_partition_gradients_reduce_epilogue = epilogue

    def fix_reduce_owner_check(self):
        """A DeepSpeed defect found by E5c 5862600, fixed on the object (DeepSpeed's source unchanged). With ZeRO stage 2 and
        use_multi_rank_bucket_allreduce=false, each bucket slice is REDUCED to its owner (dist.reduce), and the owner then
        copies the result back into its gradient. allreduce_and_copy decides who copies with
            if rank is None or rank == dist.get_rank(group=self.dp_process_group):      (stage_1_and_2.py:1548)
        where `rank` is the owner's rank in the bucket's own process group: for an expert bucket, the expert data-parallel
        group (3 ranks on X-MoE 10B), not the dense group (24). The wrong ranks copy, and the real owners keep their local,
        unreduced gradient (E5c: grad norm 14.817 against 15.815 at step 1). The all-reduce path compares in the bucket's
        own group (stage_1_and_2.py:1023). This wrapper is DeepSpeed's body (stage_1_and_2.py:1529-1550) with that one test
        made in the bucket's own group: `dist.get_rank(group=process_group)`."""
        from deepspeed.runtime.zero import stage_1_and_2 as s12
        z = self.zero

        def allreduce_and_copy(small_bucket, rank=None, log=None, divide=True, process_group=None):
            process_group = z.dp_process_group if process_group is None else process_group
            if z.overlap_comm:
                if not s12.get_accelerator().resolves_data_dependency():
                    s12.get_accelerator().synchronize()
                z._clear_previous_reduced_grads()
                stream = z.reduction_stream
            else:
                stream = s12.get_accelerator().current_stream()
            with s12.get_accelerator().stream(stream):
                allreduced = z.allreduce_bucket(small_bucket, rank=rank, log=log, divide=divide, process_group=process_group)
                if rank is None or rank == s12.dist.get_rank(group=process_group):
                    for buf, synced in zip(small_bucket, z.unflatten(allreduced, small_bucket)):
                        buf.copy_(synced)
        allreduce_and_copy._ftv2_wrapped = True
        z.allreduce_and_copy = allreduce_and_copy

    def wrap_reduce_keep(self, owner_global, pieces):
        """Milestone 2, "all-reduce + keep" (FTV2_COMMIT_MODE=keep; the user's comparison with "reduce + p2p"): with
        use_multi_rank_bucket_allreduce, ZeRO stage 2 ALL-REDUCES every bucket of every micro-batch over the data-parallel group
        and each rank then copies only the slices it owns (stage_1_and_2.py:1014-1024, 1129-1150). The partner (rank + stride)
        is in its owner's data-parallel groups, so it already receives the owner's reduced slices and drops them. This hook
        keeps them: each slice of a parameter owned by `owner_global` is copied into pieces[i] (group i, the owner's partition
        layout) at DeepSpeed's own insertion offset (grad_partition_insertion_offset, stage_1_and_2.py:826-874). No network.
        The walk over the bucket repeats DeepSpeed's own (average_tensor, stage_1_and_2.py:1075-1127), per parameter slice
        and before DeepSpeed merges neighbouring slices of one owner (neighbours in the bucket need not be neighbours in the
        partition). Both methods are wrapped on the object; DeepSpeed's source is unchanged."""
        import torch.distributed as dist
        from deepspeed.moe.utils import is_moe_param
        z = self.zero
        assert z.use_multi_rank_bucket_allreduce and z.reduce_scatter, \
            'the keep path needs ZeRO stage 2 with use_multi_rank_bucket_allreduce (every rank receives every slice)'
        orig_avg = z.average_tensor
        orig_arc = z.allreduce_and_copy_with_multiple_ranks
        owner_in = {}
        ctx = {'base': None, 'slices': []}

        def owner_rank(pg):
            k = id(pg)
            if k not in owner_in:
                ranks = dist.get_process_group_ranks(pg)
                owner_in[k] = ranks.index(owner_global) if owner_global in ranks else -1
            return owner_in[k]

        def average_tensor(tensor):
            slices, curr = [], 0
            for i, param, param_id in z.params_in_ipg_bucket:
                pg = z.dp_process_group
                if z.ipg_bucket_has_moe_params:
                    pg = z.expert_dp_process_group[param.group_name] if is_moe_param(param) else z.dp_process_group
                o = owner_rank(pg)
                w = sorted(((p, z.grad_start_offset[i][p][param_id]) for p in z.param_to_partition_ids[i][param_id]),
                           key=lambda t: t[1])
                for idx, (p, off) in enumerate(w):
                    # DeepSpeed's offsets can be floats (partition_size = len(flat) / world, stage_1_and_2.py:421; it casts with
                    # int() where it uses them, :1129): E5c 5862600's first keep arm stopped on a float slice index
                    n = int((param.numel() - off) if idx == len(w) - 1 else (w[idx + 1][1] - off))
                    if p == o:
                        slices.append((curr, n, i, int(z.grad_partition_insertion_offset[i][p][param_id])))
                    curr += n
            ctx['base'], ctx['slices'] = tensor, slices
            try:
                return orig_avg(tensor)
            finally:
                ctx['base'], ctx['slices'] = None, []

        def allreduce_and_copy_with_multiple_ranks(small_bucket, log=None, divide=True, process_group=None, bucket_ranks=None):
            if not ctx['slices']:
                return orig_arc(small_bucket, log=log, divide=divide, process_group=process_group, bucket_ranks=bucket_ranks)
            # DeepSpeed's own body (stage_1_and_2.py:1020-1024), plus the partner's copy of its owner's slices
            pg = z.dp_process_group if process_group is None else process_group
            allreduced = z.allreduce_bucket(small_bucket, log=log, divide=divide, process_group=pg)
            o = owner_rank(pg)
            base, es = ctx['base'], ctx['base'].element_size()
            for buf, synced, bucket_rank in zip(small_bucket, z.unflatten(allreduced, small_bucket), bucket_ranks):
                if dist.get_rank(group=pg) == bucket_rank:
                    buf.copy_(synced)
                if bucket_rank == o:
                    b0 = (buf.data_ptr() - base.data_ptr()) // es
                    b1 = b0 + buf.numel()
                    for (c, n, i, ins) in ctx['slices']:
                        if b0 <= c and c + n <= b1:
                            pieces[i][ins:ins + n].copy_(synced.view(-1)[c - b0:c - b0 + n])
        average_tensor._ftv2_wrapped = True
        allreduce_and_copy_with_multiple_ranks._ftv2_wrapped = True
        z.average_tensor = average_tensor
        z.allreduce_and_copy_with_multiple_ranks = allreduce_and_copy_with_multiple_ranks

    # ------------------------------------------------------------------ milestone 3: reductions every k micro-batches under the
    # pipeline engine (ZeRO stage 1; m3_map question 4, option 2, "spaced reductions")
    def is_pipe(self):
        from deepspeed.runtime.pipe.engine import PipelineEngine
        return isinstance(self.engine, PipelineEngine)

    def pipe_tied_groups(self):
        """Weights shared by two stages (tied embeddings): their gradients are summed across stages at the step's end
        (ReduceTiedGrads, pipe/engine.py:276-292), after a spaced reduction would have cleared them."""
        try:
            return len(self.engine.module.get_tied_weights_and_groups())
        except Exception:
            return 0

    def install_pipe_commits(self, k, start, overlap=None):
        """Reduce the stage's gradients after every k-th backward pass, with ZeRO-1's own code, in place of the one reduction
        after the last backward. Why it is DeepSpeed's code and nothing else (stage_1_and_2.py, DeepSpeed 0.15.5):
          - under the pipeline DeepSpeed turns overlap_comm off (engine.py:1711), so ZeRO-1 has no gradient hooks and the
            gradients of every micro-batch so far sit in each parameter's .grad;
          - reduce_gradients(pipeline_parallel=True) (:709-729) reduces them bucket by bucket only when the optimizer's
            is_gradient_accumulation_boundary is set (:1426), and its epilogue ADDS the reduced partition into
            averaged_gradients (:795-803), releases the partition buffer (:679-683) and sets .grad to None (:810);
          - the engine's own reduction after the last backward (pipe/engine.py:294-313) then adds the last micro-batches.
        The sum is the same as one reduction per step; only the order of additions (and so the rounding) differs, so a clean run
        compares with a clean run at the same k.
        Wrapped on the object: the engine's instruction map (an instance copy with BackwardPass replaced: _exec_schedule looks
        the instruction up in self._INSTRUCTION_MAP, pipe/engine.py:1447-1451) and train_batch (the start of a resumed step).
        start() is called at every train_batch and returns (J, drop_until): the step runs micro-batches J..m-1 (the engine's
        micro_batches set to m - J for this call; the loss scale stays 1/m, DeepSpeedEngine.backward), and a stage whose
        sums already hold micro-batches J..drop_until-1 drops their gradients instead of reducing them.
        Returns the state dict {'pos': micro-batches of this step done here, 'spaced': True inside a spaced reduction}.

        overlap (FTV2_COMMIT_OVERLAP=1; None otherwise): {'side': a stream, 'before': fn, 'end': fn}. The reduction then runs on
        the side stream and the host goes on to the gradient send and the next forward without waiting for it:
          - 'before'() first (the controller promotes the commit still in flight, or raises the stop): one in flight at a time;
          - an event on the compute stream at the backward's end; every gradient handed to the reduction is marked as used by the
            side stream (record_stream), because ZeRO frees it on the host as soon as it is copied into its bucket
            (stage_1_and_2.py:965-966) and the next backward's allocations on the compute stream must not reuse its memory
            before the side stream has read it; the side stream waits for the event, then ZeRO's own reduce_gradients and its
            epilogue (with the commit's work, rc_commit) are enqueued there: DeepSpeed takes the current stream for its
            buckets, collectives and copies when overlap_comm is off (stage_1_and_2.py:1054-1061, :1384-1392, :1531-1538), so nothing of
            DeepSpeed changes. After it every .grad is None (the epilogue's zero_grad), so the next backward writes new tensors;
          - the step's end (ReduceTiedGrads, ReduceGrads) calls 'end'() and makes the compute stream wait for the side stream
            before anything reads the partition sums (the tied exchange, the last reduction's adds, the optimizer step).
        FTV2_COMMIT_AT=send is ignored with overlap (nothing is kept waiting for a reduction)."""
        import os
        from deepspeed.runtime.pipe import schedule as pipe_sched
        eng, z = self.engine, self.zero
        # where the reduction runs (FTV2_COMMIT_AT): 'backward' right after the backward pass; 'send' after the next
        # instruction that sends this micro-batch's input gradient to the previous stage (SendGrad, issued at the stage's next
        # forward step, pipe/schedule.py:213-216), so the previous stage is not kept waiting for its gradient while this stage
        # reduces. The first stage sends no gradient: it reduces right after its backward either way.
        at = os.environ.get('FTV2_COMMIT_AT', 'backward') if overlap is None else 'backward'
        defer = at == 'send' and not eng.is_first_stage()
        st = {'pos': 0, 'drop_until': 0, 'm': int(eng.micro_batches), 'spaced': False, 'k': int(k), 'pending': False,
              'at': 'send' if defer else 'backward'}
        orig_bwd = type(eng)._INSTRUCTION_MAP[pipe_sched.BackwardPass]
        orig_sgrad = type(eng)._INSTRUCTION_MAP[pipe_sched.SendGrad]

        def reduce_now():
            st['red_t0'] = time.time()                    # milestone 3 C: was this reduction cut by a loss after it began?
            prev = z.is_gradient_accumulation_boundary
            z.is_gradient_accumulation_boundary = True
            st['spaced'] = True
            try:
                if overlap is None:
                    ta = trace_ev()
                    z.reduce_gradients(pipeline_parallel=True)
                    trace_add(ta)
                else:
                    reduce_on_side()
            finally:
                st['spaced'] = False
                z.is_gradient_accumulation_boundary = prev

        def trace_ev():                                   # FTV2_EVENT_TRACE (rc_controller.trace_pipe): an event on the current stream
            tr = st.get('trace')
            if tr is None or not tr.get('on'):
                return None
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e

        def trace_add(ta):
            if ta is not None:
                tb = torch.cuda.Event(enable_timing=True)
                tb.record()
                st['trace']['ev'].append(('red', st['trace']['step'], ta, tb, None, None))

        def reduce_on_side():
            overlap['before']()                           # the previous commit promoted (or the stop raised)
            side = overlap['side']
            ev = torch.cuda.Event()
            ev.record()                                   # the compute stream, after this backward
            n = 0
            for g in z.bit16_groups:
                for p in g:
                    for t in (p.grad, getattr(p, 'grad_accum', None)):
                        if t is not None:
                            t.record_stream(side)
                            n += 1
            st['side_marked'] = n
            with torch.cuda.stream(side):
                side.wait_event(ev)
                t0 = torch.cuda.Event(enable_timing=True)
                t0.record()
                st['side_t0'] = t0                        # the reduction's start on the side stream (rc_controller reads it)
                ta = trace_ev()
                z.reduce_gradients(pipeline_parallel=True)
                trace_add(ta)

        def settle_side():
            if overlap is not None:
                overlap['end']()
                torch.cuda.current_stream().wait_stream(overlap['side'])

        def after_bwd():
            p = st['pos']
            if p >= st['m'] or p % st['k']:
                return
            if p <= st['drop_until']:
                z.zero_grad(set_to_none=True)             # these micro-batches are in this stage's sums already
                return
            if defer:
                st['pending'] = True
            else:
                reduce_now()

        def backward(self_, buffer_id):
            orig_bwd(self_, buffer_id)
            st['pos'] += 1
            if st.get('c_bwd_only'):                      # milestone 3 C (rc_pipe_c): the commit runs apart, so that it can
                return                                    # run again without the backward
            after_bwd()
        st['after_bwd'] = after_bwd

        def send_grad(self_, buffer_id):
            orig_sgrad(self_, buffer_id)
            if st['pending']:
                st['pending'] = False
                reduce_now()
        # Weights tied across stages (the embedding table of the first stage and the output layer of the last one, X-MoE's
        # TiedLayerSpec('embed'): the template unties them only for DENSE models). DeepSpeed sums their gradients across the
        # stages after the last backward (ReduceTiedGrads: an all-reduce of each tied weight's .grad over its tie group, pipe/
        # engine.py:276-292), then ReduceGrads reduces over the data-parallel group. The spaced reductions have by then moved
        # every earlier micro-batch's tied gradient into averaged_gradients, the stage's own part only. So, before the engine's
        # ReduceTiedGrads (where the stages already meet), each stage rebuilds its whole data-parallel sum of the tied weight from
        # the disjoint ZeRO partitions (an all-reduce of zeros plus each owner's slice), adds the other stage's over the tie
        # group, and writes its slice back. Sums commute: DPavg(g0) + DPavg(g1) = DPavg(g0 + g1); only the rounding order
        # differs from the original, so a clean run compares with a clean run at the same k.
        tied = []
        try:
            tied = list(eng.module.get_tied_weights_and_groups())
        except Exception:
            tied = []
        st['tied'] = len(tied)

        def tied_exchange():
            import torch.distributed as dist
            for w, tgroup in tied:
                gi = next((i for i, g in enumerate(z.bit16_groups) if any(q is w for q in g)), None)
                if gi is None:
                    continue
                if not z.averaged_gradients.get(gi):
                    self.zero2_set_partial(gi, torch.zeros(self.zero2_partition_size(gi), dtype=self.zero2_dtype(),
                                                           device=z.device))
                full = torch.zeros(w.numel(), dtype=self.zero2_dtype(), device=z.device)
                li = next((j for j, q in enumerate(z.params_in_partition[gi]) if q is w), None)
                if li is not None:
                    piece = z.averaged_gradients[gi][li].view(-1)
                    start = int(z.first_offset[gi]) if li == 0 else 0
                    full[start:start + piece.numel()].copy_(piece)
                dist.all_reduce(full, group=z.real_dp_process_group[gi])      # disjoint owners: the stage's whole sum
                dist.all_reduce(full, group=tgroup)                            # plus the other stage's
                if li is not None:
                    piece = z.averaged_gradients[gi][li]
                    start = int(z.first_offset[gi]) if li == 0 else 0
                    piece.copy_(full[start:start + piece.numel()].view_as(piece))

        orig_tied = type(eng)._INSTRUCTION_MAP[pipe_sched.ReduceTiedGrads]

        def reduce_tied(self_):
            settle_side()
            if tied and st['k'] < st['m']:
                tied_exchange()
            orig_tied(self_)
        orig_reduce = type(eng)._INSTRUCTION_MAP[pipe_sched.ReduceGrads]

        def reduce_grads(self_):
            settle_side()
            orig_reduce(self_)
        reduce_grads._ftv2_wrapped = True
        reduce_tied._ftv2_wrapped = True
        backward._ftv2_wrapped = True
        send_grad._ftv2_wrapped = True
        imap = dict(type(eng)._INSTRUCTION_MAP)
        imap[pipe_sched.BackwardPass] = backward
        imap[pipe_sched.ReduceTiedGrads] = reduce_tied
        if overlap is not None:
            imap[pipe_sched.ReduceGrads] = reduce_grads
        if defer:
            imap[pipe_sched.SendGrad] = send_grad
        eng._INSTRUCTION_MAP = imap
        orig_tb = eng.train_batch

        def train_batch(data_iter=None):
            J, drop = start()
            st['pos'], st['drop_until'], st['pending'] = int(J), int(drop), False
            if st.get('c_replay_pos') is not None:        # milestone 3 C's joiner: the whole schedule, positions from its plan
                st['pos'], st['drop_until'] = st['c_replay_pos']
                st['c_replay_pos'], J = None, 0
            if J:
                eng.micro_batches = st['m'] - int(J)
            try:
                return orig_tb(data_iter=data_iter)
            finally:
                eng.micro_batches = st['m']
        train_batch._ftv2_wrapped = True
        eng.train_batch = train_batch
        return st

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
