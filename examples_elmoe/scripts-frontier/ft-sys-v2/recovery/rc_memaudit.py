"""rc_memaudit.py (2026-11-07, L2 5874149): what holds GPU memory after a survivor releases its old engine (rc_controller.recover).

B's rebuild at 16k kept about 12.7 GB on stage 0 beyond the kept state (1.89 GB) after 'released the old engine'; at seq 2048 about
5.6 GB; at PP 1 / ZeRO-2 (M2R 5867849) 13.8 GB. FTV2_MEM_AUDIT=1 turns this on (a test setting): torch's allocation history is
recorded from the first training step (start), and after the release (audit) each audited rank writes <out>/ft/memaudit/
rank_<r>_g<g>.txt with
  - allocated bytes, the live Python CUDA tensors (one per storage) and their sum, and the rest (no Python owner: autograd's saved
    tensors, or tensors held only by C++);
  - the live tensors with an autograd graph (a graph keeps every tensor it saved for its backward);
  - for the largest storages and the graph tensors: chains of gc referrers up to a named holder (an attribute of an object, a
    module global, a closure cell, a frame, a traceback);
  - the active blocks of torch.cuda.memory._snapshot() summed by the Python allocation site outside torch.
FTV2_MEM_AUDIT_RANKS (default '0'; ';' or ',' between ranks) names the ranks that walk referrers (all ranks write the totals)."""
import gc
import os
import sys
import types

import torch


def start():
    if os.environ.get('FTV2_MEM_AUDIT') != '1' or not torch.cuda.is_available():
        return
    try:
        torch.cuda.memory._record_memory_history(max_entries=200000)
    except Exception as ex:                     # an older signature
        print(f'[ftv2 memaudit] record_memory_history: {ex!r}', flush=True)


def _describe(r, child, ignore):
    """(text, the object to continue from) for a referrer r of child."""
    if isinstance(r, dict):
        key = next((k for k, v in r.items() if v is child), '?')
        for o in gc.get_referrers(r):
            if id(o) in ignore:
                continue
            if isinstance(o, types.ModuleType) and o.__dict__ is r:
                return f'global {o.__name__}.{key}', None
            if getattr(o, '__dict__', None) is r:
                return f'{type(o).__module__}.{type(o).__name__}.{key}', o
        return f'dict[{key!r}]', r
    if isinstance(r, types.FrameType):
        return f'frame {os.path.basename(r.f_code.co_filename)}:{r.f_lineno} {r.f_code.co_name}()', r
    if isinstance(r, types.TracebackType):
        return 'traceback', r
    if type(r).__name__ == 'cell':
        return 'closure cell', r
    if isinstance(r, types.FunctionType):
        return f'function {r.__module__}.{r.__qualname__}', r
    if isinstance(r, types.MethodType):
        return f'bound method {r.__qualname__}', r
    if isinstance(r, (list, tuple, set)):
        return f'{type(r).__name__}(len {len(r)})', r
    return f'{type(r).__module__}.{type(r).__name__}', r


def _chains(obj, ignore, depth=8, width=2):
    out = []
    ignore.add(id(out))

    def walk(o, path, d):
        if len(out) >= 6:
            return
        if d == depth:
            out.append(path + ['...'])
            return
        got = gc.get_referrers(o)
        ignore.add(id(got))
        refs = [r for r in got if id(r) not in ignore and not (isinstance(r, types.FrameType)
                                                               and r.f_code.co_filename == __file__)]
        ignore.add(id(refs))
        if not refs:
            out.append(path + ['(no referrer)'])
            return
        for i in range(min(width, len(refs))):          # no slice: a new list would itself refer to the referrers
            r = refs[i]
            text, nxt = _describe(r, o, ignore)
            if nxt is None or text.startswith('global'):
                out.append(path + [text])
            else:
                ignore.add(id(r))
                walk(nxt, path + [text], d + 1)
    walk(obj, [], 0)
    return out


def _ptrs(obj, out, depth=0):
    """Storage pointers of every CUDA tensor inside obj (tensors, dicts, lists, tuples, objects with a shadow / state)."""
    if depth > 6 or obj is None:
        return
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            out.add(obj.untyped_storage().data_ptr())
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _ptrs(k, out, depth + 1)
            _ptrs(v, out, depth + 1)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _ptrs(v, out, depth + 1)
    elif hasattr(obj, 'state') and isinstance(getattr(obj, 'state'), dict):      # an optimizer: its state
        _ptrs(obj.state, out, depth + 1)


def census(objs, kinds, limit=10000):
    """{id: object} of every object of the given types found as an item of a dict, list or tuple among objs. Each container is read
    on its own and never fails the census: a dict subclass whose values() is not the plain one (M3MA4 5875028: 'dictionary update
    sequence element #0 has length 22; 2 is required' ended the whole census), a container that changes while it is read
    (RuntimeError), or any other error skips that container only. dict.values is called unbound, so an override is not used."""
    import warnings
    found = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')          # isinstance on some module attributes warns (torch.distributed.reduce_op)
        _census_into(objs, kinds, limit, found)
    return found


def _census_into(objs, kinds, limit, found):
    for o in objs:
        try:
            if isinstance(o, dict):
                items = list(dict.values(o))[:limit]
            elif isinstance(o, (list, tuple)):
                items = list(o)[:limit]
            else:
                continue
        except Exception:
            continue
        for v in items:
            try:
                if isinstance(v, kinds) and id(v) not in found:
                    found[id(v)] = v
            except Exception:
                continue


HP_ATTRS = ('get_full_hp_param', 'get_full_hp_grad', 'set_full_hp_param', 'set_full_hp_grad')


def break_hp_cycles(objs=None):
    """Clear DeepSpeed's per-parameter links on every live Parameter that has them (2026-11-07, M3MA6 5875594 / M3MA7): ZeRO's
    link_hp_params gives each low-precision parameter p a tensor_fragment (p._hp_mapping) whose lp_fragment is a view made with
    autograd on (p.flatten().narrow(...), deepspeed/utils/tensor_fragment.py:327), so its grad_fn chain ends in p's AccumulateGrad
    node, which holds p from C++: p -> _hp_mapping -> lp_fragment -> grad_fn -> AccumulateGrad -> p. gc cannot see through the
    C++ nodes, so the cycle (with the flat bf16 buffers and the gradient_dict it reaches) is never freed unless DeepSpeed's own
    destroy() clears _hp_mapping, which it does only for the parameters of its bit16 groups. Returns the number cleared."""
    import gc
    import warnings
    if objs is None:
        objs = gc.get_objects()
    n = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')          # isinstance on some module attributes warns (torch.distributed.reduce_op)
        return n + _break_into(objs)


def _break_into(objs):
    n = 0
    for o in objs:
        try:
            if isinstance(o, torch.nn.Parameter) and o.__dict__.get('_hp_mapping') is not None:
                o._hp_mapping = None
                for nm in HP_ATTRS:
                    o.__dict__.pop(nm, None)
                n += 1
        except Exception:
            continue
    return n


def _hp_holders(objs, lines, limit=4):
    """The C++ holder test (the coordinator, 11-07): for live Parameters that still carry _hp_mapping, does lp_fragment's grad_fn
    chain end in an AccumulateGrad holding that same Parameter (a cycle through C++)? Also its refcount against its Python
    referrers, its hooks, any live optimizer that lists it, and the Python referrers of the fragment view and of its _base."""
    import gc
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        params = [o for o in objs if isinstance(o, torch.nn.Parameter) and o.__dict__.get('_hp_mapping') is not None]
        opts = [o for o in objs if isinstance(o, torch.optim.Optimizer)]
    lines.append(f'parameters still carrying _hp_mapping: {len(params)}, '
                 f'{sum(p.untyped_storage().nbytes() for p in params if p.is_cuda) / 1e9:.3f} GB of storages (shared ones counted '
                 f'once each per parameter); live optimizers: {len(opts)} ({sorted(set(type(o).__name__ for o in opts))})')
    for p in params[:limit]:
        frag = p.__dict__.get('_hp_mapping')
        lf = getattr(frag, 'lp_fragment', None)
        node, closes, hops = getattr(lf, 'grad_fn', None), False, 0
        while node is not None and hops < 8:
            if type(node).__name__ == 'AccumulateGrad':
                closes = getattr(node, 'variable', None) is p
                break
            nxt = [f for f, _ in node.next_functions if f is not None]
            node, hops = (nxt[0] if nxt else None), hops + 1
        refs = [r for r in gc.get_referrers(p) if r is not params and r is not objs and not isinstance(r, type(sys._getframe()))]
        in_opt = [type(o).__name__ for o in opts if any(p is q for g in o.param_groups for q in g.get('params', [])) or p in o.state]
        bref = gc.get_referrers(lf._base) if (lf is not None and getattr(lf, '_base', None) is not None) else []
        lines.append(f'-- held Parameter {tuple(p.shape)} {p.dtype}: refcount {sys.getrefcount(p) - 1}, Python referrers '
                     f'{len(refs)} ({sorted(set(type(r).__name__ for r in refs))}); lp_fragment grad_fn '
                     f'{type(getattr(lf, "grad_fn", None)).__name__} -> AccumulateGrad holding this Parameter: {closes} '
                     f'({hops} hops); hooks: backward {len(getattr(p, "_backward_hooks", None) or {})}, post-accumulate '
                     f'{len(getattr(p, "_post_accumulate_grad_hooks", None) or {})}; in a live optimizer: {in_opt}; '
                     f'Python referrers of the fragment view {len(gc.get_referrers(lf)) if lf is not None else None}, '
                     f'of its _base {len(bref)}')


def audit(rank, device, out_dir, gen, log=print, intended=(), streams=None):
    if os.environ.get('FTV2_MEM_AUDIT') != '1' or not torch.cuda.is_available():
        return
    walk_ranks = {int(x) for x in os.environ.get('FTV2_MEM_AUDIT_RANKS', '0').replace(';', ',').split(',') if x.strip()}
    lines = []
    gc.collect()
    alloc = torch.cuda.memory_allocated(device)
    objs = gc.get_objects()
    stor = {}
    for o in objs:
        try:
            if isinstance(o, torch.Tensor) and o.is_cuda:
                s = o.untyped_storage()
                k = s.data_ptr()
                if k not in stor or (stor[k][1].grad_fn is None and o.grad_fn is not None):
                    stor[k] = [s.nbytes(), o]
        except Exception:
            continue
    py = sum(v[0] for v in stor.values())
    graph = [v[1] for v in stor.values() if v[1].grad_fn is not None]
    try:
        _hp_holders(objs, lines)
    except Exception as ex:
        lines.append(f'C++ holder test: {ex!r}')
    keep_ptrs = set()                                  # what the recovery keeps on purpose (the kept state, the GPU shadow)
    for obj in intended:
        _ptrs(obj, keep_ptrs)
    rest = {k: v for k, v in stor.items() if k not in keep_ptrs}
    lines.append(f'rank {rank} gen {gen}: allocated {alloc / 1e9:.3f} GB; live Python CUDA tensors {len(stor)} storages '
                 f'{py / 1e9:.3f} GB; without a Python owner {(alloc - py) / 1e9:.3f} GB; tensors with a graph {len(graph)}; '
                 f'sys.last_traceback {getattr(sys, "last_traceback", None) is not None}; exc_info {sys.exc_info()[0]}; '
                 f'kept on purpose {sum(stor[k][0] for k in keep_ptrs if k in stor) / 1e9:.3f} GB; the rest {len(rest)} storages '
                 f'{sum(v[0] for v in rest.values()) / 1e9:.3f} GB')
    if rank in walk_ranks:
        top = sorted(rest.values(), key=lambda v: -v[0])[:16]     # the largest storages not kept on purpose
        seen = {id(v[1]) for v in top}
        top += [[t.untyped_storage().nbytes(), t] for t in graph[:6] if id(t) not in seen]
        ignore = {id(objs), id(stor), id(top), id(graph), id(lines), id(sys._getframe())}
        for v in stor.values():
            ignore.add(id(v))
        for nbytes, t in top:
            lines.append(f'-- {nbytes / 1e6:.1f} MB {tuple(t.shape)} {t.dtype} requires_grad={t.requires_grad} '
                         f'grad_fn={type(t.grad_fn).__name__ if t.grad_fn is not None else None}')
            for ch in _chains(t, set(ignore)):
                lines.append('     <- ' + ' <- '.join(ch))
    # the aborted world's objects that Python still holds (2026-11-07, the coordinator: is A's leftover a cut reduction's Work
    # inside an aborted backend?): ProcessGroup / Backend / Work objects found in Python containers, with a chain for a few
    try:
        import torch.distributed as dist
        from torch._C import _distributed_c10d as c10d_c
        kinds = tuple(t for t in (getattr(dist, 'ProcessGroup', None), getattr(c10d_c, 'Backend', None),
                                  getattr(c10d_c, 'Work', None)) if t is not None)
        found = census(objs, kinds)
        reg = set(id(g) for g in dist.distributed_c10d._world.pg_map)
        n_pg = sum(isinstance(v, kinds[0]) for v in found.values()) if kinds else 0
        lines.append(f'python-held distributed objects: {len(found)} ({n_pg} process groups; in torch registry now '
                     f'{sum(1 for v in found.values() if id(v) in reg)}); types '
                     f'{sorted(set(type(v).__name__ for v in found.values()))}')
        if rank in walk_ranks:
            ignore2 = {id(objs), id(stor), id(lines), id(found), id(sys._getframe())}
            for v in list(found.values())[:6]:
                lines.append(f'-- {type(v).__name__} (registered {id(v) in reg})')
                for ch in _chains(v, set(ignore2), depth=6):
                    lines.append('     <- ' + ' <- '.join(ch))
    except Exception as ex:
        lines.append(f'distributed census: {ex!r}')
    del objs
    try:
        snap = torch.cuda.memory._snapshot()
        sites, n = {}, 0
        by_state, pending = {}, {}                  # (2026-11-07, the coordinator: A's leftover in blocks freed but waiting on a
        for seg in snap.get('segments', []):        # stream's events, e.g. record_stream on an aborted NCCL stream?)
            if seg.get('device') not in (None, device.index):
                continue
            for b in seg.get('blocks', []):
                st = b.get('state')
                by_state[st] = by_state.get(st, 0) + b.get('size', 0)
                if st == 'active_pending_free':
                    k = seg.get('stream')
                    pending[k] = pending.get(k, 0) + b.get('size', 0)
        cur = torch.cuda.current_stream(device).cuda_stream
        names = {cur: 'current'}
        for nm, stv in (streams or {}).items():
            try:
                names[int(stv.cuda_stream)] = nm
            except Exception:
                pass
        lines.append('snapshot by block state: ' + ', '.join(f'{k} {v / 1e9:.3f} GB' for k, v in sorted(by_state.items(), key=str)))
        lines.append(f'pending-free by allocation stream (the current stream is {cur:#x}): '
                     + (', '.join((f'{k:#x} ({names.get(k, "not ours: an NCCL or other side stream")}): {v / 1e9:.3f} GB'
                                   if isinstance(k, int) else f'{k}: {v / 1e9:.3f} GB') for k, v in pending.items()) or 'none'))
        for seg in snap.get('segments', []):
            if seg.get('device') not in (None, device.index):
                continue
            for b in seg.get('blocks', []):
                if b.get('state') != 'active_allocated':
                    continue
                fr = [f for f in (b.get('frames') or []) if '/torch/' not in f.get('filename', '')][:3]
                key = ' < '.join(f"{os.path.basename(f['filename'])}:{f['line']} {f['name']}" for f in fr) or '(no stack)'
                sites[key] = sites.get(key, 0) + b.get('size', 0)
                n += 1
        lines.append(f'snapshot: {n} active blocks')
        for k, v in sorted(sites.items(), key=lambda kv: -kv[1])[:30]:
            lines.append(f'site {v / 1e6:9.1f} MB  {k}')
    except Exception as ex:
        lines.append(f'snapshot: {ex!r}')
    d = os.path.join(out_dir, 'ft', 'memaudit')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f'rank_{rank}_g{gen}.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    log(lines[0][:300])
