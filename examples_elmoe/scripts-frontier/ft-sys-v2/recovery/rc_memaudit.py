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
FTV2_MEM_AUDIT_RANKS (default '0') names the ranks that walk referrers (all ranks write the totals)."""
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


def audit(rank, device, out_dir, gen, log=print):
    if os.environ.get('FTV2_MEM_AUDIT') != '1' or not torch.cuda.is_available():
        return
    walk_ranks = {int(x) for x in os.environ.get('FTV2_MEM_AUDIT_RANKS', '0').split(',') if x.strip()}
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
    lines.append(f'rank {rank} gen {gen}: allocated {alloc / 1e9:.3f} GB; live Python CUDA tensors {len(stor)} storages '
                 f'{py / 1e9:.3f} GB; without a Python owner {(alloc - py) / 1e9:.3f} GB; tensors with a graph {len(graph)}; '
                 f'sys.last_traceback {getattr(sys, "last_traceback", None) is not None}; exc_info {sys.exc_info()[0]}')
    if rank in walk_ranks:
        top = sorted(stor.values(), key=lambda v: -v[0])[:12]
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
    del objs
    try:
        snap = torch.cuda.memory._snapshot()
        sites, n = {}, 0
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
