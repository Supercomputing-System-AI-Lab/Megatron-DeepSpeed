"""rc_swap.py (ft-sys-v2 recovery, milestone 3 C, 2026-11-06): new backends inside the existing process-group objects (m3_map
question 6, option 2), so that every object holding a group (the engine, ZeRO, the MoE layers, the pipeline, Megatron's registry)
keeps working without being re-pointed.

Why not _register_backend with the group's own type: ProcessGroup::setBackend (torch 2.13, ProcessGroup.hpp:912-935) keeps an
existing backend of the same type. A backend registered under a type the group does not use yet, then made the default, carries
every collective of that group object (tests/cpu_pg_swap_test.py, gloo, 2 losses in a row).

swap_groups(store, lost, device, backend, timeout, joiner)
  survivors (joiner=False): every registered group that holds a lost rank gets a new backend; the others keep theirs (their
                            communicators never saw the lost node);
  a joiner (joiner=True):   every registered group gets one (it built its groups with torch's "fake" backend, so that its model
                            and optimizer build, which runs collectives, needs no partner while the survivors keep training).
The two sides meet through the store: key = <hash of the group's global ranks>/<its creation order among the groups with the same
ranks>. Both sides create their groups in the same order (the same initialization code), so the keys match."""
import hashlib

import torch
import torch.distributed as dist

_USED = {}                # id(group) -> backend types already registered on it (setBackend never replaces one)
_ORDER = ('CUSTOM', 'UNDEFINED', 'XCCL', 'UCC', 'MPI', 'NCCL', 'GLOO')
_NAME_TYPE = {'nccl': 'NCCL', 'gloo': 'GLOO', 'custom': 'CUSTOM', 'fake': 'CUSTOM', 'undefined': 'UNDEFINED', 'mpi': 'MPI',
              'ucc': 'UCC', 'xccl': 'XCCL'}


def _free_type(pg):
    used = _USED.setdefault(id(pg), set())
    try:
        used.add(_NAME_TYPE.get(str(pg._get_backend_name()).lower(), 'CUSTOM'))
    except Exception:
        pass
    for t in _ORDER:
        if t not in used:
            used.add(t)
            return getattr(dist.ProcessGroup.BackendType, t)
    raise RuntimeError('no backend type left to register a new backend on this group')


def _new_backend(backend, store, rank, size, timeout):
    if backend == 'gloo':
        return dist.ProcessGroupGloo(store, rank, size, timeout)
    opts = dist.ProcessGroupNCCL.Options()
    opts._timeout = timeout
    return dist.ProcessGroupNCCL(store, rank, size, opts)


def groups_in_order():
    """[(group, global ranks)] in creation order (the registry is an insertion-ordered dict; the world group first)."""
    c = dist.distributed_c10d
    return [(pg, list(c.get_process_group_ranks(pg))) for pg in list(c._world.pg_map.keys())]


def group_keys():
    """{id(group): '<hash of its global ranks>/<its creation order among the groups with the same ranks>'}: the same on every
    member (both sides create their groups in the same order)."""
    seen, out = {}, {}
    for pg, ranks in groups_in_order():
        h = hashlib.md5(','.join(map(str, sorted(ranks))).encode()).hexdigest()[:12]
        occ = seen.get(h, 0)
        seen[h] = occ + 1
        out[id(pg)] = f'{h}/{occ}'
    return out


def swap_groups(store, lost, device, backend='nccl', timeout=None, joiner=False, log=None, extra=()):
    """extra: keys (group_keys) of groups without a lost rank that get a new backend as well (survivors: a group whose last
    operation was cut because one of its members parked before entering it, rc_c_flow.cut_healthy_groups)."""
    me = dist.get_rank()
    lost = set(int(r) for r in lost)
    extra = set(extra)
    keys = group_keys()
    n = 0
    for pg, ranks in groups_in_order():
        if not joiner and not (lost & set(ranks)) and keys[id(pg)] not in extra:
            continue                                   # no lost member and nothing cut: its communicators are intact, keep them
        key = keys[id(pg)] + '/'
        b = _new_backend(backend, dist.PrefixStore(key, store), ranks.index(me), len(ranks), timeout)
        t = _free_type(pg)
        pg._register_backend(device, t, b)
        pg._set_default_backend(t)
        if joiner:                                     # the registry's name: dist.get_backend(group) reports the real backend
            c = dist.distributed_c10d
            old = c._world.pg_map.get(pg)
            if old is not None:
                c._world.pg_map[pg] = (backend, old[1])
        n += 1
        if log:
            log(f'swap: group of {len(ranks)} ranks ({key[:-1]}) has a new {backend} backend (type {t})')
    return n
