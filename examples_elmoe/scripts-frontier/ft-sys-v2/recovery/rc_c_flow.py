"""rc_c_flow.py (ft-sys-v2 recovery, milestone 3 C, 2026-11-06): the order of a C recovery, survivors and joiner (rc_pipe_c.py has
the engine side). FTV2_RESUME_LEVEL=C with FTV2_COMMIT=1 under a pipeline.

Survivor, monitor thread (c_on_failure): stop flag set (the main thread parks at its next communication call, or inside its next
stream wait, hb_monitor C mode / rc_pipe_c) -> revoke and abort the backends of the groups that hold a lost rank, except the world
group (no operation waits on it) -> wait until the main thread is parked -> report (commit position, pipeline counters) -> the
generation's store -> rc_swap (every group holding a lost rank gets a new backend in its old object) -> the plan (J of the lost
stage; per lost rank: start, what its neighbours hold) -> ready.
Survivor, main thread (c_hold, when parked): the partner of a lost rank sends the joiner its state (Keep's shadow of it, the bf16
weights, the sums at J) and seeds Keep and the commit copies with it; stage s - 1 sends the activation meta on the new pair and
re-sends its kept activations; stage s + 1 re-sends its kept gradients; then the stop flag is cleared and the cut instruction, if
any, runs again.
Joiner (c_join): torch's fake backend, the model and optimizer built without partners, the store, rc_swap(joiner) -> the state
from the partner in the same order -> Keep and commit copies seeded -> data from micro-batch `start` -> replay."""
import json
import os
import threading
import time
from datetime import timedelta

import torch
import torch.distributed as dist

import rc_pairs
import rc_stop
import rc_swap
from rc_commit import decide_microbatch_pipe
from rc_pipe_c import plan_for, use_pair_groups


def level():
    return os.environ.get('FTV2_RESUME_LEVEL', 'B').upper()


# ---------------------------------------------------------------------- survivor, monitor thread
RED_OPS = ('all_reduce', 'reduce_scatter_tensor', '_reduce_scatter_base', 'reduce_scatter', 'reduce')


def lost_groups(lost):
    """Every registered group that holds a lost rank, except the world group (no operation waits on it)."""
    lost = set(lost)
    world = dist.distributed_c10d._get_default_group()
    return [pg for pg, ranks in rc_swap.groups_in_order() if pg is not world and lost & set(ranks)]


def _threads(fns, limit_s):
    """Run each fn in a thread of its own; wait up to limit_s in all. Returns [seconds, or None if it did not return]."""
    out = [None] * len(fns)

    def run(i, fn):
        t0 = time.time()
        try:
            fn()
        except Exception:
            pass
        out[i] = round(time.time() - t0, 3)
    ths = [threading.Thread(target=run, args=(i, fn), daemon=True) for i, fn in enumerate(fns)]
    t0 = time.time()
    [t.start() for t in ths]
    for t in ths:
        t.join(timeout=max(0.1, limit_s - (time.time() - t0)))
    return out


def _backends(gs, device):
    bs = []
    for pg in gs:
        try:
            bs.append(pg._get_backend(device))
        except Exception:
            pass
    return bs


def revoke_these(gs, device, log, limit_s, why):
    """ncclCommRevoke on the current backend of each group, one thread each: the kernels in flight on it end, the resources stay
    (no device synchronization, so it returns even when another kernel keeps this GPU busy). Returns False if RCCL has no revoke."""
    lib = rc_stop.rccl()
    if lib is None or not hasattr(lib, 'ncclCommRevoke'):
        return False
    import ctypes

    def revoke(b):
        p = b._comm_ptr()
        if p:
            lib.ncclCommRevoke(ctypes.c_void_p(p), 0)
    bs = _backends(gs, device)
    secs = _threads([lambda b=b: revoke(b) for b in bs], limit_s)
    log(f'C stop: {len(gs)} groups {why}: revoked {len(bs)} backends in {secs} s (None = did not return)')
    return True


def abort_these(gs, device, log, limit_s, why):
    """Abort the current backend of each group in gs, all at the same moment (one thread each), within the limit; each abort's
    time is logged. Returns gs."""
    bs = _backends(gs, device)
    rc_stop._LEFT_BEHIND.append(bs)
    secs = _threads([b.abort for b in bs], limit_s)
    log(f'C: {len(gs)} groups {why}: aborted {len(bs)} backends, seconds per abort {secs} (None = did not return)')
    return gs


def abort_batch(cgs, lgs, device, log, limit_s):
    """The cut groups and the lost groups aborted all at the same moment, one thread each (each abort's time logged). An abort
    frees device memory (hipFree, which waits for an idle GPU), so it waits behind any kernel still running on that GPU; in one
    batch the abort that ends that kernel runs too. One after the other, a rank whose GPU a lost group's kernel keeps busy (a
    revoke does not end it) waits in the cut groups' abort until its limit (M3C0c 5870764: ranks 6, 7, 30 s)."""
    gs = list(cgs) + [pg for pg in lgs if pg not in cgs]
    bs = _backends(gs, device)
    rc_stop._LEFT_BEHIND.append(bs)
    secs = _threads([b.abort for b in bs], limit_s)
    log(f'C: {len(cgs)} cut and {len(gs) - len(cgs)} lost groups aborted at once; seconds per abort {secs} (None = did not '
        f'return)')
    return secs


def abort_groups(lost, device, log, limit_s):
    """Revoke and abort the backend of every registered group that holds a lost rank, except the world group. Returns the groups."""
    gs = lost_groups(lost)
    revoke_these(gs, device, log, float(os.environ.get('FTV2_REVOKE_WAIT_S', '10')), 'hold a lost rank')
    return abort_these(gs, device, log, limit_s, 'hold a lost rank')


# ---------------------------------------------------------------------- the operations a stop cut on healthy groups
def cut_entry(ctl, settle_s):
    """This survivor's part of the single list (non-blocking queries only; the monitor thread never waits on the GPU).
    Per tracked group: n = index of its last operation, m = its index at this rank's last commit, p = still running after the
    settle, op = its name, s = where this rank can start the group's operations again in place: a reduction it is parked in
    (c_sync after the launch: from its last commit, m + 1), a pipeline send or receive it is parked in (from the instruction's
    first operation on that group); None = it cannot."""
    S = ctl.hb.S
    keys = rc_swap.group_keys()
    lostg = set((getattr(ctl, 'c', None) or {}).get('aborted', ()))   # groups with a lost rank: the commit positions decide
    pend = {id(gr) for gr, _, _ in ctl.hb.pending_groups(settle_s, skip=lostg)}
    mark, imark, where = dict(S.get('c_mark') or {}), dict(S.get('c_inst_mark') or {}), S.get('c_where')
    out = {}
    for i, (gr, name, n, ended) in ctl.hb.last_ops().items():
        k = keys.get(i)
        if k is None or i in lostg:
            continue
        m = mark.get(i, 0)
        p = i in pend
        s = None
        if where == 'red' and name in RED_OPS and n > m:
            s = m + 1
        elif where == 'p2p' and name in ('broadcast', 'send', 'recv', 'isend', 'irecv') and p:
            s = imark.get(i, n - 1) + 1
        out[k] = {'n': n, 'm': m, 'p': p, 'op': name, 's': s}
    return {'groups': out, 'where': where}


def decide_cut(entries, can_redo_last=False):
    """Pure: the single list. entries = {rank: cut_entry}. A group is listed when an operation on it is still running on some
    member. Its cut index = the earliest place a running member can start again (min s); after the repair every member's next
    operation on it must be that index:
      'redo'      the member starts again at the cut index (its reduction from its last commit, or its pipeline instruction);
      'none'      its next operation is the cut index anyway (it parked before entering it);
      'redo_last' it committed past the cut index: its last commit must be taken back first (the engine cannot yet: level B).
    Anything else (an operation that cannot start again in place, such as an all-to-all inside a forward or backward, or a member
    behind the cut index) makes the failure fall back to level B. Returns {'mode', 'cuts', 'actions', 'why'}."""
    listed = {k for e in entries.values() for k, v in e['groups'].items() if v['p']}
    # a reduction starts again as a whole (from the member's last commit), so every group it touched since that commit goes
    # with it (the dense and the expert data-parallel groups of one reduce_gradients call)
    grow = True
    while grow:
        grow = False
        for e in entries.values():
            if e.get('where') != 'red':
                continue
            red = {k for k, v in e['groups'].items() if v['s'] is not None and v['n'] > v['m']}
            if red & listed and not red <= listed:
                listed |= red
                grow = True
    listed = sorted(listed)
    actions = {str(r): {} for r in entries}
    cuts, why = {}, []
    for k in listed:
        mem = {r: e['groups'][k] for r, e in entries.items() if k in e['groups']}
        run = {r: v for r, v in mem.items() if v['p']}
        bad = [f'{v["op"]} on rank {r}' for r, v in run.items() if v['s'] is None]
        if bad:
            why.append(f'{k}: cannot start again in place: {", ".join(bad)}')
            continue
        starts = [v['s'] for v in (run or mem).values() if v['s'] is not None]
        if not starts:
            why.append(f'{k}: no member can start it again')
            continue
        cut = min(starts)
        cuts[k] = cut
        for r, v in mem.items():
            nxt = v['n'] + 1
            if v['s'] == cut and (v['p'] or nxt > cut):
                a = 'redo'
            elif not v['p'] and nxt == cut:
                a = 'none'
            elif v['m'] >= cut:
                a = 'redo_last'
            else:
                why.append(f'{k}: rank {r} at {v["n"]} (last commit {v["m"]}, {v["op"]}) does not meet the cut index {cut}')
                continue
            actions[str(r)][k] = a
    if not can_redo_last:
        why += [f'{k}: rank {r} committed past the cut index (taking a commit back is not built)'
                for r, a in actions.items() for k, x in a.items() if x == 'redo_last']
    return {'mode': 'B' if why else 'C', 'cuts': cuts, 'actions': actions, 'why': why}


def single_list(ctl, e, g, survivors, limit=300):
    """Each survivor writes its entry; the lowest survivor decides (decide_cut) and writes the one list rc/<g>/cutlist; every
    survivor and the joiner read that list."""
    e.put(f'rc/{g}/cut/{ctl.rank}', json.dumps(cut_entry(ctl, float(os.environ.get('FTV2_C_SETTLE_S', '3')))))
    t0 = time.time()
    if ctl.rank == min(survivors):
        while True:
            got, _ = e.range(f'rc/{g}/cut/')
            reps = {int(k.rsplit('/', 1)[1]): json.loads(v) for k, v in got.items()}
            if set(survivors) <= set(reps):
                break
            if time.time() - t0 > limit:
                raise RuntimeError(f'C: list entries from {sorted(reps)} only')
            time.sleep(0.05)
        d = decide_cut({r: reps[r] for r in survivors})
        e.put(f'rc/{g}/cutlist', json.dumps(d))
        return d
    return wait_list(e, g, limit)


def wait_list(e, g, limit=600):
    t0 = time.time()
    while time.time() - t0 < limit:
        v, _ = e.get(f'rc/{g}/cutlist')
        if v is not None:
            return json.loads(v)
        time.sleep(0.05)
    raise RuntimeError('C: no list')


def _positions(ctl, e, g, survivors, inst, in_inst, limit=120):
    """The agreed instruction of this rank's stage-replica (the ranks that share its all-to-alls: FTV2_C_SR_SIZE consecutive
    ranks, 8 = EP 8 inside a node): the largest position any of them had reached when the stop began."""
    size = int(os.environ.get('FTV2_C_SR_SIZE', '8'))
    e.put(f'rc/{g}/pos/{ctl.rank}', json.dumps([inst, in_inst]))
    mine = [r for r in survivors if r // size == ctl.rank // size]
    t0 = time.time()
    while True:
        got, _ = e.range(f'rc/{g}/pos/')
        pos = {int(k.rsplit('/', 1)[1]): json.loads(v) for k, v in got.items()}
        if set(mine) <= set(pos) or time.time() - t0 > limit:
            break
        time.sleep(0.02)
    return max(pos[r][0] for r in mine if r in pos), {r: pos.get(r) for r in mine}


def c_on_failure(ctl, f):
    """Survivor, monitor thread. Returns 'B' when this failure falls back to level B (the caller runs B's stop), else None.
    No blocking call on the GPU before the cut groups are aborted: revokes and aborts run in threads of their own, the report
    reads host copies, the list reads non-blocking queries."""
    g = f['gen']
    lost = sorted(int(r) for r in f['ranks'])
    S = ctl.hb.S
    ctl.c = {'gen': g, 'lost': lost, 'ready': threading.Event(), 'parked': threading.Event(), 'done': False,
             'aborted': set(), 'abort_t': None, 'redo': set(), 'fallback': False}
    inst, in_inst = ctl.hb.stop_and_position()
    ctl.mark(g, 'list_seen')
    if ctl.device.type == 'cuda':
        torch.cuda.set_device(ctl.device)
    e = ctl.etcd()
    world = ctl.world
    survivors = [r for r in range(world) if r not in lost]
    lgs = lost_groups(lost)
    # the cut time first: an operation on these groups that began before it is cut (a kernel ended by the revoke below must
    # never be taken for a finished one)
    ctl.c['aborted'] = {id(pg) for pg in lgs}
    ctl.c['abort_t'] = time.time()
    # 1. the groups with a lost rank: revoked only (their kernels end; their resources are freed after the cut groups' aborts,
    #    because an abort frees device memory and so waits for a GPU that a cut healthy operation keeps busy: M3C1e 5869712)
    if not revoke_these(lgs, ctl.device, ctl.log, float(os.environ.get('FTV2_REVOKE_WAIT_S', '10')), 'hold a lost rank'):
        abort_these(lgs, ctl.device, ctl.log, float(os.environ.get('FTV2_ABORT_WAIT_S', '30')), 'hold a lost rank')
    ctl.mark(g, 'abort_done')
    # 2. the agreed instruction of my stage-replica, then wait for the main thread to park at it
    S['c_target'], sr = _positions(ctl, e, g, survivors, inst, in_inst)
    ctl.log(f'C stop: my position {inst}{" (inside)" if in_inst else ""}; my stage-replica parks after instruction '
            f'{S["c_target"]} ({sr})')
    if not ctl.c['parked'].wait(timeout=float(os.environ.get('FTV2_C_PARK_WAIT_S', '60'))):
        ctl.log('C stop: the main thread did not park in time')
    ctl.mark(g, 'main_caught')
    rep = dict(ctl.commit.report() if ctl.commit is not None else {}, c=ctl.pipec.report() if ctl.pipec else None,
               base_iteration=ctl.step_snap.get('iteration'), base=ctl.step_snap.get('counters'), inc=ctl.step_snap.get('inc'),
               where=S.get('c_where'), inst=S['c_inst'])
    e.put(f'rc/{g}/c/{ctl.rank}', json.dumps(rep))
    ctl.mark(g, 'reported')
    reports = _reports(e, g, survivors)                        # every survivor is parked from here on
    # 3. the single list of operations the stop cut on groups without a lost rank (M3C1e: ranks 16-23's stage-1 reduction)
    d = single_list(ctl, e, g, survivors)
    if d['mode'] == 'B':
        ctl.c_fallbacks = getattr(ctl, 'c_fallbacks', 0) + 1
        ctl.log(f'C falls back to level B ({ctl.c_fallbacks} so far in this run): {"; ".join(d["why"])[:600]}')
        ctl.c.update(fallback=True, done=True)
        ctl.c['ready'].set()
        return 'B'
    keys = rc_swap.group_keys()
    mine = d['actions'].get(str(ctl.rank), {})
    cgs = [pg for pg, _ in rc_swap.groups_in_order() if keys[id(pg)] in d['cuts']]
    ctl.c['redo'] = {id(pg) for pg in cgs if mine.get(keys[id(pg)]) == 'redo'}
    if d['cuts']:
        ctl.log(f'C cut: {d["cuts"]} (group: index); my actions {mine}')
    # 4. abort the cut groups and the lost groups at once (abort_batch); before it, no blocking call on the GPU
    abort_batch(cgs, lgs, ctl.device, ctl.log, float(os.environ.get('FTV2_ABORT_WAIT_S', '30')))
    ctl.mark(g, 'cut_done')
    store = _world_store(ctl, g, min(survivors), world)
    _log_groups(ctl, 'survivor')
    n = rc_swap.swap_groups(dist.PrefixStore(f'ftv2/c{g}/', store), lost, ctl.device, 'nccl',
                            timedelta(seconds=float(os.environ.get('FTV2_PG_TIMEOUT_S', '1800'))),
                            log=ctl.log if os.environ.get('FTV2_C_DEBUG') == '1' else None, extra=list(d['cuts']))
    first_collectives(ctl, [dist.distributed_c10d._get_default_group(), ctl.commit.pg])
    ctl.mark(g, 'world_up')
    ctl.c.update(plan=make_plan(reports, lost, world, ctl.commit.step if ctl.commit else None), store=store, swapped=n)
    ctl.log(f'C plan {g}: {json.dumps(ctl.c["plan"])[:400]}')
    ctl.mark(g, 'plan_done')
    ctl.c['ready'].set()
    return None


def first_collectives(ctl, groups, limit_s=None):
    """One all-reduce on each swapped group that batched point-to-point rounds use (the world: Keep and the state transfer; the
    commit's group: the copies), every member at once, before any such round: torch's batch_isend_irecv needs the group's
    first operation to involve every rank (M3C1d 5869419). On a stream of its own; its end is polled (an event query, never a
    synchronize). It runs after the cut groups are aborted, when no kernel keeps a GPU busy: RCCL's communicator setup inside
    the first all-reduce makes synchronous HIP calls (M3C1e 5869712)."""
    S = ctl.hb.S
    prev = S.get('c_repairing')
    S['c_repairing'] = True
    limit_s = float(os.environ.get('FTV2_PG_TIMEOUT_S', '1800')) if limit_s is None else limit_s
    try:
        st = torch.cuda.Stream(device=ctl.device)
        with torch.cuda.stream(st):
            for g in groups:
                t = torch.ones(1, device=ctl.device)
                dist.all_reduce(t, group=g)
            ev = torch.cuda.Event()
            ev.record(st)
        t0 = time.time()
        while not ev.query():
            if time.time() - t0 > limit_s:
                raise RuntimeError(f'C: the first all-reduces did not end in {limit_s:.0f} s')
            time.sleep(0.001)
    finally:
        S['c_repairing'] = prev


def _log_groups(ctl, who):
    """The registry in creation order, one line (the swap's keys need the same order of groups with equal ranks on both sides)."""
    try:
        items = [f'{len(r)}:{min(r)}-{max(r)}:{dist.distributed_c10d._world.pg_map[pg][0]}' for pg, r in rc_swap.groups_in_order()]
        ctl.log(f'C groups ({who}, {len(items)}): {" ".join(items)}'[:3000])
    except Exception as ex:
        ctl.log(f'C groups ({who}): {ex!r}')


def _world_store(ctl, g, host, world):
    e = ctl.etcd()
    key = f'cworld/{g}'
    to = timedelta(seconds=float(os.environ.get('FTV2_WORLD_TIMEOUT_S', '1800')))
    if ctl.rank == host:
        import rc_world
        ip = rc_world.node_address()
        store = dist.TCPStore(ip, 0, world, is_master=True, timeout=to, wait_for_workers=False)
        e.put(key, f'{ip}:{store.port}')
    else:
        v = None
        while v is None:
            v, _ = e.get(key)
            time.sleep(0.05)
        ip, port = v.rsplit(':', 1)
        store = dist.TCPStore(ip, int(port), world, is_master=False, timeout=to)
    import rc_world
    rc_world._STORES.append(store)
    return store


def _reports(e, g, survivors, limit=300):
    t0 = time.time()
    while time.time() - t0 < limit:
        got, _ = e.range(f'rc/{g}/c/')
        reps = {int(k.rsplit('/', 1)[1]): json.loads(v) for k, v in got.items()}
        if set(survivors) <= set(reps):
            return reps
        time.sleep(0.1)
    raise RuntimeError(f'C: reports from {sorted(reps)} only')


def make_plan(reports, lost, world, step):
    """Pure: J of the lost stage (decide_microbatch_pipe on the commit reports), then plan_for each lost rank."""
    stride = int(os.environ.get('FTV2_PARTNER_STRIDE', '8'))
    if step is None:
        step = next(r['commit_step'] for r in reports.values() if r.get('commit_step') is not None)
    ok, J, Js, why = decide_microbatch_pipe(reports, lost, world, stride, step)
    if not ok:
        raise RuntimeError(f'C: no commit to resume from: {why}')
    stages = rc_pairs.pp()
    R = world // stages
    s = lost[0] // R
    plans = [dict(plan_for(L, reports, world, stages, Js[s]), partner=rc_pairs.partner(L, world, stride)) for L in lost]
    return {'step': step, 'stage': s, 'J': Js[s], 'Js': {str(k): v for k, v in Js.items()}, 'why': why, 'joiners': plans}


# ---------------------------------------------------------------------- survivor, main thread
def c_hold(ctl):
    """The main thread, parked at a communication call, an instruction boundary or inside a stream wait: do this rank's part of
    the repair once. If the failure fell back to level B, leave the step (StopForRecovery) for B's recovery."""
    c = getattr(ctl, 'c', None)
    S = ctl.hb.S
    if c is not None and c.get('fallback') and S.get('stop'):
        raise ctl.hb.StopForRecovery('C fell back to level B')
    if c is None or c['done']:
        return
    c['parked'].set()
    c['ready'].wait()
    if c.get('fallback'):
        raise ctl.hb.StopForRecovery('C fell back to level B')
    S['c_repairing'] = True
    try:
        plan, store = c['plan'], c['store']
        for p in plan['joiners']:
            if p['partner'] == ctl.rank:
                _send_state(ctl, p, plan)
            if p['next'] == ctl.rank and p['grad_end'] > p['start']:
                ctl.pipec.resend('grad', p['rank'], p['start'], p['grad_end'], dist.PrefixStore(f'ftv2/c{c["gen"]}/', store),
                                 p['grad_key'])
            if p['prev'] == ctl.rank:
                eng = ctl.adapter.engine
                kept = ctl.pipec.kept_act
                last = kept[next(reversed(kept))]
                eng._send_tensor_meta(last[0] if len(last) == 1 else tuple(last), eng.next_stage)
                if p['act_end'] > p['start']:
                    ctl.pipec.resend('act', p['rank'], p['start'], p['act_end'],
                                     dist.PrefixStore(f'ftv2/c{c["gen"]}/', store), p['act_key'])
        ctl.mark(c['gen'], 'ready')
    finally:
        S['c_repairing'] = False
        S['stop'] = False
        c['done'] = True
        ctl.recovered += 1
        ctl.log(f'C recovery {c["gen"]}: my part done; this rank goes on from where it stopped')


def _send_state(ctl, p, plan):
    """The partner -> the joiner, in this order (c_join receives in the same order): Keep's shadow of the lost rank; the bf16
    flat groups; the sums at J (the copy); then Keep and the commit copies seeded pairwise."""
    L = p['rank']
    ctl.log(f'C: sending rank {L} its state (Keep shadow, bf16 weights, sums at micro-batch {plan["J"]})')
    ctl.keep.send_shadow(L)
    ctl.log('C: shadow sent')
    z = ctl.adapter.zero
    for t in z.bit16_groups_flat:
        dist.send(t, L)
    ctl.log('C: weights sent')
    cp = ctl.commit.copy_of(plan['step'], plan['J'])
    if plan['J'] > 0:
        if cp is None:
            raise RuntimeError(f'C: no copy of rank {L} at [{plan["step"]}, {plan["J"]}]')
        for t in cp:
            dist.send(t, L)
    torch.cuda.synchronize(ctl.device)
    ctl.keep.seed()
    ctl.commit.seed(ctl.commit.step, ctl.commit.c, ctl.commit.snap_cur)
    ctl.log(f'C: state, weights and sums at micro-batch {plan["J"]} sent to rank {L}; Keep and copies seeded with it')


# ---------------------------------------------------------------------- the joiner
def c_join(ctl, f):
    """The standby process becomes the lost rank: fake-backend build, swap, state from the partner, replay. Returns the box."""
    from torch.testing._internal.distributed.fake_pg import FakeStore
    from adapters.deepspeed_zero import ZeroAdapter
    from rc_keep import Keep
    g = f['gen']
    lost = sorted(int(r) for r in f['ranks'])
    world = int(os.environ.get('FTV2_WORLD', ctl.world or 0)) or ctl.world
    ctl.world = world
    survivors = [r for r in range(world) if r not in lost]
    e = ctl.etcd()
    use_pair_groups()
    dist.init_process_group('fake', store=FakeStore(), rank=ctl.rank, world_size=world)
    ctl.tr.reinit_parallel(ZeroAdapter)
    model, optimizer, sched = ctl.tr.build_model()
    ctl.mark(g, 'model_built')
    adapter = ZeroAdapter(model[0])
    ctl._ds_fixes(adapter)
    ctl.c_pg = dist.new_group(list(range(world)))     # the commit's own group, created where the survivors created theirs
    _log_groups(ctl, 'joiner')
    store = _world_store(ctl, g, min(survivors), world)
    rc_swap.swap_groups(dist.PrefixStore(f'ftv2/c{g}/', store), lost, ctl.device, 'nccl',
                        timedelta(seconds=float(os.environ.get('FTV2_PG_TIMEOUT_S', '1800'))), joiner=True, log=ctl.log)
    first_collectives(ctl, [dist.distributed_c10d._get_default_group(), ctl.c_pg])
    ctl.mark(g, 'world_up')
    reports = _reports(e, g, survivors)
    plan = make_plan(reports, lost, world, None)
    p = next(x for x in plan['joiners'] if x['rank'] == ctl.rank)
    holder = p['partner']
    ctl.log(f'C join {g}: plan {p}; receiving from rank {holder}')
    states = Keep.recv_state(adapter.slots(), holder, ctl.device)
    adapter.put_state(states)
    ctl.log('C join: state received')
    z = adapter.zero
    for t in z.bit16_groups_flat:
        dist.recv(t, holder)
    ctl.log('C join: weights received')
    flats = None
    if plan['J'] > 0:
        flats = [torch.empty(adapter.zero2_partition_size(i), dtype=adapter.zero2_dtype(), device=ctl.device)
                 for i in adapter.zero2_groups()]
        for t in flats:
            dist.recv(t, holder)
    torch.cuda.synchronize(ctl.device)
    for n, i in enumerate(adapter.zero2_groups()):
        adapter.zero2_set_partial(i, None if flats is None else flats[n])
    ctl.mark(g, 'state_moved')
    step = plan['step']
    hrep = reports[holder]
    base, inc = hrep['base'], hrep['inc']                      # the counters at this step's start (as in _join)
    ctl.tr.set_counters(base['iteration'], base['samples'], base['tokens'])
    ctl.tr.set_scheduler(sched, base['samples'], max(0, base['tokens'] - inc['tokens']))
    adapter.set_counters(base['iteration'], base['samples'], base['iteration'] * inc.get('micro', 1))
    keep = Keep(adapter, ctl.rank, world, ctl.device, ctl.log, step,
                out_dir=os.path.join(ctl.out, 'ft') if ctl.out else None)
    keep.seed()
    keep.install()
    ctl.log('C join: Keep seeded')
    ctl.c_replay = (p, store, g)
    ctl._start_commit(adapter, step, plan['J'], flats)
    train_it, valid_it, test_it = ctl.tr.build_data()
    if train_it is not None:
        for _ in range(p['start']):
            next(train_it)
    ctl.adapter, ctl.keep = adapter, keep
    ctl.hb.S['on_step'] = ctl._on_step
    ctl.recovered += 1
    ctl.tr.reset_timers()
    ctl.hb.begin_generation(f'{ctl.base_run}.g{g}', rank=ctl.rank, role='train')
    ctl.role = 'train'
    ctl.mark(g, 'ready')
    ctl.log(f'C join {g}: rank {ctl.rank} built alone (fake backend), swapped in, state from rank {holder}, replays step {step} '
            f'from micro-batch {p["start"]} (commit {plan["J"]})')
    return [model, optimizer, sched, train_it, valid_it]
