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
def abort_groups(lost, device, log, limit_s):
    """Revoke and abort the backend of every registered group that holds a lost rank, except the world group. Returns the groups."""
    lost = set(lost)
    world = dist.distributed_c10d._get_default_group()
    gs = [pg for pg, ranks in rc_swap.groups_in_order() if pg is not world and lost & set(ranks)]
    bs = []
    for pg in gs:
        try:
            bs.append(pg._get_backend(device))
        except Exception:
            pass
    lib = rc_stop.rccl()
    import ctypes

    def revoke(b):
        try:
            p = b._comm_ptr()
            if p:
                lib.ncclCommRevoke(ctypes.c_void_p(p), 0)
        except Exception:
            pass
    t0 = time.time()
    if lib is not None:
        # one thread each, with a limit: ncclCommRevoke blocks on a point-to-point communicator with a send or receive in flight
        # to the lost peer (M3C1a 5868473: the monitor thread stayed in commRevokeAsync)
        rth = [threading.Thread(target=revoke, args=(b,), daemon=True) for b in bs]
        [t.start() for t in rth]
        rlim = float(os.environ.get('FTV2_REVOKE_WAIT_S', '10'))
        for t in rth:
            t.join(timeout=max(0.1, rlim - (time.time() - t0)))
    t0 = time.time()
    ths = [threading.Thread(target=b.abort, daemon=True) for b in bs]
    [t.start() for t in ths]
    for t in ths:
        t.join(timeout=max(0.1, limit_s - (time.time() - t0)))
    stuck = sum(t.is_alive() for t in ths)
    rc_stop._LEFT_BEHIND.append(bs)
    log(f'C stop: {len(gs)} groups hold a lost rank; aborted {len(bs)} backends, {stuck} did not return, {time.time() - t0:.2f} s')
    return gs


def c_on_failure(ctl, f):
    g = f['gen']
    lost = sorted(int(r) for r in f['ranks'])
    S = ctl.hb.S
    ctl.c = {'gen': g, 'lost': lost, 'ready': threading.Event(), 'parked': threading.Event(), 'done': False,
             'aborted': set(), 'abort_t': None}
    S['stop'] = True
    ctl.mark(g, 'list_seen')
    if ctl.device.type == 'cuda':
        torch.cuda.set_device(ctl.device)
    gs = abort_groups(lost, ctl.device, ctl.log, float(os.environ.get('FTV2_ABORT_WAIT_S', '30')))
    ctl.c['aborted'] = {id(pg) for pg in gs}
    ctl.c['abort_t'] = time.time()
    ctl.mark(g, 'abort_done')
    if not ctl.c['parked'].wait(timeout=float(os.environ.get('FTV2_C_PARK_WAIT_S', '300'))):
        ctl.log('C stop: the main thread did not park in time')
    ctl.mark(g, 'main_caught')
    e = ctl.etcd()
    rep = dict(ctl.commit.report() if ctl.commit is not None else {}, c=ctl.pipec.report() if ctl.pipec else None,
               base_iteration=ctl.step_snap.get('iteration'), base=ctl.step_snap.get('counters'), inc=ctl.step_snap.get('inc'))
    e.put(f'rc/{g}/c/{ctl.rank}', json.dumps(rep))
    ctl.mark(g, 'reported')
    world = ctl.world
    survivors = [r for r in range(world) if r not in lost]
    store = _world_store(ctl, g, min(survivors), world)
    _log_groups(ctl, 'survivor')
    n = rc_swap.swap_groups(dist.PrefixStore(f'ftv2/c{g}/', store), lost, ctl.device, 'nccl',
                            timedelta(seconds=float(os.environ.get('FTV2_PG_TIMEOUT_S', '1800'))),
                            log=ctl.log if os.environ.get('FTV2_C_DEBUG') == '1' else None)
    ctl.mark(g, 'world_up')
    reports = _reports(e, g, survivors)
    ctl.c.update(plan=make_plan(reports, lost, world, ctl.commit.step if ctl.commit else None), store=store, swapped=n)
    ctl.log(f'C plan {g}: {json.dumps(ctl.c["plan"])[:400]}')
    ctl.mark(g, 'plan_done')
    ctl.c['ready'].set()


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
    """The main thread, parked at a communication call or inside a stream wait: do this rank's part of the repair once."""
    c = getattr(ctl, 'c', None)
    if c is None or c['done']:
        return
    S = ctl.hb.S
    c['parked'].set()
    c['ready'].wait()
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
    ctl.keep.send_shadow(L)
    z = ctl.adapter.zero
    for t in z.bit16_groups_flat:
        dist.send(t, L)
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
    ctl.mark(g, 'world_up')
    reports = _reports(e, g, survivors)
    plan = make_plan(reports, lost, world, None)
    p = next(x for x in plan['joiners'] if x['rank'] == ctl.rank)
    holder = p['partner']
    states = Keep.recv_state(adapter.slots(), holder, ctl.device)
    adapter.put_state(states)
    z = adapter.zero
    for t in z.bit16_groups_flat:
        dist.recv(t, holder)
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
