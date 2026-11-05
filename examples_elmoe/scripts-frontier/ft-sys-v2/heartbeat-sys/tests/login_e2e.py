#!/usr/bin/env python3
"""login_e2e.py (ft-sys-v2 heartbeat system tests, 2026-11-05): the whole heartbeat system on one login node, CPU only.

A real etcd of 3 members (local ports), three node agents (nA and nB train, nS is the standby), and the toy trainer on 4 ranks with
gloo (ranks 0-1 on "node" nA, 2-3 on nB; each node is a host-memory folder of its own). Times are shortened (beat 2 s, lease 6 s,
hang limit 8 s). Each case is one training run with one injected event and the expected verdict in etcd:
  hang      rank 3 stops before a call                      -> failed list: hang, suspect 3, on nB (two signals)
  kill      rank 1 killed by SIGKILL                        -> failed list: crash on nA, rank 1 the root
  pause     every rank sleeps 12 s at the same call         -> no failed list (equal counts)
  slow      rank 2 sleeps 300 ms before each call           -> no failed list (counts keep moving)
  lost      nB's agent and its two ranks killed             -> failed list: node lost, nB
  takeover  the coordinator's agent killed, then rank 0 hangs -> a warning (alive in training), a new coordinator, then a hang
            failure; an etcd member is killed on the way
    python3 login_e2e.py --python <python with torch> --etcd-bin <etcd> [--cases hang,kill,...] [--out <dir>]"""
import argparse
import resource
import json
import os
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HB = os.path.dirname(HERE)
sys.path.insert(0, HB)
from hb_etcd import Etcd

ap = argparse.ArgumentParser()
ap.add_argument('--python', required=True)
ap.add_argument('--etcd-bin', required=True)
ap.add_argument('--cases', default='hang,kill,oom,oomstuck,abort,exit,stuck,pause,slow,lost,takeover,silent')
ap.add_argument('--out', default=None)
a = ap.parse_args()

TAG = f'e2e{os.getpid()}'
OUT = a.out or os.path.join(os.environ.get('TMPDIR', '/tmp'), TAG)
os.makedirs(OUT, exist_ok=True)
SHM = {n: f'/dev/shm/ftv2_{TAG}_{n}' for n in ('nA', 'nB', 'nS')}
PORTS = [(23891, 23991), (23892, 23992), (23893, 23993)]
ENDPOINTS = ','.join(f'127.0.0.1:{c}' for c, _ in PORTS)
FAST = {'FTV2_FAST_BEAT_S': '2', 'FTV2_LEASE_S': '6', 'FTV2_RANK_BEAT_S': '1', 'FTV2_HANG_STILL_S': '8',
        'FTV2_LOST_CONFIRM_S': '4', 'FTV2_CRASH_SETTLE_S': '10', 'FTV2_SLOW_BEAT_S': '10', 'FTV2_EQUAL_STILL_S': '60'}
procs = {}
results = []


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)


def start_etcd():
    cluster = ','.join(f'm{i}=http://127.0.0.1:{p}' for i, (_, p) in enumerate(PORTS))
    for i, (c, p) in enumerate(PORTS):
        d = os.path.join(OUT, f'etcd_m{i}')
        shutil.rmtree(d, ignore_errors=True)
        cmd = [a.etcd_bin, '--name', f'm{i}', '--data-dir', d, '--listen-client-urls', f'http://127.0.0.1:{c}',
               '--advertise-client-urls', f'http://127.0.0.1:{c}', '--listen-peer-urls', f'http://127.0.0.1:{p}',
               '--initial-advertise-peer-urls', f'http://127.0.0.1:{p}', '--initial-cluster', cluster,
               '--initial-cluster-state', 'new', '--log-level', 'warn']
        procs[f'etcd{i}'] = subprocess.Popen(cmd, stdout=open(d + '.log', 'w'), stderr=subprocess.STDOUT)
    e = Etcd(ENDPOINTS.split(','))
    for _ in range(60):
        try:
            e.put('probe', '1')
            return e
        except Exception:
            time.sleep(0.5)
    raise SystemExit('etcd did not start')


def start_agent(node, role='train'):
    env = dict(os.environ, FTV2_NODE=node, FTV2_SHM_DIR=SHM[node], **FAST)
    os.makedirs(SHM[node], exist_ok=True)
    cmd = [a.python, '-u', os.path.join(HB, 'hb_agent.py'), '--etcd', ENDPOINTS, '--job-dir', OUT, '--role', role, '--gpus', '0']
    procs[f'agent_{node}'] = subprocess.Popen(cmd, env=env, stdout=open(os.path.join(OUT, f'agent_{node}.log'), 'a'),
                                              stderr=subprocess.STDOUT)


def start_run(run, inject='', steps=400):
    port = 29600 + int(time.time()) % 300
    out = []
    for r in range(4):
        node = 'nA' if r < 2 else 'nB'
        env = dict(os.environ, RANK=str(r), WORLD_SIZE='4', LOCAL_RANK=str(r % 2), MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port),
                   FTV2_NODE=node, FTV2_SHM_DIR=SHM[node], FTV2_RUN=run, FTV2_INJECT=inject, **FAST)
        cmd = [a.python, '-u', os.path.join(HERE, 'hb_toy_trainer.py'), '--backend', 'gloo', '--per-node', '2',
               '--micro-batches', '4', '--compute-ms', '30', '--steps', str(steps)]
        p = subprocess.Popen(cmd, env=env, stdout=open(os.path.join(OUT, f'{run}_rank{r}.log'), 'w'), stderr=subprocess.STDOUT,
                             preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))   # the abort case dumped 44 GB
        procs[f'{run}_rank{r}'] = p
        out.append(p)
    return out


def stop(ps):
    for p in ps:
        if p.poll() is None:
            p.kill()
    for p in ps:
        try:
            p.wait(timeout=10)
        except Exception:
            pass


def wait_training(run, limit=90):
    """Until all 4 ranks of run have written a line with calls > 0 (training has started)."""
    t0 = time.time()
    while time.time() - t0 < limit:
        ok = 0
        for r in range(4):
            node = 'nA' if r < 2 else 'nB'
            try:
                ln = json.load(open(os.path.join(SHM[node], f'rank_{r}.json')))
                ok += ln.get('run') == run and ln.get('calls', 0) > 0
            except (OSError, ValueError):
                pass
        if ok == 4:
            return round(time.time() - t0, 1)
        time.sleep(0.5)
    raise SystemExit(f'{run}: training did not start in {limit} s')


def failed_for(e, run):
    f, _ = e.range('failed/')
    for k, v in sorted(f.items()):
        d = json.loads(v)
        if d.get('run') == run:
            return k, d
    return None, None


def wait_failed(e, run, limit):
    t0 = time.time()
    while time.time() - t0 < limit:
        k, d = failed_for(e, run)
        if d:
            return k, d, round(time.time() - t0, 1)
        time.sleep(0.5)
    return None, None, None


def check(name, ok, detail):
    results.append((name, ok, detail))
    log(f'{"PASS" if ok else "FAIL"} {name}: {detail}')


def coordinator(e):
    v, _ = e.get('coordinator')
    return v


try:
    e = start_etcd()
    log(f'etcd up: {ENDPOINTS}; out {OUT}')
    for n, role in (('nA', 'train'), ('nB', 'train'), ('nS', 'standby')):
        start_agent(n, role)
    time.sleep(4)
    log(f'coordinator: {coordinator(e)}')
    cases = a.cases.split(',')
    import threading

    def record_beats():                 # every node beat in etcd, every 2 s, for the post-mortem of a failed case
        ee = Etcd(ENDPOINTS.split(','))
        with open(os.path.join(OUT, 'beats.log'), 'a') as fh:
            while True:
                try:
                    kv, rev = ee.range('node/')
                    fh.write(f'{time.strftime("%H:%M:%S")} rev {rev} ' + ' | '.join(f'{k} {v}' for k, v in sorted(kv.items())) + '\n')
                    fh.flush()
                except Exception as ex:
                    fh.write(f'{time.strftime("%H:%M:%S")} {ex}\n')
                time.sleep(2)
    threading.Thread(target=record_beats, daemon=True).start()

    if 'hang' in cases:
        ps = start_run('r1', 'hang:3:5')
        t_inj = time.time()
        k, d, dt = wait_failed(e, 'r1', 90)
        check('hang', bool(d) and d['kind'] == 'hang' and d['suspect'] == 3 and d['node'] == 'nB',
              f'{k} {d and {x: d[x] for x in ("kind", "node", "suspect", "why", "plan")}} after {dt} s')
        stop(ps)

    if 'kill' in cases:
        ps = start_run('r2', 'kill:1:5')
        k, d, dt = wait_failed(e, 'r2', 90)
        check('kill', bool(d) and d['kind'] == 'crash' and d['node'] == 'nA' and 1 in (d['suspect'] or []),
              f'{k} {d and {x: d[x] for x in ("kind", "node", "suspect", "why")}} after {dt} s')
        stop(ps)

    if 'pause' in cases:
        ps = start_run('r3', 'pause:all:5:12', steps=40)
        k, d, dt = wait_failed(e, 'r3', 45)
        ends = [p.wait(timeout=120) for p in ps]
        check('pause', d is None and ends == [0, 0, 0, 0], f'failure list {k}; exit codes {ends}')

    if 'slow' in cases:
        ps = start_run('r4', 'slow:2:3:300', steps=30)
        k, d, dt = wait_failed(e, 'r4', 40)
        ends = [p.wait(timeout=300) for p in ps]
        check('slow', d is None and ends == [0, 0, 0, 0], f'failure list {k}; exit codes {ends}')

    # stuck: every rank still with equal counts past the first steps -> stacks of every node, then kind stuck (the user, 11-05:
    # "for a stuck over the nccl or anything, I will give it 3 mins"; here FTV2_EQUAL_STILL_S of FAST)
    if 'stuck' in cases:
        ps = start_run('r12', 'pause:all:5:120')
        k, d, dt = wait_failed(e, 'r12', 150)
        check('stuck', bool(d) and d['kind'] == 'stuck' and d['plan'].startswith('stop the training'),
              f'{k} {d and {x: d[x] for x in ("kind", "node", "why", "plan")}} after {dt} s')
        stop(ps)

    # oomstuck: the process lives on after its exception, its exit blocked (HB3 5855079: PMPI_Finalize's barrier)
    for case, run, inject, node, rank, kind in (('oom', 'r7', 'oom:2:5', 'nB', 2, 'out of memory'),
                                                ('oomstuck', 'r10', 'oomstuck:2:5', 'nB', 2, 'out of memory'),
                                                ('abort', 'r8', 'abort:1:5', 'nA', 1, 'crash'),
                                                ('exit', 'r11', 'exit:1:5', 'nA', 1, 'exception')):
        if case not in cases:
            continue
        ps = start_run(run, inject)
        k, d, dt = wait_failed(e, run, 90)
        check(case, bool(d) and d['kind'] == kind and d['node'] == node and rank in (d['suspect'] or []),
              f'{k} {d and {x: d[x] for x in ("kind", "node", "suspect", "why", "plan")}} after {dt} s')
        stop(ps)

    if 'lost' in cases:
        ps = start_run('r5')
        log(f'r5 training started after {wait_training("r5")} s')
        time.sleep(4)
        log('losing node nB: its agent and its ranks 2, 3')
        for name in ('agent_nB', 'r5_rank2', 'r5_rank3'):
            procs[name].send_signal(signal.SIGKILL)
        k, d, dt = wait_failed(e, 'r5', 90)
        check('lost', bool(d) and d['kind'] == 'node lost' and d['node'] == 'nB',
              f'{k} {d and {x: d[x] for x in ("kind", "node", "why", "plan")}} after {dt} s')
        stop(ps)
        start_agent('nB')
        time.sleep(4)

    if 'takeover' in cases:
        ps = start_run('r6', 'hang:3:250')
        log(f'r6 training started after {wait_training("r6")} s')
        time.sleep(3)
        c = coordinator(e)
        log(f'killing the coordinator\'s agent ({c}) and etcd member m0; rank 3 on nB hangs at step 250')
        procs[f'agent_{c}'].send_signal(signal.SIGKILL)
        procs['etcd0'].send_signal(signal.SIGKILL)
        t_k = time.time()
        new = None
        while time.time() - t_k < 30:
            new = coordinator(e)
            if new and new != c:
                break
            time.sleep(0.5)
        check('takeover', bool(new) and new != c, f'old {c}, new {new} after {round(time.time() - t_k, 1)} s')
        time.sleep(8)
        rec = open(os.path.join(OUT, 'ft', 'record.log')).read()
        check('alive-warning', f'WARNING {c}: lease ended' in rec and not failed_for(e, 'r6')[1],
              f'record.log warns about {c}, and no failure list for r6 yet')
        k, d, dt = wait_failed(e, 'r6', 150)
        want = 'node lost' if c == 'nB' else 'hang'
        check('hang-after-takeover', bool(d) and d['kind'] == want and d['node'] == 'nB' and d['by'] == new,
              f'{k} {d and {x: d[x] for x in ("kind", "node", "suspect", "by", "why")}} after {dt} s (expected {want})')
        stop(ps)
        if 'silent' in cases and c != 'nS':
            ps = start_run('r9', f'hang:{0 if c == "nA" else 2}:150')
            log(f'r9 ({c} has no agent) started after {wait_training("r9")} s')
            k, d, dt = wait_failed(e, 'r9', 150)
            check('silent-then-stall', bool(d) and d['kind'] == 'node lost' and d['node'] == c,
                  f'{k} {d and {x: d[x] for x in ("kind", "node", "why")}} after {dt} s')
            stop(ps)
finally:
    for name, p in procs.items():
        if p.poll() is None:
            p.kill()
    for d in SHM.values():
        shutil.rmtree(d, ignore_errors=True)
    print('\nRESULTS')
    for name, ok, detail in results:
        print(f'  {"PASS" if ok else "FAIL"}  {name}: {detail}')
    print(f'record: {OUT}/ft/record.log')
