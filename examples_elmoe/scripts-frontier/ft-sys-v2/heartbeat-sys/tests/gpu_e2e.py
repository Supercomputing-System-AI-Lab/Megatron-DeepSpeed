#!/usr/bin/env python3
"""gpu_e2e.py (ft-sys-v2 heartbeat system tests, 2026-11-05): the heartbeat system on Frontier GPUs, run from a batch script.

Nodes of the allocation: the first three train (24 ranks, all-to-all inside each node, all-reduce across nodes, RCCL), the fourth
is the standby. etcd runs on the first three (one member each, client port 2379 on the node's hsn0 address), one agent on every
node, each in its own srun step (--overlap). Each case is one training run (srun step) with one injected event; the expected
failure list (or none) is read from etcd, then the run is aborted (no recovery yet: the user, 11-05).
  hang   rank 13 stops before a call          -> hang, suspect 13, node 1          kill   rank 5 SIGKILL     -> crash, node 0
  oom    rank 21 allocates 2 TB on its GPU   -> out of memory, node 2             abort  rank 9 SIGABRT     -> crash, node 1
  pause  every rank sleeps 25 s at one call  -> nothing                            slow   rank 17 +200 ms/call -> nothing
  lost   node 2: its agent, etcd member and ranks killed               -> node lost, node 2
  takeover  the coordinator's agent killed; later rank 12 hangs        -> a warning for that node, a new coordinator, a hang
    python3 gpu_e2e.py --python <python> --preload "<libs>" --out <dir> [--cases ...]"""
import argparse
import json
import os
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
ap.add_argument('--preload', default='')
ap.add_argument('--etcd-bin', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--cases', default='hang,kill,oom,abort,pause,slow,lost,takeover,collide')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
JOB = os.environ['SLURM_JOB_ID']
results, procs = [], {}


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


nodes = sh('scontrol show hostnames $SLURM_JOB_NODELIST').stdout.split()
assert len(nodes) >= 4, nodes
TRAIN, STANDBY = nodes[:3], nodes[3]
ips = {}
for n in nodes:
    r = sh(f"srun --overlap -N1 -n1 -w {n} bash -c \"ip -4 addr show hsn0 | awk '/inet /{{print \\$2}}' | cut -d/ -f1 | head -1\"")
    ips[n] = r.stdout.strip()
log(f'nodes {nodes}; hsn0 {ips}')
ENDPOINTS = ','.join(f'{ips[n]}:2379' for n in TRAIN)


def start_etcd():
    cluster = ','.join(f'm{i}=http://{ips[n]}:2380' for i, n in enumerate(TRAIN))
    for i, n in enumerate(TRAIN):
        ip = ips[n]
        # without the proxy variables: etcd's peer transport would send through them (hb_start.sh, HB2 5854331)
        cmd = (f'srun --overlap -N1 -n1 -w {n} env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy '
               f'-u ALL_PROXY {a.etcd_bin} --name m{i} --data-dir /dev/shm/ftv2_etcd_{JOB}_m{i} '
               f'--listen-client-urls http://{ip}:2379 --advertise-client-urls http://{ip}:2379 '
               f'--listen-peer-urls http://{ip}:2380 --initial-advertise-peer-urls http://{ip}:2380 '
               f'--initial-cluster {cluster} --initial-cluster-state new --log-level warn')
        procs[f'etcd_{n}'] = subprocess.Popen(cmd, shell=True, stdout=open(f'{a.out}/etcd_m{i}.log', 'w'), stderr=subprocess.STDOUT)
    e = Etcd(ENDPOINTS.split(','))
    for _ in range(120):
        try:
            e.put('probe', '1')
            return e
        except Exception:
            time.sleep(0.5)
    raise SystemExit('etcd did not start')


def start_agent(n, role='train'):
    cmd = (f'srun --overlap -N1 -n1 -c2 -w {n} {a.python} -I -u {HB}/hb_agent.py --etcd {ENDPOINTS} --job-dir {a.out} '
           f'--role {role} --gpus 8')
    procs[f'agent_{n}'] = subprocess.Popen(cmd, shell=True, stdout=open(f'{a.out}/agent_{n}.log', 'a'), stderr=subprocess.STDOUT)


def kill_on(n, pattern):
    """SIGKILL the processes matching pattern on node n ([x]yz patterns, so pkill does not match its own command line)."""
    sh(f"srun --overlap -N1 -n1 -w {n} bash -c \"pkill -9 -u $USER -f '{pattern}'\"")


def start_run(run, inject='', steps=100000):
    port = 29500 + len([k for k in procs if k.startswith('run_')]) * 7
    env = (f'env LD_PRELOAD="{a.preload}" MASTER_ADDR={ips[TRAIN[0]]} MASTER_PORT={port} FTV2_RUN={run} FTV2_INJECT={inject}')
    cmd = (f'srun --overlap -N3 -n24 -c6 --ntasks-per-node=8 --gpus-per-node=8 --kill-on-bad-exit=0 --gpu-bind=none '
           f'-w {",".join(TRAIN)} {env} {a.python} -I -u {HERE}/hb_toy_trainer.py --backend nccl --per-node 8 '
           f'--micro-batches 8 --steps {steps}')
    p = subprocess.Popen(cmd, shell=True, stdout=open(f'{a.out}/{run}.log', 'w'), stderr=subprocess.STDOUT, start_new_session=True)
    procs[f'run_{run}'] = p
    return p


def stop(p):
    """Abort a training run: SIGTERM to the srun of the step (Slurm ends its tasks), SIGKILL after 30 s."""
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=30)


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
        time.sleep(1)
    return None, None, None


def wait_training(e, run, limit=180):
    """Until every training node's beat in etcd shows this run with counts above zero."""
    t0 = time.time()
    while time.time() - t0 < limit:
        kv, _ = e.range('node/')
        b = [json.loads(v) for v in kv.values()]
        ok = [x for x in b if x.get('run') == run and (x.get('low') or 0) > 0]
        if len(ok) >= 3:
            return round(time.time() - t0, 1)
        time.sleep(1)
    raise SystemExit(f'{run}: training did not start in {limit} s')


def check(name, ok, detail):
    results.append((name, ok, detail))
    log(f'{"PASS" if ok else "FAIL"} {name}: {detail}')


def show(d):
    return d and {x: d.get(x) for x in ('kind', 'node', 'suspect', 'why', 'plan', 'by')}


try:
    e = start_etcd()
    log(f'etcd up: {ENDPOINTS}')
    for n in TRAIN:
        start_agent(n)
    start_agent(STANDBY, 'standby')
    time.sleep(8)
    log(f'coordinator: {e.get("coordinator")[0]}')
    cases = a.cases.split(',')
    node_of = lambda r: TRAIN[r // 8]
    for case, run, inject, rank, kind in (('hang', 'hang', 'hang:13:20', 13, 'hang'), ('kill', 'kill', 'kill:5:20', 5, 'crash'),
                                          ('oom', 'oom', 'oom:21:20', 21, 'out of memory'), ('abort', 'abort', 'abort:9:20', 9, 'crash')):
        if case not in cases:
            continue
        p = start_run(run, inject)
        k, d, dt = wait_failed(e, run, 240)
        sus = d and d.get('suspect')
        sus = sus if isinstance(sus, list) else [sus]
        check(case, bool(d) and d['kind'] == kind and d['node'] == node_of(rank) and rank in sus, f'{k} {show(d)} after {dt} s')
        stop(p)
    for case, run, inject, steps in (('pause', 'pause', 'pause:all:20:25', 120), ('slow', 'slow', 'slow:17:10:200', 60)):
        if case not in cases:
            continue
        p = start_run(run, inject, steps=steps)
        t0 = time.time()
        while p.poll() is None and time.time() - t0 < 300:
            time.sleep(2)
        k, d = failed_for(e, run)
        check(case, d is None and p.poll() == 0, f'failure list {k}; srun exit {p.poll()} after {round(time.time() - t0)} s')
        stop(p)
    if 'lost' in cases:
        p = start_run('lost')
        log(f'lost: training started after {wait_training(e, "lost")} s')
        time.sleep(10)
        n2 = TRAIN[2]
        log(f'losing {n2}: its agent, its etcd member and its ranks')
        kill_on(n2, '[h]b_agent.py')
        kill_on(n2, '[e]tcd --name')
        kill_on(n2, '[h]b_toy_trainer.py')
        k, d, dt = wait_failed(e, 'lost', 120)
        check('lost', bool(d) and d['kind'] == 'node lost' and d['node'] == n2, f'{k} {show(d)} after {dt} s')
        stop(p)
        start_agent(n2)
        time.sleep(8)
    if 'takeover' in cases:
        p = start_run('takeover', 'hang:12:3000')
        log(f'takeover: training started after {wait_training(e, "takeover")} s')
        c = e.get('coordinator')[0]
        log(f'killing the coordinator\'s agent on {c}; rank 12 (node 1) hangs at step 3000 (about 75 s later)')
        kill_on(c, '[h]b_agent.py')
        t0 = time.time()
        new = None
        while time.time() - t0 < 60:
            new = e.get('coordinator')[0]
            if new and new != c:
                break
            time.sleep(1)
        check('takeover', bool(new) and new != c, f'old {c}, new {new} after {round(time.time() - t0, 1)} s')
        time.sleep(15)
        rec = open(f'{a.out}/ft/record.log').read()
        check('alive-warning', f'WARNING {c}: lease ended' in rec and not failed_for(e, 'takeover')[1],
              f'record.log warns about {c}; no failure list yet')
        k, d, dt = wait_failed(e, 'takeover', 300)
        want = ('hang', TRAIN[1]) if c != TRAIN[1] else ('node lost', TRAIN[1])
        check('after-takeover', bool(d) and (d['kind'], d['node']) == want and d['by'] == new,
              f'{k} {show(d)} after {dt} s (expected {want})')
        stop(p)
        start_agent(c, 'standby' if c == STANDBY else 'train')   # back for the next case (HB1b 5855361: collide waited for it)
        time.sleep(8)
    if 'collide' in cases:
        # two things at once (found by HB1 5854211): an agent dies on one node while a rank hangs on another; the failure is the hang
        p = start_run('collide', 'hang:12:300')
        log(f'collide: training started after {wait_training(e, "collide")} s')
        c = e.get('coordinator')[0]
        victim = [n for n in (TRAIN[0], TRAIN[2]) if n != c][0]
        log(f'killing the agent on {victim} as rank 12 on {TRAIN[1]} hangs (step 300)')
        kill_on(victim, '[h]b_agent.py')
        k, d, dt = wait_failed(e, 'collide', 240)
        check('collide', bool(d) and d['kind'] == 'hang' and d['node'] == TRAIN[1] and d['suspect'] == 12,
              f'{k} {show(d)} after {dt} s (expected the hang of rank 12 on {TRAIN[1]}, not {victim})')
        stop(p)
finally:
    for name, p in procs.items():
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL) if name.startswith('run_') else p.kill()
            except Exception:
                pass
    for n in nodes:
        sh(f'srun --overlap -N1 -n1 -w {n} bash -c "rm -rf /dev/shm/ftv2_{JOB} /dev/shm/ftv2_etcd_{JOB}_m*"', timeout=60)
    print('\nRESULTS')
    for name, ok, detail in results:
        print(f'  {"PASS" if ok else "FAIL"}  {name}: {detail}')
    print(f'record: {a.out}/ft/record.log')
