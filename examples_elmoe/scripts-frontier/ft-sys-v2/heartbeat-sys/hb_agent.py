#!/usr/bin/env python3
"""hb_agent.py (ft-sys-v2 heartbeat system, 2026-11-05): the node agent, one process per node, CPU only.

Started by its own srun step, so a crashing training process cannot take it down (design map 1.4). It does:
  every BEAT (5 s)   read its ranks' lines from host memory; see which processes are gone; renew its etcd lease; write the node's
                     small beat to etcd (node/<node>, tied to the lease)                                   the fast heartbeat
  every SLOW (60 s)  append every rank's full line, plus the node's host memory, ECC and xGMI counts, to
                     <job>/ft/nodes/<node>.log on Lustre                                                    the slow heartbeat
  on sample/<g>      if the request names this node (or every node, '*'): py-spy on its ranks twice, STACK_GAP apart, the stacks
                     to Lustre, the verdict to stacks/<g>/<node>
  on failed/<g>      copy the failure list into host memory (failed.json) for its ranks' monitor threads, and to Lustre;
                     a list that replaces a node (2026-11-06, the recovery): the replaced node's agent ends its ranks still
                     alive after FTV2_VICTIM_GRACE_S (a hung rank cannot leave by itself), and the standby's agent turns its role
                     into train
  on coordinator     when the key is deleted (the coordinator's node died), try to take the role (create-if-absent)
Slow work (rocm-smi for ECC, py-spy) runs in threads of its own so the beat is never late. The ECC and xGMI readers follow
ft-sys-v1's hb.py (_ecc_smi, _xgmi, host_mem_pct); here they run in the agent, outside the training processes (map 1.5).

    python3 hb_agent.py --etcd h1:2379,h2:2379,h3:2379 --job-dir <job dir> [--role train|standby] [--beat 5] [--lease 15]"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hb_stacks
from hb_beat import exit_path, failed_path, node_beat, node_name, read_json, shm_dir, thread_path, write_json
from hb_coordinator import Coordinator
from hb_etcd import Etcd, EtcdError

STACK_GAP = float(os.environ.get('FTV2_STACK_GAP_S', '5'))   # seconds between the two py-spy dumps of a hang (the user, 11-05 11:50)


def host_mem_pct():
    try:
        m = {}
        for ln in open('/proc/meminfo'):
            k, v = ln.split(':', 1)
            m[k] = int(v.split()[0])
        return round(100.0 * (m['MemTotal'] - m['MemAvailable']) / m['MemTotal'], 1)
    except Exception:
        return -1.0


def find_smi():
    for c in (os.environ.get('ROCM_PATH', '') + '/bin/rocm-smi', '/opt/rocm/bin/rocm-smi'):
        if c.startswith('/') and os.access(c, os.X_OK):
            return c
    return shutil.which('rocm-smi')


def ecc_smi(gcd):
    """(correctable, uncorrectable) summed over the enabled blocks of one GPU's RAS table (ft-sys-v1 hb.py _ecc_smi); -1 on failure."""
    smi = find_smi()
    if not smi:
        return -1, -1
    try:
        r = subprocess.run([smi, '-d', str(gcd), '--showrasinfo', 'all'], capture_output=True, text=True, timeout=20)
    except Exception:
        return -1, -1
    ce = ue = 0
    seen = False
    for ln in r.stdout.splitlines():
        f = ln.split()
        if len(f) >= 2 and f[1] in ('ENABLED', 'DISABLED') and f[0].isupper():
            seen = True
            if f[1] == 'ENABLED' and len(f) >= 4 and f[2].isdigit() and f[3].isdigit():
                ce += int(f[2])
                ue += int(f[3])
    return (ce, ue) if seen else (-1, -1)


def xgmi(card):
    try:
        return int(open(os.path.join(card, 'xgmi_error')).read().split()[0]) if card else -1
    except Exception:
        return -1


def proc_state(pid, tid):
    try:
        return open(f'/proc/{pid}/task/{tid}/stat').read().rsplit(')', 1)[1].split()[0]
    except Exception:
        return 'gone'


def alive(pid):
    """The process exists and is not a zombie (ended, not yet reaped by its parent: found in the login-node test, 11-05)."""
    try:
        st = open(f'/proc/{int(pid)}/stat').read()
        return st.rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False
    except Exception:
        try:
            os.kill(int(pid), 0)
            return True
        except ProcessLookupError:
            return False
        except Exception:
            return True


class Agent:
    def __init__(self, a):
        self.a = a
        self.node = node_name()
        self.dir = shm_dir()
        self.job_dir = a.job_dir
        self.etcd = Etcd(a.etcd.split(','))
        self.lease = None
        self.coord = None
        self.lock = threading.Lock()
        self.ecc = {}
        self.seen_gone = set()     # (rank, pid) already logged as gone
        self.rec = os.path.join(self.job_dir, 'ft')
        os.makedirs(os.path.join(self.rec, 'nodes'), exist_ok=True)

    def log(self, msg):
        print(f'[ftv2 agent {self.node}] {time.strftime("%H:%M:%S")} {msg}', flush=True)

    def take_lease(self):
        while True:
            try:
                self.lease = self.etcd.grant(self.a.lease)
                return
            except EtcdError as e:
                self.log(f'lease grant failed ({e}); retrying')
                time.sleep(1)

    def try_coordinate(self):
        with self.lock:
            if self.coord is not None:
                return
            try:
                if self.etcd.create_if_absent('coordinator', self.node, lease=self.lease):
                    c = Coordinator(self.etcd, self.node, self.job_dir, self.log)
                    c.resume()
                    self.coord = c
            except EtcdError as e:
                self.log(f'coordinator election failed: {e}')

    # watches
    def on_coordinator(self, kind, key, value, rev):
        if key == 'coordinator' and kind == 'DELETE':
            self.log('the coordinator key was deleted: trying to take the role')
            self.try_coordinate()

    def on_failed(self, kind, key, value, rev):
        if kind != 'PUT':
            return
        try:
            f = json.loads(value)
        except ValueError:
            return
        write_json(failed_path(self.dir), f)
        self.log(f'failure list {f.get("gen")} copied to {failed_path(self.dir)}: {f.get("kind")} on {f.get("node")}')
        if f.get('replace') and f.get('node') == self.node:
            grace = float(os.environ.get('FTV2_VICTIM_GRACE_S', '20'))
            threading.Timer(grace, self.end_my_ranks, args=(f,)).start()
        if f.get('standby') == self.node:
            # the node's role in its beat follows its ranks' lines (fast_beat): the joined ranks write role train
            self.log(f'this standby node takes the place of {f.get("node")} (list {f.get("gen")})')

    def end_my_ranks(self, f):
        """This node was replaced: SIGKILL its training ranks that are still alive (their monitors exit by themselves after the
        stop; a rank hung in a GPU call or in a sleep cannot, HB2c)."""
        import signal
        for ln in self.read_lines():
            pid = ln.get('pid')
            if str(ln.get('run')) != str(f.get('run')):
                continue                       # a later run's process on this node (the next test arm): not a victim
            if pid and alive(pid):
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    self.log(f'replaced node: rank {ln.get("rank")} (pid {pid}) still alive {time.time() - f.get("t", 0):.0f} s '
                             f'after list {f.get("gen")}: SIGKILL')
                except OSError as e:
                    self.log(f'could not end pid {pid}: {e}')

    def on_sample(self, kind, key, value, rev):
        if kind != 'PUT':
            return
        try:
            req = json.loads(value)
        except ValueError:
            return
        if req.get('node') in (self.node, '*'):
            threading.Thread(target=self.sample, args=(key.split('/')[1], req), daemon=True).start()

    def sample(self, g, req):
        """Signal 2's evidence: py-spy on this node's ranks twice, STACK_GAP seconds apart; the dumps and the last full lines to
        Lustre; the verdict to etcd (confirmed only if both dumps confirm and the suspect has not moved, hb_stacks.verdict_twice)."""
        lines = self.read_lines()
        pyspy = self.a.pyspy or shutil.which('py-spy') or os.path.expanduser('~/.local/bin/py-spy')
        first = self.dump_all(lines, pyspy)
        time.sleep(STACK_GAP)
        second = self.dump_all(lines, pyspy)
        decision, detail = hb_stacks.verdict_twice(first, second, req.get('rank'))
        d = os.path.join(self.rec, 'failures', g)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f'stacks_{self.node}.txt'), 'w') as fh:
            for k, dumps in (('dump 1', first), (f'dump 2, {STACK_GAP:.0f} s later', second)):
                for r in sorted(dumps):
                    fh.write(f'=== rank {r}, {k} ===\n{dumps[r]}\n')
        with open(os.path.join(d, f'lines_{self.node}.json'), 'w') as fh:
            json.dump(lines, fh, indent=1)
        self.etcd.put(f'stacks/{g}/{self.node}', json.dumps({'decision': decision, 'detail': detail, 't': round(time.time(), 3)}))
        self.log(f'stacks for {g}: {decision} {detail}')

    def dump_all(self, lines, pyspy):
        """One py-spy dump of each live rank of this node, all at once: {rank: text}."""
        dumps = {}

        def one(ln):
            try:
                r = subprocess.run([pyspy, 'dump', '--native', '--pid', str(ln['pid'])], capture_output=True, text=True, timeout=50)
                dumps[ln['rank']] = r.stdout + r.stderr
            except Exception as e:
                dumps[ln['rank']] = f'py-spy failed: {e}'
        ts = [threading.Thread(target=one, args=(ln,)) for ln in lines if alive(ln.get('pid', 0))]
        [t.start() for t in ts]
        [t.join() for t in ts]
        return dumps

    # the beats
    def read_lines(self):
        lines = [read_json(p) for p in glob.glob(os.path.join(self.dir, 'rank_*.json'))]
        lines = [ln for ln in lines if ln]
        if not lines:
            return []
        run = max(lines, key=lambda ln: ln.get('t', 0)).get('run')
        return sorted((ln for ln in lines if ln.get('run') == run), key=lambda ln: ln['rank'])

    def gone(self, lines):
        """{rank: why} of the ranks whose process is gone: 'signal' without an exit record (a kill, a segfault: atexit and the
        exception hook never ran), else the record's reason. A record saying 'finished' is a normal end, not a death.
        A record with an exception counts at once, even while the process lives on: the exception hook runs only after the
        main program has ended, and the exit itself can block for good (HB3 5855079: rank 21 sat in PMPI_Finalize's barrier,
        reached from Py_FinalizeEx, waiting for ranks that still trained)."""
        dead = {}
        for ln in lines:
            r = ln['rank']
            ex = read_json(exit_path(self.dir, r))
            if ex and ex.get('pid') == ln.get('pid') and str(ex.get('reason', '')).startswith('exception'):
                dead[r] = ex['reason']
                continue
            if alive(ln.get('pid', 0)):
                continue
            if ex and ex.get('pid') == ln.get('pid'):
                if ex.get('reason') != 'finished':
                    dead[r] = ex.get('reason', '?')
                continue
            dead[r] = 'signal'
        return dead

    def fast_beat(self):
        lines = self.read_lines()
        why = self.gone(lines)
        pids = {ln['rank']: ln.get('pid') for ln in lines}
        for r, w in sorted(why.items()):
            if (r, pids.get(r)) not in self.seen_gone:   # the time this agent first saw it: a crash's signal, for the record
                self.seen_gone.add((r, pids.get(r)))
                self.log(f'rank {r} gone: {w[:160]}')
        finished = [ln['rank'] for ln in lines if not alive(ln.get('pid', 0)) and ln['rank'] not in why]
        lines_live = [ln for ln in lines if ln['rank'] not in finished]
        terr = {}
        for ln in lines_live:                          # exceptions that ended a non-main thread: evidence, not deaths
            te = read_json(thread_path(self.dir, ln['rank']))
            if te and te.get('pid') == ln.get('pid'):
                terr[str(ln['rank'])] = f"thread {te.get('thread')}: {te.get('reason', '?')}"[:200]
                if (ln['rank'], te.get('t')) not in self.seen_gone:
                    self.seen_gone.add((ln['rank'], te.get('t')))
                    self.log(f"rank {ln['rank']}: exception in {terr[str(ln['rank'])]}")
        roles = {ln.get('role') for ln in lines_live}
        role = 'standby' if roles == {'standby'} else ('train' if 'train' in roles else self.a.role)   # the ranks' own role
        beat = node_beat(self.node, lines_live, list(why), time.time(), role=role, dead_why={str(k): v for k, v in why.items()},
                         thread_errors=terr)
        try:
            if self.etcd.keepalive(self.lease) <= 0:
                self.log('my lease had ended: taking a new one (the coordinator may already have acted on my silence)')
                with self.lock:
                    self.coord = None
                self.take_lease()
            self.etcd.put(f'node/{self.node}', json.dumps(beat), lease=self.lease)
        except EtcdError as e:
            self.log(f'fast beat failed: {e}')
        return lines

    def slow_beat(self, lines):
        hm = host_mem_pct()
        out = []
        for ln in lines:
            ce, ue = self.ecc.get(ln.get('local_rank'), (-1, -1))
            out.append(dict(ln, host_mem_pct=hm, ecc_ce=ce, ecc_ue=ue, xgmi_err=xgmi(ln.get('card')),
                            proc_state=proc_state(ln.get('pid'), ln.get('tid'))))
        with open(os.path.join(self.rec, 'nodes', f'{self.node}.log'), 'a') as fh:
            for ln in out:
                fh.write(json.dumps(ln, separators=(',', ':')) + '\n')

    def ecc_loop(self):
        while True:
            for gcd in range(self.a.gpus):
                self.ecc[gcd] = ecc_smi(gcd)
            time.sleep(self.a.slow)

    def run(self):
        self.take_lease()
        self.etcd.put(f'members/{self.node}', json.dumps({'role': self.a.role, 't': round(time.time(), 3)}))   # no lease: it stays
        self.etcd.put(f'node/{self.node}', json.dumps(node_beat(self.node, [], [], time.time(), role=self.a.role)), lease=self.lease)
        self.etcd.watch('coordinator', self.on_coordinator, name='watch-coordinator')
        self.etcd.watch('failed/', self.on_failed, name='watch-failed')
        self.etcd.watch('sample/', self.on_sample, name='watch-sample')
        self.try_coordinate()
        if self.a.gpus > 0 and self.a.role == 'train':
            threading.Thread(target=self.ecc_loop, name='ecc', daemon=True).start()
        self.log(f'agent on: etcd {self.a.etcd}; lease {self.a.lease} s; fast beat every {self.a.beat} s; slow every {self.a.slow} s; '
                 f'shm {self.dir}; role {self.a.role}; coordinator {self.coord is not None}')
        next_slow = time.time() + self.a.slow
        while True:
            t0 = time.time()
            lines = self.fast_beat()
            c = self.coord
            if c is not None:
                try:
                    c.step()
                except EtcdError as e:
                    self.log(f'coordinator step failed: {e}')
            if t0 >= next_slow:
                next_slow = t0 + self.a.slow
                try:
                    self.slow_beat(lines)
                except OSError as e:
                    self.log(f'slow beat failed: {e}')
            time.sleep(max(0.0, self.a.beat - (time.time() - t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--etcd', required=True, help='client endpoints of the etcd members, host:port,host:port,...')
    ap.add_argument('--job-dir', required=True, help='the job folder on Lustre; the slow heartbeat goes to <job-dir>/ft/')
    ap.add_argument('--role', default='train', choices=['train', 'standby'])
    ap.add_argument('--beat', type=float, default=float(os.environ.get('FTV2_FAST_BEAT_S', '5')))
    ap.add_argument('--lease', type=int, default=int(os.environ.get('FTV2_LEASE_S', '15')))
    ap.add_argument('--slow', type=float, default=float(os.environ.get('FTV2_SLOW_BEAT_S', '60')))
    ap.add_argument('--gpus', type=int, default=8, help='GPUs on the node, for the ECC counts (0 on a CPU test)')
    ap.add_argument('--pyspy', default=None)
    Agent(ap.parse_args()).run()


if __name__ == '__main__':
    main()
