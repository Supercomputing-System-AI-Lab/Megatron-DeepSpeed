"""hb_coordinator.py (ft-sys-v2 heartbeat system, 2026-11-05): the coordinator role, held by one node agent at a time.

Why one agent (design map 1.4): a hang or a lost node shows only when nodes are compared, and the decision must be written once.
The role goes to whichever agent created the etcd key "coordinator" first, tied to that agent's lease; when its node dies the key
goes with the lease and another agent takes the role (hb_agent.try_coordinate). Every decision and request lives in etcd, so a new
coordinator continues where the old one stopped (resume()).

Every 5 s (step()): read every node's beat in one read, then, for the current training run:
  crash      a node's agent reports gone ranks (two beats, or an exit record)               -> the failure list
  node lost  a node's beat disappeared (its lease ended) and the others stopped progressing -> the failure list
             (the others still progress: a warning; the node is alive in training)
  hang       signal 1 (hb_rules.hang_suspect) -> ask the suspect node's agent for stacks (key sample/<g>);
             signal 2 (its answer stacks/<g>/<node>, hb_stacks.verdict) 'confirmed'      -> the failure list
             'ambiguous' or no answer: an alarm on Lustre, no replacement
The failure list failed/<g> is written with create-if-absent fenced on "coordinator still names me", so an old coordinator that is
cut off cannot write one. One failure per training run tonight (no recovery yet): later signals of that run are its consequences."""
import json
import os
import time

import hb_rules

STACK_WAIT = 90.0        # seconds to wait for the suspect node's stacks before an alarm
HOLD = 60.0              # seconds before the same suspect is sampled again after an ambiguous answer
SUMMARY = float(os.environ.get('FTV2_SLOW_BEAT_S', '60'))


def _g(n):
    return f'{n:04d}'


class Coordinator:
    def __init__(self, etcd, me, job_dir, log):
        self.etcd, self.me, self.job_dir, self.log = etcd, me, job_dir, log
        self.track = hb_rules.Track()
        self.incident = None
        self.done_runs = set()
        self.warned = set()
        self.hold = {}
        self.next_summary = 0.0
        self.rec_dir = os.path.join(job_dir, 'ft')
        os.makedirs(self.rec_dir, exist_ok=True)

    def record(self, text):
        """One line for people, in <job>/ft/record.log on Lustre."""
        with open(os.path.join(self.rec_dir, 'record.log'), 'a') as fh:
            fh.write(f'{time.strftime("%F %T")} {text}\n')

    def resume(self):
        """A new coordinator continues from etcd: runs that already have a failure list, and a stack request still open."""
        failed, _ = self.etcd.range('failed/')
        for v in failed.values():
            try:
                self.done_runs.add(json.loads(v).get('run'))
            except ValueError:
                pass
        samples, _ = self.etcd.range('sample/')
        closed, _ = self.etcd.range('closed/')
        for k, v in sorted(samples.items()):
            g = k.split('/')[1]
            if f'failed/{g}' not in failed and f'closed/{g}' not in closed:
                s = json.loads(v)
                if s.get('run') not in self.done_runs:
                    self.incident = dict(s, gen=g, t_sample=time.time(), resumed=True)
        self.record(f'COORDINATOR {self.me} took the role; runs with a failure list: {sorted(map(str, self.done_runs))}; '
                    f'open stack request: {self.incident["gen"] if self.incident else "none"}')
        self.log(f'coordinator role taken; resumed incident: {self.incident}')

    def _next_gen(self):
        failed, _ = self.etcd.range('failed/')
        samples, _ = self.etcd.range('sample/')
        closed, _ = self.etcd.range('closed/')
        used = {k.split('/')[1] for k in list(failed) + list(samples) + list(closed)}
        n = 1
        while _g(n) in used:
            n += 1
        return _g(n)

    def _plan(self, beats, node, kind=None):
        if kind == 'out of memory':
            return 'change the parallelism with the OOM ladder (design map 4.1); no node replaced'
        standbys = sorted(n for n, b in beats.items() if b.get('role') == 'standby' and n != node)
        return f'standby {standbys[0]} takes node {node}' if standbys else 'no standby free: stop, or a layout with fewer GPUs (config)'

    def declare(self, beats, run, kind, node, suspect, why, gen=None):
        b = beats.get(node) or self.track.nodes.get(node, {}).get('beat', {})
        g = gen or self._next_gen()
        val = {'gen': g, 'run': run, 'kind': kind, 'node': node, 'suspect': suspect, 'ranks': b.get('rank_ids', []),
               'why': why, 'plan': self._plan(beats, node, kind), 't': round(time.time(), 3), 'by': self.me}
        ok = self.etcd.create_if_absent(f'failed/{g}', json.dumps(val), fence=('coordinator', self.me))
        if ok:
            self.done_runs.add(run)
            self.incident = None
            self.record(f'FAILURE {g}: {kind} on {node} (suspect {suspect}); {why}; plan: {val["plan"]}')
            self.log(f'failure list {g} written: {val}')
        else:
            self.log(f'failure list {g} not written (another writer, or I am no longer the coordinator)')
        return ok

    def step(self, now=None):
        now = time.time() if now is None else now
        raw, _ = self.etcd.range('node/')
        beats = {}
        for k, v in raw.items():
            try:
                beats[k.split('/', 1)[1]] = json.loads(v)
            except ValueError:
                pass
        gone = self.track.update(beats, now)
        mraw, _ = self.etcd.range('members/')
        members = {}
        for k, v in mraw.items():
            try:
                members[k.split('/', 1)[1]] = json.loads(v).get('role', 'train')
            except ValueError:
                pass
        gone += self.track.note_absent(members, beats, now)
        for n in gone:
            self.log(f'beat of {n} disappeared (its lease ended)')
        runs = [b.get('run') for b in beats.values() if b.get('role', 'train') == 'train' and b.get('run') is not None]
        run = max(set(runs), key=runs.count) if runs else None
        if now >= self.next_summary:
            self.next_summary = now + SUMMARY
            alive = sorted(beats)
            lows = ', '.join(f'{n} {beats[n].get("low")}' for n in alive if beats[n].get('role', 'train') == 'train')
            self.record(f'beat | {len(alive)} nodes with a beat: {" ".join(alive)} | run {run} | lowest counts {lows} | '
                        f'coordinator {self.me}')
        if run is None or run in self.done_runs:
            return
        if self.incident:
            self._progress(beats, run, now)
            return
        roots, conseq = [], []
        for n, b in sorted(beats.items()):
            if b.get('run') != run:
                continue
            for r in hb_rules.crash_report(b):
                why = b.get('dead_why', {}).get(str(r), 'signal')
                (roots if hb_rules.death_cause(why) == 'root' else conseq).append((n, r, why))
        if roots:
            # a root with a message (an exception that is not about communication) first: when Slurm ends every task after one
            # exits (--kill-on-bad-exit=1), the others die by signal at the same moment and would hide the real cause
            roots.sort(key=lambda x: x[2] == 'signal')
            n0, r0, why0 = roots[0]
            others = '; also gone: ' + ', '.join(f'rank {r} on {n} ({w[:60]})' for n, r, w in roots[1:] + conseq) if len(roots) + len(conseq) > 1 else ''
            kind = hb_rules.death_kind(why0)
            self.declare(beats, run, kind, n0, [r for n, r, _ in roots if n == n0], f'rank {r0} on {n0} gone: {why0[:160]}{others}')
            return
        if conseq:
            self.crash_seen = getattr(self, 'crash_seen', None) or now
        for n in sorted(self.track.missing):
            v = hb_rules.lost_node_verdict(self.track, n, now)
            if conseq and v not in ('alive', 'standby'):
                # ranks died of communication errors with no root death reported: their peers went away, and a training node
                # whose lease ended is where they went. The counts cannot say it when the lost node held the coordinator role
                # (the new coordinator never saw it beat) and the survivors are dead (no counts): login e2e, 11-05 06:06
                self.declare(beats, run, 'node lost', n, None,
                             f'lease of {n} ended and ranks elsewhere died of communication errors: '
                             + ', '.join(f'rank {r} on {m}' for m, r, _ in conseq[:4]))
                return
            if v in ('lost', 'lost-stalled') and hb_rules.counts_differ(self.track):
                # a beating rank is behind: a hang there would stall everyone too, so the stacks decide (the hang path below);
                # if they show every sampled rank waiting, the silent node is the cause (_progress)
                if (n, 'deferred') not in self.warned:
                    self.warned.add((n, 'deferred'))
                    self.record(f'SILENT {n}: lease ended and the training stalled, but a beating rank is behind: '
                                f'the stacks decide')
                continue
            if v == 'lost':
                self.declare(beats, run, 'node lost', n, None,
                             f'lease of {n} ended and the other nodes stopped progressing')
                return
            if v == 'lost-stalled' and hb_rules.hang_suspect(self.track, now) is None:
                self.declare(beats, run, 'node lost', n, None,
                             f'{n} had lost its agent but trained on; now the training has stalled for '
                             f'{hb_rules.STILL:.0f} s and no beating node is the suspect')
                return
            if v in ('alive', 'standby') and (n, v) not in self.warned:
                self.warned.add((n, v))
                msg = ('its agent stopped, but the other nodes still progress: alive in training, not replaced' if v == 'alive'
                       else 'a standby node: nothing to replace')
                self.record(f'WARNING {n}: lease ended; {msg}')
                self.log(f'warning {n}: {msg}')
        if conseq:
            if now - self.crash_seen >= hb_rules.SETTLE and ('conseq', run) not in self.warned:
                self.warned.add(('conseq', run))
                self.record(f'ALARM: ranks gone with communication errors only, no root death and no lost node in '
                            f'{hb_rules.SETTLE:.0f} s: {conseq[:8]}; no replacement')
            return
        s = hb_rules.hang_suspect(self.track, now)
        if s and now >= self.hold.get(s[:2], 0):
            node, rank, still_for = s
            g = self._next_gen()
            req = {'node': node, 'rank': rank, 'run': run, 'still_s': still_for, 't': round(now, 3), 'by': self.me}
            if self.etcd.create_if_absent(f'sample/{g}', json.dumps(req), fence=('coordinator', self.me)):
                self.incident = dict(req, gen=g, t_sample=now)
                self.record(f'SUSPECT {g}: rank {rank} on {node}: no count moved for {still_for} s and the counts differ; '
                            f'stacks requested')
                self.log(f'signal 1: suspect rank {rank} on {node}; stacks requested as sample/{g}')
        elif hb_rules.all_still_equal(self.track, now) and ('equal', run) not in self.warned:
            self.warned.add(('equal', run))
            self.record(f'ALARM: every rank still for {hb_rules.LONG:.0f} s with equal counts: no single suspect (open case)')

    def _progress(self, beats, run, now):
        inc = self.incident
        g, node, rank = inc['gen'], inc['node'], inc['rank']
        if node in self.track.missing:                 # the suspect's node lost its lease meanwhile: the lost-node rule decides
            if hb_rules.lost_node_verdict(self.track, node, now) == 'lost':
                self.declare(beats, run, 'node lost', node, rank, f'lease of {node} ended while its stacks were awaited', gen=g)
            return
        ans, _ = self.etcd.get(f'stacks/{g}/{node}')
        if ans is None:
            if now - inc['t_sample'] > STACK_WAIT:
                self.record(f'ALARM {g}: no stacks from {node} in {STACK_WAIT:.0f} s; no replacement')
                self.etcd.put(f'closed/{g}', 'no stacks')
                self.hold[(node, rank)] = now + HOLD
                self.incident = None
            return
        a = json.loads(ans)
        if a.get('decision') == 'confirmed':
            self.declare(beats, run, 'hang', node, rank,
                         f'signal 1: counts still for {inc.get("still_s")} s and differ; signal 2: rank {rank} outside any '
                         f'communication wait, {len(a.get("detail", {}).get("waiting", []))} others waiting', gen=g)
        else:
            silent = [m for m in sorted(self.track.missing)
                      if hb_rules.lost_node_verdict(self.track, m, now) in ('lost', 'lost-stalled')]
            if silent and a.get('decision') == 'ambiguous' and not a.get('detail', {}).get('outside'):
                self.declare(beats, run, 'node lost', silent[0], None,
                             f'lease of {silent[0]} ended and the training stalled; the stacks of {node} show every rank '
                             f'waiting, so the silent node is the cause', gen=g)
                return
            self.record(f'ALARM {g}: stacks of {node} do not confirm rank {rank} ({a.get("decision")}: {a.get("detail")}); '
                        f'no replacement')
            self.log(f'signal 2 not confirmed for rank {rank}: {a}')
            self.etcd.put(f'closed/{g}', a.get('decision', '?'))
            self.hold[(node, rank)] = now + HOLD
            self.incident = None
