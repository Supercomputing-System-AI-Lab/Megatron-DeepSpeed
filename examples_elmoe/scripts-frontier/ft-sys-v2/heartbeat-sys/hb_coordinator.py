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
             'ambiguous', 'moved' or no answer: an alarm on Lustre, no replacement
  stuck      every rank still with equal counts for 3 minutes (hb_rules.all_still_equal): stacks from every node (sample/<g>
             for node '*'), then the failure list "stuck"; the plan is to stop (the gate checks that find the node come later)
The plan follows the kind (the user, 11-05 11:50): out of memory -> the OOM ladder (FTV2_OOM_RECONFIG=1, the default; 0 -> stop);
another exception -> stop for debugging; a hang, a lost node, a death by a signal or a hardware fault -> replace the node.
The failure list failed/<g> is written with create-if-absent fenced on "coordinator still names me", so an old coordinator that is
cut off cannot write one. One failure per training run: later signals of that run are its consequences.
Recovery (2026-11-06, ft-sys-v2/recovery/): a list whose plan replaces a node also names the standby node in the field 'standby'
(and 'replace': true); that is the whole layout of a warm recovery (the standby's GPUs take the lost node's ranks, in order).
The standby is then taken and the replaced node retired: neither is used or judged again. The recovered training writes its
lines under a new run id (<run>.g<gen>), which the coordinator watches as a new run."""
import json
import re
import os
import time

import hb_rules

STACK_WAIT = float(os.environ.get('FTV2_STACK_WAIT_S', '150'))   # seconds to wait for the suspect node's stacks and evidence
                                                                #   (two dumps of at most 15 s, the evidence: a probe of up to 2 min)
HOLD = 60.0              # seconds before the same suspect is sampled again after an ambiguous answer
SUMMARY = float(os.environ.get('FTV2_SLOW_BEAT_S', '60'))
STUCK_REDO = os.environ.get('FTV2_STUCK_REDO') == '1'           # experiment 5 (2026-11-12): a stall with no culprit is redone in place
STUCK_REDO_MAX = int(os.environ.get('FTV2_STUCK_REDO_MAX', '1'))  # redos per run before a stall stops it (off by default)
OOM_RECONFIG = os.environ.get('FTV2_OOM_RECONFIG', '1') == '1'   # out of memory: re-configure with the OOM ladder (on by default)


def _g(n):
    return f'{n:04d}'


class Coordinator:
    def __init__(self, etcd, me, job_dir, log):
        self.etcd, self.me, self.job_dir, self.log = etcd, me, job_dir, log
        self.track = hb_rules.Track()
        self.incident = None
        self.done_runs = set()
        self.redos = {}            # experiment 5: base run -> redos done
        self.warned = set()
        self.hold = {}
        self.next_summary = 0.0
        self.taken = set()       # standby nodes a failure list already gave a role
        self.retired = set()     # nodes a failure list replaced: never judged again
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
                f = json.loads(v)
            except ValueError:
                continue
            self.done_runs.add(f.get('run'))
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
            return ('change the parallelism with the OOM ladder (design map 4.1); no node replaced' if OOM_RECONFIG
                    else 'stop the training: out of memory, and the OOM ladder is off (FTV2_OOM_RECONFIG=0)')
        if kind == 'exception':
            return 'stop the training for debugging: an error in the program; no node replaced'
        if kind == 'stuck':
            return ('stop the training for debugging: a stall no evidence explains (every rank still with equal counts, or a '
                    'suspect with no stack and no fault found, every node check passed); the stacks are saved')
        sb = self._standby(beats, node, kind)
        return f'standby {sb} takes node {node}' if sb else 'no standby free: stop, or a layout with fewer GPUs (config)'

    def _standby(self, beats, node, kind):
        """The standby node that takes `node`'s place, or None (a kind that replaces no node, or no standby free)."""
        if kind in ('out of memory', 'exception', 'stuck'):
            return None
        standbys = sorted(n for n, b in beats.items() if b.get('role') == 'standby' and n != node and n not in self.taken)
        return standbys[0] if standbys else None

    def declare(self, beats, run, kind, node, suspect, why, gen=None):
        b = beats.get(node) or self.track.nodes.get(node, {}).get('beat', {})
        g = gen or self._next_gen()
        val = {'gen': g, 'run': run, 'kind': kind, 'node': node, 'suspect': suspect, 'ranks': b.get('rank_ids', []),
               'why': why, 'plan': self._plan(beats, node, kind), 't': round(time.time(), 3), 'by': self.me}
        sb = self._standby(beats, node, kind)
        base = re.sub(r'(\.g\d+)+$', '', str(run))
        redo = kind == 'stuck' and STUCK_REDO and self.redos.get(base, 0) < STUCK_REDO_MAX
        if redo:                       # experiment 5: no node is replaced and no rank is lost; every rank stops, rebuilds, redoes
            val.update(ranks=[], plan=f'redo the step in place (FTV2_STUCK_REDO=1): every communicator aborted, the same world '
                       f'rebuilt, each rank restores its own state; redo {self.redos.get(base, 0) + 1} of {STUCK_REDO_MAX}; no node replaced')
        val.update(standby=sb, replace=bool(sb) or redo, redo=redo)
        ok = self.etcd.create_if_absent(f'failed/{g}', json.dumps(val), fence=('coordinator', self.me))
        if ok:
            self.done_runs.add(run)
            if redo:
                self.redos[base] = self.redos.get(base, 0) + 1
            if sb:
                self.taken.add(sb)
                self.retired.add(node)
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
        runs0 = [b.get('run') for b in beats.values() if b.get('role', 'train') == 'train' and b.get('run') is not None]
        arm = str(max(set(runs0), key=runs0.count)).split('.g')[0] if runs0 else None
        if arm is not None and arm != getattr(self, 'arm', None):
            # a new training run: its standbys and nodes are free again, except those its own lists (an earlier generation
            # of the same run, or a coordinator that took over) already used
            if getattr(self, 'arm', None) is not None:
                self.record(f'NEW RUN {arm}: standbys and replaced nodes of {self.arm} are free again')
            self.taken.clear()
            self.retired.clear()
            for v in self.etcd.range('failed/')[0].values():
                try:
                    f = json.loads(v)
                except ValueError:
                    continue
                if str(f.get('run')).split('.g')[0] == arm and f.get('standby'):
                    self.taken.add(f['standby'])
                    self.retired.add(f.get('node'))
            self.arm = arm
        beats = {n: b for n, b in beats.items() if n not in self.retired}     # a replaced node is never judged again
        gone = self.track.update(beats, now)
        mraw, _ = self.etcd.range('members/')
        members = {}
        for k, v in mraw.items():
            try:
                members[k.split('/', 1)[1]] = json.loads(v).get('role', 'train')
            except ValueError:
                pass
        members = {n: r for n, r in members.items() if n not in self.retired}
        for n in self.retired:
            self.track.nodes.pop(n, None)
            self.track.missing.pop(n, None)
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
        if run is not None and run not in self.done_runs:
            # a list written by another writer (a test's null recovery, rc_controller) also ends the judging of its run
            for v in self.etcd.range('failed/')[0].values():
                try:
                    if json.loads(v).get('run') == run:
                        self.done_runs.add(run)
                except ValueError:
                    pass
        if run is None or run in self.done_runs:
            return
        if self.incident:
            self._progress(beats, run, now)
            return
        roots, conseq = [], []
        for n, b in sorted(beats.items()):
            if b.get('run') != run:
                continue
            if b.get('role') == 'standby':
                # a standby's processes are not in the training: their death is a lost standby, not a training failure
                # (P1b 5861716: the standby's exception was written as a failure of the run and the training was stopped)
                for r in hb_rules.crash_report(b):
                    if ('standby-gone', n, run) not in self.warned:
                        self.warned.add(('standby-gone', n, run))
                        self.record(f'WARNING {n}: standby process {r} gone ({b.get("dead_why", {}).get(str(r), "signal")[:120]}); '
                                    f'not a training failure')
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
                # no list will come for this run, so nothing would tell its standby that the run is over: mark the run done
                # (the key the recovery's standby polls, rc_controller.standby_wait), so that the standby leaves and the job's
                # step can end (E5c 5862600: the standby waited until the wall after every training rank had gone)
                base = str(run).split('.g')[0]
                self.etcd.put(f'run_done/{base}', json.dumps({'t': round(now, 3), 'by': self.me,
                                                              'stopped': 'ranks gone with communication errors only, no list'}))
                self.record(f'RUN DONE {base}: written, so that its standby leaves (no list will come)')
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
            g = self._next_gen()
            req = {'node': '*', 'rank': None, 'run': run, 'still_s': hb_rules.LONG, 't': round(now, 3), 'by': self.me}
            if self.etcd.create_if_absent(f'sample/{g}', json.dumps(req), fence=('coordinator', self.me)):
                self.incident = dict(req, gen=g, t_sample=now)
                self.record(f'STUCK {g}: every rank still for {hb_rules.LONG:.0f} s with equal counts; stacks requested from '
                            f'every node')

    def _start_stuck_rule(self, now, why):
        inc = self.incident
        inc['stuck_rule'] = True
        inc['stall_start'] = inc['t_sample'] - float(inc.get('still_s') or 0)
        inc['stuck_why'] = why
        left = max(0.0, hb_rules.LONG - (now - inc['stall_start']))
        self.record(f'STUCK RULE {inc["gen"]}: no evidence of a fault ({why}); node checks on every training node once the '
                    f'stall reaches {hb_rules.LONG:.0f} s (in {left:.0f} s)')

    def _stuck_rule(self, beats, run, now):
        """The user's stuck rule (#5, 11-05: "for a stuck over the nccl or anything, I will give it 3 mins") for a stall the
        evidence could not explain: at LONG seconds of stall every training node checks its GPUs (the agents' check/<g>); a
        failing node is replaced (kind "hardware fault"); if every node passes, kind "stuck": stop for debugging."""
        inc = self.incident
        g, node, rank = inc['gen'], inc['node'], inc['rank']
        if now - inc['stall_start'] < hb_rules.LONG:
            return
        want = set(self.track.training())
        if 'check_t' not in inc:
            req = {'node': '*', 'run': run, 'for': g, 't': round(now, 3), 'by': self.me}
            self.etcd.put(f'check/{g}', json.dumps(req))
            inc['check_t'] = now
            self.record(f'NODE CHECKS {g}: stall of {now - inc["stall_start"]:.0f} s: every training node probes its GPUs '
                        f'({", ".join(sorted(want))})')
            return
        got, _ = self.etcd.range(f'check/{g}/')
        answers = {}
        for k, v in got.items():
            try:
                answers[k.split('/')[2]] = json.loads(v)
            except (ValueError, IndexError):
                pass
        if not (want <= set(answers)) and now - inc['check_t'] < hb_rules.CHECK_WAIT:
            return
        failing = hb_rules.node_check_verdict(answers, want)
        if failing:
            bad = node if node in failing else failing[0]
            d = answers.get(bad, {}).get('detail', 'no answer in time')
            self.declare(beats, run, 'hardware fault', bad, rank if bad == node else None,
                         f'stuck rule: stall of {now - inc["stall_start"]:.0f} s with no evidence on {node} '
                         f'({inc.get("stuck_why")}); the node check of {bad} failed: {json.dumps(d)[:300]}', gen=g)
            return
        self.declare(beats, run, 'stuck', node, rank,
                     f'stuck rule: stall of {now - inc["stall_start"]:.0f} s; no stacks and no evidence of a fault on {node} '
                     f'({inc.get("stuck_why")}); the node checks of {len(answers)} training nodes passed: stop for debugging',
                     gen=g)

    def _progress(self, beats, run, now):
        inc = self.incident
        g, node, rank = inc['gen'], inc['node'], inc['rank']
        if node == '*':                                # stuck: wait for every training node's stacks, then the list
            got, _ = self.etcd.range(f'stacks/{g}/')
            have = {k.split('/')[2] for k in got}
            want = set(self.track.training())
            if want <= have or now - inc['t_sample'] > STACK_WAIT:
                self.declare(beats, run, 'stuck', 'every node', None,
                             f'every rank still for {inc.get("still_s", hb_rules.LONG):.0f} s with equal counts; stacks of '
                             f'{len(have & want)} of {len(want)} training nodes in ft/failures/{g}/', gen=g)
            return
        if node in self.track.missing:                 # the suspect's node lost its lease meanwhile: the lost-node rule decides
            if hb_rules.lost_node_verdict(self.track, node, now) == 'lost':
                self.declare(beats, run, 'node lost', node, rank, f'lease of {node} ended while its stacks were awaited', gen=g)
            return
        if inc.get('stuck_rule'):                      # no stacks and no evidence of a fault: the stuck rule decides
            self._stuck_rule(beats, run, now)
            return
        ans, _ = self.etcd.get(f'stacks/{g}/{node}')
        if ans is None:
            if now - inc['t_sample'] > STACK_WAIT:
                self.record(f'NO ANSWER {g}: no stacks or evidence from {node} in {STACK_WAIT:.0f} s: the stuck rule decides')
                self._start_stuck_rule(now, f'no answer from {node} in {STACK_WAIT:.0f} s')
            return
        a = json.loads(ans)
        if a.get('decision') == 'confirmed':
            te = (beats.get(node) or {}).get('thread_errors', {}).get(str(rank))
            self.declare(beats, run, 'hang', node, rank,
                         f'signal 1: counts still for {inc.get("still_s")} s and differ; signal 2: rank {rank} outside any '
                         f'communication wait in two dumps, {len(a.get("detail", {}).get("waiting", []))} others waiting'
                         + (f'; earlier, an exception ended its {te}' if te else ''), gen=g)
        else:
            silent = [m for m in sorted(self.track.missing)
                      if hb_rules.lost_node_verdict(self.track, m, now) in ('lost', 'lost-stalled')]
            if silent and a.get('decision') == 'ambiguous' and not a.get('detail', {}).get('outside'):
                self.declare(beats, run, 'node lost', silent[0], None,
                             f'lease of {silent[0]} ended and the training stalled; the stacks of {node} show every rank '
                             f'waiting, so the silent node is the cause', gen=g)
                return
            if a.get('decision') == 'moved':
                # the suspect is not blocked (its stack changed between the dumps): no fault to act on yet
                self.record(f'ALARM {g}: stacks of {node} do not confirm rank {rank} ({a.get("decision")}: {a.get("detail")}); '
                            f'no replacement')
                self.log(f'signal 2 not confirmed for rank {rank}: {a}')
                self.etcd.put(f'closed/{g}', a.get('decision', '?'))
                self.hold[(node, rank)] = now + HOLD
                self.incident = None
                return
            # no confirming stacks (none could be taken, or every rank waits): the evidence decides, else the stuck rule
            # (2026-11-06, the real fault of V3-2L 5863006: "no stacks, no replacement" waited until PyTorch's watchdog
            # killed the survivors; a decision must come in bounded time)
            fault, why = hb_rules.evidence_verdict(a.get('evidence') or {})
            if fault:
                self.declare(beats, run, 'hardware fault', node, rank,
                             f'GPU stuck: signal 1: counts still for {inc.get("still_s")} s and differ; the stacks of {node} '
                             f'were not taken ({a.get("decision")}: no stack from {a.get("detail", {}).get("no_stack")}); '
                             f'evidence: {why}', gen=g)
                return
            self.record(f'UNCONFIRMED {g}: stacks of {node} do not confirm rank {rank} ({a.get("decision")}: '
                        f'{a.get("detail")}); {why}: the stuck rule decides')
            self._start_stuck_rule(now, why)
