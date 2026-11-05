#!/usr/bin/env python3
"""The heartbeat aggregator (design 3.1, 3.7, 3.11, 5.3): reads <job>/data/hb/<node>.tsv every 10 s and writes <job>/heartbeat.log
and <job>/STATUS.md.

WHAT IT WRITES
  heartbeat.log = a header block that lists every recovery so far (the RECOVERED banners, written by the job script into
  <job>/data/banners.txt) + one BLOCK per beat: a beat line and five fixed lines, every one a full sentence (the user's rule of
  2026-10-09 evening: "each sentence is clearly stated instead of being too abbreviated ... more information costs us nothing"),
  plus ALARM lines. STATUS.md (one screen) is rewritten at every beat and by every --compose. Also data/agg_last.json (the last
  table, read by the job script to name the node) and data/aggregator.alive.
THE RULES
  stillness: a rank's last line is older than the threshold (30 s; 120 s in a save window, see below).
  minority: in a group of G consecutive ranks where nobody has moved for the threshold, the ranks behind the group's collective
  count are named (30 s; 120 s in a save window).
  slow rank: busy time above 1.3 x the median of its stage mates. Counter deltas (ECC, xGMI). Host memory maximum.
THE SAVE WINDOW (2026-10-09 late evening; found in the 260-node job 5645677 at 22:34: the last pipeline stage's ranks finished
  their checkpoint write 53 s after rank 0; the save's Python-level host copy holds the GIL and starves the heartbeat thread, so
  their lines went 49-59 s stale and their collective count fell behind their group; both rules fired and a stack sample was
  taken for nothing). A group is in the save window when any of its ranks reports a save phase or the save_bg flag, or when such
  a line was seen less than 120 s ago. Inside the window the stillness rule and the minority rule both use 120 s, and the ALARM
  line says so. Outside the window nothing changes. RULE FOR FUTURE DETECTORS: any early-kill rule must check in_save_window()
  and never act inside the window.
THE DELIVERY LAG (2026-10-10 00:45; found in the 260-node job 5646348: at one beat 195 ranks on about two dozen nodes were "32 s old"
  at once, every rank of each such node equally old, and two seconds later every rank was fresh again, while the rank files show the
  ranks wrote every 10 s throughout; the lines were late on the path from the node to the aggregator, the node writer's append to
  Lustre or Lustre itself, so a stack sample of frontier01343 paused 8 ranks for nothing). The rule: when EVERY rank of a node is
  equally old (the spread of their ages is within LAG_SPREAD_S) and older than the threshold, the node file's own modification time
  is compared with the lines: when the file itself stopped changing when the lines did (the file is as old as its newest line), the
  lines inside are fresh for the file and the gap is in the delivery, not in a rank. Such a node is a DELIVERY LAG: it is written in
  the beat block as "delivery lag N s on M nodes (Lustre), not a rank", its ranks are not named by the stillness rule or by the minority
  rule, and no stack sample is taken. A lag is a lag of SOME nodes while at least one rank somewhere is fresh; when every node is
  stale at once (the end of a step, a global stall) nothing is called a delivery lag and the plain rules apply. A lag that persists LAG_ESCALATE_BEATS beats is said to be more likely a dead node writer or a
  lost node (which looks the same from here) and is left to the loop's srun watch; still no sample, because a lost node cannot be sampled.
THE TWO-BEAT RULE (2026-10-10 00:45): a rank is named by the stillness rule only when it is older than the threshold on TWO
  consecutive beats; the first beat is written as "stillness pending" in the beat block and raises nothing. The minority rule is
  unchanged (it already needs the whole group frozen for the threshold).
On a new alarm: one full-sentence line in the TOP-LEVEL file <job>/alarms.log (the user, 2026-10-09 late evening: the alarms must be
  visible at the top level), the marker data/alarm/ALARM (for scripts), a stack sample of the named ranks with py-spy through
  `srun --overlap` (timeout 60 s) -> the TOP-LEVEL folder <job>/alarms/<node>_<HHMMSS>.txt, and heatmap.png (drawn ONLY on an alarm,
  at the end of an attempt, or on demand: heatmap.py --job-dir; the user, 2026-10-09 13:55: once a minute is too expensive). When a
  named rank is no longer named, alarms.log gets a second line ("cleared at ... after N s"); when the attempt ends in a failure, the
  job script adds the line "became failure k at ...". STATUS.md counts the alarms so far.
`--compose` only re-renders heartbeat.log, STATUS.md and the banner area (top 20 lines) of a-monitor.txt; the job script calls it.
"""
import argparse, collections, json, os, statistics, subprocess, sys, time

F = ['time', 'node', 'rank', 'step', 'phase', 'coll_seq', 'step_time_s', 'busy_s', 'hbm_gb', 'host_mem_pct', 'ecc_ce', 'ecc_ue',
     'xgmi_err', 'proc_state', 'loss', 'attempt', 'pid', 'flag', 'epoch']
BANNER_LINES = 20
SAVE_WINDOW_S = 120.0     # the threshold of both rules inside a save window, and how long the window stays open after a save line
LAG_SPREAD_S = 3.0        # the ranks of a node count as "equally old" when their ages lie within this many seconds of each other
LAG_ESCALATE_BEATS = 3    # a delivery lag that persists this many beats is said to look like a dead writer or a lost node


def now_s(t=None):
    return time.strftime('%H:%M:%S', time.localtime(t or time.time()))


def read_state(job):
    st = {}
    try:
        for line in open(os.path.join(job, 'data', 'state')):
            k, _, v = line.strip().partition('=')
            st[k] = v
    except Exception:
        pass
    return st


def _atomic(path, text):
    tmp = path + '.tmp%d' % os.getpid()
    with open(tmp, 'w') as fh:
        fh.write(text)
    os.replace(tmp, path)


def compose(job):
    """heartbeat.log = recoveries header + body; a-monitor.txt top 20 lines = the banner area."""
    try:
        banners = [l.rstrip('\n') for l in open(os.path.join(job, 'data', 'banners.txt')) if l.strip()]
    except Exception:
        banners = []
    try:
        body = open(os.path.join(job, 'data', 'heartbeat_body.log')).read()
    except Exception:
        body = ''
    if banners:
        head = [f'===== RECOVERIES SO FAR: {len(banners)} (the newest last; one block per failure is in recovery.log) ====='] + banners + ['===== END RECOVERIES =====', '']
    else:
        head = ['===== RECOVERIES SO FAR: none (no failure has been recovered in this job yet) =====', '']
    _atomic(os.path.join(job, 'heartbeat.log'), '\n'.join(head) + '\n' + body)
    # a-monitor.txt: a fixed 20-line banner area on top, the monitor's own text below
    am = os.path.join(job, 'a-monitor.txt')
    start, end = '##### RESILIENCE BANNER AREA (top 20 lines; newest recovery last) #####', '##### END BANNER AREA #####'
    rest = ''
    if os.path.exists(am):
        lines = open(am).read().split('\n')
        if lines and lines[0] == start and end in lines:
            lines = lines[lines.index(end) + 1:]
        rest = '\n'.join(lines)
    keep = banners[-(BANNER_LINES - 2):] if banners else ['(no recovery yet)']
    if len(banners) > BANNER_LINES - 2:
        keep = [f'({len(banners) - (BANNER_LINES - 3)} earlier recoveries: see heartbeat.log)'] + banners[-(BANNER_LINES - 3):]
    area = [start] + keep + [''] * (BANNER_LINES - 2 - len(keep)) + [end]
    _atomic(am, '\n'.join(area) + '\n' + rest)
    status_md(job, banners)


ALARMS_HEADER = ('# alarms.log of job {job}: one line per alarm event, written by the heartbeat aggregator; the time, the rule (stillness or\n'
                 '# minority), the rank and node named, its phase and collective count, how old its line was, whether it is in a save window, and\n'
                 '# the stack sample taken (the folder alarms/); a second line when the alarm clears, or when it became a failure (recovery.log).\n')


def alarm_log(job, text):
    """Append one full-sentence line to the top-level alarms.log (created with its header on the first line)."""
    p = os.path.join(job, 'alarms.log')
    new = not os.path.exists(p)
    with open(p, 'a') as fh:
        if new:
            fh.write(ALARMS_HEADER.format(job=os.environ.get('SLURM_JOB_ID', os.path.basename(job))))
        fh.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {text}\n')


def alarm_count(job):
    """The number of alarm events so far (the lines of alarms.log that open an alarm)."""
    try:
        return sum(1 for l in open(os.path.join(job, 'alarms.log')) if ' ALARM ' in l)
    except OSError:
        return 0


def _tail(path, n=1):
    try:
        with open(path, 'rb') as fh:
            fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - 8192))
            return [l for l in fh.read().decode(errors='replace').split('\n') if l.strip()][-n:]
    except Exception:
        return []


def _plural(n, one, many=None):
    return f'{n} {one if n == 1 else (many or one + "s")}'


def status_md(job, banners=None):
    """STATUS.md: one screen (design 3.11), every row a full sentence: the state, the attempt, the last step, the last save, the
    nodes (training, spare, degraded, excluded), the open alarm, the loop's next action."""
    st = read_state(job)
    if banners is None:
        try:
            banners = [l for l in open(os.path.join(job, 'data', 'banners.txt')) if l.strip()]
        except Exception:
            banners = []
    att = st.get('attempt', '-')
    tl = [l for l in _tail(os.path.join(job, 'training.log'), 40) if '|' in l and not l.startswith('#') and not l.startswith('time')]
    last = None
    for l in reversed(tl):
        v = [x.strip() for x in l.split('|')]
        if len(v) >= 7 and v[2].isdigit():
            last = v; break
    sv = _tail(os.path.join(job, 'data', 'save_times.tsv'), 1)
    sv = sv[0].split('\t') if sv else None
    train = [x for x in st.get('train_nodes', '').split(',') if x]
    excl = [x for x in st.get('excluded', '').split(',') if x]
    alln = [x for x in st.get('all_nodes', '').split(',') if x]
    degraded = [x for x in st.get('degraded', '').split(',') if x]        # node:test:value:threshold
    deg_nodes = [x.split(':')[0] for x in degraded]
    spare = [x for x in alln if x not in train and x not in excl]
    alarm = 'none: in the last beat no rank was still and no rank was behind its group.'
    try:
        t = json.load(open(os.path.join(job, 'data', 'agg_last.json')))
        if str(t.get('attempt')) == att and t.get('alarms') and st.get('status') == 'running':
            alarm = 'there is an open alarm: ' + ' | '.join(t['alarms'][:3]) + f'  (see heartbeat.log, the beat of {now_s(t["time"])}).'
    except Exception:
        pass
    states = {'gate': 'the gate is running (the nodes are being tested before the first attempt)', 'running': f'attempt {att} is training',
              'restarting': f'attempt {att} ended and the loop is classifying the failure and deciding the restart',
              'done': 'the loop has finished (the planned exit, a stop, or the wall)'}
    status = st.get('status', '-')
    deg_txt = ', '.join(f'{x.split(":")[0]} ({x.split(":")[1]} {x.split(":")[2]} against the threshold {x.split(":")[3]})' if x.count(':') >= 3 else x for x in degraded)
    L = [f'# STATUS of job {os.environ.get("SLURM_JOB_ID", os.path.basename(job))}, written {time.strftime("%Y-%m-%d %H:%M:%S")} by the heartbeat aggregator; it is rewritten every 10 s.', '',
         '```',
         f'state          the loop\'s state is "{status}": {states.get(status, "unknown")}.',
         f'attempt        the current attempt is {att}; ' + (f'{_plural(len(banners), "failure")} {"was" if len(banners) == 1 else "were"} recovered so far (recovery.log has one block per failure).' if banners else 'no failure has happened so far.'),
         (f'step           the last logged step is {last[2]} with loss {last[3]}; the throughput is {last[4]} TFLOP/s per GCD; the step took {last[5]} seconds; '
          f'the HBM allocated is at most {last[6]} GB on any rank (the training.log line of {last[0][11:]}, attempt {last[1]}).'
          if last else 'step           no step has been logged yet in training.log.'),
         (f'last save      the last save was the save of step {sv[0]}; it took {float(sv[1]):.0f} s and finished at {sv[2][11:] if len(sv) > 2 else "-"}.' if sv and len(sv) >= 2 else 'last save      no save has finished yet.'),
         f'nodes          {_plural(len(train), "node")} {"trains" if len(train) == 1 else "train"}: {", ".join(train) or "none"}.',
         f'               {_plural(len(spare), "spare node")} {"is" if len(spare) == 1 else "are"} left: {", ".join(spare) or "none"}' + (f' (the duty node, which runs the aggregator, is {st.get("duty")}).' if st.get('duty') else '.'),
         f'               degraded, usable as a last resort when no healthy spare is left: {deg_txt or "none"}.',
         f'               {_plural(len(excl), "node")} {"is" if len(excl) == 1 else "are"} excluded: {", ".join(x + (" (degraded)" if x in deg_nodes else "") for x in excl) or "none"}.',
         f'open alarm     {alarm}',
         f'alarms so far  {alarm_count(job)} alarm event(s) were raised in this job so far (one line each in alarms.log; the stack samples are in alarms/).',
         f'next action    {st.get("next", "-")}']
    # [2026-10-10, item UCP workers] the conversion duty's one-line state, composed by the restart loop's watcher (res_ucp_watch) into
    # data/ucp_state as a full sentence; absent when AUTORUN_UCP_WORKERS is 0
    try:
        ucp = open(os.path.join(job, 'data', 'ucp_state')).read().strip()
        if ucp:
            L.append(f'conversion     {ucp}')
    except OSError:
        pass
    L.append('```')
    _atomic(os.path.join(job, 'STATUS.md'), '\n'.join(L) + '\n')


class Agg:
    def __init__(self, a):
        self.a, self.job = a, a.job_dir
        self.offs, self.last, self.changed = {}, {}, {}
        self.prev_cnt = {}
        self.alarmed = set()
        self.attempt = None
        self.save_seen = {}          # rank -> epoch of the last line that showed a save (phase save* or flag save_bg)
        self.alarm_open = {}         # rank -> (epoch when the alarm opened, the rule) for the 'cleared' line of alarms.log
        self.file_mtime = {}         # node -> the node file's modification time at the last read (the delivery-lag rule)
        self.still_prev = set()      # the ranks older than the threshold at the previous beat (the two-beat rule)
        self.lag_beats = {}          # node -> consecutive beats in a delivery lag (the escalation sentence)
        for d in ('alarm', 'hb'):
            os.makedirs(os.path.join(self.job, 'data', d), exist_ok=True)
        os.makedirs(os.path.join(self.job, 'alarms'), exist_ok=True)      # the stack samples, at the top level
        if not os.path.exists(os.path.join(self.job, 'alarms.log')):
            open(os.path.join(self.job, 'alarms.log'), 'w').write(ALARMS_HEADER.format(job=os.environ.get('SLURM_JOB_ID', os.path.basename(self.job))))
        self.body = open(os.path.join(self.job, 'data', 'heartbeat_body.log'), 'a')
        self.prev_status = None

    def read(self):
        hbd = os.path.join(self.job, 'data', 'hb')
        for fn in sorted(os.listdir(hbd)):
            if not fn.endswith('.tsv'):
                continue
            p = os.path.join(hbd, fn)
            try:
                self.file_mtime[fn[:-4]] = os.stat(p).st_mtime     # the node file's own age, for the delivery-lag rule
                with open(p) as fh:
                    fh.seek(self.offs.get(p, 0))
                    data = fh.read()
                    # keep a partial last line for the next read
                    cut = data.rfind('\n') + 1
                    self.offs[p] = self.offs.get(p, 0) + len(data[:cut].encode())
                    data = data[:cut]
            except Exception:
                continue
            for line in data.split('\n'):
                if not line or line.startswith('#') or line.startswith('time'):     # the header lines of the files
                    continue
                v = line.split('\t')
                if len(v) < len(F):
                    continue
                d = dict(zip(F, v))
                try:
                    r = int(d['rank']); ep = float(d['epoch'])
                except ValueError:
                    continue
                if self.attempt is not None and d['attempt'] != self.attempt:
                    continue
                key = (d['step'], d['coll_seq'], d['phase'] if d['phase'].startswith('save') else '')
                old = self.last.get(r)
                if old is None or float(old['epoch']) <= ep:
                    if old is None or (old['step'], old['coll_seq'], old['phase'] if old['phase'].startswith('save') else '') != key:
                        self.changed[r] = ep
                    self.last[r] = d
                if d['phase'].startswith('save') or d['flag'].startswith('save'):
                    self.save_seen[r] = max(self.save_seen.get(r, 0.0), ep)

    def in_save_window(self, ranks, L, now):
        """True when any of these ranks reports a save phase or the save_bg flag now, or showed one less than SAVE_WINDOW_S ago.
        Every early-kill rule must check this and never act inside the window."""
        for r in ranks:
            d = L.get(r)
            if d is not None and (d['phase'].startswith('save') or d['flag'].startswith('save')):
                return True
            if now - self.save_seen.get(r, 0.0) < SAVE_WINDOW_S:
                return True
        return False

    def beat(self):
        st = read_state(self.job)
        att = st.get('attempt', '0')
        if att != self.attempt:                        # a new attempt: forget the old one's ranks and alarms
            self.attempt, self.last, self.changed, self.alarmed, self.prev_cnt, self.save_seen, self.alarm_open = att, {}, {}, set(), {}, {}, {}
            self.still_prev, self.lag_beats = set(), {}
        self.read()
        now = time.time()
        excl = set(x for x in st.get('excluded', '').replace(',', ' ').split() if x)
        world = int(st.get('world', self.a.world or 0) or 0)
        G = int(st.get('group', self.a.group))
        pp = int(st.get('pp', self.a.pp))
        L = {r: d for r, d in self.last.items() if d['node'] not in excl}
        status = st.get('status', 'running')
        out = []
        if self.prev_status == 'running' and status != 'running':
            self.heatmap(f'end of attempt {self.attempt}')      # the end of an attempt: the last running table
        self.prev_status = status
        if status != 'running' or not L:
            out.append(f'beat at {now_s(now)} (attempt {att}, the loop\'s state is {status}): ' +
                       ('no rank has reported a heartbeat line yet in this attempt.' if not L else f'{len(L)} ranks have reported; the aggregator waits for the state to be running.'))
            self._emit(out, None, st)
            return
        ages = {r: now - float(d['epoch']) for r, d in L.items()}
        missing = [r for r in range(world) if r not in L] if world else []
        steps = collections.Counter(d['step'] for d in L.values())
        step = steps.most_common(1)[0][0]
        oldest = max(ages.values())
        oldest_rank = max(ages, key=ages.get)
        rep = sum(1 for a in ages.values() if a <= self.a.still)
        # the groups and their save windows (used by the beat line, the stillness rule and the minority rule)
        groups = collections.defaultdict(list)
        for r in L:
            groups[r // G].append(r)
        save_groups = {g: self.in_save_window(rs, L, now) for g, rs in groups.items()}
        thr_of = {r: (max(SAVE_WINDOW_S, self.a.still) if save_groups.get(r // G) else self.a.still) for r in L}
        # THE DELIVERY-LAG RULE (see the docstring): per node, every rank equally old and older than the threshold, and the node file
        # itself as old as its lines -> the lines were late on their way here, not stale on the ranks
        by_node = collections.defaultdict(list)
        for r, d in L.items():
            by_node[d['node']].append(r)
        lagging = {}                                   # node -> the file's age in seconds
        # a delivery lag is a lag of SOME nodes while others are fresh (the 260-node event: about two dozen nodes late, the rest fresh);
        # when every node is stale at once (the end of a step, a global stall), nothing is a delivery lag and the plain rules apply
        any_fresh = any(ages[r] <= thr_of[r] for r in L)
        for node, rs in by_node.items():
            if not any_fresh:
                break
            a = [ages[r] for r in rs]
            if min(a) <= min(thr_of[r] for r in rs) or max(a) - min(a) > LAG_SPREAD_S:
                continue                               # a fresh rank on the node, or ranks of different ages: not a node-wide delivery lag
            fm = self.file_mtime.get(node)
            if fm is not None and (now - fm) >= min(a) - LAG_SPREAD_S:
                lagging[node] = now - fm
        for node in list(self.lag_beats):
            if node not in lagging:
                del self.lag_beats[node]
        for node in lagging:
            self.lag_beats[node] = self.lag_beats.get(node, 0) + 1
        lag_ranks = set(r for node in lagging for r in by_node[node])
        out.append(f'beat at {now_s(now)} (attempt {att}, step {step}): {rep} of {world or len(L)} ranks reported a heartbeat line within the last {self.a.still:.0f} s; '
                   f'the oldest line is {oldest:.0f} s old (rank {oldest_rank} on {L[oldest_rank]["node"]})' +
                   (f', which is older than the {thr_of[oldest_rank]:.0f}-s threshold' if oldest > thr_of[oldest_rank] else '') +
                   (' (a delivery lag of that node\'s file, not the rank: see the delivery-lag line)' if oldest_rank in lag_ranks else '') +
                   (f'; the ranks {missing[:8]}{"..." if len(missing) > 8 else ""} have not reported at all' if missing else '') +
                   (f'; {sum(1 for v in save_groups.values() if v)} of {len(groups)} groups are in a save window (threshold {SAVE_WINDOW_S:.0f} s)' if any(save_groups.values()) else '') + '.')
        ph = collections.Counter(d['phase'] for d in L.values())
        out.append('  where the ranks are:      ' + ', '.join(f'{v} {"rank is" if v == 1 else "ranks are"} in the phase {k}' for k, v in ph.most_common()) + '.')
        stt = [float(d['step_time_s']) for d in L.values() if d['step_time_s'] not in ('-', '')]
        busy = {r: float(d['busy_s']) for r, d in L.items() if d['busy_s'] not in ('-', '')}
        per_stage = max(1, (world or len(L)) // max(1, pp))
        stage_busy = collections.defaultdict(list)
        for r, b in busy.items():
            stage_busy[r // per_stage].append(b)
        if stt:
            st_med = statistics.median(stt)
            pred = f' (the prediction was {self.a.pred:.1f} s)' if self.a.pred else ''
            sb = {s: statistics.median(v) for s, v in stage_busy.items()}
            if sb and st_med > 0:
                hi = max(sb, key=sb.get); lo = min(sb, key=sb.get)
                stage_txt = (f'the busiest pipeline stage is stage {hi}, busy {sb[hi]:.1f} s of the {st_med:.1f} s step (a bubble of {100 * (1 - sb[hi] / st_med):.0f} percent)'
                             + (f'; the least busy stage is stage {lo}, busy {sb[lo]:.1f} s' if lo != hi else ''))
            else:
                stage_txt = 'the busy time per stage is not known yet'
            out.append(f'  step time:                the median step time over the reporting ranks is {st_med:.2f} s{pred}; {stage_txt}.')
        else:
            out.append('  step time:                no step has finished yet in this attempt.')
        slow = []
        for r, b in busy.items():
            mates = [x for rr, x in busy.items() if rr // per_stage == r // per_stage and rr != r]
            if len(mates) >= 2:
                m = statistics.median(mates)
                if m > 0.2 and b > 1.3 * m:
                    slow.append((r, b, m))
        out.append('  slowest rank:             ' + ('no rank is busier than 1.3 x the median of its stage mates.' if not slow else
                   '; '.join(f'rank {r} on {L[r]["node"]} is busy {b:.1f} s against {m:.1f} s for its stage mates ({b / m:.1f} x)' for r, b, m in sorted(slow, key=lambda x: -x[1] / x[2])[:4]) + '.'))
        dl = {'ECC correctable': 'ecc_ce', 'ECC uncorrectable': 'ecc_ue', 'xGMI': 'xgmi_err'}
        parts = []
        for lab, k in dl.items():
            tot, where = 0, set()
            for r, d in L.items():
                try:
                    v = int(d[k])
                except ValueError:
                    continue
                p = self.prev_cnt.get((r, k))
                if p is not None and v >= 0 and p >= 0 and v > p:
                    tot += v - p; where.add(d['node'])
                self.prev_cnt[(r, k)] = v
            parts.append(f'{lab} +{tot}' + (f' on {",".join(sorted(where))}' if where else ''))
        out.append('  counters since last beat: ' + ', '.join(parts) + ' (the sums of the growth over all reporting ranks; a node is named when its counter grew).')
        hm = [(float(d['host_mem_pct']), d['node']) for d in L.values() if d['host_mem_pct'] not in ('-', '')]
        if hm:
            mx = max(hm)
            hbm = max(float(d['hbm_gb']) for d in L.values())
            out.append(f'  host memory:              the highest host memory use is {mx[0]:.0f} percent, on {mx[1]}; the highest HBM allocated on any rank is {hbm:.1f} GB.')
        # ---- the delivery lag (in the beat block, never an alarm) and the first beat of a stillness (pending, named at the next beat)
        if lagging:
            parts = []
            for node, lag in sorted(lagging.items()):
                k = self.lag_beats.get(node, 1)
                parts.append(f'{node} ({lag:.0f} s, {k} beat{"s" if k != 1 else ""}' +
                             (f': a lag this long is more likely a dead node writer or a lost node than Lustre, which the loop\'s srun watch decides' if k >= LAG_ESCALATE_BEATS else '') + ')')
            out.append(f'  delivery lag:             delivery lag {max(lagging.values()):.0f} s on {len(lagging)} node{"s" if len(lagging) != 1 else ""} (Lustre), not a rank: '
                       f'every rank of the node is equally old and the node file itself stopped changing with its lines, so the lines are late on their way here; '
                       f'these ranks are not named by the stillness or the minority rule and no stack sample is taken: ' + '; '.join(parts) + '.')
        # ---- the alarms
        alarms = []
        save_note = '; this rank is in a save; the threshold is 120 s; the last stage\'s ranks write the largest files'
        still_now = sorted(r for r, a in ages.items() if a > thr_of[r] and r not in lag_ranks)
        still = sorted(r for r in still_now if r in self.still_prev)          # the two-beat rule: still at the previous beat too
        pending = [r for r in still_now if r not in self.still_prev]
        self.still_prev = set(still_now)
        if pending:
            out.append(f'  stillness pending:        {len(pending)} rank{"s are" if len(pending) != 1 else " is"} older than the threshold for the first beat '
                       f'(ranks {pending[:8]}{"..." if len(pending) > 8 else ""}, on {",".join(sorted(set(L[r]["node"] for r in pending)))}); '
                       f'a rank is named only when it is still on the next beat too.')
        for r in still:
            d = L[r]
            alarms.append(('stillness', r, f'rank {r} on {d["node"]}: its last heartbeat line is {ages[r]:.0f} s old (written at {d["time"][11:]}), '
                                           f'older than the threshold of {thr_of[r]:.0f} s on two consecutive beats; its phase is {d["phase"]}, its collective count {d["coll_seq"]}, '
                                           f'its process state {d["proc_state"]}' + (save_note if save_groups.get(r // G) else '')))
        minority = set()
        for g, rs in sorted(groups.items()):
            # the minority rule's threshold: 30 s, or 120 s when the group is in a save window (a save legitimately holds ranks at
            # different collective counts: some write, some wait in its barriers; the save deadline D3 is the save's own detector)
            thr = max(SAVE_WINDOW_S, self.a.still) if save_groups.get(g) else self.a.still
            if len(rs) < 2 or any(now - self.changed.get(r, now) < thr for r in rs):
                continue
            if any(r in lag_ranks for r in rs):
                continue                               # a group with a rank on a lagging node: its counts are late, not behind
            cs = {r: int(L[r]['coll_seq']) for r in rs}
            mx = max(cs.values())
            behind = sorted(r for r in rs if cs[r] < mx)
            if behind and len(behind) < len(rs):
                minority.update(behind)
                nodes = sorted(set(L[r]['node'] for r in behind))
                frozen = min(now - self.changed.get(r, now) for r in rs)
                alarms.append(('minority', behind[0], f'ranks {behind} (on {",".join(nodes)}) are at collective {cs[behind[0]]}, phase {L[behind[0]]["phase"]}; '
                               f'the other ranks of group {g} (ranks {min(rs)}-{max(rs)}) are at collective {mx}, phase {L[[r for r in rs if cs[r] == mx][0]]["phase"]}; '
                               f'nobody in the group moved for {frozen:.0f} s (threshold {thr:.0f} s)' + (save_note if save_groups.get(g) else '')))
        for kind, r, txt in alarms:
            out.append(f'  ALARM {kind:10s} {txt}')
        named = set(still) | minority
        new = named - self.alarmed
        cleared = self.alarmed - named
        for r in sorted(cleared):          # the alarm on this rank is over: it moved again, or its group caught up
            self.alarmed.discard(r)
            t_open, rule = self.alarm_open.pop(r, (now, '?'))
            d = L.get(r)
            where = f'rank {r}' + (f' on {d["node"]}' if d else '')
            alarm_log(self.job, f'cleared at {now_s(now)} after {now - t_open:.0f} s: the {rule} alarm on {where} is over ' +
                      ('(the rank wrote a fresh line and is no longer behind its group)' if d else '(the rank is no longer reporting in this attempt)') + '.')
        if new:
            self.alarmed |= new
            open(os.path.join(self.job, 'data', 'alarm', 'ALARM'), 'a').write(f'{now_s(now)} attempt {att}: ranks {sorted(new)}\n')
            bynode = collections.defaultdict(list)
            for r in new:
                bynode[L[r]['node']].append(L[r]['pid'])
            samples = {}
            for node, pids in sorted(bynode.items()):
                fn = os.path.join(self.job, 'alarms', f'{node}_{time.strftime("%H%M%S")}.txt')
                self.sample(node, pids, fn)
                samples[node] = fn
                out.append(f'  action:                   a stack sample of {node} (process ids {",".join(pids)}) is taken through srun --overlap with a 60-s timeout; it goes to alarms/{os.path.basename(fn)}.')
            for r in sorted(new):
                d = L[r]
                rule = 'minority' if r in minority else 'stillness'
                self.alarm_open[r] = (now, rule)
                in_save = bool(save_groups.get(r // G))
                alarm_log(self.job, f'ALARM {rule} at {now_s(now)} (attempt {att}, step {step}): rank {r} on node {d["node"]} was named; its phase is {d["phase"]}, '
                          f'its collective count {d["coll_seq"]}; its last heartbeat line was {ages[r]:.0f} s old (process state {d["proc_state"]}); '
                          + ('it is in a save window, so the threshold was 120 s (the last stage\'s ranks write the largest files)' if in_save else f'it is not in a save window (threshold {thr_of[r]:.0f} s)')
                          + f'; a stack sample was taken to alarms/{os.path.basename(samples.get(d["node"], "-"))}.')
        table = {'time': now, 'attempt': att, 'step': step, 'still': still, 'still_pending': pending, 'lag': {n: round(v, 1) for n, v in lagging.items()},
                 'minority': sorted(minority), 'slow': [r for r, _, _ in slow],
                 'ranks': {r: {k: L[r][k] for k in ('node', 'phase', 'step', 'coll_seq', 'proc_state', 'pid')} | {'age': round(ages[r], 1)} for r in L},
                 'alarms': [f'{k}: {t}' for k, r, t in alarms], 'save_window_groups': sorted(g for g, v in save_groups.items() if v)}
        self._emit(out, table, st)
        if new:
            self.heatmap(f'alarm on ranks {sorted(new)}')

    def sample(self, node, pids, fn):
        env = {k: v for k, v in os.environ.items() if not k.startswith('SLURM_STEP') and k not in ('SLURM_JOB_NUM_NODES',)}
        pyspy = os.path.expanduser('~/.local/bin/py-spy')
        inner = f'for p in {" ".join(pids)}; do echo "=== pid $p ==="; timeout 50 {pyspy} dump --pid $p --native 2>&1 | head -80; done'
        cmd = ['timeout', '70', 'srun', '--overlap', '--jobid', os.environ.get('SLURM_JOB_ID', ''), '-w', node, '-N1', '-n1',
               '--cpu-bind=none', 'bash', '-c', inner]
        try:
            fh = open(fn, 'w')
            fh.write(f'# {time.ctime()} {" ".join(cmd)}\n'); fh.flush()
            subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        except Exception as e:
            open(fn, 'a').write(f'sampler failed: {e}\n')

    def heatmap(self, why):
        """heatmap.png from data/agg_last.json, in a child process (drawn on an alarm and at the end of an attempt only)."""
        try:
            subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'heatmap.py'), '--job-dir', self.job,
                              '--why', why], stdout=subprocess.DEVNULL, stderr=open(os.path.join(self.job, 'data', 'heatmap.err'), 'a'))
        except Exception:
            pass

    def _emit(self, out, table, st):
        self.body.write('\n'.join(out) + '\n'); self.body.flush()
        if table is not None:
            _atomic(os.path.join(self.job, 'data', 'agg_last.json'), json.dumps(table))
        compose(self.job)
        _atomic(os.path.join(self.job, 'data', 'aggregator.alive'), f'{time.time():.0f} {os.uname().nodename} {os.getpid()}\n')


def _dashboard_once(job, timeout):
    """[the run console, 2026-10-10, sandbox] one rewrite of <job>/dashboard.html by dashboard_html.py, in a separate process at
    nice 19 and the idle I/O class, so the beats never wait on it and the page never competes with anything else on this node."""
    import shutil
    pre = (['nice', '-n', '19'] if shutil.which('nice') else []) + (['ionice', '-c', '3'] if shutil.which('ionice') else [])
    try:
        r = subprocess.run(pre + [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard_html.py'), job],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stderr or '').strip().split('\n')[-1][:300]
    except Exception as e:
        return -1, f'{type(e).__name__}: {e}'


def _dashboard_loop(job, every):
    """the page every `every` seconds; the first five failures are written to data/dashboard.err, then it keeps quiet"""
    errs = 0
    while True:
        t0 = time.time()
        rc, msg = _dashboard_once(job, max(120, 4 * every))
        try:                                         # when each rewrite ran: the evidence that it does not slow the steps it overlaps
            with open(os.path.join(job, 'data', 'dashboard_times.tsv'), 'a') as fh:
                fh.write(f'{t0:.2f}\t{time.time():.2f}\t{rc}\n')
        except OSError:
            pass
        if rc != 0 and errs < 5:
            errs += 1
            with open(os.path.join(job, 'data', 'dashboard.err'), 'a') as fh:
                fh.write(f'{now_s()} the page was not rewritten (exit {rc}): {msg}\n')
        time.sleep(max(5.0, every - (time.time() - t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job-dir', dest='job', required=True)
    ap.add_argument('--world', type=int, default=0)
    ap.add_argument('--group', type=int, default=8)
    ap.add_argument('--pp', type=int, default=1)
    ap.add_argument('--interval', type=float, default=10)
    ap.add_argument('--still', type=float, default=30)
    ap.add_argument('--pred', type=float, default=0)
    ap.add_argument('--compose', action='store_true')
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--dashboard-every', type=float, default=0, help='rewrite <job>/dashboard.html every N s (0 = off)')
    a = ap.parse_args()
    a.job_dir = a.job
    if a.compose:
        compose(a.job); return
    g = Agg(a)
    print(f'[aggregator {time.ctime()}] on {os.uname().nodename} pid {os.getpid()} job dir {a.job}', flush=True)
    if a.dashboard_every > 0 and not a.once:
        import threading
        threading.Thread(target=_dashboard_loop, args=(a.job, a.dashboard_every), daemon=True, name='dashboard').start()
        print(f'[aggregator] the run console {a.job}/dashboard.html is rewritten every {a.dashboard_every:.0f} s (nice 19)', flush=True)
    while True:
        t0 = time.time()
        try:
            g.beat()
        except Exception as e:
            import traceback; traceback.print_exc()
            g.body.write(f'beat at {now_s()}: the aggregator hit an error: {type(e).__name__}: {e}\n'); g.body.flush()
        if a.once:
            break
        st = read_state(a.job)
        if st.get('status') == 'done':
            if a.dashboard_every > 0:
                _dashboard_once(a.job, 300)          # the last page shows the end of the job
            break
        time.sleep(max(1.0, a.interval - (time.time() - t0)))


if __name__ == '__main__':
    main()
