#!/usr/bin/env python3
"""dashboard.py (2026-10-10, stage 1 of the run dashboard): one text view of a resilient job, read from its job folder only.
READ-ONLY by design (the user, 2026-10-10 13:05: "make sure the probing originates from existing logs and do not ever ever interrupt
the training"): it only opens files the job already writes, never writes into the job folder, never signals or queries the job.

    python3 dashboard.py <job folder> [--out <file>] [--width 100]

It never writes into the job folder (the user's rule for running jobs): the view goes to --out (default: stdout).
The layout is the machine as the model sees it: one row per pipeline stage; in each row the data-parallel replicas; in each
replica the nodes (one character per node, 8 GCDs each). Under the grid: the spare, degraded, excluded and UCP-worker nodes,
then a one-line timeline. Every value names the file it came from in the source below; nothing is estimated except where the
line says so.

Inputs (all under the job folder):
  data/state                  pp, world, attempt, status (written by the restart loop)
  attempts/attempt_<a>/ranks/rank_0.log   "using world size: W, data-parallel-size: D, sequence-parallel size: U,
                              tensor-model-parallel size: T, pipeline-model-parallel size: P" and the save directory
  data/hb/<node>.tsv          one line per rank every 10 s: time node rank step phase coll_seq ... epoch
  data/agg_last.json          the heartbeat reader's last beat: still, still_pending, minority, slow, lag
  STATUS.md                   the nodes block (training / spare / degraded / excluded / duty) and the conversion line
  training.log                one line per step: time | attempt | step | loss | TFLOP/s | step s | HBM GB | note
  alarms.log                  one line per alarm event
  data/loop.log               the gate's duration, the attempts, the job's end time
  data/save_times.tsv         step, seconds, time of every save
  <save dir>/latest, latest_universal
"""
import argparse, glob, json, os, re, sys, time

PHASE_CHAR = [('forward', 'f'), ('backward', 'b'), ('save', 's'), ('optim', 'o'), ('step', 'o'), ('init', 'i'), ('load', 'i'),
              ('build', 'i'), ('all_to_all', 'c'), ('broadcast', 'c'), ('all_reduce', 'c'), ('allreduce', 'c'), ('barrier', 'c'),
              ('p2p', 'c'), ('send', 'c'), ('recv', 'c')]
STALE_S = 60


def read(path, default=''):
    try:
        return open(path, errors='replace').read()
    except OSError:
        return default


def tail_lines(path, nbytes=65536):
    try:
        with open(path, 'rb') as fh:
            fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - nbytes))
            data = fh.read().decode(errors='replace')
    except OSError:
        return []
    lines = data.split('\n')
    return lines[1:] if size > nbytes else lines


def phase_char(phase):
    p = (phase or '').lower()
    for key, ch in PHASE_CHAR:
        if key in p:
            return ch
    return '.'


def layout(job):
    st = dict(l.split('=', 1) for l in read(os.path.join(job, 'data', 'state')).split('\n') if '=' in l)
    att = st.get('attempt', '0')
    r0 = os.path.join(job, 'attempts', f'attempt_{att}', 'ranks', 'rank_0.log')
    txt = ''
    try:                                     # the argument dump is in the first ~2 MB of rank 0's log: read the head once per call
        with open(r0, 'rb') as fh:
            txt = fh.read(2 * 1024 * 1024).decode(errors='replace')
    except OSError:
        pass
    m = re.search(r'using world size: (\d+), data-parallel-size: (\d+), sequence-parallel size: (\d+), '
                  r'tensor-model-parallel size: (\d+), pipeline-model-parallel size: (\d+)', txt)
    if m:
        W, D, U, T, P = map(int, m.groups())
    else:
        W = int(st.get('world', 0) or 0); P = int(st.get('pp', 1) or 1); D = U = T = 1
    s = re.search(r'^\s*save \.+ (\S+)', txt, re.M)
    return dict(world=W, dp=D, u=U, tp=T, pp=P, attempt=att, status=st.get('status', '?'), save=s.group(1) if s else None)


def heartbeats(job):
    """{rank: dict(node, step, phase, epoch)} from the newest line of every rank."""
    last = {}
    for f in glob.glob(os.path.join(job, 'data', 'hb', '*.tsv')):
        for line in tail_lines(f, 8192):
            v = line.split('\t')
            if len(v) < 19 or v[0] == 'time':
                continue
            try:
                r, ep = int(v[2]), float(v[18])
            except ValueError:
                continue
            if r not in last or ep >= last[r]['epoch']:
                last[r] = dict(node=v[1], step=v[3], phase=v[4], epoch=ep, hbm=v[8])
    return last


def status_nodes(job):
    txt = read(os.path.join(job, 'STATUS.md'))
    out = dict(spares=[], duty=None, degraded=[], excluded=[], conversion='')
    m = re.search(r'spare nodes? (?:is|are) left: ([^\n(]*)', txt)
    if m:
        out['spares'] = [x.strip() for x in m.group(1).split(',') if x.strip().startswith('frontier')]
    m = re.search(r'the duty node, which runs the aggregator, is (frontier\d+)', txt)
    out['duty'] = m.group(1) if m else None
    m = re.search(r'degraded, usable as a last resort[^:]*: ([^\n]*)', txt)
    if m and 'none' not in m.group(1)[:6]:
        out['degraded'] = re.findall(r'(frontier\d+) \(([^)]*)\)', m.group(1))
    m = re.search(r'(\d+) nodes? (?:is|are) excluded: ([^\n]*)', txt)
    if m and m.group(2).strip() != 'none.':
        out['excluded'] = re.findall(r'(frontier\d+) \(([^)]*)\)', m.group(2))
    m = re.search(r'^conversion\s+(.*)$', txt, re.M)
    out['conversion'] = m.group(1).strip() if m else ''
    return out


def last_step(job):
    for line in reversed(tail_lines(os.path.join(job, 'training.log'), 16384)):
        v = [x.strip() for x in line.split('|')]
        if len(v) >= 7 and v[2].isdigit():
            return dict(time=v[0], attempt=v[1], step=int(v[2]), loss=v[3], tflops=v[4], step_s=v[5], hbm=v[6])
    return None


def ucp_workers(job):
    m = re.search(r'the 2 worker node\(s\) ([\w ]+?) were taken|worker node\(s\) (frontier[\d ]+frontier\d+)', read(os.path.join(job, 'data', 'loop.log')))
    if not m:
        return []
    return re.findall(r'frontier\d+', m.group(0))


def timeline(job):
    loop = read(os.path.join(job, 'data', 'loop.log'))
    g = re.search(r'gate: the gate ended after (\d+) s on (\d+) nodes', loop)
    e = re.search(r"the job's end time is (\S+ \S+)", loop)
    atts = re.findall(r'^(\S+ \S+) attempt (\d+) started', loop, re.M)
    saves = []
    for line in read(os.path.join(job, 'data', 'save_times.tsv')).split('\n'):
        v = line.split('\t')
        if len(v) >= 3 and v[0].isdigit() and v[0] not in saves:
            saves.append(v[0])
    return dict(gate=(g.group(1), g.group(2)) if g else None, end=e.group(1) if e else None, attempts=atts, saves=saves)


def conv_short(c, now):
    """the STATUS.md conversion sentence in a few words: 'converting step N since HH:MM (M min); last universal K'"""
    m = re.search(r'conversion of step (\d+) to the universal form is running since (\d+:\d+):?(\d+)?', c)
    u = re.search(r'last complete universal is (\S+?)[ .(]', c + ' ')
    if m:
        t0 = time.mktime(time.strptime(time.strftime('%Y-%m-%d ') + m.group(2) + ':' + (m.group(3) or '00'), '%Y-%m-%d %H:%M:%S'))
        return f"converting step {m.group(1)} since {m.group(2)} ({(now - t0) / 60:.0f} min); last complete universal {u.group(1) if u else '?'}"
    return c[:90]


def ckpt_tags(save):
    if not save:
        return '-', '-'
    rd = lambda n: (read(os.path.join(save, n)).strip() or '-').replace('global_step', '').replace('_universal', '')
    return rd('latest'), rd('latest_universal')


def render(job, width=100):
    now = time.time()
    L = layout(job); hb = heartbeats(job); S = status_nodes(job); st = last_step(job); T = timeline(job)
    agg = {}
    try:
        agg = json.load(open(os.path.join(job, 'data', 'agg_last.json')))
    except (OSError, ValueError):
        pass
    flagged = set()
    for k in ('still', 'minority', 'slow'):
        for r in agg.get(k) or []:
            try:
                flagged.add(int(r if not isinstance(r, dict) else r.get('rank')))
            except (TypeError, ValueError):
                pass
    nalarm = read(os.path.join(job, 'alarms.log')).count(' ALARM ')
    latest, latest_u = ckpt_tags(L['save'])
    jobid = re.search(r'job_(\d+)', os.path.basename(job.rstrip('/')))
    out = []
    head = f"job {jobid.group(1) if jobid else '?'}   attempt {L['attempt']} ({L['status']})"
    if st:
        age = (now - time.mktime(time.strptime(st['time'], '%Y-%m-%d %H:%M:%S'))) / 60
        head += (f"   step {st['step']} -> {st['step'] + 1} ({age:.0f} min in)   loss {st['loss']}   {st['tflops']} TFLOP/s/GCD"
                 f"   step {float(st['step_s']) / 60:.1f} min   HBM {st['hbm']} GB")
    out.append(head)
    out.append(f"latest = step {latest}    latest_universal = {latest_u}    alarms so far {nalarm}    "
               f"wall {T['end'][11:16] if T['end'] else '?'}    written {time.strftime('%H:%M:%S')}")
    out.append('')
    W, P, D = L['world'], L['pp'], L['dp']
    per_stage = W // P if P else W
    per_rep = per_stage // D if D else per_stage
    nodes_per_rep = max(1, per_rep // 8)
    out.append(f"PP {P} stages x DP {D} replicas x {nodes_per_rep} nodes x 8 GCDs   (each replica: TP {L['tp']} x Ulysses {L['u']} = {L['tp'] * L['u']} GCDs)")
    out.append(f"{'stage':<8}" + ' '.join(f"{'replica ' + str(r):<{nodes_per_rep + 2}}" for r in range(D)) + '   phase (ranks)   step')
    for s in range(P):
        cells = []
        phases = {}
        steps = set()
        for rep in range(D):
            row = ''
            for n in range(nodes_per_rep):
                r0 = s * per_stage + rep * per_rep + n * 8
                rs = [hb.get(r) for r in range(r0, r0 + 8)]
                rs_ok = [x for x in rs if x]
                if not rs_ok:
                    row += ' '; continue
                newest = max(x['epoch'] for x in rs_ok)
                chars = [phase_char(x['phase']) for x in rs_ok]
                for x in rs_ok:
                    phases[x['phase']] = phases.get(x['phase'], 0) + 1; steps.add(x['step'])
                ch = max(set(chars), key=chars.count)
                if any(r in flagged for r in range(r0, r0 + 8)):
                    ch = '!'
                elif now - newest > STALE_S:
                    ch = '?'
                row += ch
            cells.append(f"[{row}]")
        ph = ', '.join(f'{k} {v}' for k, v in sorted(phases.items(), key=lambda kv: -kv[1])[:2])
        out.append(f"{s:<8}" + ' '.join(cells) + f"   {ph}   {'/'.join(sorted(steps)) or '-'}")
    out.append("        one character = one node: f forward  b backward  c collective  s save  o optimizer  i start-up  "
               ". other")
    out.append("        ! named by the heartbeat reader (still / behind its group / slow)   ? no heartbeat line for 60 s")
    out.append('')
    duty = S['duty']
    sp = ' '.join(f"[{n.replace('frontier', '')}{' duty' if n == duty else ''}]" for n in S['spares']) or '(none)'
    out.append(f"SPARES       {sp}")
    def _val(why):
        m = re.search(r'([\d.]+) against', why)
        return float(m.group(1)) if m else 0.0
    deg = sorted(S['degraded'], key=lambda nw: -_val(nw[1]))           # the loop takes the highest value first
    thr = re.search(r'against the threshold ([\d.]+)', S['degraded'][0][1]) if S['degraded'] else None
    dg = ' '.join(f"[{n.replace('frontier', '')} {why.split(' against')[0]}]" for n, why in deg) or '(none)'
    if thr:
        dg += f"   threshold {thr.group(1)}"
    out_conv = conv_short(S['conversion'], now)
    out.append(f"DEGRADED     {dg}   (last resort, best first)")
    ex = ' '.join(f"[{n.replace('frontier', '')} {why}]" for n, why in S['excluded']) or '(none)'
    out.append(f"EXCLUDED     {ex}")
    uw = ' '.join(f"[{n.replace('frontier', '')}]" for n in ucp_workers(job)) or '(none)'
    out.append(f"UCP WORKERS  {uw}   {out_conv}")
    tl = []
    if T['gate']:
        tl.append(f"gate {T['gate'][0]} s on {T['gate'][1]} nodes")
    tl += [f"attempt {a} {t[11:16]}" for t, a in T['attempts']]
    if T['saves']:
        tl.append('saves ' + ' '.join(T['saves'][-6:]))
    if T['end']:
        tl.append(f"wall {T['end'][11:16]}")
    out.append('TIMELINE     ' + ' | '.join(tl))
    return '\n'.join(l[:max(width, 140)] for l in out) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('job'); ap.add_argument('--out'); ap.add_argument('--width', type=int, default=100)
    a = ap.parse_args()
    text = render(a.job, a.width)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        tmp = a.out + '.tmp'
        open(tmp, 'w').write(text); os.replace(tmp, a.out)
    else:
        sys.stdout.write(text)


if __name__ == '__main__':
    main()
