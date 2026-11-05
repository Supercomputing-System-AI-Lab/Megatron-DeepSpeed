#!/usr/bin/env python3
"""dashboard_html.py (2026-10-10, sandbox): the run console, one self-contained HTML page written into the job folder.

    python3 dashboard_html.py <job folder> [--out <file>] [--history <json>] [--evals <json>] [--title <text>]

READ-ONLY on the job (the user's rule: "do not ever ever interrupt the training"). It opens only files the job already writes
(data/hb/<node>.tsv, heartbeat.log, gate.log, data/gate_report.tsv, data/mem_gate.txt, STATUS.md, training.log, data/loop.log,
data/save_times.tsv, data/save_async_times.tsv, alarms.log, the head of rank logs, the tail of the last rank's log, the checkpoint
folder) and writes two files: <job>/dashboard.html (atomically, through a .tmp) and data/dashboard_cache.json, its own memory.

The cost per call stays small at any scale, because nothing is read whole:
  - every node file: its first 8 KB once (the ECC count at job start, kept in the cache) and its last ~62 KB (30 beats per rank);
  - heartbeat.log and the last rank's log: only the bytes added since the previous call (an offset in the cache);
  - the rank-log heads (the stage labels, the argument dump): once per attempt (kept in the cache).
The heartbeat reader (hb_aggregator.py --dashboard-every N) runs it as a separate process at nice 19 and the idle I/O class, on
the duty node (a spare, which never trains) or, without a spare, on the batch node; it never runs on a training node.

--history: a JSON file {"meta": [{"k", "name", "dash"?}], "series": {k: [points]}} with earlier jobs of the same campaign, drawn
under this job's curves (a point: step, loss, tflops, step_s, gnorm, job). Default: $AUTORUN_DASHBOARD_HISTORY.
--evals: a JSON list [{"run": <checkpoint folder name>, "step": n, "tasks": {task: {"score": x}}, "jobs": [...]}] with the
evaluation results of this run's checkpoints. Default: $AUTORUN_DASHBOARD_EVALS.
"""
import argparse, glob, json, os, re, statistics, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dashboard as D

PH = D.phase_char


def _head(path, n):
    try:
        with open(path, 'rb') as fh:
            return fh.read(n).decode(errors='replace')
    except (OSError, TypeError):
        return ''


def _new_text(path, cache, key, first_cap=8 * 1024 * 1024):
    """the text added to `path` since the offset kept in cache[key]; the first call reads at most the last first_cap bytes"""
    try:
        size = os.path.getsize(path)
    except OSError:
        return ''
    off = cache.get(key, max(0, size - first_cap))
    if off > size:                                         # the file was replaced: start again from its end window
        off = max(0, size - first_cap)
    try:
        with open(path, 'rb') as fh:
            fh.seek(off); b = fh.read(size - off)
    except OSError:
        return ''
    cut = b.rfind(b'\n') + 1                               # keep a partial last line for the next call
    cache[key] = off + cut
    return b[:cut].decode(errors='replace')


def _ranklog(job, att, r):
    p = os.path.join(job, 'attempts', f'attempt_{att}', 'ranks', f'rank_{r}.log')
    return p if os.path.exists(p) else None


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None



def _gate_details(txt, names):
    """[2026-10-13, the user: the gate table raised questions it should answer] per test, from the job's own gate.log DETAIL blocks
    (model-agnostic, log-only): what it checks, its calibration and pass line, its time allowance, and every node that did not pass
    (value, verdict, reason, run time). Returns {test: dict(desc, budget, rule, worst, bad=[[node, value, verdict, reason, run_s]])}."""
    out, cur = {}, None
    for line in txt.split('\n'):
        m = re.match(r'^([a-z_0-9]+): (.+)$', line)
        if m and m.group(1) in names:
            cur = out.setdefault(m.group(1), dict(desc=m.group(2).strip(), budget='', rule='', worst='', bad=[]))
            continue
        if cur is None:
            continue
        t = line.strip()
        if t.startswith('budget:'):
            cur['budget'] = t[len('budget:'):].strip()
        elif t.startswith('the worst node'):
            cur['worst'] = t
        elif re.match(r'^frontier\d+ ', t) and '|' in t:
            c = [x.strip() for x in t.split('|')]
            node, val = c[0].split()[0], (c[0].split()[1] if len(c[0].split()) > 1 else '')
            if not cur['rule'] and len(c) > 2 and c[2] and c[2] != '-':
                cur['rule'] = c[2]
            if len(c) > 3 and c[3].lower() not in ('pass', 'ok', 'n/t'):        # n/t = not tested: the row's verdict says why
                cur['bad'].append([node, val, c[3], c[4] if len(c) > 4 else '', c[5] if len(c) > 5 else ''])
        elif not line.startswith(' ') and t and not t.startswith('...'):
            cur = None
    for v in out.values():
        if len(v['bad']) > 12:                       # a test that failed everywhere: the table says how many, the rows name the first 12
            v['bad_more'] = len(v['bad']) - 12; v['bad'] = v['bad'][:12]
    return out


DEV_GIB = 64.0          # the device memory of one MI250X GCD, GiB (rocm-smi's total); the monitor records used / total in percent
GIB = 1.073741824       # GB per GiB: the trainer logs allocated memory in GB (bytes / 1e9)


def vram_samples(job, C, att_start):
    """[2026-10-13, the user: "I only want to care about the memory usage shown on the GPU HBM, not how the model allocated"] the device
    memory used per GCD (VRAM: rocm-smi --showmeminfo vram, used / total, in whole percent, so 0.64 GiB steps) as monitor_run.sh
    records it in data/monitor/monitor.tsv for every rank on every sweep (every 60 s, then every 180 s); read incrementally. Keeps in
    the cache: last[rank] = [epoch, GiB], peak[rank] = GiB since the current attempt started, ts[epoch] = the largest GiB of that sweep."""
    M = C.setdefault('mon', dict(last={}, peak={}, ts={}, att=None))
    if M.get('att') != att_start:                          # a new attempt: its peaks start again (a new layout may hold less or more)
        M['peak'] = {}; M['att'] = att_start
    for line in _new_text(os.path.join(job, 'data', 'monitor', 'monitor.tsv'), C, 'mon_offset').split('\n'):
        v = line.split('\t')
        if len(v) < 8 or not v[1].isdigit() or not v[3].isdigit() or not v[7].isdigit():
            continue                                       # the header, an unsampled rank ('?'), a partial line
        e, r, g = int(v[1]), v[3], round(int(v[7]) * DEV_GIB / 100, 2)
        if r not in M['last'] or e >= M['last'][r][0]:
            M['last'][r] = [e, g]
        if att_start is None or e >= att_start:
            M['peak'][r] = max(M['peak'].get(r, 0), g)
        M['ts'][str(e)] = max(M['ts'].get(str(e), 0), g)
    if len(M['ts']) > 4000:
        M['ts'] = dict(sorted(M['ts'].items(), key=lambda kv: int(kv[0]))[-4000:])
    return M


def _epoch(t):
    try:
        return time.mktime(time.strptime(t[:19].replace('T', ' '), '%Y-%m-%d %H:%M:%S'))
    except (TypeError, ValueError):
        return None


def build(job, history=None, evals=None, title=None, cpath=None):
    now = time.time()
    cpath = cpath or os.path.join(job, 'data', 'dashboard_cache.json')
    try:
        C = json.load(open(cpath))
    except (OSError, ValueError):
        C = {}
    if C.get('job') != os.path.abspath(job):              # a cache belongs to one job folder: another job's facts are never reused
        C = {'job': os.path.abspath(job)}
    L = D.layout(job); S = D.status_nodes(job); T = D.timeline(job)
    if L['save'] and not os.path.isabs(L['save']):        # the trainer's --save is relative to the launch folder (scripts-frontier = the
        L['save'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(job))), L['save'])   # parent of logs/)
    W, P, DP, U, TP = L['world'], max(1, L['pp']), max(1, L['dp']), L['u'], L['tp']
    att = L['attempt']; per_stage = max(1, W // P); per_rep = max(1, per_stage // DP)
    # ---- once per attempt: the argument dump and the stage labels, from the heads of the rank logs
    A = C.get('attempt_facts', {})
    # [2026-10-13] re-read until the argument dump is there: a first build before rank 0 wrote it cached ep 1 / experts 0 / seq 0
    # and, with one pipeline stage, marked them final for the whole attempt (job 5662696's page said 'EP 1' and drew no map)
    if A.get('attempt') != att or not A.get('labels_done') or not A.get('seq'):
        h0 = _head(_ranklog(job, att, 0), 2 * 1024 * 1024)
        g = lambda pat, d: int((re.search(pat, h0, re.M) or [0, d])[1])
        A = dict(attempt=att, ep=g(r'^\s*moe_expert_parallel_size \.+ (\d+)', 1), experts=g(r'^\s*num_experts \.+ \[?(\d+)', 0),
                 seq=g(r'^\s*seq_length \.+ (\d+)', 0), gbs=g(r'^\s*global_batch_size \.+ (\d+)', 0), labels=[], labels_done=False)
        lab = []
        for s in range(P):
            m = re.search(r'STAGE=%d LAYERS=(\d+) \[(\d+), (\d+)\)' % s, _head(_ranklog(job, att, s * per_stage), 4 * 1024 * 1024))
            lab.append([f'pipe items {m.group(2)}–{int(m.group(3)) - 1}', f'{m.group(1)} layers'] if m else ['', ''])
        A['labels'] = lab; A['labels_done'] = (all(x[0] for x in lab) or P == 1) and A['seq'] > 0
        C['attempt_facts'] = A
    ep, ne = max(1, A['ep']), A['experts']
    G = max(per_rep, ep, 8)                                # the column group: a replica, or an EP group when a replica is smaller
    groups, npg = max(1, per_stage // G), max(1, G // 8)
    glabel = 'replica' if G == per_rep else 'EP group'
    # ---- the node files: the ECC count at job start once (cache), the last 30 beats per rank every call
    ce0 = C.setdefault('ce0', {}); ceg = C.setdefault('ce_first_growth', {})
    hist = {}
    for f in glob.glob(os.path.join(job, 'data', 'hb', '*.tsv')):
        node = os.path.basename(f)[:-4]
        if node not in C.setdefault('ce0_nodes', []):
            for line in _head(f, 8192).split('\n')[:-1]:
                v = line.split('\t')
                if len(v) >= 19 and v[0] != 'time' and not v[0].startswith('#') and v[2].isdigit() and v[10].lstrip('-').isdigit() and int(v[10]) >= 0:
                    ce0.setdefault(v[2], int(v[10]))
            C['ce0_nodes'].append(node)
        for line in D.tail_lines(f, 8 * 30 * 260):
            v = line.split('\t')
            if len(v) < 19 or v[0] == 'time':
                continue
            try:
                r, e = int(v[2]), float(v[18])
            except ValueError:
                continue
            hist.setdefault(r, []).append((e, v))
    last = {}
    AP = C.setdefault('alloc_peak', dict(att=None, r={}))       # [2026-10-13] the most PyTorch allocated per rank over this attempt's beats
    if AP.get('att') != att:
        AP['att'] = att; AP['r'] = {}
    for r, h in hist.items():
        h.sort(); v = h[-1][1]
        hv = [_num(x[1][8]) for x in h if _num(x[1][8]) is not None]
        if hv:
            AP['r'][str(r)] = max(AP['r'].get(str(r), 0), max(hv))
        last[r] = dict(node=v[1], step=v[3], phase=v[4], coll=int(v[5]) if v[5].isdigit() else None, st=_num(v[6]), busy=_num(v[7]), hbm=_num(v[8]),
                       host=v[9], ce=v[10], ue=v[11], xgmi=v[12], flag=v[17], epoch=h[-1][0], strip=''.join(PH(x[1][4]) for x in h[-30:]))
        c0 = ce0.get(str(r))
        if c0 is not None and v[10].isdigit() and int(v[10]) > c0 and str(r) not in ceg:
            ceg[str(r)] = next((x[1][0] for x in h if x[1][10].isdigit() and int(x[1][10]) > c0), v[0])
    att_t = (T.get('attempts') or [[None]])[-1][0] if T.get('attempts') else None
    MV = vram_samples(job, C, _epoch(att_t) if att_t else None)
    vlast = {r: x[1] for r, x in MV['last'].items()}; vpeak = MV['peak']
    gate = {}
    for line in D.read(os.path.join(job, 'data', 'gate_report.tsv')).split('\n')[1:]:
        v = line.split('\t')
        if len(v) >= 11:
            gate.setdefault(v[0], {})[v[1]] = dict(value=v[2], base=v[3], thr=v[4], ok=v[5], detail=v[10][:220])
    rec = {}
    for line in D.read(os.path.join(HERE, '..', 'failure_record.tsv')).split('\n')[1:]:
        v = line.split('\t')
        if len(v) >= 9:
            rec.setdefault(v[4], []).append([v[1][:16], v[2], v[7]])
    stage_coll, stage_busy, stage_hbm = {}, {}, {}
    for r, x in last.items():
        s = r // per_stage
        if x['coll'] is not None: stage_coll.setdefault(s, []).append(x['coll'])
        if x['busy'] is not None and x['st']: stage_busy.setdefault(s, []).append(x['busy'] / x['st'])
        if x['hbm'] is not None: stage_hbm.setdefault(s, []).append(x['hbm'])
    nodes, epr = [], max(1, ne // ep) if ne else 1
    for s in range(P):
        med = statistics.median(stage_coll.get(s, [0]))
        for grp in range(groups):
            for n in range(npg):
                r0 = s * per_stage + grp * G + n * 8
                rs = [last.get(r) for r in range(r0, r0 + 8)]; ok = [x for x in rs if x]
                if not ok:
                    continue
                node = ok[0]['node']; g = gate.get(node, {})
                gcd = []
                for i, x in enumerate(rs):
                    r = r0 + i; m = (r % per_stage) % per_rep; e = (r % per_stage) % ep
                    gcd.append([r, m % TP, m // TP, e, f'{e * epr}-{(e + 1) * epr - 1}' if ne else '-', x['phase'] if x else '?', x['coll'] if x else None,
                                x['hbm'] if x else None, round(x['busy'] / x['st'] * 100, 1) if x and x['busy'] is not None and x['st'] else None,
                                x['strip'] if x else '', int(x['ce']) if x and x['ce'].isdigit() else 0, ce0.get(str(r)),
                                vlast.get(str(r)), vpeak.get(str(r))])
                colls = [x['coll'] for x in ok if x['coll'] is not None]
                gm = re.findall(r'gcd(\d)=([\d.]+)', g.get('gemm', {}).get('detail', ''))
                hosts = [float(x['host']) for x in ok if _num(x['host']) is not None]
                vn = [vlast.get(str(r)) for r in range(r0, r0 + 8) if vlast.get(str(r)) is not None]
                vp = [vpeak.get(str(r)) for r in range(r0, r0 + 8) if vpeak.get(str(r)) is not None]
                nodes.append(dict(s=s, rep=grp, slot=n, node=node.replace('frontier', ''), hbm=max((x['hbm'] or 0) for x in ok),
                                  vram=max(vn) if vn else None, vram_peak=max(vp) if vp else None,
                                  lag=round(med - min(colls)) if colls else 0, age=round(now - max(x['epoch'] for x in ok)), host=max(hosts) if hosts else 0,
                                  ce=sum(int(x['ce']) for x in ok if x['ce'].isdigit()),
                                  ce_grow=sum(max(0, c[10] - c[11]) for c in gcd if c[11] is not None),
                                  ue=sum(int(x['ue']) for x in ok if x['ue'].isdigit()), xgmi=sum(int(x['xgmi']) for x in ok if x['xgmi'].isdigit()),
                                  flag=sorted({x['flag'] for x in ok}), gemm=g.get('gemm', {}).get('value'), gemm_thr=g.get('gemm', {}).get('thr'),
                                  gemm_gcd=[float(v) for _, v in gm], xgmi_a2a=g.get('xgmi_a2a', {}).get('value'), pair=g.get('pair', {}).get('value'),
                                  record=rec.get(node, []), gcd=gcd))
    ecc = []
    for r, x in sorted(last.items()):
        if x['ce'].isdigit() and int(x['ce']) > 0:
            c0 = ce0.get(str(r), int(x['ce']))
            ecc.append(dict(node=x['node'].replace('frontier', ''), gcd=r % 8, rank=r, stage=r // per_stage, start=c0, now=int(x['ce']),
                            grow=int(x['ce']) - c0, first_growth=ceg.get(str(r), '').replace('T', ' ')))
    dist = {}
    for t in ('gemm', 'xgmi_a2a', 'pair'):
        vals = [float(v[t]['value']) for v in gate.values() if t in v and re.match(r'^[\d.]+$', v[t]['value'] or '')]
        dist[t] = dict(values=vals, thr=next((v[t]['thr'] for v in gate.values() if t in v), ''), base=next((v[t]['base'] for v in gate.values() if t in v), ''))
    gsum = [[c.strip() for c in l.split('|')] for l in D.read(os.path.join(job, 'gate.log')).split('\n') if re.match(r'^[a-z_0-9]+ +\|', l) and not l.startswith('test ')]
    gate_tests = _gate_details(D.read(os.path.join(job, 'gate.log')), [g[0] for g in gsum])   # [2026-10-13] the page explains each test
    mem = D.read(os.path.join(job, 'data', 'mem_gate.txt'))
    pred = {int(m.group(1)): dict(alloc=float(m.group(2)), vram=float(m.group(3)), line=float(m.group(4)))
            for m in re.finditer(r'stage (\d+) \([^)]*\): allocated ([\d.]+), reserved [\d.]+, VRAM ([\d.]+), out-of-memory line ([\d.]+)', mem)}
    # ---- heartbeat.log: only the new beats; the ECC growth events and the save-window count accumulate in the cache
    hb = C.setdefault('hb', dict(beats=0, sw=0, ce_events=[], last=''))
    txt = _new_text(os.path.join(job, 'heartbeat.log'), C, 'hb_offset')
    for b in re.findall(r'^beat at .*?(?=^beat at |\Z)', txt, re.M | re.S):
        hb['beats'] += 1; hb['last'] = b.strip()
        if re.search(r'\d+ of \d+ groups are in a save window', b):
            hb['sw'] += 1
        m = re.search(r'^beat at (\S+)', b); c = re.search(r'ECC correctable \+(\d+) on ([\w,]+)', b)
        if m and c:
            hb['ce_events'].append([m.group(1), int(c.group(1)), c.group(2).replace('frontier', '')])
    hb['ce_events'] = hb['ce_events'][-2000:]
    saves = [l.split('\t') for l in D.read(os.path.join(job, 'data', 'save_times.tsv')).split('\n') if l and l[0].isdigit()]
    asy = [l.split('\t') for l in D.read(os.path.join(job, 'data', 'save_async_times.tsv')).split('\n') if l.startswith('global_step')]
    loop = D.read(os.path.join(job, 'data', 'loop.log'))
    ucp = []
    for line in loop.split('\n'):
        a = re.match(r'^(\S+ \S+) ucp: the conversion of the save of step (\d+) to the universal form starts', line)
        b = re.match(r'^(\S+ \S+) ucp: the universal of step (\d+) is complete.*?the conversion took \d+ min \((\d+) s\)', line)
        if a: ucp.append(dict(time=a.group(1), what='start', step=int(a.group(2))))
        if b: ucp.append(dict(time=b.group(1), what='done', step=int(b.group(2)), seconds=int(b.group(3))))
    steps = []
    for line in D.read(os.path.join(job, 'training.log')).split('\n'):
        v = [x.strip() for x in line.split('|')]
        if len(v) >= 7 and v[2].isdigit() and _num(v[3]) is not None:   # [2026-10-11] a step line without a loss ('-') is skipped, not fatal
            steps.append(dict(time=v[0], step=int(v[2]), loss=float(v[3]), tflops=_num(v[4]) or 0, step_s=_num(v[5]) or 0, hbm=_num(v[6]) or 0, attempt=int(v[1]) if v[1].isdigit() else None))
    # ---- the gradient norm per step: the new lines of the last rank's log only
    gn = C.setdefault('gnorm', {}); lr = _ranklog(job, att, W - 1)
    if lr:
        if C.get('gnorm_file') != lr:
            C['gnorm_file'] = lr; C.pop('gn_offset', None)
        for line in _new_text(lr, C, 'gn_offset', 4 * 1024 * 1024).split('\n'):
            m = re.search(r'iteration +(\d+)/', line); g2 = re.search(r'grad norm: ([\d.E+-]+)', line)
            if 'lm loss' in line and m and g2:
                gn[m.group(1)] = float(g2.group(1))
    latest, latest_u = D.ckpt_tags(L['save'])
    sv = lambda D_, s: [v for r, v in D_.items() if r.isdigit() and int(r) // per_stage == s]
    stage = [dict(busy=round(statistics.median(stage_busy.get(s, [0])) * 100, 1), hbm=round(max(stage_hbm.get(s, [0])), 1),
                  alloc=round(max(sv(AP['r'], s), default=0) / GIB, 1), vram=max(sv(vpeak, s), default=None), vram_now=max(sv(vlast, s), default=None),
                  pred=pred.get(s)) for s in range(P)]
    # the VRAM of each step: the largest GCD of the newest monitor sweep at or before the step's end (within 10 minutes)
    tsk = sorted((int(k), v) for k, v in MV['ts'].items())
    import bisect
    tse = [k for k, _ in tsk]
    for st_ in steps:
        e = _epoch(st_['time']); i = bisect.bisect_right(tse, e) - 1 if e else -1
        st_['vram'] = tsk[i][1] if i >= 0 and e - tse[i] <= 600 else None
        st_['alloc'] = round(st_['hbm'] / GIB, 2) if st_.get('hbm') else None
    jid = (re.search(r'job_(\d+)', os.path.basename(job.rstrip('/'))) or [None, '?'])[1]
    start = (re.match(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)', loop) or [None, None])[1]
    taken = time.strftime('%Y-%m-%d %H:%M:%S')
    end = T['end'] or time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now + 3600))
    T2 = dict(T); T2.update(start=start, end=end, gate=T['gate'] or ['0', '0'], attempts=T['attempts'] or [[start or taken, '0']])
    L2 = dict(L); L2.update(ep=ep, experts=ne, experts_per_rank=epr, seq=A['seq'], gbs=A['gbs'], groups=groups, nodes_per_group=npg, group_label=glabel)
    run = dict(job=jid, label=os.path.basename((L['save'] or '').rstrip('/')) or 'this run', taken=taken, layout=L2, spares=[x.replace('frontier', '') for x in S['spares']],
               duty=(S['duty'] or '').replace('frontier', ''), degraded=[[n.replace('frontier', ''), w] for n, w in S['degraded']],
               ucp_workers=[x.replace('frontier', '') for x in D.ucp_workers(job)], ucp=ucp, timeline=T2, latest=latest, latest_universal=latest_u,
               alarms=[l for l in D.read(os.path.join(job, 'alarms.log')).split('\n') if ' ALARM ' in l][-1000:], nodes=nodes,
               alarm_log=[l for l in D.read(os.path.join(job, 'alarms.log')).split('\n') if l.strip() and not l.startswith('#')][-1000:],
               vram=dict(dev=DEV_GIB, now=max(vlast.values(), default=None), peak=max(vpeak.values(), default=None),
                         at=time.strftime('%H:%M:%S', time.localtime(max((x[0] for x in MV['last'].values()), default=0))) if MV['last'] else None,
                         sweeps=len(MV['ts'])),
               steps=steps or [dict(time=taken, step=0, loss=0, tflops=0, step_s=0, hbm=0)], stage=stage, dist=dist, gsum=gsum, gate_tests=gate_tests,
               lastbeat=hb['last'], ce_events=hb['ce_events'], save_window_beats=hb['sw'], beats=hb['beats'], saves=saves, async_saves=asy,
               memgate=(re.search(r'the colour: ([^\n]*)', mem) or [None, 'not run for this job'])[1], ecc=ecc, ckpts=ckpts(L['save'], evals),
               phases=_try(lambda: phases_of(job, C, steps, loop), []), batch_conv=_try(lambda: batch_conversions(loop), []),
               done='the resilient loop finished' in loop, end_known=bool(T['end']), loop_last=_try(lambda: re.findall(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ', loop, re.M)[-1], None))
    meta = [dict(k='this', name=f'this job ({jid})')]
    series = dict(this=[dict(step=s['step'], loss=s['loss'], tflops=s['tflops'] or None, step_s=s['step_s'], gnorm=gn.get(str(s['step'])),
                             vram=s.get('vram'), alloc=s.get('alloc'), job=jid) for s in steps])
    if history:
        try:
            h = json.load(open(history))
            meta = h.get('meta', []) + meta; series.update(h.get('series', {}))
        except (OSError, ValueError):
            pass
    try:
        tmp = cpath + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(C, fh)
        os.replace(tmp, cpath)
    except OSError:
        pass
    return dict(runs={'this': run}, series=series, series_meta=meta, stage_labels=A['labels'], title=title or f'Run console · job {jid}', live=True,
                eval_note='The evaluation scores come from the file named by --evals or AUTORUN_DASHBOARD_EVALS.' if evals else '')


def ckpts(save, evals_path):
    """one row per step in the checkpoint folder: the ZeRO save (trainable), the universal (complete or converting), the scores"""
    rows = {}
    for d in glob.glob(os.path.join(save or '/nonexistent', 'global_step*')):
        m = re.match(r'^global_step(\d+)(_universal)?$', os.path.basename(d))
        if not m:
            continue
        x = rows.setdefault(int(m.group(1)), dict(step=int(m.group(1)), zero=False, universal='-'))
        if m.group(2):
            x['universal'] = 'complete' if glob.glob(d + '/mp_rank_*_model_states.pt') and not glob.glob(d + '/tmp_*') else 'converting'
        else:
            x['zero'] = True
    if evals_path:
        try:
            run = os.path.basename((save or '').rstrip('/'))
            for e in json.load(open(evals_path)):
                if e.get('run') != run:
                    continue
                x = rows.setdefault(e['step'], dict(step=e['step'], zero=False, universal='-'))
                x['ruler'] = {k: v['score'] for k, v in e.get('tasks', {}).items()}
                x['eval'] = e.get('state') or ('done' if len(x['ruler']) >= 4 else f'{len(x["ruler"])} of 4 sub-tasks')
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return [rows[k] for k in sorted(rows)]


_PLAIN = {'collective_hang': 'a collective hung (the watchdog aborted it)', 'gpu_oom': 'GPU out of memory', 'filesystem_hang': 'a checkpoint save missed its deadline',
          'loss_nan_spike': 'NaN or spiking loss', 'driver_hip_error': 'GPU driver/runtime (HIP) error', 'framework_exception': 'another exception (software bug)',
          'init_hang': 'start-up hang (no first step in time)', 'host_oom': 'host memory above 97 %', 'node_lost': 'a node was lost',
          'rank_died': 'a rank was killed or ended with an unknown code'}


def failures_of(job):
    """[2026-10-11, the user: the page showed failures only as step-time spikes] every failure of the job from its own records (read-only):
    data/failures.tsv (what, where, step, injected, times), data/attempts.tsv (the nodes of each attempt), recovery.log (the decision lines
    under each '===== failure k' block and the final stop line). Returns a list of dicts, oldest first."""
    import time as _t
    def rows(path):
        try:
            L = [l.rstrip('\n').split('\t') for l in open(path, errors='replace') if l.strip() and not l.startswith('#')]
        except OSError:
            return []
        if not L:
            return []
        h = L[0]
        return [dict(zip(h, r)) for r in L[1:]]
    F = rows(os.path.join(job, 'data', 'failures.tsv')); A = {r.get('attempt'): r for r in rows(os.path.join(job, 'data', 'attempts.tsv'))}
    blocks, cur = [], None
    try:
        for l in open(os.path.join(job, 'recovery.log'), errors='replace'):
            if l.startswith('===== failure '):
                cur = []; blocks.append(cur)
            elif cur is not None and re.match(r'\s+(decision|no spare|checkpoint|node):', l):
                cur.append(l.strip())
            elif 'THE RUN STOPS' in l and cur is not None:
                cur.append(l.strip('= \n'))
    except OSError:
        pass
    hm = lambda x: _t.strftime('%H:%M:%S', _t.localtime(float(x))) if x and str(x).replace('.', '', 1).isdigit() else '-'
    out = []
    for i, f in enumerate(F):
        nxt = A.get(str(int(f.get('attempt', '0') or 0) + 1)) if str(f.get('attempt', '')).isdigit() else None
        dec = blocks[i] if i < len(blocks) else []
        rec = f.get('first_iter_after', '')
        rec_s = int(float(rec) - float(f['exit_time'])) if rec and rec.replace('.', '', 1).isdigit() and f.get('exit_time', '').replace('.', '', 1).isdigit() else None
        out.append(dict(n=i + 1, attempt=f.get('attempt'), time=hm(f.get('exit_time')), cls=f.get('class', ''), what=_PLAIN.get(f.get('class', ''), f.get('class', '')),
                        code=f.get('code', ''), node=f.get('node', '-'), rank=f.get('rank', '-'), step=f.get('step', '-'), injected=f.get('injected', '') == 'yes',
                        case=f.get('case', ''), excluded=f.get('excluded', ''), recovered_s=rec_s, evidence=f.get('evidence', ''),
                        resumed_at=hm(rec) if rec_s is not None else '-', decisions=dec,
                        next_nodes=len(nxt['nodes'].split(',')) if nxt and nxt.get('nodes') else None,
                        decision=' | '.join(re.sub(r'^(decision|no spare|checkpoint|node):\s*', lambda m: m.group(1) + ': ', d) for d in dec)))
    return out


def failures_table(fails):
    """[2026-10-11, the user: "too dense ... a simplified version ... when click into it, it shows a nicely formatted and clean layout"]
    one compact line per failure; clicking a line opens its card (HTML <details>: no script, so it also opens in the VS Code preview)"""
    import html as _h
    E = _h.escape
    css = ('<style>#failures{background:#fff;border:1px solid #d5dde5;border-radius:6px;padding:8px 12px;margin:10px 0}'
           '#failures h2{font-size:15px;margin:0 0 6px}#failures .fl-h,#failures summary{display:grid;grid-template-columns:2.2em 5.6em minmax(12em,1.6fr) minmax(10em,1fr) 3.4em minmax(13em,1.2fr);'
           'gap:8px;align-items:center;font-size:13px}#failures .fl-h{color:#5a6b7b;font-size:11px;text-transform:uppercase;letter-spacing:.04em;padding:0 8px 4px}'
           '#failures details{border-top:1px solid #e7edf2}#failures summary{cursor:pointer;list-style:none;padding:6px 8px}#failures summary::-webkit-details-marker{display:none}'
           '#failures summary:hover{background:#f4f7f9}#failures details[open] summary{background:#eef5f7}'
           '#failures .tag{display:inline-block;border-radius:10px;padding:1px 8px;font-size:12px}#failures .ok{background:#e3f4e8;color:#1e6b34}#failures .stop{background:#fbe3e1;color:#9b2c22}'
           '#failures .inj{color:#8a63d2;font-size:11px}#failures .card{margin:4px 8px 10px 2.6em;padding:8px 12px;border-left:3px solid #0b7285;background:#fafcfd}'
           '#failures .card dl{display:grid;grid-template-columns:11em 1fr;gap:4px 12px;margin:0;font-size:13px}#failures .card dt{color:#5a6b7b}#failures .card dd{margin:0}'
           '#failures .card ol{margin:0;padding-left:1.2em}#failures .card li{margin:2px 0}#failures .hint{color:#5a6b7b;font-size:11px;font-weight:normal}</style>')
    if not fails:
        return css + '<section id="failures"><h2>Failures and recoveries</h2><div style="color:#5a6b7b;font-size:13px">No failure so far.</div></section>'
    def outcome(f):
        if f['recovered_s'] is not None:
            return f'<span class="tag ok">resumed after {f["recovered_s"]} s' + (f' on {f["next_nodes"]} nodes' if f['next_nodes'] else '') + '</span>'
        return '<span class="tag stop">' + ('run stopped' if any('NO RESTART' in d or 'THE RUN STOPS' in d for d in f.get('decisions') or []) else 'not recovered') + '</span>'
    def clean(d):
        d = re.sub(r'^(decision|checkpoint|node|no spare):\s*', '', d)
        return d[0].upper() + d[1:] if d else d
    rows = []
    for f in fails:
        dec = [x for x in (f.get('decisions') or []) if not x.startswith('node:')]
        card = ('<div class="card"><dl>'
                f'<dt>Ended</dt><dd>attempt {E(str(f["attempt"]))} ended at {E(f["time"])}</dd>'
                f'<dt>What failed</dt><dd>{E(f["what"])} &nbsp;<span class="hint">class {E(f["cls"])}, exit code {E(str(f["code"]))}</span>'
                + (f' &nbsp;<span class="inj">injected test case: {E(f["case"])}</span>' if f['injected'] else '') + '</dd>'
                f'<dt>Where</dt><dd>node {E(f["node"])}, rank {E(str(f["rank"]))}, step {E(str(f["step"]))}</dd>'
                f'<dt>How it was seen</dt><dd>{E((f.get("evidence") or "-")[:400])}</dd>'
                f'<dt>The node</dt><dd>{E(f["excluded"] or "-")}</dd>'
                '<dt>Decisions</dt><dd><ol>' + ''.join(f'<li>{E(clean(d))}</li>' for d in dec) + '</ol></dd>'
                f'<dt>Training</dt><dd>' + (f'resumed at {E(f["resumed_at"])}, {f["recovered_s"]} s after the failure' + (f', on {f["next_nodes"]} nodes' if f['next_nodes'] else '')
                                            if f['recovered_s'] is not None else 'did not resume in this job') + '</dd>'
                '</dl></div>')
        rows.append(f'<details><summary><span>#{f["n"]}</span><span>{E(f["time"])}</span><span><b>{E(f["what"])}</b>' + (' <span class="inj">(injected)</span>' if f['injected'] else '')
                    + f'</span><span>{E(f["node"])} · rank {E(str(f["rank"]))}</span><span>{E(str(f["step"]))}</span><span>{outcome(f)}</span></summary>{card}</details>')
    return (css + f'<section id="failures"><h2>Failures and recoveries ({len(fails)}) <span class="hint">click a line for the details</span></h2>'
            '<div class="fl-h"><span>#</span><span>ended at</span><span>what failed</span><span>where</span><span>step</span><span>outcome</span></div>' + ''.join(rows) + '</section>')


def _svg_chart(pts, key, label, color='#0b7285', w=640, h=170, marks=()):
    """[2026-10-11] a script-free line chart (inline SVG) of one metric against the step, for viewers that run no JavaScript"""
    xs = [(p['step'], p[key]) for p in pts if isinstance(p.get(key), (int, float))]
    if len(xs) < 2:
        return f'<div class="sv-empty">{label}: not enough steps yet</div>'
    L, R, T, B = 52, 12, 22, 26
    x0, x1 = min(x for x, _ in xs), max(x for x, _ in xs)
    ys = sorted(y for _, y in xs); lo, hi = ys[0], ys[-1]
    if key in ('tflops', 'step_s', 'gnorm') and len(ys) > 8:         # the first steps (JIT, warm-up) would flatten the rest
        hi = ys[int(0.95 * (len(ys) - 1))] * 1.15 if key != 'tflops' else hi
    if hi <= lo:
        hi = lo + 1.0
    X = lambda v: L + (v - x0) / max(1, x1 - x0) * (w - L - R)
    Y = lambda v: T + (1 - (min(v, hi) - lo) / (hi - lo)) * (h - T - B)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{label}">',
           f'<rect x="0" y="0" width="{w}" height="{h}" fill="#fff"/>',
           f'<text x="{L}" y="14" font-size="12" font-family="sans-serif" fill="#122130">{label}</text>']
    for i in range(5):
        v = lo + (hi - lo) * i / 4; y = Y(v)
        out.append(f'<line x1="{L}" x2="{w - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e3e8ee"/>'
                   f'<text x="{L - 6}" y="{y + 4:.1f}" font-size="10" text-anchor="end" font-family="sans-serif" fill="#5a6b7b">{v:.3g}</text>')
    for i in range(6):
        v = x0 + (x1 - x0) * i / 5; x = X(v)
        out.append(f'<text x="{x:.1f}" y="{h - 8}" font-size="10" text-anchor="middle" font-family="sans-serif" fill="#5a6b7b">{v:.0f}</text>')
    for st_, lab_ in marks:                                   # [2026-10-11] a dashed red line at each failure's step
        if x0 <= st_ <= x1:
            out.append(f'<line x1="{X(st_):.1f}" x2="{X(st_):.1f}" y1="{T}" y2="{h - B}" stroke="#c0392b" stroke-dasharray="3 3"/>'
                       + (f'<text x="{X(st_) - 3:.1f}" y="{T + 10}" font-size="9" text-anchor="end" font-family="sans-serif" fill="#c0392b">{lab_}</text>'
                          if X(st_) > w - R - 24 else f'<text x="{X(st_) + 3:.1f}" y="{T + 10}" font-size="9" font-family="sans-serif" fill="#c0392b">{lab_}</text>'))
    out.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="' + ' '.join(f'{X(x):.1f},{Y(y):.1f}' for x, y in xs) + '"/>')
    out.append('</svg>')
    return ''.join(out)


# ---------------- the job timeline with every failure and recovery ----------------
# [2026-10-11, the user: "the xxx-hour timeline, include the failure and recovery process (e.g. how a new training progress needs to
# restart or cpu load checkpoint again)"] The timeline is drawn here, in Python, as inline SVG, so the browser console and the script-free
# VS Code view show the same picture. Read-only, like the rest of this file: data/attempts.tsv and data/failures.tsv (one row per ended
# attempt and per failure, written by restart_loop.sh), data/loop.log and training.log (already read by build()), and the head of each
# attempt's rank-0 log, read until its checkpoint-load lines appear and then kept in the cache (at 264 nodes they sit in the first 0.3 MB).
_LD = dict(load0=re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[,.]\d+\] \[INFO\] [^\n]*?Loading checkpoint from', re.M),
           load1=re.compile(r'^\[after model, optimizer, and learning rate scheduler are built\] datetime: (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)', re.M),
           data1=re.compile(r'^\[after dataloaders are built\] datetime: (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)', re.M))


def _try(fn, default):
    """the new timeline facts must never stop the page: on any error, the default"""
    try:
        return fn()
    except Exception:
        return default


def _tsv(path):
    try:
        L = [l.rstrip('\n').split('\t') for l in open(path, errors='replace') if l.strip() and not l.startswith('#')]
    except OSError:
        return []
    return [dict(zip(L[0], r)) for r in L[1:]] if L else []


def _ep(s):
    """'YYYY-mm-dd HH:MM:SS' (or with a T), or epoch seconds as text -> epoch seconds; None when absent"""
    s = str(s or '').strip()
    if re.match(r'^\d+(\.\d+)?$', s):
        return float(s)
    try:
        return time.mktime(time.strptime(s[:19].replace('T', ' '), '%Y-%m-%d %H:%M:%S'))
    except ValueError:
        return None


def phases_of(job, C, steps, loop):
    """one dict per attempt: srun start, checkpoint load (start, end, step, kind), data loaders, first step, its steps, end, failure"""
    rows = {int(a['attempt']): a for a in _tsv(os.path.join(job, 'data', 'attempts.tsv')) if str(a.get('attempt', '')).isdigit()}
    F = {str(f.get('attempt')): f for f in _tsv(os.path.join(job, 'data', 'failures.tsv'))}
    started = {int(k): t for t, k in re.findall(r'^(\S+ \S+) attempt (\d+) started', loop, re.M)}
    firstl = {int(k): t for t, k in re.findall(r'^(\S+ \S+) attempt (\d+) showed its first iteration line', loop, re.M)}
    LD = C.setdefault('att_load', {})
    out = []
    for k in sorted(set(rows) | set(started)):
        a = rows.get(k, {})
        st, fi, en = _ep(a.get('start')) or _ep(started.get(k)), _ep(a.get('first_iter')) or _ep(firstl.get(k)), _ep(a.get('end'))
        ld = LD.get(str(k))
        if not ld or not ld.get('done'):                   # the running attempt's start-up: read again until the loader's lines appear
            h = _head(_ranklog(job, k, 0), 4 * 1024 * 1024)
            g = {n: (rx.search(h) or [None, None])[1] for n, rx in _LD.items()}
            m = re.search(r'successfully loaded checkpoint from \S+ at iteration (\d+)', h)
            ld = dict(load0=g['load0'], load1=g['load1'], data1=g['data1'], step=int(m.group(1)) if m else None,
                      univ=bool(re.search(r'loaded universal zero checkpoints|load_universal_checkpoint \.+ True', h)), done=bool(g['data1'] or fi or en))
            LD[str(k)] = ld
        mine = [s for s in steps if s.get('attempt') == k]
        rf = a.get('resumed_from', '')
        out.append(dict(k=k, start=st, first=fi, end=en, nodes=len([x for x in (a.get('nodes') or '').split(',') if x]) or None,
                        resumed_from=int(rf) if str(rf).isdigit() else (ld['step'] if ld.get('step') is not None else None),
                        load0=_ep(ld.get('load0')), load1=_ep(ld.get('load1')), data1=_ep(ld.get('data1')), ckpt_step=ld.get('step'), univ=ld.get('univ'),
                        steps=[(_ep(s['time']), s['step'], s['step_s']) for s in mine], fail=F.get(str(k))))
    return out


def batch_conversions(loop):
    """the OOM or shrink rule's conversions on the batch node (a job without UCP workers): [step, start, end]"""
    out = []
    for t, n in re.findall(r'^(\S+ \S+) no UCP worker node is in this job, so the OOM rule converts the save of step (\d+)', loop, re.M):
        w = re.search(r'the universal checkpoint of step %s loads \(the rule waited (\d+) s' % n, loop)
        t0 = _ep(t)
        out.append([int(n), t0, t0 + int(w.group(1)) if w else None])
    return out


def _hms(t):
    return time.strftime('%H:%M:%S', time.localtime(t)) if t else '-'


def _dur(s):
    s = int(round(max(0, s or 0)))
    return f'{s} s' if s < 120 else (f'{s // 60} min {s % 60:02d} s' if s < 3600 else f'{s // 3600} h {s % 3600 // 60:02d} min')


_TLC = dict(gate='var(--ph-init,#8d99a5)', launch='var(--ph-other,#c3ccd5)', load='var(--ph-bwd,#8a63d2)', first='var(--train,#3a7cc4)',
            train='var(--train,#3a7cc4)', stop='var(--crit,#cf2d5a)', recover='#eb9db3', save='var(--ph-save,#d99a00)',
            conv='var(--ph-col,#139a91)', convrun='var(--worker,#d99a00)', ecc='var(--warn,#d98f0a)', now='var(--accent,#0b7285)')


def _tl_span(r):
    """(start, end, now, over) of the time axis: the job's wall while it runs; once it is over (the loop finished, the end time has
    passed, or an old job without the end-time line), its own first and last events, so a short or old job fills the width"""
    T = r.get('timeline') or {}; P = r.get('phases') or []
    taken = _ep(r.get('taken')) or time.time()
    jend = _ep(T.get('end')) if r.get('end_known', True) else None    # build() puts 'now + 1 h' in place of an unknown end
    t0 = _ep(T.get('start')) or (P[0]['start'] if P and P[0]['start'] else taken - 3600)
    ev = [x for p in P for x in [p['start'], p['first'], p['end']] + [s[0] for s in p['steps'][-1:]] if x] + [x for x in [_ep(r.get('loop_last'))] if x]
    last = max(ev) if ev else None
    over = bool(r.get('done')) or bool(jend and taken > jend) or bool(not jend and last and ((P and P[-1].get('end')) or taken - last > 1800))
    if over and last:
        return t0, last + max(30, 0.02 * (last - t0)), last, True
    return t0, jend or taken + 3600, taken, False


def _short_decisions(f, extra=()):
    """the recovery decisions of one failure as a few short clauses (the full text stays in the failure's card)"""
    items = [f['excluded']] if f.get('excluded') else []
    for d in f.get('decisions') or []:
        if d.startswith('node:'):
            continue
        for c in re.split(r';\s+', re.sub(r'^(decision|checkpoint|no spare):\s*', '', d)):
            c = c.strip().rstrip('.')
            if not c or re.search(r'retried \d|the decision took|heaviest stage|tokens per step|stay idle|green by|predicted at', c):
                continue
            if len(c) > 140 and ': ' in c:
                c = c.split(': ')[0]
            c = c if len(c) <= 160 else c[:157] + '…'
            if c not in items:
                items.append(c)
    return (items + list(extra))[:7]


def recoveries(r, fails):
    """one dict per failure: the phases from the last step before it to the first step after the restart, in seconds"""
    P = r.get('phases') or []
    byatt = {str(f.get('attempt')): f for f in fails}
    _, _, now, over = _tl_span(r)
    live = None if over else now                          # a running job: the phase in progress is drawn up to now
    out = []
    for i, p in enumerate(P):
        f = byatt.get(str(p['k']))
        if not f or not p.get('end'):
            continue
        nx = P[i + 1] if i + 1 < len(P) else None
        last = p['steps'][-1] if p['steps'] else None
        t_last = last[0] if last else p['end']
        ph = [('no progress until srun returned', t_last, p['end'],
               f"from the last logged step ({last[1]}) to the end of the attempt: the failure, its detection and the teardown of all ranks")] if last else \
             [('failed during start-up', p['start'], p['end'], f"attempt {p['k']} ended before its first step")]
        redo = None
        if not nx and live and live > p['end']:
            ph.append(('recovery decisions', p['end'], live, 'in progress: ' + (f.get('excluded') or 'the restart loop is deciding')))
        if nx and nx.get('start'):
            conv = [c for c in r.get('batch_conv') or [] if c[1] and p['end'] - 5 <= c[1] <= nx['start']]
            ph.append(('recovery decisions', p['end'], nx['start'], _short_decisions(f, [f'the save of step {c[0]} was converted to the universal form on the batch node'
                                                                                          + (f' ({_dur(c[2] - c[1])})' if c[2] else '') for c in conv])))
            l_end = nx['load0'] or nx['load1'] or nx['first'] or nx.get('end') or live
            if l_end:
                ph.append(('srun launch and model build', nx['start'], l_end, 'all ranks start, import, set up the communicators, build the model and the optimizer'))
            if nx['load0'] and (nx['load1'] or live):
                kind = 'universal checkpoint' if nx['univ'] else 'ZeRO save'
                ph.append((f"checkpoint load (step {nx['ckpt_step'] if nx['ckpt_step'] is not None else '?'}, {kind})", nx['load0'], nx['load1'] or live,
                           'every rank reads its shard from Lustre into host memory and copies it to its GPU'))
            if (nx['first'] or (live and nx['load1'])) and (nx['load1'] or l_end):
                ph.append(('data loaders and the first step', nx['load1'] or l_end, nx['first'] or live, 'the data loaders are rebuilt; the first step after the restart runs'))
            if last and nx.get('resumed_from') is not None and last[1] > nx['resumed_from']:
                again = [s for s in nx['steps'] if s[1] <= last[1]]
                redo = dict(first=nx['resumed_from'] + 1, last=last[1], n=last[1] - nx['resumed_from'],
                            t0=nx['first'], t1=again[-1][0] if again else None)
        out.append(dict(n=f['n'], f=f, phases=[x for x in ph if x[1] and x[2] and x[2] >= x[1]], t_last=t_last, end=p['end'],
                        next_failed=bool(nx and nx.get('end') and not nx.get('first')), next_k=nx['k'] if nx else None, live=bool(live),
                        resumed=nx['first'] if nx else None, resume_step=(nx['steps'][0][1] if nx and nx['steps'] else None), redo=redo,
                        restarted=bool(nx and nx.get('start'))))
    return out


def timeline_svg(r, fails, W=620, pid=''):
    """the whole job on one time axis: GPU training (start-up, checkpoint load, steps, each failure and its recovery), the CPU async saves,
    the UCP conversions, the corrected ECC counts; every bar carries a <title> with its times"""
    import html as _h
    E = lambda s: _h.escape(str(s), quote=True)
    T = r.get('timeline') or {}; P = r.get('phases') or []
    t0, t1, now, done = _tl_span(r)
    ml, mr = 104, 10
    X = lambda t: ml + (min(max(t, t0), t1) - t0) / max(1, t1 - t0) * (W - ml - mr)
    g = []
    def bar(a, b, y, h, kind, title, extra=''):
        if a is None or b is None or b < a:
            return
        g.append(f'<rect x="{X(a):.1f}" y="{y}" width="{max(1.0, X(b) - X(a)):.1f}" height="{h}" style="fill:{_TLC[kind]}" {extra}><title>{E(title)}</title></rect>')
    lanes = [(20, 'GPU training'), (48, 'CPU async save'), (68, 'UCP conversion'), (88, 'ECC corrected')]
    for y, lab in lanes:
        g.append(f'<text x="4" y="{y + 11}" style="fill:var(--ink,#122130)">{lab}</text><line x1="{ml}" x2="{W - mr}" y1="{y + 16}" y2="{y + 16}" style="stroke:var(--line,#e3e8ee)"/>')
    gate = (T.get('gate') or ['0'])[0]
    for i, p in enumerate(P):
        if i == 0 and p['start'] and t0 < p['start']:
            bar(t0, p['start'], 20, 16, 'gate', f"{_hms(t0)}-{_hms(p['start'])}: the pre-run tests ({gate} s) and the job's start-up")
        st = p['start']
        if not st:
            continue
        run_end = p['end'] or now
        l_end = p['load0'] or p['load1'] or p['first'] or run_end
        bar(st, l_end, 20, 16, 'launch', f"attempt {p['k']}, {_hms(st)}-{_hms(l_end)} ({_dur(l_end - st)}): srun launch and model build" + (f" on {p['nodes']} nodes" if p['nodes'] else ''))
        if p['load0']:
            le = p['load1'] or (None if p['end'] else now)
            bar(p['load0'], le, 20, 16, 'load', f"attempt {p['k']}, {_hms(p['load0'])}-{_hms(le)} ({_dur((le or p['load0']) - p['load0'])}): checkpoint load"
                + (f" of step {p['ckpt_step']}" if p['ckpt_step'] is not None else '') + (' (universal checkpoint)' if p['univ'] else ' (ZeRO save)')
                + ' from Lustre into host memory, then the GPUs')
        if p['first']:
            a = p['load1'] or l_end
            bar(a, p['first'], 20, 16, 'first', f"attempt {p['k']}, {_hms(a)}-{_hms(p['first'])}: data loaders and the first step", 'opacity=".45"')
        S = p['steps']
        if S:
            span_px = X(S[-1][0]) - X(S[0][0])
            if len(S) > 1 and span_px / max(1, len(S) - 1) < 3:   # steps under 3 px apart: one band per attempt
                bar(S[0][0], S[-1][0], 20, 16, 'train', f"attempt {p['k']}: steps {S[0][1]}-{S[-1][1]}, {_hms(S[0][0])}-{_hms(S[-1][0])}")
            else:
                for j in range(1, len(S)):
                    bar(S[j - 1][0], S[j][0] - 0.0001, 20, 16, 'train', f"step {S[j][1]}: {_dur(S[j][0] - S[j - 1][0])}")
        if not p['end'] and not done and S:                   # the steps expected until the end of the job
            ss = statistics.median([s[2] for s in S if s[2]] or [0]) or None
            if ss:
                g.append(f'<rect x="{X(S[-1][0]):.1f}" y="20" width="{max(0, X(t1) - X(S[-1][0])):.1f}" height="16" style="fill:{_TLC["train"]};fill-opacity:.1;'
                         f'stroke:{_TLC["train"]};stroke-dasharray:3 2"><title>about {int((t1 - S[-1][0]) // ss)} more steps fit before the end of the job</title></rect>')
    lab_x = -99
    for rc in recoveries(r, fails):
        for name, a, b, why in rc['phases']:
            kind = {'no progress until srun returned': 'stop', 'recovery decisions': 'recover'}.get(name)   # a start-up failure keeps its launch bar
            if kind:
                bar(a, b, 20, 16, kind, f"failure #{rc['n']}: {name}, {_hms(a)}-{_hms(b)} ({_dur(b - a)}): " + ('; '.join(why) if isinstance(why, list) else why))
        x = X(rc['t_last'])
        g.append(f'<path d="M{x - 4:.1f},6 L{x + 4:.1f},6 L{x:.1f},15 Z" style="fill:{_TLC["stop"]}"><title>failure #{rc["n"]}: {E(rc["f"].get("what", ""))} on '
                 f'{E(rc["f"].get("node", ""))}, rank {E(rc["f"].get("rank", ""))}</title></path>'
                 + (f'<text x="{x + 5:.1f}" y="13" style="fill:{_TLC["stop"]};font-weight:600">#{rc["n"]}</text>' if x - lab_x >= 22 else ''))
        lab_x = x if x - lab_x >= 22 else lab_x
        if rc['redo'] and rc['redo'].get('t1'):
            d = rc['redo']
            g.append(f'<rect x="{X(d["t0"]):.1f}" y="20" width="{max(1.0, X(d["t1"]) - X(d["t0"])):.1f}" height="16" style="fill:url(#tlHatch{pid})">'
                     f'<title>steps {d["first"]}-{d["last"]} computed again: they ran before failure #{rc["n"]} but were not in the save the restart loaded</title></rect>')
    adict = {x[0]: x for x in r.get('async_saves') or []}
    for sv in r.get('saves') or []:
        a = _ep(sv[2]); b = adict.get('global_step' + sv[0])
        e = _ep(b[2]) if b and len(b) > 2 else None
        bar(a, e or (a + 60 if a else None), 48, 12, 'save', f"save of step {sv[0]} at {_hms(a)}: the GPUs paused {sv[1]} s for the copy to pinned host memory; "
            f"CPU threads wrote it to Lustre" + (f" ({b[1]} s after the step)" if b else ''))
    opn = {}
    for u in r.get('ucp') or []:
        if u['what'] == 'start':
            opn[u['step']] = _ep(u['time'])
        elif u['step'] in opn:
            a = opn.pop(u['step']); b = _ep(u['time'])
            bar(a, b, 68, 12, 'conv', f"step {u['step']} converted to the universal form by the UCP workers in {_dur(b - a)}")
    for st, a in opn.items():
        bar(a, now, 68, 12, 'convrun', f"step {st}: " + (f"the conversion started at {_hms(a)} and did not finish before the job ended" if done else f"the UCP workers are converting it since {_hms(a)}"))
    for c in r.get('batch_conv') or []:
        bar(c[1], c[2] or (c[1] + 60), 68, 12, 'conv', f"step {c[0]} converted to the universal form on the batch node (no UCP worker in this job)"
            + (f" in {_dur(c[2] - c[1])}" if c[2] else ''))
    day = (r.get('taken') or '')[:10]
    for e in (r.get('ce_events') or [])[-400:]:
        tt = _ep(f'{day} {e[0]}')
        if tt:
            g.append(f'<circle cx="{X(tt):.1f}" cy="94" r="3" style="fill:{_TLC["ecc"]}"><title>{E(e[0])}: +{E(e[1])} corrected ECC errors on {E(e[2])}</title></circle>')
    xn = X(now)
    g.append(f'<line x1="{xn:.1f}" x2="{xn:.1f}" y1="4" y2="108" style="stroke:{_TLC["now"]};stroke-width:2"/><text x="{xn - 4 if xn > W - 90 else xn + 4:.1f}" y="118" '
             f'text-anchor="{"end" if xn > W - 90 else "start"}" style="fill:{_TLC["now"]}">{"ended" if done else "now"} {_hms(now)[:5]}</text>')
    g.append(f'<line x1="{ml}" x2="{W - mr}" y1="124" y2="124" style="stroke:var(--muted,#5a6b7b)"/>')
    span = (t1 - t0) / 60
    iv = next((m for m in (5, 10, 15, 30, 60, 120, 180, 240, 360) if span / m <= 6), max(1, int(span / 6) + 1))
    m = 0
    while m <= span + 0.01:
        anc = 'start' if m == 0 else ('end' if m + iv > span + 0.01 else 'middle')
        g.append(f'<text x="{X(t0 + m * 60):.1f}" y="138" text-anchor="{anc}" style="fill:var(--muted,#5a6b7b)">{_hms(t0 + m * 60)[:5]}</text>')
        m += iv
    defs = (f'<defs><pattern id="tlHatch{pid}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            '<rect width="6" height="6" style="fill:var(--train,#3a7cc4)"/><line x1="0" y1="0" x2="0" y2="6" style="stroke:#fff;stroke-width:2.4"/></pattern></defs>')
    return (f'<svg id="tlSrv{pid}" viewBox="0 0 {W} 144" width="100%" role="img" aria-label="Job timeline" style="font:10.5px sans-serif;display:block;max-width:{int(W * 1.6)}px">'
            + defs + ''.join(g) + '</svg>')


def timeline_html(r, fails, pid=''):
    """the timeline panel's body: the SVG, its legend, one summary line, and per failure the recovery process (a strip and a phase table)"""
    import html as _h
    E = _h.escape
    T = r.get('timeline') or {}; P = r.get('phases') or []
    sw = lambda kind, lab, extra='': (f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:12px"><span style="display:inline-block;width:12px;height:10px;'
                                      f'border-radius:2px;background:{_TLC[kind]};{extra}"></span>{lab}</span>')
    legend = ('<div style="font-size:12px;color:var(--muted,#5a6b7b);margin:4px 0 6px;line-height:1.9">' + sw('gate', 'pre-run tests') + sw('launch', 'srun launch and model build')
              + sw('load', 'checkpoint load') + sw('train', 'training steps') + sw('stop', 'failure until srun returned') + sw('recover', 'recovery decisions')
              + sw('train', 'steps computed again', 'background:repeating-linear-gradient(45deg,var(--train,#3a7cc4) 0 3px,#fff 3px 5px)')
              + sw('save', 'async save') + sw('conv', 'universal conversion') + sw('ecc', 'corrected ECC count grew') + sw('now', 'now') + '</div>')
    R = recoveries(r, fails)
    S = [s for p in P for s in p['steps']]
    still = sum((rc['resumed'] - rc['t_last']) for rc in R if rc['resumed'])
    ss = statistics.median([s[2] for s in S if s[2]] or [0])
    line = (f"gate {(T.get('gate') or ['0'])[0]} s · first step {_hms(S[0][0])[:5] if S else '-'} · {_dur(ss) if ss else '-'} per step · "
            f"{len(S)} steps logged in {len(P)} attempt{'s' if len(P) != 1 else ''}"
            + (f" · {len(R)} failure{'s' if len(R) != 1 else ''}" if R else ' · no failure')
            + (f"; the {sum(1 for rc in R if rc['resumed'])} recover{'ies' if sum(1 for rc in R if rc['resumed']) != 1 else 'y'} cost {_dur(still)} without progress" if any(rc['resumed'] for rc in R) else '')
            + (f"; the run stopped after #{R[-1]['n']}" if R and not R[-1]['restarted'] and not R[-1]['live'] else ''))
    cards = []
    for rc in R:
        f = rc['f']
        tot = sum(b - a for _, a, b, _ in rc['phases']) or 1
        x, strip = 0.0, []
        for name, a, b, why in rc['phases']:
            kind = 'stop' if name.startswith(('no progress', 'failed during')) else 'recover' if name.startswith('recovery') else 'launch' if name.startswith('srun') else 'load' if name.startswith('checkpoint') else 'first'
            w = max(0.6, (b - a) / tot * 100)
            strip.append(f'<span title="{E(name)}: {_dur(b - a)}" style="display:block;flex:0 0 {w:.2f}%;background:{_TLC[kind]};{"opacity:.45;" if kind == "first" else ""}"></span>')
        rows = ''.join(f'<tr><td>{i + 1}</td><td>{E(name)}</td><td class="num" style="text-align:right;white-space:nowrap">{_dur(b - a)}</td>'
                       f'<td style="white-space:nowrap;color:var(--muted,#5a6b7b)">{_hms(a)} → {_hms(b)}</td><td style="color:var(--muted,#5a6b7b)">'
                       + ('<ul style="margin:0;padding-left:1.1em">' + ''.join(f'<li>{E(x)}</li>' for x in why) + '</ul>' if isinstance(why, list) else E(why)) + '</td></tr>'
                       for i, (name, a, b, why) in enumerate(rc['phases']))
        if rc['redo']:
            d = rc['redo']
            rows += (f'<tr><td>{len(rc["phases"]) + 1}</td><td>steps computed again</td><td class="num" style="text-align:right">{d["n"]} step{"s" if d["n"] != 1 else ""}</td>'
                     f'<td></td><td style="color:var(--muted,#5a6b7b)">steps {d["first"]}–{d["last"]} ran before the failure but were not in the save the restart loaded</td></tr>')
        if rc['resumed']:
            res = (f"training resumed at {_hms(rc['resumed'])}" + (f" with step {rc['resume_step']}" if rc['resume_step'] is not None else '')
                   + f", {_dur(rc['resumed'] - rc['t_last'])} after the last step before the failure ({_dur(rc['resumed'] - rc['end'])} after srun returned)")
        elif rc['next_failed']:
            res = f"the restart (attempt {rc['next_k']}) ended before its first step: see #{rc['n'] + 1}"
        elif rc['restarted']:
            res = f"the restart is in its phase '{rc['phases'][-1][0]}' since {_hms(rc['phases'][-1][1])}" if rc['phases'] else 'the restart has not reached its first step yet'
        elif rc['live']:
            res = f"recovering now: the restart loop is deciding since {_hms(rc['end'])}"
        else:
            res = 'no restart: ' + next((re.sub(r'^(decision:\s*)?(NO RESTART:\s*|THE RUN STOPS[^:]*:\s*)?', '', d) for d in f.get('decisions') or [] if 'NO RESTART' in d or 'STOPS' in d), 'the run stopped')
        cards.append(f'<details style="border-top:1px solid var(--line,#e3e8ee);padding:4px 0"><summary style="cursor:pointer;display:grid;grid-template-columns:2.4em minmax(0,1fr);gap:6px;align-items:center;font-size:13px">'
                     f'<b style="color:{_TLC["stop"]}">#{rc["n"]}</b><span><span style="display:flex;height:12px;border-radius:3px;overflow:hidden;margin:2px 0 3px">{"".join(strip)}</span>'
                     f'{E(f.get("what", ""))} on {E(f.get("node", ""))} at {_hms(rc["t_last"])} · {E(res)}</span></summary>'
                     f'<table style="border-collapse:collapse;font-size:12px;margin:4px 0 6px 2.4em" class="tl-ph">{rows}</table></details>')
    rec = ''
    if cards:
        rec = ('<div style="margin-top:6px"><div style="font-weight:600;font-size:13px;margin-bottom:2px">Recovery after each failure <span style="font-weight:normal;color:var(--muted,#5a6b7b);font-size:11px">'
               'from the last step before it to the first step after the restart; click a line for the phases</span></div>' + ''.join(cards) + '</div>')
    css = '<style>.tl-ph td{padding:2px 8px;border-bottom:1px solid var(--line,#eef2f5);vertical-align:top}</style>'
    t0, t1, _, _ = _tl_span(r)
    hours = round((t1 - t0) / 3600, 1) if t0 and t1 else None
    return (css + f'<div id="tlServer{pid}" data-title="{(f"{round((t1 - t0) / 60)}-minute timeline" if t1 - t0 < 3600 else str(hours).rstrip("0").rstrip(".") + "-hour timeline") if hours is not None else "Job timeline"}" '
            f'data-sum="{_hms(t0)[:5]} → {_hms(t1)[:5]}">' + timeline_svg(r, fails, pid=pid) + legend
            + f'<div style="font-size:12px;color:var(--muted,#5a6b7b)">{E(line)}</div>' + rec + '</div>')


def static_block(data):
    """[2026-10-11, the user: the page showed no curve in VS Code] the same run as plain HTML + inline SVG, visible in a viewer that runs no
    scripts (the VS Code preview); a browser removes it and shows the interactive console instead."""
    import html as _h
    r = data['runs']['this']; pts = data.get('series', {}).get('this', [])
    tl = r.get('timeline') or {}
    att = ', '.join(f'attempt {a} at {t[11:19]}' for t, a in tl.get('attempts', []))
    rows = [('job', f"{r.get('job')} ({r.get('label', '')})"), ('last beat', r.get('lastbeat', '')), ('attempts', att or '-'),
            ('latest save', f"step {r.get('latest', '-')}"), ('latest universal checkpoint', f"step {r.get('latest_universal', '-')}"),
            ('spare nodes left', str(len(r.get('spares') or []))), ('the page was written', r.get('taken', ''))]
    last = pts[-10:]
    steps = ''.join(f"<tr><td>{p['step']}</td><td>{p.get('loss', '')}</td><td>{p.get('tflops', '')}</td><td>{p.get('step_s', '')}</td><td>{p.get('gnorm', '')}</td></tr>" for p in last)
    fails = r.get('failures') or []
    marks = [(int(f['step']), f"#{f['n']}") for f in fails if str(f.get('step', '')).isdigit()]
    charts = ''.join(f'<div class="sv-c">{_svg_chart(pts, k, lab, col, marks=marks)}</div>' for k, lab, col in
                     (('loss', 'training loss', '#c0392b'), ('tflops', 'TFLOP/s per GCD (peak 191.5)', '#0b7285'),
                      ('step_s', 'seconds per step', '#8a63d2'), ('gnorm', 'gradient norm', '#d99a00'),
                      ('vram', 'device memory used (VRAM), GiB on the fullest GCD (64 GiB per GCD)', '#1f5fa8'),
                      ('alloc', 'allocated by PyTorch, GiB (a part of the VRAM)', '#8a98a6')))
    import html as _h2
    al = list(reversed(r.get('alarm_log') or []))
    alarms = ('<h2 style="font-size:15px;margin:10px 0 2px">Alarms (' + str(len(r.get('alarms') or [])) + ' raised), newest first</h2>'
              '<div style="max-height:240px;overflow-y:auto;border:1px solid #e3e8ee;border-radius:6px;font-size:12px">'
              + (''.join(f'<div style="padding:3px 8px;border-bottom:1px solid #eef2f5">{_h2.escape(l)}</div>' for l in al)
                 or '<div style="padding:6px 8px;color:#5a6b7b">No alarm: alarms.log holds no alarm line.</div>') + '</div>')
    vr = r.get('vram') or {}
    if vr.get('peak') is not None:
        rows.insert(3, ('device memory (VRAM)', f"peak {vr['peak']:.1f} GiB of {vr['dev']:.0f} GiB on the fullest GCD this attempt; now {vr['now']:.1f} (monitor sample {vr['at']})"))
    return ('<div id="static-view" style="background:#fff;color:#122130;font:14px/1.45 sans-serif;padding:14px 16px;max-width:1340px">'
            '<style>.wrap{display:none}#static-view table{border-collapse:collapse;margin:6px 0 12px}#static-view td,#static-view th{border-bottom:1px solid #e3e8ee;'
            'padding:3px 10px;text-align:left}.sv-c{display:inline-block;margin:0 10px 8px 0;border:1px solid #e3e8ee}.sv-empty{color:#5a6b7b;padding:8px}</style>'
            f'<h1 style="font-size:19px;margin:0 0 4px">Run console of job {_h.escape(str(r.get("job")))}: static view</h1>'
            '<p style="color:#5a6b7b;margin:0 0 10px">This viewer runs no scripts, so it shows this plain copy. Open the same file through the login-node '
            'server in a browser for the interactive console (machine map, node inspector, gate, timeline).</p>'
            '<table>' + ''.join(f'<tr><th>{_h.escape(k)}</th><td>{_h.escape(str(v))}</td></tr>' for k, v in rows) + '</table>'
            + alarms + failures_table(fails)
            + _try(lambda: '<h2 style="font-size:15px;margin:10px 0 2px">Job timeline</h2>' + timeline_html(r, fails, 'S'), '')
            + charts +
            '<h2 style="font-size:15px;margin:8px 0 2px">The last steps</h2><table><tr><th>step</th><th>loss</th><th>TFLOP/s per GCD</th><th>s per step</th>'
            '<th>gradient norm</th></tr>' + steps + '</table></div>'
            '<script>document.addEventListener("DOMContentLoaded",function(){var s=document.getElementById("static-view");if(s)s.remove();'
            'var w=document.querySelector(".wrap");if(w)w.style.display="block";});</script>\n')


def write(job, out=None, history=None, evals=None, title=None):
    # with --out elsewhere (a test, or a page for others), the cache sits next to the page, so nothing is written into the job
    data = build(job, history, evals, title, os.path.join(os.path.dirname(os.path.abspath(out)), '.dashboard_cache.json') if out else None)
    try:
        data['runs']['this']['failures'] = failures_of(job)       # [2026-10-11] read-only: failures.tsv, attempts.tsv, recovery.log
    except Exception:
        data['runs']['this']['failures'] = []
    tpl = open(os.path.join(HERE, 'dashboard_template.html')).read()
    html = tpl.replace('__DATA__', json.dumps(data, separators=(',', ':')).replace('</', '<\\/'))
    html = html.replace('<title>V3 256k Run Console</title>', f'<title>Run console {data["runs"]["this"]["job"]}</title>', 1)
    html = '<!doctype html>\n<meta http-equiv="refresh" content="30">\n' + html      # the template opens with its charset line
    try:                                                     # [2026-10-11] the script-free copy right after the template's style block
        i = html.index('<div class="wrap">')
        html = html[:i] + static_block(data) + html[i:]
        j = html.index('</header>', html.index('<div class="wrap">'))       # the same table in the interactive console
        html = html[:j + 9] + failures_table(data['runs']['this'].get('failures') or []) + html[j + 9:]
        a, b = html.find('<!--TL_OLD-->'), html.find('<!--/TL_OLD-->')    # [2026-10-11] the timeline with the failures and recoveries;
        tl = _try(lambda: timeline_html(data['runs']['this'], data['runs']['this'].get('failures') or []), None)   # on an error the browser
        if 0 <= a < b and tl:                                                                                    # draws the old timeline
            html = html[:a] + tl + html[b + len('<!--/TL_OLD-->'):]
    except Exception as e:                                   # the static copy must never stop the page
        html = html.replace('<div class="wrap">', f'<!-- static view failed: {type(e).__name__} -->\n<div class="wrap">', 1)
    out = out or os.path.join(job, 'dashboard.html')
    tmp = out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(html)
    os.replace(tmp, out)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('job'); ap.add_argument('--out'); ap.add_argument('--title')
    ap.add_argument('--history', default=os.environ.get('AUTORUN_DASHBOARD_HISTORY') or None)
    ap.add_argument('--evals', default=os.environ.get('AUTORUN_DASHBOARD_EVALS') or None)
    a = ap.parse_args()
    print(write(a.job, a.out, a.history, a.evals, a.title))

