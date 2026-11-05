#!/usr/bin/env python3
"""job_index_watch.py (2026-10-13, sandbox): the index of every job that runs, served on login10 port 8766.

    python3 job_index_watch.py --out <folder> [--tsv <jobs.tsv> ...] [--every 120] [--once]

The user's standing rule (2026-10-13): every job that runs from now on is recorded here. The jobs come from the experiment's jobs.tsv
(header `label job submitted nodes qos what`, tab-separated; extra columns are ignored; short rows are tolerated). Every --every
seconds it:
  1. reads the jobs.tsv files and asks Slurm once (one sacct call for the listed ids, one squeue call for the user); an empty or failed
     answer is UNKNOWN: the last known state is kept, a job is never shown as ended because Slurm did not answer;
  2. builds a run console for every training job or probe whose folder (logs/job_<id>_* in the sandbox or the production tree) is a
     resilient run (data/loop.log or heartbeat.log): the sandbox's dashboard_html.py writes <out>/job_<id>/index.html, with its cache
     beside it (nothing is written into the job folder); rebuilt every cycle while the job runs, once more after it ends, then left;
     each build is a separate process with a time limit, so one bad job cannot stop the index;
  3. reads the answers so far of every evaluation job from its dispatch journal (eval/logs/job_<id>_*/dispatch.journal.jsonl), read-only;
  4. writes <out>/index.html atomically (static, no scripts; it reloads itself every 120 s) and <out>/.index_state.json (its memory).
Row kinds by label: EVAL_<checkpoint>_<sub-task> = evaluation; UCP_<arm>_s<N> = conversion to the universal form; anything else =
training, probe or data job. READ-ONLY on jobs, job folders and caches. Run it at nice 19 (bin/job_index_serve.sh).
"""
import argparse, glob, html, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SB_SF = '/lustre/orion/gen150/scratch/zixianw4/X-MoE-resilience/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier'
PR_SF = '/lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier'
LOG_DIRS = [f'{SB_SF}/logs', f'{PR_SF}/logs']
EVAL_LOGS = f'{PR_SF}/eval/logs'
DEFAULT_TSV = ('/lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/assistant_deep_dive/'
               '2026-10-13-ds16b_data_quality/jobs.tsv')
PY = sys.executable
USER = os.environ.get('USER', 'zixianw4')
E = lambda s: html.escape(str(s), quote=True)
FINAL = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE')
SUB = {'niah3': 'NIAH single 3', 'mk3': 'NIAH multikey 3', 'mq': 'NIAH multiquery', 'mv': 'NIAH multivalue',
       'qahotpot': 'QA HotpotQA', 'qasquad': 'QA SQuAD', 'gsm8k': 'GSM8K', 'humaneval': 'HumanEval'}
ARM = {'base': 'baseline lc128k-v2', 'new': 'arm NEW', 'old': 'arm OLD (control)', 'full': 'arm OLD-FULL (no intra-doc)',
       'y80base': 'DeepSeek-V3 671B released base read at YaRN 80 (RULER-256k)',
       'base': 'baseline', 's24': 'v3-lc256k-run1 step 24', 's35': 'v3-lc256k-run1 step 35'}   # 2026-10-15: the E1-E5 gain tests (labels E<k>_<ckpt>_<task>)   # 2026-10-14: the factor-80 reading of the base (scope 2026-10-14-rope_yarn_stages)
EVAL_RX = re.compile(r'^EVAL_(?P<arm>[a-z][a-z0-9]*)_s(?P<step>\d+)_(?P<sub>[a-z0-9]+)', re.I)   # 2026-10-14: an arm name may carry digits (y80base)
UCP_RX = re.compile(r'^UCP_(?P<arm>[a-z]+)_s(?P<step>\d+)', re.I)
TRAINISH = re.compile(r'train|probe|\barm\b|72-node|resilient', re.I)


def log(msg):
    print(f'{time.strftime("%F %T")} {msg}', flush=True)


def sh(cmd, timeout=60):
    """(returncode, stdout); returncode None on a timeout or an OS error"""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout
    except (subprocess.SubprocessError, OSError):
        return None, ''


def read_tsv(paths):
    rows = []
    for p in paths:
        try:
            lines = open(p, errors='replace').read().splitlines()
        except OSError:
            continue
        for ln in lines:
            v = ln.split('\t')
            if not v or not v[0].strip() or v[0] == 'label':
                continue
            v += [''] * (6 - len(v))
            job = v[1].strip()
            if not re.fullmatch(r'\d+', job):
                continue
            rows.append(dict(label=v[0].strip(), job=job, submitted=v[2].strip(), nodes=v[3].strip(), qos=v[4].strip(),
                             what=v[5].strip(), tsv=p))
    return rows


def slurm_states(ids, known):
    """{id: {state, start, end, elapsed, limit, nodes, name}} from one sacct call and one squeue call; unknown answers keep `known`"""
    out = {i: dict(known.get(i, {})) for i in ids}
    ok_sacct = ok_squeue = False
    if ids:
        rc, txt = sh(['sacct', '-j', ','.join(sorted(ids)), '-X', '-n', '-P', '-o',
                      'JobID,JobName,State,Start,End,Elapsed,NNodes,Timelimit'], timeout=90)
        if rc == 0 and txt.strip():
            ok_sacct = True
            for l in txt.splitlines():
                v = l.split('|')
                if len(v) < 8 or v[0] not in out:
                    continue
                out[v[0]].update(name=v[1], state=(v[2].split() or ['?'])[0], start=v[3], end=v[4], elapsed=v[5], nodes=v[6],
                                 limit=v[7], src='sacct')
    rc, txt = sh(['squeue', '-u', USER, '-h', '-o', '%i|%T|%M|%l|%D|%S|%r|%j'], timeout=60)
    if rc == 0:
        ok_squeue = True
        for l in txt.splitlines():
            v = l.split('|')
            if len(v) < 8 or v[0] not in out:
                continue
            out[v[0]].update(state=v[1], elapsed=v[2], limit=v[3], nodes=v[4], start=v[5], reason=v[6], name=v[7], src='squeue')
    return out, ok_sacct, ok_squeue


def job_folder(jid):
    for d in LOG_DIRS:
        c = sorted(glob.glob(f'{d}/job_{jid}_*'))
        if c:
            return c[0]
    return None


def is_resilient(folder):
    return bool(folder) and (os.path.exists(f'{folder}/data/loop.log') or os.path.exists(f'{folder}/heartbeat.log'))


def journal_answers(jid):
    """(answered, total) from the eval job's dispatch journal, or None when there is none"""
    c = sorted(glob.glob(f'{EVAL_LOGS}/job_{jid}_*/dispatch.journal.jsonl'))
    if not c:
        return None
    nblocks, done = {}, {}
    try:
        with open(c[0], errors='replace') as fh:
            for l in fh:
                try:
                    r = json.loads(l)
                except ValueError:
                    continue
                ph = r.get('phase', '')
                if r.get('event') == 'open':
                    nblocks[ph] = int(r.get('nblocks') or 0)
                elif r.get('event') == 'complete':
                    done[(ph, r.get('block'))] = int(r.get('n') or 1)
    except OSError:
        return None
    return sum(done.values()), sum(nblocks.values())


def build_console(folder, jid, out, timeout=300):
    d = f'{out}/job_{jid}'
    os.makedirs(d, exist_ok=True)
    rc, txt = sh(['nice', '-n', '19', PY, f'{HERE}/dashboard_html.py', folder, '--out', f'{d}/index.html'], timeout=timeout)
    return rc == 0, txt.strip()[-200:]


def ck_name(arm, step):
    a = ARM.get(arm.lower(), f'arm {arm}')
    return f'{a} step {int(step)}'


def st_cell(s):
    s = (s or 'unknown').split()[0]
    return f'<span class="st st-{E(s.lower())}">{E(s.lower())}</span>'


def when(j):
    st = j.get('state', '')
    hm = lambda x: (x or '').replace('T', ' ')[5:16] if re.match(r'^\d{4}-', x or '') else '-'
    if st == 'RUNNING':
        return f'started {hm(j.get("start"))}'
    if st == 'PENDING':
        exp = hm(j.get('start'))
        return (f'queued, expected {exp}' if exp != '-' else 'queued') + (f' ({E(j.get("reason"))})' if j.get('reason') else '')
    if st in FINAL:
        return f'ended {hm(j.get("end"))}'
    return '-'


HAND_RX = re.compile(r'<!-- hand-over -->.*?<!-- /hand-over -->', re.S)
REFRESH_RX = re.compile(r'<meta http-equiv="refresh"[^>]*>', re.I)


def write_atomic(path, text):
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(text)
    os.replace(tmp, path)


def run_dir_name(name):
    return re.sub(r'[^A-Za-z0-9._-]', '_', name)


def hand_over(out, S, jobs):
    """[2026-10-13 16:00, the user: "the local host is not updated automatically"] One run can need several Slurm jobs (a stop, then a
    resubmission that resumes the save); a browser left on the console of the ended job then shows a page that never changes. The jobs
    of one run share their Slurm job name. For each name: run/<name>/index.html always opens the console of the newest job of that name;
    the console of every OLDER job that has ended carries a banner and moves to the newest job's console after 20 s; the ended job's own
    page stays readable at job_<id>/record.html (no banner, no reload). Model-agnostic: it uses only the Slurm names and states.
    Returns {job id: newest job id of its name} for every job of a name with more than one job."""
    by = {}
    for j in jobs:
        n = (S.get(j, {}) or {}).get('name')
        if n:
            by.setdefault(n, []).append(j)
    newest_of = {}
    for n, L in by.items():
        L = sorted(set(L), key=int)
        newest = L[-1]
        for j in L:
            newest_of[j] = newest
        have = os.path.exists(f'{out}/job_{newest}/index.html')
        nst = ((S.get(newest, {}) or {}).get('state') or 'unknown').split()[0].lower()
        rd = f'{out}/run/{run_dir_name(n)}'
        os.makedirs(rd, exist_ok=True)
        tgt = f'../../job_{newest}/index.html' if have else '../../index.html'
        write_atomic(f'{rd}/index.html', f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" '
                     f'content="0; url={tgt}"><title>Run {E(n)}</title><style>html,body{{background:#fff;font:14px system-ui,sans-serif}}'
                     f'</style></head><body><p>Opening the console of job {E(newest)} ({E(nst)}), the newest job of the run {E(n)}: '
                     f'<a href="{tgt}">open it</a>.</p></body></html>\n')
        if len(L) < 2:
            continue
        for j in L[:-1]:
            st = ((S.get(j, {}) or {}).get('state') or '').split()[0] if (S.get(j, {}) or {}).get('state') else ''
            page, rec = f'{out}/job_{j}/index.html', f'{out}/job_{j}/record.html'
            if st not in FINAL or not os.path.exists(page):
                continue
            if not os.path.exists(rec):
                write_atomic(rec, REFRESH_RX.sub('', HAND_RX.sub('', open(page, encoding='utf-8', errors='replace').read())))
            base = open(rec, encoding='utf-8', errors='replace').read()
            end = ((S.get(j, {}) or {}).get('end') or '').replace('T', ' ')[5:16]
            if have:
                meta = f'<meta http-equiv="refresh" content="20; url=../job_{newest}/index.html">'
                move = (f'This page moves to the console of job {E(newest)} in 20 s (<a href="../job_{E(newest)}/index.html">go now</a>).')
            else:
                meta = '<meta http-equiv="refresh" content="30">'
                move = f'Job {E(newest)} is {E(nst)}; this page moves to its console once it starts.'
            banner = (f'<!-- hand-over --><div style="background:#fff4d6;border:1px solid #e0b84a;border-radius:8px;padding:10px 14px;'
                      f'margin:10px 16px;font:14px/1.45 system-ui,sans-serif;color:#3d2e00">'
                      f'<b>This job ended</b> ({E(st.lower())}, {E(end or "-")}). The run <b>{E(n)}</b> continues in job <b>{E(newest)}</b> '
                      f'({E(nst)}). {move} This job\'s own record, without the move: <a href="record.html">record.html</a>. To follow the '
                      f'run across its jobs, open <a href="../run/{E(run_dir_name(n))}/index.html">run/{E(run_dir_name(n))}/</a>.</div>'
                      f'<!-- /hand-over -->')
            new = REFRESH_RX.sub('', base)
            # the console pages omit <head> and <body> (HTML allows it): the reload goes right after the doctype (or a real <head>), the
            # banner right after <body> or, without one, before the first <div> (the first element of the page's content)
            if re.search(r'<head(\s[^>]*)?>', new, re.I):
                new = re.sub(r'(<head(\s[^>]*)?>)', lambda m: m.group(1) + meta, new, count=1, flags=re.I)
            elif re.match(r'\s*<!doctype[^>]*>', new, re.I):
                new = re.sub(r'^(\s*<!doctype[^>]*>)', lambda m: m.group(1) + '\n' + meta, new, count=1, flags=re.I)
            else:
                new = meta + new
            if re.search(r'<body(\s[^>]*)?>', new, re.I):
                new = re.sub(r'(<body(\s[^>]*)?>)', lambda m: m.group(1) + banner, new, count=1, flags=re.I)
            elif '<div' in new:
                i = new.index('<div'); new = new[:i] + banner + new[i:]
            else:
                new = new + banner
            old = open(page, encoding='utf-8', errors='replace').read()
            if new != old:
                write_atomic(page, new)
    return newest_of


def cycle(args, state):
    rows = read_tsv(args.tsv)
    ids = {r['job'] for r in rows}
    S, ok_a, ok_q = slurm_states(ids, state.get('jobs', {}))
    state['jobs'] = {**state.get('jobs', {}), **S}
    built = state.setdefault('built_final', [])
    t_read = time.strftime('%H:%M:%S')
    train, evals, convs, other = [], [], [], []
    consoled = []
    for r in rows:
        j = S.get(r['job'], {})
        stv = j.get('state') or 'unknown'
        el = f'{j.get("elapsed", "-") or "-"} / {j.get("limit", "-") or "-"}'
        nodes = j.get('nodes') or r['nodes'] or '-'
        m = EVAL_RX.match(r['label'])
        if m:
            key = f'ans_{r["job"]}'
            a = state.get(key)
            if a is None or stv not in FINAL or not a.get('final'):
                got = journal_answers(r['job'])
                a = dict(v=list(got) if got else None, final=stv in FINAL)
                state[key] = a
            ans = f'{a["v"][0]} of {a["v"][1]}' if a.get('v') else '–'
            evals.append((stv, r, (f'<td class="mono">{E(r["job"])}</td><td>{E(ck_name(m["arm"], m["step"]))}</td>'
                                   f'<td>{E(SUB.get(m["sub"].lower(), m["sub"]))}</td><td>{st_cell(stv)}</td><td class="num">{E(nodes)}</td>'
                                   f'<td class="num">{E(el)}</td><td>{when(j)}</td><td class="num">{E(ans)}</td>'
                                   f'<td class="lab">{E(r["label"])}</td>')))
            continue
        m = UCP_RX.match(r['label'])
        if m:
            convs.append((stv, r, (f'<td class="mono">{E(r["job"])}</td><td>{E(ck_name(m["arm"], m["step"]))}</td><td>{st_cell(stv)}</td>'
                                   f'<td class="num">{E(nodes)}</td><td class="num">{E(el)}</td><td>{when(j)}</td><td class="lab">{E(r["label"])}</td>')))
            continue
        folder = job_folder(r['job'])
        link = ''
        if is_resilient(folder):
            page = f'{args.out}/job_{r["job"]}/index.html'
            need = stv == 'RUNNING' or not os.path.exists(page) or (stv in FINAL and r['job'] not in built)
            if need:
                ok, msg = build_console(folder, r['job'], args.out)
                if ok and stv in FINAL and r['job'] not in built:
                    built.append(r['job'])
                if not ok:
                    log(f'console build of job {r["job"]} failed: {msg}')
            if os.path.exists(page):
                link = f'<a href="job_{E(r["job"])}/index.html">run console</a>'
        cells = (f'<td class="mono">{E(r["job"])}</td><td>{st_cell(stv)}</td><td class="num">{E(nodes)}</td><td class="num">{E(el)}</td>'
                 f'<td>{when(j)}</td><td>{link or "–"}</td><td class="wrapc">{E(r["what"])}</td><td class="lab">{E(r["label"])}</td>')
        if link:
            consoled.append(r['job'])
        (train if (link or folder and is_resilient(folder) or TRAINISH.search(r['label'] + ' ' + r['what'])) else other).append((stv, r, cells))
    newest_of = hand_over(args.out, S, consoled)
    def follow(x):
        stv, r, cells = x
        nw = newest_of.get(r['job'])
        if not nw:
            return x
        nm = run_dir_name((S.get(r['job'], {}) or {}).get('name', ''))
        add = (f' · <a href="run/{E(nm)}/index.html">follow the run</a>' if nw == r['job'] else
               f' · continues in <a href="job_{E(nw)}/index.html">{E(nw)}</a>')
        return (stv, r, cells.replace('>run console</a>', '>run console</a>' + add, 1))
    train = [follow(x) for x in train]
    order = {'RUNNING': 0, 'PENDING': 1}
    srt = lambda L: sorted(L, key=lambda x: (order.get(x[0], 2), -int(x[1]['job'])))
    tbl = lambda L, hdr, empty: ('<div class="scroll"><table class="t"><tr>' + ''.join(f'<th>{h}</th>' for h in hdr) + '</tr>'
                                 + (''.join(f'<tr>{c}</tr>' for _, _, c in srt(L)) or f'<tr><td colspan="{len(hdr)}" class="na">{empty}</td></tr>')
                                 + '</table></div>')
    n_run = sum(1 for L in (train, evals, convs, other) for x in L if x[0] == 'RUNNING')
    n_pd = sum(1 for L in (train, evals, convs, other) for x in L if x[0] == 'PENDING')
    slurm_note = ('' if ok_a and ok_q else
                  f'<div class="note">Slurm did not answer this cycle ({"sacct" if not ok_a else ""}{" and " if not ok_a and not ok_q else ""}'
                  f'{"squeue" if not ok_q else ""}): the states shown are the last known ones, not a sign that a job ended.</div>')
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="{int(args.every)}"><title>Job index</title>
<style>
html,body{{background:#fff;color:#122130;margin:0;font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}}
.wrap{{max-width:1340px;margin:0 auto;padding:16px}}h1{{font-size:20px;margin:0 0 4px}}h2{{font-size:16px;margin:22px 0 6px}}
.lead{{color:#33475b;margin:4px 0 12px}}a{{color:#1f5fa8}}
.pills span{{display:inline-block;border:1px solid #d5dde5;border-radius:12px;padding:2px 10px;margin:0 6px 6px 0;font-size:13px}}
.scroll{{overflow-x:auto;border:1px solid #e3e8ee;border-radius:8px}}
table.t{{border-collapse:collapse;width:100%;font-size:13px}}table.t th,table.t td{{border-bottom:1px solid #eef2f5;padding:4px 8px;text-align:left;white-space:nowrap;vertical-align:top}}
table.t th{{background:#f6f8fa;font-weight:600;color:#33475b}}td.num{{text-align:right;font-variant-numeric:tabular-nums}}.mono{{font-family:ui-monospace,monospace}}
table.t td.wrapc{{white-space:normal;min-width:280px;max-width:640px}}td.lab{{color:#8a98a6;font-size:11.5px;font-family:ui-monospace,monospace}}td.na{{color:#9aa6b2;text-align:center}}
.st{{display:inline-block;border-radius:10px;padding:0 8px;font-size:12px;background:#eef1f4;color:#33475b}}.st-running{{background:#d3eef2;color:#0b5560}}
.st-completed{{background:#e3f4e8;color:#1e6b34}}.st-timeout,.st-failed,.st-node_fail,.st-out_of_memory,.st-cancelled{{background:#fbe3e1;color:#9b2c22}}
.note{{border-left:3px solid #cf2d5a;background:#fdf1f3;padding:8px 12px;border-radius:0 6px 6px 0;margin:8px 0;font-size:13px}}
.how{{border-left:3px solid #0b7285;background:#eef6f7;padding:8px 12px;border-radius:0 6px 6px 0;margin:16px 0 8px;font-size:13px}}
@media (max-width:640px){{.wrap{{padding:10px 16px}}}}
</style></head><body><div class="wrap">
<h1>Job index: every job that runs</h1>
<div class="lead">This page lists every Slurm job of the current experiments, one row per job, read from the experiments' job tables
({E(', '.join(args.tsv))}). The rule (the user, 2026-10-13): every job that runs from now on is recorded here. A training run or probe
links to its run console (machine map, failures, timeline, loss), rebuilt every {int(args.every)} s while it runs; an evaluation row shows how many
answers its job has written so far. Written {E(now)} on the login node; job states read from Slurm at {E(t_read)}; the page reloads itself every
{int(args.every)} s.</div>
{slurm_note}
<div class="pills"><span>{n_run} running</span><span>{n_pd} queued</span><span>{len(rows)} jobs recorded</span></div>
<h2>Training runs and probes</h2>
{tbl(train, ['job', 'state', 'nodes', 'elapsed / limit', 'time', 'console', 'what it is for', 'label'], 'no training run or probe is recorded yet')}
<h2>Evaluation jobs</h2>
<div class="lead">One job evaluates one sub-task (or GSM8K, or HumanEval) of one checkpoint. "Answers" counts the requests its job has answered,
read from the job's dispatch journal (for RULER one request is one document); "–" means the job has not started answering.</div>
{tbl(evals, ['job', 'checkpoint', 'sub-task', 'state', 'nodes', 'elapsed / limit', 'time', 'answers', 'label'], 'no evaluation job is recorded yet')}
<h2>Conversions to the universal form</h2>
{tbl(convs, ['job', 'checkpoint', 'state', 'nodes', 'elapsed / limit', 'time', 'label'], 'no conversion job is recorded yet')}
<h2>Other jobs (data preparation and the rest)</h2>
{tbl(other, ['job', 'state', 'nodes', 'elapsed / limit', 'time', 'console', 'what it is for', 'label'], 'none')}
<h2>Earlier pages</h2>
<div class="scroll"><table class="t"><tr><th>page</th><th>what it holds</th></tr>
<tr><td><a href="tests-2026-10-11/index.html">Production port tests and the 8-hour run, 2026-10-11</a></td><td class="wrapc">the run consoles of the
resilient-training port tests (features, time left, page cost on and off) and of the DeepSeek-V3 671B 8-hour run at 262,144 tokens</td></tr></table></div>
<div class="how">Open: <span class="mono">ssh -N -L 8766:localhost:8766 &lt;user&gt;@login10.frontier.olcf.ornl.gov</span>, then http://localhost:8766/ .
Source: {E(HERE)}/job_index_watch.py (read-only on jobs and job folders).</div>
</div></body></html>
'''
    tmp = f'{args.out}/index.html.{os.getpid()}.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(page)
    os.replace(tmp, f'{args.out}/index.html')
    return len(rows), ok_a, ok_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--tsv', action='append', default=None)
    ap.add_argument('--every', type=float, default=120)
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    args.tsv = args.tsv or [DEFAULT_TSV]
    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)
    spath = f'{args.out}/.index_state.json'
    while True:
        t0 = time.time()
        try:
            state = json.load(open(spath))
        except (OSError, ValueError):
            state = {}
        try:
            n, ok_a, ok_q = cycle(args, state)
            tmp = f'{spath}.{os.getpid()}.tmp'
            json.dump(state, open(tmp, 'w')); os.replace(tmp, spath)
            log(f'index written: {n} jobs, sacct {"ok" if ok_a else "UNKNOWN"}, squeue {"ok" if ok_q else "UNKNOWN"}, {time.time() - t0:.1f} s')
        except Exception as e:                       # the loop must never die on one bad cycle
            log(f'cycle failed: {type(e).__name__}: {e}')
        if args.once:
            return
        time.sleep(max(5, args.every - (time.time() - t0)))


if __name__ == '__main__':
    main()
