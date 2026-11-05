#!/usr/bin/env python3
"""dashboard_watch.py (2026-10-10, sandbox tree): a run console for RUNNING jobs, built only from their logs, served on localhost.

    python3 dashboard_watch.py --out <folder> --job y64=<job folder> --job y40=<job folder> \
        [--label y64="YaRN 64"] [--history y64=<jobid>,<jobid>] [--history p128k=<jobid>,...] [--history-name p128k="128k parent run"] \
        [--evals-tsv <jobs.tsv>] [--base-run <checkpoint folder name>] [--phase-note <text>] [--title <text>] [--every 120] [--port 8765]

What it does, every --every seconds (default 120), at nice 19 and the idle I/O class:
  1. for each --job, dashboard_html.build() reads the job folder READ-ONLY (the heads and tails of the heartbeat files, only the new
     bytes of heartbeat.log and of the last rank's log, training.log, gate files, loop.log, the checkpoint folder listing); its cache
     lives in <out>/.cache_<label>.json, so NOTHING is written into the job folder or the production tree;
  2. the earlier jobs of each stream (--history) are read once at start from their last rank's log and kept in <out>/.history.json;
  3. the evaluation results come from the results.json of the eval jobs listed in --evals-tsv (EVAL rows), with squeue for the
     state of the ones still in flight;
  4. it writes <out>/index.html atomically (the page reloads itself every --every seconds) and serves <out> on 127.0.0.1:<port>.
Reach it from a laptop with: ssh -N -L <port>:localhost:<port> <user>@<this login node>, then http://localhost:<port>/
(VS Code Remote-SSH can forward the port from its Ports panel). Stop with Ctrl-C or kill; it never touches the jobs.
"""
import argparse, glob, http.server, json, os, re, subprocess, sys, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dashboard_html as H


def kv(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition('=')
        out[k] = v
    return out


def last_rank_log(jobdir):
    c = sorted(glob.glob(f'{jobdir}/rank_*.log') + glob.glob(f'{jobdir}/attempts/attempt_*/ranks/rank_*.log'),
               key=lambda p: int(re.search(r'rank_(\d+)\.log', p).group(1)))
    return c[-1:] if c else []


def history_points(logs_dir, jobids):
    """every logged step of finished jobs, from their last rank's log (loss, TFLOP/s, step time, gradient norm)"""
    pts = {}
    for j in jobids:
        for d in glob.glob(f'{logs_dir}/job_{j}_*'):
            for f in last_rank_log(d):
                for line in open(f, 'rb'):
                    if b'lm loss' not in line:
                        continue
                    t = line.decode(errors='replace')
                    g = lambda k: (re.search(k + r':?\s*([\d.E+-]+)', t) or [None, None])[1]
                    m = re.search(r'iteration +(\d+)/', t)
                    if not m:
                        continue
                    it = int(m.group(1))
                    pts[it] = dict(step=it, job=j, loss=float(g('lm loss')), tflops=float(g('TFLOPs') or 0) or None,
                                   step_s=round(float(g(r'elapsed time per iteration \(ms\)') or 0) / 1000, 1), gnorm=float(g('grad norm') or 0) or None)
    return [pts[k] for k in sorted(pts)]


def evals(jobs_tsv, eval_logs, results_json=None):
    """[{run, step, tasks{task: {score}}, jobs, state}] from the EVAL rows of jobs.tsv and the jobs' results.json"""
    rows = {}
    for line in open(jobs_tsv, errors='replace'):
        v = line.rstrip('\n').split('\t')
        if len(v) < 6 or not v[0].startswith('EVAL'):
            continue
        m = re.search(r' on (\S+?)/global_step(\d+)_universal', v[5])
        if not m:
            continue
        res = None
        d = glob.glob(f'{eval_logs}/job_{v[1]}_*')
        if d and os.path.exists(d[0] + '/results.json'):
            try:
                for t, x in json.load(open(d[0] + '/results.json'))['results'].items():
                    res = (t, x.get('262144,none'))
            except (OSError, ValueError, KeyError):
                pass
        rows.setdefault((m.group(1), int(m.group(2))), []).append((v[1], res))
    ids = sorted({j for js in rows.values() for j, _ in js})
    q = {}
    try:
        o = subprocess.run(['squeue', '-h', '-j', ','.join(ids), '-o', '%i %T'], capture_output=True, text=True, timeout=30).stdout if ids else ''
        q = dict(l.split() for l in o.strip().split('\n') if l.strip())
    except Exception:
        pass
    over = {}
    if results_json and os.path.exists(results_json):
        # [2026-10-11] cells scored from the response cache (complete or stopped) override what the job folders alone can show
        R = json.load(open(results_json))
        for name, cells in R.get('streams', {}).items():
            m = re.search(r'\((\S+?),', name)
            if not m:
                continue
            for st, c in cells.items():
                if not st.split()[0].isdigit() or 'released' in st:
                    continue
                n = sum(c.get('docs_scored', {}).values())
                over[(m.group(1), int(st.split()[0]))] = (dict((t, dict(score=v)) for t, v in c['scores'].items()),
                                                          'done' if n >= 200 else f'stopped: {n} of 200 documents scored (from the cache)')
    out = []
    for (run, step), js in sorted(rows.items()):
        tasks = {r[0]: dict(score=r[1]) for _, r in js if r}
        live = [q[j] for j, _ in js if j in q]
        state = 'done' if len(tasks) >= 4 else ('running' if 'RUNNING' in live else 'queued' if live else f'{len(tasks)} of 4 sub-tasks')
        if (run, step) in over:
            tasks, state = over.pop((run, step))
        out.append(dict(run=run, step=step, tasks=tasks, jobs=[j for j, _ in js], state=state))
    for (run, step), (tasks, state) in sorted(over.items()):
        out.append(dict(run=run, step=step, tasks=tasks, jobs=[], state=state))
    return sorted(out, key=lambda e: (e['run'], e['step']))


def build_once(a, jobs, labels, hist, out):
    ev_path = None
    ev = []
    if a.evals_tsv:
        try:
            ev = evals(a.evals_tsv, a.eval_logs, a.results_json)
            ev_path = os.path.join(out, '.evals.json')
            json.dump(ev, open(ev_path + '.tmp', 'w')); os.replace(ev_path + '.tmp', ev_path)
        except Exception as e:
            print(f'[watch] evals skipped: {type(e).__name__}: {e}', flush=True)
    runs, series, labels_stage = {}, {}, None
    for k, jd in jobs.items():
        d = H.build(jd, None, ev_path, None, os.path.join(out, f'.cache_{k}.json'))
        r = d['runs']['this']; r['label'] = labels.get(k, k)
        runs[k] = r
        series[k] = hist.get(k, []) + [p for p in d['series']['this'] if p['step'] not in {q['step'] for q in hist.get(k, [])}]
        labels_stage = labels_stage or d['stage_labels']
    for k, pts in hist.items():
        if k not in jobs:
            series[k] = pts
    meta = [dict(k=k, name=a.hist_names.get(k, k), dash='4 3') for k in hist if k not in jobs] + \
           [dict(k=k, name=labels.get(k, k)) for k in jobs]
    k0 = next(iter(jobs))                          # [2026-10-12, the user] the newest sandbox page style (dashboard_html.write): the
    try:                                           # failures table, the failure-aware timeline and the script-free static view of
        runs[k0]['failures'] = H.failures_of(jobs[k0])   # the primary job (the first --job); read-only
    except Exception:
        runs[k0]['failures'] = []
    data = dict(runs=runs, series=series, series_meta=meta, stage_labels=labels_stage, title=a.title, live=True)
    base = next((e for e in ev if e['run'] == a.base_run), None) if a.base_run else None
    if base:
        data['eval_base'] = dict(label=a.base_label, ruler={t: x['score'] for t, x in base['tasks'].items()})
    if ev:
        data['eval_note'] = a.eval_note
    if a.phase_note:
        data['phase_note'] = a.phase_note
    tpl = open(os.path.join(HERE, 'dashboard_template.html')).read()
    html = tpl.replace('__DATA__', json.dumps(data, separators=(',', ':')).replace('</', '<\\/'))
    html = html.replace('<title>V3 256k Run Console</title>', f'<title>{a.title}</title>', 1)
    html = f'<!doctype html>\n<meta http-equiv="refresh" content="{int(a.every)}">\n' + html
    try:
        sd = dict(data, runs={'this': runs[k0]}, series={'this': series.get(k0, [])})
        i = html.index('<div class="wrap">'); html = html[:i] + H.static_block(sd) + html[i:]
        j = html.index('</header>', html.index('<div class="wrap">')); html = html[:j + 9] + H.failures_table(runs[k0]['failures']) + html[j + 9:]
        x, y = html.find('<!--TL_OLD-->'), html.find('<!--/TL_OLD-->')
        tl = H._try(lambda: H.timeline_html(runs[k0], runs[k0]['failures']), None)
        if 0 <= x < y and tl:
            html = html[:x] + tl + html[y + len('<!--/TL_OLD-->'):]
    except Exception as e:                         # the static copy must never stop the page
        html = html.replace('<div class="wrap">', f'<!-- static view failed: {type(e).__name__} -->\n<div class="wrap">', 1)
    if a.hide_ckpt_panel:                          # [2026-10-12, the user] no checkpoints-and-evaluations panel; hidden, not removed,
        html = html.replace('<div class="wrap">', '<style>section[aria-label="Checkpoints and evaluations"]{display:none!important}</style>\n<div class="wrap">', 1)
    tmp = os.path.join(out, 'index.html.tmp')
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(html)
    os.replace(tmp, os.path.join(out, 'index.html'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--job', action='append', required=True)
    ap.add_argument('--label', action='append'); ap.add_argument('--history', action='append'); ap.add_argument('--history-name', action='append')
    ap.add_argument('--logs', default=None, help='the logs/ folder of the history jobs (default: the parent of the first --job)')
    ap.add_argument('--evals-tsv'); ap.add_argument('--eval-logs', default=None)
    ap.add_argument('--base-run'); ap.add_argument('--base-label', default='the starting checkpoint')
    ap.add_argument('--eval-note', default=''); ap.add_argument('--phase-note', default='')
    ap.add_argument('--title', default='Run console'); ap.add_argument('--every', type=float, default=120); ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--hide-ckpt-panel', action='store_true', help='hide the checkpoints-and-evaluations panel (its markup stays: the script fills it)')
    ap.add_argument('--results-json', default=None, help='cells scored from the response cache (assistant_deep_dive/2026-10-08-lc256k/results/ruler256k_results.json)')
    a = ap.parse_args()
    try:
        os.nice(19)
    except OSError:
        pass
    try:
        subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], capture_output=True, timeout=10)
    except Exception:
        pass
    jobs, labels = kv(a.job), kv(a.label)
    a.hist_names = kv(a.history_name)
    out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    for jd in jobs.values():
        assert not os.path.abspath(out).startswith(os.path.abspath(jd)), 'the output folder must not be inside a job folder'
    logs = a.logs or os.path.dirname(os.path.abspath(list(jobs.values())[0]))
    a.eval_logs = a.eval_logs or os.path.join(os.path.dirname(logs), 'eval', 'logs')
    hpath = os.path.join(out, '.history.json')
    want = {k: v.split(',') for k, v in kv(a.history).items()}
    try:
        hist = json.load(open(hpath))
        assert hist.get('_want') == want
    except Exception:
        hist = {k: history_points(logs, ids) for k, ids in want.items()}; hist['_want'] = want
        json.dump(hist, open(hpath, 'w'))
    hist = {k: v for k, v in hist.items() if k != '_want'}
    if not a.once:
        handler = lambda *x, **y: http.server.SimpleHTTPRequestHandler(*x, directory=out, **y)
        http.server.SimpleHTTPRequestHandler.log_message = lambda *x: None
        srv = http.server.ThreadingHTTPServer(('127.0.0.1', a.port), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f'[watch] serving {out} on http://127.0.0.1:{a.port}/ (host {os.uname().nodename}); rebuilding every {a.every:.0f} s', flush=True)
    while True:
        t0 = time.time()
        try:
            build_once(a, jobs, labels, hist, out)
            print(f'[watch] {time.strftime("%H:%M:%S")} index.html rebuilt in {time.time() - t0:.1f} s', flush=True)
        except Exception as e:
            print(f'[watch] {time.strftime("%H:%M:%S")} rebuild failed: {type(e).__name__}: {e}', flush=True)
        if a.once:
            return
        time.sleep(max(10.0, a.every - (time.time() - t0)))


if __name__ == '__main__':
    main()
