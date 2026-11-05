#!/usr/bin/env python3
"""RESULTS.md of a resilient job: one row per injected case with PASS/FAIL, the time to detect (the injection marker to the
srun return), the time to recover (the srun return to the first iteration of the next attempt), the node named and excluded,
and the evidence; plus, for a job with XMOE_GATE_INJECT, one row per injected gate fault with the judge's verdict. Reads
<job>/data/{failures.tsv, attempts.tsv, inject/*.fired, banners.txt, heartbeat_body.log, gate_inject.txt, gate/excluded_reasons.tsv}.
Writes <job>/RESULTS.md and <job>/data/results.tsv."""
import argparse, csv, glob, json, os, time

EXPECT = {'hang': 'collective_hang', 'exit': 'rank_died', 'kill': 'rank_died', 'oom': 'gpu_oom', 'fshang': 'filesystem_hang',
          'nan': 'loss_nan_spike', 'inithang': 'init_hang', 'slow': None,
          'hostmem': 'host_oom', 'realhostoom': 'host_oom'}   # [2026-10-11, sandbox] the 97 % guard's test reading and a real kernel OOM kill
EXCLUDES = {'collective_hang', 'rank_died', 'filesystem_hang', 'init_hang', 'driver_hip_error', 'node_lost', 'slow'}


def tsv(p):
    try:
        return list(csv.DictReader(open(p), delimiter='\t'))
    except Exception:
        return []


def hms(t):
    return time.strftime('%H:%M:%S', time.localtime(float(t))) if t not in (None, '', '-', 'PENDING') else '-'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--job-dir', required=True); a = ap.parse_args()
    J = a.job_dir; R = os.path.join(J, "data")
    fails, atts = tsv(os.path.join(R, 'failures.tsv')), tsv(os.path.join(R, 'attempts.tsv'))
    fired = sorted((json.load(open(f)) for f in glob.glob(os.path.join(R, 'inject', '*.fired'))), key=lambda d: d['time'])
    sched = os.environ.get('XMOE_INJECT_SCHEDULE', '') or open(os.path.join(R, 'schedule.txt')).read().strip() if os.path.exists(os.path.join(R, 'schedule.txt')) else os.environ.get('XMOE_INJECT_SCHEDULE', '')
    body = open(os.path.join(R, 'heartbeat_body.log')).read() if os.path.exists(os.path.join(R, 'heartbeat_body.log')) else ''
    rows = []
    for d in fired:
        case, att = d['case'], str(d['attempt'])
        f = next((x for x in fails if x['attempt'] == att and x['case'] == case), None)
        exp = EXPECT.get(case)
        if case == 'slow':
            named = f"rank {d['rank']} on {d['node']}" in body
            rows.append(dict(case=f"slow:{d['rank']}:{d['step_planned']}", node=d['node'], fired=f"step {d['step']} {hms(d['time'])}",
                             cls='(no exit by design: the slow-step flag + the aggregator slow_rank)', named=d['node'] if named else '-', excl='-',
                             det='-', rec='-', res='PASS' if named else 'FAIL',
                             ev='heartbeat.log "slowest rank" line names the rank' if named else 'no slowest-rank line named it'))
            continue
        if f is None:
            rows.append(dict(case=f"{case}:{d['rank']}:{d['step_planned']}", node=d['node'], fired=f"step {d['step']} {hms(d['time'])}", cls='-',
                             named='-', excl='-', det='-', rec='-', res='FAIL', ev='fired but no failure line for its attempt (job ended?)'))
            continue
        det = float(f['exit_time']) - float(d['time'])
        rec = (float(f['first_iter_after']) - float(f['exit_time'])) if f['first_iter_after'] not in ('PENDING', '-', '') else None
        ok_cls = f['class'] == exp
        ok_node = (f['node'] == d['node'])
        ok_excl = (f['class'] not in EXCLUDES) or ('excluded' in f['excluded'] and f['node'] in f['excluded'])
        verdict = 'PASS' if (ok_cls and ok_node and ok_excl and rec is not None) else \
                  ('DETECTED, NOT RECOVERED' if ok_cls and ok_node and rec is None else 'FAIL')
        why = []
        if not ok_cls: why.append(f'class {f["class"]} != expected {exp}')
        if not ok_node: why.append(f'node named {f["node"]} != injected {d["node"]}')
        if rec is None: why.append('no first iteration after the restart')
        rows.append(dict(case=f"{case}:{d['rank']}:{d['step_planned']}", node=d['node'], fired=f"step {d['step']} {hms(d['time'])}",
                         cls=f"{f['code']} {f['class']}", named=f"{f['node']} (rank {f['rank']})", excl=f['excluded'],
                         det=f'{det:.0f}', rec=('%.0f' % rec) if rec is not None else '-', res=verdict,
                         ev=('; '.join(why) + ' | ' if why else '') + f['evidence'][:220]))
    out = [f'# RESULTS: resilient job {os.path.basename(J)}', '',
           f'Written {time.strftime("%Y-%m-%d %H:%M:%S")} by results.py. Schedule: `{sched or "none"}`. '
           'Time to detect = from the injection marker to the srun return; time to recover = from the srun return to the first '
           'iteration line of the next attempt. Both are measured from the job folder\'s files, not estimated. Attempt k = '
           'attempts/attempt_k (attempt 0 is the first run). The machine-readable copy: data/results.tsv.', '',
           '| case (case:rank:step) | injected on | fired | detected class | node named | exclusion | detect s | recover s | result | evidence |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        out.append(f"| {r['case']} | {r['node']} | {r['fired']} | {r['cls']} | {r['named']} | {r['excl']} | {r['det']} | {r['rec']} | **{r['res']}** | {r['ev'].replace('|', '/')} |")
    planned = [x for x in sched.replace('+', ',').split(',') if x]
    fired_keys = {f"{d['case']}:{d['rank']}:{d['step_planned']}" for d in fired}
    notfired = [x for x in planned if x not in fired_keys]
    if notfired:
        out += ['', f'Not fired (the run ended before their step): {", ".join(notfired)}']
    out += ['', '## Attempts', '', '| attempt | nodes | start | first iteration | end | rc | class | node | rank | step | resumed from save |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for x in atts:
        out.append(f"| {x['attempt']} | {x['nodes']} | {hms(x['start'])} | {hms(x['first_iter'])} | {hms(x['end'])} | {x['rc']} | {x['code']} {x['class']} | {x['node']} | {x['rank']} | {x['step']} | {x['resumed_from']} |")
    other = [f for f in fails if not f['injected'] == 'yes']
    if other:
        out += ['', '## Failures without an injection (real or unexplained)', '']
        out += [f"- attempt {f['attempt']}: {f['code']} {f['class']} on {f['node']} rank {f['rank']}: {f['evidence'][:300]}" for f in other]
    # [2026-10-09 late evening] the gate's injected faults (XMOE_GATE_INJECT, test only): one row per case with the judge's verdict
    gi = os.path.join(R, 'gate_inject.txt')
    if os.path.exists(gi) and open(gi).read().strip():
        reasons = {}
        for r in tsv(os.path.join(R, 'gate', 'excluded_reasons.tsv')):
            reasons[r['node']] = r
        out += ['', '## The gate\'s injected faults (XMOE_GATE_INJECT)', '',
                'Expected: pair_silent -> excluded as dead by the pair test (named by two different partners); gemm_slow -> excluded as dead by the '
                'majority rule (a timeout while the others passed); gemm_half -> excluded as degraded (below the GEMM threshold), kept as a spare of last resort.', '',
                '| case | node | excluded | tier | test | the judge\'s sentence | result |', '|---|---|---|---|---|---|---|']
        for item in open(gi).read().strip().replace(',', '+').split('+'):
            if ':' not in item:
                continue
            case, node = item.split(':', 1)
            r = reasons.get(node)
            exp_test = {'pair_silent': 'pair', 'gemm_slow': 'gemm', 'gemm_half': 'gemm'}.get(case)
            exp_tier = {'pair_silent': 'dead', 'gemm_slow': 'dead', 'gemm_half': 'degraded'}.get(case)
            ok = r is not None and r['test'] == exp_test and r['tier'] == exp_tier
            out.append(f"| {case} | {node} | {'yes' if r else 'no'} | {r['tier'] if r else '-'} | {r['test'] if r else '-'} | {(r['sentence'][:300] if r else 'the node was not excluded').replace('|', '/')} | **{'PASS' if ok else 'FAIL'}** |")
    # [2026-10-10] the wall-aware exit (XMOE_WALL_AWARE_EXIT=1): one row per exit line of data/wall_exit.tsv with the five checks of
    # the test: exit code 0 as the planned exit, 'latest' names the step, the save complete on every rank, the SIGUSR1 handler did
    # not fire, and the srun returned before the job's end time
    we = os.path.join(R, 'wall_exit.tsv')
    if os.path.exists(we):
        loop = open(os.path.join(R, 'loop.log')).read() if os.path.exists(os.path.join(R, 'loop.log')) else ''
        handler_fired = 'the wall is near (SIGUSR1' in loop or 'timeout guard: the wall is near' in loop
        # [2026-10-10 02:30] the two ways the handler can fire are told apart (job 5646541: the trainer had decided and published its
        # save, but the processes did not end by themselves, so the handler stopped the step: a defect of the exit path, not of the rule)
        handler_txt = ('FIRED after the trainer had decided (the processes did not end by themselves: a defect)' if handler_fired and 'a defect to report' in loop
                       else 'FIRED (the trainer had not decided)' if handler_fired else 'did not fire')
        out += ['', '## The wall-aware exit (XMOE_WALL_AWARE_EXIT=1)', '',
                'The trainer must exit by itself, with code 0, after the last step whose successor and save would not fit before the Slurm end '
                'time, and only once the save of that step is complete on every rank. Columns: the step, the seconds that remained at the '
                'decision, the predicted cost of one more step (its arithmetic), what \'latest\' names, the save complete on all ranks, '
                'the attempt\'s exit, the SIGUSR1 handler, the seconds left when the srun returned, and the result.', '',
                '| step | remaining s | one more step would need | latest | save complete | attempt exit | USR1 handler | left at srun return s | result |',
                '|---|---|---|---|---|---|---|---|---|']
        for ln in open(we):
            if ln.startswith('#') or not ln.strip():
                continue
            c = ln.rstrip('\n').split('\t')
            att, step, rem, step_s, fac, save_s, mar, cost, end_ep, src, latest, complete = c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12]
            a = next((x for x in atts if x['attempt'] == att), None)
            planned = a is not None and a['class'] == 'planned_exit' and a['rc'] == '0'
            left = (float(end_ep) - float(a['end'])) if (a and end_ep not in ('-', '') and a['end'] not in ('-', '')) else None
            ok_latest = latest == f'global_step{step}'
            ok = planned and ok_latest and complete == 'yes' and not handler_fired and (left is None or left > 0)
            out.append(f"| {step} | {rem} | {step_s} x {float(fac):.1f} + {save_s} + {mar} = {cost} s | {latest} | {complete} | "
                       f"{('planned exit (rc 0)' if planned else (a['class'] + ' rc ' + a['rc'] if a else 'no attempt line'))} | "
                       f"{handler_txt} | {('%.0f' % left) if left is not None else '-'} | **{'PASS' if ok else 'FAIL'}** |")
        out += ['', f'End time source: {src}. The sentence of the exit: {c[14] if len(c) > 14 else "-"}']
    # [2026-10-10] the UCP worker duty (AUTORUN_UCP_WORKERS): one row per conversion of data/ucp_conversions.tsv with the completeness
    # checks, then the retention's lines of loop.log, the STATUS.md conversion line, and the step time of the steps without a save
    uc = os.path.join(R, 'ucp_conversions.tsv')
    if os.path.exists(uc):
        rows_u = [l.rstrip('\n').split('\t') for l in open(uc) if l.strip() and not l.startswith('#') and not l.startswith('step\t')]
        ck = None
        try:
            st = dict(l.strip().partition('=')[::2] for l in open(os.path.join(R, 'state')) if '=' in l)
        except Exception:
            st = {}
        loop = open(os.path.join(R, 'loop.log')).read() if os.path.exists(os.path.join(R, 'loop.log')) else ''
        out += ['', '## The UCP worker duty (AUTORUN_UCP_WORKERS)', '',
                'The worker nodes convert every complete save to the universal form inside the job, one conversion at a time; a conversion '
                'PASSES when its universal is complete (atoms > 0, zero/optimizer_state.pt, no tmp folder, the finalize lock) and its record '
                'says so. Columns: the step, the start and end times, the seconds, the workers, the share exit codes, the atoms, the result.', '',
                '| step | start | end | seconds | workers | share exit codes | atoms | result |', '|---|---|---|---|---|---|---|---|']
        n_ok = 0
        for c in rows_u:
            if len(c) < 9:
                continue
            ok = c[8] == 'complete' and c[7].isdigit() and int(c[7]) > 0
            n_ok += ok
            out.append(f"| {c[0]} | {c[1][11:]} | {c[2][11:]} | {c[3]} | {c[4]} ({c[5]}) | {c[6]} | {c[7]} | **{'PASS' if ok else 'FAIL'}** ({c[8]}) |")
        ret = [l.strip() for l in loop.split('\n') if 'ucp: retention:' in l]
        skipped = [l.strip() for l in loop.split('\n') if 'skipped for now' in l]
        status_line = ''
        try:
            status_line = next((l for l in open(os.path.join(J, 'STATUS.md')) if l.startswith('conversion')), '').strip()
        except Exception:
            pass
        # the step time of the steps without a save, from training.log (the user's reference for 16B_4L: 0.44-0.55 s)
        steps = []
        try:
            for l in open(os.path.join(J, 'training.log')):
                v = [x.strip() for x in l.split('|')]
                if len(v) >= 8 and v[2].isdigit() and 'save' not in v[7] and 'first step' not in v[7]:
                    steps.append(float(v[5]))
        except Exception:
            pass
        med = sorted(steps)[len(steps) // 2] if steps else None
        out += ['', f'{n_ok} of {len(rows_u)} conversion(s) complete ({"at least two: PASS" if n_ok >= 2 else "fewer than two: FAIL"}).',
                f'Retention lines in loop.log: {len(ret)}' + (' (' + ' / '.join(r.split(" ucp: ", 1)[-1][:160] for r in ret[:4]) + ')' if ret else ' (none: no ZeRO save was deleted)') + '.',
                f'Saves skipped while the converter was busy: {len(skipped)} line(s) in loop.log.',
                f'STATUS.md conversion line: {status_line or "absent"}.',
                (f'The step time of the {len(steps)} steps without a save: median {med:.2f} s, min {min(steps):.2f}, max {max(steps):.2f} (the reference of the earlier 16B_4L runs is 0.44-0.55 s).' if steps else 'No step without a save was logged.')]
    # [2026-10-10, round 6] the OOM re-configuration (AUTORUN_OOM_RECONFIG=1) and the memory gate: the gate's verdict, one row per decision of
    # the rule (data/oom_reconfig.tsv), the layouts of every attempt with the predicted and the measured peaks (data/attempt_layouts.tsv,
    # data/mem_peaks_a<k>.tsv), the conversion time, and the pass checks of the test (the user's definition of 2026-10-10)
    og = os.path.join(R, 'oom_reconfig.tsv'); mg = os.path.join(R, 'mem_gate.txt')
    if os.path.exists(og) or os.path.exists(mg):
        gate_log = open(os.path.join(J, 'gate.log')).read() if os.path.exists(os.path.join(J, 'gate.log')) else ''
        rec = open(os.path.join(J, 'recovery.log')).read() if os.path.exists(os.path.join(J, 'recovery.log')) else ''
        warn = next((l.strip() for l in gate_log.split('\n') if l.startswith('WARNING (the memory gate)')), '')
        try:   # the file opens with a '#' comment line, then the header
            lays = list(csv.DictReader([l for l in open(os.path.join(R, 'attempt_layouts.tsv')) if not l.startswith('#')], delimiter='\t'))
        except Exception:
            lays = []
        decs = [l.rstrip('\n').split('\t') for l in open(og) if l.strip() and not l.startswith('#') and not l.startswith('attempt\t')] if os.path.exists(og) else []
        out += ['', '## The OOM re-configuration (AUTORUN_OOM_RECONFIG=1) and the memory gate', '',
                'The memory gate predicts every stage\'s peak with the memory model and reads the layout heatmap\'s colour before the first attempt; a red layout '
                'gets a WARNING in gate.log and the run proceeds. Upon exit code 11 the rule takes one rung of the fixed ladder (micro-batch 1; d aligned to EP with the '
                'fewest-bytes split; EP and d doubled; PP doubled; then stop with mail), loads the universal checkpoint of \'latest\' when the parallelism changed, and '
                'writes its block into recovery.log; the memory profiler\'s per-rank peak of every attempt stands next to the prediction.', '',
                f'The memory gate\'s warning in gate.log: {("`" + warn + "`") if warn else "none (no WARNING line)"}.', '',
                '| attempt | layout | loads | predicted per stage (allocated / VRAM / line, GiB) | measured peak (GiB, rank, stage) | first iteration | end | class |',
                '|---|---|---|---|---|---|---|---|']
        def peak_of(att):
            p = os.path.join(R, f'mem_peaks_a{att}.tsv')
            try:
                rows_p = [l.rstrip('\n').split('\t') for l in open(p) if l.strip() and not l.startswith('#') and not l.startswith('rank\t')]
                b = max(rows_p, key=lambda c: float(c[2]))
                return f'{float(b[2]):.1f} on rank {b[0]} (stage {b[1]}; {b[4].split(" (")[0]})'
            except Exception:
                return '-'
        for l in lays:
            a = next((x for x in atts if x['attempt'] == l['attempt']), None)
            out.append(f"| {l['attempt']} | {l['text']} | {l['loads']} | {l['pred'].replace(';', '; ')} | {peak_of(l['attempt'])} | {hms(a['first_iter']) if a else '-'} | {hms(a['end']) if a else '-'} | {(a['code'] + ' ' + a['class']) if a else '(running or cut by the wall)'} |")
        if decs:
            out += ['', '| failed attempt | OOM at step / stage / rank / HBM GB | rule step | old layout | new layout | bytes GiB per step (across nodes) | measured peak of the failed attempt | universal step | waited s | decided at | decision s |',
                    '|---|---|---|---|---|---|---|---|---|---|---|']
            for c in decs:
                if len(c) < 19: continue
                out.append(f"| {c[0]} | {c[1]} / {c[2]} / {c[3]} / {c[4]} | {c[6]} | {c[7]} | {c[8]} | {c[9]} ({c[10]}) | {c[12]} GiB | {c[15]} | {c[16]} | {c[17][11:]} | {c[18]} |")
        # the pass checks of the user's test (2026-10-10): the warning; the first attempt ends with code 11; the rule's block; the universal or the initial
        # state; the next attempt runs with the rule's layout and reaches two steps with the measured peak under the line
        a0 = next((x for x in atts if x['attempt'] == '0'), None)
        c1 = bool(warn); c2 = a0 is not None and a0['code'] == '11'
        c3 = 'the rule:    upon the out of memory' in rec
        c4 = bool(decs) and (decs[0][15] != '-' or 'no save exists yet' in rec)
        a1 = next((x for x in atts if x['attempt'] == '1'), None); l1 = next((x for x in lays if x['attempt'] == '1'), None)
        steps1 = []
        try:
            for l in open(os.path.join(J, 'training.log')):
                v = [x.strip() for x in l.split('|')]
                if len(v) >= 7 and v[1] == '1' and v[2].isdigit(): steps1.append(v)
        except Exception:
            pass
        pk1 = peak_of('1'); line1 = None
        try:
            line1 = min(float(x.split('/')[2]) for x in l1['pred'].split(';')) if l1 else None
            pk1v = float(pk1.split(' ')[0]) if pk1 != '-' else None
        except Exception:
            pk1v = None
        c5 = l1 is not None and decs and l1['text'] == decs[0][8] and len(steps1) >= 2 and (pk1v is not None and line1 is not None and pk1v < 64.0)
        out += ['', 'The pass checks:',
                f'- gate.log carries the memory gate\'s warning: **{"PASS" if c1 else "FAIL"}**',
                f'- the first attempt ended with code 11 (gpu_oom): **{"PASS" if c2 else "FAIL"}** ({(a0["code"] + " " + a0["class"]) if a0 else "no attempt 0 line"})',
                f'- recovery.log carries the rule\'s block ("the rule: upon the out of memory ..."): **{"PASS" if c3 else "FAIL"}**',
                f'- the universal checkpoint was converted by the workers, or the restart loaded the initial state when no save existed: **{"PASS" if c4 else "FAIL"}** ({("universal of step " + decs[0][15] + ", waited " + decs[0][16] + " s") if decs and decs[0][15] != "-" else ("no save existed: the initial state" if c4 else "-")})',
                f'- attempt 1 ran with the rule\'s layout and reached at least 2 steps with the profiler\'s peak under the 64 GiB of a GCD: **{"PASS" if c5 else "FAIL"}** '
                f'(layout {l1["text"] if l1 else "-"}; {len(steps1)} step line(s) in training.log; measured peak {pk1}; the model\'s line {("%.1f" % line1) if line1 else "-"})']
    try:
        out += ['', '## The banners (first lines of heartbeat.log and a-monitor.txt)', '', '```'] + \
               [l.rstrip() for l in open(os.path.join(R, 'banners.txt'))] + ['```']
    except Exception:
        out += ['', 'No RECOVERED banner was written.']
    with open(os.path.join(R, 'results.tsv'), 'w') as fh:
        keys = ['case', 'node', 'fired', 'cls', 'named', 'excl', 'det', 'rec', 'res', 'ev']
        fh.write('\t'.join(keys) + '\n')
        for r in rows:
            fh.write('\t'.join(str(r[k]).replace('\t', ' ') for k in keys) + '\n')
    open(os.path.join(J, 'RESULTS.md'), 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
