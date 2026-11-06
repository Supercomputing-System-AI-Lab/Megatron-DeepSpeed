#!/usr/bin/env python3
"""rc_report.py (ft-sys-v2 recovery, 2026-11-06): what a recovery run left, as tables.

    python rc_report.py losses <arm dir> [<arm dir> ...]   loss, grad norm and ms per iteration of each arm, side by side, with the
                                                           relative loss difference of each arm to the first one
    python rc_report.py throughput <arm dir> [...]         TFLOP/s per GCD and ms per iteration after 8 warm-up iterations
    python rc_report.py keep <arm dir>                     Keep's per-step times (sums over the groups, the median over ranks)
    python rc_report.py final <arm dir A> <arm dir B>      the end-of-run fp32 masters rank by rank (md5, sampled difference)
    python rc_report.py inits <arm dir> <rank log name>     RCCL communicator setups in one rank's log, with their times
    python rc_report.py restore <arm dir>                  restored optimizer state vs the same rank's state at that step end
    python rc_report.py recovery <arm dir>                 each recovery's phases, seconds after the failure list was written
                                                           (median and max over ranks), and each rank's outcome
An arm dir is <job dir>/<arm> (it holds rank_<r>.log, standby_<l>.log and ft/). The iteration lines come from Megatron's last rank:
rank_<W-1>.log, or after a join the standby log of that rank (standby_<l>.log); a later line of the same iteration wins (a redo)."""
import glob
import json
import os
import re
import statistics
import sys

IT = re.compile(r'iteration\s+(\d+)/\s*(\d+) \|.*?elapsed time per iteration \(ms\): ([\d.]+).*?lm loss: ([\d.E+-]+)'
                r'(?:.*?grad norm: ([\d.]+))?')


def iterations(arm):
    files = glob.glob(os.path.join(arm, 'rank_*.log')) + glob.glob(os.path.join(arm, 'standby_*.log'))
    out = {}
    for f in sorted(files, key=os.path.getmtime):
        try:
            text = open(f, errors='replace').read()
        except OSError:
            continue
        for m in IT.finditer(text):
            it = int(m.group(1))
            tf = re.search(r'TFLOPs: ([\d.]+)', text[m.start():m.start() + 2000].split('\n')[0])
            out[it] = {'loss': float(m.group(4)), 'ms': float(m.group(3)), 'gn': float(m.group(5)) if m.group(5) else None,
                       'tflops': float(tf.group(1)) if tf else None, 'file': os.path.basename(f)}
    return out


def losses(arms):
    data = [iterations(a) for a in arms]
    its = sorted(set().union(*[d.keys() for d in data]))
    head = ['iter'] + [f'{os.path.basename(a.rstrip("/"))} loss' for a in arms] + \
        [f'rel diff {os.path.basename(a.rstrip("/"))}' for a in arms[1:]] + [f'gn {os.path.basename(a.rstrip("/"))}' for a in arms] + \
        [f'ms {os.path.basename(a.rstrip("/"))}' for a in arms]
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    worst = [0.0] * (len(arms) - 1)
    for it in its:
        row = [str(it)]
        ls = [d.get(it, {}).get('loss') for d in data]
        row += [f'{x:.6f}' if x is not None else '-' for x in ls]
        for k in range(1, len(arms)):
            if ls[0] is not None and ls[k] is not None:
                r = abs(ls[k] - ls[0]) / abs(ls[0])
                worst[k - 1] = max(worst[k - 1], r)
                row.append(f'{r:.2e}')
            else:
                row.append('-')
        row += [f'{d.get(it, {}).get("gn"):.4f}' if d.get(it, {}).get('gn') is not None else '-' for d in data]
        row += [f'{d.get(it, {}).get("ms"):.0f}' if d.get(it) else '-' for d in data]
        print('| ' + ' | '.join(row) + ' |')
    for k in range(1, len(arms)):
        print(f'largest relative loss difference, {os.path.basename(arms[k].rstrip("/"))} vs {os.path.basename(arms[0].rstrip("/"))}: '
              f'{worst[k - 1]:.3e}')


def throughput(arms, warmup=8):
    """Model TFLOP/s per GCD (Megatron's line) and ms per iteration, mean +/- std over the iterations after `warmup`."""
    print(f'| arm | iterations averaged | TFLOP/s per GCD (mean +/- std) | share of 191.5 peak | ms per iteration (mean +/- std) |')
    print('|---|---|---|---|---|')
    for a in arms:
        d = iterations(a)
        its = [i for i in sorted(d) if i > warmup]
        tf = [d[i]['tflops'] for i in its if d[i]['tflops'] is not None]
        ms = [d[i]['ms'] for i in its]
        if not its:
            continue
        sd = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0
        print(f'| {os.path.basename(a.rstrip("/"))} | {its[0]}-{its[-1]} ({len(its)}) | '
              f'{statistics.mean(tf):.2f} +/- {sd(tf):.2f} | {100 * statistics.mean(tf) / 191.5:.1f}% | '
              f'{statistics.mean(ms):.0f} +/- {sd(ms):.0f} |' if tf else f'| {a} | - | - | - | - |')


def keep(arm):
    rows = {}
    for f in glob.glob(os.path.join(arm, 'ft', 'keep', 'rank_*.jsonl')):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            rows.setdefault(r['step'], []).append(r)
    keys = ['p2p_ms', 'own_adam_ms', 'd2h_ms', 'cpu_adam_ms', 'shadow_ms', 'wait_ms']
    print('| step | ranks | ' + ' | '.join(f'{k} (median / max)' for k in keys) + ' | check max abs |')
    print('|' + '---|' * (len(keys) + 3))
    for s in sorted(rows):
        rs = rows[s]
        cells = []
        for k in keys:
            v = [r['sum'].get(k, 0) for r in rs]
            cells.append(f'{statistics.median(v):.1f} / {max(v):.1f}' if any(v) else '-')
        chk = [r.get('check_max_abs') for r in rs if r.get('check_max_abs') is not None]
        print(f'| {s} | {len(rs)} | ' + ' | '.join(cells) + f' | {max(chk):.3e} |' if chk else f'| {s} | {len(rs)} | ' + ' | '.join(cells) + ' | - |')


PHASES = ['list_seen', 'revoke_done', 'barrier_done', 'abort_done', 'main_caught', 'reported', 'released', 'reports_in', 'world_up',
          'parallel_state_done', 'model_built', 'state_moved', 'restore_done', 'data_built', 'reseed_done', 'ready', 'first_step_done']


def recovery(arm):
    for gdir in sorted(glob.glob(os.path.join(arm, 'ft', 'recovery', '*'))):
        recs = []
        for f in glob.glob(os.path.join(gdir, 'rank_*.json')):
            try:
                recs.append(json.load(open(f)))
            except ValueError:
                pass
        if not recs:
            continue
        t0s = [r['times'].get('list_written') for r in recs if r['times'].get('list_written')]
        t0 = min(t0s) if t0s else None
        print(f'\n### recovery {os.path.basename(gdir)}: {len(recs)} rank records; list written at {t0}')
        ex = sorted({(r.get('exit'), r.get('code')) for r in recs if r.get('exit')}, key=str)
        if ex:
            print(f'exits: {ex}')
        done = [r for r in recs if r.get('resume_step') is not None]
        if done:
            r0 = done[0]
            print(f'resume step {r0["resume_step"]}; kind {r0.get("kind")}; node {r0.get("node")} -> standby {r0.get("standby")}; '
                  f'lost {r0.get("lost")}; state from: {sorted({r.get("state_from") for r in done})}; rng equal: {r0.get("rng_same")}')
        print('| phase | ranks | median s after the list | max s |')
        print('|---|---|---|---|')
        for p in PHASES:
            v = [r['times'][p] - t0 for r in recs if t0 and r['times'].get(p)]
            if v:
                print(f'| {p} | {len(v)} | {statistics.median(v):.2f} | {max(v):.2f} |')
        sent = {r['checksums']['sent_to']: r['checksums']['sent'] for r in recs if (r.get('checksums') or {}).get('sent')}
        got = {r['rank']: r['checksums']['received'] for r in recs if (r.get('checksums') or {}).get('received')}
        if sent or got:
            same = [o for o in sent if got.get(o) == sent[o]]
            print(f'shadow transfer: {len(same)} of {len(sent)} joiners received exactly what their shadow holder sent '
                  f'(integer checksums of fp32, exp_avg, exp_avg_sq per group); joiners reporting: {len(got)}')
        st = [r.get('stop', {}) for r in recs if r.get('stop')]
        if st:
            rv = [s['revoke']['s'] for s in st if 'revoke' in s]
            ab = [s['abort']['s'] for s in st if 'abort' in s]
            n = [s['revoke']['revoked'] for s in st if 'revoke' in s]
            print(f'revoke: {statistics.median(rv):.3f} s median over {len(rv)} ranks, communicators per rank {min(n)}-{max(n)}; '
                  f'abort {statistics.median(ab):.3f} s median, results {sorted({s["abort"]["abort"] for s in st if "abort" in s})}')


def final(arms):
    """The end-of-run fp32 masters of two arms, rank by rank: md5 equal? the largest difference on the sample."""
    import torch
    a, b = arms[0], arms[1]
    print(f'| rank | groups | md5 equal | max abs diff (sample) | max abs value | rel sum diff |')
    print('|---|---|---|---|---|---|')
    worst, eq = 0.0, 0
    files = sorted(glob.glob(os.path.join(a, 'ft', 'final', 'rank_*.pt')), key=lambda f: int(re.findall(r'(\d+)', os.path.basename(f))[0]))
    for f in files:
        g = os.path.join(b, 'ft', 'final', os.path.basename(f))
        if not os.path.exists(g):
            continue
        x, y = torch.load(f), torch.load(g)
        same = all(p['md5'] == q['md5'] for p, q in zip(x['groups'], y['groups']))
        eq += same
        d = max(float((p['sample'] - q['sample']).abs().max()) for p, q in zip(x['groups'], y['groups']))
        m = max(float(p['sample'].abs().max()) for p in x['groups'])
        rs = max(abs(p['sum'] - q['sum']) / max(abs(p['sum']), 1e-30) for p, q in zip(x['groups'], y['groups']))
        worst = max(worst, d)
        print(f'| {x["rank"]} | {len(x["groups"])} | {same} | {d:.3e} | {m:.3e} | {rs:.2e} |')
    print(f'ranks with every md5 equal: {eq} of {len(files)}; largest sampled difference {worst:.3e}')


def inits(arm, rank_file):
    """RCCL communicator setups in one rank's log (needs NCCL_DEBUG=INFO, SUBSYS=INIT, timestamps on every level): for each
    communicator its size, Init START and COMPLETE times; and per second, how many were set up."""
    import datetime as dt
    f = os.path.join(arm, rank_file)
    pat = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]]*\].*comm (0x[0-9a-f]+) rank (\d+) nranks (\d+).* - Init (START|COMPLETE)')
    rows = {}
    for ln in open(f, errors='replace'):
        m = pat.search(ln)
        if m:
            t = dt.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
            r = rows.setdefault(m.group(2), {'nranks': int(m.group(4))})
            r[m.group(5)] = t
    print(f'| comm | nranks | START | COMPLETE | seconds |')
    print('|---|---|---|---|---|')
    for c, r in sorted(rows.items(), key=lambda kv: kv[1].get('START', dt.datetime.min)):
        d = (r['COMPLETE'] - r['START']).total_seconds() if 'COMPLETE' in r and 'START' in r else None
        print(f'| {c} | {r["nranks"]} | {r.get("START")} | {r.get("COMPLETE")} | {d} |')


def restore(arm):
    """M1-B's exactness: each rank's restored optimizer state (fp32 master, exp_avg, exp_avg_sq per group) against that rank's own
    state at the same step end, written before the failure by the process that held it (a lost rank's dead process included)."""
    import torch
    d = os.path.join(arm, 'ft', 'sums')
    print('| rank | restored from | version | groups | fp32 equal | exp_avg equal | exp_avg_sq equal | max abs diff fp32 / m / v (sample) |')
    print('|---|---|---|---|---|---|---|---|')
    tot = {'ranks': 0, 'exact': 0}
    for f in sorted(glob.glob(os.path.join(d, 'rank_*_restored_*.pt')), key=lambda x: int(re.findall(r'rank_(\d+)_', x)[0])):
        r = torch.load(f)
        v = r['tag'].rsplit('_v', 1)[1]
        g = os.path.join(d, f'rank_{r["rank"]}_v{v}.pt')
        if not os.path.exists(g):
            print(f'| {r["rank"]} | {r.get("from")} | {v} | - | no step-end record | | | |')
            continue
        o = torch.load(g)
        eq = [all(a[j] == b[j] for a, b in zip(r['sums'], o['sums'])) for j in range(3)]
        dif = [max(float((a[j] - b[j]).abs().max()) for a, b in zip(r['samples'], o['samples'])) for j in range(3)]
        tot['ranks'] += 1
        tot['exact'] += all(eq)
        print(f'| {r["rank"]} | {r.get("from")} | {v} | {len(r["sums"])} | {eq[0]} | {eq[1]} | {eq[2]} | '
              f'{dif[0]:.2e} / {dif[1]:.2e} / {dif[2]:.2e} |')
    print(f'ranks restored bit for bit (every group, all three tensors): {tot["exact"]} of {tot["ranks"]}')


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    cmd, args = sys.argv[1], sys.argv[2:]
    {'losses': lambda: losses(args), 'keep': lambda: keep(args[0]), 'recovery': lambda: recovery(args[0]),
     'final': lambda: final(args), 'throughput': lambda: throughput(args), 'inits': lambda: inits(args[0], args[1]),
     'restore': lambda: restore(args[0])}[cmd]()


if __name__ == '__main__':
    main()
