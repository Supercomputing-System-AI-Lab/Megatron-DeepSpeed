#!/usr/bin/env python3
"""perf_pair.py (2026-10-11): the in-job page's cost, from the two perf16 arms' own logs (read-only).
  perf_pair.py <job dir ON> <job dir OFF>
THRESHOLD (written before the runs, run_matrix.sh perf16): (a) within the ON arm, the median step time of the steps that overlap a page
rewrite (data/dashboard_times.tsv, padded by 1 s for training.log's 1-s timestamps) is within 1 % of the other steps' median; (b) the two arms'
mean TFLOP/s per GCD over the steps after 8 warm-up steps, saves and the step after a save excluded, differ by less than 2 %."""
import sys, statistics as st, time, os
def steps(jd):
    out = []
    for l in open(os.path.join(jd, 'training.log')):
        if l.startswith('#') or l.startswith('time') or '|' not in l: continue
        f = [x.strip() for x in l.split('|')]
        try:
            t = time.mktime(time.strptime(f[0], '%Y-%m-%d %H:%M:%S')); step = int(f[2]); tf = float(f[4]); s = float(f[5])
        except ValueError:
            continue
        out.append(dict(t=t, step=step, tflops=tf, s=s, note=f[7] if len(f) > 7 else ''))
    return out
def clean(ss):
    save_steps = {p['step'] for p in ss if 'save of step' in p['note']}
    return [p for p in ss if p['step'] > 8 and p['step'] not in save_steps and (p['step'] - 1) not in save_steps]
on, off = sys.argv[1], sys.argv[2]
A, B = clean(steps(on)), clean(steps(off))
ma, mb = st.mean(p['tflops'] for p in A), st.mean(p['tflops'] for p in B)
print(f'ON  {os.path.basename(on)[:12]}: {len(A)} steps, TFLOP/s per GCD mean {ma:.2f} +/- {st.pstdev([p["tflops"] for p in A]):.2f} ({ma / 191.5 * 100:.1f} % of 191.5), median s/step {st.median(p["s"] for p in A):.3f}')
print(f'OFF {os.path.basename(off)[:12]}: {len(B)} steps, TFLOP/s per GCD mean {mb:.2f} +/- {st.pstdev([p["tflops"] for p in B]):.2f} ({mb / 191.5 * 100:.1f} % of 191.5), median s/step {st.median(p["s"] for p in B):.3f}')
d = (ma - mb) / mb * 100
print(f'(b) ON vs OFF mean TFLOP/s: {d:+.2f} %  -> {"PASS" if abs(d) < 2 else "FAIL"} (threshold 2 %)')
R = []
for l in open(os.path.join(on, 'data', 'dashboard_times.tsv')):
    v = l.split()
    try: R.append((float(v[0]), float(v[1])))
    except (ValueError, IndexError): pass
ov = [p for p in A if any(r0 <= p['t'] + 1 and r1 >= p['t'] - p['s'] - 1 for r0, r1 in R)]
rest = [p for p in A if p not in ov]
mo, mr = st.median(p['s'] for p in ov), st.median(p['s'] for p in rest)
print(f'page rewrites in the ON arm: {len(R)} (median duration {st.median(r1 - r0 for r0, r1 in R):.2f} s); steps overlapping a rewrite: {len(ov)}, others: {len(rest)}')
print(f'(a) median s/step overlapping {mo:.3f} vs others {mr:.3f}: {(mo - mr) / mr * 100:+.2f} %  -> {"PASS" if abs(mo - mr) / mr < 0.01 else "FAIL"} (threshold 1 %)')
