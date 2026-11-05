#!/usr/bin/env python3
"""Login-node check of bin/wall_exit.py, the arithmetic of the trainer's wall-aware exit (2026-10-10). No torch, no Slurm:
  A. the end-time chain: SLURM_JOB_END_TIME first; then scontrol's EndTime (parsed from an injected text); then the old
     --exit-duration-in-mins turned into an end time; else none (the rule is off);
  B. the decision at the boundary of the user's example: step 2,584 s, save 68 s, margin 120 s -> cost 2,584 x 1.1 + 68 + 120 = 3,030 s;
     3,100 s left -> training continues; 1,300 s left -> exit; the slowest of the last 3 steps is used;
  C. the save cost: the estimate before any save; the synchronous part + the asynchronous part once measured; the seed from the job
     folder's save_times.tsv / save_async_times.tsv for an attempt that has not saved yet;
  D. the sentence carries every number of the arithmetic, the end time as HH:MM:SS, the source, the rank count and 'latest';
  E. the record line has the 15 columns of the header and round-trips through the restart loop's awk/cut (attempt and sentence)."""
import os, sys, tempfile, time, subprocess
BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
sys.path.insert(0, BIN)
import wall_exit as W

ok = True


def check(name, cond, extra=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else '  ' + str(extra)[:600])); ok &= bool(cond)


# A. the end-time chain
now = time.time()
t, src, note = W.resolve_end_time(env={'SLURM_JOB_END_TIME': str(int(now) + 5400), 'SLURM_JOB_ID': '1'}, job_id='1', train_start=now, exit_duration_min=44)
check('A1: SLURM_JOB_END_TIME wins', t == float(int(now) + 5400) and src == 'SLURM_JOB_END_TIME' and 'SLURM_JOB_END_TIME' in note, (t, src, note))
sc_text = 'JobId=5646336 JobName=res-x\n   RunTime=00:06:10 TimeLimit=00:40:00 TimeMin=N/A\n   SubmitTime=2026-10-09T23:38:00 EligibleTime=2026-10-09T23:38:00\n   StartTime=2026-10-09T23:39:32 EndTime=2026-10-10T00:19:32 Deadline=N/A\n'
exp = time.mktime(time.strptime('2026-10-10T00:19:32', '%Y-%m-%dT%H:%M:%S'))
check('A2: scontrol EndTime parses to the local epoch', W.parse_scontrol_endtime(sc_text) == exp, W.parse_scontrol_endtime(sc_text))
check('A2b: a pending job (EndTime=Unknown) gives None', W.parse_scontrol_endtime('StartTime=Unknown EndTime=Unknown') is None, '')
t, src, note = W.resolve_end_time(env={'SLURM_JOB_ID': '5646336'}, job_id='5646336', train_start=now, exit_duration_min=44, run=lambda cmd: sc_text if cmd[:3] == ['scontrol', 'show', 'job'] else '')
check('A3: without the variable, scontrol is the source', t == exp and src == 'scontrol show job EndTime' and '5646336' in note, (t, src, note))
t, src, note = W.resolve_end_time(env={}, job_id=None, train_start=now, exit_duration_min=44, run=lambda cmd: '')
check('A4: without Slurm, --exit-duration-in-mins is the fallback end time', abs(t - (now + 44 * 60)) < 1e-6 and src == '--exit-duration-in-mins' and 'fallback' in note, (t, src, note))
t, src, note = W.resolve_end_time(env={}, job_id=None, train_start=now, exit_duration_min=None, run=lambda cmd: '')
check('A5: nothing known -> None and the rule is off', t is None and src is None and 'off' in note, (t, src, note))

# B. the decision at the user's example
P = W.Predictor(env={})
check('B0: the defaults are 0.1, 120 s, 120 s', (P.margin_frac, P.margin_s, P.save_est_s) == (0.1, 120.0, 120.0), (P.margin_frac, P.margin_s, P.save_est_s))
for s in (2500.0, 2584.0, 2400.0):
    P.record_step(s)
P.record_save(sync_s=0.0, async_s=68.0)
end = 1_000_000.0
d = P.decide(now=end - 3100.0, end=end)
check('B1: 3,100 s left against a cost of 3,030 s -> training continues', d['done'] is False and abs(d['cost'] - 3030.4) < 0.01 and d['step_s'] == 2584.0, d)
check('B1b: the arithmetic text is the user\'s sentence', W.arithmetic(d) == '2,584 x 1.1 + 68 + 120 = 3,030 s', W.arithmetic(d))
d = P.decide(now=end - 1300.0, end=end)
check('B2: 1,300 s left -> exit', d['done'] is True and abs(d['remaining'] - 1300.0) < 1e-6, d)
P.record_step(100.0)   # the history keeps the last 3: 2584, 2400, 100 -> the slowest is still 2584
d = P.decide(now=end - 3100.0, end=end)
check('B3: the slowest of the last 3 iterations is the basis', d['step_s'] == 2584.0 and 'last 3' in d['step_basis'], d)
P.record_step(100.0); P.record_step(100.0)   # now 100, 100, 100
d = P.decide(now=end - 400.0, end=end)
check('B4: after 3 fast steps the cost falls (100 x 1.1 + 68 + 120 = 298) and 400 s left continues', d['done'] is False and abs(d['cost'] - 298.0) < 0.01, d)
d = P.decide(now=end - 290.0, end=end)
check('B5: 290 s left against 298 s -> exit', d['done'] is True, d)
d = W.Predictor(env={}).decide(now=end - 10.0, end=None)
check('B6: no end time -> never done', d['done'] is False and d['remaining'] is None and d['cost'] is None, d)
d = W.Predictor(env={}).decide(now=end - 10.0, end=end)
check('B7: before the first iteration there is no prediction and no exit', d['done'] is False and d['cost'] is None and 'no prediction' in W.arithmetic(d), d)

# C. the save cost
P = W.Predictor(env={'XMOE_SAVE_EST_S': '200', 'XMOE_WALL_MARGIN_S': '60', 'XMOE_WALL_MARGIN_FRAC': '0.25'})
check('C1: the environment sets the margins', (P.margin_frac, P.margin_s, P.save_est_s) == (0.25, 60.0, 200.0), (P.margin_frac, P.margin_s, P.save_est_s))
s, srcs = P.save_cost()
check('C2: before any save the estimate is used', s == 200.0 and 'XMOE_SAVE_EST_S' in srcs, (s, srcs))
P.record_save(sync_s=12.5)
s, srcs = P.save_cost()
check('C3: the synchronous part alone once measured', s == 12.5, (s, srcs))
P.record_save(async_s=68.6)
s, srcs = P.save_cost()
check('C4: synchronous + asynchronous parts', abs(s - 81.1) < 1e-9, (s, srcs))
T = tempfile.mkdtemp(prefix='wallexit_', dir=os.environ.get('TMPD', '/tmp'))
open(os.path.join(T, 'save_times.tsv'), 'w').write('2\t20.50\t2026-10-10T00:00:00\n4\t18.25\t2026-10-10T00:01:00\n')
open(os.path.join(T, 'save_async_times.tsv'), 'w').write('# header\nglobal_step2\t30.0\t2026-10-10T00:00:30\nglobal_step4\t25.5\t2026-10-10T00:01:30\n')
P = W.Predictor(env={}); P.seed_from_files(T)
s, srcs = P.save_cost()
check('C5: the seed from the job folder takes the LAST line of each file (18.25 + 25.5)', abs(s - 43.75) < 1e-9 and 'previous attempt' in srcs, (s, srcs))
check('C5b: a missing folder seeds nothing', W.read_save_files('/nonexistent/dir') == (None, None), W.read_save_files('/nonexistent/dir'))

# D. the sentence
P = W.Predictor(env={})
for s in (2500.0, 2584.0, 2400.0):
    P.record_step(s)
P.record_save(sync_s=0.0, async_s=68.0)
d = P.decide(now=end - 1300.0, end=end)
sent = W.exit_sentence(5, d, end, 'SLURM_JOB_END_TIME', 2048, 'global_step5', True, 71.2, True)
check('D1: the sentence has the step, the remaining seconds, the end time, the source and the arithmetic',
      sent.startswith('wall-aware exit at step 5: 1,300 s remain before the Slurm end time ' + W.hms(end)) and '(source SLURM_JOB_END_TIME)' in sent
      and '2,584 x 1.1 + 68 + 120 = 3,030 s' in sent, sent)
check('D2: the sentence says the save is complete on all 2,048 ranks and names latest, and ends with code 0',
      "the save of step 5 is complete on all 2048 ranks ('latest' names global_step5, 71 s after the save started)" in sent and sent.endswith('exiting with code 0'), sent)
sent2 = W.exit_sentence(5, d, end, 'SLURM_JOB_END_TIME', 16, 'global_step4', False, 300.0, False)
check('D3: an incomplete save is said so, with what latest names', 'is NOT complete after 300 s' in sent2 and "'latest' names global_step4" in sent2, sent2)
cont = W.continue_sentence(4, P.decide(now=end - 3100.0, end=end), end, 'SLURM_JOB_END_TIME')
check('D4: the continue line says training continues with the arithmetic', cont.startswith('wall-aware check at step 4: 3,100 s remain') and cont.endswith('training continues') and '= 3,030 s' in cont, cont)

# E. the record line and the restart loop's reading of it (awk by attempt, cut -f3 the step, cut -f15 the sentence)
line = W.record_line('2', 5, d, end, 'SLURM_JOB_END_TIME', 'global_step5', True, 71.2, sent)
cols = line.rstrip('\n').split('\t')
names = W.RECORD_HEADER.split('Columns:')[1].strip().rstrip('\n').split(', ')
check('E1: 15 columns, as many as the header names', len(cols) == 15 and len(names) == 15 and names[2] == 'step' and names[14] == 'sentence', (len(cols), names))
check('E2: the columns are plain numbers (no thousands separators) and yes/no', cols[3] == '1300' and cols[6] == '68' and cols[8] == '3030' and cols[12] == 'yes' and cols[1] == '2' and cols[2] == '5', cols)
p = os.path.join(T, 'wall_exit.tsv'); open(p, 'w').write(W.RECORD_HEADER + line)
awk = subprocess.run(['bash', '-c', f"awk -F'\\t' -v a=2 '!/^#/ && $2==a {{s=$0}} END{{print s}}' {p} | cut -f3,15"], capture_output=True, text=True).stdout.strip()
check('E3: the restart loop\'s awk/cut recovers the step and the sentence', awk.startswith('5\twall-aware exit at step 5:') and awk.endswith('exiting with code 0'), awk)
awk0 = subprocess.run(['bash', '-c', f"awk -F'\\t' -v a=0 '!/^#/ && $2==a {{s=$0}} END{{print s}}' {p}"], capture_output=True, text=True).stdout.strip()
check('E4: another attempt finds no line', awk0 == '', awk0)
check('E5: fmt and hms', W.fmt(1300) == '1,300' and W.fmt(2584.4) == '2,584' and W.fmt(None) == '-' and len(W.hms(end)) == 8, (W.fmt(1300), W.hms(end)))

print('ALL PASS' if ok else 'SOME FAILED')
sys.exit(0 if ok else 1)
