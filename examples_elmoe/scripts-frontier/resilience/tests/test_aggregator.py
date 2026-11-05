#!/usr/bin/env python3
"""Login-node unit check of hb_aggregator.py with synthetic heartbeat lines: 16 ranks on 2 nodes (2 groups of 8).
Case A (quiet): all ranks move. Case B (hang): rank 5 stays at collective 100 while ranks 0-4,6,7 are at 101 for 40 s ->
the minority alarm names rank 5 and its node. Case C (dead rank): rank 12's lines stop -> stillness names rank 12, but only at
the SECOND beat (the two-beat rule of 2026-10-10: the first beat is "stillness pending", no alarm, no stack sample).
Case D (slow): rank 7 busy 5.0 s vs 2.0 for its stage mates -> slowest rank names rank 7. Also checks the banner compose.
Case E (delivery lag, 2026-10-10 00:45): every rank of nodeB is equally 35 s old and nodeB's file itself is 35 s old -> the beat
block says "delivery lag 35 s on 1 node (Lustre), not a rank", no ALARM on any beat, no stack sample; after three beats the line
says the lag looks like a dead writer or a lost node. Case F: the same ages but a FRESH node file (the writer appends, the ranks
do not write) -> that is a stillness of the ranks, named at the second beat."""
import os, sys, time, json, tempfile, shutil, subprocess
BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
sys.path.insert(0, BIN)
import hb_aggregator as A

F = A.F
def ln(t, node, r, step, phase, cs, st='3.00', busy='2.00', ce=0, att='1'):
    return '\t'.join([time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t)), node, str(r), str(step), phase, str(cs), st, busy,
                      '10.00', '40.0', str(ce), '0', '0', 'R', '2.5', att, str(1000 + r), '-', '%.1f' % t]) + '\n'

def run(case, beats=1):
    job = tempfile.mkdtemp(prefix='aggtest_', dir=os.environ.get('TMPD', '/tmp'))
    os.makedirs(job + '/data/hb')
    open(job + '/data/state', 'w').write('attempt=1\nstatus=running\nworld=16\ngroup=8\npp=1\nexcluded=\ntrain_nodes=nodeA,nodeB\nall_nodes=nodeA,nodeB,nodeC\nduty=nodeC\nnext=watching attempt 1\n')
    now = time.time()
    with open(job + '/data/hb/nodeA.tsv', 'w') as a, open(job + '/data/hb/nodeB.tsv', 'w') as b:
        for k, t in enumerate([now - 50, now - 40, now - 30, now - 20, now - 10, now - 1]):
            for r in range(16):
                fh = a if r < 8 else b
                node = 'nodeA' if r < 8 else 'nodeB'
                if case == 'quiet':
                    fh.write(ln(t, node, r, 3 + k, 'forward', 100 + 10 * k))
                elif case == 'hang':
                    cs = 100 if r == 5 else (101 if r < 8 else 150)
                    fh.write(ln(t, node, r, 3, 'forward' if r == 5 else 'all_to_all', cs))
                elif case == 'dead':
                    if r == 12 and t > now - 45:
                        continue
                    fh.write(ln(t, node, r, 3 + k, 'forward', 100 + 10 * k))
                elif case == 'slow':
                    fh.write(ln(t, node, r, 3 + k, 'forward', 100 + 10 * k, busy='5.00' if r == 7 else '2.00'))
                elif case == 'ecc':
                    fh.write(ln(t, node, r, 3 + k, 'forward', 100 + 10 * k, ce=(k if r == 9 else 0)))
    class O: pass
    o = O(); o.job = o.job_dir = job; o.world = 16; o.group = 8; o.pp = 1; o.interval = 10; o.still = 30; o.pred = 0
    os.environ['SLURM_JOB_ID'] = ''
    g = A.Agg(o)
    # pretend the 'changed' times are the line times (the reader sets them at first sight = the line epoch)
    g.beat()
    for _ in range(beats - 1):
        g.beat()
    if case == 'ecc':                                  # a second beat: rank 9's correctable count grows by 2
        with open(job + '/data/hb/nodeB.tsv', 'a') as b:
            b.write(ln(time.time(), 'nodeB', 9, 9, 'forward', 170, ce=7))
            for r in range(8, 16):
                if r != 9: b.write(ln(time.time(), 'nodeB', r, 9, 'forward', 170, ce=0))
        g.beat()
    body = open(job + '/heartbeat.log').read()
    last = json.load(open(job + '/data/agg_last.json'))
    return job, body, last

ok = True
def check(name, cond, extra=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else '  ' + extra))
    ok &= bool(cond)

job, body, last = run('quiet')
check('quiet: no ALARM', 'ALARM' not in body, body)
check('quiet: no heatmap.png (drawn only on an alarm or at the end of an attempt)', not os.path.exists(job + '/heatmap.png'), '')
check('quiet: five fixed lines', all(s in body for s in ('beat ', 'where the ranks are', 'step time', 'slowest rank', 'counters since last beat', 'host memory')), body)
job, body, last = run('hang')
check('hang: minority names rank 5', last['minority'] == [5], str(last['minority']) + body)
check('hang: ALARM line names nodeA', 'ALARM minority' in body and 'nodeA' in body, body)
st = open(job + '/STATUS.md').read()
al = open(job + '/alarms.log').read()
check('hang: alarms.log has one full-sentence ALARM line naming rank 5 on nodeA and the stack sample in alarms/', ' ALARM minority at ' in al and 'rank 5 on node nodeA was named' in al and 'alarms/nodeA_' in al and 'not in a save window' in al, al)
check('hang: STATUS.md counts the alarms so far', 'alarms so far  1 alarm event(s)' in open(job + '/STATUS.md').read(), open(job + '/STATUS.md').read())
check('hang: STATUS.md shows the open alarm, the nodes and the next action', 'open alarm     there is an open alarm: minority' in st and '1 spare node is left: nodeC' in st and 'next action    watching attempt 1' in st, st)
for _ in range(40):            # the child process imports matplotlib: about 4 s on a loaded login node (1.5 s was too short)
    if os.path.exists(job + '/heatmap.png'):
        break
    time.sleep(0.5)
check('hang: heatmap.png drawn on the alarm (the aggregator, a child process)', os.path.exists(job + '/heatmap.png') or 'matplotlib' in open(job + '/data/heatmap.err').read(), open(job + '/data/heatmap.err').read()[-300:] if os.path.exists(job + '/data/heatmap.err') else 'no err file')
job, body, last = run('dead')
check('dead, one beat: nobody is named yet (the two-beat rule); the beat block says stillness pending for rank 12',
      last['still'] == [] and last['still_pending'] == [12] and 'stillness pending' in body and 'ALARM' not in body, str(last['still']) + body)
job, body, last = run('dead', beats=2)
check('dead, two beats: stillness names rank 12 and the ALARM line says two consecutive beats', last['still'] == [12] and 'ALARM stillness' in body and 'two consecutive beats' in body, str(last['still']) + body)
job, body, last = run('slow')
check('slow: rank 7 named', last['slow'] == [7] and 'rank 7 on nodeA' in body, body)
job, body, last = run('ecc')
check('ecc: correctable delta on nodeB', 'ECC correctable +2 on nodeB' in body, body)
# the save window (2026-10-09 late evening): rank 5 behind its group while a rank of the group is in a save -> no alarm at 40 s
# (the threshold is 120 s inside a save window); the same without a save -> the minority alarm (case 'hang' above)
def run_save():
    job = tempfile.mkdtemp(prefix='aggtest_', dir=os.environ.get('TMPD', '/tmp'))
    os.makedirs(job + '/data/hb')
    open(job + '/data/state', 'w').write('attempt=1\nstatus=running\nworld=16\ngroup=8\npp=1\nexcluded=\ntrain_nodes=nodeA,nodeB\nall_nodes=nodeA,nodeB,nodeC\nduty=nodeC\nnext=watching attempt 1\n')
    now = time.time()
    with open(job + '/data/hb/nodeA.tsv', 'w') as a, open(job + '/data/hb/nodeB.tsv', 'w') as b:
        for t in [now - 50, now - 40, now - 30, now - 20, now - 10, now - 1]:
            for r in range(16):
                fh = a if r < 8 else b
                node = 'nodeA' if r < 8 else 'nodeB'
                cs = 100 if r == 5 else (101 if r < 8 else 150)
                phase = 'save' if r == 5 else ('save_barrier' if r < 8 else 'all_to_all')
                fh.write(ln(t, node, r, 3, phase, cs))
    class O: pass
    o = O(); o.job = o.job_dir = job; o.world = 16; o.group = 8; o.pp = 1; o.interval = 10; o.still = 30; o.pred = 0
    g = A.Agg(o); g.beat()
    return open(job + '/heartbeat.log').read(), json.load(open(job + '/data/agg_last.json'))
body_s, last_s = run_save()
check('save window: no minority alarm at 40 s while the group is in a save (threshold 120 s)', last_s['minority'] == [] and 'ALARM' not in body_s and '1 of 2 groups are in a save window' in body_s, body_s)
# the delivery lag (2026-10-10 00:45): nodeB's 8 ranks all 35 s old and nodeB's file itself 35 s old -> a lag, never an alarm;
# the same ages with a fresh file -> a real stillness of the ranks (named at the second beat)
def run_lag(file_old, beats):
    job = tempfile.mkdtemp(prefix='aggtest_', dir=os.environ.get('TMPD', '/tmp'))
    os.makedirs(job + '/data/hb')
    open(job + '/data/state', 'w').write('attempt=1\nstatus=running\nworld=16\ngroup=8\npp=1\nexcluded=\ntrain_nodes=nodeA,nodeB\nall_nodes=nodeA,nodeB,nodeC\nduty=nodeC\nnext=watching attempt 1\n')
    now = time.time()
    with open(job + '/data/hb/nodeA.tsv', 'w') as a, open(job + '/data/hb/nodeB.tsv', 'w') as b:
        for k, t in enumerate([now - 55, now - 45, now - 35, now - 25, now - 15, now - 1]):
            for r in range(16):
                if r >= 8 and t > now - 35:
                    continue                            # nodeB's ranks: the newest line of every rank is 35 s old
                (a if r < 8 else b).write(ln(t, 'nodeA' if r < 8 else 'nodeB', r, 3 + k, 'forward', 100 + 10 * k))
    if file_old:
        os.utime(job + '/data/hb/nodeB.tsv', (now - 35, now - 35))     # the file stopped changing with its lines: a delivery lag
    class O: pass
    o = O(); o.job = o.job_dir = job; o.world = 16; o.group = 8; o.pp = 1; o.interval = 10; o.still = 30; o.pred = 0
    g = A.Agg(o)
    for _ in range(beats):
        g.beat()
    return job, open(job + '/heartbeat.log').read(), json.load(open(job + '/data/agg_last.json'))
job, body, last = run_lag(True, 2)
check('lag: two beats of an equally old node with an equally old file -> no ALARM, no stack sample, the beat block names the delivery lag',
      'ALARM' not in body and 'delivery lag 3' in body and 'on 1 node (Lustre), not a rank' in body and last['still'] == [] and 'nodeB' in last['lag']
      and not os.listdir(job + '/alarms'), body)
job, body, last = run_lag(True, 3)
check('lag: after three beats the line says the lag looks like a dead writer or a lost node, still no ALARM',
      'ALARM' not in body and 'more likely a dead node writer or a lost node' in body, body)
def run_all_stale(beats):
    """every rank of BOTH nodes equally 35 s old with both files 35 s old: not a delivery lag (nobody is fresh), so the plain rules apply"""
    job = tempfile.mkdtemp(prefix='aggtest_', dir=os.environ.get('TMPD', '/tmp'))
    os.makedirs(job + '/data/hb')
    open(job + '/data/state', 'w').write('attempt=1\nstatus=running\nworld=16\ngroup=8\npp=1\nexcluded=\ntrain_nodes=nodeA,nodeB\nall_nodes=nodeA,nodeB,nodeC\nduty=nodeC\nnext=watching attempt 1\n')
    now = time.time()
    with open(job + '/data/hb/nodeA.tsv', 'w') as a, open(job + '/data/hb/nodeB.tsv', 'w') as b:
        for k, t in enumerate([now - 55, now - 45, now - 35]):
            for r in range(16):
                (a if r < 8 else b).write(ln(t, 'nodeA' if r < 8 else 'nodeB', r, 3, 'all_to_all', 100 if r == 5 else (101 if r < 8 else 150)))
    for n in ('nodeA', 'nodeB'):
        os.utime(job + f'/data/hb/{n}.tsv', (now - 35, now - 35))
    class O: pass
    o = O(); o.job = o.job_dir = job; o.world = 16; o.group = 8; o.pp = 1; o.interval = 10; o.still = 30; o.pred = 0
    g = A.Agg(o)
    for _ in range(beats):
        g.beat()
    return open(job + '/heartbeat.log').read(), json.load(open(job + '/data/agg_last.json'))
body_a, last_a = run_all_stale(1)
check('all nodes stale at once: no delivery lag is called, the minority rule still names rank 5 (the end of a step is not a Lustre lag)',
      last_a['lag'] == {} and 'delivery lag' not in body_a and last_a['minority'] == [5], str(last_a) + body_a[-800:])
job, body, last = run_lag(False, 2)
check('fresh file, old lines: a stillness of the ranks, named at the second beat (not a delivery lag)',
      last['still'] == list(range(8, 16)) and 'ALARM stillness' in body and 'delivery lag' not in body and last['lag'] == {}, str(last['still']) + body[-1500:])
# banner compose
open(job + '/data/banners.txt', 'w').write('##### RECOVERED 04:10:00: collective_hang on nodeA (rank 5) at step 3; restart 1 inside the same allocation; nodeA excluded; training continues from the save of step 2 #####\n')
open(job + '/a-monitor.txt', 'w').write('monitor line 1\nmonitor line 2\n')
A.compose(job)
hl = open(job + '/heartbeat.log').read().split('\n')
am = open(job + '/a-monitor.txt').read().split('\n')
check('compose: heartbeat.log starts with the recoveries header + banner', hl[0].startswith('===== RECOVERIES SO FAR: 1') and hl[1].startswith('##### RECOVERED'), '\n'.join(hl[:3]))
check('compose: a-monitor.txt banner area is 20 lines and keeps the monitor text', am[1].startswith('##### RECOVERED') and am[19] == '##### END BANNER AREA #####' and am[20] == 'monitor line 1', '\n'.join(am[:22]))
A.compose(job)
am2 = open(job + '/a-monitor.txt').read().split('\n')
check('compose: idempotent (no second banner area)', am2 == am, '')
print('ALL PASS' if ok else 'SOME FAILED')
sys.exit(0 if ok else 1)
