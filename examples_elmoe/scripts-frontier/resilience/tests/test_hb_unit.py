#!/usr/bin/env python3
"""Login-node unit checks of hb.py: one heartbeat line has the 19 fields; the injection schedule parses; D7 fires exit 13 after 3
bad losses (in a child process); D3 fires exit 12 when a save outlives its deadline (in a child process); the node writer copies."""
import os, sys, subprocess, tempfile, time, json
BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
sys.path.insert(0, BIN)
T = tempfile.mkdtemp(prefix='hbunit_', dir=os.environ.get('TMPD', '/tmp'))
ok = True
def check(name, cond, extra=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else '  ' + str(extra))); ok &= bool(cond)

import hb
hb._ctx.update(rank=3, local_rank=0, job_dir=T)
ln, hm = hb.line()
f = ln.rstrip('\n').split('\t')
check('line has 19 fields', len(f) == 19 == len(hb.HB_FIELDS), f)
check('proc_state is a letter', f[13] in 'RSDZT', f[13])
check('host_mem_pct parsed', 0 < hm < 100, hm)
os.environ['XMOE_INJECT_SCHEDULE'] = 'hang:5:3+exit:9:6,oom:2:9'
hb._inj['cases'] = None
check('schedule parses', hb._cases() == [('hang', 5, 3), ('exit', 9, 6), ('oom', 2, 9)], hb._cases())
child = '''
import os, sys; sys.path.insert(0, %r); import hb
hb._ctx.update(rank=0, local_rank=0, job_dir=%r, detect=True)
mode = sys.argv[1]
if mode == 'loss':
    for s, l in enumerate([3.0, 2.9, 2.8, 2.7, 2.6, 2.5, float('nan'), float('nan'), float('nan'), 2.4]):
        hb.loss_rule(s, l)
if mode == 'save':
    os.environ['XMOE_SAVE_DEADLINE_S'] = '2'
    import torch.distributed     # [2026-10-10] the trainer has torch imported long before a save; SaveDeadline.__enter__ imports
                                 # torch.distributed to wrap its collectives, and a COLD import from Lustre took 8.8 s on login10
                                 # tonight, which sat inside the measured window and failed D3 three times; it is paid here instead
    with hb.SaveDeadline(12):
        import time; time.sleep(10)
print('not reached')
''' % (BIN, T)
r = subprocess.run([sys.executable, '-c', child, 'loss'], capture_output=True, text=True)
check('D7: exit 13 after 3 NaN losses', r.returncode == 13, (r.returncode, r.stdout[-300:]))
rec = json.load(open(os.path.join(T, 'data', 'exit', 'attempt_0', 'r0.json')))   # no RES_EXIT_DIR: data/exit/attempt_<k>/
check('D7: exit record class loss_nan_spike', rec['class'] == 'loss_nan_spike', rec)
t0 = time.time()
r = subprocess.run([sys.executable, '-c', child, 'save'], capture_output=True, text=True)
# the deadline is judged from the child's own clock (its DETECTOR EXIT line says "not done after N s (deadline 2 s)", N measured from
# the start of the save): the parent's clock also holds the python start and the torch import, which are not what D3 measures
import re as _re
_m = _re.search(r'not done after (\d+) s', r.stdout)
_n = int(_m.group(1)) if _m else None
check('D3: exit 12 at the 2 s deadline', r.returncode == 12 and _n is not None and _n <= 5, (r.returncode, _n, time.time() - t0, r.stdout[-300:]))
# D9 (2026-10-16, the port of the failed-save class; job 5664539): an uncaught exception whose text carries the asynchronous engine's
# "[AsyncPinned] a checkpoint write failed" exits 12 (filesystem: restart from the last complete save); any other exception text still 17
exc_child = '''
import os, sys; sys.path.insert(0, %r); import hb
hb._ctx.update(rank=0, local_rank=0, job_dir=%r, detect=True)
try:
    raise RuntimeError(sys.argv[1])
except RuntimeError:
    hb.excepthook(*sys.exc_info())
print('not reached')
''' % (BIN, T)
for msg, want, cls in (('[AsyncPinned] a checkpoint write failed: unexpected pos 1048576 (rank 27, global_step20)', 12, 'filesystem_hang'),
                       ('a code defect in the trainer', 17, 'framework_exception')):
    ed = tempfile.mkdtemp(prefix='exc_', dir=T)
    r = subprocess.run([sys.executable, '-c', exc_child, msg], capture_output=True, text=True, env=dict(os.environ, RES_EXIT_DIR=ed))
    try:
        rec = json.load(open(os.path.join(ed, 'r0.json')))
    except (OSError, ValueError):
        rec = {}
    check(f'D9: an exception "{msg[:45]}" exits {want} ({cls})', r.returncode == want and rec.get('code') == want and rec.get('class') == cls,
          (r.returncode, rec, r.stderr[-300:]))
# node writer: copies new bytes of rank files into the node file, then stops when its parent is gone
src = os.path.join(T, 'shm'); os.makedirs(src); open(os.path.join(src, 'rank_0.tsv'), 'w').write('\t'.join(hb.HB_FIELDS) + '\n# ecc source\na\tb\n')
dst = os.path.join(T, 'node.tsv')
p = subprocess.Popen([sys.executable, os.path.join(BIN, 'hb.py'), '--node-writer', src, dst, '0.5', '0', str(os.getpid())])
time.sleep(1.5); open(os.path.join(src, 'rank_0.tsv'), 'a').write('c\td\n'); time.sleep(1.5); p.kill()
check('node writer: the header once, the rank file\'s header lines dropped, both data lines copied once',
      open(dst).read() == '\t'.join(hb.HB_FIELDS) + '\na\tb\nc\td\n', open(dst).read())
check('header_lines: the field names first, then the ECC source line', hb.header_lines().startswith('time\tnode\trank') and '# ecc_ce/ecc_ue:' in hb.header_lines(), hb.header_lines())
ce, ue, umc, note = hb._ecc_smi(0)
check('ecc via rocm-smi returns numbers or -1 with a note (login node: rocm-smi may be absent)', (ce >= 0 and ue >= 0) or (ce == -1 and note), (ce, ue, umc, note))
# training.log: one line per step, the first line of an attempt names the save it resumed from; a save that follows is noted
hb._ctx.update(rank=0, job_dir=T, attempt='1', resumed_from=4); hb._ctx.pop('tl_first_done', None); hb._loss['bad'] = 0; hb.S['flag'] = '-'
hb.training_log_line(5, 9.92, 36.25, 2.41, 41.3, False)
hb.training_log_line(6, 9.81, 36.40, 2.40, 41.3, True)
tl = open(os.path.join(T, 'training.log')).read().split('\n')
check('training.log: the column-name row first, then two step lines with the columns', len(tl) == 4 and tl[0].startswith('time') and tl[0].split('|')[2].strip() == 'step'
      and tl[1].split('|')[2].strip() == '5' and tl[1].split('|')[4].strip() == '36.25', tl)
check('training.log: the column-name row lines up with the step lines (every | at the same column)',
      [i for i, c in enumerate(tl[0]) if c == '|'][:7] == [i for i, c in enumerate(tl[1]) if c == '|'][:7], tl[:2])
check('training.log: notes (first step of attempt 1 resumed from 4; save of step 6 follows)',
      'first step of attempt 1, resumed from the save of step 4' in tl[1] and 'save of step 6 follows' in tl[2], tl)
for k in range(7, 7 + hb.TL_HEAD_EVERY):
    hb.training_log_line(k, 9.8, 36.4, 2.4, 41.3, False)
tl = open(os.path.join(T, 'training.log')).read().split('\n')
check('training.log: the column-name row repeats every TL_HEAD_EVERY step lines', sum(1 for l in tl if l.startswith('time')) == 2, sum(1 for l in tl if l.startswith('time')))
print('ALL PASS' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)
