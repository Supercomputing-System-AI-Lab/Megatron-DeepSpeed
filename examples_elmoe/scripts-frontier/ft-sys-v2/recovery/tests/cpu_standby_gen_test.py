#!/usr/bin/env python3
"""cpu_standby_gen_test.py (2026-11-07, M3MA4 5875028): a waiting standby takes the failure lists of every later generation of the
run it started under (hb_monitor.list_applies); a training process only those of its own generation."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'heartbeat-sys'))
import hb_monitor as hb  # noqa: E402

cases = [
    ('standby, its own run', {'run': '5875028.a_two'}, '5875028.a_two', 'standby', True),
    ('standby, a later generation (the M3MA4 case)', {'run': '5875028.a_two.g0001'}, '5875028.a_two', 'standby', True),
    ('standby, two generations later', {'run': '5875028.a_two.g0002'}, '5875028.a_two', 'standby', True),
    ('standby, another run of the job', {'run': '5875028.b_lost'}, '5875028.a_two', 'standby', False),
    ('standby, a run whose name only starts the same', {'run': '5875028.a_two_x'}, '5875028.a_two', 'standby', False),
    ('training, its own generation', {'run': '5875028.a_two.g0001'}, '5875028.a_two.g0001', 'train', True),
    ('training, an earlier generation', {'run': '5875028.a_two'}, '5875028.a_two.g0001', 'train', False),
    ('training, a later generation is not followed', {'run': '5875028.a_two.g0002'}, '5875028.a_two.g0001', 'train', False),
]
fails = 0
for name, f, mine, role, want in cases:
    got = hb.list_applies(f, mine, role)
    print(f'{"PASS" if got == want else "FAIL"}  {name}: {got}')
    fails += got != want
print('all passed' if not fails else f'{fails} failed')
sys.exit(1 if fails else 0)
