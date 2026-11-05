#!/usr/bin/env python3
"""test_oom_reconfig.py: the unit tests of the OOM re-configuration rule (bin/oom_reconfig.py --selftest) plus the CLI contract the loop
relies on (the `next --shell` assignments and the `gate --shell` assignments). Login node, no GPU, under a second.
Cases: the 16B test layout (64 nodes, 128k, PP2 EP16 TP1 U1 mbs 1: red at the gate; the ladder 1 -> 2 -> 3 with TP1 x U16, EP32 alone
because the 16 heads bound d, PP4; the fourth OOM stops), the heatmap's 'batch' cell for d2 (no colour; computed green), the 671B
production layout (PP8 EP32 TP2 U16 mbs 2 at 256 nodes: step 0, step 1 skipped, EP64 d64, PP16; the TP2 x U16 split where TP1 is
red), and the user's example chain shape PP2 EP16 d2 -> PP2 EP16 d16 -> PP2 EP32 d32 -> PP4 EP32 d32 on a 128-head model."""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__)); BIN = os.path.join(HERE, '..', 'bin')
sys.path.insert(0, BIN)
import oom_reconfig as o   # noqa: E402

ok = o.selftest() == 0
PY = sys.executable
# the CLI contract: next --shell for the test layout upon the first OOM
r = subprocess.run([PY, os.path.join(BIN, 'oom_reconfig.py'), 'next', '--model', 'DeepSeek_16B', '--nodes', '64', '--seq', '131072', '--gbs', '256', '--pp', '2', '--ep', '16',
                    '--tp', '1', '--ulysses', '1', '--mbs', '1', '--chunked-ce', '2048', '--capacity', '1.0', '--taken', '0', '--oom-step', '1', '--oom-stage', '1',
                    '--oom-rank', '300', '--oom-hbm', '63.8', '--oom-attempt', '0', '--shell'], capture_output=True, text=True)
env = dict(l.split('=', 1) for l in r.stdout.strip().split('\n') if '=' in l)
want = dict(OOM_NEXT="1", OOM_STEP="1", NEW_PP="2", NEW_EP="16", NEW_TP="1", NEW_U="16", NEW_MBS="1", NEW_D="16", NEW_SP="0", OOM_NEEDS_UNIVERSAL="1", NEW_DP_MOE="16", NEW_DP_NON_MOE="16", NEW_GA="16")
for k, v in want.items():
    got = env.get(k, '').strip("'")
    print(('PASS ' if got == v else 'FAIL ') + f'next --shell {k}={got} (want {v})'); ok = ok and got == v
print(('PASS ' if 'the rule takes step 1' in env.get('OOM_TEXT', '') and 'TP1 x U16' in env.get('OOM_TEXT', '') else 'FAIL ') + 'the block names step 1 and TP1 x U16')
ok = ok and 'the rule takes step 1' in env.get('OOM_TEXT', '')
# gate --shell for the same layout: a WARNING and the per-stage prediction
r = subprocess.run([PY, os.path.join(BIN, 'oom_reconfig.py'), 'gate', '--model', 'DeepSeek_16B', '--nodes', '64', '--seq', '131072', '--gbs', '256', '--pp', '2', '--ep', '16',
                    '--tp', '1', '--ulysses', '1', '--mbs', '1', '--chunked-ce', '2048', '--capacity', '1.0', '--shell'], capture_output=True, text=True)
env = dict(l.split('=', 1) for l in r.stdout.strip().split('\n') if l.startswith('MG_'))
c = env.get('MG_COLOUR', '').strip("'") == 'red' and 'WARNING' in env.get('MG_WARNING', '') and env.get('MG_PRED', '').count(':') == 2
print(('PASS ' if c else 'FAIL ') + f"gate --shell: colour {env.get('MG_COLOUR')}, warning present, two stages predicted ({env.get('MG_PRED', '')[:60]})"); ok = ok and c
# the same gate for the d2 layout at GBS 128: no red, the heatmap cell is 'batch'
r = subprocess.run([PY, os.path.join(BIN, 'oom_reconfig.py'), 'gate', '--model', 'DeepSeek_16B', '--nodes', '64', '--seq', '131072', '--gbs', '128', '--pp', '2', '--ep', '16',
                    '--tp', '1', '--ulysses', '2', '--mbs', '1', '--chunked-ce', '2048', '--capacity', '1.0'], capture_output=True, text=True)
c = "reads 'batch'" in r.stdout and 'green' in r.stdout
print(('PASS ' if c else 'FAIL ') + "gate for PP2 EP16 TP1 U2 at GBS 128: the heatmap cell reads 'batch' and the computed colour is green"); ok = ok and c
print('OOM RECONFIG UNIT: ' + ('ALL PASS' if ok else 'SOME FAILED'))
sys.exit(0 if ok else 1)
