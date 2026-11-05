#!/usr/bin/env python3
# ft-sys-v2 copy (2026-11-05) of examples_elmoe/scripts-frontier/resilience/tests/test_gate_judge.py (ft-sys-v1, commit 7fb97edf). Changed: only the software-stack paths, now settings with the new stack (ROCm 7.1.1, torch 2.13 venv, aws-ofi-nccl 1.22.0) as the default (GATE_PYTHON, GATE_OFI_DIR, ROCM_PATH); the tests, the rules and the judge are v1's. The tests point at the copies in ..
"""Login-node check of gate_judge.py with SYNTHETIC rows for the rules of 2026-10-09 evening (4 nodes n0..n3, every test calibrated):
  A. the pair naming rule: n1 named by n0 in round 1 and by n3 in round 2 -> excluded as DEAD ('did not answer two different partners');
  B. named once + link errors: n2 named once, its switch row shows cxi_err=3 -> excluded as DEAD ('link errors on its cards');
  C. named once, no link errors -> a SUSPECT PAIR listed, nobody excluded; with --exclude-suspect-pair --spares 2 -> both excluded;
  D. the majority rule: gemm 'timeout' on n2 while n0, n1, n3 pass -> excluded as DEAD; two timeouts (n2, n3) -> not excluded;
  E. the two tiers: gemm below threshold on n2 -> excluded as DEGRADED (excluded_reasons.tsv says so; gate.log lists it under degraded);
  F. gate.log prints a command line and a column header under every test."""
import os, subprocess, sys, tempfile, csv
BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PY = sys.executable
NODES = ['n0', 'n1', 'n2', 'n3']
ok = True


def check(name, cond, extra=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else '  ' + str(extra)[:800])); ok &= bool(cond)


def pair_detail(gcds, named_by='', partners='round 1: partner nA; round 2: partner nB', per_result=''):
    """named_by: [(node, round)] (the old form, kind timeout) or [(node, round, kind)] with kind 'timeout' or 'gone' (2026-10-10);
    per_result: the 'per result:' text of the row (what THIS node recorded about its partners), e.g. 'round 1.1 GCD 0 with n1: timeout (...)'."""
    named_in = len(set(x[1] for x in named_by)) if named_by else 0
    nb = ','.join(f'{x[0]}(round {x[1]}' + (f', {x[2]})' if len(x) > 2 else ')') for x in named_by) if named_by else 'nobody'
    return (f'the slowest of 8 GCDs over 2 rounds x 2 sub-rounds (16 all-reduce results ok, 0 timed out, 0 partner gone, 0 wrong data, 0 skipped); '
            + ','.join(f'gcd{g}={v}' for g, v in gcds.items()) + f'; card 0: GCD 0 {gcds[0]}, GCD 1 {gcds[1]}; {partners}; named_in={named_in} named_by={nb}; '
            f'path host-buffer (no /GDRDMA on the 48 via-NET lines; NCCL_NET_GDR_LEVEL unset (RCCL default)); OFI device per GCD: gcd0->0'
            + (f'; 64 MiB fp32 x 20 per all-reduce, deadline 30 s; per result: {per_result}' if per_result else ''))


def switch_detail(cxi_err=0):
    return f'busbw min of 8 ranks of this node (max 11.9); 32 ranks x 64 MiB x 10; cxi_err={cxi_err} (link errors {cxi_err}, replays 0); pause frames 10'


def run(case, pair=None, switch=None, gemm=None, extra_args=(), cal_nodes=None, net_marker=None):
    """pair/switch/gemm: node -> (value, status, detail) overrides; returns (report rows, failed, reasons, gate.log text, stdout)."""
    T = tempfile.mkdtemp(prefix='judge_', dir=os.environ.get('TMPD', '/tmp'))
    raw = os.path.join(T, 'raw'); os.makedirs(raw)
    if net_marker:                                   # [2026-10-11] net_launch_retry.sh's not-launched marker
        os.makedirs(os.path.join(raw, 'net')); open(os.path.join(raw, 'net', 'not_launched'), 'w').write(net_marker)
    base = os.path.join(T, 'gate_baseline.tsv')
    with open(base, 'w') as fh:
        fh.write('test\tbaseline\trun_s\tdate\tjob\tnodes\tnote\n')
        for t, b, r in [('answers_srun', 'yes', '0.1'), ('rocm_smi_ids', '8', '0.9'), ('ras_ue', '0', '0.2'), ('xgmi_err', '0', '0.2'),
                        ('rocm_version', '6.4.1-83', '0.0'), ('rccl_md5', 'abc:1', '0.1'), ('ofi_plugin_md5', 'def:2', '0.0'), ('libfabric', 'fi 2.3.1', '0.0'),
                        ('shm_free_gb', '> 1', '0.0'), ('gemm', '113.0', '94.8'), ('xgmi_a2a', '65.8', '32.4'), ('pair', '17.0', '20.0'), ('switch', '11.4', '0.4'),
                        ('counters_after', 'same', '0.1')]:
            fh.write(f'{t}\t{b}\t{r}\t2026-10-09\t999\t{(cal_nodes or {}).get(t, 4)}\tsynthetic\n')
    ts = '2026-10-09 23:00:00'
    for n in NODES:
        rows = [('counters_before', 'ue=-1,ce=-1,xgmi=0', '0.1', 'ok', '-'), ('rocm_smi_ids', '8', '0.3', 'ok', '-'), ('ras_ue', '0', '0.2', 'ok', '-'),
                ('xgmi_err', '0', '0.2', 'ok', '-'), ('rocm_version', '6.4.1-83', '0.0', 'ok', '-'), ('rccl_md5', 'abc:1', '0.1', 'ok', 'f'),
                ('ofi_plugin_md5', 'def:2', '0.0', 'ok', 'f'), ('libfabric', 'fi 2.3.1', '0.0', 'ok', '-'), ('gpu_direct_env', 'x', '0', 'ok', '-'),
                ('dmesg', 'not permitted', '0.1', 'ok', '-'), ('shm_free_gb', '250.0', '0.0', 'ok', '-'),
                ('xgmi_a2a', '65.9', '32.3', 'ok', 'busbw'), ('counters_after', 'same', '0.1', 'ok', '-'), ('node_done', 'yes', '0', 'ok', '-')]
        g = (gemm or {}).get(n, ('110.0', '90.0', 'ok', 'min of 8 GCDs; gcd0=110.0,gcd1=112.0,gcd2=111.0,gcd3=113.0,gcd4=114.0,gcd5=115.0,gcd6=116.0,gcd7=112.0; import 4 s'))
        rows.append(('gemm',) + tuple(g))
        with open(os.path.join(raw, f'{n}.raw.tsv'), 'w') as fh:
            for r in rows:
                fh.write('\t'.join([ts, n] + list(r)) + '\n')
        with open(os.path.join(raw, f'{n}.probe.tsv'), 'w') as fh:
            fh.write('\t'.join([ts, n, 'answers_srun', 'yes', '0', 'ok', 'the node wrote its gate rows']) + '\n')
        p = (pair or {}).get(n, ('16.8', 'ok', pair_detail({g: 16.8 + 0.1 * g for g in range(8)})))
        s = (switch or {}).get(n, ('11.5', 'ok', switch_detail()))
        with open(os.path.join(raw, f'{n}.net.tsv'), 'w') as fh:
            fh.write('\t'.join([ts, n, 'pair', p[0], '20.0', p[1], p[2]]) + '\n')
            fh.write('\t'.join([ts, n, 'switch', s[0], '0.4', s[1], s[2]]) + '\n')
    cmd = [PY, os.path.join(BIN, 'gate_judge.py'), '--raw', raw, '--nodes', ' '.join(NODES), '--mode', 'full', '--baseline', base,
           '--report', os.path.join(T, 'report.tsv'), '--failed', os.path.join(raw, 'failed_nodes.txt'), '--log', os.path.join(T, 'gate.log'),
           '--job', '999', '--dry-rc', '0', '--t0', '0', '--budgets', 'gemm=220,pair=250,switch=31'] + list(extra_args)
    r = subprocess.run(cmd, capture_output=True, text=True)
    rep = list(csv.DictReader(open(os.path.join(T, 'report.tsv')), delimiter='\t'))
    failed = open(os.path.join(raw, 'failed_nodes.txt')).read().split()
    reasons = {x['node']: x for x in csv.DictReader(open(os.path.join(raw, 'excluded_reasons.tsv')), delimiter='\t')}
    return rep, failed, reasons, open(os.path.join(T, 'gate.log')).read(), r


def verdict(rep, node, test):
    return next(x for x in rep if x['node'] == node and x['test'] == test)


# A. named in both rounds -> dead
rep, failed, reasons, log, r = run('A', pair={'n1': ('16.8', 'ok', pair_detail({g: 16.8 for g in range(8)}, named_by=[('n0', 1), ('n3', 2)]))})
v = verdict(rep, 'n1', 'pair')
check('A: n1 named by two different partners is excluded', failed == ['n1'], (failed, r.stdout[-500:], r.stderr[-500:]))
check('A: the reason names both partners', 'did not answer two different partners' in v['reason'] and 'n0 in round 1' in v['reason'] and 'n3 in round 2' in v['reason'], v['reason'])
check('A: tier dead', reasons.get('n1', {}).get('tier') == 'dead', reasons)
# B. named once + link errors on its own cards -> dead
rep, failed, reasons, log, r = run('B', pair={'n2': ('16.8', 'ok', pair_detail({g: 16.8 for g in range(8)}, named_by=[('n3', 2)]))},
                                   switch={'n2': ('11.5', 'ok', switch_detail(cxi_err=3))})
check('B: n2 named once with link errors is excluded', 'n2' in failed, (failed, r.stderr[-300:]))
check('B: the pair reason says link errors, named by a partner', 'link errors on its cards' in verdict(rep, 'n2', 'pair')['reason'] and 'n3' in verdict(rep, 'n2', 'pair')['reason'], verdict(rep, 'n2', 'pair')['reason'])
check('B: tier dead (the switch row also fails it)', reasons.get('n2', {}).get('tier') == 'dead', reasons)
# C. named once, no link errors -> suspect pair, nobody excluded; with the flag and 2 spares -> both excluded
rep, failed, reasons, log, r = run('C', pair={'n2': ('16.8', 'ok', pair_detail({g: 16.8 for g in range(8)}, named_by=[('n3', 2)]))})
check('C: suspect pair listed, nobody excluded', failed == [] and 'SUSPECT PAIR n2 and n3' in log and 'not excluded' in log, (failed, [l for l in log.split('\n') if 'suspect' in l.lower()]))
check('C: the pair row of n2 passes with the suspect note', verdict(rep, 'n2', 'pair')['pass'] == 'yes' and 'suspect pair with n3' in verdict(rep, 'n2', 'pair')['reason'], verdict(rep, 'n2', 'pair'))
rep, failed, reasons, log, r = run('C2', pair={'n2': ('16.8', 'ok', pair_detail({g: 16.8 for g in range(8)}, named_by=[('n3', 2)]))},
                                   extra_args=['--exclude-suspect-pair', '--spares', '2'])
check('C2: with GATE_EXCLUDE_SUSPECT_PAIR and 2 spares both nodes are excluded', sorted(failed) == ['n2', 'n3'] and 'BOTH EXCLUDED' in log, (failed, [l for l in log.split('\n') if 'suspect' in l.lower()]))
rep, failed, reasons, log, r = run('C3', pair={'n2': ('16.8', 'ok', pair_detail({g: 16.8 for g in range(8)}, named_by=[('n3', 2)]))},
                                   extra_args=['--exclude-suspect-pair', '--spares', '1'])
check('C3: with the flag but 1 spare nobody is excluded', failed == [] and 'fewer than 2 spares' in log, (failed,))
# D. the majority rule
rep, failed, reasons, log, r = run('D', gemm={'n2': ('105.0', '206.0', 'timeout', '7 of 8 GCDs reported before the deadline of 206 s; the GCDs that did not answer: gcd7; gcd0=105.0; import 4 s')})
check('D: one gemm timeout while the 3 others passed is excluded by the majority rule', failed == ['n2'] and 'majority rule' in reasons.get('n2', {}).get('sentence', ''), (failed, reasons))
check('D: tier dead', reasons.get('n2', {}).get('tier') == 'dead', reasons)
check('D: gate.log says the majority rule applied with the counts', 'majority rule on gemm: the majority rule applies: the test passed on 3 of the other 3 nodes' in log, [l for l in log.split('\n') if 'majority' in l])
rep, failed, reasons, log, r = run('D2', gemm={'n2': ('105.0', '206.0', 'timeout', '7 of 8 GCDs reported; gcd0=105.0'), 'n3': ('-', '211.0', 'timeout', '0 of 8 GCDs reported before the budget of 211 s; none')})
check('D2: two gemm timeouts of 4 are not excluded (2 of 3 others passed, below 90 percent)', failed == [] and 'not excluded' in log, (failed, [l for l in log.split('\n') if l.startswith('gemm')]))
# E. the two tiers: below threshold on a performance test -> degraded
rep, failed, reasons, log, r = run('E', gemm={'n2': ('60.0', '95.0', 'ok', 'min of 8 GCDs; gcd0=110.0,gcd1=112.0,gcd2=111.0,gcd3=113.0,gcd4=114.0,gcd5=115.0,gcd6=116.0,gcd7=60.0; import 4 s')})
check('E: gemm below threshold is excluded as degraded', failed == ['n2'] and reasons.get('n2', {}).get('tier') == 'degraded' and reasons['n2']['test'] == 'gemm' and reasons['n2']['value'] == '60.0', reasons)
check('E: gate.log lists it under degraded, usable as a spare of last resort', 'excluded as degraded (a performance test below its threshold; usable as spares of last resort' in log and 'n2 (gemm 60.0 against >= 101.7)' in log, [l for l in log.split('\n') if 'degraded' in l])
# G. THE GONE PARTNER (2026-10-10, from job 5646336): a partner that left counts like a timeout
#    G1: n1 (the silent node) named by n0 in round 1 (timeout) and by n3 in round 2 (gone) -> excluded as DEAD, the reason shows both kinds
rep, failed, reasons, log, r = run('G1', pair={'n1': ('16.8', 'ok', pair_detail({g: 16.8 for g in range(8)}, named_by=[('n0', 1, 'timeout'), ('n3', 2, 'gone')]))})
v = verdict(rep, 'n1', 'pair')
check('G1: a timeout in round 1 and a gone partner in round 2 exclude the node as dead', failed == ['n1'] and reasons.get('n1', {}).get('tier') == 'dead', (failed, reasons, r.stderr[-300:]))
check('G1: the reason names both partners with their kinds', 'n0 in round 1 (timeout)' in v['reason'] and 'n3 in round 2 (gone)' in v['reason'], v['reason'])
#    G2: the EXPLAINED departure: n0 timed out on n1 in round 1 (and left, by design); in round 2 n2 found n0 gone and named it once ->
#        n0 is NOT a suspect pair, the reason says the departure is explained by n1, which is excluded as dead
rep, failed, reasons, log, r = run('G2', pair={
    'n1': ('16.8', 'ok', pair_detail({g: 16.8 for g in range(8)}, named_by=[('n0', 1, 'timeout'), ('n3', 2, 'gone')])),
    'n0': ('16.8', 'ok', pair_detail({g: 16.8 for g in range(8)}, named_by=[('n2', 2, 'gone')],
                                     per_result='round 1.1 GCD 0 with n1: timeout (the first all-reduce did not complete within 30 s: partner n1 did not answer) | round 2.1 GCD 0 with n2: ok 12.3 GB/s'))})
v0 = verdict(rep, 'n0', 'pair')
check('G2: only the dead node is excluded', failed == ['n1'], (failed, r.stderr[-300:]))
check('G2: n0 passes with the explained-departure note, and no suspect pair is listed', v0['pass'] == 'yes' and 'explained departure' in v0['reason'] and 'n1 in round 1' in v0['reason']
      and 'suspect pairs: none' in log, (v0, [l for l in log.split('\n') if 'suspect' in l.lower()]))
#    G3: a single gone naming WITHOUT an explanation (n2 never named anybody) stays a suspect pair, as a single timeout does
rep, failed, reasons, log, r = run('G3', pair={'n2': ('16.8', 'ok', pair_detail({g: 16.8 for g in range(8)}, named_by=[('n3', 2, 'gone')]))})
check('G3: an unexplained single gone naming is a suspect pair, nobody excluded', failed == [] and 'SUSPECT PAIR n2 and n3' in log, (failed, [l for l in log.split('\n') if 'suspect' in l.lower()]))
#    G4: the old named_by form without a kind still reads as a timeout
sys.path.insert(0, BIN); import gate_judge as GJ
check('G4: the old form n0(round 1) parses as a timeout', GJ.named_kinds_of('named_in=1 named_by=n0(round 1); x') == [('n0', 1, 'timeout')]
      and GJ.named_kinds_of('named_in=2 named_by=n0(round 1, timeout),n3(round 2, gone); x') == [('n0', 1, 'timeout'), ('n3', 2, 'gone')]
      and GJ.own_timeouts_of('per result: round 1.1 GCD 0 with n1: timeout (x) | round 2.1 GCD 0 with n2: gone (y) | round 2.2 GCD 1 with n2: ok 12.3 GB/s') == [(1, 'n1', 'timeout'), (2, 'n2', 'gone')], '')
# F. gate.log shows the command and the column header under every test
check('F: every test block prints its command and its column header', log.count('  command: ') >= 14 and log.count('measured value') >= 14 and 'calibrated by (job, date)' in log and 'nodes passed' in log, (log.count('  command: '), log.count('measured value')))
check('F: the pair test block names the deadline and the sub-rounds', 'GATE_PAIR_DEADLINE_S = 30 s' in log and 'even GCDs, then odd GCDs' in log, '')
# H. the switch test's full-node rule (job 5646617): a node that lost a rank is 'not tested' for the switch: neither a pass nor a fail
NT = ('-', 'not tested', 'this node is not in the switch test because its rank(s) [0] left the pair rounds (after a partner that did not answer, or silent), and a communicator over a node with 7 ranks needs RCCL\'s slow generic topology search; the switch test runs over the 2 full node(s); this node is judged by its pair rows; cxi_err=-1 (no counters read)')
rep, failed, reasons, log, r = run('H1', switch={'n0': NT, 'n1': NT})
v = verdict(rep, 'n0', 'switch')
check('H1: a not-tested switch row excludes nobody', failed == [], (failed, r.stderr[-300:]))
check('H1: the row reads not tested with the reason, the table says n/t', v['pass'] == 'not tested' and v['reason'].startswith('not tested: this node is not in the switch test') and '| n/t  |' in log, (v, [l for l in log.split('\n') if 'n0 ' in l and 'switch' not in l][:0]))
check('H1: the switch verdict is ok for the tested nodes (no TEST DEFECT)', any(l.startswith('switch') and 'TEST DEFECT' not in l for l in log.split('\n')) and 'TEST DEFECT' not in [l for l in log.split('\n') if l.startswith('switch')][0], [l for l in log.split('\n') if l.startswith('switch')][:1])
rep, failed, reasons, log, r = run('H2', switch={n: NT for n in NODES})
check('H2: all nodes not tested -> "not run", no TEST DEFECT, nobody excluded', failed == [] and any('not run: the test was not run on any node' in l for l in log.split('\n')) and not any(l.startswith('switch') and 'TEST DEFECT' in l for l in log.split('\n')), (failed, [l for l in log.split('\n') if 'not run' in l or 'DEFECT' in l][:3]))
# I. the scale guard (job 5646880): a pair baseline calibrated at 260 nodes is not applied to a 4-node job; a below-threshold value is recorded, nobody excluded
rep, failed, reasons, log, r = run('I1', pair={'n2': ('9.3', 'ok', pair_detail({g: 9.3 for g in range(8)}))}, cal_nodes={'pair': 260})
check('I1: a below-threshold pair value under a far-scale baseline excludes nobody', failed == [], (failed, r.stderr[-300:]))
check('I1: the pair verdict says the threshold is not applied', any(l.startswith('pair') and 'threshold is not applied' in l for l in log.split('\n')), [l for l in log.split('\n') if l.startswith('pair')][:1])
rep, failed, reasons, log, r = run('I2', pair={'n2': ('9.3', 'ok', pair_detail({g: 9.3 for g in range(8)}))}, cal_nodes={'pair': 8})
check('I2: the same value under a baseline within 4x of the scale is degraded as before', failed == ['n2'] and reasons.get('n2', {}).get('tier') == 'degraded', (failed, reasons.get('n2')))
check('I2: gemm (a per-node test) is never scale-guarded', not any(l.startswith('gemm') and 'threshold is not applied' in l for l in log.split('\n')), '')
# J. (2026-10-11, job 5650829) Slurm did not start the network step: every node's pair/switch row is missing
MISS = ('-', 'missing', 'no rank of this node ran the network phase (no output files)')
rep, failed, reasons, log, r = run('J1', pair={n: MISS for n in NODES}, switch={n: MISS for n in NODES},
                                   net_marker='3\tsrun: error: Unable to confirm allocation for job 999: Socket timed out on send/recv operation\n')
check('J1: with the not-launched marker, pair and switch read NOT LAUNCHED, not TEST DEFECT',
      any(l.startswith('pair') and 'NOT LAUNCHED: Slurm did not start the network step after 3 tries' in l for l in log.split('\n'))
      and any(l.startswith('switch') and 'NOT LAUNCHED' in l for l in log.split('\n'))
      and not any(l.startswith('pair') and 'TEST DEFECT' in l for l in log.split('\n')), [l for l in log.split('\n') if l.startswith(('pair', 'switch'))])
check('J1: nobody excluded, the summary line names them, the exit code is 4 (AUTORUN_GATE_STRICT=1 can stop the job)',
      failed == [] and 'NOT LAUNCHED: pair; switch' in log and 'TEST DEFECTS: none' in log and r.returncode == 4, (failed, r.returncode))
rep, failed, reasons, log, r = run('J2', pair={n: MISS for n in NODES}, switch={n: MISS for n in NODES})
check('J2: the same rows WITHOUT the marker stay TEST DEFECT (the old rule)',
      any(l.startswith('pair') and 'TEST DEFECT: no output on any' in l for l in log.split('\n')) and 'NOT LAUNCHED: none' in log and failed == [], '')
print('ALL PASS' if ok else 'SOME FAILED')
sys.exit(0 if ok else 1)
