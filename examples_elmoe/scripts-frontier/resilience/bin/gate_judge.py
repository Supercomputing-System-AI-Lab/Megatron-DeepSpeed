#!/usr/bin/env python3
"""gate_judge.py: the gate's verdicts under the five rules of design 3.11 (2026-10-09 13:45), its calibration, and the rules
added on 2026-10-09 evening (the user's decisions): the pair test's naming rules, the majority rule, the suspect pair, and the
two tiers of exclusion (degraded against dead).

WHAT THIS PROGRAM DOES, IN PLAIN WORDS
  The gate's measuring scripts (gate_node.sh on every node, gate_net.py across the nodes) write raw rows and never judge.
  This program reads every raw row, compares each value with the calibrated baseline of its test, and writes one verdict per
  node per test. From the verdicts it decides which nodes the job must not train on, and it writes gate.log (for people),
  gate_report.tsv (for scripts), failed_nodes.txt (the excluded nodes) and excluded_reasons.tsv (one sentence per excluded
  node, with its tier: degraded or dead; the restart loop copies that sentence into the failure record).

INPUT
  <raw dir>/<node>.raw.tsv   the per-node rows of gate_node.sh:   time node test value run_s status detail
  <raw dir>/<node>.net.tsv   the cross-node rows of gate_net.py --summarize (tests pair, switch, and rdma when GATE_RDMA_PHASE=1)
  <raw dir>/<node>.probe.tsv the re-probe row of gate.sh (test answers_srun)
  the dry run's exit code, and the calibration file gate_baseline.tsv:   test baseline run_s date job nodes note

THE RULES
 1. calibration: a test counts only when gate_baseline.tsv holds its baseline AND its run time, measured by this test code on
    known-good nodes, with a date and a job id; a test without that line is "not calibrated: informational" and fails nobody.
 2. one verdict per node per test: value | baseline | threshold | pass | reason; the reason is one of: below threshold, timeout,
    no output, error <text>.
 3. TEST DEFECT: a test failing on more than half of the nodes, or producing no output anywhere, is a defect of the test: it is
    said with the counts, it excludes nobody, the job continues (gate.sh exits 4 so that AUTORUN_GATE_STRICT=1 can stop it).
 4. gate.log opens with the summary table: test | calibrated by | budget | nodes passed | nodes failed | verdict.
 5. a node is excluded only by a calibrated test, with the reason "below threshold" or "error", or by the majority rule below.
 6. THE MAJORITY RULE (2026-10-09 evening): a node whose row says "timeout" or "no output" IS excluded when the same test passed
    within its budget on at least 90 percent of the OTHER nodes (the test has proven itself healthy in this job, so the silence
    is the node's own); otherwise the old rule holds: a timeout is never a node's fault alone. "Of the other nodes" means the
    denominator is K - 1: at 260 nodes it is the same as 90 percent of all nodes; at 4 nodes with one timeout it reads 3 of 3,
    so the one silent node is excluded (the 4-node test job relies on this reading).
 7. THE PAIR TEST'S NAMING RULES (gate_net.py: a two-rank all-reduce in two rounds of disjoint pairs; a rank that waits past
    the deadline writes the partner node's name): a node named by partners in BOTH rounds is excluded ("did not answer two
    different partners"); a node named in ONE round is not excluded by itself: the pair goes to the switch test's card counters:
    the node whose cards show link errors or replays is excluded ("link errors on its cards, named by a partner"); when neither
    node shows them, both are listed in gate.log as a SUSPECT PAIR and excluded only with GATE_EXCLUDE_SUSPECT_PAIR=1 and when
    the spares allow it (--spares). A pair value below 0.8 x the calibrated GB/s is "below threshold" as for every rate test.
 8. TWO TIERS OF EXCLUSION (2026-10-09 late evening, the user: a slow node "can still work and is better than node failures or
    no nodes to be used"): a node excluded ONLY by a performance test (gemm, xgmi_a2a, pair, switch, rdma) with the reason
    "below threshold" is DEGRADED: the restart loop keeps it as a spare of last resort (AUTORUN_USE_DEGRADED=1). A node excluded
    for an error (does not answer, uncorrectable memory errors, xGMI errors, link errors or replays, a version mismatch, no GCDs,
    wrong data, a timeout under the majority rule) is DEAD and is never used. Both tiers are written to excluded_reasons.tsv.

--calibrate: before judging, every test that ran (status ok) on at least half of the nodes gets a new calibration line (the
baseline from this job: the median for a throughput, the majority for a version, the fixed expectation for a check; the run time =
the slowest node's); the old line moves to gate_baseline_history.tsv. The nodes are then judged against it.
Output: <data>/gate_report.tsv (one row per node per test), <data>/gate/failed_nodes.txt (every excluded node),
<data>/gate/excluded_reasons.tsv (node, tier, test, value, threshold, sentence), <job>/gate.log.
"""
import argparse, collections, fcntl, glob, math, os, re, statistics, sys, time

# THE TEST TABLE: one entry per test, read by gate.sh (the budgets) and by this program (the rules and gate.log).
#   kind: 'perf' (value >= factor x baseline), 'equal' (value == baseline), 'expect' (a check whose expectation is fixed by the
#         test), 'info' (recorded, never judged).
#   default_budget: the budget in seconds when the test is not calibrated; extra_budget: seconds added to the calibrated budget
#         (the pair test's deadlines and done-file waits are paid only when a rank is silent, so they are not in its run time).
#   what: one sentence for gate.log; command: the EXACT command the test runs on the node (printed under every test's header,
#         so that a reader of one block knows what was measured and how).
TESTS = collections.OrderedDict([
    ('answers_srun',   dict(kind='expect', default_budget=60,  what='the node runs the gate step (re-probed with srun hostname when it wrote no row)',
                            command='srun -N1 -n1 -w <node> --overlap hostname -s   (only for a node whose gate step wrote no row)')),
    ('rocm_smi_ids',   dict(kind='expect', default_budget=30,  what='rocm-smi --showid: the GCDs that answer (8 expected)',
                            command='rocm-smi --showid   (counts the distinct GPU[k] lines)')),
    ('ras_ue',         dict(kind='expect', default_budget=30,  what='rocm-smi --showrasinfo: uncorrectable ECC errors (0 expected)',
                            command="rocm-smi --showrasinfo all   (the sum of the uncorrectable column over every block of every GCD)")),
    ('xgmi_err',       dict(kind='expect', default_budget=30,  what='rocm-smi --showxgmierr: GCDs with xGMI errors (0 expected)',
                            command='rocm-smi --showxgmierr   (counts the GCDs whose XGMI Error count is above 0)')),
    ('rocm_version',   dict(kind='equal',  default_budget=10,  what='the ROCm version string, equal on every node',
                            command='head -1 $ROCM_PATH/.info/version')),
    ('rccl_md5',       dict(kind='equal',  default_budget=40,  what='librccl.so of the env: md5 of the first 16 MiB + the size, equal on every node',
                            command='head -c 16777216 <site-packages>/torch/lib/librccl.so | md5sum; stat -Lc %s <the same file>')),
    ('ofi_plugin_md5', dict(kind='equal',  default_budget=40,  what='the aws-ofi-rccl plugin: md5 of the first 16 MiB + the size, equal on every node',
                            command='head -c 16777216 <aws-ofi-rccl lib>/librccl-net.so | md5sum; stat -Lc %s <the same file>')),
    ('libfabric',      dict(kind='equal',  default_budget=15,  what='the libfabric version, equal on every node',
                            command='fi_info --version | head -1   (else the realpath of libfabric.so.1 on LD_LIBRARY_PATH)')),
    ('shm_free_gb',    dict(kind='expect', default_budget=10,  what='/dev/shm free GB (more than 1 GB expected)',
                            command='df -Pk /dev/shm   (the available kilobytes / 1048576)')),
    ('gemm',           dict(kind='perf',   default_budget=300, factor=0.9, unit='TFLOP/s',
                            what='bf16 matmul per GCD, one process per GCD, timed with CUDA events; the value is the slowest GCD of the node',
                            command='python gate_gemm.py: torch.matmul of two 8192 x 8192 bf16 matrices, 3 warm-ups + 20 timed, one forked process per GCD, all 8 start together at a barrier, timed with CUDA events')),
    ('xgmi_a2a',       dict(kind='perf',   default_budget=120, factor=0.8, unit='GB/s',
                            what='rccl-tests alltoall_perf over the 8 GCDs of the node at 256 MB; the value is the busbw',
                            command='alltoall_perf -b 256M -e 256M -f 2 -g 8 -w 3 -n 10 -c 0   (one process driving 8 GCDs, MPI singleton)')),
    # [2026-10-09 evening] the cross-node tests (gate_net.py, one srun step over all gate nodes, 8 tasks per node; rows in <node>.net.tsv)
    ('pair',           dict(kind='perf',   default_budget=300, extra_budget=180, factor=0.8, unit='GB/s',
                            what='the pair test: a two-rank all-reduce in two rounds of disjoint node pairs, each round in two sub-rounds (the even GCDs 0, 2, 4, 6, then the odd GCDs 1, 3, 5, 7: GCD k of node a with GCD k of node b, so that every network card carries one stream per sub-round and all 8 GCDs are tested); the value is the node\'s slowest GCD in GB/s (busbw = bytes / time); a rank that waits past the deadline names its partner node',
                            command='python gate_net.py --phase net: per round, per sub-round (even GCDs, then odd GCDs), per GCD, dist.new_group of the two ranks (GCD k of both nodes), 3 warm-up + 20 timed dist.all_reduce of 64 MiB fp32 (async_op=True, polled with is_completed() under a deadline of GATE_PAIR_DEADLINE_S = 30 s each), the result checked against the known sum; rounds separated by done files (60-s wait); NCCL_NET_GDR_LEVEL as the training run has it')),
    ('rdma',           dict(kind='perf',   default_budget=120, factor=0.8, unit='GB/s', optional=True,
                            what='the RDMA test (only with GATE_RDMA_PHASE=1): the old NIC-to-NIC ring transfer with GPU-direct requested (NCCL_NET_GDR_LEVEL=PHB); the detail names the path RCCL took; the value is the slowest NIC in GB/s',
                            command='python gate_net.py --phase gdr: dist.batch_isend_irecv of 64 MB bf16 x 20 to the ring-neighbour node on GCDs 0, 2, 4, 6, with NCCL_NET_GDR_LEVEL=PHB')),
    ('switch',         dict(kind='perf',   default_budget=120, factor=0.8, unit='GB/s',
                            what='the switch test: one all-to-all over every gate node x 8 GCDs with the Slingshot NIC (cxi) counters read before and after; the value is the node\'s slowest rank busbw; link errors or replays (cxi_err above 0) name the node',
                            command='python gate_net.py --phase net (after the pair rounds): dist.all_to_all_single of 64 MiB bf16 per rank, 3 warm-ups + 10 timed with CUDA events, over the ranks that survived the pair rounds; /sys/class/cxi/*/device/telemetry read before and after')),
    ('counters_after', dict(kind='expect', default_budget=10,  what='the uncorrectable ECC and xGMI counters after the tests (no growth expected)',
                            command='cat /sys/class/drm/card*/device/ras/umc_err_count and /sys/class/drm/card*/device/xgmi_error, before and after the tests')),
    ('dmesg',          dict(kind='info',   default_budget=20,  what='kernel log lines about hardware errors (dmesg is not permitted for users on Frontier)',
                            command='dmesg -T | grep -i -E "mce|ecc|xgmi|hardware error|kfd|amdgpu.*(fail|error)"')),
    ('gpu_direct_env', dict(kind='info',   default_budget=5,   what='the GPU-direct environment variables (RCCL decides at its first communicator; see the training.log headers)',
                            command='echo NCCL_NET_GDR_LEVEL NCCL_DMABUF_ENABLE FI_HMEM, and whether the plugin directory is on LD_LIBRARY_PATH')),
    ('counters_before', dict(kind='info',  default_budget=10,  what='the counters before the tests',
                            command='cat /sys/class/drm/card*/device/ras/umc_err_count and /sys/class/drm/card*/device/xgmi_error')),
])
EXPECT = {'answers_srun': 'yes', 'rocm_smi_ids': '8', 'ras_ue': '0', 'xgmi_err': '0', 'shm_free_gb': '> 1', 'counters_after': 'same'}
NET_TESTS = {'pair', 'rdma', 'switch'}          # the cross-node step: rows in <node>.net.tsv; they need 2 nodes or more
FULL_ONLY = {'gemm', 'xgmi_a2a'} | NET_TESTS
PER_UNIT = {'gemm': 'GCDs', 'pair': 'GCDs', 'rdma': 'NICs'}   # tests whose detail carries one value per unit (k=v,k=v in the 2nd ';' field)
PERF_TESTS = {t for t, d in TESTS.items() if d['kind'] == 'perf'}
MAJORITY_SHARE = 0.9     # rule 6: the share of the OTHER nodes that must have passed for a timeout to be the node's own


def budget_of(test, cal):
    """The test's budget in seconds: 2 x its calibrated run time + 30 s (the run time is measured by the same code), plus the
    test's extra_budget when it has one (the pair test's deadlines and waits); else its default budget."""
    c = cal.get(test)
    extra = TESTS[test].get('extra_budget', 0)
    if c and c.get('run_s') not in (None, '', '-'):
        return int(math.ceil(2.0 * float(c['run_s']) + 30)) + extra
    return TESTS[test]['default_budget']


def read_cal(path):
    cal = {}
    try:
        for line in open(path):
            if line.startswith('#') or line.startswith('test\t') or not line.strip():
                continue
            v = line.rstrip('\n').split('\t')
            if len(v) >= 6 and v[2] not in ('', '-') and v[3] not in ('', '-') and v[4] not in ('', '-'):
                cal[v[0]] = dict(baseline=v[1], run_s=v[2], date=v[3], job=v[4], nodes=v[5], note=v[6] if len(v) > 6 else '')
    except FileNotFoundError:
        pass
    return cal


def numeric(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def cxi_err_of(row):
    """The link errors + replays of a node during the switch test, from its switch row's detail ('cxi_err=N'); -1 when unknown."""
    if not row:
        return -1
    m = re.search(r'cxi_err=(-?\d+)', row.get('detail', ''))
    return int(m.group(1)) if m else -1


def named_by_of(detail):
    """The pair row's naming fields: named_in=N and named_by=A(round 1, timeout),B(round 2, gone) -> (N, [(node, round), ...]).
    The old form A(round 1) (no kind) is read too. The kinds are returned by named_kinds_of."""
    return _named(detail)[0:2]


def _named(detail):
    m = re.search(r'named_in=(\d+)', detail or '')
    n = int(m.group(1)) if m else 0
    by, kinds = [], []
    m2 = re.search(r'named_by=([^;]*)', detail or '')
    if m2:
        # whole items are matched (not a split on commas: the kind sits inside the parentheses, "n0(round 1, timeout)")
        for nd, r, k in re.findall(r'([^\s,()]+)\(round (\d)(?:, *(\w+))?\)', m2.group(1)):
            by.append((nd, int(r)))
            kinds.append(k or 'timeout')
    return n, by, kinds


def named_kinds_of(detail):
    """[(node, round, kind)] with kind 'timeout' or 'gone' (2026-10-10: a gone partner counts like a timeout; the kind explains it)."""
    n, by, kinds = _named(detail)
    return [(nd, r, k) for (nd, r), k in zip(by, kinds)]


def own_timeouts_of(detail):
    """The nodes THIS node named as not answering, from its pair row's 'per result' list: [(round, partner node, kind)], where the
    kind is 'timeout' or 'gone' (the text 'round 1.1 GCD 0 with frontier04379: timeout (...)')."""
    out = []
    for m in re.finditer(r'round (\d)\.\d GCD \d+ with (\S+): (timeout|gone)', detail or ''):
        out.append((int(m.group(1)), m.group(2), m.group(3)))
    return out


def scale_off(c, K):
    """True when the calibration's node count and this job's differ by more than a factor of 4 (the scale guard; see the verdict loop)."""
    try:
        n = int(str(c.get('nodes', '')).strip())
    except ValueError:
        return False
    return n > 0 and (K > 4 * n or n > 4 * K)


def judge_one(test, row, cal):
    """-> (value, baseline, threshold, pass(bool|None), reason). The generic rules (rule 2); the pair test's naming rules are
    applied afterwards in judge_pair, because they need the rows of other nodes and of the switch test."""
    kind = TESTS[test]['kind']
    if row is None:
        return '-', '-', '-', False, 'no output'
    value, status, detail = row['value'], row['status'], row['detail']
    c = cal.get(test)
    base = c['baseline'] if c else '-'
    if status == 'timeout':
        return value, base, '-', False, 'timeout'
    if status == 'missing':
        return value, base, '-', False, 'no output'
    if status == 'not tested':    # the switch test's full-node rule (gate_net.py): the node lost a rank in the pair rounds, its pair rows decide
        return value, base, '-', None, 'not tested: ' + (detail.split(';')[0][:200] if detail else 'the test did not run on this node')
    if status == 'error':
        return value, base, '-', False, f'error {detail[:160]}'
    if kind == 'info':
        return value, '-', '-', None, 'informational'
    if kind == 'expect':
        exp = EXPECT[test]
        if test == 'shm_free_gb':
            v = numeric(value)
            return value, exp, '> 1 GB', (v is not None and v > 1), ('' if (v is not None and v > 1) else 'below threshold')
        if test in ('ras_ue', 'xgmi_err') and value == 'unparsed':
            return value, exp, '= ' + exp, False, 'no output'      # rocm-smi answered, but with no table this code can read
        if test == 'counters_after' and value == 'unreadable':
            return value, exp, 'no growth', None, 'informational (no counter files)'
        if value == exp:
            return value, exp, '= ' + exp, True, ''
        if test == 'rocm_smi_ids':
            return value, exp, '= 8', False, 'below threshold'
        why = {'ras_ue': f'error {value} uncorrectable ECC errors', 'xgmi_err': f'error {value} GCDs report xGMI errors',
               'counters_after': f'error the counters grew during the gate ({detail})', 'answers_srun': f'error {detail}'}
        return value, exp, '= ' + exp, False, why.get(test, f'error {value}')
    if kind == 'equal':
        if not c:
            return value, '-', '= baseline', None, ''
        ok = value == c['baseline']
        return value, c['baseline'], '= baseline', ok, '' if ok else f'error {value} differs from the baseline {c["baseline"]}'
    if kind == 'perf':
        v = numeric(value)
        b = numeric(c['baseline']) if c else None
        thr_txt = f'>= {TESTS[test]["factor"] * b:.1f}' if b is not None else '-'
        if test == 'switch':   # the NIC link counters: link errors or replays during the all-to-all name the node (an error, not a rate)
            m = re.search(r'cxi_err=(\d+)', detail)
            if m and int(m.group(1)) > 0:
                return value, base, thr_txt, False, f'error {m.group(1)} link errors/replays on the NICs of this node during the all-to-all ({detail.split(";", 1)[-1].strip()[:140]})'
        if not c:
            return value, '-', '-', None, ''
        thr = TESTS[test]['factor'] * b
        if v is None:
            return value, c['baseline'], f'>= {thr:.1f}', False, f'error no number ({detail[:120]})'
        return value, c['baseline'], f'>= {thr:.1f}', v >= thr, '' if v >= thr else 'below threshold'
    return value, base, '-', None, ''


def judge_pair(nodes, rows, cal, V):
    """Rule 7, applied after judge_one filled V['pair']: the naming rules of the pair test. Returns the list of suspect pairs
    [(node, partner, round)] for gate.log; it rewrites V['pair'][node] where a naming rule decides."""
    suspects = []
    # the nodes named in two rounds (excluded as dead below) are known first, so that an EXPLAINED departure can refer to them:
    # (2026-10-10) a healthy rank that timed out on a dead node leaves the test (by design, so that it cannot be trapped), and its
    # round-2 partner then records it as 'gone'; that single 'gone' naming is explained by the dead node and is not a suspect pair
    dead_named = set()
    for n in nodes:
        row = rows[n].get('pair')
        if row is not None and len(set(r for _, r in named_by_of(row['detail'])[1])) >= 2:
            dead_named.add(n)
    for n in nodes:
        row = rows[n].get('pair')
        if row is None:
            continue
        val, base, thr, ok, why = V['pair'][n]
        named_in, by = named_by_of(row['detail'])
        kinds = {(p, r): k for p, r, k in named_kinds_of(row['detail'])}
        rounds = sorted(set(r for _, r in by))
        if len(rounds) >= 2:
            partners = ', '.join(f'{p} in round {r} ({kinds.get((p, r), "timeout")})' for p, r in by)
            V['pair'][n] = (val, base, thr, False, f'error did not answer two different partners ({partners})')
            continue
        if len(rounds) == 1 and by:
            partner, r = by[0]
            kind = kinds.get((partner, r), 'timeout')
            if kind == 'gone':
                mine = [(r0, p0, k0) for r0, p0, k0 in own_timeouts_of(row['detail']) if r0 < r and p0 in dead_named]
                if mine:
                    r0, p0, k0 = mine[0]
                    if ok is not False:
                        V['pair'][n] = (val, base, thr, ok, (why + '; ' if why else '') +
                                        f'named once by {partner} (round {r}, gone) because it had left the test after naming {p0} in round {r0}, '
                                        f'which is excluded as dead: an explained departure, not a suspect pair')
                    continue
            my_err = cxi_err_of(rows[n].get('switch'))
            partner_err = cxi_err_of(rows.get(partner, {}).get('switch'))
            if my_err > 0:
                V['pair'][n] = (val, base, thr, False, f'error link errors on its cards ({my_err} link errors or replays during the switch all-to-all), named by a partner ({partner} in round {r})')
                continue
            if partner_err > 0:
                # the partner's own switch row carries the link errors; the partner is excluded by the switch rule, this node is not named for it
                if ok is not False:
                    V['pair'][n] = (val, base, thr, ok, (why + '; ' if why else '') + f'named once by {partner} (round {r}), whose own cards show {partner_err} link errors or replays: the partner is the suspect')
                continue
            if ok is not False:
                suspects.append((n, partner, r))
                V['pair'][n] = (val, base, thr, ok, (why + '; ' if why else '') + f'suspect pair with {partner}: named once (round {r}), no link errors on either node')
    return suspects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='folder of <node>.raw.tsv')
    ap.add_argument('--nodes', required=True)
    ap.add_argument('--mode', default='quick')
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--report', required=True, help='gate_report.tsv')
    ap.add_argument('--failed', required=True, help='failed_nodes.txt (every excluded node); excluded_reasons.tsv is written next to it')
    ap.add_argument('--log', required=True, help='gate.log')
    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--job', default=os.environ.get('SLURM_JOB_ID', 'nojob'))
    ap.add_argument('--dry-rc', default='')
    ap.add_argument('--dry-log', default='')
    ap.add_argument('--t0', type=float, default=time.time())
    ap.add_argument('--budgets', default='', help='test=s,... the budgets used')
    ap.add_argument('--rdma', action='store_true', help='the optional rdma phase ran (GATE_RDMA_PHASE=1): judge its rows')
    ap.add_argument('--spares', type=int, default=0, help='the spare nodes of the job: a suspect pair is excluded only when 2 spares are left for it')
    ap.add_argument('--exclude-suspect-pair', action='store_true', help='GATE_EXCLUDE_SUSPECT_PAIR=1: exclude both nodes of a suspect pair when the spares allow')
    a = ap.parse_args()
    nodes = a.nodes.split()
    K = len(nodes)
    tests = [t for t in TESTS if a.mode == 'full' or t not in FULL_ONLY]
    if not a.rdma:
        tests = [t for t in tests if not TESTS[t].get('optional')]     # the rdma phase runs only with GATE_RDMA_PHASE=1
    if K < 2:
        tests = [t for t in tests if t not in NET_TESTS]     # the cross-node step needs a partner; one node is not a defect
    rows = collections.defaultdict(dict)       # node -> test -> row
    for n in nodes:
        for p in (os.path.join(a.raw, f'{n}.raw.tsv'), os.path.join(a.raw, f'{n}.probe.tsv'), os.path.join(a.raw, f'{n}.net.tsv')):
            try:
                for line in open(p):
                    v = line.rstrip('\n').split('\t')
                    if len(v) >= 6:
                        rows[n][v[2]] = dict(time=v[0], value=v[3], run_s=v[4], status=v[5], detail=v[6] if len(v) > 6 else '')
            except FileNotFoundError:
                pass
    budgets = dict(x.split('=') for x in a.budgets.split(',') if '=' in x)
    # ---- calibration (rule 1): measured by this code on these nodes, with the date and the job id
    cal_new = []
    if a.calibrate:
        with open(a.baseline, 'a+') as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.seek(0)
            old_lines = fh.read().split('\n')
            old = read_cal(a.baseline)
            new = {}
            for t in tests:
                if TESTS[t]['kind'] == 'info':
                    continue
                ok = [rows[n][t] for n in nodes if t in rows[n] and rows[n][t]['status'] == 'ok'
                      and rows[n][t]['value'] not in ('unparsed', 'unreadable', '-', '')]     # a test that works gives a value
                if len(ok) * 2 < K or not ok:
                    continue
                kind = TESTS[t]['kind']
                if kind == 'perf':
                    if t in PER_UNIT:    # the median over every unit (GCD, card, NIC) of every node (the detail's 2nd field holds unit=v,...)
                        vals = [float(x.split('=')[1]) for r in ok for x in r['detail'].split(';')[1].strip().split(',')
                                if '=' in x and numeric(x.split('=')[1]) is not None]
                    else:
                        vals = [float(r['value']) for r in ok if numeric(r['value']) is not None]
                    if not vals:
                        continue
                    base = '%.1f' % statistics.median(vals)
                    note = f'median of {len(vals)} values ({PER_UNIT.get(t, "nodes")}), min {min(vals):.1f}, max {max(vals):.1f}'
                elif kind == 'equal':
                    base, cnt = collections.Counter(r['value'] for r in ok).most_common(1)[0]
                    note = f'majority value ({cnt} of {len(ok)} nodes)'
                else:
                    base = EXPECT[t]; note = 'fixed expectation of the test; run time measured'
                run_s = max(numeric(r['run_s']) or 0.0 for r in ok)
                new[t] = dict(baseline=base, run_s='%.1f' % run_s, date=time.strftime('%Y-%m-%d'), job=a.job, nodes=str(len(ok)), note=note)
                cal_new.append(t)
            keep = [l for l in old_lines if l.strip() and not l.startswith('test\t') and l.split('\t')[0] not in new]
            hist = os.path.join(os.path.dirname(a.baseline), 'gate_baseline_history.tsv')
            with open(hist, 'a') as hf:
                for l in old_lines:
                    if l.strip() and not l.startswith('test\t') and l.split('\t')[0] in new:
                        hf.write(f'{l}\treplaced by job {a.job} on {time.strftime("%Y-%m-%d %H:%M")}\n')
            fh.seek(0); fh.truncate()
            fh.write('test\tbaseline\trun_s\tdate\tjob\tnodes\tnote\n')
            for l in keep:
                fh.write(l + '\n')
            for t, c in new.items():
                fh.write('\t'.join([t, c['baseline'], c['run_s'], c['date'], c['job'], c['nodes'], c['note']]) + '\n')
            fcntl.flock(fh, fcntl.LOCK_UN)
    cal = read_cal(a.baseline)
    # ---- one verdict per node per test (rule 2), then the pair test's naming rules (rule 7)
    V = collections.defaultdict(dict)
    for t in tests:
        for n in nodes:
            V[t][n] = judge_one(t, rows[n].get(t), cal)
    suspects = judge_pair(nodes, rows, cal, V) if 'pair' in tests else []
    # ---- per-test verdicts (rules 3, 5, 6 and 8)
    summary, excluded = [], collections.defaultdict(list)
    tier = {}            # node -> 'dead' | 'degraded'  (rule 8; dead wins over degraded)
    sentences = collections.defaultdict(list)   # node -> the sentences of its exclusions (for excluded_reasons.tsv and the failure record)
    majority_notes = {}

    def exclude(n, t, why, kind_tier, val, thr):
        excluded[n].append(t)
        tier[n] = 'dead' if (tier.get(n) == 'dead' or kind_tier == 'dead') else 'degraded'
        sentences[n].append((kind_tier, t, str(val), str(thr), why))

    not_launched = None                     # [2026-10-11, sandbox] net_launch_retry.sh's marker: (tries, Slurm's words)
    try:
        _nl = open(os.path.join(a.raw, 'net', 'not_launched')).read().strip().split('\t', 1)
        not_launched = (_nl[0], _nl[1] if len(_nl) > 1 else 'no message')
    except OSError:
        pass
    for t in tests:
        kind = TESTS[t]['kind']
        npass = sum(1 for n in nodes if V[t][n][3] is True)
        nfail = sum(1 for n in nodes if V[t][n][3] is False)
        nout = sum(1 for n in nodes if t in rows[n] and rows[n][t]['status'] == 'ok' and V[t][n][4] != 'no output')
        reasons = collections.Counter(V[t][n][4].split(' ')[0] if V[t][n][4].startswith('error') else V[t][n][4]
                                      for n in nodes if V[t][n][3] is False)
        rtxt = ', '.join(f'{k} {v}' for k, v in reasons.most_common())
        nnt = sum(1 for n in nodes if t in rows[n] and rows[n][t]['status'] == 'not tested')
        if kind == 'info':
            verdict = 'informational'
        elif nout == 0 and nnt and nnt + nfail == K:
            verdict = (f'not run: the test was not run on any node ({nnt} not tested' + (f', {nfail} failed: {rtxt}' if nfail else '') +
                       '); excludes nobody (' + next(V[t][n][4] for n in nodes if V[t][n][3] is None and V[t][n][4].startswith('not tested'))[:220] + ')')
        elif nout == 0 and t in NET_TESTS and not_launched:
            # [2026-10-11, sandbox] Slurm did not start the network step (net/not_launched from net_launch_retry.sh): the test never ran,
            # so it is not a defect of the test and says nothing about the nodes
            verdict = f'NOT LAUNCHED: Slurm did not start the network step after {not_launched[0]} tries ({not_launched[1]}); excludes nobody'
        elif nout == 0:
            verdict = f'TEST DEFECT: no output on any of the {K} nodes ({rtxt}); excludes nobody; marked for repair'
        elif K >= 2 and nfail * 2 > K:
            verdict = f'TEST DEFECT: failed on {nfail} of {K} nodes (more than half; {rtxt}); excludes nobody; marked for repair'
        elif t not in cal:
            verdict = 'not calibrated: informational' + (f' ({nfail} of {K} would fail: {rtxt})' if nfail else '')
        elif t in NET_TESTS and scale_off(cal[t], K):
            # THE SCALE GUARD (2026-10-10, job 5646880): the pair and switch rates depend on how many nodes take part (260 nodes: the
            # pair test measured 9.3-13.9 GB/s and the 3-node baseline of 12.1 with its 0.8 factor marked 8 nodes degraded; the
            # switch all-to-all over 40 nodes gave 0.47 against a 3-node baseline of 11.2). A baseline from a job whose node count is
            # more than 4x away from this one is not applied: the values are recorded, nobody is excluded, and with calibration on
            # this job writes the baseline for its own scale.
            verdict = (f'calibrated at {cal[t]["nodes"]} node(s) by job {cal[t]["job"]} and this job has {K}: the threshold is not applied '
                       f'(the rate depends on the scale); recorded only' + (f'; {nfail} of {K} would fail: {rtxt}' if nfail else ''))
        else:
            ex = [n for n in nodes if V[t][n][3] is False and (V[t][n][4] == 'below threshold' or V[t][n][4].startswith('error'))]
            soft = [n for n in nodes if V[t][n][3] is False and n not in ex]
            # rule 6, the majority rule: the test passed on at least 90 percent of the OTHER nodes -> the silence is the node's own
            maj = []
            if soft and K >= 2 and npass >= MAJORITY_SHARE * (K - 1):
                maj = soft; soft = []
                majority_notes[t] = (f'the majority rule applies: the test passed on {npass} of the other {K - 1} nodes '
                                     f'(at least {int(MAJORITY_SHARE * 100)} percent), so a timeout or no output is the node\'s own')
            for n in ex:
                val, base, thr, ok, why = V[t][n]
                kt = 'degraded' if (t in PERF_TESTS and why == 'below threshold') else 'dead'
                exclude(n, t, f'{t}: {why} (value {val}, {"threshold " + thr if thr != "-" else "no threshold"})', kt, val, thr)
            for n in maj:
                val, base, thr, ok, why = V[t][n]
                exclude(n, t, f'{t}: {why}; excluded by the majority rule ({majority_notes[t]})', 'dead', val, thr)
            verdict = ('ok' if not (ex or maj) else 'excluded: ' + ' '.join(ex + maj)) + \
                      (f' ({len(maj)} of them by the majority rule: {" ".join(maj)})' if maj else '') + \
                      (f'; {len(soft)} node(s) with timeout/no output, not excluded (the test passed on fewer than {int(MAJORITY_SHARE * 100)} percent of the other nodes, so the silence is not proven to be the node\'s own): {" ".join(soft)}' if soft else '')
        c = cal.get(t)
        summary.append((t, (f'{c["job"]} {c["date"]}' + (' (this job)' if t in cal_new else '')) if c else 'no',
                        budgets.get(t, str(budget_of(t, cal))), npass, nfail, verdict))
    # the suspect pairs (rule 7): listed always; excluded only on request and when the spares allow it
    suspect_lines = []
    spares_left = a.spares - len(excluded)
    for n, partner, r in suspects:
        line = f'SUSPECT PAIR {n} and {partner}: {partner} named {n} as not answering in round {r} only, and neither node shows link errors or replays on its cards'
        if a.exclude_suspect_pair and spares_left >= 2 and n not in excluded and partner not in excluded:
            for x in (n, partner):
                val, base, thr, ok, why = V['pair'][x]
                exclude(x, 'pair', f'pair: suspect pair ({n}, {partner}), named once in round {r} with no link errors on either node; excluded because GATE_EXCLUDE_SUSPECT_PAIR=1 and {spares_left} spares were left', 'dead', val, thr)
            spares_left -= 2
            line += f'; BOTH EXCLUDED (GATE_EXCLUDE_SUSPECT_PAIR=1, spares allowed it)'
        else:
            line += '; not excluded' + (' (GATE_EXCLUDE_SUSPECT_PAIR=1 but fewer than 2 spares are left)' if a.exclude_suspect_pair else ' (set GATE_EXCLUDE_SUSPECT_PAIR=1 to exclude such pairs when the spares allow)')
        suspect_lines.append(line)
    # ---- the report table
    with open(a.report, 'w') as fh:
        fh.write('node\ttest\tvalue\tbaseline\tthreshold\tpass\treason\trun_s\tcalibrated\tstatus\tdetail\n')
        for t in tests:
            for n in nodes:
                val, base, thr, ok, why = V[t][n]
                r = rows[n].get(t) or {}
                fh.write('\t'.join([n, t, str(val), str(base), str(thr), {True: 'yes', False: 'no', None: ('not tested' if (why or '').startswith('not tested') else 'info')}[ok], why or '-',
                                    r.get('run_s', '-'), 'yes' if t in cal else 'no', r.get('status', 'none'), (r.get('detail') or '-')[:300]]) + '\n')
    with open(a.failed, 'w') as fh:
        for n in nodes:
            if n in excluded:
                fh.write(n + '\n')
    # one sentence per excluded node, with its tier (rule 8): the restart loop reads this file for the failure record and the spare list
    with open(os.path.join(os.path.dirname(a.failed), 'excluded_reasons.tsv'), 'w') as fh:
        fh.write('node\ttier\ttest\tvalue\tthreshold\tsentence\n')
        for n in nodes:
            if n not in excluded:
                continue
            kinds = sentences[n]
            first = kinds[0]
            sent = f'the gate excluded {n} as {tier[n]}: ' + '; '.join(s[4] for s in kinds)
            fh.write('\t'.join([n, tier[n], first[1], first[2], first[3], sent.replace('\t', ' ')]) + '\n')
    # ---- gate.log (rule 4: the summary first)
    defects = [s for s in summary if s[5].startswith('TEST DEFECT')]
    notl = [s for s in summary if s[5].startswith('NOT LAUNCHED')]
    W = []
    W.append(f'GATE of job {a.job}: the gate ran in mode {a.mode} on {K} nodes from {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(a.t0))} to '
             f'{time.strftime("%H:%M:%S")} ({time.time() - a.t0:.0f} s); the calibration file is {a.baseline}')
    if a.calibrate:
        W.append(f'CALIBRATION in this job (AUTORUN_GATE_CALIBRATE=1): {len(cal_new)} test(s) were calibrated on these {K} nodes: {", ".join(cal_new) or "none"}')
    W.append('')
    W.append('SUMMARY: one row per test; the columns are the test, the job and date that calibrated it, its budget in seconds, the number of nodes that passed, the number that failed, and the verdict')
    hdr = ('test', 'calibrated by (job, date)', 'budget (s)', 'nodes passed', 'nodes failed', 'verdict')
    w = [max(len(str(x[i])) for x in summary + [hdr]) for i in range(5)]
    W.append(' | '.join(str(h).ljust(w[i]) if i < 5 else h for i, h in enumerate(hdr)))
    W.append('-+-'.join('-' * w[i] for i in range(5)) + '-+-' + '-' * 20)
    if a.dry_rc != '':
        W.append(' | '.join(['dry_run'.ljust(w[0]), '-'.ljust(w[1]), '300'.ljust(w[2]), ('1' if a.dry_rc == '0' else '0').ljust(w[3]),
                             ('0' if a.dry_rc == '0' else '1').ljust(w[4]),
                             'ok (exit code 0, on the batch node)' if a.dry_rc == '0' else f'FAILED with exit code {a.dry_rc} ({a.dry_log}); a run-level failure, no node is excluded']))
    for s in summary:
        W.append(' | '.join(str(x).ljust(w[i]) if i < 5 else str(x) for i, x in enumerate(s)))
    W.append('')
    deg = [n for n in nodes if tier.get(n) == 'degraded']
    dead = [n for n in nodes if tier.get(n) == 'dead']
    W.append('nodes excluded by the gate: ' + (', '.join(f'{n} ({" ".join(ts)})' for n, ts in excluded.items()) or 'none'))
    W.append('excluded as degraded (a performance test below its threshold; usable as spares of last resort when no healthy spare is left, AUTORUN_USE_DEGRADED=1): ' +
             (', '.join(f'{n} ({s[1]} {s[2]} against {s[3]})' for n in deg for s in sentences[n][:1]) or 'none'))
    W.append('excluded as dead (an error, or a timeout under the majority rule; never used): ' +
             (', '.join(f'{n} ({sentences[n][0][1]})' for n in dead) or 'none'))
    W.append('suspect pairs: ' + ('; '.join(suspect_lines) if suspect_lines else 'none'))
    W.append('TEST DEFECTS: ' + ('; '.join(s[0] for s in defects) + ' (the job continues unless AUTORUN_GATE_STRICT=1)' if defects else 'none'))
    W.append('NOT LAUNCHED: ' + ('; '.join(s[0] for s in notl) + ' (Slurm did not start the network step; the job continues unless AUTORUN_GATE_STRICT=1)' if notl else 'none'))
    for t, note in majority_notes.items():
        W.append(f'majority rule on {t}: {note}')
    W.append('')
    W.append('DETAIL per test, the failed nodes first. Under every test: what it measures, the exact command it runs, its budget and calibration, then one line per node.')
    for t in tests:
        c = cal.get(t)
        W.append('')
        W.append(f'{t}: {TESTS[t]["what"]}')
        W.append(f'  command: {TESTS[t].get("command", "-")}')
        W.append(f'  budget: {budgets.get(t, budget_of(t, cal))} s ' +
                 (f'(2 x the calibrated run time of {c["run_s"]} s + 30 s' + (f' + {TESTS[t]["extra_budget"]} s for the deadlines and waits' if TESTS[t].get('extra_budget') else '') +
                  f'); calibrated by job {c["job"]} on {c["date"]} over {c["nodes"]} nodes; baseline {c["baseline"]} ({c["note"]})'
                  if c else '(the default: the test is not calibrated, so it is informational and excludes nobody)'))
        nums = [(numeric(V[t][n][0]), n) for n in nodes if numeric(V[t][n][0]) is not None]
        if TESTS[t]['kind'] == 'perf' and nums and any(t in rows[n] for n in nodes):
            lo = min(nums)
            W.append(f'  the worst node is {lo[1]} at {lo[0]:.1f} {TESTS[t].get("unit", "")}; the run times span '
                     f'{min(numeric(rows[n][t]["run_s"]) or 0 for n in nodes if t in rows[n]):.1f}-{max(numeric(rows[n][t]["run_s"]) or 0 for n in nodes if t in rows[n]):.1f} s')
        W.append(f'  {"node":16s} {"measured value":>12} | {"baseline":>10} | {"passes if":>12} | {"verdict":4} | reason | run time (s)')
        order = sorted(nodes, key=lambda n: (V[t][n][3] is not False, n))
        shown = 0
        for n in order:
            val, base, thr, ok, why = V[t][n]
            if ok is False or K <= 16 or shown < 3:
                r = rows[n].get(t) or {}
                W.append(f'  {n:16s} {str(val)[:40]:>12} | {str(base)[:24]:>10} | {thr:>12} | {({True: "pass", False: "FAIL", None: ("n/t" if (why or "").startswith("not tested") else "info")})[ok]:4} | '
                         f'{why or "-"} | {r.get("run_s", "-")}')
                shown += 1
        if shown < K:
            W.append(f'  ... {K - shown} more passing nodes are in data/gate_report.tsv')
    open(a.log, 'w').write('\n'.join(W) + '\n')
    print('\n'.join(W[:len(summary) + 16]))
    sys.exit(4 if (defects or notl) else 0)


if __name__ == '__main__':
    main()
