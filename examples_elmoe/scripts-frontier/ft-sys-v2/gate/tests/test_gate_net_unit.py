#!/usr/bin/env python3
# ft-sys-v2 copy (2026-11-05) of examples_elmoe/scripts-frontier/resilience/tests/test_gate_net_unit.py (ft-sys-v1, commit 7fb97edf). Changed: only the software-stack paths, now settings with the new stack (ROCm 7.1.1, torch 2.13 venv, aws-ofi-nccl 1.22.0) as the default (GATE_PYTHON, GATE_OFI_DIR, ROCM_PATH); the tests, the rules and the judge are v1's. The tests point at the copies in ..
"""Login-node unit check of the torch-free helpers of bin/gate_net.py (2026-10-10, after job 5646502's switch-test stall):
  A. pair_schedule: two rounds of disjoint pairs for K = 2..6 nodes, every node in every round exactly once per round (a repeated
     node carries a note), and a node never pairs with itself;
  B. dummy_groups_needed: a rank with fewer process groups than the largest count among the survivors creates exactly the
     difference (the hashed group name of torch 2.9 includes the per-rank group count, so the counts must be equal);
  C. ready_counts: the ready files done/r<rank>.ready.<count> parse back to {rank: count} and other done files are ignored;
  D. the summary's left-out list is a set (job 5646502 printed [0, 0, 0, 0, 0, 0, 0] for one rank)."""
import os, sys, tempfile
BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, BIN)
import gate_net as G

ok = True


def check(name, cond, extra=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else '  ' + str(extra)[:400])); ok &= bool(cond)


# A. the schedule
for K in range(2, 7):
    r1, r2 = G.pair_schedule(K)
    for rnd, name in ((r1, 'round 1'), (r2, 'round 2')):
        seen = [i for i, j, _ in rnd] + [j for i, j, _ in rnd]
        check(f'A K={K} {name}: every node appears, no self pair', set(seen) == set(range(K)) and all(i != j for i, j, _ in rnd), rnd)
        repeated = [n for n in set(seen) if seen.count(n) > 1]
        check(f'A K={K} {name}: a repeated node carries a note', all(any(nt for i, j, nt in rnd if n in (i, j)) for n in repeated), (repeated, rnd))
r1, r2 = G.pair_schedule(4)
check('A K=4: round 1 (0,1)(2,3), round 2 (1,2)(3,0) (the other matching of the cycle)', [(i, j) for i, j, _ in r1] == [(0, 1), (2, 3)] and [(i, j) for i, j, _ in r2] == [(1, 2), (3, 0)], (r1, r2))
for K in (4, 6, 8, 260, 262, 264):
    r1, r2 = G.pair_schedule(K)
    used = [x for i, j, _ in r2 for x in (i, j)]
    p1 = {frozenset((i, j)) for i, j, _ in r1}; p2 = {frozenset((i, j)) for i, j, _ in r2}
    check(f'A K={K} (even): round 2 uses every node exactly once and repeats no pair of round 1', sorted(used) == list(range(K)) and not (p1 & p2) and all(not nt for _, _, nt in r2), (K, r2[:4]))
# A2. the group-count table: both members of every pair have created the same number of groups when they form it
def counts_agree(K):
    """Simulate what the ranks do: a rank of parity s (even GCDs: sub-round 0; odd GCDs: sub-round 1) creates a group only in its
    own sub-round, so the counts live per (parity, node); both members of every pair must agree when they form the group."""
    sched = G.pair_schedule(K); pads = G.pair_pads(sched); cnt = {}
    for r_idx, rnd in enumerate(sched):
        for s_idx in range(2):
            for p_idx, (i, j, _) in enumerate(rnd):
                pi, pj = pads[(r_idx, s_idx, p_idx)]
                ci, cj = cnt.get((s_idx, i), 0) + pi, cnt.get((s_idx, j), 0) + pj
                if ci != cj:
                    return False
                cnt[(s_idx, i)] = ci + 1; cnt[(s_idx, j)] = cj + 1
    return True
# the defect of job 5646685 (K = 3): the odd-GCD ranks of the repeated node's partners must NOT be padded in round 1
pads3 = G.pair_pads(G.pair_schedule(3))
check('A2 K=3: round 1 sub-round 2 (the odd GCDs) pads exactly as sub-round 1 does (the counts are per rank parity, not per node)',
      pads3[(0, 1, 0)] == pads3[(0, 0, 0)] and pads3[(0, 1, 1)] == pads3[(0, 0, 1)], pads3)
for K in (3, 4, 5, 6, 7, 262):
    check(f'A2 K={K}: the pads make both members of every pair agree on the group count', counts_agree(K), K)
check('A2 K=6: no pad at all with the even-K schedule', all(v == (0, 0) for v in G.pair_pads(G.pair_schedule(6)).values()), '')
check('A2 K=5: the repeated node\'s partners pad (job 5646651 was K=6 under the old block scheme: node 0 in two pairs of round 2)', any(v != (0, 0) for v in G.pair_pads(G.pair_schedule(5)).values()), '')

# B. the group count
check('B: equal counts need no dummy', G.dummy_groups_needed(3, [3, 3, 3]) == 0, '')
check('B: a rank one short creates one dummy (rank 16 of job 5646502: 2 against 3)', G.dummy_groups_needed(2, [3]) == 1, '')
check('B: a rank above the largest creates none', G.dummy_groups_needed(5, [3, 4]) == 0, '')
check('B: an unknown count (-1) creates none', G.dummy_groups_needed(-1, [3]) == 0, '')
check('B: no counts -> none', G.dummy_groups_needed(3, []) == 0, '')

# C. the ready files
T = tempfile.mkdtemp(prefix='gatenet_', dir=os.environ.get('TMPD', '/tmp'))
for n in ('r1.ready.3', 'r16.ready.2', 'r9.ready.3', 'r0.gone', 'r1.round2', 'r7.ready.-1'):
    open(os.path.join(T, n), 'w').close()
check('C: ready files parse to {rank: count}, other done files ignored', G.ready_counts(T) == {1: 3, 16: 2, 9: 3, 7: -1}, G.ready_counts(T))
check('C: an empty folder gives {}', G.ready_counts(os.path.join(T, 'none')) == {}, '')

# D. the left-out list is a set in the summary (the expression as written in summarize)
nj = [dict(switch_left_out=[0, 8, 24]) for _ in range(7)]
RN = {0: 'nA', 8: 'nB', 24: 'nC'}
left = sorted(set(r for j in nj for r in (j.get('switch_left_out') or []) if RN.get(r) == 'nA'))
check('D: one rank of this node appears once, not seven times', left == [0], left)

# E. the full-node rule (job 5646617: three nodes lost one rank each; the switch group holds the full node(s) only)
node_of = {r: ['nA', 'nB', 'nC', 'nD'][r // 8] for r in range(32)}
members = [r for r in range(32) if r not in (0, 8, 24)]
mf, fulls, partial = G.full_node_members(members, node_of, 8)
check('E: only the full node forms the switch group', fulls == ['nC'] and mf == list(range(16, 24)), (fulls, mf))
check('E: the partial nodes name their missing ranks', partial == {'nA': [0], 'nB': [8], 'nD': [24]}, partial)
mf, fulls, partial = G.full_node_members(list(range(32)), node_of, 8)
check('E: with no departure every node is full', fulls == ['nA', 'nB', 'nC', 'nD'] and mf == list(range(32)) and partial == {}, (fulls, partial))
mf, fulls, partial = G.full_node_members([r for r in range(32) if r >= 8], node_of, 8)
check('E: a node that lost all 8 ranks is partial with all 8 missing', partial == {'nA': list(range(8))} and fulls == ['nB', 'nC', 'nD'], partial)

print('ALL PASS' if ok else 'SOME FAILED')
sys.exit(0 if ok else 1)
