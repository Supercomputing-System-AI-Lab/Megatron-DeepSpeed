#!/usr/bin/env python3
# ft-sys-v2 copy (2026-11-05) of examples_elmoe/scripts-frontier/resilience/bin/gate_net.py (ft-sys-v1, commit 7fb97edf). Changed: only the software-stack paths, now settings with the new stack (ROCm 7.1.1, torch 2.13 venv, aws-ofi-nccl 1.22.0) as the default (GATE_PYTHON, GATE_OFI_DIR, ROCM_PATH); the tests, the rules and the judge are v1's.
"""gate_net.py: the gate's cross-node network tests (2026-10-09 evening; rewritten 2026-10-09 late evening to the user's decisions).

THE DESIGN IN PLAIN WORDS
  The tests run as ONE srun step over the K gate nodes, 8 tasks per node (one per GCD), launched by bin/gate_net.sh, which runs
  this file once per task in phase 'net' (and once more in phase 'gdr' only when GATE_RDMA_PHASE=1). Afterwards gate.sh runs
  `--summarize` on the batch node, which turns the per-rank json files into one row per node per test in the 7 columns of
  gate_node.sh (time node test value run_s status detail). The rows never judge: gate_judge.py does.

  pair (phase net)   THE PAIR TEST: a two-rank all-reduce in two rounds.
      Per round the nodes are paired in DISJOINT pairs: round 1 pairs node i with node i+1 (i even); round 2 pairs node i with
      node i+2 inside blocks of four (0-2, 1-3, 4-6, 5-7, ...), so every node meets a DIFFERENT partner. With an odd node count the
      leftover node is paired with the first node a second time in round 1 and with the second node in round 2, and the detail says
      so (pair_schedule below lists every case). Each round runs in TWO SUB-ROUNDS: first the even GCDs (0, 2, 4, 6), then the odd
      GCDs (1, 3, 5, 7): GCD k of node a forms a two-rank group with GCD k of node b, so that in every sub-round each of the 4
      network cards carries exactly one stream (card 0 serves GCDs 0 and 1, card 1 GCDs 2 and 3, card 2 GCDs 4 and 5, card 3
      GCDs 6 and 7: the gate's measured mapping) and all 8 GCDs are tested, the odd ones over their own path to the card.
      The groups are created with dist.new_group (every rank calls it for every group; a rank that is not a member gets None at
      once because use_local_synchronization=True; the groups are disjoint, so their communicators form in parallel). Each group
      runs 3 warm-up all-reduces and then 20 timed all-reduces of 64 MiB fp32 (fp32, not bf16: the data check needs exact
      integers, and fp32 holds every integer below 2^24 exactly) with async_op=True, polled with is_completed() under a DEADLINE of
      GATE_PAIR_DEADLINE_S = 30 s per all-reduce (the user's choice for a shared cluster: a too-long deadline costs 30 s only when
      a rank is really silent, a too-short one would exclude a healthy node under congestion). The first all-reduce also builds
      the communicator; a guard thread applies the same deadline to it, because that call blocks and cannot be polled.
      THE DATA IS CHECKED: each rank fills its buffer with (its rank + 1); the result must equal the sum of the two; a mismatch
      is status 'error' with the reason 'wrong data'.
      NO dist.barrier anywhere in this test: between the rounds and at the end the ranks synchronize through DONE FILES in the
      net dir (done/r<rank>.<tag>), with a wait of GATE_SYNC_WAIT_S = 60 s that ends either way. A rank whose all-reduce does not
      complete by the deadline writes its json with status 'timeout' and the PARTNER NODE'S NAME, writes done/r<rank>.gone, and
      leaves with os._exit (no process-group teardown), so it can never trap the others.
      THE GONE PARTNER (2026-10-10, from job 5646336): a partner that is GONE counts as "did not answer" exactly like a timeout.
      Before a round a rank checks whether its partner rank has left (a .gone file): the result is status 'gone' with the
      partner's own reason in the note; a group formation or an all-reduce that fails at once (the partner's process exited, its
      communicator closed) is also 'gone', and this rank CONTINUES with its next pair and round instead of leaving. The summary
      names a node for 'timeout' and 'gone' alike (named_by=node(round r, kind)); gate_judge.py excludes a node named in both
      rounds, and recognizes an EXPLAINED departure (a rank that left after naming a node that is excluded as dead) so that it
      does not become a suspect pair.
      The value of a node is its SLOWEST GCD in GB/s, as the busbw of a two-rank all-reduce = bytes / time (for 2 ranks busbw
      equals algbw). The detail lists the 8 per-GCD rates grouped by card ("card 0: GCD 0 17.2, GCD 1 17.0; ...") so that an odd
      GCD slower than its even partner on the same card is visible, the partners of both rounds, named_in = in how many rounds a
      partner wrote this node down as not answering, named_by = who did so, and the path RCCL took (from the "via NET" lines of
      the RCCL log: "/GDRDMA" = GPU-direct, else the host-buffer path). The test runs on the SAME network path as the training
      run: NCCL_NET_GDR_LEVEL is read from the job's environment, never forced.
  switch (phase net, after the pair rounds)   unchanged in spirit: one all-to-all over the gate nodes x 8 GCDs (64 MiB per rank,
      3 warm-ups + 10 timed, CUDA events) with the Slingshot NIC (cxi) telemetry counters read before and after on every node.
      It runs over the ranks that SURVIVED the pair rounds (a rank that left after a timeout is not waited for): after the final
      done-file wait the lowest surviving rank writes survivors.json, everyone creates that group and runs the all-to-all in it.
      The counters are read by the lowest surviving local rank of each node. A guard thread bounds every blocking step by
      GATE_SWITCH_DEADLINE_S = 60 s (status 'timeout', no partner: it is a group operation); the FIRST all-to-all, which also forms
      the communicator over every survivor, has its own budget GATE_SWITCH_INIT_DEADLINE_S = 120 s. HONEST LIMIT: the switch itself
      is not addressable from a compute node; this test sees it only through the all-to-all rate and the NIC link counters.
      THE GROUP COUNT (2026-10-10 02:00, from job 5646502, where the switch all-to-all over 29 survivors never completed on any rank):
      torch names a group made with use_local_synchronization=True by a hash of its ranks AND of the number of process groups THIS
      rank has created so far (_hash_ranks_to_str appends len(_world.pg_names)); the hash is the prefix under which the group's rank 0
      publishes the NCCL unique id in the store and the others read it. Rank 16 had skipped its round-2 pair group (its partner, rank
      0, was gone: the old code did `continue` before new_group), so it had created 2 groups where every other survivor had 3; its name
      for the survivors' group differed, its store read waited for a key nobody set, and the other 28 ranks waited in RCCL's bootstrap
      for the 29th (their RCCL logs show the init past "Using network" and never reaching "Init START"; rank 16's log shows no init
      at all: 2 "Using network" lines against 4). TWO FIXES: (1) a pair group is created even when the partner is gone (a local,
      lazy call that talks to nobody), so every living rank creates exactly one group per round and sub-round it belongs to; (2) before
      the survivors' group, every survivor writes a READY file with its group count, the lowest survivor waits for every survivor's file
      and writes the FINAL membership with the largest count, and a rank with fewer groups creates one-rank dummy groups until its count
      equals the largest, so the hashed names agree by construction; the counts are recorded in the json. On a switch timeout the rank
      also dumps its own stack with py-spy (a child process, bounded) into stack.r<rank>.txt before it leaves, so that the next such
      stall is read from a stack and not inferred from log line counts.
      THE FULL-NODE RULE (2026-10-10 03:00, from job 5646617, where the two fixes above worked, the 29 survivors all reached
      "Init START" and "Connected all rings", and the init still did not finish within the 120-s budget): a node that lost one rank
      in the pair rounds has 7 ranks left, and RCCL has no ready topology model for 7 GCDs of an MI250X node, so it runs its generic
      topology search; every rank of the three 7-GCD nodes printed ~13,400 "GPU Direct RDMA Disabled for GPU x / HCA y" search lines
      against 1-13 on the full node and 25-29 in the 32-rank world communicator (RCCL log, data/gate/net/rccl_net.*). The search is
      the slow step, not a deadlock. So the switch test's group is formed from the FULL nodes only: the nodes whose 8 ranks all
      survived the pair rounds. A node that lost a rank is reported "not tested" for the switch test (its pair rows decide its fate;
      in a real failure such a node has usually lost all 8 ranks, because its partner node was silent on all 8 GCDs), and a switch
      test needs at least 2 full nodes to run. The communicator's formation is then the known 8-GCD case (2-3 s at 4 nodes).
  rdma (phase gdr, ONLY with GATE_RDMA_PHASE=1)   the old NIC-to-NIC ring transfer (dist.batch_isend_irecv, 64 MB bf16 x 20 to the
      ring-neighbour node on GCDs 0, 2, 4, 6) with GPU-direct requested (NCCL_NET_GDR_LEVEL=PHB); kept reachable, off by default,
      because the pair test now runs on the training's own path and the separate phase would only add time.

FAULT INJECTION (TEST ONLY): XMOE_GATE_INJECT="pair_silent:<node>[+gemm_slow:<node>+gemm_half:<node>]". On the named node the rank
  of GCD 0 never joins any group or all-reduce (it sleeps until the others moved on, writes a json saying 'injected silent' and
  leaves), so its partner in round 1 and its partner in round 2 both write it down: the judge excludes it by the naming rule.
  gemm_slow and gemm_half are gate_gemm.py's cases. The restart loop resolves '@<k>' to the k-th node of the allocation.

FILES in the net dir: r<rank>.<phase>.json (the measurements), r<rank>.<phase>.out (stdout+stderr), r<rank>.<phase>.rc (exit code),
  the RCCL log rccl_<phase>.<node>.<pid> (NCCL_DEBUG_FILE), done/ (the done files), survivors.r<rank>.json.
Device choice: with srun --gpus-per-node=8 every task sees all 8 GCDs (lessons.md 2026-10-09 trap 3), so each task selects
  torch.cuda.set_device(SLURM_LOCALID) itself; the OFI device index in the RCCL "via NET" lines verifies the card.
"""
import argparse, glob, json, os, re, socket, sys, threading, time

MB = 1024 * 1024
PAIR_BYTES = 64 * MB                 # 64 MiB per all-reduce buffer = 16,777,216 fp32 values
PAIR_WARMUP, PAIR_ITERS = 3, 20
PAIR_DEADLINE_S = float(os.environ.get('GATE_PAIR_DEADLINE_S', '30'))     # per all-reduce (and for the first one, which also builds the communicator)
SYNC_WAIT_S = float(os.environ.get('GATE_SYNC_WAIT_S', '60'))             # the done-file wait between rounds and at the end
SWITCH_DEADLINE_S = float(os.environ.get('GATE_SWITCH_DEADLINE_S', '60'))  # every blocking step of the switch test
SWITCH_INIT_DEADLINE_S = float(os.environ.get('GATE_SWITCH_INIT_DEADLINE_S', '120'))  # the first all-to-all: it also forms the communicator over every survivor
READY_WAIT_S = float(os.environ.get('GATE_READY_WAIT_S', '30'))            # the wait for every survivor's ready file before the switch group is formed
PYSPY = os.path.expanduser(os.environ.get('GATE_PYSPY', '~/.local/bin/py-spy'))
A2A_BYTES, A2A_ITERS, A2A_WARMUP = 64 * MB, 10, 3
RING_BYTES, RING_ITERS = 64 * MB, 20                                        # the old ring transfer (phase gdr only)
CARD_OF_GCD = {g: g // 2 for g in range(8)}                                 # the gate's measured mapping: GCDs 2c and 2c+1 share card c
SUBROUNDS = [(0, 2, 4, 6), (1, 3, 5, 7)]                                    # sub-round 1: the even GCDs; sub-round 2: the odd GCDs

# the cxi telemetry counters this test reads (per NIC): link errors, replays, pauses, drops, octets (names of Slingshot 11 on Frontier)
CXI_RX = re.compile(r'^(hni_pcs_uncorrected_cw|hni_pcs_corrected_cw|hni_pcs_fecl_errors_\d|hni_llr_(tx|rx)_replay_event|'
                    r'hni_llr_rx_bad_(lossless|lossy)|hni_llr_tx_discard|hni_discard_cntr_\d|hni_sts_(tx|rx)_(ok|bad)_octets|'
                    r'hni_pause_sent|hni_pause_xoff_sent_\d|hni_pause_recv_\d|hni_rx_paused_std|hni_tx_stall_llr|'
                    r'ixe_rx_pkt_drop_\w+|pct_rsp_dropped_\w+|pct_req_tct_tmout_drops|pct_(sct|spt|tct)_timeouts)$')
CXI_GROUPS = {   # group -> the counters summed into it (a regex on the name)
    'err': re.compile(r'^(hni_pcs_uncorrected_cw|hni_llr_rx_bad_(lossless|lossy)|hni_sts_(tx|rx)_bad_octets)$'),
    'replay': re.compile(r'^hni_llr_(tx|rx)_replay_event$'),
    'pause': re.compile(r'^(hni_pause_sent|hni_pause_xoff_sent_\d|hni_pause_recv_\d)$'),
    'paused_std': re.compile(r'^hni_rx_paused_std$'),
    'tx_stall_llr': re.compile(r'^hni_tx_stall_llr$'),
    'drop': re.compile(r'^(hni_llr_tx_discard|hni_discard_cntr_\d|ixe_rx_pkt_drop_\w+|pct_rsp_dropped_\w+|pct_req_tct_tmout_drops|pct_(sct|spt|tct)_timeouts)$'),
    'corrected': re.compile(r'^(hni_pcs_corrected_cw|hni_pcs_fecl_errors_\d)$'),
    'tx_octets': re.compile(r'^hni_sts_tx_ok_octets$'),
    'rx_octets': re.compile(r'^hni_sts_rx_ok_octets$'),
}


def cxi_read():
    """{nic: {counter: value}} for the counters of CXI_RX on every NIC of this node (value@time files; the value part)."""
    out = {}
    for nic in sorted(glob.glob('/sys/class/cxi/cxi*')):
        tdir = os.path.join(nic, 'device', 'telemetry')
        try:
            names = os.listdir(tdir)
        except OSError:
            continue
        d = {}
        for nm in names:
            if CXI_RX.match(nm):
                try:
                    d[nm] = float(open(os.path.join(tdir, nm)).read().split('@')[0])
                except (OSError, ValueError):
                    pass
        out[os.path.basename(nic)] = d
    return out


def cxi_delta(before, after):
    """{nic: {group: delta}} summed per group of CXI_GROUPS."""
    out = {}
    for nic, d1 in (after or {}).items():
        d0 = (before or {}).get(nic, {})
        g = {k: 0.0 for k in CXI_GROUPS}
        for nm, v1 in d1.items():
            dv = v1 - d0.get(nm, v1)
            for k, rx in CXI_GROUPS.items():
                if rx.match(nm):
                    g[k] += dv
        out[nic] = g
    return out


# ------------------------------------------------------------------------------------------------------ the pairing schedule
def pair_schedule(K):
    """The two rounds of disjoint pairs over K nodes (indices 0..K-1). Returns [round_1, round_2]; a round is a list of
    (i, j, note): node i is paired with node j; the note is '' for a regular pair and a sentence when a node had to be used a
    second time in the round (an odd count, a trailing block that is not a full block of four, or only two nodes).
      K=4: round 1 (0,1) (2,3); round 2 (0,2) (1,3).      K=3: round 1 (0,1) (2,0); round 2 (1,2) (0,1 again, said so).
      K=5: round 1 (0,1) (2,3) (4,0); round 2 (0,2) (1,3) (4,1).    K=6: round 2 adds (4,0) (5,1) for the trailing block of two."""
    r1, r2 = [], []
    for i in range(0, K - 1, 2):
        r1.append((i, i + 1, ''))
    if K % 2 == 1 and K > 1:
        r1.append((K - 1, 0, 'the leftover node of an odd count, paired with the first node (node 0), which is used a second time in this round'))
    if K == 2:
        r2.append((0, 1, 'only two nodes: the same pair a second time'))
    elif K % 2 == 0:
        # EVEN K (2026-10-10 03:10, job 5646651): the other perfect matching of the cycle 0-1-2-...-(K-1)-0, (1,2) (3,4) ... (K-1,0):
        # every node gets a second partner and NO node is used twice in the round. The old block-of-four scheme paired a trailing
        # block of two nodes (K = 6, 10, ..., 262 = 256 + 6 spares) with nodes 0 and 1 a second time, and a node in two groups of one
        # sub-round creates two process groups where its partners create one, so torch's hashed group name (which counts the groups a
        # rank has created) differed between the two members and both timed out (see pair_pads below for the general cure).
        for i in range(0, K, 2):
            r2.append((i + 1, (i + 2) % K, ''))
    else:
        for b in range(0, K, 4):
            blk = list(range(b, min(b + 4, K)))
            if len(blk) == 4:
                r2 += [(blk[0], blk[2], ''), (blk[1], blk[3], '')]
            elif len(blk) == 3:
                r2.append((blk[1], blk[2], ''))
                r2.append((blk[0], 1, 'the first node of a trailing block of three, paired with the second node (node 1), which is used a second time in this round'
                           + (' (with three nodes this repeats a pair of round 1)' if blk[0] == 0 else '')))
            elif len(blk) == 2:
                r2.append((blk[0], 0, 'a trailing block of two: paired with the first node (node 0), which is used a second time in this round'))
                r2.append((blk[1], 1, 'a trailing block of two: paired with the second node (node 1), which is used a second time in this round'))
            else:
                r2.append((blk[0], 1, 'the leftover node, paired with the second node (node 1), which is used a second time in this round'))
    return [r1, r2]


def pair_pads(sched, n_sub=2):
    """THE GROUP-COUNT TABLE (2026-10-10 03:10, job 5646651). torch names a group made with use_local_synchronization=True by a hash of
    its ranks AND of the number of groups the calling rank has created so far (distributed_c10d._hash_ranks_to_str), and a rank that
    is not a member of a group registers nothing when it calls new_group for it. So the two members of a pair must have created the
    SAME number of groups when they form their group, or each waits for a unique id the other publishes under a different key (both
    time out: ranks 1-7 of frontier05911 against 33-39 of frontier05929 in job 5646651, where node 0 was in two pairs of round 2 and
    node 4 in one). Every rank simulates the whole schedule here, in the order the groups are created (round, sub-round, pair), and
    pads with one-rank dummy groups where its node has created fewer groups than its partner, so that the counts agree by
    construction and no file synchronization is needed. -> {(round_idx, sub_idx, pair_idx): (pad_i, pad_j)}; with the even-K
    schedule every pad is 0; with a repeated node (odd K) the repeated node's partners pad by one."""
    # THE COUNT IS PER RANK, NOT PER NODE (2026-10-10 03:30, job 5646685, the 3-node production probe): the even GCDs of a node
    # create groups in sub-round 1 only and the odd GCDs in sub-round 2 only, so a node's two rank classes carry separate counts;
    # the first version carried one count per node across the sub-rounds, padded the odd-GCD ranks of the repeated node's
    # partners by one, and every odd-GCD pair of round 1 timed out (ranks 1,3,5,7 against 9,11,13,15). The counts of sub-round s
    # carry over from the previous round's sub-round s only.
    cnt = {}
    pads = {}
    for r_idx, rnd in enumerate(sched):
        for s_idx in range(n_sub):
            for p_idx, (i, j, note) in enumerate(rnd):
                ci, cj = cnt.get((s_idx, i), 0), cnt.get((s_idx, j), 0)
                m = max(ci, cj)
                pads[(r_idx, s_idx, p_idx)] = (m - ci, m - cj)
                cnt[(s_idx, i)] = cnt[(s_idx, j)] = m + 1
    return pads


# ------------------------------------------------------------------------------------------------------ the group count
def pg_count(dist):
    """The number of process groups this rank has created so far (the default group included): torch hashes it into the name
    of the next group made with use_local_synchronization=True (torch 2.9.1, distributed_c10d._hash_ranks_to_str), so two ranks
    with different counts name the same group differently and their NCCL unique-id exchange never meets."""
    try:
        return len(dist.distributed_c10d._world.pg_names)
    except Exception:
        return -1


def dummy_groups_needed(my_count, counts):
    """How many one-rank dummy groups this rank must create so that its count reaches the largest count among the survivors."""
    if my_count < 0 or not counts:
        return 0
    return max(0, max(counts) - my_count)


def make_dummy_groups(dist, rank, n):
    """Create n one-rank groups (local, lazy: no collective, no store traffic) to raise this rank's group count."""
    for _ in range(n):
        dist.new_group(ranks=[rank], use_local_synchronization=True)


def full_node_members(members, node_of, ranks_per_node):
    """The switch test's members: the ranks of the nodes whose ranks ALL survived (the full-node rule). -> (members_full, full_nodes,
    partial: {node: [missing ranks]}). node_of maps every rank of the world to its node; ranks_per_node is the world's per-node count."""
    per_node = {}
    for r, nd in node_of.items():
        per_node.setdefault(nd, set()).add(r)
    mset = set(members)
    full_nodes = sorted(nd for nd, rs in per_node.items() if rs <= mset and len(rs) == ranks_per_node)
    partial = {nd: sorted(rs - mset) for nd, rs in per_node.items() if nd not in full_nodes}
    members_full = sorted(r for r in members if node_of.get(r) in full_nodes)
    return members_full, full_nodes, partial


def ready_counts(done_dir):
    """{rank: count} from the ready files done/r<rank>.ready.<count>."""
    out = {}
    try:
        names = os.listdir(done_dir)
    except OSError:
        names = []
    for n in names:
        m = re.match(r'r(\d+)\.ready\.(-?\d+)$', n)
        if m:
            out[int(m.group(1))] = int(m.group(2))
    return out


def self_stack_dump(path, seconds=30):
    """py-spy dump of THIS process (a child process does it; bounded), for the stall that the guard is about to end."""
    try:
        import subprocess
        with open(path, 'w') as fh:
            fh.write(f'# {time.strftime("%Y-%m-%d %H:%M:%S")} py-spy dump --pid {os.getpid()} --native (the main thread is the one in the blocking call)\n')
            fh.flush()
            subprocess.run([PYSPY, 'dump', '--pid', str(os.getpid()), '--native'], stdout=fh, stderr=subprocess.STDOUT, timeout=seconds)
    except Exception as e:
        try:
            open(path, 'a').write(f'# the stack dump failed: {type(e).__name__}: {e}\n')
        except Exception:
            pass


# ------------------------------------------------------------------------------------------------------ the deadline guard
class Guard:
    """A daemon thread that ends this process when the main thread is still inside a blocking call after the deadline.
    arm(seconds, **what) before the call, disarm() after it. On the deadline it calls on_timeout(what), which writes the json
    (status timeout, the partner's name), writes the .gone file and leaves with os._exit: a blocked rank can never trap the others."""
    def __init__(self, on_timeout):
        self.deadline, self.what, self.on_timeout = None, None, on_timeout
        self.lock = threading.Lock()
        threading.Thread(target=self._run, name='gate-net-guard', daemon=True).start()

    def arm(self, seconds, **what):
        with self.lock:
            self.deadline, self.what = time.time() + seconds, what

    def disarm(self):
        with self.lock:
            self.deadline, self.what = None, None

    def _run(self):
        while True:
            time.sleep(0.2)
            with self.lock:
                d, w = self.deadline, self.what
            if d is not None and time.time() > d:
                self.on_timeout(w)


# ------------------------------------------------------------------------------------------------------ the phases
def run_phase(a):
    t_py0 = time.time()
    rank = int(os.environ.get('SLURM_PROCID', '0')); world = int(os.environ.get('SLURM_NTASKS', '1'))
    local = int(os.environ.get('SLURM_LOCALID', '0'))
    node = socket.gethostname().split('.')[0]
    nodes = a.nodes.split()
    done_dir = os.path.join(a.net_dir, 'done')
    os.makedirs(done_dir, exist_ok=True)
    rec = dict(phase=a.phase, rank=rank, world=world, local=local, node=node, pid=os.getpid(), nodes=nodes, error=None, exited=None,
               t_import_s=None, t_init_s=None, pair=None, ring=None, a2a=None, cxi_before=None, cxi_after=None, t_pair_end_s=None, t_total_s=None,
               env=dict(NCCL_NET_GDR_LEVEL=os.environ.get('NCCL_NET_GDR_LEVEL'), NCCL_SOCKET_IFNAME=os.environ.get('NCCL_SOCKET_IFNAME'),
                        MASTER_ADDR=os.environ.get('MASTER_ADDR'), MASTER_PORT=os.environ.get('MASTER_PORT'),
                        ROCR_VISIBLE_DEVICES=os.environ.get('ROCR_VISIBLE_DEVICES')))
    out_json = os.path.join(a.net_dir, f'r{rank}.{a.phase}.json')

    def dump():
        rec['t_total_s'] = round(time.time() - t_py0, 2)
        tmp = out_json + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(rec, fh)
        os.replace(tmp, out_json)

    def touch(name):
        open(os.path.join(done_dir, f'r{rank}.{name}'), 'w').close()

    def leave(code):
        """Write the json and the .gone marker, then end the process at once (no teardown of the process group)."""
        rec['exited'] = rec.get('exited') or 'left early'
        try:
            dump(); touch('gone')
            sys.stdout.flush(); sys.stderr.flush()
        except Exception:
            pass
        os._exit(code)

    def on_guard_timeout(what):
        """The guard thread's action: the main thread is stuck in a blocking call past the deadline."""
        what = what or {}
        if what.get('kind') == 'pair':
            rr = what['result']
            rr.update(status='timeout', note=f"{what.get('step', 'the all-reduce')} did not complete within {PAIR_DEADLINE_S:.0f} s: "
                                            f"partner {rr['partner_node']} (rank {rr['partner_rank']}, GCD {rr['gcd']}) did not answer")
            rec['exited'] = f"timeout in round {rr['round']} waiting for {rr['partner_node']}"
        else:
            rec['a2a'] = dict(status='timeout', note=f"{what.get('step', 'a step of the switch test')} did not complete within {what.get('deadline', SWITCH_DEADLINE_S):.0f} s")
            rec['exited'] = 'timeout in the switch test'
            # the stall is read from the stack next time, not inferred from log line counts (job 5646502)
            self_stack_dump(os.path.join(a.net_dir, f'stack.r{rank}.txt'))
            rec['stack_dump'] = f'stack.r{rank}.txt'
        print(f'gate_net.py rank {rank} on {node}: {rec["exited"]}; leaving', file=sys.stderr, flush=True)
        leave(3)

    guard = Guard(on_guard_timeout)
    dist = None
    try:
        t0 = time.time()
        import torch
        import torch.distributed as dist
        from datetime import timedelta
        rec['t_import_s'] = round(time.time() - t0, 2)
        rec['torch'] = torch.__version__
        ndev = torch.cuda.device_count()
        dev_idx = local if local < ndev else local % max(ndev, 1)
        rec['device_count'] = ndev; rec['device'] = dev_idx
        torch.cuda.set_device(dev_idx)
        dev = torch.device('cuda', dev_idx)
        try:
            rec['gpu_bus'] = getattr(torch.cuda.get_device_properties(dev_idx), 'pci_bus_id', None)
        except Exception:
            rec['gpu_bus'] = None
        # the world group, WITHOUT device_id: the NCCL communicators then form lazily inside the first collective of each group,
        # which puts the pair groups' formation under the guard's deadline (an eager init in new_group could not be bounded)
        t0 = time.time()
        dist.init_process_group('nccl', rank=rank, world_size=world, timeout=timedelta(seconds=a.init_timeout))
        table = [None] * world
        dist.all_gather_object(table, (rank, node, local))      # who is where: the partner ranks follow from the hostnames
        rec['t_init_s'] = round(time.time() - t0, 2)
        rank_of = {(n, l): r for r, n, l in table}
        if node not in nodes:
            raise RuntimeError(f'this node {node} is not in the step list {nodes}')
        K = len(nodes); ni = nodes.index(node)

        if a.phase == 'gdr':
            rec['ring'] = run_ring(dist, torch, dev, rank, local, node, nodes, rank_of, K, ni)
            dump()
            leave(0)

        # ---- the pair test (phase net)
        silent = f'pair_silent:{node}' in os.environ.get('XMOE_GATE_INJECT', '').replace(',', '+').split('+') and local == 0
        sched = pair_schedule(K)
        pair = dict(deadline_s=PAIR_DEADLINE_S, bytes=PAIR_BYTES, iters=PAIR_ITERS, warmup=PAIR_WARMUP, dtype='fp32', gcd=local,
                    card=CARD_OF_GCD.get(local), rounds=[], injected_silent=bool(silent), schedule=[[(nodes[i], nodes[j], nt) for i, j, nt in r] for r in sched])
        rec['pair'] = pair
        if silent:
            # TEST ONLY: this rank never joins a group or an all-reduce; it waits until the others have moved to the switch test
            print(f'gate_net.py rank {rank} on {node}: XMOE_GATE_INJECT pair_silent: this rank stays silent', file=sys.stderr, flush=True)
            t_end = time.time() + 2 * 2 * (PAIR_DEADLINE_S + SYNC_WAIT_S) + SWITCH_DEADLINE_S + 60
            while time.time() < t_end and not glob.glob(os.path.join(a.net_dir, 'survivors.r*.json')):
                time.sleep(1)
            time.sleep(20)
            rec['exited'] = 'injected silent (XMOE_GATE_INJECT pair_silent): never joined a group or an all-reduce'
            leave(0)
        buf = torch.empty(PAIR_BYTES // 4, dtype=torch.float32, device=dev)
        pads = pair_pads(sched, len(SUBROUNDS))
        rec['pair_pads'] = {f'{k[0] + 1}.{k[1] + 1}.{k[2]}': v for k, v in pads.items() if v != (0, 0)}
        for r_idx, rnd in enumerate(sched):
            round_no = r_idx + 1
            for s_idx, gcds in enumerate(SUBROUNDS):
                # a rank creates the groups it is a member of, in the schedule's order, padded by pair_pads so that both members of a
                # pair have created the same number of groups (the old code also called new_group for the groups it was NOT in, which
                # registers nothing and so kept no count equal: job 5646651)
                my_groups = []
                for p_idx, (i, j, note) in enumerate(rnd):
                    for g in gcds:
                        ra, rb = rank_of.get((nodes[i], g)), rank_of.get((nodes[j], g))
                        if ra is None or rb is None:
                            continue           # a node with fewer than 8 tasks: no group for this GCD
                        members = sorted([ra, rb])
                        if rank in members:
                            partner_rank = rb if rank == ra else ra
                            partner_node = nodes[j] if rank == ra else nodes[i]
                            res = dict(round=round_no, subround=s_idx + 1, gcd=local, card=CARD_OF_GCD.get(local), partner_node=partner_node,
                                       partner_rank=partner_rank, members=members, note=note, status=None, gbps=None, seconds=None)
                            pair['rounds'].append(res)
                            # THE GONE PARTNER (2026-10-10, from job 5646336): a partner that is GONE (its rank left the test, so the
                            # group cannot be formed or the all-reduce fails at once) counts as "did not answer" exactly like a timeout:
                            # status 'gone', named in the summary. The partner's own reason (its json) goes into the note, so that the
                            # judge can tell a rank that left after naming a dead node from a rank that simply died.
                            partner_gone = os.path.exists(os.path.join(done_dir, f'r{partner_rank}.gone'))
                            # THE GROUP IS CREATED EVEN WHEN THE PARTNER IS GONE (2026-10-10, job 5646502): new_group with
                            # use_local_synchronization=True is a local, lazy call (no collective, no store traffic), and torch hashes
                            # the number of groups this rank has created into the name of every later group, so a skipped creation
                            # here would make this rank name the survivors' group differently from everyone else (see the docstring)
                            pad = pads[(r_idx, s_idx, p_idx)][0 if rank == ra else 1]
                            if pad:
                                res['pad'] = pad
                                make_dummy_groups(dist, rank, pad)      # the partner has created more groups so far (a repeated node in the round)
                            guard.arm(PAIR_DEADLINE_S, kind='pair', result=res, step='forming the two-rank group')
                            try:
                                pg = dist.new_group(ranks=members, use_local_synchronization=True)
                            except Exception as e:
                                guard.disarm()
                                pg = None
                                try:
                                    make_dummy_groups(dist, rank, 1)      # keep the count equal to the ranks whose creation succeeded
                                except Exception:
                                    pass
                                if not partner_gone:
                                    res.update(status='gone', note=(note + '; ' if note else '') + f'forming the two-rank group failed at once ({type(e).__name__}: {str(e)[:160]}): '
                                               f'partner {partner_node} (rank {partner_rank}, GCD {local}) did not answer, so it is named like a timeout; this rank continues')
                            guard.disarm()
                            if partner_gone:
                                res.update(status='gone', note=(note + '; ' if note else '') + f'partner rank {partner_rank} on {partner_node} had already left the test before this round '
                                           f'(its own reason: {partner_reason(a.net_dir, partner_rank, a.phase)}): it did not answer, so it is named like a timeout; '
                                           f'the two-rank group was still created (unused) so that this rank\'s group count stays equal to the others\'')
                                continue
                            if pg is None:
                                continue
                            my_groups.append((pg, res))
                        # not a member of this group: nothing to create (a non-member's new_group registers no group, so it would not
                        # keep the count equal; the padding above does)
                for pg, res in my_groups:
                    try:
                        pair_allreduce(dist, torch, dev, buf, pg, res, rank, guard)
                    except Exception as e:
                        # the collective failed at once (the partner's process is gone, or its communicator closed): named like a timeout,
                        # and this rank goes on to its next pair and round instead of leaving (the old code left the whole test here)
                        guard.disarm()
                        res.update(status='gone', note=f'the all-reduce failed at once ({type(e).__name__}: {str(e)[:160]}): partner {res["partner_node"]} '
                                   f'(rank {res["partner_rank"]}, GCD {res["gcd"]}) did not answer, so it is named like a timeout; this rank continues')
                    dump()
            # the end of the round: this rank is done; wait (at most SYNC_WAIT_S) for every rank that has not left
            touch(f'round{round_no}')
            dump()
            waited = wait_done(done_dir, f'round{round_no}', world, SYNC_WAIT_S)
            pair.setdefault('sync', []).append(dict(round=round_no, **waited))
        rec['t_pair_end_s'] = round(time.time() - t_py0, 2)
        del buf
        dump()

        # ---- the switch test over the survivors of the pair rounds
        if os.environ.get('GATE_SKIP_SWITCH') == '1':      # the scale guard of gate.sh (job 5646880 at 260 nodes; see gate.sh)
            rec['a2a'] = dict(status='not_tested', note='the switch test is skipped at this node count (gate.sh GATE_NET_MAX_NODES): its survivor wait and its '
                              'baseline were validated up to 6 nodes only; the pair rows judge this node')
            rec['t_pair_end_s'] = rec.get('t_pair_end_s') or round(time.time() - t_py0, 2)
            dump(); leave(0)
        present = set(wait_done(done_dir, 'round2', world, 5.0)['present'])
        if rank == min(present):
            with open(os.path.join(a.net_dir, f'survivors.r{rank}.json.tmp'), 'w') as fh:
                json.dump(sorted(present), fh)
            os.replace(os.path.join(a.net_dir, f'survivors.r{rank}.json.tmp'), os.path.join(a.net_dir, f'survivors.r{rank}.json'))
        t_end = time.time() + 20
        files = []
        while time.time() < t_end:
            files = glob.glob(os.path.join(a.net_dir, 'survivors.r*.json'))
            if files:
                time.sleep(2)
                files = glob.glob(os.path.join(a.net_dir, 'survivors.r*.json'))
                break
            time.sleep(0.5)
        if not files:
            rec['a2a'] = dict(status='error', note='no survivors list appeared within 20 s after the pair rounds: the switch test did not run on this rank')
            dump(); leave(0)
        files.sort(key=lambda p: int(re.search(r'survivors\.r(\d+)\.json', p).group(1)))
        survivors = json.load(open(files[0]))
        rec['switch_survivors'] = len(survivors)
        rec['switch_left_out'] = sorted(set(range(world)) - set(survivors))
        if rank not in survivors:
            rec['a2a'] = dict(status='error', note=f'this rank is not in the survivors list written by rank {min(survivors)} (its round-2 done file was late): the switch test did not run on this rank')
            dump(); leave(0)
        # ---- THE READY FILES AND THE EQUAL GROUP COUNT (2026-10-10, job 5646502; see the docstring): every survivor writes its group
        # count; the lowest survivor waits for every survivor's file (READY_WAIT_S) and writes the FINAL membership (the ranks that
        # answered) with the largest count; everyone waits for the final file, checks its membership, raises its own count to the
        # largest with one-rank dummy groups, and only then forms the survivors' group
        my_count = pg_count(dist)
        touch(f'ready.{my_count}')
        final_p = os.path.join(a.net_dir, 'survivors_final.json')
        if rank == min(survivors):
            t_end = time.time() + READY_WAIT_S
            while time.time() < t_end:
                rc_ = ready_counts(done_dir)
                if all(r in rc_ for r in survivors):
                    break
                time.sleep(0.5)
            rc_ = ready_counts(done_dir)
            final = dict(ranks=sorted(r for r in survivors if r in rc_), counts={str(r): rc_[r] for r in survivors if r in rc_},
                         max_count=max([rc_[r] for r in survivors if r in rc_] or [my_count]),
                         not_ready=sorted(r for r in survivors if r not in rc_), written_by=rank, time=time.time())
            with open(final_p + '.tmp', 'w') as fh:
                json.dump(final, fh)
            os.replace(final_p + '.tmp', final_p)
        t_end = time.time() + READY_WAIT_S + 20
        final = None
        while time.time() < t_end:
            try:
                final = json.load(open(final_p))
                break
            except Exception:
                time.sleep(0.5)
        if final is None:
            rec['a2a'] = dict(status='error', note=f'the final membership file of the switch test did not appear within {READY_WAIT_S + 20:.0f} s (rank {min(survivors)} writes it): the switch test did not run on this rank')
            dump(); leave(0)
        members = final['ranks']
        rec['switch_pg_count'] = dict(mine=my_count, max=final['max_count'], counts=final['counts'], not_ready=final['not_ready'])
        rec['switch_survivors'] = len(members)
        rec['switch_left_out'] = sorted(set(range(world)) - set(members))
        if rank not in members:
            rec['a2a'] = dict(status='error', note=f'this rank wrote its ready file too late for the final membership (written by rank {final["written_by"]}): the switch test did not run on this rank')
            dump(); leave(0)
        # ---- THE FULL-NODE RULE (2026-10-10, job 5646617; see the docstring): only the nodes whose 8 ranks all survived form the
        # switch group; a rank on a node that lost a rank records 'not tested' and leaves; fewer than 2 full nodes: no switch test
        node_of = {r: nd for (nd, l), r in rank_of.items()}
        members_full, full_nodes, partial = full_node_members(members, node_of, len([1 for (nd, l) in rank_of if nd == node]))
        rec['switch_full_nodes'] = full_nodes
        rec['switch_partial_nodes'] = partial
        if node not in full_nodes:
            rec['a2a'] = dict(status='not_tested', note=f'this node is not in the switch test because its rank(s) {partial.get(node, [])} left the pair '
                              f'rounds (after a partner that did not answer, or silent), and a communicator over a node with 7 ranks needs RCCL\'s slow generic '
                              f'topology search (job 5646617: ~13,400 search lines per rank, the 120-s budget ran out); the switch test runs over the '
                              f'{len(full_nodes)} full node(s) {full_nodes}; this node is judged by its pair rows')
            print(f'gate_net.py rank {rank} on {node}: {rec["a2a"]["note"]}', file=sys.stderr, flush=True)
            dump(); leave(0)
        if len(full_nodes) < 2:
            rec['a2a'] = dict(status='not_tested', note=f'the switch test did not run: only {len(full_nodes)} full node(s) survived the pair rounds and an all-to-all needs 2; '
                              f'the nodes that lost ranks: {partial}')
            dump(); leave(0)
        max_full = max(int(final['counts'].get(str(r), my_count)) for r in members_full)
        need = dummy_groups_needed(my_count, [max_full])
        if need:
            print(f'gate_net.py rank {rank} on {node}: this rank created {my_count} process groups, the largest count among the survivors is {final["max_count"]}: '
                  f'{need} one-rank dummy group(s) are created so that the survivors\' group gets the same hashed name on every rank', file=sys.stderr, flush=True)
            make_dummy_groups(dist, rank, need)
            rec['switch_pg_count']['dummies'] = need
        rec['switch_survivors'] = len(members_full)
        run_switch(dist, torch, dev, rank, local, node, members_full, rank_of, rec, guard)
        dump()
    except Exception as e:  # the row becomes 'error' with this text; never a traceback-less silence
        guard.disarm()
        rec['error'] = f'{type(e).__name__}: {str(e)[:300]}'
        print(f'gate_net.py phase {a.phase} rank {rank} on {node}: {rec["error"]}', file=sys.stderr, flush=True)
        dump()
        leave(1)
    guard.disarm()
    dump()
    leave(0)     # no destroy_process_group: a teardown can wait for ranks that left; the json is the result


def pair_allreduce(dist, torch, dev, buf, pg, res, rank, guard):
    """One two-rank group: 3 warm-up + 20 timed all-reduces of the 64 MiB fp32 buffer, async, polled under the deadline; the
    timing by CUDA events on the current stream (work.wait() orders the stream after the NCCL stream); the data checked at the end."""
    me, other = rank + 1, res['partner_rank'] + 1
    expected = float(me + other)
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    for i in range(PAIR_WARMUP + PAIR_ITERS):
        buf.fill_(float(me))
        if i == PAIR_WARMUP:
            e0.record()
        step = ('the first all-reduce (it also forms the communicator)' if i == 0 else f'all-reduce {i + 1} of {PAIR_WARMUP + PAIR_ITERS}')
        guard.arm(PAIR_DEADLINE_S, kind='pair', result=res, step=step)
        work = dist.all_reduce(buf, group=pg, async_op=True)
        t_dead = time.time() + PAIR_DEADLINE_S
        while not work.is_completed():
            if time.time() > t_dead:
                guard.disarm()
                res.update(status='timeout', note=f'{step} did not complete within {PAIR_DEADLINE_S:.0f} s: partner {res["partner_node"]} (rank {res["partner_rank"]}, GCD {res["gcd"]}) did not answer')
                guard.on_timeout(dict(kind='pair', result=res, step=step))
            time.sleep(0.001)
        work.wait()
        guard.disarm()
    e1.record()
    guard.arm(PAIR_DEADLINE_S, kind='pair', result=res, step='the final device synchronization')
    torch.cuda.synchronize()
    guard.disarm()
    dt = e0.elapsed_time(e1) / 1000.0
    mn, mx = float(buf.min().item()), float(buf.max().item())
    res.update(seconds=round(dt, 4), gbps=round(PAIR_BYTES * PAIR_ITERS / dt / 1e9, 2) if dt > 0 else None)
    if mn != expected or mx != expected:
        res.update(status='error', note=f'wrong data: expected every value to be {expected:.0f} ({me} + {other}), saw min {mn} max {mx}')
    else:
        res.update(status='ok', note=(res.get('note') or '') + (f'; data checked: every value equals {expected:.0f}' if res.get('note') else f'data checked: every value equals {expected:.0f}'))


def partner_reason(net_dir, partner_rank, phase='net'):
    """Why the partner rank left, from its own json ('exited' and 'error'), for the note of a 'gone' result; a short text."""
    try:
        j = json.load(open(os.path.join(net_dir, f'r{partner_rank}.{phase}.json')))
        ex, er = j.get('exited') or 'no reason recorded', j.get('error')
        return (ex + (f'; error {er[:100]}' if er else ''))[:220]
    except Exception:
        return 'its json is not readable'


def wait_done(done_dir, tag, world, max_wait):
    """Wait until every rank that has not left (.gone) wrote done/r<rank>.<tag>, or max_wait seconds passed. One listdir per poll
    (never one read per rank). Returns what was seen, for the json."""
    t0 = time.time()
    present, gone = set(), set()
    while True:
        try:
            names = os.listdir(done_dir)
        except OSError:
            names = []
        present = set(int(m.group(1)) for n in names for m in [re.match(rf'r(\d+)\.{tag}$', n)] if m)
        gone = set(int(m.group(1)) for n in names for m in [re.match(r'r(\d+)\.gone$', n)] if m)
        if len(present | gone) >= world or time.time() - t0 > max_wait:
            break
        time.sleep(0.5)
    missing = sorted(set(range(world)) - present - gone)
    return dict(tag=tag, waited_s=round(time.time() - t0, 1), present=sorted(present), gone=sorted(gone), missing=missing,
                complete=not missing)


def run_switch(dist, torch, dev, rank, local, node, survivors, rank_of, rec, guard):
    """The switch test: one all-to-all over the survivors' group; the cxi counters read on each node by its lowest surviving local
    rank; CUDA events time the 10 all-to-alls; every blocking step under the guard (status timeout, no partner)."""
    def arm(step, deadline=SWITCH_DEADLINE_S):
        guard.arm(deadline, kind='switch', step=step, deadline=deadline)
    arm('forming the survivors group')
    pg = dist.new_group(ranks=sorted(survivors), use_local_synchronization=True)
    guard.disarm()
    G = len(survivors)
    n = (A2A_BYTES // 2 // G) * G
    inp = torch.randn(n, dtype=torch.bfloat16, device=dev); out = torch.empty_like(inp)
    my_node_locals = sorted(l for (nd, l), r in rank_of.items() if nd == node and r in survivors)
    reader = bool(my_node_locals) and local == my_node_locals[0]
    t_init0 = time.time()
    for i in range(A2A_WARMUP):
        if i == 0:
            # the first all-to-all also forms the NCCL communicator over every survivor (the unique-id exchange through the store, RCCL's
            # bootstrap over K nodes, the topology and the connections): it gets its own budget, separate from the data deadline
            arm(f'warm-up all-to-all 1 of {A2A_WARMUP} (it also forms the communicator over the {G} survivors)', SWITCH_INIT_DEADLINE_S)
        else:
            arm(f'warm-up all-to-all {i + 1} of {A2A_WARMUP}')
        dist.all_to_all_single(out, inp, group=pg)
        if i == 0:
            torch.cuda.synchronize()
            rec['switch_init_s'] = round(time.time() - t_init0, 2)
        guard.disarm()
    torch.cuda.synchronize()
    bounded_sync(dist, torch, dev, pg, arm, guard, 'the synchronization before the counters are read')
    if reader:
        rec['cxi_before'] = cxi_read()
    bounded_sync(dist, torch, dev, pg, arm, guard, 'the synchronization before the timed all-to-alls')
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(A2A_ITERS)]
    for i in range(A2A_ITERS):
        arm(f'timed all-to-all {i + 1} of {A2A_ITERS}')
        ev[i][0].record(); dist.all_to_all_single(out, inp, group=pg); ev[i][1].record()
        guard.disarm()
    arm('the device synchronization after the timed all-to-alls')
    torch.cuda.synchronize()
    guard.disarm()
    bounded_sync(dist, torch, dev, pg, arm, guard, 'the synchronization before the counters are read again')
    if reader:
        rec['cxi_after'] = cxi_read()
    ms = sorted(e0.elapsed_time(e1) for e0, e1 in ev)
    med = ms[len(ms) // 2] / 1e3
    nbytes = n * 2
    rec['a2a'] = dict(status='ok', world=G, bytes=nbytes, iters=A2A_ITERS, ms_median=round(med * 1e3, 3), ms_min=round(ms[0], 3), ms_max=round(ms[-1], 3),
                      algbw=round(nbytes / med / 1e9, 2), busbw=round(nbytes / med / 1e9 * (G - 1) / G, 2), counters_read_by_this_rank=reader)
    del inp, out


def bounded_sync(dist, torch, dev, pg, arm, guard, what):
    """A synchronization of the group that cannot trap: a one-element all-reduce, async, polled under the switch deadline
    (dist.barrier is not used anywhere in this test)."""
    t = torch.zeros(1, device=dev)
    arm(what)
    w = dist.all_reduce(t, group=pg, async_op=True)
    t_dead = time.time() + SWITCH_DEADLINE_S
    while not w.is_completed():
        if time.time() > t_dead:
            guard.on_timeout(dict(kind='switch', step=what))
        time.sleep(0.001)
    w.wait(); torch.cuda.synchronize()
    guard.disarm()


def run_ring(dist, torch, dev, rank, local, node, nodes, rank_of, K, ni):
    """Phase gdr only (GATE_RDMA_PHASE=1): the old NIC-to-NIC ring transfer on GCDs 0, 2, 4, 6 (one per card), 64 MB bf16 x 20 to
    the next node and from the previous one, with GPU-direct requested by gate_net.sh (NCCL_NET_GDR_LEVEL=PHB). Kept as it was."""
    active = (local % 2 == 0) and K >= 2
    ring = dict(active=active, nic=local // 2, dst_node=None, src_node=None, dst_rank=None, src_rank=None, bytes=RING_BYTES, iters=RING_ITERS, seconds=None, gbps=None)
    if active:
        nxt, prv = nodes[(ni + 1) % K], nodes[(ni - 1) % K]
        if (nxt, local) not in rank_of or (prv, local) not in rank_of:
            raise RuntimeError(f'no rank for GCD {local} on {nxt} or {prv} in the step')
        dst, src = rank_of[(nxt, local)], rank_of[(prv, local)]
        ring.update(dst_node=nxt, src_node=prv, dst_rank=dst, src_rank=src)
        n = RING_BYTES // 2
        sbuf = torch.ones(n, dtype=torch.bfloat16, device=dev); rbuf = torch.empty(n, dtype=torch.bfloat16, device=dev)

        def xfer():
            for r in dist.batch_isend_irecv([dist.P2POp(dist.isend, sbuf, dst), dist.P2POp(dist.irecv, rbuf, src)]):
                r.wait()
        for _ in range(3):
            xfer()
        torch.cuda.synchronize()
    dist.barrier(); torch.cuda.synchronize()     # the old phase keeps its barriers: it is off by default
    if active:
        t0 = time.perf_counter()
        for _ in range(RING_ITERS):
            xfer()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        ring['seconds'] = round(dt, 4); ring['gbps'] = round(RING_BYTES * RING_ITERS / dt / 1e9, 2)
        del sbuf, rbuf
    dist.barrier()
    return ring


# ------------------------------------------------------------------------------------------------------ the summary
def _read_jsons(net_dir, phase):
    out = {}
    for p in glob.glob(os.path.join(net_dir, f'r*.{phase}.json')):
        try:
            j = json.load(open(p)); out[j['rank']] = j
        except Exception:
            pass
    return out


def _rank_nodes(net_dir, phase):
    """rank -> node from the wrapper's header line of r<rank>.<phase>.out (exists even when the phase died at once)."""
    out = {}
    for p in glob.glob(os.path.join(net_dir, f'r*.{phase}.out')):
        try:
            head = open(p, errors='replace').readline()
            m = re.search(r'node=(\S+) rank=(\d+)', head)
            if m:
                out[int(m.group(2))] = m.group(1)
        except OSError:
            pass
    return out


def _rc(net_dir, rank, phase):
    try:
        return int(open(os.path.join(net_dir, f'r{rank}.{phase}.rc')).read().strip())
    except (OSError, ValueError):
        return None


def _out_error(net_dir, rank, phase):
    try:
        lines = open(os.path.join(net_dir, f'r{rank}.{phase}.out'), errors='replace').read().splitlines()
    except OSError:
        return 'no output file'
    for ln in reversed(lines):
        if re.search(r'error|Error|Traceback|Killed|Terminated', ln):
            return ln.strip()[:160]
    return (lines[-1].strip()[:160] if lines else 'empty output')


def _rccl_lines(net_dir, phase, node, pids):
    """the RCCL log lines of the given pids of a node (NCCL_DEBUG_FILE = rccl_<phase>.<node>.<pid>)."""
    lines = []
    for pid in pids:
        p = os.path.join(net_dir, f'rccl_{phase}.{node}.{pid}')
        try:
            lines += open(p, errors='replace').read().splitlines()
        except OSError:
            pass
    return lines


def _path_text(lines, env_level):
    vianet = [l for l in lines if 'via NET/' in l]
    ngdr = sum(1 for l in vianet if '/GDRDMA' in l)
    lvl = f'NCCL_NET_GDR_LEVEL {env_level if env_level else "unset (RCCL default)"}'
    if not vianet:
        return f'path unknown (no via-NET line in the RCCL log; {lvl})'
    if ngdr == len(vianet):
        return f'path GPU-direct (/GDRDMA on {ngdr} of {len(vianet)} via-NET lines; {lvl})'
    if ngdr == 0:
        return f'path host-buffer (no /GDRDMA on the {len(vianet)} via-NET lines; {lvl})'
    return f'path mixed (/GDRDMA on {ngdr} of {len(vianet)} via-NET lines; {lvl})'


def _fmt(v):
    return '-' if v is None else (f'{v:.2f}' if isinstance(v, float) else str(v))


def summarize(a):
    nodes = a.nodes.split()
    budgets = dict(x.split('=') for x in a.budgets.split(',') if '=' in x)
    JN, JG = _read_jsons(a.net_dir, 'net'), _read_jsons(a.net_dir, 'gdr')
    RN, RG = _rank_nodes(a.net_dir, 'net'), _rank_nodes(a.net_dir, 'gdr')
    for j in JN.values():
        RN[j['rank']] = j['node']
    for j in JG.values():
        RG[j['rank']] = j['node']
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    # who named whom: node -> [(namer node, round, namer rank, gcd, kind)] from every rank's round results with status timeout or gone
    # (2026-10-10: a GONE partner counts as "did not answer" exactly like a timeout; the kind is kept so that the judge can explain it)
    named = {n: [] for n in nodes}
    for j in JN.values():
        for res in ((j.get('pair') or {}).get('rounds') or []):
            if res.get('status') in ('timeout', 'gone') and res.get('partner_node') in named:
                named[res['partner_node']].append((j['node'], res['round'], j['rank'], res.get('gcd'), res['status']))
    for node in nodes:
        rows = []

        def row(test, value, run_s, status, detail):
            rows.append('\t'.join([ts, node, test, str(value), str(run_s), status, detail.replace('\t', ' ').replace('\n', ' ')]))
        rn = sorted(r for r, n in RN.items() if n == node); rg = sorted(r for r, n in RG.items() if n == node)
        nj = [JN[r] for r in rn if r in JN]
        rcs = [_rc(a.net_dir, r, 'net') for r in rn]
        killed = any(rc is not None and 124 <= rc <= 137 for rc in rcs)      # the phase's timeout killed a rank of this node
        env_level = next((j['env'].get('NCCL_NET_GDR_LEVEL') for j in nj if j.get('env')), None)
        # ---- pair
        if not rn:
            row('pair', '-', 0, 'timeout' if killed else 'missing', 'no rank of this node ran the network phase (no output files)')
            row('switch', '-', 0, 'timeout' if killed else 'missing', 'no rank of this node ran the network phase (no output files)')
        else:
            per_gcd = {}          # gcd -> min gbps over the rounds (ok results only)
            round_txt = []
            n_ok = n_to = n_err = n_skip = n_gone = 0
            silent = [j['local'] for j in nj if (j.get('pair') or {}).get('injected_silent')]
            for j in sorted(nj, key=lambda j: j['local']):
                for res in ((j.get('pair') or {}).get('rounds') or []):
                    st = res.get('status')
                    if st == 'ok' and res.get('gbps') is not None:
                        n_ok += 1
                        per_gcd[j['local']] = min(per_gcd.get(j['local'], 1e9), res['gbps'])
                    elif st == 'timeout':
                        n_to += 1
                    elif st == 'gone':
                        n_gone += 1
                    elif st == 'error':
                        n_err += 1
                    elif st == 'skipped':
                        n_skip += 1
                    round_txt.append(f"round {res['round']}.{res['subround']} GCD {res['gcd']} with {res['partner_node']}: {st or 'no result'}"
                                     + (f" {res['gbps']} GB/s" if res.get('gbps') is not None else '') + (f" ({res['note'][:160]})" if st in ('timeout', 'gone', 'error', 'skipped') and res.get('note') else ''))
            missing_ranks = [r for r in rn if r not in JN]
            no_result = [j['local'] for j in nj if j.get('pair') is None or not (j['pair'].get('rounds'))]
            run_pair = max([j.get('t_pair_end_s') or j.get('t_total_s') or 0 for j in nj] or [0])
            partners = {}
            for j in nj:
                for res in ((j.get('pair') or {}).get('rounds') or []):
                    partners.setdefault(res['round'], set()).add(res['partner_node'])
            partner_txt = '; '.join(f"round {r}: partner {', '.join(sorted(p))}" for r, p in sorted(partners.items())) or 'no partner recorded'
            notes = sorted(set(res['note'] for j in nj for res in ((j.get('pair') or {}).get('rounds') or []) if res.get('note') and res.get('status') == 'ok' and 'second time' in res['note']))
            by = named[node]
            named_rounds = sorted(set(r for _, r, _, _, _ in by))
            # named_by = namer(round r, kind),...: the kind is timeout or gone (gate_judge.named_by_of parses both forms)
            named_by = ','.join(f'{nm}(round {r}, {k})' for nm, r, k in sorted(set((nm, r, k) for nm, r, _, _, k in by)))
            lines_rccl = _rccl_lines(a.net_dir, 'net', node, [j['pid'] for j in nj])
            devs = {}
            for j in nj:
                m = set(re.findall(r'\[send\] via NET/[^/]+/(\d+)', '\n'.join(_rccl_lines(a.net_dir, 'net', node, [j['pid']]))))
                if m:
                    devs[j['local']] = ','.join(sorted(m))
            cards = '; '.join(f'card {c}: ' + ', '.join(f'GCD {g} {_fmt(per_gcd.get(g))}' for g in (2 * c, 2 * c + 1)) for c in range(4))
            gcd_list = ','.join(f'gcd{g}={per_gcd[g]}' for g in sorted(per_gcd))
            value = min(per_gcd.values()) if per_gcd else '-'
            head = (f'the slowest of {len(per_gcd)} GCDs over 2 rounds x 2 sub-rounds ({n_ok} all-reduce results ok, {n_to} timed out, {n_gone} partner gone, {n_err} wrong data, {n_skip} skipped)'
                    + (f'; {len(missing_ranks)} rank(s) wrote no json' if missing_ranks else '') + (f'; GCD {silent} injected silent (XMOE_GATE_INJECT)' if silent else ''))
            tail = (f'{cards}; {partner_txt}; named_in={len(named_rounds)} named_by={named_by or "nobody"}; '
                    f'{_path_text(lines_rccl, env_level)}; OFI device per GCD: ' + (' '.join(f'gcd{l}->{d}' for l, d in sorted(devs.items())) or 'not in the RCCL log')
                    + (f'; {PAIR_BYTES // MB} MiB fp32 x {PAIR_ITERS} per all-reduce, deadline {PAIR_DEADLINE_S:.0f} s' )
                    + (f'; schedule notes: {" | ".join(notes)}' if notes else '') + f'; per result: {" | ".join(round_txt)[:900]}')
            if n_err:
                row('pair', value, run_pair, 'error', f'wrong data on {n_err} all-reduce result(s) of this node: {head}; {gcd_list or "none"}; {tail}')
            elif per_gcd:
                row('pair', value, run_pair, 'ok', f'{head}; {gcd_list}; {tail}')
            elif killed or n_to or missing_ranks or no_result:
                row('pair', '-', run_pair, 'timeout', f'no GCD of this node completed an all-reduce ({head}; the phase budget of {budgets.get("pair", "?")} s{" killed the step" if killed else ""}); none; {tail}')
            else:
                row('pair', '-', run_pair, 'error', f'no result at all: {head}; none; {tail}')
            # ---- switch (the survivors' all-to-all)
            a2a = [j for j in nj if (j.get('a2a') or {}).get('busbw') is not None]
            run_sw = max([(j.get('t_total_s') or 0) - (j.get('t_pair_end_s') or 0) for j in nj] or [0])
            readers = [j for j in nj if (j.get('a2a') or {}).get('counters_read_by_this_rank') and j.get('cxi_after')]
            cx = cxi_delta(readers[0]['cxi_before'], readers[0]['cxi_after']) if readers else {}
            tot = {k: sum(g[k] for g in cx.values()) for k in CXI_GROUPS} if cx else {}
            cxi_err = int(tot.get('err', 0) + tot.get('replay', 0)) if cx else -1
            cxi_txt = (f'cxi_err={cxi_err} (link errors {int(tot["err"])}, replays {int(tot["replay"])}); pause frames {int(tot["pause"])}, '
                       f'rx paused {int(tot["paused_std"])}, tx stall llr {int(tot["tx_stall_llr"])}; drops {int(tot["drop"])}; corrected {int(tot["corrected"])}; '
                       f'octets tx {tot["tx_octets"] / 1e9:.2f} GB rx {tot["rx_octets"] / 1e9:.2f} GB; per NIC: ' +
                       ' '.join(f'{nic}[err {int(g["err"])} replay {int(g["replay"])} pause {int(g["pause"])} drop {int(g["drop"])} tx {g["tx_octets"] / 1e9:.2f}GB]'
                                for nic, g in sorted(cx.items()))) if cx else 'cxi_err=-1 (no counters read: /sys/class/cxi unreadable or no surviving rank of this node read them)'
            surv = next((j.get('switch_survivors') for j in nj if j.get('switch_survivors')), None)
            left = sorted(set(r for j in nj for r in (j.get('switch_left_out') or []) if RN.get(r) == node))
            left_txt = f'; ranks of this node that left before the switch test (after a pair timeout or silent): {left}' if left else ''
            cnts = [j.get('switch_pg_count') for j in nj if j.get('switch_pg_count')]
            if cnts:
                dum = sum(int(c.get('dummies', 0) or 0) for c in cnts)
                left_txt += (f'; group counts: {len(cnts)} ranks of this node reported, largest count {cnts[0].get("max")}' +
                             (f', {dum} dummy group(s) created on this node to equalize' if dum else ', all equal'))
            inits = [j.get('switch_init_s') for j in nj if j.get('switch_init_s') is not None]
            if inits:
                left_txt += f'; the first all-to-all (communicator formation) took {max(inits):.1f} s on this node'
            n_here = len([j for j in nj if j['rank'] not in set(left)])
            to_sw = [j for j in nj if (j.get('a2a') or {}).get('status') == 'timeout']
            not_tested = [j for j in nj if (j.get('a2a') or {}).get('status') == 'not_tested']
            fulls = next((j.get('switch_full_nodes') for j in nj if j.get('switch_full_nodes') is not None), None)
            if fulls is not None:
                left_txt += f'; the switch group held the {len(fulls)} full node(s) {fulls}'
            all_left = nj and all(j.get('exited') and j.get('a2a') is None for j in nj)
            if all_left:
                # the cause is named here (2026-10-10: in job 5646336 every rank left with "DistBackendError: ... retrieving ncclUniqueId" after
                # the store server was killed, and the row only said "left early")
                errs = sorted(set((j.get('error') or '').split(':')[0] for j in nj if j.get('error')))
                row('switch', '-', round(run_sw, 1), 'timeout', f'no rank of this node reached the switch test: all {len(nj)} left during the pair rounds ({nj[0].get("exited")}'
                    + (f'; errors: {", ".join(errs)}: {next(j["error"] for j in nj if j.get("error"))[:200]}' if errs else '') + f'); {cxi_txt}')
            elif not_tested and not a2a:
                row('switch', '-', round(run_sw, 1), 'not tested', f'{not_tested[0]["a2a"]["note"]}; {cxi_txt}{left_txt}')
            elif a2a and len(a2a) == n_here:
                vals = [j['a2a']['busbw'] for j in a2a]
                row('switch', min(vals), round(run_sw, 1), 'ok',
                    f'busbw min of {len(vals)} ranks of this node (max {max(vals)}); {a2a[0]["a2a"]["world"]} ranks x {a2a[0]["a2a"]["bytes"] // MB} MiB x {A2A_ITERS}; {cxi_txt}{left_txt}')
            elif to_sw or killed:
                mn = min([j['a2a']['busbw'] for j in a2a], default='-')
                row('switch', mn, round(run_sw, 1), 'timeout', f'{len(a2a)} of {n_here} surviving ranks reported a busbw; {len(to_sw)} rank(s) hit the switch deadline of {SWITCH_DEADLINE_S:.0f} s'
                    + (f' ({to_sw[0]["a2a"].get("note", "")[:120]})' if to_sw else '') + f'; {cxi_txt}{left_txt}')
            else:
                mn = min([j['a2a']['busbw'] for j in a2a], default='-')
                errs = [j['a2a'].get('note', '') for j in nj if (j.get('a2a') or {}).get('status') == 'error'] + [j['error'] for j in nj if j.get('error')]
                row('switch', mn, round(run_sw, 1), 'error', f'{len(a2a)} of {n_here} surviving ranks reported a busbw ({errs[0][:160] if errs else "no number"}); {cxi_txt}{left_txt}')
        # ---- rdma (phase gdr, only when it ran)
        if rg or JG:
            gj = [JG[r] for r in rg if r in JG]
            rcs_g = [_rc(a.net_dir, r, 'gdr') for r in rg]
            killed_g = any(rc is not None and 124 <= rc <= 137 for rc in rcs_g)
            act = sorted([j for j in gj if j.get('ring') and j['ring'].get('active') and j['ring'].get('gbps') is not None], key=lambda j: j['local'])
            per_nic = ','.join(f'nic{j["ring"]["nic"]}={j["ring"]["gbps"]}' for j in act)
            run_g = max([j.get('t_total_s') or 0 for j in gj] or [0])
            lines = _rccl_lines(a.net_dir, 'gdr', node, [j['pid'] for j in gj])
            ofi = [l for l in lines if 'NET/OFI' in l]
            prov = next((re.search(r'(Selected Provider is \S+)', l).group(1) for l in ofi if 'Selected Provider' in l), 'provider line not found')
            if not rg:
                row('rdma', '-', 0, 'timeout' if killed_g else 'missing', 'no rank of this node ran the gdr phase (no output files)')
            elif gj and not ofi:
                row('rdma', '-', run_g, 'missing', f'no NET/OFI line in the RCCL log of this node: the aws-ofi-rccl plugin is not loaded; {per_nic or "no transfer"}')
            elif len(act) == 4:
                row('rdma', min(j['ring']['gbps'] for j in act), run_g, 'ok',
                    f'min of 4 NICs (GPU buffers, GPU-direct requested: NCCL_NET_GDR_LEVEL=PHB); {per_nic}; {_path_text(lines, "PHB")}; {prov}')
            else:
                mn = min([j['ring']['gbps'] for j in act], default='-')
                errs = [j['error'] for j in gj if j.get('error')]
                row('rdma', mn, run_g, 'timeout' if killed_g else 'error', f'{len(act)} of 4 NICs reported ({errs[0][:120] if errs else "a ring rank gave no number"}); {per_nic or "none"}; {_path_text(lines, "PHB")}')
        with open(os.path.join(a.out, f'{node}.net.tsv'), 'w') as fh:
            fh.write('\n'.join(rows) + '\n')
        print('\n'.join(rows))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['net', 'gdr'])
    ap.add_argument('--net-dir', required=True)
    ap.add_argument('--nodes', required=True, help='the step nodes, in Slurm order')
    ap.add_argument('--init-timeout', type=int, default=120)
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--out', default=None, help='summarize: the raw dir that receives <node>.net.tsv')
    ap.add_argument('--budgets', default='', help='summarize: pair=s,switch=s,rdma=s (for the timeout notes)')
    ap.add_argument('--schedule', type=int, default=0, help='print the pairing schedule for K nodes and exit')
    a = ap.parse_args()
    if a.schedule:
        for r, rnd in enumerate(pair_schedule(a.schedule)):
            print(f'round {r + 1}: ' + '  '.join(f'({i},{j})' + ('*' if nt else '') for i, j, nt in rnd))
        return
    if a.summarize:
        a.out = a.out or os.path.dirname(os.path.abspath(a.net_dir))
        sys.exit(summarize(a))
    if not a.phase:
        ap.error('--phase net|gdr or --summarize')
    sys.exit(run_phase(a))


if __name__ == '__main__':
    main()
