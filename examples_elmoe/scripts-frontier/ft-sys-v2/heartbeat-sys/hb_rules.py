"""hb_rules.py (ft-sys-v2 heartbeat system, 2026-11-05): the coordinator's decisions, as pure functions of what it has read.

No input or output here, so every case runs as a test on the login node (tests/test_rules.py). The rules are the ones decided
on the design map (11-04):
  2.1 hang       signal 1: no call count moved anywhere for STILL seconds AND the counts differ -> the lowest is the suspect;
                 signal 2: the suspect's stack is outside any communication wait while the others wait (hb_stacks.verdict).
  2.4 node lost  signal 1: its lease ended (etcd deleted its beat); signal 2: the other nodes stopped progressing.
                 If they still progress, the node is alive in training: a warning, not a replacement (2.5).
  crash          the node's agent saw the process gone, or found its exit record. Its kind decides the plan (the user, 11-05 11:50):
                 out of memory -> the OOM ladder; a hardware fault reported as an exception -> replace the node; any other
                 exception -> stop the training for debugging; a death by a signal -> replace the node.
  pause          every rank still with equal counts (an optimizer step, a save): nothing for LONG seconds; then, once training is
                 past its first steps, kind "stuck" (the user, 11-05 12:0x: "for a stuck over the nccl or anything, I will give it
                 3 mins"): every node's stacks are saved and training stops; finding the node or link (gate checks) comes later.
  no stacks      (2026-11-06, the real fault of V3-2L 5863006) the suspect's stacks cannot be taken: the agent gathers other
                 evidence (evidence_verdict): a rank in state D or a GPU that does not answer -> kind "hardware fault", replace
                 the node; no evidence -> the stuck rule at LONG seconds of stall: every training node checks its GPUs
                 (node_check_verdict), a failing node is replaced, none -> kind "stuck", stop. A decision in bounded time.
Times are the coordinator's own clock: it compares what it read at different moments, never times written on other nodes."""

import os
import re

STILL = float(os.environ.get('FTV2_HANG_STILL_S', '15'))       # seconds without any count moving before a hang is suspected
                                                                #   (the toy's stuck rule, recovery_core.py)
STILL_SAVE = float(os.environ.get('FTV2_SAVE_STILL_S', '300'))  # the same while any rank reports the save phase (ft-sys-v1 lost a
                                                                #   healthy spare to a slow save)
CONFIRM = float(os.environ.get('FTV2_LOST_CONFIRM_S', '10'))    # seconds the other nodes must stay still after a lost lease
LONG = float(os.environ.get('FTV2_EQUAL_STILL_S', '180'))      # everyone still with equal counts: stuck after this long
LONG_FROM_STEP = 2                                              # ... once every node is past its first steps (a long first-step
                                                                #   compile or autotune is still, with equal counts, and healthy)
SETTLE = float(os.environ.get('FTV2_CRASH_SETTLE_S', '25'))    # with only communication deaths, how long to wait for a lost node
CHECK_WAIT = float(os.environ.get('FTV2_CHECK_WAIT_S', '120'))  # the stuck rule's node checks: how long to wait for every answer


def evidence_verdict(evidence):
    """The evidence an agent gathered on ranks it could not stack-dump (2026-11-06, the real fault of V3-2L 5863006: two ranks
    inside the GPU driver, unkillable, py-spy could not attach). evidence: {rank: {'state': the main thread's /proc state letter,
    'wchan': ..., 'probe': 'ok' | 'fail' | 'timeout' | 'unavailable', 'nonblocking': 'outside' | 'waiting' | 'none'}}.
    Returns (fault, why): fault True when a rank's main thread is in state D (uninterruptible: inside the kernel, as a driver call
    that does not return) or its GPU's probe failed or did not answer; else False (no evidence of a fault: the stuck rule)."""
    reasons = []
    for r, e in sorted((evidence or {}).items(), key=lambda kv: str(kv[0])):
        if str(e.get('state', '')).startswith('D'):
            reasons.append(f'rank {r} in state D (wchan {e.get("wchan") or "?"})')
        if e.get('probe') in ('fail', 'timeout'):
            reasons.append(f'rank {r}: its GPU probe {e.get("probe")} ({e.get("probe_detail", "")[:80]})')
    if reasons:
        return True, '; '.join(reasons)
    seen = ', '.join(f'rank {r}: state {e.get("state", "?")}, probe {e.get("probe", "?")}, nonblocking stack '
                     f'{e.get("nonblocking", "?")}' for r, e in sorted((evidence or {}).items(), key=lambda kv: str(kv[0])))
    return False, f'no evidence of a fault ({seen or "nothing gathered"})'


def node_check_verdict(answers, want):
    """The stuck rule's node checks (every training node's agent probes each of its GPUs and reads its RAS counters).
    answers: {node: {'ok': bool, 'detail': ...}}; want: the training nodes. Returns the failing nodes, in order: a node whose check
    failed, then a node that did not answer (its probe hung, or its agent did)."""
    failed = [n for n in sorted(want) if n in answers and not answers[n].get('ok')]
    silent = [n for n in sorted(want) if n not in answers]
    return failed + silent

# A death whose exception names a communication failure is a consequence: a peer died and this rank's call failed with it.
# Whole words only: case-insensitive "peer" matched inside "TyPEError", so every TypeError counted as a communication error
# (E5c 5862600: 24 ranks gone with a TypeError, read as "communication errors only", no list for 10 minutes).
COMM_ERROR = re.compile(r'\b(?:DistBackendError|DistNetworkError|NCCL|RCCL|Gloo|Connection (?:closed|reset)|remote process|'
                        r'peer|Socket|timed out|Timeout|ProcessGroup\w*)\b', re.I)


class Track:
    """What the coordinator remembers per node between reads: its last counts and when they last changed (its own clock)."""

    def __init__(self):
        self.nodes = {}           # node -> {'low', 'high', 'changed', 'beat', 'seen'}
        self.missing = {}         # node -> the time its beat disappeared

    def update(self, beats, now):
        """beats: {node: node beat} as read from etcd at time now. Returns the nodes whose beat disappeared since the last read."""
        gone = []
        for node, b in beats.items():
            s = self.nodes.get(node)
            if s is None or b.get('run') != s['beat'].get('run'):
                self.nodes[node] = {'low': b.get('low'), 'high': b.get('high'), 'changed': now, 'beat': b, 'seen': now}
                continue
            if (b.get('low'), b.get('high')) != (s['low'], s['high']):
                s.update(low=b.get('low'), high=b.get('high'), changed=now)
            s.update(beat=b, seen=now)
            self.missing.pop(node, None)
        for node in list(self.nodes):
            if node not in beats and node not in self.missing:
                self.missing[node] = now
                gone.append(node)
        return gone

    def note_absent(self, members, beats, now):
        """Members (lasting keys) with no beat: missing too, even if this coordinator never saw them (they vanished before it
        took the role, the old coordinator's own node for one). Returns the newly missing."""
        new = []
        for node, role in members.items():
            if node in beats or node in self.missing:
                continue
            if node not in self.nodes:
                self.nodes[node] = {'low': None, 'high': None, 'changed': now, 'beat': {'role': role}, 'seen': None}
            self.missing[node] = now
            new.append(node)
        return new

    def training(self, exclude=()):
        return {n: s for n, s in self.nodes.items()
                if n not in self.missing and n not in exclude and s['beat'].get('role', 'train') == 'train'
                and s['low'] is not None}


def by_stage(nodes):
    """{stage: {node: state}}. Under a pipeline the stages make different numbers of communication calls by design, so counts
    are only compared inside a stage (m3_map question 3; design map 2.1). Without a pipeline every node is stage 0."""
    out = {}
    for n, s in nodes.items():
        st = s['beat'].get('stage')
        out.setdefault(0 if st is None else st, {})[n] = s
    return out


def hang_suspect(track, now, still=STILL, still_save=STILL_SAVE):
    """Signal 1 of a hang: (node, rank, still_for) or None. Every training node still for STILL seconds, and inside some stage
    the counts differ: the suspect is the node furthest behind its own stage's highest count."""
    nodes = track.training()
    if not nodes:
        return None
    last_move = max(s['changed'] for s in nodes.values())
    saving = any('save' in s['beat'].get('phases', []) for s in nodes.values())
    limit = still_save if saving else still
    still_for = now - last_move
    if still_for < limit:
        return None
    best = None
    for st, group in sorted(by_stage(nodes).items(), key=lambda kv: str(kv[0])):
        high = max(s['high'] for s in group.values())
        for n, s in sorted(group.items()):
            gap = high - s['low']
            if gap > 0 and (best is None or gap > best[0]):
                best = (gap, n)
    if best is None:
        return None                                   # equal counts in every stage: a pause, or the open case (all_still_equal)
    node = best[1]
    return node, nodes[node]['beat'].get('low_rank'), round(still_for, 1)


def all_still_equal(track, now, long=LONG):
    """Everyone still with equal counts (inside each stage) for LONG seconds: an alarm only (no single suspect)."""
    nodes = track.training()
    if not nodes:
        return False
    last_move = max(s['changed'] for s in nodes.values())
    equal = all(min(s['low'] for s in g.values()) == max(s['high'] for s in g.values()) for g in by_stage(nodes).values())
    return (now - last_move >= long and equal and not any('save' in s['beat'].get('phases', []) for s in nodes.values())
            and min(s['beat'].get('step', -1) for s in nodes.values()) >= LONG_FROM_STEP)


def lost_node_verdict(track, node, now, confirm=CONFIRM, confirm_save=STILL_SAVE):
    """After a node's beat disappeared (its lease ended): 'lost' if the other training nodes stopped progressing for CONFIRM
    seconds, 'alive' if any of them moved after the beat disappeared (the agent died, the node still trains), 'wait' until one
    of the two is known. A standby node that disappears is 'standby' (nothing to replace). While any node saves, everyone is
    still for a long time anyway, so the wait is STILL_SAVE: an agent that dies during a save is not taken for a lost node."""
    s = track.nodes.get(node)
    if s and s['beat'].get('role') == 'standby':
        return 'standby'
    if s and s.get('seen') is None and not track.training(exclude=(node,)):
        return 'unknown'                  # never seen beating and no training to compare with: nothing to decide
    t_gone = track.missing.get(node, now)
    others = track.training(exclude=(node,))
    if not others:
        return 'lost'
    saving = any('save' in o['beat'].get('phases', []) for o in others.values())
    if saving:
        confirm = confirm_save
    last_move = max(o['changed'] for o in others.values())
    if any(o['changed'] > t_gone for o in others.values()):
        # the node trained on without its agent. It cannot be checked any more (no agent, no stacks), so if the training later
        # stalls for the hang limit it is the silent node's turn (the coordinator first asks whether a beating node is the suspect)
        return 'lost-stalled' if now - last_move >= max(STILL_SAVE if saving else STILL, confirm) else 'alive'
    if now - max(last_move, t_gone - confirm) >= confirm and now - t_gone >= 0:
        return 'lost'
    return 'wait'


def crash_report(beat):
    """The ranks a node's agent reports as gone (two beats in a row, or an exit record): [] when none."""
    return list(beat.get('dead', []))


def death_cause(why):
    """'root' for a death by signal (no exit record) or by an error that is not about communication; 'consequence' for an
    exception that names a communication failure (a peer was gone first)."""
    if not why or why == 'signal':
        return 'root'
    if OOM.search(why):
        return 'root'                  # an out-of-memory error inside the communication library (a buffer it could not get) is the
                                       #   rank's own memory, not a peer that went away (the user, 11-05)
    return 'consequence' if COMM_ERROR.search(why) else 'root'


OOM = re.compile(r'OutOfMemoryError|out of memory|HIP out of memory|CUDA out of memory|hipErrorOutOfMemory', re.I)
# a fault of the GPU or its links that the runtime reports as an exception: the node is the cause, not the program (the user, 11-05).
# "illegal memory access" is left out on purpose: it is most often a kernel bug, which a new node would hit again.
HARDWARE = re.compile(r'\bECC\b|uncorrectable|xGMI|HW_EXCEPTION|hardware exception|GPU (?:is )?lost|device (?:is )?lost|'
                      r'hipErrorECCNotCorrectable', re.I)


def death_kind(why):
    """The failure kind of a root death:
    'out of memory'  OutOfMemoryError and the like (plan: the OOM ladder, design map 4.1, no node replaced)
    'hardware fault' an exception that names a GPU or link fault (plan: replace the node)
    'exception'      any other exception: an error in the program (plan: stop the training for debugging)
    'killed'         a death by a signal, or any other reason (plan: replace the node)"""
    if why and OOM.search(why):
        return 'out of memory'
    if why and str(why).startswith('exception'):
        return 'hardware fault' if HARDWARE.search(why) else 'exception'
    return 'killed'                     # was 'crash' until 11-05 (the user: one name for one thing, as Figure 16)


def counts_differ(track):
    """Among the nodes that still beat, is some rank behind the others? Then a silent node is not yet proven to be the cause:
    a hang on a beating node stalls everyone too (found on GPUs, job 5854211: an agent killed while another node's rank hung)."""
    nodes = track.training()
    if not nodes:
        return False
    return min(s['low'] for s in nodes.values()) < max(s['high'] for s in nodes.values())
