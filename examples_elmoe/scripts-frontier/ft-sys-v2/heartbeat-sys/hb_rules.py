"""hb_rules.py (ft-sys-v2 heartbeat system, 2026-11-05): the coordinator's decisions, as pure functions of what it has read.

No input or output here, so every case runs as a test on the login node (tests/test_rules.py). The rules are the ones decided
on the design map (11-04):
  2.1 hang       signal 1: no call count moved anywhere for STILL seconds AND the counts differ -> the lowest is the suspect;
                 signal 2: the suspect's stack is outside any communication wait while the others wait (hb_stacks.verdict).
  2.4 node lost  signal 1: its lease ended (etcd deleted its beat); signal 2: the other nodes stopped progressing.
                 If they still progress, the node is alive in training: a warning, not a replacement (2.5).
  crash          the node's agent saw the process gone on two beats in a row, or found its exit record.
  pause          every rank still with equal counts (an optimizer step, a save): nothing; after LONG seconds an alarm only
                 (the open "equal counts" case, design map 2.1).
Times are the coordinator's own clock: it compares what it read at different moments, never times written on other nodes."""

import os
import re

STILL = float(os.environ.get('FTV2_HANG_STILL_S', '15'))       # seconds without any count moving before a hang is suspected
                                                                #   (the toy's stuck rule, recovery_core.py)
STILL_SAVE = float(os.environ.get('FTV2_SAVE_STILL_S', '300'))  # the same while any rank reports the save phase (ft-sys-v1 lost a
                                                                #   healthy spare to a slow save)
CONFIRM = float(os.environ.get('FTV2_LOST_CONFIRM_S', '10'))    # seconds the other nodes must stay still after a lost lease
LONG = float(os.environ.get('FTV2_EQUAL_STILL_S', '120'))      # everyone still with equal counts: an alarm after this long
SETTLE = float(os.environ.get('FTV2_CRASH_SETTLE_S', '25'))    # with only communication deaths, how long to wait for a lost node

# A death whose exception names a communication failure is a consequence: a peer died and this rank's call failed with it.
COMM_ERROR = re.compile(r'DistBackendError|DistNetworkError|NCCL|RCCL|Gloo|gloo|Connection (closed|reset)|connection reset|'
                        r'remote process|peer|Socket|socket|timed out|Timeout|ProcessGroup', re.I)


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


def hang_suspect(track, now, still=STILL, still_save=STILL_SAVE):
    """Signal 1 of a hang: (node, rank, still_for) or None."""
    nodes = track.training()
    if not nodes:
        return None
    last_move = max(s['changed'] for s in nodes.values())
    saving = any('save' in s['beat'].get('phases', []) for s in nodes.values())
    limit = still_save if saving else still
    still_for = now - last_move
    if still_for < limit:
        return None
    low = min(s['low'] for s in nodes.values())
    high = max(s['high'] for s in nodes.values())
    if low >= high:
        return None                                   # equal counts: a pause, or the open case (see all_still_equal)
    node = sorted(n for n, s in nodes.items() if s['low'] == low)[0]
    return node, nodes[node]['beat'].get('low_rank'), round(still_for, 1)


def all_still_equal(track, now, long=LONG):
    """Everyone still with equal counts for LONG seconds: an alarm only (no single suspect)."""
    nodes = track.training()
    if not nodes:
        return False
    last_move = max(s['changed'] for s in nodes.values())
    low = min(s['low'] for s in nodes.values())
    high = max(s['high'] for s in nodes.values())
    return now - last_move >= long and low == high and not any('save' in s['beat'].get('phases', []) for s in nodes.values())


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
    return 'consequence' if COMM_ERROR.search(why) else 'root'


OOM = re.compile(r'OutOfMemoryError|out of memory|HIP out of memory|CUDA out of memory', re.I)


def death_kind(why):
    """The failure kind of a root death: 'out of memory' (the plan is the OOM ladder, design map 4.1, no node replaced) or 'crash'."""
    return 'out of memory' if why and OOM.search(why) else 'crash'


def counts_differ(track):
    """Among the nodes that still beat, is some rank behind the others? Then a silent node is not yet proven to be the cause:
    a hang on a beating node stalls everyone too (found on GPUs, job 5854211: an agent killed while another node's rank hung)."""
    nodes = track.training()
    if not nodes:
        return False
    return min(s['low'] for s in nodes.values()) < max(s['high'] for s in nodes.values())
