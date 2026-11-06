#!/usr/bin/env python3
"""test_rules.py (ft-sys-v2 heartbeat system tests, 2026-11-05): the coordinator's decisions as pure functions, no etcd, no GPU.
Each case is a sequence of node beats at given times on the coordinator's clock, and the expected verdict.
    python3 -I tests/test_rules.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import hb_rules as R
import hb_stacks as K

fails = []


def check(name, got, want):
    ok = got == want
    print(f'{"PASS" if ok else "FAIL"}  {name}: got {got!r}, want {want!r}')
    if not ok:
        fails.append(name)


def beat(low, high, low_rank=0, run='r', phases=('forward',), role='train', step=5):
    return {'low': low, 'high': high, 'low_rank': low_rank, 'run': run, 'phases': list(phases), 'role': role, 'step': step}


def track(seq):
    t = R.Track()
    for now, beats in seq:
        t.update(beats, now)
    return t


# hang, signal 1: nobody moved for STILL seconds and the counts differ -> the lowest is the suspect
t = track([(0, {'n0': beat(100, 101), 'n1': beat(100, 101)}), (5, {'n0': beat(158, 158), 'n1': beat(155, 158, 13)}),
           (10, {'n0': beat(158, 158), 'n1': beat(155, 158, 13)}), (20, {'n0': beat(158, 158), 'n1': beat(155, 158, 13)})])
check('hang suspect after 15 s', R.hang_suspect(t, 20.1), ('n1', 13, 15.1))
check('no suspect before 15 s', R.hang_suspect(t, 19.0), None)
# a pause: everyone still with equal counts -> no suspect; an alarm only after LONG
t = track([(0, {'n0': beat(100, 101), 'n1': beat(100, 101)}), (5, {'n0': beat(150, 150), 'n1': beat(150, 150)}),
           (60, {'n0': beat(150, 150), 'n1': beat(150, 150)})])
check('pause: no suspect', R.hang_suspect(t, 60), None)
check('pause: no alarm before LONG', R.all_still_equal(t, 60), False)
check('pause: stuck after LONG (3 min)', R.all_still_equal(t, 5 + R.LONG), True)
check('stuck limit is 180 s', R.LONG, 180.0)
t = track([(0, {'n0': beat(100, 101, step=1), 'n1': beat(100, 101, step=1)}), (5, {'n0': beat(150, 150, step=1), 'n1': beat(150, 150, step=1)})])
check('still with equal counts in the first steps (a compile): not stuck', R.all_still_equal(t, 5 + R.LONG), False)
# a slow rank: the counts differ but keep moving -> nothing
t = track([(0, {'n0': beat(100, 101), 'n1': beat(99, 101)}), (5, {'n0': beat(180, 181), 'n1': beat(179, 181)}),
           (10, {'n0': beat(260, 261), 'n1': beat(259, 261)}), (15, {'n0': beat(340, 341), 'n1': beat(339, 341)})])
check('slow rank: no suspect', R.hang_suspect(t, 15), None)
# a save: the longer limit
t = track([(0, {'n0': beat(100, 101), 'n1': beat(100, 101)}), (5, {'n0': beat(120, 120, phases=('save',)), 'n1': beat(119, 120)})])
check('save: no suspect at 60 s', R.hang_suspect(t, 65), None)
check('save: suspect after the save limit', R.hang_suspect(t, 5 + R.STILL_SAVE)[0], 'n1')
# a lost node: its beat disappears; the others stopped -> lost; the others still move -> alive
seq = [(0, {'n0': beat(100, 100), 'n1': beat(100, 100), 'n2': beat(100, 100)}), (5, {'n0': beat(130, 130), 'n1': beat(130, 130)})]
t = track(seq)
check('lost: others stopped', R.lost_node_verdict(t, 'n2', 5 + R.CONFIRM), 'lost')
t = track(seq + [(10, {'n0': beat(170, 170), 'n1': beat(170, 170)})])
check('agent dead, node trains: alive', R.lost_node_verdict(t, 'n2', 10), 'alive')
t = track(seq + [(10, {'n0': beat(170, 170), 'n1': beat(170, 170)}), (40, {'n0': beat(170, 170), 'n1': beat(170, 170)})])
check('silent node, training then stalls: lost-stalled', R.lost_node_verdict(t, 'n2', 40), 'lost-stalled')
t = track([(0, {'n0': beat(100, 100), 'sb': beat(None, None, role='standby')}), (5, {'n0': beat(130, 130)})])
check('standby lost: nothing to replace', R.lost_node_verdict(t, 'sb', 5), 'standby')
# the cause and kind of a death
check('signal is a root', R.death_cause('signal'), 'root')
check('communication error is a consequence', R.death_cause('exception: RuntimeError: Connection reset by peer'), 'consequence')
check('NCCL error is a consequence', R.death_cause('exception: DistBackendError: NCCL error in: ...'), 'consequence')
check('ValueError is a root', R.death_cause('exception: ValueError: bad shape'), 'root')
check('evidence: a rank in state D is a fault', R.evidence_verdict({20: {'state': 'D (disk sleep)', 'probe': 'ok'}})[0], True)
check('evidence: a GPU probe that timed out is a fault', R.evidence_verdict({20: {'state': 'S (sleeping)', 'probe': 'timeout'}})[0],
      True)
check('evidence: a failed GPU probe is a fault', R.evidence_verdict({20: {'state': 'S', 'probe': 'fail'}})[0], True)
check('evidence: sleeping, probe ok: no fault', R.evidence_verdict({20: {'state': 'S (sleeping)', 'probe': 'ok'}})[0], False)
check('evidence: probe unavailable is not a fault', R.evidence_verdict({20: {'state': 'S', 'probe': 'unavailable'}})[0], False)
check('evidence: nothing gathered: no fault', R.evidence_verdict({})[0], False)
check('node checks: a failed node first, then a silent one',
      R.node_check_verdict({'a': {'ok': True}, 'b': {'ok': False}}, {'a', 'b', 'c'}), ['b', 'c'])
check('node checks: all passed', R.node_check_verdict({'a': {'ok': True}, 'b': {'ok': True}}, {'a', 'b'}), [])
check('TypeError is a root ("peer" inside "TypeError", E5c 5862600)',
      R.death_cause('exception: TypeError: slice indices must be integers or None or have an __index__ method'), 'root')
check('ProcessGroupNCCL text is a consequence', R.death_cause('exception: RuntimeError: ProcessGroupNCCL is aborted'), 'consequence')
check('a socket timeout is a consequence', R.death_cause('exception: RuntimeError: socket timed out after 600 s'), 'consequence')
check('gloo connection closed is a consequence',
      R.death_cause('exception: RuntimeError: [../third_party/gloo/gloo/transport/tcp/pair.cc:534] Connection closed by peer'),
      'consequence')
check('DistNetworkError is a consequence', R.death_cause('exception: DistNetworkError: failed to recv'), 'consequence')
check('OOM kind', R.death_kind('exception: OutOfMemoryError: HIP out of memory. Tried to allocate 2 TiB'), 'out of memory')
check('killed kind', R.death_kind('signal'), 'killed')
check('a program error: kind exception (stop for debugging)', R.death_kind('exception: RuntimeError: ftv2 injected exit'), 'exception')
check('a hardware fault raised as an exception: replace the node',
      R.death_kind('exception: RuntimeError: HIP error: uncorrectable ECC error encountered'), 'hardware fault')
check('an illegal memory access stays a program error', R.death_kind('exception: RuntimeError: HIP error: an illegal memory access was encountered'), 'exception')
check('OOM inside RCCL is the rank\'s own memory: a root', R.death_cause('exception: DistBackendError: NCCL error: hipErrorOutOfMemory'), 'root')
check('OOM inside RCCL: kind out of memory', R.death_kind('exception: DistBackendError: NCCL error: hipErrorOutOfMemory'), 'out of memory')
# signal 2, py-spy stacks
hung = 'Thread 7 (idle): "MainThread"\n    _inject (hb_monitor.py:62)\n    wrapped (hb_monitor.py:91)\n    forward (moe.py:10)\n'
wait = 'Thread 8 (active): "MainThread"\n    synchronize (torch/cuda/__init__.py:985)\n    all_to_all_single (c10d.py:4)\n'
check('stacks: confirmed', K.verdict({13: hung, 12: wait, 14: wait}, 13)[0], 'confirmed')
check('stacks: every rank waits -> ambiguous', K.verdict({13: wait, 12: wait}, 13)[0], 'ambiguous')
check('stacks: two outside -> ambiguous', K.verdict({13: hung, 12: hung}, 13)[0], 'ambiguous')
check('stacks: suspect gone -> no-stack', K.verdict({13: 'Error: no such process', 12: wait}, 13)[0], 'no-stack')
# the real X-MoE stacks of HB2b 5855002 (frontier08106, sample 0001), cut after a few frames: rank 13 asleep in the injection under
# broadcast_data (a collective name deeper in the stack), rank 12 in the HIP runtime's signal wait under the MoE layer
hung_x = ('Thread 1916988 (idle): "MainThread"\n    clock_nanosleep@GLIBC_2.2.5 (libc.so.6)\n    time_sleep (python3.11)\n'
          '    _inject (hb_monitor.py:59)\n    _count (hb_monitor.py:77)\n    wrapped (hb_monitor.py:93)\n'
          '    _build_key_size_numel_dictionaries (megatron/core/tensor_parallel/data.py:50)\n'
          '    broadcast_data (megatron/core/tensor_parallel/data.py:101)\n    get_batch (pretrain_gpt_deepspeed.py:463)\n')
wait_x = ('Thread 1916987 (active): "MainThread"\n    rocr::core::BusyWaitSignal::WaitRelaxed (libhsa-runtime64.so.1.21.0)\n'
          '    forward (deepspeed/moe/moe_v2.py:426)\n    _call_impl (torch/nn/modules/module.py:1789)\n'
          '    forward (deepspeed/moe/layer.py:280)\n')
check('X-MoE: asleep under broadcast_data is outside', K.is_waiting(K.main_thread_frames(hung_x)), False)
check('X-MoE: HIP signal wait is waiting', K.is_waiting(K.main_thread_frames(wait_x)), True)
check('X-MoE stacks: confirmed', K.verdict({12: wait_x, 13: hung_x, 14: wait_x}, 13)[0], 'confirmed')
# two dumps a few seconds apart (the user, 11-05): confirmed only if the suspect is stuck at the same place in both
busy = hung_x.replace('_inject (hb_monitor.py:59)', 'tokenize (data.py:120)').replace('clock_nanosleep@GLIBC_2.2.5 (libc.so.6)\n    time_sleep (python3.11)\n    ', '')
d = {12: wait_x, 13: hung_x, 14: wait_x}
check('two dumps, same place: confirmed', K.verdict_twice(d, dict(d), 13)[0], 'confirmed')
check('two dumps, the suspect moved (busy, not stuck): moved', K.verdict_twice(d, {12: wait_x, 13: busy, 14: wait_x}, 13)[0], 'moved')
check('two dumps, the second has every rank waiting: ambiguous', K.verdict_twice(d, {12: wait_x, 13: wait_x, 14: wait_x}, 13)[0], 'ambiguous')
print(f'\n{len(fails)} failed' if fails else '\nall passed')
sys.exit(1 if fails else 0)
