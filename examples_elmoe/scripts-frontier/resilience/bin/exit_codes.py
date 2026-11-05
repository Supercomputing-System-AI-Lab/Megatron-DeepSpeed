#!/usr/bin/env python3
"""The exit-code table of the resilient training system (design 1009_2026_RESILIENCE_DESIGN.md, section 3.4).

One table, read by the trainer (megatron/training.py imports it through hb.py) and by the job script
(`python exit_codes.py --bash` prints bash arrays). Columns: code, class, restart policy, whether the node is
excluded on a restart, hard/soft for the lemon list.
"""
import sys

# code: (class, retries allowed, exclude the named node?, lemon kind, plain meaning)
TABLE = {
    0:  ('planned_exit',        0, False, '-',    'the planned exit (wall - margin or train_iters reached)'),
    10: ('collective_hang',     2, True,  'soft', 'a collective did not finish: the watchdog fired'),
    11: ('gpu_oom',             2, False, '-',    'GPU out of memory caught in the rank'),
    12: ('filesystem_hang',     2, True,  'soft', 'the save did not finish by its deadline'),
    13: ('loss_nan_spike',      2, False, '-',    'loss NaN or spike in 3 steps in succession'),
    14: ('replicas_differ',     2, False, '-',    'the replica hash differs (not built)'),
    15: ('slow',                2, True,  'soft', 'step time above 2x the median'),
    16: ('driver_hip_error',    2, True,  'soft', 'an exception whose text names HIP'),   # [2026-10-11, sandbox, the user: 2 retries]
    17: ('framework_exception', 0, False, '-',    'any other exception: a code or data defect, no restart'),
    18: ('init_hang',           1, True,  'soft', 'no first iteration within T_init (the job script)'),
    19: ('host_oom',            1, False, '-',    'host memory above 97 percent (retried once: a second time is a defect of the system or the framework)'),
    20: ('node_lost',           2, True,  'hard', 'srun reports a lost node / Slurm marks it down'),
    21: ('rank_died',           2, True,  'soft', 'a rank exited with an unknown code or by a signal (srun --kill-on-bad-exit)'),
    # [2026-10-09 late evening] not an exit code of the trainer: the record class of a node the pre-run gate excluded (the record
    # line carries gate_<test> as its class, e.g. gate_gemm or gate_pair; the loop writes it with this code)
    30: ('gate_excluded',       0, True,  'soft', 'the pre-run gate excluded the node (class gate_<test> in the failure record)'),
}
CLASS_TO_CODE = {v[0]: k for k, v in TABLE.items()}


def name(code):
    return TABLE.get(int(code), ('unknown',))[0]


def bash():
    import shlex
    out = ['declare -A RES_CLASS RES_RETRIES RES_EXCLUDE RES_KIND RES_MEANING']
    for c, (cls, r, ex, kind, m) in sorted(TABLE.items()):
        out.append(f'RES_CLASS[{c}]={cls}; RES_RETRIES[{c}]={r}; RES_EXCLUDE[{c}]={1 if ex else 0}; RES_KIND[{c}]={kind}; '
                   f'RES_MEANING[{c}]={shlex.quote(m)}')
    return '\n'.join(out)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--bash':
        print(bash())
    elif len(sys.argv) > 2 and sys.argv[1] == 'name':
        print(name(sys.argv[2]))
    else:
        for c, (cls, r, ex, kind, m) in sorted(TABLE.items()):
            print(f'{c:3d}  {cls:20s} retries={r} exclude_node={int(ex)} lemon={kind:4s}  {m}')
