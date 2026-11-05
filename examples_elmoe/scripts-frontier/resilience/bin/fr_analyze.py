#!/usr/bin/env python3
"""fr_analyze.py (2026-10-08): read PyTorch flight-recorder dumps (TORCH_NCCL_DUMP_ON_TIMEOUT, one pickle per rank) and name the
ranks that never entered the collective the others wait in. torch 2.9.1+rocm6.4 here ships no torch.distributed.flight_recorder
module, so this is the analyzer. For every process group and collective sequence number: which ranks recorded it, in which state.
A collective that is 'scheduled' on some ranks and absent on others names the culprits: the absent ranks.
    python3 fr_analyze.py <dir with nccl_trace_rank_*> [--prefix nccl_trace_rank_]
Validated on job 5639642 (one injected sleeper): "pg 10, seq 2497 all_to_all: scheduled on 15 ranks, ABSENT on rank 5"."""
import os, sys, glob, pickle, collections
d = sys.argv[1]; pre = sys.argv[sys.argv.index('--prefix') + 1] if '--prefix' in sys.argv else 'nccl_trace_rank_'
files = sorted(glob.glob(os.path.join(d, pre + '*')), key=lambda f: int(f.rsplit('_', 1)[-1]))
ranks = {}
for f in files:
    r = int(f.rsplit('_', 1)[-1])
    try: ranks[r] = pickle.load(open(f, 'rb'))
    except Exception as e: print(f'rank {r}: unreadable ({str(e)[:60]})')
print(f'{len(ranks)} rank dumps read from {d}')
# the newest entry per (pg, seq) per rank, and the set of ranks that hold it
last = collections.defaultdict(dict)          # (pg_name, seq) -> rank -> (state, name)
pg_size = {}
for r, dmp in ranks.items():
    for pg, cfg in (dmp.get('pg_config') or {}).items():
        rk = cfg.get('ranks') if isinstance(cfg, dict) else None       # a string like "[0, 1, 2]" in torch 2.9
        if isinstance(rk, str):
            import ast
            try: rk = ast.literal_eval(rk)
            except Exception: rk = None
        if rk is not None: pg_size[str(pg)] = list(rk) if hasattr(rk, '__iter__') else rk
    for e in dmp.get('entries', []):
        pg = str(e.get('pg_id', e.get('pg_name'))); key = (pg, int(e.get('collective_seq_id', e.get('seq_id', -1))))
        last[key][r] = (e.get('state'), e.get('profiling_name'))
        if pg not in pg_size:                      # torch 2.9 keys pg_config by the group's uid, carried in the entry as process_group "('130', 'undefined')"
            pgrp = e.get('process_group'); uid = None
            if isinstance(pgrp, (tuple, list)) and pgrp: uid = str(pgrp[0])
            elif isinstance(pgrp, str):
                import ast
                try: uid = str(ast.literal_eval(pgrp)[0])
                except Exception: uid = None
            if uid is not None and uid in pg_size: pg_size[pg] = pg_size[uid]
# a collective is suspicious if any rank holds it in a non-completed state
sus = []
for key, byrank in last.items():
    states = collections.Counter(st for st, _ in byrank.values())
    if any(st != 'completed' for st in states):
        name = next(iter(byrank.values()))[1]
        members = pg_size.get(key[0])
        expect = set(members) if isinstance(members, list) else (set(range(members)) if isinstance(members, int) else None)
        absent = sorted(expect - set(byrank)) if expect else sorted(set(ranks) - set(byrank))   # without the group's member list: the dumped ranks that lack the entry
        sus.append((key, name, dict(states), sorted(byrank), absent))
if not sus:
    print('no incomplete collective in any dump'); sys.exit(0)
for (pg, seq), name, states, holders, absent in sorted(sus, key=lambda x: -x[0][1])[:5]:
    print(f'pg {pg}, seq {seq}, {name}: states {states}; recorded on ranks {holders if len(holders) < 20 else str(len(holders)) + " ranks"}'
          + (f'; ABSENT on ranks {absent}  <- the ranks that never entered the collective' + ('' if expect else ' (group members not in the dump: absent among the dumped ranks)') if absent else '; every member recorded it'))
