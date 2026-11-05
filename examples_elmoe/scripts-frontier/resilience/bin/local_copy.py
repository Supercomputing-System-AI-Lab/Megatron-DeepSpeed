#!/usr/bin/env python3
"""local_copy.py: one line in words about the node-local checkpoint copies of an attempt (design 3.5; AUTORUN_LOCAL_COPIES >= 1),
from <job dir>/data/local_copy.tsv, written by the async checkpoint engine (deepspeed/runtime/checkpoint_engine/
async_pinned_checkpoint_engine.py): rows `time attempt rank node tag kind file bytes seconds shm_used_gb shm_free_gb`, kind in
shm_write | shm_skip | load_local | load_lustre | load_summary_local | load_summary_lustre.
  local_copy.py summary --data-dir <job dir>/data --attempt <k>      the loads of attempt k, per source, as one line (recovery.log)
  local_copy.py saves   --data-dir <job dir>/data                    the local writes: per tag the files, GB, seconds, shm used/free
restart_loop.sh calls both; the lines are prose for recovery.log, the tsv stays the record.
"""
import argparse, collections, os, sys


def rows(p):
    out = []
    try:
        for ln in open(p):
            v = ln.rstrip('\n').split('\t')
            if len(v) >= 11:
                try:
                    out.append(dict(time=v[0], attempt=v[1], rank=int(v[2]), node=v[3], tag=v[4], kind=v[5], file=v[6],
                                    bytes=float(v[7]), s=float(v[8]), used=float(v[9]), free=float(v[10])))
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    return out


def summary(R, attempt):
    sel = [r for r in R if r['attempt'] == str(attempt) and r['kind'] in ('load_local', 'load_lustre')]
    if not sel:
        return f'local copy: attempt {attempt}: no load rows in data/local_copy.tsv (a first run loads nothing, or the engine is off)'
    tag = sel[0]['tag']
    parts = []
    for kind, name in (('load_local', '/dev/shm'), ('load_lustre', 'Lustre')):
        per_rank = collections.defaultdict(lambda: [0, 0.0, 0.0])
        for r in sel:
            if r['kind'] == kind:
                a = per_rank[(r['rank'], r['node'])]; a[0] += 1; a[1] += r['bytes']; a[2] += r['s']
        if per_rank:
            secs = [a[2] for a in per_rank.values()]
            parts.append(f'{len(per_rank)} ranks read {sum(a[0] for a in per_rank.values())} files ({sum(a[1] for a in per_rank.values()) / 1e9:.2f} GB) '
                         f'from {name} in {min(secs):.2f}-{max(secs):.2f} s per rank (sum {sum(secs):.1f} s)')
    nodes_local = sorted({r['node'] for r in sel if r['kind'] == 'load_local'})
    nodes_lustre_only = sorted({r['node'] for r in sel if r['kind'] == 'load_lustre'} - set(nodes_local))
    return (f'local copy: attempt {attempt} loaded step {tag}: ' + '; '.join(parts) +
            (f'; nodes with a local hit: {" ".join(nodes_local)}' if nodes_local else '; no node had a local copy') +
            (f'; Lustre only: {" ".join(nodes_lustre_only)}' if nodes_lustre_only else ''))


def saves(R):
    w = [r for r in R if r['kind'] == 'shm_write']
    sk = [r for r in R if r['kind'] == 'shm_skip']
    if not w and not sk:
        return 'local copy: no local write rows in data/local_copy.tsv'
    by_tag = collections.defaultdict(list)
    for r in w:
        by_tag[r['tag']].append(r)
    lines = []
    for tag in sorted(by_tag, key=lambda t: int(''.join(c for c in t if c.isdigit()) or 0)):
        rr = by_tag[tag]
        per_node = collections.defaultdict(float)
        for r in rr:
            per_node[r['node']] += r['bytes']
        lines.append(f'local copy: {tag}: {len(rr)} files, {sum(r["bytes"] for r in rr) / 1e9:.2f} GB in /dev/shm over {len(per_node)} nodes '
                     f'({max(per_node.values()) / 1e9:.2f} GB on the fullest node), write {min(r["s"] for r in rr):.2f}-{max(r["s"] for r in rr):.2f} s per file; '
                     f'/dev/shm after: used {max(r["used"] for r in rr):.1f} GB, free {min(r["free"] for r in rr):.1f} GB (worst node)')
    if sk:
        lines.append(f'local copy: {len(sk)} local writes SKIPPED for space (tags {", ".join(sorted({r["tag"] for r in sk}))})')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['summary', 'saves'])
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--attempt', default='0')
    a = ap.parse_args()
    R = rows(os.path.join(a.data_dir, 'local_copy.tsv'))
    print(summary(R, a.attempt) if a.what == 'summary' else saves(R))


if __name__ == '__main__':
    main()
