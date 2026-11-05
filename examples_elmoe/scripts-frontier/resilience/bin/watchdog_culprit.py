#!/usr/bin/env python3
"""Name the stopped rank of a collective hang from the watchdog lines of the job's stderr (2026-10-09, after H16 5641684).
The ranks of the stuck group fire first; the group is a block of G consecutive ranks (the expert group at EP G, the TP group at
TP G); the member of the first-firing block that did NOT fire is the one that never entered the collective.
    watchdog_culprit.py <stderr file> <byte offset> <G>   ->  prints 'rank <r> block <b> fired <list>' or nothing
fr_analyze.py keys collectives by (process-group id, sequence number), but the id is numbered per process, so at TP > 1 groups
of different nodes share an id and its ABSENT sets mix groups (H16: it named ranks 0-7 instead of 13)."""
import re, sys
from datetime import datetime
f, off, G = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
pat = re.compile(r'^\[rank(\d+)\]:\[E(\d{4}) (\d\d:\d\d:\d\d\.\d+) .*Watchdog caught collective operation timeout')
fired = []
with open(f, 'rb') as fh:
    fh.seek(off)
    for raw in fh:
        m = pat.match(raw.decode('utf-8', 'replace'))
        if m:
            t = datetime.strptime(m.group(3)[:15], '%H:%M:%S.%f')
            fired.append((t, int(m.group(1))))
if not fired:
    sys.exit(0)
fired.sort()
t0 = fired[0][0]
first = sorted({r for t, r in fired if (t - t0).total_seconds() <= 5})
blk = fired[0][1] // G                      # the block of the EARLIEST firing rank
inblk = sorted(r for r in first if r // G == blk)
missing = [r for r in range(blk * G, blk * G + G) if r not in inblk]
if len(missing) >= 1 and len(missing) < G:
    print(f'rank {missing[0]} block {blk} fired {inblk} missing {missing}')
