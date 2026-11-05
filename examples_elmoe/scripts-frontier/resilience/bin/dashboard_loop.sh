#!/bin/bash
# dashboard_loop.sh <job folder> <out dir> [interval s]: rewrite <out dir>/dashboard.log every interval (default 30 s) from the job's own
# files (dashboard.py is read-only on the job). Stops by itself when the job's state file says it is done or its STATUS.md has not
# changed for 15 min (the job is over). Writes nothing into the job folder and never touches the job.
JOB=$1; OUT=$2; INT=${3:-30}; HERE=$(cd "$(dirname "$0")" && pwd); mkdir -p "$OUT"
while :; do
  nice -n 19 python3 "$HERE/dashboard.py" "$JOB" --out "$OUT/dashboard.log" 2>> "$OUT/dashboard.err"
  grep -q '^status=done' "$JOB/data/state" 2>/dev/null && break
  [ $(( $(date +%s) - $(stat -c %Y "$JOB/STATUS.md" 2>/dev/null || date +%s) )) -gt 900 ] && break
  sleep "$INT"
done
echo "$(date '+%F %T') the job is over; the last view stays in $OUT/dashboard.log" >> "$OUT/dashboard.err"
