#!/bin/bash
# job_index_serve.sh (2026-10-13, sandbox): the job index on login10 port 8766 (loopback only): one http.server serving <out> and the
# watcher job_index_watch.py (every 120 s, nice 19, idle I/O class), which rewrites <out>/index.html and the run consoles.
#   Start:  setsid nohup bash job_index_serve.sh [out] [extra --tsv files...] > <out>/serve.log 2>&1 < /dev/null &
#   Stop:   kill the pids in <out>/serve.pids (by pid, never pkill -f)
#   Laptop: ssh -N -L 8766:localhost:8766 <user>@login10.frontier.olcf.ornl.gov, then http://localhost:8766/
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
# [2026-10-16, the production port] default <out>: THIS tree's scripts-frontier/dashboards/jobs (production when run from production,
# the sandbox when run from the sandbox); it was the sandbox's folder, hard-coded
OUT=${1:-$(cd "$HERE/../.." && pwd)/dashboards/jobs}
shift || true
TSV=(); for t in "$@"; do TSV+=(--tsv "$t"); done
[ ${#TSV[@]} -gt 0 ] && TSV=(--tsv /lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/assistant_deep_dive/2026-10-13-ds16b_data_quality/jobs.tsv "${TSV[@]}")
mkdir -p "$OUT"
python3 -m http.server 8766 --bind 127.0.0.1 --directory "$OUT" >> "$OUT/http.log" 2>&1 &
HP=$!
( cd "$HERE" && exec nice -n 19 ionice -c3 python3 job_index_watch.py --out "$OUT" --every 120 ${TSV[@]+"${TSV[@]}"} ) >> "$OUT/watch.log" 2>&1 &
WP=$!
echo "$HP $WP $$" > "$OUT/serve.pids"
echo "$(date '+%F %T') http.server pid $HP on 127.0.0.1:8766 serving $OUT; watcher pid $WP; this shell $$"
wait
