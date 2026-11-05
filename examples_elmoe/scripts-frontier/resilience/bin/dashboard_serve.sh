#!/bin/bash
# dashboard_serve.sh (2026-10-10, sandbox): serve a run console folder from ANY login node, on the loopback address only.
# The page itself is written to Lustre (by the heartbeat reader inside a job, or by dashboard_watch.py on another login node), and every
# login node sees Lustre, so this only serves the file; it builds nothing and touches no job.
#   bash dashboard_serve.sh [folder] [port]      default folder: the 256k console (dashboards/lc256k), default port 8765 (next free one if taken)
# Then: VS Code Ports panel -> Forward a Port -> the port printed below -> open http://localhost:<port>/ on the laptop
#   or, from a laptop terminal:  ssh -N -L <port>:localhost:<port> <user>@<this login node>   (prints nothing while the tunnel is open)
set -u
J=${1:-/lustre/orion/gen150/scratch/zixianw4/X-MoE-resilience/Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier/dashboards/lc256k}
PORT=${2:-8765}
while python3 -c "import socket,sys; s=socket.socket(); s.bind(('127.0.0.1',$PORT))" 2>/dev/null; [ $? -ne 0 ]; do PORT=$((PORT+1)); done
PAGE=index.html; [ -f "$J/index.html" ] || PAGE=dashboard.html
[ -f "$J/$PAGE" ] || echo "note: $J has no index.html or dashboard.html yet"
echo "serving $J on this node ($(hostname)) at http://127.0.0.1:$PORT/$PAGE"
echo "laptop: forward port $PORT in VS Code (Ports panel), then open http://localhost:$PORT/$PAGE"
exec python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$J"
