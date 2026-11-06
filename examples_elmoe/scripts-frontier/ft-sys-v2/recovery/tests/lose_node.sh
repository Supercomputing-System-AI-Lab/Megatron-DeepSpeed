#!/bin/bash
# lose_node.sh <node index> <step> (ft-sys-v2 recovery tests, 2026-11-06): a node that loses power, for one arm of a recovery run.
# The heartbeat tests' runs/ftv2_lose_node.sh (2026-11-05, HB6) with one change: it fires only when that node's etcd beat reports
# the CURRENT arm's run (FTV2_RUN) at step >= <step>, so it can be one arm of several in one allocation. Then, on that node, SIGKILL:
# its agent, its etcd member and its training ranks (to the others: no beat, no lease renewal, peers that never answer). The
# patterns are anchored at the start of the command line, so they never match an srun client on the batch node.
# Started in the background by launch/ftv2_srun.sh when an arm sets FTV2_LOSE_NODE=<index>:<step>; needs FTV2_ETCD, FTV2_PY, FTV2_RUN.
set -u
i=$1; s=$2
hb="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../heartbeat-sys" && pwd)"
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
n=${nodes[$i]}
until "$FTV2_PY" -I -c "import sys, json; sys.path.insert(0, '$hb'); from hb_etcd import Etcd; v, _ = Etcd('$FTV2_ETCD'.split(',')).get('node/$n'); b = json.loads(v) if v else {}; sys.exit(0 if str(b.get('run')) == '$FTV2_RUN' and b.get('step', -1) >= $s else 1)" 2>/dev/null; do
  sleep 1
done
echo "[ftv2 test] $(date '+%T') losing $n (run $FTV2_RUN, step >= $s): its agent, its etcd member and its training ranks"
command srun --overlap -N1 -n1 -w "$n" bash -c "pkill -9 -u $USER -f '^[^ ]*python[^ ]* -I -u [^ ]*hb_agent\.py'; pkill -9 -u $USER -f '^[^ ]*/etcd --name'; pkill -9 -u $USER -f '^[^ ]*python[^ ]* \.\./ELMoE_launch\.py'; echo \"[ftv2 test] \$(date '+%T') killed on \$(hostname)\""
