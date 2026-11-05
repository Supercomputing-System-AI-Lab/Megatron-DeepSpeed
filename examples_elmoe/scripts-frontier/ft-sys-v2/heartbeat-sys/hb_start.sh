#!/bin/bash
# hb_start.sh (ft-sys-v2 heartbeat system, 2026-11-05): start the heartbeat system inside a Slurm allocation, from the batch script,
# before the training srun. Source it (". hb_start.sh") so FTV2_ETCD reaches the training's environment:
#   etcd, one member on each of the first 3 nodes (client port 2379 on the node's hsn0 address), data in /dev/shm;
#   one agent per node (the last node is the standby when FTV2_STANDBY=1).
# Every process is its own srun step with --overlap, so it shares the nodes with training and survives a training crash.
# FTV2_PY: the Python for the agents (no torch needed); FTV2_ETCD_BIN: the etcd binary (get_etcd.sh fetches a checked one);
# FTV2_OUT: the job folder on Lustre (the slow heartbeat goes to $FTV2_OUT/ft/). FTV2_ABORT_ON_FAILURE=<seconds> (tests only):
# once a failure list exists, wait that long for the records, then end the job (no recovery yet).
# FTV2_GATE=quick|full: first the pre-run gate (../gate/ftv2_gate.sh, ft-sys-v1's gate on the new stack) on every node; its result
# goes to $FTV2_OUT/ft/gate/ and, once etcd is up, to the etcd key gate/result.
HB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FTV2_PY=${FTV2_PY:-python3}
FTV2_ETCD_BIN=${FTV2_ETCD_BIN:?set FTV2_ETCD_BIN to the etcd binary}
FTV2_OUT=${FTV2_OUT:?set FTV2_OUT to the job folder}
mkdir -p "$FTV2_OUT/ft"
_hb_nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
[ -n "${FTV2_GATE:-}" ] && . "$HB_DIR/../gate/ftv2_gate.sh"
_hb_ip() { srun --overlap -N1 -n1 -w "$1" bash -c "ip -4 addr show hsn0 | awk '/inet /{print \$2}' | cut -d/ -f1 | head -1"; }
_hb_cluster=""; FTV2_ETCD=""
for i in 0 1 2; do
  n=${_hb_nodes[$i]}; ip=$(_hb_ip $n)
  _hb_cluster="${_hb_cluster:+$_hb_cluster,}m$i=http://$ip:2380"; FTV2_ETCD="${FTV2_ETCD:+$FTV2_ETCD,}$ip:2379"
  eval "_hb_ip_$i=$ip"
done
export FTV2_ETCD
# etcd's peer transport (Go net/http) sends through http_proxy/https_proxy when they are set; the X-MoE environment sets them
# (proxy.ccs.ornl.gov:3128) and its no_proxy does not list the hsn0 addresses, so the members never reached each other (HB2
# 5854331, 2026-11-05). The members talk only inside the allocation: they run without the proxy variables.
_hb_noproxy="env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY"
for i in 0 1 2; do
  n=${_hb_nodes[$i]}; eval "ip=\$_hb_ip_$i"
  srun --overlap -N1 -n1 -w $n $_hb_noproxy "$FTV2_ETCD_BIN" --name m$i --data-dir /dev/shm/ftv2_etcd_${SLURM_JOB_ID}_m$i \
    --listen-client-urls http://$ip:2379 --advertise-client-urls http://$ip:2379 --listen-peer-urls http://$ip:2380 \
    --initial-advertise-peer-urls http://$ip:2380 --initial-cluster "$_hb_cluster" --initial-cluster-state new --log-level warn \
    > "$FTV2_OUT/ft/etcd_m$i.log" 2>&1 &
done
_hb_up=0; _hb_t0=$(date +%s)
for k in $(seq 1 60); do
  "$FTV2_PY" -I -c "import sys; sys.path.insert(0, '$HB_DIR'); from hb_etcd import Etcd; Etcd('$FTV2_ETCD'.split(',')).put('probe', '1')" 2>/dev/null && { _hb_up=1; break; }
  sleep 1
done
if [ $_hb_up = 0 ]; then
  echo "[hb_start] ERROR $(date '+%T'): etcd took no write in $(( $(date +%s) - _hb_t0 )) s; see $FTV2_OUT/ft/etcd_m*.log"
  # a test run cannot answer its question without the heartbeat system: end it instead of holding the nodes to the wall
  [ -n "${FTV2_ABORT_ON_FAILURE:-}" ] && { echo "[hb_start] test run: ending the job"; scancel "$SLURM_JOB_ID"; }
fi
_hb_last=$(( ${#_hb_nodes[@]} - 1 ))
for i in $(seq 0 $_hb_last); do
  role=train; [ "${FTV2_STANDBY:-0}" = 1 ] && [ $i = $_hb_last ] && role=standby
  srun --overlap -N1 -n1 -c2 -w ${_hb_nodes[$i]} "$FTV2_PY" -I -u "$HB_DIR/hb_agent.py" --etcd "$FTV2_ETCD" --job-dir "$FTV2_OUT" \
    --role $role --gpus 8 > "$FTV2_OUT/ft/agent_${_hb_nodes[$i]}.log" 2>&1 &
done
[ -f "$FTV2_OUT/ft/gate/result.json" ] && "$FTV2_PY" -I -c "import sys; sys.path.insert(0, '$HB_DIR'); from hb_etcd import Etcd; Etcd('$FTV2_ETCD'.split(',')).put('gate/result', open('$FTV2_OUT/ft/gate/result.json').read())" 2>/dev/null
echo "[hb_start] etcd $FTV2_ETCD; agents on ${_hb_nodes[*]}; records in $FTV2_OUT/ft"
if [ -n "${FTV2_ABORT_ON_FAILURE:-}" ]; then
  ( while ! "$FTV2_PY" -I -c "import sys; sys.path.insert(0, '$HB_DIR'); from hb_etcd import Etcd; sys.exit(0 if Etcd('$FTV2_ETCD'.split(',')).range('failed/')[0] else 1)" 2>/dev/null; do sleep 5; done
    echo "[hb_start] $(date '+%T') a failure list exists; ending the job in ${FTV2_ABORT_ON_FAILURE} s (no recovery yet)"
    sleep "$FTV2_ABORT_ON_FAILURE"; scancel "$SLURM_JOB_ID" ) &
fi
