# ftv2_srun.sh (ft-sys-v2 recovery, 2026-11-06): sourced in the batch shell through AUTORUN_PRE_CMD, after hb_start.sh. It makes
# `srun` a function of the batch shell that changes two of the template's srun calls and passes every other one through:
#   the Triton cache warm-up (cache_setup.sh)   runs on the standby node(s) too
#   the training srun (ELMoE_launch.py)         becomes, for each arm of FTV2_ARMS, ONE step over every node of the allocation:
#        tasks 0 .. W-1 train (W = FTV2_WORLD = the template's -n), tasks W .. are the standby (FTV2_STANDBY_NODES nodes, the last
#        ones), --kill-on-bad-exit=0 (a dead rank does not end the others). One step, because on Frontier each Slurm step has its
#        own Slingshot VNI (SwitchParameters job_vni=user): processes of two steps cannot reach each other over the network (P1c
#        5861771: a standby started as its own step failed the first collective of the new world). Every process splits MPI's
#        COMM_WORLD by role at start (rc_controller.split_mpi_world), so Megatron sees the W training ranks only.
#   Each arm: FTV2_RUN=<job>.<arm>, its logs in ${JOB_DIR}/<arm>/ (rank_<task>.log; the standby's tasks are W .. W+7), its own
#   environment (the arm's K=V list).
# FTV2_ARMS="name1:K=V,K=V name2:K=V": the arms, one after another in the same allocation (default: one arm "main").
# An arm with FTV2_NO_STANDBY=1 runs on the training nodes only (for an arm with the recovery off, XMOE_FT_V2=0);
# FTV2_LOSE_NODE=<i>:<step> loses node i at that step (tests/lose_node.sh); FTV2_DROP_ARGS="--a --b" removes those
# arguments from the training command (the cold-restart arm drops the template's --finetune to resume fully).
srun() {
  local x is_train=0 is_cache=0
  for x in "$@"; do
    case "$x" in *ELMoE_launch.py*) is_train=1 ;; *cache_setup.sh*) is_cache=1 ;; esac
  done
  if [ $is_train = 0 ] && [ $is_cache = 0 ]; then command srun "$@"; return $?; fi
  local nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
  local ns=${FTV2_STANDBY_NODES:-1} nn=${#nodes[@]} g=${FTV2_GPUS_PER_NODE:-8}
  local nt=$((nn - ns)) args=("$@") i
  if [ $is_cache = 1 ]; then
    local a=()
    for ((i = 0; i < ${#args[@]}; i++)); do
      case "${args[$i]}" in -N|-n) a+=("${args[$i]}" "$nn"); i=$((i + 1)) ;; *) a+=("${args[$i]}") ;; esac
    done
    echo "[ftv2 srun] $(date '+%T') cache warm-up on all $nn nodes" >&2
    command srun "${a[@]}"; return $?
  fi
  local tw sw aw
  tw=$(IFS=,; echo "${nodes[*]:0:$nt}"); sw=$(IFS=,; echo "${nodes[*]:$nt}"); aw=$(IFS=,; echo "${nodes[*]}")
  if [ -n "${PERSISTENT_TRITON_CACHE:-}" ] && [ -n "${TMP_TRITON_DIR:-}" ] && [ "${PROFILING_SKIP_CACHE_WARM:-}" != 1 ]; then
    # the template warmed the Triton cache on the training nodes before this file was sourced; the standby gets the same
    echo "[ftv2 srun] $(date '+%T') Triton cache warm-up on the standby node(s) $sw" >&2
    command srun -N "$ns" -n "$ns" --ntasks-per-node=1 -w "$sw" bash cache_setup.sh "$PERSISTENT_TRITON_CACHE" "$TMP_TRITON_DIR"
  fi
  local arms=(${FTV2_ARMS:-main:}) arm rc=0
  for arm in "${arms[@]}"; do
    (
      set +u
      name=${arm%%:*}; kv=${arm#*:}
      mkdir -p "$JOB_DIR/$name" "$FTV2_OUT/$name/ft"
      export FTV2_ARM=$name FTV2_ARM_OUT=$FTV2_OUT/$name FTV2_RUN=${SLURM_JOB_ID}.$name FTV2_WORLD=$((nt * g)) FTV2_ONE_STEP=1
      export NCCL_DEBUG_TIMESTAMP_LEVELS=${NCCL_DEBUG_TIMESTAMP_LEVELS:-ALL}   # a time on every RCCL line (communicator setup times)
      IFS=, read -ra kvs <<< "$kv"
      for e in "${kvs[@]}"; do [ -n "$e" ] && export "$e"; done
      # a failure arm records the optimizer state's fingerprint at the step ends around the failure (M1-B's restore check)
      fs=""; [ -n "${FTV2_INJECT:-}" ] && fs=${FTV2_INJECT##*:}; [ -n "${FTV2_LOSE_NODE:-}" ] && fs=${FTV2_LOSE_NODE##*:}
      if [ -n "$fs" ] && [ -z "${FTV2_STEP_SUMS_FROM:-}" ]; then export FTV2_STEP_SUMS_FROM=$((fs - 1)) FTV2_STEP_SUMS_TO=$((fs + 1)); fi
      if [ "${FTV2_NO_STANDBY:-0}" = 1 ]; then nodes_n=$nt; w=$tw; else nodes_n=$nn; w=$aw; fi
      ta=()
      for ((i = 0; i < ${#args[@]}; i++)); do
        x=${args[$i]}
        case "$x" in
          -N) ta+=(-N "$nodes_n"); i=$((i + 1)); continue ;;
          -n) ta+=(-n "$((nodes_n * g))"); i=$((i + 1)); continue ;;
          --kill-on-bad-exit=1) ta+=(--kill-on-bad-exit=0); continue ;;
        esac
        if [[ "$x" == *ELMoE_launch.py* ]]; then
          x="${x//\/rank_/\/$name\/rank_}"
          for d in ${FTV2_DROP_ARGS:-}; do x="${x// $d / }"; done     # test arms: e.g. --finetune off for a full resume
          ta+=("$x")
        else ta+=("$x"); fi
      done
      echo "[ftv2 srun] $(date '+%T') arm $name ($kv): one step on $w ($((nodes_n * g)) tasks): training tasks 0-$((nt * g - 1)) on $tw$([ "$nodes_n" -gt "$nt" ] && echo ", standby tasks $((nt * g))-$((nodes_n * g - 1)) on $sw"); run $FTV2_RUN" >&2
      if [ -n "${FTV2_LOSE_NODE:-}" ]; then      # test: a node lost at a step (tests/lose_node.sh <index> <step>)
        bash "$(dirname "${BASH_SOURCE[0]}")/../tests/lose_node.sh" ${FTV2_LOSE_NODE//:/ } > "$FTV2_OUT/$name/ft/lose_node.log" 2>&1 &
      fi
      command srun -w "$w" "${ta[@]}"
      r=$?
      echo "[ftv2 srun] $(date '+%T') arm $name: step ended ($r)" >&2
      exit $r
    )
    rc=$?
  done
  return $rc
}
