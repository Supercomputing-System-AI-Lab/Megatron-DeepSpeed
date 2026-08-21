#!/bin/bash
# monitor_run.sh — async, per-rank pipeline-stall monitor for the NON-SLURM (torchrun) ELMoE runs.
#
# CUDA/torchrun counterpart of scripts-frontier/monitor_run.sh. Same CLI, same output
# contract (a-monitor.txt / node{idx}.txt / monitor.tsv), same categorize() + ACTIVE_RE,
# so anything you learned reading the Frontier files transfers verbatim. Differences:
#   * host list      : ${HOSTFILE} (or purely local when NODES=1) instead of scontrol/SLURM_NODELIST
#   * job id         : ${JOB_ID} (minted by autorun.sh) instead of ${SLURM_JOB_ID}
#   * rank identity  : RANK / LOCAL_RANK (torchrun) instead of SLURM_PROCID / SLURM_LOCALID
#   * GPU filter     : /dev/nvidia* open fd instead of /dev/kfd
#   * telemetry      : nvidia-smi instead of rocm-smi; HBM is read PER-PROCESS via
#                      --query-compute-apps, which is exact and immune to
#                      CUDA_VISIBLE_DEVICES remapping (rocm-smi had to key off local rank)
#   * local fast path: when a host is this machine, the collector runs directly — no ssh.
#                      This is what makes single-node monitoring work with no key setup.
#   * dmesg scrape   : Xid/NVRM patterns instead of amdgpu/kfd
#
# Usage:  bash monitor_run.sh <JOB_DIR> [FAST_INTERVAL] [SLOW_INTERVAL] [FAST_WINDOW] [TRACE_FRAMES]
# Run BACKGROUNDED from elmoe.sh.template so it never blocks training:
#     bash "${DIR}/monitor_run.sh" "${JOB_DIR}" 60 180 300 20 &
#
# Output:  a-monitor.txt at <JOB_DIR>/ (top level, quick access); rest under <JOB_DIR>/monitor/ :
#   a-monitor.txt   — CENTRALIZED quick-glance file: per interval = a state HISTOGRAM across all
#                     ranks, a LIVE x/world liveness line (GONE/suspect/DEAD-HOST + last/peak HBM%
#                     & RAM_GB of any rank that died), the full per-node tables, then the culprit
#                     stack, then a one-shot dmesg scrape on any node that just lost a rank.
#   node{index}.txt — same per-node table but just that node (index = hostfile line order).
#   monitor.tsv     — machine view, one row per (interval,rank).
#
# Per-rank columns: STATUS RANK GPU PID WATTS HBM% RAM_GB TRACE(deepest<-outer).
#   WATTS   = GPU power draw (nvidia-smi power.draw, by GPU index = LOCAL_RANK)
#   HBM%    = this PROCESS's VRAM / the GPU's total, percent (nvidia-smi --query-compute-apps)
#   RAM_GB  = host resident RAM of this rank's process in GiB (ps rss)
#             flat RAM_GB + low WATTS + FROZEN = hung
# TRACE depth is the 5th arg (TRACE_FRAMES, default 20 frames, deepest first).
#
# frozen_ticks = consecutive unchanged-trace intervals.
# Cadence: FAST_INTERVAL for the first FAST_WINDOW sec, then SLOW_INTERVAL.
#
# Rank discovery uses `pgrep -u "$(id -u)"`, NOT $USER: $USER is unset under `env -i`,
# in cron/systemd units, and in non-login shells, and `pgrep -u ""` then errors out and
# silently reports zero ranks while training runs perfectly well.
#
# Requires: py-spy on PATH or at ~/.local/bin/py-spy (`pip install py-spy`), on EVERY node.
#           ptrace must be permitted: /proc/sys/kernel/yama/ptrace_scope = 0.
# Env in  : JOB_ID, NODES, TOTAL_GPUS, HOSTFILE  (all exported by elmoe.sh.template).

JOB_DIR="${1:?usage: monitor_run.sh <JOB_DIR>}"
FAST_INTERVAL="${2:-60}"
SLOW_INTERVAL="${3:-180}"
FAST_WINDOW="${4:-300}"
TRACE_FRAMES="${5:-20}"   # stack frames kept per rank in TRACE (deepest first)

JOB_ID="${JOB_ID:-0}"
NODES="${NODES:-1}"
HOSTFILE="${HOSTFILE:-$PWD/hostfile}"

MONDIR="${JOB_DIR}/monitor"
SUMMARY="${JOB_DIR}/a-monitor.txt"   # top-level for quick access (per-node files & tsv stay in monitor/)
TSV="${MONDIR}/monitor.tsv"
SSH="ssh -q -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o LogLevel=ERROR"

mkdir -p "${MONDIR}"

# --- host list: hostfile for multi-node, else just this machine ---
NODES_LIST=()
if [ "${NODES}" -gt 1 ] && [ -f "${HOSTFILE}" ]; then
    while read -r h; do
        [[ -z "$h" || "$h" =~ ^# ]] && continue
        NODES_LIST+=("$h")
    done < "${HOSTFILE}"
else
    NODES_LIST=("$(hostname)")
fi

# --- which of those names mean "this machine"? Those skip ssh entirely. ---
SELF_NAMES=" $(hostname) $(hostname -s 2>/dev/null) localhost 127.0.0.1 $(hostname -I 2>/dev/null) "
is_self() { case "${SELF_NAMES}" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

TMPROOT="/tmp/.monitor_${JOB_ID}_$$"
mkdir -p "${TMPROOT}"
trap 'rm -rf "${TMPROOT}"' EXIT

[ -s "${TSV}" ] || printf 'timestamp\tepoch\tnode\trank\tgpu\tpid\twatts\thbm_pct\tram_gb\ttrace\tcategory\todd\tfrozen_ticks\n' > "${TSV}"

ACTIVE_RE='_exec_forward_pass|_exec_backward_pass|do_bench|dropped_scatter|drop_kernel|all_to_all_single|a2a_single|moe_v2|barrier|get_ltor|broadcast_data'

categorize() {
    case "$1" in
        *p2p.py*) case "$1" in *send*) echo "p2p-send";; *) echo "p2p-recv";; esac ;;
        *moe_v2*|*a2a_single*|*drop_kernel*|*do_bench*)                 echo "moe/expert" ;;
        *barrier*)                                                      echo "barrier" ;;
        *get_ltor*|*get_batch*|*broadcast_data*|*_next_batch*|*load_micro_batch*|*data.py*) echo "data-load" ;;
        *_exec_backward*|*"backward ("*)                                echo "backward" ;;
        *transformer.py*|*flash_attn*|*layernorm*|*attention*)          echo "compute-fwd" ;;
        *synchronize*|*timers.py*)                                      echo "sync/timer" ;;
        *unsamplable*)                                                  echo "unsamplable" ;;
        *)                                                              echo "other" ;;
    esac
}

# --- collector: "rank<TAB>gpu<TAB>pid<TAB>watts<TAB>hbm_pct<TAB>ram_gb<TAB>chain" per GPU rank ---
# (single-quoted payload; ONLY double quotes inside, so it survives both `bash -c` locally
#  and the ssh command line unchanged). TRACE_FRAMES is injected by the caller.
REMOTE='
  : "${TRACE_FRAMES:=20}"
  PYSPY=$(command -v py-spy 2>/dev/null); [ -x "$PYSPY" ] || PYSPY="$HOME/.local/bin/py-spy"
  # Two independent nvidia-smi queries, each guarded by timeout so a wedged driver cannot stall
  # the monitor. NEVER chain with && -- a failed power read must not blank every HBM%.
  NVG=""; NVP=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    NVG=$(timeout 15 nvidia-smi --query-gpu=index,power.draw,memory.total --format=csv,noheader,nounits 2>/dev/null)
    NVP=$(timeout 15 nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null)
  fi
  for p in $(pgrep -u "$(id -u)" -f ELMoE_launch.py); do
    ls -l /proc/$p/fd 2>/dev/null | grep -q /dev/nvidia || continue
    e=$(tr "\0" "\n" < /proc/$p/environ 2>/dev/null)
    rk=$(printf "%s\n" "$e" | sed -n "s/^RANK=//p" | head -1); [ -z "$rk" ] && rk="?"
    lr=$(printf "%s\n" "$e" | sed -n "s/^LOCAL_RANK=//p" | head -1); [ -z "$lr" ] && lr="?"
    mem=$(ps -o rss= -p $p 2>/dev/null | awk "{printf \"%.1f\", \$1/1048576}"); [ -z "$mem" ] && mem="?"
    # WATTS by GPU index = LOCAL_RANK.
    pw=$(printf "%s\n" "$NVG" | awk -F", *" -v i="$lr" "\$1==i {printf \"%.0f\", \$2}"); [ -z "$pw" ] && pw="?"
    # HBM% = THIS process VRAM / that GPU total. Keyed by pid, so it stays correct even if
    # CUDA_VISIBLE_DEVICES remaps devices. Total falls back to GPU 0 (homogeneous node).
    tot=$(printf "%s\n" "$NVG" | awk -F", *" -v i="$lr" "\$1==i {print \$3}")
    [ -z "$tot" ] && tot=$(printf "%s\n" "$NVG" | awk -F", *" "NR==1{print \$3}")
    # MAX, not head -1: --query-compute-apps emits one row per (process, GPU), and every
    # rank holds a small (~400 MiB) NCCL/CUDA-IPC buffer on each PEER GPU besides its own
    # real allocation. head -1 returns whichever GPU is listed first, so every rank but the
    # one on GPU 0 reported that peer buffer (~1%) instead of its true footprint. A rank
    # always allocates far more on its own GPU than on any peer, so the max is correct.
    # NOTE: no apostrophes anywhere in this block -- it lives inside a single-quoted
    # shell string, and a single stray quote silently terminates the whole payload.
    usd=$(printf "%s\n" "$NVP" | awk -F", *" -v q="$p" "\$1==q && \$2+0>m {m=\$2+0} END{ if(m>0) print m }")
    hbm=$(awk -v u="$usd" -v t="$tot" "BEGIN{ if(t+0>0 && u!=\"\") printf \"%d\", (u*100)/t }")
    [ -z "$hbm" ] && hbm="?"
    d=$(timeout 15 "$PYSPY" dump --pid $p --nonblocking 2>/dev/null)
    chain=$(printf "%s" "$d" \
      | grep -oE "[A-Za-z_][A-Za-z0-9_]* \([^()]*\.py:[0-9]+\)|device_synchronize|all_to_all_single|nccl[A-Za-z]+|c10d::[A-Za-z_:]+|do_bench|barrier|synchronize" \
      | sed -E "s#\([^)]*/#(#" \
      | head -${TRACE_FRAMES} | tr "\n" "|" | sed "s/|$//")
    [ -z "$chain" ] && chain="unsamplable"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$rk" "$lr" "$p" "$pw" "$hbm" "$mem" "$chain"
  done
'

declare -A PREV FROZEN
# Liveness + death-cause ledger (persist across intervals). EVER_ALIVE=every rank seen at least once;
# LAST_*/PEAK_*=last and peak HBM%/RAM_GB per rank (so OOM is visible AFTER the process is gone);
# MISSING_STREAK=consecutive absent intervals; DEATH_REPORTED/DMESG_DONE=fire-once guards.
declare -A EVER_ALIVE LAST_HOST LAST_HBM LAST_MEM PEAK_HBM PEAK_MEM MISSING_STREAK DEATH_REPORTED DMESG_DONE NODE_HAD_RANKS NODE_EMPTY_STREAK
# World size: TOTAL_GPUS is authoritative here (the template exports it), but we still keep a
# high-water mark of concurrently-sampled ranks so the liveness line is right even if it is unset.
WORLD_HWM=0
SNT="${TOTAL_GPUS:-${SLURM_NTASKS:-0}}"; case "$SNT" in ''|*[!0-9]*) SNT=0 ;; esac
MON_START=$(date '+%s')
JOB_START=${MON_START}

while true; do
    ts=$(date '+%F %T'); ep=$(date '+%s')

    # ---- 1. fan out: collect every node IN PARALLEL (locally where possible) ----
    i=0; declare -A NODEFILE NODEIDX
    for n in "${NODES_LIST[@]}"; do
        NODEFILE[$n]="${TMPROOT}/node.${i}"; NODEIDX[$n]=$i
        if is_self "$n"; then
            ( TRACE_FRAMES=${TRACE_FRAMES} bash -c "${REMOTE}" > "${NODEFILE[$n]}" 2>/dev/null ) &
        else
            ( ${SSH} "${n}" "TRACE_FRAMES=${TRACE_FRAMES}; ${REMOTE}" > "${NODEFILE[$n]}" 2>/dev/null ) &
        fi
        i=$((i+1))
    done
    wait

    # ---- 2. build per-node tables (-> node{idx}.txt AND a master block), tally aggregate ----
    declare -A CAT THIS_ALIVE
    ALL="${TMPROOT}/allblocks"; : > "${ALL}"
    tot=0; unsamp=0; nfrozen=0; nodd=0; nnodes=0; maxfrz=0; maxfrz_who=""; culprit=""; unsamp_list=""

    for n in "${NODES_LIST[@]}"; do
        idx=${NODEIDX[$n]}; f="${NODEFILE[$n]}"; nf="${MONDIR}/node${idx}.txt"; blk="${TMPROOT}/blk"
        if [ ! -s "$f" ]; then
            NODE_EMPTY_STREAK["$n"]=$(( ${NODE_EMPTY_STREAK[$n]:-0} + 1 ))
            { printf '\n  %s  node%s  %s   (no GPU ranks sampled)\n' "${ts}" "${idx}" "${n}"; } | tee -a "$nf" >> "${ALL}"
            continue
        fi
        nnodes=$((nnodes+1)); NODE_HAD_RANKS["$n"]=1; NODE_EMPTY_STREAK["$n"]=0
        sort -n "$f" -o "$f"
        ngpu=$(grep -c . "$f")
        mode=$(cut -f7 "$f" | sort | uniq -c | sort -rn | head -1 | sed 's/^ *[0-9]* //')

        {
            printf '\n  node%-3s %-16s (%s ranks)   %s\n' "${idx}" "${n}" "${ngpu}" "${ts}"
            printf '     %-14s %-5s %-4s %-9s %-6s %-6s %-8s %s\n' "STATUS" "RANK" "GPU" "PID" "WATTS" "HBM%" "RAM_GB" "TRACE (deepest <- outer)"
            while IFS=$'\t' read -r rk gpu pid pw hbm mem chain; do
                [ -z "$pid" ] && continue
                cat=$(categorize "$chain")
                CAT["$cat"]=$(( ${CAT["$cat"]:-0} + 1 )); tot=$((tot+1))
                [ "$cat" = "unsamplable" ] && { unsamp=$((unsamp+1)); unsamp_list="${unsamp_list} ${n}:r${rk}"; }
                # liveness + resource ledger (brace-group runs in this shell, so these persist across intervals)
                THIS_ALIVE["$rk"]=1; EVER_ALIVE["$rk"]=1; LAST_HOST["$rk"]="$n"
                LAST_HBM["$rk"]="$hbm"; LAST_MEM["$rk"]="$mem"
                case "$hbm" in ''|*[!0-9]*) ;; *) [ "$hbm" -gt "${PEAK_HBM[$rk]:-0}" ] 2>/dev/null && PEAK_HBM["$rk"]="$hbm" ;; esac
                case "$mem" in ''|*[!0-9.]*) ;; *) PEAK_MEM["$rk"]=$(awk -v a="${PEAK_MEM[$rk]:-0}" -v b="$mem" 'BEGIN{print (b+0>a+0)?b:a}') ;; esac
                key="${n}:${pid}"
                if [ "${PREV[$key]}" = "$chain" ]; then FROZEN[$key]=$(( ${FROZEN[$key]:-0} + 1 )); else FROZEN[$key]=0; fi
                PREV[$key]="$chain"; frz=${FROZEN[$key]}
                odd=0; [ "$chain" != "$mode" ] && odd=1
                [ "$odd" = "1" ] && nodd=$((nodd+1))
                [ "$frz" -ge 2 ] && nfrozen=$((nfrozen+1))
                if [ "$frz" -gt "$maxfrz" ]; then maxfrz=$frz; maxfrz_who="${n}:r${rk} (${cat})"; fi
                [ -z "$culprit" ] && printf '%s' "$chain" | grep -qE "$ACTIVE_RE" && culprit="$n"
                flag=""; [ "$odd" = "1" ] && flag="ODD "
                [ "$frz" -ge 2 ] && flag="${flag}FROZENx${frz}"; [ -z "$flag" ] && flag="-"
                pretty=$(printf '%s' "$chain" | sed 's/|/ <- /g')
                printf '     %-14s %-5s %-4s %-9s %-6s %-6s %-8s %s\n' "$flag" "$rk" "$gpu" "$pid" "$pw" "$hbm" "$mem" "$pretty"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                       "$ts" "$ep" "$n" "$rk" "$gpu" "$pid" "$pw" "$hbm" "$mem" "$chain" "$cat" "$odd" "$frz" >> "${TSV}"
            done < "$f"
        } > "$blk"
        cat "$blk" >> "$nf"
        cat "$blk" >> "${ALL}"
    done

    # ---- 2b. liveness ledger: compare live set vs world; confirm death over 2 intervals, capture stats ----
    [ "${tot}" -gt "${WORLD_HWM}" ] && WORLD_HWM=${tot}
    WORLD=${WORLD_HWM}; [ "${SNT}" -gt "${WORLD}" ] && WORLD=${SNT}
    GONE=""; SUSPECT=""; NEWDEATHS=""; DEATH_HOSTS_NEW=""
    for rk in $(printf '%s\n' "${!EVER_ALIVE[@]}" | sort -n); do
        if [ -n "${THIS_ALIVE[$rk]}" ]; then MISSING_STREAK["$rk"]=0; continue; fi
        st=$(( ${MISSING_STREAK[$rk]:-0} + 1 )); MISSING_STREAK["$rk"]=$st
        h="${LAST_HOST[$rk]}"
        if [ "$st" -ge 2 ]; then
            GONE="${GONE} ${h}:r${rk}"
            if [ -z "${DEATH_REPORTED[$rk]}" ]; then
                DEATH_REPORTED["$rk"]=1
                NEWDEATHS="${NEWDEATHS}\n##    r${rk} on ${h} GONE — last HBM% ${LAST_HBM[$rk]:-?} (peak ${PEAK_HBM[$rk]:-?}) | RAM ${LAST_MEM[$rk]:-?}G (peak ${PEAK_MEM[$rk]:-?}G)"
                DEATH_HOSTS_NEW="${DEATH_HOSTS_NEW} ${h}"
            fi
        else
            SUSPECT="${SUSPECT} ${h}:r${rk}"   # absent 1 interval — could be a transient blip
        fi
    done
    DEAD_HOST=""
    for n in "${NODES_LIST[@]}"; do
        [ -n "${NODE_HAD_RANKS[$n]}" ] && [ "${NODE_EMPTY_STREAK[$n]:-0}" -ge 2 ] && DEAD_HOST="${DEAD_HOST} ${n}"
    done

    # ---- 3. centralized a-monitor.txt: interval banner + HISTOGRAM + all node tables + culprit ----
    {
        elm=$(( (ep - JOB_START) / 60 )); elh=$(( elm / 60 )); elmm=$(( elm % 60 ))
        printf '\n\n############################################################################\n'
        printf '##  %s    job %s    +%dm (%dh%02dm) since monitor start\n' "${ts}" "${JOB_ID}" "${elm}" "${elh}" "${elmm}"
        printf '##  %s ranks across %s nodes   (%s unsamplable)\n' "${tot}" "${nnodes}" "${unsamp}"
        printf '##  LIVE %s/%s ranks' "${tot}" "${WORLD}"
        [ -n "${GONE}" ]      && printf '   GONE:%s' "${GONE}"
        [ -n "${SUSPECT}" ]   && printf '   suspect:%s' "${SUSPECT}"
        [ -n "${DEAD_HOST}" ] && printf '   DEAD-HOST:%s' "${DEAD_HOST}"
        printf '\n'
        [ -n "${NEWDEATHS}" ] && printf '%b\n' "${NEWDEATHS}"
        if [ "${tot}" -gt 0 ]; then
            printf '##  FROZENx>=2: %s/%s (%s%%)    ODD: %s    longest-frozen: FROZENx%s %s\n' \
                   "${nfrozen}" "${tot}" "$(( nfrozen * 100 / tot ))" "${nodd}" "${maxfrz}" "${maxfrz_who:-none}"
        fi
        printf '##  STATE HISTOGRAM (ranks by phase):\n'
        for k in "${!CAT[@]}"; do printf '%8d  %s\n' "${CAT[$k]}" "$k"; done | sort -rn | sed 's/^/##     /'
        [ -n "${unsamp_list}" ] && printf '##  unsamplable:%s\n' "${unsamp_list}"
        printf '############################################################################\n'
        cat "${ALL}"
    } >> "${SUMMARY}"

    if [ -n "${culprit}" ]; then
        printf '\n  ---- full native stack: %s ----\n\n' "${culprit}" >> "${SUMMARY}"
        NATIVE='p=$(pgrep -u "$(id -u)" -f ELMoE_launch.py | while read pid; do
                  PYSPY=$(command -v py-spy 2>/dev/null); [ -x "$PYSPY" ] || PYSPY="$HOME/.local/bin/py-spy"
                  "$PYSPY" dump --pid $pid 2>/dev/null | grep -q train_batch && { echo $pid; break; }
                done)
                PYSPY=$(command -v py-spy 2>/dev/null); [ -x "$PYSPY" ] || PYSPY="$HOME/.local/bin/py-spy"
                timeout 20 "$PYSPY" dump --pid $p --native 2>/dev/null'
        if is_self "${culprit}"; then out=$(bash -c "${NATIVE}" 2>/dev/null); else out=$(${SSH} "${culprit}" "${NATIVE}" 2>/dev/null); fi
        printf '%s\n' "$out" \
            | grep -E "Thread|_exec_|moe_v2|a2a_single|drop_|kernels\.py|engine\.py|transformer\.py|training\.py|do_bench|autotuner|synchronize|barrier|all_to_all|recv|send|\.py:[0-9]" \
            | cut -c1-200 | sed 's/^/      /' >> "${SUMMARY}"
    fi

    # ---- death-cause: one-shot dmesg scrape on each node that just lost a rank (kernel OOM-killer and
    #      NVRM/Xid faults log to dmesg, NOT to the rank log — this is where silent deaths leave a trace) ----
    DMESG_CMD='dmesg 2>/dev/null | grep -iE "out of memory|oom-kill|oom_reap|killed process|Xid|NVRM|nvidia|gpu has fallen off|hung task" | tail -n 25'
    for h in $(printf '%s\n' ${DEATH_HOSTS_NEW} | sort -u); do
        [ -z "$h" ] && continue
        [ -n "${DMESG_DONE[$h]}" ] && continue
        DMESG_DONE["$h"]=1
        if is_self "$h"; then dm=$(bash -c "${DMESG_CMD}" 2>/dev/null); else dm=$(${SSH} "$h" "${DMESG_CMD}" 2>/dev/null); fi
        {
            printf '\n  ---- death-cause dmesg scrape: %s ----\n' "$h"
            if [ -n "$dm" ]; then printf '%s\n' "$dm" | sed 's/^/      /'
            else printf '      (no oom/Xid lines — dmesg may be root-restricted here, or it was a clean/uncaught-signal exit; check that rank log)\n'
            fi
        } >> "${SUMMARY}"
    done

    unset NODEFILE NODEIDX CAT THIS_ALIVE
    if [ $(( $(date '+%s') - MON_START )) -lt "${FAST_WINDOW}" ]; then sleep "${FAST_INTERVAL}"; else sleep "${SLOW_INTERVAL}"; fi
done
