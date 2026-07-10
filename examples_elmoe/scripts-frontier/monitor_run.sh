#!/bin/bash
# monitor_run.sh — async, per-rank pipeline-stall monitor for the ELMoE SLURM runs.
#
# Usage:  bash monitor_run.sh <JOB_DIR> [FAST_INTERVAL] [SLOW_INTERVAL] [FAST_WINDOW] [TRACE_FRAMES]
# Run BACKGROUNDED from the slurm template so it never blocks training:
#     bash "${DIR}/monitor_run.sh" "${JOB_DIR}" 60 180 300 &
#
# Each interval it snapshots EVERY GPU rank on EVERY node with py-spy, IN PARALLEL across nodes.
# Output:  a-monitor.txt at <JOB_DIR>/ (top level, quick access); everything else under <JOB_DIR>/monitor/ :
#   a-monitor.txt   — CENTRALIZED quick-glance file: per interval = a state HISTOGRAM across all
#                     ranks, a LIVE x/world liveness line (GONE/suspect/DEAD-HOST + last/peak HBM%
#                     & RAM_GB of any rank that died), the full per-node tables, then the culprit
#                     stack, then a one-shot dmesg scrape on any node that just lost a rank.
#   node{index}.txt — same per-node table but just that node, for drilling in (index = node order).
#   monitor.tsv     — machine view, one row per (interval,rank).
#
# Per-rank columns: STATUS RANK GPU PID WATTS HBM% RAM_GB TRACE(deepest<-outer).
#   WATTS   = GPU pkg power (rocm-smi; "N/A" on MI250X secondary dies — power is per-card)
#   HBM%    = VRAM used on this rank's GCD, percent (rocm-smi --showmeminfo vram used/total, by GPU index = local rank)
#   RAM_GB  = host resident RAM of this rank's process in GiB (ps rss) — flat RAM_GB + low WATTS + FROZEN = hung
# TRACE depth is the 5th arg (TRACE_FRAMES, default 14 frames, deepest first).
#
# Identity: rank=SLURM_PROCID, gpu=SLURM_LOCALID (per-task, /proc/<pid>/environ). Real ranks =
# python procs holding a GPU (/dev/kfd open). frozen_ticks = consecutive unchanged-trace intervals.
# Cadence: FAST_INTERVAL for the first FAST_WINDOW sec, then SLOW_INTERVAL.
#
# Requires: ~/.local/bin/py-spy on $HOME (shared across nodes); SLURM_NODELIST + SLURM_JOB_ID.

JOB_DIR="${1:?usage: monitor_run.sh <JOB_DIR>}"
FAST_INTERVAL="${2:-60}"
SLOW_INTERVAL="${3:-180}"
FAST_WINDOW="${4:-300}"
TRACE_FRAMES="${5:-20}"   # stack frames kept per rank in TRACE (deepest first)

MONDIR="${JOB_DIR}/monitor"
SUMMARY="${JOB_DIR}/a-monitor.txt"   # top-level for quick access (per-node files & tsv stay in monitor/)
TSV="${MONDIR}/monitor.tsv"
SSH="ssh -q -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o LogLevel=ERROR"

mkdir -p "${MONDIR}"
NODES=$(scontrol show hostnames "${SLURM_NODELIST}")
TMPROOT="/tmp/.monitor_${SLURM_JOB_ID}_$$"
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

# --- remote collector: "rank<TAB>gpu<TAB>pid<TAB>watts<TAB>hbm_pct<TAB>ram_gb<TAB>chain" per GPU rank ---
# (single-quoted ssh payload; ONLY double quotes inside). rocm-smi is best-effort (-> "?" if absent).
# TRACE_FRAMES is injected by the ssh caller (prepended assignment); default kept here for safety.
REMOTE='
  : "${TRACE_FRAMES:=14}"
  # Pick a rocm-smi that MATCHES the node driver: the one on PATH (loaded module) if any, else the
  # site default symlink, else the NEWEST installed (sort -V). NOT "head -1", which picks the oldest
  # (rocm-5.6.0) lexicographically and mis-reads MI250X GCDs under a newer driver -> "?" everywhere.
  SMI=$(command -v rocm-smi 2>/dev/null)
  [ -x "$SMI" ] || SMI=/opt/rocm-default/bin/rocm-smi
  [ -x "$SMI" ] || SMI=$(ls /opt/rocm-*/bin/rocm-smi 2>/dev/null | sort -V | tail -1)
  # Query power and VRAM INDEPENDENTLY (never chain with && -- a non-zero --showpower must not skip
  # the VRAM read and blank every HBM%). Each guarded by timeout so a wedged rocm-smi cannot stall.
  RP=""; RV=""
  if [ -x "$SMI" ]; then
    RP=$(timeout 15 "$SMI" --showpower 2>/dev/null)
    RV=$(timeout 15 "$SMI" --showmeminfo vram 2>/dev/null)
  fi
  for p in $(pgrep -u $USER -f ELMoE_launch.py); do
    ls -l /proc/$p/fd 2>/dev/null | grep -q /dev/kfd || continue
    e=$(tr "\0" "\n" < /proc/$p/environ 2>/dev/null)
    rk=$(printf "%s\n" "$e" | sed -n "s/^SLURM_PROCID=//p" | head -1); [ -z "$rk" ] && rk=$(printf "%s\n" "$e" | sed -n "s/^RANK=//p" | head -1); [ -z "$rk" ] && rk="?"
    lr=$(printf "%s\n" "$e" | sed -n "s/^SLURM_LOCALID=//p" | head -1); [ -z "$lr" ] && lr=$(printf "%s\n" "$e" | sed -n "s/^LOCAL_RANK=//p" | head -1); [ -z "$lr" ] && lr="?"
    mem=$(ps -o rss= -p $p 2>/dev/null | awk "{printf \"%.1f\", \$1/1048576}"); [ -z "$mem" ] && mem="?"
    # HBM% from --showmeminfo vram: index-keyed by GPU[<localrank>] and version-stable (bytes, not a
    # positional concise column that shifts between rocm-smi builds). hbm = used/total*100, matched by GPU idx.
    tot=$(printf "%s\n" "$RV" | grep -E "GPU\[$lr\]" | grep -i "Total Memory" | grep -oE "[0-9]+" | tail -1)
    usd=$(printf "%s\n" "$RV" | grep -E "GPU\[$lr\]" | grep -i "Used Memory"  | grep -oE "[0-9]+" | tail -1)
    hbm=$(awk -v u="$usd" -v t="$tot" "BEGIN{ if(t+0>0) printf \"%d\", (u*100)/t }")
    [ -z "$hbm" ] && hbm="?"
    pw=$(printf "%s\n" "$RP" | grep -E "GPU\[$lr\]" | sed -n "s/.*(W)://p" | grep -oE "[0-9.]+|N/A" | head -1); [ -z "$pw" ] && pw="?"
    d=$(timeout 15 ~/.local/bin/py-spy dump --pid $p --nonblocking 2>/dev/null)
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
# World size: SLURM_NTASKS in THIS (batch-script) context is the allocation's task count (often
# 1/node), NOT the inner training srun's rank count — so we self-calibrate to a high-water mark of
# ranks ever concurrently sampled, and only fall back to SLURM_NTASKS if it's somehow larger.
WORLD_HWM=0
SNT="${SLURM_NTASKS:-0}"; case "$SNT" in ''|*[!0-9]*) SNT=0 ;; esac
MON_START=$(date '+%s')
# SLURM job start (epoch), captured once. Falls back to monitor-start if scontrol can't be parsed.
JS=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | grep -oE 'StartTime=[0-9T:-]+' | head -1 | cut -d= -f2)
JOB_START=$(date -d "${JS}" '+%s' 2>/dev/null); [ -z "${JOB_START}" ] && JOB_START=${MON_START}

while true; do
    ts=$(date '+%F %T'); ep=$(date '+%s')

    # ---- 1. fan out: collect every node IN PARALLEL ----
    i=0; declare -A NODEFILE NODEIDX
    for n in ${NODES}; do
        NODEFILE[$n]="${TMPROOT}/node.${i}"; NODEIDX[$n]=$i
        ( ${SSH} "${n}" "TRACE_FRAMES=${TRACE_FRAMES}; ${REMOTE}" > "${NODEFILE[$n]}" 2>/dev/null ) &
        i=$((i+1))
    done
    wait

    # ---- 2. build per-node tables (-> node{idx}.txt AND a master block), tally aggregate ----
    declare -A CAT THIS_ALIVE
    ALL="${TMPROOT}/allblocks"; : > "${ALL}"
    tot=0; unsamp=0; nfrozen=0; nodd=0; nnodes=0; maxfrz=0; maxfrz_who=""; culprit=""; unsamp_list=""

    for n in ${NODES}; do
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
    [ "${tot}" -gt "${WORLD_HWM}" ] && WORLD_HWM=${tot}      # self-calibrating world size (peak concurrent ranks)
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
            SUSPECT="${SUSPECT} ${h}:r${rk}"   # absent 1 interval — could be a transient ssh blip
        fi
    done
    DEAD_HOST=""
    for n in ${NODES}; do
        [ -n "${NODE_HAD_RANKS[$n]}" ] && [ "${NODE_EMPTY_STREAK[$n]:-0}" -ge 2 ] && DEAD_HOST="${DEAD_HOST} ${n}"
    done

    # ---- 3. centralized a-monitor.txt: interval banner + HISTOGRAM + all node tables + culprit ----
    {
        elm=$(( (ep - JOB_START) / 60 )); elh=$(( elm / 60 )); elmm=$(( elm % 60 ))
        printf '\n\n############################################################################\n'
        printf '##  %s    job %s    +%dm (%dh%02dm) since job start\n' "${ts}" "${SLURM_JOB_ID}" "${elm}" "${elh}" "${elmm}"
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
        ${SSH} "${culprit}" 'p=$(pgrep -u $USER -f ELMoE_launch.py | while read pid; do
                  ~/.local/bin/py-spy dump --pid $pid 2>/dev/null | grep -q train_batch && { echo $pid; break; }
                done); timeout 20 ~/.local/bin/py-spy dump --pid $p --native 2>/dev/null' 2>/dev/null \
            | grep -E "Thread|_exec_|moe_v2|a2a_single|drop_|kernels\.py|engine\.py|transformer\.py|training\.py|do_bench|autotuner|synchronize|barrier|all_to_all|recv|send|\.py:[0-9]" \
            | cut -c1-200 | sed 's/^/      /' >> "${SUMMARY}"
    fi

    # ---- death-cause: one-shot dmesg scrape on each node that just lost a rank (kernel OOM-killer and
    #      amdgpu/kfd faults log to dmesg, NOT to the job .e — this is where silent deaths leave a trace) ----
    for h in $(printf '%s\n' ${DEATH_HOSTS_NEW} | sort -u); do
        [ -z "$h" ] && continue
        [ -n "${DMESG_DONE[$h]}" ] && continue
        DMESG_DONE["$h"]=1
        dm=$(${SSH} "$h" 'dmesg 2>/dev/null | grep -iE "out of memory|oom-kill|oom_reap|killed process|amdgpu|gpu reset|ring [a-z0-9]+ timeout|kfd|xid|hung task" | tail -n 25' 2>/dev/null)
        {
            printf '\n  ---- death-cause dmesg scrape: %s ----\n' "$h"
            if [ -n "$dm" ]; then printf '%s\n' "$dm" | sed 's/^/      /'
            else printf '      (no oom/gpu-fault lines — dmesg may be root-restricted here, or it was a clean/uncaught-signal exit; check that rank stderr + the .e log)\n'
            fi
        } >> "${SUMMARY}"
    done

    unset NODEFILE NODEIDX CAT THIS_ALIVE
    if [ $(( $(date '+%s') - MON_START )) -lt "${FAST_WINDOW}" ]; then sleep "${FAST_INTERVAL}"; else sleep "${SLOW_INTERVAL}"; fi
done
