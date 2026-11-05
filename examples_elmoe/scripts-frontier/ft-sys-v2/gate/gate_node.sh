#!/bin/bash
# ft-sys-v2 copy (2026-11-05) of examples_elmoe/scripts-frontier/resilience/bin/gate_node.sh (ft-sys-v1, commit 7fb97edf). Changed: only the software-stack paths, now settings with the new stack (ROCm 7.1.1, torch 2.13 venv, aws-ofi-nccl 1.22.0) as the default (GATE_PYTHON, GATE_OFI_DIR, ROCM_PATH); the tests, the rules and the judge are v1's.
# gate_node.sh <quick|full> <outdir>: the per-node part of the pre-run gate (design G4/G5/G6, rules of 3.11).
# WHAT THIS SCRIPT IS: one srun task per node runs it (gate.sh launches the step; the restart loop also runs it alone on a spare
# before a swap: the 'spare check', mode quick). It MEASURES and never judges: one raw row per test,
#     time  node  test  value  run_s  status  detail
# status = ok | timeout (the command ran past its budget, or a GCD did not answer by the deadline) | error | missing (the tool is
# absent: "no output"); gate_judge.py applies the baselines and the rules. The exact command of every test is also listed in the
# TESTS table of gate_judge.py (printed in gate.log under each test), so keep the two in step when a command changes.
# Raw outputs go to <outdir>/<node>_{ras,xgmi,dmesg,gemm,a2a}.txt. Every command runs under its budget GATE_BUDGET_<test>
# (seconds; gate.sh derives it from the calibrated run time); safe on a login node without GPUs.
MODE=${1:-quick}; OUT=${2:-.}
mkdir -p "$OUT" || exit 0
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
N=$(hostname -s 2>/dev/null || hostname)
TSV=$OUT/$N.raw.tsv; : > "$TSV"
PY=${GATE_PYTHON:-/lustre/orion/world-shared/gen150/zixianw4/envs/TORCH2.13-ROCM7.14-XMOE_venv/bin/python}
A2A_BIN=${GATE_A2A_BIN:-/lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/assistant_deep_dive/2026-10-09-collective_stack/tools/rccl-tests/build/alltoall_perf}
RCCL_SHIM=/lustre/orion/gen150/scratch/zixianw4/X-MoE/Megatron-DeepSpeed-X-MoE/examples_elmoe/assistant_deep_dive/2026-10-09-collective_stack/tools/rccl_torch_lib
OFI_DIR=${GATE_OFI_DIR:-/lustre/orion/gen150/scratch/zixianw4/aws-ofi-nccl-v1.22.0/install/lib}
ts() { date '+%F %H:%M:%S'; }
now() { date +%s.%N; }
el() { awk -v a="$1" -v b="$(now)" 'BEGIN{printf "%.1f", b-a}'; }
budget() { local v="GATE_BUDGET_$1"; echo "${!v:-$2}"; }
row() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(ts)" "$N" "$1" "$2" "$3" "$4" "${5:--}" >> "$TSV"; }   # test value run_s status detail
st_of() { [ "$1" -ge 124 ] && [ "$1" -le 137 ] && echo timeout || { [ "$1" = 0 ] && echo ok || echo error; }; }
counters() {  # ue=<sum>,ce=<sum>,xgmi=<sum>; -1 where no file exists
  local ue=-1 ce=-1 xg=-1 f v
  for f in /sys/class/drm/card*/device/ras/umc_err_count; do
    [ -r "$f" ] || continue
    v=$(timeout 5 awk '$1=="ue:"{u+=$2} $1=="ce:"{c+=$2} END{printf "%d %d", u, c}' "$f" 2>/dev/null) || continue
    [ $ue -lt 0 ] && ue=0 && ce=0
    ue=$((ue + ${v% *})); ce=$((ce + ${v#* }))
  done
  for f in /sys/class/drm/card*/device/xgmi_error; do
    [ -r "$f" ] || continue
    v=$(timeout 5 cat "$f" 2>/dev/null | grep -o -E '[0-9]+' | head -1)
    [ -n "$v" ] || continue
    [ $xg -lt 0 ] && xg=0; xg=$((xg + v))
  done
  echo "ue=$ue,ce=$ce,xgmi=$xg"
}
cval() { echo "$1" | tr ',' '\n' | awk -F= -v k="$2" '$1==k{print $2}'; }
t=$(now); CB=$(counters); row counters_before "$CB" "$(el $t)" ok

# ---- rocm-smi: the GCDs that answer, the RAS and xGMI error counts
SMI=$(command -v rocm-smi 2>/dev/null)
for c in "$SMI" "${ROCM_PATH:+$ROCM_PATH/bin/rocm-smi}" /opt/rocm/bin/rocm-smi /opt/rocm-7.1.1/bin/rocm-smi /opt/rocm-6.4.1/bin/rocm-smi; do [ -n "$c" ] && [ -x "$c" ] && { SMI=$c; break; }; done
[ -x "$SMI" ] || SMI=""
if [ -z "$SMI" ]; then
  row rocm_smi_ids - 0 missing "rocm-smi not found"; row ras_ue - 0 missing "rocm-smi not found"; row xgmi_err - 0 missing "rocm-smi not found"
else
  t=$(now); out=$(timeout -k 5 "$(budget rocm_smi_ids 30)" "$SMI" --showid 2>&1); rc=$?
  n=$(echo "$out" | grep -o -E '^GPU\[[0-9]+\]' | sort -u | wc -l)
  row rocm_smi_ids "$n" "$(el $t)" "$(st_of $rc)" "$( [ $rc != 0 ] && echo "rc $rc: $(echo "$out" | tail -1 | head -c 120)")"
  t=$(now); timeout -k 5 "$(budget ras_ue 30)" "$SMI" --showrasinfo all > "$OUT/${N}_ras.txt" 2>&1; rc=$?
  ue=$(awk 'NF>=4 && $(NF-1)~/^[0-9]+$/ && $NF~/^[0-9]+$/ {s+=$NF; k++} END{if(k)print s}' "$OUT/${N}_ras.txt")
  row ras_ue "${ue:-unparsed}" "$(el $t)" "$(st_of $rc)" "$( [ -z "$ue" ] && echo "no counter table in ${N}_ras.txt")"
  t=$(now); timeout -k 5 "$(budget xgmi_err 30)" "$SMI" --showxgmierr > "$OUT/${N}_xgmi.txt" 2>&1; rc=$?
  xv=$(grep -a -o -E '^GPU\[[0-9]+\].*XGMI Error count: *[0-9]+' "$OUT/${N}_xgmi.txt" | awk '{k++; if($NF+0>0)nz++} END{if(k)print nz+0}')
  row xgmi_err "${xv:-unparsed}" "$(el $t)" "$(st_of $rc)" "$( [ -z "$xv" ] && echo "no per-GCD counts in ${N}_xgmi.txt")"
fi

# ---- versions: the values; the judge compares them with the calibrated baseline
V=missing; [ -n "$SMI" ] && [ -z "$ROCM_PATH" ] && ROCM_PATH=$(dirname "$(dirname "$SMI")")
t=$(now); for f in "${ROCM_PATH:+$ROCM_PATH/.info/version}" /opt/rocm/.info/version; do [ -n "$f" ] && [ -r "$f" ] && { V=$(head -1 "$f"); break; }; done
row rocm_version "$V" "$(el $t)" "$( [ "$V" = missing ] && echo missing || echo ok)"
fp() {  # cheap fingerprint: md5 of the first 16 MiB + size (librccl.so is 2.1 GB; a full md5 on every node is a Lustre burst)
  [ -r "$1" ] || { echo missing; return; }
  local m s; m=$(timeout 30 head -c 16777216 "$1" | md5sum | cut -c1-12); s=$(stat -Lc %s "$1" 2>/dev/null)
  echo "${m}:${s}"
}
PYENV=$(dirname "$(dirname "$PY")")
t=$(now); f=$(ls -1 $PYENV/lib/python3*/site-packages/torch/lib/librccl.so* 2>/dev/null | head -1); v=$(fp "$f")
row rccl_md5 "$v" "$(el $t)" "$( [ "$v" = missing ] && echo missing || echo ok)" "$f"
t=$(now); f=$(ls -1 $OFI_DIR/librccl-net*.so 2>/dev/null | head -1); v=$(fp "$f")
row ofi_plugin_md5 "$v" "$(el $t)" "$( [ "$v" = missing ] && echo missing || echo ok)" "$f"
t=$(now); L=$(timeout 10 fi_info --version 2>/dev/null | head -1)
if [ -z "$L" ]; then
  for d in ${LD_LIBRARY_PATH//:/ }; do [ -e "$d/libfabric.so.1" ] && { L=$(realpath "$d/libfabric.so.1"); break; }; done
  [ -z "$L" ] && L=$(ls -1 /opt/cray/libfabric/*/lib64/libfabric.so.1 2>/dev/null | tail -1) && [ -n "$L" ] && L=$(realpath "$L")
fi
row libfabric "${L:-missing}" "$(el $t)" "$( [ -z "$L" ] && echo missing || echo ok)"
# the GPU-direct environment (informational: RCCL decides at its first communicator; the training.log header reads its NET lines)
row gpu_direct_env "NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-unset},NCCL_DMABUF_ENABLE=${NCCL_DMABUF_ENABLE:-unset},FI_HMEM=${FI_HMEM:-unset},plugin_on_LD_LIBRARY_PATH=$(echo ":$LD_LIBRARY_PATH:" | grep -q ":$OFI_DIR:" && echo yes || echo no)" 0 ok

# ---- dmesg (informational: not permitted for users on Frontier)
t=$(now); out=$(timeout 20 dmesg -T 2>&1); rc=$?
if echo "$out" | head -3 | grep -q -i "not permitted"; then
  echo "$out" | head -3 > "$OUT/${N}_dmesg.txt"; row dmesg "not permitted" "$(el $t)" ok
else
  echo "$out" | grep -i -E "mce|ecc|xgmi|hardware error|kfd|amdgpu.*(fail|error)" | tail -50 > "$OUT/${N}_dmesg.txt"
  row dmesg "$(wc -l < "$OUT/${N}_dmesg.txt")" "$(el $t)" "$(st_of $rc)"
fi

# ---- /dev/shm
t=$(now); k=$(timeout 10 df -Pk /dev/shm 2>/dev/null | awk 'NR==2{print $4}')
if [ -n "$k" ]; then row shm_free_gb "$(awk -v k="$k" 'BEGIN{printf "%.1f", k/1048576}')" "$(el $t)" ok
else row shm_free_gb - "$(el $t)" error "df /dev/shm gave nothing"; fi

if [ "$MODE" == full ]; then
  # ---- GEMM per GCD: one process per GCD, forked from one parent that imported torch once (gate_gemm.py)
  B=$(budget gemm 300)
  t=$(now); GATE_GEMM_DEADLINE_S=$((B - 5)) timeout -k 5 "$B" "$PY" "$HERE/gate_gemm.py" > "$OUT/${N}_gemm.txt" 2>&1; rc=$?
  vals=$(awk -F'\t' '$1~/^gcd[0-7]$/ && $2~/^[0-9.]+$/ {printf "%s%s=%s", (n++?",":""), $1, $2}' "$OUT/${N}_gemm.txt")
  ng=$(echo "$vals" | tr ',' '\n' | grep -c '=')
  mn=$(echo "$vals" | tr ',' '\n' | awk -F= 'NF==2{if(m==""||$2<m)m=$2} END{print m}')
  imp=$(awk -F'\t' '$1=="import_s"{print $2}' "$OUT/${N}_gemm.txt")
  # [2026-10-09 late evening] a GCD whose child was still alive at gate_gemm.py's deadline prints 'gcdK timeout': the row is then a
  # timeout (the judge's majority rule can act on it), not an error; 'error' stays for a child that died or a non-zero exit
  tos=$(awk -F'\t' '$1~/^gcd[0-7]$/ && $2=="timeout" {printf "%s%s", (n++?",":""), $1}' "$OUT/${N}_gemm.txt")
  if [ "$ng" = 8 ]; then row gemm "$mn" "$(el $t)" ok "min of 8 GCDs; $vals; import ${imp:-?} s"
  elif [ "$(st_of $rc)" = timeout ]; then row gemm "${mn:--}" "$(el $t)" timeout "$ng of 8 GCDs reported before the budget of $B s; ${vals:-none}; import ${imp:-?} s"
  elif [ -n "$tos" ]; then row gemm "${mn:--}" "$(el $t)" timeout "$ng of 8 GCDs reported before the deadline of $((B - 5)) s; the GCDs that did not answer: $tos; ${vals:-none}; import ${imp:-?} s"
  else row gemm "${mn:--}" "$(el $t)" error "$ng of 8 GCDs reported (rc $rc): $(grep -a -m1 -E 'error|Error' "$OUT/${N}_gemm.txt" | head -c 160)"; fi
  # ---- in-node all-to-all over xGMI: rccl-tests, one process driving 8 GCDs (-g 8), MPI singleton (Slurm/PMI env stripped)
  if [ -x "$A2A_BIN" ]; then
    UNS=(); for v in $(compgen -e); do case $v in PMI_*|PMIX_*|SLURM_*|CRAY_PMI*|PALS_*) UNS+=(-u "$v");; esac; done
    R=${ROCM_PATH:-/opt/rocm-7.1.1}
    t=$(now)
    env "${UNS[@]}" LD_LIBRARY_PATH="$R/lib:$RCCL_SHIM:$OFI_DIR:/opt/cray/pe/lib64:${CRAY_LD_LIBRARY_PATH:+$CRAY_LD_LIBRARY_PATH:}${LD_LIBRARY_PATH:-}" \
      NCCL_DEBUG=WARN timeout -k 5 "$(budget xgmi_a2a 120)" "$A2A_BIN" -b 256M -e 256M -f 2 -g 8 -w 3 -n 10 -c 0 > "$OUT/${N}_a2a.txt" 2>&1; rc=$?
    v=$(awk '$1~/^[0-9]+$/ && NF>=8 {b=$8} END{print b}' "$OUT/${N}_a2a.txt")
    if [ $rc = 0 ] && [[ "$v" =~ ^[0-9]+([.][0-9]+)?$ ]]; then row xgmi_a2a "$v" "$(el $t)" ok "busbw GB/s at 256 MB, 8 GCDs"
    else row xgmi_a2a "${v:--}" "$(el $t)" "$(st_of $rc)" "alltoall_perf rc $rc: $(grep -a -m1 -i -E 'error|fail' "$OUT/${N}_a2a.txt" | head -c 160)"; fi
  else
    row xgmi_a2a - 0 missing "alltoall_perf not found at $A2A_BIN"
  fi
fi

# ---- counters after: ue and xgmi must not grow during the gate
t=$(now); CA=$(counters)
if [ "$(cval "$CA" ue)" == -1 ] && [ "$(cval "$CA" xgmi)" == -1 ]; then row counters_after unreadable "$(el $t)" ok "$CB -> $CA"
elif [ "$(cval "$CA" ue)" -gt "$(cval "$CB" ue)" ] || [ "$(cval "$CA" xgmi)" -gt "$(cval "$CB" xgmi)" ]; then row counters_after grew "$(el $t)" ok "$CB -> $CA"
else row counters_after same "$(el $t)" ok "$CB -> $CA"; fi
row node_done yes 0 ok
exit 0
