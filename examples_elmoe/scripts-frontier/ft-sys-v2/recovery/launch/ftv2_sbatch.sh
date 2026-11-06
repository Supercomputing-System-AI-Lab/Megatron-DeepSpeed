#!/bin/bash
# ftv2_sbatch.sh <rendered slurm script> (ft-sys-v2 recovery, 2026-11-06): the 0623 launcher's AUTORUN_SBATCH_CMD
# (local_autorun_frontier_0623_2026.sh:1026, 1121). It submits a copy of the rendered script with FTV2_STANDBY_NODES (default 1)
# more nodes and #SBATCH --no-kill (a lost node does not end the job); the template's NODES stays the training node count, and
# launch/ftv2_srun.sh (sourced through AUTORUN_PRE_CMD) starts the standby on the extra node(s). Nothing else in the script changes.
set -euo pipefail
f=$1
extra=${FTV2_STANDBY_NODES:-1}
out="${f%.slurm}_ftv2.slurm"
n=$(grep -m1 '^#SBATCH -N ' "$f" | awk '{print $3}')
[ -n "$n" ] || { echo "FATAL [ftv2_sbatch]: no '#SBATCH -N' line in $f"; exit 1; }
awk -v n="$n" -v m="$((n + extra))" '
  $0 == "#SBATCH -N " n && !done { print "#SBATCH -N " m; print "#SBATCH --no-kill   # [ftv2_sbatch] a lost node does not end the job"; done = 1; next }
  { print }' "$f" > "$out"
grep -q "^#SBATCH -N $((n + extra))\$" "$out" && grep -q '^#SBATCH --no-kill' "$out" || { echo "FATAL [ftv2_sbatch]: the node line did not change in $out"; exit 1; }
# the template's geometry guard compares SLURM_NNODES with NODES (the training nodes); the allocation now has the standby too
g='[ "${SLURM_NNODES}" -ne "${NODES}" ]'
grep -qF "$g" "$out" || { echo "FATAL [ftv2_sbatch]: the geometry guard was not found in $out"; exit 1; }
python3 - "$out" "$g" "[ \"\${SLURM_NNODES}\" -ne \"\$(( NODES + $extra ))\" ]" <<'PY'
import sys
f, a, b = sys.argv[1:4]
s = open(f).read()
assert s.count(a) == 1, 'guard not unique'
open(f, 'w').write(s.replace(a, b))
PY
grep -qF "\$(( NODES + $extra ))" "$out" || { echo "FATAL [ftv2_sbatch]: the geometry guard did not change in $out"; exit 1; }
echo "[ftv2_sbatch] $out: nodes $n -> $((n + extra)) (standby $extra), --no-kill"
sbatch "$out"
