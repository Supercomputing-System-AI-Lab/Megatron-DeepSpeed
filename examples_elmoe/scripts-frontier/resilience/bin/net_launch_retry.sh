#!/bin/bash
# net_launch_retry.sh <srun log> <net dir> <tries> <wait s> -- <launch command ...>     (2026-10-11, sandbox; the user approved the fix)
# Runs the pre-run network step's launch. When the launch fails because Slurm's controller did not start the step, and no rank wrote any
# file, it waits and launches again. Why: in job 5650829 (264 nodes) the srun gave up after Slurm's 35-s MessageTimeout with
# "srun: error: Unable to confirm allocation for job 5650829: Socket timed out on send/recv operation"; no rank ran, every node read
# "missing", and the judge called the pair test a TEST DEFECT; the training's srun 23 s later started normally. After the last try it writes
# <net dir>/not_launched (tries <TAB> Slurm's words), which gate_judge.py turns into "NOT LAUNCHED" instead of "TEST DEFECT".
# Prints one line per extra try for gate.sh to log. Exit code: the last launch's.
LOG=$1; NET=$2; TRIES=$3; WAIT=$4; shift 4; [ "${1:-}" = "--" ] && shift
RE='Unable to confirm allocation|Socket timed out on send/recv|Job step creation temporarily disabled|Unable to create step|Expired or invalid job'
rm -f "$NET/not_launched"
rc=0
for try in $(seq 1 "$TRIES"); do
  "$@" > "$LOG" 2>&1; rc=$?
  produced=$(find "$NET" -maxdepth 1 -type f ! -name 'srun*' ! -name 'summarize.out' ! -name 'not_launched' ! -name 'launch_tries.txt' 2>/dev/null | head -1)
  if [ $rc -ne 0 ] && [ -z "$produced" ] && grep -q -a -E "$RE" "$LOG"; then
    msg=$(grep -a -m1 -E "$RE" "$LOG" | tr '\t' ' ' | head -c 220)
    cp "$LOG" "${LOG%.log}_try$try.log"
    if [ "$try" -lt "$TRIES" ]; then
      echo "try $try of $TRIES: Slurm did not start the network step ($msg); launching again in $WAIT s"
      sleep "$WAIT"; continue
    fi
    echo "try $try of $TRIES: Slurm did not start the network step ($msg); the tests were not launched"
    printf '%s\t%s\n' "$TRIES" "$msg" > "$NET/not_launched"
  elif [ "$try" -gt 1 ]; then
    echo "try $try of $TRIES: the network step was launched (exit code $rc)"
  fi
  break
done
exit $rc
