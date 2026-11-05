#!/bin/bash
# test_net_launch_retry.sh (2026-10-11, sandbox): bin/net_launch_retry.sh with a fake launch command on the login node.
#   a. Slurm refuses twice, the third launch runs (a rank writes a file) -> 3 tries, no marker, exit 0
#   b. Slurm refuses three times -> marker "3 <TAB> srun: error: Unable to confirm allocation ...", exit 1
#   c. another failure (the test itself crashed) -> no retry, no marker
#   d. Slurm's words in the log but a rank already wrote a file -> no retry (the ranks ran)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); B=$HERE/../bin; ok=1
T=$(mktemp -d "${TMPD:-/tmp}/netretry.XXXX")
fake() {   # fake <net dir> <counter file> <refusals> <mode>: refuse <refusals> times with Slurm's words, then succeed (or crash: mode crash)
  local net=$1 cnt=$2 nref=$3 mode=$4; local n=$(( $(cat "$cnt" 2>/dev/null || echo 0) + 1 )); echo $n > "$cnt"
  if [ "$mode" = crash ]; then echo "Traceback: gate_net.py crashed"; exit 1; fi
  if [ "$mode" = ranklater ]; then echo x > "$net/net_frontier1_r0.json"; echo "srun: error: Unable to confirm allocation for job 1: Socket timed out on send/recv operation"; exit 1; fi
  if [ $n -le $nref ]; then echo "srun: error: Unable to confirm allocation for job 1: Socket timed out on send/recv operation"; echo "srun: Check SLURM_JOB_ID environment variable. Expired or invalid job 1"; exit 1; fi
  echo x > "$net/net_frontier1_r0.json"; echo "ranks ran"; exit 0
}
export -f fake
run() {   # run <case> <refusals> <mode>
  local net=$T/$1; mkdir -p $net; rm -f $T/$1.cnt
  bash $B/net_launch_retry.sh $net/srun.log $net 3 1 -- bash -c "fake $net $T/$1.cnt $2 $3" > $net/launch_tries.txt; echo $?
}
rc=$(run a 2 ok); tries=$(cat $T/a.cnt)
[ "$rc" = 0 ] && [ "$tries" = 3 ] && [ ! -f $T/a/not_launched ] && grep -q "try 3 of 3: the network step was launched" $T/a/launch_tries.txt && [ -f $T/a/srun_try1.log ] && [ -f $T/a/srun_try2.log ] \
  && echo "PASS a: two refusals, the third launch ran" || { echo "FAIL a: rc $rc tries $tries $(cat $T/a/launch_tries.txt)"; ok=0; }
rc=$(run b 5 ok); tries=$(cat $T/b.cnt)
[ "$rc" = 1 ] && [ "$tries" = 3 ] && grep -q -P "^3\tsrun: error: Unable to confirm allocation" $T/b/not_launched && grep -q "the tests were not launched" $T/b/launch_tries.txt \
  && echo "PASS b: three refusals -> the not-launched marker" || { echo "FAIL b: rc $rc tries $tries marker $(cat $T/b/not_launched 2>/dev/null)"; ok=0; }
rc=$(run c 0 crash); tries=$(cat $T/c.cnt)
[ "$rc" = 1 ] && [ "$tries" = 1 ] && [ ! -f $T/c/not_launched ] && echo "PASS c: another failure is not retried" || { echo "FAIL c: rc $rc tries $tries"; ok=0; }
rc=$(run d 0 ranklater); tries=$(cat $T/d.cnt)
[ "$tries" = 1 ] && [ ! -f $T/d/not_launched ] && echo "PASS d: ranks already wrote output -> not retried" || { echo "FAIL d: tries $tries"; ok=0; }
rm -rf $T
[ $ok = 1 ] && echo "NET LAUNCH RETRY: ALL PASS" || echo "NET LAUNCH RETRY: SOME FAILED"
