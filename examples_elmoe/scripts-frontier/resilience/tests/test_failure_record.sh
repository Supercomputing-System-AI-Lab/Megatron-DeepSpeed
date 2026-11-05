#!/bin/bash
# test_failure_record.sh: synthetic record -> exact lemon list. Prints PASS/FAIL per assertion; exit 1 on any FAIL.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FR="python3 $HERE/../bin/failure_record.py"
T=${1:-/tmp/claude-20159/-lustre-orion-gen150-scratch-zixianw4/0ae86a26-aee4-4885-ac5f-8b4de977350f/scratchpad/frtest}
rm -rf "$T"; mkdir -p "$T"
A=$T/a/failures.tsv; B=$T/b/failures.tsv
NOW='2026-10-09 12:00:00'
NF=0
check() {  # name expected actual
  if [ "$2" == "$3" ]; then echo "PASS $1"; else echo "FAIL $1"; echo "  expected: [$2]"; echo "  actual:   [$3]"; NF=$((NF+1)); fi
}
ap() { $FR append --record "$A" "$@" ; }
# 1 hard node_lost on n1
out=$(ap --job 101 --cls node_lost --code 137 --node frontier00001 --rank 8 --step 40 --injected no --evidence $'srun: node\tfrontier00001 not responding\nline2' --time '2026-10-05 10:00:00')
check "append prints prefixed line" "[failure_record] 101	2026-10-05 10:00:00	node_lost	137	frontier00001	8	40	no	srun: node frontier00001 not responding line2" "$out"
# 2 soft collective_hang on n2 within 30 days; the second written to BOTH records (duplicate must count once)
ap --job 102 --cls collective_hang --code 1 --node frontier00002 --rank 3 --step 50 --injected no --evidence 'watchdog 600 s' --time '2026-10-01 08:00:00' >/dev/null
$FR append --record "$A" --record "$B" --job 103 --cls collective_hang --code 1 --node frontier00002 --rank 3 --step 55 --injected no --evidence 'watchdog' --time '2026-10-08 08:00:00' >/dev/null
# n3: one soft line older than 30 days + one soft line inside, the inside one duplicated in both files -> still 1 in window
ap --job 90 --cls collective_hang --code 1 --node frontier00003 --rank 0 --step 9 --injected no --evidence old --time '2026-08-01 00:00:00' >/dev/null
$FR append --record "$A" --record "$B" --job 104 --cls slow_rank --code 0 --node frontier00003 --rank 0 --step 9 --injected no --evidence slow --time '2026-10-07 00:00:00' >/dev/null
# n4: an injected hard fault (test, node healthy)
ap --job 105 --cls node_lost --code 137 --node frontier00004 --rank 16 --step 60 --injected yes --evidence 'inject kill' --time '2026-10-06 00:00:00' >/dev/null
# a hard line with no node: ignored
ap --job 106 --cls xgmi_error --code 1 --node - --rank - --step - --injected no --evidence 'no node' --time '2026-10-06 00:00:00' >/dev/null
# [2026-10-09 late evening] n5: two gate exclusions (gate_gemm, gate_pair; code 30) in 30 days -> a soft lemon; n6: one -> not a lemon
ap --job 107 --cls gate_gemm --code 30 --node frontier00005 --rank - --step - --injected no --evidence 'the gate excluded it as degraded: gemm below threshold' --time '2026-10-03 00:00:00' >/dev/null
ap --job 108 --cls gate_pair --code 30 --node frontier00005 --rank - --step - --injected no --evidence 'the gate excluded it as dead: did not answer two partners' --time '2026-10-08 00:00:00' >/dev/null
ap --job 109 --cls gate_gemm --code 30 --node frontier00006 --rank - --step - --injected no --evidence 'the gate excluded it as degraded' --time '2026-10-08 00:00:00' >/dev/null

check "header once in A" "1" "$(grep -c '^job	time	class	code	node	rank	step	injected	evidence$' $A)"
check "A has 10 data lines" "10" "$(($(wc -l < $A)-1))"
check "B has header + 2 lines" "3" "$(wc -l < $B)"
check "every A line has 9 fields" "9" "$(awk -F'\t' '{print NF}' $A | sort -u)"

check "lemons nodes-only (30 d; the gate classes are soft: two make n5 a lemon, one does not make n6 one)" $'frontier00001\nfrontier00002\nfrontier00005' "$($FR lemons --days 30 --record $A --record $B --now "$NOW" --nodes-only)"
check "lemons full lines (30 d)" $'frontier00001\thard\tnode_lost@2026-10-05 10:00:00\nfrontier00002\tsoft\tcollective_hang@2026-10-01 08:00:00; collective_hang@2026-10-08 08:00:00\nfrontier00005\tsoft\tgate_gemm@2026-10-03 00:00:00; gate_pair@2026-10-08 00:00:00' \
      "$($FR lemons --days 30 --record $A --record $B --now "$NOW")"
check "include-injected adds n4" $'frontier00001\nfrontier00002\nfrontier00004\nfrontier00005' "$($FR lemons --days 30 --record $A --record $B --now "$NOW" --include-injected --nodes-only)"
check "100-day window makes n3 soft" $'frontier00001\nfrontier00002\nfrontier00003\nfrontier00005' "$($FR lemons --days 100 --record $A --record $B --now "$NOW" --nodes-only)"
check "hard line 30+ d old and 1 soft in window -> no lemon" "" "$($FR lemons --days 30 --record $A --now "2026-11-04 12:00:00" --nodes-only)"
check "missing record file -> empty, exit 0" "|0" "$($FR lemons --days 30 --record $T/none.tsv; echo "|$?")"
# importable API
check "import API" "[('frontier00001', 'hard'), ('frontier00002', 'soft'), ('frontier00005', 'soft')]" \
  "$(cd $HERE/../bin && python3 -c "import failure_record as f; print([(n,k) for n,k,_ in f.lemons(['$A','$B'],30,now='$NOW')])")"
# concurrency: 40 parallel appends to one file -> 40 whole lines
C=$T/c/failures.tsv
for i in $(seq 1 40); do $FR append --record $C --job $i --cls rank_died --code 1 --node n$i --rank 0 --step 0 --injected no --evidence "parallel $i" >/dev/null & done; wait
check "40 parallel appends, 1 header, 40 lines of 9 fields" "1 40 9" "$(grep -c '^job	' $C) $(grep -vc '^job	' $C) $(awk -F'\t' '{print NF}' $C | sort -u)"
echo "failures: $NF"; [ $NF -eq 0 ]
