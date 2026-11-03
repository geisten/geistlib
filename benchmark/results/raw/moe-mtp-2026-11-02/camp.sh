#!/bin/bash
# camp.sh <out.jsonl> <cycles> <spec>...   spec = BIN:VARIANT:PROMPT:SEQ[:ENV=..,ENV=..]
# Waits for load < 3 (up to 30 min), warms the model up, then runs the specs
# in rotated order per cycle; one JSON line per run (+ bin/env/cycle fields).
W=~/work-moe-246; cd $W
out=$1; cyc=$2; shift 2
specs=("$@"); n=${#specs[@]}
for i in $(seq 180); do awk -v l=$(cut -d" " -f1 /proc/loadavg) 'BEGIN{exit !(l<3)}' && break; sleep 10; done
echo "# start load $(cut -d' ' -f1-3 /proc/loadavg) $(date +%T)" >> $out
BIN=$W/new_bench ./run.sh OFF prompt.md 512 8 >/dev/null 2>&1
for c in $(seq ${OFF0:-1} $((${OFF0:-1}+cyc-1))); do
  for k in $(seq 0 $((n-1))); do
    sp=${specs[$(( (k + c) % n ))]}
    IFS=: read -r b v p s e <<< "$sp"
    xe=$(echo "$e" | tr , ' ')
    line=$(BIN=$W/$b XENV="$xe" ./run.sh $v $p $s ${DN:-128} --emit-tokens 2>/dev/null)
    echo "{\"cycle\":$c,\"bin\":\"$b\",\"env\":\"$e\",${line#\{}" >> $out
  done
done
echo "# end load $(cut -d' ' -f1-3 /proc/loadavg) $(date +%T)" >> $out
