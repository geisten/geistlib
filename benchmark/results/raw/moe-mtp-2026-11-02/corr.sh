#!/bin/bash
# corr.sh <out.jsonl> <prompt:seq>... : greedy equivalence, 128 tokens per prompt:
# base OFF (reference, with the runner-up margins), new OFF, new MTP2, new MTP3.
W=~/work-moe-246; cd $W
out=$1; shift
for i in $(seq 180); do awk -v l=$(cut -d" " -f1 /proc/loadavg) 'BEGIN{exit !(l<3)}' && break; sleep 10; done
echo "# start load $(cut -d' ' -f1-3 /proc/loadavg) $(date +%T)" >> $out
for ps in "$@"; do
  p=${ps%%:*}; s=${ps##*:}
  for bv in base_bench:OFF new_bench:OFF new_bench:MTP2 new_bench:MTP3; do
    b=${bv%%:*}; v=${bv##*:}; ex=""
    [ $b = base_bench ] && ex="--emit-margins"
    line=$(BIN=$W/$b ./run.sh $v $p $s 128 --emit-tokens $ex 2>/dev/null)
    echo "{\"cycle\":0,\"bin\":\"$b\",\"env\":\"\",${line#\{}" >> $out
  done
done
echo "# end load $(cut -d' ' -f1-3 /proc/loadavg) $(date +%T)" >> $out
