#!/bin/bash
# cycle.sh <n> <out.jsonl>: one interleaved cycle on the 27B Metal, both prompts.
C=${C:-.}
n=$1; out=$2
echo "# cycle $n start $(date +%T) load $(sysctl -n vm.loadavg | tr , .)" >> $out
for pf in ARCHITECTURE.md OFFLOAD.md; do
 for bv in base:OFF new:OFF base:MTP2 new:MTP2 new:MTP3 new:MTP4 ctl:OFF; do
  b=${bv%%:*}; v=${bv##*:}; ex=""; sp=""
  case $v in MTP*) ex="GEIST_MTP=1"; sp="--spec ${v#MTP}";; esac
  r=$(env $ex GEIST_BENCH_BACKEND=metal $C/${b}_bench --gguf $HOME/geist-models/qwen3.8-27b-q4_0.gguf --seq-lens 512 --decode-n 128 --warmup 8 --repeats 1 --prompt-file $C/$pf --emit-tokens $sp 2>/dev/null | tail -1)
  echo "{\"cycle\":$n,\"prompt\":\"$pf\",\"bin\":\"$b\",\"v\":\"$v\",\"load\":\"$(sysctl -n vm.loadavg | tr , . | cut -d' ' -f2)\",\"r\":${r:-null}}" >> $out
 done
done
echo "# cycle $n end $(date +%T) load $(sysctl -n vm.loadavg | tr , .)" >> $out
