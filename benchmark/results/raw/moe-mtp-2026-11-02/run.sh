#!/bin/bash
# run.sh <variant> <prompt-file> <seq> <decode-n> [extra bench args]: one bench_perf_sweep run, one JSON line.
# variants: OFF (GEIST_MTP unset, decode_step), SEQ (GEIST_MTP=1, decode_step),
#           MTP<k> (GEIST_MTP=1, decode_speculative k), NG<k> (MTP unset, decode_speculative k = n-gram)
W=~/work-moe-246; cd $W
M=${M:-Qwen3.6-35B-A3B-UD-Q4_K_XL}
BIN=${BIN:-$W/branch/bin/linux/release/tests/bench_perf_sweep}
P=$W/branch/benchmark/results/raw/2026-10-23_qwen36_expert_usage_run1830.txt
v=$1; pf=$2; seq=$3; dn=$4; shift 4
ex=""; sp=""
case $v in
  OFF) ;;
  SEQ) ex="GEIST_MTP=1";;
  MTP*) ex="GEIST_MTP=1"; sp="--spec ${v#MTP}";;
  NG*) sp="--spec ${v#NG}";;
esac
log=$(mktemp)
l=$(cut -d" " -f1 /proc/loadavg)
r=$(env $ex $XENV GEIST_VERBOSE=1 GEIST_MOE_PROFILE=$P GEIST_BENCH_BACKEND=vulkan $BIN --gguf ~/geist-models/$M.gguf --seq-lens $seq --decode-n $dn --warmup 8 --repeats 1 --prompt-file $pf --m-max 512 $sp "$@" 2>$log | tail -1)
hot=$(grep -oE "[0-9]+ of [0-9]+ experts \([0-9]+ MiB\) on [a-z]+" $log | head -1)
mtp=$(grep -oE "autoregressive=[0-9]+, mtp=[0-9]+" $log | head -1)
[ -n "$KEEPLOG" ] && cp $log $KEEPLOG
[ -z "$r" ] && { echo "{\"v\":\"$v\",\"error\":true}"; tail -5 $log >&2; rm -f $log; exit 1; }
echo "{\"v\":\"$v\",\"prompt\":\"$(basename $pf)\",\"load\":$l,\"hot\":\"$hot\",\"mtp\":\"$mtp\",\"r\":$r}"
rm -f $log
