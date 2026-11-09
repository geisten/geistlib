#!/bin/sh
# #374 profile (throwaway, profile branch only): prefill and decode of one
# model on the CPU, headline numbers, forward-stage profile, the MoE
# experts' thread utilization (GEIST_MOE_UTIL), the 128-row chunk, and
# llama-bench beside it.
#   scripts/profile_374.sh <bench_perf_sweep> <model.gguf> <backend> <threads> <llama-bench or ->
set -u
B=$1 M=$2 BE=$3 T=$4 LB=$5
P=docs/OFFLOAD.md
export GEIST_BENCH_BACKEND=$BE
echo "== $(basename "$M") backend $BE threads $T  load $(uptime | sed 's/.*average[s]*: //')"
echo "-- headline pp512/tg128 (2 repeats)"
$B --gguf "$M" --seq-lens 512 --decode-n 128 --warmup 1 --repeats 2 --prompt-file $P 2>/dev/null
echo "-- prefill-only profile (512, 2 repeats, no warmup)"
GEIST_PROFILE_FORWARD=1 GEIST_MOE_UTIL=1 $B --gguf "$M" --seq-lens 512 --decode-n 0 --warmup 0 --repeats 2 --prompt-file $P 2>&1 | grep -v '^geist'
echo "-- prefill+decode profile (512 + 128, 2 repeats, no warmup)"
GEIST_PROFILE_FORWARD=1 GEIST_MOE_UTIL=1 $B --gguf "$M" --seq-lens 512 --decode-n 128 --warmup 0 --repeats 2 --prompt-file $P 2>&1 | grep -v '^geist'
echo "-- GEIST_M_MAX=128 (prefill chunk at the CPU cap)"
GEIST_M_MAX=128 GEIST_MOE_UTIL=1 $B --gguf "$M" --seq-lens 512 --decode-n 0 --warmup 1 --repeats 2 --prompt-file $P 2>&1 | grep -v '^geist'
if [ "$LB" != "-" ]; then
    echo "-- llama-bench"
    "$LB" -m "$M" -p 512 -n 128 -r 2 -t "$T" -o jsonl 2>/dev/null | grep -o '"n_prompt": [0-9]*, "n_gen": [0-9]*.*"avg_ts": [0-9.]*'
fi
