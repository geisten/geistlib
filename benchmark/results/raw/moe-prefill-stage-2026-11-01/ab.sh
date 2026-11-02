#!/bin/bash
# ab.sh <cycles> <variant-env>... : interleaved, rotated order; one JSON line per run
cd ~/work-moe-239
B=${B:-./bin/linux/release/tests/bench_perf_sweep}
P36=benchmark/results/raw/2026-10-23_qwen36_expert_usage_run1830.txt
P30=benchmark/results/raw/2026-10-15_moe_expert_usage_vulkan_run1225.txt
M=${M:-Qwen3.6-35B-A3B-UD-Q4_K_XL}
P=$P36; [ "$M" = Qwen3-30B-A3B-Q4_K_M ] && P=$P30
SEQ=${SEQ:-2726}
cyc=$1; shift
vs=("$@"); n=${#vs[@]}
for i in $(seq 90); do awk -v l=$(cut -d" " -f1 /proc/loadavg) 'BEGIN{exit !(l<3)}' && break; sleep 10; done
# warm-up: the model file into the page cache (a model switch evicts the other)
env GEIST_BENCH_BACKEND=vulkan GEIST_MOE_PROFILE=$P $B --gguf ~/geist-models/$M.gguf --seq-lens 512 --decode-n 8 --warmup 0 --repeats 1 --prompt-file docs/OFFLOAD.md --m-max 512 >/dev/null 2>&1
echo "# start load $(cut -d' ' -f1-3 /proc/loadavg) $(date +%T) model $M"
for c in $(seq $cyc); do
  for k in $(seq 0 $((n-1))); do
    v=${vs[$(( (k + c) % n ))]}
    l=$(cut -d" " -f1 /proc/loadavg)
    log=$(mktemp)
    bin=$B; case $v in MAIN*) bin=$HOME/work-moe-239-main/bin/linux/release/tests/bench_perf_sweep;; esac
    r=$(env $v GEIST_VERBOSE=1 GEIST_MOE_PROFILE=$P GEIST_BENCH_BACKEND=vulkan $bin --gguf ~/geist-models/$M.gguf --seq-lens $SEQ --decode-n 128 --warmup 8 --repeats 1 --prompt-file docs/OFFLOAD.md --m-max 512 2>$log | tail -1)
    th=$(grep -oE "threshold [0-9]+" $log | head -1); st=$(grep -oE "[0-9]+ cold experts staged" $log | head -1)
    echo "{\"cycle\":$c,\"v\":\"$v\",\"load\":$l,\"th\":\"$th\",\"staged\":\"$st\",\"r\":$r}"
    rm -f $log
  done
done
echo "# end load $(cut -d' ' -f1-3 /proc/loadavg) $(date +%T)"
