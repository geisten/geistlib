#!/bin/bash
# ceiling: prefill with vs without the CPU (cold) experts, GPU mode, 2726-token request
cd ~/work-moe-239
B=./bin/linux/release/tests/bench_perf_sweep
P36=benchmark/results/raw/2026-10-23_qwen36_expert_usage_run1830.txt
M=Qwen3.6-35B-A3B-UD-Q4_K_XL
# wait (bounded) for running CI jobs to drain
for i in $(seq 90); do awk -v l=$(cut -d" " -f1 /proc/loadavg) 'BEGIN{exit !(l<2.5)}' && break; sleep 10; done
echo "start load $(cut -d' ' -f1-3 /proc/loadavg)"
out=~/work-moe-239/ceil-$(date +%m%d-%H%M).jsonl
run() { env "$@" GEIST_MOE_PROFILE=$P36 GEIST_BENCH_BACKEND=vulkan $B --gguf ~/geist-models/$M.gguf --seq-lens 2726 --decode-n 128 --warmup 8 --repeats 1 --prompt-file docs/OFFLOAD.md --m-max 512 2>$LOG | tail -1; }
LOG=~/work-moe-239/hist.log run XMOE_HIST=1 GEIST_VERBOSE=1; grep -E "XMOE|experts \(|rows ran" ~/work-moe-239/hist.log
for c in 1 2 3; do
  if [ $((c % 2)) = 1 ]; then vs="base skip"; else vs="skip base"; fi
  for v in $vs; do
    x="X=0"; [ $v = skip ] && x="XMOE_SKIP_COLD=1"
    l=$(cut -d" " -f1 /proc/loadavg)
    LOG=/dev/null; r=$(run $x)
    echo "{\"cycle\":$c,\"v\":\"$v\",\"load\":$l,\"r\":$r}" | tee -a $out
  done
done
echo "end load $(cut -d' ' -f1-3 /proc/loadavg)"
