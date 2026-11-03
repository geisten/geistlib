#!/bin/bash
# mkprof.sh: one request of the prompt on the branch with the profile cache on -> cache-<kind>-<model>/
W=~/work-mpp; cd $W
for M in Qwen3.6-35B-A3B-UD-Q4_K_XL Qwen3-30B-A3B-Q4_K_M; do
  P=$W/main/benchmark/results/raw/2026-10-23_qwen36_expert_usage_run1830.txt
  [ "$M" = Qwen3-30B-A3B-Q4_K_M ] && P=$W/main/benchmark/results/raw/2026-10-15_moe_expert_usage_vulkan_run1225.txt
  for kind in same cross; do
    f=prompt.md; [ $kind = cross ] && f=main/docs/ARCHITECTURE.md
    rm -rf cache-$kind-$M; mkdir cache-$kind-$M
    env GEIST_VERBOSE=1 GEIST_MOE_PROFILE=$P GEIST_MOE_PROFILE_CACHE=$W/cache-$kind-$M GEIST_BENCH_BACKEND=vulkan branch/bin/linux/release/tests/bench_perf_sweep --gguf ~/geist-models/$M.gguf --seq-lens 2726 --decode-n 128 --warmup 0 --repeats 1 --prompt-file $f --m-max 512 2>cache-$kind-$M.log | tail -1 | cut -c1-200
    grep -E "threshold|chosen by|prompt:" cache-$kind-$M.log
    head -c 300 cache-$kind-$M/*.txt; echo; grep -c PREFILL cache-$kind-$M/*.txt; grep STAGE cache-$kind-$M/*.txt
  done
done
