#!/bin/bash
# perplexity: tokens [ctx, ctx+1023] of a text, context prefilled in 512-row chunks; main (4 GiB, decode-order copies) vs branch (prefill-row copies from cache-same)
W=~/work-mpp; cd $W
for M in Qwen3.6-35B-A3B-UD-Q4_K_XL Qwen3-30B-A3B-Q4_K_M; do
  P=$W/main/benchmark/results/raw/2026-10-23_qwen36_expert_usage_run1830.txt
  [ $M = Qwen3-30B-A3B-Q4_K_M ] && P=$W/main/benchmark/results/raw/2026-10-15_moe_expert_usage_vulkan_run1225.txt
  for win in "prompt.md 3072 2048" "prompt.md 2048 1024" "main/docs/ARCHITECTURE.md 3072 2048" "main/docs/ARCHITECTURE.md 2048 1024"; do
    for v in MAIN NEW; do
      cd=$(mktemp -d); bin=main/bin/linux/release/tests/bench_nll; ex=""
      [ $v = NEW ] && { bin=branch/bin/linux/release/tests/bench_nll; cp cache-same-$M/*.txt $cd/; ex="GEIST_MOE_PROFILE_CACHE=$cd"; }
      r=$(env $ex GEIST_VERBOSE=1 GEIST_NLL_PREFILL=1 GEIST_NLL_M_MAX=512 GEIST_MOE_PROFILE=$P GEIST_BENCH_BACKEND=vulkan $bin ~/geist-models/$M.gguf $win 2>ppl.err | tail -1)
      echo "{\"model\":\"$M\",\"win\":\"$win\",\"v\":\"$v\",\"staged\":\"$(grep -oE '[0-9]+ cold experts staged on vulkan for [0-9]+ rows' ppl.err)\",\"r\":$r}"
      rm -rf $cd
    done
  done
done
