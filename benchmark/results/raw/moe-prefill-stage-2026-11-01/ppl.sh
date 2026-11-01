#!/bin/bash
# perplexity of docs/OFFLOAD.md tokens 2048..3071, context [0,2048) prefilled in 512-row chunks
cd ~/work-moe-239
B=./bin/linux/release/tests/bench_nll
for M in Qwen3.6-35B-A3B-UD-Q4_K_XL Qwen3-30B-A3B-Q4_K_M; do
  P=benchmark/results/raw/2026-10-23_qwen36_expert_usage_run1830.txt
  [ $M = Qwen3-30B-A3B-Q4_K_M ] && P=benchmark/results/raw/2026-10-15_moe_expert_usage_vulkan_run1225.txt
  for v in GEIST_MOE_STAGE=0 X=default; do
    r=$(env $v GEIST_VERBOSE=1 GEIST_NLL_PREFILL=1 GEIST_NLL_M_MAX=512 GEIST_MOE_PROFILE=$P GEIST_BENCH_BACKEND=vulkan $B ~/geist-models/$M.gguf docs/OFFLOAD.md 3072 2048 2>ppl.err | tail -1)
    echo "{\"model\":\"$M\",\"v\":\"$v\",\"staged\":\"$(grep -oE '[0-9]+ cold experts staged on vulkan for [0-9]+ rows' ppl.err)\",\"r\":$r}"
  done
done
