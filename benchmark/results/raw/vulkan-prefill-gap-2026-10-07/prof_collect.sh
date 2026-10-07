#!/usr/bin/env bash
# Per-op profiles, pp512, geist (M_MAX 64/128) and llama.cpp (ub 64/128/512).
S=$(cd "$(dirname "$0")" && pwd)
B=$S/wt/main-vk/bin/linux/release/tests/bench_perf_sweep
P=$S/prof; mkdir -p $P
export GGML_VK_VISIBLE_DEVICES=0
declare -A F=(
  [gemma4-e2b]=~/workplace/geistlib/gguf_artifacts/gemma4-e2b-Q4_K_M.gguf
  [qwen35-4b]=~/gguf-campaign/qwen3.5-4b-q4_0.gguf
  [qwen3-0.6b]=~/gguf-campaign/qwen3-0.6b-q8_0.gguf
  [bonsai2-27b]=~/gguf-campaign/Ternary-Bonsai-2-27B-PQ2_0.gguf
)
for k in gemma4-e2b qwen35-4b qwen3-0.6b bonsai2-27b; do
  L=~/llama.cpp-cross-engine/build-vulkan/bin/llama-bench
  [ $k = bonsai2-27b ] && L=~/llama.cpp-cross-engine-PrismML-Eng_llama.cpp/build-vulkan/bin/llama-bench
  for mm in 64 128; do
    GEIST_M_MAX=$mm GEIST_VK_PROFILE=1 GEIST_VK_PROFILE_SHAPES=1 GEIST_BENCH_BACKEND=vulkan GEIST_TEXT_ONLY=1 \
      $B --gguf ${F[$k]} --seq-lens 512 --decode-n 1 --warmup 64 --repeats 1 --threads 4 --emit-jsonl \
      > $P/geist_${k}_m$mm.jsonl 2> $P/geist_${k}_m$mm.err
    echo "geist $k m$mm rc=$?"
  done
  for ub in ${UBS-64 128 512}; do
    GGML_VK_PERF_LOGGER=1 $L -m ${F[$k]} -p 512 -n 0 -r 2 -ub $ub -ngl 99 -t 4 -o jsonl \
      > $P/llama_${k}_ub$ub.jsonl 2> $P/llama_${k}_ub$ub.err
    echo "llama $k ub$ub rc=$?"
  done
done
