#!/bin/bash
cd ~/work-moe-239
./ab.sh 3 MAIN=1 X=default > final36.jsonl 2>&1
M=Qwen3-30B-A3B-Q4_K_M ./ab.sh 3 MAIN=1 X=default > final30.jsonl 2>&1
./ppl.sh > ppl.jsonl 2>&1
# §6 gate: a dense model on the card, main vs branch, built from scratch
[ -f synth-1b.gguf ] || python3 tools/gen_synth_gguf.py synth-1b.gguf --preset llama32-1b-q4_k > /dev/null
python3 tools/bench_revision_ab.py --rev base=6642260d --rev new=b239 --gguf synth-1b.gguf --seq-lens 128,512 --cycles 6 \
  --env GEIST_BENCH_BACKEND=vulkan --make-arg TARGET=linux --make-arg 'BACKENDS=vulkan cpu_x86 cpu_scalar' --make-arg GEMM_PROVIDER=native \
  --out-dir ab-gate > gate.out 2>&1
