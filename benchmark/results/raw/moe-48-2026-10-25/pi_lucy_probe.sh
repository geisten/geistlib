#!/bin/bash
# #48: does Qwen3-MoE 4x1.7B Q4_K_M (3.0 GB) fit the 4 GB Pi 5? A memory
# cap of 3000 MB (systemd scope, no swap) keeps the board's other services
# safe; geist cpu_neon and llama.cpp CPU, 32-token prompt, 16 decoded.
M=~/geist-models/moe/Qwen3-MOE-4x1.7B-6.8B-Lucys-Song-Uncensored.Q4_K_M.gguf
cd ~/geistlib-48
until [ "$(cat /sys/class/thermal/thermal_zone0/temp)" -lt 60000 ] && awk '{exit !($1<0.5)}' /proc/loadavg; do sleep 5; done
echo "== gate $(cat /sys/class/thermal/thermal_zone0/temp) $(cut -d' ' -f1 /proc/loadavg); free: $(free -m | awk '/Mem/{print $7}') MB available"
cat $M > /dev/null
echo "== geist cpu_neon"
timeout 900 systemd-run --user --scope -q -p MemoryMax=3000M -p MemorySwapMax=0 /usr/bin/time -v env GEIST_BENCH_BACKEND=cpu_neon bin/pi5/release/tests/bench_perf_sweep --gguf $M --seq-lens 32 --decode-n 16 --warmup 2 --repeats 1 --threads 4 --prompt-file docs/OFFLOAD.md 2>&1 | grep -E '^\{|Maximum resident|Major|Exit status|Killed|error|failed' | cut -c1-400
echo "== llama.cpp CPU"
timeout 900 systemd-run --user --scope -q -p MemoryMax=3000M -p MemorySwapMax=0 /usr/bin/time -v ~/llama.cpp-cross-engine/build-cpu/bin/llama-bench -m $M -p 32 -n 16 -r 1 -t 4 -o jsonl 2>&1 | grep -E '^\{|Maximum resident|Major|Exit status|Killed|error' | cut -c1-400
