#!/bin/bash
# #48: llama.cpp on the 4x1.7B MoE under the same 3,000 MB cap (no swap) as
# geist. llama-server did not come up there in pi_moe_bench.sh; this logs
# why, then measures with llama-bench (pseudo-random tokens, 512 + 128,
# 2 repetitions) and geist on the same synthetic shape for a like-for-like row.
M=~/geist-models/moe/Qwen3-MOE-4x1.7B-6.8B-Lucys-Song-Uncensored.Q4_K_M.gguf
CAP="systemd-run --user --scope -q -p MemoryMax=3000M -p MemorySwapMax=0"
gate() { until [ "$(cat /sys/class/thermal/thermal_zone0/temp)" -lt 60000 ] && awk '{exit !($1<0.5)}' /proc/loadavg; do sleep 5; done
         echo "== gate $(cat /sys/class/thermal/thermal_zone0/temp) $(cut -d' ' -f1 /proc/loadavg)"; }
cd ~/geistlib-48
gate
echo "== llama-server under the cap"
timeout 120 $CAP /usr/bin/time -v ~/llama.cpp-cross-engine/build-cpu/bin/llama-server -m $M -ngl 0 -c 4096 -np 1 -t 4 --port 8099 > /tmp/lucy-server.log 2>&1 &
sleep 60; curl -s -m 5 http://127.0.0.1:8099/health; echo; pkill -f "llama-server -m $M"; wait
grep -iE "error|fail|killed|memory|Maximum resident|Exit status|signal" /tmp/lucy-server.log | tail -8
gate
echo "== llama-bench under the cap"
$CAP /usr/bin/time -v ~/llama.cpp-cross-engine/build-cpu/bin/llama-bench -m $M -p 512 -n 128 -r 2 -t 4 -o jsonl 2>&1 |
    python3 -c '
import json,sys
for l in sys.stdin:
    if l.startswith("{"):
        d=json.loads(l); print(json.dumps({k:d[k] for k in ("n_prompt","n_gen","avg_ts","stddev_ts","samples_ts")}))
    elif "Maximum resident" in l or "Major" in l or "Exit status" in l: print(l.strip())'
gate
echo "== geist under the cap, same synthetic shape (no --prompt-file)"
$CAP /usr/bin/time -v env GEIST_BENCH_BACKEND=cpu_neon bin/pi5/release/tests/bench_perf_sweep --gguf $M --seq-lens 512 --decode-n 128 --warmup 2 --repeats 2 --threads 4 2>&1 |
    python3 -c '
import json,sys
for l in sys.stdin:
    if l.startswith("{"):
        d=json.loads(l); print(json.dumps({k:d[k] for k in ("seq_len","decode_n","prefill_tps","decode_tps","rss_mb")}))
    elif "Maximum resident" in l or "Major" in l or "Exit status" in l: print(l.strip())'
