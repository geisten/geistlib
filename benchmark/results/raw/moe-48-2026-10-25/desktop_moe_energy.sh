#!/bin/bash
# #48 research note: decode energy of Qwen3-30B-A3B on the RTX 2080 Ti box,
# geist GPU mode (#48 profile, experts split, device route #228) against
# llama.cpp Vulkan --n-cpu-moe 24 (its best split that fits, run 1741).
#
#   desktop_moe_energy.sh <gguf> <out.jsonl>
#
# Hold /tmp/geist-bench.lock and /tmp/geist-gpu.lock around it. Power:
# nvidia-smi power.draw every 100 ms (the card only). The CPU package
# (RAPL, /sys/class/powercap/intel-rapl:0/energy_uj) is root-only (0400),
# so the host side is not measured. Per engine, a 32-token prompt with 128
# and 640 decoded tokens; GPU decode power = (E_640 - E_128) / (T_640 -
# T_128). Throughput rows (512-token prompt of docs/OFFLOAD.md, 128
# decoded) are recorded beside it.
set -u
MODEL=$1 OUT=$2
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
GB=$(ls "$ROOT"/bin/linux*/release/tests/bench_perf_sweep | head -1)
LS=$HOME/llama.cpp-cross-engine/build-vulkan/bin/llama-server
PROMPT=$ROOT/docs/OFFLOAD.md
PROF=$ROOT/benchmark/results/raw/2026-10-15_moe_expert_usage_vulkan_run1225.txt
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"; pkill -P $$ 2>/dev/null' EXIT
now() { date +%s.%N; }
mon_start() {
    nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,pstate --format=csv,noheader,nounits -lms 100 > "$TMP/mon.csv" &
    MON=$!
}
mon_stop() {
    kill $MON 2>/dev/null; wait $MON 2>/dev/null
    python3 - "$TMP/mon.csv" <<'PY'
import json, sys
w = [float(l.split(",")[1]) for l in open(sys.argv[1]) if l.count(",") >= 3]
print(json.dumps({"e_gpu_j": sum(w) * 0.1, "mean_w": sum(w) / len(w) if w else 0, "samples": len(w)}))
PY
}
geist_run() { # seq decode-n repeats
    env GEIST_BENCH_BACKEND=vulkan GEIST_MOE_PROFILE=$PROF "$GB" --gguf "$MODEL" --seq-lens "$1" \
        --decode-n "$2" --warmup 2 --repeats "$3" --m-max 512 --prompt-file "$PROMPT" 2> "$TMP/g.err" |
        tail -1 | python3 -c '
import json,sys
d=json.loads(sys.stdin.read()); print(json.dumps({"prefill_tps": d["prefill_tps"], "decode_tps": d["decode_tps"], "raw": d}))'
}
llama_up() {
    "$LS" -m "$MODEL" --device Vulkan0 -ngl 99 --n-cpu-moe 24 -c 4096 -np 1 --port 8099 > "$TMP/s.log" 2>&1 &
    SRV=$!
    for _ in $(seq 600); do
        curl -s -m 2 http://127.0.0.1:8099/health | grep -q ok && return 0
        sleep 1
    done
    return 1
}
llama_req() {
    python3 - "$PROMPT" "$1" "$2" <<'PY'
import json, sys, urllib.request
base = "http://127.0.0.1:8099"
def post(p, b):
    r = urllib.request.Request(base + p, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=1200))
ids = post("/tokenize", {"content": open(sys.argv[1]).read()})["tokens"][: int(sys.argv[2])]
t = post("/completion", {"prompt": ids, "n_predict": int(sys.argv[3]), "temperature": 0,
                         "cache_prompt": False, "ignore_eos": True})["timings"]
print(json.dumps({"prefill_tps": t["prompt_per_second"], "decode_tps": t["predicted_per_second"]}))
PY
}
rec() { echo "{\"model\": \"qwen3-30b-a3b-q4km\", \"cfg\": \"$1\", \"kind\": \"$2\", \"decode_n\": $3, \"load1\": $(cut -d' ' -f1 /proc/loadavg), \"wall_s\": $4, \"energy\": $5, \"result\": $6}" | tee -a "$OUT"; }

cat "$MODEL" > /dev/null
for cfg in geist-gpu llama-vulkan-ncmoe24; do
    [ $cfg = llama-vulkan-ncmoe24 ] && { llama_up && llama_req 32 16 > /dev/null; }
    if [ $cfg = geist-gpu ]; then r=$(geist_run 512 128 3); else
        r=$(for k in 1 2 3; do llama_req 512 128; done | python3 -c '
import json,sys
rs=[json.loads(l) for l in sys.stdin]
print(json.dumps({"prefill_tps": sum(x["prefill_tps"] for x in rs)/len(rs), "decode_tps": sum(x["decode_tps"] for x in rs)/len(rs), "raw": rs}))'); fi
    rec $cfg throughput 128 0 null "$r"
    for nd in 128 640; do
        mon_start; t0=$(now)
        if [ $cfg = geist-gpu ]; then r=$(geist_run 32 $nd 1); else r=$(llama_req 32 $nd); fi
        t1=$(now); e=$(mon_stop)
        rec $cfg energy $nd "$(echo "$t1 - $t0" | bc | sed "s/^\./0./")" "$e" "$r"
    done
    [ $cfg = llama-vulkan-ncmoe24 ] && { kill $SRV; wait $SRV 2>/dev/null; }
done
