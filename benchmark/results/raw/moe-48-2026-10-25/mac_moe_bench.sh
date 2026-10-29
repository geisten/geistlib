#!/bin/bash
# #48 research note: one model on the M1 Max, geist (Metal, optionally
# cpu_neon) against llama.cpp (Metal, optionally CPU), interleaved cycles,
# plus a decode-energy pair per configuration from macmon.
#
#   mac_moe_bench.sh <label> <gguf> <cycles> <cfgs> <out.jsonl>
#   cfgs: ;-separated of geist-metal geist-neon llama-metal llama-cpu
#
# Run it under ~/bin/with-mac-lock. Throughput: bench_perf_sweep on the
# first 512 tokens of docs/OFFLOAD.md, 128 decoded, warmup 2, repeats 3;
# llama-server /completion on the same token ids (one warmup request, then
# 3). Before every run: GPU and CPU below 65 C and CPU busy < 30 % (macmon,
# 3 s), up to 60 s. Energy: two runs per configuration, 32-token prompt
# with 128 and 640 decoded tokens; macmon all_power (CPU + GPU + ANE) is
# integrated over each, and the decode power is
# (E_640 - E_128) / (T_640 - T_128): load, prefill and warmup cancel. The
# idle power before each run is recorded too (other users of the Mac).
set -u
LABEL=$1 MODEL=$2 CYCLES=$3 CFGS=$4 OUT=$5
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
GB=$ROOT/bin/mac-omp/release/tests/bench_perf_sweep
LS=${LLAMA_SERVER:-$HOME/llama.cpp-cross-engine/build-metal/bin/llama-server}
PROMPT=$ROOT/docs/OFFLOAD.md
MACMON=/opt/homebrew/bin/macmon
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"; pkill -P $$ 2>/dev/null' EXIT

gate() { # cool + idle, at most ~60 s; prints the reading
    local i line
    for i in $(seq 20); do
        line=$($MACMON pipe -i 1000 -s 3 2>/dev/null | python3 -c '
import json,sys
r=[json.loads(l) for l in sys.stdin if l.strip()]
g=max(x["temp"]["gpu_temp_avg"] for x in r); c=max(x["temp"]["cpu_temp_avg"] for x in r)
u=sum(x["cpu_usage_pct"] for x in r)/len(r); p=sum(x["all_power"] for x in r)/len(r)
print(f"{g:.1f} {c:.1f} {u:.3f} {p:.2f}")')
        set -- $line
        if awk -v g="$1" -v c="$2" -v u="$3" 'BEGIN{exit !(g<65 && c<65 && u<0.30)}'; then
            echo "$line ok"; return
        fi
    done
    echo "$line timeout"
}

mon_start() { $MACMON pipe -i 100 > "$TMP/mon.jsonl" 2>/dev/null & MON=$!; }
mon_stop() { # integrates all_power, cpu_power, gpu_power (J) over the samples
    kill $MON 2>/dev/null; wait $MON 2>/dev/null
    python3 - "$TMP/mon.jsonl" <<'PY'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
dt = 0.1
print(json.dumps({"e_all_j": sum(x["all_power"] for x in r) * dt,
                  "e_cpu_j": sum(x["cpu_power"] for x in r) * dt,
                  "e_gpu_j": sum(x["gpu_power"] for x in r) * dt,
                  "samples": len(r),
                  "gpu_mhz_med": sorted(x["gpu_freq_mhz"] for x in r)[len(r) // 2] if r else 0}))
PY
}

geist_env() {
    case $1 in
    geist-metal) echo "GEIST_BENCH_BACKEND=metal" ;;
    geist-neon) echo "GEIST_BENCH_BACKEND=cpu_neon" ;;
    esac
}

llama_up() { # $1 = cfg; starts the server, waits for health
    local ngl=99
    [ "$1" = llama-cpu ] && ngl=0
    "$LS" -m "$MODEL" -ngl $ngl -c 4096 -np 1 --port 8099 > "$TMP/server.log" 2>&1 &
    SRV=$!
    for _ in $(seq 300); do
        curl -s -m 2 http://127.0.0.1:8099/health | grep -q ok && return 0
        sleep 1
    done
    return 1
}
llama_down() { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; }

llama_req() { # $1 = prompt tokens (512 of OFFLOAD.md or 32), $2 = n_predict -> timings json
    python3 - "$PROMPT" "$1" "$2" <<'PY'
import json, sys, urllib.request
base = "http://127.0.0.1:8099"
def post(p, b):
    r = urllib.request.Request(base + p, data=json.dumps(b).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=1200))
ids = post("/tokenize", {"content": open(sys.argv[1]).read()})["tokens"][: int(sys.argv[2])]
t = post("/completion", {"prompt": ids, "n_predict": int(sys.argv[3]), "temperature": 0,
                         "cache_prompt": False, "ignore_eos": True})["timings"]
print(json.dumps({"prefill_tps": t["prompt_per_second"], "decode_tps": t["predicted_per_second"],
                  "n_prompt": t["prompt_n"], "n_decoded": t["predicted_n"]}))
PY
}

geist_run() { # $1 cfg, $2 seq, $3 decode-n, $4 repeats -> mean prefill/decode tok/s json
    local mm=512 # one 512-token chunk on Metal; cpu_neon takes at most 128
    [ "$1" = geist-neon ] && mm=128
    env $(geist_env "$1") "$GB" --gguf "$MODEL" --seq-lens "$2" --decode-n "$3" --warmup 2 \
        --repeats "$4" --m-max $mm --prompt-file "$PROMPT" 2> "$TMP/geist.err" | tail -1 |
        python3 -c '
import json,sys
d=json.loads(sys.stdin.read())
print(json.dumps({"prefill_tps": d.get("prefill_tps"), "decode_tps": d.get("decode_tps"),
                  "raw": d}))'
}

# warm the page cache after the model switch (bench-matrix trap)
cat "$MODEL" > /dev/null

# ENERGY_ONLY=1: skip the throughput cycles (a rerun of the energy pair)
[ "${ENERGY_ONLY:-0}" = 1 ] && CYCLES=0
for c in $( [ "$CYCLES" -gt 0 ] && seq "$CYCLES"); do
    # rotate the order every cycle
    set -- $(echo "$CFGS" | tr ';' ' ')
    n=$#
    order=()
    for i in $(seq 0 $((n - 1))); do
        j=$(((i + c - 1) % n + 1))
        order+=("${!j}")
    done
    for cfg in "${order[@]}"; do
        g=$(gate)
        t0=$(python3 -c "import time;print(time.time())")
        case $cfg in
        geist-*) r=$(geist_run "$cfg" 512 128 3) ;;
        llama-*)
            if llama_up "$cfg"; then
                llama_req 512 128 > /dev/null # warmup
                r=$(for k in 1 2 3; do llama_req 512 128; done | python3 -c '
import json,sys
rs=[json.loads(l) for l in sys.stdin]
print(json.dumps({"prefill_tps": sum(x["prefill_tps"] for x in rs)/len(rs),
                  "decode_tps": sum(x["decode_tps"] for x in rs)/len(rs), "raw": rs}))')
            else
                r='{"error": "server did not come up"}'
            fi
            llama_down
            ;;
        esac
        t1=$(python3 -c "import time;print(time.time())")
        echo "{\"model\": \"$LABEL\", \"cfg\": \"$cfg\", \"cycle\": $c, \"kind\": \"throughput\", \"gate\": \"$g\", \"wall_s\": $(echo "$t1 - $t0" | bc | sed "s/^\./0./"), \"result\": $r}" | tee -a "$OUT"
    done
done

# energy: decode power from a 128/640 pair per configuration
for cfg in $(echo "$CFGS" | tr ';' ' '); do
    [ "$cfg" = llama-cpu ] || [ "$cfg" = llama-metal ] && { llama_up "$cfg" && llama_req 32 16 > /dev/null; }
    for nd in 128 640; do
        g=$(gate)
        mon_start
        t0=$(python3 -c "import time;print(time.time())")
        case $cfg in
        geist-*) r=$(geist_run "$cfg" 32 $nd 1) ;;
        llama-*) r=$(llama_req 32 $nd) ;;
        esac
        t1=$(python3 -c "import time;print(time.time())")
        e=$(mon_stop)
        echo "{\"model\": \"$LABEL\", \"cfg\": \"$cfg\", \"kind\": \"energy\", \"decode_n\": $nd, \"gate\": \"$g\", \"wall_s\": $(echo "$t1 - $t0" | bc | sed "s/^\./0./"), \"energy\": $e, \"result\": $r}" | tee -a "$OUT"
    done
    case $cfg in llama-*) llama_down ;; esac
done
