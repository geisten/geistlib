#!/bin/bash
# #48 research note: one model on the Raspberry Pi 5, geist cpu_neon
# against llama.cpp CPU (llama-server), interleaved cycles, plus a
# decode-energy pair per engine from the PMIC (vcgencmd pmic_read_adc, the
# sum of I x V over the rails: internal telemetry, not a wall meter).
#
#   pi_moe_bench.sh <label> <gguf> <cycles> <out.jsonl>
#
# Run it under ~/bin/with-pi-lock. Throughput: bench_perf_sweep on the
# first 512 tokens of docs/OFFLOAD.md, 128 decoded, warmup 2, repeats 2;
# llama-server /completion on the same token ids (one warmup request, then
# 2). Before every run: SoC below 60 C and load-1m below 0.5, up to 10 min
# (the board's thermal rule). Energy: 32-token prompt with 64 and 320
# decoded tokens; decode power = (E_320 - E_64) / (T_320 - T_64), so load,
# prefill and warmup cancel; the idle board power before each run is
# recorded too.
set -u
LABEL=$1 MODEL=$2 CYCLES=$3 OUT=$4
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
GB=$(ls "$ROOT"/bin/*/release/tests/bench_perf_sweep | head -1)
LS=${LLAMA_SERVER:-$HOME/llama.cpp-cross-engine/build-cpu/bin/llama-server}
PROMPT=$ROOT/docs/OFFLOAD.md
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"; pkill -P $$ 2>/dev/null' EXIT

watts() {
    vcgencmd pmic_read_adc | python3 -c '
import re,sys
a,v={},{}
for r,k,x in re.findall(r"(\S+)_(?:A|V)\s+(current|volt)\(\d+\)=([\d.]+)", sys.stdin.read()):
    (a if k=="current" else v)[r]=float(x)
print(sum(c*v[r] for r,c in a.items() if r in v))'
}

gate() {
    local i t l
    for i in $(seq 120); do
        t=$(cat /sys/class/thermal/thermal_zone0/temp)
        l=$(cut -d' ' -f1 /proc/loadavg)
        if [ "$t" -lt 60000 ] && awk -v l="$l" 'BEGIN{exit !(l<0.5)}'; then
            echo "$t $l $(watts) ok"; return
        fi
        sleep 5
    done
    echo "$t $l $(watts) timeout"
}

mon_start() {
    ( while :; do echo "$(date +%s.%N) $(watts)"; sleep 0.2; done ) > "$TMP/mon.txt" &
    MON=$!
}
mon_stop() { # trapezoid integral of the board power (J), mean W, samples
    kill $MON 2>/dev/null; wait $MON 2>/dev/null
    python3 - "$TMP/mon.txt" <<'PY'
import json, sys
s = [tuple(map(float, l.split())) for l in open(sys.argv[1]) if len(l.split()) == 2]
e = sum((t1 - t0) * (w0 + w1) / 2 for (t0, w0), (t1, w1) in zip(s, s[1:]))
T = s[-1][0] - s[0][0] if len(s) > 1 else 0
print(json.dumps({"e_board_j": e, "mean_w": e / T if T else 0, "samples": len(s), "span_s": T}))
PY
}

llama_up() {
    "$LS" -m "$MODEL" -ngl 0 -c 4096 -np 1 -t 4 --port 8099 > "$TMP/server.log" 2>&1 &
    SRV=$!
    for _ in $(seq 600); do
        curl -s -m 2 http://127.0.0.1:8099/health | grep -q ok && return 0
        kill -0 $SRV 2>/dev/null || return 1
        sleep 1
    done
    return 1
}
llama_down() { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; }
llama_req() {
    python3 - "$PROMPT" "$1" "$2" <<'PY'
import json, sys, urllib.request
base = "http://127.0.0.1:8099"
def post(p, b):
    r = urllib.request.Request(base + p, data=json.dumps(b).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))
ids = post("/tokenize", {"content": open(sys.argv[1]).read()})["tokens"][: int(sys.argv[2])]
t = post("/completion", {"prompt": ids, "n_predict": int(sys.argv[3]), "temperature": 0,
                         "cache_prompt": False, "ignore_eos": True})["timings"]
print(json.dumps({"prefill_tps": t["prompt_per_second"], "decode_tps": t["predicted_per_second"],
                  "n_prompt": t["prompt_n"], "n_decoded": t["predicted_n"]}))
PY
}
geist_run() { # seq decode-n repeats
    /usr/bin/time -v env GEIST_BENCH_BACKEND=cpu_neon "$GB" --gguf "$MODEL" --seq-lens "$1" \
        --decode-n "$2" --warmup 2 --repeats "$3" --threads 4 --prompt-file "$PROMPT" \
        2> "$TMP/geist.err" | tail -1 | python3 -c '
import json,re,sys
d=json.loads(sys.stdin.read())
err=open(sys.argv[1]).read()
mx=re.search(r"Maximum resident set size \(kbytes\): (\d+)", err)
mj=re.search(r"Major \(requiring I/O\) page faults: (\d+)", err)
print(json.dumps({"prefill_tps": d.get("prefill_tps"), "decode_tps": d.get("decode_tps"),
                  "maxrss_mb": int(mx.group(1))/1024 if mx else None,
                  "major_faults": int(mj.group(1)) if mj else None, "raw": d}))' "$TMP/geist.err"
}
now() { date +%s.%N; }

cat "$MODEL" > /dev/null # page cache after the model switch (as far as it fits)
for c in $(seq "$CYCLES"); do
    if [ $((c % 2)) -eq 1 ]; then order="geist-neon llama-cpu"; else order="llama-cpu geist-neon"; fi
    for cfg in $order; do
        g=$(gate)
        t0=$(now)
        if [ $cfg = geist-neon ]; then
            r=$(geist_run 512 128 2)
        elif llama_up; then
            llama_req 512 16 > /dev/null
            r=$(for k in 1 2; do llama_req 512 128; done | python3 -c '
import json,sys
rs=[json.loads(l) for l in sys.stdin]
print(json.dumps({"prefill_tps": sum(x["prefill_tps"] for x in rs)/len(rs),
                  "decode_tps": sum(x["decode_tps"] for x in rs)/len(rs), "raw": rs}))')
            llama_down
        else
            r='{"error": "server did not come up"}'
            llama_down
        fi
        echo "{\"model\": \"$LABEL\", \"cfg\": \"$cfg\", \"cycle\": $c, \"kind\": \"throughput\", \"gate\": \"$g\", \"wall_s\": $(python3 -c "print(round($(now) - $t0, 3))"), \"result\": $r}" | tee -a "$OUT"
    done
done

for cfg in geist-neon llama-cpu; do
    [ $cfg = llama-cpu ] && { llama_up && llama_req 32 8 > /dev/null; }
    for nd in 64 320; do
        g=$(gate)
        mon_start
        t0=$(now)
        if [ $cfg = geist-neon ]; then r=$(geist_run 32 $nd 1); else r=$(llama_req 32 $nd); fi
        t1=$(now)
        e=$(mon_stop)
        echo "{\"model\": \"$LABEL\", \"cfg\": \"$cfg\", \"kind\": \"energy\", \"decode_n\": $nd, \"gate\": \"$g\", \"wall_s\": $(python3 -c "print(round($t1 - $t0, 3))"), \"energy\": $e, \"result\": $r}" | tee -a "$OUT"
    done
    [ $cfg = llama-cpu ] && llama_down
done
