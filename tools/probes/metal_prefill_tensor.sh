#!/bin/sh
# metal_prefill_tensor.sh MODEL [CYCLES] — Metal prefill with and without the
# tensor API (MetalPerformancePrimitives matmul2d), geist against llama.cpp,
# on one model (#314). Run by .gitea/workflows/mac-probe.yml under the host
# lock; writes to $PROBE_OUT.
#
# Four arms, interleaved CYCLES times (order rotated per cycle):
#   geist   GEIST_METAL_TENSOR=0 / =1   (tests/bench_perf_sweep)
#   llama   default (tensor API on M5+) / GGML_METAL_TENSOR_DISABLE=1
# then one GEIST_METAL_PROFILE=1 prefill per geist arm (dispatch counts and
# GPU time per command buffer). The arms' ratio says how much of each
# engine's prefill the matrix units carry on this GPU.
set -eu
MODEL=$1
CYCLES=${2:-3}
OUT=${PROBE_OUT:-probe-out}
SEQ=${PROBE_SEQ:-128,512,1024}
REPS=${PROBE_REPS:-3}
COOL=${PROBE_COOL:-15}
mkdir -p "$OUT"

{
    git rev-parse HEAD
    sw_vers
    sysctl -n machdep.cpu.brand_string hw.memsize
    system_profiler SPDisplaysDataType 2>/dev/null | sed -n '1,14p'
    pmset -g batt 2>/dev/null | head -2
    pmset -g 2>/dev/null | grep -i lowpowermode || true
} >"$OUT/env.txt" 2>&1

make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" bin >"$OUT/build.log" 2>&1
GEIST=bin/mac-omp/release/tests/bench_perf_sweep
LLAMA=$(scripts/ensure-llama-bench.sh metal 2>"$OUT/llama-build.log")
git -C "$(dirname "$(dirname "$(dirname "$LLAMA")")")" rev-parse HEAD >>"$OUT/env.txt" 2>&1 || true

geist_run() { # arm-name tensor
    GEIST_BENCH_BACKEND=metal GEIST_METAL_TENSOR=$2 "$GEIST" --gguf "$MODEL" --seq-lens "$SEQ" \
        --decode-n 1 --warmup 1 --repeats "$REPS" 2>>"$OUT/$1.err" |
        sed "s/^{/{\"arm\":\"$1\",/" >>"$OUT/runs.jsonl"
}
llama_run() { # arm-name [env...]
    arm=$1
    shift
    env "$@" "$LLAMA" -m "$MODEL" -ngl 99 -p "$SEQ" -n 0 -r "$REPS" -t 4 -o jsonl 2>>"$OUT/$arm.err" |
        sed "s/^{/{\"arm\":\"$arm\",/" >>"$OUT/runs.jsonl"
}

: >"$OUT/runs.jsonl"
c=0
while [ "$c" -lt "$CYCLES" ]; do
    case $((c % 4)) in
    0) order="g0 g1 l1 l0" ;;
    1) order="l0 g1 g0 l1" ;;
    2) order="g1 l1 l0 g0" ;;
    *) order="l1 g0 g1 l0" ;;
    esac
    for a in $order; do
        sleep "$COOL"
        case $a in
        g0) geist_run geist_simdgroup 0 ;;
        g1) geist_run geist_tensor 1 ;;
        l1) llama_run llama_tensor ;;
        l0) llama_run llama_simdgroup GGML_METAL_TENSOR_DISABLE=1 ;;
        esac
    done
    c=$((c + 1))
done

for t in 0 1; do
    sleep "$COOL"
    GEIST_BENCH_BACKEND=metal GEIST_METAL_TENSOR=$t GEIST_METAL_PROFILE=1 "$GEIST" --gguf "$MODEL" \
        --seq-lens 512 --decode-n 1 --warmup 1 --repeats 1 >"$OUT/profile_tensor$t.txt" 2>&1 || true
done

python3 - "$OUT" <<'PY'
import json, statistics, sys, os
out = sys.argv[1]
tps = {}
for line in open(os.path.join(out, "runs.jsonl")):
    line = line.strip()
    if not line.startswith("{"):
        continue
    r = json.loads(line)
    arm = r["arm"]
    if arm.startswith("geist"):
        seq, v = r["seq_len"], r["prefill_tps"]
    else:
        if r.get("n_prompt", 0) == 0:
            continue
        seq, v = r["n_prompt"], r["avg_ts"]
    tps.setdefault((arm, seq), []).append(v)
arms = ["geist_simdgroup", "geist_tensor", "llama_simdgroup", "llama_tensor"]
seqs = sorted({s for _, s in tps})
med = {k: statistics.median(v) for k, v in tps.items()}
lines = ["## Metal prefill tok/s, tensor API off/on (median of cycles)", "",
         "| seq | " + " | ".join(arms) + " | geist tensor/simd | llama tensor/simd | geist/llama (tensor) |",
         "|--:|" + "--:|" * (len(arms) + 3)]
for s in seqs:
    row = [med.get((a, s), float("nan")) for a in arms]
    lines.append("| %d | %s | %.2fx | %.2fx | %.2fx |" % (
        s, " | ".join("%.1f" % v for v in row),
        row[1] / row[0], row[3] / row[2], row[1] / row[3]))
lines += ["", "Spread (min-max per arm/seq): " + "; ".join(
    "%s@%d %.0f-%.0f" % (a, s, min(v), max(v)) for (a, s), v in sorted(tps.items())), ""]
open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY
