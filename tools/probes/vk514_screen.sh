#!/bin/sh
# vk514_screen.sh MODEL [CYCLES] — decode on the RTX 2080 Ti for each
# revision in $REVS (label=rev): the device unit tests ($PROBE_TESTS),
# 64 greedy tokens against the first revision's, geist bench_perf_sweep
# decode tok/s alternating with llama-bench tg128 (pinned llama.cpp,
# Vulkan), then one GEIST_VK_PROFILE run per revision and llama.cpp's
# GGML_VK_PERF_LOGGER. Builds outside the locks (#514 screening).
set -u
MODEL=$1
CYCLES=${2:-4}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
ROOT=$(pwd)
LABELS=""
for spec in $REVS; do
    label=${spec%%=*}
    rev=${spec#*=}
    wt="$RUNNER_TEMP/wt-$label"
    rm -rf "$wt"
    git worktree prune
    git worktree add -f --detach "$wt" "$rev" >/dev/null 2>&1 || { echo "no rev $rev"; exit 1; }
    (cd "$wt" && git log -1 --oneline && make -j16 TARGET=linux BACKENDS="vulkan cpu_x86 cpu_scalar" GEMM_PROVIDER=native bin) \
        >"$OUT/build_$label.log" 2>&1 || { echo "build $label failed"; tail -30 "$OUT/build_$label.log"; exit 1; }
    LABELS="$LABELS $label"
done
cat "$MODEL" >/dev/null # page cache
LB=$(sh "$ROOT/scripts/ensure-llama-bench.sh" vulkan 2>"$OUT/llama-build.log") || { tail -20 "$OUT/llama-build.log"; exit 1; }
export GEIST_QUIET=1
. "$ROOT/scripts/gpu-lock.sh"
uptime | tee "$OUT/load.txt"
for label in $LABELS; do # unit tests on the device, and 64 greedy tokens
    for t in ${PROBE_TESTS:-test_backend_vulkan_ops_unit test_backend_vulkan_linear_parity test_moe_device_unit test_moe_grouped_unit}; do
        "$RUNNER_TEMP/wt-$label/bin/linux/release/tests/$t" >"$OUT/unit_${label}_$t.txt" 2>&1
        echo "unit $label $t rc=$?" >>"$OUT/unit.txt"
    done
    GEIST_BENCH_BACKEND=vulkan "$RUNNER_TEMP/wt-$label/bin/linux/release/tests/bench_perf_sweep" \
        --gguf "$MODEL" --seq-lens 16 --decode-n 64 --warmup 0 --repeats 1 --emit-tokens \
        >"$OUT/tokens_$label.jsonl" 2>/dev/null
done
for wt_label in $LABELS; do # warm the page cache and the pipeline cache
    GEIST_BENCH_BACKEND=vulkan "$RUNNER_TEMP/wt-$wt_label/bin/linux/release/tests/bench_perf_sweep" \
        --gguf "$MODEL" --seq-lens 16 --decode-n 16 --warmup 1 --repeats 1 >/dev/null 2>&1
done
c=1
while [ $c -le "$CYCLES" ]; do
    for label in $LABELS; do
        GEIST_BENCH_BACKEND=vulkan "$RUNNER_TEMP/wt-$label/bin/linux/release/tests/bench_perf_sweep" \
            --gguf "$MODEL" --seq-lens 16 --decode-n 128 --warmup 1 --repeats 3 \
            >"$OUT/geist_${label}_$c.jsonl" 2>"$OUT/geist_${label}_$c.err"
    done
    GGML_VK_VISIBLE_DEVICES=0 "$LB" -m "$MODEL" -ngl 99 -p 0 -n 128 -r 3 -o jsonl \
        >"$OUT/llama_$c.jsonl" 2>"$OUT/llama_$c.err"
    c=$((c + 1))
done
for label in $LABELS; do
    GEIST_VK_PROFILE=1 GEIST_BENCH_BACKEND=vulkan "$RUNNER_TEMP/wt-$label/bin/linux/release/tests/bench_perf_sweep" \
        --gguf "$MODEL" --seq-lens 16 --decode-n 128 --warmup 0 --repeats 1 \
        >"$OUT/profile_$label.jsonl" 2>"$OUT/profile_$label.txt"
done
GGML_VK_PERF_LOGGER=1 GGML_VK_VISIBLE_DEVICES=0 "$LB" -m "$MODEL" -ngl 99 -p 0 -n 32 -r 1 \
    >"$OUT/llama_perf.txt" 2>&1
python3 - "$OUT" $LABELS <<'PY' >"$OUT/summary.md"
import json, sys, glob, statistics as S
out = sys.argv[1]
print("| engine | decode tok/s per cycle | median |\n|---|---|---|")
for lab in sys.argv[2:]:
    v = []
    for f in sorted(glob.glob(f"{out}/geist_{lab}_*.jsonl")):
        for line in open(f):
            if line.startswith("{"):
                v.append(json.loads(line)["decode_tps"])
    print(f"| geist {lab} | {' '.join(f'{x:.1f}' for x in v)} | {S.median(v) if v else 0:.1f} |")
v = []
for f in sorted(glob.glob(f"{out}/llama_*.jsonl")):
    for line in open(f):
        if line.startswith("{"):
            v.append(json.loads(line)["avg_ts"])
print(f"| llama.cpp tg128 | {' '.join(f'{x:.1f}' for x in v)} | {S.median(v) if v else 0:.1f} |")
print("\n```"); print(open(f"{out}/unit.txt").read()); print("```")
toks = {}
for lab in sys.argv[2:]:
    for line in open(f"{out}/tokens_{lab}.jsonl"):
        if line.startswith("{"):
            toks[lab] = json.loads(line).get("tokens", [])
base = sys.argv[2]
for lab in sys.argv[3:]:
    a, b = toks.get(base, []), toks.get(lab, [])
    same = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    print(f"\ngreedy tokens {lab} vs {base}: first {same} of {len(a)} identical")
lines = open(f"{out}/llama_perf.txt", errors="replace").read().splitlines()
print("\n### llama.cpp perf logger (last block)\n```")
starts = [i for i, l in enumerate(lines) if "Vulkan Timings" in l]
print("\n".join(lines[starts[-2] if len(starts) > 1 else 0:][:80]))
print("```")
for lab in sys.argv[2:]:
    print(f"\n### profile {lab}\n```")
    print("".join(l for l in open(f"{out}/profile_{lab}.txt") if l.startswith("  ") or "profile" in l))
    print("```")
PY
cat "$OUT/summary.md"
