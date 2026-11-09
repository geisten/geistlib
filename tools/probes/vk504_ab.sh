#!/bin/sh
# vk504_ab.sh [CYCLES] [MODELS] — #504 speed on the RTX 2080 Ti, under the
# bench + GPU locks (GEIST_QUIET=1 scripts/gpu-lock.sh):
#   - A/B (tools/bench_revision_ab.py, base = 2nd of $REVS, new = 1st) on
#     the files main also runs on Vulkan: DeepSeek-V2-Lite and GigaChat3
#     decompressed (device MLA now), Llama-3.2-3B (no-regression);
#   - the absorbed files (main refuses them): geist new vs llama-bench
#     Vulkan, interleaved per cycle, pp512 / pp2048 and decode after each.
# MODELS: space separated subset of: v2 gigac llama giga v2s (default all).
. tools/probes/vk504_common.sh
CYCLES=${1:-6}
MODELS=${2:-"v2 gigac llama giga v2s"}
NEW=$(echo $REVS | cut -d' ' -f1); NEW=${NEW#*=}
BASE=$(echo $REVS | cut -d' ' -f2); BASE=${BASE#*=}
REVS="new=$NEW"
build_revs
llama_tools
SW=$RUNNER_TEMP/wt-new/bin/linux/release/tests/bench_perf_sweep
export GEIST_QUIET=1
. "$ROOT/scripts/gpu-lock.sh"
uptime | tee "$OUT/load.txt"
for m in $MODELS; do
    case $m in
    v2) f=$V2 ;; gigac) f=$GIGA_C ;; llama) f=$HOME/geist-models/llama-3.2-3b-Q4_K_M.gguf ;;
    giga) f=$GIGA ;; v2s) f=$V2_S ;;
    esac
    cat "$f" >/dev/null # page cache
    case $m in
    v2|gigac|llama)
        python3 "$ROOT/tools/bench_revision_ab.py" --rev "base=$BASE" --rev "new=$NEW" \
            --gguf "$f" --seq-lens 512,2048 --decode-n 64 --repeats 1 --warmup 1 --cycles "$CYCLES" \
            --env GEIST_BENCH_BACKEND=vulkan --make-arg TARGET=linux \
            --make-arg "BACKENDS=vulkan cpu_x86 cpu_scalar" --make-arg GEMM_PROVIDER=native \
            --jobs 16 --out-dir "$OUT/ab-$m" >"$OUT/ab-$m.txt" 2>&1
        { echo "### A/B $m ($(basename "$f"))"; echo '```'; cat "$OUT/ab-$m.txt" | tail -40; echo '```'; } >>"$OUT/summary.md"
        ;;
    giga|v2s)
        case $m in v2s) ngl="-ngl 99 --n-cpu-moe 4" ;; *) ngl="-ngl 99" ;; esac
        GEIST_BENCH_BACKEND=vulkan "$SW" --gguf "$f" --seq-lens 64 --decode-n 4 --warmup 1 --repeats 1 >/dev/null 2>&1
        c=1
        while [ $c -le "$CYCLES" ]; do
            GEIST_BENCH_BACKEND=vulkan "$SW" --gguf "$f" --seq-lens 512,2048 --decode-n 64 --warmup 1 --repeats 1 \
                >"$OUT/geist_${m}_$c.jsonl" 2>"$OUT/geist_${m}_$c.err"
            # shellcheck disable=SC2086
            GGML_VK_VISIBLE_DEVICES=0 "$LB" -m "$f" $ngl -p 512,2048 -n 64 -r 1 -o jsonl \
                >"$OUT/llama_${m}_$c.jsonl" 2>"$OUT/llama_${m}_$c.err"
            # llama-bench's tg runs on an empty cache: decode at depth too
            # shellcheck disable=SC2086
            GGML_VK_VISIBLE_DEVICES=0 "$LB" -m "$f" $ngl -p 0 -n 64 -d 512,2048 -r 1 -o jsonl \
                >"$OUT/llama_${m}_d_$c.jsonl" 2>"$OUT/llama_${m}_d_$c.err"
            c=$((c + 1))
        done
        python3 - "$OUT" "$m" <<'PY' >>"$OUT/summary.md"
import glob, json, statistics as S, sys
out, m = sys.argv[1], sys.argv[2]
g = {}
for f in sorted(glob.glob(f"{out}/geist_{m}_*.jsonl")):
    for line in open(f):
        if line.startswith("{"):
            d = json.loads(line)
            n = d.get("seq_len") or d.get("prompt_tokens")
            g.setdefault(("pp", n), []).append(d.get("prefill_tps"))
            g.setdefault(("tg", n), []).append(d.get("decode_tps"))
l = {}
for f in sorted(glob.glob(f"{out}/llama_{m}_*.jsonl")):
    for line in open(f):
        if line.startswith("{"):
            d = json.loads(line)
            key = ("pp", d["n_prompt"]) if d["n_prompt"] else ("tg", d.get("n_depth", 0))
            l.setdefault(key, []).append(d["avg_ts"])
print(f"### {m}: geist new vs llama.cpp Vulkan (tok/s, median of cycles)\n")
print("| what | geist | llama.cpp | ratio |\n|---|---|---|---|")
for k in sorted(set(g) | set(l), key=str):
    gv = [x for x in g.get(k, []) if x is not None]
    gm = S.median(gv) if gv else float("nan")
    lm = S.median(l[k]) if l.get(k) else float("nan")
    print(f"| {k[0]} @{k[1]} | {gm:.1f} | {lm:.1f} | {gm / lm if lm == lm and lm else float('nan'):.2f} |")
print("\nraw geist keys:", sorted(g, key=str), "\n")
PY
        ;;
    esac
done
cat "$OUT/summary.md"
