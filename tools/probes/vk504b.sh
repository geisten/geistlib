#!/bin/sh
# vk504b.sh MODE... — #504 follow-up (absorbed-MLA prefill attention) on the
# desktop's RTX 2080 Ti. The workflow step holds the bench + GPU locks for
# the whole job; nothing here locks again. Every measurement first waits for
# a quiet host (quiet_gate). $REVS: "new=<rev> base=<rev>".
# MODE: unit | prof | ab[=CYCLES] | vsllama[=CYCLES] | parity[=CHUNKS]
set -u
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
ROOT=$(pwd)
M=$HOME/geist-models/moe
GIGA=$M/ai-sage_GigaChat3-10B-A1.8B-Q4_K_M.gguf    # absorbed (split k_b / v_b)
GIGA_C=$M/GigaChat3-10B-A1.8B-Q4_K_M.combined.gguf # decompressed (kv_b)
V2=$M/DeepSeek-V2-Lite.Q4_K_M.gguf                 # decompressed (kv_b)
V2_S=$M/DeepSeek-V2-Lite.Q4_K_M.split.gguf         # absorbed
LLAMA3=$HOME/geist-models/llama-3.2-3b-Q4_K_M.gguf
LLAMA_DIR=$HOME/llama.cpp-cross-engine
NEW=$(echo $REVS | cut -d' ' -f1); NEW=${NEW#*=}
BASE=$(echo $REVS | cut -d' ' -f2); BASE=${BASE#*=}
MK="TARGET=linux GEMM_PROVIDER=native"

# Wait until no CI task container runs and the CPU is >= 97 % idle over
# 10 s (30 min at most), then log the top processes.
quiet_gate() {
    t0=$(date +%s)
    while :; do
        n=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -c GITEA-ACTIONS-TASK)
        idle=$(python3 -c '
import time
def r():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return v[3] + v[4], sum(v)
a = r(); time.sleep(10); b = r()
print(int(100 * (b[0] - a[0]) / max(1, b[1] - a[1])))')
        echo "gate $(date -u +%T): task containers $n, idle $idle % ($1)"
        [ "$n" = 0 ] && [ "$idle" -ge 97 ] && break
        if [ $(($(date +%s) - t0)) -gt 1800 ]; then
            echo "gate: not quiet after 30 min, measuring anyway"
            break
        fi
    done
    ps -eo pid,user,pcpu,etime,comm --sort=-pcpu | head -8
}

build() { # label rev
    wt="$RUNNER_TEMP/wt-$1"
    rm -rf "$wt"
    git worktree prune
    git worktree add -f --detach "$wt" "$2" >/dev/null 2>&1 || { echo "no rev $2"; exit 1; }
    # shellcheck disable=SC2086
    (cd "$wt" && git log -1 --oneline && make -j16 $MK BACKENDS="vulkan cpu_x86 cpu_scalar" bin) \
        >"$OUT/build_$1.log" 2>&1 || { echo "build $1 failed"; tail -30 "$OUT/build_$1.log"; exit 1; }
    echo "built $1: $(cd "$wt" && git log -1 --oneline)"
}
B() { echo "$RUNNER_TEMP/wt-$1/bin/linux/release/$2"; }

llama_tools() {
    LB=$(sh "$ROOT/scripts/ensure-llama-bench.sh" vulkan 2>"$OUT/llama-build.log") || { tail -20 "$OUT/llama-build.log"; exit 1; }
    LBIN=$(dirname "$LB")
    if [ ! -x "$LBIN/llama-perplexity" ] || [ ! -x "$LBIN/llama-tokenize" ]; then
        cmake --build "$(dirname "$LBIN")" -j16 --target llama-perplexity llama-tokenize >>"$OUT/llama-build.log" 2>&1 ||
            { tail -20 "$OUT/llama-build.log"; exit 1; }
    fi
    echo "llama.cpp: $LBIN ($(git -C "$LLAMA_DIR" rev-parse --short HEAD))"
}

uptime
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader 2>&1
build new "$NEW"
build base "$BASE"

for mode in "$@"; do
    arg=${mode#*=}
    case $mode in
    unit)
        for lab in base new newx; do
            l=${lab%x}; x=0; [ "$lab" = newx ] && x=1
            for t in test_backend_vulkan_ops_unit test_deepseek2_unit; do
                echo "=== $lab $t"
                GEIST_VK_MLA_EXACT=$x "$(B "$l" "tests/$t")" 2>&1 | tee "$OUT/unit_${lab}_$t.txt" | grep -i "MLA\|x576\|x520\|FAIL\|PASS\|SKIP"
            done
        done
        ;;
    prof)
        for f in "$GIGA" "$V2_S"; do
            for lab in base new newx; do
                l=${lab%x}; x=0; [ "$lab" = newx ] && x=1
                quiet_gate "prof $lab $(basename "$f")"
                echo "### prof $lab $(basename "$f") prefill 512" | tee -a "$OUT/summary.md"
                GEIST_VK_MLA_EXACT=$x GEIST_VK_PROFILE=1 GEIST_BENCH_BACKEND=vulkan "$(B "$l" tests/bench_perf_sweep)" --gguf "$f" \
                    --seq-lens 512,2048 --decode-n 1 --warmup 1 --repeats 1 >"$OUT/p.jsonl" 2>"$OUT/p_${lab}.txt"
                { echo '```'; grep -v "^geist vulkan: device memory" "$OUT/p_${lab}.txt" | head -40; echo '```'; } | tee -a "$OUT/summary.md"
            done
        done
        ;;
    ab | ab=*)
        [ "$arg" = ab ] && arg=6
        for m in giga v2s v2 llama gigac; do
            case $m in
            giga) f=$GIGA ;; v2s) f=$V2_S ;; v2) f=$V2 ;; llama) f=$LLAMA3 ;; gigac) f=$GIGA_C ;;
            esac
            cat "$f" >/dev/null # page cache
            quiet_gate "ab $m"
            # shellcheck disable=SC2086
            python3 "$ROOT/tools/bench_revision_ab.py" --rev "base=$BASE" --rev "new=$NEW" \
                --gguf "$f" --seq-lens 512,2048 --decode-n 64 --repeats 1 --warmup 1 --cycles "$arg" \
                --env GEIST_BENCH_BACKEND=vulkan --make-arg TARGET=linux \
                --make-arg "BACKENDS=vulkan cpu_x86 cpu_scalar" --make-arg GEMM_PROVIDER=native \
                --jobs 16 --out-dir "$OUT/ab-$m" >"$OUT/ab-$m.txt" 2>&1
            { echo "### A/B $m ($(basename "$f"))"; echo '```'; tail -40 "$OUT/ab-$m.txt"; echo '```'; } | tee -a "$OUT/summary.md"
        done
        ;;
    vsllama | vsllama=*)
        [ "$arg" = vsllama ] && arg=4
        llama_tools
        for m in giga v2s; do
            case $m in giga) f=$GIGA; ngl="-ngl 99" ;; v2s) f=$V2_S; ngl="-ngl 99 --n-cpu-moe 4" ;; esac
            cat "$f" >/dev/null
            c=1
            while [ "$c" -le "$arg" ]; do
                for lab in base new; do
                    quiet_gate "geist $lab $m $c"
                    GEIST_BENCH_BACKEND=vulkan "$(B "$lab" tests/bench_perf_sweep)" --gguf "$f" --seq-lens 512,2048 \
                        --decode-n 64 --warmup 1 --repeats 1 >"$OUT/geist_${lab}_${m}_$c.jsonl" 2>"$OUT/geist_${lab}_${m}_$c.err"
                done
                quiet_gate "llama $m $c"
                # shellcheck disable=SC2086
                GGML_VK_VISIBLE_DEVICES=0 "$LB" -m "$f" $ngl -p 512,2048 -n 0 -r 1 -o jsonl \
                    >"$OUT/llama_${m}_$c.jsonl" 2>"$OUT/llama_${m}_$c.err"
                c=$((c + 1))
            done
            python3 - "$OUT" "$m" <<'PY' | tee -a "$OUT/summary.md"
import glob, json, statistics as S, sys
out, m = sys.argv[1], sys.argv[2]
def geist(lab):
    g = {}
    for f in sorted(glob.glob(f"{out}/geist_{lab}_{m}_*.jsonl")):
        for line in open(f):
            if line.startswith("{"):
                d = json.loads(line)
                n = d.get("seq_len") or d.get("prompt_tokens")
                g.setdefault(("pp", n), []).append(d.get("prefill_tps"))
                g.setdefault(("tg", n), []).append(d.get("decode_tps"))
    return {k: S.median([x for x in v if x is not None]) for k, v in g.items() if any(x is not None for x in v)}
l = {}
for f in sorted(glob.glob(f"{out}/llama_{m}_*.jsonl")):
    for line in open(f):
        if line.startswith("{"):
            d = json.loads(line)
            if d["n_prompt"]:
                l.setdefault(("pp", d["n_prompt"]), []).append(d["avg_ts"])
l = {k: S.median(v) for k, v in l.items()}
gb, gn = geist("base"), geist("new")
print(f"### {m}: geist base / new vs llama.cpp Vulkan (tok/s, median of cycles)\n")
print("| what | geist base | geist new | llama.cpp | base/llama | new/llama |\n|---|---|---|---|---|---|")
for k in sorted(set(gb) | set(gn), key=str):
    lm = l.get(k, float("nan"))
    b, n = gb.get(k, float("nan")), gn.get(k, float("nan"))
    print(f"| {k[0]} @{k[1]} | {b:.1f} | {n:.1f} | {lm:.1f} | {b / lm:.2f} | {n / lm:.2f} |")
print()
PY
        done
        ;;
    parity | parity=*)
        [ "$arg" = parity ] && arg=8
        llama_tools
        TEXT=$OUT/offload.txt
        git show b98a5c4b:docs/OFFLOAD.md >"$TEXT"
        echo "| model | engine | ppl | vs llama.cpp Vulkan | tokens agree |" >>"$OUT/summary.md"
        echo "|---|---|---|---|---|" >>"$OUT/summary.md"
        for model in "$GIGA" "$V2_S"; do
            case $model in *V2-Lite*) ngl="-ngl 99 --n-cpu-moe 4" ;; *) ngl="-ngl 99" ;; esac
            # shellcheck disable=SC2086
            GGML_VK_VISIBLE_DEVICES=0 "$LBIN/llama-perplexity" -m "$model" -f "$TEXT" -c 512 --chunks "$arg" $ngl \
                >"$OUT/llama_$(basename "$model" .gguf).log" 2>&1
            LPPL=$(grep -o 'Final estimate: PPL = [0-9.]*' "$OUT/llama_$(basename "$model" .gguf).log" | grep -o '[0-9.]*$')
            echo "| $(basename "$model") | llama.cpp Vulkan ($ngl) | $LPPL | | |" | tee -a "$OUT/summary.md"
            [ -n "$LPPL" ] || LPPL=1
            for lab in new base; do
                name="$(basename "$model" .gguf)_$lab"
                GEIST_BACKEND=vulkan python3 "$ROOT/tools/ppl_vs_llama.py" "$model" "$TEXT" --llama-bin "$LBIN" \
                    --eval-geist "$(B "$lab" tools/eval_geist)" --chunks "$arg" --skip-llama "$LPPL" \
                    >"$OUT/$name.json" 2>"$OUT/$name.err"
                echo "| $(basename "$model") | geist $lab vulkan | $(python3 -c "import json; d=json.load(open('$OUT/$name.json')); print(d['geist_ppl'], '|', f\"{d['rel_diff']*100:+.2f} %\", '|', d['tokenizer_agrees'])" 2>/dev/null || echo "FAILED: $(tail -3 "$OUT/$name.err" | tr '\n' ' ')") |" | tee -a "$OUT/summary.md"
            done
        done
        ;;
    *) echo "unknown mode $mode" ;;
    esac
done
cat "$OUT/summary.md" 2>/dev/null
