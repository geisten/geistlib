#!/bin/sh
# moe482_m6.sh MODEL NEW [ROUNDS [WHAT]] — #482 on the M6 with NEW built
# once (Metal): WHAT is a comma list of
#   units  the MoE / Metal unit tests with GEIST_METAL_TENSOR=0 and =1;
#   parity next-token logits and 32 greedy steps after a 512-token prompt
#          (head of docs/OFFLOAD.md), device-routed prefill against the host
#          route (GEIST_MOE_DEVICE_PREFILL=0), and greedy against llama.cpp
#          Metal (benchmark/dump_llamacpp_greedy.c) on the same ids;
#   ppl    tools/ppl_vs_llama.py (8 x 512, docs/OFFLOAD.md) against
#          llama.cpp Metal, geist with and without the device prefill;
#   speed  geist (bench_perf_sweep) against llama-bench, pp128 / pp512 /
#          pp2048 and tg128, ROUNDS (default 3) alternating rounds.
# Models: OLMoE-1B-7B, DeepSeek-V2-Lite, GigaChat3-10B (Q4_K_M) and
# gpt-oss-20b (MXFP4), sha256-gated into the host cache. MODEL (mac-probe's)
# is unused. Run by .gitea/workflows/mac-probe.yml under the host lock;
# writes to $PROBE_OUT.
set -u
shift
NEW=${1:?NEW revision, e.g. origin/agent/issue-482}
ROUNDS=${2:-3}
WHAT=${3:-units,parity,ppl,speed}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
has() { case ",$WHAT," in *",$1,"*) return 0 ;; esac; return 1; }
{
    sw_vers
    sysctl -n machdep.cpu.brand_string
    echo "memory: $(($(sysctl -n hw.memsize) / 1073741824)) GiB"
    LC_ALL=C sysctl -n vm.loadavg
} >"$OUT/env.txt" 2>&1
git fetch --quiet origin "+refs/heads/${NEW#origin/}:refs/remotes/$NEW"
WT="$(mktemp -d)/new"
git worktree add --detach "$WT" "$NEW" >"$OUT/worktree.log" 2>&1
git -C "$WT" rev-parse HEAD >"$OUT/new_sha.txt"
(cd "$WT" && make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" bin) >"$OUT/build.log" 2>&1 || {
    echo "build of $NEW failed" >"$OUT/summary.md"
    exit 1
}
B="$WT/bin/mac-omp/release"
echo "## #482 on the M6: $NEW ($(cat "$OUT/new_sha.txt"))" >"$OUT/summary.md"
if has units; then
    {
        echo
        echo "| test | GEIST_METAL_TENSOR | exit |"
        echo "|---|--:|--:|"
    } >>"$OUT/summary.md"
    for t in test_moe_grouped_unit test_moe_device_unit test_deepseek2_unit test_gpt_oss_unit \
        test_qwen3moe_unit test_olmoe_unit test_backend_metal_parity_unit; do
        for v in 0 1; do
            GEIST_METAL_TENSOR=$v "$B/tests/$t" >"$OUT/${t}_tc$v.log" 2>&1
            echo "| $t | $v | $? |" >>"$OUT/summary.md"
        done
    done
fi
fetch() { python3 tools/fetch_bench_model.py --url "$1" --sha256 "$2"; }
OL=$(fetch https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct-GGUF/resolve/2ac1d2731792/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf \
    4ddc0e53159ed512b8dd67914a66e27bc618f694672ba43a9a0454eabd9c684f)
DS=$(fetch https://huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF/resolve/0f37fdf276e8094747457f0ae4d40f2e8d2521f9/DeepSeek-V2-Lite.Q4_K_M.gguf \
    70985c39ba3e51a53e6ad6ae25ce29a3efc92b4802f61645760e1bc1eb89ad1a)
GC=$(fetch https://huggingface.co/bartowski/ai-sage_GigaChat3-10B-A1.8B-GGUF/resolve/5f329324103300ba9aff1e2fc68ddbb8a682a2c0/ai-sage_GigaChat3-10B-A1.8B-Q4_K_M.gguf \
    88505918a1af4e2174994e406ae5a6a06c5867227796a4370c18321c6481e9a1)
GO=$(fetch https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/15dcfc7e7c608c6161507211ed43c5b590801b4b/gpt-oss-20b-MXFP4.gguf \
    27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901)
LB=$(scripts/ensure-llama-bench.sh metal 2>"$OUT/llama-build.log") || {
    echo "llama-bench build failed" >>"$OUT/summary.md"
    exit 1
}
LBD=$(dirname "$LB")
LROOT=$(dirname "$(dirname "$LBD")")
cmake --build "$(dirname "$LBD")" --target llama-perplexity llama-tokenize llama \
    -j 8 >>"$OUT/llama-build.log" 2>&1
cc -std=c11 -O2 -I"$LROOT/include" -I"$LROOT/ggml/include" -o "$OUT/dump_llamacpp_greedy" \
    benchmark/dump_llamacpp_greedy.c -L"$LBD" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$LBD" >>"$OUT/llama-build.log" 2>&1
head -c 2400 docs/OFFLOAD.md >"$OUT/prompt.txt"
if has parity; then
    {
        echo
        echo "### Device prefill against the host route, and greedy against llama.cpp Metal"
        echo
    } >>"$OUT/summary.md"
    for m in "$OL" "$DS" "$GC" "$GO"; do
        n=$(basename "$m" .gguf)
        for f in 0 1; do
            GEIST_DUMP_BACKEND=metal GEIST_MOE_DEVICE_PREFILL=$f "$B/tools/dump_geist_logits" "$m" \
                "$OUT/prompt.txt" "$OUT/$n.ids" "$OUT/${n}_logits_$f.bin" >"$OUT/${n}_logits_$f.log" 2>&1
            GEIST_DUMP_BACKEND=metal GEIST_MOE_DEVICE_PREFILL=$f "$B/tools/dump_geist_greedy" "$m" \
                "$OUT/$n.ids" 32 >"$OUT/${n}_greedy_$f.txt" 2>"$OUT/${n}_greedy_$f.err"
        done
        "$OUT/dump_llamacpp_greedy" "$m" "$OUT/$n.ids" 32 >"$OUT/${n}_greedy_llama.txt" \
            2>"$OUT/${n}_greedy_llama.err"
        python3 - "$OUT" "$n" >>"$OUT/summary.md" <<'EOF'
import json, subprocess, sys
out, n = sys.argv[1], sys.argv[2]
cmp = subprocess.run([sys.executable, "tools/compare_logits.py", f"{out}/{n}_logits_0.bin",
                      f"{out}/{n}_logits_1.bin"], capture_output=True, text=True).stdout.strip()
def toks(p):
    try:
        return [l.split()[1] for l in open(p) if l.strip() and l.split()[0].isdigit()]
    except OSError:
        return []
def same(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k
h, d, l = toks(f"{out}/{n}_greedy_0.txt"), toks(f"{out}/{n}_greedy_1.txt"), toks(f"{out}/{n}_greedy_llama.txt")
print(f"- **{n}** logits device vs host: `{cmp}`")
print(f"  greedy identical steps: device vs host {same(d, h)}/{len(d)}, device vs llama.cpp {same(d, l)}/{len(l)}, host vs llama.cpp {same(h, l)}/{len(l)}")
EOF
    done
fi
if has ppl; then
    {
        echo
        echo "| model | geist device prefill | geist host route | llama.cpp Metal |"
        echo "|---|--:|--:|--:|"
    } >>"$OUT/summary.md"
    for m in "$OL" "$DS" "$GC" "$GO"; do
        n=$(basename "$m" .gguf)
        GEIST_BACKEND=metal python3 tools/ppl_vs_llama.py "$m" docs/OFFLOAD.md --llama-bin "$LBD" \
            --ngl 99 --eval-geist "$B/tools/eval_geist" >"$OUT/${n}_ppl_1.json" 2>"$OUT/${n}_ppl_1.err"
        LP=$(python3 -c 'import json,sys;print(json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])["llama_ppl"])' "$OUT/${n}_ppl_1.json" 2>/dev/null || echo "")
        GEIST_MOE_DEVICE_PREFILL=0 GEIST_BACKEND=metal python3 tools/ppl_vs_llama.py "$m" docs/OFFLOAD.md \
            --llama-bin "$LBD" --ngl 99 --eval-geist "$B/tools/eval_geist" ${LP:+--skip-llama "$LP"} \
            >"$OUT/${n}_ppl_0.json" 2>"$OUT/${n}_ppl_0.err"
        python3 - "$OUT" "$n" >>"$OUT/summary.md" <<'EOF'
import json, sys
out, n = sys.argv[1], sys.argv[2]
def get(f):
    try:
        return json.loads(open(f"{out}/{n}_ppl_{f}.json").read().strip().splitlines()[-1])
    except Exception:
        return {}
a, b = get(1), get(0)
print(f"| {n} | {a.get('geist_ppl', '?')} | {b.get('geist_ppl', '?')} | {a.get('llama_ppl', '?')} |")
EOF
    done
fi
if has speed; then
    {
        echo
        echo "| model | round | engine | pp128 | pp512 | pp2048 | tg128 |"
        echo "|---|--:|---|--:|--:|--:|--:|"
    } >>"$OUT/summary.md"
    for m in "$OL" "$DS" "$GC" "$GO"; do
        n=$(basename "$m" .gguf)
        i=1
        while [ "$i" -le "$ROUNDS" ]; do
            sleep 20
            "$LB" -m "$m" -p 128,512,2048 -n 128 -r 3 -ngl 99 -o json >"$OUT/llama_${n}_$i.json" \
                2>"$OUT/llama_${n}_$i.err"
            python3 - "$OUT/llama_${n}_$i.json" "$n" "$i" >>"$OUT/summary.md" <<'EOF'
import json, sys
rows = json.load(open(sys.argv[1]))
pp = {r["n_prompt"]: r["avg_ts"] for r in rows if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0}
tg = next((r["avg_ts"] for r in rows if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0), 0.0)
print(f"| {sys.argv[2]} | {sys.argv[3]} | llama.cpp | {pp.get(128, 0):.1f} | {pp.get(512, 0):.1f} | {pp.get(2048, 0):.1f} | {tg:.1f} |")
EOF
            sleep 20
            GEIST_BENCH_BACKEND=metal "$B/tests/bench_perf_sweep" --gguf "$m" \
                --seq-lens 1,128,512,2048 --decode-n 128 --warmup 16 --repeats 3 --emit-jsonl \
                >"$OUT/geist_${n}_$i.jsonl" 2>"$OUT/geist_${n}_$i.err"
            python3 - "$OUT/geist_${n}_$i.jsonl" "$n" "$i" >>"$OUT/summary.md" <<'EOF'
import json, sys
rows = {r.get("seq_len"): r for r in (json.loads(l) for l in open(sys.argv[1]) if l.startswith("{"))}
pp = lambda s: rows.get(s, {}).get("prefill_tps", 0.0)
tg = rows.get(1, {}).get("decode_tps", 0.0)
print(f"| {sys.argv[2]} | {sys.argv[3]} | geist | {pp(128):.1f} | {pp(512):.1f} | {pp(2048):.1f} | {tg:.1f} |")
EOF
            i=$((i + 1))
        done
    done
fi
git worktree remove --force "$WT" >/dev/null 2>&1 || true
