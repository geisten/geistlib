#!/bin/sh
# attn487_m6.sh MODEL NEW [WHAT [WHICH [BASE [ROUNDS]]]] — #487 on the M6,
# NEW and BASE (default origin/main) built once each (Metal). WHAT is a
# comma list of
#   units  the attention / Metal unit tests of NEW, GEIST_METAL_TENSOR=0 and 1;
#   parity next-token logits after a ~512- and a ~2300-token prompt (heads of
#          docs/OFFLOAD.md) NEW against BASE, and 32 greedy steps of NEW,
#          BASE and llama.cpp Metal (dump_llamacpp_greedy) on the same ids;
#   ppl    tools/ppl_vs_llama.py, 8 x 512 and 4 x 2048, NEW, BASE, llama.cpp;
#   speed  NEW (bench_perf_sweep) against llama-bench -fa 1, pp512 / pp2048 /
#          pp4096 and tg128, ROUNDS (default 2) alternating rounds.
# WHICH: comma list of go (gpt-oss-20b), gc (GigaChat3-10B), ds
# (DeepSeek-V2-Lite), q3 (mac-probe's MODEL, e.g. qwen3-0.6b-q8). Run by
# .gitea/workflows/mac-probe.yml under the host lock; writes to $PROBE_OUT.
set -u
MODEL=$1
shift
NEW=${1:?NEW revision}
WHAT=${2:-units,parity,ppl,speed}
WHICH=${3:-go,gc,ds,q3}
BASE=${4:-origin/main}
ROUNDS=${5:-2}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
has() { case ",$WHAT," in *",$1,"*) return 0 ;; esac; return 1; }
hasm() { case ",$WHICH," in *",$1,"*) return 0 ;; esac; return 1; }
{
    sw_vers
    sysctl -n machdep.cpu.brand_string
    LC_ALL=C sysctl -n vm.loadavg
} >"$OUT/env.txt" 2>&1
git fetch --quiet origin "+refs/heads/${NEW#origin/}:refs/remotes/$NEW" \
    "+refs/heads/${BASE#origin/}:refs/remotes/$BASE"
TMP=$(mktemp -d)
for r in new base; do
    if [ $r = new ]; then REV=$NEW; else REV=$BASE; fi
    git worktree add --detach "$TMP/$r" "$REV" >>"$OUT/worktree.log" 2>&1
    git -C "$TMP/$r" rev-parse HEAD >"$OUT/${r}_sha.txt"
    (cd "$TMP/$r" && make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" bin) >"$OUT/build_$r.log" 2>&1 || {
        echo "build of $REV failed" >"$OUT/summary.md"
        exit 1
    }
done
BN="$TMP/new/bin/mac-omp/release"
BB="$TMP/base/bin/mac-omp/release"
echo "## #487 on the M6: new $(cat "$OUT/new_sha.txt"), base $(cat "$OUT/base_sha.txt")" >"$OUT/summary.md"
if has units; then
    {
        echo
        echo "| test | GEIST_METAL_TENSOR | exit |"
        echo "|---|--:|--:|"
    } >>"$OUT/summary.md"
    for t in test_gpt_oss_ops_unit test_backend_metal_parity_unit test_gpt_oss_unit test_deepseek2_unit \
        test_attention_f32_unit test_moe_prefill_unit; do
        for v in 0 1; do
            GEIST_METAL_TENSOR=$v "$BN/tests/$t" >"$OUT/${t}_tc$v.log" 2>&1
            echo "| $t | $v | $? |" >>"$OUT/summary.md"
        done
    done
fi
fetch() { python3 tools/fetch_bench_model.py --url "$1" --sha256 "$2"; }
MODELS=""
hasm go && MODELS="$MODELS $(fetch https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/15dcfc7e7c608c6161507211ed43c5b590801b4b/gpt-oss-20b-MXFP4.gguf \
    27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901)"
hasm gc && MODELS="$MODELS $(fetch https://huggingface.co/bartowski/ai-sage_GigaChat3-10B-A1.8B-GGUF/resolve/5f329324103300ba9aff1e2fc68ddbb8a682a2c0/ai-sage_GigaChat3-10B-A1.8B-Q4_K_M.gguf \
    88505918a1af4e2174994e406ae5a6a06c5867227796a4370c18321c6481e9a1)"
hasm ds && MODELS="$MODELS $(fetch https://huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF/resolve/0f37fdf276e8094747457f0ae4d40f2e8d2521f9/DeepSeek-V2-Lite.Q4_K_M.gguf \
    70985c39ba3e51a53e6ad6ae25ce29a3efc92b4802f61645760e1bc1eb89ad1a)"
hasm q3 && MODELS="$MODELS $MODEL"
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
head -c 2400 docs/OFFLOAD.md >"$OUT/prompt_s.txt"
head -c 10000 docs/OFFLOAD.md >"$OUT/prompt_l.txt"
if has parity; then
    {
        echo
        echo "### Logits NEW vs BASE, greedy NEW / BASE / llama.cpp Metal"
        echo
    } >>"$OUT/summary.md"
    for m in $MODELS; do
        n=$(basename "$m" .gguf)
        for p in s l; do
            for r in new base; do
                if [ $r = new ]; then B=$BN; else B=$BB; fi
                GEIST_DUMP_BACKEND=metal "$B/tools/dump_geist_logits" "$m" "$OUT/prompt_$p.txt" \
                    "$OUT/${n}_$p.ids" "$OUT/${n}_${p}_logits_$r.bin" >"$OUT/${n}_${p}_logits_$r.log" 2>&1
                GEIST_DUMP_BACKEND=metal "$B/tools/dump_geist_greedy" "$m" "$OUT/${n}_$p.ids" 32 \
                    >"$OUT/${n}_${p}_greedy_$r.txt" 2>"$OUT/${n}_${p}_greedy_$r.err"
            done
            "$OUT/dump_llamacpp_greedy" "$m" "$OUT/${n}_$p.ids" 32 >"$OUT/${n}_${p}_greedy_llama.txt" \
                2>"$OUT/${n}_${p}_greedy_llama.err"
            python3 - "$OUT" "$n" "$p" >>"$OUT/summary.md" <<'EOF'
import subprocess, sys
out, n, p = sys.argv[1:4]
cmp = subprocess.run([sys.executable, "tools/compare_logits.py", f"{out}/{n}_{p}_logits_base.bin",
                      f"{out}/{n}_{p}_logits_new.bin"], capture_output=True, text=True).stdout.strip()
def toks(f):
    try:
        return [l.split()[1] for l in open(f) if l.strip() and l.split()[0].isdigit()]
    except OSError:
        return []
def same(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k
try:
    nids = len(open(f"{out}/{n}_{p}.ids").read().split())
except OSError:
    nids = 0
nw, b, l = toks(f"{out}/{n}_{p}_greedy_new.txt"), toks(f"{out}/{n}_{p}_greedy_base.txt"), toks(f"{out}/{n}_{p}_greedy_llama.txt")
print(f"- **{n}** prompt {nids} tokens: logits new vs base `{cmp}`")
print(f"  greedy identical steps: new vs base {same(nw, b)}/{len(nw)}, new vs llama.cpp {same(nw, l)}/{len(l)}, base vs llama.cpp {same(b, l)}/{len(l)}")
EOF
        done
    done
fi
if has ppl; then
    mkdir -p "$OUT/pt"
    cp tools/eval_repl.py "$OUT/pt/"
    python3 - tools/ppl_vs_llama.py "$OUT/pt/ppl_vs_llama.py" <<'PYEOF'
import ast, sys
src = open(sys.argv[1]).read()
doc = ast.get_docstring(ast.parse(src), clean=False)
end = src.index('"""', src.index('"""') + 3) + 3 if doc is not None else 0
open(sys.argv[2], "w").write(src[:end] + "\nfrom __future__ import annotations\n" + src[end:])
PYEOF
    {
        echo
        echo "| model | ctx x chunks | geist NEW | geist BASE | NEW vs BASE | llama.cpp Metal |"
        echo "|---|---|--:|--:|--:|--:|"
    } >>"$OUT/summary.md"
    for m in $MODELS; do
        n=$(basename "$m" .gguf)
        for cc in 512:8 2048:4; do
            c=${cc%:*}
            k=${cc#*:}
            GEIST_BACKEND=metal python3 "$OUT/pt/ppl_vs_llama.py" "$m" docs/OFFLOAD.md --llama-bin "$LBD" \
                --ngl 99 --ctx "$c" --chunks "$k" --eval-geist "$BN/tools/eval_geist" \
                >"$OUT/${n}_ppl_${c}_new.json" 2>"$OUT/${n}_ppl_${c}_new.err"
            LP=$(python3 -c 'import json,sys;print(json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])["llama_ppl"])' "$OUT/${n}_ppl_${c}_new.json" 2>/dev/null || echo "")
            GEIST_BACKEND=metal python3 "$OUT/pt/ppl_vs_llama.py" "$m" docs/OFFLOAD.md --llama-bin "$LBD" \
                --ngl 99 --ctx "$c" --chunks "$k" --eval-geist "$BB/tools/eval_geist" ${LP:+--skip-llama "$LP"} \
                >"$OUT/${n}_ppl_${c}_base.json" 2>"$OUT/${n}_ppl_${c}_base.err"
            python3 - "$OUT" "$n" "$c" "$k" >>"$OUT/summary.md" <<'EOF'
import json, sys
out, n, c, k = sys.argv[1:5]
def get(r):
    try:
        return json.loads(open(f"{out}/{n}_ppl_{c}_{r}.json").read().strip().splitlines()[-1])
    except Exception:
        return {}
a, b = get("new"), get("base")
ga, gb = a.get("geist_ppl"), b.get("geist_ppl")
d = f"{(ga / gb - 1) * 100:+.3f} %" if ga and gb else "?"
print(f"| {n} | {c} x {k} | {ga} | {gb} | {d} | {a.get('llama_ppl', '?')} |")
EOF
        done
    done
fi
if has speed; then
    {
        echo
        echo "| model | round | engine | pp512 | pp2048 | pp4096 | tg128 |"
        echo "|---|--:|---|--:|--:|--:|--:|"
    } >>"$OUT/summary.md"
    for m in $MODELS; do
        n=$(basename "$m" .gguf)
        i=1
        while [ "$i" -le "$ROUNDS" ]; do
            sleep 20
            "$LB" -m "$m" -p 512,2048,4096 -n 128 -r 3 -ngl 99 -fa 1 -o json >"$OUT/llama_${n}_$i.json" \
                2>"$OUT/llama_${n}_$i.err"
            python3 - "$OUT/llama_${n}_$i.json" "$n" "$i" >>"$OUT/summary.md" <<'EOF'
import json, sys
try:
    rows = json.load(open(sys.argv[1]))
except Exception:
    rows = []
pp = {r["n_prompt"]: r["avg_ts"] for r in rows if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0}
tg = next((r["avg_ts"] for r in rows if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0), 0.0)
print(f"| {sys.argv[2]} | {sys.argv[3]} | llama.cpp | {pp.get(512, 0):.1f} | {pp.get(2048, 0):.1f} | {pp.get(4096, 0):.1f} | {tg:.1f} |")
EOF
            sleep 20
            GEIST_BENCH_BACKEND=metal "$BN/tests/bench_perf_sweep" --gguf "$m" \
                --seq-lens 1,512,2048,4096 --decode-n 128 --warmup 16 --repeats 3 --emit-jsonl \
                >"$OUT/geist_${n}_$i.jsonl" 2>"$OUT/geist_${n}_$i.err"
            python3 - "$OUT/geist_${n}_$i.jsonl" "$n" "$i" >>"$OUT/summary.md" <<'EOF'
import json, sys
rows = {r.get("seq_len"): r for r in (json.loads(l) for l in open(sys.argv[1]) if l.startswith("{"))}
pp = lambda s: rows.get(s, {}).get("prefill_tps", 0.0)
tg = rows.get(1, {}).get("decode_tps", 0.0)
print(f"| {sys.argv[2]} | {sys.argv[3]} | geist | {pp(512):.1f} | {pp(2048):.1f} | {pp(4096):.1f} | {tg:.1f} |")
EOF
            i=$((i + 1))
        done
    done
fi
git worktree remove --force "$TMP/new" >/dev/null 2>&1 || true
git worktree remove --force "$TMP/base" >/dev/null 2>&1 || true
