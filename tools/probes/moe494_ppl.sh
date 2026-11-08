#!/bin/sh
# moe494_ppl.sh MODEL NEW [WHICH [CHUNKS [BASE]]] — #494 on the M6:
# tools/ppl_vs_llama.py (ctx 512, docs/OFFLOAD.md) of NEW, of NEW with
# GEIST_M_MAX=256 (main's prefill chunk: isolates the chunk size from the
# kernels) and of BASE (default origin/main), plus llama.cpp Metal, for
# each chunk count of CHUNKS (comma list, default 8,24). WHICH: ol, ds, gc,
# go (default gc). MODEL (mac-probe's) is unused. Run by
# .gitea/workflows/mac-probe.yml under the host lock; writes to $PROBE_OUT.
set -u
shift
NEW=${1:?NEW revision}
WHICH=${2:-gc}
CHUNKS=${3:-8,24}
BASE=${4:-origin/main}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT/pt"
git fetch --quiet origin "+refs/heads/${NEW#origin/}:refs/remotes/$NEW" \
    "+refs/heads/${BASE#origin/}:refs/remotes/$BASE"
TMP=$(mktemp -d)
for r in new base; do
    if [ "$r" = new ]; then rev=$NEW; else rev=$BASE; fi
    git worktree add --detach "$TMP/$r" "$rev" >>"$OUT/worktree.log" 2>&1
    echo "$r $(git -C "$TMP/$r" rev-parse HEAD)" >>"$OUT/shas.txt"
    (cd "$TMP/$r" && make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" bin) \
        >"$OUT/build_$r.log" 2>&1 || {
        echo "build of $rev failed" >"$OUT/summary.md"
        exit 1
    }
done
fetch() { python3 tools/fetch_bench_model.py --url "$1" --sha256 "$2"; }
case $WHICH in
ol) M=$(fetch https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct-GGUF/resolve/2ac1d2731792/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf \
    4ddc0e53159ed512b8dd67914a66e27bc618f694672ba43a9a0454eabd9c684f) ;;
ds) M=$(fetch https://huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF/resolve/0f37fdf276e8094747457f0ae4d40f2e8d2521f9/DeepSeek-V2-Lite.Q4_K_M.gguf \
    70985c39ba3e51a53e6ad6ae25ce29a3efc92b4802f61645760e1bc1eb89ad1a) ;;
gc) M=$(fetch https://huggingface.co/bartowski/ai-sage_GigaChat3-10B-A1.8B-GGUF/resolve/5f329324103300ba9aff1e2fc68ddbb8a682a2c0/ai-sage_GigaChat3-10B-A1.8B-Q4_K_M.gguf \
    88505918a1af4e2174994e406ae5a6a06c5867227796a4370c18321c6481e9a1) ;;
go) M=$(fetch https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/15dcfc7e7c608c6161507211ed43c5b590801b4b/gpt-oss-20b-MXFP4.gguf \
    27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901) ;;
esac
LB=$(scripts/ensure-llama-bench.sh metal 2>"$OUT/llama-build.log")
LBD=$(dirname "$LB")
cmake --build "$(dirname "$LBD")" --target llama-perplexity -j 8 >>"$OUT/llama-build.log" 2>&1
# The M6's python3 is 3.9: postponed annotations for ppl_vs_llama.py.
cp tools/eval_repl.py "$OUT/pt/"
python3 - tools/ppl_vs_llama.py "$OUT/pt/ppl_vs_llama.py" <<'PYEOF'
import ast, sys
src = open(sys.argv[1]).read()
doc = ast.get_docstring(ast.parse(src), clean=False)
end = src.index('"""', src.index('"""') + 3) + 3 if doc is not None else 0
open(sys.argv[2], "w").write(src[:end] + "\nfrom __future__ import annotations\n" + src[end:])
PYEOF
{
    echo "## #494 ppl on the M6, $(basename "$M") (ctx 512, docs/OFFLOAD.md)"
    echo
    echo '```'
    cat "$OUT/shas.txt"
    echo '```'
    echo
    echo "| chunks | NEW | NEW, GEIST_M_MAX=256 | BASE | NEW vs BASE | NEW@256 vs BASE | llama.cpp |"
    echo "|--:|--:|--:|--:|--:|--:|--:|"
} >"$OUT/summary.md"
get() { python3 -c 'import json,sys;print(json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])[sys.argv[2]])' "$1" "$2" 2>/dev/null; }
for c in $(echo "$CHUNKS" | tr ',' ' '); do
    GEIST_BACKEND=metal python3 "$OUT/pt/ppl_vs_llama.py" "$M" docs/OFFLOAD.md --llama-bin "$LBD" --ngl 99 \
        --chunks "$c" --eval-geist "$TMP/new/bin/mac-omp/release/tools/eval_geist" >"$OUT/new_$c.json" 2>"$OUT/new_$c.err"
    LP=$(get "$OUT/new_$c.json" llama_ppl)
    GEIST_M_MAX=256 GEIST_BACKEND=metal python3 "$OUT/pt/ppl_vs_llama.py" "$M" docs/OFFLOAD.md --llama-bin "$LBD" \
        --ngl 99 --chunks "$c" --eval-geist "$TMP/new/bin/mac-omp/release/tools/eval_geist" ${LP:+--skip-llama "$LP"} \
        >"$OUT/n256_$c.json" 2>"$OUT/n256_$c.err"
    GEIST_BACKEND=metal python3 "$OUT/pt/ppl_vs_llama.py" "$M" docs/OFFLOAD.md --llama-bin "$LBD" --ngl 99 \
        --chunks "$c" --eval-geist "$TMP/base/bin/mac-omp/release/tools/eval_geist" ${LP:+--skip-llama "$LP"} \
        >"$OUT/base_$c.json" 2>"$OUT/base_$c.err"
    python3 - "$c" "$(get "$OUT/new_$c.json" geist_ppl)" "$(get "$OUT/n256_$c.json" geist_ppl)" \
        "$(get "$OUT/base_$c.json" geist_ppl)" "$LP" >>"$OUT/summary.md" <<'EOF'
import sys
c, n, n2, b, l = sys.argv[1:6]
def rel(a, b):
    try:
        return f"{(float(a) / float(b) - 1) * 100:+.3f} %"
    except ValueError:
        return "?"
print(f"| {c} | {n} | {n2} | {b} | {rel(n, b)} | {rel(n2, b)} | {l} |")
EOF
done
for r in new base; do git worktree remove --force "$TMP/$r" >/dev/null 2>&1 || true; done
