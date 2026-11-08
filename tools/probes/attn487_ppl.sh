#!/bin/sh
# attn487_ppl.sh MODEL NEW [WHICH [CTX [CHUNKS [BASE]]]] — #487 on the M6:
# tools/ppl_vs_llama.py on docs/OFFLOAD.md, CTX x CHUNKS (default 512 x
# 24), geist NEW with GEIST_METAL_TENSOR=1 and =0 and BASE (default
# origin/main), against llama.cpp Metal. WHICH: gc (GigaChat3-10B), go
# (gpt-oss-20b), ds (DeepSeek-V2-Lite) or model (mac-probe's MODEL). Run
# by .gitea/workflows/mac-probe.yml under the host lock; writes to $PROBE_OUT.
set -u
MODEL=$1
shift
NEW=${1:?NEW revision}
WHICH=${2:-gc}
CTX=${3:-512}
CHUNKS=${4:-24}
BASE=${5:-origin/main}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
git fetch --quiet origin "+refs/heads/${NEW#origin/}:refs/remotes/$NEW" \
    "+refs/heads/${BASE#origin/}:refs/remotes/$BASE"
TMP=$(mktemp -d)
for r in new base; do
    if [ $r = new ]; then REV=$NEW; else REV=$BASE; fi
    git worktree add --detach "$TMP/$r" "$REV" >>"$OUT/worktree.log" 2>&1
    (cd "$TMP/$r" && make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" bin) >"$OUT/build_$r.log" 2>&1 || exit 1
done
fetch() { python3 tools/fetch_bench_model.py --url "$1" --sha256 "$2"; }
case $WHICH in
go) M=$(fetch https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/15dcfc7e7c608c6161507211ed43c5b590801b4b/gpt-oss-20b-MXFP4.gguf \
    27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901) ;;
gc) M=$(fetch https://huggingface.co/bartowski/ai-sage_GigaChat3-10B-A1.8B-GGUF/resolve/5f329324103300ba9aff1e2fc68ddbb8a682a2c0/ai-sage_GigaChat3-10B-A1.8B-Q4_K_M.gguf \
    88505918a1af4e2174994e406ae5a6a06c5867227796a4370c18321c6481e9a1) ;;
ds) M=$(fetch https://huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF/resolve/0f37fdf276e8094747457f0ae4d40f2e8d2521f9/DeepSeek-V2-Lite.Q4_K_M.gguf \
    70985c39ba3e51a53e6ad6ae25ce29a3efc92b4802f61645760e1bc1eb89ad1a) ;;
*) M=$MODEL ;;
esac
LB=$(scripts/ensure-llama-bench.sh metal 2>"$OUT/llama-build.log") || exit 1
LBD=$(dirname "$LB")
cmake --build "$(dirname "$LBD")" --target llama-perplexity llama-tokenize -j 8 >>"$OUT/llama-build.log" 2>&1
mkdir -p "$OUT/pt"
cp tools/eval_repl.py "$OUT/pt/"
python3 - tools/ppl_vs_llama.py "$OUT/pt/ppl_vs_llama.py" <<'PYEOF'
import ast, sys
src = open(sys.argv[1]).read()
doc = ast.get_docstring(ast.parse(src), clean=False)
end = src.index('"""', src.index('"""') + 3) + 3 if doc is not None else 0
open(sys.argv[2], "w").write(src[:end] + "\nfrom __future__ import annotations\n" + src[end:])
PYEOF
LP=""
echo "## #487 ppl $(basename "$M") $CTX x $CHUNKS" >"$OUT/summary.md"
for v in new:1 new:0 base:1; do
    r=${v%:*}
    t=${v#*:}
    GEIST_METAL_TENSOR=$t GEIST_BACKEND=metal python3 "$OUT/pt/ppl_vs_llama.py" "$M" docs/OFFLOAD.md \
        --llama-bin "$LBD" --ngl 99 --ctx "$CTX" --chunks "$CHUNKS" --eval-geist "$TMP/$r/bin/mac-omp/release/tools/eval_geist" \
        ${LP:+--skip-llama "$LP"} >"$OUT/ppl_${r}_tc$t.json" 2>"$OUT/ppl_${r}_tc$t.err"
    LP=$(python3 -c 'import json,sys;print(json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])["llama_ppl"])' "$OUT/ppl_${r}_tc$t.json" 2>/dev/null || echo "")
    echo "- $r tensor=$t: $(tail -1 "$OUT/ppl_${r}_tc$t.json")" >>"$OUT/summary.md"
done
git worktree remove --force "$TMP/new" >/dev/null 2>&1 || true
git worktree remove --force "$TMP/base" >/dev/null 2>&1 || true
