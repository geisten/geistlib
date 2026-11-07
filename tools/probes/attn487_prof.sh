#!/bin/sh
# attn487_prof.sh MODEL NEW [SEQS [WHICH]] — #487 on the M6: where a long
# Metal prefill of NEW spends its GPU time, against llama.cpp Metal.
# For each model of WHICH (comma list of go = gpt-oss-20b MXFP4, gc =
# GigaChat3-10B Q4_K_M, ds = DeepSeek-V2-Lite Q4_K_M; default all) and each
# prompt length of SEQS (default 2048,4096):
#   - GEIST_METAL_PROFILE=1 with GEIST_METAL_PIPELINE_PREFILL=1 (one op per
#     command buffer: the shares per kernel count, not the absolute times);
#   - a default run (the total);
#   - llama-bench -p SEQS -n 0 -fa 0,1 -ngl 99 (the pinned build).
# Plus bench_metal_flash_attn (NEW's attention microbench) and llama.cpp's
# test-backend-ops perf for FLASH_ATTN_EXT, best effort. MODEL
# (mac-probe.yml's) is unused. Writes to $PROBE_OUT.
set -u
shift
NEW=${1:?NEW revision}
SEQS=${2:-2048,4096}
WHICH=${3:-go,gc,ds}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
has() { case ",$WHICH," in *",$1,"*) return 0 ;; esac; return 1; }
{
    sw_vers
    sysctl -n machdep.cpu.brand_string
    LC_ALL=C sysctl -n vm.loadavg
} >"$OUT/env.txt" 2>&1
git fetch --quiet origin "+refs/heads/${NEW#origin/}:refs/remotes/$NEW"
WT="$(mktemp -d)/new"
git worktree add --detach "$WT" "$NEW" >"$OUT/worktree.log" 2>&1
(cd "$WT" && make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" bin) >"$OUT/build.log" 2>&1 || {
    echo "build of $NEW failed" >"$OUT/summary.md"
    exit 1
}
G="$WT/bin/mac-omp/release/tests/bench_perf_sweep"
fetch() { python3 tools/fetch_bench_model.py --url "$1" --sha256 "$2"; }
MODELS=""
has go && MODELS="$MODELS $(fetch https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/15dcfc7e7c608c6161507211ed43c5b590801b4b/gpt-oss-20b-MXFP4.gguf \
    27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901)"
has gc && MODELS="$MODELS $(fetch https://huggingface.co/bartowski/ai-sage_GigaChat3-10B-A1.8B-GGUF/resolve/5f329324103300ba9aff1e2fc68ddbb8a682a2c0/ai-sage_GigaChat3-10B-A1.8B-Q4_K_M.gguf \
    88505918a1af4e2174994e406ae5a6a06c5867227796a4370c18321c6481e9a1)"
has ds && MODELS="$MODELS $(fetch https://huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF/resolve/0f37fdf276e8094747457f0ae4d40f2e8d2521f9/DeepSeek-V2-Lite.Q4_K_M.gguf \
    70985c39ba3e51a53e6ad6ae25ce29a3efc92b4802f61645760e1bc1eb89ad1a)"
LB=$(scripts/ensure-llama-bench.sh metal 2>"$OUT/llama-build.log") || LB=""
SEQ_SP=$(echo "$SEQS" | tr ',' ' ')
echo "## #487 prefill profile: $(git -C "$WT" rev-parse HEAD)" >"$OUT/summary.md"
for m in $MODELS; do
    n=$(basename "$m" .gguf)
    for seq in $SEQ_SP; do
        sleep 20
        GEIST_BENCH_BACKEND=metal GEIST_METAL_PROFILE=1 GEIST_METAL_PIPELINE_PREFILL=1 "$G" \
            --gguf "$m" --seq-lens "$seq" --decode-n 1 --warmup 1 --repeats 2 \
            >"$OUT/prof_${n}_${seq}_rot1.txt" 2>&1 || true
        sleep 20
        GEIST_BENCH_BACKEND=metal "$G" --gguf "$m" --seq-lens "$seq" --decode-n 16 --warmup 2 \
            --repeats 3 --emit-jsonl >"$OUT/total_${n}_${seq}.jsonl" 2>"$OUT/total_${n}_${seq}.err" || true
    done
    if [ -n "$LB" ]; then
        sleep 20
        "$LB" -m "$m" -p "$SEQS" -n 16 -fa 0,1 -r 3 -ngl 99 >"$OUT/llama_${n}.md" 2>"$OUT/llama_${n}.err" || true
    fi
done
sleep 10
"$WT/bin/mac-omp/release/tests/bench_metal_flash_attn" 4 5 f16 2048 4096 >"$OUT/flash_bench.txt" 2>&1 || true
if [ -n "$LB" ]; then
    LBD=$(dirname "$(dirname "$LB")")
    cmake --build "$LBD" --target test-backend-ops -j 8 >>"$OUT/llama-build.log" 2>&1 &&
        "$LBD/bin/test-backend-ops" perf -o FLASH_ATTN_EXT >"$OUT/llama_fa_ops.txt" 2>&1 || true
fi
{
    for f in "$OUT"/total_*.jsonl; do
        echo
        echo "### $(basename "$f")"
        echo '```'
        grep -h '^{' "$f" | cut -c1-400
        echo '```'
    done
    for f in "$OUT"/llama_*.md; do
        echo
        echo "### $(basename "$f")"
        cat "$f"
    done
    for f in "$OUT"/prof_*.txt "$OUT/flash_bench.txt"; do
        echo
        echo "### $(basename "$f")"
        echo '```'
        grep -vE '^[[:space:]]*$' "$f" | grep -v '^{' | cut -c1-200 | head -80
        echo '```'
    done
    echo
    echo "### llama.cpp FLASH_ATTN_EXT (hsk 64 / 576)"
    echo '```'
    grep -E 'hsk=(64|576)' "$OUT/llama_fa_ops.txt" 2>/dev/null | grep -E 'kv=(512|1024|4096|8192)' | head -60
    echo '```'
} >>"$OUT/summary.md"
git worktree remove --force "$WT" >/dev/null 2>&1 || true
