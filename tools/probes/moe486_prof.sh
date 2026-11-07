#!/bin/sh
# moe486_prof.sh MODEL NEW [SEQS [MODELS]] — #486 on the M6: GPU time per
# MoE prefill kernel of NEW (Metal). For each model (comma list of ol, ds,
# gc, go: OLMoE-1B-7B, DeepSeek-V2-Lite, GigaChat3-10B, gpt-oss-20b; default
# ol) and prompt length (comma list, default 512): a profile run with
# GEIST_METAL_PIPELINE_PREFILL=1 (one op per command buffer: the shares of
# the GPU time count, not the absolute sum) and a default run for the
# prefill total. NEW is a branch with the MoE ops in stages of their own
# (agent/issue-486-prof-*). MODEL (mac-probe.yml's) is unused. Run by
# .gitea/workflows/mac-probe.yml under the host lock; writes to $PROBE_OUT.
set -u
shift
NEW=${1:?NEW revision}
SEQS=${2:-512}
MODELS=${3:-ol}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
LC_ALL=C sysctl -n vm.loadavg >"$OUT/env.txt" 2>&1
git fetch --quiet origin "+refs/heads/${NEW#origin/}:refs/remotes/$NEW"
WT="$(mktemp -d)/new"
git worktree add --detach "$WT" "$NEW" >"$OUT/worktree.log" 2>&1
(cd "$WT" && make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" bin) >"$OUT/build.log" 2>&1 || {
    echo "build of $NEW failed" >"$OUT/summary.md"
    exit 1
}
G="$WT/bin/mac-omp/release/tests/bench_perf_sweep"
fetch() { python3 tools/fetch_bench_model.py --url "$1" --sha256 "$2"; }
path() {
    case "$1" in
    ol) fetch https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct-GGUF/resolve/2ac1d2731792/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf \
        4ddc0e53159ed512b8dd67914a66e27bc618f694672ba43a9a0454eabd9c684f ;;
    ds) fetch https://huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF/resolve/0f37fdf276e8094747457f0ae4d40f2e8d2521f9/DeepSeek-V2-Lite.Q4_K_M.gguf \
        70985c39ba3e51a53e6ad6ae25ce29a3efc92b4802f61645760e1bc1eb89ad1a ;;
    gc) fetch https://huggingface.co/bartowski/ai-sage_GigaChat3-10B-A1.8B-GGUF/resolve/5f329324103300ba9aff1e2fc68ddbb8a682a2c0/ai-sage_GigaChat3-10B-A1.8B-Q4_K_M.gguf \
        88505918a1af4e2174994e406ae5a6a06c5867227796a4370c18321c6481e9a1 ;;
    go) fetch https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/15dcfc7e7c608c6161507211ed43c5b590801b4b/gpt-oss-20b-MXFP4.gguf \
        27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901 ;;
    esac
}
echo "## #486 kernel profile: $NEW ($(git -C "$WT" rev-parse --short HEAD))" >"$OUT/summary.md"
for key in $(echo "$MODELS" | tr ',' ' '); do
    m=$(path "$key")
    for seq in $(echo "$SEQS" | tr ',' ' '); do
        sleep 15
        GEIST_BENCH_BACKEND=metal "$G" --gguf "$m" --seq-lens "$seq" --decode-n 1 --warmup 2 \
            --repeats 5 >"$OUT/total_${key}_$seq.txt" 2>&1 || true
        sleep 15
        GEIST_BENCH_BACKEND=metal GEIST_METAL_PROFILE=1 GEIST_METAL_PIPELINE_PREFILL=1 "$G" \
            --gguf "$m" --seq-lens "$seq" --decode-n 1 --warmup 2 --repeats 3 \
            >"$OUT/prof_${key}_$seq.txt" 2>&1 || true
        {
            echo
            echo "### $(basename "$m") @ $seq"
            echo
            echo '```'
            grep -o '"prefill_ms":[0-9.]*' "$OUT/total_${key}_$seq.txt" | head -1
            grep -E '^  gpu' "$OUT/prof_${key}_$seq.txt" | head -24
            echo '```'
        } >>"$OUT/summary.md"
    done
done
git worktree remove --force "$WT" >/dev/null 2>&1 || true
