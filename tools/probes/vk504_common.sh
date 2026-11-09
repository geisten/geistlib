# Shared by the vk504_* probes: build revisions ($REVS, label=rev) in
# worktrees under $RUNNER_TEMP, locate models and llama.cpp tools.
set -u
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
ROOT=$(pwd)
M=$HOME/geist-models/moe
GIGA=$M/ai-sage_GigaChat3-10B-A1.8B-Q4_K_M.gguf          # absorbed (split k_b / v_b)
GIGA_C=$M/GigaChat3-10B-A1.8B-Q4_K_M.combined.gguf       # decompressed (kv_b)
V2=$M/DeepSeek-V2-Lite.Q4_K_M.gguf                       # decompressed (kv_b)
V2_S=$M/DeepSeek-V2-Lite.Q4_K_M.split.gguf               # absorbed
LLAMA_DIR=$HOME/llama.cpp-cross-engine
LB=""
LABELS=""

build_revs() {
    for spec in $REVS; do
        label=${spec%%=*}
        rev=${spec#*=}
        wt="$RUNNER_TEMP/wt-$label"
        rm -rf "$wt"
        git worktree prune
        git worktree add -f --detach "$wt" "$rev" >/dev/null 2>&1 || { echo "no rev $rev"; exit 1; }
        (cd "$wt" && git log -1 --oneline && make -j16 TARGET=linux BACKENDS="vulkan cpu_x86 cpu_scalar" GEMM_PROVIDER=native bin) \
            >"$OUT/build_$label.log" 2>&1 || { echo "build $label failed"; tail -30 "$OUT/build_$label.log"; exit 1; }
        echo "built $label: $(cd "$wt" && git log -1 --oneline)"
        LABELS="$LABELS $label"
    done
}

# llama-bench (pinned, Vulkan) and its perplexity / tokenize tools.
llama_tools() {
    LB=$(sh "$ROOT/scripts/ensure-llama-bench.sh" vulkan 2>"$OUT/llama-build.log") || { tail -20 "$OUT/llama-build.log"; exit 1; }
    LBIN=$(dirname "$LB")
    if [ ! -x "$LBIN/llama-perplexity" ] || [ ! -x "$LBIN/llama-tokenize" ]; then
        cmake --build "$(dirname "$LBIN")" -j16 --target llama-perplexity llama-tokenize >>"$OUT/llama-build.log" 2>&1 ||
            { tail -20 "$OUT/llama-build.log"; exit 1; }
    fi
    echo "llama.cpp: $LBIN ($(git -C "$LLAMA_DIR" rev-parse --short HEAD))"
}
