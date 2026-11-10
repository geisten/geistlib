#!/bin/sh
# vk514_ab_set.sh MODEL_PATH(unused) SET — several bench_revision_ab A/Bs in
# ONE job under ONE hold of the bench + GPU locks (GEIST_QUIET=1
# scripts/gpu-lock.sh), so desktop CI cannot start between them; before
# each A/B it waits (bounded, logged) until no Gitea job container runs and
# the CPUs are under 5 % busy. SET names a list of model:cycles:seq:dn:revs
# (revs comma separated label=rev, the first the baseline).
set -eu
SET=$2
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
MV=base=origin/agent/vulkan-moe-gateup,new=origin/agent/issue-514-vk-dense-mv
RA=base=origin/main,new=origin/agent/issue-514-vk-rope-append
case "$SET" in
mv) SPECS="olmoe:10:128:256:$MV llama3b:8:128:128:$MV qwen3moe:8:128:64:$MV gptoss:8:128:64:$MV" ;;
ra) SPECS="olmoe:10:128:256:$RA llama3b:8:512:128:$RA qwen3moe:8:128:64:$RA gptoss:8:128:64:$RA" ;;
all) SPECS="olmoe:10:128:256:$MV llama3b:8:128:128:$MV qwen3moe:8:128:64:$MV gptoss:8:128:64:$MV olmoe:10:128:256:$RA llama3b:8:512:128:$RA qwen3moe:8:128:64:$RA gptoss:8:128:64:$RA" ;;
*) SPECS=$SET ;;
esac
model_path() {
    case "$1" in
    olmoe) python3 tools/fetch_bench_model.py --url https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct-GGUF/resolve/2ac1d2731792/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf --sha256 4ddc0e53159ed512b8dd67914a66e27bc618f694672ba43a9a0454eabd9c684f ;;
    qwen3moe) python3 tools/fetch_bench_model.py --url https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/e4d4bafdfb96a411a163846265362aceb0b9c63a/Qwen3-30B-A3B-Q4_K_M.gguf --sha256 0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5 ;;
    gptoss) python3 tools/fetch_bench_model.py --url https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/15dcfc7e7c608c6161507211ed43c5b590801b4b/gpt-oss-20b-MXFP4.gguf --sha256 27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901 ;;
    llama3b) echo "$HOME/geist-models/llama-3.2-3b-Q4_K_M.gguf" ;;
    esac
}
quiet() { # up to 20 min: no CI job container, 1-min load < 2
    i=0
    while :; do
        n=$(docker ps --filter name=GITEA-ACTIONS-TASK -q 2>/dev/null | wc -l)
        # CPU busy % over 3 s (/proc/stat): the hung iGPU processes sit in D
        # state, which the load average counts though they use no CPU
        a=$(head -1 /proc/stat); sleep 3; b=$(head -1 /proc/stat)
        l=$(echo "$a $b" | awk '{t1=0; for(i=2;i<=11;i++) t1+=$i; t2=0; for(i=13;i<=22;i++) t2+=$i; idle=($16+$17)-($5+$6); printf "%.1f", 100*(1-idle/(t2-t1))}')
        if [ "$n" -eq 0 ] && awk -v l="$l" 'BEGIN{exit !(l < 5)}'; then
            echo "quiet: $(date -u +%T) containers $n cpu busy $l % (waited ${i}x13 s)" | tee -a "$OUT/load.txt"
            return 0
        fi
        i=$((i + 1))
        if [ $i -ge 120 ]; then
            echo "NOT quiet after ~26 min: containers $n cpu busy $l %" | tee -a "$OUT/load.txt"
            ps -eo pcpu,etime,user,comm --sort=-pcpu | head -12 | tee -a "$OUT/load.txt"
            return 0
        fi
        sleep 10
    done
}
export GEIST_QUIET=1
. scripts/gpu-lock.sh
k=0
: >"$OUT/summary.md"
for spec in $SPECS; do
    k=$((k + 1))
    m=${spec%%:*}; r=${spec#*:}
    cyc=${r%%:*}; r=${r#*:}
    seq=${r%%:*}; r=${r#*:}
    dn=${r%%:*}; revs=${r#*:}
    f=$(model_path "$m")
    test -f "$f"
    cat "$f" >/dev/null
    revargs=""
    for x in $(echo "$revs" | tr ',' ' '); do revargs="$revargs --rev $x"; done
    echo "=== $k $m $revs" | tee -a "$OUT/load.txt"
    quiet
    # shellcheck disable=SC2086
    python3 tools/bench_revision_ab.py $revargs --gguf "$f" --seq-lens "$seq" \
        --decode-n "$dn" --repeats 1 --warmup 1 --cycles "$cyc" \
        --env GEIST_BENCH_BACKEND=vulkan \
        --make-arg TARGET=linux --make-arg "BACKENDS=vulkan cpu_x86 cpu_scalar" \
        --make-arg GEMM_PROVIDER=native --jobs 16 \
        --out-dir "$OUT/ab$k" >"$OUT/report$k.txt" 2>&1 || echo "A/B $k failed" | tee -a "$OUT/load.txt"
    echo "after $k: $(cut -d' ' -f1-3 /proc/loadavg) containers $(docker ps --filter name=GITEA-ACTIONS-TASK -q 2>/dev/null | wc -l)" | tee -a "$OUT/load.txt"
    { echo "### $k $m ($revs)"; echo '```'; grep -E "^Revisions|^Protocol|^\| (prefill|decode)" "$OUT/report$k.txt"; echo '```'; } >>"$OUT/summary.md"
done
{ echo '```'; cat "$OUT/load.txt"; echo '```'; } >>"$OUT/summary.md"
cat "$OUT/summary.md"
