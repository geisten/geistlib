#!/bin/sh
# moe374_ab.sh MODEL [CYCLES] — #374 on the Apple M6 (mac-probe.yml, under
# the host lock): tools/bench_revision_ab.py of the four #374 branches
# against origin/main on gpt-oss-20b MXFP4 and DeepSeek-V2-Lite Q4_K_M,
# cpu_neon, pp512 of docs/OFFLOAD.md + 32 decode steps. MODEL (mac-probe's)
# is unused; the models come by URL + sha256 into the host cache.
set -eu
shift
CYCLES=${1:-6}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
{ sw_vers; sysctl -n machdep.cpu.brand_string; echo "memory: $(($(sysctl -n hw.memsize) / 1073741824)) GiB"; LC_ALL=C sysctl -n vm.loadavg; } >"$OUT/env.txt" 2>&1
B="agent/issue-374-moe-cpu-prefill agent/issue-374-moe-cpu-chunk agent/issue-374-moe-decode-wide agent/issue-374-sinks-grouped"
REFS="+refs/heads/main:refs/remotes/origin/main"
for b in $B; do REFS="$REFS +refs/heads/$b:refs/remotes/origin/$b"; done
# shellcheck disable=SC2086
git fetch --quiet origin $REFS
GPTOSS=$(python3 tools/fetch_bench_model.py --url https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/15dcfc7e7c608c6161507211ed43c5b590801b4b/gpt-oss-20b-MXFP4.gguf --sha256 27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901)
DEEPSEEK=$(python3 tools/fetch_bench_model.py --url https://huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF/resolve/0f37fdf276e8094747457f0ae4d40f2e8d2521f9/DeepSeek-V2-Lite.Q4_K_M.gguf --sha256 70985c39ba3e51a53e6ad6ae25ce29a3efc92b4802f61645760e1bc1eb89ad1a)
for m in "$GPTOSS" "$DEEPSEEK"; do
    n=$(basename "$m" .gguf)
    echo "== $n" | tee -a "$OUT/revision_ab.md"
    python3 tools/bench_revision_ab.py --rev base=origin/main \
        --rev wide=origin/agent/issue-374-moe-cpu-prefill --rev chunk=origin/agent/issue-374-moe-cpu-chunk \
        --rev dwide=origin/agent/issue-374-moe-decode-wide --rev sinks=origin/agent/issue-374-sinks-grouped \
        --gguf "$m" --make-arg TARGET=mac-omp --make-arg "BACKENDS=cpu_neon cpu_scalar" \
        --seq-lens 512 --decode-n 32 --warmup 1 --repeats 1 --cycles "$CYCLES" \
        --prompt-file docs/OFFLOAD.md --env GEIST_BENCH_BACKEND=cpu_neon \
        --out-dir "$OUT/ab-$n" 2>&1 | tee "$OUT/ab-$n.log" | grep -E '^\||Host|Protocol' >>"$OUT/revision_ab.md"
done
