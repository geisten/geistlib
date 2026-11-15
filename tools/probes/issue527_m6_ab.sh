#!/bin/sh
# issue527_m6_ab.sh MODEL NEW MOE SEQ_LENS [CYCLES [DECODE_N [BASE]]] —
# tools/bench_revision_ab.py of NEW against BASE (default origin/main) on
# cpu_neon, for #527 / #528. MOE picks the GGUF by URL + sha256 into the host
# cache (deepseek | qwen3moe | gptoss), or `protocol` takes MODEL, the path
# mac-probe.yml resolved from its model key. Run by mac-probe.yml under the
# host lock; writes to $PROBE_OUT. Probe branch only, not for main.
set -eu
MODEL=$1
NEW=${2:?NEW revision}
MOE=${3:?deepseek | qwen3moe | gptoss | protocol}
SEQ=${4:?prefill lengths, e.g. 16,128,512}
CYCLES=${5:-10}
DN=${6:-64}
BASE=${7:-origin/main}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
case $MOE in
deepseek)
    URL=https://huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF/resolve/0f37fdf276e8094747457f0ae4d40f2e8d2521f9/DeepSeek-V2-Lite.Q4_K_M.gguf
    SHA=70985c39ba3e51a53e6ad6ae25ce29a3efc92b4802f61645760e1bc1eb89ad1a ;;
qwen3moe)
    URL=https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/e4d4bafdfb96a411a163846265362aceb0b9c63a/Qwen3-30B-A3B-Q4_K_M.gguf
    SHA=0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5 ;;
gptoss)
    URL=https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/15dcfc7e7c608c6161507211ed43c5b590801b4b/gpt-oss-20b-MXFP4.gguf
    SHA=27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901 ;;
protocol) URL= ;;
*) echo "unknown MOE $MOE" >&2; exit 2 ;;
esac
{
    sw_vers
    sysctl -n machdep.cpu.brand_string
    echo "memory: $(($(sysctl -n hw.memsize) / 1073741824)) GiB"
    LC_ALL=C sysctl -n vm.loadavg
} >"$OUT/env.txt" 2>&1
if [ -n "$URL" ]; then
    GGUF=$(python3 tools/fetch_bench_model.py --url "$URL" --sha256 "$SHA")
else
    GGUF=$MODEL
fi
for r in "$BASE" "$NEW"; do
    git fetch --quiet origin "+refs/heads/${r#origin/}:refs/remotes/$r"
done
python3 tools/bench_revision_ab.py --rev "base=$BASE" --rev "new=$NEW" --gguf "$GGUF" \
    --make-arg TARGET=mac-omp --make-arg "BACKENDS=cpu_neon cpu_scalar" \
    --seq-lens "$SEQ" --decode-n "$DN" --warmup 1 --cycles "$CYCLES" --prompt-file docs/OFFLOAD.md \
    --env GEIST_BENCH_BACKEND=cpu_neon --out-dir "$OUT/ab" 2>&1 | tee "$OUT/ab.log"
{
    echo "### $MOE, $NEW vs $BASE, cpu_neon"
    cat "$OUT"/ab/*/revision_ab.md
} >"$OUT/revision_ab.md"
