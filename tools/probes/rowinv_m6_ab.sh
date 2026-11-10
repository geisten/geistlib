#!/bin/sh
# rowinv_m6_ab.sh MODEL NEW [CYCLES [URL SHA256]] — tools/bench_revision_ab.py
# of NEW against origin/main on cpu_neon (prefill of the first part of
# docs/OFFLOAD.md, 512 tokens, then 64 decoded tokens), default model
# DeepSeek-V2-Lite Q4_K_M (its Q5_0 experts take the dequant trampoline that
# #527 changes). MODEL (mac-probe.yml's) is unused; the model comes by URL +
# sha256 into the host cache. Run by mac-probe.yml under the host lock;
# writes to $PROBE_OUT.
set -eu
shift
NEW=${1:?NEW revision, e.g. origin/agent/moe-gemm-row-invariant}
CYCLES=${2:-10}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
URL=${3:-https://huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF/resolve/0f37fdf276e8094747457f0ae4d40f2e8d2521f9/DeepSeek-V2-Lite.Q4_K_M.gguf}
SHA=${4:-70985c39ba3e51a53e6ad6ae25ce29a3efc92b4802f61645760e1bc1eb89ad1a}
{
    sw_vers
    sysctl -n machdep.cpu.brand_string
    echo "memory: $(($(sysctl -n hw.memsize) / 1073741824)) GiB"
    LC_ALL=C sysctl -n vm.loadavg
} >"$OUT/env.txt" 2>&1
GGUF=$(python3 tools/fetch_bench_model.py --url "$URL" --sha256 "$SHA")
git fetch --quiet origin "+refs/heads/main:refs/remotes/origin/main" \
    "+refs/heads/${NEW#origin/}:refs/remotes/$NEW"
python3 tools/bench_revision_ab.py --rev base=origin/main --rev "new=$NEW" --gguf "$GGUF" \
    --make-arg TARGET=mac-omp --make-arg "BACKENDS=cpu_neon cpu_scalar" \
    --seq-lens 512 --decode-n 64 --warmup 1 --cycles "$CYCLES" --prompt-file docs/OFFLOAD.md \
    --env GEIST_BENCH_BACKEND=cpu_neon --out-dir "$OUT/ab" 2>&1 | tee "$OUT/ab.log"
cat "$OUT"/ab/*/revision_ab.md >"$OUT/revision_ab.md"
