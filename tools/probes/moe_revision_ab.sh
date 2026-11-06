#!/bin/sh
# moe_revision_ab.sh MODEL NEW [CYCLES] — tools/bench_revision_ab.py of NEW
# against origin/main on Qwen3-30B-A3B Q4_K_M with Metal (512-token prompt,
# 128 decoded tokens, cool gate on): the MoE no-regression gate of #471
# (#373), which mac-ab.yml cannot run (MoE models are not in the
# cross-engine protocol). The model comes by URL + sha256 into the host
# cache, as in mac-batch-ab.yml. MODEL (mac-probe.yml's) is unused. Run by
# .gitea/workflows/mac-probe.yml under the host lock; writes to $PROBE_OUT.
set -eu
shift
NEW=${1:?NEW revision, e.g. origin/agent/issue-373}
CYCLES=${2:-10}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
URL=https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/e4d4bafdfb96a411a163846265362aceb0b9c63a/Qwen3-30B-A3B-Q4_K_M.gguf
SHA=0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5
{
    sw_vers
    sysctl -n machdep.cpu.brand_string
    echo "memory: $(($(sysctl -n hw.memsize) / 1073741824)) GiB"
    LC_ALL=C sysctl -n vm.loadavg
} >"$OUT/env.txt" 2>&1
GGUF=$(python3 tools/fetch_bench_model.py --url "$URL" --sha256 "$SHA")
# mac-probe checks out one revision shallowly: both must resolve here.
git fetch --quiet origin "+refs/heads/main:refs/remotes/origin/main" \
    "+refs/heads/${NEW#origin/}:refs/remotes/$NEW"
python3 tools/bench_revision_ab.py --rev base=origin/main --rev "new=$NEW" --gguf "$GGUF" \
    --make-arg TARGET=mac-omp --make-arg "BACKENDS=cpu_neon cpu_scalar metal" \
    --seq-lens 512 --decode-n 128 --warmup 16 --cycles "$CYCLES" \
    --env GEIST_BENCH_BACKEND=metal --out-dir "$OUT/ab" 2>&1 | tee "$OUT/ab.log"
cat "$OUT"/ab/*/revision_ab.md >"$OUT/revision_ab.md"
