#!/bin/sh
# mlx_vs_geist.sh MODEL [driver options] — mlx-lm against geist (Metal) and
# llama.cpp (Metal) on the same models: prefill at 512 prompt tokens, 128
# greedy tokens, time to ready (cold and warm), peak memory. Engines run
# interleaved, three cycles per model, every cell behind a power, idle and
# temperature gate. MODEL (mac-probe.yml's) only names the host's GGUF cache
# directory; the model pairs and every pin are in mlx_vs_geist_models.json.
#
# MLX lives in a directory of its own (default ~/geist-bench-cache/mlx,
# GEIST_MLX_CACHE overrides): a uv binary, a standalone CPython, a virtual
# environment with mlx and mlx-lm, and the Hugging Face cache of the MLX
# snapshots. Nothing else on the host is touched; a rerun reuses it.
#
#   -i script=tools/probes/mlx_vs_geist.sh -i args="--smoke --models qwen35-0.8b-8bit"
#   -i script=tools/probes/mlx_vs_geist.sh -i args="--models qwen35-0.8b-8bit,llama32-3b-4bit"
#
# Run by .gitea/workflows/mac-probe.yml under the host lock; writes
# results.jsonl (every cell, raw) and summary.md to $PROBE_OUT.
set -eu
MODEL=$1
shift
OUT=${PROBE_OUT:-probe-out}
CACHE=${GEIST_MLX_CACHE:-$HOME/geist-bench-cache/mlx}
mkdir -p "$OUT"
make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" bin >"$OUT/build.log" 2>&1 || {
    tail -30 "$OUT/build.log"
    echo "geist build failed" >"$OUT/summary.md"
    exit 1
}
df -g "$HOME" | tail -1
python3 tools/probes/mlx_vs_geist.py --cache "$CACHE" --gguf-cache "$(dirname "$MODEL")" \
    --out "$OUT" "$@"
