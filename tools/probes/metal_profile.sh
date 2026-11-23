#!/bin/sh
# metal_profile.sh MODEL [driver options] — where geist's Metal time goes
# against mlx-lm's (#590): both engines on the pairs of
# mlx_vs_geist_models.json with tools/probes/metal_trace.m loaded, which
# records every Metal command buffer (GPU start and end, dispatches per
# kernel). MODEL (mac-probe.yml's) only names the host's GGUF cache
# directory; MLX lives where mlx_vs_geist.sh put it (~/geist-bench-cache/mlx,
# GEIST_MLX_CACHE overrides).
#
#   -i script=tools/probes/metal_profile.sh -i args="--models qwen35-0.8b-8bit"
#
# Run by .gitea/workflows/mac-probe.yml under the host lock; writes the
# traces, cells.jsonl and summary.md to $PROBE_OUT.
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
cc -std=gnu23 -dynamiclib -fobjc-arc -O2 -o "$OUT/libmetal_trace.dylib" \
    tools/probes/metal_trace.m -framework Metal -framework Foundation
python3 tools/probes/metal_profile.py --cache "$CACHE" --gguf-cache "$(dirname "$MODEL")" \
    --lib "$OUT/libmetal_trace.dylib" --out "$OUT" "$@"
rm -f "$OUT/libmetal_trace.dylib"
