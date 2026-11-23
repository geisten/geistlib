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
# origin/main's bench_nll for the logits check, with this revision's
# bench_nll.c (it prints the logits hash; the library is main's).
MAIN=$(dirname "$OUT")/metal-profile-main
MAIN_TESTS=""
AB_BASE=""
rm -rf "$MAIN"
git worktree prune
if git fetch -q --depth 1 origin main && git worktree add -f --detach "$MAIN" FETCH_HEAD >/dev/null 2>&1; then
    AB_BASE="--ab-base $(git rev-parse FETCH_HEAD)"
    cp tests/bench_nll.c "$MAIN/tests/bench_nll.c"
    if make -C "$MAIN" -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" \
        bin/mac-omp/release/tests/bench_nll >"$OUT/build-main.log" 2>&1; then
        MAIN_TESTS="--main-tests $MAIN/bin/mac-omp/release/tests"
    fi
fi
# shellcheck disable=SC2086 # MAIN_TESTS and AB_BASE are empty or two words
python3 tools/probes/metal_profile.py --cache "$CACHE" --gguf-cache "$(dirname "$MODEL")" \
    --lib "$OUT/libmetal_trace.dylib" --out "$OUT" $MAIN_TESTS $AB_BASE "$@"
rm -f "$OUT/libmetal_trace.dylib"
git worktree remove --force "$MAIN" 2>/dev/null || rm -rf "$MAIN"
