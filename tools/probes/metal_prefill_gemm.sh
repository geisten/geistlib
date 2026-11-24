#!/bin/sh
# metal_prefill_gemm.sh MODEL [driver options] — the tensor-API GEMM of
# fragments (#590) in the models: origin/main, this checkout and this
# checkout with GEIST_METAL_TC_COOP=1 against mlx-lm, on the pairs of
# mlx_vs_geist_models.json (512 prompt tokens, 128 greedy tokens,
# interleaved cycles behind mlx_vs_geist.py's gate), and whether the three
# geist arms compute the same logits (tests/bench_nll, decode and prefill
# path) and tokens; then tools/bench_revision_ab.py, origin/main against
# this checkout, on each pair's GGUF. Before the models: the Metal unit
# tests with either kernel, metal_gemm_kernels.sh and metal_dn_kernels.sh.
# MODEL (mac-probe.yml's) only names the host's GGUF cache directory.
#
#   -i script=tools/probes/metal_prefill_gemm.sh -i args="--models qwen35-0.8b-8bit,llama32-3b-4bit"
#
# Run by .gitea/workflows/mac-probe.yml under the host lock; writes
# cells.jsonl and summary.md to $PROBE_OUT.
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
# The Metal unit tests on this GPU, with the fragment kernels and with the
# cooperative ones.
{
    echo "## Metal unit tests"
    echo
    echo "| test | GEIST_METAL_TC_COOP | exit | FAIL lines |"
    echo "| :-- | --: | --: | --: |"
    for t in bin/mac-omp/release/tests/test_*metal*_unit bin/mac-omp/release/tests/test_decode_ahead_unit; do
        for coop in 0 1; do
            log="$OUT/unit_$(basename "$t")_coop$coop.log"
            rc=0
            GEIST_METAL_TC_COOP=$coop "$t" >"$log" 2>&1 || rc=$?
            echo "| $(basename "$t") | $coop | $rc | $(grep -c FAIL "$log" || true) |"
        done
    done
    echo
    grep -h "tensor GEMM kernels:" "$OUT/unit_test_backend_metal_parity_unit_coop0.log" || true
    echo
} >"$OUT/units.md"
# One GEMM at a time, both kernels and mlx's; the DeltaNet mixer and
# mlx-lm's recurrence kernel.
PROBE_OUT="$OUT/kernels" sh tools/probes/metal_gemm_kernels.sh "$MODEL" >/dev/null 2>&1 || true
PROBE_OUT="$OUT/dn" sh tools/probes/metal_dn_kernels.sh "$MODEL" >/dev/null 2>&1 || true
# origin/main's library with this revision's bench_nll.c (the logits hash
# and dump).
MAIN=$(dirname "$OUT")/metal-prefill-gemm-main
git worktree remove --force "$MAIN" 2>/dev/null || true
git worktree prune
git fetch -q --depth 1 origin main
git worktree add -f --detach "$MAIN" FETCH_HEAD >/dev/null 2>&1
cp tests/bench_nll.c "$MAIN/tests/bench_nll.c"
make -C "$MAIN" -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" \
    bin/mac-omp/release/tests/bench_nll bin/mac-omp/release/tests/bench_perf_sweep \
    >"$OUT/build-main.log" 2>&1 || {
    tail -30 "$OUT/build-main.log"
    echo "origin/main build failed" >"$OUT/summary.md"
    exit 1
}
python3 tools/probes/metal_prefill_gemm.py --cache "$CACHE" --gguf-cache "$(dirname "$MODEL")" \
    --lib "$OUT/libmetal_trace.dylib" --out "$OUT" --main-rev "$(git rev-parse FETCH_HEAD)" \
    --main-tests "$MAIN/bin/mac-omp/release/tests" "$@"
cat "$OUT/units.md" "$OUT/kernels/summary.md" "$OUT/dn/summary.md" >>"$OUT/summary.md" \
    2>/dev/null || true
git worktree remove --force "$MAIN" 2>/dev/null || true
