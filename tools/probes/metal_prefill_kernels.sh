#!/bin/sh
# metal_prefill_kernels.sh MODEL — GPU time per prefill kernel on Metal (#314).
# GEIST_METAL_PIPELINE_PREFILL=1 puts every op of the prefill in its own
# command buffer, and GEIST_METAL_PROFILE=1 books each buffer's GPU time
# under its kernel mask, so the profile reads per kernel. One-op buffers
# change the GPU's clocking (#77), so the shares are what this reads, not
# the absolute times; a default-rotation run per length gives the total.
# Run by .gitea/workflows/mac-probe.yml under the host lock.
set -eu
MODEL=$1
OUT=${PROBE_OUT:-probe-out}
COOL=${PROBE_COOL:-20}
mkdir -p "$OUT"
make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" bin >"$OUT/build.log" 2>&1
GEIST=bin/mac-omp/release/tests/bench_perf_sweep
for seq in 512 1024; do
    for split in 16 1; do
        sleep "$COOL"
        GEIST_BENCH_BACKEND=metal GEIST_METAL_TENSOR=1 GEIST_METAL_PROFILE=1 \
            GEIST_METAL_PIPELINE_PREFILL=$split "$GEIST" --gguf "$MODEL" --seq-lens "$seq" \
            --decode-n 1 --warmup 1 --repeats 1 >"$OUT/profile_seq${seq}_rot${split}.txt" 2>&1 || true
    done
done
grep -h '"prefill_tps"' "$OUT"/profile_*.txt | head -8 >"$OUT/tps.txt" || true
