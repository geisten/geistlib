#!/bin/sh
# vk514_ab.sh MODEL [CYCLES] [SEQ] [DECODE_N] — vulkan-ab.yml's A/B step
# (tools/bench_revision_ab.py, alternating revisions, control copy of the
# first) on the discrete GPU, under the bench + GPU locks, for a real GGUF.
# $REVS: label=rev ..., the first is the baseline.
set -eu
MODEL=$1
CYCLES=${2:-8}
SEQ=${3:-128}
DN=${4:-128}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
revargs=""
for spec in $REVS; do revargs="$revargs --rev $spec"; done
export GEIST_QUIET=1
. scripts/gpu-lock.sh
uptime | tee "$OUT/load.txt"
cat "$MODEL" >/dev/null # page cache
# shellcheck disable=SC2086
python3 tools/bench_revision_ab.py $revargs \
    --gguf "$MODEL" --seq-lens "$SEQ" \
    --decode-n "$DN" --repeats 1 --warmup 1 --cycles "$CYCLES" \
    --env GEIST_BENCH_BACKEND=vulkan \
    --make-arg TARGET=linux --make-arg "BACKENDS=vulkan cpu_x86 cpu_scalar" \
    --make-arg GEMM_PROVIDER=native --jobs 16 \
    --out-dir "$OUT/ab" | tee "$OUT/report.txt"
uptime | tee -a "$OUT/load.txt"
{ echo "### $(basename "$MODEL")"; echo '```'; cat "$OUT/report.txt"; cat "$OUT/load.txt"; echo '```'; } >"$OUT/summary.md"
