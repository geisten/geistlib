#!/bin/sh
# moe494_combo.sh MODEL NEW [MODELS] — #494 on the M6 in one queue slot:
# moe494_kern.sh (Q6_K rows, main against NEW, 128 / 512 / 2048) and then
# moe494_mmax.sh (NEW's prefill at GEIST_M_MAX 256 and 512 against
# llama-bench, MODELS default all). Writes to $PROBE_OUT/kern and
# $PROBE_OUT/mmax; the summaries are concatenated.
set -u
MODEL=$1
NEW=${2:?NEW revision}
MODELS=${3:-ol,ds,gc,go}
OUT=${PROBE_OUT:-probe-out}
D=$(dirname "$0")
PROBE_OUT="$OUT/kern" sh "$D/moe494_kern.sh" "$MODEL" "main=origin/main,new=$NEW" 3 128,512,2048 Q6_K
PROBE_OUT="$OUT/mmax" sh "$D/moe494_mmax.sh" "$MODEL" "$NEW" 3 256,512 "$MODELS"
cat "$OUT/kern/summary.md" "$OUT/mmax/summary.md" >"$OUT/summary.md" 2>/dev/null
