#!/bin/sh
# issue527_m6_all.sh MODEL NEW MOES SEQ_LENS [CYCLES [DECODE_N [MB]]] — one
# mac-probe.yml job for several measurements (the runner's concurrency group
# keeps a single pending run): with MB=mb the Q5_0 microbench first, then
# issue527_m6_ab.sh for every model of MOES (comma-separated). Probe branch
# only, not for main.
set -eu
MODEL=$1 NEW=$2 MOES=$3 SEQ=$4
CYCLES=${5:-10} DN=${6:-64} MB=${7:-}
OUT=${PROBE_OUT:-probe-out}
if [ "$MB" = mb ]; then
    sh tools/probes/issue528_q5_0_mb.sh "$MODEL" "$NEW"
fi
for m in $(echo "$MOES" | tr ',' ' '); do
    PROBE_OUT="$OUT/$m" sh tools/probes/issue527_m6_ab.sh "$MODEL" "$NEW" "$m" "$SEQ" "$CYCLES" "$DN"
    cat "$OUT/$m/revision_ab.md" >>"$OUT/ab_all.md"
done
