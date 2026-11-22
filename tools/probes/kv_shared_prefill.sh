#!/bin/sh
# kv_shared_prefill.sh MODEL — the bit-identity test of #591 on this host:
# a prefill that leaves the KV-shared layers out for its cache rows against
# one that runs them for every row (GEIST_KV_SHARED_FULL=1), on Metal and on
# cpu_neon (tests/test_kv_shared_prefill_int.c). MODEL is a Gemma 4 GGUF
# (-i model=gemma4-e2b-q4km or gemma4-e4b-q4km). Run by
# .gitea/workflows/mac-probe.yml under the host lock; writes to $PROBE_OUT.
set -u
MODEL=$1
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
{
    sw_vers
    sysctl -n machdep.cpu.brand_string
    git rev-parse HEAD
} >"$OUT/env.txt" 2>&1
make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" bin >"$OUT/build.log" 2>&1 || {
    tail -30 "$OUT/build.log"
    echo "geist build failed" >"$OUT/summary.md"
    exit 1
}
rc_all=0
{
    echo "| model | backend | rc | result |"
    echo "|---|---|---|---|"
} >"$OUT/summary.md"
for be in metal cpu_neon; do
    log="$OUT/kv_shared_$be.log"
    GEIST_BACKEND=$be GEIST_GGUF_PATH=$MODEL \
        bin/mac-omp/release/tests/test_kv_shared_prefill_int >"$log" 2>&1
    rc=$?
    [ $rc -eq 0 ] || rc_all=1
    echo "| $(basename "$MODEL") | $be | $rc | $(grep '^backend' "$log" | tail -1) |" >>"$OUT/summary.md"
    grep -E 'FAIL|logit ' "$log" | head -20
done
exit $rc_all
