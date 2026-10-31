#!/bin/sh
# metal_tc_tiles.sh MODEL [CYCLES] — Metal prefill per tensor-GEMM tile shape
# (GEIST_METAL_TC_TILE="nra,nrb,nk,nsg"), interleaved CYCLES times with the
# order rotated (#314 tile sweep). Run by mac-probe.yml under the host lock.
set -eu
MODEL=$1
CYCLES=${2:-2}
OUT=${PROBE_OUT:-probe-out}
SEQ=${PROBE_SEQ:-128,512,1024}
COOL=${PROBE_COOL:-10}
TILES=${PROBE_TILES:-"64,128,32,4 64,128,64,4 64,128,128,4 128,128,32,8 128,128,64,8 64,64,32,4 128,64,32,4 64,256,32,8 128,256,32,8 32,128,32,4 64,128,32,8 32,64,32,2"}
mkdir -p "$OUT"
make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" bin >"$OUT/build.log" 2>&1
GEIST=bin/mac-omp/release/tests/bench_perf_sweep
: >"$OUT/runs.jsonl"
n=$(echo $TILES | wc -w)
c=0
while [ "$c" -lt "$CYCLES" ]; do
    set -- $TILES
    k=0
    while [ "$k" -lt "$n" ]; do
        idx=$(((k + c * 5) % n + 1))
        eval t=\${$idx}
        sleep "$COOL"
        GEIST_BENCH_BACKEND=metal GEIST_METAL_TENSOR=1 GEIST_METAL_TC_TILE=$t "$GEIST" --gguf "$MODEL" \
            --seq-lens "$SEQ" --decode-n 1 --warmup 1 --repeats 3 2>>"$OUT/err.txt" |
            sed "s/^{/{\"tile\":\"$t\",/" >>"$OUT/runs.jsonl"
        k=$((k + 1))
    done
    c=$((c + 1))
done
python3 - "$OUT" <<'PY'
import json, statistics, sys, os
out = sys.argv[1]
d = {}
for line in open(os.path.join(out, "runs.jsonl")):
    if line.startswith("{"):
        r = json.loads(line)
        d.setdefault((r["tile"], r["seq_len"]), []).append(r["prefill_tps"])
tiles = []
for t, _ in d:
    if t not in tiles:
        tiles.append(t)
seqs = sorted({s for _, s in d})
base = tiles[0] if "64,128,32,4" not in tiles else "64,128,32,4"
lines = ["## tc GEMM tile sweep, prefill tok/s (median; vs %s)" % base, "",
         "| tile nra,nrb,nk,nsg | " + " | ".join(str(s) for s in seqs) + " |",
         "|--|" + "--:|" * len(seqs)]
for t in tiles:
    cells = []
    for s in seqs:
        v = statistics.median(d[(t, s)])
        b = statistics.median(d[(base, s)])
        cells.append("%.1f (%+.1f %%, %.0f-%.0f)" % (v, 100 * (v / b - 1), min(d[(t, s)]), max(d[(t, s)])))
    lines.append("| %s | %s |" % (t, " | ".join(cells)))
open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY
