#!/bin/sh
# metal_tc_tiles.sh MODEL [CYCLES [TILE...]] — Metal prefill per tensor-GEMM tile shape
# (GEIST_METAL_TC_TILE="nra,nrb,nk,nsg"), interleaved CYCLES times with the
# order rotated (#314 tile sweep). Run by mac-probe.yml under the host lock.
set -eu
MODEL=$1
CYCLES=${2:-2}
OUT=${PROBE_OUT:-probe-out}
SEQ=${PROBE_SEQ:-128,512,1024}
COOL=${PROBE_COOL:-10}
TILES=${PROBE_TILES:-"64,128,32,4 64,128,64,4 64,128,128,4 128,128,32,8 128,128,64,8 64,64,32,4 128,64,32,4 64,256,32,8 128,256,32,8 32,128,32,4 64,128,32,8 32,64,32,2"}
if [ $# -gt 2 ]; then
    shift 2
    TILES=$(echo "$*" | tr ':' ',') # tea's -i splits on commas: pass 64:128:32:4
fi
mkdir -p "$OUT"
make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" bin >"$OUT/build.log" 2>&1
GEIST=bin/mac-omp/release/tests/bench_perf_sweep
: >"$OUT/runs.jsonl"
n=$(echo $TILES | wc -w)
c=0
while [ "$c" -lt "$CYCLES" ]; do
    k=0
    while [ "$k" -lt "$n" ]; do
        idx=$(((k + c * 5) % n + 1))
        set -- $TILES
        eval t=\${$idx}
        sleep "$COOL"
        # an arm is GEMM_TILE[/GATE_UP_TILE|/off]: off keeps the unfused FFN front
        gt=${t%%/*}
        gu=default
        case $t in */*) gu=${t#*/} ;; esac
        set -- GEIST_METAL_TC_TILE=$gt
        if [ "$gu" = off ]; then
            set -- "$@" GEIST_FFN_GATE_UP_MN=0
        elif [ "$gu" != default ]; then
            set -- "$@" GEIST_METAL_TC_TILE_GU=$gu
        fi
        env GEIST_BENCH_BACKEND=metal GEIST_METAL_TENSOR=1 "$@" "$GEIST" --gguf "$MODEL" \
            --seq-lens "$SEQ" --decode-n 1 --warmup 1 --repeats 3 2>>"$OUT/err.txt" |
            sed "s|^{|{\"tile\":\"$t\",|" >>"$OUT/runs.jsonl"
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
base = tiles[0]
lines = ["## tc tile sweep (GEMM/gate_up), prefill tok/s (median; vs %s)" % base, "",
         "| arm nra,nrb,nk,nsg[,db] | " + " | ".join(str(s) for s in seqs) + " |",
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
