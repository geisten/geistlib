#!/bin/sh
# metal_dn_kernels.sh MODEL — the DeltaNet prefill mixer against mlx-lm's
# recurrence kernel (#590): tests/bench_metal_deltanet on the shapes of
# Qwen3.5 0.8B and 4B at 256 and 512 rows (one call is the whole mixer: conv,
# norms, the chunked delta rule, the gated norm), and mlx-lm's
# gated_delta_kernel on the same shapes (metal_dn_kernels_mlx.py with the
# venv of mlx_vs_geist.sh; the recurrence alone, its conv and norms are
# other mlx ops). MODEL is unused (mac-probe.yml passes it). Run by
# .gitea/workflows/mac-probe.yml under the host lock; writes runs.txt,
# mlx.txt and summary.md to $PROBE_OUT.
set -eu
OUT=${PROBE_OUT:-probe-out}
CACHE=${GEIST_MLX_CACHE:-$HOME/geist-bench-cache/mlx}
mkdir -p "$OUT"
make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" \
    bin/mac-omp/release/tests/bench_metal_deltanet >"$OUT/build.log" 2>&1 || {
    tail -30 "$OUT/build.log"
    echo "geist build failed" >"$OUT/summary.md"
    exit 1
}
: >"$OUT/runs.txt"
: >"$OUT/mlx.txt"
for cycle in 1 2 3; do
    bin/mac-omp/release/tests/bench_metal_deltanet 16 9 256 512 2>&1 | grep q35 >>"$OUT/runs.txt"
    "$CACHE/venv/bin/python" tools/probes/metal_dn_kernels_mlx.py 16:16:128:128:256 \
        16:16:128:128:512 16:32:128:128:256 16:32:128:128:512 >>"$OUT/mlx.txt" 2>>"$OUT/mlx.err" || true
    sleep 5
done
python3 - "$OUT" <<'PY'
import collections, re, statistics, sys
out = sys.argv[1]
geist, mlx = collections.defaultdict(list), collections.defaultdict(list)
for line in open(out + "/runs.txt"):
    m = re.match(r"(\S+)\s+m=(\d+)\s+heads=(\d+)/(\d+)\s+d=\d+\s+([\d.]+) ms/call", line)
    if m:
        geist[int(m[4]), int(m[2])].append(float(m[5]))
for line in open(out + "/mlx.txt"):
    m = re.match(r"mlx-gdn heads=\d+/(\d+) d=\S+ m=(\d+) ([\d.]+) ms/call", line)
    if m:
        mlx[int(m[1]), int(m[2])].append(float(m[3]))
lines = ["## DeltaNet prefill: geist's mixer, mlx-lm's recurrence kernel (#590)", "",
         "ms per call, median of three cycles. geist's call is the whole mixer, mlx-lm's the "
         "recurrence alone.", "",
         "| value heads | rows | geist mixer | mlx-lm gated_delta_kernel |", "|--:|--:|--:|--:|"]
for key in sorted(geist):
    lines.append("| %d | %d | %.3f | %s |" % (
        key[0], key[1], statistics.median(geist[key]),
        "%.3f" % statistics.median(mlx[key]) if key in mlx else "-"))
open(out + "/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY
