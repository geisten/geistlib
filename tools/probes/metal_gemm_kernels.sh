#!/bin/sh
# metal_gemm_kernels.sh MODEL — one prefill GEMM at a time (#590): the two
# tensor-API kernels (fragments, the default, and GEIST_METAL_TC_COOP=1's
# cooperative matmul2d) through tests/bench_metal_gemm on the shapes of
# Llama 3.2 3B (Q4_K, Q6_K), Qwen3.5 0.8B (Q8_0) and Qwen3.5 4B (Q4_0), at
# 256 and 512 rows, and mlx's quantized GEMM on the same shapes
# (metal_gemm_kernels_mlx.py with the venv of mlx_vs_geist.sh). MODEL is
# unused (mac-probe.yml passes it). Run by .gitea/workflows/mac-probe.yml
# under the host lock; writes runs.txt, mlx.txt and summary.md to $PROBE_OUT.
set -eu
OUT=${PROBE_OUT:-probe-out}
CACHE=${GEIST_MLX_CACHE:-$HOME/geist-bench-cache/mlx}
mkdir -p "$OUT"
{
    git rev-parse HEAD
    sw_vers
    sysctl -n machdep.cpu.brand_string hw.memsize
    system_profiler SPDisplaysDataType 2>/dev/null | sed -n '1,14p'
    pmset -g batt 2>/dev/null | head -2
} >"$OUT/env.txt" 2>&1
make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" \
    bin/mac-omp/release/tests/bench_metal_gemm >"$OUT/build.log" 2>&1 || {
    tail -30 "$OUT/build.log"
    echo "geist build failed" >"$OUT/summary.md"
    exit 1
}
B=bin/mac-omp/release/tests/bench_metal_gemm
SHAPES="q4_k:3072:3072 q4_k:3072:1024 q4_k:3072:8192 q4_k:8192:3072 q6_k:3072:1024 q6_k:8192:3072
q8_0:1024:6144 q8_0:1024:3584 q8_0:3584:1024 q8_0:2048:1024 q4_0:2560:9216 q4_0:9216:2560
q4_0:2560:8192"
: >"$OUT/runs.txt"
for cycle in 1 2 3; do
    for coop in 0 1; do
        # The first GEMM of a process sequence runs on a cold GPU.
        GEIST_METAL_TC_COOP=$coop "$B" q4_k 3072 3072 20 3 256 >/dev/null 2>&1 || true
        for s in $SHAPES; do
            f=${s%%:*}
            r=${s#*:}
            GEIST_METAL_TC_COOP=$coop "$B" "$f" "${r%%:*}" "${r#*:}" 20 7 256 512 2>&1 |
                sed "s/^/c$cycle coop$coop /" >>"$OUT/runs.txt" || true
        done
    done
    sleep 5
done
MS=""
for s in $SHAPES; do
    f=${s%%:*}
    r=${s#*:}
    case $f in
    q8_0) bits=8 dt=bfloat16 ;;
    q6_k) bits=6 dt=float16 ;;
    *) bits=4 dt=float16 ;;
    esac
    MS="$MS $bits:${r%%:*}:${r#*:}:256:$dt $bits:${r%%:*}:${r#*:}:512:$dt"
done
: >"$OUT/mlx.txt"
for cycle in 1 2 3; do
    # shellcheck disable=SC2086 # MS is a word list
    "$CACHE/venv/bin/python" tools/probes/metal_gemm_kernels_mlx.py $MS \
        >>"$OUT/mlx.txt" 2>>"$OUT/mlx.err" || true
done
python3 - "$OUT" <<'PY'
import collections, re, statistics, sys
out = sys.argv[1]
t = collections.defaultdict(list)
fnv, err = {}, {}
for line in open(out + "/runs.txt"):
    m = re.match(r"c\d+ coop(\d) (\S+)\s+n_in=(\d+)\s+n_out=(\d+)\s+m=(\d+)\s+([\d.]+) ms"
                 r".*ref_err=(\S+) y_fnv=(\w+)", line)
    if m:
        key = (m[2], int(m[3]), int(m[4]), int(m[5]))
        t[key, int(m[1])].append(float(m[6]))
        err[key, int(m[1])] = m[7]
        fnv[key, int(m[1])] = m[8]
mlx = collections.defaultdict(list)
for line in open(out + "/mlx.txt"):
    m = re.match(r"mlx(\d+) n_in=(\d+) n_out=(\d+) m=(\d+) (\S+) ([\d.]+) ms", line)
    if m:
        mlx[int(m[1]), int(m[2]), int(m[3]), int(m[4])].append(float(m[6]))
keys = sorted({k for k, _ in t})
bits = {"q4_k": 4, "q6_k": 6, "q8_0": 8, "q4_0": 4}
lines = ["## One prefill GEMM on Metal: cooperative kernel, fragments, mlx (#590)", "",
         "ms per GEMM, median of three cycles; TFLOP/s = 2 x rows x n_in x n_out / time. mlx's "
         "weights are affine groups of 64 at the format's bits, its activations 16-bit floats.", "",
         "| format n_in x n_out, rows | cooperative ms | fragments ms | change | same output | "
         "cooperative TFLOP/s | fragments TFLOP/s | mlx ms | mlx TFLOP/s | "
         "difference to cpu_scalar (coop, fragments) |",
         "|:--|--:|--:|--:|:--|--:|--:|--:|--:|:--|"]
for k in keys:
    if (k, 0) not in t or (k, 1) not in t:
        continue
    fr, co = statistics.median(t[k, 0]), statistics.median(t[k, 1])
    fl = 2.0 * k[1] * k[2] * k[3] * 1e-9
    mk = (bits[k[0]], k[1], k[2], k[3])
    mm = statistics.median(mlx[mk]) if mk in mlx else float("nan")
    lines.append("| %s %d x %d, %d | %.3f | %.3f | %+.0f %% | %s | %.1f | %.1f | %.3f | %.1f | %s, %s |" % (
        k[0], k[1], k[2], k[3], co, fr, 100.0 * (fr / co - 1.0),
        "yes" if fnv[k, 0] == fnv[k, 1] else "NO", fl / co, fl / fr, mm, fl / mm,
        err[k, 1], err[k, 0]))
open(out + "/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY
