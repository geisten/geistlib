#!/bin/sh
# moe494_kern.sh MODEL REVS [ROUNDS [SEQS [FILTER]]] — #494 on the M6: the
# grouped expert GEMM per kernel, geist against llama.cpp's mul_mm_id.
# REVS: comma list of name=revision (e.g. main=origin/main,new=origin/agent/
# issue-494); each is built (Metal) and runs tests/bench_metal_moe_mm, on
# the tensor path and with GEIST_METAL_TENSOR=0 as name.sg (llama.cpp's
# with GGML_METAL_TENSOR_DISABLE=1 as llama.sg) (the
# NEW revision's source of it is copied into every tree, so all time the
# same rows). llama.cpp: the pinned build's test-backend-ops compiled from
# a patched copy of its source whose perf cases are the bench's shapes
# (OLMoE gate / down, Qwen3-30B, gpt-oss gate / down at 128 / 512 / 2048
# tokens; Q4_K, Q6_K, Q8_0, Q5_0, MXFP4). ROUNDS (default 3) rounds per
# prompt length of SEQS (default 128,512,2048), the variants' order rotating
# each round; medians. FILTER: the bench's row filter. MODEL (mac-probe's)
# is unused. Run by .gitea/workflows/mac-probe.yml under the host lock;
# writes to $PROBE_OUT.
set -u
shift
REVS=${1:?REVS name=rev,...}
ROUNDS=${2:-3}
SEQS=${3:-128,512,2048}
FILTER=${4:-}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
{
    sw_vers
    sysctl -n machdep.cpu.brand_string
    LC_ALL=C sysctl -n vm.loadavg
} >"$OUT/env.txt" 2>&1
TMP=$(mktemp -d)
NAMES=""
SRC=""
for nr in $(echo "$REVS" | tr ',' ' '); do
    n=${nr%%=*}
    rev=${nr#*=}
    case $rev in origin/*) git fetch --quiet origin "+refs/heads/${rev#origin/}:refs/remotes/$rev" ;; esac
    git worktree add --detach "$TMP/$n" "$rev" >>"$OUT/worktree.log" 2>&1
    echo "$n $(git -C "$TMP/$n" rev-parse HEAD)" >>"$OUT/shas.txt"
    [ -n "$SRC" ] || SRC="$TMP/$n/tests/bench_metal_moe_mm.c"
    [ -f "$TMP/$n/tests/bench_metal_moe_mm.c" ] && SRC="$TMP/$n/tests/bench_metal_moe_mm.c"
    NAMES="$NAMES $n"
done
for n in $NAMES; do
    [ -f "$TMP/$n/tests/bench_metal_moe_mm.c" ] || cp "$SRC" "$TMP/$n/tests/"
    (cd "$TMP/$n" && make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" bin) \
        >"$OUT/build_$n.log" 2>&1 || {
        echo "build of $n failed" >"$OUT/summary.md"
        exit 1
    }
done
GN=""
for n in $NAMES; do GN="$GN $n $n.sg"; done
NAMES=$GN
LB=$(scripts/ensure-llama-bench.sh metal 2>"$OUT/llama-build.log") || LB=""
TBO=""
if [ -n "$LB" ]; then
    LBD=$(dirname "$LB")
    L=$(dirname "$(dirname "$LBD")")
    python3 - "$L/tests/test-backend-ops.cpp" "$TMP/tbo.cpp" <<'EOF'
import sys
s = open(sys.argv[1]).read()
i = s.index("static std::vector<std::unique_ptr<test_case>> make_test_cases_perf() {\n")
j = s.index("\n", s.index("std::vector<std::unique_ptr<test_case>> test_cases;", i)) + 1
add = """
    for (int bs : {128, 512, 2048}) {
        for (ggml_type t : {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_MXFP4}) {
            test_cases.emplace_back(new test_mul_mat_id(t, GGML_TYPE_F32, 64, 8, true, 1024, bs, 2048));
            test_cases.emplace_back(new test_mul_mat_id(t, GGML_TYPE_F32, 64, 8, false, 2048, bs, 1024));
            test_cases.emplace_back(new test_mul_mat_id(t, GGML_TYPE_F32, 128, 8, false, 768, bs, 2048));
            if (ggml_blck_size(t) == 32) {
                test_cases.emplace_back(new test_mul_mat_id(t, GGML_TYPE_F32, 32, 4, true, 2880, bs, 2880));
                test_cases.emplace_back(new test_mul_mat_id(t, GGML_TYPE_F32, 32, 4, false, 2880, bs, 2880));
            }
        }
    }
    return test_cases;
"""
open(sys.argv[2], "w").write(s[:j] + add + s[j:])
EOF
    B=$(dirname "$LBD")
    if c++ -O3 -DNDEBUG -std=gnu++17 -arch arm64 -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_BLAS \
        -DGGML_USE_CPU -DGGML_USE_METAL -DLLAMA_SHARED -DLLAMA_SUBPROCESS -I"$L/include" \
        -I"$L/ggml/include" -I"$L/common" -I"$L/vendor" -w -c "$TMP/tbo.cpp" -o "$TMP/tbo.o" \
        >>"$OUT/llama-build.log" 2>&1 &&
        c++ -O3 -arch arm64 "$TMP/tbo.o" -o "$TMP/tbo" -Wl,-rpath,"$LBD" "$LBD"/libllama-common.0.*.dylib \
            "$LBD"/libllama.0.*.dylib "$LBD"/libggml.0.*.dylib "$LBD"/libggml-cpu.0.*.dylib \
            "$LBD"/libggml-blas.0.*.dylib "$LBD"/libggml-metal.0.*.dylib "$LBD"/libggml-base.0.*.dylib \
            "$B/common/libllama-common-base.a" >>"$OUT/llama-build.log" 2>&1; then
        TBO="$TMP/tbo"
        NAMES="$NAMES llama llama.sg"
    fi
fi
: >"$OUT/raw.txt"
for M in $(echo "$SEQS" | tr ',' ' '); do
    set -- $NAMES
    i=0
    while [ $i -lt "$ROUNDS" ]; do
        sleep 10
        for v in "$@"; do
            case $v in
            llama)
                "$TBO" perf -b MTL0 -o MUL_MAT_ID -p "n=$M," 2>&1 | grep 'MUL_MAT_ID(' |
                    sed "s/^/llama r$i /" >>"$OUT/raw.txt"
                ;;
            llama.sg)
                GGML_METAL_TENSOR_DISABLE=1 "$TBO" perf -b MTL0 -o MUL_MAT_ID -p "n=$M," 2>&1 |
                    grep 'MUL_MAT_ID(' | sed "s/^/llama.sg r$i /" >>"$OUT/raw.txt"
                ;;
            *.sg)
                GEIST_METAL_TENSOR=0 "$TMP/${v%.sg}/bin/mac-omp/release/tests/bench_metal_moe_mm" \
                    8 7 "$M" $FILTER 2>&1 | sed "s/^/$v r$i /" >>"$OUT/raw.txt"
                ;;
            *)
                "$TMP/$v/bin/mac-omp/release/tests/bench_metal_moe_mm" 8 7 "$M" $FILTER 2>&1 |
                    sed "s/^/$v r$i /" >>"$OUT/raw.txt"
                ;;
            esac
        done
        first=$1
        shift
        set -- "$@" "$first"
        i=$((i + 1))
    done
done
LC_ALL=C sysctl -n vm.loadavg >>"$OUT/env.txt"
{
    echo "## #494 grouped expert GEMM per kernel (us, median of $ROUNDS rounds)"
    echo
    echo '```'
    cat "$OUT/shas.txt"
    cat "$OUT/env.txt"
    echo '```'
    echo
    python3 - "$OUT/raw.txt" "$(echo $NAMES | cut -d' ' -f1)" <<'EOF'
import re, sys, statistics as st
from collections import defaultdict
f, base = sys.argv[1], sys.argv[2]
d = defaultdict(list); order = []
LT = {"q4_K": "Q4_K", "q6_K": "Q6_K", "q8_0": "Q8_0", "q5_0": "Q5_0", "mxfp4": "MXFP4"}
SH = {("64", "8", "1", "1024", "2048"): "olmoe.gate", ("64", "8", "0", "2048", "1024"): "olmoe.down",
      ("128", "8", "0", "768", "2048"): "q30b", ("32", "4", "1", "2880", "2880"): "gptoss.gate",
      ("32", "4", "0", "2880", "2880"): "gptoss.down"}
for line in open(f):
    m = re.match(r"(\S+) r\d+ (\S+)\s+(\S+)\s+(\S*)\s*n=(\d+)\s+k=(\d+) m=(\d+)\s+(\d+x\d+) (\S+)\s+([\d.]+) us", line)
    if m:
        row = (int(m.group(7)), f"{m.group(2)} {m.group(3)}")
        d[(m.group(1), row)].append(float(m.group(10)))
    else:
        m = re.match(r"llama r\d+\s+MUL_MAT_ID\(type_a=(\w+),type_b=f32,n_mats=(\d+),n_used=(\d+),b=(\d),m=(\d+),n=(\d+),k=(\d+).*?([\d.]+) us/run", line)
        if not m: continue
        sh = SH.get((m.group(2), m.group(3), m.group(4), m.group(5), m.group(7)))
        if sh is None or m.group(1) not in LT: continue
        row = (int(m.group(6)), f"{sh} {LT[m.group(1)]}")
        d[("llama", row)].append(float(m.group(8)))
    if row not in order: order.append(row)
vs = sorted({v for v, _ in d}, key=lambda x: (x != base, x == "llama", x))
print("| m | row | " + " | ".join(vs) + " |")
print("|--:|---|" + "--:|" * len(vs))
for r in sorted(order):
    b = st.median(d[(base, r)]) if d[(base, r)] else None
    cells = []
    for v in vs:
        xs = d[(v, r)]
        if not xs: cells.append("-"); continue
        md = st.median(xs)
        s = f"{md:.1f}"
        if b and v != base: s += f" ({100 * (md / b - 1):+.1f} %)"
        cells.append(s)
    print(f"| {r[0]} | {r[1]} | " + " | ".join(cells) + " |")
EOF
} >"$OUT/summary.md"
for n in $NAMES; do
    case $n in llama* | *.sg) continue ;; esac
    git worktree remove --force "$TMP/$n" >/dev/null 2>&1 || true
done
