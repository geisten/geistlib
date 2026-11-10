#!/bin/sh
# #522 probe (probe branch only): gpt-oss-20b MXFP4, the first $4 bytes of
# docs/OFFLOAD.md, on cpu_scalar and cpu_x86 of two builds ($1 base, $2
# new: bin/linux/release dirs). Reports each run's status and compares the
# final logits. Never fails the job for a run that fails: that is the
# finding.
BASE=$1 NEW=$2 MODEL=$3 BYTES=$4 OUT=$5
head -c "$BYTES" docs/OFFLOAD.md > "$OUT/prompt.txt"
for rev in base new; do
    [ $rev = base ] && B=$BASE || B=$NEW
    echo "== unit test, $rev"
    "$B/tests/test_gpt_oss_unit" > "$OUT/unit-$rev.log" 2>&1
    echo "test_gpt_oss_unit rc=$?"
    grep -E "FAIL|not finite|deep gates" "$OUT/unit-$rev.log"
    for be in cpu_scalar cpu_x86; do
        echo "== gpt-oss, $rev, $be"
        t0=$(date +%s)
        GEIST_DUMP_BACKEND=$be "$B/tools/dump_geist_logits" "$MODEL" "$OUT/prompt.txt" \
            "$OUT/ids-$rev-$be.bin" "$OUT/logits-$rev-$be.bin" > "$OUT/run-$rev-$be.log" 2>&1
        echo "rc=$? after $(($(date +%s) - t0)) s"
        grep -v DBG522 "$OUT/run-$rev-$be.log" | tail -4
        # The trace build's lines, per layer: rows, gates below -52.13,
        # the lowest gate, non-finite values after the activation.
        grep DBG522 "$OUT/run-$rev-$be.log" | awk '{r[$3]++; b[$3]+=$9; n[$3]+=$11; if (!($3 in m) || $7 < m[$3]) m[$3]=$7}
            END {for (l in r) printf "  trace layer %2d: %d expert rows with a gate below -52.13 (%d gates, lowest %.2f), %d non-finite after the activation\n", l, r[l], b[l], m[l], n[l]}' | sort -k3n
    done
done
python3 - "$OUT" << 'PY'
import array, math, os, sys
out = sys.argv[1]
def load(name):
    p = os.path.join(out, "logits-%s.bin" % name)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    a = array.array("f")
    with open(p, "rb") as f:
        a.frombytes(f.read())
    return a
def ntok(name):
    p = os.path.join(out, "ids-%s.bin" % name)
    return os.path.getsize(p) // 4 if os.path.exists(p) else 0
v = {n: load(n) for n in ("base-cpu_scalar", "base-cpu_x86", "new-cpu_scalar", "new-cpu_x86")}
for n, a in v.items():
    if a is None:
        print("%s: no logits" % n)
    else:
        bad = sum(1 for x in a if x != x or abs(x) == math.inf)
        print("%s: %d logits, %d not finite, argmax %d, prompt tokens %d"
              % (n, len(a), bad, max(range(len(a)), key=a.__getitem__), ntok(n)))
def rel(x, y):
    a, b = v[x], v[y]
    if a is None or b is None:
        print("rel L2 %s vs %s: n/a" % (x, y))
        return
    num = math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))
    den = math.sqrt(sum(q * q for q in b))
    print("rel L2 %s vs %s: %.4g" % (x, y, num / den))
rel("base-cpu_x86", "new-cpu_scalar")
rel("new-cpu_x86", "new-cpu_scalar")
rel("base-cpu_x86", "new-cpu_x86")
rel("base-cpu_scalar", "new-cpu_scalar")
PY
