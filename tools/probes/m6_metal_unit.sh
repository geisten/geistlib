#!/bin/sh
# m6_metal_unit.sh MODEL [REV...] — the Metal unit tests that fail on the
# M6 (#361), per revision, with the tensor-API path off and on
# (GEIST_METAL_TENSOR=0/1). No REV: this checkout only. Each REV is fetched
# from origin and built in its own worktree. MODEL is unused (mac-probe.yml
# passes it). Run by .gitea/workflows/mac-probe.yml; writes to $PROBE_OUT.
set -u
shift
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
TESTS=${PROBE_TESTS:-"test_backend_metal_parity_unit test_decode_batch_unit test_olmoe_unit test_qwen35moe_unit"}
BUILD='make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" bin'
{
    sw_vers
    sysctl -n machdep.cpu.brand_string
} >"$OUT/env.txt" 2>&1

run_tree() { # label root
    label=$1
    root=$2
    (cd "$root" && git rev-parse HEAD && sh -c "$BUILD") >"$OUT/build_$label.log" 2>&1 || {
        echo "| $label | build failed | |" >>"$OUT/summary.md"
        return
    }
    for t in $TESTS; do
        for v in 0 1; do
            log="$OUT/${label}_${t}_tc$v.log"
            GEIST_METAL_TENSOR=$v "$root/bin/mac-omp/release/tests/$t" >"$log" 2>&1
            rc=$?
            fl=$(grep -c FAIL "$log")
            echo "| $label | $t | tc=$v | rc=$rc | FAIL lines $fl |" >>"$OUT/summary.md"
        done
    done
}

echo "| rev | test | tensor | rc | fails |" >"$OUT/summary.md"
echo "|---|---|---|---|---|" >>"$OUT/summary.md"
if [ $# -eq 0 ]; then
    run_tree head .
fi
for rev in "$@"; do
    git fetch -q --depth 1 origin "$rev" || {
        echo "| $rev | fetch failed | |" >>"$OUT/summary.md"
        continue
    }
    l=$(printf %.8s "$rev")
    rm -rf "../m6u-$l"
    git worktree add -q --detach "../m6u-$l" FETCH_HEAD
    run_tree "$l" "../m6u-$l"
done
exit 0
