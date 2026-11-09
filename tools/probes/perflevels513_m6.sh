#!/bin/sh
# perflevels513_m6.sh MODEL NEW [BASE] — #513 on the M6: the perf-level
# topology, the OMP pool size geist's cpu_neon backend sets (OMP_DISPLAY_ENV)
# and a gpt-oss-20b CPU pp512/tg128, BASE (default origin/main) against NEW.
# MODEL (mac-probe.yml's) is unused. Writes to $PROBE_OUT.
set -u
shift
NEW=${1:?NEW revision}
BASE=${2:-origin/main}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
{
    sw_vers
    sysctl -n machdep.cpu.brand_string
    sysctl hw.ncpu hw.nperflevels
    i=0
    while [ "$i" -lt "$(sysctl -n hw.nperflevels)" ]; do
        sysctl "hw.perflevel$i.name" "hw.perflevel$i.physicalcpu"
        i=$((i + 1))
    done
    LC_ALL=C sysctl -n vm.loadavg
} >"$OUT/env.txt" 2>&1
git fetch --quiet origin "+refs/heads/${NEW#origin/}:refs/remotes/$NEW" \
    "+refs/heads/${BASE#origin/}:refs/remotes/$BASE"
TMP=$(mktemp -d)
for r in new base; do
    if [ $r = new ]; then REV=$NEW; else REV=$BASE; fi
    git worktree add --detach "$TMP/$r" "$REV" >>"$OUT/worktree.log" 2>&1
    git -C "$TMP/$r" rev-parse HEAD >"$OUT/${r}_sha.txt"
    (cd "$TMP/$r" && make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar" bin) >"$OUT/build_$r.log" 2>&1 || {
        echo "build of $REV failed" >"$OUT/summary.md"
        exit 1
    }
done
M=$(python3 tools/fetch_bench_model.py --url https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/15dcfc7e7c608c6161507211ed43c5b590801b4b/gpt-oss-20b-MXFP4.gguf \
    --sha256 27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901)
for round in 1 2; do
    for r in base new; do
        G="$TMP/$r/bin/mac-omp/release/tests/bench_perf_sweep"
        sleep 20
        GEIST_BENCH_BACKEND=cpu_neon OMP_DISPLAY_ENV=true "$G" --gguf "$M" --seq-lens 512 \
            --decode-n 128 --warmup 2 --repeats 3 --emit-jsonl \
            >"$OUT/${r}_$round.jsonl" 2>"$OUT/${r}_$round.err" || true
    done
done
{
    echo "## #513 M6 perf levels: new $(cat "$OUT/new_sha.txt"), base $(cat "$OUT/base_sha.txt")"
    echo '```'
    cat "$OUT/env.txt"
    echo '```'
    for f in "$OUT"/*.jsonl; do
        b=$(basename "$f" .jsonl)
        echo
        echo "### $b"
        echo '```'
        grep -h 'OMP_NUM_THREADS\|OMP_WAIT_POLICY' "$OUT/$b.err" "$f" | head -4
        grep -h '^{' "$f" | cut -c1-500
        echo '```'
    done
} >"$OUT/summary.md"
for r in new base; do git worktree remove --force "$TMP/$r" >/dev/null 2>&1 || true; done
