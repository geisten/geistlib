#!/bin/sh
# issue528_q5_0_mb.sh MODEL NEW [BASE] — the Q5_0 expert-shape microbench
# (issue528_q5_0_mb.c) against libgeist.a of BASE (default origin/main) and
# of NEW, three alternating rounds. MODEL (mac-probe.yml's) is unused. Run
# by mac-probe.yml under the host lock; writes to $PROBE_OUT. Probe branch
# only, not for main.
set -eu
shift
NEW=${1:?NEW revision}
BASE=${2:-origin/main}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
W=$(mktemp -d)
OMP=$(brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
for v in base new; do
    if [ $v = base ]; then r=$BASE; else r=$NEW; fi
    git fetch --quiet origin "+refs/heads/${r#origin/}:refs/remotes/$r"
    git worktree add --detach "$W/$v" "$r" >/dev/null 2>&1
    git -C "$W/$v" log --oneline -1 | sed "s/^/$v: /" >>"$OUT/head.txt"
    make -C "$W/$v" TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar" -j8 lib >"$OUT/build_$v.log" 2>&1 ||
        { tail -30 "$OUT/build_$v.log"; exit 1; }
    cc -std=c23 -O3 -Xpreprocessor -fopenmp -I"$OMP/include" -I"$W/$v/include" -I"$W/$v" \
        -I"$W/$v/src/base" -I"$W/$v/src/quant" -I"$W/$v/tests" tools/probes/issue528_q5_0_mb.c \
        "$W/$v/lib/mac-omp/release/libgeist.a" -framework Accelerate -L"$OMP/lib" -lomp -lm \
        -o "$W/mb_$v"
done
{
    echo "### Q5_0 1408 -> 2048 (DeepSeek-V2-Lite ffn_down_exps), cpu_neon"
    echo '```'
    cat "$OUT/head.txt"
    sysctl -n machdep.cpu.brand_string
    for round in 1 2 3; do
        for v in base new; do
            echo "== $v (round $round)"
            "$W/mb_$v"
        done
    done
    echo '```'
} | tee "$OUT/q5_0_mb.md"
git worktree remove --force "$W/base" >/dev/null 2>&1 || true
git worktree remove --force "$W/new" >/dev/null 2>&1 || true
