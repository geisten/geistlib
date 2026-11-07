#!/bin/sh
# moe482_sample.sh MODEL NEW URL SHA256 [SEQ] — #482 on the M6: a CPU
# sample (macOS `sample`, 8 s) of NEW's bench_perf_sweep repeating
# SEQ-token prefills (default 512) of the MoE GGUF at URL on Metal, the
# device-routed prefill on and off, to see where the host spends a prefill.
# MODEL (mac-probe.yml's) is unused. Writes to $PROBE_OUT.
set -u
shift
NEW=${1:?NEW revision}
URL=${2:?model URL}
SHA=${3:?model sha256}
SEQ=${4:-512}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
GGUF=$(python3 tools/fetch_bench_model.py --url "$URL" --sha256 "$SHA")
git fetch --quiet origin "+refs/heads/${NEW#origin/}:refs/remotes/$NEW"
WT="$(mktemp -d)/new"
git worktree add --detach "$WT" "$NEW" >"$OUT/worktree.log" 2>&1
(cd "$WT" && make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" bin) >"$OUT/build.log" 2>&1 || exit 1
G="$WT/bin/mac-omp/release/tests/bench_perf_sweep"
: >"$OUT/summary.md"
for dev in 1 0; do
    sleep 20
    GEIST_MOE_DEVICE_PREFILL=$dev GEIST_BENCH_BACKEND=metal "$G" --gguf "$GGUF" --seq-lens "$SEQ" \
        --decode-n 1 --warmup 2 --repeats 60 >"$OUT/bench_dev$dev.txt" 2>&1 &
    P=$!
    sleep 10
    sample "$P" 8 -file "$OUT/sample_dev$dev.txt" >/dev/null 2>&1
    wait "$P"
    {
        echo "## dev=$dev $(grep -o '"prefill_ms":[0-9.]*' "$OUT/bench_dev$dev.txt")"
        echo '```'
        sed -n '/Sort by top of stack/,$p' "$OUT/sample_dev$dev.txt" | head -60
        echo '```'
    } >>"$OUT/summary.md"
done
git worktree remove --force "$WT" >/dev/null 2>&1 || true
