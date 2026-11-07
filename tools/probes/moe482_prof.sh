#!/bin/sh
# moe482_prof.sh MODEL NEW URL SHA256 [SEQ] — #482 on the M6: GPU time per
# prefill kernel of NEW (Metal) on the MoE GGUF at URL, a SEQ-token prompt
# (default 512), the device-routed prefill on and off
# (GEIST_MOE_DEVICE_PREFILL), as metal_prefill_kernels.sh reads it
# (GEIST_METAL_PIPELINE_PREFILL=1: one op per command buffer, so the shares
# count, not the absolute times), plus a default run of each for the total.
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
for dev in 1 0; do
    for split in 1 16; do
        sleep 20
        GEIST_MOE_DEVICE_PREFILL=$dev GEIST_BENCH_BACKEND=metal GEIST_METAL_PROFILE=1 \
            GEIST_METAL_PIPELINE_PREFILL=$split "$G" --gguf "$GGUF" --seq-lens "$SEQ" \
            --decode-n 1 --warmup 2 --repeats 3 >"$OUT/prof_dev${dev}_rot${split}.txt" 2>&1 || true
    done
done
{
    echo "## #482 kernel profile: $(basename "$GGUF") @ $SEQ, $(git -C "$WT" rev-parse --short HEAD)"
    for f in "$OUT"/prof_*.txt; do
        echo
        echo "### $(basename "$f")"
        echo '```'
        grep -E 'prefill_tps|gpu|dispatch|wait|moe' "$f" | head -80
        echo '```'
    done
} >"$OUT/summary.md"
git worktree remove --force "$WT" >/dev/null 2>&1 || true
