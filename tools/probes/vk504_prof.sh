#!/bin/sh
# vk504_prof.sh — GEIST_VK_PROFILE of one 512-token prefill and 64 decode
# steps of the absorbed files on the new revision (where the time goes).
. tools/probes/vk504_common.sh
REVS=$(echo $REVS | cut -d' ' -f1)
build_revs
SW=$RUNNER_TEMP/wt-new/bin/linux/release/tests/bench_perf_sweep
export GEIST_QUIET=1
. "$ROOT/scripts/gpu-lock.sh"
for f in "$GIGA" "$V2_S"; do
    for mode in "--seq-lens 512 --decode-n 1" "--seq-lens 512 --decode-n 64"; do
        echo "### $(basename "$f") $mode" | tee -a "$OUT/summary.md"
        # shellcheck disable=SC2086
        GEIST_VK_PROFILE=1 GEIST_BENCH_BACKEND=vulkan "$SW" --gguf "$f" $mode --warmup 1 --repeats 1 \
            >"$OUT/p.jsonl" 2>"$OUT/p.txt"
        { echo '```'; grep -v "^geist vulkan: device memory" "$OUT/p.txt" | head -60; echo '```'; } | tee -a "$OUT/summary.md"
    done
done
