#!/bin/sh
# metal_ple_tc_quality.sh MODEL — does the F32 tensor GEMM (Gemma's PLE
# weights rounded to half per tile, #314) change the output? origin/main
# (PLE in exact F32) against this revision, both with the tensor path on:
#   - bench_quality completions (all prompts, incl. the one past 64 tokens),
#     compared byte for byte;
#   - the next-token logits after a short and a long prompt
#     (tools/dump_geist_logits, tools/compare_logits.py).
# Run by .gitea/workflows/mac-probe.yml under the host lock; writes to $PROBE_OUT.
set -eu
MODEL=$1
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
BUILD='make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" bin'
sh -c "$BUILD" >"$OUT/build_new.log" 2>&1
git fetch -q --depth 1 origin main # the job checkout is shallow, without origin/main
rm -rf ../ple-base && git worktree add -q --detach ../ple-base FETCH_HEAD
(cd ../ple-base && sh -c "$BUILD") >"$OUT/build_base.log" 2>&1

printf 'The capital of France is' >"$OUT/prompt_short.txt"
head -c 3000 README.md >"$OUT/prompt_long.txt"

for rev in base new; do
    root=.
    [ "$rev" = base ] && root=../ple-base
    B=$root/bin/mac-omp/release
    GEIST_BENCH_BACKEND=metal GEIST_METAL_TENSOR=1 "$B/tests/bench_quality" "$MODEL" \
        >"$OUT/quality_$rev.txt" 2>"$OUT/quality_$rev.err" || true
    for p in short long; do
        GEIST_DUMP_BACKEND=metal GEIST_METAL_TENSOR=1 "$B/tools/dump_geist_logits" "$MODEL" \
            "$OUT/prompt_$p.txt" "$OUT/ids_${p}_$rev.bin" "$OUT/logits_${p}_$rev.bin" \
            >"$OUT/dump_${p}_$rev.log" 2>&1 || true
    done
done

{
    if cmp -s "$OUT/quality_base.txt" "$OUT/quality_new.txt"; then
        echo "bench_quality completions, main vs this revision: identical ($(wc -l <"$OUT/quality_new.txt") lines)"
    else
        echo "bench_quality completions, main vs this revision: DIFFERENT"
    fi
    for p in short long; do
        printf '%s prompt (%s tokens): ' "$p" "$(($(wc -c <"$OUT/ids_${p}_new.bin") / 4))"
        python3 tools/compare_logits.py "$OUT/logits_${p}_base.bin" "$OUT/logits_${p}_new.bin" ||
            echo "compare failed"
    done
} | tee "$OUT/summary.md"
git worktree remove --force ../ple-base
