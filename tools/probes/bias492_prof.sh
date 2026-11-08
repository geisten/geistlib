#!/bin/sh
# bias492_prof.sh MODEL NEW [SEQS [BASE]] — #492 on the M6: where a gpt-oss-20b
# Metal prefill spends its GPU time, BASE (default origin/main) against NEW,
# at each prompt length of SEQS (default 512,2048):
#   - GEIST_METAL_PROFILE=1 with GEIST_METAL_PIPELINE_PREFILL=1 (one op per
#     command buffer, as #487's profile: the shares and dispatch counts per
#     kernel count, not the absolute times);
#   - a default run of each (the total, 3 repeats).
# MODEL (mac-probe.yml's) is unused. Writes to $PROBE_OUT.
set -u
shift
NEW=${1:?NEW revision}
SEQS=${2:-512,2048}
BASE=${3:-origin/main}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
{
    sw_vers
    sysctl -n machdep.cpu.brand_string
    LC_ALL=C sysctl -n vm.loadavg
} >"$OUT/env.txt" 2>&1
git fetch --quiet origin "+refs/heads/${NEW#origin/}:refs/remotes/$NEW" \
    "+refs/heads/${BASE#origin/}:refs/remotes/$BASE"
TMP=$(mktemp -d)
for r in new base; do
    if [ $r = new ]; then REV=$NEW; else REV=$BASE; fi
    git worktree add --detach "$TMP/$r" "$REV" >>"$OUT/worktree.log" 2>&1
    git -C "$TMP/$r" rev-parse HEAD >"$OUT/${r}_sha.txt"
    (cd "$TMP/$r" && make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" bin) >"$OUT/build_$r.log" 2>&1 || {
        echo "build of $REV failed" >"$OUT/summary.md"
        exit 1
    }
done
M=$(python3 tools/fetch_bench_model.py --url https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/15dcfc7e7c608c6161507211ed43c5b590801b4b/gpt-oss-20b-MXFP4.gguf \
    --sha256 27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901)
echo "## #492 gpt-oss-20b prefill profile: new $(cat "$OUT/new_sha.txt"), base $(cat "$OUT/base_sha.txt")" >"$OUT/summary.md"
for seq in $(echo "$SEQS" | tr ',' ' '); do
    for r in base new; do
        G="$TMP/$r/bin/mac-omp/release/tests/bench_perf_sweep"
        sleep 15
        GEIST_BENCH_BACKEND=metal GEIST_METAL_PROFILE=1 GEIST_METAL_PIPELINE_PREFILL=1 "$G" \
            --gguf "$M" --seq-lens "$seq" --decode-n 1 --warmup 1 --repeats 2 \
            >"$OUT/prof_${r}_${seq}.txt" 2>&1 || true
        sleep 15
        GEIST_BENCH_BACKEND=metal "$G" --gguf "$M" --seq-lens "$seq" --decode-n 16 --warmup 2 \
            --repeats 3 --emit-jsonl >"$OUT/total_${r}_${seq}.jsonl" 2>"$OUT/total_${r}_${seq}.err" || true
    done
done
{
    for f in "$OUT"/total_*.jsonl; do
        echo
        echo "### $(basename "$f")"
        echo '```'
        grep -h '^{' "$f" | cut -c1-400
        echo '```'
    done
    for f in "$OUT"/prof_*.txt; do
        echo
        echo "### $(basename "$f")"
        echo '```'
        grep -vE '^[[:space:]]*$' "$f" | grep -v '^{' | cut -c1-200 | head -90
        echo '```'
    done
} >>"$OUT/summary.md"
for r in new base; do git worktree remove --force "$TMP/$r" >/dev/null 2>&1 || true; done
