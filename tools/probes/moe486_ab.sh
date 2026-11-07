#!/bin/sh
# moe486_ab.sh MODEL NEW CYCLES SEQS URL SHA256 [BASE] — #486 on the M6:
# tools/bench_revision_ab.py of NEW against BASE (default origin/main) with
# Metal on the MoE GGUF at URL (sha256-gated into the host cache), prompts
# of SEQS tokens (comma list) and 128 decoded tokens, a control copy of the
# base, cool gate on. MODEL (mac-probe.yml's) is unused. Run by
# .gitea/workflows/mac-probe.yml under the host lock; writes to $PROBE_OUT.
set -eu
shift
NEW=${1:?NEW revision, e.g. origin/agent/issue-486}
CYCLES=${2:-10}
SEQS=${3:-512}
URL=${4:?model URL}
SHA=${5:?model sha256}
BASE=${6:-origin/main}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
{
    sw_vers
    sysctl -n machdep.cpu.brand_string
    echo "memory: $(($(sysctl -n hw.memsize) / 1073741824)) GiB"
    LC_ALL=C sysctl -n vm.loadavg
} >"$OUT/env.txt" 2>&1
GGUF=$(python3 tools/fetch_bench_model.py --url "$URL" --sha256 "$SHA")
git fetch --quiet origin "+refs/heads/${BASE#origin/}:refs/remotes/$BASE" \
    "+refs/heads/${NEW#origin/}:refs/remotes/$NEW"
git rev-parse "$BASE" "$NEW" >"$OUT/shas.txt"
python3 tools/bench_revision_ab.py --rev "base=$BASE" --rev "new=$NEW" --gguf "$GGUF" \
    --make-arg TARGET=mac-omp --make-arg "BACKENDS=cpu_neon cpu_scalar metal" \
    --seq-lens "$SEQS" --decode-n 128 --warmup 16 --cycles "$CYCLES" \
    --env GEIST_BENCH_BACKEND=metal --out-dir "$OUT/ab" 2>&1 | tee "$OUT/ab.log"
cat "$OUT"/ab/*/revision_ab.md >"$OUT/revision_ab.md"
