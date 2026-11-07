#!/bin/sh
# attn487_ab.sh MODEL NEW CYCLES SEQS WHICH [BASE] — #487 on the M6:
# tools/bench_revision_ab.py of NEW against BASE (default origin/main) on
# Metal, prompts of SEQS tokens (comma list) and 128 decoded tokens, a
# control copy of the base, cool gate on. WHICH picks the GGUF: go
# (gpt-oss-20b MXFP4), gc (GigaChat3-10B Q4_K_M), ds (DeepSeek-V2-Lite
# Q4_K_M) or model (mac-probe.yml's MODEL, e.g. qwen3-0.6b-q8). Run by
# .gitea/workflows/mac-probe.yml under the host lock; writes to $PROBE_OUT.
set -eu
MODEL=$1
shift
NEW=${1:?NEW revision, e.g. origin/agent/issue-487}
CYCLES=${2:-8}
SEQS=${3:-512,2048,4096}
WHICH=${4:-model}
BASE=${5:-origin/main}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
{
    sw_vers
    sysctl -n machdep.cpu.brand_string
    echo "memory: $(($(sysctl -n hw.memsize) / 1073741824)) GiB"
    LC_ALL=C sysctl -n vm.loadavg
} >"$OUT/env.txt" 2>&1
fetch() { python3 tools/fetch_bench_model.py --url "$1" --sha256 "$2"; }
case $WHICH in
go) GGUF=$(fetch https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/15dcfc7e7c608c6161507211ed43c5b590801b4b/gpt-oss-20b-MXFP4.gguf \
    27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901) ;;
gc) GGUF=$(fetch https://huggingface.co/bartowski/ai-sage_GigaChat3-10B-A1.8B-GGUF/resolve/5f329324103300ba9aff1e2fc68ddbb8a682a2c0/ai-sage_GigaChat3-10B-A1.8B-Q4_K_M.gguf \
    88505918a1af4e2174994e406ae5a6a06c5867227796a4370c18321c6481e9a1) ;;
ds) GGUF=$(fetch https://huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF/resolve/0f37fdf276e8094747457f0ae4d40f2e8d2521f9/DeepSeek-V2-Lite.Q4_K_M.gguf \
    70985c39ba3e51a53e6ad6ae25ce29a3efc92b4802f61645760e1bc1eb89ad1a) ;;
*) GGUF=$MODEL ;;
esac
git fetch --quiet origin "+refs/heads/${BASE#origin/}:refs/remotes/$BASE" \
    "+refs/heads/${NEW#origin/}:refs/remotes/$NEW"
git rev-parse "$BASE" "$NEW" >"$OUT/shas.txt"
python3 tools/bench_revision_ab.py --rev "base=$BASE" --rev "new=$NEW" --gguf "$GGUF" \
    --make-arg TARGET=mac-omp --make-arg "BACKENDS=cpu_neon cpu_scalar metal" \
    --seq-lens "$SEQS" --decode-n 128 --warmup 16 --cycles "$CYCLES" \
    --env GEIST_BENCH_BACKEND=metal --out-dir "$OUT/ab" 2>&1 | tee "$OUT/ab.log"
cat "$OUT"/ab/*/revision_ab.md >"$OUT/revision_ab.md"
cp "$OUT/revision_ab.md" "$OUT/summary.md"
