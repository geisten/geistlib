#!/bin/sh
# ds_m6.sh MODEL NEW [ROUNDS] — #376 on the M6: the Metal unit tests of NEW
# with the tensor-API path off and on (GEIST_METAL_TENSOR=0/1), then geist
# (NEW, Metal) against llama.cpp Metal (scripts/ensure-llama-bench.sh's pin)
# on DeepSeek-V2-Lite and GigaChat3 Q4_K_M, pp512 / tg128, ROUNDS (default 3)
# alternating rounds. MODEL (mac-probe.yml's) is unused. Run by
# .gitea/workflows/mac-probe.yml under the host lock; writes to $PROBE_OUT.
set -u
shift
NEW=${1:?NEW revision, e.g. origin/agent/issue-376}
ROUNDS=${2:-3}
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
{
    sw_vers
    sysctl -n machdep.cpu.brand_string
    echo "memory: $(($(sysctl -n hw.memsize) / 1073741824)) GiB"
    LC_ALL=C sysctl -n vm.loadavg
} >"$OUT/env.txt" 2>&1
git fetch --quiet origin "+refs/heads/${NEW#origin/}:refs/remotes/$NEW"
WT="$(mktemp -d)/new"
git worktree add --detach "$WT" "$NEW" >"$OUT/worktree.log" 2>&1
git -C "$WT" rev-parse HEAD >"$OUT/new_sha.txt"
(cd "$WT" && make -j8 TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar metal" bin) >"$OUT/build.log" 2>&1 || {
    echo "build of $NEW failed" >"$OUT/summary.md"
    exit 1
}
{
    echo "## #376 on the M6: $NEW ($(cat "$OUT/new_sha.txt"))"
    echo
    echo "| test | GEIST_METAL_TENSOR | exit |"
    echo "|---|--:|--:|"
} >"$OUT/summary.md"
for t in test_deepseek2_unit test_moe_device_unit test_backend_metal_parity_unit; do
    for v in 0 1; do
        GEIST_METAL_TENSOR=$v "$WT/bin/mac-omp/release/tests/$t" >"$OUT/${t}_tc$v.log" 2>&1
        echo "| $t | $v | $? |" >>"$OUT/summary.md"
    done
done
LB=$(scripts/ensure-llama-bench.sh metal 2>"$OUT/llama-build.log") || {
    echo "llama-bench build failed" >>"$OUT/summary.md"
    exit 1
}
DS=$(python3 tools/fetch_bench_model.py \
    --url https://huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF/resolve/0f37fdf276e8094747457f0ae4d40f2e8d2521f9/DeepSeek-V2-Lite.Q4_K_M.gguf \
    --sha256 70985c39ba3e51a53e6ad6ae25ce29a3efc92b4802f61645760e1bc1eb89ad1a)
GC=$(python3 tools/fetch_bench_model.py \
    --url https://huggingface.co/bartowski/ai-sage_GigaChat3-10B-A1.8B-GGUF/resolve/5f329324103300ba9aff1e2fc68ddbb8a682a2c0/ai-sage_GigaChat3-10B-A1.8B-Q4_K_M.gguf \
    --sha256 88505918a1af4e2174994e406ae5a6a06c5867227796a4370c18321c6481e9a1)
{
    echo
    echo "| model | round | engine | pp512 tok/s | tg128 tok/s |"
    echo "|---|--:|---|--:|--:|"
} >>"$OUT/summary.md"
for m in "$DS" "$GC"; do
    name=$(basename "$m" .gguf)
    i=1
    while [ "$i" -le "$ROUNDS" ]; do
        sleep 20
        "$LB" -m "$m" -p 512 -n 128 -r 3 -ngl 99 -o json >"$OUT/llama_${name}_$i.json" 2>"$OUT/llama_${name}_$i.err"
        python3 - "$OUT/llama_${name}_$i.json" "$name" "$i" >>"$OUT/summary.md" <<'EOF'
import json, sys
rows = json.load(open(sys.argv[1]))
pp = next((r["avg_ts"] for r in rows if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0), 0.0)
tg = next((r["avg_ts"] for r in rows if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0), 0.0)
print(f"| {sys.argv[2]} | {sys.argv[3]} | llama.cpp | {pp:.1f} | {tg:.1f} |")
EOF
        sleep 20
        GEIST_BENCH_BACKEND=metal "$WT/bin/mac-omp/release/tests/bench_perf_sweep" --gguf "$m" \
            --seq-lens 1,512 --decode-n 128 --warmup 16 --repeats 3 --emit-jsonl \
            >"$OUT/geist_${name}_$i.jsonl" 2>"$OUT/geist_${name}_$i.err"
        python3 - "$OUT/geist_${name}_$i.jsonl" "$name" "$i" >>"$OUT/summary.md" <<'EOF'
import json, sys
rows = {r.get("seq_len"): r for r in (json.loads(l) for l in open(sys.argv[1]) if l.startswith("{"))}
pp = rows.get(512, {}).get("prefill_tps", 0.0)  # a 512-token prompt
tg = rows.get(1, {}).get("decode_tps", 0.0)  # 128 tokens after one, as tg128
print(f"| {sys.argv[2]} | {sys.argv[3]} | geist | {pp:.1f} | {tg:.1f} |")
EOF
        i=$((i + 1))
    done
done
git worktree remove --force "$WT" >/dev/null 2>&1 || true
