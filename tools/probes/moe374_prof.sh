#!/bin/sh
# moe374_prof.sh MODEL — the #374 profile (scripts/profile_374.sh) of
# gpt-oss-20b MXFP4 and DeepSeek-V2-Lite Q4_K_M on the Apple M6's CPU
# (cpu_neon), with llama-bench (pinned, CPU build) beside it. Run by
# mac-probe.yml under the host lock; MODEL is unused.
set -eu
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
{ sw_vers; sysctl -n machdep.cpu.brand_string; sysctl -n hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu 2>/dev/null; LC_ALL=C sysctl -n vm.loadavg; git rev-parse HEAD; } >"$OUT/env.txt" 2>&1
make TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar" -j8 lib bin >"$OUT/build.log" 2>&1
LLAMA=$(scripts/ensure-llama-bench.sh cpu 2>>"$OUT/build.log")
GPTOSS=$(python3 tools/fetch_bench_model.py --url https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/15dcfc7e7c608c6161507211ed43c5b590801b4b/gpt-oss-20b-MXFP4.gguf --sha256 27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901)
DEEPSEEK=$(python3 tools/fetch_bench_model.py --url https://huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF/resolve/0f37fdf276e8094747457f0ae4d40f2e8d2521f9/DeepSeek-V2-Lite.Q4_K_M.gguf --sha256 70985c39ba3e51a53e6ad6ae25ce29a3efc92b4802f61645760e1bc1eb89ad1a)
T=$(sysctl -n hw.ncpu)
for m in "$GPTOSS" "$DEEPSEEK"; do
    sh scripts/profile_374.sh bin/mac-omp/release/tests/bench_perf_sweep "$m" cpu_neon "$T" "$LLAMA" 2>&1 | tee -a "$OUT/profile.txt"
done
{ echo '```'; grep -E '^==|^--|prefill_tps|moe util|sinks attention|  attention|  ffn |  experts|avg_ts' "$OUT/profile.txt"; echo '```'; } >"$OUT/profile.md"
