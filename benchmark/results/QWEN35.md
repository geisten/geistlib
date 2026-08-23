# geist Benchmarks — qwen35 hybrid family (Qwen3.5 / 3.8)

The qwen35 models are hybrids: 3 of every 4 layers are gated-DeltaNet linear
attention (recurrent conv + delta-rule state), the rest softmax attention.
That split shows up directly in the numbers — **decode** is recurrence-friendly
and competitive with llama.cpp already, while **prefill** pays for geist's
phase-1 *sequential* per-token recurrence (llama.cpp runs the chunked
formulation; that is the tracked next step in #281).

> ⚠️ Same measurement rules as [PI5.md](PI5.md): quiesced machine, warmed
> caches, and on the Pi a **thermally settled** board — the per-sweep
> temperatures are logged below. All numbers CPU-only, identical GGUF files
> for both engines.

## Setup

- **Mac:** Apple M-series (8 P-cores used), macOS. geist `make` (clang,
  Accelerate GEMM, OpenMP). Reference: llama.cpp `3fc4e10` (b9820, Homebrew),
  `llama-bench -ngl 0 -t 8` (CPU-only; its Metal numbers are noted separately).
- **Pi 5:** Model B Rev 1.1, 4× Cortex-A76, 4 GB, 64-bit RPi OS. geist
  `make TARGET=pi5 CC=gcc`. Reference: llama.cpp `acd79d6` on-board build (OpenBLAS, `-t 4`).
- **Models:** Qwen3.5-0.8B **Q8_0** (`make fetch-qwen35-model`, SHA-pinned;
  the CI fixture) and Qwen3.5-4B **Q4_0** (unsloth). The 27B (Mac only,
  15 GiB) is Qwen3.8-27B **Q4_0**.
- **geist protocol:** `bench_perf_sweep --gguf <M> --seq-lens ... --decode-n ...
  --warmup ... --repeats ...` — mean over repeats after a discarded warm-up.
  Thread pool + wait policy are set by the backend since #286 — **no env
  variables needed** (that alone was 6.1 → 22.3 t/s on the 4B; see the PR).
- **llama protocol:** `llama-bench -ngl 0 -p 128[,512] -n 16|32 -r 2..3`.

## Mac (M-series, 8 threads, CPU-only) — measured 2026-08-24

| model | metric | geist | llama.cpp CPU | ratio |
| :-- | :-- | ---: | ---: | :-- |
| 0.8B Q8_0 | prefill pp128 | 139.1 | 496.7 | llama 3.6× |
| 0.8B Q8_0 | **decode** | **91.8** | 77.9 | **geist 1.18×** |
| 4B Q4_0 | prefill pp128 | 44.1 | 142.4 | llama 3.2× |
| 4B Q4_0 | prefill pp512 | 40.5 | 144.9 | llama 3.6× |
| 4B Q4_0 | **decode** | 22.4 | 27.8 | llama 1.24× |
| 27B Q4_0 | prefill pp128 | 6.6 | 26.5 | llama 4.0× |
| 27B Q4_0 | **decode** | 4.3 | 5.8 | llama 1.35× |

(llama.cpp Metal, for context: 4B pp128 865 / tg 69.5; 27B pp512 106 /
tg 19.0. GPU is out of scope for geist — see the #281 positioning.)

**Reading:** decode is already at or past parity on the small model and
within 1.2–1.4× on the larger ones; the recurrence itself vectorizes well
and the W8A8 Q4_0/Q4_1 kernels (#285) closed the FFN gap. Prefill is
consistently ~3–4× behind — entirely the sequential delta-rule prefill;
the chunked formulation (spec §5b in #281) converts that into GEMM work.

## Raspberry Pi 5 (4 GB, quiesced, thermally gated) — measured 2026-08-24

geist `9b36c1d`, `TARGET=pi5`, gcc 14.2. Sweep temperatures: 0.8B started
at 48.5 °C, ended 60.6 °C; the 4B sweep started after cooling to 51.3 °C,
ended 65.0 °C (soft limit not reached; low throttle nibble 0 throughout).

| model | metric | geist | llama.cpp (on-board) | notes |
| :-- | :-- | ---: | ---: | :-- |
| 0.8B Q8_0 | prefill pp128 | 61.8 | 90.6 | RSS 1.0 GB |
| 0.8B Q8_0 | prefill pp256 | 62.3 | — | |
| 0.8B Q8_0 | **decode** | **13.0** | 9.3 | |
| 4B Q4_0 | prefill pp128 | 8.2 | 24.8 | RSS 2.9 GB — fits the 4 GB board |
| 4B Q4_0 | **decode** | 3.8 | 3.3 | |

**Reading:** on the design target, **geist wins decode on both models** —
0.8B 13.0 vs 9.3 t/s (**1.39×**), 4B 3.8 vs 3.3 t/s (**1.14×**) — the same
pattern as the BitNet results: specialized int8 kernels plus no per-layer
graph overhead. Prefill stays llama's (1.5× / 3.0×) until the chunked
delta-rule prefill lands. The 0.8B hybrid is fully usable on the Pi
(13 t/s decode, 1 GB resident); the 4B fits the 4 GB board and runs at
reading speed. The 27B (15 GiB) is not attempted on the Pi.

## Reproduce

```sh
# Mac
make && make fetch-qwen35-model
bin/mac-omp/release/tests/bench_perf_sweep \
  --gguf gguf_artifacts/qwen3.5-0.8b-q8_0.gguf \
  --seq-lens 128 --decode-n 32 --warmup 16 --repeats 5
llama-bench -m gguf_artifacts/qwen3.5-0.8b-q8_0.gguf -ngl 0 -t 8 -p 128 -n 32 -r 3

# Pi 5 (quiesced, < 60 °C between engines; see PI5.md)
make TARGET=pi5 CC=gcc && make fetch-qwen35-model
bin/pi5/release/tests/bench_perf_sweep \
  --gguf gguf_artifacts/qwen3.5-0.8b-q8_0.gguf \
  --seq-lens 128,256 --decode-n 16 --warmup 8 --repeats 5
```
