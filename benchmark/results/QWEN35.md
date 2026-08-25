# geist Benchmarks — qwen35 hybrid family (Qwen3.5 / 3.8)

The qwen35 models are hybrids: 3 of every 4 layers are gated-DeltaNet linear
attention (recurrent conv + delta-rule state), the rest softmax attention.
**Decode** is recurrence-friendly and competitive with llama.cpp; the
original 3–4× **prefill** gap turned out to be the mN linear kernels, not
the recurrence (#287 falsified that attribution) — #288 closed most of it
on Mac by unserializing the dequant+SGEMM prefill path.

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

## Mac (M-series, 8 threads, CPU-only) — measured 2026-08-25, geist `16b5807`

Post-#288 (parallel dequant+SGEMM prefill for Q8_0/Q4_0). Pre-#288
prefill for reference: 0.8B 139.1, 4B 44.1/40.5, 27B 6.6.

| model | metric | geist | llama.cpp CPU | ratio |
| :-- | :-- | ---: | ---: | :-- |
| 0.8B Q8_0 | prefill pp128 | 389.3 | 496.7 | llama 1.28× |
| 0.8B Q8_0 | prefill pp512 | 383.0 | — | |
| 0.8B Q8_0 | **decode** | **94.0** | 77.9 | **geist 1.21×** |
| 4B Q4_0 | prefill pp128 | 84.0 | 142.4 | llama 1.70× |
| 4B Q4_0 | prefill pp512 | 79.4 | 144.9 | llama 1.83× |
| 4B Q4_0 | **decode** | 22.6 | 27.8 | llama 1.23× |
| 27B Q4_0 | prefill pp128 | 13.2 | 26.5 | llama 2.0× |
| 27B Q4_0 | **decode** | 4.5 | 5.8 | llama 1.29× |

(llama.cpp Metal, for context: 4B pp128 865 / tg 69.5; 27B pp512 106 /
tg 19.0. GPU is out of scope for geist — see the #281 positioning.)

**Reading:** decode is at or past parity on the small model and within
1.2–1.3× on the larger ones. Prefill went from 3–4× behind to 1.3–2×
via two findings: chunking the delta-rule recurrence (#287) moved it
only ~0–7 % — falsifying the original recurrence attribution — while
the real culprit was the *serial* Mac dequant+SGEMM tile loop, fixed
in #288. The remaining 1.3–2× is mN quantized-GEMM throughput
(per-call dequant+SGEMM vs llama.cpp's repacked int8 GEMM) — the next
lever if prefill matters more than it does today.

### Chunked delta-rule prefill (#287) — measured 2026-08-25, pre-#288

A/B on the same build, `GEIST_DN_SEQ_PREFILL=1` forcing the sequential
path (Mac, 8 threads, load < 2 at start):

| model | metric | chunked | sequential | Δ |
| :-- | :-- | ---: | ---: | :-- |
| 0.8B Q8_0 | prefill pp128 | 145.3 | 138.3 | +5 % |
| 0.8B Q8_0 | prefill pp512 | 146.6 | 136.6 | +7 % |
| 4B Q4_0 | prefill pp128 | 40.0 | 42.3 | −5 % (noisy run) |
| 4B Q4_0 | prefill pp512 | 42.8 | 41.8 | +2.5 % |

The chunked recurrence is kept: correct (pinned by
test_deltanet_chunk_unit at f32 precision), never slower outside
noise, and load-bearing now that #288 stopped the mN kernels from
dominating prefill time.

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
graph overhead. Prefill stays llama's (1.5× / 3.0×); per the #287
findings above, closing it is an mN-kernel problem, not a recurrence
problem (Pi A/B of the chunked prefill pending — the balance may
differ on 4 cores without Accelerate). The 0.8B hybrid is fully usable on the Pi
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
