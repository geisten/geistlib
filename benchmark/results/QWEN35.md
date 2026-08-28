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

## Mac (M-series, 8 threads, CPU-only) — measured 2026-08-26, geist `f3ef6a8`

Full #287 + #288 + #289 + #291 stack (chunked delta-rule prefill,
parallel dequant+SGEMM prefill, Q6_K x8 lm_head GEMV, Q4_0 x8 decode
GEMV). Pre-stack reference (2026-08-24): prefill 0.8B 139.1,
4B 44.1/40.5, 27B 6.6; decode 0.8B 91.8, 4B 22.4, 27B 4.3.

| model | metric | geist | llama.cpp CPU | ratio |
| :-- | :-- | ---: | ---: | :-- |
| 0.8B Q8_0 | prefill pp128 | 467.3 | 496.7 | llama 1.06× |
| 0.8B Q8_0 | prefill pp512 | 448.8 | — | |
| 0.8B Q8_0 | **decode** | **99.9** | 77.9 | **geist 1.28×** |
| 4B Q4_0 | prefill pp128 | 88.7 | 142.4 | llama 1.61× |
| 4B Q4_0 | prefill pp512 | 89.7 | 144.9 | llama 1.62× |
| 4B Q4_0 | **decode** | **27.1** | 27.8 | ~parity (0.97×) |
| 27B Q4_0 | prefill pp128 | 13.5 | 26.5 | llama 2.0× |
| 27B Q4_0 | **decode** | **5.6** | 5.8 | ~parity (0.96×) |

Decode is measured after a 128/512-token prefill (real KV + recurrent
state); 4B decode at 512 context is 25.5. RSS includes the packed x8
copies (#289 lm_head + #291 all Q4_0 projections): 4B 5.7 GB,
27B 27.7 GB — `GEIST_Q4_0_X8_GEMV=0` / `GEIST_Q6K_X8_GEMV=0` trade
the speed back for the memory. #291 A/B, same run: 4B decode
27.1 vs 25.4 (+7 %), 27B 5.6 vs 4.9 (+14 %); prefill untouched.

(llama.cpp Metal, for context: 4B pp128 865 / tg 69.5; 27B pp512 106 /
tg 19.0. The geist Metal backend has since caught up — see the Metal
section below.)

**Reading:** the 0.8B beats llama.cpp on decode by 1.28× at prefill
parity, and the 4B/27B reach decode parity (0.96–0.97×) with #291's
interleaved Q4_0 GEMV (kernel-level +61–72 % GB/s, end-to-end +7–14 %
— the rest of the budget is the Q6_K head near its compute bound and
the DN recurrence). Prefill went
from 3–4× behind to 1.06–2× via two findings: chunking the delta-rule
recurrence (#287) moved it only ~0–7 % in isolation — falsifying the
original recurrence attribution — while the real culprit was the
*serial* Mac dequant+SGEMM tile loop, fixed in #288 (the chunked
recurrence became load-bearing once #288 landed: both together give
the 0.8B its 467 t/s). #289 then took the 4B/27B lm_head (Q6_K,
a third of 4B decode) from 33 to ~45 GB/s. Remaining levers, in
diminishing order: mN quantized-GEMM throughput on the 4B/27B
prefill (per-tile dequant+SGEMM vs llama.cpp's repacked int8 GEMM)
and Q4_0 m1 GEMV interleaving for the last ~20 % of large-model
decode.

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

## Mac Metal backend (M1 Max GPU) — measured 2026-08-28, `feat/296-qwen35-metal` @ `aa3eb4d`

The #300–#308 stack: simdgroup GEMV/GEMM kernels for
Q4_0/Q4_1/Q8_0/Q5_K/IQ4_XS/IQ4_NL/Q3_K/IQ3_S (llama mul_mv / mul_mm
structures), chunked DeltaNet prefill (port of the CPU chunk recipe;
its serial predecessor was 72 % of prefill wall), and the IQ4 loader
that makes the UD mixed quant run at all. Starting point on this
branch was 5.3 pp / 2.7 tg on the 27B (correctness-first kernels).

**Protocol:** cool state = 240 s GPU cooldown **and** a resident model —
the two fight each other when several 15 GiB files rotate through the
page cache, so every run is preceded by a full `cat model > /dev/null`
pre-touch. An eviction shows up as RSS below the model size and drags
both engines equally (measured: UD at 35 pp with 9 GB resident vs 79 pp
at 33 GB). `bench_perf_sweep` pp512/tg64, mean of 2 repeats after a
discarded 64-token warmup. Reference: llama.cpp Metal `3fc4e10`
(b9820, Homebrew), same GGUF, back-to-back on the same protocol.

| model | metric | geist Metal | llama.cpp Metal | ratio |
| :-- | :-- | ---: | ---: | :-- |
| 27B Q4_0 | prefill pp512 | **95.4** | 93.1 ±15 | ~parity |
| 27B Q4_0 | **decode tg64** | **12.1** | 8.2 | **geist 1.48×** |
| 27B UD-Q4_K_M | prefill pp512 | 78.7 | 105.2 ±7 | llama 1.34× |
| 27B UD-Q4_K_M | **decode tg64** | **7.8** | 8.1 | ~parity (0.97×) |
| 4B Q4_0 | prefill pp512 | 470.5 | 926 (warm) | llama ~2× |
| 4B Q4_0 | decode tg64 | 52.0 | 61.8 (warm) | llama 1.19× |
| gemma4-e2b Q4_K_M | prefill pp512 | 992.4 | 1540 (2026-07 ref) | llama 1.55× |
| gemma4-e2b Q4_K_M | decode tg64 | 79.3 | 92.8 (2026-07 ref) | llama 1.17× |

**Reading:** the 27B — the model this branch is for — beats llama.cpp
Metal on decode by 1.48× at prefill parity. The UD mixed quant reaches
decode parity; its prefill gap is the first-pass IQ4_XS GEMM (llama's
IQ kernels are mature). The 4B prefill gap is small-shape GEMM
efficiency plus the remaining DeltaNet chain (~360 ms per 512 tokens
after the barrier fix; simdgroup-MMA for the chunk matmuls and a panel
substitution are the documented follow-ups). gemma4 numbers are the
old 2026-07 program state restored (the PLE probe regression had
silently zeroed them) on the new kernel stack.

Attribution highlights, for whoever continues: prefill wall = max of
overlapping GPU chains, not their sum — single-category skips of the
subtractive profiler show nothing, combinations do
(`GEIST_METAL_PROFILE=1`, repeats=1 only). The whole 1.4 s DeltaNet
chain on the 4B was 2·C `mem_device` threadgroup barriers at ~54 µs
each, not compute. Half-staged GEMM weights make unpinned-scale parity
noise of √n·ulp size — the test pins are deliberate.

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
