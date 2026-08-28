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

## Mac (M-series, 8 threads, CPU-only) — measured 2026-08-27, geist #295 branch

Full #287 + #288 + #289 + #291 + #295 stack (chunked delta-rule
prefill, parallel dequant+SGEMM prefill, Q6_K x8 lm_head GEMV, Q4_0
x8 decode GEMV, int8 mN GEMM on the x8 layout). Pre-stack reference
(2026-08-24): prefill 0.8B 139.1, 4B 44.1/40.5, 27B 6.6; decode
0.8B 91.8, 4B 22.4, 27B 4.3.

| model | metric | geist | llama.cpp CPU | ratio |
| :-- | :-- | ---: | ---: | :-- |
| 0.8B Q8_0 | prefill pp128 | 467.3 | 496.7 | llama 1.06× |
| 0.8B Q8_0 | prefill pp512 | 448.8 | — | |
| 0.8B Q8_0 | **decode** | **99.9** | 77.9 | **geist 1.28×** |
| 4B Q4_0 | prefill pp128 | 114.0 | 142.4 | llama 1.25× |
| 4B Q4_0 | prefill pp512 | 113.3 | 144.9 | llama 1.28× |
| 4B Q4_0 | **decode** | **27.6** | 27.8 | ~parity |
| 27B Q4_0 | prefill pp128 | 18.4 | 26.5 | llama 1.44× |
| 27B Q4_0 | **decode** | **5.7** | 5.8 | ~parity |

Decode is measured after a 128/512-token prefill (real KV + recurrent
state); 4B decode at 512 context is 25.5. RSS includes the packed x8
copies (#289 lm_head + #291 all Q4_0 projections): 4B 5.7 GB,
27B 27.7 GB — `GEIST_Q4_0_X8_GEMV=0` / `GEIST_Q6K_X8_GEMV=0` trade
the speed back for the memory. #291 A/B, same run: 4B decode
27.1 vs 25.4 (+7 %), 27B 5.6 vs 4.9 (+14 %); prefill untouched.

### Transactional speculative-state baseline (#296 PR 2) — 2026-08-28

Before wiring the MTP head, `bench_speculative` was run with the existing
n-gram drafter and `GEIST_SPEC_MIN_L=1`, deliberately forcing the new
DeltaNet checkpoint/restore path. This is a transaction-cost baseline, not
an MTP result. Greedy output equivalence is covered separately by
`test_speculative_primitives_int`, including byte-identical conv and S state
for every accepted prefix (0..4 on 0.8B, 0..2 on 27B).

| model | sequential | transactional spec | speedup | tokens/spec call |
| :-- | ---: | ---: | ---: | ---: |
| 0.8B Q8_0 | 31.7 tok/s | 19.3 tok/s | 0.61× | 4.63 |
| 27B Q4_0 | 1.7 tok/s | 1.1 tok/s | 0.67× | 1.74 |

The forced n-gram path is a loss on both sizes. On 27B, a lazy checkpoint is
about 150 MiB and low draft acceptance compounds that cost. PR 2 therefore
establishes correctness and bounded memory; later MTP PRs must beat this
baseline through higher acceptance and should avoid replay on the common
full-accept path.

(llama.cpp Metal, for context: 4B pp128 865 / tg 69.5; 27B pp512 106 /
tg 19.0. GPU is out of scope for geist — see the #281 positioning.)

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
