# geist Benchmarks

geist vs **llama.cpp** and Microsoft's **bitnet.cpp** — the **identical** GGUF on
both engines, each at its best thread count, after a discarded warm-up.

| | [Raspberry Pi 5](results/PI5.md) | [Apple M1 Max](results/APPLE.md) |
| :-- | :-- | :-- |
| Role | **primary optimization target** (edge) | dev box / Apple-AMX reference |
| Quant matmul path | native int8 (W4A8) | native int8 |
| Dense fp32 path | native NEON (BLAS-free) | Accelerate / **AMX** |
| Measurement | quiesced + cool → **mean of 10** | live desktop → **best of 10** |

> **TL;DR** — On **Apple Silicon** geist wins prefill at *every* length and the
> lead **widens** with context (1.48× at 1024 tokens). On the **Pi 5** llama.cpp's
> OpenBLAS prefill leads geist by ~10–15 % across the sweep (both flat, ~37–39 vs
> ~32–34 t/s); **decode ties on the sweep and edges ahead** with the spec-decode
> head (7.5 vs 6.8 t/s). geist's Pi value is the dependency-free static binary +
> decode parity, not raw prefill — the A76's mature OpenBLAS fp32 path is the bar
> geist is still chasing there. On **ternary BitNet** geist decodes ~2× bitnet.cpp
> on the Pi 5 and beats it on x86 too.

## Don't trust the numbers — run them

Every published speedup is one command away from being checked on *your*
hardware:

```sh
make bench
```

It downloads the BitNet GGUF, runs a frozen protocol (warm-up, then 10 measured
repeats at 32/128/512-token prompts), and — if it finds a `llama.cpp` or
`bitnet.cpp` binary on your machine — measures **that engine in the same run,
against the byte-identical GGUF, after the board has cooled back to the
temperature geist started from**. It prints run-to-run spread alongside every
number and writes nothing to the repository. Absolute t/s is hardware-bound;
the **ratio on your own box** is the number that travels.

Greedy output is bit-identical to the `cpu_scalar` reference before any speedup
is quoted — a faster engine that produces different tokens is not iso-quality.
The protocol is frozen in [`../tools/bench_reproduce.py`](../tools/bench_reproduce.py);
every row in [`reference_runs.json`](reference_runs.json) came from it.

## Historical headline CPU numbers

These rows are useful within each linked campaign, but they are **not one
cross-system table under identical conditions**: models, prompt/decode shapes,
engine commits and aggregation rules differ. The strict current-coverage table
is [results/MATRIX.md](results/MATRIX.md); it leaves incompatible or unmeasured
cells blank instead of combining them.

| model | platform | metric | **geistlib** | baseline |
| :-- | :-- | :-- | --: | --: |
| Gemma 4 E2B-it (Q4_K_M) | **Pi 5** | total t/s (32p+128d) | **8.8** | 8.2 *(llama.cpp)* |
| Gemma 4 E2B-it (Q4_K_M) | **Pi 5** | decode t/s | **7.5** | 6.8 *(llama.cpp)* |
| Gemma 4 E2B-it (Q4_K_M) | **M1 Max** | prefill t/s (pp1024) | **144** | 97 *(llama.cpp)* |
| BitNet b1.58 2B-4T (`i2_s`) | **Pi 5** | decode t/s (32p) | **17.9** | 9.1 *(bitnet.cpp)* |
| BitNet b1.58 2B-4T (`i2_s`) | **Pi 5** | decode t/s (512p) | **15.0** | — *(no like-for-like)* |
| BitNet b1.58 2B-4T (`i2_s`) | **Pi 5** | prefill t/s (512p) | 45.5 | 45.8 *(bitnet.cpp)* |
| BitNet b1.58 2B-4T (`i2_s`) | **AMD 9950X** | prefill t/s (pp128) | **1098** | 679.9 *(bitnet.cpp)* |
| BitNet b1.58 2B-4T (`i2_s`) | **AMD 9950X** | decode t/s (tg128) | **103.1** | 54.3 *(bitnet.cpp)* |
| Gemma 4 E2B-it (Q4_K_M) | **AMD 9950X** | prefill t/s | **512** | 495 *(llama.cpp)* |
| Gemma 4 E2B-it (Q4_K_M) | **AMD 9950X** | decode t/s | **48.6** | 44.1 *(llama.cpp)* |
| Llama 3.2 3B (Q4_K_M) | **AMD 9950X** | prefill t/s | **351** | 346 *(llama.cpp)* |
| Llama 3.2 3B (Q4_K_M) | **AMD 9950X** | decode t/s | 34.1 | 34.5 *(llama.cpp)* |

<sub>**Baseline versions:** llama.cpp `d05fe1d` (Pi 5, M1 Max) · `b9827` (x86) ·
bitnet.cpp = [microsoft/BitNet](https://github.com/microsoft/BitNet) `master`.
GPU (Metal/Vulkan) numbers: [`../docs/BACKENDS.md`](../docs/BACKENDS.md).
Sub-parity rows are kept on purpose — nothing cherry-picked.</sub>

## Which file answers which question

| Question | File |
|---|---|
| **How do I produce a trustworthy number?** | [METHODOLOGY.md](METHODOLOGY.md) — protocol, per-machine aggregation, quality/MMLU procedures, the correctness gate |
| Which results share the **same current protocol**? | [results/MATRIX.md](results/MATRIX.md) — strict coverage matrix; missing and incompatible cells remain explicit |
| What came out on the **Raspberry Pi 5**? | [results/PI5.md](results/PI5.md) — the edge target: full sweep, thread placement, int8-vs-OpenBLAS analysis |
| …on **Apple M1 Max**? | [results/APPLE.md](results/APPLE.md) — AMX reference, the auto-recorded `make bench-*` table, decode-kernel investigation |
| …on **AMD x86-64** (AVX-512)? | [results/X86.md](results/X86.md) — Gemma & Llama vs llama.cpp, BitNet vs bitnet.cpp |
| …on the **Apple GPU** (Metal, experimental)? | [results/METAL.md](results/METAL.md) — M1 Max vs llama.cpp Metal, the two scalar-fallback fixes, and why the program closed |
| …on an **NVIDIA GPU** (Vulkan, experimental)? | [results/VULKAN.md](results/VULKAN.md) — RTX 2080 Ti vs llama.cpp Vulkan, incl. the measured dead ends |
| What about **ternary BitNet**? | [results/TERNARY.md](results/TERNARY.md) — supported formats, conversion, the per-token byte budget, the ternary measurement protocol |

## Running them

```sh
make bench-small        # quick check — records a table row + a raw JSONL record
make bench-detailed     # more repeats, tighter spread
make bench-mmlu         # quality (MMLU), self-contained
```

Raw run artifacts land in `~/bench-geistlib/` — outside the repo, so the working
tree stays clean. Read [METHODOLOGY.md](METHODOLOGY.md) before trusting a number:
throughput swings with core count, thread policy and board temperature, and a
recorded row states its own spread for exactly that reason.

## The tools here

| File | Purpose |
| :-- | :-- |
| `perf_gate.py` | CI cliff detector — fails the build on a catastrophic regression (scalar fallback, `-O0`, OpenMP off), not on drift |
| `total_tps.py` | cross-engine end-to-end measurement (geist vs llama.cpp), thermally quiesced on a Pi |
| `compare_ternary_pi5.sh` | three-engine ternary head-to-head: geist vs llama.cpp vs bitnet.cpp, one protocol |
| `compare_metal.sh` | cool-state Metal A/B against llama.cpp |
| `chart_headline.py` | renders `headline_results.json` into the README scoreboard SVG |
| `headline_results.json` | curated headline numbers — each row names its baseline engine and pinned version; sub-parity rows are kept on purpose |
