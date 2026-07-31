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

## Which file answers which question

| Question | File |
|---|---|
| **How do I produce a trustworthy number?** | [METHODOLOGY.md](METHODOLOGY.md) — protocol, per-machine aggregation, quality/MMLU procedures, the correctness gate |
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
