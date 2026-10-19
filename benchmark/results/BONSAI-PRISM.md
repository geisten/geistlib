# Ternary Bonsai 2 27B: geistlib against the PrismML llama.cpp fork

**Result:** on `PQ2_0`, geistlib decodes faster than PrismML's own
llama.cpp fork on every machine measured: 1.53× on the M1 Max CPU, 1.14×
on the Ryzen 9 9950X, 1.38× on an RTX 2080 Ti (Vulkan) and 1.23× on the
M1 Max GPU (Metal). Prefill is ahead on both CPUs (1.18×, 1.28×) and on
Vulkan (1.14×), and behind on Metal (0.81×).

| Machine, backend | geistlib revision | Decode, short context (tok/s) | Prefill, 512 tokens (tok/s) |
| :-- | :-- | --: | --: |
| M1 Max, CPU (`cpu_neon`) | `f8da3d1f` (#83) | 9.97 vs 6.52, **1.53×** | 25.6 vs 21.6, **1.18×** |
| Ryzen 9 9950X, CPU (`cpu_x86`) | decode `ccd11418`, prefill `88b35a64` (#86) | 8.89 vs 7.83, **1.14×** | 58.1 vs 45.4, **1.28×** |
| RTX 2080 Ti, Vulkan | `ccd11418` | 41.3 vs 30.0, **1.38×** | 648 vs 569, **1.14×** |
| M1 Max, Metal | `30c00b32` (main, incl. #81) | 18.9 vs 15.4, **1.23×** | 88.5 vs 109.0, 0.81× |

geistlib first, the fork second; the ratio is geistlib divided by the fork.
The Metal prefill of the fork spreads widely between cycles (96 to 114
tokens/s, geistlib 87 to 96), so that ratio is the least precise in the
table.

## Setup

- Model: `prism-ml/Ternary-Bonsai-2-27B-gguf` @ `b072e1d3`,
  `Ternary-Bonsai-2-27B-PQ2_0.gguf`, 26.9 B parameters, 6.70 GiB, sha256
  `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`
  (pinned in `benchmark/cross_engine_gpu_protocol.json`).
- Fork: PrismML-Eng/llama.cpp `0082ac24a89ae62132041b40e2c0170e278a98f4`,
  built by `scripts/ensure-llama-bench.sh` (CPU, Metal, Vulkan variants);
  every log prints the commit recorded in the build's `.geist-built-from`.
- geistlib: `bench_perf_sweep --seq-lens 32,512 --decode-n 64 --warmup 1
  --repeats 2`. Decode is the 32-token-prompt row, prefill the 512 row.
- Fork: `llama-bench -p 512 -n 64 -d 0,512 -r 2` (`-ngl 99` on a GPU).
  Decode is the `-d 0` row, prefill `pp512`.
- Three cycles, the two engines interleaved within each cycle; the table
  gives the median of the three cycles. Exception: the Ryzen prefill row
  comes from the A/B run below (geistlib: median of six cycles; fork: mean of
  two `llama-bench` runs in the same job).
- M1 Max (10 CPU cores, Darwin 27.0.0) under the machine's
  benchmark lock. Ryzen 9 9950X and RTX 2080 Ti (driver 595.91.07) on the
  geist-vulkan runner under `scripts/gpu-lock.sh`, CI runs 1381 and 1384.

## Integer kernels against the float path (CPU prefill)

Before #83 and #86, the `PQ2_0` CPU prefill expanded the weights to fp32 and
called a general SGEMM. The two pull requests run it on int8 dot products
(NEON `SDOT` on the M1 Max, AVX-512 VNNI on the Ryzen).
`tools/bench_revision_ab.py`, six cycles in rotating order, with a control
run of the baseline binary:

| Machine | Baseline | New | Prefill 512 (ms) | tok/s | Greedy check |
| :-- | :-- | :-- | --: | --: | :-- |
| M1 Max, `cpu_neon` | `d5681183` (main) | `f8da3d1f` (#83) | 34079 → 16335, −51.8 % | 15.0 → 31.3 | 64 of 64 tokens identical |
| Ryzen 9 9950X, `cpu_x86` | main | `88b35a64` (#86) | 14336 → 8820, −38.2 % | 35.7 → 58.1 | 64 of 64 identical, KL 8.0e-5 (VNNI against forced AVX2) |

Decode is unchanged within noise in both runs.

## Raw data

`raw/bonsai-prism-2026-10-18/`:

- `m1max-cpu-metal-f8da3d1f.txt`: M1 Max, CPU rows (and the Metal rows of
  that build, superseded by the next file).
- `m1max-metal-main-30c00b32.txt` (and `-geist.jsonl`, `-fork.jsonl`): M1
  Max, Metal rows on main with #81.
- `geist-vulkan-run1381.log`: Ryzen decode and RTX 2080 Ti rows (full CI log).
- `geist-vulkan-run1384-x86-ab.log`: Ryzen A/B, greedy check and fork prefill.
- `m1max-cpu_neon-ab-f8da3d1f.md`, `.jsonl`: M1 Max A/B.
- `m1max-cpu_neon-greedy-f8da3d1f.txt`: M1 Max greedy check.
