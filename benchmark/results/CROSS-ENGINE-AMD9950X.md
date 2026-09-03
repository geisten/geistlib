# Current CPU head-to-head: geist vs llama.cpp

Model: `gemma4-e2b-Q4_K_M.gguf` (`740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8`)

geist `7b08b590cc99`; llama.cpp `2d8d612e4c68`.

Host profile `amd_9950x`: 16-thread prefill, 15-thread decode, geist on `cpu_x86`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-30-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 515.68 ± 3.90 | 286.19 ± 18.74 | +80.19% | 50.59 ± 0.06 | 44.43 ± 0.16 | +13.86% |
| 256 | 510.69 ± 2.28 | 366.93 ± 6.86 | +39.18% | 50.18 ± 0.07 | 43.79 ± 0.14 | +14.61% |
| 512 | 491.93 ± 1.92 | 423.69 ± 9.52 | +16.11% | 49.54 ± 0.09 | 43.06 ± 0.13 | +15.05% |
| 1024 | 468.26 ± 1.81 | 408.22 ± 7.74 | +14.71% | 49.04 ± 0.05 | 42.87 ± 0.18 | +14.39% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.

## Reproducibility: two runs of the same protocol disagree by more than their MAD

This cell was measured twice, same source, same protocol, same host, 33 minutes
apart. Both artifacts are kept.

| cell | run A `2026-09-02T070548Z_g` | run B `2026-09-02T073807Z_g` | drift |
| :-- | --: | --: | --: |
| pp128 | +71.1% | +80.2% | +9.1 |
| tg128 | +12.9% | +13.9% | +1.0 |
| pp256 | +41.0% | +39.2% | -1.8 |
| tg256 | +14.7% | +14.6% | -0.1 |
| pp512 | +18.3% | +16.1% | -2.2 |
| tg512 | +14.6% | +15.1% | +0.4 |
| pp1024 | +17.4% | +14.7% | -2.7 |
| tg1024 | +14.3% | +14.4% | +0.1 |

geist reproduces almost exactly (pp512 491.93 vs 492.95 tok/s, decode within
0.1 tok/s). The drift is llama.cpp's prefill: 286.19 ± 18.74 versus
303.88 ± 9.34 tok/s at pp128, 6.5% MAD within run B alone.

The consequence matters more than the number: **the ± MAD in the table above
is the spread of 12 draws inside one run, and it understates the real
uncertainty.** Between-run variation at pp128 is 9 percentage points of ratio,
several times the within-run dispersion. Read pp512 as "geist leads by roughly
15–18%", not as a figure good to one decimal. A future revision of the protocol
should either treat a run as one draw and require replicate runs, or state the
between-run component explicitly. Until then, no cell in the matrix should be
quoted to a precision this data does not carry.

## Provenance

- protocol: [`cross_engine_cpu_protocol.json`](../cross_engine_cpu_protocol.json)
  (`2e27bafcc4903fe3`), host profile `amd_9950x`
- raw samples, every draw retained: [`raw/2026-09-02T073807Z_geist_llama_cpu_amd_9950x.jsonl`](raw/2026-09-02T073807Z_geist_llama_cpu_amd_9950x.jsonl)
  and [`raw/2026-09-02T070548Z_geist_llama_cpu_amd_9950x.jsonl`](raw/2026-09-02T070548Z_geist_llama_cpu_amd_9950x.jsonl)
- geist `7b08b590cc99`, binary `9a53a6297098ea8b`
- llama.cpp `2d8d612e4c68`, binary `8fba0648f0b58e8b`,
  built `-DGGML_NATIVE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS`, CPU-only
- geist built `make TARGET=linux` (default `cpu_x86 cpu_scalar`, OpenBLAS GEMM,
  `-march=x86-64-v3` with runtime AVX-512/VNNI dispatch)

Both engines therefore reach the same BLAS library and the same core count. The
thread profile is rule-based (physical cores for prefill, one fewer for decode),
matched between engines, and not tuned per engine — see the note in the protocol.

