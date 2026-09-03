# Current CPU head-to-head: geist vs llama.cpp

Model: `gemma4-e2b-Q4KM-740185b2.gguf` (`740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8`)

geist `2d7f532c4745`; llama.cpp `c7bda030e7fa`.

Host profile `pi5`: 4-thread prefill, 3-thread decode, geist on `cpu_neon`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: Cortex-A76 (unavailable physical cores), Linux 6.18.33+rpt-rpi-2712.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 32.86 ± 0.13 | 34.85 ± 1.11 | -5.69% | 7.21 ± 0.06 | 6.61 ± 0.04 | +9.12% |
| 256 | 32.39 ± 0.07 | 36.66 ± 0.32 | -11.67% | 7.09 ± 0.01 | 6.39 ± 0.07 | +11.07% |
| 512 | 31.27 ± 0.07 | 35.76 ± 0.26 | -12.54% | 6.95 ± 0.01 | 6.27 ± 0.09 | +10.96% |
| 1024 | 29.94 ± 0.07 | 34.22 ± 0.39 | -12.51% | 6.85 ± 0.01 | 6.21 ± 0.06 | +10.39% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.

Thermal provenance: measured under the protocol's temp gate (measurements
start below 62 °C; the passively cooled board throttles at ~80 °C — an
ungated first attempt reached 80 °C mid-run and was discarded). Board:
Raspberry Pi 5 Model B Rev 1.1, 4 GB, reference host of PI5.md.
