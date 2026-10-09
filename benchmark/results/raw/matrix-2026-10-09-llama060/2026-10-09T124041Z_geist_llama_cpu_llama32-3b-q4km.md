# Current CPU head-to-head: geist vs llama.cpp

Model: `llama-3.2-3b-Q4_K_M.gguf` (`6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff`)

geist `8489cb953774`; llama.cpp `d81235049384`.

Host profile `amd_9950x`: 16-thread prefill, 15-thread decode, geist on `cpu_x86`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 381.00 ± 4.69 | 494.14 ± 2.97 | -22.90% | 37.48 ± 0.09 | 35.53 ± 0.02 | +5.48% |
| 256 | 380.14 ± 3.33 | 503.78 ± 2.52 | -24.54% | 37.36 ± 0.06 | 35.02 ± 0.08 | +6.69% |
| 512 | 378.91 ± 2.87 | 508.87 ± 1.37 | -25.54% | 37.01 ± 0.09 | 34.42 ± 0.05 | +7.51% |
| 1024 | 374.41 ± 2.21 | 494.68 ± 1.74 | -24.31% | 36.43 ± 0.08 | 33.20 ± 0.03 | +9.75% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
