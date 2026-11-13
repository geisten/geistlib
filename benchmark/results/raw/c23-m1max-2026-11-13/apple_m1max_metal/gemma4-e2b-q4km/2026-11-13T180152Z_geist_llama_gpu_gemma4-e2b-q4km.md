# Current GPU head-to-head: geist vs llama.cpp

Model: `gemma4-e2b-Q4_K_M.gguf` (`740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8`)

geist `6642260d55a1`; llama.cpp `d81235049384`.

Host profile `apple_m1max_metal`: 4-thread prefill, 4-thread decode, geist on `metal`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. Full GPU offload on Apple M1 Max, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: Apple M1 Max (8 physical cores), Darwin 27.2.0.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 1387.02 ± 4.63 | 1404.62 ± 2.88 | -1.25% | 120.70 ± 1.18 | 99.31 ± 0.10 | +21.55% |
| 256 | 1557.58 ± 2.71 | 1524.34 ± 2.02 | +2.18% | 119.87 ± 1.15 | 99.33 ± 0.17 | +20.67% |
| 512 | 1556.16 ± 2.41 | 1549.73 ± 0.86 | +0.42% | 118.44 ± 1.00 | 99.25 ± 0.12 | +19.33% |
| 1024 | 1540.98 ± 2.26 | 1484.78 ± 1.42 | +3.79% | 117.19 ± 1.48 | 97.46 ± 0.12 | +20.25% |

Every run started on AC power with Low Power Mode off (power gate, pmset); each run's machine state records it.

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
