# Current CPU head-to-head: geist vs llama.cpp

Model: `gemma4-e2b-Q4_K_M.gguf` (`740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8`)

geist `8489cb953774`; llama.cpp `d81235049384`.

Host profile `amd_9950x`: 16-thread prefill, 15-thread decode, geist on `cpu_x86`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 525.33 ± 2.63 | 591.97 ± 4.72 | -11.26% | 52.00 ± 0.37 | 44.46 ± 0.14 | +16.95% |
| 256 | 529.18 ± 1.84 | 619.95 ± 4.33 | -14.64% | 51.63 ± 0.24 | 43.67 ± 0.23 | +18.23% |
| 512 | 528.04 ± 1.15 | 626.65 ± 2.67 | -15.74% | 51.49 ± 0.23 | 43.32 ± 0.10 | +18.87% |
| 1024 | 521.80 ± 2.08 | 610.09 ± 3.81 | -14.47% | 51.14 ± 0.08 | 43.04 ± 0.07 | +18.81% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
