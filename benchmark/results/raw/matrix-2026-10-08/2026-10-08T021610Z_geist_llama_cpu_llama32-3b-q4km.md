# Current CPU head-to-head: geist vs llama.cpp

Model: `llama-3.2-3b-Q4_K_M.gguf` (`6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff`)

geist `c696fbe7a144`; llama.cpp `2d8d612e4c68`.

Host profile `amd_9950x`: 16-thread prefill, 15-thread decode, geist on `cpu_x86`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 331.05 ± 50.54 | 399.25 ± 17.84 | -17.08% | 36.98 ± 0.52 | 35.08 ± 0.57 | +5.40% |
| 256 | 346.58 ± 35.56 | 396.63 ± 18.34 | -12.62% | 36.67 ± 0.62 | 34.62 ± 0.56 | +5.93% |
| 512 | 329.39 ± 47.96 | 396.74 ± 14.98 | -16.98% | 36.21 ± 0.62 | 33.82 ± 0.55 | +7.07% |
| 1024 | 342.14 ± 32.69 | 387.85 ± 16.95 | -11.78% | 35.57 ± 0.62 | 32.60 ± 0.63 | +9.09% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
