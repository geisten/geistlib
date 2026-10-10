# Current GPU head-to-head: geist vs llama.cpp

Model: `qwen3-0.6b-q8_0.gguf` (`9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`)

geist `8489cb953774`; llama.cpp `d81235049384`.

Host profile `nvidia_2080ti_vulkan`: 4-thread prefill, 4-thread decode, geist on `vulkan`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. Full GPU offload on NVIDIA GeForce RTX 2080 Ti, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 9423.20 ± 8.68 | 10392.55 ± 1607.60 | -9.33% | 320.55 ± 2.12 | 390.44 ± 0.39 | -17.90% |
| 256 | 13971.51 ± 7.25 | 16610.00 ± 391.60 | -15.88% | 327.10 ± 0.60 | 357.01 ± 0.59 | -8.38% |
| 512 | 18552.74 ± 21.87 | 16598.85 ± 120.15 | +11.77% | 314.93 ± 0.13 | 345.20 ± 2.78 | -8.77% |
| 1024 | 16645.53 ± 12.58 | 15263.70 ± 24.80 | +9.05% | 305.76 ± 0.23 | 331.67 ± 3.90 | -7.81% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
