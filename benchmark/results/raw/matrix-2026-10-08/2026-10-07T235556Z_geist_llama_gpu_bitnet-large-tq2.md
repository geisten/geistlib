# Current GPU head-to-head: geist vs llama.cpp

Model: `bitnet_b1_58-large-TQ2_0.gguf` (`281aafb18a9f4a3124c10a1d8683e2296f0cfe8a2944da0a5667d17488a951bb`)

geist `c696fbe7a144`; llama.cpp `2d8d612e4c68`.

Host profile `nvidia_2080ti_vulkan`: 4-thread prefill, 4-thread decode, geist on `vulkan`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. Full GPU offload on NVIDIA GeForce RTX 2080 Ti, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 1345.49 ± 17.53 | 7558.28 ± 1401.45 | -82.20% | 343.17 ± 4.15 | 351.69 ± 25.25 | -2.42% |
| 256 | 1334.86 ± 2.63 | 10080.99 ± 1710.43 | -86.76% | 343.35 ± 1.19 | 369.41 ± 13.82 | -7.05% |
| 512 | 1287.02 ± 6.28 | 12389.55 ± 164.25 | -89.61% | 333.76 ± 3.79 | 360.12 ± 8.42 | -7.32% |
| 1024 | 1190.66 ± 15.62 | 11811.00 ± 191.40 | -89.92% | 313.93 ± 4.94 | 345.10 ± 8.03 | -9.03% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
