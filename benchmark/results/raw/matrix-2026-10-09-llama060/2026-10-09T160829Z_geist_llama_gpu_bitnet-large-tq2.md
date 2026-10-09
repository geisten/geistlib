# Current GPU head-to-head: geist vs llama.cpp

Model: `bitnet_b1_58-large-TQ2_0.gguf` (`281aafb18a9f4a3124c10a1d8683e2296f0cfe8a2944da0a5667d17488a951bb`)

geist `8489cb953774`; llama.cpp `d81235049384`.

Host profile `nvidia_2080ti_vulkan`: 4-thread prefill, 4-thread decode, geist on `vulkan`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. Full GPU offload on NVIDIA GeForce RTX 2080 Ti, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 8054.11 ± 58.45 | 8344.85 ± 1377.74 | -3.48% | 349.44 ± 1.19 | 385.10 ± 4.00 | -9.26% |
| 256 | 7393.61 ± 5.23 | 11388.25 ± 1849.02 | -35.08% | 348.55 ± 0.68 | 379.31 ± 2.72 | -8.11% |
| 512 | 6202.87 ± 4.51 | 14074.15 ± 40.05 | -55.93% | 342.00 ± 0.51 | 368.94 ± 1.38 | -7.30% |
| 1024 | 4693.29 ± 8.13 | 13141.45 ± 39.95 | -64.29% | 324.45 ± 0.30 | 350.92 ± 2.82 | -7.54% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
