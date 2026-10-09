# Current GPU head-to-head: geist vs llama.cpp

Model: `bitnet_b1_58-large-TQ2_0.gguf` (`281aafb18a9f4a3124c10a1d8683e2296f0cfe8a2944da0a5667d17488a951bb`)

geist `db7f140d5728`; llama.cpp `d81235049384`.

Host profile `nvidia_2080ti_vulkan`: 4-thread prefill, 4-thread decode, geist on `vulkan`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. Full GPU offload on NVIDIA GeForce RTX 2080 Ti, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 8564.74 ± 57.11 | 8280.00 ± 1362.44 | +3.44% | 348.84 ± 3.18 | 384.10 ± 3.58 | -9.18% |
| 256 | 11023.32 ± 6.65 | 11626.55 ± 1521.55 | -5.19% | 349.75 ± 1.21 | 379.12 ± 3.91 | -7.75% |
| 512 | 13261.16 ± 4.12 | 14180.25 ± 42.00 | -6.48% | 342.27 ± 1.02 | 367.19 ± 3.22 | -6.79% |
| 1024 | 11918.83 ± 7.07 | 13216.85 ± 28.10 | -9.82% | 325.50 ± 1.33 | 348.54 ± 4.87 | -6.61% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
