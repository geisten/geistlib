# Current GPU head-to-head: geist vs llama.cpp

Model: `llama-3.2-3b-Q4_K_M.gguf` (`6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff`)

geist `c696fbe7a144`; llama.cpp `2d8d612e4c68`.

Host profile `nvidia_2080ti_vulkan`: 4-thread prefill, 4-thread decode, geist on `vulkan`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. Full GPU offload on NVIDIA GeForce RTX 2080 Ti, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 1358.09 ± 44.10 | 3583.05 ± 9.84 | -62.10% | 153.99 ± 2.05 | 192.50 ± 0.13 | -20.00% |
| 256 | 1351.77 ± 16.98 | 3956.59 ± 6.41 | -65.84% | 153.80 ± 1.02 | 185.22 ± 0.33 | -16.97% |
| 512 | 1269.92 ± 1.65 | 4624.78 ± 9.97 | -72.54% | 150.22 ± 0.74 | 183.47 ± 0.52 | -18.12% |
| 1024 | 1116.32 ± 0.89 | 4468.30 ± 11.66 | -75.02% | 145.53 ± 0.58 | 180.96 ± 0.62 | -19.58% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
