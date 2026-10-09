# Current GPU head-to-head: geist vs llama.cpp

Model: `llama-3.2-3b-Q4_K_M.gguf` (`6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff`)

geist `8489cb953774`; llama.cpp `d81235049384`.

Host profile `nvidia_2080ti_vulkan`: 4-thread prefill, 4-thread decode, geist on `vulkan`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. Full GPU offload on NVIDIA GeForce RTX 2080 Ti, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 2621.77 ± 8.99 | 3724.74 ± 28.33 | -29.61% | 156.27 ± 0.21 | 192.82 ± 0.17 | -18.95% |
| 256 | 4065.34 ± 12.08 | 4168.71 ± 10.67 | -2.48% | 154.31 ± 0.14 | 183.34 ± 1.34 | -15.84% |
| 512 | 4910.19 ± 17.79 | 4727.09 ± 10.71 | +3.87% | 150.85 ± 0.11 | 180.50 ± 1.13 | -16.42% |
| 1024 | 4695.55 ± 3.52 | 4556.19 ± 2.30 | +3.06% | 149.09 ± 0.06 | 177.79 ± 1.10 | -16.15% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
