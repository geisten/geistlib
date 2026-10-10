# Current GPU head-to-head: geist vs llama.cpp

Model: `qwen3.5-0.8b-q8_0.gguf` (`0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c`)

geist `8489cb953774`; llama.cpp `d81235049384`.

Host profile `nvidia_2080ti_vulkan`: 4-thread prefill, 4-thread decode, geist on `vulkan`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. Full GPU offload on NVIDIA GeForce RTX 2080 Ti, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 9040.51 ± 45.89 | 7507.16 ± 1122.14 | +20.43% | 325.97 ± 0.37 | 308.21 ± 1.34 | +5.76% |
| 256 | 12284.36 ± 33.21 | 11482.70 ± 55.40 | +6.98% | 327.11 ± 0.33 | 288.56 ± 1.72 | +13.36% |
| 512 | 15673.55 ± 17.52 | 12039.90 ± 16.25 | +30.18% | 326.37 ± 0.37 | 288.08 ± 1.78 | +13.29% |
| 1024 | 15414.37 ± 7.66 | 11937.05 ± 31.60 | +29.13% | 325.20 ± 0.30 | 286.41 ± 2.27 | +13.54% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
