# Current GPU head-to-head: geist vs llama.cpp

Model: `gemma4-e2b-Q4_K_M.gguf` (`740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8`)

geist `8489cb953774`; llama.cpp `d81235049384`.

Host profile `nvidia_2080ti_vulkan`: 4-thread prefill, 4-thread decode, geist on `vulkan`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. Full GPU offload on NVIDIA GeForce RTX 2080 Ti, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 2580.44 ± 14.46 | 3780.55 ± 41.18 | -31.74% | 158.71 ± 0.46 | 173.34 ± 0.49 | -8.44% |
| 256 | 4100.23 ± 6.34 | 4671.74 ± 10.53 | -12.23% | 159.20 ± 0.06 | 165.24 ± 0.84 | -3.66% |
| 512 | 5766.77 ± 19.98 | 5153.81 ± 5.17 | +11.89% | 158.31 ± 0.14 | 164.57 ± 0.33 | -3.80% |
| 1024 | 5597.85 ± 10.03 | 4614.71 ± 13.62 | +21.30% | 158.10 ± 0.39 | 163.40 ± 0.80 | -3.25% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
