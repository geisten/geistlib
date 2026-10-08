# Current GPU head-to-head: geist vs llama.cpp

Model: `gemma4-e4b-Q4_K_M.gguf` (`85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87`)

geist `c696fbe7a144`; llama.cpp `2d8d612e4c68`.

Host profile `nvidia_2080ti_vulkan`: 4-thread prefill, 4-thread decode, geist on `vulkan`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. Full GPU offload on NVIDIA GeForce RTX 2080 Ti, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 330.53 ± 0.80 | 2110.19 ± 7.98 | -84.34% | 70.77 ± 0.05 | 98.51 ± 0.38 | -28.16% |
| 256 | 299.69 ± 0.81 | 2762.45 ± 4.40 | -89.15% | 76.28 ± 0.06 | 98.03 ± 0.22 | -22.18% |
| 512 | 271.81 ± 0.53 | 2851.80 ± 4.73 | -90.47% | 75.98 ± 0.05 | 97.26 ± 0.30 | -21.88% |
| 1024 | 264.81 ± 0.39 | 2669.93 ± 2.90 | -90.08% | 74.59 ± 0.03 | 96.85 ± 0.23 | -22.99% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
