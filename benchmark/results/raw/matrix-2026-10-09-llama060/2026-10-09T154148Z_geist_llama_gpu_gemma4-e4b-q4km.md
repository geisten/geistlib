# Current GPU head-to-head: geist vs llama.cpp

Model: `gemma4-e4b-Q4_K_M.gguf` (`85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87`)

geist `8489cb953774`; llama.cpp `d81235049384`.

Host profile `nvidia_2080ti_vulkan`: 4-thread prefill, 4-thread decode, geist on `vulkan`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. Full GPU offload on NVIDIA GeForce RTX 2080 Ti, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 1555.95 ± 0.64 | 2364.07 ± 11.62 | -34.18% | 97.52 ± 0.08 | 104.15 ± 0.26 | -6.37% |
| 256 | 2551.17 ± 3.05 | 3019.61 ± 10.31 | -15.51% | 97.22 ± 0.03 | 103.37 ± 0.16 | -5.95% |
| 512 | 2940.65 ± 11.89 | 3034.45 ± 8.34 | -3.09% | 96.78 ± 0.10 | 102.74 ± 0.30 | -5.80% |
| 1024 | 2897.41 ± 3.41 | 2837.14 ± 7.74 | +2.12% | 96.36 ± 0.07 | 102.41 ± 0.11 | -5.90% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
