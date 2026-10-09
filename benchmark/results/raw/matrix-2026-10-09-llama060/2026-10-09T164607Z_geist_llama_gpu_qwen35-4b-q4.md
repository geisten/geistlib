# Current GPU head-to-head: geist vs llama.cpp

Model: `qwen3.5-4b-q4_0.gguf` (`298fcb5fe7a77ccc79745ae24751560c5ac56874caff4bb39b1f2055bd72b8bb`)

geist `8489cb953774`; llama.cpp `d81235049384`.

Host profile `nvidia_2080ti_vulkan`: 4-thread prefill, 4-thread decode, geist on `vulkan`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. Full GPU offload on NVIDIA GeForce RTX 2080 Ti, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 1994.33 ± 6.86 | 2685.28 ± 6.26 | -25.73% | 122.79 ± 0.12 | 124.95 ± 0.27 | -1.72% |
| 256 | 3012.88 ± 3.95 | 3369.34 ± 4.84 | -10.58% | 122.60 ± 0.18 | 124.93 ± 0.44 | -1.87% |
| 512 | 3520.07 ± 4.14 | 3356.98 ± 4.23 | +4.86% | 122.18 ± 0.12 | 124.29 ± 0.45 | -1.70% |
| 1024 | 3484.81 ± 4.85 | 3320.11 ± 6.05 | +4.96% | 121.43 ± 0.19 | 122.24 ± 0.30 | -0.66% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
