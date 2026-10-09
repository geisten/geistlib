# Current CPU head-to-head: geist vs llama.cpp

Model: `qwen3.5-4b-q4_0.gguf` (`298fcb5fe7a77ccc79745ae24751560c5ac56874caff4bb39b1f2055bd72b8bb`)

geist `8489cb953774`; llama.cpp `d81235049384`.

Host profile `amd_9950x`: 16-thread prefill, 15-thread decode, geist on `cpu_x86`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 256.14 ± 2.34 | 303.76 ± 4.31 | -15.68% | 26.05 ± 0.04 | 23.13 ± 0.22 | +12.62% |
| 256 | 255.14 ± 3.96 | 313.12 ± 1.95 | -18.52% | 26.02 ± 0.02 | 23.06 ± 0.25 | +12.85% |
| 512 | 256.94 ± 1.77 | 314.37 ± 2.44 | -18.27% | 25.98 ± 0.02 | 22.92 ± 0.08 | +13.35% |
| 1024 | 256.14 ± 1.37 | 309.40 ± 3.79 | -17.21% | 25.95 ± 0.04 | 22.91 ± 0.22 | +13.29% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
