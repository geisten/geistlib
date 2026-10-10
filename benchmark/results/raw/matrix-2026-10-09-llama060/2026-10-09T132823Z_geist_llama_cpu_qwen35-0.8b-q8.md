# Current CPU head-to-head: geist vs llama.cpp

Model: `qwen3.5-0.8b-q8_0.gguf` (`0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c`)

geist `8489cb953774`; llama.cpp `d81235049384`.

Host profile `amd_9950x`: 16-thread prefill, 15-thread decode, geist on `cpu_x86`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 1713.14 ± 12.55 | 982.88 ± 5.08 | +74.30% | 81.24 ± 0.39 | 69.26 ± 0.46 | +17.29% |
| 256 | 1752.91 ± 14.20 | 1014.18 ± 7.41 | +72.84% | 81.40 ± 0.49 | 69.92 ± 0.28 | +16.42% |
| 512 | 1756.37 ± 18.02 | 1022.97 ± 15.71 | +71.69% | 81.42 ± 0.41 | 69.76 ± 0.71 | +16.71% |
| 1024 | 1738.05 ± 28.38 | 1027.86 ± 6.56 | +69.09% | 80.60 ± 0.34 | 68.55 ± 0.50 | +17.57% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
