# Current CPU head-to-head: geist vs llama.cpp

Model: `qwen3-0.6b-q8_0.gguf` (`9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`)

geist `8489cb953774`; llama.cpp `d81235049384`.

Host profile `amd_9950x`: 16-thread prefill, 15-thread decode, geist on `cpu_x86`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 2034.86 ± 56.40 | 1599.43 ± 12.20 | +27.22% | 115.98 ± 1.67 | 105.61 ± 0.68 | +9.82% |
| 256 | 2044.32 ± 25.84 | 1664.16 ± 8.42 | +22.84% | 114.44 ± 1.19 | 106.45 ± 0.64 | +7.50% |
| 512 | 1950.94 ± 23.46 | 1655.59 ± 12.21 | +17.84% | 110.72 ± 0.54 | 101.89 ± 0.49 | +8.67% |
| 1024 | 1835.06 ± 10.31 | 1524.81 ± 10.77 | +20.35% | 104.84 ± 0.39 | 91.57 ± 0.39 | +14.49% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
