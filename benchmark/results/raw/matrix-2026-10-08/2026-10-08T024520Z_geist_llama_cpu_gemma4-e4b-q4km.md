# Current CPU head-to-head: geist vs llama.cpp

Model: `gemma4-e4b-Q4_K_M.gguf` (`85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87`)

geist `c696fbe7a144`; llama.cpp `2d8d612e4c68`.

Host profile `amd_9950x`: 16-thread prefill, 15-thread decode, geist on `cpu_x86`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 269.02 ± 1.14 | 262.50 ± 0.99 | +2.48% | 26.19 ± 0.05 | 22.95 ± 0.06 | +14.13% |
| 256 | 268.93 ± 0.71 | 266.22 ± 1.65 | +1.02% | 26.09 ± 0.05 | 22.71 ± 0.05 | +14.87% |
| 512 | 266.57 ± 0.85 | 268.54 ± 0.89 | -0.74% | 26.00 ± 0.03 | 22.51 ± 0.05 | +15.51% |
| 1024 | 264.76 ± 0.67 | 265.15 ± 0.94 | -0.15% | 25.93 ± 0.06 | 22.32 ± 0.03 | +16.14% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
