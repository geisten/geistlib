# Current CPU head-to-head: geist vs llama.cpp

Model: `gemma4-e4b-Q4_K_M.gguf` (`85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87`)

geist `8489cb953774`; llama.cpp `d81235049384`.

Host profile `amd_9950x`: 16-thread prefill, 15-thread decode, geist on `cpu_x86`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 264.22 ± 4.27 | 317.29 ± 1.30 | -16.73% | 26.13 ± 0.11 | 22.92 ± 0.03 | +14.00% |
| 256 | 268.20 ± 1.23 | 320.40 ± 2.16 | -16.29% | 26.14 ± 0.05 | 22.71 ± 0.03 | +15.11% |
| 512 | 267.45 ± 1.34 | 328.55 ± 1.58 | -18.60% | 26.07 ± 0.05 | 22.42 ± 0.05 | +16.28% |
| 1024 | 261.60 ± 4.29 | 322.94 ± 3.76 | -18.99% | 25.98 ± 0.08 | 22.34 ± 0.06 | +16.31% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
