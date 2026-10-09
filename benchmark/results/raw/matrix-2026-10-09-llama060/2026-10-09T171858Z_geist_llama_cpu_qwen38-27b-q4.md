# Current CPU head-to-head: geist vs llama.cpp

Model: `qwen3.8-27b-q4_0.gguf` (`ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d`)

geist `8489cb953774`; llama.cpp `d81235049384`.

Host profile `amd_9950x`: 16-thread prefill, 15-thread decode, geist on `cpu_x86`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 39.00 ± 0.21 | 55.01 ± 0.22 | -29.11% | 4.84 ± 0.00 | 4.48 ± 0.01 | +8.12% |
| 256 | 39.04 ± 0.09 | 55.53 ± 0.12 | -29.70% | 4.83 ± 0.00 | 4.47 ± 0.01 | +8.11% |
| 512 | 38.92 ± 0.12 | 54.03 ± 1.07 | -27.97% | 4.83 ± 0.00 | 4.47 ± 0.00 | +8.22% |
| 1024 | 38.86 ± 0.10 | 53.58 ± 0.92 | -27.48% | 4.83 ± 0.01 | 4.45 ± 0.01 | +8.53% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
