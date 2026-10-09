# Current CPU head-to-head: geist vs llama.cpp

Model: `bitnet_b1_58-large-TQ2_0.gguf` (`281aafb18a9f4a3124c10a1d8683e2296f0cfe8a2944da0a5667d17488a951bb`)

geist `8489cb953774`; llama.cpp `d81235049384`.

Host profile `amd_9950x`: 16-thread prefill, 15-thread decode, geist on `cpu_x86`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 2315.97 ± 73.36 | 1359.82 ± 3.74 | +70.32% | 320.18 ± 3.03 | 229.05 ± 1.00 | +39.79% |
| 256 | 2325.25 ± 23.03 | 1402.07 ± 6.97 | +65.84% | 302.48 ± 1.02 | 216.84 ± 2.23 | +39.49% |
| 512 | 2207.15 ± 13.86 | 1372.89 ± 5.16 | +60.77% | 270.40 ± 1.84 | 184.29 ± 0.98 | +46.72% |
| 1024 | 1998.18 ± 4.00 | 1324.68 ± 0.16 | +50.84% | 218.92 ± 1.89 | 144.43 ± 1.34 | +51.57% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
