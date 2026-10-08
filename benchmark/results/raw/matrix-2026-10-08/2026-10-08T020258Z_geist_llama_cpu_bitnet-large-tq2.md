# Current CPU head-to-head: geist vs llama.cpp

Model: `bitnet_b1_58-large-TQ2_0.gguf` (`281aafb18a9f4a3124c10a1d8683e2296f0cfe8a2944da0a5667d17488a951bb`)

geist `c696fbe7a144`; llama.cpp `2d8d612e4c68`.

Host profile `amd_9950x`: 16-thread prefill, 15-thread decode, geist on `cpu_x86`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-34-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 2358.38 ± 55.91 | 1363.08 ± 4.84 | +73.02% | 323.86 ± 2.55 | 245.95 ± 1.43 | +31.68% |
| 256 | 2346.83 ± 11.85 | 1404.81 ± 8.42 | +67.06% | 303.85 ± 0.53 | 203.77 ± 3.84 | +49.11% |
| 512 | 2214.14 ± 15.14 | 1380.40 ± 4.50 | +60.40% | 270.16 ± 0.65 | 173.29 ± 2.60 | +55.90% |
| 1024 | 2010.34 ± 12.55 | 1327.50 ± 2.68 | +51.44% | 217.10 ± 0.36 | 135.35 ± 0.78 | +60.40% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
