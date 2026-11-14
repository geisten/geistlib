# Current GPU head-to-head: geist vs llama.cpp

Model: `Ternary-Bonsai-2-27B-PQ2_0.gguf` (`3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`)

geist `6642260d55a1`; llama.cpp `adfffbe41b2c`.

Host profile `apple_m1max_metal`: 4-thread prefill, 4-thread decode, geist on `metal`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. Full GPU offload on Apple M1 Max, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: Apple M1 Max (8 physical cores), Darwin 27.2.0.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 159.34 ± 0.15 | 141.93 ± 0.02 | +12.26% | 21.42 ± 0.05 | 21.14 ± 0.02 | +1.33% |
| 256 | 166.28 ± 0.07 | 147.52 ± 0.08 | +12.72% | 21.39 ± 0.03 | 21.04 ± 0.17 | +1.65% |
| 512 | 161.86 ± 0.05 | 125.47 ± 11.36 | +29.00% | 21.23 ± 0.05 | 19.30 ± 1.78 | +10.00% |
| 1024 | 161.48 ± 0.07 | 116.08 ± 1.41 | +39.11% | 21.07 ± 0.08 | 17.16 ± 0.26 | +22.78% |

Every run started on AC power with Low Power Mode off (power gate, pmset); each run's machine state records it.

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.
