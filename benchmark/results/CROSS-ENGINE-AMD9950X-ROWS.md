# AMD AVX-512 CPU column: remaining matrix rows

The AMD AVX-512 CPU column for the remaining #364 rows, all under the frozen
CPU protocol (profile `amd_9950x`, 16/15 threads), measured by the on-demand
`cross-engine-campaign` workflow on the org box. Two kinds of cells live here:
competitive ones (Gemma 4 E4B, Llama 3.2 3B) and cells that quantify geist's
missing x86 kernel coverage for TQ2_0 and the qwen arches — there geist runs
its scalar fallback 10-100x behind the reference; see
[#410](https://github.com/geisten/geistlib/issues/410). Honest cells either
way: same model bytes, CPU-only verified, every sample retained. The 27B row
is being re-measured under a longer job timeout.

## Gemma 4 E4B-it Q4_K_M

Model: `gemma4-e4b-Q4_K_M.gguf` (`85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87`)

geist `b78df97fdb09`; llama.cpp `2d8d612e4c68`.

Host profile `amd_9950x`: 16-thread prefill, 15-thread decode, geist on `cpu_x86`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-30-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 265.14 ± 1.05 | 262.87 ± 1.54 | +0.86% | 25.47 ± 0.01 | 22.94 ± 0.04 | +11.02% |
| 256 | 262.35 ± 0.80 | 268.74 ± 1.89 | -2.38% | 25.29 ± 0.01 | 22.72 ± 0.04 | +11.30% |
| 512 | 255.21 ± 0.62 | 269.77 ± 0.99 | -5.40% | 24.93 ± 0.01 | 22.44 ± 0.06 | +11.09% |
| 1024 | 246.43 ± 0.39 | 264.82 ± 1.14 | -6.94% | 24.72 ± 0.01 | 22.34 ± 0.06 | +10.67% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.

Raw samples, every draw retained: [`raw/2026-09-08T021027Z_geist_llama_cpu_gemma4-e4b-q4km_amd_9950x.jsonl`](raw/2026-09-08T021027Z_geist_llama_cpu_gemma4-e4b-q4km_amd_9950x.jsonl)

## Llama 3.2 3B Q4_K_M

Model: `llama-3.2-3b-Q4_K_M.gguf` (`6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff`)

geist `b78df97fdb09`; llama.cpp `2d8d612e4c68`.

Host profile `amd_9950x`: 16-thread prefill, 15-thread decode, geist on `cpu_x86`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-30-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 361.36 ± 1.48 | 420.39 ± 3.33 | -14.04% | 36.15 ± 0.02 | 35.47 ± 0.03 | +1.89% |
| 256 | 358.85 ± 1.08 | 418.37 ± 1.65 | -14.23% | 35.69 ± 0.03 | 34.97 ± 0.03 | +2.06% |
| 512 | 345.39 ± 1.49 | 415.06 ± 1.47 | -16.78% | 34.88 ± 0.05 | 34.36 ± 0.02 | +1.50% |
| 1024 | 321.36 ± 2.79 | 405.10 ± 0.87 | -20.67% | 33.41 ± 0.07 | 33.19 ± 0.07 | +0.68% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.

Raw samples, every draw retained: [`raw/2026-09-08T023725Z_geist_llama_cpu_llama32-3b-q4km_amd_9950x.jsonl`](raw/2026-09-08T023725Z_geist_llama_cpu_llama32-3b-q4km_amd_9950x.jsonl)

## BitNet b1.58-large TQ2_0

Model: `bitnet_b1_58-large-TQ2_0.gguf` (`281aafb18a9f4a3124c10a1d8683e2296f0cfe8a2944da0a5667d17488a951bb`)

geist `b78df97fdb09`; llama.cpp `2d8d612e4c68`.

Host profile `amd_9950x`: 16-thread prefill, 15-thread decode, geist on `cpu_x86`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-30-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 14.26 ± 0.45 | 1361.53 ± 0.71 | -98.95% | 8.43 ± 0.27 | 247.36 ± 4.74 | -96.59% |
| 256 | 14.14 ± 0.49 | 1405.85 ± 10.35 | -98.99% | 8.39 ± 0.25 | 210.73 ± 3.38 | -96.02% |
| 512 | 14.12 ± 0.45 | 1374.49 ± 2.24 | -98.97% | 8.34 ± 0.26 | 184.20 ± 0.83 | -95.47% |
| 1024 | 14.10 ± 0.45 | 1324.81 ± 1.75 | -98.94% | 8.28 ± 0.25 | 143.64 ± 0.85 | -94.23% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.

Raw samples, every draw retained: [`raw/2026-09-08T025853Z_geist_llama_cpu_bitnet-large-tq2_amd_9950x.jsonl`](raw/2026-09-08T025853Z_geist_llama_cpu_bitnet-large-tq2_amd_9950x.jsonl)

## Qwen3 0.6B Q8_0

Model: `qwen3-0.6b-q8_0.gguf` (`9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`)

geist `b78df97fdb09`; llama.cpp `2d8d612e4c68`.

Host profile `amd_9950x`: 16-thread prefill, 15-thread decode, geist on `cpu_x86`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-30-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 22.56 ± 0.06 | 1567.01 ± 23.94 | -98.56% | 11.21 ± 0.03 | 104.89 ± 0.60 | -89.31% |
| 256 | 22.58 ± 0.02 | 1661.78 ± 11.30 | -98.64% | 11.17 ± 0.03 | 105.85 ± 0.91 | -89.45% |
| 512 | 22.55 ± 0.03 | 1616.58 ± 8.67 | -98.61% | 11.10 ± 0.03 | 99.32 ± 0.31 | -88.83% |
| 1024 | 22.44 ± 0.03 | 1505.51 ± 7.28 | -98.51% | 10.95 ± 0.04 | 91.15 ± 0.32 | -87.98% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.

Raw samples, every draw retained: [`raw/2026-09-08T034948Z_geist_llama_cpu_qwen3-0.6b-q8_amd_9950x.jsonl`](raw/2026-09-08T034948Z_geist_llama_cpu_qwen3-0.6b-q8_amd_9950x.jsonl)

## Qwen3.5 0.8B Q8_0

Model: `qwen3.5-0.8b-q8_0.gguf` (`0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c`)

geist `b78df97fdb09`; llama.cpp `2d8d612e4c68`.

Host profile `amd_9950x`: 16-thread prefill, 15-thread decode, geist on `cpu_x86`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-30-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 19.53 ± 0.06 | 977.42 ± 4.50 | -98.00% | 8.73 ± 0.02 | 70.49 ± 0.39 | -87.61% |
| 256 | 19.58 ± 0.07 | 1009.35 ± 5.03 | -98.06% | 8.74 ± 0.03 | 70.37 ± 0.45 | -87.58% |
| 512 | 19.61 ± 0.06 | 979.69 ± 34.66 | -98.00% | 8.73 ± 0.03 | 68.94 ± 0.28 | -87.34% |
| 1024 | 19.61 ± 0.06 | 1010.16 ± 12.66 | -98.06% | 8.72 ± 0.02 | 68.30 ± 0.41 | -87.24% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.

Raw samples, every draw retained: [`raw/2026-09-08T042937Z_geist_llama_cpu_qwen35-0.8b-q8_amd_9950x.jsonl`](raw/2026-09-08T042937Z_geist_llama_cpu_qwen35-0.8b-q8_amd_9950x.jsonl)

## Qwen3.5 4B Q4_0

Model: `qwen3.5-4b-q4_0.gguf` (`298fcb5fe7a77ccc79745ae24751560c5ac56874caff4bb39b1f2055bd72b8bb`)

geist `b78df97fdb09`; llama.cpp `2d8d612e4c68`.

Host profile `amd_9950x`: 16-thread prefill, 15-thread decode, geist on `cpu_x86`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-30-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 2.79 ± 0.00 | 267.69 ± 1.06 | -98.96% | 1.58 ± 0.00 | 23.19 ± 0.06 | -93.17% |
| 256 | 2.79 ± 0.00 | 273.92 ± 0.95 | -98.98% | 1.58 ± 0.00 | 23.19 ± 0.14 | -93.17% |
| 512 | 2.79 ± 0.00 | 273.05 ± 1.36 | -98.98% | 1.58 ± 0.00 | 23.12 ± 0.14 | -93.15% |
| 1024 | 2.78 ± 0.00 | 272.43 ± 0.68 | -98.98% | 1.58 ± 0.00 | 22.89 ± 0.07 | -93.09% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.

Raw samples, every draw retained: [`raw/2026-09-08T051228Z_geist_llama_cpu_qwen35-4b-q4_amd_9950x.jsonl`](raw/2026-09-08T051228Z_geist_llama_cpu_qwen35-4b-q4_amd_9950x.jsonl)
