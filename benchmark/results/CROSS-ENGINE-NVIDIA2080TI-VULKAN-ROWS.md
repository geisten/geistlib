# NVIDIA Vulkan column: remaining matrix rows

The NVIDIA Vulkan column for the remaining #364 rows. Only Llama 3.2 3B
produced a full campaign, and it measures geist's missing tuned llama-arch
path on Vulkan (scalar-fallback speed,
[#410](https://github.com/geisten/geistlib/issues/410)). The other rows are
explicit non-cells, each with a filed defect
([#409](https://github.com/geisten/geistlib/issues/409)):

| model | outcome |
| :-- | :-- |
| Gemma 4 E4B Q4_K_M | geist `GEIST_E_BACKEND` at pp1024, all repeats (pp128-512 fine) |
| BitNet b1.58-large TQ2_0 | geist `GEIST_E_UNSUPPORTED` at pp1024 (same shape passes on cpu_neon) |
| Qwen3 0.6B Q8_0 | geist hang in the first measured prefill (6 h, no output) |
| Qwen3.5 0.8B / 4B | geist Vulkan lacks the qwen3.5 arch (`state_create` fails at load) |
| Qwen3.8 27B Q4_0 | does not fit the 11 GB device |

Failed gates and failed engines publish no numbers; the cells stay explicit.

## Llama 3.2 3B Q4_K_M

Model: `llama-3.2-3b-Q4_K_M.gguf` (`6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff`)

geist `0f9008692139`; llama.cpp `2d8d612e4c68`.

Host profile `nvidia_2080ti_vulkan`: 4-thread prefill, 4-thread decode, geist on `vulkan`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. Full GPU offload on NVIDIA GeForce RTX 2080 Ti, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-30-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 2.36 ± 0.00 | 3600.72 ± 19.69 | -99.93% | 1.63 ± 0.00 | 192.44 ± 0.69 | -99.16% |
| 256 | 2.36 ± 0.00 | 3967.08 ± 16.39 | -99.94% | 1.62 ± 0.00 | 187.48 ± 0.56 | -99.13% |
| 512 | 2.36 ± 0.00 | 4637.96 ± 9.84 | -99.95% | 1.62 ± 0.00 | 185.09 ± 0.44 | -99.12% |
| 1024 | 2.35 ± 0.00 | 4475.85 ± 13.17 | -99.95% | 1.62 ± 0.00 | 181.50 ± 0.60 | -99.11% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.

Raw samples, every draw retained: [`raw/2026-09-07T162202Z_geist_llama_gpu_llama32-3b-q4km_nvidia_2080ti_vulkan.jsonl`](raw/2026-09-07T162202Z_geist_llama_gpu_llama32-3b-q4km_nvidia_2080ti_vulkan.jsonl)
