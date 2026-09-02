# Current CPU head-to-head: geist vs llama.cpp

Model: `gemma4-e2b-Q4_K_M.gguf` (`740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8`)

geist `b4637a2822c9`; llama.cpp `2d8d612e4c68`.

Host profile `amd_9950x`: 16-thread prefill, 15-thread decode, geist on `cpu_x86`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-30-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 520.00 ± 2.03 | 303.88 ± 9.34 | +71.12% | 50.56 ± 0.11 | 44.79 ± 0.06 | +12.90% |
| 256 | 512.58 ± 0.75 | 363.56 ± 13.79 | +40.99% | 50.22 ± 0.06 | 43.76 ± 0.10 | +14.74% |
| 512 | 492.95 ± 1.78 | 416.67 ± 6.14 | +18.31% | 49.56 ± 0.09 | 43.24 ± 0.13 | +14.63% |
| 1024 | 471.25 ± 0.48 | 401.31 ± 4.83 | +17.43% | 49.02 ± 0.07 | 42.88 ± 0.13 | +14.31% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.

## Provenance

- protocol: [`cross_engine_cpu_protocol.json`](../cross_engine_cpu_protocol.json)
  (`2e27bafcc4903fe3`), host profile `amd_9950x`
- raw samples, every draw retained: [`raw/2026-09-02T070548Z_geist_llama_cpu_amd_9950x.jsonl`](raw/2026-09-02T070548Z_geist_llama_cpu_amd_9950x.jsonl)
- geist `b4637a2822c9`, binary `9a53a6297098ea8b`
- llama.cpp `2d8d612e4c68`, binary `8fba0648f0b58e8b`,
  built `-DGGML_NATIVE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS`, CPU-only
- geist built `make TARGET=linux` (default `cpu_x86 cpu_scalar`, OpenBLAS GEMM,
  `-march=x86-64-v3` with runtime AVX-512/VNNI dispatch)

Both engines therefore reach the same BLAS library and the same core count. The
thread profile is rule-based (physical cores for prefill, one fewer for decode),
matched between engines, and not tuned per engine — see the note in the protocol.

