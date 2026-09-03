# Current CPU head-to-head: geist vs llama.cpp

Model: `gemma4-e2b-Q4_K_M.gguf` (`740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8`)

geist `2d7f532c4745`; llama.cpp `2d8d612e4c68`.

Host profile `apple_m1max`: 8-thread prefill, 7-thread decode, geist on `cpu_neon`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: Apple M1 Max (8 physical cores), Darwin 25.5.0.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 171.43 ± 0.29 | 306.26 ± 0.68 | -44.03% | 44.18 ± 0.37 | 50.82 ± 0.20 | -13.08% |
| 256 | 168.01 ± 0.55 | 309.40 ± 1.17 | -45.70% | 39.83 ± 0.77 | 49.46 ± 0.26 | -19.48% |
| 512 | 161.27 ± 0.46 | 302.38 ± 4.25 | -46.67% | 34.80 ± 1.01 | 48.57 ± 0.15 | -28.35% |
| 1024 | 153.15 ± 0.28 | 293.81 ± 3.00 | -47.88% | 32.65 ± 0.27 | 47.48 ± 0.22 | -31.23% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.

## Reading the cell

This is the first cell in the matrix where the reference engine leads, and it
does so at every shape: prefill −44…−48 %, decode −13…−31 %, the gap widening
with depth. Within-run dispersion is tight (MAD ≤ 1.1 % on geist, ≤ 1.4 % on
llama.cpp), and the order alternated every cycle, so the ranking is not a
scheduling artefact. What it is not: a statement about geist on Apple silicon
in general — the Metal backend is a different column, and the historical
best-of numbers in [`APPLE.md`](APPLE.md) were taken under other thread and
aggregation rules and are not comparable to this row (see the matrix note on
historical evidence).

The host gate was met the hard way: this is a desktop, and its idle load
(WindowServer, screen-sharing, editors) hovers around the 0.2/core limit. The
runner waited at the gate for most of the ~3 h wall time, and several
30-second quiet windows were followed by a cooldown re-check that failed and
sent it back to waiting. No sample was admitted outside the gate — the JSONL
carries the machine state recorded before every engine member.

## Provenance

- protocol: [`cross_engine_cpu_protocol.json`](../cross_engine_cpu_protocol.json)
  (`2fa87de84bdf7c2e`), host profile `apple_m1max` (8 P-core prefill,
  7-thread decode, `OMP_WAIT_POLICY=active`, `GEIST_TEXT_ONLY=1`)
- raw samples, every draw retained, host state per member:
  [`raw/2026-09-03T100308Z_geist_llama_cpu_apple_m1max.jsonl`](raw/2026-09-03T100308Z_geist_llama_cpu_apple_m1max.jsonl)
- geist `2d7f532c4745`, binary `f535711689c7c7ba`, built `make` (TARGET
  `mac-omp`: `cpu_neon cpu_scalar`, Accelerate GEMM, libomp), Apple clang 21
- llama.cpp `2d8d612e4c68`, binary `d162ebe53fe518ed`, built
  `-DGGML_NATIVE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple -DGGML_METAL=OFF`
  (Accelerate BLAS, `libggml-cpu` + `libggml-blas` only; no Metal, Vulkan or
  CUDA library linked — verified from the binary's load commands)
- host: Apple M1 Max, 8 performance + 2 efficiency cores, 64 GB, macOS 26.5.1
  (Darwin 25.5.0), mains power

Both engines therefore reach the same BLAS library (Accelerate) and the same
core count; the thread profile is the protocol's rule (P-cores for prefill,
one fewer for decode), matched between engines and not tuned per engine.
