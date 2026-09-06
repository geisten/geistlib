# Current GPU head-to-head: geist vs llama.cpp

Model: `gemma4-e2b-Q4_K_M.gguf` (`740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8`)

geist `f1097f1043c2`; llama.cpp `2d8d612e4c68`.

Host profile `apple_m1max_metal`: 4-thread prefill, 4-thread decode, geist on `metal`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. Full GPU offload on Apple M1 Max, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: Apple M1 Max (8 physical cores), Darwin 25.6.0.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 1082.11 ± 7.19 | 1411.58 ± 2.40 | -23.34% | 85.69 ± 0.71 | 97.28 ± 0.28 | -11.91% |
| 256 | 1049.88 ± 1.79 | 1532.62 ± 6.06 | -31.50% | 83.37 ± 0.66 | 96.81 ± 0.35 | -13.89% |
| 512 | 991.12 ± 1.58 | 1557.01 ± 2.36 | -36.34% | 79.40 ± 0.42 | 96.69 ± 0.33 | -17.88% |
| 1024 | 921.22 ± 0.93 | 1345.20 ± 147.31 | -31.52% | 76.31 ± 0.25 | 94.82 ± 0.50 | -19.51% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.

## Reading the cell

The reference engine leads on Metal as it does on this host's CPU column:
prefill −23…−36 %, decode −12…−20 %, the decode gap widening with depth.
Dispersion is tight (MAD ≤ 0.9 % everywhere) with one exception: llama.cpp's
pp1024 member scattered to ±147 tok/s (MAD 11 %) across cycles — its slowest
cycle sits well below the other three. The raw JSONL retains every ordered
sample, so the cell publishes the median over all twelve draws rather than
excluding the wide member. geist's own pp1024 spread stayed at ±0.9.

This cell and [`CROSS-ENGINE-APPLE-M1MAX.md`](CROSS-ENGINE-APPLE-M1MAX.md)
describe the same machine under two backends; the GPU column's absolute
numbers are ~6× (prefill) and ~2× (decode) the CPU column's, which is
capacity data about the M1 Max, not part of the engine comparison. Engine
ratios are only comparable within one column.

The host gate was the expensive part, more so than for the CPU cell: the
shared desktop's load sat above the 0.2/core limit for four full 4-hour gate
timeouts before a quiet window opened. Each timeout left the artifact
unmeasured rather than admitting a loaded sample; the four resume records in
the raw JSONL document those restarts, and the machine state recorded before
every member shows all eight measured under 0.2/core.

## Provenance

- protocol: [`cross_engine_gpu_protocol.json`](../cross_engine_gpu_protocol.json)
  (`b6e8fc036844a705`), host profile `apple_m1max_metal` (4/4 dispatch-side
  threads, `OMP_WAIT_POLICY=active`, `GEIST_TEXT_ONLY=1`,
  `GEIST_BENCH_BACKEND=metal`)
- raw samples, every draw retained, host state per member:
  [`raw/2026-09-05T082222Z_geist_llama_gpu_apple_m1max_metal.jsonl`](raw/2026-09-05T082222Z_geist_llama_gpu_apple_m1max_metal.jsonl)
- geist `f1097f1043c2`, binary `3694bf8e010cfc3f`, built
  `make BACKENDS="cpu_neon cpu_scalar metal"` (TARGET `mac-omp`), Apple
  clang 21; backend `metal` verified from the bench's stderr backend line
- llama.cpp `2d8d612e4c68`, binary `3f3a912702bf6785`, built
  `-DGGML_METAL=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple`
  (`libggml-metal` linked and reported as `MTL,BLAS`); full offload
  (`-ngl 99`) and device `Apple M1 Max` verified from every JSONL row,
  KV cache GPU-resident (`no_kv_offload: false`)
- host: Apple M1 Max, 32-core GPU, unified 64 GB, macOS 26.6.2
  (Darwin 25.6.0), mains power

Both engines drive the same GPU from the same unified memory; the thread
profile only fixes the dispatch side and is matched between engines, not
tuned per engine.
