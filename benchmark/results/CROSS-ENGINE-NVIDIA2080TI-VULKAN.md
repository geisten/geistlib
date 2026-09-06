# Current GPU head-to-head: geist vs llama.cpp

Model: `gemma4-e2b-Q4_K_M.gguf` (`740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8`)

geist `2276d33899da`; llama.cpp `2d8d612e4c68`.

Host profile `nvidia_2080ti_vulkan`: 4-thread prefill, 4-thread decode, geist on `vulkan`, identical counts for both engines.

4 alternating A/B cycles, 3 raw samples per cell and cycle; median ± MAD. Full GPU offload on NVIDIA GeForce RTX 2080 Ti, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.

Host: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores), Linux 7.0.0-30-generic.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 1588.57 ± 4.81 | 3533.99 ± 10.96 | -55.05% | 138.71 ± 2.10 | 157.90 ± 0.42 | -12.16% |
| 256 | 1529.60 ± 1.23 | 4393.28 ± 20.02 | -65.18% | 140.26 ± 0.66 | 152.82 ± 0.42 | -8.22% |
| 512 | 1415.91 ± 0.49 | 4829.78 ± 10.99 | -70.68% | 137.14 ± 2.35 | 153.05 ± 0.43 | -10.40% |
| 1024 | 1290.12 ± 0.79 | 4354.90 ± 6.79 | -70.38% | 136.09 ± 1.87 | 151.65 ± 0.72 | -10.26% |

Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. Publish speedups only together with the repository's separate quality-parity gate.

## Reading the cell

The widest prefill gap in the matrix so far: llama.cpp leads by −55 % at
pp128 growing to −71 % at pp512/1024, while decode stays a much closer
−8…−12 %. That shape matches the backend's known state
([`VULKAN.md`](VULKAN.md)): geist's Vulkan decode path has been within reach
of llama.cpp since the 2026-07 port, and prefill has been the open GEMM work
ever since. Dispersion is tight on both engines (MAD ≤ 0.4 % for llama.cpp,
≤ 1.7 % for geist), and llama.cpp's prefill peaking at depth 512 rather than
1024 is its own curve, reproduced across all four cycles.

Unlike the two desktop cells, the gate cost nothing here: the CI box idles
near zero, and all eight members measured at 0.005–0.09 load/core on the
first attempt — a fixed rig behind a queue is the shape a campaign wants.
Absolute numbers describe the RTX 2080 Ti + 9950X box, not the engines;
ratios are only comparable within this column.

## Provenance

Produced by the on-demand
[`cross-engine-vulkan.yml`](../../.github/workflows/cross-engine-vulkan.yml)
workflow (run 34061852975) on the org's `geist-vulkan` runner.

- protocol: [`cross_engine_gpu_protocol.json`](../cross_engine_gpu_protocol.json)
  (`2223183217d536ed`), host profile `nvidia_2080ti_vulkan` (4/4
  dispatch-side threads, `OMP_WAIT_POLICY=active`, `GEIST_TEXT_ONLY=1`,
  `GEIST_BENCH_BACKEND=vulkan`, `GGML_VK_VISIBLE_DEVICES=0` so llama.cpp
  cannot bind the host's Raphael iGPU)
- raw samples, every draw retained, host state per member:
  [`raw/2026-09-06T214316Z_geist_llama_gpu_nvidia_2080ti_vulkan.jsonl`](raw/2026-09-06T214316Z_geist_llama_gpu_nvidia_2080ti_vulkan.jsonl)
- geist `2276d33899da` (main at dispatch), binary `cd0f36e8933334bc`, built
  `make TARGET=linux BACKENDS="vulkan cpu_x86 cpu_scalar"
  GEMM_PROVIDER=native`, gcc 15.2; backend `vulkan` verified from the
  bench's stderr backend line, discrete device preferred by `vk_pick_device`
- llama.cpp `2d8d612e4c68`, binary `571bc46cc86c3162`, built
  `-DGGML_VULKAN=ON` at the same pin as every other cell; backend token
  `Vulkan`, full offload (`-ngl 99`) and device
  `NVIDIA GeForce RTX 2080 Ti` verified from every JSONL row, KV cache
  GPU-resident (`no_kv_offload: false`)
- host: AMD Ryzen 9 9950X, RTX 2080 Ti 11 GB (Turing), Ubuntu 26.04.1
  (Linux 7.0.0-30), mains power; same physical box as the
  `amd_9950x` CPU cell — a different backend column of the same machine

The discrete-GPU guard ran before the campaign (`vulkaninfo` must
enumerate a `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`), so a driver failure
cannot demote this cell to llvmpipe silently.
