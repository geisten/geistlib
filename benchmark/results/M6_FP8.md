# M6 native FP8 experiment — 2026-10-31

**The new native FP8 path is slower on all three measured models and does not reduce Metal buffer allocations. Keep it opt-in.** This is a working native FP8 implementation and an experiment, not a performance improvement to enable by default.

Both operands are E4M3 tensors with E8M0 scales per 32 elements, consumed directly by MPP `matmul2d` and accumulated in FP32. Packed GGUF weights stay packed in memory. Decode, attention and short-batch selection retain the existing kernels.

## Protocol

Apple M6; ProductName:		macOS; ProductVersion:		27.0; BuildVersion:		26A425. Kernel source commit `15a088d72531`, based on main `6b3b72e1`. The later benchmark metadata fix and report commit do not change the kernels or measured commands.

Executable SHA-256: `af70c1543995fa751eb1eceb10c892d849b555cfc4979de855794f18bef2bc36`. Model SHA-256 values are in [metadata.json](raw/m6-fp8-2026-10-31/metadata.json).

Six cycles alternate Metal 4/FP8 and FP8/Metal 4 on the **same executable**, changing only `GEIST_METAL_FP8`; Metal tensors are enabled in both arms. Each process warms up with 16 prefill tokens and four decode steps, then measures 128 and 512 prompt tokens and 32 greedy decode steps, one repetition per point. Four OpenMP threads. Prompts use the fixed pseudo-random token stream from `bench_perf_sweep`. The host lease excludes other heavy jobs. Every arm passed AC-power/Low-Power-Mode and <=65 °C CPU/GPU start-temperature gates; all recorded thermal checks used sensors, not timeout-based pauses.

72 timing rows, 36 process invocations. Tables show medians across six cycles; speed ratio is the median of paired per-cycle FP8/Metal-4 prefill ratios. Ratios below 1 mean FP8 is slower. This compares two runtime modes with identical code layout; it is not a comparison of incremental versus clean builds.

## Throughput

| Model | Prompt tokens | Metal 4 prefill tok/s | FP8 prefill tok/s | Paired FP8 / Metal 4 | Metal 4 decode tok/s | FP8 decode tok/s |
| :-- | --: | --: | --: | --: | --: | --: |
| Llama 3.2 3B Q4_K_M | 128 | 1953.8 | 1039.9 | 0.532x | 70.35 | 70.38 |
| Llama 3.2 3B Q4_K_M | 512 | 2048.9 | 1074.2 | 0.524x | 68.83 | 68.98 |
| Qwen 3.5 0.8B Q8_0 | 128 | 5878.4 | 3874.5 | 0.659x | 155.12 | 155.01 |
| Qwen 3.5 0.8B Q8_0 | 512 | 7138.4 | 4452.2 | 0.624x | 154.96 | 155.06 |
| Bonsai 2 27B PQ2_0 | 128 | 255.1 | 125.5 | 0.492x | 19.05 | 19.22 |
| Bonsai 2 27B PQ2_0 | 512 | 224.0 | 110.8 | 0.494x | 19.15 | 19.14 |

At 512 prompt tokens FP8 lost in **all six paired cycles for each model**. Decode differences are small and do not establish a speedup. The new tile stages and quantizes activations as well as weights, whereas the existing path reads activations directly. Conversion, staging and occupancy are plausible costs; these measurements do not isolate their individual contributions or measure the chip’s peak FP8 arithmetic throughput.

## Memory

| Model, 512-token prompt | Metal 4 buffers MiB | FP8 buffers MiB | Metal 4 RSS MiB | FP8 RSS MiB |
| :-- | --: | --: | --: | --: |
| Llama 3.2 3B Q4_K_M | 2086.14 | 2086.14 | 604.38 | 604.64 |
| Qwen 3.5 0.8B Q8_0 | 901.45 | 901.45 | 159.91 | 160.10 |
| Bonsai 2 27B PQ2_0 | 7480.00 | 7480.00 | 487.34 | 487.45 |

Metal bytes come from the device’s `currentAllocatedSize` snapshot through `geist_backend_resources_snapshot`; RSS is the process-resident snapshot after the measured point. Neither is a peak-memory measurement. Unified-memory mappings overlap these accounting domains: **do not add them**. There is no model-buffer saving from this FP8 compute mode. The native tile needs 25,344 bytes of on-chip threadgroup storage, versus 8,192 for the existing tensor GEMM. No full-model FP8 weight copy or full activation copy is introduced by the new kernel.

The separate scratch PR #353 removes an existing unused PQ2 activation buffer from both tensor modes. This experiment uses the same main baseline in both arms and does not include that independent PR; its saving must not be attributed to FP8.

## Output quality

Twelve self-authored completion tasks cover simple factual prose, arithmetic, German, Python, JSON and short context following. Both modes score the same teacher-forced continuation tokens after prefill, then decode 24 greedy steps. The conditional perplexity ratio is `exp(mean_NLL_fp8 - mean_NLL_metal4)`; lower is better **on this corpus only**.

| Model | Continuation tokens | Mean NLL Metal 4 | Mean NLL FP8 | Conditional PPL change | Identical 24-step token streams | First-token matches |
| :-- | --: | --: | --: | --: | --: | --: |
| Llama 3.2 3B Q4_K_M | 168 | 1.39939 | 1.39738 | -0.20% | 8/12 | 12/12 |
| Qwen 3.5 0.8B Q8_0 | 158 | 1.31252 | 1.32081 | +0.83% | 8/12 | 12/12 |
| Bonsai 2 27B PQ2_0 | 158 | 1.03326 | 1.01530 | -1.78% | 10/12 | 12/12 |

This smoke corpus shows small NLL movement in both directions and changed continuations. It does **not** establish quality equivalence or an improvement, and is not WikiText perplexity, MMLU, a coding benchmark or a statistical quality guarantee. The 24-step token comparison includes steps after EOS because the existing DECODE protocol always executes the requested count; the separately recorded GEN text stops at EOS. GEN text in the JSONL is whitespace-normalized by the REPL parser.

Full-vocabulary next-token logits were also compared for three prompts per model (English, German, Python), with identical recorded token IDs in each pair:

| Model | Top-1 agreement | Minimum centered cosine | Maximum Jensen-Shannon divergence, nats |
| :-- | --: | --: | --: |
| Llama 3.2 3B Q4_K_M | 3/3 | 0.997494 | 0.000638 |
| Qwen 3.5 0.8B Q8_0 | 3/3 | 0.996709 | 0.008367 |
| Bonsai 2 27B PQ2_0 | 3/3 | 0.998837 | 0.000203 |

## Validation and decision

- Release and ASan/UBSan unit suites: 108 passed, 54 skipped, no failures (162 tests each; sandboxed GPU and other-hardware tests skip).
- Real M6 GPU: default tensor parity, FP8 parity, the tensor-disabled fallback, and FP8 parity under ASan/UBSan pass.
- Independent CPU E4M3 oracle: NRMSE zero on fixtures covering zero blocks, K=32/160/256, partial output and row tiles, offsets, untouched output guards and activation scales 2^-32 / 2^32; also passes under ASan/UBSan.
- Default-path tolerance stays unchanged. Explicit FP8 parity uses a separate 6% NRMSE smoke budget; this is not an application-quality promise. The independent oracle additionally checks FP8 arithmetic correctness.
- clang-format 22, changelog and whitespace checks pass.

Keep `GEIST_METAL_FP8=0` (the default). The native execution plumbing and measurements are useful for further kernel experiments, but the measured implementation has neither a throughput nor a model-memory benefit. A replacement tile/staging strategy and broader quality evaluation are needed before enabling it by default.

## Reproduce and inspect

```sh
make -j8 bin/mac-omp/release/tests/bench_perf_sweep \
  bin/mac-omp/release/tests/test_metal_fp8_unit \
  bin/mac-omp/release/tools/eval_geist \
  bin/mac-omp/release/tools/dump_geist_logits
~/bin/with-macmini-lock python3 tools/bench_metal_fp8.py \
  --model llama=/path/to/llama-3.2-3b-Q4_K_M.gguf \
  --model qwen=/path/to/qwen3.5-0.8b-q8_0.gguf \
  --model bonsai=/path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --out /tmp/m6-fp8-comparison --cycles 6
```

[Raw timings](raw/m6-fp8-2026-10-31/runs.jsonl), [quality inputs, scores and outputs](raw/m6-fp8-2026-10-31/quality.jsonl), [machine-generated summary](raw/m6-fp8-2026-10-31/summary.json), [validation summary](raw/m6-fp8-2026-10-31/validation.json), [logit/ID fingerprints](raw/m6-fp8-2026-10-31/logit-manifest.json). Full logit binaries remain local to the run directory and can be regenerated; binary dumps are not included in the PR.
