### Changed
- **Metal on Apple M5/M6: the tensor-API prefill GEMM is tiled as mlx's, with
  the same output bit for bit (#590).** Each simdgroup multiplies 16 x 32 x
  16 fragments held in its registers instead of eight simdgroups sharing one
  `matmul2d` over tensors in memory. On the Mac mini M6 against `main`
  (`bench_revision_ab.py`, 10 cycles, run 4153, control within ±0.3 %): the
  prefill of 512 tokens takes 6.1 % less on Qwen3.5 0.8B Q8_0 (69.2 to
  65.0 ms), 9.7 % less on Llama 3.2 3B Q4_K_M (248.6 to 224.6 ms), 7.8 %
  less on Qwen3.5 4B Q4_0 and 6.7 % less on Gemma 4 E2B Q4_K_M; decode is
  within noise on all four. The logits hash of `bench_nll` equals main's on
  the decode and the prefill path of all four models, and
  `tests/test_backend_metal_parity_unit.c` compares the two kernels bit for
  bit on six formats. `GEIST_METAL_TC_COOP=1` keeps the previous kernel,
  which also stays the one `GEIST_METAL_TENSOR=1` runs on a GPU without
  matrix units. One shape is slower: Q6_K 8192 to 3072 at 512 rows in one
  call (+19 %), which the default 256 rows per pass do not reach.
  `benchmark/results/METAL_PREFILL_GEMM_M6_2026-11-24.md` has every kernel
  of the prefill against mlx's on the same shapes: the gap of #590's
  profile is the quantized GEMM (1.4x mlx's before, 1.1-1.25x now) and not
  the DeltaNet, and what remains is mlx's 16-bit activations and cheaper
  weight formats. It also records that the matrix units take FP32
  activations as FP16 truncated toward zero under `relaxed_precision`.
  `tests/bench_metal_gemm` times one quantized GEMM on Metal;
  `bench_nll` can dump its logits (`GEIST_NLL_DUMP`).
