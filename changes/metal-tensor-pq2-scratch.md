### Changed
- **Metal tensor prefill avoids unused PQ2_0 activation scratch on M5/M6.**
  Batches above eight rows no longer allocate or grow the simdgroup GEMM's
  FP16 copy when the compiled tensor pipeline reads FP32 directly. This
  avoids 10.625 MiB at 320 x 17408; short-batch kernels and the fallback
  when Metal 4 is unavailable retain their scratch. Numerical kernels are
  unchanged. The parity gate covers fresh tensor prefill and prefill after
  a short batch, as well as the simdgroup path with tensors disabled.
