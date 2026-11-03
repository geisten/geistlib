### Fixed
- **MTP drafting on Metal no longer cuts a dense model's decode to a third
  (#401).** A speculative verify runs its k rows through every linear, and
  for Q4_0, Q4_1, Q8_0 and Q5_K weights 2-7 rows took the naive
  `linear_*` kernels (one threadgroup per output and row, byte loads):
  2 rows of a 5120x17408 Q4_0 matrix cost 0.97 ms against 0.14 for one
  decode row, and a 2-row verify of Qwen3.8-27B Q4_0 ~650 ms against a
  ~100 ms decode step. New `matvec_*_r2` / `_r3` kernels keep the decode
  GEMV's grid, load each row's activations of a block once, dequantize a
  weight row once and dot it with every row (more rows go in chunks of 3
  and 2): 2 rows now cost 0.19 ms, 3 rows 0.28-0.40. Qwen3.8-27B Q4_0 on
  the M1 Max, 128 tokens, medians of 3 interleaved cycles: MTP K=2
  3.7 → 11.9 tok/s on `docs/ARCHITECTURE.md` (MTP off 11.5) and
  4.1 → 12.9 on `docs/OFFLOAD.md` (off 11.1, +8-16 %); K=3 11.1 / 12.4;
  K=4 still loses (8.1 / 10.3). Greedy output is token-identical to MTP
  off in all runs. Eight rows and more keep the GEMM; `GEIST_METAL_MV_M8=0`
  runs one decode GEMV per row instead.
