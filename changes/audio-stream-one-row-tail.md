### Fixed
- **A streaming audio turn could flip its greedy prediction when the encoder
  worker woke on the last real frame.** The tail then leaves as a
  one-sub-token block, and `linear_fp32` sent a one-row product down the
  BLAS's gemv path, which sums in another order than the gemm the batch
  encode uses (OpenBLAS on Neoverse-N1, Accelerate for any M=1 sgemm): one
  soft token 2e-6 off, a 1.3-logit shift in the LM, and
  `test_audio_stream_session_int` failed on the new arm64 cloud runner,
  whose timing lands the worker there. `linear_fp32` now runs one row as a
  two-row gemm (zero second row, thread-local scratch), so a row gets the
  same bits whatever the row count; `test_linear_fp32_rows_unit` checks
  that on every GEMM provider, and the audio-smoke job pins the worker
  snapshot to the last frame.
