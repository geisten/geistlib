### Changed
- **Metal: Gemma 4's decode PLE gate and the F32 matvec take a simdgroup
  per row (#196).** `ple_gate_f32` and `matmul_f32` gave each output row a
  threadgroup of 256 (1024) threads and a barrier reduction; Gemma 4 E2B
  runs both 35 times per decode token on its 1536 x 256 F32 PLE matrices,
  and the subtractive profiler put them at about 8 and 15 us per call. The
  new `ple_gate_f32_sg` and `matvec_f32_rows` put four rows in a
  128-thread threadgroup, read float4 chunks and finish with one
  `simd_sum`. They run for one-row F32 matvecs whose length and offsets
  are multiples of 4. The sums are in another order: Gemma 4 E2B's logits
  move by rounding, its greedy output is unchanged; Llama 3.2 3B and
  Qwen3 0.6B use no F32 matvec at decode and are bit-identical.
