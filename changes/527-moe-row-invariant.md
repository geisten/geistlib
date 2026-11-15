### Fixed
- **A routed MoE model's prefill no longer depends on how the prompt is
  chunked, on `cpu_x86` (#527).** A prefill chunk gathers the rows routed
  to one expert into one m-row linear, so m is how many of the chunk's
  tokens chose it, and several kernels picked a code path, and with it a
  summation order, by m. On `cpu_x86` (AVX-512 VNNI) the Q4_K tail rows
  (m % 4) and the Q6_K tail and small calls now quantize and sum as the
  tiled GEMM does (`q4kx8_gemv_row_tiled`, `q6k_gemm_tiled_order`), every
  prefill Q8_K activation goes through one quantizer, and the AVX2 IQ4 /
  MXFP4 4-row dot keeps the single-row dot's two accumulators. On
  `cpu_neon` the Q4_K / Q6_K / Q8_0 prefill float step has one fused
  spelling (`kq_epilogue.h`): with real fp16 scales 96 of 128 row counts
  differed. A prefill chunk runs a one-row expert on the M>1 kernels; the
  decode kernels are unchanged. `test_moe_row_invariant_unit` asserts, at
  the expert shapes of Qwen3-30B-A3B, DeepSeek-V2-Lite and gpt-oss-20b,
  that each row of an m-row call equals that row of the 128-row call for
  every m in 1..128. Final and teacher-forced logits at 64- against
  128-row chunks (`GEIST_M_MAX`) are bit-identical for all three models on
  the Ryzen 9 9950X (`cpu_x86`), and for Qwen3-30B-A3B and gpt-oss-20b on
  the M1 Max (`cpu_neon`); 18 % of Qwen3-30B-A3B's expert selections
  differed before (#507). DeepSeek-V2-Lite on `cpu_neon` still differs
  (max |d| 0.93): its Q5_0 experts go through the BLAS, #528.
  `tools/bench_revision_ab.py` against main, prefill of 16 / 128 / 512
  tokens and 64 decoded tokens: within the control's noise on the 9950X
  (Qwen3-30B-A3B, gpt-oss-20b, Llama 3.2 3B) and on the Mac mini M6
  (gpt-oss-20b, Llama 3.2 3B), except DeepSeek-V2-Lite's 16-token prefill,
  where the one-row experts now take the M>1 kernels: +1.0 % on the 9950X
  (131.2 -> 132.9 ms) and +1.8 % on the M6 (196.8 -> 200.5 ms; its
  128-token prefill is 1.2 % faster). Qwen3-30B-A3B has no usable Apple
  timing: it does not fit the M6's memory, and the M1 Max run's control
  moved by +-64 %.
