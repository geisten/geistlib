### Changed
- **Metal decode: q, k and v in one dispatch, and gate, up and SiLU in one
  (#17).** A new fused op, `linear_t_triple` (`geist_backend_fused`), runs
  the three attention projections of a decode step as one grid: each
  threadgroup runs the matvec body of its own weight, so the weights may
  differ in dtype and width (Gemma 4 E2B and Llama 3.2 Q4_K_M: q, k Q4_K,
  v Q6_K; Qwen3, SmolLM2: Q8_0). `ffn_gate_up_silu` (probed as
  `GEIST_FUSED_FFN_GATE_UP_SILU`) is the SwiGLU twin of `ffn_gate_up`, for
  Q4_K and Q8_0; `ffn_gate_up` now also takes Q8_0. Per decode token
  this removes 22 of 469 dispatches on Gemma 4 E2B, 112 of 511 on Qwen3
  0.6B and 98 of 441 on Llama 3.2 3B. The new kernels give the bits of the
  matvecs they replace (the parity test compares bytes, with block scales
  and activations that round); where the Q4_K-only `pair_q4k_n4` kernel
  took k and v before, logits move by rounding, and greedy output of
  Gemma 4 E2B, Llama 3.2 3B, Qwen3 0.6B and Qwen3.5 0.8B is unchanged.
  Vulkan and the CPU backends are unaffected.
  `bench_revision_ab` against main on the M1 Max (Metal, 10 cycles, cool
  start, control copy within noise): decode Qwen3 0.6B −6.8 % at 128
  (95 % interval −8.3..−4.9 %) and −6.9 % at 512, Llama 3.2 3B −2.3 %
  (−3.5..−1.6 %) and −2.9 %, Gemma 4 E2B unchanged; no prefill row moved.
