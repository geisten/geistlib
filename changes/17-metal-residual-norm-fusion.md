### Changed
- **Metal decode drops about 100 dispatches per Gemma 4 token: residual
  steps and the next RMSNorm share one kernel (#17).** The new fused op
  `rmsnorm_add_norm` (`geist_backend_fused`, probed as
  `GEIST_FUSED_RMSNORM_ADD_NORM`) computes `y = (res + rmsnorm(x) * w) *
  scale` (or `res + x`) and `z = rmsnorm(y) * z_w` in one pass. Llama,
  Qwen3, Gemma 3 and Gemma 4 use it after attention, where it also writes
  the FFN's input norm, and at the end of a layer, where it also writes the
  next layer's attention norm. For Gemma 4 this happens after the PLE
  projection, with the layer scale folded in. Metal's `ple_block` takes
  `norm_w = nullptr` to stop at the projection. Gemma 4 E2B goes from about
  580 to about 477 dispatches per decode token. The results are
  bit-identical: the parity test compares bytes, and teacher-forced logits
  and greedy output of Gemma 4 E2B, Llama 3.2 3B and Qwen3 0.6B match
  main. Decode on the M1 Max (`bench_revision_ab`, 14 cycles) is 1.5 %
  faster for Gemma 4 E2B (95 % interval −1.9..−0.7 %) and unchanged for
  Qwen3 0.6B and Llama 3.2 3B. Vulkan and the CPU backends have no such
  op and are unaffected.
