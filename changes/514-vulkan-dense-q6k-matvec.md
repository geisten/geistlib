### Changed
- **The dense Q6_K matvec on Vulkan reads x where it lies and sums each warp
  in registers (#514).** `matvec_q6k` staged x in 24 KB of shared memory per
  256-thread workgroup, which held Turing to two workgroups per SM, and
  summed each warp through a shared-memory tree; it now takes the
  `MV_X_GLOBAL` and `MV_SG_REDUCE` paths #505 gave `matvec_q6k_sel`. This is
  the lm_head of every Q4_K_M model with a Q6_K output tensor, plus the Q6_K
  attention projections: OLMoE-1B-7B's Q6_K matvecs went from 46 to 31 us per
  call on the RTX 2080 Ti. `bench_revision_ab` against #509's branch (one hold
  of the bench lock, quiet host): decode OLMoE-1B-7B Q4_K_M -4.8 % [-6.5,
  -2.5] (10/10), Llama-3.2-3B Q4_K_M -8.4 % [-14.4, -7.1] (8/8),
  Qwen3-30B-A3B -2.1 % [-5.7, -0.1], gpt-oss-20b within noise; prefill
  within noise. Greedy tokens identical; `test_backend_vulkan_linear_parity`
  passes on the device.
