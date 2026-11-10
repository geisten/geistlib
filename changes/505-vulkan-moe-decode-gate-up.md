### Changed
- **A MoE decode step on Vulkan runs the experts' gate, up and GLU as one
  dispatch, and its expert matvecs reach more of the memory bandwidth
  (#505).** Vulkan now has `fused->moe_ffn_ids` for up to 8 rows: the new
  `moe_gate_up_q4k_sel` reads x once for a slot's gate and up weight (Q4_K)
  and writes `act(gate + bias, up + bias)` (SwiGLU or swiglu_oai), then down
  runs as `linear_t_ids`' `_sel` matvec -- four dispatches per MoE layer
  instead of six. A chunk (from 9 rows) and other dtypes return
  `GEIST_E_UNSUPPORTED` and keep the grouped GEMMs. The `_sel` matvecs
  (`matvec_q4k_sel`, `matvec_q6k_sel`) sum each warp with `subgroupAdd`
  where a subgroup is a 32-lane warp, and `matvec_q6k_sel` reads x from
  memory instead of staging it in 24 KB of shared memory, which held Turing
  to two workgroups per SM: OLMoE's Q6_K down projection 61 -> 37 us per
  layer. The dense and grouped matvecs' SPIR-V is unchanged. RTX 2080 Ti,
  OLMoE-1B-7B Q4_K_M, `bench_revision_ab` against main, 10 cycles: decode
  of 256 tokens -5.3 % [-8.5, -1.6], 9 of 10 cycles faster (~288 -> ~303
  tok/s; llama.cpp Vulkan tg128 371); Qwen3-30B-A3B (hot experts, #162
  path) and gpt-oss-20b within noise. Greedy tokens identical to main;
  `test_moe_grouped_unit` checks the fused FFN at decode rows against the
  four calls (1e-5) and a double FFN.
