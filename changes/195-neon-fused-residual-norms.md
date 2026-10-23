### Changed
- **cpu_neon runs the residual norms and the attention prep as fused ops
  on all threads (#195).** Gemma 4 E2B prefill on the M1 Max spent ~11 %
  of its main thread outside parallel regions; per layer and 64-token
  chunk the post-attention norm + residual add + pre-FFN norm, the
  post-FFN norm + add, the PLE tail's norm + add + layer scale + next
  pre-norm and the q/k/v head norms, RoPE and KV append each ran as a
  serial pass. cpu_neon now answers the fusions the exec plan already
  binds on Metal (`rmsnorm_add`, `rmsnorm_add_norm`, `attn_qkv_prep`,
  `attn_rope_append`), each on the same per-row kernels as the separate
  ops, with rows split across threads from 8 on. Logits are byte-identical
  (teacher-forced, 560 + 40 steps, on Gemma 4 E2B, Llama 3.2 3B, Qwen3
  0.6B and Qwen3.5 0.8B); `test_res_norm_neon_unit` compares each fusion
  with the separate ops byte for byte. Gemma 4 E2B prefill at 512 tokens
  -4.2 % time (`bench_revision_ab`, 10/10 cycles, control within ±6 %).
