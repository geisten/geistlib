### Changed
- **A MoE prefill chunk on Metal runs gate and up with the activation as one
  grouped GEMM, buckets its routed slots once per layer and launches only
  the tiles that have rows (#486).** The device route (#482) ran
  `linear_t_ids` for gate, up and down -- three bucketings (`moe_map`) --
  plus a GLU dispatch over the gate / up rows in memory. A new fused slot,
  `geist_backend_fused.moe_ffn_ids` (EXPERIMENTAL, `struct
  geist_moe_ffn_args`), takes the experts' whole FFN: Metal buckets once,
  runs `moe_mm_glu_<T>` (gate and up rows interleaved in moe_mm's 64 x 32
  tile, SwiGLU or swiglu_oai with both biases in the epilogue) and the down
  GEMM, over a tile list `moe_map` now writes (grid floor(m*k/32) + min(n,
  m*k) instead of ceil(m/32) x n). `GEIST_E_UNSUPPORTED` (a decode step's
  rows, gate and up of different dtypes) keeps the four calls. Results are
  bit-identical to the four calls (unit test, every dtype and activation;
  logits and perplexity identical to main on OLMoE, DeepSeek-V2-Lite,
  GigaChat3 and gpt-oss-20b). Mac mini M6, `bench_revision_ab` against
  main, 8 cycles: prefill @128 / @512 / @2048 OLMoE-1B-7B -3.1 / -3.3 /
  -3.3 %, DeepSeek-V2-Lite -2.3 / -2.5 / -2.6 %, GigaChat3-10B -1.6 / -1.7 /
  -1.1 %, gpt-oss-20b -1.5 / -1.3 / -0.9 %; decode within noise. A
  128-row tile holding both products was 40 % slower and was dropped.
