### Fixed
- **Metal on M5/M6-class Macs keeps F32 weights in F32 again (#361).** Since
  #340 every F32 matmul of 8 or more rows went to the tensor-API GEMM with
  its weights rounded to half, not only Gemma's PLE projections it was
  meant for: MoE routers, DeltaNet projections and every F32 fixture lost
  precision. On the Apple M6 batched decode drifted 1.64 logits from
  decode_step (`test_decode_batch_unit`, tolerance 1e-2), and OLMoE /
  Qwen3.5-MoE chunked prefill missed their parity gates. F32 weights take
  the simdgroup GEMMs again (exact F32 staging); the quantized tensor-API
  GEMMs and flash attention are unchanged. Cost: #340's gain on the M6,
  Gemma 4 E2B Q4_K_M prefill +0.9 % at 128 and +1.8 % at 512 tokens
  (mac-ab run 2467), decode unchanged. `test_backend_metal_parity_unit` now
  checks F32 weights with full mantissas, also with the tensor path forced
  on.
