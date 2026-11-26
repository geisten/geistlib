### Changed
- **Gemma 4 on Metal with an INT8 or INT4 KV cache: the last prefill row
  alone runs the KV-shared layers (#591).** As FP32, F16 and KIVI caches
  already do. The pending logits move by up to 8.6e-3 of the largest logit
  against a prefill that runs every layer for every row (Gemma 4 E2B
  Q4_K_M, M1 Max, eight prompt lengths; the attention rounds each query to
  8 bits); the pending token, the K/V rows and every decode step are the
  same. `GEIST_KV_SHARED_FULL=1` is the full prefill.
- **Gemma 4 on Vulkan: the last prefill row alone runs the KV-shared
  layers (#591).** Measured on an RTX 2080 Ti with Gemma 4 E2B Q4_K_M at
  eight prompt lengths against the full prefill: pending logits within
  2.1e-3 of the largest logit with an FP32 or F16 cache and 7.0e-3 with
  INT8 or INT4, the same pending token, every decode step bit for bit. A
  KIVI cache on Vulkan is not measured and keeps the full last chunk.
