### Added
- **Vulkan runs DeepSeek-type latent attention on the GPU, in both layouts
  (#504).** GigaChat3 and DeepSeek-V3 / Kimi K2 files (the split
  `attn_k_b` / `attn_v_b`, absorbed MLA) no longer refuse to load on Vulkan,
  and DeepSeek-V2-Lite's decompressed latent attention is no longer
  assembled on the host. Vulkan now sets `caps.mla_device`: `rmsnorm` and
  `scale_f32` take row views of wider rows, attention runs at 576 dims
  (tensor-core prefill in column thirds from an F16 or the FP32 cache,
  split-KV decode, V read from the K cache as on Metal) and at 192
  (tensor-core prefill), and `linear_t_ids` runs k_b / v_b per head in
  dispatches of at most 65535 slots. RTX 2080 Ti, `tools/bench_revision_ab.py`
  against main, 6 cycles under the bench and GPU locks: DeepSeek-V2-Lite
  Q4_K_M prefill @512 14.0 s -> 0.41 s, decode of 64 tokens 1.62 -> 0.60 s;
  GigaChat3 (decompressed) prefill @512 27.5 s -> 0.21 s, decode 2.24 ->
  0.50 s; Llama-3.2-3B unchanged (within +0.2 %). Perplexity
  (`tools/ppl_vs_llama.py`, 8 x 512) within -0.43 / +0.05 % of llama.cpp
  Vulkan on all four files. The absorbed files run at 0.84-0.92x of
  llama.cpp Vulkan's decode and 0.26-0.44x of its prefill.
