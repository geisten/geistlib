### Added
- **deepseek2 GGUFs with the split `attn_k_b` / `attn_v_b` load and run
  (#375).** Every DeepSeek-V3 / R1, Kimi K2 and GigaChat3 file converted
  since 2025 stores latent attention for the absorbed form, which the loader
  refused. It now runs as llama.cpp's `is_mla` path on the CPU backends: one
  KV head of 576 (`[k_pe | latent]`, for K and V), each query head's nope
  part taken into the latent by its `k_b` slice, each output head's latent
  dims out by its `v_b` slice; a GPU backend refuses the layout (#376). A
  model with heads past 512 gets the FP32 KV cache by default and F16 for a
  quantized request, whose stack arrays stop at 512. GigaChat3-10B-A1.8B
  Q4_K_M: perplexity within 0.07 % of `llama-perplexity` on cpu_neon, 32
  greedy steps identical; DeepSeek-V2-Lite rewritten to the split layout
  gives the unsplit file's logits on cpu_scalar to 3e-7.
