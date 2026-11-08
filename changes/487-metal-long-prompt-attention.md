### Changed
- **Metal: long-prompt attention for gpt-oss and MLA models (#487).**
  GPT-OSS's attention with sink logits ran a one-simdgroup-per-(row, head)
  kernel, about half of a 2048-token prefill's GPU time; it now takes the
  flash prefill kernels (which skip the key blocks before the sliding
  window) and the split-KV decode kernels, with the sink folded into the
  flash epilogue and the decode combine. Absorbed MLA (GigaChat3,
  DeepSeek-V3: one KV head of 576) and DeepSeek-V2-Lite's 192-dim heads
  get tensor-API flash kernels on GPUs with matrix units (M5/M6), and
  absorbed MLA reads V from the K cache (V is K before RoPE, which touches
  only the dims the output drops), so a key block is read once for both
  products.
