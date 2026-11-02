### Added
- **Mixtral 8x7B loads (#241).** The file says `general.architecture` =
  `llama`, so the family table selects a Mixtral variant of the llama family
  when `llama.expert_count` is set. It runs Llama's attention with the
  mixture-of-experts FFN of the other MoE families. Early GGUFs store one
  tensor per expert, which the loader refuses by name;
  `tools/gguf_merge_experts.py` rewrites them to the merged layout without
  touching the quantized data. Perplexity on a 255-token text: 2.07 on Metal.
  The comparison with `llama-perplexity` is still open.
