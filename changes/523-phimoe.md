### Added
- **Phi-MoE loads (#523).** `general.architecture` `phimoe` (Phi-mini-MoE),
  as the model's reference implementation runs it: the attention, FFN and
  output norms are LayerNorms with a bias, q/k/v/o and the lm_head have
  biases, and the router is SparseMixer's inference rule (the two largest
  logits, each weighted by its softmax among the logits within 2 % of it;
  usually both weights are 1). Three additions carry it: a backend
  primitive `layernorm` (EXPERIMENTAL `geist_backend_primitives`, nullable;
  on cpu_scalar, cpu_neon, cpu_x86, Metal and Vulkan), an output-norm bias
  and an lm_head bias in the forward core, and the router. llama.cpp runs
  this architecture with RMSNorm plus bias and softmax-renormalised expert
  weights, so its perplexity differs: Phi-mini-MoE Q4_K_M on wikitext-2
  (8 x 512, M1 Max) scores 6.792 on Metal and 6.793 on cpu_neon here,
  8.983 in `llama-perplexity`; with llama.cpp's two choices switched in,
  geist scores 8.993 (+0.12 %), so the rest of the graph agrees. A
  double-precision reference of the graph agrees with cpu_scalar, cpu_neon
  and Metal (experts on the GPU, any share of them hot) to 2e-7. A model
  with an lm_head bias gets neither the speculative sketch head nor the
  decision readout. Files with LongRoPE factors (Phi-3.5-MoE) are refused;
  `attention.sliding_window` is not applied, as in llama.cpp. On a GPU the
  host routes (the device routers have no SparseMixer weights), so decode
  pays one readback per layer.
