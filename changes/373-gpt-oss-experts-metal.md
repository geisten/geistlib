### Changed
- **GPT-OSS's experts run on Metal (#373).** Until now every GPT-OSS
  expert ran on the CPU, also with the model on Metal: the GPU expert
  kernels had no router bias, no softmax over the selected experts, no
  expert biases, no `swiglu_oai` and (on Metal's device route) no MXFP4.
  `fused->moe_route` now takes `struct geist_moe_route_args` (gating
  `SOFTMAX` or `SOFTMAX_WEIGHT`, an optional bias on the logits),
  `fused->moe_weighted_sum` the experts' output biases by the routed ids,
  and the new `fused->glu` runs the expert activation with its biases
  (`SILU` or `SWIGLU_OAI`). Metal implements all of them plus an
  id-indexed MXFP4 matvec; Vulkan refuses the new router options and keeps
  GPT-OSS's experts on the CPU. gpt-oss-20b on the M1 Max (Metal, 512-token
  prompt, 64 decoded tokens, 6 interleaved cycles against main):
  prefill 4.3x faster, decode 3.1x faster; perplexity on an in-distribution
  text 29.34 (main 29.33, llama.cpp Metal 29.33).
