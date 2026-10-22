### Added
- **Qwen3.5/3.6 MoE (`qwen35moe`, Qwen3.6-35B-A3B) loads and runs (#169).**
  The Qwen3.5 hybrid (DeltaNet and gated attention) with a mixture of
  experts: a softmax router over 256 experts, 8 per token, and a shared
  expert every token runs, weighted by sigmoid(x · its gate). The shared
  expert is routed as one more expert, so every placement of the experts
  takes it: the CPU, hot copies on a GPU (the shared ones first). On a
  synthetic model whose MoE equals a dense qwen35 the logits match the
  dense model's to 1.5e-7.
