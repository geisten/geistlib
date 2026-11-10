### Added
- **GLM-4.5 loads (#242).** `general.architecture` `glm4moe` (GLM-4.5,
  GLM-4.5-Air, GLM-4.6), as llama.cpp's graph runs it: q/k/v biases,
  per-head QK norms where the file has them, NEOX RoPE on the first
  `rope.dimension_count` dims of a head, dense lead layers, then a sigmoid
  router with a selection bias (DeepSeek-V3's, without groups), renormalised
  and scaled, plus the shared experts. The NextN (MTP) layer counted in
  `block_count` is not loaded; M-RoPE (GLM-4.5V) is refused. The `glm4` /
  `glm5` pretokenizers run the Llama 3 regex and add no BOS, as llama.cpp.
  A double-precision reference of the graph agrees with cpu_scalar to
  1e-7 and with Metal (experts on the GPU) to 2e-7. A random-weight
  conversion of `tiny-random/glm-4-moe` scores the same per-token
  log-probabilities as `llama-perplexity` (8 x 512, cpu_scalar, cpu_neon
  and Metal: NLL sums equal to 7 digits) with identical token ids. No real
  GLM-4.5 checkpoint was run.
