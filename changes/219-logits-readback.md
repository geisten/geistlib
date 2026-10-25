### Fixed
- **Vulkan: logits the host reads come from cached memory -- sampled
  decode 17.8 -> 75 tok/s (#219).** `peek_logits` and the host sampler
  (temperature > 0, repetition penalties) read the logits row in place, in
  the BAR window, which the CPU reads uncached (~90 MB/s). A session on a
  discrete GPU now copies the row on the device into a readback buffer in
  cached system memory first (allocated with the session). Qwen3-30B-A3B on
  an RTX 2080 Ti, GPU mode: decode at temperature 0.8 17.8 -> 74.5-75.1
  tok/s; a token-by-token feed (`bench_nll`) 16.7 -> 73.4 tok/s, its
  softmax over 151,936 logits 36.4 -> 0.11 ms a token (runs 1987, 1992,
  1993).
