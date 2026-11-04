### Changed
- **Greedy argmax is vectorized on macOS too (#414).** `geist_sampler_argmax`
  takes each 256-logit block's maximum with `fmaxf` instead of
  `x > m ? x : m`. clang vectorizes the select only when it may assume no
  NaNs, and the macOS build keeps NaNs defined (`-fno-finite-math-only`), so
  there the pass stayed a scalar compare/select chain over the whole
  vocabulary: ~0.19 ms per greedy token on Llama 3.2 3B (M1 Max), also paid
  per row in MTP drafting and the speculative head. `fmaxf` vectorizes with
  clang and gcc on arm64 and x86-64. The result is unchanged: the lowest
  index of the maximum wins, held to a plain scan by `test_sampler_unit`
  (now also signed-zero ties, masked `-inf` logits and many repeated maxima).
