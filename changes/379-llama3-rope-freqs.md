### Fixed
- **Llama 3.1-3.3 apply their RoPE frequency factors (#379).** The llama
  family ignored `rope_freqs.weight`, the per-pair divisors the converter
  writes for `rope_scaling` type "llama3", so the low-frequency pairs turned
  up to 32 times too fast. The family now loads the tensor and builds the
  shared RoPE tables with it (`rope_compute_ff_at`, ggml's freq_factors), so
  every backend gets it; files without the tensor are unchanged. Llama 3.2
  3B Q4_K_M against `llama-perplexity` on the same wikitext-2 tokens: at
  16384 tokens 196.6 vs 6.31 before, 6.314 vs 6.312 after (+0.04 %); at
  512 tokens -0.60 % before, +0.14 % after. Greedy continuations now match
  llama.cpp for 32 steps after a 256- and a 12288-token prompt (before:
  different from the first token).
