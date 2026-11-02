### Added
- **GPT-OSS loads and runs (#243).** The GGUF architecture `gpt-oss`
  (gpt-oss-20b / -120b) as llama.cpp's graph runs it: GQA with q/k/v/o
  biases, NEOX RoPE with YaRN, a learned sink logit per head in every
  softmax, even layers sliding over 128 keys, and a mixture of experts whose
  router picks the 4 largest biased logits and weights them by the softmax
  over those alone, each expert with biases and llama.cpp's `swiglu_oai`.
  The experts are MXFP4 (see that entry). Perplexity against
  `llama-perplexity` on the same tokens (512-token windows, 8 chunks) on the
  Ryzen 9 9950X, cpu_x86: 654.96 vs 656.34 on `tools/data/ppl_alice.txt`
  (-0.21 %), 29.33 vs 29.44 on a technical text (-0.38 %).
- **`fused->attention_sinks` (EXPERIMENTAL, `geist_backend.h`).**
  `prims->attention` with a sink logit per head; Metal implements it, other
  backends fall back to the architecture's host loop.
- **YaRN RoPE** (`rope.scaling.type = yarn`) as ggml's `rope_yarn` computes
  it, for any family that reads it (`family_populate_yarn`).
