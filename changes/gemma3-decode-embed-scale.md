### Fixed
- **Gemma 3 generates with the right token embeddings again.** The decode
  step multiplied the token embedding by sqrt(d_model) only for models with
  per-layer embeddings (`has_ple`), while prefill used the embedding-scale
  flag. Gemma 3 has the scale and no per-layer embeddings, so every decoded
  token entered the model ~25x too small (gemma-3-270m: d_model 640) and
  generation degraded after the prompt; perplexity, scored through prefill,
  never showed it. On gemma-3-270m-it Q4_K_M the logits after a decode step
  differed from a prefill of the same tokens by up to 55; now they are
  identical. `test_gemma3_decode_parity_int` pins it. Llama, Qwen 3 and
  Gemma 4 output is bit-identical.
