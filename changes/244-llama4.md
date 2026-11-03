### Added
- **Llama 4 loads and runs (#244).** The GGUF architecture `llama4` (Scout
  17B-16E; Maverick's layout too) as llama.cpp's graph runs it: GQA with
  interleaved RoPE and the file's `rope_freqs.weight` frequency factors, q
  and k RMS-normalised after RoPE, every fourth layer without RoPE (its q
  scaled by llama.cpp's temperature) and full attention, the others chunked
  attention over 8192 keys; `interleave_moe_layer_step` MoE layers whose
  router picks by logit and scales the expert's input by the sigmoid weight,
  plus a shared expert, the other layers dense. Chunked attention runs every
  backend's attention on cache views that start at the chunk. Perplexity
  against `llama-perplexity` on the same tokens, Scout Q3_K_S on the Ryzen 9
  9950X, cpu_x86: 25.79 vs 25.86 on a technical text (-0.26 %), 1.7916 vs
  1.7799 on `tools/data/ppl_alice.txt` (+0.66 %, where geist's logits are
  3.7 times closer than llama.cpp's to an exact cpu_scalar run).
