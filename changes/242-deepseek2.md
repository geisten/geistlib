### Added
- **DeepSeek-V2 / V2-Lite load and run (#242).** The GGUF architecture
  `deepseek2` with the unsplit `attn_kv_b` (DeepSeek-V2-Lite GGUFs made
  before 2025), as llama.cpp's non-absorbed graph runs it: multi-head latent
  attention decompressed per token (the latent RMS-normalised and expanded
  by `kv_b` to every head's K and V, RoPE on the 64 pe dims only, YaRN with
  DeepSeek's log multiplier folded into the q scale), the first
  `leading_dense_block_count` layers dense, then the router plus the shared
  experts as one dense FFN. Files with the split `attn_k_b` / `attn_v_b`
  (absorbed MLA) are refused by name. Perplexity against `llama-perplexity`
  on the same tokens on the Ryzen 9 9950X, cpu_x86: 1.2227 vs 1.2224 on
  `tools/data/ppl_alice.txt`, 21.106 vs 21.088 on a technical text.
- **DeepSeek-V3's router** in `moe_route`: sigmoid scores, the
  `exp_probs_b` selection bias, group-limited selection (a group scores its
  two best; `expert_group_used_count` groups stay), renormalisation and
  `expert_weights_scale`, as llama.cpp's `build_moe_ffn`. Tested against a
  double reference; no V3-size model ran here.
