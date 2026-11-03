### Fixed
- **Llama 3.x GGUFs without `tokenizer.ggml.add_bos_token` get their BOS
  (#385).** `geist_model_add_bos` now decides as llama.cpp does: the flag
  when the file has one, else the default of the vocabulary (SentencePiece
  and WordPiece add BOS) and, for byte-BPE, of the pretokenizer (`llama-bpe`,
  `llama3`, `tekken`, `chameleon`, ... add it; Qwen's `qwen2`/`qwen35` do
  not, #281), and always for Gemma 4. Before, every byte-BPE file without
  the flag got none, so Llama 3.2 3B Instruct ran without BOS in
  `geist_session_set_prompt` and in every harness that asks eval_geist's
  `BOS` command, where llama-perplexity puts one at the start of each chunk.
  Llama 3.2 3B Q4_K_M perplexity at 512 tokens (wikitext-2, 8 chunks,
  cpu_neon, F16 KV, `ppl_vs_llama.py`) against llama.cpp's 11.2082: 12.3604
  (+10.3 %) before, 11.1411 (-0.6 %) after, 11.2236 (+0.14 %) with the
  `rope_freqs` fix of #379 as well, with no BOS forced in the harness. A chat
  template that renders its own `<|begin_of_text|>` still gets one BOS
  (#263). No other GGUF in the test set changes, so no logit golden moves.
