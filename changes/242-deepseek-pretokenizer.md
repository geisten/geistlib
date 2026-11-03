### Added
- **DeepSeek tokenizers (#242).** GGUFs with `tokenizer.ggml.pre` =
  `deepseek-llm` (DeepSeek-V2, V2-Lite) or `deepseek-v3` (DeepSeek-V3, R1)
  are split as llama.cpp splits them: its list of regexes, each over the
  pieces the previous one left. Token ids match `llama-tokenize` on
  `tools/data/ppl_alice.txt` and 214 other texts for both vocabularies.
