### Fixed
- **Control-looking vocab strings are special tokens, as in llama.cpp
  (#404).** llama.cpp marks a list of control-looking strings CONTROL
  whatever their `tokenizer.ggml.token_type` (its end-of-generation names
  such as `</s>`, `<eos>`, `<|im_end|>`, and the FIM names when the file has
  no id for that role), and matches UNKNOWN entries such as `<unk>` too.
  geist matched only CONTROL and USER_DEFINED entries, so the text `</s>`
  was `<` `/s` `>` for Qwen3 (llama.cpp: its NORMAL token 128247) and
  `<eos>` three pieces for Gemma 4 (llama.cpp: token 1). geist now builds the
  same special set, including llama.cpp's Gemma 4 exception that keeps
  `</s>` plain text. Every vocab entry as input and the #399 corpus (the
  wikitext-2 test text, 2000 of its lines, edge cases and random strings)
  now tokenize as `llama-tokenize` does for SmolLM2, Llama 3.2, Qwen3,
  Qwen3.5, Gemma 4, gpt-oss, Llama 4 Scout and DeepSeek-V2-Lite; before,
  Qwen3 and Gemma 4 each differed on one vocab entry. No test prompt
  contains these strings, so no golden changed. Tokenizing speed is
  unchanged.
