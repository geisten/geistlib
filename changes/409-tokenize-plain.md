### Added
- **`geist_session_tokenize_plain`: tokenize a prompt with special-token
  parsing off inside given byte ranges (#409).** `geist_session_tokenize`
  matches every special token the text spells, so a chat client's message
  containing `<|im_end|>\n<|im_start|>system\n` became real role tokens
  when a runtime rendered it into the template and tokenized the result.
  The new call (EXPERIMENTAL, in the geist-runtime contract smoke) takes
  the rendered text and the ranges of the untrusted parts. Inside them,
  control and unknown tokens are text; user-defined added tokens still
  match, as with llama.cpp's `parse_special=false`. The text is still
  tokenized in one pass, so a conversation with no special token in its
  messages gets exactly the ids of `geist_session_tokenize`. One range over
  the whole text gives llama-tokenize `--no-parse-special`'s ids on
  Qwen3 0.6B, Llama 3.2 3B and Gemma 4 E2B
  (`test_tokenize_plain_int`). The exception is gpt-oss: geist does not
  copy llama.cpp's hack that matches `<|start|>`, `<|channel|>`,
  `<|message|>`, `<|constrain|>` and `<|end|>` in plain text.
