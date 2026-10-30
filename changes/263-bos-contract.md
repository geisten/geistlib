### Fixed
- **`geist_session_set_prompt` starts with exactly one BOS on both tokenizer
  paths (#263).** With a GGUF tokenizer it prepended BOS in front of a chat
  template's own (`<bos><|turn>user…` prefilled two BOS tokens; the Gemma 4
  templates in `bench_quality`, `bench_speculative` and the chat tests do
  this). With a `tokenizer.bin` it never prepended one, although
  `geist_model_add_bos` reported true. Now, when the model expects BOS,
  `set_prompt` prepends it unless the prompt's tokens already begin with it;
  `geist_session_tokenize` still adds none. A `tokenizer.bin` prompt that does
  not spell its BOS now gets one. The rule is in `include/geist.h` and
  `docs/API_CONTRACT.md`; `test_bos_contract_unit` checks it on both paths.
