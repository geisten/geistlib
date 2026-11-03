### Changed
- **The GGUF tokenizer skips its special-token scan at bytes no special
  token starts with (#409).** Every encode compared each text position
  against every special token's first byte; a table built at load now
  answers that with one lookup. For an 8 KB, 13-turn chat on an M1 Max
  (min of 200 runs, three interleaved rounds, ids identical):

  | model | before | after |
  |---|--:|--:|
  | Llama 3.2 3B (256 specials) | 1153 µs | 144 µs |
  | Gemma 4 E2B | 1504 µs | 1078 µs |
  | Qwen3 0.6B | 485 µs | 392 µs |
  | SmolLM2 360M | 432 µs | 381 µs |
