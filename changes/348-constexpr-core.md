### Changed
- **base, engine and io constants are typed `constexpr` objects (#348).**
  The object-like `#define` and one-member `enum { }` constants of
  `src/base/`, `src/engine/` and `src/io/` (heap alignment, error buffer,
  tokenizer and GGUF magic numbers, reader limits) became `constexpr`, each
  with its literal's type; the object code is byte-identical.
