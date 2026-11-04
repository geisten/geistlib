### Changed
- **src/base, src/io and src/engine build with `-Wconversion
  -Wsign-conversion` (#343).** An implicit conversion that can change a
  value or its sign is a build error there. The one site it found, in the
  tokenizer's UTF-8 decode, is a real bug, filed as #427; the conversion is
  written out until that fix lands with its parity test.
