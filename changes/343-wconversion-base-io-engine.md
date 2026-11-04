### Changed
- **Seven source directories and tools/ build with `-Wconversion
  -Wsign-conversion` (#343).** An implicit conversion that can change a
  value or its sign is a build error in src/base, src/io, src/engine,
  src/archs/vision_siglip, src/backends/{cpu_scalar,metal,vulkan} and
  tools/. The one site it found, in the tokenizer's UTF-8 decode, is a real
  bug, filed as #427; the conversion is written out until that fix lands
  with its parity test. The other directories stay without it: their 433
  sites are index and geometry arithmetic that only casts would silence,
  and none was a bug.
