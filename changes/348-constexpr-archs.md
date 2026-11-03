### Changed
- **The arch layers' constants are typed `constexpr` objects (#348).** The
  object-like `#define` and one-member `enum { }` constants of
  `src/archs/` (audio conformer, vision SigLIP, transformer) became
  `constexpr`, each with the type its literal had, so the arithmetic and the
  object code are unchanged (byte-identical on Apple clang, gcc-14 and clang
  x86-64). Where a comment stated a relation (`HEAD_DIM = AUDIO_HIDDEN /
  N_HEADS`, `MEL_N_FFT_BINS = MEL_FFT_LENGTH / 2 + 1`) the initializer now
  computes it.
