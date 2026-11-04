### Changed
- **43 redundant `(void *)` casts are gone (#347).** The AVX-512/AMX kernels
  cast every pointer they hand to `_mm512_load*_si512` /
  `_mm512_store*_si512`, whose parameters already are `void const *` /
  `void *`; the casts hid nothing but could hide a wrong pointer. The 23
  casts left each do something: int to pointer (`mmap`, thread ids),
  dropping `const` or `_Atomic` for `safe_free` and the `geist_par_for`
  context, unifying a ternary, or the `(T *) (void *)` alignment idiom.
  Object code is unchanged.
