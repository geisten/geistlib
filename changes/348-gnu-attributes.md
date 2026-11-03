### Changed
- **GNU attributes have one spelling, `[[gnu::...]]` (#348).** The last
  `__attribute__((always_inline))`, `__attribute__((target(...)))` and
  `__attribute__((constructor))` became `[[gnu::...]]`;
  `tests/test_spelling_py.py` now fails on any `__attribute__` in code. The
  object code is byte-identical (Apple clang arm64, gcc-14 arm64, clang
  x86-64).
