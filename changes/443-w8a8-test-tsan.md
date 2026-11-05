### Fixed
- **`test_w8a8_gemm_unit` builds again under gcc-14 at `-O1` (`MODE=tsan`)
  (#443).** gcc threaded the zero-trip path of the test's quantize loop into
  `M == 0`, saw `Yg = malloc(0)` against `w8a8_gemm`'s
  `out[static n_tokens * n_rows]` and stopped the build with
  `-Wstringop-overflow` under `-Werror`. No scenario passes a zero dimension
  and every buffer is sized exactly, so it was no out-of-bounds write; the
  test now states the kernels' precondition (`M`, `N`, `K / 16` at least 1)
  and returns failure otherwise, which removes the path without a pragma.
  It fired with plain `-O1` too; `MODE=asan` hid it because UBSan's checks
  block the threading, and CI's TSan job builds only the multi-session tests.
