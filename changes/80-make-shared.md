### Fixed
- **`make shared` builds the FFI shared library (#80).** It links
  `libgeist.so` / `libgeist.dylib` with the GEMM provider's libraries; the
  hand-written line in `examples/ffi/README.md` and CI left `cblas_sgemm`
  unresolved on Linux, and CI's FFI step piped the failure into `head`, so
  it passed without running. The step now fails with the example and
  checks for a non-empty continuation.
