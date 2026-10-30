### Fixed
- **The cpu_neon Q6_K decode GEMV no longer returns success with `y`
  unwritten when its scratch cannot be allocated (#274).**
  `linear_q6k_decode_w6a8_pre` and `linear_q6k_decode_w6a8_x8_pre` keep
  thread-local scratch (the sub-block activation sums; the x8 form also its
  activation blocks) and returned silently when growing it failed, so under
  out-of-memory a decode step read whatever `y` held before. Both now return
  `[[nodiscard]] bool`, and `cpu_neon_w_q6k_m1` computes `y` with
  `geist_linear_ref` when they report false, as the other kernels' scratch
  failures already did. Reachable whenever the backend's shared workspace was
  already big enough (a previous `linear_mN`, or a recycled thread id) but
  the kernel's own buffer was not; `test_linear_fallback_unit` now runs that
  case and failed 18 checks before the fix. The row loops are unchanged
  instruction for instruction; the wrappers gain one compare.
