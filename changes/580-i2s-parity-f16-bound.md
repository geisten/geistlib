### Fixed
- **`test_i2_s_parity_unit` no longer fails now and then on the macOS GCD
  leg (#580).** Its F16 check compared the kernel with a float reference
  relative to the row's sum. The test's `rand()` sequence is not the same
  in every run of the no-OpenMP build, and in about one run in sixty row
  19's 2560 terms cancelled to -0.23, where float rounding alone is 2.3e-3
  of the sum against a bound of 2e-3. The reference is now summed in double
  and the difference is measured against the sum of the terms' magnitudes
  (bound 1e-6; one wrong term is about 1e-4). 400 runs of the ASan build:
  0 failures, 5 of 300 before.
