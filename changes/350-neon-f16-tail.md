### Fixed
- **cpu_neon F16 prefill converts a subnormal weight in a row's last
  `n_in % 4` columns correctly (#350).** The dequant trampoline's scalar
  tail had its own F16 bit-twiddle, which turned a subnormal `m * 2^-24`
  into `2^-14 * (1 + m * 2^-10)`; it now calls quant.h's `fp16_to_fp32`.
  Only rows whose length is not a multiple of 4 reach that tail, so no
  shipped model was affected. `test_resolve_weight_unit` covers it and fails
  on the old code (`y = 1027.5`, want `3.5`).
