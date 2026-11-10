### Fixed
- **GPT-OSS's expert activation no longer turns a strongly negative gate
  into NaN on x86 gcc builds (#522).** The host's swiglu_oai computed
  `g / (1 + expf(-alpha * g))`; below `g = -52.13` the `expf` overflows and
  the divisor is infinite. gcc's `-ffast-math` on x86 vectorizes a float
  division as `rcpps` plus a Newton-Raphson step, which gives `0 * inf =
  NaN` there instead of 0. cpu_scalar then stopped with `moe: layer 19
  router logits are not finite` on gpt-oss-20b MXFP4; cpu_x86 quantizes the
  expert activations to int8, which hid the NaN and left wrong values in
  those gate units instead. arm64 and clang builds were not affected. The
  activation (and the plain SwiGLU host path beside it) now uses
  `silu_fp32_ooo`'s form with the exp argument never above 0.
  `test_gpt_oss_unit` runs the fixture with gate biases down to -100: on an
  x86 gcc build cpu_scalar failed with the router error before the fix and
  matches the double reference within 1e-5 after it, and cpu_x86's relative
  L2 to that reference went from 0.82 to 0.009.
