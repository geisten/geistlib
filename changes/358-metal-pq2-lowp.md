### Added
- **Opt-in M6 PQ2 prefill experiments for Bonsai (#358).** Compare FP8,
  INT8, packed INT2, register operands and prepared full-K matrices with
  the established Metal tensor path. Defaults, exact weights, decode and
  non-PQ2 models retain their existing kernels. CPU oracles and measured
  performance, memory and quality determine which configurations are useful.
