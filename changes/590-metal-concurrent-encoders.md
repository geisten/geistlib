### Changed
- **Metal: a prefill chunk or decode step is encoded into concurrent compute
  encoders (#590).** A serial encoder runs its dispatches one after the
  other; the sequence's encoder is now a concurrent one, with a memory
  barrier before a dispatch only when it reads a buffer that a dispatch since
  the last barrier wrote, or writes one that such a dispatch read or wrote.
  What a kernel reads and writes comes from its pipeline's reflection, and
  buffers are compared as whole `MTLBuffer`s. Outputs are unchanged bit for
  bit (logits hash of `bench_nll` against main on eight models, decode and
  prefill path; `tests/test_metal_concurrent_unit.c` for the barrier
  placement). `GEIST_METAL_CONCURRENT=0` keeps the serial encoder. M1 Max,
  Qwen3.5 0.8B Q8_0: decode 183.5 to 194.1 tok/s, prefill at 512 tokens 5025
  to 5197 tok/s, with a barrier left before 85 % of the dispatches. The
  first sequence of a backend takes about 7-10 ms longer on that machine.
  `benchmark/c23_metrics.json`: `void *` ceiling 1300 to 1326, for the two
  Objective-C message wrappers the reflection and the encoder need.
