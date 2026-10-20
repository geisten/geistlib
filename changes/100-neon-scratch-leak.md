### Fixed
- **cpu_neon kernels no longer leak their scratch with every thread that
  ends (#100).** The q4_K/q6_K/TL1 kernels' `_Thread_local` high-water
  buffers grow through `heap_tl_grow`, which frees them at thread exit; a
  host that runs each request on a new thread leaked up to the largest
  prefill's packed activations per thread. LeakSanitizer reported them,
  so the ASan jobs failed on `test_linear_fallback_unit`. Steady state
  unchanged: the same capacity compare, no call, no allocation.
