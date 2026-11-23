### Added
- **A probe that records every Metal command buffer of an engine (#590).**
  `tools/probes/metal_trace.m` is a library that replaces a handful of
  methods on the device's Metal classes and writes, per command buffer, its
  commit, GPU start and end, the wait on it and its dispatches per kernel.
  It loads into geist (`DYLD_INSERT_LIBRARIES`) and into mlx-lm (`ctypes`),
  so both are measured the same way; it can also turn an engine's serial
  encoders into concurrent ones. `tools/probes/metal_profile.sh` runs the
  model pairs of the mlx-lm comparison under it on the Mac mini,
  `metal_trace_report.py` prints the tables. `bench_perf_sweep` stamps its
  phases into the trace when the library is loaded, and `bench_nll` prints a
  hash of every scored logits vector (`logits_fnv`), equal between two builds
  exactly when their logits are.
