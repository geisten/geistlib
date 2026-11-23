### Added
- **A profile of geist's Metal backend against mlx-lm on the Mac mini M6,
  and the probe that took it (#590).**
  `benchmark/results/METAL_PROFILE_M6_2026-11-23.md`: a decode step is bound
  by memory on both engines (6.32 against 6.24 ms of GPU time on the 8-bit
  Qwen3.5 0.8B pair) and geist's GPU has no command buffer for 0.28 ms
  between steps, which mlx-lm avoids by queueing the next step early; the
  4-bit pair's remaining decode gap is the GGUF's weight bytes; geist's
  prefill is GPU time only, 1.45–1.47x that of mlx-lm's batched pass.
  `tools/probes/metal_trace.m` is a library that replaces a handful of
  methods on the device's Metal classes and writes, per command buffer, its
  commit, GPU start and end, the wait on it and its dispatches per kernel.
  It loads into geist (`DYLD_INSERT_LIBRARIES`) and into mlx-lm (`ctypes`),
  so both are measured the same way; it can also turn an engine's serial
  encoders into concurrent ones. `tools/probes/metal_profile.sh` runs the
  model pairs of the mlx-lm comparison under it, `metal_trace_report.py`
  prints the tables. `bench_perf_sweep` stamps its phases into the trace
  when the library is loaded, and `bench_nll` prints a hash of every scored
  logits vector (`logits_fnv`), equal between two builds exactly when their
  logits are.
