### Changed
- **Metal: a prefill chunk or decode step is encoded into concurrent compute
  encoders, and a decode step commits its first command buffer early
  (#590).** A serial encoder runs its dispatches one after the other; the
  sequence's encoder is now a concurrent one, with a memory barrier before a
  dispatch only when it reads a buffer that a dispatch since the last barrier
  wrote, or writes one that such a dispatch read or wrote. What a kernel
  reads and writes comes from its pipeline's reflection, and buffers are
  compared as whole `MTLBuffer`s. A kind of sequence that spares fewer than
  one barrier in twenty (Llama 3.2 3B decode: one chain) goes back to serial
  encoders after 4096 dispatches. `GEIST_METAL_CONCURRENT=0` keeps the serial
  encoder throughout. The first command buffer of a decode step is committed
  after 16 dispatches instead of a full period of 192, so the GPU does not
  wait for the encoding (`GEIST_METAL_PIPELINE_FIRST`, `0` for the old
  behaviour). Outputs are unchanged bit for bit: the logits hash of
  `bench_nll` equals main's on the decode and the prefill path, on eight
  models on an M1 Max and on two on the M6 with the tensor kernels;
  `tests/test_metal_concurrent_unit.c` checks where the barriers land.
  `benchmark/c23_metrics.json`: `void *` ceiling 1300 to 1326, for the two
  Objective-C message wrappers the reflection and the encoder need.
