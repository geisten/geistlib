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
  behaviour). On the Mac mini M6 against `main` (`bench_revision_ab.py`, 10
  cycles, run 4094): Qwen3.5 0.8B Q8_0 prefill −2.9 % [−3.0 %, −2.8 %] and
  decode −2.7 % [−2.8 %, −2.7 %] in time (7151 to 7367 and 152.4 to 156.7
  tok/s); Llama 3.2 3B Q4_K_M prefill within noise, decode −0.6 %
  [−0.8 %, −0.2 %]. Outputs are unchanged bit for bit: the logits hash of
  `bench_nll` equals main's on the decode and the prefill path, on both
  models on the M6 and on eight models on an M1 Max;
  `tests/test_metal_concurrent_unit.c` checks where the barriers land. A
  backend's first sequence takes about 2 ms longer (process start to first
  token 56.8 to 58.6 ms and 97.7 to 100.0 ms). `benchmark/c23_metrics.json`:
  `void *` ceiling 1300 to 1326, for the two Objective-C message wrappers
  the reflection and the encoder need.
