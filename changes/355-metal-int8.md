### Added
- **Opt-in M6 INT8 prefill and an isolated FP16/FP8/INT8 benchmark (#355).**
  Q8_0, Q4_0 and Q4_K retain their packed weights and block scales while
  sharing one blockwise activation quantization across output tiles.
  Independent CPU oracles check scaling, tails, scratch reuse and bypasses.
  Reproducible runners separate GEMM throughput from conversion/calibration
  and compare full-model speed, memory and output quality with Metal FP16.
