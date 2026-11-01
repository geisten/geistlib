### Added
- **Opt-in M6 INT8 prefill and an isolated FP16/FP8/INT8 benchmark (#355).**
  Q8_0, Q4_0 and Q4_K retain their mapped original weights while sharing
  rowwise INT8 preparation across output tiles and using full-K INT32 dots.
  This approximate path is opt-in and adds prepared-operand scratch.
  Independent CPU oracles check scaling, tails, scratch reuse and bypasses.
  Reproducible runners separate GEMM throughput from conversion/calibration
  and compare full-model speed, memory and output quality with Metal FP16.
