### Fixed
- **cpu_x86: prefill and decode give a row the same bits on the W8A8
  kernels (#476).** The one-row W8A8 dot and the many-row GEMMs (w8x8 /
  w8x16, and under `-ffast-math` the row-major GEMM too) rounded a block's
  contribution in different orders, so a row's output differed in the last
  bit between decode and chunked prefill; a following int8 activation
  quantization could turn that into a whole grid step (7e-3 of the logits'
  scale on a DeepSeek fixture in the coverage build). Every W8A8 kernel now
  rounds in one order, written as fused multiply-adds the compiler cannot
  reassociate. F32 weights run as W8A8 (Gemma 4's per-layer projections,
  Qwen3.5/3.6's `ssm_alpha` / `ssm_beta`) also decode on the lane-parallel
  kernel they prefill on, and keep one layout instead of two (half their
  memory). Prefill and decode speed unchanged (A/B on a Ryzen 9 9950X).
