### Added
- **Experimental native FP8 tensor prefill on M6 (#352).**
  `GEIST_METAL_FP8=1` stages packed quantized weights and FP32 activations
  into native E4M3 tensors with E8M0 block scales, accumulating in FP32.
  Apple GPU family 11 and Metal 4.1 are required; unsupported devices and
  compilation failures retain the existing path. Decode and default
  behavior are unchanged. Includes an independent rounding oracle and a
  reproducible throughput, memory and quality comparison. Remains off by
  default: native FP8 alone does not guarantee a throughput or RAM benefit;
  see `benchmark/results/M6_FP8.md` for measurements and quality limits.
