### Changed
- **cpu_neon: Q8_0 prefill runs the SDOT kernel on Apple too (#16).** Apple
  took dequant + Accelerate sgemm for Q8_0 prefill: every weight
  dequantized to F32 per 64-token chunk, eight single-threaded sgemm calls
  sharing two AMX units. The register-tiled SDOT kernel of #7, the default
  everywhere else, is now the Apple default too; `GEIST_Q8_0_NATIVE_MN=0`
  restores the old path. M1 Max, cpu_neon, 8 interleaved cycles with a
  control: Qwen3.5 0.8B prefill -42.1 % time at 128 tokens, -39.5 % at 512
  (8/8 cycles each), about llama.cpp's speed now (870 vs 860-890 tok/s at
  pp512); Qwen3 0.6B -40.7 % / -30.3 % (8/8); decode within noise. The
  activations are int8 per 32 in prefill as they already were in decode:
  next-token NLL over 128 teacher-forced steps after a 512-token prompt
  2.5594 against 2.5567 for sgemm and 2.5570 for cpu_scalar (llama.cpp
  2.5658), top-1 agreement with cpu_scalar 97.7 % against 96.1 %.
