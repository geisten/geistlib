### Changed
- **cpu_neon: SiLU and the SwiGLU product run on all threads in prefill
  (#16).** `silu` ran on the calling thread with libm's `expf`, and the
  product with `up` was a second serial pass: 11 % of the main thread's
  Llama 3.2 3B prefill on the M1 Max while the other threads waited.
  cpu_neon now binds the fused `silu_mul` and splits both over the team
  from 16384 floats on (a decode token stays on the calling thread).
  Logits byte-identical. M1 Max, 10 interleaved cycles with a control:
  prefill time -10.3 % at 512 tokens on Llama 3.2 3B Q4_K_M (10/10),
  -15.3 % / -15.7 % at 128 / 512 on Qwen3.5 0.8B Q8_0 (10/10 each);
  Gemma 4 E2B (GEGLU) and decode within noise.
