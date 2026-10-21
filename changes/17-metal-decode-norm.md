### Changed
- **Metal decode RMSNorm about 30 % faster per call (#17).** A
  single-row `rmsnorm` / `rmsnorm_add` keeps its x, w and residual values
  in registers (up to four per thread) and runs `cols / 4` threads instead
  of 1024, so every load is in flight before the reduction and fewer
  simdgroups meet at the barrier. A chain of 1536-wide decode norms on the
  M1 Max: 8.0 -> 5.6 us per dispatch. Gemma 4 E2B issues about 180 of them
  per token. Prefill rows keep the old loop and its bits; decode results
  differ in summation order only (Metal vs cpu_scalar within 5e-7, greedy
  output of Gemma 4 E2B, Qwen3 0.6B and Llama 3.2 3B unchanged).
