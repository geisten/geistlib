### Changed
- **Mixture-of-experts numbers are measured on real text (#78).** On a
  pseudo-random prompt, as `bench_perf_sweep` and llama-bench use, Qwen3-MoE
  routes 40 % of its expert rows to experts the hot placement leaves on the
  CPU, against ~12 % on text, and the few it picks repeat and stay in the
  CPU's cache. `bench_perf_sweep --prompt-file` takes a real prompt,
  `bench_nll` reports its feed's tokens per second, and `moe-baseline.yml`
  sweeps llama.cpp's `--n-cpu-moe` through llama-server on the same prompt
  tokens. On real text (Qwen3-30B-A3B Q4_K_M, RTX 2080 Ti + Ryzen 9 9950X):
  llama.cpp's best split 52.7 tok/s decode, 490 prefill; geistlib on the CPU
  35, with its hot experts on the GPU 42-43 (robust); the model on the GPU
  57-60 in two runs but 2.3-2.5 in two others and 11-16 fed token by token,
  so it stays experimental. Prefill in every geist mode is about half of
  llama.cpp's.
