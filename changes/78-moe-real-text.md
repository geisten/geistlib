### Changed
- **Mixture-of-experts numbers are measured on real text (#78).** On a
  pseudo-random prompt, as `bench_perf_sweep` and llama-bench use, Qwen3-MoE
  routes 40 % of its expert rows to experts the hot placement leaves on the
  CPU, against ~12 % on text, and the few it picks repeat and stay in the
  CPU's cache. `bench_perf_sweep --prompt-file` takes a real prompt and
  `bench_nll` reports its feed's tokens per second. On real text
  (Qwen3-30B-A3B Q4_K_M, RTX 2080 Ti + Ryzen 9 9950X) the model on the CPU
  with its hot experts on the GPU decodes at 41-43 tok/s against 34 on the
  CPU alone; the model on the GPU, 49-58 tok/s on the pseudo-random prompt,
  drops to 2.5-16 tok/s (its card idles at low clocks, and the CPU's OpenMP
  workers fall asleep between waits on it) and is experimental.
