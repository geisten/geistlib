### Added
- **mlx-lm measured against geist's Metal backend on the Mac mini M6.**
  `benchmark/results/MLX_VS_GEIST_M6_2026-11-22.md`: six model pairs (GGUF
  for geist and llama.cpp, an MLX conversion for mlx-lm; Qwen3.5 0.8B at 8
  and 4 bit, Llama 3.2 3B, Qwen3.5 4B, Gemma 4 E2B, Qwen3.5 9B), three
  interleaved cycles each. mlx-lm 0.32.2 is faster on every pair: prefill at
  512 tokens 1.18–1.39x (2.88x on Gemma 4 E2B, where its prefill does not
  compute the 20 KV-shared layers for prompt tokens), decode 1.08–1.30x.
  geist reaches its first token 9–19x sooner from a warm page cache
  (0.05–0.09 s against 0.79–1.29 s). The pairs are different quantizations,
  so this compares engine and format together. The probe is
  `tools/probes/mlx_vs_geist.sh`; it also found that `bench_perf_sweep`
  prints `"decode_tps":inf` when a decode time reads zero.
