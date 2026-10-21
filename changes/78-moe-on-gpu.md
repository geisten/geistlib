### Added
- **A mixture-of-experts model runs on a GPU, its hottest experts there
  and the rest on the CPU (#78 phase 3).** Loaded on a GPU backend,
  Qwen3-MoE runs attention, router, embedding and head on the GPU; its
  experts are resolved on a CPU backend of the model's own, and the GPU gets
  copies of the experts the router picks most often (`GEIST_MOE_PROFILE`,
  the #48 usage lines) in the memory the rest left. Per layer the hot
  experts run on the GPU as one batch while the CPU runs the cold ones; at
  decode that batch is four dispatches through the new
  `fused->linear_t_grouped` (Vulkan: grouped matvecs reading the weights by
  device address). Qwen3-30B-A3B Q4_K_M on an RTX 2080 Ti + Ryzen 9 9950X
  (11 GB card, 18.6 GB model, 2,807 experts on the GPU): decode 49-52
  tok/s against 34-36 on the CPU alone and 39.5-39.7 for llama.cpp
  `-ot exps=CPU`; perplexity 30.06 against 29.90 on the CPU. Prefill is
  still slower than on the CPU (215 vs 267 tok/s). Loaded on the CPU with a
  GPU as split backend, the model keeps its experts on the CPU and gives
  the GPU the hottest for decode steps.
