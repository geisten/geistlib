### Changed
- **A mixture of experts on the GPU prefills at llama.cpp's pace (#163).**
  With the model on the GPU, a prefill chunk's hot experts read their rows
  gathered from x on the device (`fused->rows_gather`, new) and the GPU
  adds their outputs per token onto the cold experts' sums
  (`fused->rows_weighted_sum`, now for any number of tokens); only the cold
  experts' rows cross the bus, and the hot batch runs while the CPU runs the
  cold experts. On a discrete GPU such a model prefills in chunks of 512
  rows (from 128): its experts get ~32 rows each instead of ~8.
  Qwen3-30B-A3B Q4_K_M, RTX 2080 Ti + Ryzen 9 9950X, real prompt: prefill
  490-498 tok/s from 166 (llama.cpp Vulkan's best split: 490), decode
  53-57 tok/s; perplexity 30.10 against 30.06 for cpu_scalar's exact
  arithmetic (cpu_x86, with int8 activations: 29.90).
