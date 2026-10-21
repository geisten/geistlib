### Changed
- **A mixture-of-experts decode step on the GPU waits once per layer, not
  twice (#161).** The hot experts' batch is started without waiting
  (`fused->submit`), the CPU runs the cold experts meanwhile, and the GPU
  adds the hot outputs, weighted, onto the cold experts' sum the host
  hands it in host memory (`fused->rows_weighted_sum`; Vulkan: a ring the
  work in flight does not read). With two waits the card idled between
  them, ~30 % busy, and NVIDIA's driver kept it at 300-700 MHz after a short
  prompt: 3-5 tok/s with Qwen3-30B-A3B on an RTX 2080 Ti instead of 49-52.
