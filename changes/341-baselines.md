### Added
- **Phase 0 baselines of the C23 review and `tools/text_size.py` (#341).**
  `benchmark/c23_baseline.json` and `benchmark/results/C23_BASELINE.md`
  record main at 6642260d, before the first #333 change: `.text` per layer
  and library (macOS arm64, Linux arm64, Linux x86_64), clean build time,
  the bench-matrix cells at that commit (M6 CPU/Metal, Ryzen 9950X,
  RTX 2080 Ti Vulkan, Pi 5), Qwen3-30B-A3B and Qwen3.6-35B-A3B on the
  desktop, and the synthetic Llama 3.2 1B row. `text_size.py compare` is
  the +1 % `.text` gate of #333.
