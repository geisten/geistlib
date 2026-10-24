### Changed
- **MoE prefill on a GPU: the hot experts' GEMMs run on the tensor cores
  (#167).** The Vulkan tensor-core tiles need a multiple of 16 rows; a
  prefill chunk spreads its rows over the experts unevenly, so most hot
  experts' GEMMs fell back to the scalar kernel. Each expert's rows are now
  padded to a multiple of 16 with a repeated token that is never summed.
  Qwen3.6-35B-A3B on an RTX 2080 Ti, the model on the GPU with its expert
  profile, 2,048 prompt tokens: prefill 531 -> 687 tok/s (runs 1898, 1902).
