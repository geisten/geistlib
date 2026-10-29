### Changed
- **MoE prefill on a GPU: the hot experts' GEMMs run on the tensor cores
  (#167).** The Vulkan tensor-core tiles need a multiple of 16 rows; a
  prefill chunk spreads its rows over the experts unevenly, so most hot
  experts' GEMMs fell back to the scalar kernel. Each expert's rows are now
  padded to a multiple of 16 with a repeated token that is never summed.
  Qwen3.6-35B-A3B UD-Q4_K_XL on an RTX 2080 Ti, the model on the GPU with
  its expert profile, Strata's request (2,726 prompt tokens, 128 decoded),
  4 interleaved rounds (run 1930): prefill 522 -> 655 tok/s, decode
  unchanged (71.6 / 71.3); Strata on the same box 605 / 62.2.
