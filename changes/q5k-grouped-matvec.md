### Changed
- **Vulkan runs Q5_K experts in the grouped decode matvec (#169).**
  `linear_t_grouped` took F32, Q4_K and Q6_K; a layer whose down experts
  are Q5_K (most of Qwen3.6-35B-A3B UD-Q4_K_M's) fell back to one dispatch
  per expert and projection. `matvec_q5k_id` is `matvec_q5k` with the
  weight read by device address, one workgroup row per expert.
