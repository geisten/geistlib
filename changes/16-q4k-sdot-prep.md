### Changed
- **cpu_neon: the Q4_K SDOT prefill kernel prepares a weight tile once
  (#16).** It transposed each 4-row tile's weights and gathered their
  scales again for every group of 8 tokens; it now does that once per tile
  (16 blocks at a time, on the stack). Logits byte-identical (Gemma 4 E2B,
  Llama 3.2 3B). M1 Max, 8 interleaved cycles with a control: prefill
  time -11.9 % / -11.6 % at 128 / 512 tokens on Gemma 4 E2B Q4_K_M,
  -14.1 % / -13.7 % on Llama 3.2 3B Q4_K_M (8/8 cycles each); decode
  within noise.
