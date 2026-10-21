### Changed
- **cpu_neon: FP32 attention works a KV head's query heads together
  (#16).** The NEON rows read each K and V row once per query head and
  sent every key's V update through memory. Up to four query heads of one
  KV head now share each K and V load and keep their outputs in registers
  over a block of keys, with the same arithmetic per head in the same
  order: logits byte-identical (Gemma 4 E2B, Llama 3.2 3B, Qwen3 0.6B).
  It runs in prefill, and in decode where the KV heads give every thread
  one (Llama 3.2 3B; Gemma 4 E2B decode keeps its single-head rows). M1
  Max, 8 interleaved cycles with a control: Gemma 4 E2B prefill of 1024
  tokens -12.1 % time (8/8); Llama 3.2 3B prefill of 1024 -7.9 % (8/8),
  decode at depth 1024 -12.0 % and at 128 -4.1 % (8/8); the rest within
  noise.
