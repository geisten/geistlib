### Changed
- **The speculative output head computes its finalists' logits on all
  threads (#16).** Phase 3 dequantized and dotted 512 (Llama 3.2) or 4096
  (Gemma 4) lm_head rows on one thread while the rest waited: on an M1 Max
  Llama 3.2 3B decode, 8.6 % of the main thread's samples. It now runs on
  geist_par_for with a scratch row per thread. Greedy tokens and logits
  byte-identical.
