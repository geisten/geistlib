### Changed
- **Metal decode attention 2x faster on Gemma 4's head sizes (#17).** The
  split-KV decode kernel walks the cache in 32-key blocks per simdgroup
  (after llama.cpp's `flash_attn_ext_vec`), so a lane has a block's K/V
  loads in flight at once instead of one key's chain of load, reduce and
  softmax per iteration; head_dim 64/128/256/512 take it, other sizes keep
  the per-key kernel. The pass that merges the splits now spreads each
  head over 64-element threadgroups with the split weights computed once
  per simdgroup, instead of one thread looping over the splits per element
  (18-37 us per call before, 7-10 us after). One call on the M1 Max at 512
  keys, 8 query heads on 1 KV head: head_dim 256 43 -> 24 us, head_dim 512
  93 -> 38 us (2048 keys: 266 -> 55 us). Not bit-identical (summation
  order): max |diff| 3e-6 against cpu_scalar at one query row, greedy
  output of Gemma 4 E2B, Qwen3 0.6B and Llama 3.2 3B unchanged.
