### Added
- **cpu_neon: opt-in split decode attention, `GEIST_ATTN_SPLIT_DECODE=1`
  (#16).** A decode step had one grouped attention item per KV head:
  Llama 3.2 3B's 8 on 7 decode threads (one thread ran two), Gemma 4
  E2B's 2. With the option, from 256 keys on each item's range is cut at
  fixed multiples of 128 keys (flash-decoding) and the parts are merged in
  double; deterministic across thread counts. M1 Max, decode at depth
  1024: about 9 % faster on Llama 3.2 3B, Gemma 4 E2B and Qwen3.5 0.8B.
  Off by default: it is not byte-identical, and cpu_neon's int8 decode
  activations make greedy output depart from the default's at near-ties
  (docs/BACKENDS.md). The default path is unchanged, bit for bit.
