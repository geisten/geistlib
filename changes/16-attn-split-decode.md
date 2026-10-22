### Changed
- **cpu_neon: a decode step's attention splits long key ranges across
  threads (#16).** A decode step had one grouped attention item per KV
  head: Llama 3.2 3B's 8 on 7 decode threads (one thread ran two), Gemma
  4 E2B's 2. From 256 keys on, each item's range is now cut into parts at
  fixed multiples of 128 keys (flash-decoding), and the parts are merged
  with their max and sum in double. The cut points depend on the key
  positions only, so a step gives the same bits on any thread count. Not
  byte-identical to the unsplit kernel (the parts' sums are added, not
  carried through one running sum); short contexts and prefill are
  unchanged, bit for bit.
