### Changed
- **Metal: decode attention blocks in four simdgroups for small grids
  (#196).** `attention_decv` walks each 32-key block with one simdgroup,
  a key at a time with a shuffle reduction per key: 13 us per call at
  head_dim 256 and 28 us at 512 on the M1 Max, the same at 128 and 2048
  keys. `attention_decq` scores eight keys per simdgroup with every K load
  issued before the reductions, and gives each thread a float4 of the
  output over the block's 32 keys. It runs for head_dim 128/256/512 while
  the grid has at most 256 threadgroups (Gemma 4: 8 heads); larger grids
  keep `attention_decv`. Gemma 4 E2B shapes, split + combine: sliding
  layers 19 -> 14.5 us at 128 keys, global layers 33.5 -> 14.5 us.
