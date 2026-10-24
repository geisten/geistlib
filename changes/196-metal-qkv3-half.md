### Changed
- **Metal: the fused q/k/v matvec takes the half-simdgroup Q4_K layout
  for q (#196).** Where `linear_t` runs `matvec_q4k_h` (K = 512 mod
  1024, at least 1024 rows: Gemma 4's q projection), `mv3` now does too,
  so the triple keeps the bits of three `linear_t`. Gemma 4 E2B's qkv3
  dispatch 19.8 -> 13.3 us (global layers 29 -> 22 us) on the M1 Max.
