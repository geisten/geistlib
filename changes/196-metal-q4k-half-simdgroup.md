### Changed
- **Metal: Q4_K decode matvecs at K = 1536 run 25-35 % faster (#196).**
  The one-row Q4_K kernels let four 8-lane groups stride the 256-wide
  blocks by four; at Gemma 4's d_model 1536 (six blocks) half the groups
  idled on the second stride. `matvec_q4k_h` and the GeGLU `gate_up_q4k_h`
  give each half simdgroup its own rows and stride by two, for K a
  multiple of 512 but not 1024 and at least 1024 output rows. On the
  M1 Max: the 262144 x 1536 output projection 889 -> 646 us, the
  12288 x 1536 gate/up pair 88 -> 66 us. Sums round in another order.
