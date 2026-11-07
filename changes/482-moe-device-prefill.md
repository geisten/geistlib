### Changed
- **A MoE prefill chunk on Metal routes and runs its experts on the device
  (#482).** With every expert of a layer on the GPU (the model on Metal),
  a prefill chunk took the host route: the host waited for each layer's
  router logits, routed, gathered the rows and ran each expert as GEMMs of
  its own. Now the chunk takes the decode's device route: `moe_route` over
  all rows, and `linear_t_ids` from 9 rows on buckets the routed slots by
  expert on the device (`moe_map`, slot order kept) and runs every
  expert's rows as one GEMM tile column (`moe_mm_<T>`, llama.cpp's
  `mul_mm_id`: the 64 x 32 simdgroup tile, and on GPUs with matrix units
  (M5/M6) the tensor-API one), for Q4_K, Q6_K, Q8_0, MXFP4 and Q5_0. No
  host round trip per layer remains. `GEIST_MOE_DEVICE_PREFILL=0` keeps
  the host route. Decode and a batched decode's rows (up to 8) keep the
  id-indexed matvec.
