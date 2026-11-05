### Fixed
- **Metal: the DeltaNet decode kernels check their pipeline's thread limit
  (#266).** `dn_dec_v_rows` (speculative verify, 2–8 rows) runs 4 × head_v
  threads per threadgroup, and its compiled limit on the M1 Max is exactly
  512, the 4 × 128 that Bonsai 2 27B and Qwen3.5 need. A dispatch over
  the limit is dropped without an error, so a kernel that needs a few more
  registers left z and the states untouched and still returned OK. Both
  decode paths (`deltanet_mix`, `deltanet_mix_states`) now take the
  fallback when `maxTotalThreadsPerThreadgroup` is below the thread count.
