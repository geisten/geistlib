### Changed
- **PQ2_0 linears with a handful of rows no longer pay for a full tile
  (#105).** Speculative verify and the replay after a partial accept run
  2–8 rows through every PQ2_0 linear. Metal took the 32-row simdgroup GEMM
  for them; Vulkan took the register-tiled GEMM, whose cost grew by a GEMV
  per row. New kernels decode each weight once and run the rows on the
  matrix units:
  - Metal: `matmul_pq2_m8` for 2–8 rows. Codes go straight into
    `simdgroup_half8x8` thread elements, with no staging.
  - Vulkan: `mm_pq2_0_m16_cm` for 2–15 tokens on the tensor cores. It uses
    32 weight rows per workgroup and 4 k-slices, and loads whole blocks with
    the next one in flight.

  On Bonsai 2 27B, an m-row pass (`prefill_tokens(m)`, median of 7) costs:

  | Backend | m | before | after |
  |---|--:|--:|--:|
  | M1 Max, Metal | 2 | 434 ms | 138 ms |
  | M1 Max, Metal | 8 | 485 ms | 133 ms |
  | RTX 2080 Ti, Vulkan | 2 | 74 ms | 71 ms |
  | RTX 2080 Ti, Vulkan | 8 | 221 ms | 89 ms |

  Speculative decode with n-gram drafts (k_max 4):
  - Metal (3 prompts): 3.96 → 6.71 tok/s.
  - Vulkan (12 prompts): 12.4 → 14.6–14.8 tok/s.

  Both still lose to plain decode (17 and 38 tok/s). What remains is the
  prefill path's fixed cost and the replay; see #105. Parity against
  cpu_scalar covers 2, 4, 5, 7, 8 and 15 rows, including ragged output tiles.
