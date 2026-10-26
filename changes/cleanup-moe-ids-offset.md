### Changed
- **`geist_backend_fused`: `moe_route`, `linear_t_ids` and `linear_t_sel`
  lose their `ids_offset` parameter (EXPERIMENTAL backend table).** Every
  caller passed 0; the ids now start at the beginning of the `ids` buffer.
  The Metal and Vulkan implementations drop the matching alignment and
  overflow checks. Output unchanged.
