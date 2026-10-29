### Changed
- **A mixture of experts split between a discrete GPU and the CPU routes its
  decode steps on the GPU (#162, Vulkan).** The GPU ranks the router's
  experts as the host does (`fused->moe_route`, a one-pass rank shader) and
  starts the experts it holds by those ids (`fused->linear_t_sel`) in the
  same submission as the router, while the host waits for the router alone
  (`fused->submit_mark` / `map_marked`, an event in the command buffer) and
  runs the cold experts. The ~40 us per layer the GPU sat idle between the
  router and the hot experts is gone: decode +7 % on Qwen3-30B-A3B and +9 %
  on Qwen3.6-35B-A3B (RTX 2080 Ti, 512 tokens after a 512-token prompt;
  busy host +4 % / +14 %), perplexity unchanged. `GEIST_MOE_GPU_ROUTE=0`
  routes on the host as before.
