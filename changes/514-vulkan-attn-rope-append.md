### Changed
- **Vulkan runs the attention prep of Llama, Qwen3, GraniteMoE and OLMoE in
  one dispatch (#514).** The new `fused->attn_rope_append` on Vulkan
  (`rope_append_{f32,f16}`, one workgroup per row and head) does what up to
  seven dispatches did: the per-head q / k RMSNorms where the family has
  them, both RoPEs (half-split over the first n_rot elements), the q scale
  and the F16 or F32 KV append. Same arithmetic in the same order: on the RTX
  2080 Ti it is bit-identical to the separate ops
  (`test_backend_vulkan_ops_unit`, six geometries). OLMoE's attention block
  takes the plan-bound op after its whole-row QK norms on every backend that
  has it (Metal: greedy tokens identical to before). On the RTX 2080 Ti the
  four OLMoE ops took ~12 us per layer, the fused one 3.6-4.4; `bench_revision_ab`
  against main (one hold of the bench lock, quiet host): decode Llama-3.2-3B
  Q4_K_M -1.5 % [-3.5, -1.3] (8/8), OLMoE-1B-7B -2.9 % [-6.2, +1.4] (8/10),
  Qwen3-30B-A3B and gpt-oss-20b (unchanged paths) and prefill within noise.
