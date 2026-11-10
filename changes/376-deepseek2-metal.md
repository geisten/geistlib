### Changed
- **DeepSeek-V2 / V3-layout models run on Metal: experts on the GPU, latent
  attention on the device (#376).** Until now every expert of a `deepseek2`
  model ran on the CPU, also with the model on Metal (the hot-expert code
  assumed every layer has experts; DeepSeek's first is dense), the latent
  attention was assembled on the host once per layer, and Metal refused the
  split `attn_k_b` / `attn_v_b` layout (GigaChat3, DeepSeek-V3 / R1, Kimi
  K2). Now:
  - `moe_gpu.c` skips layers without experts; `geist_moe_route_args` gains
    SIGMOID gating, the selection bias, the weight scale and expert groups
    (DeepSeek-V3's router), which Metal runs on the device (`moe_topk_g`)
    and Vulkan refuses. The MOE_DEVICE probe names the router
    (`geist_fusion_query.moe_route`); a model with dense lead layers or
    such a router puts experts only on a GPU whose probe accepts it.
  - Metal runs Q5_0 (matvec, simdgroup GEMM, MoE id matvec): DeepSeek-V2-Lite
    Q4_K_M stores 14 of its 26 expert layers' down projections as Q5_0.
    Explicit `metal` now loads Q5_0 models instead of refusing them.
  - MLA on a GPU with strided copies and `linear_t_ids` (Metal): the
    latent's RMSNorm, K/V assembly and, absorbed, `k_b` / `v_b` of every
    head as one `linear_t_ids` call. Metal attention takes heads up to 576.
    Vulkan keeps the decompressed layout on the host and refuses the split
    one.

  DeepSeek-V2-Lite Q4_K_M on Metal, M1 Max: decode 17 → 89 tok/s, prefill
  178 → 338 tok/s (128-token prompt); perplexity within 0.03 % of main's
  Metal, 32 greedy steps identical to llama.cpp Metal.
