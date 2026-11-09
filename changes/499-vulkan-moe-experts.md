### Added
- **Vulkan runs GPT-OSS's and DeepSeek-type models' experts on the GPU
  (#499).** What Metal got in #373 / #376 / #482: the device router takes
  GPT-OSS's router bias and softmax over the selected and DeepSeek's sigmoid,
  selection bias, expert groups and weight scale (ids exact against the
  host's `moe_route`); new `fused->glu` (SwiGLU / `swiglu_oai` with the
  experts' biases), `moe_weighted_sum` (output biases by id) and
  `linear_t_ids` (id-indexed matvecs for decode, `moe_map` plus a grouped
  tensor-core GEMM for a prefill chunk); Q5_0 and MXFP4 get Vulkan kernels
  (matvec, register-tiled and tensor-core GEMM, id-indexed matvecs). A
  model whose experts all fit routes on the device (`GEIST_FUSED_MOE_DEVICE`),
  one that does not keeps the hot/cold split with its hot experts on the
  GPU, GPT-OSS's routed there as well (#162). On an RTX 2080 Ti
  (`tools/bench_revision_ab.py`, 8 cycles, quiet host): OLMoE-1B-7B
  Q4_K_M prefills 512 tokens 50.5 % faster [-50.9, -49.7] (6.1k tok/s,
  llama.cpp Vulkan 6.0k), decode within noise to 5.5 % faster; gpt-oss-20b
  (620 of 768 experts on the 11 GiB card) prefills 89.5 % faster
  [-89.7, -89.4] and decodes 51.8 % faster [-53.4, -50.5] than with every
  expert on the CPU. Perplexity (8 x 512, docs/OFFLOAD.md): OLMoE 45.891
  (main 45.958, llama.cpp Vulkan 45.834), gpt-oss-20b 29.402 (main 29.280
  with its CPU experts' int8 activations, llama.cpp Vulkan 29.502). New
  `caps.mla_device` names the backends that run deepseek2's latent
  attention on the device (Metal); Vulkan still refuses the absorbed layout
  (GigaChat3, DeepSeek-V3).
