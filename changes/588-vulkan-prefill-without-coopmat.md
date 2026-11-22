### Changed
- **Vulkan: prefill on a GPU without cooperative matrices runs attention and
  the dense GEMMs in register tiles; MoE prefill on an integrated Radeon is
  1.9-2.8x faster (#588).** On the Ryzen 9950X's integrated Radeon (RADV,
  RDNA2, shared memory, no `VK_KHR_cooperative_matrix`) MoE prefill ran at
  0.42-0.55x of llama.cpp Vulkan on the same GPU. `GEIST_VK_PROFILE` showed
  where: the host waits for the GPU all of the time (one flush per prefill),
  and the GPU spent it in the per-row kernels, attention one workgroup per
  (query row, head) and the GEMMs one output row per subgroup. OLMoE-1B-7B's
  512 tokens took 7.6 s: 32 % attention, 28 % dense GEMMs, 38 % the grouped
  expert GEMM (run 4027); Qwen3-30B-A3B's 26.8 s: 43 %, 27 % and 29 % (run
  4028); Qwen3.6-35B-A3B's 31.7 s: 55 % in its Q8_0 GEMMs and 34 % in the
  GEMMs of single experts (run 4031). Such a GPU now runs a batch of query
  rows through `attention_f16_rt.comp` (16 rows per workgroup, 64 keys per
  step; F16 KV cache, heads up to 256) and Q4_K / Q5_K / Q6_K / Q8_0 GEMMs of
  16 batch rows or more through a 128 x 64 tile (`mm_rt.glsl`). A tile reads
  a key, a value and a batch row once, where the per-row kernels read them
  once per query row and per weight row; operands stay f32, so the results
  are the per-row kernels' up to the order of the sums. The pipelines are
  created only without cooperative matrices: a GPU with tensor cores creates
  and runs what it did. pp512 on that GPU, before -> now, llama.cpp Vulkan
  (d8123504) beside it: OLMoE-1B-7B 67.3 -> 124.6 tok/s (123.1), Qwen3-30B-A3B
  18.7 -> 41.0 (37.5), runs 4030; Qwen3.6-35B-A3B 16.2 -> 45.8 (38.1), run
  4031. Decode is unchanged (27.86 -> 27.88, 10.05 -> 10.05, 9.09 -> 9.08
  tok/s). Perplexity of the 511 tokens after a prefilled context of 511
  (`GEIST_NLL_PREFILL=1 bench_nll <gguf> docs/OFFLOAD.md 1024 512`), cpu_x86 /
  Vulkan before / Vulkan now: 26.60 / 26.96 / 26.89, 32.76 / 32.88 / 32.92,
  17.01 / 16.93 / 16.90. On the RTX 2080 Ti the two revisions measure the
  same (OLMoE-1B-7B pp512 82.8 against 82.7 ms, 95 % interval of the change
  [-0.2 %, +0.3 %] over 12 interleaved cycles, the same pipelines in both
  profiles), and with `GEIST_VK_NO_COOPMAT=1` it prefills that model at about
  2480 instead of 1540 tok/s (run 4031). Other weight types, heads wider
  than 256 and an FP32 KV cache keep the per-row kernels.
  `test_backend_vulkan_ops_unit` now runs in the lavapipe CI leg, where the
  attention tile is taken.
