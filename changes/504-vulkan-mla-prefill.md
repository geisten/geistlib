### Changed
- **Vulkan's absorbed-MLA prefill attention shares each K tile across the
  q heads; GigaChat3 prefills 2.1-4.2x faster (#504).** GigaChat3 and
  DeepSeek-V3-type files have one latent KV head for all q heads. The
  tensor-core prefill kernel walked it once per q head, computed QK^T twice
  in each of three column thirds and ran one warp per SM; it took ~47 % of
  GigaChat3's 512-token prefill. With an F16 cache (the default) prefill
  now runs the multi-subgroup kernel with the q heads of a KV head folded
  into the 16 rows of a tile: one token's heads read each K / V chunk once
  and compute the scores once, in one online-softmax pass with an exact
  running max. RTX 2080 Ti, `tools/bench_revision_ab.py` against #508, 6
  cycles under the bench and GPU locks: GigaChat3 Q4_K_M prefill @512
  498 -> 240 ms, @2048 4857 -> 1143 ms; DeepSeek-V2-Lite split @512
  1287 -> 1145 ms, @2048 6774 -> 4773 ms; the decompressed files and
  Llama-3.2-3B unchanged. Against llama.cpp Vulkan GigaChat3 prefills at
  0.74x (512 tokens, was 0.35x) and 1.88x (2048, was 0.44x). Perplexity
  (8 x 512) -0.32 % (GigaChat3) and -0.00 % (V2-Lite split) from
  llama.cpp Vulkan; the attention unit cases stay at or below the old
  kernel's 2.2-2.8e-4.
