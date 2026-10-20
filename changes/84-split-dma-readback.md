### Changed
- **Layer split: the GPU stage's output rows come back by DMA (#84).** The
  split read them on the host straight out of the stage's scratch, which on
  Vulkan is the uncached BAR window (24-90 MB/s, #78). A split session now
  also gets an IO buffer (cached host memory) on the split backend; a
  device copy fills it and the host reads that. Qwen3-14B Q8_0, split 26,
  RTX 2080 Ti + 9950X, 16 interleaved cycles: prefill -3.9 % time
  ([-4.6 %, -1.5 %], 15/16 cycles faster), decode within noise; two
  cpu_x86 instances still byte-identical. `split-ab.yml` runs the split's
  tests and a parent/patch A/B on the GPU box.
