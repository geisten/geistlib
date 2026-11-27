### Fixed
- **Vulkan on a discrete GPU: a KIVI or INT8 KV cache no longer runs its
  attention out of the card's BAR window (#605).** Vulkan has no kernel for
  these caches; their attention runs on the host. The cache buffers were
  scratch buffers, which Vulkan places in the BAR window, where every host
  read is an uncached read over PCIe, and the KIVI kernel read a head's
  query row from a scratch slot once per cache position. The cache buffers
  are in system memory now and the query row is copied once. Gemma 4 E2B
  Q4_K_M on an RTX 2080 Ti, a 512-token prefill, one or two runs each: KIVI
  7.5 to 8.6 s where it did not end within five minutes, INT8 1.4 to 2.1 s
  where it took 9.9 to 10.1 s, an INT8 decode step 196 to 335 ms where it
  took 738 to 839 ms. An F16 cache, which the card reads itself, is 45 to
  49 ms and 5.5 to 5.9 ms a step, so the quantized caches stay far behind
  it on Vulkan. KIVI logits are bit for bit the same on Metal, cpu_neon
  and cpu_scalar (two prompt lengths); x86-64 was not compared.
