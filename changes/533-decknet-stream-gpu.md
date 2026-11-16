### Added
- **decknet streams: the integer part of a frame on a GPU, sixth step of
  #533 (EXPERIMENTAL, `DECKNET=1`).** `geist_decknet_stream_set_backend(s,
  be)` puts a stream on the GPU of a backend that has a decknet path, today
  `metal`; `nullptr` takes it back to the CPU. The GPU converts an NV12 or
  YUYV frame, samples the crops and sums the kernels' pixels, all in 32-bit
  integers, in two command buffers; the gains, the 1x1 units and the
  pooling need F64 and stay on the CPU, which computes the floor planes
  from the first command buffer while the GPU runs the second. The results
  are the bits of the CPU stream and of `geist_decknet_run`
  (`test_decknet_stream_unit` runs its frames once more on the GPU and
  compares with `memcmp`; `bench_stream --backend metal` hashes every
  output of every frame). A frame's rows are copied to memory the GPU
  reads; a gated frame is computed on the CPU; a frame allocates nothing in
  the library. `include/geist_decknet_gpu.h` is what a backend implements
  (a plan, tables of 32-bit words, two stages), `src/backends/metal/decknet.c`
  implements it. Apple M1 Max, 1920 x 1080, `tools/bench_stream` (an
  invented network), behind a camera at 60 frames per second, 10 runs of
  600 frames per arm in turn, RGB8 / NV12 / YUYV, median of p50 and
  processor time of the process for 600 frames: CPU on one thread 5.25 /
  5.31 / 5.38 ms and 3.4 to 3.5 s, CPU on two threads 2.85 / 2.85 / 2.92 ms
  and 3.6 to 3.7 s, Metal with one thread for the CPU's part 2.95 / 2.90 /
  2.87 ms and 1.7 s, with two 2.54 / 2.52 / 2.51 ms and 1.8 s. The
  processor time does not see what the GPU draws; no power was measured.
  **Nothing was measured on a Raspberry Pi 5, and there is no Vulkan
  code**: `docs/DECKNET.md` ("On a GPU") has the parts of a frame, what was
  tried and dropped, what a Vulkan port carries over and what only the
  board can say. `tools/bench_stream` takes `--backend cpu|metal` and
  reports the processor time of each row.
