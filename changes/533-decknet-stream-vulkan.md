### Added
- **decknet streams: the GPU path on Vulkan, measured on the Raspberry Pi
  5's GPU, seventh step of #533 (EXPERIMENTAL, `DECKNET=1`).**
  `geist_decknet_stream_set_backend(s, be)` takes a `vulkan` backend as it
  takes a `metal` one: `src/backends/vulkan/decknet.c` implements
  `include/geist_decknet_gpu.h` with the two integer kernels as compute
  shaders (`decknet_sample.comp`, `decknet_sums.comp`), a submission with a
  fence per stage, and host-visible mapped buffers, so nothing is uploaded
  but the frame's rows. A frame's bytes are read out of 32-bit words (the
  V3D has no 8- or 16-bit types in a shader). The results are the bits of
  the CPU stream: `test_decknet_stream_unit` runs its stream tests once more
  on a Vulkan backend where one has a device (on the Pi 5 on the V3D and on
  Mesa's llvmpipe; in CI on lavapipe, in the `vulkan-lavapipe` leg), and
  `bench_stream --backend vulkan` hashes every output of every frame.
  The backend refuses the V3D for models (#92), so
  **`GEIST_VK_DECKNET_ONLY=1`** makes a Vulkan backend for decknet streams
  alone: no subgroup requirement, no model pipeline, every weight refused;
  without the variable the backend is what it was. **The measurement says
  not to use it on the board.** Raspberry Pi 5, 1920 x 1080,
  `tools/bench_stream` (an invented network) behind a camera at 60 frames
  per second, 10 runs of 600 frames per arm in turn, bound to two cores,
  RGB8 / NV12 / YUYV, median of p50 from offer to result: CPU on two
  threads 11.2 / 11.6 / 11.9 ms with p99 below a frame period in each of
  the 30 runs (median p99 13.9 / 12.4 / 12.9 ms) and 22 to 24 ms of
  processor time per frame; V3D with one thread for the CPU's part 81.5 /
  88.5 / 90.6 ms (p99 91.7 / 98.3 / 99.7 ms, no run below a frame period,
  125 to 140 of 600 frames computed, 40 ms of processor time per computed
  frame), with two threads 77.7 / 85.0 / 86.4 ms. The GPU's own time is
  67 ms a frame (35 ns a sample), and the CPU's part takes 39 ms on a
  mapped V3D buffer where it takes 8 ms on ordinary memory. No power was
  measured. `docs/DECKNET.md` ("Vulkan, and the Raspberry Pi 5") has the
  table, where a frame goes and what was not tried; a discrete GPU was not
  run. `tools/bench_stream` takes `--backend vulkan`.
