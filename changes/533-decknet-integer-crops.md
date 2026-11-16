### Changed
- **decknet computes on integer crops, second step of #533 (EXPERIMENTAL,
  `DECKNET=1`).** A crop pixel is now an integer, for `geist_decknet_run`
  and a stream alike: the bilinear sample of the four camera bytes in fixed
  point (weights in 1/2^24, the value in 1/65536 of a byte), before any
  gain. The gain is applied behind the kernels, which are linear, and a
  kernel is the sum over its distinct weights of weight times the integer
  sum of the pixels that have it. An integer sum is the same in any order,
  so the compiler's vector types carry it (no intrinsics, no platform
  `#if`), and a stream can keep it from frame to frame while the brightness
  moves. The 1x1 layer computes four rows at a time and its `relu` selects
  instead of branching; its values are the bits they were. The outputs stay
  within the 1e-4 of the numpy reference that the parity test allows,
  unchanged; on the invented network the largest deviation went from 2.8e-7
  to 5.4e-7. The device's network was not run with it by the author (its
  fixture is not in the repository); CI runs it where the fixture is.
  An NV12 or YUYV frame of a stream is now converted tile by tile (16 x 16
  pixels, only the tiles a sample reads), each pixel once and in 16-bit
  arithmetic that gives the bytes of the documented formula for all 2^24
  inputs, instead of four times per sample.
  Raspberry Pi 5, 1920 x 1080, `tools/bench_stream` (an invented network,
  six slots; 300 frames; p50 in ms per frame, one thread): a stream 43.3 to
  21.4 for RGB8, 52.7 to 23.6 for NV12, 50.5 to 24.5 for YUYV (p99 22.6,
  24.4, 24.8); `geist_decknet_run` once per frame 106.5 to 62.9 for RGB8.
  The default build is unchanged: it compiles none of the files this
  touches. See `docs/DECKNET.md`, "Integer crops" and "Stream".
