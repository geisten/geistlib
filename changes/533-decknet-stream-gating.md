### Added
- **decknet streams: temporal gating and a second thread, second step of
  #533 (EXPERIMENTAL, `DECKNET=1`).**
  `geist_decknet_stream_set_gate(s, on, threshold)`: a gated stream keeps
  the 16 x 16 pixel tiles the network reads; a tile whose bytes differ from
  the kept ones by at most `threshold` (the sum of absolute differences over
  all its bytes, colour included) keeps them, and what the pass computed
  from them (samples, kernel sums, 1x1 units) is reused. The result is bit
  for bit that of `geist_decknet_run` on the image of the kept tiles, so
  threshold 0 is a stream without gating. `tests/test_decknet_stream_unit.c`
  builds that image with its own tiles and its own sum and compares every
  output with `memcmp`, at threshold 0 and above, and at the edges (a kernel
  window across a cell of the crop, a tile exactly at the threshold, a frame
  that is not checkable, the seams of the sample tables);
  `tests/fuzz_decknet_stream.c` runs a gated stream against one without.
  The first call that turns gating on allocates what it keeps; no frame
  allocates (counted, compared with zero).
  `geist_decknet_stream_set_threads(s, 2)` computes one half of the slots,
  and of an NV12 or YUYV frame's conversion, on a second thread; the same
  bits. `geist_decknet_stream_result` also reports the tiles the network
  reads, the tiles taken from the frame and the samples computed, and
  `tools/bench_stream` prints them with the frame bytes they stand for.
  Raspberry Pi 5, 1920 x 1080, `tools/bench_stream` (an invented network,
  six slots; 300 frames; ms per frame), RGB8 / NV12 / YUYV: a stream on two
  threads p50 13.1 / 15.0 / 15.4 (21.4 / 23.6 / 24.5 on one), p99 13.3 /
  15.3 / 15.7 in one run and 15.0 / 20.4 / 15.9 in a second: RGB8 is inside
  the 16.7 ms of the 60 frames per second #533 aims at, NV12 and YUYV are at
  it with about 1 ms to spare. Gated at threshold 0 on one thread, a block
  moving over a still scene (4.3 % of the tiles change): p50 5.9 / 5.5 /
  5.7, p99 6.3 / 5.8 / 6.0. When every tile changes, gating costs about a
  third more than a stream without it (29.2 / 31.3 / 33.2 on one thread). The
  drop-oldest queue and a GPU path are not in this step. See
  `docs/DECKNET.md`, "Gating", "Two threads" and "Measured".
