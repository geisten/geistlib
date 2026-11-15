### Added
- **decknet: a stream handle for camera frames, first step of #533
  (EXPERIMENTAL, `DECKNET=1`).** `geist_decknet_stream_create`,
  `_set_reference`, `_push_frame`, `_result`, `_slot` and `_destroy` run the
  deck-check network frame after frame from one camera. A frame is the
  caller's memory, described by format, size, stride and extent, read in
  place and never kept; the formats are RGB8, NV12 and YUYV (BT.601 limited
  range, the integer formula in `docs/DECKNET.md`). A stream keeps what does
  not change between frames: per homography a table of where each crop pixel
  samples the frame, only for the crop pixels the pass reads (region of
  interest), and the empty deck sampled once. The result is that of
  `geist_decknet_run` on the same two images bit for bit (RGB8 as it is, NV12
  and YUYV as converted by that formula); `tests/test_decknet_stream_unit.c`
  compares every output with `memcmp`. After the create nothing allocates,
  the first frame included (counted, compared with zero). A frame whose
  description does not fit its buffer is refused with `GEIST_E_INVALID_ARG`
  before anything is read; `tests/fuzz_decknet_stream.c` fuzzes format, size,
  stride and extent on exact-size buffers and checks the verdict, not only
  the absence of a crash. `tools/bench_stream` measures frames per second
  and p50/p99 per frame against the still path called once per frame, on
  frames from memory and a network invented in the bench.
  On a Raspberry Pi 5, one thread, 1920 x 1080: 9.4 to 23.1 frames/s for
  RGB8 (p50 106.5 to 43.3 ms, p99 107.2 to 44.1 ms), 8.7 to 19.0 frames/s
  for NV12 (p50 115.4 to 52.7 ms), 8.6 to 19.8 frames/s for YUYV; each of the
  three parts (frame in place with the empty deck sampled once, tables,
  region) was measured to gain on its own. That is not yet the 60 frames/s
  the issue aims at: the rest of a frame is the pass itself.
  Temporal gating, integer kernels, the drop-oldest queue and a GPU path are
  not in this step. The default build is unchanged (all 445 objects and 307
  binaries byte-identical on macOS arm64). See `docs/DECKNET.md`, "Stream".
