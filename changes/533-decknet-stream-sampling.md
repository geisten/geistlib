### Changed
- **decknet streams sample faster, and a gated frame that changed all over
  no longer costs a third more, third step of #533 (EXPERIMENTAL,
  `DECKNET=1`).** A bilinear sample computes its three colours at once in
  four 32-bit lanes (the compilers' vector types, no intrinsics, no platform
  `#if`: two 16-byte row loads, a byte shuffle per pixel, three
  multiplications by the differences from the first pixel, which is the
  integer `decknet_blend` gives with four), and a sample-table entry is 16
  bytes instead of 20: what a plane reads of its crop is runs of
  neighbouring pixels, so the table no longer names the crop pixel. A sample
  at the last pixels of a frame whose buffer ends with them is computed from
  those pixels alone. A gated frame in which more than 7 of 8 tiles were
  taken is computed as a frame that is not gated, from the kept tiles, and
  the next frame that mostly stands still fills what the stream keeps
  again; the taken tiles of a row are copied and converted in one piece.
  The results are the bits they were: stream, gated stream and
  `geist_decknet_run` still agree under `memcmp`, the bench's hashes over
  every output are unchanged, and the tolerance against the numpy reference
  was not touched. Raspberry Pi 5, 1920 x 1080, `tools/bench_stream` (an
  invented network; 300 frames; ms per frame, p50), RGB8 / NV12 / YUYV: a
  stream on one thread 21.2 / 23.5 / 24.0 before and 19.1 / 21.6 / 22.6
  after, on two threads 13.2 / 15.2 / 14.9 and 12.0 / 14.1 / 14.0; a gated
  stream when every tile changes, one thread, 25.7 / 28.3 / 29.0 and
  20.9 / 23.1 / 23.4, against 19.2 / 22.1 / 22.3 for a stream that is not
  gated. `make fuzz-libfuzzer` now links with the target's `LDFLAGS`
  (`fuzz_decknet_stream` needs the OpenMP runtime on macOS).
