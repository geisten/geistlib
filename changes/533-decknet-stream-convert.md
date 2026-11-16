### Changed
- **decknet streams: an NV12 or YUYV frame costs little more than an RGB8
  one, fifth step of #533 (EXPERIMENTAL, `DECKNET=1`).** The 2 ms such a
  frame took longer than an RGB8 frame on two threads were its conversion,
  a pass of its own that wrote an RGB image of the frame and had the slots
  read it back. A stream that is not gated now converts slot by slot: the
  thread that computes a slot converts the tiles that slot reads right
  before it samples them, into a packed image of just those tiles, which
  its next slot overwrites; and the conversion computes 16 pixels at a time
  in the compiler's vector types (no intrinsics) instead of as loops for
  the compiler to make vectors of. The bytes are the ones the documented
  formula gives for all 2^24 inputs, and the results are the bits they were
  (stream and `geist_decknet_run` agree under `memcmp`, the bench's hashes
  over every output are unchanged). A stream allocates one more image of
  the frame's size in its create (6 MB at 1920 x 1080). Raspberry Pi 5,
  1920 x 1080, `tools/bench_stream` (an invented network), two threads
  bound to two cores, behind a camera at 60 frames per second, 10 runs of
  600 frames before and after in turn, RGB8 / NV12 / YUYV: median of p50
  11.2 / 13.1 / 13.4 ms before and 11.2 / 11.6 / 11.9 ms after; median of
  p99 12.0 / 13.7 / 17.2 and 12.0 / 12.8 / 12.5 ms; runs with p99 below a
  frame period 10 / 9 / 4 of 10 and 10 / 10 / 10; late results of 6000
  9 / 21 / 67 and 10 / 1 / 9. One thread, back to back: 17.6 / 19.9 / 20.3
  before, 17.3 / 17.9 / 18.2 after. `docs/DECKNET.md` ("Where the
  conversion's 2 ms went") has the parts of a frame before and after and
  what was tried and dropped.
