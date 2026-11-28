### Added
- **decknet streams read full-range and BT.709 frames, second step of
  #607.** `struct geist_decknet_frame` has a `colour`: BT.601 or BT.709, each
  in limited ("video") range or full range (`enum geist_decknet_colour`),
  as a camera says of its buffers. Zero is BT.601 limited range, which is
  what every NV12, YUYV and UYVY frame was read as until now, so a caller
  that does not set the field gets the same bits as before. A caller whose
  camera delivers `420f` or BT.709 material no longer converts the whole
  frame first or accepts shifted colours. The four conversions are one
  integer formula with six numbers each (`docs/DECKNET.md`, "Frames"),
  exact for all 2^24 values of (Y, U, V) on the CPU's 16-pixel path (clang
  and gcc 14 on arm64, checked outside the tree) and bit for bit the still
  path in `tests/test_decknet_stream_unit.c` on the CPU and on Metal; the
  Vulkan stream runs in CI. A gated stream given the same bytes with
  another colour takes every tile again. `tools/bench_stream --colour`
  measures them: on an M1 Max a 1920 x 1080 frame of BT.601 limited range
  costs what it cost (NV12 5.02 ms against 5.03, YUYV 5.11 against 5.12,
  one thread, median of five alternating rounds) and one of the other
  three 0.08 to 0.11 ms more. A value the enum does not name is refused with
  `GEIST_E_INVALID_ARG`. BGRA is still not read.
