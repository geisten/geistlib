### Added
- **decknet streams read UYVY frames, first step of #607.**
  `GEIST_DECKNET_FORMAT_UYVY` is the 4:2:2 layout U Y0 V Y1 (`2vuy` from a
  camera on macOS): the samples of YUYV in another byte order, read in
  place by the same conversion, on the CPU and by the Metal and Vulkan
  shaders. A caller with such a camera no longer swaps the bytes of every
  frame first. Every output is bit for bit that of the still path
  (`tests/test_decknet_stream_unit.c`, on the CPU and on Metal here; the
  Vulkan stream runs in CI), and a frame costs what a YUYV frame costs
  (M1 Max, 1920 x 1080, one thread: 5.16 ms against 5.26 ms at the median,
  the runs overlapping). Full range, BT.709 and BGRA are still not read.
  See `docs/DECKNET.md`, "Frames" and "Measured".
