### Changed
- **decknet streams: what a camera's own buffers could save on Metal was
  measured, and the surface path was not built, last step of #533
  (documentation only).** On Apple a camera's pixel buffers lie on
  IOSurfaces; a stream on the Metal backend copies the rows of a frame into
  the GPU's memory, and a surface handed to the GPU would save that copy and
  nothing else. On an M1 Max (1920 x 1080, `tools/bench_stream --backend
  metal`, an invented network, seven rounds of 600 frames per arm in turn)
  the stream was run as it is and with the copy left out, behind a camera
  at 60 frames per second and back to back, with one and two threads for
  the CPU's part: the copy is 0.075 to 0.078 ms of an NV12 frame and 0.097
  to 0.098 ms of a YUYV frame, 2.2 to 4.2 % of frames that take 2.3 to
  3.4 ms, and leaving it out makes a frame shorter by just that (0.08 to
  0.12 ms at the median of the paired runs); p99 and the results later than
  a frame period do not tell the arms apart. A copy out of a locked
  IOSurface-backed pixel buffer costs what the same copy out of plain
  memory costs (0.08 ms for NV12, 0.09 ms for YUYV, the lock 0.001 ms). No
  camera was opened, and none was attached to the Raspberry Pi 5, so V4L2
  buffers were not fed to a stream there. No code changed. See
  `docs/DECKNET.md`, "The copy a surface would save".
