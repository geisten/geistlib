### Changed
- **decknet streams sum a kernel's rows of equal weights from prefix sums,
  and the slow passes behind a camera are accounted for, fourth step of #533
  (EXPERIMENTAL, `DECKNET=1`).** A kernel of few weights (a disc and a rim:
  two) is now also kept as spans, the neighbours of one kernel row, and a
  stream that samples a plane whole writes prefix sums along the plane's
  runs of crop pixels in place of the pixels, so that a span is the
  difference of two of them: 38 reads for a 9 x 9 kernel where the taps are
  81, and the same integer. The still path and a gated stream's kept crops
  sum tap by tap as before; the results are the bits they were (stream,
  gated stream and `geist_decknet_run` agree under `memcmp`, now also on a
  network whose kernels have spans; the bench's hashes over every output
  are unchanged; the tolerance against the numpy reference was not
  touched). Raspberry Pi 5, 1920 x 1080, `tools/bench_stream` (an invented
  network; 300 frames; ms per frame, p50), RGB8 / NV12 / YUYV: one thread
  19.4 / 22.1 / 22.4 before and 17.5 / 19.9 / 20.4 after, two threads
  12.0 / 13.9 / 14.2 and 11.2 / 13.1 / 13.7. The passes of 17 to 36 ms that
  a stream has a few of in 600 behind a camera at 60 frames per second are
  the board's and not the pass's: of 221 late results in 26 399 traced
  passes, 59 % follow a pass during which the kernel's memory compaction
  migrated pages (every 2.05 s on that board, the stream's pages among
  them), 34 % a pass next to another process at work, and neither the
  frequency, the temperature, a core taken away nor the second thread's
  wake-up had a part. `tools/bench_stream --camera FPS --trace` prints each
  slow pass with the process's minor faults and involuntary switches and
  the pages the kernel migrated meanwhile. Binding the two threads to cores
  needs `OMP_NUM_THREADS=2` next to `OMP_PROC_BIND` or `GOMP_CPU_AFFINITY`,
  or the second thread is not used; `docs/DECKNET.md` ("Two threads", "The
  tail behind a camera") has that, the runs and what a deployment can
  change on its board.
