### Added
- **decknet streams: a queue of one between the camera and the pass, third
  step of #533 (EXPERIMENTAL, `DECKNET=1`).**
  `geist_decknet_stream_offer(s, frame, &dropped)` is for the thread that
  captures: one atomic exchange of a pointer, so it never waits and never
  allocates. The frame becomes the waiting one; if a frame was waiting
  already, the pass was slower than the camera, and that older frame is
  dropped, handed back in `*dropped` and counted
  (`geist_decknet_stream_result.n_dropped`).
  `geist_decknet_stream_push_offered(s, &done)` is `push_frame` on the
  waiting frame, for the thread that computes; it does not wait for a frame
  (when to call it is the caller's: the stream has no thread and no lock),
  and hands the frame back in `*done`. Nothing is copied: a frame, the
  struct and its bytes, is lent to the stream from the offer until that
  pointer comes back, which happens exactly once, as `*dropped` or as
  `*done`. The result carries the timestamp of the frame it was computed
  from, as before. `tests/test_decknet_stream_unit.c` runs it step by step,
  with a consumer that takes one pass for every three frames a second
  thread offers (the third is computed, two are dropped, in each round) and
  with that thread offering 3000 frames without waiting: every frame comes
  back once, the computed ones in order and with the bits of a `push_frame`
  on the same frame; clean under ThreadSanitizer. The no-allocation ratchet
  and `tests/fuzz_decknet_stream.c` include both calls.
  `tools/bench_stream --camera FPS` offers the frames from a second thread
  at that rate and reports frames computed and dropped and the time from
  the offer to the result; also new there: `--size 720|1080`, and
  `--scene half` and `--scene burst`.
