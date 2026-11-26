### Changed
- **decknet streams: the Vulkan path keeps its buffers in a discrete GPU's
  own memory, and a stream's power is measured, eighth step of #533
  (EXPERIMENTAL, `DECKNET=1`).** On a discrete GPU
  `src/backends/vulkan/decknet.c` now makes the buffers its shaders work on
  device-local: the frame goes in through a staging buffer with the first
  stage's command buffer, and only what the CPU reads comes back (the
  plan's new `read` spans of the pixel memory per stage in
  `include/geist_decknet_gpu.h`, and the kernel sums). An integrated GPU,
  the Raspberry Pi 5's V3D and a software device keep mapped host memory;
  `GEIST_VK_DECKNET_MEMORY=host|device` overrides the choice. The results
  are the CPU's bits either way: `test_decknet_stream_unit` runs its Vulkan
  part with both placements. **Measured on an RTX 2080 Ti next to a Ryzen 9
  9950X** (1920 x 1080, `tools/bench_stream`, an invented network, bound to
  two cores, RGB8 / NV12 / YUYV): frames back to back take 6.9 / 6.6 /
  6.7 ms with host memory and 2.2 / 1.9 / 2.0 ms device-local (the second
  stage is waited for 4.7 ms instead of 0), and the card draws 57 W instead
  of 27 to 35 W. **Behind a camera at 60 frames per second the CPU is
  still the faster path there**: ten rounds of 600 frames per arm in turn,
  median p50 from offer to result 3.58 / 3.85 / 3.88 ms on one CPU thread,
  2.00 / 2.19 / 2.22 ms on two, 5.33 / 4.76 / 4.99 ms on the card (p99
  11.2 / 9.9 / 10.4 ms, below a frame period in every run), because the
  card is slow to answer after the pause between two frames (10.1 ms a
  frame behind a camera at 30, 1.9 ms at 240 and back to back). The card's
  path uses 0.9 s of processor time per 600 frames where one CPU thread
  uses 2.2 to 2.4 s, and costs the card 7.5 W above its idle 19 W (125 mJ a
  frame). **Energy:** `tools/stream_energy.py` samples what a host reports
  about itself from outside the bench, at most ten times a second and not
  in step with the frames, and lays it over each row (`bench_stream
  --window`, new, as is `--format`): the Raspberry Pi 5's PMIC rails, an
  NVIDIA card's board power, the RAPL package counter where it is readable
  (on the desktop it is root's, so the processor's energy there is not
  measured), a Mac's battery telemetry. On the Pi 5 behind a camera at 60
  (rails only: without the PMIC's losses and what hangs on 5 V; idle
  1.6 W): two CPU threads 3.56 / 3.58 / 3.75 W, 33 / 33 / 35 mJ a frame
  above idle; one thread 3.34 / 3.33 / 3.22 W for 546 to 573 of 600 frames;
  the V3D 2.3 W for 125 to 141 of 600 frames, 52 to 59 mJ per computed
  frame above idle. Sampling five times a second does not move the Pi's
  latency (p50 within 0.03 ms), ten times a second moves it by 1 %. On the
  M1 Max no power was measured: the machine was not quiet enough for its
  one-minute telemetry. `docs/DECKNET.md` ("Vulkan on a desktop GPU",
  "Energy") has the tables and what each number leaves out.
