### Added
- **decknet: a hand-written convolutional net for images, behind `DECKNET=1`
  (#532, EXPERIMENTAL).** The geistbox deck-check network is the first model
  in the library that is not a language model: two RGB8 camera images in (the
  deck now, the empty deck), one state vector per deck position out
  (`occ_front`, `occ_rear`, `wall_front`, `rack`, `plate`, `deep`, `tube_n`,
  `tube_r`, 96 tips, 2 x 8 tubes). `include/geist_decknet.h` has its own
  entry points, `geist_decknet_load` / `_load_from_memory`, `_info`,
  `_set_homographies`, `_run`, `_slot`, `_destroy`, and no tokenizer, session
  or backend handle; `src/archs/decknet/` holds the file contract and the
  forward pass (homography resample, 1x1 and small dense 2D convolutions,
  ReLU, sparse weighted pooling) in portable C, the same code on every
  platform. Every output matches the numpy reference (geistbox
  `decknet.py`) to about 1e-6; the issue allows 1e-4. The parity test runs
  everywhere on `tests/data/decknet_synth/`, a network whose every number is
  invented, and on the device's network where `GEIST_DECKNET_FIXTURE` names
  it: that fixture holds the deck geometry and is not in the repository. The
  homographies of either image can be replaced at run time, so a camera pose
  measured per run needs no new file. The loader fails closed: a file that
  differs from the contract in any way (a missing or extra tensor, a pooling
  row whose pixel is out of range, fractional or NaN, rows out of order, a
  non-finite weight, any truncation) is refused with a status and a message
  naming the tensor and the row; `tests/fuzz_decknet.c` fuzzes the loader and
  runs whatever it accepts. No call after the load allocates. The default
  build compiles none of it and is byte for byte what it was (all 138
  library objects, `libgeist.dylib` and all 307 binaries compared on macOS
  arm64); `make test-decknet` builds and tests it. `geist_model_load` keeps
  refusing a decknet file with `GEIST_E_UNSUPPORTED`. See `docs/DECKNET.md`.
