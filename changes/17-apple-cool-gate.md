### Changed
- **Mac benchmarks start every run on a cool chip (#17).** On the M1 Max the
  GPU holds 1293 MHz only up to ~78-83 °C and back-to-back runs reach
  92-96 °C, so results depended on run order and load time: llama-bench's
  slower model load gave the GPU time to recover and made geist look 4-7 %
  slower in prefill, with equal tok/s per MHz. `tools/bench_host.py` reads
  the GPU and CPU temperature with macmon (no root) and an `apple_temp_gate`
  in the host gate waits until they are below its limits, or pauses 45 s
  where macmon is missing and records that it did. The Apple profiles of
  the cross-engine protocols (GPU 65 °C for Metal, CPU 65 °C for the CPU)
  and of `apple_cpu_protocol.json` enable it; `tools/bench_revision_ab.py`
  applies it on macOS before every run (`--no-cool` to skip) and reports the
  start temperatures. See
  `benchmark/METHODOLOGY.md`, "Cool start on Apple Silicon".
