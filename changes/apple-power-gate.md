### Changed
- **Mac benchmarks require AC power with Low Power Mode off.** The M1 Max
  was found on battery in Low Power Mode, measuring about 1.5x slower with
  no other sign in the results. `tools/bench_host.py` reads `pmset -g batt`
  and `pmset -g` before every run (a Mac without a battery reads as AC) and
  an `apple_power_gate` waits, polling every 30 s, until the Mac is on AC
  with Low Power Mode off, failing the run as busy after the gate's timeout;
  each run records the power state. The Apple profiles of the cross-engine
  protocols and `apple_cpu_protocol.json` enable it,
  `tools/bench_revision_ab.py` applies it on macOS (`--no-power-gate` to
  skip), and the decision campaign's latency gate checks it on macOS. See
  `benchmark/METHODOLOGY.md`, "Power gate on Apple Silicon".
