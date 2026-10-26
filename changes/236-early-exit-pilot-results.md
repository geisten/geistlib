### Added
- **#236 early-exit pilot on Bonsai 2 27B (50 calibration questions).**
  `benchmark/results/DECISION_EARLY_EXIT_PILOT_2026-10-26.md`:
  - answer probes during reasoning leave the generation unchanged and cost
    3.2 decode steps each;
  - stopping at the first probe with top-1 probability >= 0.8 matches full
    reasoning (78 %) at 0.40x the tokens.
  The paired intervals at n = 50 are +/-8-15 pp, so no policy is frozen yet.
