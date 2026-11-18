### Added
- **#467 result: early exit at tau 0.9 is non-inferior at the 2 pp margin**
  (`benchmark/results/DECISION_EARLY_EXIT_TAU09_2026-11-18.md`).
  - On 3,000 new held-out MMLU questions at cap 8192, disjoint from the
    4,000 of #171 and #286, early exit at tau 0.9 scored 84.2 % vs 81.7 %
    for full reasoning; paired [+1.17, +3.88] pp.
  - It uses 0.35x the generated tokens. No latency run was declared.
  - Descriptively, tau 0.8 scored 83.1 % at 0.24x on the same questions;
    the two thresholds were not tested against each other.
