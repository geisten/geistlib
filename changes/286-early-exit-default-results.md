### Added
- **#286 result: early exit is non-inferior at the 2 pp margin**
  (`benchmark/results/DECISION_EARLY_EXIT_DEFAULT_2026-11-05.md`).
  - On 3,500 new held-out MMLU questions at cap 8192, early exit at tau 0.8
    scored 84.6 % vs 82.8 % for full reasoning; paired [+0.27, +3.27] pp.
  - It uses 0.24x the generated tokens. Live, p95 latency is 285 s vs
    1,178 s, and the answers match the offline ones on 300/300.
  - The gain comes from answers full reasoning loses to the cap. On
    questions it finishes, early exit is about 2.5 pp worse.
