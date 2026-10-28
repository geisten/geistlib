### Added
- **#236 result at cap 8192:** `benchmark/results/DECISION_EARLY_EXIT_CAP8192_2026-10-28.md`.
  - On the 500 held-out MMLU questions, early exit at tau 0.8 matches full
    reasoning: 85.6 % each, paired [-4.74, +4.74] pp. The 2 pp margin is not
    established.
  - It uses 0.23x the generated tokens.
  - Of the 41 questions full reasoning leaves capped at 8192, the policy
    answers 39 (25 correctly).
