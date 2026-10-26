### Added
- **#26 shared-state decision results on Bonsai 2 27B.**
  `benchmark/results/DECISION_SHARED_STATE_2026-10-25.md` covers three results:
  - per-subject prefix scoring on the 500 held-out MMLU questions: bit-identical
    to independent calls (500/500) at 0.40x the time;
  - forked reasoning generation from a prefix snapshot: token-identical;
  - a batched multi-sequence decode prototype: 3.47x aggregate tokens/s at 8
    sequences, with bit-exact invariance to the other batch members.
  It also contains an API proposal for batched decode.
