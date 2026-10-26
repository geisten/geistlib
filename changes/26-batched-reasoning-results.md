### Added
- **#26 batched reasoning accuracy on Bonsai 2 27B.**
  `benchmark/results/DECISION_BATCHED_REASONING_2026-10-26.md`: the 500
  held-out MMLU questions at cap 2048, decoded 8 at a time through
  `geist_session_decode_batch`.
  - Accuracy 78.8 %, against 79.0 % for #171: -0.2 pp, interval
    [-1.27, +0.87] pp, non-inferior at 2 pp.
  - 471/500 generations are token-identical to #171; the one changed answer
    comes from batching.
  - 58.4 tokens/s aggregate decode against about 15 single-sequence;
    0.29x the end-to-end time.
