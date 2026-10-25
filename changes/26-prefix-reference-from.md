### Added
- **`tools/eval_decision_prefix.py --reference-from` re-measures the prefix arm
  under another cut alignment against an earlier run's independent arm (#26).**
  The earlier run must share the model, binary, cases, profile, backend, threads
  and environment, and the runner checks this. Only `decision_prefix` runs; the
  earlier run's `decision_dense` samples are the reference. The amendment in
  `docs/DECISION_CAMPAIGN_PLAN.json` explains why it was needed. Bonsai 2 27B
  resolves a 320-token prefill chunk on Metal, not 256. With 256-aligned cuts,
  28/500 option-logit vectors differed (by at most 0.003) wherever one of the
  two prefill splits ended in a pass of 8 rows or fewer.
