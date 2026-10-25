### Added
- **`bench_decision --prefix` and `tools/eval_decision_prefix.py` measure
  shared-prefix decision scoring against independent calls (#26).** With
  `--prefix` the wire input carries a cached-token count per case, and the new
  `decision_prefix` arm scores each prompt from a `geist_decision_prefix_prepare`
  snapshot of its first tokens. A snapshot is prepared when a case's cached
  tokens differ from the previous case's, and that preparation is reported as
  its own `prefix_prepare` record. The runner groups the questions by subject
  and cuts each subject's common prefix at the engine's prefill chunk. It runs
  the independent and prefix arms in one process, in resumable chunks under a
  wall-clock deadline, and reports bit-identity of the option logits, accuracy,
  p50/p95, end-to-end time, snapshot bytes and peak RSS. The campaign is declared
  in `docs/DECISION_CAMPAIGN_PLAN.json` (`shared_prefix_campaign_26`).
