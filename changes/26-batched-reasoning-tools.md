### Added
- **`tools/bench_reasoning_batch` and `tools/eval_decision_batched_reasoning.py`
  run greedy reasoning through `geist_session_decode_batch` (#26).**
  - The driver keeps up to 8 sessions decoding together and refills a slot as
    soon as its question finishes (continuous batching). Each subject's prompt
    prefix is prefilled once and restored per question.
  - Per question it reports the generated tokens and how many came from each
    batch size.
  - The runner is chunked and resumable, with a frozen manifest. It scores
    every answer exactly as the #171 campaign did.
  - The campaign is declared in `docs/DECISION_CAMPAIGN_PLAN.json`
    (`batched_reasoning_26`).
