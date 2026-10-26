### Added
- **`tools/bench_reasoning_batch` probes answers and can exit early (#236).**
  - With `interval tau`, each session's answer is probed right after its
    prompt and every `interval` tokens, between batched steps: save the state,
    prefill the thinking close, read the options, restore.
  - With tau > 0, a question stops at the first probe from `interval` on
    whose top option probability reaches tau.
  - `tools/eval_decision_batched_reasoning.py` passes the new options through
    (`--probe-interval`, `--exit-tau`, `--probe`, `--question-ids`). It keeps
    every probe and scores early exits by the probe's option.
  - The batched pilot (verify, confirm on calibration, freeze rule, held-out
    run) is declared in `docs/DECISION_CAMPAIGN_PLAN.json`
    (`early_exit_batched_236`).
