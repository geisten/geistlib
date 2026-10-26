### Added
- **`tools/bench_reasoning_probe` and `tools/eval_decision_probe_pilot.py` probe
  the answer during reasoning (#236).**
  - Every N generated tokens, the session's state is saved, the
    thinking-close tokens are prefilled, the option logits are read and the
    state is restored. The generation therefore continues unchanged, which a
    transparency re-run without probes checks.
  - The runner samples a calibration split, runs in resumable chunks, and
    analyzes early-exit thresholds offline.
  - The pilot is declared in `docs/DECISION_CAMPAIGN_PLAN.json`
    (`early_exit_pilot_236`).
