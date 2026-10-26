### Changed
- **The #236 early-exit policy is frozen from calibration: tau = 0.7.**
  `docs/DECISION_CAMPAIGN_PLAN.json` (`early_exit_batched_236.frozen_policy`)
  records it before any test-set inference. The declared rule was applied to
  the 200 calibration questions not in the pilot. On those, the policy
  reaches 83.0 % against 77.5 % for full reasoning, with 0.32x the generated
  tokens.
