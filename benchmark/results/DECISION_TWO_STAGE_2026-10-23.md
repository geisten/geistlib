# Bonsai 2 two-stage decision: direct scoring first, reasoning when unsure (M1 Max, Metal, 2026-10-23)

Tracking: geistlib#204 (follow-up of #171, prerequisite for #26).

**The two-stage policy matches reasoning's accuracy, but it saves almost nothing.** The policy was declared in advance, and its threshold was fitted on calibration data only. It answered 31 % of the 500 held-out questions from direct scoring and sent 69 % to reasoning. That gave **78.8 %** accuracy against **79.0 %** for always-reasoning at cap 2,048. The paired difference is −0.2 pp, with a conservative 95 % interval of [−2.36, +1.97] pp, so the 2 pp margin is narrowly *not* established. The policy used **0.97×** the time of always-reasoning and 1.32× its tokens, with 1.69 model calls per question.

The confident questions are the ones reasoning already answers quickly. Reasoning takes 22 s and 265 generated tokens on average for the questions the policy answers directly, against 60 s and 788 tokens for the ones it routes. Skipping reasoning on the easy third therefore removes only 15 % of reasoning time, and the extra direct call on every question gives most of that back.

No new inference was run. Everything reuses the #171 campaign data ([held-out writeup](DECISION_MMLU_HELDOUT_2026-10-23.md)).

## Declaration (before any test-set evaluation)

The policy, the threshold rule and the thresholds themselves were committed in `docs/DECISION_CAMPAIGN_PLAN.json` → `two_stage_policy_204` (PR #206) before the test questions were evaluated. `thresholds_204.py` reads only the calibration split.

- **Policy:** always run chat_direct `decision_selected`. Let c be the top option probability after temperature scaling (T = 1.0546, the #171 calibration fit). If c ≥ threshold, answer with the direct argmax. Otherwise run chat_reasoning `generate_long` at cap 2,048; a capped or unparseable answer counts as wrong, as in #171.
- **Primary threshold:** target a direct-answer precision of 0.95 on the 250 calibration questions. The threshold is the smallest calibration confidence t such that every accepted set {c ≥ t′}, for t′ ≥ t with at least 25 questions, has calibration precision ≥ 0.95. Result: **t = 0.7809**. On calibration, 68 questions (27.2 %) are answered directly, with 95.6 % precision.
- **Secondary (not the decision):** route the least-confident 30 / 50 / 70 % of calibration questions. Thresholds: 0.532 / 0.651 / 0.765.
- **Reference:** always-reasoning at cap 2,048 on the same 500 questions. That is the cap-512 output where it finished and the amendment-3 re-run otherwise, which are equivalent by #171's determinism check.
- **Comparison:** `paired_quality` exactly as in #171 (exact Clopper-Pearson for gains and losses, Bonferroni, conservative joint 95 % interval) with a 2 pp margin.
- **Capped answers:** 87 answers still hit cap 2,048. In the *as measured* figures they count as wrong in both arms. The *pessimistic* figures count all of them as correct in both arms; this can only help the reference where the policy answered directly. Non-inferiority is called only if both lower bounds clear −2 pp.
- **Known at declaration:** only the #171 aggregates published in #171 and PRs #193/#203. Test-set confidence against correctness had not been computed.

## Result: primary policy

| | Two-stage (primary) | Always-reasoning, cap 2,048 | Always-direct |
| --- | --- | --- | --- |
| Accuracy | **78.8 %** (394/500) | **79.0 %** (395) | 67.8 % (339) |
| Routed to reasoning | 343 (68.6 %) | 500 | 0 |
| Answered directly | 157, precision 94.9 % (reasoning on the same: 95.5 %) | — | 500 |
| Model calls | 843 (1.69 per question) | 500 | 500 |
| Prompt tokens | 661,793 | 394,065 | 374,065 |
| Generated tokens | 270,248 | 311,800 | 0 |
| Time, sum | 23,138 s (**0.968×**) | 23,894 s | 2,725 s (0.114×) |
| Time per question, mean / p50 / p95 | 46.3 / 19.2 / 153.4 s | 47.8 / 18.4 / 146.4 s | 5.5 / 4.4 / 12.3 s |

**Paired difference, policy minus reasoning:**

- **As measured:** gains 4, losses 5, **−0.2 pp, [−2.36, +1.97] pp**.
- **Pessimistic:** gains 0, losses 7, −1.4 pp, [−3.10, +0.38] pp.
- **Non-inferior at 2 pp: no.** Both lower bounds fall just below −2 pp.

Routed questions agree with the reference by construction, so only 9 of 500 questions are discordant. The conservative Clopper-Pearson/Bonferroni interval stays about ±2 pp wide even at this count; at 0 gains and 0 losses it would still be ±0.9 pp.

**The 87 capped answers:** the policy routes 81 of them to reasoning, where they count as wrong in both arms. It answers 6 directly. All 4 gains are capped questions that direct scoring got right, which is why the pessimistic difference is lower.

Time is the measured sum of per-call times: the selected-row call for every question, plus the reasoning call (cap 2,048) when routed. These are sustained-load Metal figures, as in #171. There is no prefix or KV reuse between the two calls. The direct and reasoning prompts differ from the second token onward, so reuse would need a redesigned prompt. Even if the direct call were free on routed questions, the policy would take 0.885× the time.

## Secondary policies (declared, not the decision)

| Threshold rule | Threshold | Routed (test) | Accuracy | Difference, as measured | Pessimistic interval | Time vs reasoning |
| --- | --- | --- | --- | --- | --- | --- |
| Route least-confident 30 % of calibration | 0.532 | 132 (26 %) | 73.6 % | −5.4 pp [−11.0, +0.3] | [−18.7, −10.5] | 0.50× |
| Route least-confident 50 % of calibration | 0.651 | 246 (49 %) | 76.4 % | −2.6 pp [−6.7, +1.6] | [−10.7, −4.3] | 0.75× |
| Route least-confident 70 % of calibration | 0.765 | 337 (67 %) | 79.0 % | 0.0 pp [−2.3, +2.3] | [−3.1, +0.4] | 0.95× |

None establishes non-inferiority. Halving the time costs 5.4 pp of point accuracy, and under the pessimistic handling at least 10.5 pp.

## Exploratory: accuracy/cost curve on the test set (descriptive, not a decision)

> **Exploratory.** The thresholds here are quantiles of the *test* confidences, which is exactly what the declaration forbids for choosing a policy. The curve only shows the shape of the trade-off.

| Routed (least confident) | Accuracy | Difference, as measured [95 %] | Pessimistic [95 %] | Mean time | Time vs reasoning |
| --- | --- | --- | --- | --- | --- |
| 0 % (direct only) | 67.8 % | −11.2 [−18.2, −4.0] | [−33.8, −22.9] | 5.5 s | 0.11× |
| 10 % | 70.8 % | −8.2 [−14.8, −1.5] | [−27.1, −16.9] | 13.3 s | 0.28× |
| 20 % | 72.2 % | −6.8 [−12.7, −0.8] | [−22.0, −12.8] | 20.2 s | 0.42× |
| 30 % | 75.0 % | −4.0 [−9.3, +1.4] | [−16.1, −8.4] | 26.5 s | 0.55× |
| 40 % | 76.6 % | −2.4 [−7.2, +2.4] | [−12.7, −5.8] | 31.5 s | 0.66× |
| 50 % | 76.6 % | −2.4 [−6.5, +1.7] | [−10.5, −4.1] | 36.5 s | 0.76× |
| 60 % | 77.8 % | −1.2 [−4.3, +2.0] | [−6.4, −1.4] | 41.7 s | 0.87× |
| 65 % | 78.6 % | −0.4 [−2.8, +2.0] | [−3.6, +0.1] | 44.8 s | 0.94× |
| 70 % | 79.2 % | +0.2 [−1.8, +2.1] | [−2.5, +0.6] | 46.7 s | 0.98× |
| 75 % | 79.4 % | +0.4 [−1.2, +1.9] | [−1.6, +0.8] | 48.4 s | 1.01× |
| 80 % | 79.2 % | +0.2 [−0.9, +1.3] | [−1.3, +0.9] | 50.4 s | 1.05× |
| 100 % (direct + reasoning always) | 79.0 % | 0.0 [−0.9, +0.9] | [−0.9, +0.9] | 53.2 s | 1.11× |

The full 5 % grid is in the JSON. Even with the threshold picked after the fact, the curve establishes non-inferiority only from about 75 % routed onward. By then the policy costs *more* than always-reasoning, because of the extra direct call. Every point that saves at least 25 % of the time is about 2.4 pp or more below reasoning, with intervals reaching −6.5 pp or lower.

## Is this decision-grade?

**Yes, for the question #204 asks.** The primary threshold was fixed from calibration data and committed before the test evaluation, so its test estimate is unbiased. Its outcome is clear on cost: accuracy at reasoning level (−0.2 pp; the margin is narrowly not established) at **97 % of reasoning's time**, not "a large fraction of cost saved".

**Reasoning outputs on the calibration split are not needed.** They would only let a threshold be fitted against reasoning accuracy, which moves the operating point along the curve above. That curve's cost ceiling is structural: the confident questions are the cheap ones for reasoning too. A better-placed threshold cannot buy a large saving at reasoning-level accuracy on this data. For the record, running chat_reasoning at cap 2,048 on the 250 calibration questions would take about 3.3 h of Mac time at the measured test mean of 47.8 s per question (about 4 h with model loads and cool gates, in about 10 chunks). At 150 s per question, the rate for questions that hit the cap, it would take at most 10.4 h. I did not start it and do not recommend it.

## What this means for #26

**Do not build #26 on the two-stage policy.** At reasoning-level accuracy, the policy's cost is 88 % reasoning on the 69 % of questions direct scoring is unsure about. #26 (shared-state multi-question scoring) speeds up only the direct stage, which is about 12 % of the policy's total time here. Even a free direct stage would leave the policy at about 0.89× of always-reasoning.

#26 pays off only where direct-only accuracy is acceptable (67.8 % here against 79.0 %), or if a scorer at direct cost closes most of the 11 pp gap. Of the paths named in #204, only a trained head (old GitHub #588) could do that, and nothing measured so far shows it can: direct scoring reaches 51 % on the questions where reasoning needs more than 512 tokens. The recommendation is **neither as a foundation for #26 now**. Gate #26 on a #588 result that recovers most of the gap at direct cost, or on a consumer that accepts direct-only accuracy.

**Scope:** five-shot classic MMLU (knowledge questions), Bonsai 2 27B PQ2_0, Metal on the M1 Max, greedy decoding, and sustained-load latency. On tasks where reasoning is long even for easy items, the cost picture could differ.

## Raw evidence

- [Compact JSON](DECISION_TWO_STAGE_2026-10-23.json): calibration thresholds, primary and secondary results, the full exploratory grid, and one row per question (calibrated confidence, direct and reasoning correctness, capped flag, per-call times and token counts, primary routing).
- [`thresholds_204.py`](raw/decision-two-stage-2026-10-23/thresholds_204.py): calibration-only threshold fit.
- [`analyze_204.py`](raw/decision-two-stage-2026-10-23/analyze_204.py): test evaluation. Both scripts read the #171 campaign directory with the pinned tools checkout (`src` at `d159d350`), and re-running them reproduces their parts of the JSON exactly (checked).
