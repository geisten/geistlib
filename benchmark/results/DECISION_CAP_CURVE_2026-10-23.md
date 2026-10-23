# Bonsai 2 decision: accuracy and time over the reasoning cap (M1 Max, Metal, 2026-10-23, offline)

Tracking: geistlib#210 B (follow-up of #204 and #171).

> **Exploratory.** No policy was declared for this analysis. The thresholds come from the calibration split only, as in #204. The Pareto selection below is made on the 500 test questions, so it describes the trade-off and does not choose an operating point.

**Main finding: the reasoning cap is a second lever. A moderate cap combined with routing beats either one alone, but it buys only a small saving at reasoning-level accuracy.** Above 0.80× the time of always-reasoning at cap 2,048, the best points are within about 1 pp of it. Below about 0.75×, every point loses at least 2 pp:

- **Routing 70 % at cap 1,536:** 77.8 % (−1.2 pp, interval [−3.9, +1.5]) at **0.82×** the time.
- **Routing 80 % at cap 1,536:** 78.2 % (−0.8 pp, interval [−3.0, +1.4]) at **0.88×**.
- **Always-reasoning at cap 1,536:** 77.8 % at 0.86×. It is dominated by routing 70 % at the same cap.
- **The 2 pp non-inferiority margin** is established only by points that cost *more* than the reference: routing 80 % or 90 % at cap 2,048 (1.02× and 1.08×).
- **Shortening the cap below 1,024 hurts more than routing does.** Always-reasoning at cap 512 reaches only 65.4 %, below direct scoring alone (67.8 %).

No new inference was run. The analysis reuses the #171 campaign data ([held-out writeup](DECISION_MMLU_HELDOUT_2026-10-23.md), [two-stage writeup](DECISION_TWO_STAGE_2026-10-23.md)): Bonsai 2 27B PQ2_0, Metal on the M1 Max, five-shot classic MMLU, greedy decoding, FP32 KV, sustained-load latencies.

## Method

**Outcomes at cap k.** Greedy decoding is deterministic (#171), so reasoning at cap k is the first k tokens of the cap-2,048 generation. A generation of L ≤ k tokens is unchanged. A longer one is cut to k tokens (stop = limit) and scored by the campaign's own `eval_decision.generation_outcome`, so a capped answer counts as wrong, as in #171. The cut prefix is turned back into text from the model's GGUF vocabulary. Three checks pass before any number is used:

- The detokenizer reproduces all 667 recorded generations byte for byte: the 500 cap-512 calls and the 167 cap-2,048 re-runs. EOS has no surface bytes.
- All 164 cap-2,048 re-runs start with exactly the 512 tokens of their measured cap-512 run, and the 3 determinism questions are identical. #171 had checked 3.
- **Truncating the cap-2,048 generations to 512 tokens reproduces every measured cap-512 outcome.** That covers all 500 questions: predicted letter, correctness, validity, invalid reason, truncation flag and token count, with 327 correct and 164 capped. At cap 2,048 the result is 395 correct and 87 capped, as in #171.

**Time at cap k.** For each question, the reasoning-call time is a piecewise-linear function of the number of generated tokens, through measured points:

- **Prefill** (0 tokens): the selected-row call time × the reasoning/direct prompt-token ratio. A least-squares fit of the measured cap-512 times on this estimate and the generated tokens gives 1.07 × prefill + 56.4 ms/token (R² = 0.98).
- **The measured cap-512 call:** at min(L, 512) tokens.
- **The measured cap-2,048 re-run:** at L tokens, for the 164 re-run questions.

Caps 512 and 2,048 are therefore exact sums of measured call times, and they reproduce #204's 23,894 s and its primary 0.968×. Caps 256, 768, 1,024 and 1,536 are interpolations. Decode slows as the context grows, so a straight line between two measured points slightly overstates the time in between.

Two things put bounds on that. First, a single-rate model that extrapolates each re-run's average rate back to 512 tokens overstates the measured cap-512 time by 16 %. The re-runs ran on a hotter GPU at a longer context, at about 9–11 tok/s against about 18 tok/s. That is why measured points are used where they exist. Second, as a sensitivity check, the JSON field `…_fast_decode_sensitivity` decodes the tokens past 512 at the cap-512 call's own rate. This lowers the interpolated times by up to 0.065× (always-reasoning at cap 1,536: 0.855× → 0.790×; routing 70 % at cap 1,536: 0.82× → 0.77×). Three re-runs took less time than their cap-512 call; they are kept as measured.

**Two-stage policy.** The policy is the one declared in #204. It always runs the selected-row direct call. If the calibrated top-1 probability (T = 1.0546) is at or above the threshold, it answers directly. Otherwise it runs reasoning at cap k. The thresholds route the least-confident 10, 20, …, 90 % of the 250 calibration questions, using the #204 rule; the 30 / 50 / 70 % thresholds equal #204's. The table also includes #204's primary threshold (0.7809) and both single-stage policies. Each policy is evaluated at caps 256, 512, 768, 1,024, 1,536 and 2,048, which gives 67 points.

**Comparison.** Every point is compared with always-reasoning at cap 2,048 using `paired_quality`, exactly as in #171 and #204: exact Clopper-Pearson intervals for gains and losses, Bonferroni, a conservative joint 95 % interval and a 2 pp margin. The *pessimistic* interval follows #204: the 87 questions that no tested cap completes count as correct in the reference and in any arm that reasons on them. Non-inferiority needs both lower bounds at or above −2 pp. The intervals are per point and not adjusted across the 67 points.

## Pareto front (accuracy as measured vs summed time)

A point is on the front if no other point is at least as accurate and at most as slow, with one of the two strictly better. "Routed" is the share of *test* questions sent to reasoning.

| Policy | Cap | Routed | Accuracy | Difference vs reasoning at 2,048, pp [95 %] | Pessimistic [95 %] | Time vs reasoning at 2,048 | Mean / p95 per question | Calls per question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct only | — | 0 % | 67.8 % | −11.2 [−18.2, −4.0] | [−33.8, −22.9] | 0.11× | 5.5 / 12 s | 1.00 |
| route 10 % | 512 | 11 % | 68.2 % | −10.8 [−17.6, −3.8] | [−29.8, −19.3] | 0.18× | 8.7 / 38 s | 1.11 |
| route 10 % | 768 | 11 % | 69.8 % | −9.2 [−15.9, −2.4] | [−28.1, −17.8] | 0.21× | 9.8 / 52 s | 1.11 |
| route 10 % | 1,024 | 11 % | 70.4 % | −8.6 [−15.2, −1.8] | [−27.5, −17.3] | 0.22× | 10.7 / 61 s | 1.11 |
| route 10 % | 1,536 | 11 % | 71.2 % | −7.8 [−14.4, −1.1] | [−26.6, −16.5] | 0.25× | 12.2 / 61 s | 1.11 |
| route 20 % | 1,024 | 21 % | 71.6 % | −7.4 [−13.3, −1.3] | [−22.6, −13.4] | 0.31× | 15.0 / 80 s | 1.21 |
| route 30 % | 1,024 | 26 % | 72.2 % | −6.8 [−12.6, −0.9] | [−20.2, −11.8] | 0.36× | 17.2 / 80 s | 1.26 |
| route 20 % | 1,536 | 21 % | 72.6 % | −6.4 [−12.2, −0.5] | [−21.5, −12.5] | 0.37× | 17.9 / 107 s | 1.21 |
| route 20 % | 2,048 | 21 % | 72.8 % | −6.2 [−12.0, −0.3] | [−21.3, −12.3] | 0.43× | 20.5 / 139 s | 1.21 |
| route 30 % | 1,536 | 26 % | 73.2 % | −5.8 [−11.4, −0.1] | [−19.2, −10.9] | 0.43× | 20.6 / 108 s | 1.26 |
| route 40 % | 1,024 | 41 % | 74.0 % | −5.0 [−10.0, +0.1] | [−15.0, −7.5] | 0.48× | 23.0 / 86 s | 1.41 |
| route 40 % | 1,536 | 41 % | 75.4 % | −3.6 [−8.4, +1.3] | [−13.4, −6.3] | 0.59× | 28.0 / 113 s | 1.41 |
| route 50 % | 1,536 | 49 % | 75.6 % | −3.4 [−7.7, +0.9] | [−11.6, −5.0] | 0.65× | 31.2 / 115 s | 1.49 |
| route 40 % | 2,048 | 41 % | 76.2 % | −2.8 [−7.5, +1.9] | [−12.5, −5.6] | 0.68× | 32.3 / 143 s | 1.41 |
| route 50 % | 2,048 | 49 % | 76.4 % | −2.6 [−6.7, +1.6] | [−10.7, −4.3] | 0.75× | 36.0 / 144 s | 1.49 |
| route 60 % | 1,536 | 60 % | 76.8 % | −2.2 [−5.6, +1.2] | [−7.6, −2.2] | 0.76× | 36.1 / 119 s | 1.60 |
| **route 70 %** | **1,536** | 67 % | **77.8 %** | **−1.2 [−3.9, +1.5]** | [−4.7, −0.4] | **0.82×** | 39.0 / 120 s | 1.67 |
| **route 80 %** | **1,536** | 76 % | **78.2 %** | **−0.8 [−3.0, +1.4]** | [−3.4, +0.3] | **0.88×** | 42.1 / 128 s | 1.76 |
| route 70 % | 2,048 | 67 % | 79.0 % | 0.0 [−2.3, +2.3] | [−3.1, +0.4] | 0.95× | 45.2 / 152 s | 1.67 |
| route 80 % | 2,048 | 76 % | 79.4 % | +0.4 [−1.2, +1.9] | [−1.6, +0.8] | 1.02× | 48.6 / 158 s | 1.76 |

Always-reasoning is not on the front at any cap: routing 70–80 % at the same cap matches its accuracy for less time. Only route 80 % at cap 2,048 establishes non-inferiority (route 90 % at cap 2,048 also does, at 1.08×, but it is not on the front).

## Always-reasoning by cap

| Cap | Accuracy | Capped | Difference vs cap 2,048, pp [95 %] | Time vs cap 2,048 | Mean / p50 / p95 per question | Generated tokens |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | 53.2 % | 231 | −25.8 [−30.5, −20.6] | 0.34× | 16.4 / 17.2 / 29 s | 93,409 |
| 512 | 65.4 % | 164 | −13.6 [−17.4, −9.5] | 0.46× | 22.1 / 18.4 / 44 s | 143,058 |
| 768 | 71.0 % | 134 | −8.0 [−11.1, −4.6] | 0.59× | 28.3 / 18.4 / 67 s | 181,139 |
| 1,024 | 74.8 % | 112 | −4.2 [−6.7, −1.6] | 0.69× | 32.9 / 18.4 / 84 s | 212,318 |
| 1,536 | 77.8 % | 94 | −1.2 [−2.8, +0.5] | 0.86× | 40.9 / 18.4 / 117 s | 265,577 |
| 2,048 | 79.0 % | 87 | reference | 1.00× | 47.8 / 18.4 / 146 s | 311,800 |

The p50 does not move because half the questions finish within about 225 tokens. The cap only shortens the long tail, and each step down loses questions that reasoning would have answered correctly.

## Full grid (accuracy · time vs always-reasoning at cap 2,048)

| Policy (routed on test) | cap 256 | cap 512 | cap 768 | cap 1,024 | cap 1,536 | cap 2,048 |
| --- | --- | --- | --- | --- | --- | --- |
| route 10 % (11 %) | 66.8 % · 0.16× | 68.2 % · 0.18× | 69.8 % · 0.21× | 70.4 % · 0.22× | 71.2 % · 0.25× | 71.2 % · 0.28× |
| route 20 % (21 %) | 66.2 % · 0.20× | 68.6 % · 0.24× | 70.8 % · 0.28× | 71.6 % · 0.31× | 72.6 % · 0.37× | 72.8 % · 0.43× |
| route 30 % (26 %) | 65.0 % · 0.22× | 68.6 % · 0.27× | 70.8 % · 0.32× | 72.2 % · 0.36× | 73.2 % · 0.43× | 73.6 % · 0.50× |
| route 40 % (41 %) | 63.2 % · 0.28× | 69.2 % · 0.35× | 72.2 % · 0.42× | 74.0 % · 0.48× | 75.4 % · 0.59× | 76.2 % · 0.68× |
| route 50 % (49 %) | 61.6 % · 0.31× | 68.4 % · 0.39× | 71.4 % · 0.47× | 73.6 % · 0.54× | 75.6 % · 0.65× | 76.4 % · 0.75× |
| route 60 % (60 %) | 60.2 % · 0.34× | 67.8 % · 0.44× | 71.6 % · 0.54× | 74.4 % · 0.62× | 76.8 % · 0.76× | 77.6 % · 0.87× |
| route 70 % (67 %) | 59.6 % · 0.37× | 68.4 % · 0.47× | 72.4 % · 0.58× | 75.4 % · 0.67× | 77.8 % · 0.82× | 79.0 % · 0.95× |
| #204 primary (69 %) | 58.8 % · 0.37× | 67.8 % · 0.48× | 72.0 % · 0.59× | 75.0 % · 0.68× | 77.6 % · 0.84× | 78.8 % · 0.97× |
| route 80 % (76 %) | 56.6 % · 0.40× | 67.0 % · 0.51× | 71.8 % · 0.63× | 75.2 % · 0.72× | 78.2 % · 0.88× | 79.4 % · 1.02× |
| route 90 % (87 %) | 54.2 % · 0.43× | 65.6 % · 0.55× | 71.0 % · 0.67× | 74.8 % · 0.77× | 77.8 % · 0.93× | 79.0 % · 1.08× |
| reasoning only (100 %) | 53.2 % · 0.34× | 65.4 % · 0.46× | 71.0 % · 0.59× | 74.8 % · 0.69× | 77.8 % · 0.86× | 79.0 % · 1.00× |

At caps of 512 and below, routing *more* questions lowers accuracy. The routed questions are the hard ones, and short-capped reasoning on them does worse than the direct answer: direct scoring gets 51 % of the questions that need more than 512 reasoning tokens, and capped reasoning gets 0 %.

## What this means

- **For an application choosing a policy** (#210 A): two settings matter, the routed share and the cap. On this task and model, the useful region is a cap of 1,536–2,048 with 40–80 % routed. Its range runs from 0.59× time at −3.6 pp (interval down to −8.4 pp) to 0.88× at −0.8 pp (interval down to −3.0 pp). Caps below 1,024 are never worth it with reasoning on more than about a third of the questions.
- **It does not change #204's conclusion for #26.** A saving of 12–18 % at about −1 pp is the best this policy family offers here, and the 2 pp margin is not established there. A large saving still needs a direct-cost scorer that closes the 11 pp gap (#588).
- **The curve is task- and model-specific.** It depends on how long reasoning runs on easy and hard items, and on how well the direct probabilities separate them. On JevBench, direct scoring alone reaches 85.3 % (#171), so its curve would sit elsewhere. Every application has to measure its own curve on its own calibration data before picking a point.

**Not decision-grade:** the Pareto selection uses the test set, and the times at caps 768–1,536 are interpolated (sensitivity above). A chosen (share, cap) point would need its own declared threshold and a held-out check.

## Raw evidence

- [Compact JSON](DECISION_CAP_CURVE_2026-10-23.json): thresholds, checks, all 67 points with both intervals, time ratios and the fast-decode sensitivity, and one row per question with confidence, direct correctness, correctness at every cap and the modelled reasoning time at every cap.
- [`analyze_210.py`](raw/decision-cap-curve-2026-10-23/analyze_210.py) and [`ggufvocab.py`](raw/decision-cap-curve-2026-10-23/ggufvocab.py) produce the JSON. Run `python3 analyze_210.py docs/DECISION_CAMPAIGN_PLAN.json <Ternary-Bonsai-2-27B-PQ2_0.gguf>`. The script reads the #171 campaign directory with the pinned tools checkout (`src` at `d159d350`) and asserts every check above.
