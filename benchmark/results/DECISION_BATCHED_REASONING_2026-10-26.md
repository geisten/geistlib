# Batched reasoning on Bonsai 2 27B: accuracy against #171 (M1 Max, Metal, 2026-10-26)

Tracking: geistlib#26. Declared in `docs/DECISION_CAMPAIGN_PLAN.json` → `batched_reasoning_26` (PR #261) before any inference.

**Batched decode keeps #171's reasoning accuracy.**
- **Setup:** the 500 held-out MMLU questions, reasoning at cap 2048, decoded 8 at a time through `geist_session_decode_batch` (PRs #259/#260).
- **Accuracy:** **78.8 %** (394/500), against **79.0 %** (395/500) for #171's unbatched run at cap 2048.
- **Paired difference:** gains 0, losses 1, **−0.2 pp, conservative 95 % interval [−1.27, +0.87] pp**. The run is non-inferior at the 2 pp margin.
- **Identical outputs:** 471 of 500 generations are token-identical to #171's.
- **The one changed answer comes from batching.** An unbatched re-run on the same binary reproduces #171 token for token.
- **Throughput:** aggregate decode ran at **58.4 tokens/s**, against about 15 tokens/s single-sequence in #171. The whole run took 6,901 s of lease time, against 23,894 s summed over #171's questions.

## Setup (frozen before inference)

- **Questions:** the #171 test bundle (`cases-test/test.jsonl` sha256 108f815e…), chat_reasoning prompts, in subject order.
- **Model and backend:** Bonsai 2 27B PQ2_0 (sha256 3907dc16…), Metal, FP32 KV, 6 threads, `OMP_WAIT_POLICY=passive`.
- **Generation:** greedy, cap 2048, stop at EOS, as #171's `generate_long`.
- **Batching:** `tools/bench_reasoning_batch --batch 8` at main cd62fc60.
  - Continuous batching: a finished question's slot takes the next question at once.
  - **94 %** of the generated tokens came from 8-session passes; the rest came from the tails as chunks drained.
- **Prefill:** a per-subject prefix snapshot, cut at the common prefix minus one token, aligned down to the engine's 320-token chunk, and restored per question. This is bit-identical to a single prefill (#26 steps 1–2).
- **Runner:** `tools/eval_decision_batched_reasoning.py`.
  - 6 chunks under 25-min wall-clock deadlines, `LOCK_MAX=30`, cool start (GPU < 65 °C), frozen manifest.
  - Two chunks reached their deadline; their in-flight questions were re-run from the start, so no partial generation is used.
- **Scoring:** `eval_decision.generation_outcome` with thinking on, exactly as #171. Capped or unparseable answers count as wrong.

## Result

| | batched (this run) | #171 always-reasoning, cap 2048 |
|---|---|---|
| accuracy | **78.8 %** (394) | **79.0 %** (395) |
| capped at 2048 | 89 | 87 |
| generated tokens | 312,466 | 311,800 |
| per-question latency p50 / p95 | 42.2 / 327 s (8 in flight) | 18.4 / 146 s (one at a time) |
| aggregate decode tokens/s | **58.4** (complete runs) | ≈ 15 |

**Paired comparison**, batched minus #171, using `decision_metrics.paired_quality`, which is #171's conservative paired interval:
- gains 0, losses 1;
- **−0.2 pp, [−1.27, +0.87] pp**;
- the lower bound clears −2 pp, so the run is **non-inferior**.

**Token agreement with #171's generations:**
- 471/500 are identical.
- 29 diverge. The first differing token is at position 879 at the median, 18 at the earliest; the rounding of the batched rows takes a while to change a greedy choice.
- Of the 29 divergent questions, 28 still reach the same correctness.

**Attribution (declared: unbatched same-binary re-run of every question whose correctness changed, at most 30).**
- One question changed: `high_school_microeconomics/4701`.
  - #171 and the unbatched re-run produce the same 1,800 tokens and are correct.
  - The batched run diverges at token 1201 and hits the cap.
- The unbatched re-run reproduces #171 exactly, so the engine has not drifted since #171 for this question. The change comes from the batched rows' rounding.

## Time

Per question, latency is higher, because 8 questions share each pass. Throughput is what improves:
- **Decode:** 58.4 tokens/s aggregate in the complete driver runs, against about 15 tokens/s single-sequence (#171: 311,800 tokens in about 20,800 s of decode). That is **≈ 3.9×**, and it includes the partly filled passes at the end of each chunk.
- **End to end:** 6,901 s of lease time for all 500 questions (**0.29×** of #171's 23,894 s).
  - The lease time includes six model loads, cool gates and the re-runs after two deadline stops.
  - The prefill side also uses the per-subject prefix (step 1: 0.40×).

## What this means

Batched decode keeps the always-reasoning accuracy of #171 within the declared margin while cutting its cost to about 0.3×. That is the reasoning cost #204 and #210 assumed fixed: their curves were all measured against an unbatched baseline of 1.00×. The two-stage and cap trade-offs still apply on top. They now scale a much smaller base.

Not measured here:
- **Rollout batches 3–5:** single-dispatch attention and DeltaNet for B sequences, other families and the CPU path, and continuous batching in geist-runtime. They wait for the user's decision.
- **Other tasks or models.**

Raw data: `raw/decision-batched-reasoning-2026-10-26/`. It holds the manifest, `report.json` with every question's outcome, per-chunk driver lines and temperatures, the attribution re-run, and `analyze_batched.py`, which produced the comparison. The compact numbers are in `DECISION_BATCHED_REASONING_2026-10-26.json`.
