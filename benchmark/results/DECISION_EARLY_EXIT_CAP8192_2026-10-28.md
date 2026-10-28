# Early exit from reasoning at cap 8192 (Bonsai 2 27B, M1 Max, Metal, 2026-10-28)

Tracking: geistlib#236. Declared in `docs/DECISION_CAMPAIGN_PLAN.json` → `early_exit_batched_236`:
- the amendment of PR #270 (cap 8192, constant batch size);
- the freeze of PR #275 (τ\* = 0.8), recorded before the held-out policy run.

No new inference was run for this analysis.

**On the 500 held-out MMLU questions, early exit at τ = 0.8 matches full reasoning's accuracy with 0.23× the generated tokens. The 2 pp margin is not established.**
- **Accuracy:** 85.6 % (428/500) for both.
- **Paired comparison:** gains 27, losses 27, **0.0 pp, conservative 95 % interval [−4.74, +4.74] pp**. The lower bound is below −2 pp: 500 questions with 54 discordant answers cannot show non-inferiority at 2 pp.
- **Exits:** 330 of 500 questions exit early, two thirds of them at the first probe (token 128).
- **Questions full reasoning loses to the cap:** the policy answers 39 of the 41, and gets 25 of them right.

## Setup

- **Model and generation:** Bonsai 2 27B PQ2_0, Metal, FP32 KV, greedy, chat_reasoning prompts, cap **8192**.
- **Batching:** batch **8** in pad mode. Every `geist_session_decode_batch` step has exactly 8 sessions, so each question's tokens are independent of its batch companions (per-n invariance, #26).
- **Full-reasoning baseline at cap 8192:**
  - the #264 batched outputs, kept where the generation finished under 2048 using only 8-session steps;
  - the declared re-runs at cap 8192: the 89 capped questions, the 18 with tail-batch tokens, and 8 spot-checks.
  - All 8 spot-checks reproduced their earlier output token for token (8/8, and 8/8 on calibration).
- **Policy:**
  - probe right after the prompt and every 128 generated tokens (save state, prefill `\n</think>\n\n`, read the A–D logits, restore);
  - stop at the first probe from token 128 whose top option probability is ≥ **0.8** and answer that option;
  - otherwise reason to EOS or the cap.
  - τ = 0.8 is the declared freeze rule's result on the 200 calibration questions at cap 8192 (#275): there it gave 85.0 % vs 84.5 %, 0.21× tokens.
- **Runs:**
  - **Calibration re-run:** 54 questions, 2.55 h of lease.
  - **Test baseline re-run:** 115 questions, 5.16 h.
  - **Test policy run:** 500 questions, 32 chunks of 16, 2.55 h of lease. After 2026-10-27 it ran 21:00–07:00 only, by the user's decision.
  - The superseded cap-2048 τ = 0.7 run (420 questions) is kept apart and not used.

## Accuracy

| | accuracy | vs full reasoning at cap 8192, pp [paired 95 %] | capped |
|---|---|---|---|
| full reasoning, cap 8192 (batched) | **85.6 %** | — | 41 |
| early exit, τ = 0.8, cap 8192 | **85.6 %** | **0.0 [−4.74, +4.74]**; gains 27, losses 27 | 2 |
| #171 always-reasoning, cap 2048 (unbatched, for context) | 79.0 % | early exit vs #171: +6.6 [+1.43, +11.65] | 87 |

- **Paired comparison:** `decision_metrics.paired_quality`, #171's conservative interval, 2 pp margin.
- **Non-inferiority at 2 pp: not established.** With 54 discordant answers out of 500, the conservative interval is ±4.7 pp. Showing −2 pp at this discordance rate would take about 2,800 questions (the half-width shrinks with 1/√n).

**Where the gains and losses come from:**
- **Exits** (330 questions): the early answer equals full reasoning's in 260. Accuracy is the same, 264 correct each.
  - Of the 39 exits on questions that full reasoning runs into the 8192 cap, 25 are correct. These are most of the 27 gains.
  - The 27 losses are exits whose early answer differs from a correct full answer.
- **No exit** (170 questions): reasoning ran to EOS (168) or the cap (2). Accuracy is the same as full reasoning (164 correct each).

**The 41 questions full reasoning leaves capped at 8192:** the policy exits on 39 of them (25 correct) and runs 2 into the cap.

## Exits by probe position

| first confident probe at | 128 | 256 | 384 | 512–1024 | 1025–2048 | 2049–8192 |
|---|---|---|---|---|---|---|
| exits | 221 | 29 | 23 | 34 | 14 | 9 |

The median exit is at token 128, the first probe the policy may use.

## Tokens and time

| | generated tokens | lease time | notes |
|---|---|---|---|
| full reasoning, cap 8192 | 676,706 | no single run: #264 (6,901 s, cap 2048) + re-runs of 115 questions at 8192 (18,561 s) | a clean batched run would take about 11,600 s of decode at #264's 58.4 tok/s, plus prefill |
| early exit, τ = 0.8 | **152,657 (0.23×)** | **9,192 s** (32 chunks, incl. model loads, cool starts, pad waste) | 1,586 probes, 202 s in total (median 118 ms each) |
| #171 always-reasoning, cap 2048, unbatched | 311,800 | 23,894 s summed | |

- **The token saving is 4.4×; the time saving is much smaller.**
  - **Pad mode:** it keeps every step at 8 sessions, so a chunk lasts as long as its longest question, and the 7 finished slots decode padding meanwhile.
  - **Short chunks:** the policy run's chunks of 16 questions add a model load and a cool start each.
  - Its decode ran at 20.8 useful tok/s, against 58.4 for #264's continuous batching without padding.
- **Time with continuous batching:** without the constant-n requirement, a refilled batch would take the 152,657 tokens in about 2,600 s of decode. That is roughly 0.2× of full reasoning's time as well. Pad mode is a measurement device for bit-level invariance, not how a deployment would run.

## Conclusion

- **The measurement is now usable.** At cap 8192, early exit at τ = 0.8 gives the same accuracy as full reasoning (85.6 % each) with 0.23× the generated tokens.
- **Its gains** come mostly from questions where full reasoning runs out of budget.
- **Its losses** come from early answers that full reasoning would have corrected.
- **The 2 pp non-inferiority margin is not established** at n = 500 (interval ±4.7 pp).
- **Against #171's cap-2048 reasoning,** the policy is better by 6.6 pp, with an interval of [+1.4, +11.7].

The next decision is the user's: accept the policy at this evidence level, or test it on a larger held-out sample (the full cleaned test split has 13,904 questions).

Raw data: `raw/decision-early-exit-cap8192-2026-10-28/`. It holds the three runs' manifests, reports and driver lines per chunk, the merge/freeze/analysis scripts and their outputs. The compact numbers are in `DECISION_EARLY_EXIT_CAP8192_2026-10-28.json`.
