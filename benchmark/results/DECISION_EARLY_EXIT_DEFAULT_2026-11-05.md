# Early exit as the default of decision mode: held-out validation (Bonsai 2 27B, M1 Max, Metal, 2026-11-05)

Tracking: geistlib#286. Declared in `docs/DECISION_CAMPAIGN_PLAN.json` → `early_exit_default_286` (PR #299, d358ec3a), with the latency amendment of PR #322 (127451eb), both before the inference they govern. The 3,500 questions were analysed once, after the last one finished.

Raw bundle: `benchmark/results/raw/decision-early-exit-default-2026-11-05/`, with every number below from `analysis.py` → `DECISION_EARLY_EXIT_DEFAULT_2026-11-05.json`. It holds the three runs' manifests and reports (all probe logits), the targets, the campaign scripts and the power logs.

**The declared gate passes.**
- **Non-inferiority:** at τ 0.8 with probes every 128 tokens, early exit is non-inferior to full reasoning at cap 8192 on the 3,500 held-out questions, at the 2 pp margin.
  - Accuracy: 84.57 % vs 82.80 %. Gains 170, losses 108.
  - Conservative paired 95 % interval: **[+0.27, +3.27] pp**.
  - Generated tokens: 0.24×.
- **Live check:** live early exit gives the offline answers on **300 of 300** questions, with p95 latency 285 s vs 1,178 s.

**What the pass rests on:** the gain comes from questions full reasoning loses to the cap. On the questions full reasoning finishes, early exit is about 2.5 pp worse. Some subjects lose much more; moral_scenarios lost 12.4 pp.

## Setup

- **Sample:** 3,500 held-out questions.
  - Drawn label-independent and subject-stratified from the cleaned classic-MMLU test split (13,904), excluding the 500 of #171/#236.
  - Seed `geist-decision-test-sample-286-v1`; `question_ids_sha256` `ac034b06…`, as declared.
- **Model and generation:** Bonsai 2 27B PQ2_0 (`3907dc16…`), Metal on the M1 Max, FP32 KV, 6 threads, greedy, chat_reasoning prompts (5-shot), cap **8192**.
  - Batch 8, refilled as questions finish, no padding; per-subject prefix snapshots.
  - Source 4da9fdb2; `bench_reasoning_batch` `35a600d6…` for all three runs.
- **One run gives both arms:** full reasoning recorded the option logits at a probe every 128 generated tokens (threshold 0). Probes are transparent (#236).
  - **Early-exit arm:** the frozen rule applied to those probes. The first probe at position ≥ 128 whose top softmax probability over the four options is ≥ 0.8 answers with its argmax; without one, the full answer stands.
  - **Full arm:** a capped answer without a final letter is wrong (#171 scoring).
- **Runs:** night-only, chunks of at most 28 min under `with-mac-lock` (`LOCK_PRIO=batch`).
  - **Main:** 103 chunks, 44.2 h of lease, 2026-10-29 to 11-04.
  - **Latency check:** 300 of the 3,500, both arms run live (amendment): early exit (τ 0.8), then full reasoning.
- **Power:** all 501 samples inside the latency-phase chunk windows were on AC with Low Power Mode off, so no chunk was discarded. Some main-run nights had Low Power Mode samples; that changes speed only (amendment).

## Primary: non-inferiority at 2 pp

| | accuracy | generated tokens | capped without an answer |
|---|---|---|---|
| full reasoning, cap 8192 | 82.80 % (2,898) | 4,784,285 | 277 |
| early exit, τ 0.8 | **84.57 %** (2,960) | **1,163,236 (0.243×)** | 15 |

- **`paired_quality(early exit, full, 2 pp)`:** gains 170, losses 108, +1.77 pp, interval **[+0.27, +3.27]**. The lower bound is above −2 pp, so the result is **non-inferior: PASS**.
  - The lower bound is also above 0. Superiority was not declared, so that is a description, not a claim.
- **Exits:** 2,186 of 3,500 questions exit early, 1,320 of them at the first probe.
- **Probes:** 38,935 were recorded. 8,251 fall after the prompt and up to the exit or the end, which is what live early exit runs.

## Secondary: where gains and losses come from

| subset | n | full reasoning | early exit | gains | losses | paired 95 % |
|---|---|---|---|---|---|---|
| full reasoning hits the cap | 277 | 0 (scored wrong) | 141 | 141 | 0 | |
| full reasoning finishes (EOS) | 3,223 | 89.92 % | 87.47 % | 29 | 108 | −2.45 pp [−3.56, −1.33] |

The second row is conditioned on the full arm's outcome (post hoc), so it is no test. It is the cost a caller pays on a task whose reasoning finishes within the cap.

- **Losses:**
  - all 108 are on questions full reasoning finishes;
  - 73 exit at token 128;
  - 71 exit at a probability in [0.8, 0.9).
- **Capped questions:** early exit answers 262 of the 277 (141 correctly); 15 stay capped (8 of them professional_law).
- **Exit accuracy by position** (full reasoning's accuracy on the same questions in brackets): 128: 83.7 % (85.8 %); 256: 72.1 % (66.6 %); 384: 75.3 % (73.4 %); 512: 62.8 % (67.4 %). The 1,314 questions that never exit: 96.0 %.
- **By subject, net (gains − losses):**
  - **worst:** moral_scenarios −28 (147 → 119 of 225; 32 of its 36 losses exit at token 128), professional_psychology −6, high_school_physics −5, high_school_statistics −4;
  - **best:** professional_law +70 (208 → 278 of 386; 139 of the 277 capped questions are law), moral_disputes +10, professional_medicine +6, security_studies +5.
  - 19 subjects net negative, 13 even, 25 positive.

## Latency check (live, same 300, both arms under the power gate)

| | accuracy | tokens | capped | p50 | p95 | mean | answers / h of lease |
|---|---|---|---|---|---|---|---|
| early exit τ 0.8 | 85.0 % | 96,607 | 1 | 39.5 s | **285.5 s** | 79.0 s | **282** |
| full reasoning | 85.0 % | 445,366 | 27 | 37.9 s | 1,177.7 s | 217.3 s | 97 |
| main run, full (reference; power not verifiable) | 85.0 % | | | 42.1 s | 1,238.9 s | 228.8 s | |

- **Latency** runs from prompt to answer, prefill included; probe time is included. Without probes: early exit 39.3 / 283.9 s, full 37.7 / 1,168.1 s.
- **Tokens/s:**
  - over all chunk walls: early exit 25.2, full 40.0. The full arm lost 3 chunks to the deadline; their questions were re-run.
  - over the complete driver runs: 25.4 vs 41.8 wall, 39.5 vs 49.2 decode.
  - Early exit's rate is lower because probes and prefills take a larger share of fewer tokens. Per answer it is still 4.6× fewer tokens and 2.9× less time.
- **Agreement, live early exit vs offline:** 300/300 in answer, stop reason and exit position.
- **Live full vs the main run:** 300/300 in answer and generated tokens.
- **Live paired:** gains 14, losses 14.

## Exploratory (chosen after seeing the test data; not usable for any selection)

| τ | accuracy | gains / losses | paired 95 % | tokens vs full | exits |
|---|---|---|---|---|---|
| 0.6 | 80.51 % | 168 / 248 | [−4.09, −0.47] | 0.123× | 2,290 |
| 0.7 | 83.00 % | 171 / 164 | [−1.45, +1.85] | 0.167× | 2,247 |
| **0.8** | **84.57 %** | **170 / 108** | **[+0.27, +3.27]** | **0.243×** | 2,186 |
| 0.9 | 85.69 % | 151 / 50 | [+1.63, +4.13] | 0.361× | 2,078 |
| 0.95 | 85.74 % | 128 / 25 | [+1.88, +4.00] | 0.486× | 1,959 |
| 0.99 | 84.74 % | 73 / 5 | [+1.22, +2.66] | 0.727× | 1,549 |

Greedy trajectories are prefix-identical, so a smaller cap is a truncation of the same runs:

| cap | full reasoning | early exit τ 0.8 | paired 95 % | tokens |
|---|---|---|---|---|
| 2048 | 75.80 % | 83.11 % | [+5.68, +8.93] | 0.43× |
| 4096 (`geistr decide --max-tokens` default) | 79.63 % | 84.11 % | [+2.89, +6.07] | 0.32× |
| 8192 (declared) | 82.80 % | 84.57 % | [+0.27, +3.27] | 0.24× |

τ 0.9 is followed up by its own pre-registered validation in #467.

## Decision (owner, 2026-11-05)

- **Default for Bonsai 2 27B only:** early exit becomes the default of `geistr decide --escalate chat` for the verified Bonsai 2 27B profile only (geist-runtime#284). Every other model keeps full reasoning by default; `--early-exit` and `--no-early-exit` work for every model.
- **The cost is accepted:** about −2.5 pp on questions that full reasoning finishes, and −12.4 pp on moral_scenarios. It is documented next to `--no-early-exit`.
