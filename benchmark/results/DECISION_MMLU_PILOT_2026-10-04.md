# Bonsai 2 classic-MMLU development pilot — Apple M1 Max, 2026-10-04

The optional decision path works on the existing ternary Bonsai checkpoint: selected rows reproduce DENSE logits exactly on every paired trial, and all within-backend deterministic-repeat checks pass. This pilot **does not establish a quality-matched speedup or Jev equivalence**. Eight development questions cannot establish the declared two-percentage-point margin, and the 512-token reasoning baseline contains incomplete answers.

## Frozen scope and provenance

- Checkpoint: `Ternary-Bonsai-2-27B-PQ2_0.gguf`, 7,206,168,928 bytes, SHA-256 `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`.
- Classic `cais/mmlu` revision `c30699e8356da336a370243923dbaf21066bb9fe`, five fixed subject-specific dev examples, explicit exact-duplicate/earlier-split exclusion. This is a **cleaned development subset**, not a stock-MMLU test score or MMLU-Redux result.
- One question from each of eight subjects chosen deterministically, independently of labels. No option rotations. Quality N=8; timing repeats do not increase N.
- Apple M1 Max, 64 GiB; serial Metal first, then CPU NEON; six threads, concurrency 1, `OMP_WAIT_POLICY=passive`, FP32 KV, greedy decoding.
- Actual GGUF chat template, frozen date 2026-10-04; chat reasoning xhigh, direct chat thinking disabled. Standard MMLU cloze uses spaced answer letters. All 96 complete-prefix/candidate boundaries passed native tokenization checks.
- A 512-token cap; `--warmup 0 --repeats 1`: one post-setup first call and one warm call per arm/question. Eight arms across three profiles, 128 model calls per backend, 256 total.
- Clean source revision `13261ac73fb2841ef565bcabda5473eacd1e8ce0` for both runs. Driver SHA-256 `63f56ff7def18aa503474cf983c9b54b5ce11b9dcad94f5ba8352d70092f419a`; tokenizer binary SHA-256 `e39ecc4d6374048656de0fbb682fa86f5d1fa825f534edbc2cfbd00c04ec6d39`.
- Manifest and exact commands were frozen before labelled inference. Model, driver, tokenizer and evaluator source hashes were checked again after both runs.

## Exploratory quality and descriptive timing

Latencies below are warm inference calls, including reset, prefill and score/generation. p50/p95 are across eight questions with one warm sample each. The reasoning/selected ratio uses the sum of corresponding question medians; it is **not** a ratio of the table p50 values. It compares these capped policies only and is ineligible as a quality-matched speedup.

| Backend | Direct chat selected | MMLU cloze selected | Reasoning | Invalid/capped reasoning | Reasoning/selected time ratio |
| --- | --- | --- | --- | --- | --- |
| metal | 4/8 | 6/8 | 4/8 | 4/8 invalid; 4/8 capped | 2.168 (descriptive) |
| cpu_neon | 4/8 | 6/8 | 4/8 | 4/8 invalid; 4/8 capped | 2.328 (descriptive) |

| Backend | Profile / arm | Correct /8 | Invalid /8 | Capped /8 | Warm p50 s | Warm p95 s | Mean s | Sequential QPS estimate | Peak process RSS GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metal | chat_direct / decision_dense | 4 | — | — | 16.256 | 22.953 | 17.295 | 0.0578 | 2.021 |
| metal | chat_direct / decision_selected | 4 | — | — | 17.197 | 24.929 | 18.666 | 0.0536 | 2.021 |
| metal | chat_direct / scorealt_dense | 4 | — | — | 17.839 | 23.071 | 17.866 | 0.0560 | 2.021 |
| metal | chat_direct / generate_1 | 2 | 5 | 8 | 17.890 | 23.880 | 18.771 | 0.0533 | 2.021 |
| metal | cloze / decision_dense | 6 | — | — | 12.758 | 16.442 | 12.818 | 0.0780 | 2.029 |
| metal | cloze / decision_selected | 6 | — | — | 12.614 | 14.717 | 11.966 | 0.0836 | 2.029 |
| metal | cloze / scorealt_dense | 6 | — | — | 12.699 | 14.707 | 11.868 | 0.0843 | 2.029 |
| metal | chat_reasoning / generate_long | 4 | 4 | 4 | 44.147 | 59.638 | 40.466 | 0.0247 | 1.378 |
| cpu_neon | chat_direct / decision_dense | 4 | — | — | 40.247 | 46.131 | 40.172 | 0.0249 | 13.957 |
| cpu_neon | chat_direct / decision_selected | 4 | — | — | 41.319 | 48.122 | 40.478 | 0.0247 | 13.957 |
| cpu_neon | chat_direct / scorealt_dense | 4 | — | — | 42.130 | 46.619 | 40.435 | 0.0247 | 13.957 |
| cpu_neon | chat_direct / generate_1 | 2 | 5 | 8 | 41.624 | 47.353 | 40.300 | 0.0248 | 13.957 |
| cpu_neon | cloze / decision_dense | 6 | — | — | 38.881 | 44.254 | 37.617 | 0.0266 | 14.652 |
| cpu_neon | cloze / decision_selected | 6 | — | — | 38.766 | 45.517 | 38.243 | 0.0261 | 14.652 |
| cpu_neon | cloze / scorealt_dense | 6 | — | — | 39.442 | 43.439 | 38.770 | 0.0258 | 14.652 |
| cpu_neon | chat_reasoning / generate_long | 4 | 4 | 4 | 98.937 | 127.428 | 94.215 | 0.0106 | 14.111 |

Numeric score probabilities are **uncalibrated and conditional on the four supplied alternatives**. No temperature was fitted. Log-loss, Brier sum over classes, ECE and reliability bins are retained in each raw `report.json`; their eight-question values are exploratory.

The unrestricted single-generated-token arm is reported separately: a one-token budget does not guarantee a valid answer letter. Invalid surfaces count as wrong; no letter is guessed from reasoning text. For reasoning, an answer requires exactly one `</think>` marker followed by a valid final letter (optionally `Answer:`).

## Numerical checks and quality gate

- metal: 32 exact selected/DENSE logit pairs across both numeric profiles and both trials; independent dense reference parity and within-backend numeric/generation repeat checks passed. Paired chat-direct minus reasoning accuracy: 0.0pp; conservative joint 95% interval [-57.39, 57.39]pp, target lower bound −2pp not established.
- cpu_neon: 32 exact selected/DENSE logit pairs across both numeric profiles and both trials; independent dense reference parity and within-backend numeric/generation repeat checks passed. Paired chat-direct minus reasoning accuracy: 0.0pp; conservative joint 95% interval [-57.39, 57.39]pp, target lower bound −2pp not established.

Cross-backend prediction agreement (not bitwise or external-runtime parity): chat_direct 8/8; cloze 8/8; generate_1 8/8; generate_long 8/8.

| Backend | Chat selected projected rows / readback bytes | Cloze selected projected rows / readback bytes | DENSE rows / readback bytes |
| --- | --- | --- | --- |
| metal | [16] / [64] | [64] / [256] | [248320] / [993280] |
| cpu_neon | [8] / [32] | [32] / [128] | [248320] / [993280] |

Selected rows reduce output projection/readback work; the unchanged 27B backbone still processes every prompt token. This pilot does not establish a reliable whole-query acceleration from selected rows alone. Prefill, arm order, thermal/host variation and the limited warm sample count all affect whole-query timings.

## Campaign costs and measurement limits

Preparation took 22.029s, including model/tokenizer startup, checksum work and 120 TOK calls (boundary validation included).

| Backend | Run start UTC | Full campaign minutes | Direct profile minutes | Cloze profile minutes | Reasoning profile minutes |
| --- | --- | --- | --- | --- | --- |
| metal | 2026-10-04T14:51:32.895061+00:00 | 40.265 | 19.194 | 10.001 | 10.982 |
| cpu_neon | 2026-10-04T15:32:27.041231+00:00 | 99.224 | 43.237 | 30.647 | 25.274 |

Full campaign time includes setup/validation and all reference arms/repeats. These are measured costs; the earlier 18–26min Metal / 88–124min CPU estimates in the frozen plan were planning spans based on older rates and an all-cap workload. In particular, the Metal planning span underestimated this campaign.

No own builds or other owned model benchmarks overlapped these runs. The shared desktop was not thermally isolated or continuously audited for external load. There was no cache eviction; first calls are post-setup, not cold cache. Peak process RSS includes retained handles and does not capture all Metal resource memory. Sequential QPS is a reciprocal-latency estimate. Additive preprocessing/surface/parse estimates in the raw reports are not measured service end-to-end latency. There is no abstention/fallback route.

The measurements apply to the frozen source revision above. The remote branch
subsequently integrated main changes to session KV sizing, device-copied weight
residency and session snapshots. This is a record of the measured binary, not
a performance measurement of that later merged tree. A new acceptance campaign
must bind and measure the then-current binary again.

## Raw evidence and next evaluation boundary

- [Compact machine-readable result](DECISION_MMLU_PILOT_2026-10-04.json), including canonical per-question predictions and generated-token/stop summaries.
- [Raw bundle and SHA-256 index](raw/decision-mmlu-pilot-2026-10-04/README.md): exact prepared prompts/IDs, excluded identities, manifests, per-call driver JSONL, component timings and full reports; source files remain unchanged during the measured campaign.
- [Frozen approved plan](../../docs/DECISION_CAMPAIGN_PLAN.json) and [evaluation protocol](../../docs/DECISION_EVALUATION.md).

The next boundary is pinned native Prism same-token parity (including longer/high-entropy prompts and full recurrent-state reset), followed by a development-only generation-budget study that obtains complete reasoning answers. Any revised cap/policy must be frozen before calibration and held-out evaluation. Broader negative/ambiguous cases, label rotations, calibration and a full cleaned held-out run remain outstanding in #587. These eight questions do not justify choosing a trained head or backbone adaptation; #588 remains gated.

No production rollout follows automatically. The implementation remains optional and default-off behind `DECISION=1`.
