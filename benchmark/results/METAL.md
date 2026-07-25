# Metal (Apple GPU) — M1 Max vs llama.cpp Metal

**Goal:** greedy-decode parity with llama.cpp's Metal backend on the same model
and the same machine, without giving up bit-exactness against the `cpu_scalar`
reference.

**Status: closed 2026-07-04, re-confirmed 2026-07-05.** Decode lands within
**12 %** of llama.cpp; prefill remains **1.56×** behind. The backend is
`EXPERIMENTAL` — build it with `BACKENDS="… metal"`.

---

## Final numbers

Cool-state `compare_metal.sh`, gemma4-e2b-Q4_K_M, M1 Max, both engines in the
same run:

| | geist Metal | llama.cpp Metal | ratio |
| :-- | --: | --: | --: |
| Prefill `pp512` (t/s) | 1006 | 1540 | 1.53× behind |
| Decode `tg64` (t/s) | 81.6 | 92.8 | 1.14× behind |
| Total (t/s) | 445 | — | |
| Decode `tg64` @ kv 2100 (t/s) | 73 | — | holds past the 4096 default |

The 2026-07-05 re-run (`1020d45`) came after the #58/#60 Metal work and the
#62/#63 dead-flag cleanup and reads within noise of the 2026-07-04 close (987 /
81.2 / 441 vs llama 1542 / 91.3) — the flag, kernel and replay removals cost no
performance.

Greedy decode is **bit-exact against `cpu_scalar`**; that gate ran on every step
of the program, and steps that broke it were reverted regardless of their speed.

## What produced the gain

Two invisible scalar-kernel fallbacks, found by scaling-curve triage plus
`sample` — not by reading code:

- **sg8 prefill flash** — the prefill flash-attention path fell back to scalar.
- **dec512 split-KV** — head_dim-512 full-attention layers did the same on decode.

Three plausible theories were falsified by measurement first: the clock theory
(`powermetrics` showed both engines pinned at 1296 MHz), `m_max 512` (a
regression when tried), and MQA K/V redundancy (23 ms of 671 ms attention — the
flash kernel is compute-bound at ~0.7 TF effective, so removing redundancy could
not have paid).

## What is left, and why it stopped here

Both remaining gaps are kernel-efficiency questions, and both are *measured*
rather than assumed:

- the q4_K GEMM plateau at ~6 TF — llama.cpp sits on the same plateau,
- flash-kernel throughput at ~0.7 TF vs llama-class ~2 TF.

Answering either needs limiter/occupancy counters. M1 exposes per-dispatch
durations but no limiter counters (M3/A17+ do). Every lever in the program that
was measurable on an M1 has been executed and either landed or reverted with
numbers, so the program was closed rather than left open.

## Reproducing

```sh
benchmark/compare_metal.sh        # cool-state A/B against llama.cpp
```

Read [METHODOLOGY.md](../METHODOLOGY.md) first — a warm GPU reads differently,
and the numbers above are cool-state.

The full session-by-session lab log (phase plans, falsified theories, the
execution ledger, the long-context MQA root-cause hunt) is kept outside the
repository in `~/artikel-geistlib/metal-labor.md`.
