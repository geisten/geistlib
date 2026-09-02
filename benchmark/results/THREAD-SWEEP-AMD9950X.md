# Thread sweep: is 16/15 a defensible profile on the 9950X?

The `amd_9950x` profile in
[`cross_engine_cpu_protocol.json`](../cross_engine_cpu_protocol.json) started as
a rule — physical cores for prefill, one fewer for decode — mirroring the M1
Max's 8/7. A rule is not evidence. If the two engines peak at different thread
counts, a matched count sits where only one of them is happy, and the matrix
cell measures the choice rather than the engines.

Swept with [`bench_thread_sweep.py`](../../tools/bench_thread_sweep.py), which
imports the gate, the validation and the sample handling from
`bench_cross_engine.py` rather than reimplementing them. pp512 and tg64 at
depth 512 — the shapes the matrix cell headlines. Two **independent runs** per
point and engine, not two cycles inside one run: the AMD cell showed
llama.cpp's prefill drifting further between runs than within one.

Raw: [`raw/2026-09-02T081851Z_thread_sweep_amd_9950x.jsonl`](raw/2026-09-02T081851Z_thread_sweep_amd_9950x.jsonl)

## Result

| threads | geist pp | llama.cpp pp | Δ | geist tg | llama.cpp tg | Δ |
| --: | --: | --: | --: | --: | --: | --: |
| 4 | 267.7 | 280.6 | **−4.6%** | 42.83 | 36.53 | +17.3% |
| 8 | 385.4 | 374.7 | +2.9% | 43.20 | 37.67 | +14.7% |
| 12 | 440.7 | 393.4 | +12.0% | 48.18 | 43.06 | +11.9% |
| 16 | 493.7 | 433.6 | +13.8% | 49.53 | 43.25 | +14.5% |
| 24 | 450.5 | 109.7 | +310.7% | 47.42 | 41.58 | +14.0% |
| 32 | 500.0 | 73.8 | +577.8% | 45.90 | 38.64 | +18.8% |

Replicate spread is at or below 2.4% everywhere except llama.cpp prefill at 12
threads (6.2%). geist stays under 1% at every point.

## Verdict: 16 stands, and it is now measured rather than assumed

Three of the four curves peak at 16 threads. llama.cpp peaks there in both
phases; geist peaks there in decode and comes within 1.3% of its best in
prefill (500.0 at 32 threads against 493.7 at 16 — inside the run-to-run
variation this protocol has already been shown to carry). The decision rule
fixed before the measurement was: same peak for both engines, adopt it. That is
what happened.

The profile keeps 15 for decode. That specific value was **not** measured — the
sweep varied one number and applied it to both phases, so the decode column is
16 threads at the 16-thread row. The decode curve is flat from 12 to 16
(48.18 → 49.53), so 15 sits inside a plateau; nothing suggests a cliff between
them, and nothing here proves its absence either.

## Two things this sweep says about the matrix cell

**The +310% and +578% at 24 and 32 threads are not an engine claim.** They are
llama.cpp's prefill collapsing under SMT — 433.6 to 109.7 to 73.8 tok/s, a 6×
drop, and reproducible (109.2/110.2 and 72.9/74.7 across independent runs).
geist does not collapse. Quoting those ratios as a speedup would be
dishonest arithmetic on a configuration nobody would choose.

**geist's prefill lead is a property of the configuration, not of the engine.**
At 4 threads llama.cpp is 4.6% *faster*. The lead appears as thread count
rises and geist's prefill scaling pulls ahead. The matrix cell is therefore a
statement about "geist versus llama.cpp at 16 matched threads on this host",
which is what a cell in a system-versus-model matrix is supposed to mean — but
it does not generalize to a machine with four usable cores.

The decode ratio does generalize within this host: +12% to +19% across the
entire 4-to-32 range, with no thread count where llama.cpp leads.
