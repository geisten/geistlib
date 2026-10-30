# Skipping the router's low-weight experts (#240)

**Result: not worth it.** The issue's rule was a recommended setting only if
perplexity stays within 0.5 %. The settings that hold the bar remove under
2 % of the routed selections, and every setting that removes real work misses
it. The experiment was not merged; main still runs all k selected experts.

## Setup

- Branch `moe/240-tail-skip` (an experiment, not pushed):
  `GEIST_MOE_TAIL=top:<r> | min:<t> | cum:<p>` on the host route. Dropped
  selections are not computed and the kept weights are renormalised.
- The device route is off while the tail is set, and in the baseline too
  (`GEIST_MOE_GPU_ROUTE=0`), so both sides use the same arithmetic.
- `bench_nll`: 1,023 tokens fed one at a time, GPU mode on the RTX 2080 Ti,
  the #48 / run-1830 expert profiles.
- Two texts: `docs/OFFLOAD.md` and a held-out English Wikipedia excerpt
  (simple-wiki, 14 KB).
- Raw per-position dumps: desktop `~/moe245/q240/`.

## Quality

Δ perplexity against the same run without a tail. Top-1 is agreement per
position with that run. "Kept" is counted over 512 wiki tokens.

| setting | kept of 8 | Qwen3-30B, OFFLOAD | Qwen3-30B, wiki | Qwen3.6-35B, OFFLOAD | Qwen3.6-35B, wiki |
|:--|--:|--:|--:|--:|--:|
| `top:7` (Strata's `ROUTE_TAIL_SKIP=7`) | 7.00 | **+3.30 %**, 90.8 % top-1 | −0.39 %, 93.9 % | +0.27 %, 92.3 % | **+1.83 %**, 94.4 % |
| `top:6` | 6.00 | +6.41 % | +1.71 % | +5.70 % | +3.98 % |
| `top:5` | 5.00 | +9.48 % | +4.12 % | +11.87 % | +8.64 % |
| `top:4` | 4.00 | +30.0 % | +17.2 % | +23.9 % | +15.8 % |
| `cum:0.95` | 7.87 / 7.89 | +0.60 %, 96.7 % | −0.01 %, 97.7 % | −0.68 %, 95.7 % | −0.17 %, 97.9 % |
| `cum:0.9` | | +3.12 % | +0.40 % | +1.43 % | +2.01 % |
| `cum:0.8` | | +5.88 % | +3.19 % | +7.32 % | +5.26 % |
| `min:0.03` | 7.97 / 7.97 | +0.22 %, 97.7 % | −0.14 %, 97.6 % | −0.75 %, 97.7 % | +0.30 %, 97.4 % |
| `min:0.06` | | +1.11 % | +1.70 % | +1.24 % | +1.91 % |

Base perplexities:

| | OFFLOAD | wiki |
|:--|--:|--:|
| Qwen3-30B | 29.94 | 8.04 |
| Qwen3.6-35B | 17.15 | 5.42 |

## Reading

- **Only `min:0.03` holds the bar on every model and text;** `cum:0.95` misses
  it on one cell (+0.60 %). They drop 0.3–0.4 % and 1.4–1.7 % of the routed
  selections, so under ~1 % of a decode step.
- **Every setting that removes real work fails.** Strata's skip-7 removes
  12.5 % of the selections and costs up to +3.3 % perplexity (Qwen3-30B,
  OFFLOAD) and +1.8 % (Qwen3.6, wiki), with 6–9 % of top-1 predictions changed.
- **These routers have no cheap tail.** Qwen3's renormalised 8th weight is
  rarely below a few percent.

## Speed

Not measured. The `tok/s` column in the raw data is not a measurement: the
desktop was at load 35–55 from CI throughout. No setting qualified, so
device-side support was not built; it would need the rank kernels of #162
(Vulkan) and #201 (Metal) to emit a variable k, and the host route loses
#162's +7–9 %. The upper bound per setting is the share of expert work
removed, at most 12.5 % of the expert part for `top:7`.

If it is ever wanted as an explicitly lossy option, `top:7` is the candidate,
at a stated +0.3 to +3.3 % perplexity. It is not a default.
