# Forward-only fine-tuning (ZO gains)

geist can adapt a model to a task **without a backward pass**: no autodiff, no
gradient buffers, no optimizer state, no FP16 shadow weights. Training memory
equals inference memory, and the result is the unmodified GGUF plus a few
hundred floats.

This exists because a ternary model has no gradient to propagate. The weights
are trits — `{-1, 0, +1}` — and the quantization function has a derivative of
zero almost everywhere. The usual answer is quantization-aware training with a
straight-through estimator, which needs full-precision shadow weights and a
backward pass, i.e. exactly the two things a 4 GB Raspberry Pi does not have.

The answer here is the other one: estimate the gradient from forward passes
alone, and shrink the search space until that is cheap enough to converge.

---

## 1. What is trainable

Weights stay frozen. What moves is a **gain** — one `float` per linear weight,
`1.0` at load, multiplied into that weight's output:

```
y = gain · (W̄ x)        W̄ frozen, gain trainable
```

For a 30-layer model that is `9 · 30 + 2 = 272` trainable scalars. Slot order:

| slots | tensor |
| :-- | :-- |
| `9·l + 0 … 8` | layer `l`: q, k, v, o, gate, up, down, per_layer_gate, per_layer_proj |
| `9·n_layers + 0` | lm_head (embed table) |
| `9·n_layers + 1` | model_proj |

The order depends on `n_layers` alone, so a sidecar written by one process is
readable by another on the same model.

### What that can and cannot learn

A per-tensor gain reweights how much each linear contributes to the residual
stream. It can turn a head or an FFN branch up, down, or off; it can
recalibrate a quantization-damaged layer; it can shift the output
distribution. Combined with the lm_head gain — which in BitNet is a
full-precision tensor — it has real control over what the model emits.

It cannot flip the sign of a weight, cannot revive a zero trit, and cannot add
a direction that is not already in the frozen ternary basis. **No new
knowledge, full control over weighting and output.** For task adaptation on a
few hundred examples that is usually the binding constraint anyway; for
teaching the model facts it does not have, it is not enough.

The upgrade path, if per-tensor capacity runs out, is per-block scales: TQ2_0
carries one f16 scale per 256 trits, so a `[2560, 2560]` weight has 25 600 of
them — roughly a rank-10 LoRA's parameter budget. They sit at a 66-byte stride
inside the packed weight bytes and reaching them needs a backend-layout-aware
accessor, which is why the shipped version stops at per-tensor.

---

## 2. The optimizer

Two-sided SPSA, the form MeZO uses:

```
d = (L(θ + εz) − L(θ − εz)) / 2ε        z ∈ {−1,+1}ⁿ
θ ← θ − lr · d · z
```

Two forward passes produce one scalar `d`. That scalar times the same random
direction `z` is the update. Three properties make it work here:

**`z` is never stored.** It is regenerated from a 64-bit seed on each of the
three uses (both probes and the update), so the perturbation costs no memory.

**`base` is the only running state.** Each probe is written *from* `base`
rather than added to the live array, so no float rounding accumulates across
steps and a restore is exact by construction.

**`d` is clipped.** With a batch of 1–4 the estimate is noisy, and one outlier
divided by `2ε` is large enough to wreck the vector. This is QZO's directional
derivative clipping.

### Why the dimension matters

Zeroth-order convergence degrades with the number of parameters being
optimized. MeZO needs ~20 000 steps where backprop needs ~625, and that is on
a full parameter vector in a low-effective-rank regime near initialization.

Training 272 scalars instead of 2·10⁹ is what makes the method practical
rather than theoretical. It is also why the perturbation loop itself stays
free: perturbing two billion weights three times per step would dominate two
forward passes on an A76, while perturbing 272 floats is not measurable.

This is the one place where ternary is an *advantage* rather than a
constraint. The frozen trits are not a limitation to work around — they are
what collapses the search space.

---

## 3. The loss

Teacher-forced negative log-likelihood of a target given a context, per
target token:

```
reset()                          → back to the pinned prefix
prefill(ctx suffix)              → one prefill
for each target token k:
    logits = peek_logits()       → NLL contribution
    prefill(target[k])           → one token, decode cost, warm KV
```

One example costs about one prefill. The target tokens run against a warm KV
cache at decode cost.

Everything here is the public `geist_util.h` surface —
`geist_session_prefill_tokens`, `geist_session_peek_logits`. No engine
internals, no autodiff hooks.

---

## 4. Shared-prefix pinning

Training examples usually share a system prompt. Re-prefilling it for every
example, twice per step, is most of the run.

`geist_session_pin_prefix` puts the shared tokens in the KV cache once and
makes `reset()` truncate back to them instead of to empty. The trainer
computes the longest token prefix common to all examples, pins it, and
prefills only suffixes.

**BitNet 2B-4T, 68-token shared prompt, same seed, quiesced board:**

| | prefill / example | wall clock |
| :-- | --: | --: |
| Pi 5, unpinned, 20 steps | 76.4 tokens | 94.7 s |
| Pi 5, pinned, 20 steps | 8.4 tokens (−89 %) | **29.0 s — 3.27×** |
| M1 Max, unpinned, 40 steps | 76.4 tokens | 59.5 s |
| M1 Max, pinned, 40 steps | 8.4 tokens | **15.8 s — 3.77×** |

Pinning is worth more than the raw token ratio suggests on neither machine
and less than it on both, because scoring an example is a reset plus a
prefill plus one call per target token — the per-call overhead does not
shrink.

### When pinning pays — and when it does not

The payoff is not set by how many prompt families you have. It is set by
**what fraction of the prefilled context is shared**, and that varies by two
orders of magnitude between plausible-looking tasks.

Measured on BitNet 2B-4T, 20 steps, batch 1, same seed:

| task shape | shared | wall clock | |
| :-- | --: | --: | --: |
| home routing — fixed system prompt, short requests | 89 % | 94.7 s → 29.0 s | **3.27×** |
| tool calling, one family — [hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) glaive | 16 % | 490 s → 419 s | **1.17×** |
| tool calling, all families mixed | 1 % | — | ~1.00× |

The saving passes through almost 1:1 at long contexts (a 16 % prefill cut
buys 14.5 % of wall clock) and better than 1:1 at short ones, where the
per-call overhead pinning removes is a larger share of the total.

What makes the difference is where the bulk of the context sits. A fixed
system prompt with short user turns puts it in the shared part. A tool-calling
prompt puts it in the per-request tool list — the boilerplate before `<tools>`
is shared and at the front, which is the right place, but it is only ~300 of
~1 500 characters.

Grouping examples by family and pinning per family therefore buys at most
~1.15× on data of this shape, before paying for re-pins on family switches
(~44 % of steps at realistic family sizes) and for the correlated batches that
family-blocked sampling forces on a gradient estimator that is already noisy.
**That is why the tool pins one prefix and stops.**

The corollary is worth more than the measurement: if your own agent sends a
*stable* tool schema — as a fixed home deployment does — that schema is shared
prefix again, and you are back in the first row. Hermes's diversity is
synthetic; a real deployment usually is not that diverse.

### Two invariants that bite

Both of these were found by measurement, not by review, and both produce
loss curves that look entirely healthy while being wrong.

**`pin_len` is not an on/off switch.** Once a prefix is pinned, `reset()`
restores to it unconditionally. Passing `pin_len = 0` while a prefix is
pinned prefills the shared context a *second time on top of itself*.
Measured: the holdout read 3.26 instead of 2.79, and the only reason it was
caught is that two runs which had to agree on their step-0 value did not.
Pass `0` only when nothing is pinned.

**The pinned KV is a function of the weights, and the gains are weights.**
The cached prefix reflects the gains that were live when it was pinned, while
suffix and target are scored with the current ones — so training makes its
own cache stale.

It is not fatal: within a step both probes see the same prefix, so the finite
difference is taken at a consistent, if slightly displaced, operating point,
and a run tuned this way still generalizes. But it is measurable, and it
flatters the number being reported. After 400 steps the in-run holdout read
**0.1556** against a prefix pinned at `gains = 1.0`, and **0.2249** once
re-pinned at the tuned gains — 0.07 nats of pure bookkeeping error.

The trainer therefore re-pins every `REPIN_EVERY_STEPS` (25) and again before
every holdout evaluation. One prefill of the shared prefix per 25 steps is far
below what the pin saves, and it bounds the drift instead of letting it
accumulate over the run. With that in place the in-run final holdout and a
fresh process loading the same sidecar agree exactly:

```
training run, step 400 holdout  0.4562
fresh process, --init g2.bin    0.4562
```

### The verification gate

`geist_session_pin_prefix` returns `GEIST_OK` even when the architecture's
`pin_prefix` failed underneath — `session.c` discards that status. An
unnoticed failure would mean training on truncated contexts behind a loss
curve that still looks reasonable.

So the pin is not trusted, it is **checked**: one example is scored with a
full prefill, then scored again on top of the pinned prefix. The two must
agree.

They agree to a tolerance, not exactly, and the tolerance is measured rather
than guessed:

| | Δ (nats) |
| :-- | --: |
| pinned vs unpinned, Apple/Accelerate | 0.011 |
| pinned vs unpinned, Pi 5 native int8 kernel | < 0.0001 |
| **context actually missing** | **6.19** |

The Apple discrepancy is prefill chunking: `arch_ops.c` prefills in
`m_max`(=64)-token sub-batches, so 76 tokens in one call split 64+12 while
pinned-68 plus an 8-token suffix splits 64+4 then 8. Accelerate's sgemm sums
different batch shapes in a different order and the last bits move. On the
Pi's own int8 kernel the accumulation order does not depend on the batch
shape and the difference vanishes — which confirms the diagnosis.

The threshold is **0.1 nats, absolute**. Roughly 10× above the noise and 60×
below the fault.

Absolute rather than relative, and that is not a style choice: once tuning
drives the loss to ~0.02, the same 0.003-nat wobble reads as 11 % and a
relative gate rejects a pin that is perfectly fine. This was found by running
the round-trip test, not by reasoning about it.

---

## 5. Deployment

A tuned model is the base GGUF plus `n_gains` floats — **1 088 bytes** for
BitNet 2B-4T. Not a second model on the SD card, no requantization.

```c
float *g; size_t n;
if (geist_model_gains(model, &g, &n) == GEIST_OK) {
    FILE *f = fopen("home-routing.bin", "rb");
    if (f) { (void) fread(g, sizeof *g, n, f); fclose(f); }
}
```

A write takes effect on the next forward pass. Switching between tuning
profiles at runtime is a `memcpy` — no reload, no second instance. Hold as
many profiles in RAM as you like and select per context.

**Limit:** the array is model-global, not per-session. With concurrent
sessions a switch is visible to all of them immediately. One session at a
time is fine; two profiles served simultaneously needs two model instances.

---

## 6. Build switch

The gain multiply is behind `GEIST_TUNE`. Without it, `apply_gain` expands to
`((void) 0)`, no gains array is allocated, and `geist_model_gains` returns
`GEIST_E_UNSUPPORTED` — so the shipped binary is provably unchanged rather
than argued to be.

```sh
make clean && make EXTRA_CFLAGS=-DGEIST_TUNE
```

`make` does not invalidate objects when `EXTRA_CFLAGS` changes. `make clean`
first, or you will silently link a stale mix.

On a Pi 5 the target expects gcc ≥ 14 for `-std=c23`. If the board only has
gcc 12, clang 19 works with three flags the pi5 target assumes gcc for:

```sh
make TARGET=pi5 CC=clang-19 \
     EXTRA_CFLAGS="-Wno-unknown-warning-option -D_GNU_SOURCE -fno-finite-math-only -DGEIST_TUNE"
```

`-Wno-unknown-warning-option` covers the gcc-only `-Wno-nonnull-compare` /
`-Wno-vla-parameter` in `mk/target-pi5.mk`, `-D_GNU_SOURCE` exposes
`clock_gettime`, and `-fno-finite-math-only` is what the mac target already
pairs with `-ffast-math`. Absolute throughput from a clang build is not
comparable to the gcc numbers in `../benchmark/results/PI5.md`.

`op_gains` also refuses a backend that runs the fused tensor linear path
(GPU): that path returns before the gain is applied, so handing out a
writable array would silently produce ungained output. CPU backends only.

---

## 7. Running it

```sh
bin/<target>/release/tools/zo_tune \
    --gguf model.gguf --data train.jsonl --out gains.bin \
    --steps 400 --batch 2 --lr 3e-3 --eps 1e-2 --holdout 4
```

Data is one JSON object per line:

```json
{"prompt": "turn on the kitchen light", "completion": "light.kitchen"}
```

The loss is the NLL of `completion` given BOS + `prompt`. The two are
tokenized separately, so the split lands on a token boundary by construction —
write the completion the way the model should emit it, leading space included
if that is what the tokenizer produces.

| flag | |
| :-- | :-- |
| `--init PATH` | continue from an existing sidecar |
| `--no-pin` | disable shared-prefix pinning |
| `--holdout N` | hold out the last N examples (default 10 %) |
| `--eval-every N` | holdout eval interval, 0 = off |
| `--clip F` | directional-derivative clip (default 10) |
| `--self-check` | run the unit checks, no model needed |

`--self-check` covers the perturbation algebra, the JSON escape decoding and
the shared-prefix length rule. It uses an always-on `CHECK` macro rather than
`assert`, because the shipped build is `-DNDEBUG` and asserts would compile
away into a self-check that reports success on no evidence.

### Batch size is the trap

Batch multiplies every step. ZO papers use 8–16 for stability; on a Pi that
turns a 20-minute run into a 3-hour one. Use 1–4 and compensate with steps.

---

## 8. Cost

Per step: 2 forward passes × batch examples. Wall clock scales with prefill
throughput and, decisively, with how much of the context is pinned.

Measured on BitNet 2B-4T (`i2_s`, 30 layers, 272 gains) with a 68-token
shared prompt, 16 examples of which 4 held out, pinned:

| | Pi 5 (A76, 4 threads, quiesced) | M1 Max |
| :-- | --: | --: |
| 20 steps, batch 1 | 29.0 s | — |
| 40 steps, batch 1 | — | 15.8 s |
| 400 steps, batch 2 | 14 min 47 s | ~4 min |

The Pi 5 long run carries a caveat its short runs do not: it holds all four
cores busy for fifteen minutes and drives a passively cooled board from
55 °C to 77 °C, where the soft temperature limit trips. Treat it as what a
real unattended run costs, not as a throttle-free measurement — and check
`vcgencmd measure_temp` before comparing two of them.

**A full tuning run fits on the Pi.** Fifteen minutes, unattended, inside
4 GB, producing a 1 088-byte sidecar. Nothing has to leave the board.

**Scaling to your data.** Steps, not examples, set the cost: a ZO step is two
forward passes over `batch` examples regardless of how large the training set
is. 1 000 examples cost the same per step as 16. What changes is how many
steps you need, and how much context can be shared and therefore pinned.

For short examples the run is dominated by per-call overhead rather than
throughput — a scored example is a reset, a prefill, and one call per target
token. Pinning attacks exactly that, which is why it is worth 3.3×.

---

## 9. A verified run

BitNet 2B-4T on the 4 GB Pi 5. Task: map a natural-language home-automation
request to one device identifier. 16 examples, 4 held out, 68-token shared
system prompt. 400 steps, batch 2, `lr 3e-3`, `eps 1e-2`.

```
pin: 68-token prefix verified (loss delta 0.0000 nats)
272 gains, 12 train + 4 holdout examples
pinned 68 shared context tokens, prefill drops 89% (76.4 -> 8.4 tokens)
step   0  holdout 3.2640
step 100  holdout 2.1413
step 200  holdout 0.9464
step 300  holdout 0.6447
step 400  holdout 0.8430
14m47s — wrote 272 gains (1088 bytes)
```

Reloading that sidecar in a fresh process reproduces `0.8430` exactly, and
the file is byte-identical after the round trip.

Two things worth reading off this rather than past:

**Holdout NLL fell 3.26 → 0.64 on held-out examples**, from 272 trainable
floats and not one backward pass. The model was not shown these four
requests.

**Step 400 is worse than step 300.** There is no learning-rate schedule — a
constant `lr` bounces around the minimum once it gets there, and the run
happened to end on an upswing. Quoting 0.6447 as "the result" would be
picking the best point of a noisy curve. If you need the best vector rather
than the last one, checkpoint on holdout improvement; the tool currently
writes whatever the final step produced.

---

## 10. What this is not

ZO methods work *near* the initialization point. MeZO's own theory depends on
it, and the gain formulation makes it structural: multiplicative reweighting
of frozen circuits cannot travel far. Expect calibration, formatting, routing
and behaviour suppression. Do not expect new facts, new vocabulary behaviour,
or new reasoning.

If a run plateaus above where you need it, the answer is not more steps. It is
more capacity — per-block scales, or a real QAT pipeline on a real machine.

---

## References

- Malladi et al., *Fine-Tuning Language Models with Just Forward Passes* (MeZO), [arXiv:2305.17333](https://arxiv.org/abs/2305.17333)
- *Fine-tuning Quantized Neural Networks with Zeroth-order Optimization* (QZO), [arXiv:2505.13430](https://arxiv.org/html/2505.13430v3)
- *QuZO: Quantized Zeroth-Order Fine-Tuning*, [arXiv:2502.12346](https://arxiv.org/pdf/2502.12346)
- *FZOO: Fast Zeroth-Order Optimizer*, [arXiv:2506.09034](https://arxiv.org/pdf/2506.09034)
- *BitNet Distillation*, [arXiv:2510.13998](https://arxiv.org/html/2510.13998v1)
