# A model-specialized forward: upper bound on the Pi 5 (#37, part A)

**Result: negative.** A forward specialized to BitNet b1.58 2B-4T can save at
most the time the generic forward spends outside its kernels. On the Pi 5
that is about **0.5 %** of decode, far below the 5 % in #37's decision rule.
So the specialized forward was not built.

## Why an upper bound is enough

A specialized forward uses constexpr geometry, no family flags and direct
kernel calls. It calls the **same** `linear_m1` / `linear_mN`, attention and
norm kernels; #37 asks for bit-identical output. The only time it can remove
is the time between kernel calls: flag checks, dispatch through the backend
vtable, tensor views, and the per-layer bookkeeping. This measures that time
directly.

## Measurement

- Raspberry Pi 5 (4 GB), `with-pi-lock`, thermal gate < 56 °C at start.
- geistlib `527bf29d` (main, 2026-10-14), `TARGET=pi5 GEMM_PROVIDER=native`.
- Model: `bitnet-b1.58-2B-4T-i2_s.gguf`, microsoft/BitNet-b1.58-2B-4T-gguf @
  `a1f2f1c7`, sha256 `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`.
- Gitea run 1139.

| | tok/s |
| :-- | --: |
| decode after a 32-token prompt (10 repeats) | 19.15 |
| decode after a 512-token prompt (10 repeats) | 15.98 |
| prefill 512 | 47.1 |

**Stage coverage of decode** (`GEIST_PROFILE_FORWARD=1`). Prompt 512 with 64
decode steps, minus the same run with 1 decode step, leaves the decode alone.
Both runs: 3 repeats and the warm-up, 63 decode steps each.

| stage group | decode share of the profile |
| :-- | --: |
| attention block (norm, qkv, q/kv prep, store, core, o_proj, post) | 4162 ms |
| FFN block (norm, gate_up, act, mul, SubLN, down, post) | 6271 ms |
| head (lm_head) | 1575 ms |
| **sum of stages** | **12 008 ms** |
| decode wall time (3 x 4024 ms) | 12 072 ms |

The stages cover **99.5 %** of decode. The stages themselves are mostly
kernel calls. `perf record` of a 128-token decode gives:

| symbol | share |
| :-- | --: |
| `i2s_m1_rows` (I2_S GEMV) | 65.0 % |
| `i2s_mN_rows` (I2_S prefill GEMM) | 11.2 % |
| `spec_rough_rows` (lm_head shortlist) | 10.7 % |
| `nkv_items` (attention) | 2.9 % |
| `spec_build_rows` | 1.3 % |
| `transformer_spec_head_try` | 0.9 % |
| unresolved (libgomp / libc) | ~6 % |

No `transformer_*` forward function other than the head shortlist reaches 0.5 %.

## Conclusion

- **H1 holds.** Decode is kernel- and bandwidth-bound; the generic forward
  costs about 0.5 %, so a specialized one cannot reach the 5 % threshold.
- **H2 (prefill, small models)** is not measured here. The prefill stages are
  the same GEMMs, and their share only grows with the batch, so the bound is
  not looser there.
- Speed is therefore no reason for model-specialized forwards. Faster
  kernels are, as in #7, #16 and #18.
- **Part B of #37 (per-family forwards for maintainability) stays open.** It
  is a refactor of the forward, judged by the removal diff and performance
  neutrality, not by speed.
