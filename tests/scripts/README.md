# Numerical validators

Python companions to the C tests in `tests/`. Each runs a C binary that writes
one tensor as raw FP32, then compares it against a reference in NumPy.

**Only one of these runs today.** The rest are the bring-up ladder from the
original Gemma-3 port and are kept as a record of how the engine was proven
correct — read the section below before assuming any of them still works.

## `validate_gguf_dequant.py` — runs

```sh
make test-dequant                                        # every GGUF in gguf_artifacts/
make test-dequant DEQUANT_MODELS=path/to/model.gguf      # one specific file
```

Compares geist's dequant kernels against **gguf-py**, the canonical reference,
and requires equality to the bit — both sides decode the same bytes, so a
tolerance would only hide a divergence. Cases are derived from whatever GGUF is
present (one tensor per distinct dtype, smallest of each) rather than hardcoded,
because a fixed tensor list goes stale when a re-quantised model changes a
dtype, and then reports its own staleness as a kernel failure.

Large tensors are compared as a prefix of whole rows, capped at ~4 M elements
per side. Without that cap a single embedding table dominates everything:
gemma-4-E2B's `per_layer_token_embd` is 2.3e9 elements, and dequantising it on
both sides peaked at **29.9 GB** — more than a Pi 5 has. Row-wise it costs 1.0 GB
and runs in ~17 s, and it exercises `gguf_dequant_row_to_fp32`, the path the PLE
loader takes on exactly such a board.

It exits 77 (SKIPPED) when the `gguf` package, the model, or the built binary is
missing, so it is safe to run anywhere. Two limits are inherent: gguf-py cannot
parse geist's ternary `i2_s` (type 36), so those files are skipped whole, and it
has no BF16 dequant, so the reference for BF16 is computed here.

Coverage as of 2026-07: F32, Q4_K, Q5_K, Q6_K, BF16 and **TQ2_0**, all bit-exact.

It has already earned its keep: the row path supported every dtype the
whole-tensor path did **except BF16**, so a BF16 PLE table — which is what
gemma-4-E2B ships — failed there while the whole-tensor path succeeded. Fixed in
`src/formats/gguf/common.c`; the two dtype switches now agree.

## The bring-up ladder — retired

`validate_step1` … `validate_step12to14`, `validate_layers0to4`,
`validate_layers0to14`, `validate_full_logits`, `validate_full_logits_gguf`,
`validate_greedy`, `validate_vs_llamacpp`.

They proved the Gemma-3 forward pass one tensor at a time against a HuggingFace
reference — embedding lookup, `input_layernorm`, `q_proj`, QKV + per-head
RMSNorm, RoPE + sliding-window MQA + `o_proj`, MLP and its norms, PLE
pre-compute and `layer_scalar` — then the same pieces chained (layers 0-4, 0-14),
then full logits, then greedy decode, then a head-to-head against llama.cpp.
Exactly the right shape for a port: every tensor alone before any of them
together.

**Why they no longer run.** Three independent reasons, in order of severity:

1. **13 of the 14 need `dumps/*.npz`** — pre-computed PyTorch activations. They
   only `np.load` those files; nothing in this repository produces them, and the
   generator was never committed. Without it they cannot be revived, at any
   effort.
2. **11 also need `gemma-4-E2B-it/model.safetensors`** (~5 GB). `make
   fetch-model` fetches GGUF, not safetensors.
3. **All 14 look for their C binary at the repository root** (e.g.
   `ROOT_DIR / "test_step2_input_layernorm"`). Binaries have long lived under
   `bin/<target>/<mode>/tests/` and carry a `_unit` suffix.

They are kept rather than deleted because the C tests they drive still exist and
their comments name them ("Companion `validate_step1.py` loads
`dumps/T1.npz`…"). Deleting them would leave those comments pointing at nothing.

**If you ever need this again**, the missing piece is a script that dumps HF
reference activations to `dumps/*.npz`; fixing the binary paths afterwards is
mechanical. Reviving one is only worth it while chasing a specific numerical
bug — for ongoing correctness the C tests under `tests/` and the bit-exact
`cpu_scalar` reference gate cover the same ground without a PyTorch install.
