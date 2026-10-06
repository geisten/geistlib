# Numerical validators

Python companions to the C tests in `tests/`. Each runs a C binary that writes
one tensor as raw FP32, then compares it against a reference in NumPy.

**Only `validate_gguf_dequant.py` runs.** The others are the retired bring-up
ladder of the Gemma-3 port; see below before assuming any of them works.

## `validate_gguf_dequant.py`

```sh
make test-dequant                                        # every GGUF in gguf_artifacts/
make test-dequant DEQUANT_MODELS=path/to/model.gguf      # one specific file
```

Compares geist's dequant kernels against **gguf-py**, the canonical reference,
and requires equality to the bit — both sides decode the same bytes, so a
tolerance would only hide a divergence. Cases are derived from whatever GGUF is
present (one tensor per distinct dtype, smallest of each) rather than hardcoded,
so a re-quantised model cannot turn a stale tensor list into a false failure.

Large tensors are compared as a prefix of whole rows, capped at ~4 M elements
per side: gemma-4-E2B's `per_layer_token_embd` (2.3e9 elements) would otherwise
need ~30 GB, more than a Pi 5 has. The row path is `gguf_dequant_row_to_fp32`,
the one the PLE loader uses.

It exits 77 (SKIPPED) when the `gguf` package, the model, or the built binary is
missing, so it is safe to run anywhere. gguf-py cannot parse geist's ternary
`i2_s` (type 36), so those files are skipped whole, and it has no BF16 dequant,
so the BF16 reference is computed here.

Covered dtypes: F32, Q4_K, Q5_K, Q6_K, BF16 and TQ2_0, all bit-exact.

## The bring-up ladder — retired

`validate_step1` … `validate_step12to14`, `validate_layers0to4`,
`validate_layers0to14`, `validate_full_logits`, `validate_full_logits_gguf`,
`validate_greedy`, `validate_vs_llamacpp`.

They check the Gemma-3 forward pass one tensor at a time against a HuggingFace
reference — embedding lookup, `input_layernorm`, `q_proj`, QKV + per-head
RMSNorm, RoPE + sliding-window MQA + `o_proj`, MLP and its norms, PLE
pre-compute and `layer_scalar` — then the same pieces chained (layers 0-4, 0-14),
then full logits, then greedy decode, then a head-to-head against llama.cpp.

**Why they do not run:**

1. **13 of the 14 need `dumps/*.npz`** — pre-computed PyTorch activations that
   nothing in this repository produces.
2. **11 also need `gemma-4-E2B-it/model.safetensors`** (~5 GB). `make
   fetch-model` fetches GGUF, not safetensors.
3. **All 14 look for their C binary at the repository root** (e.g.
   `ROOT_DIR / "test_step2_input_layernorm"`); binaries live under
   `bin/<target>/<mode>/tests/` with a `_unit` suffix.

They are kept because the C tests they drive still exist and name them in their
comments. Reviving one needs a script that dumps HF reference activations to
`dumps/*.npz`, then fixing the binary paths; for ongoing correctness the C tests
and the bit-exact `cpu_scalar` reference gate cover the same ground.
