# Numerical validators

Python companions to the C tests in `tests/`. Each runs a C binary that writes
one tensor as raw FP32, then compares it against a reference in NumPy.

`embedding_e2e_smoke.py` drives `dump_geist_embedding` on a synthetic GGUF.

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

## The bring-up ladder — removed

`validate_step1` … `validate_vs_llamacpp` checked the Gemma-3 forward pass one
tensor at a time against HuggingFace reference activations (`dumps/*.npz`) that
nothing in this repository produces, so they could not run and were deleted.
The C tests they drove remain; for ongoing correctness those and the bit-exact
`cpu_scalar` reference gate cover the same ground.
