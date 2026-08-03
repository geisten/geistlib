# Models

Any GGUF that carries its own tokenizer works. Two models are first-class and
one-download-and-go; the rest of the table is what the test suite and the
benchmarks exercise.

| Model | Modality | Quant | ~Size | RAM | Best on | Get it |
| :-- | :-- | :-- | --: | --: | :-- | :-- |
| **BitNet b1.58 2B-4T** | text (ternary) | `i2_s` | 1.1 GiB | ≥ 4 GB | **Pi 5 · x86** | `make fetch-bench-model` · [⬇ gguf](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf) — or the [self-contained `geist-bitnet` binary](https://github.com/geisten/geistlib/releases/latest) |
| **Gemma 4 E2B-it** | text · vision · audio | `Q4_K_M` | 2.9 GB | ≥ 4 GB | Mac / Pi 5 | `make fetch-model` · [⬇ gguf](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf) |
| Gemma 4 E4B-it | text · vision · audio | `Q4_K_M` | 4.6 GB | ≥ 6 GB | Mac | [⬇ gguf](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf) |
| Llama family (e.g. SmolLM2, Llama 3.2) | text | any GGUF quant | varies | varies | everywhere | e.g. [⬇ SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf) (the CI reference) |
| BitNet b1.58-large | text (ternary) | `TQ2_0` | 207 MB | ≥ 1 GB | smallest footprint | convert from [1bitLLM ↗](https://huggingface.co/1bitLLM/bitnet_b1_58-large) |

```bash
make fetch-bench-model   # BitNet b1.58 2B-4T — the ~2× decode win on a Pi 5
make fetch-model         # Gemma 4 E2B-it — text + vision + audio towers
```

## Vision & audio

Multimodality rides on the Gemma 4 models — the engine has a SigLIP vision
tower and a Conformer audio tower built in, both running on the CPU backends.
See [`ARCHITECTURE.md`](ARCHITECTURE.md) for attaching image/audio inputs
(`EXPERIMENTAL` API surface).

## Ternary (1.58-bit) as a first-class citizen

geistlib runs Microsoft's BitNet b1.58 (canonical `i2_s` and compact `TQ2_0`)
with integer-only dot products — ARM SDOT (add/subtract, no multiplies) and
x86 AVX-512 VNNI. It beats Microsoft's own bitnet.cpp on both a Pi 5 (~2×
decode) and an AMD 9950X; exact numbers in
[`../benchmark/README.md`](../benchmark/README.md). BitNet is ternary, so the
whole 2B-4T model is 1.1 GiB — about a third the footprint of a comparable
4-bit model, small enough to be roomy on a 4 GB Pi.

`TQ2_0` has no canonical GGUF upstream yet — conversion recipe in
[`../benchmark/results/TERNARY.md`](../benchmark/results/TERNARY.md).
