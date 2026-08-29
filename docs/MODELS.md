# Models

A GGUF must use one of the registered architectures (`gemma4`, `llama`,
`qwen3`, `qwen35`, `bitnet-b1.58` or `bitnet`) and dtypes supported by the
selected backend. The table lists the model families exercised by tests and
benchmarks. Text weights are one-download-and-go; Gemma vision/audio additionally
need their matching tower weights.

| Model | Modality | Quant | ~Size | RAM | Best on | Get it |
| :-- | :-- | :-- | --: | --: | :-- | :-- |
| **BitNet b1.58 2B-4T** | text (ternary) | `i2_s` | 1.1 GiB | ≥ 4 GB | **Pi 5 · x86** | `make fetch-bench-model` · [⬇ gguf](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf) — or the [self-contained `geist-bitnet` binary](https://github.com/geisten/geistlib/releases/latest) |
| **Gemma 4 E2B-it** | text · vision · audio | `Q4_K_M` | 3.1 GB | ≥ 4 GB | Mac / Pi 5 | `make fetch-model` · [⬇ gguf](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf) |
| Gemma 4 E4B-it | text · audio | `Q4_K_M` | 4.6 GB | ≥ 8 GB | Mac | `make fetch-e4b-model` · [⬇ gguf](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf) |
| Llama family (e.g. SmolLM2, Llama 3.2) | text | supported GGUF dtypes | varies | varies | everywhere | e.g. [⬇ SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf) (the CI reference) |
| Qwen3 (0.6B / 1.7B / 4B) | text | supported GGUF dtypes | 0.6–4 GB | ≥ 1 GB | everywhere | `make fetch-qwen3-model` · [⬇ Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf) (the CI reference) |
| Qwen3.5/3.6/3.8 dense (0.8B – 27B) | text | supported GGUF dtypes incl. UD mixed (IQ4_XS/IQ4_NL/Q3_K/IQ3_S) | 0.8–16 GB | ≥ 2 GB | CPU; Metal on Mac (27B: ≥ 32 GB) | `make fetch-qwen35-model` · [⬇ Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf) (the CI reference) — hybrid DeltaNet/attention; MoE variants (A3B etc.) unsupported |
| BitNet b1.58-large | text (ternary) | `TQ2_0` | 207 MB | ≥ 1 GB | smallest footprint | convert from [1bitLLM ↗](https://huggingface.co/1bitLLM/bitnet_b1_58-large) |

```bash
make fetch-bench-model   # BitNet b1.58 2B-4T — the ~2× decode win on a Pi 5
make fetch-model         # Gemma 4 E2B-it text model; fetch towers separately
make fetch-audio-tower   # matching E2B audio tower
```

## Vision & audio

Multimodality rides on the Gemma 4 models — the engine has a SigLIP vision
tower and a Conformer audio tower built in, both running on the CPU backends.
See [`ARCHITECTURE.md`](ARCHITECTURE.md) for attaching image/audio inputs
(`EXPERIMENTAL` API surface).

The towers are per-variant: their final projection targets the text model's
residual stream (1536 on E2B, 2560 on E4B), and the engine refuses a tower
whose width doesn't match the loaded GGUF. E4B audio is gated weekly in CI
(`e4b-smoke`); extract its tower with
`python3 tools/fetch_audio_tower.py --url .../gemma-4-E4B-it/resolve/main/model.safetensors`.
E4B vision is unlisted until a test executes it — the loader is
shape-driven, but nobody has run the E4B vision tower yet
(`tools/dump_vision_tower.py` extracts it if you want to be first).

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
