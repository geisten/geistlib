# Expert usage of MoE models: how skewed is the router? (#48, phase 0)

**Result.** The skew grows with the number of experts per layer.
- With 40 or 64 experts (Granite 3.1 3B-A800M, OLMoE-1B-7B), the router
  spreads its choices almost evenly. The 18.3 % most-used experts serve only
  40–41 % of the selections on held-out text.
- With 128 experts (Qwen3-30B-A3B), the same share serves **54 %**. That
  reaches Strata's reference point: a static, profile-filled cache holding
  4,500 of 24,576 experts (18.3 %) serves 50 % of the routed experts on
  held-out prompts (Strata paper, section 3.4, Figure 3 and finding 4).
- Strata reports no entropy, so that coverage is the yardstick here.
  Normalised entropy is listed alongside: 0.92–0.95 for the small models,
  0.87 for Qwen3-30B-A3B.

## Measurement

- Router selections are counted inside llama.cpp `d81235049` with
  `tools/moe_usage/moe_usage.cpp`. An eval callback reads the
  `ffn_moe_topk-<layer>` tensor, the expert ids the router chose for every
  token.
- Logits are requested for every token, because llama.cpp computes the last
  layer only for tokens whose logits are wanted. Check: every (file, layer)
  has exactly top-k = 8 selections per token.
- Hosts:
  - **RTX 2080 Ti box** (`geist-vulkan` runner; Ryzen 9 9950X, 62 GB RAM,
    RTX 2080 Ti 11 GB): llama.cpp Vulkan build (`GGML_VULKAN=ON`,
    `GGML_NATIVE=ON`, system `cc`/`c++`); `moe_usage` built with
    Ubuntu clang 21.1.8 `-O2`. Gitea runs 1224 (OLMoE and Granite, all
    layers on the GPU) and 1225 (Qwen3-30B-A3B, 20 of 48 layers on the GPU).
  - **The same box, CPU only:** llama.cpp CPU build, Gitea run 1217. Its
    counts differ from the Vulkan run in 0.36 % (OLMoE) and 0.38 % (Granite)
    of the selections, near-ties the two backends round differently. Every
    table below is unchanged to within 0.2 points.
- Models:

  | model | experts / layer | active | layers | GGUF | sha256 |
  | :-- | --: | --: | --: | :-- | :-- |
  | granite-3.1-3b-a800m-instruct Q4_K_M (2.0 GB) | 40 | 8 | 32 | bartowski/granite-3.1-3b-a800m-instruct-GGUF @ `be9a36f0` | `48e0edcd…` |
  | OLMoE-1B-7B-0125-Instruct Q4_K_M (4.2 GB) | 64 | 8 | 16 | allenai/OLMoE-1B-7B-0125-Instruct-GGUF @ `2ac1d273` | `4ddc0e53…` |
  | Qwen3-30B-A3B Q4_K_M (18.6 GB) | 128 | 8 | 48 | Qwen/Qwen3-30B-A3B-GGUF @ `e4d4bafd` | `0d003f66…` |

  Full hashes are in the raw files. Granite has no first-party GGUF.
- Corpora: `tools/moe_usage/corpus.py`, 13 files of about 3,000–4,000 tokens
  in each set. File hashes are in the raw files.
  - Profile set (the hot set is chosen on it): *Pride and Prejudice*
    (English), Kafka's *Die Verwandlung* (German), geistlib's
    `src/engine/*.c`, geistlib's `tools/*.py`.
  - Held-out set (the hot set is tested on it): *Moby-Dick*, Fontane's
    *Effi Briest*, llama.cpp's `src/llama-*.cpp`, llama.cpp's `gguf-py`.
  - Prose is from Project Gutenberg (public domain).
- Raw data:
  - `raw/2026-10-15_moe_expert_usage_vulkan_run1224.txt`
  - `raw/2026-10-15_moe_expert_usage_vulkan_run1225.txt`
  - `raw/2026-10-15_moe_expert_usage.txt` (CPU, run 1217)

## Results

The hot set is the K % most-used experts of each layer on the profile set.
Each cell is the share of the router's selections that hit the hot set. If
the router chose uniformly, every cell would equal K. The row K = 18.3 % is
Strata's point; Strata reaches 50 % there.

```

## OLMoE-1B-7B-0125-Instruct-Q4_K_M: 16 layers x 64 experts, held-out 49653 tokens
normalised entropy of held-out usage per layer: min 0.921 mean 0.947 (1.0 = uniform)
selections per token and layer: [8.0] (must be one value, the model's top-k)
experts never selected on the profile corpus: 0 of 1024
| hot K % | profile (in-sample) | held-out all | held-out en | held-out de | held-out c | held-out py | worst layer, held-out |
| --: | --: | --: | --: | --: | --: | --: | --: |
| 10 % (6) | 27.2 % | 26.1 % | 11.4 % | 20.4 % | 38.2 % | 35.6 % | 19.8 % |
| 18.3 % (12) | 42.4 % | 41.1 % | 21.7 % | 38.4 % | 54.7 % | 51.8 % | 31.6 % |
| 25 % (16) | 50.4 % | 49.0 % | 30.1 % | 47.6 % | 61.7 % | 58.2 % | 40.1 % |
| 50 % (32) | 74.2 % | 72.1 % | 57.7 % | 74.5 % | 80.6 % | 76.7 % | 65.9 % |
| 75 % (48) | 91.0 % | 89.0 % | 82.5 % | 92.0 % | 92.3 % | 89.1 % | 85.6 % |

## granite-3.1-3b-a800m-instruct-Q4_K_M: 32 layers x 40 experts, held-out 50415 tokens
normalised entropy of held-out usage per layer: min 0.761 mean 0.921 (1.0 = uniform)
selections per token and layer: [8.0] (must be one value, the model's top-k)
experts never selected on the profile corpus: 11 of 1280
| hot K % | profile (in-sample) | held-out all | held-out en | held-out de | held-out c | held-out py | worst layer, held-out |
| --: | --: | --: | --: | --: | --: | --: | --: |
| 10 % (4) | 27.0 % | 26.9 % | 25.6 % | 27.6 % | 27.5 % | 27.3 % | 16.6 % |
| 18.3 % (7) | 40.4 % | 40.2 % | 38.1 % | 41.0 % | 41.3 % | 41.0 % | 27.0 % |
| 25 % (10) | 51.6 % | 51.4 % | 47.6 % | 51.6 % | 53.9 % | 53.6 % | 36.4 % |
| 50 % (20) | 78.5 % | 78.0 % | 72.9 % | 77.2 % | 81.7 % | 81.6 % | 63.1 % |
| 75 % (30) | 93.5 % | 93.0 % | 90.5 % | 92.7 % | 95.0 % | 94.4 % | 84.3 % |

## Qwen3-30B-A3B-Q4_K_M: 48 layers x 128 experts, held-out 44547 tokens
normalised entropy of held-out usage per layer: min 0.840 mean 0.874 (1.0 = uniform)
selections per token and layer: [8.0] (must be one value, the model's top-k)
experts never selected on the profile corpus: 112 of 6144
| hot K % | profile (in-sample) | held-out all | held-out en | held-out de | held-out c | held-out py | worst layer, held-out |
| --: | --: | --: | --: | --: | --: | --: | --: |
| 10 % (13) | 41.6 % | 38.5 % | 37.5 % | 54.4 % | 33.3 % | 20.6 % | 25.3 % |
| 18.3 % (23) | 58.1 % | 54.1 % | 49.5 % | 67.6 % | 53.5 % | 38.7 % | 37.6 % |
| 25 % (32) | 68.7 % | 64.5 % | 60.3 % | 76.6 % | 64.6 % | 49.4 % | 46.6 % |
| 50 % (64) | 90.7 % | 88.3 % | 88.0 % | 94.6 % | 87.8 % | 77.8 % | 73.9 % |
| 75 % (96) | 98.5 % | 97.5 % | 97.5 % | 99.4 % | 97.4 % | 94.4 % | 92.5 % |
```

## Observations

- **More experts per layer, more skew.** At K = 18.3 % the held-out cover is
  40 % with 40 experts, 41 % with 64 and 54 % with 128. The 128-expert model
  also leaves 112 of its 6,144 experts unused on the profile set.
- **Profiling carries over.** The held-out cover is 1–4 points below the
  in-sample cover.
- **Domain matters.** In OLMoE the mixed-profile hot set covers C code far
  better than English prose: 55 % vs 22 % at K = 18.3 %. In Qwen3-30B-A3B,
  Python is the weakest domain: 39 % against 68 % for German prose.
- **The worst layer stays far below the average.** In Qwen3-30B-A3B one
  layer reaches only 38 % at K = 18.3 %, so the hot set would have to differ
  in size per layer.

## Interpretation

This part is derived, not measured.
- **Small models.** With 40–64 experts per layer, a static hot set serves too
  little. Take OLMoE on a 4 GB Pi 5 with half the experts resident: 28 % of
  the selections miss, about 130 MB per token from storage, against about
  0.47 GB of experts the token touches in all.
- **Qwen3-30B-A3B.** At 128 experts the profile is as useful as Strata's.
  But at 18.6 GB in Q4_K_M it is far beyond the 4 GB Pi; it would need the
  8 GB Pi with most experts paged, or the M1 Max. #48's question was "small
  MoE on cheap hardware". This measurement says the model has to be at
  least this fine-grained before load-time placement has something to work
  with.
- **Adaptive beats static.** Strata measures it: 50 % → 72 % with
  per-conversation swapping. The domain dependence above points the same
  way. The kernel's page cache is an adaptive LRU of the mmapped experts
  already.

## Hypotheses (not tested)

- Qwen3-Next-80B-A3B (512 experts, 10 active, the design Strata's model
  shares) should be more skewed still. It was not measured, because 128
  experts already reached Strata's point.
- A per-domain profile (code only) should cover more than the mixed profile
  on code.

## Reproduce

```sh
L=$HOME/llama.cpp-cross-engine   # scripts/ensure-llama-bench.sh vulkan (or cpu) builds it at d81235049
clang++ -std=c++17 -O2 -I$L/include -I$L/ggml/include tools/moe_usage/moe_usage.cpp \
    -L$L/build-vulkan/bin -lllama -lggml -lggml-base -Wl,-rpath,$L/build-vulkan/bin -o moe_usage
python3 tools/moe_usage/corpus.py /tmp/c48 $PWD $L
MOE_NGL=99 ./moe_usage OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf profile.json /tmp/c48/profile/*.txt
python3 tools/moe_usage/analyze.py benchmark/results/raw/2026-10-15_moe_expert_usage_vulkan_run1225.txt
```

`moe_usage` writes JSON. The raw files hold the same counts as one
`USAGE` line per (model, set, file, layer), the format `analyze.py` reads.
