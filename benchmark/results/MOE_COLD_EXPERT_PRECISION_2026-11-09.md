# Cold experts in lower precision (#247)

**Result: not worth it in geistlib. No engine support; at most an offline
file recipe for a smaller GGUF.**

- **Quality is cheap only where the saving is cheap.** On Qwen3-30B-A3B
  (128 experts) Q3_K on the 50 % least-used experts holds the 0.5 % bar of
  #240 (+0.43 % ± 0.29 perplexity, KLD 0.031), and Q2_K holds it up to 25 %.
  Beyond that, and for IQ2_S (no importance matrix) already at 25 %
  (+1.8 %), it misses. OLMoE (64 experts, a flat router) misses the bar for
  every setting from 25 % on; Q3_K at 25 % costs +1.1 %, Q2_K +3.7 %.
- **The profile is worth about a factor of four.** The same 50 % in Q3_K
  picked at random costs Qwen3 +1.73 % instead of +0.43 %: the least-used
  half serves only 11.7 % of the held-out selections.
- **Speed: slower, not faster, on the backends we have.** All experts of
  Qwen3-30B-A3B in Q3_K or IQ2_S decode at 0.26× / 0.20× of Q4_K_M on Metal
  (no Q3_K/IQ2_S MoE kernels on the device route) and at 0.84× / 0.59× on
  cpu_neon (median of three cycles). The bytes shrink by 29 % / 47 %, but
  these formats' kernels cost more than the bytes they save. Mixed files
  could only be slower than Q4_K_M by the cold share of the work.
- **Offload (derived).** Lower-precision cold experts do not let more
  experts fit in VRAM at the budgets that matter: the hot set stays in
  Q4_K_M and fills the budget before any cold expert is reached. The only
  lever left is the CPU branch, which would read 30 % (Q3_K) to 48 % (IQ2_S)
  fewer bytes per token — but on cpu_neon those bytes are not the bound, and
  at 8 GB the non-resident experts are 55 % of all, beyond the share that
  stays within the bar.

## Setup

- **Models** (the files the #48 usage profile was taken on, sha256 checked):
  Qwen3-30B-A3B Q4_K_M (`0d003f66…`, 48 layers × 128 experts) and
  OLMoE-1B-7B-0125-Instruct Q4_K_M (`4ddc0e53…`, 16 × 64).
- **Cold experts.** Per layer, the 10/25/50 % of experts with the fewest
  router selections on the #48 **profile** set (Pride and Prejudice, Die
  Verwandlung, geistlib C and Python;
  `raw/2026-10-15_moe_expert_usage_vulkan_run122{4,5}.txt`). Control: the
  same number of experts drawn at random per layer (`rnd-…`).
- **Precision.** `tools/moe_cold_requant.py` rewrites the GGUF offline.
  GGUF holds one type per expert tensor, so every expert is carried in a
  Q8_0 container: hot experts as `Q8_0(Q4_K_M)`, cold ones as
  `Q8_0(cold-type(Q4_K_M))`. Quantisation is ggml's reference code through
  ctypes, without an importance matrix (as `llama-quantize` without
  `--imatrix`). A check on OLMoE block 3: hot experts within 0.5 % RMS of
  the original, cold ones within 0.5 % of a direct Q2_K round trip.
- **Baseline** (`base`): the same container with no cold experts. Against
  the unmodified file it reads −0.02 % (Qwen) and +0.17 % (OLMoE)
  perplexity, KLD 0.002 / 0.003: the container's own noise floor.
- **Quality.** llama.cpp `d81235049` `llama-perplexity` on Metal, `-c 512`,
  KL divergence against `base`, on the #48 **held-out** set (Moby-Dick, Effi
  Briest, llama.cpp `src/llama-*.cpp`, `gguf-py`), one file per domain,
  ≈ 45 k tokens; disjoint from the profile set. The three Effi Briest chunks
  differ from October's (Project Gutenberg changed the file); the hashes are
  in `raw/moe-cold-requant-2026-11-09/heldout.sha256`.
- **ΔPPL** is the variant's perplexity over `base`'s, chunk-weighted over
  the four domains, ± the paired standard error from the KL run.
  The KL run's own `PPL(base)` comes from compressed stored logits and reads
  0.04–1.9 % low, so it is not used (`tools/moe_cold_requant_table.py`).
- **Speed.** geistlib `29a0ce4b` (`make`, mac-omp, Metal + cpu_neon),
  `bench_perf_sweep --seq-lens 512 --decode-n 64 --warmup 16 --repeats 3
  --prompt-file` (held-out English), `--m-max 512` on Metal and 128 on
  cpu_neon (its limit), three interleaved cycles of Q4_K_M, all experts in
  Q3_K and all in IQ2_S (`--cold 1 --out-type <type>`: quantised once,
  no container). Mixed files cannot be timed: a GGUF tensor has one type.
- Host: M1 Max 64 GB, every run under `with-mac-lock`. The first attempt
  was lost to a reboot; the quality runs were repeated and reproduced to
  the last digit.

## Quality

"Cold selections" is the share of the held-out router selections that land
on the cold experts.

### Qwen3-30B-A3B (128 experts per layer)

| variant | cold selections (held-out) | ΔPPL all | KLD | same top-1 | ΔPPL en | de | c | py |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| orig (no container) | – | −0.02 % ± 0.19 | 0.0018 | 98.5 % | −0.2 % | +0.1 % | −0.1 % | +0.1 % |
| Q3_K, 10 % least used | 0.4 % | −0.06 % ± 0.19 | 0.0023 | 98.4 % | −0.2 % | −0.1 % | +0.0 % | +0.2 % |
| Q3_K, 25 % | 2.5 % | −0.10 % ± 0.19 | 0.0040 | 98.0 % | −0.0 % | −0.3 % | −0.1 % | +0.1 % |
| **Q3_K, 50 %** | 11.7 % | **+0.43 % ± 0.29** | 0.0313 | 93.7 % | +1.1 % | +0.3 % | −0.2 % | +0.3 % |
| Q3_K, 50 % random | 49.1 % | +1.73 % ± 0.34 | 0.0507 | 91.7 % | +3.3 % | +1.2 % | +0.9 % | +1.1 % |
| Q2_K, 10 % | 0.4 % | −0.14 % ± 0.22 | 0.0091 | 96.5 % | +0.4 % | −0.5 % | −0.2 % | −0.5 % |
| **Q2_K, 25 %** | 2.5 % | **−0.10 % ± 0.24** | 0.0135 | 96.1 % | +1.0 % | −0.8 % | −0.2 % | −0.8 % |
| Q2_K, 50 % | 11.7 % | +5.18 % ± 0.45 | 0.0918 | 89.0 % | +13.6 % | +3.6 % | +1.1 % | +0.1 % |
| Q2_K, 25 % random | 25.1 % | +10.47 % ± 0.51 | 0.1281 | 86.3 % | +20.9 % | +11.6 % | +3.3 % | +3.0 % |
| IQ2_S, 25 % | 2.5 % | +1.82 % ± 0.25 | 0.0167 | 95.5 % | +3.9 % | +1.1 % | +0.9 % | +0.9 % |

Base perplexity (container, no cold experts): en 14.93, de 12.91, c 3.50,
py 2.25; OLMoE 11.21, 24.69, 4.05, 2.58.

### OLMoE-1B-7B (64 experts per layer)

| variant | cold selections (held-out) | ΔPPL all | KLD | same top-1 | ΔPPL en | de | c | py |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| orig (no container) | – | +0.17 % ± 0.08 | 0.0033 | 97.9 % | +0.3 % | +0.2 % | +0.1 % | +0.0 % |
| Q3_K, 10 % least used | 3.0 % | +0.26 % ± 0.11 | 0.0080 | 96.7 % | +0.8 % | +0.2 % | +0.0 % | +0.0 % |
| Q3_K, 25 % | 11.0 % | +1.08 % ± 0.17 | 0.0239 | 94.1 % | +4.0 % | +0.1 % | +0.1 % | −0.0 % |
| Q3_K, 50 % | 27.9 % | +2.68 % ± 0.25 | 0.0537 | 91.2 % | +10.5 % | +0.1 % | +0.1 % | +0.4 % |
| Q3_K, 50 % random | 49.4 % | +3.47 % ± 0.33 | 0.0877 | 87.8 % | +8.3 % | +3.6 % | +1.2 % | +0.5 % |
| Q2_K, 10 % | 3.0 % | +0.44 % ± 0.16 | 0.0200 | 94.6 % | +1.6 % | +0.1 % | +0.1 % | −0.2 % |
| Q2_K, 25 % | 11.0 % | +3.74 % ± 0.30 | 0.0733 | 90.1 % | +14.0 % | +0.4 % | +0.5 % | +0.3 % |
| Q2_K, 50 % | 27.9 % | +9.04 % ± 0.47 | 0.1726 | 84.7 % | +33.2 % | +3.3 % | +0.7 % | +1.7 % |
| Q2_K, 25 % random | 22.9 % | +10.76 % ± 0.52 | 0.2157 | 81.4 % | +28.4 % | +8.1 % | +4.3 % | +2.6 % |
| IQ2_S, 25 % | 11.0 % | +5.97 % ± 0.38 | 0.0885 | 90.0 % | +21.6 % | +1.6 % | +0.9 % | +0.8 % |

### Reading

- **Perplexity alone hides the damage; KLD does not.** Qwen3 Q2_K at
  10–25 % reads −0.1 % perplexity, inside the noise, while KLD rises 4–7×
  over the container's floor (0.009–0.014 vs 0.002) and 3.5–4 % of the
  top-1 tokens change. The bar holds; the distribution moves anyway.
- **English prose takes the hit** in both models. The profile set's prose
  (Austen, Kafka) does not cover Moby-Dick's experts well, and #48 already
  showed English as OLMoE's worst-covered domain (22 % at K = 18.3 %).
  Q3_K at 50 % costs Qwen3 +1.1 % on English and nothing on code.
- **The flat router is the problem for OLMoE.** Its least-used 25 % still
  serve 11 % of the held-out selections (Qwen3: 2.5 %), so there are no
  experts cold enough to degrade for free.
- **2-bit without an importance matrix is too coarse.** Cold Q2_K experts
  are 30 % RMS away from the Q4_K_M weights (OLMoE block 3). IQ2_S without
  one is worse than Q2_K at the same share (+1.8 vs −0.1 %). An imatrix
  would help both; it was out of scope, and geistlib cannot run Q2_K anyway
  (no kernels on any backend).

## Speed

Qwen3-30B-A3B, prompt 512, 64 decode steps; median (range) of three
interleaved cycles, tok/s. RSS from `bench_perf_sweep`.

| backend | experts | expert bytes | prefill 512 | decode | RSS |
| :-- | :-- | --: | --: | --: | --: |
| Metal | Q4_K_M (as shipped) | 17.55 GB | **849 (837–849)** | **68.5 (65.3–72.5)** | 0.3 GB |
| Metal | all Q3_K | 12.48 GB | 220 (189–221) | 18.1 (17.8–18.6) | 0.4 GB |
| Metal | all IQ2_S | 9.29 GB | 30 (25–32) | 13.4 (12.2–13.6) | 6.7 GB |
| cpu_neon | Q4_K_M | 17.55 GB | **79 (79–138)** | **20.4 (19.3–33.0)** | 36–43 GB |
| cpu_neon | all Q3_K | 12.48 GB | 48 (46–58) | 17.1 (16.4–19.5) | 10.5 GB |
| cpu_neon | all IQ2_S | 9.29 GB | 47 (45–48) | 12.0 (11.5–12.2) | 8.5 GB |

- **Metal: no device kernels for these types.** The MoE device route and
  the grouped prefill GEMM (`metal_moe_ids_pipe`, `metal_moe_mm_pipe`) cover
  Q4_K, Q6_K, Q8_0, MXFP4, Q5_0 and F32. Q3_K and IQ2_S experts fall back
  to the host route. These rows measure the missing kernels, not the
  formats.
- **cpu_neon: the formats themselves are slower.** Both have decode and
  prefill kernels there. Q4_K_M's cpu_neon numbers are noisy: its experts
  are repacked at load (RSS 36–43 GB against 8–11 GB for the formats read
  from the mapping), and cycles 2–3 ran at 20 tok/s against 33 in cycle 1
  (#213 measured 30). Even the slowest Q4_K_M cycle (19.3) is above Q3_K's
  median and its prefill (79) above both low formats' best. Fewer bytes do
  not buy speed here: unpacking 3- and 2-bit codes costs more than the
  memory bandwidth it saves.
- **Upper bound for a mixed file, if the formats were as fast per byte as
  Q4_K:** decode reads cold experts in proportion to their selections. The
  50 % least-used experts of Qwen3 carry 11.7 % of the held-out selections,
  so Q3_K there would save 11.7 % × 29 % ≈ 3.4 % of the expert bytes per
  token. Measured, the formats are slower, so the sign of the change is
  negative.
- Not measured: cpu_x86 (the CPU branch of the RTX 2080 Ti offload) and
  Vulkan. Vulkan's MoE kernels (`matvec_{f32,q4k,q6k}_id`) do not cover
  Q3_K/IQ2_S either.
- The first Metal lease also tried cpu_neon at `--m-max 512`, which it
  rejects (`speed.err`); `speed-cpu.jsonl` is the rerun at 128.

## Offload mode (derived, not measured)

`tools/moe_cold_offload_model.py`: Qwen3-30B-A3B, a VRAM budget for experts
split evenly over the layers and filled hottest first (profile order), the
held-out selections counted against it. Expert size per layer and type from
the GGUF: 2.86 MB in Q4_K_M (Q6_K `ffn_down` in half the layers), 2.03 MB in
Q3_K, 1.51 MB in IQ2_S. For scale: geistlib's GPU mode on the RTX 2080 Ti
holds 2,807 experts (8.2 GB, 45.7 %) (`docs/OFFLOAD.md`).

| VRAM for experts | cold experts | resident | GPU share of selections | CPU expert MB / token |
| --: | :-- | --: | --: | --: |
| 6.0 GB | none (all Q4_K_M) | 34.0 % | 75.0 % | 277 |
| 6.0 GB | non-resident in Q3_K | 34.0 % | 75.0 % | 194 |
| 6.0 GB | non-resident in IQ2_S | 34.0 % | 75.0 % | 145 |
| 8.0 GB | none (all Q4_K_M) | 45.3 % | 85.0 % | 167 |
| 8.0 GB | non-resident in Q3_K | 45.3 % | 85.0 % | 116 |
| 8.0 GB | non-resident in IQ2_S | 45.3 % | 85.0 % | 87 |
| 8.0 GB | least-used 50 % in Q3_K | 45.3 % | 85.0 % | 129 |
| 9.5 GB | none (all Q4_K_M) | 53.9 % | 90.1 % | 110 |
| 9.5 GB | least-used 50 % in Q3_K | 55.5 % | 90.8 % | 71 |
| 9.5 GB | least-used 50 % in IQ2_S | 57.4 % | 91.5 % | 49 |

The full table (all schemes at every budget) is
`raw/moe-cold-requant-2026-11-09/offload_model.txt`.

- **No more experts fit.** With every expert ranked by the profile, the
  budget is spent on hot Q4_K_M experts first. A cold half in lower
  precision changes the resident set only once the budget reaches beyond
  the hot half: +1.6 points of resident experts (+0.7 points of selections)
  at 9.5 GB in Q3_K, nothing at 6 or 8 GB.
- **The CPU branch reads fewer bytes**, if every non-resident expert is in
  the low type (HOBBIT's design): at 8 GB, 167 → 116 MB per token (Q3_K) or
  87 MB (IQ2_S). But the non-resident share is then 55 % of the experts, and
  holding the bar was measured only up to 50 % in Q3_K; and on cpu_neon the
  low formats are slower per token than Q4_K_M, so fewer bytes need not
  mean a shorter branch. Whether cpu_x86 with AVX-512 runs Q3_K faster than
  Q4_K per expert would decide it; nothing measured here suggests so.

## Recommendation

**Not worth it; do not build a mixed-precision expert cache.**

- No setting both saves real work and holds the bar: the settings that hold
  it (Qwen3: Q3_K ≤ 50 %, Q2_K ≤ 25 %) touch ≤ 11.7 % of the selections,
  and the low formats are slower per token on every backend measured.
- It would also need engine work that conflicts with the loader: one GGUF
  tensor holds one type, so per-expert precision means split expert tensors
  or a second set of them, and Metal/Vulkan would need Q3_K/IQ2_S MoE
  kernels first.
- **What remains is a memory recipe, not a speed one.** Where the model does
  not fit (Qwen3-30B-A3B on a 16 GB machine), an offline GGUF with the 50 %
  least-used experts in Q3_K cuts the experts from 17.55 to about 15 GB for
  +0.4 % perplexity. That is `llama-quantize` territory (per-tensor types
  plus a per-expert split), not geistlib's. Revisit only if (a) cpu_x86 or
  a device kernel runs Q3_K/IQ2_S at least as fast per byte as Q4_K, and
  (b) an importance matrix brings 2-bit cold experts within the bar.

## Reproduce

```sh
L=$HOME/llama.cpp-cross-engine/build-metal/bin   # llama.cpp d81235049, Metal build
python3 tools/moe_usage/corpus.py /tmp/c48 $PWD ~/llama.cpp-cross-engine   # held-out texts
U=benchmark/results/raw/2026-10-15_moe_expert_usage_vulkan_run1225.txt
python3 tools/moe_cold_requant.py Qwen3-30B-A3B-Q4_K_M.gguf base.gguf --usage $U \
    --model Qwen3-30B-A3B-Q4_K_M --cold 0 --libggml $L/libggml-base.dylib
python3 tools/moe_cold_requant.py Qwen3-30B-A3B-Q4_K_M.gguf q3k50.gguf --usage $U \
    --model Qwen3-30B-A3B-Q4_K_M --cold 0.5 --cold-type Q3_K --libggml $L/libggml-base.dylib
$L/llama-perplexity -m base.gguf -f en.txt -c 512 --kl-divergence-base base-en.kld
$L/llama-perplexity -m q3k50.gguf -c 512 --kl-divergence-base base-en.kld --kl-divergence
python3 tools/moe_cold_requant_table.py <dir of per-variant logs>
```

The per-variant logs (rewrite summary plus the four domains' KL blocks) are
in `raw/moe-cold-requant-2026-11-09/{qwen,olmoe}/`.
