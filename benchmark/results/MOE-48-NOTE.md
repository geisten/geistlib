# A small Mixture-of-Experts model on cheap hardware: research note (#48)

Question (#48): can a **small** Mixture-of-Experts model run usefully on
inexpensive hardware when geistlib decides **at load time** where each
expert lives?

This note collects the answer from phase 0 (#51) to the GPU expert split
(#78 phase 3, #158–#228) and adds the measurements #48 still lacked: the
M1 Max on current main, the Raspberry Pi 5, energy, and a determinism test
for the placement. Sections separate **measurements**, **observations**,
**interpretation** and **hypotheses**. Failed attempts are listed under
negative findings.

## Answer in short

- **"Small MoE" and "load-time placement" do not meet.**
  - The small, natively trained MoEs (Granite 3.1 3B-A800M, OLMoE-1B-7B) route
    almost uniformly. Their 18.3 % hottest experts cover only 40–41 % of
    held-out selections (`MOE-EXPERT-USAGE.md`).
  - geistlib does not support those architectures (`granitemoe`, `olmoe`).
  - The small MoEs it does load (`qwen3moe`, 3–4 experts) are merges of
    dense Qwen3 models. Every expert there is hot.
- **Placement pays from about 128 experts per layer.** Qwen3-30B-A3B (128
  experts, 54 % held-out cover at 18.3 %) and Qwen3.6-35B-A3B (256 experts)
  are the models where a static hot set works. They need a GPU or ≥ 20 GB
  of RAM, not a Pi.
- **Where it works, it works well:**
  - **RTX 2080 Ti + 9950X**, Qwen3-30B-A3B with 2,800 of 6,144 experts on
    the GPU: decode 74 tok/s with GPU routing (#228). That is 2× CPU-only
    and 1.4× llama.cpp's best `--n-cpu-moe` split on Vulkan.
  - **RTX 2080 Ti + 9950X**, Qwen3.6-35B-A3B: decode 1.06–1.15× Strata,
    prefill 1.08× Strata (#167, #216).
  - **M1 Max**, everything on Metal (unified memory, so placement is "all
    on the GPU"): Qwen3-30B-A3B decodes at **75 tok/s**, 1.16× llama.cpp
    Metal on the same requests. Prefill is 0.40× llama.cpp's (below).
- **The Pi 5 (4 GB) runs a small `qwen3moe`:**
  - Huihui-MoE-1.2B-A0.6B: 13.1 tok/s decode, the same as llama.cpp's
    13.4. Prefill is 1.5× llama.cpp's.
  - A 3.0 GB 4×1.7B MoE (4 experts, 2 active) fits only by paging. Under
    a 3,000 MB cap it decodes at 4.9–5.1 tok/s against llama.cpp's
    5.4–6.1, and prefills at 1.6× llama.cpp's speed.
  - Placement plays no role on these models; there is no cold expert to
    leave out.
- **Energy (decode, tokens per joule):**
  - On the M1 Max, geist Metal needs **23–60 % less energy per token**
    than llama.cpp Metal on all three models (0.40 vs 0.54 J/token on
    Qwen3-30B-A3B). The difference is the CPU: on Qwen3-30B and the
    4×1.7B model llama.cpp's threads draw 0.16–0.18 J/token beside the
    GPU, geist's 0.02.
  - On the Pi (board power, PMIC) the engines are level on Huihui (0.33
    vs 0.32 J/token). On the paged 4×1.7B model geist takes 18 % more
    (1.05 vs 0.89 J/token).
  - Not measured on the desktop: its NVIDIA driver and library are out of
    sync until a reboot, and RAPL is root-only.

**Proposal: close #48** with this note. The decision rule was met on the
GPU side, and the CPU-side placement policies were never built, for a
measured reason (phase 0). The open criteria are listed at the end with a
verdict for each.

## Setup

- **geistlib:** main `af668940`, plus the placement test of PR #229, which
  changes no library code.
- **llama.cpp:** `d81235049` (CPU, Metal and Vulkan builds). On the Mac
  and the Pi: `llama-server` `/completion` with the same token ids as
  geist, `cache_prompt` off, `ignore_eos`. On the desktop: as in
  `moe-baseline.yml`.
- **Prompt:** the first 512 tokens of `docs/OFFLOAD.md` (real text,
  `bench_perf_sweep --prompt-file`), then 128 decoded tokens. On the Pi:
  the same with 2 repeats.
- **Locks:** Mac under `with-mac-lock` (batch). Pi under `with-pi-lock`.
  The desktop rows come from the merged PRs; this note's own desktop run
  failed (see Energy).
- **Start gates:**
  - Mac: GPU and CPU below 65 °C and CPU < 30 % busy (macmon, 3 s).
  - Pi: SoC below 60 °C and load-1m below 0.5.
- **Order:** interleaved cycles with the order rotated. The page cache is
  warmed after every model switch (the run-order trap of #164).
- **Scripts and raw data:** `raw/moe-48-2026-10-25/`:
  - `mac_moe_bench.sh`, `pi_moe_bench.sh`, `pi_lucy_probe.sh`,
    `pi_lucy_llama.sh`, `desktop_moe_energy.sh` (its run failed, see
    Energy);
  - `summarize.py` builds the tables below from the JSONL.
- **Models** (sha256 = the HF LFS hash, checked after download):

  | model | arch | experts / active | file | sha256 |
  | :-- | :-- | :-- | :-- | :-- |
  | Qwen3-30B-A3B Q4_K_M (18.6 GB) | qwen3moe | 128 / 8 | Qwen/Qwen3-30B-A3B-GGUF @ `e4d4bafd` | `0d003f66…` |
  | Huihui-MoE-1.2B-A0.6B Q8_0 (1.37 GB) | qwen3moe | 3 / 1 | mradermacher/Huihui-MoE-1.2B-A0.6B-GGUF @ `cc691036` | `fdf5e80b…` |
  | Qwen3-MOE-4x1.7B-6.8B Q4_K_M (3.0 GB) | qwen3moe | 4 / 2 | mradermacher/Qwen3-MOE-4x1.7B-6.8B-Lucys-Song-Uncensored-GGUF @ `6ec1350d` | `6ce18acd…` |

  The two small models are merges of dense Qwen3 models (0.6B and 1.7B),
  each expert a full dense FFN. They were chosen because they are the
  `qwen3moe` files that fit a 4 GB Pi, not for their quality. Their shapes
  (few, wide experts) decide the speed, as in any model of that geometry.

## Placement strategy (what was built)

| where | placement | decided by |
| :-- | :-- | :-- |
| CPU only | every expert from the mmap; the page cache keeps what the text uses | nothing to decide (phase 0: no hot set worth pinning) |
| Discrete GPU, model on the GPU (Vulkan) | the hottest experts get a VRAM copy until the free memory less a reserve is used; the rest is resolved on the CPU only (#164/#198) | `GEIST_MOE_PROFILE` (USAGE lines of `tools/moe_usage`), else the opt-in run-time profile cache (#165), else the same share of every layer; `GEIST_MOE_HOT=<n>` overrides the count |
| Unified memory (Metal), model on the GPU | every expert that fits the working set: on the M1 Max all of them, aliasing the mapping, no copy (#168/#200) | the same ranking; on 64 GB it never binds |
| Decode route | GPU ranks the router itself and runs the experts it holds; the host runs the cold ones (#162, #228; Metal: no host in the layer at all, #201/#213) | per layer at load: all routed experts on the device → device route |

**Ranking:** count descending, ties to the lower `expert × n_layers +
layer`. A shared expert (qwen35moe) comes first, because it runs for every
token. The ranking is a pure function of the profile and the model.

**Determinism (new, PR #229).** `test_moe_profile_unit` loads the fixture
three times with the same profile and `GEIST_MOE_HOT`. The profile
contains ties. A second set of three loads has no profile.
- Every load places the same experts: the ones the ranking defines.
- Every load gives the same logits, bit for bit, both chunked and token by
  token.
- This holds with the model on the GPU and with the model on the CPU and a
  GPU split backend.
- Flipping the tie-break makes the test fail.
- **Limit:** without `GEIST_MOE_HOT`, the count comes from the free device
  memory at load. Under different memory pressure two loads can place a
  different number of experts. The order within that number is still the
  ranking's.

## Measurements

### M1 Max 64 GB, Metal: Qwen3-30B-A3B Q4_K_M, current main

3 interleaved cycles; each cycle's value is the mean of 3 repeats (geist)
or 3 requests after a warmup request (llama.cpp). Shown: median (range).

| engine | prefill 512 tok/s | decode tok/s |
| :-- | --: | --: |
| **geist Metal** (every expert on the GPU, device route) | 315 (315–316) | **75.0 (74.8–76.0)** |
| llama.cpp Metal `-ngl 99` | **793 (769–803)** | 64.9 (63.0–67.1) |

- **Decode: geist is 1.16× llama.cpp.** llama.cpp's first request of each
  cycle decodes at 75.7–78.0 tok/s. Its next two fall to 55.9–64.1 (cause
  not traced). geist's repeats stay within 2 % of each other.
  - Against llama.cpp's first requests alone, the engines are level
    (75.0 vs 76.8).
  - #213 measured 0.92× on 2026-10-23 with one request per run.
- **Prefill: geist is 0.40× llama.cpp.** #213 measured 218 tok/s, this
  run 315. This run prefills the 512 tokens as one chunk (`--m-max 512`);
  whether #213 did, and which change accounts for the rest, was not
  checked. Prefill still routes on the host: there is no device route for
  prefill chunks yet (#213, "not covered: prefill").
- geist cpu_neon (from #213, not re-measured: its RSS of 43 GB does not
  fit beside the Mac's other users today): 113 tok/s prefill, 30 tok/s
  decode.

### M1 Max: the small MoEs

| model | engine | prefill 512 tok/s | decode tok/s |
| :-- | :-- | --: | --: |
| Huihui 1.2B-A0.6B Q8_0 | **geist Metal** | 4,150 (4,116–4,209) | **186.7 (184.2–187.3)** |
| | llama.cpp Metal | **4,448 (4,422–4,471)** | 167.2 (166.9–171.2) |
| | geist cpu_neon | 325 (325–328) | **90.6 (88.6–93.1)** |
| | llama.cpp CPU | **800 (751–806)** | 88.4 (85.6–93.4) |
| 4×1.7B Q4_K_M | **geist Metal** | 1,018 (1,013–1,027) | **98.5 (96.4–101.0)** |
| | llama.cpp Metal | **1,029 (1,022–1,060)** | 83.6 (79.5–90.5) |
| | geist cpu_neon | 97.6 (96.0–97.8) | 22.8 (22.5–23.0) |
| | llama.cpp CPU | **129 (114–130)** | **34.1 (24.1–40.9)** |

- **Metal:** geist decodes 1.12–1.18× faster than llama.cpp and prefills
  at 0.93–0.99×.
- **cpu_neon:**
  - Huihui decode is level with llama.cpp.
  - The 4×1.7B model decodes at 0.67× llama.cpp, whose spread is wide.
  - Prefill is behind on both: 0.41× on Q8_0 and 0.76× on Q4_K/Q6_K.
    cpu_neon caps a chunk at 128 rows; llama.cpp ran 512.
- **Observation:** the 4×1.7B model has an RSS of 8 GB on cpu_neon for a
  3.0 GB file (`rss_mb`, Mac). The Q4_K experts are repacked for the NEON
  kernels while the file pages stay mapped, the same pattern x86 had
  before `GEIST_W_RAW_COLD` (#54); likely, not traced. On the Pi the same
  model ran under a 3,000 MB cap (below), at 2.9 GB max RSS: the
  clean file pages give way there.

### Raspberry Pi 5 (4 GB, Cortex-A76, NVMe), cpu_neon vs llama.cpp CPU

What fits: not Qwen3-30B-A3B (18.6 GB). Paged from the NVMe it would read
most of its ~3B active parameters, about 1.7 GB at Q4_K_M, from storage
per token. That is an estimate, not run. No Qwen3.5/3.6 MoE fits either. The `qwen3moe` files that fit are the
merges above.

| model | engine | prefill 512 tok/s | decode tok/s | max RSS |
| :-- | :-- | --: | --: | --: |
| Huihui 1.2B-A0.6B Q8_0 | **geist cpu_neon** | **74.7 (74.1–74.8)** | 13.13 (13.10–13.15) | 1,282 MB |
| | llama.cpp CPU, 4 threads | 50.2 (50.0–50.6) | **13.38 (13.31–13.39)** | |
| 4×1.7B Q4_K_M, 3,000 MB cap | **geist cpu_neon** | **17.0 (16.9–17.1)** | 5.12 (4.94–5.30) | 3,005 MB (16,528 major faults) |
| | llama.cpp CPU | – (server dropped the 512-token request, see below) | 5.44 (32-token prompt, energy run) | |
| same, synthetic tokens (`llama-bench` shape, 2 reps) | geist cpu_neon | **16.3** | 4.89 | 3,019 MB (12,687 major faults) |
| | llama.cpp `llama-bench` | 10.1 | **6.06** | 3,312 MB (120,858 major faults) |

- 3 cycles (Huihui) and 2 cycles (4×1.7B). The board started every run
  at 49–56 °C with load-1m < 0.5.
- **The 4×1.7B model (3.0 GB) fits the 4 GB Pi only by paging**: under a
  3,000 MB cap without swap (a systemd scope, so the board's other
  services stay safe) both engines run, at about 5–6 tok/s decode.
  - geist prefills 1.6× llama.cpp's speed and decodes at 0.81× (synthetic
    tokens; 0.94× against the energy run's real-text decode).
  - llama.cpp takes about 10× geist's major faults.
- **llama-server under the cap:** in both throughput cycles of the
  interleaved run it came up, then dropped the connection during the
  512-token request (urllib error after 17–23 s). With 32-token prompts
  (the energy phase) it ran. Presumably the prompt's buffers pushed it
  past the cap; not traced. The like-for-like
  row is therefore `llama-bench` against geist on the same synthetic
  shape.
- Fit probe for the 4×1.7B model (`pi_lucy_probe.sh`: 3,000 MB cap, no
  swap, 32-token prompt, 16 decoded):
  - geist: decode 5.35 tok/s, prefill 13.1 tok/s, max RSS 2.94 GB, 16,129
    major faults.
  - llama.cpp: max RSS 3.35 GB, 64,088 major faults. Its tok/s were cut
    off in that log and are in the table instead.

### Energy per decoded token

Method: two runs per engine, with 128 and 640 decoded tokens after a
32-token prompt (Pi: 64 and 320). Energy per decoded token is
(E_long − E_short) / (n_long − n_short). Load, prefill and warmup cancel.
The power sources:

| host | source | covers | what is missing |
| :-- | :-- | :-- | :-- |
| M1 Max | `macmon` (no sudo), 100 ms | CPU + GPU + ANE (`all_power`) | DRAM (`ram_power` reads 0 on this machine), SoC rest, display; other users of the Mac (idle 0.2–4.6 W before the runs, not subtracted) |
| Pi 5 | `vcgencmd pmic_read_adc`, Σ I·V over the PMIC rails, ~0.2 s | the board's rails (internal telemetry, not a wall meter) | the PSU, the NVMe, USB |
| RTX 2080 Ti box | `nvidia-smi power.draw`, 100 ms | the card | the CPU package: RAPL (`/sys/class/powercap/intel-rapl:0/energy_uj`) is root-only (0400) since CVE-2020-8694, so the host's share is **not measured** |

| host | model | engine | decode tok/s | J / token | tokens / J |
| :-- | :-- | :-- | --: | --: | --: |
| M1 Max | Qwen3-30B-A3B | **geist Metal** | 75.7 | **0.40** (CPU 0.02, GPU 0.39) | **2.50** |
| | | llama.cpp Metal | 68.9 | 0.54 (CPU 0.18, GPU 0.35) | 1.87 |
| M1 Max | Huihui 1.2B-A0.6B | **geist Metal** | 191.3 | **0.074** | **13.6** |
| | | llama.cpp Metal | 168.7 | 0.186 | 5.4 |
| | | geist cpu_neon | 104.5 | 0.289 | 3.5 |
| | | llama.cpp CPU | 77.7 | 0.308 | 3.2 |
| M1 Max | 4×1.7B | **geist Metal** | 101.0 | **0.353** | **2.83** |
| | | llama.cpp Metal | 85.8 | 0.458 | 2.18 |
| | | geist cpu_neon | 23.7 | 1.355 | 0.74 |
| | | llama.cpp CPU | 51.7 | 0.540 | 1.85 |
| Pi 5 | Huihui 1.2B-A0.6B | geist cpu_neon | 16.3 | 0.330 | 3.03 |
| | | llama.cpp CPU | 13.4 | 0.316 | 3.17 |
| Pi 5 | 4×1.7B, 3,000 MB cap | geist cpu_neon | 5.45 | 1.05 | 0.95 |
| | | llama.cpp CPU | 5.44 | 0.89 | 1.13 |

- **Observation (M1 Max):** the GPU's energy per token is about the same
  in both engines (0.35–0.39 J). The difference is the CPU: llama.cpp
  draws 0.16–0.18 J per token beside the GPU on Qwen3-30B and the 4×1.7B
  model, geist 0.02.
- **Hypothesis, not checked:** llama.cpp's CPU threads poll while the GPU
  works. geist's host only encodes.
- On the CPU alone, energy follows speed.
- **On the Pi:**
  - Huihui: at this short context geist decodes faster (16.3 vs 13.4
    tok/s), and both draw about the same per token.
  - 4×1.7B: at the same decode speed, geist takes 18 % more energy per
    token (1.05 vs 0.89 J). Not explained. llama.cpp has the larger page
    fault count, so faults alone do not account for it. One pair of runs,
    not repeated.
- **RTX 2080 Ti box: not measured.** The run was queued under both locks.
  When it ran (2026-10-25 ~09:00), `nvidia-smi` reported "Failed to
  initialize NVML: Driver/library version mismatch" (NVML library
  595.99 against a loaded kernel module of another version): a driver
  update without a reboot. Both engines came out far below every earlier
  run (decode: geist 6.9, llama.cpp 12.1 tok/s), and the power samples
  were empty. The numbers were discarded.
  The card needs a reboot before any GPU measurement on that box.

### Desktop (RTX 2080 Ti + 9950X), from the PRs: Qwen3-30B-A3B and Qwen3.6-35B-A3B

These rows are reported in `docs/OFFLOAD.md`. Real text, 512 + 128/512
tokens.

| | decode tok/s | prefill tok/s | source |
| :-- | --: | --: | :-- |
| Qwen3-30B geist cpu_x86 | 34.8–36.0 | 236–252 | runs 1740–1743 |
| Qwen3-30B llama.cpp Vulkan `--n-cpu-moe 24` (best that fits) | 52.7 | 491 | run 1741 |
| Qwen3-30B **geist GPU mode, #48 profile, GPU routes** | **74.0 / 73.6 / 74.4** | 749–776 (after #216) | #228; #164 re-measure |
| Qwen3.6-35B Strata `5c8dd322` CUDA, MTP off | 62.2 | 605 | run 1930 |
| Qwen3.6-35B **geist GPU mode, profile** | **71.3** (86.8–87.8 with #228, 512 tokens) | **655** | run 1930, #228 |

## Interpretation

1. **The skew of the router decides whether load-time placement can pay.**
   - With 3–64 experts, every expert is warm. Pinning a subset saves
     nothing, and the page cache already holds what the text uses.
   - From 128 experts on, a profile-ranked set of 38–46 % of the experts
     covers 79–88 % of the selections (#197, #158). Static placement then
     carries most of the decode traffic. That is Strata's static profile
     cache, decided once at load.
2. **On a discrete GPU, the placement plus a device-side route beats
   llama.cpp Vulkan and Strata in decode and is level with llama.cpp CUDA
   (run 1839).** A runtime-adaptive cache added nothing
   on top (#166, below).
3. **On unified memory the placement question disappears.** Every expert
   fits, and the gain comes from keeping the host out of the layer (#213:
   20 → 72 tok/s). Prefill lags because chunks still route on the host.
4. **"Cheap hardware" in #48's sense, the 4 GB Pi, runs only small merged
   MoEs.** On those, geist decodes level with llama.cpp or up to 19 %
   behind it (the paged 4×1.7B model) and prefills 1.5–1.6× faster.
   Expert placement has no role there beyond what the page cache does.

## Negative findings (measured, not merged)

- **Phase 0 stop rule (#51).** Granite 3B and OLMoE route near-uniformly
  (normalised entropy 0.92–0.95). Hot-resident + cold-paged CPU policies
  were therefore not built, so the criterion "every placement policy is
  byte-identical to all-resident" has no CPU policies to test.
- **Adaptive expert cache (#166):** built on `feat/166-moe-adapt`, LRU
  swaps between 64-token windows, balance-aware.
  - Qwen3-30B: +3 % quiet (within noise), −2 % busy.
  - Qwen3.6: −2 % quiet, −1 % busy.
  - The decode layer is bound by the GPU queue. Moving experts onto the
    GPU costs GPU time the per-branch probe does not see.
  - Closed as measured.
- **Helper pool for cold experts (#214, reverted in #217):** about 15×
  faster on a busy host (2 → 40 tok/s), but 5–20 % slower on a quiet one
  in every variant (2–7 helpers, spin 0–200 ms). #222's thread regime
  fixed the busy-host collapse without it.
- **Run-order trap (#164):** a "regression" of 4–6 % mean prefill and a
  worst repeat up to +55 %, first blamed on dropping the hot experts' CPU
  copies and then on transparent huge pages. Rotating the order showed the
  real cause: whichever variant ran first after a model switch re-read
  evicted file pages (up to 612k major faults; the desktop's 59 GB cannot
  cache two MoE files plus process memory).
  - #198 costs nothing and saves 4–5 GiB.
  - Method since then: warm up after a switch and rotate the order. This
    note does both.
- **Route per linear for offload (#78):** no gain, because every layer
  streamed while the CPU waited (run 1383). It was replaced by the per-step
  route.
- **Importing the GGUF mapping into Vulkan** (`VK_EXT_external_memory_host`):
  the NVIDIA 595.91 driver refuses it, and afterwards the process's other
  host allocations fail too.

## Hypotheses (untested)

- **A device-side prefill route on Metal** (the #201 ops for chunks) would
  close most of the 0.40× prefill gap on Qwen3-30B. The decode route made
  the same change for decode (3.5×).
- **Dropping the file pages of repacked Q4_K experts on cpu_neon** (as
  `GEIST_W_RAW_COLD` does on x86) would cut the 4×1.7B model's RSS from
  ~8 GB toward ~3 GB on the Mac.
- **A fine-grained MoE small enough for 8 GB** (≥ 128 experts, ≤ 6 GB at
  4 bit) would be the real Pi target for load-time placement. None in a
  supported architecture was found.

## #48 acceptance criteria: verdict

| criterion | state |
| :-- | :-- |
| Logits vs llama.cpp within the declared tolerance | met on cpu_neon (ticked); cpu_x86's single-position JS traced to its INT8 KV default, per-family F16 KV since then (`test_qwen3moe_unit`: "F16 KV cache under AUTO") |
| Greedy 256 tokens to the first near-tie | reported: first divergence is a router near-tie (margin 0.007), not a logit tie — the owner's call |
| Placement policies byte-identical to all-resident | CPU policies not built (phase 0 stop rule); GPU placement changes arithmetic by design (perplexity +0.4–0.9 % against the CPU) |
| Load refuses unknown router / missing tensor / bad profile | router and tensor refusals tested; a profile for another model is ignored with one stderr line (#165 cache key, `test_moe_profile_unit`); `GEIST_MOE_PROFILE` lines past the model's layers are skipped, not refused |
| Other families byte-identical | checked in #54 (local + CI logit goldens) |
| No STABLE API change; options EXPERIMENTAL and documented | `GEIST_MOE_*` are environment knobs in `docs/BACKENDS.md`, `GEIST_W_EXACT` EXPERIMENTAL in `geist_weight.h`; no STABLE symbol changed |
| **Placement deterministic** | **met: PR #229** |
| **Pi 5 and M1 Max vs llama.cpp** | **this note** (Qwen3-30B does not fit the 4 GB Pi; small merged MoEs do) |
| Per-policy decode/prefill/RSS/page-in/hot-set hit rate on held-out text | GPU policies in OFFLOAD.md and #197/#215 (held-out cover 81.3 %); CPU policies not built |
| **Energy per decoded token on the Pi (PMIC)** | **this note**; Mac (macmon) too; desktop: GPU run lost to a driver/library mismatch, CPU (RAPL) not readable without root |
| No regression for existing models (`pi5-ab` on BitNet) | not run as `pi5-ab`; dense-model logits are guarded on every merge by CI's logit goldens, and the MoE code lives in its own units (`families/qwen3moe.c`, `forward/moe.c`, `moe_gpu.c`) |
| Model URL/SHA, commits, compiler and flags | model URLs and SHAs above and in `moe-baseline.yml`; geist `af668940`, llama.cpp `d81235049`; compilers: Apple clang 21.0.0 (Mac, `BACKENDS="metal cpu_neon cpu_scalar"`), gcc 14.2.0 (Pi, `TARGET=pi5 GEMM_PROVIDER=native`), gcc 15.2.0 (desktop build, its run lost; `TARGET=linux BACKENDS="vulkan cpu_x86 cpu_scalar" GEMM_PROVIDER=native`); all `-std=c23`, release flags of `mk/` |
| Results in `benchmark/` and a research note | this file + `raw/moe-48-2026-10-25/` |
