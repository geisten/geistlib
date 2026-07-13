# Vulkan backend — baseline & targets (RTX 2080 Ti)

Phase-0 snapshot (2026-07-04): the numbers to beat, measured before any geist
Vulkan code exists. Protocol: pp512 + tg128, E2E total = 640 tok / (t_pp + t_tg).

## Rig

- GPU: NVIDIA RTX 2080 Ti 11 GB (Turing, sm_75), driver 595.71.05, Vulkan 1.4
  - reports: fp16 ✓, KHR_coopmat ✓, warp 32 (ggml prints `int dot: 0` — its
    accelerated-int-dot probe fails on this driver; DP4A still worth testing in ours)
- CPU: AMD Ryzen 9 9950X (16C/32T) — CPU runs use `OMP_NUM_THREADS=16`
  (SMT costs ~40% decode; 32-thread runs are invalid)
- llama.cpp pinned `d0f9d2e` (2026-06-22):
  - `build-vulkan/`: `-DGGML_VULKAN=ON` (needs SPIRV-Headers; installed to `~/.local`)
  - `build-cuda/`: `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75` (reference only)
- llama-bench: `-p 512 -n 128 -r 8`, `GGML_VK_VISIBLE_DEVICES=0` (else it also
  enumerates the Raphael iGPU)
- geist @ main 5ae89d5, `make BACKENDS="cpu_x86 cpu_scalar"`, bench via
  `tests/bench_session_throughput`

## Gemma 4 E2B-it Q4_K_M (2.88 GiB) — the H2H

| engine | pp512 tok/s | tg128 tok/s | E2E tok/s |
| :-- | --: | --: | --: |
| llama.cpp Vulkan (the bar) | 4682.6 ± 39.7 | 152.6 ± 1.2 | 675 |
| llama.cpp Vulkan (re-run 2026-07-06) | 4639.0 ± 34.5 | 154.0 ± 0.3 | 680 |
| llama.cpp CUDA (reference) | 6697.3 ± 85.7 | 181.8 ± 1.3 | 820 |
| geist cpu_x86 (16T) | 484.2 | 42.2 | 156 |
| geist vulkan @ 3c-14 (2026-07-06, median of 5) | 1150 | 132.3 | 452 |
| **geist vulkan — target ≥ 1.10×** | | | **≥ 743** |

Status 2026-07-06: tg at 86% of the llama.cpp bar, pp at 25%, E2E at 66%.
Decode is within striking distance; the remaining E2E gap is mostly the
prefill GEMMs (mm_q4k_cm/mm_cm32/mm_q6k_cm ~0.9 ms of the ~0.85 s wall) and
prefill attention. Short-context decode now equals long-context (131.7 at
pp64) — attention no longer scales the token cost through 640 positions.

## Quality gate (2026-07-06, build @ 3c-14)

Same build, same 200 shuffled MMLU questions (5-shot, seed-fixed), via
`GEIST_BACKEND=<b> tools/eval_mmlu.py`:

| backend | MMLU-200 | tooling (JSON / func) |
| :-- | --: | :-- |
| cpu_x86 | 0.490 (98/200) | — |
| vulkan | 0.520 (104/200) | 14/14 valid+schema / 14/14 fully correct |

Vulkan ≥ CPU within binomial noise (σ ≈ 7 questions at n=200) — the f16/
coopmat numeric drift (greedy decode diverges from CPU after ~20 tokens)
costs no task accuracy. Caveat discovered this round: `"auto"` backend
resolution ignored the GPU entirely (registry[0] = cpu_x86), so every CLI
"parity" check before the GEIST_BACKEND env fix compared CPU against CPU —
end-to-end GPU verification starts at commit 596c5d2.

Decode dominates E2E (~88% of wall time). At pp parity the target needs
tg ≥ 170 tok/s — i.e. ~85% of the 2080 Ti's 616 GB/s reading a 3.1 GB model
per token. llama.cpp Vulkan sits at ~50% bandwidth efficiency; that's the room.

## BitNet b1.58 2B-4T I2_S — PR-2 target

llama.cpp cannot run I2_S at all (no ggml type, no Vulkan ternary kernels),
so the bar is geist's own CPU path + bitnet.cpp:

| engine | pp512 tok/s | tg128 tok/s | E2E tok/s |
| :-- | --: | --: | --: |
| geist cpu_x86 (16T, the bar) | 909.3 | 71.1 | 271 |
| bitnet.cpp (prior H2H, same rig) | ~679 | ~56.5 | — |

## Reproduce

```sh
GGML_VK_VISIBLE_DEVICES=0 ~/llama.cpp/build-vulkan/bin/llama-bench \
  -m ~/models/gemma/gemma-4-E2B-it-Q4_K_M.gguf -p 512 -n 128 -r 8
OMP_NUM_THREADS=16 GEIST_GGUF_PATH=~/models/gemma/gemma-4-E2B-it-Q4_K_M.gguf \
  GEIST_BENCH_BACKEND=cpu_x86 GEIST_BENCH_PP=512 GEIST_BENCH_TG=128 \
  bin/linux/release/tests/bench_session_throughput
```

## Progress log

| stage | pp512 | tg128 | E2E | notes |
| :-- | --: | --: | --: | :-- |
| Phase 2 — correctness (sync per-linear dispatch) | 27.3 | 2.2 | 8.2 | Q4_K/Q6_K/F32 linears on GPU, parity bit-exact; everything else CPU; one submit+fence per linear |
| Phase 3a — batched sequence (1 flush/token) | 39.3 | 4.2 | 14.8 | full per-token op batch on GPU (linear_t, level-2 shaders, GPU argmax, device KV, x ring); ~590 dispatches/token in one submit. Next: matvec/attention shader efficiency + tiled prefill GEMM |
| Phase 3b-1 — flash attention, warp matvecs, BAR pool | 47.2 | 49.5 | 47.6 | attention 845→39 us (online-softmax, parallel positions); q4k/q6k matvec 8 rows/wg; scratch pool in BAR (GPU ops at VRAM speed, x-ring copies gone). GPU profiler (GEIST_VK_PROFILE=1). matvec still ~40 GB/s of 616 — next: llama.cpp-derived matvec + tiled prefill GEMM |
| Phase 3b-2 — register-tiled GEMM + vec4 everywhere | 330.4 | 69.8 | 189.2 | matmul q4k/q6k: 4x16 register tile (weights read once per 16 batch rows) + vec4 x loads; matvecs vec4. matmul 6474→~1000 us, matvec 59→25 us |
| Phase 3b-3 — f32 GEMM tile + fused ffn_gate_up | 379.7 | 71.4 | 203.8 | PLE f32 GEMMs tiled; decode FFN front fused to one dispatch. Decode profile: matvecs 2.3ms + gate_up 2.1 + rmsnorm 1.2 + attention 1.1 + lm_head 0.7 per token |
| Phase 3b-4 — qkv_prep fusion + 8x32 GEMM tile | 384.3 | 72.7 | 206.9 | per-head norms+rope+KV-append in one dispatch; q4k GEMM weight amplification halved again. Remaining: coopmat GEMM for prefill (~10x gap), DP4A matvecs + more fusion for decode (~2.3x gap) |
| Phase 3c-1 — tensor-core GEMMs (KHR_coopmat) | 819.6 | 72.0 | 266.4 | q4k+q6k prefill GEMMs on tensor cores: 64x64 tile, BK=32 aligned to sub-blocks, dequant->f16 shared, f32 accumulate, column-major store. Gotcha: spirv-opt (-O) miscompiles coopmat kernels to silent zeros — *_cm.comp build without -O |
| Phase 3c-2 — llama-structure matvec, probes | ~780 | ~70 | ~256 | matvec_q4k re-ported to llama.cpp's 32-thread/4-row structure (x loads amortized over rows) — measured EQUAL to the warp kernel in our harness; limiter is elsewhere. Quantified: inter-dispatch barriers cost ~21% decode / ~35% prefill (GEIST_VK_NO_BARRIER probe) → dependency-aware barrier tracking is the next lever. f16-acc coopmat: no gain on this driver (tested, reverted) |
| Phase 3c-3 — dependency-aware barriers | 819.2 | 70.7 | 262.6 | hazard tracking over byte ranges elides barriers between independent dispatches (q/k/v, kv appends, ring copies). Only ~12% elide — the decode chain is genuinely serial; the NO_BARRIER delta mostly reflects unrecoverable true-dependency drains |
| Phase 3c-4 — rolling submits, dset cache, llama-structure gate_up | 856.5 | 81.8 | 295.9 | rolling submission every 64 dispatches keeps the GPU at P0/99% during decode (governor stayed at P8 with one-submit-per-token); descriptor sets cached (98% hit); ffn_gate_up rebuilt in the 32-thread/4-row structure with 8-way x reuse (104 -> 48 us). Clock-clean decode profile: gate_up 1.6ms + q4k mv 2.3 + q6k/lm_head 1.25 + norms 2.0 + attention 1.1 per token |
| Phase 3c-5 — Q6_K 216-byte GPU repack | 801.3 | 83.2 | 294.0 | Q6_K GPU copies repacked to 4-byte-aligned 216-B blocks at resolve: word loads replace byte assembly (matvec 54->35 us); short-context decode 124.7 tok/s — tg128 is now attention/PLE context-scaling bound |
| Phase 3c-6 — f16 KV + flash-decoding | 870.8 | 104.2 | 352.3 | kv_append_f16 slot enables the F16 cache (auto-upgrade); f16 attention/qkv_prep variants; decode attention was 8 workgroups total (60 SMs idle) — split-K flash-decoding (128-pos chunks x heads partials in the x-ring + combine pass) fixes the context scaling |
| Phase 3c-7 — GEMM BK=64, x_stage in BAR | ~827 | ~104 | ~345-352 | coopmat k-loop widened to BK=64 (fewer barriers, 228->204 us); sync-path x staging BAR-resident. f16-acc re-tested under clean clocks: 4% only, driver runs both rates equal — reverted (overflow risk). Remaining: GEMM tile engineering (llama-class ~40-60 us/call), DP4A decode, norm-fusions |
| Phase 3c-8 — GEMM diagnosis round | 864.4 | 106.6 | 356.9 | double-buffered k-loop (+2%), 32-row small-N variant (+/-0), m_max>64 unlocked for vulkan but BAR (246 MB) can't hold the bigger pools — chunk growth is a dead end here. Verdict: kernel is coopMatLoad-bound (MMA:load 0.8 vs llama ~4) -> next design: 32x64-per-subgroup register tiles (BM 128). Gotcha #2 filed: the -O exclusion glob missed _cm32 -> silent zero kernel again |
| Phase 3c-9 — register-tile redesign FALSIFIED | ~815 | ~104 | ~340 | built the scoped BM=128 32x64-per-subgroup tile (2 A-frags x 4 B-frags = 8 MMAs / 6 loads, MMA:load 1.33). Bit-exact but prefill LOST ~5% (813 -> 767). The MMA:load arithmetic was the wrong model: BM=128 halves the workgroup count and pushes shared to ~30 KB -> 1 workgroup/SM instead of 2, so the dequant-staging latency stops hiding. This GPU is occupancy/latency-hiding bound at these shapes, not instruction-mix bound. Reverted. Next lever is elsewhere (DP4A decode, norm fusions) — the coopmat tile is at its local optimum on Turing |
| Phase 3c-10 — vulkan ple_block (fused PLE gate) | ~815 | 110.4 | ~355 | first backend to implement the arch's ple_block hook: gate GEMV + gelu*ple_in fused into one dispatch (7.1 us vs 14.1 for the matvec_f32 + gelu_mul pair), proj tail stays matvec + rmsnorm_add — 3 dispatches/layer instead of 4-5. Decode 9.55 -> 9.06 ms/tok. Negative result folded in: a single-workgroup proj GEMV + rmsnorm + add fusion measured 68 us vs 19 for the pair — one SM streaming the 1.5 MB proj weight is a bandwidth wall; full-vector-reduction fusions are only worth it when the weight read is small. Decode dispatch profile now: norms 2.6 ms + q4k mv 1.8 + attention 1.7 + gate_up 1.6 — next: fold standalone rmsnorms into their consumer GEMVs (lever 2), fuse q/k/v matvecs (lever 3) |
| Phase 3c-11 — norm-fold round: FFN kept, QKV falsified | ~800 | 111.4 | ~355 | ffn_norm_gate_up: pre-FFN rmsnorm folded into the gate/up matvec x-loads (each 32-thread WG recomputes inv-RMS, subgroupAdd) — measured +1.3 tok/s decode via env A/B, kept. attn_norm_qkv (norm + q/k/v in one dispatch) built, debugged and DELETED: first version lost 3 tok/s to a per-access 3-way weight-buffer select in the k-loop (hoisting the segment branch out recovered it — lesson: runtime buffer selects serialize the stream, resolve them per-loop, the ffn kernel's [[unroll]]-resolved `which` select is free); the hoisted version measured exactly NEUTRAL at PP64 and PP512 — dispatch-count reduction stopped paying after 3c-10, the decode chain is now bound by real per-kernel serial latency, not launch count. Broader lesson: fusion pays when it amortizes the prologue over >1 weight stream (ffn: 2 streams) or removes real work, not when it just concatenates dispatches |
| Phase 3c-12 — vec4 attention loads | ~1050 | 129.5 | ~432 | attn_part_f16 and attention_f16 streamed K/V/Q as SCALAR f16 loads — the whole kernel cost (attn_part 58.5 us vs a ~2 us bandwidth floor). Rewritten as f16vec4/vec4 streams (host guards offsets % 4): decode 111.4 -> 129.5 (+16%), prefill 860 -> ~1050 (+23%, attention_f16 was ~35% of prefill wall at 720 us/call), E2E 355 -> 432. Largest single win since tensor cores. Lesson: audit every kernel for scalar loads on its hot stream — the 3c-6 flash-decoding redesign fixed parallelism but silently kept the scalar loads |
| Phase 3c-13 — reduction/load micro-pass | ~1050 | 130.5 | ~438 | rmsnorm/rmsnorm_add: 14-barrier tree -> subgroupAdd two-level (+1 tok/s; the 8 us/call is dispatch-latency floor, not reduction cost). Q4_K header as one uvec4 (was 4 word loads x 16 redundant lanes), Q6_K ql/qh/d/sc as uvec2 pairs (7 -> 4 loads/lane/block): per-op times UNCHANGED — L1 absorbs the header redundancy, the matvecs are not load-issue-bound; kept anyway (fewer instructions, simpler). Remaining decode walls per token: ffn_gate_up 1.8 ms (51 us vs ~27 floor), q4k mv 1.8 ms, norms 1.35 ms (35 x 8 us dispatch floor), attention 0.95, q6k/lm_head 0.83 at ~33% BW. Next levers need occupancy/latency analysis per kernel, not more load tricks |
| Phase 3c-14 — 64-thread matvec workgroups | ~1050 | 133.2 | ~440 | Turing caps residency at 16 blocks/SM, so the 32-thread matvec workgroups topped out at 16 of 32 warps. matvec_q4k and ffn_norm_gate_up rebuilt as 2 warps x 4 rows (gx = n_out/8): gate_up 51 -> 45.8 us (its 768-1536 workgroups actually hit the block cap), small matvecs unchanged (256-512 workgroups never reach 16/SM — occupancy only binds at scale). Decode 130.5 -> 133.2 |
| Benchmark round (2026-07-06) | 1150 | 132.3 | 452 | full quality + perf pass, medians of 5 under warm clocks; llama.cpp re-measured same day (pp 4639 / tg 154 / E2E 680 — geist at 25% / 86% / 66% of the bar). MMLU-200 vulkan 0.520 vs cpu_x86 0.490 same build (binomial noise, sigma ~7 questions; gate passed), tooling suite 14/14 JSON + 14/14 function-calling. Also fixed: "auto" backend resolution ignored the GPU — CLI parity checks before 596c5d2 silently compared CPU vs CPU; GEIST_BACKEND env now overrides auto |
| Phase 3c-15 — cm32 32x32 tile + ping-pong | ~1100 (first-run 1197) | 132.8 | ~445 | the narrow-shape GEMMs (o/down/q at 1536-2048, ~1 workgroup/SM) were both starved AND serial: BN 64->32 doubles the workgroup count (215 -> 180 us), double-buffered staging hides the dequant behind the MMAs (-> ~148-180 us, run-dependent). Same two levers applied to the WIDE tiles measured negative/neutral and were reverted: BN=32 on the 64x64 tile LOSES (halves A-fragment reuse; 96-192 workgroups were not starved), ping-pong on q6k_cm/lm_head is NEUTRAL (4096 workgroups hide staging by themselves). Rule extracted: batch-tiling and ping-pong pay if and only if workgroups/SM ~ 1 — check the dispatch grid before touching the kernel. Tooling quality re-smoked 14/14 + 14/14 |
| Phase 3c-16 — DP4A matvec FALSIFIED | ~1090 | 129.1 | ~445 | int8-dot Q4_K decode matvec built, parity-verified and REVERTED (in history at `feat(vulkan): DP4A int8-dot Q4_K decode matvec`): per-4-chunk in-register x quant + dotPacked4x8EXT on the masked nibble words removed all unpack8/int->float converts (~2x less inner-loop ALU) — measured EXACTLY NEUTRAL, matvec_q4k 17.1 -> 17.0 us/call, tg128 129.0 -> 129.1 (alternating A/B x5; ~129 both sides this session vs 132.8 at 3c-15 — colder clocks, the A/B is same-session). Halving ALU at unchanged runtime proves the decode matvecs are memory-latency-bound at their occupancy, not instruction-bound; q6k/gate_up share the structure, so DP4A has no room anywhere in decode on this GPU. Toolchain findings banked: the 2080 Ti driver DOES report integerDotProduct4x8BitPackedSignedAccelerated=true — llama-bench's "int dot: 0" is its glslc build probe failing (distro glslc/shaderc predate GL_EXT_integer_dot_product; glslang >= 16 required), i.e. llama.cpp also runs WITHOUT DP4A on this rig. Remaining decode levers are latency/occupancy-class: the 8 us norm dispatch floor (~1.35 ms/tok), lm_head/q6k at ~33% BW, KV quantization for attention |
