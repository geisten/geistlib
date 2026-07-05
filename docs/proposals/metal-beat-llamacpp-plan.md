# Metal on main: closing the gap to the old engine — and passing llama.cpp

State (cool, M1 Max, gemma-E2B Q4_K_M, token parity with cpu_scalar at every step):

| | port start | now (8f0d4de) | old engine (wip) | llama.cpp Metal |
|---|---|---|---|---|
| prefill pp512 t/s | 188 | **731** | 1072 | 1546 |
| decode tg64 t/s | 3.7 | **32** | 61 (84 short-ctx) | 94 |

## 1. Why the port is slower than the old engine (measured, not guessed)

The wip engine was **GPU-first**: its arch layer was co-designed with the metal
backend. main's engine is CPU-first and decomposes the graph. Every delta below
was measured this session:

| wip engine | main engine (port) | cost today |
|---|---|---|
| f16 KV cache, K/V appended once (converting append kernel) | f32 KV cache; flash path re-converts the **whole** K/V cache per layer per chunk | conversion is cheap (~1% of savings) but K/V reads are 2× (f32) on the scalar decode path; O(kv²) conversion growth at long ctx |
| fused qnorm+rope+flash attention (1 dispatch) | rmsnorm + rope + attention (3+ dispatches, separate passes over q/k) | extra dispatches + memory passes |
| fused PLE block (1 op) | gate-linear + gelu + mul + proj-linear + rmsnorm + add (6 ops) | ~6× dispatches on 11% of FLOPs |
| device greedy head: head GEMV + argmax on GPU, 4-byte token readback | head GEMV on GPU, then **host argmax over 262 144 logits** + host sampler | forced flush + host scan per token |
| ONE command sequence per decode step (≈450 dispatches, zero mid-step host work) | ≈910 dispatches + host phases (embed dequant, PLE table dequant) → 2-3 flushes/token | decode: 10.6 ms GPU + ~20 ms host/wait tail |
| decode replay (record once, replay per token — near-zero encode) | re-encode every token | ~0.5 ms/token encode + objc overhead |

Decode is the bigger relative gap: **the GPU work is already fine (10.6 ms/tok);
~20 ms/tok is host tail + serialization**. Prefill's remaining ~340 ms over wip
is the decomposed elementwise/PLE graph + warm-state q4k (262 ms warm vs ~190
cool-capable) + per-layer cache re-conversion.

## 2. Phase plan to close the gap (each step gated on token parity + cool A/B)

### Phase A — Decode: 32 → ~55-65 t/s (recover wip level)
1. **Device greedy head** (biggest single lever): optional vtbl slot
   `greedy_head` (same additive pattern as linear_t/buffer_copy/scale_f32).
   Metal impl: head GEMV via existing matvec + re-add the small argmax kernel
   (deleted in cleanup — restore from git history `cccba03^`); engine's
   `finalize_logits_one_row` uses it when non-null. Kills the 262k host scan
   AND the pipeline flush before it (token readback = 4 bytes).
2. **Device embed + PLE table lookups**: engine's `dequant_one_row` calls
   `v->embedding_lookup` when the dtype is supported (metal kernel exists,
   handles f32/f16/q4/q5/q6). Removes the remaining per-token host phases →
   decode step becomes a single submit.
3. **Fused q/k norm+rope op** (optional slot, restore wip's
   `q_norm_rope_rows` kernel): 3 dispatches → 1 per q and k, ~140 fewer
   dispatches/token.
4. Then re-profile; if encode cost shows up (~0.5 ms/tok), consider decode
   replay (recorded command stream) as the last step — wip's machinery is in
   git history.

### Phase B — Prefill: 731 → ~1000-1200 t/s
1. **f16 KV cache mode** (main already has a session `kv_mode` enum): metal
   requests f16; kv_store appends through the converting kernel
   (`kv_append_rows_f16` is exactly that). Flash then runs with **zero
   per-layer conversion**, decode attention gets 2× bandwidth too.
2. **Fused PLE block** (optional slot, wip kernels in git history): 6 ops → 1.
   PLE was 11% of FLOPs but ~6 dispatches/layer.
3. **Batch the per-token embed dispatches** (576 → 5 per chunk).
4. m_max default 128 for metal (currently env `GEIST_M_MAX=128`).
5. Re-measure cool: q4k at capacity is ~190 ms; attention ~85; q6k ~90; PLE
   fused ~35; tail ~70 → ≈470 ms ≈ **1080 t/s** (wip level, now on the
   correct/authoritative graph).

### Phase C — Passing llama.cpp (the honest physics)
Measured ground truth from this session: **geist's q4_K GEMM kernel and
llama's kernel_mul_mm are EQUAL cool (both 5.5-6.5 TF isolated, both orders
tried)**, and llama's 1546 is fully explained by its kernels at cool clocks
plus a near-zero non-GEMM tail. So:

- **Decode: beatable.** llama's 94 t/s ≈ 287 GB/s effective on a ~400 GB/s
  part; wip already hit llama decode parity at short ctx (84.5 vs 81.8).
  Recipe: Phase A + f16 KV + fused matvec chains (gate+up in one pass —
  wip's nt8 fused kernels are in git history) + decode replay (encode-free
  tokens — something llama does not do). Target: **95-110 t/s**, i.e. past
  llama, because replay + single-submit removes overhead llama still pays
  per token (its ~1 ms/token CPU side is visible in its conc-off numbers).
- **Prefill: parity is realistic, beating needs kernel innovation.** With
  Phase B geist lands ~1080-1200 vs llama 1546. The last 1.3× is llama's
  leaner elementwise tail (its fusion) — closable with the fused-op slots —
  and both engines then sit on the SAME ~6 TF GEMM plateau. To pass llama on
  pp512 one of these must land:
  1. a q4_K GEMM **above** 6.5 TF (double-buffered threadgroup staging /
     wider NR1 with register-tiled accumulators — unproven on M1, the one
     genuine research item);
  2. gemma-3n-specific fusion llama doesn't do (PLE block + layer-scale into
     the down-GEMM epilogue) — worth ~5-8%;
  3. win on **end-to-end total** instead: with decode > llama, total
     (pp512+tg64) passes llama even at pp parity — total is the user-facing
     number this README already leads with.
- **Long context**: flash + f16 KV scale better than llama's f32 paths at
  8k+ (wip measured better relative scaling) — a second front where geist
  can lead.

### Sequencing & gates
A1 → A2 → B1 (each ~1 session, parity + cool A/B after each) → B2/B3/A3
(one session) → C-decode (replay + fused matvecs) → C-prefill (GEMM research,
timeboxed). Every engine change stays additive (nullable vtbl slot + host
fallback), CPU backends untouched — the pattern proven six times this session.

---

## 3. State 2026-07-04 s2 & the next options (fusion batch landed)

Cool, M1 Max, gemma-4-E2B Q4_K_M (`benchmark/compare_metal.sh`):

| | port start | **now (05bbbe5)** | wip (pre-re-port) | llama.cpp |
|---|---|---|---|---|
| prefill pp512 t/s | 188 | **756** | 1072 | 1548 |
| decode tg64 t/s | 3.7 | **54.8** | 61 (84 short-ctx) | 93 |
| total pp512+tg64 | 28.5 | **312** | — | — |

**All roadmap fusion slots landed** (rmsnorm_add, f16-KV, ple_block,
attn_qkv_prep, ffn_gate_up, device embed/PLE-prep). Decode ops/tok 945→~540.
**The measured lesson: op-count reduction is spent** — −43% ops bought −3%
GPU time; small ops were latency-hidden. What's left is not more fusion.

### Decode: 18.2 ms/tok → wip 16.4 → llama 10.8. Measured split:
q4k matvec **5.2** | serial command-stream floor **~6.0** | norms 1.76 |
q6k 1.05 | f32-PLE 1.05 | attention 0.74 | host tail ~1.2 | elem/copy 0.3.

1. **Command-buffer pipelining à la llama (n_cb) — THE decode lever,
   attacks the ~6 ms floor and is PROVEN on this hardware.** Source
   analysis of llama's ggml-metal (2026-07-04): per graph it splits the
   node list into n_cb+1 command buffers (optimal n_cb=1–2 on M-series,
   their comment), **enqueues them all up front**, commits the first
   immediately so the GPU starts while worker threads still encode the
   rest (`dispatch_apply` + `commandBufferWithUnretainedReferences`), and
   **never calls waitUntilCompleted after compute** — sync happens lazily
   at the logits read. Net effect: encode cost AND front-end stream
   parsing of buffer k+1 overlap kernel execution of buffer k. geist
   instead encodes → commits → waits one serial buffer, exposing the
   whole ~6 ms parse/schedule floor. Port: rotate the sequence buffer
   every ~N layers WITHOUT waiting (enqueue up front, commit as encoding
   passes the boundary), keep the existing flush-on-map as the only sync
   point. Bounded change in the sequence machinery, no kernel work, no
   ICB setBytes refactor needed. ICB stays the fallback idea if buffer
   splitting recovers less than llama's overlap suggests.
2. **Fused q/k/v projection matvec** (like ffn_gate_up): q_proj/k_proj/v_proj
   are still 3 separate GEMVs reading `normed` 3×. One kernel, one read →
   part of the 5.2 ms q4k / 1.9 ms-over-floor headroom. ~M1-executable,
   small win (~0.5 ms).
3. **q6k down_proj** (1.05 ms) already at ~6 TF structure — leave it.

### Prefill: 756 → wip 1072 → llama 1548.
4. **Extend the fusion slots to prefill (rows>1) — THE prefill lever.**
   attn_qkv_prep, ple_block and ffn_gate_up all bail on rows>1 today, so
   the *prefill* graph is still fully decomposed (separate rmsnorm+rope+
   norms per chunk, 6-op PLE) — exactly wip's 340-ms-over advantage. The
   kernels are per-row already; batching the dispatch grid (or looping rows
   in-kernel) closes most of 756→1072. Biggest prefill win, M1-executable,
   same additive pattern.
5. **q4k GEMM > 6.5 TF** — the one genuine research item, the only lever to
   pass llama on prefill. Both engines sit on the SAME ~6 TF plateau
   (measured, isolated). Double-buffered threadgroup staging / wider NR1 with
   register-tiled accumulators. Unproven on M1; needs an M3/M4 GPU-limiter
   capture to target (M1 has no per-dispatch shader counters). Timeboxed.

### Recommended order
1 (command-buffer pipelining → the decode floor, llama-proven) → 4
(prefill fusion → recover most of the wip prefill gap; pipelining helps
prefill chunks too) → 2 (q/k/v projection) → 5 (GEMM research, needs M3).
5 is the only path to actually *beat* llama prefill; end-to-end total
already leads once decode > llama.

## 4. Execution ledger (2026-07-04 s4) — every plan item accounted for

| item | status |
|---|---|
| A1 device greedy head/argmax | ✅ 084cec3 (restored argmax_f32 kernel, 4-byte readback) |
| A2 device embed + PLE lookups | ✅ 9adc04a + 2bd7664 |
| A3 fused q/k norm+rope | ✅ 9ed7d8c (attn_qkv_prep — also active for prefill, §3.4's "bails on rows>1" was wrong for this slot) |
| A4 decode replay | ✅ re-measured (<0.8 ms prize) → retired; superseded by pipelining |
| B1 f16 KV mode | ✅ a175cec |
| B2 fused PLE block | ✅ d97545b (6→3 ops; ple_proj_norm_f32 kernel measured unusable — 400 µs 1-tg GEMV) |
| B3 embed batching | ✅ 2bd7664 (device per-row lookups; literal 576→5 batching worthless — dispatches are latency-hidden, wip measurement) |
| B4 m_max=128 metal default | ✅ 91d6923 |
| B5 cool re-measure | ✅ compare_metal.sh, every step |
| C fused matvec chains | ✅ ffn_gate_up (9ed7d8c) + linear_t_pair k/v (91d6923); plain layout beats packed nt8 (measured) |
| C pipelining (replay successor) | ✅ 9a33ade (−1.85 ms/tok, sweep-tuned N=192) |
| C gemma-3n layer-scale into down-GEMM epilogue | ❌ declined with measurement: all elementwise incl. scale_rows = 0.23 ms total, latency-hidden — no lever |
| C q4_K GEMM >6.5 TF | ⏸ M3/M4-gated research (M1 has no per-dispatch shader counters) — the only remaining path past llama prefill |
| long-context front (8k+) | ⏸ not re-benchmarked on this branch yet — f16-KV + flash should scale better than llama's f32 path |

## Known issue: decode no-ops at context ≥ 4096 (found 2026-07-04 s5)

After a prefill that fills ≥4096 positions, `decode_step` returns OK but
runs no real forward (measured 0.54 ms/tok vs ~50 at kv~3072). Root cause:
the RoPE cos/sin tables are built in `allocate_runtime_rope` sized to
`st->max_seq_len`, which is set at **state create** (model-load time,
`transformer_state_create`) from the *model-load* opts — default 4096. A
session created later with a larger `max_seq_len` never resizes the
state-level tables, so position ≥4096 indexes past them and the forward
degenerates. Correct up to ~3072; the product regime (≤2048) is unaffected.
Fix scope (separate, needs multi-session parity): size the state RoPE tables
(and any position-indexed state buffer) to cover the max session window —
either raise the state default or resize on session-create when a larger
window is requested. Not Metal-specific (engine-level).

## 6. Long-context root cause FOUND (2026-07-04 s5) — MQA K/V re-streaming

Cool prefill scales O(n²): ms/tok = 0.87 + 0.00088·pp (256→3072 all fit).
At pp2048 the linear-in-seq term is 1.80 ms/tok = **69% of prefill**.
**geist's long-context attention slope is 10.7× steeper than llama's**
(geist +1.26 ms/tok over pp512→2048; llama +0.118).

Not clock (powermetrics: both pinned at 1296 MHz max, geist 9% idle-res vs
llama 5%). Not the sliding window (correctly clamped, 28 of 35 layers cap at
512). Not a head_dim bug (GGUF key_length=512 is real gemma-3n; the 7 full
layers at indices 4,9,…,34 genuinely do head_dim-512 O(seq²) attention).

**The mechanism: MQA K/V re-streaming.** gemma-3n is `head_count=8,
head_count_kv=1` (pure MQA). The prefill flash kernel `attention_flash_sg_f16`
dispatches `fgroups={q_rows/8, q_heads}` = one threadgroup per (query-tile,
head) with `kvh = h/(qh/kvh) = 0` for all 8 heads. So **all 8 query heads
stream the same K/V independently** — up to 8× the K/V device-memory
bandwidth. On the 7 full layers at kv=2048 that is ~235 MB of redundant K/V
reads per query-tile pass (vs ~29 MB shared). llama's flash_attn_ext groups
the GQA/MQA heads that share K/V into one threadgroup and loads K/V once.

**The lever (concrete, M1-executable, biggest remaining win): head-grouped
flash.** Rewrite the prefill flash so one threadgroup handles a query tile
across MULTIPLE q-heads that share a kv-head, loading each K/V tile into
threadgroup memory once and reusing it for all grouped heads' QK/PV. For
pure MQA that is all 8 heads → up to 8× less K/V traffic on the full layers,
directly attacking the 10.7× slope. Same simdgroup-matmul structure, just
an extra head loop inside the kv-tile loop + wider accumulator set. Gated on
token parity + the pp512/1024/2048 scaling curve (must flatten toward
llama's slope). Decode split-KV attention has the same MQA structure — the
grouping helps decode long-context too (tg64@kv2100 is 3.0× off llama).

This reorders the roadmap: head-grouped flash is now the #1 lever (was
"prefill fusion"). It needs no M3 and compounds with every context token.

## 5. Now vs the pre-merge wip engine — the complete diff

**Recovered from wip** (same capability, different form — optional nullable
vtbl slots on main's contract instead of wip's co-designed block ops):
f16 KV cache (now a proper `GEIST_KV_F16` session mode, AUTO-upgraded via
slot presence, half the KV memory) · fused q/k/v prep = norm+RoPE+KV-append
(`attn_qkv_prep`, 2 dispatches; wip: inside `attention_query_block`) ·
fused PLE (`ple_block`, 3 dispatches; wip: 1 block op — the historical
proj_norm kernel measured unusable and is bypassed) · fused gate+up GeGLU
(`ffn_gate_up`, the restored n4 kernel) · fused post-norm residual
(`rmsnorm_add`) · device greedy argmax (`argmax_f32`, 4-byte readback) ·
device embed/PLE lookups · batched submission · split-KV flash decode ·
simdgroup flash prefill · strided PLE slab views · the fast f32/q4k/q6k
GEMM kernels.

**Beyond wip** (the current engine has, wip never did):
command-buffer pipelining (llama n_cb-style — wip was strictly
encode→commit→wait serial; its replay only saved CPU encode, pipelining
also hides the GPU front-end stream cost, measured −1.85 ms/tok) · fused
k/v projection pair (`linear_t_pair`; wip ran k/v separately, its fused
qk variant lost and was opt-in) · empty-sequence commit-without-wait ·
full dispatch-profile attribution + GEIST_SKIP_* subtractive categories ·
`benchmark/compare_metal.sh` · m_max=128 as the metal default · device
prefill embed + PLE-prep phases (no host map flushes mid-batch).

**Deliberately NOT carried over** (measured decisions):
decode replay record+execute (prize re-measured at <0.8 ms — superseded
by pipelining) · packed nt4/nt8 weight layouts + prepare_weight_layout
pack-cache (plain layout beats packed, wip's own late finding; packed
paths remain env-gated experiments) · w4a8 (no bandwidth win on Apple) ·
the block-op backend contract itself (unmergeable with main — the reason
the re-port exists).

**Structural differences** (the fundamentals):
every step token-parity-gated against cpu_scalar on main's authoritative
graph (wip: self-checksums only) · 100% additive nullable slots, CPU
backends untouched, branch fast-forwardable onto origin/main (wip:
permanently divergent) · f16 KV integrated beside FP32/INT8/KIVI in the
public session enum · every new path has an env kill switch.

**Remaining measured gap to wip** (cool): prefill 762 vs 1072 (1.41×),
decode 55.2 vs 61 (1.11×). The residue is main's CPU-first graph shape —
more ops and intermediate passes per token than the co-designed wip graph
(~540 vs ~450 dispatches) — plus wip's decode number being measured at
shorter context. It buys correctness, mergeability, and every CPU backend.

### Why llama is beatable (source-analysis 2026-07-04)
- Its q4_K kernels are NOT faster (measured shape-for-shape equal, both
  ~6 TF); its decode matvecs run ~287 GB/s effective on a ~400 GB/s part
  — kernel headroom exists for BOTH engines.
- It computes MORE per token than geist: the full gemma-3n AltUp/Laurel
  graph (~110 extra op sites in its layer builder) — and still decodes at
  10.8 ms. Its lead is 100% engine overlap, not math or kernels.
- Its own fusion/concurrency features measured NEGATIVE for decode on M1
  (tg 76.5→92.9 with CONCURRENCY_DISABLE) — its best decode config is a
  serial encoder, same as geist's. The overlap machinery (n_cb + async
  completion) is the entire difference, and geist can adopt it AND keep
  its leaner graph (no AltUp cost) + f16-KV long-context scaling.
- It re-encodes every token on the CPU (no replay/ICB) and swings 77–93
  tg across cool runs; a pipelined geist decode has a realistic shot at
  a STABLE >93.

---

## 6. M1 program closed (2026-07-04, evening)

Final cool numbers: **987 pp512 / 81.2 tg64 / 441 total** vs llama same-run
1542 / 91.3. Decode gap 1.12× (wip's 61 passed by 33%); prefill 1.56×;
tg64@kv2100 73 t/s (was 27).

**Re-confirmed 2026-07-05** (`1020d45`, after the #58/#60 Metal work and the
#62/#63 dead-flag cleanup): cool `compare_metal.sh` reads **1006 pp512 / 81.6
tg64 / 445 total** vs llama same-run 1540 / 92.8 — within noise of the close.
Metal performance held across the flag/kernel/replay removals; no regression.

The two invisible-scalar-kernel fixes (sg8 prefill flash + dec512 split-KV,
head_dim-512 full-attention layers) were the day's breakthroughs — found by
scaling-curve triage + `sample`, after the clock, chunking and MQA theories
were each falsified by measurement (powermetrics: both engines pinned at
1296 MHz; m_max 512: regression; MQA K/V redundancy: 23 ms of 671 ms
attention — the flash kernel is compute-bound at ~0.7 TF effective).

What remains is measured to be kernel-efficiency territory: the q4_K GEMM
plateau (~6 TF, shared with llama) and flash-kernel throughput (~0.7 vs
llama-class ~2 TF). Both are limiter/occupancy WHY-questions; M1 exposes
per-dispatch durations but no limiter counters (M3/A17+). Every
measurable-on-M1 lever in this document has been executed, measured, and
either landed or reverted with numbers.
