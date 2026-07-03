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
