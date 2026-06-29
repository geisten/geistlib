# Linux/x86_64 perf profile vs llama.cpp

Hard-data comparison of geist's `cpu_x86` backend against llama.cpp on
the reference 9950X (Zen 5, 16C/32T, DDR5-6400). Measured 2026-06-22 on
identical Gemma 4 E2B-it Q4_K_M, single bench run, `OMP_NUM_THREADS=16
OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=cores`.

> **Correction (2026-06-27).** The original diagnosis below — that the gap
> was a Q4_K inner-loop compute-density problem — was only half the story
> and pointed Phase 3 at the wrong matrix. A per-stage prefill profile
> (`GEIST_PROFILE_PREFILL=1`) showed the dominant cost was the **Q6_K
> `ffn_down` projection at ~82 % of prefill**, running on the scalar
> `cpu_scalar_w_quant_mN` fallback because Q6_K had no native x86 prefill
> (`linear_mN`) kernel. In Q4_K_M, `ffn_down` and `attn_v` are stored as
> Q6_K, not Q4_K, so the whole Q4_Kx8 effort (Steps 1–3c) accelerated
> matrices that were only ~4 % of prefill. Adding `cpu_x86_linear_q6k_mN`
> (tiled W8A8 GEMM) took prefill from **29 → 130 t/s** (4.5×). See "Root
> cause and resolution" at the end. The cycle/IPC data below remains
> accurate for the *old* bf16 Q4_K path it measured.

The profile data here drives the Phase 3 optimization direction: the
gap to llama.cpp is **not** a thread-scaling problem and **not** a
memory-bandwidth problem — it is a **per-cycle compute-density** problem
in the inner loop.

## `perf stat` headline: IPC collapses 6.4×

For `pp128` only (decode=0), single bench iteration:

| Metric                  |        geist |    llama.cpp |       ratio |
| ----------------------- | -----------: | -----------: | ----------: |
| Wall time               |    7.31 s    |    1.19 s    | 6.1× slower |
| User CPU time           |   82.9 s     |    9.33 s    | 8.9× more   |
| **Instructions**        |    **201 G** |     **95 G** | **2.1× more** |
| **Cycles**              |    **426 G** |    **32 G**  | **13.5× more** |
| **IPC (insn/cycle)**    |   **0.47**   |   **3.01**   | **6.4× lower** |
| Frequency               |  5.08 GHz    |  3.29 GHz    | (we boost higher) |
| Frontend stall          |   0.48 %     |   3.08 %     | similar     |
| Branch miss             |   0.10 %     |   0.32 %     | similar     |
| Page faults             |   2.2 M      |   380 K      | 5.8× more   |

`/usr/bin/time -v`, same run:

| Metric                  |        geist |    llama.cpp |       ratio |
| ----------------------- | -----------: | -----------: | ----------: |
| `%CPU` (parallelism)    |    **1170 %** |   **931 %**  | both ~10 cores effective |
| RSS                     |    7.5 GB    |    4.3 GB    | 1.7× more   |

## Diagnosis

1. **Threading is fine**: `1170 %` CPU means ~11.7 cores actively
   executing — comparable to llama.cpp's 9.3. We were briefly suspicious
   of OMP fork/join overhead, but `OMP_WAIT_POLICY=active` confirmed
   threads are *running*, not stuck on barriers.

2. **We execute 2.1× more instructions for the same work.** That is the
   weight-layout efficiency: every VPMADDUBSW in llama.cpp's inner
   produces partial sums for **8 cells in parallel** (lane-parallel),
   while our VPDPBUSD / VDPBF16PS produces a partial sum for **1 cell**.
   For an equivalent matmul we therefore emit ~8× more cell-level dot
   product instructions, which the inner-loop overhead (loads, scale
   broadcasts, accumulator updates) inflates to a ~2.1× total instruction
   count.

3. **We run those instructions at 6.4× lower IPC.** Zen 5's 6-wide front-
   end issues ~6 instructions per cycle when the dependency graph has
   enough parallelism. llama.cpp's `acc_rows[16]` and `acc_min_rows[16]`
   provide 16 independent fp32 accumulator chains that overlap across
   iterations; the front-end stays full. Our BF16 path writes to ONE
   shared `acc_v[i, jj]` per (i, jj) inner step, and each VFMADD has a
   4-cycle latency. With 4 m-rows and the M_TILE=4 fmadd chain that's a
   **16-cycle critical path** the compiler cannot reorder out of.

4. **Net effect**: 13.5× more cycles total = 2.1× more instructions ×
   6.4× lower IPC. The wall-clock ratio is 6.1× (not 13.5×) because
   geist runs at a higher boost frequency (5.08 vs 3.29 GHz), which
   partially compensates for the cycle deficit.

## Implication for Phase 3

The fix is **architectural in the inner loop**, not tuning the existing
kernel:

- **Independent accumulators** (`acc_rows[16]`) break the dependency
  chain → expected IPC ~3 (Zen 5 peak utilization) → ~6× cycle reduction.
- **Lane-parallel weight layout** (`block_q4_Kx8`) gives 8 cells per
  VPMADDUBSW → ~2× instruction reduction.
- Compound expected lift: ~10–12× on the inner kernel, bringing prefill
  from ~28 t/s to **~150–250 t/s** — within 2–3× of llama.cpp's 514
  rather than the current 18×.

That is exactly what the Phase 3 spec lays out (`docs/LINUX_X86_SPEC.md`
Phase 1b Step 3 + Phase 2 native VDPBF16PS). The kernel port is mechanical
(the scalar reference at `kernel_q4kx8_gemm_scalar.c` is verified correct
and serves as the oracle for any AVX variant), but the shuffle patterns
in llama.cpp's `ggml_gemm_q4_K_8x8_q8_K` (~640 LOC of intrinsics) are
intricate enough that a single-turn port has consistently produced bugs.
The recommended approach is a multi-turn focused debug session where each
intermediate value (d/dmin decode, mins_01 build, iacc_0/iacc_1 lane
assignment) is cross-checked against the scalar reference before moving
on.

## Reproduce

```sh
# geist
make TARGET=linux BACKENDS="cpu_x86 cpu_scalar"
/usr/bin/time -v env OMP_NUM_THREADS=16 OMP_WAIT_POLICY=active \
    OMP_PROC_BIND=close OMP_PLACES=cores GEIST_BENCH_BACKEND=cpu_x86 \
    bin/linux/release/tests/bench_perf_sweep \
    --gguf gguf_artifacts/gemma4-e2b-Q4_K_M.gguf \
    --seq-lens 128 --decode-n 0 --warmup 8 --repeats 1 --threads 16

# llama.cpp (commit 7c082bc)
/usr/bin/time -v llama-bench -m gemma4-e2b-Q4_K_M.gguf -p 128 -n 0 \
    -t 16 -ngl 0 -r 1

# perf stat (needs perf_event_paranoid <= 1):
perf stat -- <above command>

# per-stage prefill breakdown (the diagnostic that found the real bottleneck):
GEIST_PROFILE_PREFILL=1 <above bench command>
```

## Root cause and resolution (2026-06-27)

The `GEIST_PROFILE_PREFILL=1` per-stage breakdown on the same model/host,
*before* the fix (warmup 1, repeats 1):

| Stage             | Time      | Share          |
| ----------------- | --------: | -------------- |
| ffn `down`        | 3597 ms   | 94.8 % of ffn  |
| ffn `gate_up`     |  161 ms   |  4.2 % of ffn  |
| attention `qkv`   |   74 ms   | 65 % of attn   |
| attention `o_proj`|   27 ms   | 24 % of attn   |
| **ffn total**     | **3794 ms** | **88 % of prefill** |

`ffn_down` dominated. It is **Q6_K** in Q4_K_M (confirmed by dumping the
GGUF: 34 Q6_K tensors — every `ffn_down` + `attn_v`), and the cpu_x86
backend had no Q6_K `linear_mN`. `cpu_x86_resolve_weight` calls the scalar
resolver first (which installs both `linear_m1` and `linear_mN`), then
overrides only `linear_m1` for Q6_K — so prefill silently ran the scalar
`cpu_scalar_w_quant_mN` (dequant Q6_K → fp32 → scalar matmul) at ~14.7
ms/call. The Q4_Kx8 kernel, measured in isolation, was already fast
(~2620 GFLOP/s @ 16T) but only covered the ~4 %-of-prefill Q4_K matrices,
so Steps 1–3c moved the end-to-end number by ~0 %.

**Fix:** `cpu_x86_linear_q6k_mN` (src/backends/cpu_x86/linear_q6k.c) — a
tiled W8A8 GEMM (`w8a8_gemm`, kernel_w8a8_avx512_vnni.c). Each weight row
is read once and reused across the whole token batch; `W8A8_JT=4`
independent fp32 accumulators per VPDPBUSD step keep the back-end fed.
Activations are quantized to int8 once for all tokens. Installed alongside
the existing W8A8 `linear_m1` decode kernel.

| Metric (pp128, 16T)   |   before |    after | ratio |
| --------------------- | -------: | -------: | ----: |
| prefill               | 29 t/s   | 130 t/s  | 4.5×  |
| ffn `down` stage      | 3597 ms  |  596 ms  | 6.0×  |

Correctness: `tests/test_w8a8_gemm_unit.c` cross-checks the GEMM against
the trusted per-token W8A8 GEMV (max Δ ~9e-05 over K=12288); end-to-end
generation is coherent.

## Lane-parallel W8x8 (2026-06-28)

The tiled `w8a8_gemm` still reduced each 16-block to a scalar per row (2
`hadd`) and applied the per-block fp32 scale once per row. `w8x8_gemm`
(kernel_w8a8_avx512_vnni.c) removes both: 8 output rows are interleaved
(`w8x8_repack`) so one VPDPBUSD lands 8 rows in the 8 int32 lanes — no
`hadd` — and the per-block scale/offset become one 8-wide fp32 FMA for all
8 rows. `cpu_x86_linear_q6k_mN` uses it when `n_out % 8 == 0` on a VNNI
host; otherwise it falls back to `w8a8_gemm`. The interleaved blob is built
at resolve and appended after the row-major one (~2× Q6_K memory, like the
Q4_K dual-blob; +0.37 GB RSS on this model).

| Metric (pp128, 16T)   | w8a8_gemm | w8x8_gemm | ratio |
| --------------------- | --------: | --------: | ----: |
| prefill               | 130 t/s   | 149 t/s   | 1.14× |
| ffn `down` stage      | 596 ms    | 248 ms    | 2.4×  |

`down` did exactly what the design predicted (2.4×); the smaller
end-to-end move is because `down` is no longer the bottleneck — see below.
Correctness: `test_w8a8_gemm_unit.c` cross-checks `w8x8_gemm` (incl. repack)
against the GEMV oracle (max Δ ~1e-04); generation stays coherent.

## F32 PLE projections → cblas (2026-06-28)

After W8x8, `ple` became the largest prefill stage (~59 %, ~420 ms/rep).
Sub-profiling found the cost was not the Q5_K gather but the two **F32**
per-layer projections (`inp_gate` 1536→256, `proj` 256→1536, ×35 layers):
cpu_x86 never overrode the F32 dtype, so they ran `cpu_scalar_w_f32_mN` —
a **single-threaded naive triple loop with a double accumulator** ("~10×
slower than cblas; intentional, this is the reference").

**Fix:** cpu_x86 now installs `cpu_x86_linear_f32_m1/_mN` (backend.c) that
route F32 dense through `geist_sgemm`/`geist_sgemv` (OpenBLAS), mirroring
cpu_neon. Four lines + a switch case.

| Metric (pp128, 16T)   | before | after  | ratio |
| --------------------- | -----: | -----: | ----: |
| prefill               | 149 t/s| 351 t/s| 2.35× |
| `ple` stage           | 420 ms | 48 ms  | 8.8×  |

## Vectorized FFN gelu (2026-06-28)

With PLE fixed, the FFN `act` (gelu_tanh) stage was the largest remaining
non-matmul cost (~47 ms/rep, 17 % of ffn) — a single-threaded scalar
`tanhf` per element on cpu_scalar (cpu_x86 didn't override elementwise).

**Fix:** `cpu_x86_gelu_tanh{,_mul,_mul_scaled}` (elementwise.c) — OMP over
the work, with tanh computed as `1 - 2/(e^2u+1)` so the inner `expf`
auto-vectorizes via glibc **libmvec** (`_ZGVdN8v_expf`, 8-wide) under the
project's standard `-ffast-math -fopenmp`. `u` is clamped to ±10 so e^2u
can't overflow. Same math as the reference within ~1e-6 (test_gelu_x86_unit.c).

| Metric (pp128, 16T)   | before | after  | ratio |
| --------------------- | -----: | -----: | ----: |
| prefill               | 351 t/s| 387 t/s| 1.10× |
| ffn `act` stage       | 140 ms | 8.5 ms | 16×   |

## Remaining gap to llama.cpp

llama.cpp pp128 is ~514 t/s; geist is now ~387 t/s — **~1.33× behind, down
from ~18×**. The profile is now almost entirely the quantized matmuls:

| Stage group | share | detail |
| ----------- | ----: | ------ |
| ffn         | 67 %  | gate_up (Q4_K) + down (Q6_K W8x8); act/gelu now 1 % |
| attention   | 18 %  | qkv + o_proj (Q4_K) |
| ple         | 14 %  | Q5_K gather + F32 projections (cblas) |

The non-matmul waste is gone. The profile is now matmul-bound.

## Kernel-level comparison vs llama.cpp (2026-06-28)

Direct read of llama.cpp's x86 kernels (`ggml/src/ggml-cpu/arch/x86/repack.cpp`,
`arch-fallback.h`, `quants.c`) against ours, plus isolation GFLOP/s on this host.

**Q4_K prefill (gate_up, qkv, o_proj) — at parity.** Our
`q4kx8_gemm16x16_tile_avx512` is a faithful port of
`ggml_gemm_q4_K_8x8_q8_K`: identical 16×16 register blocking, 32 live
`__m512` accumulators (16 main + 16 min), 64 `maddubs_epi16` + 16
`madd_epi16` per row-pair, int16 sub-block scales folded into the integer
accumulation, a single fp32 `d`/`dmin` multiply per super-block, q8_Kx4
activations with precomputed bsums, no prefetch, all-FMA. Measured **~2620
GFLOP/s @ 16T** — there is no structural headroom here; it *is* their kernel.

**Q6_K prefill (down) — we are ahead.** llama.cpp has **no x86 SIMD** for
Q6_K GEMM: `arch-fallback.h` maps `ggml_gemm_q6_K_8x8_q8_K` to the generic
C++ scalar template; only the M=1 decode `vec_dot` is vectorized. Our
AVX-512 `w8x8_gemm` runs **~2360 GFLOP/s @ 16T**. The one difference worth
noting: llama keeps the Q6_K sub-block scale as **int8 folded into integer
accumulation** (fp32 multiply once per 256-superblock), whereas we predecode
to **fp32 scale+offset per 16-block** → an extra 8-wide fp32 FMA-pair per
block. That is the ~10 % gap between w8x8 (2360) and q4kx8 (2620).

**Verdict: the inner loops are tuned to parity (Q4_K) or better (Q6_K).**
There is no large GEMM micro-tuning win left — the kernels are not the gap.
Remaining headroom, smallest-effort first:

1. **W8x8 integer-scale folding — tried, REVERTED (pessimization).** Built
   `block_q6_Kx8` (int8 sub-block scales + fp32 super-block `d`) and a
   `q6kx8_gemm` that folds the int8 scale into the int32 accumulator. Result:
   `down` got *slower* (289 → 319 ms), prefill 387 → 383. Reason: our 8-row
   lane-parallel layout uses `dpbusd` which yields one row per **int32** lane,
   so applying the per-row int scale needs `mullo_epi32` — high-latency and
   on the same vector-int ports as `dpbusd`. That costs more than the cheap
   fp32 `fmadd` it replaced. llama's int8-folding only wins because its Q4_K
   kernel keeps **int16** partials (`maddubs`) and folds via the cheap
   `madd_epi16`; that structure is incompatible with our dpbusd layout. The
   fp32-per-block W8x8 is the right kernel here. Conclusion: the ~10 % w8x8
   vs q4kx8 GFLOP/s gap is not the scale handling — likely 8-wide (256-bit)
   vs 16-wide (512-bit) lane parallelism. A `block_q6_Kx16` 512-bit dpbusd
   kernel is the only plausible remaining GEMM lever, and it's a large rewrite
   for ~3 % — not worth it.
2. **Redundant activation quantization.** We re-quantize the layer input to
   q8 once *per projection* (q/k/v separately, gate/up separately); llama
   quantizes once and reuses. Measured ~5 % of a matmul, so ~3 ms/rep total —
   not worth the cross-call caching plumbing.
3. **`ple` Q5_K gather** (`compute_per_layer_inputs_batch`) dequants the
   per-layer embedding rows scalar/serially; worth a look if ple stays 14 %.
4. **Memory:** the row-major W8A8 blob exists only for `m1` decode. Serving
   `m1` from W8x8 too (n_tokens=1) would drop it — removes the +0.37 GB and
   unifies the path. Touches the hot decode path, so deferred.

(Note: llama-bench at this repo's HEAD can't load the gemma4/PLE GGUF, so the
514 t/s target is from the pinned-commit run in the table above, not
reproducible on this checkout. The per-kernel GFLOP/s parity is the reliable
comparison.)

## Planned: shared activation quantization (architectural)

**Idea.** Each linear projection independently quantizes its fp32 input to
int8 inside `linear_*_mN`. Within a layer the same input feeds several
projections, so we quantize it more than once:

- attention: `attn_q`, `attn_k`, `attn_v` all read the attention-norm output;
- ffn: `ffn_gate`, `ffn_up` both read the ffn-norm output.

llama.cpp quantizes the row to q8_K once and reuses it across the projections.

**What can actually be shared (Gemma 4 E2B Q4_K_M dtypes).** The quantized
format must match the consumer's kernel:

| group | tensors | dtype | quant format | shareable? |
| ----- | ------- | ----- | ------------ | ---------- |
| attn  | q, k    | Q4_K  | Q8_Kx4       | q+k share one quant |
| attn  | v       | Q6_K  | W8A8 acts    | different format — separate |
| ffn   | gate, up| Q4_K  | Q8_Kx4       | gate+up share one quant |

So the win is **2 fewer input-quantizations per layer** (one in attn, one in
ffn). Measured input-quant ≈ 0.034 ms (m=64, n_in=1536); 2 × 35 layers ≈
**~2.4 ms/rep (~0.7 % of prefill)**. Total activation quant is only ~2.6 %,
and most of it isn't redundant.

**Design (if pursued).**
1. Add a small per-call quant cache to `cpu_x86_state`: `{const float *src;
   size_t n_in, m; void *q8kx4_blob;}`. `linear_q4k_mN` checks if `x` (pointer)
   and shape match the cached entry; on hit, skip `quantize_q8_Kx4` and reuse.
2. Invalidate at the start of each forward step (the engine already has a
   per-step boundary). Pointer-identity match is safe because the norm output
   buffer is stable within a layer and distinct per layer.
3. Q6_K `attn_v`/`down` keep their own W8A8 quant (different format) — no
   change.

**Recommendation: not worth it.** ~0.7 % for cross-call cache plumbing +
invalidation correctness risk fails the cost/benefit bar. Documented for
completeness. If activation quant ever shows up larger in the profile, the
higher-value move is to **thread `quantize_q8_Kx4`** (it is currently the
single-threaded `for mt` loop in `linear_q4k_mN`) rather than to dedupe it.

## Standing vs llama.cpp (CPU-only, 16T, 9950X) — current

Both engines on the **same** GGUF, llama.cpp built from master (b9827, CPU
backend, gemma4 arch supported). After the decode kernels:

| model              | metric  | geist | llama.cpp | result |
| ------------------ | ------- | ----: | --------: | ------ |
| Llama 3.2 3B Q4_K_M | prefill | **351** | 346     | **+1.4 %** |
| Llama 3.2 3B Q4_K_M | decode  | 34.1  | 34.5      | parity |
| **Gemma 4 E2B Q4_K_M** | prefill | **512** | 495     | **+3.4 %** |
| **Gemma 4 E2B Q4_K_M** | decode  | **48.6** | 44.1   | **geist +10 %** |

Progression of the Gemma 4 numbers across the session: prefill 29 → 452
(decode 23.6 → 48.6). Gemma 4 prefill climbed 78 → 85 % (OpenBLAS 1-thread)
→ 91 % (F32 PLE projections quantized to W8A8); decode went from behind to
**+10 % ahead** of llama.cpp. `model_proj` (BF16 on disk) is dequantized to
F32 at load and is also covered by f32q; the remaining slack is the body
Q4_K/Q6_K matmuls (already at/above parity).

### cblas validation — Gemma 4 text (2026-06-28)

ARM/Pi5 must not depend on cblas. After f32q quantizes every F32 PLE
projection (inp_gate, proj, model_proj), what still calls `geist_sgemm`/
`geist_sgemv` (= cblas on openblas builds) in Gemma 4 text inference?

- **cpu_x86: zero.** Verified with `GEIST_PROFILE_GEMM=1` — its atexit report
  registers only on the first `geist_sgemm` call and prints nothing for a
  Gemma 4 (or Llama) run → `geist_sgemm` is never reached. Every dense matmul
  goes through geist's own int8 kernels (q4kx8 / w8x8 / q6k / f32q-W8A8). The
  only `geist_sgemm` call sites left (`linear_fp32`) are vision/audio-encoder,
  not the text path. **A `GEMM_PROVIDER=native` (BLAS-free) build runs Gemma 4
  correctly.**
- **cpu_neon (ARM): not yet cblas-free.** `cpu_neon_w_f32_mN` still routes the
  PLE projections through `geist_sgemm` (cblas on the Pi5 openblas build). Two
  fixes: (1) build Pi5 `GEMM_PROVIDER=native` (native NEON `geist_sgemm`,
  cblas-free but unquantized), or (2) port f32q to cpu_neon — quantize the F32
  PLE to Q8_0 and reuse the existing NEON int8 kernels
  (`linear_q8_0_w8a8_prefill`/`_decode`), cblas-free *and* fast. (2) needs Pi5
  hardware to validate and is the recommended ARM follow-up; it was not done
  here (x86 host — shipping untested NEON into the hot path would be reckless).

**Decode: at parity (Llama) / ahead (Gemma 4).** The remaining gap is
**Gemma 4 prefill (78 %)** — gemma4-specific; Llama prefill is already 98 %.

### Gemma 4 prefill gap — investigation (2026-06-28)

Per-stage prefill (per rep): ffn 231 ms (gate_up 126, down 98), attention
63 ms (qkv 26, o_proj 23, core 5), ple 51 ms. The matmuls use the same
Q4_K (q4kx8) / Q6_K (w8x8) kernels that are at parity on Llama — and geist's
Q6_K *prefill* already beats llama's (which has no x86 SIMD for it), so the
matmuls are not the gap. Ruled out:

- **PLE/main-path elementwise (add/mul/rmsnorm) → OMP+SIMD overrides:**
  neutral on prefill (388) and *regressed decode* (44.6→42.9 — OMP fork/join
  overhead on the tiny per-token tensors). Reverted.
- **Prefill chunk size m_max 64→128:** +2.3 % only (388→396). Available via
  `GEIST_M_MAX`; not made default (scratch growth, Pi5 regressed at 128).

### PLE sub-stage profiling result (2026-06-28)

Added two gated sub-profilers (`transformer ple per-layer` in layer_ple.c;
`ple precompute per-chunk` in layer.c). Gemma 4 prefill, per rep:

| bucket | stage | ms/rep | share |
| ------ | ----- | -----: | ----- |
| per-layer `run_ple_or_copy` | **gate_lin** (F32 1536→256) | 24 | 47 % |
| | **proj_lin** (F32 256→1536) | 24 | 47 % |
| | gelu+mul+rmsnorm+add | ~3 | 6 % |
| per-chunk precompute *(was hidden)* | **model_proj** (BF16 1536→8960) | 16 | 96 % |
| | gather (Q5_K dequant) | <1 | — |

So the gemma4 prefill gap is **~64 ms/rep of skinny F32/BF16 dense matmuls
on cblas** (~73 GFLOP/s), *not* elementwise or the gather. Root cause:
OpenBLAS spins its own thread pool per call, and geist issues ~70 tiny
per-layer PLE sgemms per chunk — the per-call spawn/sync overhead dominates.

**Fix shipped: pin OpenBLAS to 1 thread** (weak `openblas_set_num_threads`,
geist parallelizes the heavy kernels itself). Gemma 4 prefill **388 → 422
t/s (+9 %)**, Llama unaffected, decode unchanged.

| model | metric | geist | llama.cpp | result |
| ----- | ------ | ----: | --------: | ------ |
| Gemma 4 E2B | prefill | **422** | 495 | 85 % |
| Gemma 4 E2B | decode | **44.7** | 44.1 | ahead |
| Llama 3.2 3B | prefill | 337 | 346 | 97 % |
| Llama 3.2 3B | decode | 34.2 | 34.5 | parity |

**Remaining gemma4 prefill lever:** the PLE projections are still ~40 ms/rep
of single-threaded cblas. Quantizing inp_gate/proj/model_proj to int8 and
running them on geist's VPDPBUSD GEMM (OMP-parallel, ~2600 GFLOP/s) would cut
that several-fold — the path to beating llama.cpp on gemma4 prefill too. It
is gemma4-specific and changes those weights' precision, so it needs accuracy
validation (cosine vs the F32 reference) before shipping.

## Fair head-to-head: Llama 3.2 3B Q4_K_M (2026-06-28)

The gemma4/PLE GGUF can't load in llama.cpp (arch name), so the "514 t/s"
target was never reproducible here. To get an apples-to-apples CPU number,
both engines were run on the **same** Llama 3.2 3B Instruct Q4_K_M GGUF
(geist supports `llama` arch natively; verified coherent output). llama.cpp
built **CPU-only** (the CUDA build reports 705 pp but that is GPU-assisted
even at `-ngl 0` — not a CPU comparison). 16 threads, 9950X.

| test          | geist | llama.cpp (CPU) | geist / llama |
| ------------- | ----: | --------------: | ------------: |
| prefill pp128 | 333 t/s | 346 t/s       | **0.96×** (near parity) |
| decode  tg32  | 23.6 t/s| 34.5 t/s      | **0.68×** (1.46× behind) |
| RSS           | 7.8 GB  | 1.87 GB       | 4.2× more |

**Prefill is essentially at parity** — the session's kernel work paid off.
**Decode is the real gap, and it is weight bandwidth.** Decode (M=1) is
memory-bound; geist achieves ~59 GB/s vs llama's ~64 GB/s effective, but it
**reads more bytes per token** because it decodes from expanded formats:

| weights | geist decode format | B/wt | native (llama) | B/wt |
| ------- | ------------------- | ---: | -------------- | ---: |
| Q4_K (gate_up, qkv, o_proj) | W4A8: 4-bit + fp32 scale + fp32 offset | 0.75 | Q4_K packed | 0.56 |
| Q6_K (down, lm_head)        | W8A8: **8-bit** + fp32 scale + fp32 offset | 1.50 | Q6_K packed | 0.65 |

Decode profile (Llama 3B): gate_up 405 ms, down 281 ms, lm_head 196 ms,
qkv 144 ms, o_proj 79 ms — all GEMV, all bandwidth-bound. The Q6_K path is
the worst offender (8-bit expansion → 2.3× native).

### Decode is mixed compute + bandwidth bound (measured 2026-06-28)

Two experiments refined the diagnosis:

- **fp16 scales/offsets in W4A8 (tried, REVERTED).** Halving the scale bytes
  (0.75→0.625 B/wt on Q4_K) left decode at **23.6 t/s unchanged** and dropped
  RSS only 0.29 GB. The branchy scalar `fp16_to_fp32` per block ate the
  bandwidth saving, and broke 3 unit tests for zero gain — reverted.
- **Thread scaling 8→16:** decode goes 18.0 → 23.3 t/s = **1.30×** for 2×
  cores. Not BW-saturated (would be ~1.0×), not compute-scaling (would be
  ~2×). So decode is **both** bound: by the per-block horizontal reduction
  (`hsum_i32_avx2`, 2× per 2-block, port-5 shuffles) in `w4a8_gemv` AND by
  the expanded weight footprint.

**The right fix addresses both at once: a lane-parallel decode GEMV.** Like
the prefill `q4kx8`/`w8x8` kernels, process **8 output rows in the 8 int32
lanes** of one VPDPBUSD, accumulate across blocks, and reduce **once at the
end** — no per-block `hsum`. Reading the **compact q4kx8 layout** (0.56 B/wt,
already resident for prefill) instead of W4A8 (0.75) also cuts BW, and lets
us **drop the W4A8 blob entirely** (~halves the 7.8 GB RSS). The blocker is
the activation format: q4kx8/w8x8 consume 4-token-interleaved `block_q8_Kx4`;
decode is M=1, so it needs a single-token int8 activation broadcast variant
(a dedicated M=1 kernel over the 8-row-interleaved weights).

Prioritized:
1. **Lane-parallel M=1 decode GEMV over q4kx8 — DONE.** `q4kx8_gemv_m1`
   (kernel_q4kx8_gemm_avx512.c) reuses `quantize_q8k_row` +
   `q4kx8_gemv_one_row_tile`: 8 output cells in the 8 fp32 lanes, reduced
   once per tile, reading the resident q4kx8 (0.56 B/wt). **Decode 23.6 →
   27.5 t/s** (geist 68% → 80% of llama). Q4_K stages: gate_up 405→276 ms,
   qkv 144→107, o_proj 79→54. Confirms Q4_K decode was hsum/compute-bound.
2. **Native 6-bit Q6_K decode** (down 247 ms + lm_head 195 ms, now the
   largest decode cost — **bandwidth-bound**: W8A8 reads 1.5 B/wt vs native
   Q6_K 0.65). Routing Q6_K decode through the lane-parallel W8x8 was
   measured **neutral** (23.6→23.56) — confirming it's BW, not hsum. The fix
   is a GEMV that decodes directly from the original Q6_K (`w->raw`, already
   mmap'd) — unpack 6-bit, VPDPBUSD vs int8 acts, int8 sub-scales folded via
   `madd_epi16`, one fp32 `d` per super-block (llama's vec_dot_q6_K_q8_K
   structure). Est. down+lm_head ~halve → decode → ~32–33 t/s.
3. **Drop the W4A8 blob** — now unused for Q4_K body matrices (both m1 and mN
   read q4kx8). Allocate only q4kx8 when n_out%8==0 → large RSS cut.
4. **Kernel BW/prefetch tuning** for the last 59→64 GB/s.

## Session summary

29 → 387 t/s (13.3×), ~18× behind llama.cpp → ~1.3×. The wins were all
removing non-matmul stalls; the GEMMs themselves were already at parity
(Q4_K is a port of llama's kernel) or ahead (Q6_K — llama has no x86 SIMD).
The remaining gap is not addressable by GEMM micro-tuning (verified: int8
folding regressed; wider 512-bit Q6_K is ~3 % for a large rewrite) — the
profile is clean, matmul-bound code.

## BitNet b1.58 2B-4T (I2_S ternary) — 2026-06

BitNet was completely non-functional on cpu_x86 before this work: cpu_scalar's
resolver returned `GEIST_E_UNSUPPORTED` for I2_S (ternary BitLinear weights)
and had no F16 dense path for the tied lm_head, so the model failed at the
first projection and emitted zero tokens. Fixed (I2_S/F16/BF16 in cpu_scalar)
then accelerated on cpu_x86.

Kernels (9950X, 16T, same `ggml-model-i2_s.gguf`):
- **kernel_i2s** — biased-u8 VPDPBUSD ternary GEMV/GEMM. Codes {0,1,2} unpacked
  in-register from the packed 0.25 B/wt stream (decode is BW-bound, so no f32
  predecode); `dot = Σcode·a − Σa`; one per-tensor fp32 scale; activations
  permuted once per token into the VPDPBUSD pairing order. M=1 decode + JT=4
  token-tiled prefill GEMM (weight row read once, reused across the tile).
- **kernel_f16_gemv** — OMP + F16C GEMV for the 657 MB tied F16 lm_head (read
  once per decode step; the NEON spec-head fast path is ARM-only). 400 → 8.6
  ms/call (~76 GB/s).

| metric | scalar (was) | geist cpu_x86 | bitnet.cpp (i2_s) |
| ------ | -----------: | ------------: | ----------------: |
| prefill pp128 | 4.6  | **472** | 686 |
| decode  tg64  | 1.0  | **49.6** | 57.0 |

Output is byte-identical to bitnet.cpp (greedy) — correctness cross-validated.
geist is at 69 % of bitnet.cpp prefill, 87 % decode. bitnet.cpp's i2_s GEMM is
plain AVX2 maddubs (not VNNI) but wins via a load-time weight transform + ggml's
mature GEMM blocking/threading. Tried and rejected as no-ops here: JT=8 (465 vs
472) and per-code-group independent accumulators (451; register-spills at 22
live zmm). Closing the gap needs the row-interleaved layout (16 rows → 16 int32
lanes, no per-cell `_mm512_reduce_add_epi32`) — a load-time repack, deferred.

### Build footgun
`backend_registry.o` is compiled with the `-DGEIST_BACKEND_*` set active at
*its* compile time; switching `BACKENDS=` without `make clean` leaves a stale
object so `"auto"` silently resolves to the wrong `registry[0]` (cpu_scalar
instead of cpu_x86). Always `make clean` when changing `BACKENDS`. The
bench_perf_sweep harness also defaults to cpu_neon→cpu_scalar; use
`GEIST_BENCH_BACKEND=cpu_x86`.
