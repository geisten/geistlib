# geist Benchmarks

geist vs **llama.cpp** and Microsoft's **bitnet.cpp**, CPU-only, the **identical**
GGUF on both engines. This page details the `Gemma 4 E2B-it Q4_K_M` sweep on the
two ARM/Apple reference machines; **ternary BitNet** (Pi 5) and the **AMD x86-64
(AVX-512)** results are linked under [Files](#files).

| | [Raspberry Pi 5](BENCHMARK_PI5.md) | [Apple M1 Max](BENCHMARK.md) |
| :-- | :-- | :-- |
| Role | **primary optimization target** (edge) | dev box / Apple-AMX reference |
| Quant matmul path | native int8 (W4A8) | native int8 |
| Dense fp32 path | native NEON (BLAS-free) | Accelerate / **AMX** |
| Measurement | quiesced + cool → **mean of 10** | live desktop → **best of 10** |

> **TL;DR** — On **Apple Silicon** geist wins prefill at *every* length and the
> lead **widens** with context (1.48× at 1024 tokens). On the **Pi 5** llama.cpp's
> OpenBLAS prefill leads geist by ~10–15 % across the sweep (both flat, ~37–39 vs
> ~32–34 t/s); **decode ties on the sweep and edges ahead** with the spec-decode
> head (7.5 vs 6.8 t/s). geist's Pi value is the
> dependency-free static binary + decode parity, not raw prefill — the A76's
> mature OpenBLAS fp32 path is the bar geist is still chasing there.

---

## Apple M1 Max (8 P-cores) — prefill (tokens/s, higher is better)

| seq_len | llama.cpp `-ngl 0` | geist | winner |
| ---: | :---: | :---: | :--- |
|  128 | 141 | **164** | **geist 1.16×** |
|  256 | 147 | **161** | **geist 1.10×** |
|  512 | 128 | **150** | **geist 1.17×** |
| 1024 |  97 | **144** | **geist 1.48×** |

```
prefill t/s   (each █ ≈ 10 t/s)            geist stays flat ·· llama drops off
 geist  128 ████████████████ 164    llama  128 ██████████████ 141
        256 ████████████████ 161           256 ███████████████ 147
        512 ███████████████ 150            512 █████████████ 128
       1024 ██████████████ 144            1024 ██████████ 97
```

**Decode:** ≈ par (~26 t/s both). On a live desktop the 16-token decode window is
too jitter-prone to rank — see the Pi for the controlled decode result.

geist's dense-fp32 path here is **Accelerate/AMX** (Apple's matrix coprocessor),
which holds ~flat from 128 → 1024 tokens (164 → 144), while llama.cpp's CPU-only
path degrades sharply past 256 (147 → 97). [Full write-up + decode-kernel
investigation →](BENCHMARK.md)

## Raspberry Pi 5 (Cortex-A76, 4 cores) — prefill (tokens/s, higher is better)

Identical protocol for both engines — **prefill-only, 8 reps, cool start
(<56 °C)**, t=4 — back-to-back in one session (the Pi throttles when heat-soaked;
see the write-up):

| seq_len | llama.cpp (OpenBLAS) | geist | winner |
| ---: | :---: | :---: | :--- |
|  128 | **37.4** | 34.8 | llama 1.07× |
|  256 | **39.4** | 34.2 | llama 1.15× |
|  512 | **37.6** | 32.9 | llama 1.14× |
| 1024 | **35.9** | 31.5 | llama 1.14× |

```
prefill t/s   (each █ ≈ 2.5 t/s)           both flat ·· llama ~10-15% ahead
 geist  128 ██████████████ 35       llama  128 ███████████████ 37
        256 ██████████████ 34               256 ████████████████ 39
        512 █████████████ 33                512 ███████████████ 38
       1024 █████████████ 31              1024 ██████████████ 36
```

**Decode:** on the prefill-sweep harness it's ≈ par (geist 6.9 vs llama.cpp 6.8,
best at 3 threads, memory-bound for both); with the **spec-decode head** geist's
end-to-end decode is **7.5 t/s** — the headline number (see the total-tps section
of the write-up). geist's prefill curve is flat thanks to a parallelized
O(n²) attention core (it used to fade to 23 t/s at 1024), but llama's mature
OpenBLAS sgemm still leads Pi prefill by ~10–15 %.
[Full write-up + the thermal correction + thread placement →](BENCHMARK_PI5.md)

---

## Reading the numbers — why the curves look the way they do

Both engines reach the Q4_K matmuls differently, and both prefill curves are now
**flat** with context on the Pi — they just sit at different heights:

- **geist runs prefill on a native int8 (W4A8) kernel** — low fixed overhead. Its
  attention is O(n²), but the SDPA core is now **parallelized across cores** (it
  used to be serial, which made the curve fade to 23 t/s at 1024); spread over 4
  A76 cores the per-token rise is absorbed, so geist's curve is flat at ~32–34 t/s.
- **llama.cpp dequantizes to fp32 and calls OpenBLAS sgemm**, a decades-tuned path
  that is genuinely fast on the A76 (which lacks `i8mm`, so geist can't use SMMLA
  to pull ahead). llama's curve is flat and ~10–15 % *higher* (~37–39 t/s) — it
  wins Pi prefill at every length. Closing that gap is geist's open A76 work.
- **On the M1 Max the picture flips to geist's favour at every length** because
  geist's dense-fp32 path is **Accelerate/AMX**, which scales flat to long
  sequences, while llama.cpp's CPU-only path (`-ngl 0`) *degrades* sharply past 256.
- **Decode is memory-bandwidth-bound** for both (streaming the weights per token
  dwarfs the compute), so the kernel differences wash out and the two tie (~6.8 t/s
  on the Pi).

> geist's flat Pi curve is recent: profiling showed the O(n²) attention stage
> climbing 22 %→45 % of prefill with its SDPA **core single-threaded**;
> parallelizing it (bit-exact) lifted pp512 +22 % and pp1024 +35 %. Separately,
> the llama Pi numbers here were re-measured after a thermal-throttling artifact
> had understated them (pp128 22→37). See
> [BENCHMARK_PI5.md](BENCHMARK_PI5.md).

**How to dig deeper.** To attribute the scaling, profile prefill *phase-by-phase*
(attention vs FFN-matmul vs PLE) at each seq_len rather than as one number: build
with `-DGEIST_PROFILE_QUANT` (per-kernel ns counters, auto-reported at exit) and
compare the attention fraction at 128 vs 1024. For llama.cpp, `llama-bench -p <n>`
plus `perf stat` (or Instruments on macOS) isolates where its CPU path loses time
past 256 tokens.

---

## Methodology — and why the two machines use different aggregation

Both machines run the **same model and quantization**, both engines CPU-only, each
at its best thread count, after a **discarded warm-up run** (the runtime pages
weights resident and spins up the OpenMP pool, so timings reflect steady state, not
cold-start). llama.cpp build `d05fe1d`.

- **Raspberry Pi 5 — `mean of 10`, cool start.** A dedicated headless box,
  genuinely quiesced (load 0.0). The mean is meaningful: spread is <2 % run-to-run.
  Crucially, **both engines are started from a cool baseline (<56 °C)** — a 4.6 B
  prefill drives this passively-cooled board to ~78 °C and trips the soft temp
  limit in under a minute, so benchmarking one engine right after the other
  throttles the second (this is exactly what understated llama's pp128 to 22 t/s
  in an earlier revision — cool, it is ~37).
- **Apple M1 Max — `best of 10`.** A developer workstation that *cannot* be
  quiesced while in use (WindowServer, browser, IDE all contend for the P-cores).
  On a contended box the **mean** is dominated by interference spikes (±20 %
  run-to-run), so we report the **best** of 10 repeats — the least-interrupted run,
  which approximates the uncontended ceiling and is stable across independent
  campaigns. Both engines use best-of here, so the comparison stays
  apples-to-apples.

**Always measure on a quiesced box.** On the 4-core Pi a single stray process
eating one core inverts the 4-thread numbers. Reproduce with `bench_perf_sweep`
(geist) and `llama-bench` (reference) — see [BENCHMARKING.md](BENCHMARKING.md) for
the exact commands, the correctness gate (bit-identical greedy output on the same
weights), and the quality/PPL caveats.

## Files

- **[BENCHMARK_PI5.md](BENCHMARK_PI5.md)** — Raspberry Pi 5 (Cortex-A76): the edge
  target. Full sweep, thread placement, the int8-vs-OpenBLAS analysis.
- **[BENCHMARK.md](BENCHMARK.md)** — Apple M1 Max: the AMX reference + the
  auto-recorded `make bench-small`/`bench-detailed` results table + the
  decode-kernel investigation.
- **[TERNARY_BITNET.md](TERNARY_BITNET.md)** — ternary BitNet b1.58 on the Pi 5:
  geist **17.4 t/s decode vs bitnet.cpp's 8.2** (~2×), plus the Cougar/bitnet.cpp
  head-to-head and the spec-head lm_head trick.
- **[BENCHMARK_X86.md](BENCHMARK_X86.md)** — AMD Ryzen 9 9950X (AVX-512): Gemma &
  Llama vs llama.cpp, and BitNet vs bitnet.cpp (prefill +30 %, decode +38 %).
- **[BENCHMARKING.md](BENCHMARKING.md)** — how to produce trustworthy numbers
  (reproduce, compare-ref, quality/MMLU procedures).
