/*
 * src/backends/cpu_x86/linear_q4k.c — cpu_x86 Q4_K M=1 (decode) wiring.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Per Q4_K weight:
 *   1. Allocate one heap-aligned blob for the W4A8 SoA: packed nibbles
 *      (n_in/2 bytes), per-block scales (n_in/32 fp32), per-block offsets
 *      (n_in/32 fp32), in that order; each row contributes n_blocks of
 *      each. The blob is owned by the weight (GEIST_W_AUX_HEAP_OWNED) so
 *      the engine frees it at model destroy.
 *   2. Predecode via q4k_to_w4a8_row, row-major over n_out rows.
 *   3. Grow the per-backend activation scratch (int8 acts + sum_a) to
 *      cover n_in if needed — at model-load, never in the hot path.
 *   4. Install cpu_x86_linear_q4k_m1 into w->linear_m1.
 *
 * The hot-path kernel reconstructs the SoA pointers from w->aux_fp32 +
 * w->n_in + w->n_out arithmetic; no per-call allocation, no branching.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "linear_q4k.h"

#include "backend_state.h"
#include "kernel_bf16_gemm.h"
#include "kernel_q4kx8_gemm.h" /* Phase 3 lane-parallel Q4_Kx8 GEMV */
#include "kernel_w4a8.h"
#include "kernel_w8a8.h" /* sum_a sized for W8A8 to also cover Q6_K */
#include "q4k_to_q4kx8.h"
#include "q4k_to_w4a8.h"
#include "q8_kx4.h"

#include "heap.h"
#include "quant.h" /* Q4_K_BLOCK_ELEMS / Q4_K_BLOCK_BYTES / dequant_q4_K_row */

#include <geist_backend.h>

#include <immintrin.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* Layout sizes for one weight (row-major SoA). Per-row blocks = n_in/32. */
static inline size_t weights_bytes_per_row(size_t n_in) {
    return (n_in / W4A8_BLOCK_ELEMS) * W4A8_BLOCK_BYTES_WEIGHTS;
}
static inline size_t scales_count_per_row(size_t n_in) {
    return n_in / W4A8_BLOCK_ELEMS;
}

/* SoA pointer reconstruction. The blob layout is:
 *   [weights      : n_out * weights_bytes_per_row(n_in)]      (W4A8 for m=1)
 *   [w_scales     : n_out * scales_count_per_row(n_in) fp32]
 *   [w_offsets    : n_out * scales_count_per_row(n_in) fp32]
 *   [q4kx8        : (n_out/8) * (n_in/256) * sizeof(block_q4_Kx8)] (Phase 3 prefill)
 * Aligned by construction. */
static size_t q4kx8_bytes_total(size_t n_in, size_t n_out) {
    return (n_out / 8) * (n_in / Q4_K_BLOCK_ELEMS) * sizeof(struct block_q4_Kx8);
}

static size_t blob_total_bytes(size_t n_in, size_t n_out) {
    const size_t weights_total = n_out * weights_bytes_per_row(n_in);
    const size_t scales_total  = n_out * scales_count_per_row(n_in) * sizeof(float);
    return weights_total + 2 * scales_total + q4kx8_bytes_total(n_in, n_out);
}

static void blob_pointers(const uint8_t              *blob,
                          size_t                      n_in,
                          size_t                      n_out,
                          const uint8_t             **weights_out,
                          const float               **scales_out,
                          const float               **offsets_out,
                          const struct block_q4_Kx8 **q4kx8_out) {
    const size_t weights_bytes = n_out * weights_bytes_per_row(n_in);
    const size_t scales_count  = n_out * scales_count_per_row(n_in);

    *weights_out = blob;
    *scales_out  = (const float *) (blob + weights_bytes);
    *offsets_out = *scales_out + scales_count;
    *q4kx8_out   = (const struct block_q4_Kx8 *) (*offsets_out + scales_count);
}

/* Grow the backend's activation scratch buffers to cover at least n_in
 * elements. Called only at resolve_weight time. Returns OK or E_OOM. */
static enum geist_status grow_scratch(struct cpu_x86_state *st, size_t n_in) {
    if (n_in <= st->scratch_cap) {
        return GEIST_OK;
    }
    int8_t *new_acts = heap_alloc_aligned(n_in * sizeof(int8_t), OPTIMAL_ALIGNMENT);
    if (new_acts == nullptr) {
        return GEIST_E_OOM;
    }
    /* Size sum_a for the SMALLEST block granularity — W8A8 (16) — so the
     * buffer covers both Q4_K (W4A8, 32-elem blocks) and Q6_K (W8A8) callers
     * sharing this scratch. */
    const size_t n_blocks  = n_in / W8A8_BLOCK_ELEMS;
    int32_t     *new_sum_a = heap_alloc_aligned(n_blocks * sizeof(int32_t), OPTIMAL_ALIGNMENT);
    if (new_sum_a == nullptr) {
        safe_free((void **) &new_acts);
        return GEIST_E_OOM;
    }
    safe_free((void **) &st->acts_scratch);
    safe_free((void **) &st->sum_a_scratch);
    st->acts_scratch  = new_acts;
    st->sum_a_scratch = new_sum_a;
    st->scratch_cap   = n_in;
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_x86_linear_q4k_resolve(struct cpu_x86_state *st,
                                                           struct geist_weight  *w) {
    if (st == nullptr || w == nullptr || w->n_in <= 0 || w->n_out <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    if (n_in % Q4_K_BLOCK_ELEMS != 0) {
        return GEIST_E_INVALID_ARG;
    }

    /* SoA blob: W4A8 nibbles + scales + offsets + packed bf16 cache. */
    const size_t blob_bytes = blob_total_bytes(n_in, n_out);

    uint8_t *blob = heap_alloc_aligned(blob_bytes, OPTIMAL_ALIGNMENT);
    if (blob == nullptr) {
        return GEIST_E_OOM;
    }
    const uint8_t             *blob_w_const;
    const float               *blob_s_const;
    const float               *blob_o_const;
    const struct block_q4_Kx8 *blob_q4kx8_const;
    blob_pointers(
            blob, n_in, n_out, &blob_w_const, &blob_s_const, &blob_o_const, &blob_q4kx8_const);
    uint8_t             *blob_w     = (uint8_t *) blob_w_const;
    float               *blob_s     = (float *) blob_s_const;
    float               *blob_o     = (float *) blob_o_const;
    struct block_q4_Kx8 *blob_q4kx8 = (struct block_q4_Kx8 *) blob_q4kx8_const;

    const size_t   q4k_row_bytes = (n_in / Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES;
    const size_t   w_row_bytes   = weights_bytes_per_row(n_in);
    const size_t   s_row_count   = scales_count_per_row(n_in);
    const uint8_t *q4k_raw       = (const uint8_t *) w->raw;
    for (size_t m = 0; m < n_out; m++) {
        q4k_to_w4a8_row(n_in,
                        q4k_raw + m * q4k_row_bytes,
                        blob_w + m * w_row_bytes,
                        blob_s + m * s_row_count,
                        blob_o + m * s_row_count);
    }

    /* Repack into Q4_Kx8 interleaved layout for prefill. n_out must be a
     * multiple of 8 — every Gemma 4 Q4_K body matrix satisfies this. */
    if (n_out % 8 == 0) {
        q4k_to_q4kx8_matrix(n_in, n_out, q4k_raw, blob_q4kx8);
    }

    /* Grow scratch to cover this n_in. */
    enum geist_status scratch_st = grow_scratch(st, n_in);
    if (scratch_st != GEIST_OK) {
        safe_free((void **) &blob);
        return scratch_st;
    }

    /* aux_fp32 reinterpreted as the blob pointer; engine frees it on
     * model destroy via heap_free / safe_free (GEIST_W_AUX_HEAP_OWNED). */
    w->aux_fp32 = (const float *) blob;
    w->aux_n    = (int32_t) blob_bytes;
    w->flags |= GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_BACKEND_REPACK;
    w->linear_m1 = cpu_x86_linear_q4k_m1;
    w->linear_mN = cpu_x86_linear_q4k_mN;
    return GEIST_OK;
}

void cpu_x86_linear_q4k_m1(const float               *x,
                           const struct geist_weight *w,
                           struct geist_backend      *be,
                           float                     *y) {
    struct cpu_x86_state *st               = (struct cpu_x86_state *) be->state;
    const size_t          n_in             = (size_t) w->n_in;
    const size_t          n_out            = (size_t) w->n_out;
    const size_t          n_blocks_per_row = n_in / W4A8_BLOCK_ELEMS;

    const uint8_t             *weights;
    const float               *w_scales;
    const float               *w_offsets;
    const struct block_q4_Kx8 *q4kx8;
    blob_pointers(
            (const uint8_t *) w->aux_fp32, n_in, n_out, &weights, &w_scales, &w_offsets, &q4kx8);

    /* Decode over the compact Q4_Kx8 layout when its blob is built (n_out a
     * multiple of 8 — every Q4_K body matrix). The 8-cell lane-parallel GEMV
     * reduces once per tile (no per-block hsum) and reads 0.56 B/wt vs W4A8's
     * 0.75 — both the compute and bandwidth limits of decode
     * (docs/LINUX_X86_PERF_PROFILE.md). */
    if (n_out % 8 == 0) {
        q4kx8_gemv_m1(n_out, n_in, x, q4kx8, y);
        return;
    }

    /* Per-row activation quantization → int8 acts + per-block sum_a. */
    const float scale_x = w4a8_quantize_acts_row(n_in, x, st->acts_scratch, st->sum_a_scratch);

    /* Multi-row GEMV fallback (n_out not a multiple of 8). OMP-parallel. */
    w4a8_gemv(n_out,
              n_blocks_per_row,
              weights,
              w_scales,
              w_offsets,
              st->acts_scratch,
              st->sum_a_scratch,
              scale_x,
              y);
}

/* These helpers + the tiled mN below use AVX-512+VNNI intrinsics. The
 * `target` attribute tells gcc to allow them in this TU even though
 * the file is compiled at baseline -march=x86-64-v3; the dispatcher
 * only calls cpu_x86_linear_q4k_mN on hosts whose cpuid confirms VNNI
 * support, so there is no SIGILL risk on AVX2-only CPUs. */
#define VNNI_TARGET "avx2,avx512f,avx512bw,avx512dq,avx512vl,avx512vnni"

/* Horizontal sum of 8 fp32 lanes → one fp32. */
__attribute__((target(VNNI_TARGET))) static inline float hsum_f32_avx(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128       s  = _mm_add_ps(lo, hi);
    s               = _mm_hadd_ps(s, s);
    s               = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/* Unpack 16 packed Q4_K bytes (= one 32-element block) into 32 unsigned
 * int8 values in element order, returned in a 256-bit register. The
 * caller will feed the result to VPDPBUSD against int8 activations. */
__attribute__((target(VNNI_TARGET))) static inline __m256i
unpack_w4a8_block_to_u8(const uint8_t weight_packed[16]) {
    const __m128i packed_128 = _mm_loadu_si128((const __m128i *) weight_packed);
    const __m256i packed_256 = _mm256_set_m128i(packed_128, packed_128);
    const __m256i lo_mask    = _mm256_set1_epi8(0x0F);
    const __m256i lo_nibs    = _mm256_and_si256(packed_256, lo_mask);
    const __m256i hi_nibs    = _mm256_and_si256(_mm256_srli_epi16(packed_256, 4), lo_mask);
    const __m256i nibs_lo    = _mm256_unpacklo_epi8(lo_nibs, hi_nibs);
    const __m256i nibs_hi    = _mm256_unpackhi_epi8(lo_nibs, hi_nibs);
    return _mm256_permute2x128_si256(nibs_lo, nibs_hi, 0x20);
}

/* Phase 1b Step 3 (c): tiled int8 GEMM for Q4_K prefill (M>1).
 *
 * Outer OMP parallelization over row-tiles. Per tile, hold M_TILE=4
 * fp32 accumulators for each of m output cells (in a stack-resident
 * acc[]), then sweep all K-blocks: per block, load 4 unpacked weight
 * blocks + 4 scale/offset pairs, iterate j∈[0,m) loading one activation
 * block + one sum_a value, and do M_TILE VPDPBUSDs + scale/offset
 * applications per j. Each weight block is read once per row-tile (vs
 * once per m tokens in the per-row gemv path), and each activation
 * block is read M_TILE times instead of n_out times — the standard
 * tiled-GEMM amortization. */
/* Phase 3: Q4_Kx8 lane-parallel GEMM via VPMADDUBSW. 8 cells per inst
 * (vs our previous 1 cell per VPDPBUSD) — the 8× compute-density lift
 * identified empirically in docs/LINUX_X86_PERF_PROFILE.md (IPC 0.47 →
 * target 3.01). The per-row acts get quantized to Q8_Kx4 (4 m-rows
 * interleaved in 8-byte stripes) in heap scratch; the GEMV-style
 * AVX kernel handles the 8-cell tile per (m, n_tile) call.
 *
 * Fallback: if n_out is not divisible by 8 (no Gemma 4 matrix is, this
 * is purely defensive), drop to the bf16 path. */
void cpu_x86_linear_q4k_mN(const float               *x,
                           const struct geist_weight *w,
                           size_t                     m,
                           struct geist_backend      *be,
                           float                     *y) {
    (void) be;
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;

    const uint8_t             *weights_unused;
    const float               *scales_unused;
    const float               *offsets_unused;
    const struct block_q4_Kx8 *q4kx8;
    blob_pointers((const uint8_t *) w->aux_fp32,
                  n_in,
                  n_out,
                  &weights_unused,
                  &scales_unused,
                  &offsets_unused,
                  &q4kx8);
    (void) weights_unused;
    (void) scales_unused;
    (void) offsets_unused;

    if (n_out % 8 != 0 || m % 4 != 0) {
        /* Defensive scalar fallback for shapes the Q4_Kx8 kernel doesn't
         * cover. Gemma 4 never hits this. */
        for (size_t row = 0; row < m; row++) {
            cpu_x86_linear_q4k_m1(x + row * n_in, w, be, y + row * n_out);
        }
        return;
    }

    /* Quantize acts into block_q8_Kx4 scratch — 4 rows interleaved per
     * super-block. Allocated per call from heap.h (small: m/4 × n_in/256
     * × ~1.2 KB ≈ 36 KB for m=128, n_in=1536). */
    const size_t         n_super_k   = n_in / 256;
    const size_t         q8kx4_count = (m / 4) * n_super_k;
    struct block_q8_Kx4 *acts =
            heap_alloc_aligned(q8kx4_count * sizeof(struct block_q8_Kx4), OPTIMAL_ALIGNMENT);
    if (acts == nullptr) {
        for (size_t row = 0; row < m; row++) {
            cpu_x86_linear_q4k_m1(x + row * n_in, w, be, y + row * n_out);
        }
        return;
    }
    for (size_t mt = 0; mt < m / 4; mt++) {
        quantize_q8_Kx4(n_in, x + mt * 4 * n_in, acts + mt * n_super_k);
    }

    q4kx8_gemm_avx512(m, n_out, n_in, acts, q4kx8, y);
    safe_free((void **) &acts);
}
