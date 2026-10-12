/*
 * src/backends/cpu_x86/kernel_q4k_tiled_avx512_vnni.c — tiled int8 Q4_K
 * GEMM, AVX-512 VNNI (#18).
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Port of llama.cpp's tiled k-quant matmul (ggml/src/ggml-cpu/tiled/
 * tiled.cpp and tiled-kernel.cpp, introduced by d834d44e6 "ggml-cpu: tiled
 * mul_mat for k-quants", at d8123504): the Q4_K instance (32-element
 * subblocks, mins, no bias), its 8x16 VPDPBUSD microkernel, the in-place
 * activation interleave and the 256x256 window driver. Original code
 * Copyright (c) 2023-2026 The ggml authors, MIT-licensed. Adapted to geist:
 * C, the Q4_Kx8 weight repack and the Q8_Kx4 activations as sources (no
 * second copy of the weights), geist_par_for and the backend-owned
 * per-thread workspace.
 *
 * Per (weight window, activation window) of up to 256 x 256 and per
 * 256-element K block:
 *   - the weight rows unpack to uint8 codes (0..15) with int32 subblock
 *     scales and mins and float d / dmin;
 *   - the activation rows unpack to int8 codes with int32 per-16 sums and
 *     float d, then each 16-row band is transposed in place into the
 *     [k/4 % 16][k/4 / 16][row][4] order VPDPBUSD reads;
 *   - 8x16 bands accumulate sum_k w*a exactly in int32 per 32-element
 *     subblock, apply the integer scales and mins, and convert to float
 *     once per K block: acc += a_d * (w_d * s1 - w_dmin * s2).
 * Every TU in this file may be EVEX (mk/backend-cpu_x86.mk); the caller
 * decides with vnni_tiles_usable().
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_q4k_tiled.h"

#include "par.h"
#include "quant.h"

#include <immintrin.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

/* NSUB_MAX: subblocks per K block, 16 for Q6_K (16 elements), 8 for Q4_K
 * (32 elements), which uses the first half of each row's slots. */
enum { TILE = 256, MICRO = 16, NSUB_MAX = 16 };

/* One thread's tiles. Rows past a window's edge hold stale or zero values;
 * the microkernel computes them and the store drops them. */
struct q4k_tiled_ws {
    alignas(64) uint8_t w_q[TILE * TILE];           /* [row][k] codes 0..15 */
    alignas(64) int32_t w_scale[TILE * NSUB_MAX];   /* [row][subblock] */
    alignas(64) int32_t w_min[TILE * NSUB_MAX];     /* [row][subblock], Q4_K only */
    alignas(64) float w_d[TILE];                    /* per row */
    alignas(64) float w_dmin[TILE];                 /* per row */
    alignas(64) int8_t a_q[TILE * TILE];            /* [row][k], then banded */
    alignas(64) int32_t a_bsum[(TILE / 16) * TILE]; /* [group of 16][row] */
    alignas(64) float a_d[TILE];                    /* per row */
    alignas(64) float acc[TILE * TILE];             /* [weight row][act row] */
};

size_t q4k_tiled_ws_bytes(void) {
    return sizeof(struct q4k_tiled_ws);
}

/* Weight rows [r0, r0 + n) of K block kb from the Q4_Kx8 repack: row r of
 * octet o is lane r % 8 of W[o * n_super + kb]. The lane's 16 eight-byte
 * stripes are its 128 bytes of nibbles, and the 6-bit scales and mins are
 * the inverse of make_block_q4_kx8 (q4k_to_q4kx8.c). */
static void unpack_w(size_t                     n_super,
                     size_t                     kb,
                     size_t                     r0,
                     int                        n,
                     const struct block_q4_Kx8 *W,
                     struct q4k_tiled_ws       *ws) {
    for (int r = 0; r < n; r++) {
        const size_t               row  = r0 + (size_t) r;
        const int                  lane = (int) (row % 8);
        const struct block_q4_Kx8 *b    = &W[(row / 8) * n_super + kb];
        ws->w_d[r]                      = fp16_to_fp32(b->d[lane]);
        ws->w_dmin[r]                   = fp16_to_fp32(b->dmin[lane]);
        for (int sb = 0; sb < 8; sb++) {
            const uint8_t *g = b->scales + sb * 12;
            int32_t        s, m;
            if (lane < 4) {
                s = g[lane] & 63;
                m = g[4 + lane] & 63;
            } else {
                s = (g[4 + lane] & 15) | ((g[lane - 4] >> 6) << 4);
                m = (g[4 + lane] >> 4) | ((g[lane] >> 6) << 4);
            }
            ws->w_scale[r * NSUB_MAX + sb] = s;
            ws->w_min[r * NSUB_MAX + sb]   = m;
        }
        /* 128 bytes of this lane: stripe s holds bytes 8s..8s+7. */
        uint8_t qs[128];
        for (int s = 0; s < 16; s++) {
            memcpy(qs + 8 * s, b->qs + (size_t) (8 * s + lane) * 8, 8);
        }
        /* Q4_K order: 32 bytes give elements c*64 + [0, 32) in their low
         * nibbles and c*64 + [32, 64) in their high nibbles. */
        uint8_t      *q    = &ws->w_q[(size_t) r * TILE];
        const __m256i mask = _mm256_set1_epi8(0x0F);
        for (int c = 0; c < 4; c++) {
            const __m256i v = _mm256_loadu_si256((const __m256i *) (qs + 32 * c));
            _mm256_storeu_si256((__m256i *) (q + 64 * c), _mm256_and_si256(v, mask));
            _mm256_storeu_si256((__m256i *) (q + 64 * c + 32),
                                _mm256_and_si256(_mm256_srli_epi16(v, 4), mask));
        }
    }
}

/* Weight rows [r0, r0 + n) of K block kb from the GGUF Q6_K bytes: codes
 * 0..63 (the value is code - 32) from 4 low bits (ql) and 2 high bits
 * (qh), in dequant_q6_K_row's element order, and the signed 8-bit scale of
 * each 16-element subblock (llama.cpp's Q6_K tiled unpack). */
static void unpack_w_q6k(size_t                     n_super,
                         size_t                     kb,
                         size_t                     r0,
                         int                        n,
                         const struct block_q6_K_t *W,
                         struct q4k_tiled_ws       *ws) {
    const __m256i lo4 = _mm256_set1_epi8(0x0F), lo2 = _mm256_set1_epi8(0x03);
    for (int r = 0; r < n; r++) {
        const struct block_q6_K_t *b = &W[(r0 + (size_t) r) * n_super + kb];
        ws->w_d[r]                   = fp16_to_fp32(b->d);
        for (int sb = 0; sb < 16; sb++) {
            ws->w_scale[r * NSUB_MAX + sb] = b->scales[sb];
        }
        uint8_t *q = &ws->w_q[(size_t) r * TILE];
        for (int half = 0; half < 2; half++) {
            uint8_t      *out = q + 128 * half;
            const __m256i l0  = _mm256_loadu_si256((const __m256i *) (b->ql + 64 * half));
            const __m256i l1  = _mm256_loadu_si256((const __m256i *) (b->ql + 64 * half + 32));
            const __m256i h   = _mm256_loadu_si256((const __m256i *) (b->qh + 32 * half));
            /* element l of each 32-run: ql low/high nibble | qh bit pair << 4 */
            const __m256i h0 = _mm256_slli_epi16(_mm256_and_si256(h, lo2), 4);
            const __m256i h1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 2), lo2), 4);
            const __m256i h2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 4), lo2), 4);
            const __m256i h3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 6), lo2), 4);
            _mm256_storeu_si256((__m256i *) (out + 0),
                                _mm256_or_si256(_mm256_and_si256(l0, lo4), h0));
            _mm256_storeu_si256((__m256i *) (out + 32),
                                _mm256_or_si256(_mm256_and_si256(l1, lo4), h1));
            _mm256_storeu_si256(
                    (__m256i *) (out + 64),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), lo4), h2));
            _mm256_storeu_si256(
                    (__m256i *) (out + 96),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), lo4), h3));
        }
    }
}

/* Activation rows [r0, r0 + n) of K block kb from Q8_Kx4: row r is lane
 * r % 4 of X[(r / 4) * n_super + kb], whose 64-element subblock sb keeps
 * the lane's 8-element stripe i at qs[sb * 256 + i * 32 + lane * 8].
 * Rows up to the next multiple of 16 are zeroed. */
static void unpack_a(size_t                     n_super,
                     size_t                     kb,
                     size_t                     r0,
                     int                        n,
                     const struct block_q8_Kx4 *X,
                     struct q4k_tiled_ws       *ws) {
    const int n_pad = (n + MICRO - 1) & ~(MICRO - 1);
    for (int r = 0; r < n_pad; r++) {
        int8_t *q = &ws->a_q[(size_t) r * TILE];
        if (r >= n) {
            memset(q, 0, TILE);
            for (int g = 0; g < 16; g++) {
                ws->a_bsum[g * TILE + r] = 0;
            }
            ws->a_d[r] = 0.0f;
            continue;
        }
        const size_t               row  = r0 + (size_t) r;
        const int                  lane = (int) (row % 4);
        const struct block_q8_Kx4 *b    = &X[(row / 4) * n_super + kb];
        for (int sb = 0; sb < 4; sb++) {
            for (int i = 0; i < 8; i++) {
                memcpy(q + sb * 64 + i * 8, b->qs + sb * 256 + i * 32 + lane * 8, 8);
            }
        }
        for (int g = 0; g < 16; g++) {
            ws->a_bsum[g * TILE + r] = b->bsums[lane * 16 + g];
        }
        ws->a_d[r] = b->d[lane];
    }
}

/* Transpose one 16-row band of activation codes in place from [row][k]
 * into [kg % 16][kg / 16][row][4], kg = k / 4: four 16x16 int32
 * transposes per band (llama.cpp's tiled_repack_src1, VNNI branch). */
static void band_interleave(int row0, struct q4k_tiled_ws *ws) {
    uint8_t *base = (uint8_t *) ws->a_q + (size_t) row0 * TILE;
    for (int t = 0; t < TILE / (MICRO * 4); t++) {
        uint8_t *cbase = base + t * (MICRO * 4);
        __m512i  v[16];
        for (int r = 0; r < 16; r++) {
            v[r] = _mm512_load_si512((const void *) (cbase + r * TILE));
        }
        {
            const __m512i a_idx =
                    _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
            const __m512i b_idx =
                    _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
            for (int i = 0; i < 16; i += 2) {
                const __m512i a = v[i], b = v[i + 1];
                v[i]     = _mm512_permutex2var_epi32(a, a_idx, b);
                v[i + 1] = _mm512_permutex2var_epi32(a, b_idx, b);
            }
        }
        {
            const __m512i a_idx =
                    _mm512_setr_epi32(0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29);
            const __m512i b_idx =
                    _mm512_setr_epi32(2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31);
            for (int i = 0; i < 16; i += 4) {
                for (int j = 0; j < 2; j++) {
                    const __m512i a = v[i + j], b = v[i + j + 2];
                    v[i + j]     = _mm512_permutex2var_epi32(a, a_idx, b);
                    v[i + j + 2] = _mm512_permutex2var_epi32(a, b_idx, b);
                }
            }
        }
        {
            const __m512i a_idx =
                    _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
            const __m512i b_idx =
                    _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
            for (int i = 0; i < 16; i += 8) {
                for (int j = 0; j < 4; j++) {
                    const __m512i a = v[i + j], b = v[i + j + 4];
                    v[i + j]     = _mm512_permutex2var_epi32(a, a_idx, b);
                    v[i + j + 4] = _mm512_permutex2var_epi32(a, b_idx, b);
                }
            }
        }
        {
            const __m512i a_idx =
                    _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
            const __m512i b_idx =
                    _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
            for (int i = 0; i < 8; i++) {
                const __m512i a = v[i], b = v[i + 8];
                v[i]     = _mm512_permutex2var_epi32(a, a_idx, b);
                v[i + 8] = _mm512_permutex2var_epi32(a, b_idx, b);
            }
        }
        static const int col_order[16] = {0, 8, 1, 9, 4, 12, 5, 13, 2, 10, 3, 11, 6, 14, 7, 15};
        for (int g = 0; g < 16; g++) {
            _mm512_store_si512((void *) (cbase + col_order[g] * TILE), v[g]);
        }
    }
}

/* acc[i0 + t][j0 .. j0 + 16) += the K block's contribution, t < 8: exact
 * int32 subblock sums by VPDPBUSD (unsigned weight codes x signed
 * activation codes), the integer scale, min and code-bias corrections,
 * then one float epilogue (llama.cpp's tiled_run_micro_vnni_8x16). The
 * format is in the constant arguments, so every call site specializes:
 *   sub   elements per subblock (32 Q4_K, 16 Q6_K)
 *   mins  the format subtracts a scaled min (Q4_K)
 *   bias  the stored code is value + bias (32 Q6_K, 0 Q4_K). */
static inline __attribute__((always_inline)) void
micro_8x16(int sub, bool mins, int bias, int i0, int j0, struct q4k_tiled_ws *ws) {
    enum { ROWS = 8 };
    const int    nsub = TILE / sub, per16 = sub / 16, groups = sub / 4;
    const __m512 a_d = _mm512_load_ps(&ws->a_d[j0]);
    __m512i      s1[ROWS], s2[ROWS];
    for (int t = 0; t < ROWS; t++) {
        s1[t] = _mm512_setzero_si512();
        s2[t] = _mm512_setzero_si512();
    }
    const uint8_t *band = (const uint8_t *) ws->a_q + (size_t) j0 * TILE;
    for (int s = 0; s < nsub; s++) {
        __m512i acc[ROWS];
        for (int t = 0; t < ROWS; t++) {
            acc[t] = _mm512_setzero_si512();
        }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC unroll 8
#endif
        for (int g = 0; g < groups; g++) {
            const int     kg    = s * groups + g;
            const __m512i codes = _mm512_load_si512(
                    (const void *) (band + (kg % MICRO) * TILE + (kg / MICRO) * (MICRO * 4)));
            for (int t = 0; t < ROWS; t++) {
                uint32_t w4;
                memcpy(&w4, &ws->w_q[(size_t) (i0 + t) * TILE + kg * 4], 4);
                acc[t] = _mm512_dpbusd_epi32(acc[t], _mm512_set1_epi32((int) w4), codes);
            }
        }
        if (bias != 0) {
            /* sum (code - bias) * a = raw - bias * (activation sum) */
            __m512i asum = _mm512_load_si512((const void *) &ws->a_bsum[(s * per16) * TILE + j0]);
            for (int u = 1; u < per16; u++) {
                asum = _mm512_add_epi32(
                        asum,
                        _mm512_load_si512((const void *) &ws->a_bsum[(s * per16 + u) * TILE + j0]));
            }
            const __m512i corr = _mm512_mullo_epi32(asum, _mm512_set1_epi32(bias));
            for (int t = 0; t < ROWS; t++) {
                acc[t] = _mm512_sub_epi32(acc[t], corr);
            }
        }
        for (int t = 0; t < ROWS; t++) {
            s1[t] = _mm512_add_epi32(
                    s1[t],
                    _mm512_mullo_epi32(acc[t],
                                       _mm512_set1_epi32(ws->w_scale[(i0 + t) * NSUB_MAX + s])));
        }
    }
    /* s2 += min * (activation sum over the subblock): no VPDPBUSD
     * dependency, so it runs after the band and costs no accumulator
     * registers there. */
    if (mins) {
        for (int s = 0; s < nsub; s++) {
            __m512i asum = _mm512_load_si512((const void *) &ws->a_bsum[(s * per16) * TILE + j0]);
            for (int u = 1; u < per16; u++) {
                asum = _mm512_add_epi32(
                        asum,
                        _mm512_load_si512((const void *) &ws->a_bsum[(s * per16 + u) * TILE + j0]));
            }
            for (int t = 0; t < ROWS; t++) {
                s2[t] = _mm512_add_epi32(
                        s2[t],
                        _mm512_mullo_epi32(asum,
                                           _mm512_set1_epi32(ws->w_min[(i0 + t) * NSUB_MAX + s])));
            }
        }
    }
    for (int t = 0; t < ROWS; t++) {
        const int r      = i0 + t;
        __m512    result = _mm512_mul_ps(_mm512_cvtepi32_ps(s1[t]), _mm512_set1_ps(ws->w_d[r]));
        if (mins) {
            result = _mm512_fnmadd_ps(
                    _mm512_set1_ps(ws->w_dmin[r]), _mm512_cvtepi32_ps(s2[t]), result);
        }
        float *p = &ws->acc[(size_t) r * TILE + j0];
        _mm512_store_ps(p, _mm512_fmadd_ps(result, a_d, _mm512_load_ps(p)));
    }
}

/* Y[a0 + j][w0 + i] = acc[i][j], i < n_w, j < n_a. */
static void
store_window(int n_w, int n_a, size_t w0, size_t a0, size_t N, const float *acc, float *Y) {
    for (int j = 0; j < n_a; j++) {
        float *dst = Y + (a0 + (size_t) j) * N + w0;
        for (int i = 0; i < n_w; i++) {
            dst[i] = acc[(size_t) i * TILE + j];
        }
    }
}

enum tiled_format { TILED_Q4KX8, TILED_Q6K };

struct tiled_job {
    enum tiled_format          format;
    size_t                     M, N, n_super;
    size_t                     dw, da, nchunk_w;
    const struct block_q8_Kx4 *X;
    const void                *W; /* block_q4_Kx8 or block_q6_K_t */
    float                     *Y;
    struct cpu_x86_state      *st;
    atomic_bool                failed;
};

/* One chunk: weight rows [w_lo, w_hi) x activation rows [a_lo, a_hi), in
 * windows of 256 x 256. */
static void run_chunk(size_t                  w_lo,
                      size_t                  w_hi,
                      size_t                  a_lo,
                      size_t                  a_hi,
                      const struct tiled_job *j,
                      struct q4k_tiled_ws    *ws) {
    for (size_t a0 = a_lo; a0 < a_hi; a0 += TILE) {
        const int n_a = (int) (a_hi - a0 < TILE ? a_hi - a0 : TILE);
        for (size_t w0 = w_lo; w0 < w_hi; w0 += TILE) {
            const int n_w = (int) (w_hi - w0 < TILE ? w_hi - w0 : TILE);
            memset(ws->acc, 0, sizeof ws->acc);
            for (size_t kb = 0; kb < j->n_super; kb++) {
                const bool q6k = j->format == TILED_Q6K;
                if (q6k) {
                    unpack_w_q6k(j->n_super, kb, w0, n_w, j->W, ws);
                } else {
                    unpack_w(j->n_super, kb, w0, n_w, j->W, ws);
                }
                unpack_a(j->n_super, kb, a0, n_a, j->X, ws);
                for (int j0 = 0; j0 < n_a; j0 += MICRO) {
                    band_interleave(j0, ws);
                    for (int i0 = 0; i0 < n_w; i0 += MICRO) {
                        if (q6k) {
                            micro_8x16(16, false, 32, i0, j0, ws);
                            micro_8x16(16, false, 32, i0 + 8, j0, ws);
                        } else {
                            micro_8x16(32, true, 0, i0, j0, ws);
                            micro_8x16(32, true, 0, i0 + 8, j0, ws);
                        }
                    }
                }
            }
            store_window(n_w, n_a, w0, a0, j->N, ws->acc, j->Y);
        }
    }
}

static void tiled_chunks(void *ctx, size_t c0, size_t c1) {
    struct tiled_job         *j   = ctx;
    struct cpu_x86_workspace *cws = cpu_x86_ws_acquire_tiled(j->st, sizeof(struct q4k_tiled_ws));
    if (cws == nullptr) {
        atomic_store(&j->failed, true);
        return;
    }
    struct q4k_tiled_ws *ws = cws->tiled;
    for (size_t c = c0; c < c1; c++) {
        const size_t cw = c % j->nchunk_w, ca = c / j->nchunk_w;
        const size_t w_lo = cw * j->dw, a_lo = ca * j->da;
        const size_t w_hi = w_lo + j->dw < j->N ? w_lo + j->dw : j->N;
        const size_t a_hi = a_lo + j->da < j->M ? a_lo + j->da : j->M;
        if (w_lo < w_hi && a_lo < a_hi) {
            run_chunk(w_lo, w_hi, a_lo, a_hi, j, ws);
        }
    }
}

static bool tiled_gemm(enum tiled_format          format,
                       size_t                     M,
                       size_t                     N,
                       size_t                     K,
                       const struct block_q8_Kx4 *X,
                       const void                *W,
                       float                     *Y,
                       struct cpu_x86_state      *st) {
    /* Chunks of 256 x 256, halved while there are fewer than four per
     * thread (llama.cpp's driver), down to one microtile. */
    const size_t threads = geist_par_max_threads();
    size_t       chunk   = TILE;
    size_t       ncw = (N + chunk - 1) / chunk, nca = (M + chunk - 1) / chunk;
    while (ncw * nca < 4 * threads && chunk > MICRO) {
        chunk /= 2;
        ncw = (N + chunk - 1) / chunk;
        nca = (M + chunk - 1) / chunk;
    }
    struct tiled_job j = {
            .format   = format,
            .M        = M,
            .N        = N,
            .n_super  = K / 256,
            .dw       = chunk,
            .da       = chunk,
            .nchunk_w = ncw,
            .X        = X,
            .W        = W,
            .Y        = Y,
            .st       = st,
    };
    atomic_init(&j.failed, false);
    geist_par_for(ncw * nca, tiled_chunks, &j);
    return !atomic_load(&j.failed);
}

bool q4k_tiled_gemm_avx512_vnni(size_t                     M,
                                size_t                     N,
                                size_t                     K,
                                const struct block_q8_Kx4 *X,
                                const struct block_q4_Kx8 *W,
                                float                     *Y,
                                struct cpu_x86_state      *st) {
    return tiled_gemm(TILED_Q4KX8, M, N, K, X, W, Y, st);
}

bool q6k_tiled_gemm_avx512_vnni(size_t                     M,
                                size_t                     N,
                                size_t                     K,
                                const struct block_q8_Kx4 *X,
                                const struct block_q6_K_t *W,
                                float                     *Y,
                                struct cpu_x86_state      *st) {
    return tiled_gemm(TILED_Q6K, M, N, K, X, W, Y, st);
}
