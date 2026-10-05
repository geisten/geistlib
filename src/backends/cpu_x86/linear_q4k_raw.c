/*
 * src/backends/cpu_x86/linear_q4k_raw.c — cpu_x86 Q4_K and Q5_K linear on the
 * GGUF bytes (AVX2), no repack.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * linear_q4k.c repacks every Q4_K weight into a Q4_Kx8 or W4A8 blob of
 * about the same size, so the original bytes and the repack are resident
 * together (#577). This kernel reads the original block layout instead.
 * The activations are quantized once per call to int8 with one scale per
 * 256 elements (llama.cpp's Q8_K) plus each 32-element sub-block's integer
 * sum S_j. A Q4_K superblock is eight 32-element sub-blocks j with 6-bit
 * scale sc_j and min m_j under the fp16 d and dmin:
 *
 *   w = d sc_j q - dmin m_j,  q in 0..15
 *   sum(w x) = d dx sum_j sc_j P_j - dmin dx sum_j m_j S_j,  P_j = sum(q xq)
 *
 * maddubs(q, xq) pairs reach 2 * 15 * 127 = 3810 (no int16 saturation),
 * madd with sc_j (<= 63) makes int32 lanes, and the sum over j stays exact
 * in int32. Rows split across OpenMP threads; M>1 tiles NR tokens per pass
 * over a weight row so each block's nibbles are unpacked once per tile.
 *
 * Q5_K is the same superblock with a fifth bit per q from 32 qh bytes (bit
 * 2j for the low nibbles of sub-block pair j, 2j + 1 for the high ones), q
 * in 0..31: the kernels OR it in after the nibble unpack and are otherwise
 * shared. Q5_K had no x86 kernel at all before; it ran the generic
 * dequantize-to-fp32 path (#410).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "linear_q4k_raw.h"

#include "backend_state.h"

#include "checked.h"
#include "linear_ref.h"
#include "quant.h"
#include "quant_blocks.h"

#include <geist_backend.h>
#include <geist_types.h>

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

constexpr size_t QK  = Q4_K_BLOCK_ELEMS; /* 256 */
constexpr size_t NSB = QK / 32;          /* sub-blocks per superblock */
static_assert(Q4_K_BLOCK_ELEMS == 256 && Q5_K_BLOCK_ELEMS == 256,
              "eight 32-element sub-blocks per superblock");

/* Activation tile height of the M>1 kernel. */
constexpr size_t NR = 4;

/* One activation row to int8 blocks of 256 (d = amax / 127, round to
 * nearest-even) plus each 32-element sub-block's integer sum. */
static void quantize_row_q8_k(size_t nb, const float *x, int8_t *qx, float *dx, int32_t *sx) {
    const __m256  abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256i perm     = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i ones_u8  = _mm256_set1_epi8(1);
    const __m256i ones_16  = _mm256_set1_epi16(1);
    for (size_t b = 0; b < nb; b++) {
        const float *xb = x + b * QK;
        __m256       m  = _mm256_setzero_ps();
        for (size_t i = 0; i < QK; i += 8) {
            m = _mm256_max_ps(m, _mm256_and_ps(_mm256_loadu_ps(xb + i), abs_mask));
        }
        __m128 m4        = _mm_max_ps(_mm256_extractf128_ps(m, 1), _mm256_castps256_ps128(m));
        m4               = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4               = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        const float amax = _mm_cvtss_f32(m4);

        dx[b]              = amax / 127.0f;
        const __m256 scale = _mm256_set1_ps(amax > 0.0f ? 127.0f / amax : 0.0f);
        for (size_t c = 0; c < NSB; c++) {
            const float *xc = xb + c * 32;
            __m256i      i0 = _mm256_cvtps_epi32(
                    _mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(xc), scale),
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i1 = _mm256_cvtps_epi32(
                    _mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(xc + 8), scale),
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i2 = _mm256_cvtps_epi32(
                    _mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(xc + 16), scale),
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i3 = _mm256_cvtps_epi32(
                    _mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(xc + 24), scale),
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            i0 = _mm256_packs_epi32(i0, i1);
            i2 = _mm256_packs_epi32(i2, i3);
            i0 = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(i0, i2), perm);
            _mm256_storeu_si256((__m256i *) (qx + b * QK + c * 32), i0);
            const __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, i0), ones_16);
            __m128i       s4 =
                    _mm_add_epi32(_mm256_castsi256_si128(s32), _mm256_extracti128_si256(s32, 1));
            s4              = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, _MM_SHUFFLE(1, 0, 3, 2)));
            s4              = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, _MM_SHUFFLE(2, 3, 0, 1)));
            sx[b * NSB + c] = _mm_cvtsi128_si32(s4);
        }
    }
}

/* The eight 6-bit scales and mins of a superblock (get_scale_min_k4's
 * packing), as bytes: sc[0..7], mn[0..7]. */
static inline void
unpack_scales(const uint8_t packed[static 12], uint8_t sc[static 8], uint8_t mn[static 8]) {
    uint32_t u[3];
    memcpy(u, packed, sizeof u);
    const uint32_t k1 = 0x3f3f3f3fu, k2 = 0x0f0f0f0fu, k3 = 0x03030303u;
    const uint32_t s0 = u[0] & k1;
    const uint32_t s1 = (u[2] & k2) | (((u[0] >> 6) & k3) << 4);
    const uint32_t m0 = u[1] & k1;
    const uint32_t m1 = ((u[2] >> 4) & k2) | (((u[1] >> 6) & k3) << 4);
    memcpy(sc, &s0, 4);
    memcpy(sc + 4, &s1, 4);
    memcpy(mn, &m0, 4);
    memcpy(mn + 4, &m1, 4);
}

static inline float hsum_ps(__m256 s) {
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    s4        = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4        = _mm_add_ss(s4, _mm_movehdup_ps(s4));
    return _mm_cvtss_f32(s4);
}

/* One superblock format: Q4_K (4-bit q) or Q5_K (q gets a fifth bit from
 * qh). Both share d, dmin and the 12 packed scale bytes at the front. */
struct fmt {
    size_t stride; /* bytes per superblock */
    size_t qs;     /* offset of the 128 nibble bytes */
    size_t qh;     /* offset of the 32 high-bit bytes (Q5_K only) */
    bool   q5;
};
static constexpr struct fmt FMT_Q4_K = {sizeof(struct block_q4_K_t), 16, 0, false};
static constexpr struct fmt FMT_Q5_K = {sizeof(struct block_q5_K_t), 48, 16, true};
static_assert(sizeof(struct block_q4_K_t) == 144 && sizeof(struct block_q5_K_t) == 176,
              "Q4_K / Q5_K superblock sizes");

static inline float blk_h(const uint8_t *p) {
    uint16_t h;
    memcpy(&h, p, sizeof h);
    return _cvtsh_ss(h);
}

/* Bit p of each qh byte, moved to bit 4 (value 16) or 0. Masking first
 * keeps the 16-bit shifts inside each byte. */
static inline __m256i q5_bit(__m256i hb, int p) {
    const __m256i m = _mm256_and_si256(hb, _mm256_set1_epi8((char) (1 << p)));
    return p <= 4 ? _mm256_slli_epi16(m, 4 - p) : _mm256_srli_epi16(m, p - 4);
}

/* The 64 q values of sub-blocks 2j and 2j + 1 (0..15, or 0..31 for Q5_K). */
[[gnu::always_inline]] static inline void
unpack_pair(struct fmt f, const uint8_t *blk, size_t j, __m256i *lo, __m256i *hi) {
    const __m256i mask = _mm256_set1_epi8(0x0F);
    const __m256i q    = _mm256_loadu_si256((const __m256i *) (blk + f.qs + j * 32));
    *lo                = _mm256_and_si256(q, mask);
    *hi                = _mm256_and_si256(_mm256_srli_epi16(q, 4), mask);
    if (f.q5) {
        const __m256i hb = _mm256_loadu_si256((const __m256i *) (blk + f.qh));
        *lo              = _mm256_or_si256(*lo, q5_bit(hb, (int) (2 * j)));
        *hi              = _mm256_or_si256(*hi, q5_bit(hb, (int) (2 * j + 1)));
    }
}

/* sum_j sc_j P_j of one superblock against one activation block, as eight
 * exact int32 lanes (Q5_K pairs reach 2 * 31 * 127 = 7874, still no int16
 * saturation). */
[[gnu::always_inline]] static inline __m256i
superblock_sumi(struct fmt f, const uint8_t *blk, const uint8_t sc[static 8], const int8_t *xq) {
    __m256i acc = _mm256_setzero_si256();
    for (size_t j = 0; j < 4; j++) {
        __m256i lo, hi;
        unpack_pair(f, blk, j, &lo, &hi);
        const __m256i x0 = _mm256_loadu_si256((const __m256i *) (xq + j * 64));
        const __m256i x1 = _mm256_loadu_si256((const __m256i *) (xq + j * 64 + 32));
        acc              = _mm256_add_epi32(
                acc, _mm256_madd_epi16(_mm256_maddubs_epi16(lo, x0), _mm256_set1_epi16(sc[2 * j])));
        acc = _mm256_add_epi32(
                acc,
                _mm256_madd_epi16(_mm256_maddubs_epi16(hi, x1), _mm256_set1_epi16(sc[2 * j + 1])));
    }
    return acc;
}

static inline int32_t min_term(const uint8_t mn[static 8], const int32_t sx[static 8]) {
    int32_t s = 0;
    for (size_t j = 0; j < NSB; j++) {
        s += (int32_t) mn[j] * sx[j];
    }
    return s;
}

/* One weight row against one activation row; the min term accumulates in a
 * scalar; fixed order. One fp32 FMA per 256 elements, so a single
 * accumulator is not the bottleneck. */
[[gnu::always_inline]] static inline float dot_row(struct fmt     f,
                                                   size_t         nb,
                                                   const uint8_t *w,
                                                   const int8_t  *qx,
                                                   const float   *dx,
                                                   const int32_t *sx) {
    __m256 acc  = _mm256_setzero_ps();
    float  mins = 0.0f;
    for (size_t b = 0; b < nb; b++) {
        const uint8_t *blk = w + b * f.stride;
        uint8_t        sc[8], mn[8];
        unpack_scales(blk + 4, sc, mn);
        const __m256i sumi = superblock_sumi(f, blk, sc, qx + b * QK);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(blk_h(blk) * dx[b]), _mm256_cvtepi32_ps(sumi), acc);
        mins += blk_h(blk + 2) * dx[b] * (float) min_term(mn, sx + b * NSB);
    }
    return hsum_ps(acc) - mins;
}

/* NR activation rows against one weight row: each superblock's q values and
 * scales unpacked once per tile. */
[[gnu::always_inline]] static inline void dot_rows(struct fmt     f,
                                                   size_t         nb,
                                                   size_t         n_in,
                                                   const uint8_t *w,
                                                   const int8_t  *qx,
                                                   const float   *dx,
                                                   const int32_t *sx,
                                                   float          out[static NR]) {
    __m256 acc[NR];
    float  mins[NR];
    for (size_t r = 0; r < NR; r++) {
        acc[r]  = _mm256_setzero_ps();
        mins[r] = 0.0f;
    }
    const size_t nsx = n_in / 32;
    for (size_t b = 0; b < nb; b++) {
        const uint8_t *blk = w + b * f.stride;
        uint8_t        sc[8], mn[8];
        unpack_scales(blk + 4, sc, mn);
        __m256i sumi[NR];
        for (size_t r = 0; r < NR; r++) {
            sumi[r] = _mm256_setzero_si256();
        }
        for (size_t j = 0; j < 4; j++) {
            __m256i lo, hi;
            unpack_pair(f, blk, j, &lo, &hi);
            const __m256i sl  = _mm256_set1_epi16(sc[2 * j]);
            const __m256i sh  = _mm256_set1_epi16(sc[2 * j + 1]);
            const size_t  off = b * QK + j * 64;
            for (size_t r = 0; r < NR; r++) {
                const int8_t *xr = qx + r * n_in + off;
                const __m256i x0 = _mm256_loadu_si256((const __m256i *) xr);
                const __m256i x1 = _mm256_loadu_si256((const __m256i *) (xr + 32));
                sumi[r] = _mm256_add_epi32(sumi[r],
                                           _mm256_madd_epi16(_mm256_maddubs_epi16(lo, x0), sl));
                sumi[r] = _mm256_add_epi32(sumi[r],
                                           _mm256_madd_epi16(_mm256_maddubs_epi16(hi, x1), sh));
            }
        }
        const float d    = blk_h(blk);
        const float dmin = blk_h(blk + 2);
        for (size_t r = 0; r < NR; r++) {
            const float dxr = dx[r * nb + b];
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d * dxr), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
            mins[r] += dmin * dxr * (float) min_term(mn, sx + r * nsx + b * NSB);
        }
    }
    for (size_t r = 0; r < NR; r++) {
        out[r] = hsum_ps(acc[r]) - mins[r];
    }
}

/* The calling thread's workspace with room for m quantized activation rows
 * (int8 values, one fp32 scale per 256 and one int32 sum per 32), or
 * nullptr. */
static struct cpu_x86_workspace *acquire_acts(struct geist_backend *be, size_t m, size_t n_in) {
    size_t acts_bytes = 0, n_blocks = 0, scale_bytes = 0, n_sums = 0, sum_bytes = 0;
    if (be == nullptr || be->state == nullptr || ckd_mul(&acts_bytes, m, n_in) ||
        ckd_mul(&n_blocks, m, n_in / QK) || ckd_mul(&scale_bytes, n_blocks, sizeof(float)) ||
        ckd_mul(&n_sums, m, n_in / 32) || ckd_mul(&sum_bytes, n_sums, sizeof(int32_t))) {
        return nullptr;
    }
    return cpu_x86_ws_acquire_mN(
            (struct cpu_x86_state *) be->state, acts_bytes, sum_bytes, scale_bytes, 0);
}

[[gnu::always_inline]] static inline void linear_m1(struct fmt                 f,
                                                    const float               *x,
                                                    const struct geist_weight *w,
                                                    struct geist_backend      *be,
                                                    float                     *y) {
    const size_t              n_in  = (size_t) w->n_in;
    const size_t              n_out = (size_t) w->n_out;
    const size_t              nb    = n_in / QK;
    struct cpu_x86_workspace *ws    = acquire_acts(be, 1, n_in);
    if (ws == nullptr) {
        geist_linear_ref(1, x, w, y); /* no scratch: the reference needs none */
        return;
    }
    quantize_row_q8_k(nb, x, ws->mN_acts, ws->mN_scale, ws->mN_sum_a);
    const int8_t  *qx = ws->mN_acts;
    const float   *dx = ws->mN_scale;
    const int32_t *sx = ws->mN_sum_a;
    const uint8_t *wb = (const uint8_t *) w->raw;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t j = 0; j < n_out; j++) {
        y[j] = dot_row(f, nb, wb + j * nb * f.stride, qx, dx, sx);
    }
}

[[gnu::always_inline]] static inline void linear_mN(struct fmt                 f,
                                                    size_t                     m,
                                                    const float               *x,
                                                    const struct geist_weight *w,
                                                    struct geist_backend      *be,
                                                    float                     *y) {
    const size_t              n_in  = (size_t) w->n_in;
    const size_t              n_out = (size_t) w->n_out;
    const size_t              nb    = n_in / QK;
    const size_t              nsx   = n_in / 32;
    struct cpu_x86_workspace *ws    = acquire_acts(be, m, n_in);
    if (ws == nullptr) {
        geist_linear_ref(m, x, w, y);
        return;
    }
    int8_t        *qx    = ws->mN_acts;
    float         *dx    = ws->mN_scale;
    int32_t       *sx    = ws->mN_sum_a;
    const uint8_t *wb    = (const uint8_t *) w->raw;
    const size_t   m_til = m - m % NR;

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (size_t i = 0; i < m; i++) {
            quantize_row_q8_k(nb, x + i * n_in, qx + i * n_in, dx + i * nb, sx + i * nsx);
        }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (size_t j = 0; j < n_out; j++) {
            const uint8_t *wr = wb + j * nb * f.stride;
            float          out[NR];
            for (size_t i = 0; i < m_til; i += NR) {
                dot_rows(f, nb, n_in, wr, qx + i * n_in, dx + i * nb, sx + i * nsx, out);
                for (size_t r = 0; r < NR; r++) {
                    y[(i + r) * n_out + j] = out[r];
                }
            }
            for (size_t i = m_til; i < m; i++) {
                y[i * n_out + j] = dot_row(f, nb, wr, qx + i * n_in, dx + i * nb, sx + i * nsx);
            }
        }
    }
}

static void cpu_x86_linear_q4k_raw_m1(const float               *x,
                                      const struct geist_weight *w,
                                      struct geist_backend      *be,
                                      float                     *y) {
    linear_m1(FMT_Q4_K, x, w, be, y);
}

static void cpu_x86_linear_q4k_raw_mN(size_t                     m,
                                      const float               *x,
                                      const struct geist_weight *w,
                                      struct geist_backend      *be,
                                      float                     *y) {
    linear_mN(FMT_Q4_K, m, x, w, be, y);
}

static void cpu_x86_linear_q5k_m1(const float               *x,
                                  const struct geist_weight *w,
                                  struct geist_backend      *be,
                                  float                     *y) {
    linear_m1(FMT_Q5_K, x, w, be, y);
}

static void cpu_x86_linear_q5k_mN(size_t                     m,
                                  const float               *x,
                                  const struct geist_weight *w,
                                  struct geist_backend      *be,
                                  float                     *y) {
    linear_mN(FMT_Q5_K, m, x, w, be, y);
}

void cpu_x86_linear_q4k_raw_bind(struct geist_weight *w) {
    w->linear_m1 = cpu_x86_linear_q4k_raw_m1;
    w->linear_mN = cpu_x86_linear_q4k_raw_mN;
}

void cpu_x86_linear_q5k_bind(struct geist_weight *w) {
    w->linear_m1 = cpu_x86_linear_q5k_m1;
    w->linear_mN = cpu_x86_linear_q5k_mN;
}
