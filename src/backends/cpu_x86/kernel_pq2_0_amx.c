/*
 * src/backends/cpu_x86/kernel_pq2_0_amx.c — PQ2_0 prefill GEMM, AMX-INT8.
 *
 * Layer: BACKEND (cpu_x86). Compiled with -mamx-tile -mamx-int8 -mavx512f
 * -mavx512bw (mk/backend-cpu_x86.mk). linear_pq2_0.c decides, at bind time
 * and outside this TU, whether the host may run it.
 *
 * The AVX2 GEMM in linear_pq2_0.c spends four maddubs per block, row and
 * token. One TDPBSSD multiplies 16 x 64 int8 by 64 x 16 int8 into 16 x 16
 * int32 in 16 cycles. Here a C tile is 16 weight rows x 16 tokens over one
 * 128-element block, two TDPBSSD (one per 64-element half):
 *
 *   - A, the weights: code - 1 as s8, in code order (the activations'
 *     order, linear_pq2_0.c): row r of half 0 is levels 0 and 1 of its 32
 *     code bytes, half 1 levels 2 and 3. Subtracting 1 here takes the
 *     code's bias out of the dot, so no activation sum is needed. A block's
 *     A is extracted once per 16 rows (two rows per zmm) and serves every
 *     token tile.
 *   - B, the activations: linear_pq2_0.c's int8 rows, packed per block into
 *     VNNI tiles (row k4 = bytes 4*k4 .. 4*k4+3 of 16 tokens) by
 *     pq2_0_amx_pack: a 16 x 16 dword transpose per half.
 *   - C: the exact int32 block dots (|dot| <= 128 * 2 * 128). The block
 *     scale is applied in fp32, acc[r][t] += d[r] * C[r][t], and y is
 *     inv[t] * acc once all blocks are in.
 *
 * C leaves the tiles through memory (TILESTORED), and a vector load of
 * those bytes waits until the store commits, after the two TDPBSSD it
 * depends on retire (about 140 cycles as a dependent chain). So C
 * alternates between two tile registers and a ring of four buffers, and
 * each step post-processes the C of two steps before. The B tiles
 * alternate with C, so a step's loads need not wait for the previous
 * step's dots to read theirs. The next block's A is extracted into the
 * other half of a double buffer while this block's steps run, so its tile
 * load never waits for the extraction's stores either; the weights are
 * requested PREFETCH_BLOCKS blocks ahead.
 *
 * Host: 4-vCPU Xeon (Sapphire Rapids class), gcc 14.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_pq2_0_amx.h"

#ifndef GEIST_NO_AMX /* the assembler knows AMX; mk/backend-cpu_x86.mk */

#include "quant.h"

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

constexpr size_t QK   = PQ2_0_BLOCK_ELEMS; /* 128 */
constexpr size_t BB   = PQ2_0_BLOCK_BYTES; /* fp16 d, then 32 code bytes */
constexpr size_t ROWS = PQ2_0_AMX_ROWS;
static_assert(PQ2_0_BLOCK_ELEMS == 128 && PQ2_0_BLOCK_BYTES == 34,
              "a block is two 64-byte K halves; its codes are one ymm load");
static_assert(PQ2_0_AMX_ROWS == 16 && PQ2_0_AMX_TOKENS == 16, "C is one 16 x 16 int32 tile");

/* Bytes of one tile (16 rows of 64), and its int32 count. */
constexpr size_t TILE_BYTES = 1024;
constexpr size_t TILE_WORDS = TILE_BYTES / sizeof(int32_t);
static_assert(PQ2_0_AMX_TILE_BYTES == 2 * TILE_BYTES, "a token tile is two B tiles");

/* Steps between a C tile's store and its post-processing, and the ring of
 * C buffers that covers them. */
constexpr size_t LAG  = 2;
constexpr size_t RING = 4;
static_assert(LAG < RING, "a C buffer must not be stored to before it is read");

/* Blocks ahead of the extraction that the weight rows are requested. 4
 * measured best of 0 / 4 / 8 with the weights streamed from DRAM. */
constexpr size_t PREFETCH_BLOCKS = 4;

/* Scratch: A double buffer (two halves of 16 rows each), the C ring, then
 * the accumulators, 16 x 16 fp32 per token tile. */
constexpr size_t A_BYTES = 2 * 2 * TILE_BYTES;
constexpr size_t C_BYTES = RING * TILE_BYTES;
static_assert(A_BYTES + C_BYTES == PQ2_0_AMX_SCRATCH_FIXED, "scratch layout");
static_assert(ROWS * PQ2_0_AMX_TOKENS * sizeof(float) == PQ2_0_AMX_SCRATCH_PER_TILE,
              "one accumulator tile per token tile");

/* LDTILECFG's 64-byte operand, palette 1. */
struct tile_config {
    uint8_t  palette_id;
    uint8_t  start_row;
    uint8_t  reserved[14];
    uint16_t colsb[16];
    uint8_t  rows[16];
};
static_assert(sizeof(struct tile_config) == 64, "LDTILECFG reads 64 bytes");

/* Tiles: C in 0 / 1 (alternating steps), A in 2 / 3 (halves 0 / 1), B in
 * 4, 5 (even steps) / 6, 7 (odd steps); all 16 rows x 64 bytes. */
static void configure_tiles(void) {
    alignas(64) struct tile_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.palette_id = 1;
    for (size_t i = 0; i < 8; i++) {
        cfg.colsb[i] = 64;
        cfg.rows[i]  = 16;
    }
    _tile_loadconfig(&cfg);
}

static inline float block_scale(const uint8_t *blk) {
    uint16_t d;
    memcpy(&d, blk, sizeof d);
    return _cvtsh_ss(d);
}

/* In-place 16 x 16 transpose of 32-bit elements: v[r][c] -> v[c][r]. */
static inline void transpose16(__m512 v[16]) {
    __m512 t[16];
    for (size_t r = 0; r < 16; r += 2) {
        t[r]     = _mm512_unpacklo_ps(v[r], v[r + 1]);
        t[r + 1] = _mm512_unpackhi_ps(v[r], v[r + 1]);
    }
    /* per 128-bit lane L: v[4q + j] = column 4L + j of rows 4q .. 4q+3 */
    for (size_t r = 0; r < 16; r += 4) {
        const __m512d a = _mm512_castps_pd(t[r]);
        const __m512d b = _mm512_castps_pd(t[r + 1]);
        const __m512d c = _mm512_castps_pd(t[r + 2]);
        const __m512d d = _mm512_castps_pd(t[r + 3]);
        v[r]            = _mm512_castpd_ps(_mm512_unpacklo_pd(a, c));
        v[r + 1]        = _mm512_castpd_ps(_mm512_unpackhi_pd(a, c));
        v[r + 2]        = _mm512_castpd_ps(_mm512_unpacklo_pd(b, d));
        v[r + 3]        = _mm512_castpd_ps(_mm512_unpackhi_pd(b, d));
    }
    /* column 4L + j = lane L of v[j], v[4 + j], v[8 + j], v[12 + j] */
    for (size_t j = 0; j < 4; j++) {
        const __m512 lo01 = _mm512_shuffle_f32x4(v[j], v[4 + j], 0x44);
        const __m512 hi01 = _mm512_shuffle_f32x4(v[j], v[4 + j], 0xEE);
        const __m512 lo23 = _mm512_shuffle_f32x4(v[8 + j], v[12 + j], 0x44);
        const __m512 hi23 = _mm512_shuffle_f32x4(v[8 + j], v[12 + j], 0xEE);
        t[j]              = _mm512_shuffle_f32x4(lo01, lo23, 0x88);
        t[4 + j]          = _mm512_shuffle_f32x4(lo01, lo23, 0xDD);
        t[8 + j]          = _mm512_shuffle_f32x4(hi01, hi23, 0x88);
        t[12 + j]         = _mm512_shuffle_f32x4(hi01, hi23, 0xDD);
    }
    for (size_t r = 0; r < 16; r++) {
        v[r] = t[r];
    }
}

void pq2_0_amx_pack(size_t tn, size_t n_tok, const int8_t *xq, int32_t *bt) {
    for (size_t tile = 0; tile < tn; tile++) {
        for (size_t h = 0; h < 2; h++) {
            __m512 v[16];
            for (size_t n = 0; n < 16; n++) {
                const size_t t = tile * 16 + n;
                v[n]           = t < n_tok ? _mm512_castsi512_ps(_mm512_loadu_si512(
                                                     (const void *) (xq + t * QK + h * 64)))
                                           : _mm512_setzero_ps();
            }
            transpose16(v);
            int32_t *b = bt + (tile * 2 + h) * TILE_WORDS;
            for (size_t k = 0; k < 16; k++) {
                _mm512_storeu_si512((void *) (b + k * 16), _mm512_castps_si512(v[k]));
            }
        }
    }
}

/* One block of the group's rows 0 .. nr-1 (row r at w + r * rb) as the two
 * A tiles at a, and their scales. Rows nr .. 15 are zero with d = 0. */
static inline void extract_a(size_t nr, size_t rb, const uint8_t *w, int8_t *a, float *d) {
    const __m512i m3  = _mm512_set1_epi8(3);
    const __m512i one = _mm512_set1_epi8(1);
    size_t        r   = 0;
    for (; r < nr; r += 2) {
        /* rows r (low 256 bits) and r + 1 (high), or r alone */
        const uint8_t *b0  = w + r * rb;
        const __m256i  lo  = _mm256_loadu_si256((const __m256i *) (b0 + 2));
        const bool     two = r + 1 < nr;
        const __m256i  hi =
                two ? _mm256_loadu_si256((const __m256i *) (b0 + rb + 2)) : _mm256_setzero_si256();
        const __m512i v  = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
        const __m512i c0 = _mm512_sub_epi8(_mm512_and_si512(v, m3), one);
        const __m512i c1 = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(v, 2), m3), one);
        const __m512i c2 = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(v, 4), m3), one);
        const __m512i c3 = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(v, 6), m3), one);
        /* row r: [level 0 | level 1] and [level 2 | level 3], row r + 1 the
         * high halves */
        _mm512_store_si512((void *) (a + r * 64), _mm512_shuffle_i64x2(c0, c1, 0x44));
        _mm512_store_si512((void *) (a + TILE_BYTES + r * 64), _mm512_shuffle_i64x2(c2, c3, 0x44));
        d[r] = block_scale(b0);
        if (two) {
            _mm512_store_si512((void *) (a + (r + 1) * 64), _mm512_shuffle_i64x2(c0, c1, 0xEE));
            _mm512_store_si512((void *) (a + TILE_BYTES + (r + 1) * 64),
                               _mm512_shuffle_i64x2(c2, c3, 0xEE));
            d[r + 1] = block_scale(b0 + rb);
        }
    }
    for (r = nr; r < ROWS; r++) {
        _mm512_store_si512((void *) (a + r * 64), _mm512_setzero_si512());
        _mm512_store_si512((void *) (a + TILE_BYTES + r * 64), _mm512_setzero_si512());
        d[r] = 0.0f;
    }
}

/* Step s: C tile (s & 1) = A x the token tile at b, both halves. */
static inline void tile_dots(size_t s, const int32_t *b) {
    if ((s & 1) != 0) {
        _tile_loadd(6, b, 64);
        _tile_loadd(7, b + TILE_WORDS, 64);
        _tile_zero(1);
        _tile_dpbssd(1, 2, 6);
        _tile_dpbssd(1, 3, 7);
    } else {
        _tile_loadd(4, b, 64);
        _tile_loadd(5, b + TILE_WORDS, 64);
        _tile_zero(0);
        _tile_dpbssd(0, 2, 4);
        _tile_dpbssd(0, 3, 5);
    }
}

static inline void tile_store(size_t s, int32_t *c) {
    if ((s & 1) != 0) {
        _tile_stored(1, c, 64);
    } else {
        _tile_stored(0, c, 64);
    }
}

/* acc[r][0 .. 15] += d[r] * C[r][0 .. 15] for one C tile. */
static inline void post(const int32_t *c, const float *d, float *acc) {
    for (size_t r = 0; r < ROWS; r++) {
        const __m512 p = _mm512_cvtepi32_ps(_mm512_load_si512((const void *) (c + r * 16)));
        _mm512_store_ps(acc + r * 16,
                        _mm512_fmadd_ps(p, _mm512_set1_ps(d[r]), _mm512_load_ps(acc + r * 16)));
    }
}

/* y[t * n_out + r] = inv[t] * acc for the group's nr rows (y at its row 0)
 * and tokens t < m; acc is 16 rows x 16 tokens per token tile. */
static void write_out(size_t       m,
                      size_t       n_out,
                      size_t       nr,
                      size_t       tn,
                      const float *acc,
                      const float *inv,
                      float       *y) {
    const __mmask16 rows = (__mmask16) ((1u << nr) - 1u);
    for (size_t tile = 0; tile < tn; tile++) {
        __m512 v[16];
        for (size_t r = 0; r < ROWS; r++) {
            v[r] = _mm512_load_ps(acc + (tile * ROWS + r) * 16);
        }
        transpose16(v);
        for (size_t n = 0; n < 16 && tile * 16 + n < m; n++) {
            const size_t t = tile * 16 + n;
            _mm512_mask_storeu_ps(y + t * n_out, rows, _mm512_mul_ps(v[n], _mm512_set1_ps(inv[t])));
        }
    }
}

void pq2_0_amx_gemm(size_t         nb,
                    size_t         n_out,
                    size_t         m,
                    size_t         tn,
                    size_t         g0,
                    size_t         g1,
                    const uint8_t *w,
                    const int32_t *bt,
                    const float   *inv,
                    uint8_t       *scratch,
                    float         *y) {
    const size_t rb    = nb * BB;
    int8_t      *abuf  = (int8_t *) scratch;
    int32_t     *cring = (int32_t *) (void *) (scratch + A_BYTES);
    float       *acc   = (float *) (void *) (scratch + A_BYTES + C_BYTES);
    /* The scales of four blocks: a post can lag into the block two back
     * while the next one's are extracted. */
    alignas(64) float dq[RING][ROWS];
    const float      *pd[RING] = {};
    float            *pa[RING] = {};
    configure_tiles();
    for (size_t g = g0; g < g1; g++) {
        const size_t   r0 = g * ROWS;
        const size_t   nr = n_out - r0 < ROWS ? n_out - r0 : ROWS;
        const uint8_t *wg = w + r0 * rb;
        memset(acc, 0, tn * PQ2_0_AMX_SCRATCH_PER_TILE);
        extract_a(nr, rb, wg, abuf, dq[0]);
        size_t s = 0; /* step: (block, token tile) in order */
        for (size_t b = 0; b < nb; b++) {
            const int8_t *ab = abuf + (b & 1) * 2 * TILE_BYTES;
            _tile_loadd(2, ab, 64);
            _tile_loadd(3, ab + TILE_BYTES, 64);
            if (b + PREFETCH_BLOCKS < nb) {
                for (size_t r = 0; r < nr; r++) {
                    _mm_prefetch((const char *) (wg + r * rb + (b + PREFETCH_BLOCKS) * BB),
                                 _MM_HINT_T0);
                }
            }
            if (b + 1 < nb) {
                extract_a(nr,
                          rb,
                          wg + (b + 1) * BB,
                          abuf + ((b + 1) & 1) * 2 * TILE_BYTES,
                          dq[(b + 1) % RING]);
            }
            const int32_t *bb = bt + b * tn * 2 * TILE_WORDS;
            for (size_t tile = 0; tile < tn; tile++, s++) {
                tile_dots(s, bb + tile * 2 * TILE_WORDS);
                if (s >= LAG) {
                    const size_t q = (s - LAG) % RING;
                    post(cring + q * TILE_WORDS, pd[q], pa[q]);
                }
                tile_store(s, cring + (s % RING) * TILE_WORDS);
                pd[s % RING] = dq[b % RING];
                pa[s % RING] = acc + tile * ROWS * 16;
            }
        }
        for (size_t q = s > LAG ? s - LAG : 0; q < s; q++) {
            post(cring + (q % RING) * TILE_WORDS, pd[q % RING], pa[q % RING]);
        }
        write_out(m, n_out, nr, tn, acc, inv, y + r0);
    }
    _tile_release();
}
#endif /* GEIST_NO_AMX */
