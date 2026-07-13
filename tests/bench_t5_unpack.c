/*
 * bench_t5_unpack — #104 Phase A feasibility spike (kill-switch).
 *
 * Question: does a base-3 "T5" weight packing (5 trits/byte, 1.6 bpw,
 * −18.75 % bytes vs the shipping 2-bit x4 layout) win or lose on the
 * 9950X-class decode GEMV, where the ternary layers stream ~520 MB/token
 * at 50-60 GB/s? The unpack is compilade's pow3 trick (ggml PR #8151):
 *
 *   pack:   n = t0*81 + t1*27 + t2*9 + t3*3 + t4   (n < 243, trits 0..2)
 *           s = ceil(n * 256 / 243)                (fits u8)
 *   unpack: m_k = s * 3^k  (mod 256 — wrapping byte adds)
 *           t_k = (m_k > 85) + (m_k > 170)         (= floor(m_k*3/256))
 *
 * Kernels compared, identical N×K sweep, OMP over 4-row groups, both
 * ending in VPDPBUSD u8-code × s8-act accumulation:
 *   (a) X4    — the shipping layout: one 64 B load = 64 cols × 4 rows,
 *               3 shifts + 4 ands unpack (i2s_x4_group_m1 structure).
 *   (b) T5    — per-row base-3 rows (stride-64 plane layout: byte c of a
 *               64 B group holds trits of columns plane*64+c), 4 rows per
 *               group share the 5 activation slices from registers.
 *
 * Output: weights/s and effective GB/s for both, plus an exact int32
 * correctness check of (b)'s unpack against the ground-truth trits.
 *
 * GATE (#104): ship Phase B/C only if T5 weights/s ≥ 1.05× X4 weights/s
 * (i.e. the byte saving survives the extra ALU). Below that, close the
 * issue with these numbers.
 */
#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

#define K_COLS 2560               /* BitNet 2B-4T hidden dim */
#define GROUP_T5 320              /* columns per 64-byte T5 group */
#define N_ROWS (4 * 96 * 1024)    /* 393216 rows -> x4: 629 MB, t5: 503 MB */
#define ROW_BYTES_X4 (K_COLS / 4) /* 640: packed 2-bit, row-major view */
#define ROW_BYTES_T5 (K_COLS / 5) /* 512: base-3, 8 groups of 64 B */
#define SWEEPS 6

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

static uint32_t prng(uint32_t *s) {
    uint32_t z = (*s += 0x9E3779B9u);
    z          = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z          = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))

#define T5_TARGET "avx512f,avx512bw,avx512vl,avx512vnni"

/* ---- (a) shipping x4 layout: byte = 4 rows x 1 col, 2 bit each ---------- */

__attribute__((target(T5_TARGET))) static void
x4_rowgroup_dot(const uint8_t *Wg, /* K/4*4 = K bytes */
                const int8_t  *xq,
                int32_t        out[4]) {
    const __m512i m3 = _mm512_set1_epi8(3);
    __m512i       a0 = _mm512_setzero_si512();
    __m512i       a1 = _mm512_setzero_si512();
    __m512i       a2 = _mm512_setzero_si512();
    __m512i       a3 = _mm512_setzero_si512();
    for (size_t cb = 0; cb < K_COLS / 64; cb++) {
        _mm_prefetch((const char *) (Wg + cb * 64 + 512), _MM_HINT_NTA);
        const __m512i w = _mm512_loadu_si512((const void *) (Wg + cb * 64));
        const __m512i a = _mm512_loadu_si512((const void *) (xq + cb * 64));
        a0              = _mm512_dpbusd_epi32(a0, _mm512_and_si512(_mm512_srli_epi16(w, 6), m3), a);
        a1              = _mm512_dpbusd_epi32(a1, _mm512_and_si512(_mm512_srli_epi16(w, 4), m3), a);
        a2              = _mm512_dpbusd_epi32(a2, _mm512_and_si512(_mm512_srli_epi16(w, 2), m3), a);
        a3              = _mm512_dpbusd_epi32(a3, _mm512_and_si512(w, m3), a);
    }
    out[0] = _mm512_reduce_add_epi32(a0);
    out[1] = _mm512_reduce_add_epi32(a1);
    out[2] = _mm512_reduce_add_epi32(a2);
    out[3] = _mm512_reduce_add_epi32(a3);
}

/* ---- (b) T5 base-3 rows: byte = 5 trits of ONE row, stride-64 planes ---- */

/* Extract trit plane from m (= s*3^k mod 256): (m>85) + (m>170). */
__attribute__((target(T5_TARGET))) static inline __m512i t5_trits(__m512i m) {
    const __m512i   one  = _mm512_set1_epi8(1);
    const __mmask64 g85  = _mm512_cmpgt_epu8_mask(m, _mm512_set1_epi8(85));
    const __mmask64 g170 = _mm512_cmpgt_epu8_mask(m, _mm512_set1_epi8((char) 170));
    __m512i         t    = _mm512_maskz_mov_epi8(g85, one);
    return _mm512_mask_add_epi8(t, g170, t, one);
}

__attribute__((target(T5_TARGET))) static void
t5_rowgroup_dot(const uint8_t *w_rows[4], const int8_t *xq, int32_t out[4]) {
    __m512i acc[4];
    for (int r = 0; r < 4; r++) {
        acc[r] = _mm512_setzero_si512();
    }
    for (size_t g = 0; g < K_COLS / GROUP_T5; g++) {
        __m512i m[4];
        for (int r = 0; r < 4; r++) {
            _mm_prefetch((const char *) (w_rows[r] + g * 64 + 512), _MM_HINT_NTA);
            m[r] = _mm512_loadu_si512((const void *) (w_rows[r] + g * 64));
        }
        for (int plane = 0; plane < 5; plane++) {
            const __m512i a =
                    _mm512_loadu_si512((const void *) (xq + g * GROUP_T5 + (size_t) plane * 64));
            for (int r = 0; r < 4; r++) {
                acc[r] = _mm512_dpbusd_epi32(acc[r], t5_trits(m[r]), a);
                if (plane < 4) { /* m *= 3 (mod 256) for the next plane */
                    m[r] = _mm512_add_epi8(_mm512_add_epi8(m[r], m[r]), m[r]);
                }
            }
        }
    }
    for (int r = 0; r < 4; r++) {
        out[r] = _mm512_reduce_add_epi32(acc[r]);
    }
}

/* ---- driver ------------------------------------------------------------- */

int main(void) {
    if (!__builtin_cpu_supports("avx512vnni") || !__builtin_cpu_supports("avx512bw")) {
        printf("SKIP: needs AVX-512 BW+VNNI\n");
        return 77;
    }

    const size_t x4_bytes = (size_t) N_ROWS * ROW_BYTES_X4;
    const size_t t5_bytes = (size_t) N_ROWS * ROW_BYTES_T5;
    uint8_t     *Wx4      = aligned_alloc(64, x4_bytes);
    uint8_t     *Wt5      = aligned_alloc(64, t5_bytes);
    int8_t      *xq       = aligned_alloc(64, K_COLS);
    int32_t     *ref      = malloc((size_t) N_ROWS * sizeof(int32_t)); /* first CHECK_ROWS used */
    if (Wx4 == NULL || Wt5 == NULL || xq == NULL || ref == NULL) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    uint32_t s = 0x104BEEFu;
    for (size_t i = 0; i < K_COLS; i++) {
        xq[i] = (int8_t) ((int) (prng(&s) % 255) - 127);
    }

    /* Fill both formats from the same ground-truth trits; keep an exact
     * int32 reference dot for the first rows. x4 group layout matches the
     * engine: x4[grp*K + c] byte packs rows 4g..4g+3 of column c at shifts
     * 6-2rb. t5: row-major rows of K/5 bytes, stride-64 planes per group. */
    const size_t CHECK_ROWS = 64;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t grp = 0; grp < N_ROWS / 4; grp++) {
        uint32_t rs = 0xABCD0123u ^ (uint32_t) grp;
        uint8_t  trit[4][K_COLS];
        for (size_t c = 0; c < K_COLS; c++) {
            uint8_t b = 0;
            for (int r = 0; r < 4; r++) {
                trit[r][c] = (uint8_t) (prng(&rs) % 3);
                b          = (uint8_t) (b | (trit[r][c] << (6 - 2 * r)));
            }
            Wx4[grp * K_COLS + c] = b;
        }
        for (int r = 0; r < 4; r++) {
            uint8_t *row = Wt5 + (grp * 4 + (size_t) r) * ROW_BYTES_T5;
            for (size_t g = 0; g < K_COLS / GROUP_T5; g++) {
                for (size_t c = 0; c < 64; c++) {
                    uint32_t n = 0;
                    for (int plane = 0; plane < 5; plane++) {
                        n = n * 3 + trit[r][g * GROUP_T5 + (size_t) plane * 64 + c];
                    }
                    row[g * 64 + c] = (uint8_t) ((n * 256 + 242) / 243);
                }
            }
            if (grp * 4 + (size_t) r < CHECK_ROWS) {
                int32_t acc = 0;
                for (size_t c = 0; c < K_COLS; c++) {
                    acc += (int32_t) trit[r][c] * (int32_t) xq[c];
                }
                ref[grp * 4 + r] = acc;
            }
        }
    }

    /* Correctness: both kernels must reproduce the ground-truth code dots. */
    for (size_t grp = 0; grp < CHECK_ROWS / 4; grp++) {
        int32_t        o_x4[4], o_t5[4];
        const uint8_t *rows[4];
        for (int r = 0; r < 4; r++) {
            rows[r] = Wt5 + (grp * 4 + (size_t) r) * ROW_BYTES_T5;
        }
        x4_rowgroup_dot(Wx4 + grp * K_COLS, xq, o_x4);
        t5_rowgroup_dot(rows, xq, o_t5);
        for (int r = 0; r < 4; r++) {
            if (o_x4[r] != ref[grp * 4 + r] || o_t5[r] != ref[grp * 4 + r]) {
                fprintf(stderr,
                        "FAIL row %zu: ref=%d x4=%d t5=%d\n",
                        grp * 4 + (size_t) r,
                        ref[grp * 4 + r],
                        o_x4[r],
                        o_t5[r]);
                return 1;
            }
        }
    }
    printf("correctness: x4 + t5 both exact on %zu rows\n", CHECK_ROWS);

    const double     weights = (double) N_ROWS * K_COLS;
    volatile int32_t sink    = 0;

    for (int pass = 0; pass < 2; pass++) { /* pass 0 = warmup, 1 = timed */
        double t_x4 = 0.0, t_t5 = 0.0;
        for (int sweep = 0; sweep < SWEEPS; sweep++) {
            double t0 = now_s();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (size_t grp = 0; grp < N_ROWS / 4; grp++) {
                int32_t o[4];
                x4_rowgroup_dot(Wx4 + grp * K_COLS, xq, o);
                if (o[0] == 0x7FFFFFFF) {
                    sink += o[0];
                }
            }
            t_x4 += now_s() - t0;

            t0 = now_s();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (size_t grp = 0; grp < N_ROWS / 4; grp++) {
                const uint8_t *rows[4];
                for (int r = 0; r < 4; r++) {
                    rows[r] = Wt5 + (grp * 4 + (size_t) r) * ROW_BYTES_T5;
                }
                int32_t o[4];
                t5_rowgroup_dot(rows, xq, o);
                if (o[0] == 0x7FFFFFFF) {
                    sink += o[0];
                }
            }
            t_t5 += now_s() - t0;
        }
        if (pass == 1) {
            const double wps_x4 = weights * SWEEPS / t_x4;
            const double wps_t5 = weights * SWEEPS / t_t5;
            printf("x4 : %7.2f Gwt/s  %6.2f GB/s  (%zu MB stream)\n",
                   wps_x4 / 1e9,
                   wps_x4 * 0.25 / 1e9,
                   x4_bytes >> 20);
            printf("t5 : %7.2f Gwt/s  %6.2f GB/s  (%zu MB stream)\n",
                   wps_t5 / 1e9,
                   wps_t5 * 0.20 / 1e9,
                   t5_bytes >> 20);
            printf("t5/x4 weights-throughput ratio: %.3f  (gate: >= 1.05)\n", wps_t5 / wps_x4);
            printf("%s\n", (wps_t5 >= 1.05 * wps_x4) ? "GATE: PASS" : "GATE: FAIL");
        }
    }
    (void) sink;
    free(Wx4);
    free(Wt5);
    free(xq);
    free(ref);
    return 0;
}

#else
int main(void) {
    printf("SKIP: x86_64 only\n");
    return 77;
}
#endif
