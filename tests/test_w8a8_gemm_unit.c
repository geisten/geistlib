/*
 * test_w8a8_gemm_unit — the cpu_x86 W8A8 kernels agree bit for bit (#476).
 *
 * A row of output must not depend on whether it is computed alone (decode)
 * or in a chunk (prefill), nor on the layout or ISA tier that computes it:
 * a last-bit difference can move a following int8 activation quantization
 * by a whole grid step. So every kernel is held to the one-row GEMV, the
 * oracle, with memcmp:
 *   - w8a8_dot_scalar (the tier below AVX-512 VNNI), row by row;
 *   - w8a8_gemm over the chunk, and over one token at a time;
 *   - on VNNI hosts, w8x8_gemm (N % 8 == 0) and w8x16_gemm (N % 16 == 0)
 *     on the repacked weights, over the chunk and one token at a time.
 * Both weight formats the kernels serve: the Q6_K predecode's u8 [0, 63]
 * with a positive offset, and F32 weights through f32_to_w8a8_row (u8
 * [0, 255], offset -min of either sign). Widths with and without a
 * multiple of 8 / 16 rows, an odd block count (the one-block tail), a
 * short last activation group, token counts on and off the JT tile.
 * The GEMV itself must stay within 1e-3 of the dequantized reference.
 *
 * Deterministic; no GGUF needed.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "../src/backends/cpu_x86/kernel_w8a8.h"
#include "../src/backends/cpu_x86/linear_f32q.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t prng_next(uint32_t *s) {
    uint32_t z = (*s += 0x9E3779B9u);
    z          = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z          = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

static float frand(uint32_t *s) {
    return 2.0f * ((prng_next(s) & 0xFFFFu) / 65536.0f) - 1.0f;
}

enum wfmt { Q6K_RANGE, F32_QUANT };

/* Bitwise equality of n floats; prints the first difference. */
static int same_bits(const char *what,
                     size_t      M,
                     size_t      N,
                     size_t      K,
                     const float got[static M * N],
                     const float ref[static M * N]) {
    if (memcmp(got, ref, M * N * sizeof(float)) == 0) {
        return 0;
    }
    size_t i = 0;
    while (memcmp(&got[i], &ref[i], sizeof(float)) == 0) {
        i++;
    }
    fprintf(stdout,
            "FAIL [%s M=%zu N=%zu K=%zu] token %zu row %zu: %.9g vs the GEMV's %.9g\n",
            what,
            M,
            N,
            K,
            i / N,
            i % N,
            (double) got[i],
            (double) ref[i]);
    return 1;
}

/* A lane-parallel GEMM's repack + run, for one layout. */
typedef void (*repack_fn)(
        size_t, size_t, const uint8_t *, const float *, const float *, uint8_t *, float *, float *);
typedef void (*gemm_fn)(size_t,
                        size_t,
                        size_t,
                        const uint8_t *,
                        const float *,
                        const float *,
                        const int8_t *,
                        const int32_t *,
                        const float *,
                        float *);

/* gemm over all M tokens, then one token at a time, each against Yref. */
static int check_gemm(const char    *what,
                      gemm_fn        gemm,
                      size_t         M,
                      size_t         N,
                      size_t         K,
                      const uint8_t *W,
                      const float   *ws,
                      const float   *wo,
                      const int8_t  *acts,
                      const int32_t *sa,
                      const float   *sx,
                      const float   *Yref,
                      float         *Y) {
    const size_t NB = K / W8A8_BLOCK_ELEMS, NG = w8a8_act_groups(NB);
    char         label[64];
    gemm(M, N, NB, W, ws, wo, acts, sa, sx, Y);
    snprintf(label, sizeof label, "%s, chunk", what);
    int fails = same_bits(label, M, N, K, Y, Yref);
    for (size_t j = 0; j < M; j++) {
        gemm(1, N, NB, W, ws, wo, acts + j * K, sa + j * NB, sx + j * NG, Y + j * N);
    }
    snprintf(label, sizeof label, "%s, one token", what);
    return fails + same_bits(label, M, N, K, Y, Yref);
}

static int check_lanes(const char    *what,
                       repack_fn      repack,
                       gemm_fn        gemm,
                       size_t         M,
                       size_t         N,
                       size_t         K,
                       const uint8_t *W,
                       const float   *ws,
                       const float   *wo,
                       const int8_t  *acts,
                       const int32_t *sa,
                       const float   *sx,
                       const float   *Yref,
                       float         *Y) {
    const size_t NB  = K / W8A8_BLOCK_ELEMS;
    uint8_t     *qs  = malloc(N * K);
    float       *qsc = malloc(N * NB * sizeof(float));
    float       *qof = malloc(N * NB * sizeof(float));
    int          fails;
    if (qs == nullptr || qsc == nullptr || qof == nullptr) {
        fprintf(stderr, "out of memory\n");
        fails = 1;
    } else {
        repack(N, K, W, ws, wo, qs, qsc, qof);
        fails = check_gemm(what, gemm, M, N, K, qs, qsc, qof, acts, sa, sx, Yref, Y);
    }
    free(qs);
    free(qsc);
    free(qof);
    return fails;
}

/* One scenario's buffers (W8A8 weights, their dequantized values, acts). */
struct bufs {
    uint8_t *W;
    float   *ws, *wo, *Wf, *X;
    int8_t  *acts;
    int32_t *sa;
    float   *sx, *Yref, *Y;
};

static int run(enum wfmt fmt, size_t M, size_t N, size_t K, uint32_t *s, const struct bufs *b) {
    const size_t NB = K / W8A8_BLOCK_ELEMS;
    const size_t NG = w8a8_act_groups(NB);
    uint8_t     *W  = b->W;
    float       *ws = b->ws, *wo = b->wo, *Wf = b->Wf, *X = b->X, *sx = b->sx;
    int8_t      *acts = b->acts;
    int32_t     *sa   = b->sa;
    float       *Yref = b->Yref, *Y = b->Y;
    int          fails = 0;

    if (fmt == F32_QUANT) {
        for (size_t i = 0; i < N * K; i++)
            Wf[i] = frand(s) * 0.05f + (i % 3 == 0 ? 0.02f : 0.0f);
        for (size_t r = 0; r < N; r++)
            f32_to_w8a8_row(K, Wf + r * K, W + r * K, ws + r * NB, wo + r * NB);
    } else {
        for (size_t i = 0; i < N * K; i++)
            W[i] = (uint8_t) (prng_next(s) & 0x3Fu);
        for (size_t i = 0; i < N * NB; i++) {
            ws[i] = 0.001f + ((prng_next(s) & 0xFFFFu) / 65536.0f) * 0.01f;
            wo[i] = ((prng_next(s) & 0xFFFFu) / 65536.0f) * 0.5f;
        }
    }
    /* What the kernels reconstruct: w = scale * u - offset. */
    for (size_t r = 0; r < N; r++)
        for (size_t k = 0; k < K; k++) {
            const size_t blk = r * NB + k / W8A8_BLOCK_ELEMS;
            Wf[r * K + k]    = ws[blk] * (float) W[r * K + k] - wo[blk];
        }

    for (size_t i = 0; i < M * K; i++)
        X[i] = frand(s);
    for (size_t j = 0; j < M; j++)
        w8a8_quantize_acts_row(K, X + j * K, acts + j * K, sa + j * NB, sx + j * NG);

    /* The oracle: the one-row GEMV, token by token ... */
    for (size_t j = 0; j < M; j++)
        w8a8_gemv(N, NB, W, ws, wo, acts + j * K, sa + j * NB, sx + j * NG, Yref + j * N);

    /* ... within rounding of the dequantized weights and activations (the
     * first two tokens: enough, and cheap in the sanitizer builds). */
    double maxd = 0.0;
    for (size_t j = 0; j < M && j < 2; j++)
        for (size_t r = 0; r < N; r++) {
            double y = 0.0;
            for (size_t k = 0; k < K; k++)
                y += (double) Wf[r * K + k] * (double) acts[j * K + k] *
                     (double) sx[j * NG + k / (W8A8_ACT_GROUP_BLOCKS * W8A8_BLOCK_ELEMS)];
            maxd = fmax(maxd, fabs(y - (double) Yref[j * N + r]));
        }
    if (maxd > 1e-3) {
        fprintf(stdout, "FAIL [gemv M=%zu N=%zu K=%zu] max |d| %g vs double\n", M, N, K, maxd);
        fails++;
    }

    /* The scalar tier's dot, row by row. */
    for (size_t j = 0; j < M; j++)
        for (size_t r = 0; r < N; r++)
            Y[j * N + r] = w8a8_dot_scalar(NB,
                                           W + r * K,
                                           ws + r * NB,
                                           wo + r * NB,
                                           acts + j * K,
                                           sa + j * NB,
                                           sx + j * NG);
    fails += same_bits("w8a8_dot_scalar", M, N, K, Y, Yref);

    fails += check_gemm("w8a8_gemm", w8a8_gemm, M, N, K, W, ws, wo, acts, sa, sx, Yref, Y);
    const bool vnni = w8a8_isa_is_vnni();
    if (vnni && N % W8X8_NROWS == 0)
        fails += check_lanes(
                "w8x8_gemm", w8x8_repack, w8x8_gemm, M, N, K, W, ws, wo, acts, sa, sx, Yref, Y);
    if (vnni && N % W8X16_NROWS == 0)
        fails += check_lanes(
                "w8x16_gemm", w8x16_repack, w8x16_gemm, M, N, K, W, ws, wo, acts, sa, sx, Yref, Y);

    fprintf(stdout,
            "[%s M=%zu N=%zu K=%zu] gemv vs double %.3g; scalar, gemm%s%s %s\n",
            fmt == F32_QUANT ? "f32q" : "q6k ",
            M,
            N,
            K,
            maxd,
            vnni && N % W8X8_NROWS == 0 ? ", w8x8" : "",
            vnni && N % W8X16_NROWS == 0 ? ", w8x16" : "",
            fails == 0 ? "bit-identical" : "DIFFER");
    return fails;
}

static int scenario(enum wfmt fmt, size_t M, size_t N, size_t K) {
    const size_t NB = K / W8A8_BLOCK_ELEMS;
    const size_t NG = w8a8_act_groups(NB);
    uint32_t     s  = 0xB16B00B5u ^ (uint32_t) (M * 131 + N * 7 + K);

    /* The kernels' [static] bounds need every dimension >= 1. Stating it
     * also keeps gcc-14 -O1 (plain or -fsanitize=thread, #443) from threading
     * the zero-trip path of the quantize loop below into M == 0, where it
     * sees Y = malloc(0) against w8a8_gemm's out[static n_tokens * n_rows]
     * and reports -Wstringop-overflow. No scenario passes a zero dimension. */
    if (M == 0 || N == 0 || NB == 0)
        return 1;

    const struct bufs b = {.W    = malloc(N * K),
                           .ws   = malloc(N * NB * sizeof(float)),
                           .wo   = malloc(N * NB * sizeof(float)),
                           .Wf   = malloc(N * K * sizeof(float)),
                           .X    = malloc(M * K * sizeof(float)),
                           .acts = malloc(M * K),
                           .sa   = malloc(M * NB * sizeof(int32_t)),
                           .sx   = malloc(M * NG * sizeof(float)),
                           .Yref = malloc(M * N * sizeof(float)),
                           .Y    = malloc(M * N * sizeof(float))};
    int               fails;
    if (!b.W || !b.ws || !b.wo || !b.Wf || !b.X || !b.acts || !b.sa || !b.sx || !b.Yref || !b.Y) {
        fprintf(stderr, "out of memory\n");
        fails = 1;
    } else {
        fails = run(fmt, M, N, K, &s, &b);
    }
    free(b.W);
    free(b.ws);
    free(b.wo);
    free(b.Wf);
    free(b.X);
    free(b.acts);
    free(b.sa);
    free(b.sx);
    free(b.Yref);
    free(b.Y);
    return fails;
}

int main(void) {
    static const struct {
        size_t M, N, K;
    } shapes[] = {
            {4, 32, 256},      /* JT-exact, small */
            {7, 40, 512},      /* M not divisible by JT (=4) → tail; 8 | N, 16 ∤ N */
            {6, 48, 256},      /* three 16-row groups */
            {5, 32, 1040},     /* K % 256 != 0: a short last scale group, 65 blocks */
            {3, 36, 528},      /* N % 8 != 0 (row-major only), odd block count */
            {21, 520, 512},    /* the DeepSeek fixture's 520-row head (8 | 520, 16 ∤ 520) */
            {9, 512, 512},     /* 512 rows (16 | 512) over 32 blocks */
            {64, 1536, 12288}, /* real Gemma 4 ffn_down shape */
    };
    int fails = 0;
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++) {
        fails += scenario(Q6K_RANGE, shapes[i].M, shapes[i].N, shapes[i].K);
        fails += scenario(F32_QUANT, shapes[i].M, shapes[i].N, shapes[i].K);
    }
    if (!w8a8_isa_is_vnni())
        fprintf(stdout, "(no AVX-512 VNNI: w8x8 / w8x16 not run)\n");
    if (fails == 0)
        fprintf(stdout, "OK\n");
    return fails == 0 ? 0 : 1;
}
