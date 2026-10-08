/*
 * test_neon_act_outlier_unit — cpu_neon's W*A8 kernels quantize activations
 * with one scale per group, not per row (#698).
 *
 * Every token row carries one |x| = 100 element in its first 256 and
 * |x| <= 1 elsewhere, the shape of Gemma 4's FFN activations. The weights
 * of those first 256 inputs are zero, so the output comes from the quiet
 * elements alone. With one int8 scale per row the quiet elements sit on a
 * grid of 100/127 and the output is off by ~40 % of its norm; with one
 * scale per 256 elements (Q8_K: K-quants, IQ formats) or per 32 (Q8_0:
 * Q4_0, Q4_1, Q8_0, IQ4_NL) — llama.cpp's vec_dot granularity — the error
 * is the int8 rounding of |x| <= 1, ~0.4 %. The bar is 2 %.
 *
 * Each dtype runs through the resolver (linear_m1, linear_mN and the pair
 * kernels, under every kernel policy that selects a different activation
 * consumer) and through the quant.h entry points whose layouts the
 * resolver only installs for large tensors (Q6_K x8, Q4_0 x8). The
 * reference is geist_linear_ref: the dequantized weights against fp32 x.
 *
 * Deterministic; no GGUF needed. Skips without cpu_neon.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if !defined(GEIST_BACKEND_CPU_NEON)
int main(void) {
    printf("SKIP: cpu_neon backend not in this build\n");
    return GEIST_TEST_SKIP;
}
#else

#define GEIST_INTERNAL_BACKEND_LAYER
#include "../src/backends/cpu_neon/internal.h"

#include "linear_ref.h"
#include "quant.h"

constexpr size_t N_IN    = 4096; /* >= 4096: the Q6_K n-tile repacks need it */
constexpr size_t N_OUT   = 16;
constexpr size_t M       = 9;   /* an 8-token tile plus a remainder token */
constexpr size_t QUIET0  = 256; /* inputs [0, QUIET0) carry the outlier, zero weights */
constexpr float  OUTLIER = 100.0f;
constexpr double BAR     = 0.02; /* relative L2 error; per-row scales give ~0.4 */

static int g_fail = 0;

static uint32_t prng(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

/* Where a block keeps its fp16 scale(s): d always, and the min (Q4_K / Q5_K
 * dmin, Q4_1 m) when the format has one. */
struct fmt {
    const char *name;
    uint16_t    dtype;
    size_t      blk_elems;
    size_t      blk_bytes;
    size_t      d_off;
    int         min_off; /* -1: none */
};

static const struct fmt FMTS[] = {
        {"Q4_K", (uint16_t) GEIST_DTYPE_Q4_K, Q4_K_BLOCK_ELEMS, Q4_K_BLOCK_BYTES, 0, 2},
        {"Q5_K", (uint16_t) GEIST_DTYPE_Q5_K, Q5_K_BLOCK_ELEMS, Q5_K_BLOCK_BYTES, 0, 2},
        {"Q6_K", (uint16_t) GEIST_DTYPE_Q6_K, Q6_K_BLOCK_ELEMS, Q6_K_BLOCK_BYTES, 208, -1},
        {"Q3_K", (uint16_t) GEIST_DTYPE_Q3_K, Q3_K_BLOCK_ELEMS, Q3_K_BLOCK_BYTES, 108, -1},
        {"IQ2_S", (uint16_t) GEIST_DTYPE_IQ2_S, IQ2_S_BLOCK_ELEMS, IQ2_S_BLOCK_BYTES, 0, -1},
        {"IQ3_S", (uint16_t) GEIST_DTYPE_IQ3_S, IQ3_S_BLOCK_ELEMS, IQ3_S_BLOCK_BYTES, 0, -1},
        {"IQ4_XS", (uint16_t) GEIST_DTYPE_IQ4_XS, IQ4_XS_BLOCK_ELEMS, IQ4_XS_BLOCK_BYTES, 0, -1},
        {"IQ4_NL", (uint16_t) GEIST_DTYPE_IQ4_NL, IQ4_NL_BLOCK_ELEMS, IQ4_NL_BLOCK_BYTES, 0, -1},
        {"Q4_0", (uint16_t) GEIST_DTYPE_Q4_0, Q4_0_BLOCK_ELEMS, Q4_0_BLOCK_BYTES, 0, -1},
        {"Q4_1", (uint16_t) GEIST_DTYPE_Q4_1, Q4_1_BLOCK_ELEMS, Q4_1_BLOCK_BYTES, 0, 2},
        {"Q8_0", (uint16_t) GEIST_DTYPE_Q8_0, Q8_0_BLOCK_ELEMS, Q8_0_BLOCK_BYTES, 0, -1},
};

static void put_u16(uint8_t *p, uint16_t v) {
    memcpy(p, &v, sizeof v);
}

/* Random blocks with sane fp16 scales (2^-8 .. 2^-6, the min of either
 * sign); the blocks covering inputs [0, QUIET0) get zero scales, so their
 * weights dequantize to exactly 0. */
static uint8_t *make_weights(const struct fmt *f, uint32_t seed) {
    const size_t nb    = N_IN / f->blk_elems;
    const size_t bytes = N_OUT * nb * f->blk_bytes;
    uint8_t     *raw   = heap_alloc_array_aligned(uint8_t, bytes);
    if (raw == nullptr) {
        return nullptr;
    }
    uint32_t s = seed;
    for (size_t i = 0; i < bytes; i++) {
        raw[i] = (uint8_t) prng(&s);
    }
    for (size_t r = 0; r < N_OUT; r++) {
        for (size_t b = 0; b < nb; b++) {
            uint8_t   *blk  = raw + (r * nb + b) * f->blk_bytes;
            const bool loud = b * f->blk_elems < QUIET0;
            put_u16(blk + f->d_off, loud ? 0 : (uint16_t) (0x1C00u | (prng(&s) & 0x7FFu)));
            if (f->min_off >= 0) {
                put_u16(blk + f->min_off, loud ? 0 : (uint16_t) (0x1C00u | (prng(&s) & 0x87FFu)));
            }
        }
    }
    return raw;
}

static void make_acts(float x[static M * N_IN]) {
    uint32_t s = 0x6E0A5C1Du;
    for (size_t i = 0; i < M * N_IN; i++) {
        x[i] = 2.0f * (float) (prng(&s) & 0xFFFFu) / 65536.0f - 1.0f;
    }
    for (size_t t = 0; t < M; t++) {
        x[t * N_IN + 17] = (t & 1u) ? -OUTLIER : OUTLIER;
    }
}

/* Relative L2 error of y against ref over m rows; reports and fails above
 * BAR. */
static void check(const char *dtype, const char *path, size_t m, const float *y, const float *ref) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < m * N_OUT; i++) {
        const double d = (double) y[i] - (double) ref[i];
        num += d * d;
        den += (double) ref[i] * (double) ref[i];
    }
    const double rel = den > 0.0 ? sqrt(num / den) : sqrt(num);
    const bool   ok  = isfinite(rel) && rel <= BAR;
    printf("  %-6s %-36s rel_l2=%.5f%s\n", dtype, path, rel, ok ? "" : "  <-- FAIL");
    if (!ok) {
        g_fail = 1;
    }
}

static void free_aux(struct geist_weight *w) {
    if ((w->flags & GEIST_W_AUX_HEAP_OWNED) != 0 && w->aux_fp32 != nullptr) {
        void *aux = (void *) w->aux_fp32;
        safe_free(&aux);
    }
    w->aux_fp32 = nullptr;
}

/* A kernel-policy variant: which fields to set before resolve_weight, and
 * (late) after it — for a kernel the resolver reaches only when the policy
 * changed after the repack was chosen. */
typedef void (*policy_fn)(struct cpu_neon_kernel_policy *p);

static void p_plain(struct cpu_neon_kernel_policy *p) {
    p->q4k_predecode             = false;
    p->q4k_mtile_prefill         = false;
    p->q4k_ntile_prefill         = false;
    p->q4k_block_q8_prefill      = false;
    p->q4k_sgemm_prefill         = false;
    p->q6k_sgemm_prefill         = false;
    p->q6k_ntile_prefill         = false;
    p->q6k_ntile4_stream_prefill = false;
    p->q6k_x8_gemv               = false;
    p->q4_0_x8_gemv              = false;
    p->q5k_native_mn             = true;
    p->q8_0_native_mn            = true;
    p->q4_01_native_mn           = true;
    p->iq4xs_native_mn           = true;
}
static void p_q4k_predecode(struct cpu_neon_kernel_policy *p) {
    p_plain(p);
    p->q4k_predecode = true;
}
static void p_q4k_mtile8(struct cpu_neon_kernel_policy *p) {
    p_q4k_predecode(p);
    p->q4k_mtile_prefill = true;
}
static void p_q4k_bscale(struct cpu_neon_kernel_policy *p) {
    p_q4k_mtile8(p);
    p->q4k_block_q8_prefill = true;
}
static void p_q4k_ntile4(struct cpu_neon_kernel_policy *p) {
    p_q4k_mtile8(p);
    p->q4k_ntile_prefill = true;
}
static void p_q6k_ntile4(struct cpu_neon_kernel_policy *p) {
    p_plain(p);
    p->q6k_ntile_prefill = true;
}
static void p_q6k_stream(struct cpu_neon_kernel_policy *p) {
    p_q6k_ntile4(p);
    p->q6k_ntile4_stream_prefill = true;
}

struct variant {
    const char *name;
    uint16_t    dtype; /* 0: every dtype */
    policy_fn   at_resolve;
    policy_fn   at_call; /* nullptr: same as at_resolve */
};

static const struct variant VARIANTS[] = {
        {"plain", 0, p_plain, nullptr},
        {"q4k predecode", (uint16_t) GEIST_DTYPE_Q4_K, p_q4k_predecode, nullptr},
        {"q4k predecode mtile8", (uint16_t) GEIST_DTYPE_Q4_K, p_q4k_mtile8, nullptr},
        {"q4k predecode mtile4_ntile4", (uint16_t) GEIST_DTYPE_Q4_K, p_q4k_mtile8, p_q4k_ntile4},
        {"q4k predecode bscale", (uint16_t) GEIST_DTYPE_Q4_K, p_q4k_bscale, nullptr},
        {"q4k ntile4 packed", (uint16_t) GEIST_DTYPE_Q4_K, p_q4k_ntile4, nullptr},
        {"q6k ntile4", (uint16_t) GEIST_DTYPE_Q6_K, p_q6k_ntile4, nullptr},
        {"q6k ntile4 stream", (uint16_t) GEIST_DTYPE_Q6_K, p_q6k_stream, nullptr},
};

static void run_resolved(struct geist_backend *be,
                         const struct fmt     *f,
                         const struct variant *v,
                         uint8_t              *raw,
                         const float          *x,
                         const float          *ref,
                         float                *y,
                         float                *y1) {
    struct cpu_neon_state              *st    = (struct cpu_neon_state *) be->state;
    const struct cpu_neon_kernel_policy saved = st->policy;
    v->at_resolve(&st->policy);
    struct geist_weight w = {
            .raw        = raw,
            .raw_nbytes = N_OUT * (N_IN / f->blk_elems) * f->blk_bytes,
            .n_in       = (int32_t) N_IN,
            .n_out      = (int32_t) N_OUT,
            .dtype      = f->dtype,
    };
    if (be->desc->vtbl->resolve_weight(be, &w) != GEIST_OK || w.linear_m1 == nullptr ||
        w.linear_mN == nullptr) {
        printf("  %-6s %-36s not resolved on this host, skipped\n", f->name, v->name);
        st->policy = saved;
        free_aux(&w);
        return;
    }
    if (v->at_call != nullptr) {
        v->at_call(&st->policy);
    }

    char path[64];
    for (size_t t = 0; t < M; t++) {
        w.linear_m1(x + t * N_IN, &w, be, y + t * N_OUT);
    }
    snprintf(path, sizeof path, "%s m1", v->name);
    check(f->name, path, M, y, ref);

    w.linear_mN(M, x, &w, be, y);
    snprintf(path, sizeof path, "%s mN", v->name);
    check(f->name, path, M, y, ref);

    if (w.linear_pair_m1 != nullptr) {
        for (size_t t = 0; t < M; t++) {
            w.linear_pair_m1(x + t * N_IN, &w, &w, be, y + t * N_OUT, y1 + t * N_OUT);
        }
        snprintf(path, sizeof path, "%s pair m1 (w0)", v->name);
        check(f->name, path, M, y, ref);
        snprintf(path, sizeof path, "%s pair m1 (w1)", v->name);
        check(f->name, path, M, y1, ref);
    }
    if (w.linear_pair_mN != nullptr) {
        w.linear_pair_mN(M, x, &w, &w, be, y, y1);
        snprintf(path, sizeof path, "%s pair mN (w0)", v->name);
        check(f->name, path, M, y, ref);
        snprintf(path, sizeof path, "%s pair mN (w1)", v->name);
        check(f->name, path, M, y1, ref);
    }
    st->policy = saved;
    free_aux(&w);
}

/* The x8 interleaved layouts the resolver installs only for wide tensors
 * (Q6_K: n_out >= 32768, Q4_0: n_out >= 1024), through their quant.h entry
 * points; and the Q4_K predecoded GEMV. */
static void
run_packed(const struct fmt *f, const uint8_t *raw, const float *x, const float *ref, float *y) {
    size_t bytes = 0;
    if (f->dtype == (uint16_t) GEIST_DTYPE_Q6_K) {
        bytes = q6k_x8_gemv_size_bytes(N_IN, N_OUT);
    } else if (f->dtype == (uint16_t) GEIST_DTYPE_Q4_0) {
        bytes = q4_0_x8_gemv_size_bytes(N_IN, N_OUT);
    } else if (f->dtype == (uint16_t) GEIST_DTYPE_Q4_K) {
        bytes = q4k_predecode_size_bytes(N_IN, N_OUT);
    } else {
        return;
    }
    void *packed = bytes > 0 ? heap_alloc_aligned(bytes, 64) : nullptr;
    if (packed == nullptr) {
        printf("  %-6s packed layout unavailable\n", f->name);
        g_fail = 1;
        return;
    }
    if (f->dtype == (uint16_t) GEIST_DTYPE_Q6_K) {
        if (q6k_x8_gemv_pack(raw, N_IN, N_OUT, packed) != 0) {
            g_fail = 1;
        }
        for (size_t t = 0; t < M; t++) {
            linear_q6k_decode_w6a8_x8(N_IN, N_OUT, x + t * N_IN, packed, y + t * N_OUT);
        }
        check(f->name, "x8 gemv m1", M, y, ref);
    } else if (f->dtype == (uint16_t) GEIST_DTYPE_Q4_0) {
        if (q4_0_x8_gemv_pack(raw, N_IN, N_OUT, packed) != 0) {
            g_fail = 1;
        }
        for (size_t t = 0; t < M; t++) {
            linear_q4_0_decode_w4a8_x8(N_IN, N_OUT, x + t * N_IN, packed, y + t * N_OUT);
        }
        check(f->name, "x8 gemv m1", M, y, ref);
        linear_q4_0_w4a8_prefill_x8(M, N_IN, N_OUT, x, packed, y);
        check(f->name, "x8 gemm mN", M, y, ref);
    } else {
        if (q4k_predecode_pack(raw, N_IN, N_OUT, packed) != 0) {
            g_fail = 1;
        }
        for (size_t t = 0; t < M; t++) {
            linear_q4k_decode_w4a8_predecoded(N_IN, N_OUT, x + t * N_IN, packed, y + t * N_OUT);
        }
        check(f->name, "predecoded gemv m1", M, y, ref);
    }
    safe_free(&packed);
}

int main(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_neon", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        printf("SKIP: no cpu_neon backend: %s\n", geist_last_create_error());
        return GEIST_TEST_SKIP;
    }
    float *x   = heap_alloc_array_aligned(float, M *N_IN);
    float *ref = heap_alloc_array_aligned(float, M *N_OUT);
    float *y   = heap_alloc_array_aligned(float, M *N_OUT);
    float *y1  = heap_alloc_array_aligned(float, M *N_OUT);
    if (x == nullptr || ref == nullptr || y == nullptr || y1 == nullptr) {
        geist_backend_destroy(be);
        return GEIST_TEST_ERROR;
    }
    make_acts(x);

    for (size_t fi = 0; fi < sizeof FMTS / sizeof FMTS[0]; fi++) {
        const struct fmt *f   = &FMTS[fi];
        uint8_t          *raw = make_weights(f, 0x9E3779B9u + (uint32_t) fi);
        if (raw == nullptr) {
            g_fail = 1;
            continue;
        }
        const struct geist_weight wref = {
                .raw        = raw,
                .raw_nbytes = N_OUT * (N_IN / f->blk_elems) * f->blk_bytes,
                .n_in       = (int32_t) N_IN,
                .n_out      = (int32_t) N_OUT,
                .dtype      = f->dtype,
        };
        geist_linear_ref(M, x, &wref, ref);
        for (size_t vi = 0; vi < sizeof VARIANTS / sizeof VARIANTS[0]; vi++) {
            if (VARIANTS[vi].dtype == 0 || VARIANTS[vi].dtype == f->dtype) {
                run_resolved(be, f, &VARIANTS[vi], raw, x, ref, y, y1);
            }
        }
        run_packed(f, raw, x, ref, y);
        void *p = raw;
        safe_free(&p);
    }

    safe_free((void **) &x);
    safe_free((void **) &ref);
    safe_free((void **) &y);
    safe_free((void **) &y1);
    geist_backend_destroy(be);
    if (g_fail) {
        printf("FAIL: an activation outlier coarsened the rest of its row (bar %.2f)\n", BAR);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: one |x| = %.0f outlier leaves the other activation groups exact to %.2f\n",
           (double) OUTLIER,
           BAR);
    return GEIST_TEST_PASS;
}

#endif /* GEIST_BACKEND_CPU_NEON */
