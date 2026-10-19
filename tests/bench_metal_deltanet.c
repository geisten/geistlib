/*
 * bench_metal_deltanet — time one prefill DeltaNet mixer call on Metal.
 *
 * One call is fused->deltanet_mix for one Gated-DeltaNet layer: causal
 * conv + SiLU, q/k L2 norms, the gated delta rule over m tokens and the
 * gated RMS norm into z, with non-zero recurrent and conv state. Shapes
 * are qwen35's linear-attention layers: Bonsai 2 27B (16 key heads, 48
 * value heads, head dims 128, conv kernel 4) and Qwen3.5 4B (16/32). m is
 * the prefill chunk (default 64 320 512; llama.cpp's ubatch is 512).
 * Each sample encodes ITERS back-to-back calls in one prefill batch, as
 * the engine's prefill submits them, and drains them with a 4-byte
 * download; the median sample over REPS gives ms per call. "per token"
 * divides by m and multiplies by the model's DeltaNet layer count (48 for
 * Bonsai, 24 for the 4B), the per-prompt-token cost of the mixer.
 *
 * Usage: bench_metal_deltanet [iters] [reps] [m ...]. SKIPs without a
 * Metal device. Timing: run it under the host's benchmark lock.
 */
#define _POSIX_C_SOURCE 200809L
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { MAX_REPS = 64, CONV_K = 4 };

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *) a, y = *(const double *) b;
    return (x > y) - (x < y);
}

struct shape {
    const char *name;
    size_t      nkh, nvh, dk, dv, layers;
};

struct dbuf {
    struct geist_buffer *b;
    float               *p;
};

static bool dev_alloc(struct geist_backend *be, size_t n, struct dbuf *out) {
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    if (v->buffer_create(be, n * sizeof(float), GEIST_BUFFER_SCRATCH, GEIST_MEMORY_AUTO, &out->b) !=
        GEIST_OK) {
        return false;
    }
    out->p = v->buffer_map(out->b);
    return out->p != nullptr;
}

static void fill(size_t n, float scale, float bias, uint32_t *rng, float x[static n]) {
    for (size_t i = 0; i < n; i++) {
        *rng = *rng * 1664525u + 1013904223u;
        x[i] = bias + ((float) (*rng >> 8) / 16777216.0f - 0.5f) * 2.0f * scale;
    }
}

static struct geist_tensor mat(struct geist_buffer *b, size_t rows, size_t cols) {
    return (struct geist_tensor) {.buffer = b,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 2,
                                  .shape  = {(int64_t) rows, (int64_t) cols},
                                  .stride = {(int64_t) cols, 1}};
}

static struct geist_tensor vec(struct geist_buffer *b, size_t n) {
    return (struct geist_tensor) {.buffer = b,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 1,
                                  .shape  = {(int64_t) n},
                                  .stride = {1}};
}

static int
bench_shape(struct geist_backend *be, const struct shape *sh, size_t m, int iters, int reps) {
    const struct geist_backend_vtbl  *v    = be->desc->vtbl;
    const struct geist_backend_fused *f    = geist_backend_fused_tbl(be);
    const size_t                      keyd = sh->nkh * sh->dk, vd = sh->nvh * sh->dv;
    const size_t                      cd    = 2u * keyd + vd;
    const size_t                      n[10] = {m * cd,
                                               m * vd,
                                               m * sh->nvh,
                                               m * sh->nvh,
                                               cd * CONV_K,
                                               sh->nvh,
                                               sh->nvh,
                                               sh->dv,
                                               (CONV_K - 1) * cd,
                                               sh->nvh * sh->dk * sh->dv};
    struct dbuf                       d[10] = {0};
    uint32_t                          rng   = 0x9e3779b9u;
    for (size_t i = 0; i < 10; i++) {
        if (!dev_alloc(be, n[i], &d[i])) {
            fprintf(stderr, "%s: backend buffer alloc failed\n", sh->name);
            return GEIST_TEST_ERROR;
        }
    }
    fill(n[0], 1.0f, 0.0f, &rng, d[0].p);  /* qkv */
    fill(n[1], 1.0f, 0.0f, &rng, d[1].p);  /* z */
    fill(n[2], 2.0f, 0.0f, &rng, d[2].p);  /* beta logits */
    fill(n[3], 2.0f, -1.0f, &rng, d[3].p); /* alpha */
    fill(n[4], 0.5f, 0.0f, &rng, d[4].p);  /* conv_w */
    fill(n[5], 0.5f, -1.0f, &rng, d[5].p); /* ssm_a (negative) */
    fill(n[6], 0.5f, 0.0f, &rng, d[6].p);  /* dt_bias */
    fill(n[7], 0.2f, 1.0f, &rng, d[7].p);  /* norm_w */
    fill(n[8], 0.5f, 0.0f, &rng, d[8].p);  /* conv state */
    fill(n[9], 0.05f, 0.0f, &rng, d[9].p); /* delta state */
    struct geist_tensor tq = mat(d[0].b, m, cd), tz = mat(d[1].b, m, vd);
    struct geist_tensor tb = mat(d[2].b, m, sh->nvh), ta = mat(d[3].b, m, sh->nvh);
    struct geist_tensor tcw = mat(d[4].b, cd, CONV_K), taw = vec(d[5].b, sh->nvh);
    struct geist_tensor tdt = vec(d[6].b, sh->nvh), tnw = vec(d[7].b, sh->dv);
    struct geist_tensor tcs = mat(d[8].b, CONV_K - 1, cd);
    struct geist_tensor tst = {.buffer = d[9].b,
                               .dtype  = GEIST_DTYPE_F32,
                               .layout = GEIST_LAYOUT_DENSE,
                               .ndim   = 3,
                               .shape  = {(int64_t) sh->nvh, (int64_t) sh->dk, (int64_t) sh->dv},
                               .stride = {(int64_t) (sh->dk * sh->dv), (int64_t) sh->dv, 1}};
    const struct geist_deltanet_mix_args args = {.qkv         = &tq,
                                                 .z           = &tz,
                                                 .beta        = &tb,
                                                 .alpha       = &ta,
                                                 .conv_w      = &tcw,
                                                 .ssm_a       = &taw,
                                                 .dt_bias     = &tdt,
                                                 .norm_w      = &tnw,
                                                 .conv_state  = &tcs,
                                                 .delta_state = &tst,
                                                 .seq         = m,
                                                 .n_k_heads   = sh->nkh,
                                                 .n_v_heads   = sh->nvh,
                                                 .head_k      = sh->dk,
                                                 .head_v      = sh->dv,
                                                 .conv_kernel = CONV_K,
                                                 .eps         = 1e-6f};
    float                                probe[1];
    double                               samples[MAX_REPS];
    for (int r = -1; r < reps; r++) { /* r == -1: warm-up */
        const double t0  = now_s();
        const int    tok = v->parallel_region_begin != nullptr
                                   ? v->parallel_region_begin(be, GEIST_REGION_PREFILL_BATCH)
                                   : 0;
        for (int i = 0; i < iters; i++) {
            if (f->deltanet_mix(be, &args) != GEIST_OK) {
                fprintf(stderr,
                        "%s: deltanet_mix failed: %s\n",
                        sh->name,
                        geist_backend_errmsg(be));
                return GEIST_TEST_ERROR;
            }
        }
        if (v->parallel_region_end != nullptr) {
            v->parallel_region_end(be, tok);
        }
        if (v->buffer_download(sizeof probe, (uint8_t *) probe, d[1].b) != GEIST_OK) {
            fprintf(stderr, "%s: download failed\n", sh->name);
            return GEIST_TEST_ERROR;
        }
        if (r >= 0) {
            samples[r] = (now_s() - t0) / iters;
        }
    }
    qsort(samples, (size_t) reps, sizeof samples[0], cmp_double);
    const double t = samples[reps / 2];
    printf("%-7s m=%-4zu heads=%zu/%zu d=%zu  %8.3f ms/call  %6.3f ms/token (x%zu layers)  "
           "(min %.3f max %.3f)%s\n",
           sh->name,
           m,
           sh->nkh,
           sh->nvh,
           sh->dk,
           t * 1e3,
           t * 1e3 / (double) m * (double) sh->layers,
           sh->layers,
           samples[0] * 1e3,
           samples[reps - 1] * 1e3,
           isfinite(probe[0]) ? "" : "  NONFINITE");
    fflush(stdout);
    for (size_t i = 0; i < 10; i++) {
        v->buffer_destroy(be, d[i].b);
    }
    return GEIST_TEST_PASS;
}

int main(int argc, char **argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 8;
    const int reps  = argc > 2 ? atoi(argv[2]) : 7;
    if (iters < 1 || reps < 1 || reps > MAX_REPS) {
        fprintf(stderr, "usage: %s [iters >= 1] [reps 1..%d] [m ...]\n", argv[0], MAX_REPS);
        return GEIST_TEST_ERROR;
    }
    struct geist_backend *be = nullptr;
    if (geist_backend_create("metal", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        GEIST_SKIP("metal backend unavailable (not built or no device)");
    }
    static const struct shape shapes[] = {
            {"bonsai", 16, 48, 128, 128, 48},
            {"q35-4b", 16, 32, 128, 128, 24},
    };
    size_t ms[16] = {64, 320, 512};
    size_t n_m    = 3;
    if (argc > 3) {
        n_m = 0;
        for (int a = 3; a < argc && n_m < sizeof ms / sizeof ms[0]; a++) {
            const long mv = atol(argv[a]);
            if (mv < 1 || mv > 512) {
                fprintf(stderr, "bad m (1..512): %s\n", argv[a]);
                geist_backend_destroy(be);
                return GEIST_TEST_ERROR;
            }
            ms[n_m++] = (size_t) mv;
        }
    }
    int rc = GEIST_TEST_PASS;
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0] && rc == GEIST_TEST_PASS; s++) {
        for (size_t j = 0; j < n_m && rc == GEIST_TEST_PASS; j++) {
            rc = bench_shape(be, &shapes[s], ms[j], iters, reps);
        }
    }
    geist_backend_destroy(be);
    return rc;
}
