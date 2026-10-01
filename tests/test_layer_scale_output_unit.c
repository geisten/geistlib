/*
 * test_layer_scale_output_unit — transformer_layer_scale_output.
 *
 * The step after a layer (layer_ple.c) multiplies the hidden state by the
 * layer's scalar: through the backend's scale_f32 where it has one (the
 * GPU backends), else in a host loop. Only Gemma 4 loads a scalar other
 * than 1, and no fixture here is a Gemma 4, so the step is checked on its
 * own, on cpu_scalar with counting wrappers around scale_f32 and
 * buffer_map:
 *
 *   - scalar 1, either path: no scale_f32 call, no map, the state
 *     unchanged to the bit;
 *   - scalars 0.5 and 1 + 2^-23 on the host loop: one map, every value
 *     scaled;
 *   - the same with a scale_f32: one call, with the scalar and the
 *     [seq, d_model] view, and no map.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "src/archs/transformer/forward/internal.h"

#include "test_helpers.h"

#include <geist.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static constexpr size_t SEQ = 3;
static constexpr size_t D   = 40;
static constexpr size_t N   = SEQ * D;

static int     g_maps;
static int     g_scales;
static float   g_scale_seen;
static int64_t g_rows_seen;
static int64_t g_cols_seen;
static void *(*g_map)(struct geist_buffer *buf);

static void *counting_map(struct geist_buffer *buf) {
    g_maps++;
    return g_map(buf);
}

static enum geist_status counting_scale(struct geist_backend      *be,
                                        const struct geist_tensor *x,
                                        float                      scale,
                                        struct geist_tensor       *y) {
    (void) be;
    (void) y;
    g_scales++;
    g_scale_seen = scale;
    g_rows_seen  = x->shape[0];
    g_cols_seen  = x->shape[1];
    return GEIST_OK;
}

/* Large: static, so zeroed and off the stack. */
static struct transformer_arch_state    g_st;
static struct transformer_layer_weights g_layer;

/* Runs the step with `scalar` on the host loop or through scale_f32, from
 * `in` into `out`. Counters are reset first. */
static enum geist_status run(struct geist_backend      *be,
                             struct geist_backend_vtbl *v,
                             struct geist_buffer       *buf,
                             bool                       device,
                             float                      scalar,
                             const float                in[static N],
                             float                      out[static N]) {
    struct geist_backend_primitives prims = *be->desc->prims;
    prims.scale_f32                       = counting_scale;
    g_st.d_model                          = D;
    g_st.model_fusions.prim_scale_f32     = device;
    g_layer.layer_scalar                  = scalar;
    v->buffer_upload(buf, N * sizeof(float), (const uint8_t *) in);
    g_maps = g_scales                        = 0;
    struct transformer_layer_forward_ctx ctx = {
            .st = &g_st, .be = be, .v = v, .prims = &prims, .L = &g_layer};
    ctx.seq                   = SEQ;
    ctx.SEQ                   = (int64_t) SEQ;
    ctx.h_out_buf             = buf;
    const enum geist_status s = transformer_layer_scale_output(&ctx);
    v->buffer_download(N * sizeof(float), (uint8_t *) out, buf);
    return s;
}

int main(void) {
    struct geist_backend *be = nullptr;
    GEIST_SKIP_IF(geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK,
                  "cpu_scalar backend not available");
    struct geist_backend_vtbl v = *be->desc->vtbl;
    g_map                       = v.buffer_map;
    v.buffer_map                = counting_map;

    struct geist_buffer *buf   = nullptr;
    int                  fails = geist_expect(
            v.buffer_create(
                    be, N * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &buf) ==
                            GEIST_OK &&
                    buf != nullptr,
            "hidden-state buffer created");
    if (buf == nullptr) {
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    float    in[N], out[N];
    uint32_t seed = 0x9E3779B9u;
    for (size_t i = 0; i < N; i++) {
        seed  = seed * 1664525u + 1013904223u;
        in[i] = ((float) (seed >> 8) / 16777216.0f - 0.5f) * 8.0f; /* [-4, 4) */
    }

    for (int device = 0; device <= 1; device++) {
        const char *path = device ? "scale_f32" : "host loop";
        char        what[96];
        fails += geist_expect(run(be, &v, buf, device, 1.0f, in, out) == GEIST_OK, "scalar 1 ran");
        snprintf(what, sizeof what, "%s, scalar 1: no scale_f32 call and no map", path);
        fails += geist_expect(g_scales == 0 && g_maps == 0, what);
        snprintf(what, sizeof what, "%s, scalar 1: the state unchanged", path);
        fails += geist_expect(memcmp(in, out, sizeof in) == 0, what);
    }

    /* 0.5, and the float just above 1 */
    const float scalars[] = {0.5f, 0x1.000002p+0f};
    for (size_t k = 0; k < sizeof scalars / sizeof scalars[0]; k++) {
        const float s = scalars[k];
        char        what[96];
        fails += geist_expect(run(be, &v, buf, false, s, in, out) == GEIST_OK, "host loop ran");
        bool scaled = true;
        for (size_t i = 0; i < N; i++) {
            scaled = scaled && out[i] == in[i] * s;
        }
        snprintf(what, sizeof what, "host loop, scalar %a: one map, every value scaled", s);
        fails += geist_expect(g_maps == 1 && scaled, what);

        fails += geist_expect(run(be, &v, buf, true, s, in, out) == GEIST_OK, "scale_f32 ran");
        snprintf(what, sizeof what, "scale_f32, scalar %a: one call on [seq, d_model], no map", s);
        fails += geist_expect(g_scales == 1 && g_scale_seen == s && g_rows_seen == (int64_t) SEQ &&
                                      g_cols_seen == (int64_t) D && g_maps == 0,
                              what);
    }

    v.buffer_destroy(be, buf);
    geist_backend_destroy(be);
    if (fails == 0) {
        printf("PASS: the layer output scale skips a scalar of 1 and applies any other\n");
    }
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
