/*
 * test_kv_int8_fused_unit — the architecture hands the attention over its
 * quantized KV caches to the backend's kernels (fused->attention_kv_int8,
 * and attention_kv_int4 for the packed INT4 cache that rides the INT8
 * storage) when the layer plan binds them, and gets the host loops' answer.
 *
 * On every CPU backend whose probe binds a kernel (cpu_x86 must bind the
 * INT8 one; cpu_neon, where built with FEAT_DotProd, both), for each kernel
 * it binds:
 *
 * 1. The call site, directly. transformer_kv_store_attention runs twice on
 *    the same query and cache, once with the plan's fuse_attn_kv_int8 set
 *    and once without (the host loop, attention_int8_via_buffers): decode
 *    at the first position and deep in the context (split), prefill chunks
 *    after a prefix, sliding windows, the rotated cache (GEIST_KV_ROT: Q
 *    rotated before, the output after, both in the architecture), and a
 *    head_dim the rotation does not cover; the INT4 cases the same over
 *    packed rows. The cache holds more rows than are live. Every output
 *    must be written (poisoned first), the query left as the host loop
 *    leaves it (bit for bit), and the outputs must agree within 1e-5 of
 *    their scale: the kernels differ only in how -ffast-math groups the
 *    softmax sums (~1e-6 measured). A view one row short, the query
 *    position off by one, or a rotation left out is off by 1e-3 or more.
 *
 * 2. A model. The llama of model_fixtures.h with 8 query heads on 2 KV
 *    heads of 64 is loaded twice, once as is and once with
 *    GEIST_KV_INT8_FUSED=0: every layer plan must bind the kernel in the
 *    first and not in the second. Both take the same 300-token prompt
 *    (several prefill chunks) and then 40 tokens one at a time, the cache
 *    plain and rotated; their logits must agree within 5e-2 of their
 *    range. That bound is loose on purpose: cpu_x86 runs this fixture's
 *    F32 weights as W8A8 (int8 activations), so a last-bit difference in
 *    an attention output flips activation roundings downstream (the host
 *    loop against itself with 1e-6 relative noise reaches 1.6e-2; a
 *    rotation left out, 0.89). Part 1 is the tight check.
 */
#define _POSIX_C_SOURCE 200809L /* setenv, unsetenv */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "test_helpers.h"
#include "model_fixtures.h"

#include "src/archs/transformer/forward/internal.h"
#include "src/archs/transformer/forward.h"
#include "src/engine/model.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include "heap.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const BACKENDS[] = {"cpu_x86", "cpu_neon", "cpu_scalar"};

/* Whether backend `name` has to bind attention_kv_int8 (attention_kv_int4
 * when int4) in this build. */
static bool must_bind(const char *name, bool int4) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (strcmp(name, "cpu_neon") == 0) {
        return true;
    }
#endif
    return !int4 && strcmp(name, "cpu_x86") == 0;
}

/* Whether `be` binds attention_kv_int8 (attention_kv_int4 when int4) at
 * this geometry. */
static bool
binds(struct geist_backend *be, bool int4, size_t n_q_heads, size_t n_kv_heads, size_t hd) {
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    const struct geist_fusion_query   q     = {.op         = int4 ? GEIST_FUSED_ATTN_KV_INT4
                                                                  : GEIST_FUSED_ATTN_KV_INT8,
                                               .m          = 64,
                                               .head_dim   = hd,
                                               .n_q_heads  = n_q_heads,
                                               .n_kv_heads = n_kv_heads};
    const bool                        slot =
            int4 ? fused->attention_kv_int4 != nullptr : fused->attention_kv_int8 != nullptr;
    return slot && fused->supported != nullptr && fused->supported(be, &q);
}

/* ---- 1. The call site ------------------------------------------------- */

struct wcase {
    size_t seq, q_position, n_q_heads, n_kv_heads, hd, window;
    bool   rot;
    bool   int4; /* the packed INT4 cache and attention_kv_int4 */
};

static const struct wcase WCASES[] = {
        /* decode: the first position; deep in the context (split); a window */
        {1, 0, 8, 2, 64, 0, false, false},
        {1, 699, 8, 2, 64, 0, false, false},
        {1, 699, 8, 2, 64, 0, true, false},
        {1, 1000, 12, 4, 128, 256, true, false},
        {1, 2047, 8, 1, 256, 0, true, false},
        /* prefill: the first chunk; chunks after a prefix */
        {64, 0, 8, 2, 64, 0, true, false},
        {37, 300, 8, 2, 64, 0, true, false},
        {20, 400, 12, 4, 128, 256, false, false},
        /* head_dim 80: the rotation does not apply, asked or not */
        {1, 500, 4, 4, 80, 0, true, false},
        {9, 120, 4, 4, 80, 64, true, false},
        /* the packed INT4 cache: the same kinds of call */
        {1, 0, 8, 2, 64, 0, true, true},
        {1, 699, 8, 2, 64, 0, false, true},
        {1, 699, 8, 2, 64, 0, true, true},
        {1, 1000, 12, 4, 128, 256, true, true},
        {37, 300, 8, 2, 64, 0, true, true},
        {9, 120, 4, 4, 80, 64, true, true},
};

static uint32_t g_rng = 0x9E3779B9u;
static uint32_t next_u32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

/* A buffer of `bytes` on `be`, filled from `src` when it is not nullptr. */
static struct geist_buffer *buffer(struct geist_backend *be, size_t bytes, const void *src) {
    const struct geist_backend_vtbl *vt = be->desc->vtbl;
    struct geist_buffer             *b  = nullptr;
    if (vt->buffer_create(be, bytes, GEIST_BUFFER_SCRATCH, 0, &b) != GEIST_OK || b == nullptr) {
        return nullptr;
    }
    if (src != nullptr) {
        memcpy(vt->buffer_map(b), src, bytes);
        vt->buffer_unmap(b);
    }
    return b;
}

static struct transformer_arch_state    g_st;
static struct transformer_arch_session  g_sess;
static struct transformer_layer_weights g_L;

/* transformer_kv_store_attention on case `c`, the kernel bound or not: the
 * query from `q` (the call rotates it in place), the output poisoned
 * first. Leaves the output and the query after the call in out / q_after. */
static enum geist_status call_site(struct geist_backend *be,
                                   const struct wcase   *c,
                                   bool                  fuse,
                                   struct geist_buffer  *bufs[static 6],
                                   size_t                arena_bytes,
                                   void                 *arena,
                                   const float          *q,
                                   const float          *poison,
                                   float                *out,
                                   float                *q_after) {
    const struct geist_backend_vtbl *vt      = be->desc->vtbl;
    const size_t                     q_elems = c->seq * c->n_q_heads * c->hd;
    memcpy(vt->buffer_map(bufs[0]), q, q_elems * sizeof *q);
    vt->buffer_unmap(bufs[0]);
    memcpy(vt->buffer_map(bufs[1]), poison, q_elems * sizeof *poison);
    vt->buffer_unmap(bufs[1]);

    g_st                                       = (struct transformer_arch_state) {0};
    g_st.n_q_heads                             = c->n_q_heads;
    g_st.n_kv_heads                            = c->n_kv_heads;
    g_sess                                     = (struct transformer_arch_session) {0};
    g_sess.kv_int8_enabled                     = true;
    g_sess.kv_int4_packed_enabled              = c->int4;
    g_sess.kv_rot_enabled                      = c->rot;
    g_sess.scratch_q                           = bufs[0];
    g_sess.scratch_attn                        = bufs[1];
    g_L                                        = (struct transformer_layer_weights) {0};
    g_L.head_dim                               = c->hd;
    g_L.sliding_window                         = c->window;
    const struct transformer_layer_exec_plan P = {.fuse_attn_kv_int8 = fuse,
                                                  .fuse_attn_kv_int4 = fuse};
    frame_arena_init(&g_sess.scratch_arena, arena, arena_bytes);
    struct transformer_layer_forward_ctx ctx = {.st                = &g_st,
                                                .sess              = &g_sess,
                                                .be                = be,
                                                .v                 = vt,
                                                .fused             = geist_backend_fused_tbl(be),
                                                .L                 = &g_L,
                                                .P                 = &P,
                                                .q_position        = c->q_position,
                                                .seq               = c->seq,
                                                .kv_int8_enabled   = true,
                                                .hd                = c->hd,
                                                .q_out             = c->n_q_heads * c->hd,
                                                .kv_out            = c->n_kv_heads * c->hd,
                                                .SEQ               = (int64_t) c->seq,
                                                .kv_len_now        = c->q_position + c->seq,
                                                .k_cache_q8_buf    = bufs[2],
                                                .v_cache_q8_buf    = bufs[3],
                                                .k_cache_scale_buf = bufs[4],
                                                .v_cache_scale_buf = bufs[5]};
    const struct geist_tensor            t_q =
            view_3d(bufs[0], (int64_t) c->seq, (int64_t) c->n_q_heads, (int64_t) c->hd);
    struct geist_tensor t_out =
            view_3d(bufs[1], (int64_t) c->seq, (int64_t) c->n_q_heads, (int64_t) c->hd);
    const enum geist_status s = transformer_kv_store_attention(&ctx, &t_q, &t_out);
    memcpy(out, vt->buffer_map(bufs[1]), q_elems * sizeof *out);
    vt->buffer_unmap(bufs[1]);
    memcpy(q_after, vt->buffer_map(bufs[0]), q_elems * sizeof *q_after);
    vt->buffer_unmap(bufs[0]);
    return s;
}

/* Case `c` both ways; *worst gets the larger of itself and max|d| / scale. */
static int
wiring_case(const char *backend, struct geist_backend *be, const struct wcase *c, double *worst) {
    const struct geist_backend_vtbl *vt = be->desc->vtbl;
    /* The cache holds 5 rows past the live ones, filled with values that
     * would show if a view reached them; INT4 rows are half as many bytes. */
    const size_t n_kv = c->q_position + c->seq, cap = n_kv + 5;
    const size_t row      = c->int4 ? c->hd / 2 : c->hd;
    const size_t q_elems  = c->seq * c->n_q_heads * c->hd;
    const size_t kv_elems = cap * c->n_kv_heads * row;
    const size_t n_sc     = cap * c->n_kv_heads;
    const size_t arena_bytes =
            attention_int8_scratch_floats(c->n_q_heads, c->hd) * sizeof(float) + 64;
    float               *q       = heap_alloc_array_aligned(float, q_elems);
    float               *poison  = heap_alloc_array_aligned(float, q_elems);
    float               *out_f   = heap_alloc_array_aligned(float, q_elems);
    float               *out_h   = heap_alloc_array_aligned(float, q_elems);
    float               *q_f     = heap_alloc_array_aligned(float, q_elems);
    float               *q_h     = heap_alloc_array_aligned(float, q_elems);
    int8_t              *k       = heap_alloc_array_aligned(int8_t, kv_elems);
    int8_t              *v       = heap_alloc_array_aligned(int8_t, kv_elems);
    float               *ks      = heap_alloc_array_aligned(float, n_sc);
    float               *vs      = heap_alloc_array_aligned(float, n_sc);
    void                *arena   = heap_alloc_aligned(arena_bytes, 64);
    struct geist_buffer *bufs[6] = {nullptr};
    int                  fails   = 0;
    char                 msg[200];
    if (q == nullptr || poison == nullptr || out_f == nullptr || out_h == nullptr ||
        q_f == nullptr || q_h == nullptr || k == nullptr || v == nullptr || ks == nullptr ||
        vs == nullptr || arena == nullptr) {
        fails += geist_expect(false, "allocation");
        goto done;
    }
    const float POISON = -7.5e30f;
    for (size_t i = 0; i < q_elems; i++) {
        q[i]      = (float) ((int) (next_u32() % 2001u) - 1000) * 1e-3f;
        poison[i] = POISON;
    }
    for (size_t i = 0; i < kv_elems; i++) {
        const bool live = i < n_kv * c->n_kv_heads * row;
        if (c->int4) { /* any two nibbles; 7 and 7 past the live rows */
            k[i] = live ? (int8_t) (next_u32() & 0xFFu) : 0x77;
            v[i] = live ? (int8_t) (next_u32() & 0xFFu) : 0x77;
        } else {
            k[i] = live ? (int8_t) ((int) (next_u32() % 255u) - 127) : 127;
            v[i] = live ? (int8_t) ((int) (next_u32() % 255u) - 127) : 127;
        }
    }
    for (size_t i = 0; i < n_sc; i++) {
        const bool live = i < n_kv * c->n_kv_heads;
        ks[i]           = live ? 0.005f + (float) (next_u32() % 1000u) * 1e-5f : 1e3f;
        vs[i]           = live ? 0.005f + (float) (next_u32() % 1000u) * 1e-5f : 1e3f;
    }
    bufs[0] = buffer(be, q_elems * sizeof(float), nullptr);
    bufs[1] = buffer(be, q_elems * sizeof(float), nullptr);
    bufs[2] = buffer(be, kv_elems, k);
    bufs[3] = buffer(be, kv_elems, v);
    bufs[4] = buffer(be, n_sc * sizeof(float), ks);
    bufs[5] = buffer(be, n_sc * sizeof(float), vs);
    for (size_t i = 0; i < 6; i++) {
        if (bufs[i] == nullptr) {
            fails += geist_expect(false, "buffers");
            goto done;
        }
    }
    const enum geist_status sf =
            call_site(be, c, true, bufs, arena_bytes, arena, q, poison, out_f, q_f);
    const enum geist_status sh =
            call_site(be, c, false, bufs, arena_bytes, arena, q, poison, out_h, q_h);
    size_t unwritten = 0;
    double scale = 0.0, max_d = 0.0;
    for (size_t i = 0; i < q_elems; i++) {
        unwritten += memcmp(&out_f[i], &POISON, sizeof POISON) == 0 ||
                     memcmp(&out_h[i], &POISON, sizeof POISON) == 0;
        const double d = fabs((double) out_f[i] - (double) out_h[i]);
        const double a = fabs((double) out_h[i]);
        max_d          = d > max_d ? d : max_d;
        scale          = a > scale ? a : scale;
    }
    snprintf(msg,
             sizeof msg,
             "%s%s seq=%zu at %zu, heads %zu/%zu, hd=%zu, window %zu%s: status %d/%d, %zu "
             "unwritten, max|d| %.2e of scale %.2e",
             backend,
             c->int4 ? " INT4" : "",
             c->seq,
             c->q_position,
             c->n_q_heads,
             c->n_kv_heads,
             c->hd,
             c->window,
             c->rot ? ", rotated" : "",
             (int) sf,
             (int) sh,
             unwritten,
             max_d,
             scale);
    fails += geist_expect(sf == GEIST_OK && sh == GEIST_OK && unwritten == 0 && scale > 0.0 &&
                                  max_d <= 1e-5 * scale,
                          msg);
    if (scale > 0.0 && max_d / scale > *worst) {
        *worst = max_d / scale;
    }
    fails += geist_expect(memcmp(q_f, q_h, q_elems * sizeof *q_f) == 0,
                          "the query after the call is the host loop's");
done:
    for (size_t i = 0; i < 6; i++) {
        if (bufs[i] != nullptr) {
            vt->buffer_destroy(be, bufs[i]);
        }
    }
    safe_free((void **) &q);
    safe_free((void **) &poison);
    safe_free((void **) &out_f);
    safe_free((void **) &out_h);
    safe_free((void **) &q_f);
    safe_free((void **) &q_h);
    safe_free((void **) &k);
    safe_free((void **) &v);
    safe_free((void **) &ks);
    safe_free((void **) &vs);
    safe_free(&arena);
    return fails;
}

static int check_call_site(const char *backend, struct geist_backend *be) {
    int    fails  = 0;
    size_t ran[2] = {0, 0};
    double worst  = 0.0;
    for (size_t i = 0; i < sizeof WCASES / sizeof WCASES[0]; i++) {
        const struct wcase *c = &WCASES[i];
        if (!binds(be, c->int4, c->n_q_heads, c->n_kv_heads, c->hd)) {
            fails += geist_expect(!must_bind(backend, c->int4), "the backend binds the kernel");
            continue;
        }
        ran[c->int4]++;
        fails += wiring_case(backend, be, c, &worst);
    }
    if (fails == 0) {
        printf("  %s: the call site gives the host loops' output within 1e-5 of its scale "
               "(%zu INT8 and %zu INT4 cases, worst %.2e)\n",
               backend,
               ran[0],
               ran[1],
               worst);
    }
    return fails;
}

/* ---- 2. A model ------------------------------------------------------- */

constexpr size_t VOCAB   = 512;
constexpr size_t PROMPT  = 300;
constexpr size_t STEPS   = 40;
constexpr double REL_TOL = 5e-2;

static geist_token_t TOKENS[PROMPT + STEPS];

/* Whether every layer plan of `m` binds the INT8 kernel (the INT4 one when
 * int4), if want, or none does. */
static bool plans_bind(struct geist_model *m, bool int4, bool want) {
    const struct transformer_arch_state *st =
            (const struct transformer_arch_state *) geist_model_internal_arch_meta(m);
    if (st == nullptr || st->layer_plans == nullptr || st->n_layers == 0) {
        return false;
    }
    for (size_t i = 0; i < st->n_layers; i++) {
        const struct transformer_layer_exec_plan *P = &st->layer_plans[i];
        if ((int4 ? P->fuse_attn_kv_int4 : P->fuse_attn_kv_int8) != want) {
            return false;
        }
    }
    return true;
}

/* The pending logits of `s`, or nullptr. */
static const float *logits(struct geist_session *s) {
    size_t       n = 0;
    const float *p = geist_session_peek_logits(&n, s);
    return n == VOCAB ? p : nullptr;
}

/* max |a - b| over the range of a; +inf if either is missing. */
static double rel_diff(const float *a, const float *b) {
    if (a == nullptr || b == nullptr) {
        return INFINITY;
    }
    float  lo = a[0], hi = a[0];
    double d = 0.0;
    for (size_t i = 0; i < VOCAB; i++) {
        lo              = a[i] < lo ? a[i] : lo;
        hi              = a[i] > hi ? a[i] : hi;
        const double di = fabs((double) a[i] - (double) b[i]);
        d               = di > d ? di : d;
    }
    return hi > lo ? d / (double) (hi - lo) : d;
}

/* The two models side by side over the prompt and the steps, on the INT8
 * cache (the packed INT4 one when int4), rotated or not. */
static int compare(const char           *backend,
                   struct geist_backend *be,
                   struct geist_model   *mf,
                   struct geist_model   *mh,
                   bool                  int4,
                   bool                  rot) {
    char msg[160];
    setenv("GEIST_KV_ROT", rot ? "1" : "0", 1); /* INT4 rotates by default */
    const struct geist_session_opts o  = {.kv_mode = int4 ? GEIST_KV_INT4 : GEIST_KV_INT8,
                                          .top_p   = 1.0f};
    struct geist_session           *sf = nullptr, *sh = nullptr;
    const bool                      ok = geist_session_create(mf, be, &o, &sf) == GEIST_OK &&
                                         geist_session_create(mh, be, &o, &sh) == GEIST_OK;
    unsetenv("GEIST_KV_ROT");
    int    fails = 0;
    double worst = 0.0;
    if (!ok) {
        snprintf(msg,
                 sizeof msg,
                 "%s %s%s: sessions",
                 backend,
                 int4 ? "INT4" : "INT8",
                 rot ? " rotated" : "");
        fails += geist_expect(false, msg);
        goto out;
    }
    fails += geist_expect(geist_session_prefill_tokens(sf, PROMPT, TOKENS) == GEIST_OK &&
                                  geist_session_prefill_tokens(sh, PROMPT, TOKENS) == GEIST_OK,
                          "prefill");
    worst = rel_diff(logits(sh), logits(sf));
    for (size_t i = 0; i < STEPS && fails == 0; i++) {
        const geist_token_t t = TOKENS[PROMPT + i];
        fails += geist_expect(geist_session_prefill_tokens(sf, 1, &t) == GEIST_OK &&
                                      geist_session_prefill_tokens(sh, 1, &t) == GEIST_OK,
                              "step");
        const double d = rel_diff(logits(sh), logits(sf));
        worst          = d > worst ? d : worst;
    }
    snprintf(msg,
             sizeof msg,
             "%s %s%s: the bound model's logits within %.0e of the host loop's (worst %.2e)",
             backend,
             int4 ? "INT4" : "INT8",
             rot ? " rotated" : "",
             REL_TOL,
             worst);
    fails += geist_expect(worst <= REL_TOL, msg);
    if (fails == 0) {
        printf("  %s\n", msg);
    }
out:
    geist_session_destroy(sf);
    geist_session_destroy(sh);
    return fails;
}

static int check_model(const char *backend, struct geist_backend *be, const struct tf_buf *g) {
    struct geist_model *mf = nullptr, *mh = nullptr;
    const bool          lf = geist_model_load_from_memory(g->b, g->n, be, &mf) == GEIST_OK;
    setenv("GEIST_KV_INT8_FUSED", "0", 1);
    const bool lh = geist_model_load_from_memory(g->b, g->n, be, &mh) == GEIST_OK;
    unsetenv("GEIST_KV_INT8_FUSED");
    int fails = 0;
    if (!lf || !lh) {
        fprintf(stderr, "FAIL: %s: model load: %s\n", backend, geist_last_create_error());
        fails++;
    } else {
        const bool int4 = binds(be, true, 8, 2, 64);
        fails += geist_expect(plans_bind(mf, false, true), "every layer plan binds the kernel");
        fails += geist_expect(plans_bind(mf, true, int4),
                              "every layer plan binds the INT4 kernel where the backend has it");
        fails += geist_expect(plans_bind(mh, false, false) && plans_bind(mh, true, false),
                              "GEIST_KV_INT8_FUSED=0: no layer plan binds them");
        fails += compare(backend, be, mf, mh, false, false);
        fails += compare(backend, be, mf, mh, false, true);
        if (int4) {
            fails += compare(backend, be, mf, mh, true, false);
            fails += compare(backend, be, mf, mh, true, true);
        }
    }
    geist_model_destroy(mf);
    geist_model_destroy(mh);
    return fails;
}

int main(void) {
    for (size_t i = 0; i < PROMPT + STEPS; i++) {
        TOKENS[i] = (geist_token_t) (2 + (i * 97 + i / 7) % (VOCAB - 2));
    }
    struct tf_buf g     = mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                                            .d_model  = 512,
                                                            .heads    = 8,
                                                            .kv_heads = 2,
                                                            .ffn      = 256,
                                                            .vocab    = VOCAB,
                                                            .context  = 512,
                                                            .seed     = 23});
    int           fails = 0, ran = 0;
    for (size_t b = 0; b < sizeof BACKENDS / sizeof BACKENDS[0]; b++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(BACKENDS[b], nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
            continue; /* not in this build */
        }
        if (!binds(be, false, 8, 2, 64)) {
            fails += geist_expect(!must_bind(BACKENDS[b], false),
                                  "the backend binds attention_kv_int8");
        } else {
            ran++;
            fails += check_call_site(BACKENDS[b], be);
            fails += check_model(BACKENDS[b], be, &g);
        }
        geist_backend_destroy(be);
    }
    free(g.b);
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    if (ran == 0) {
        printf("SKIP: no backend in this build binds attention_kv_int8\n");
        return GEIST_TEST_SKIP;
    }
    printf("PASS: the architecture's quantized-KV attention on the backends' kernels\n");
    return GEIST_TEST_PASS;
}
