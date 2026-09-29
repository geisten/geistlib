/*
 * test_kv_placement_unit — where a session puts its KV cache.
 *
 * On a backend with caps.kv_q8_block (cpu_x86), a layer's INT8/INT4 K and V
 * data are aliased slices of one buffer: K at its start, V after K's pages
 * and half a row into its page (half a page at most, a multiple of 64
 * bytes), so that a KV head's K and V rows do not take the same cache sets
 * (see alloc_kv_block in arch_state.c). With caps.kv_dense_block the dense
 * FP32 cache is placed the same way, V one KV head's slice on (rounded up
 * to 64 bytes), and not at all with one KV head. Checked on the fixture
 * llama with 1, 2 and 8 KV heads of 64, INT8, packed INT4 and FP32: the
 * slices lie where the rule puts them, do not overlap and start zeroed, and
 * a session prefills and decodes through them to finite logits. On
 * cpu_scalar, which leaves both bits unset, K and V stay buffers of their
 * own.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "test_helpers.h"
#include "model_fixtures.h"

#include "src/archs/transformer/arch_state.h"
#include "src/engine/model.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

constexpr size_t VOCAB = 512;
constexpr size_t PAGE  = 4096;

/* Whether the n bytes at p are all zero. */
static bool zeroed(size_t n, const uint8_t p[static n]) {
    for (size_t i = 0; i < n; i++) {
        if (p[i] != 0) {
            return false;
        }
    }
    return true;
}

/* The name of a cache mode in messages. */
static const char *mode_name(enum geist_kv_mode mode) {
    return mode == GEIST_KV_FP32 ? "FP32" : mode == GEIST_KV_INT4 ? "INT4" : "INT8";
}

/* The placement of every layer's cache in a session of `mode` on `m`. */
static int check_placement(const char           *backend,
                           struct geist_backend *be,
                           struct geist_model   *m,
                           enum geist_kv_mode    mode) {
    struct transformer_arch_state   *st   = geist_model_internal_arch_meta(m);
    const struct geist_session_opts  o    = {.kv_mode = mode, .top_p = 1.0f};
    struct transformer_arch_session *sess = transformer_session_alloc(st, &o);
    const char                      *name = mode_name(mode);
    char                             msg[192];
    snprintf(msg, sizeof msg, "%s %s: session", backend, name);
    int fails = geist_expect(sess != nullptr, msg);
    if (sess == nullptr) {
        return fails;
    }
    const struct geist_backend_vtbl *vt    = be->desc->vtbl;
    const bool                       dense = mode == GEIST_KV_FP32;
    const bool   block = dense ? be->desc->caps.kv_dense_block : be->desc->caps.kv_q8_block;
    const size_t elem  = sess->kv_f16_enabled ? 2 : sizeof(float); /* dense */
    for (size_t li = 0; li < st->n_layers; li++) {
        const size_t         hd    = st->layers[li].head_dim;
        const size_t         slice = dense ? hd * elem : mode == GEIST_KV_INT4 ? hd / 2 : hd;
        const size_t         row   = st->n_kv_heads * slice;
        const size_t         data  = st->max_seq_len * row;
        const size_t         span  = (data + PAGE - 1) / PAGE * PAGE;
        const size_t         skip  = dense ? (st->n_kv_heads > 1 ? (slice + 63) / 64 * 64 : 0)
                                           : (row < PAGE ? row : PAGE) / 2 / 64 * 64;
        struct geist_buffer *kb    = dense ? sess->k_cache[li] : sess->k_cache_q8[li];
        struct geist_buffer *vb    = dense ? sess->v_cache[li] : sess->v_cache_q8[li];
        uint8_t             *k     = kb != nullptr ? vt->buffer_map(kb) : nullptr;
        uint8_t             *v     = vb != nullptr ? vt->buffer_map(vb) : nullptr;
        snprintf(msg,
                 sizeof msg,
                 "%s %s layer %zu (%zu-byte rows): K and V caches",
                 backend,
                 name,
                 li,
                 row);
        fails += geist_expect(k != nullptr && v != nullptr && k != v, msg);
        if (k == nullptr || v == nullptr) {
            continue;
        }
        if (block) {
            const uint8_t *b = sess->kv_data[li] ? vt->buffer_map(sess->kv_data[li]) : nullptr;
            snprintf(msg,
                     sizeof msg,
                     "%s %s layer %zu: K at the block's start, V %zu bytes on (K's %zu "
                     "bytes, then %zu into the page)",
                     backend,
                     name,
                     li,
                     span + skip,
                     data,
                     skip);
            fails += geist_expect(b == k && v == k + span + skip, msg);
            snprintf(msg,
                     sizeof msg,
                     "%s %s layer %zu: V starts %zu bytes further into its page than K",
                     backend,
                     name,
                     li,
                     skip);
            fails += geist_expect(((uintptr_t) v - (uintptr_t) k) % PAGE == skip, msg);
        } else {
            snprintf(msg,
                     sizeof msg,
                     "%s %s layer %zu: no block, K and V buffers of their own",
                     backend,
                     name,
                     li);
            fails += geist_expect(sess->kv_data[li] == nullptr, msg);
        }
        snprintf(msg, sizeof msg, "%s %s layer %zu: zeroed", backend, name, li);
        fails += geist_expect(zeroed(data, k) && zeroed(data, v), msg);
    }
    transformer_session_free(st, sess);
    return fails;
}

/* A session of `mode` prefills 70 tokens and decodes 5 through its cache. */
static int check_run(const char           *backend,
                     struct geist_backend *be,
                     struct geist_model   *m,
                     enum geist_kv_mode    mode) {
    const struct geist_session_opts o = {.kv_mode = mode, .top_p = 1.0f};
    struct geist_session           *s = nullptr;
    geist_token_t                   toks[75];
    for (size_t i = 0; i < 75; i++) {
        toks[i] = (geist_token_t) (2 + (i * 97 + i / 7) % (VOCAB - 2));
    }
    bool ok = geist_session_create(m, be, &o, &s) == GEIST_OK &&
              geist_session_prefill_tokens(s, 70, toks) == GEIST_OK;
    for (size_t i = 70; i < 75 && ok; i++) {
        ok = geist_session_prefill_tokens(s, 1, &toks[i]) == GEIST_OK;
    }
    size_t       n = 0;
    const float *l = ok ? geist_session_peek_logits(&n, s) : nullptr;
    ok             = ok && l != nullptr && n == VOCAB;
    for (size_t i = 0; ok && i < VOCAB; i++) {
        ok = isfinite(l[i]);
    }
    geist_session_destroy(s);
    char msg[160];
    snprintf(msg,
             sizeof msg,
             "%s %s: 70 tokens prefilled and 5 decoded to finite logits",
             backend,
             mode_name(mode));
    return geist_expect(ok, msg);
}

int main(void) {
    static const char *const BACKENDS[] = {"cpu_x86", "cpu_scalar"};
    static const uint32_t    KV_HEADS[] = {1, 2, 8};
    int                      fails = 0, ran = 0;
    for (size_t b = 0; b < sizeof BACKENDS / sizeof BACKENDS[0]; b++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(BACKENDS[b], nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
            continue; /* not in this build */
        }
        for (size_t h = 0; h < sizeof KV_HEADS / sizeof KV_HEADS[0]; h++) {
            struct tf_buf       g = mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                                                      .d_model  = 512,
                                                                      .heads    = 8,
                                                                      .kv_heads = KV_HEADS[h],
                                                                      .ffn      = 256,
                                                                      .vocab    = VOCAB,
                                                                      .context  = 512,
                                                                      .seed     = 23});
            struct geist_model *m = nullptr;
            if (geist_model_load_from_memory(g.b, g.n, be, &m) != GEIST_OK) {
                fprintf(stderr,
                        "FAIL: %s: model load: %s\n",
                        BACKENDS[b],
                        geist_last_create_error());
                fails++;
            } else {
                ran++;
                static const enum geist_kv_mode MODES[] = {
                        GEIST_KV_INT8, GEIST_KV_INT4, GEIST_KV_FP32};
                for (size_t i = 0; i < sizeof MODES / sizeof MODES[0]; i++) {
                    fails += check_placement(BACKENDS[b], be, m, MODES[i]);
                    fails += check_run(BACKENDS[b], be, m, MODES[i]);
                }
            }
            geist_model_destroy(m);
            free(g.b);
        }
        geist_backend_destroy(be);
    }
    if (ran == 0) {
        printf("SKIP: no CPU backend in this build\n");
        return GEIST_TEST_SKIP;
    }
    if (fails != 0) {
        return GEIST_TEST_FAIL;
    }
    printf("PASS: INT8/INT4 and FP32 KV caches placed as the backend asks, and sessions run on "
           "them\n");
    return GEIST_TEST_PASS;
}
