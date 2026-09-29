/*
 * test_kv_placement_unit — where a session puts its INT8/INT4 KV cache.
 *
 * On a backend with caps.kv_q8_block (cpu_x86), a layer's K and V data are
 * aliased slices of one buffer: K at its start, V after K's pages and half
 * a row into its page (half a page at most, a multiple of 64 bytes), so
 * that a KV head's K and V rows do not take the same cache sets (see
 * alloc_kv_q8_block in arch_state.c). Checked on the fixture llama with 2
 * and 8 KV heads of 64 (rows of 128 and 512 bytes), INT8 and packed INT4
 * (half the row): the slices lie where the rule puts them, do not overlap
 * and start zeroed, and a session prefills and decodes through them to
 * finite logits. On cpu_scalar, which leaves the bit unset, K and V stay
 * buffers of their own.
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

/* The placement of every layer's cache in a session of `mode` on `m`. */
static int check_placement(const char           *backend,
                           struct geist_backend *be,
                           struct geist_model   *m,
                           enum geist_kv_mode    mode) {
    struct transformer_arch_state   *st   = geist_model_internal_arch_meta(m);
    const struct geist_session_opts  o    = {.kv_mode = mode, .top_p = 1.0f};
    struct transformer_arch_session *sess = transformer_session_alloc(st, &o);
    char                             msg[192];
    const bool                       int4 = mode == GEIST_KV_INT4;
    snprintf(msg, sizeof msg, "%s %s: session", backend, int4 ? "INT4" : "INT8");
    int fails = geist_expect(sess != nullptr, msg);
    if (sess == nullptr) {
        return fails;
    }
    const struct geist_backend_vtbl *vt    = be->desc->vtbl;
    const bool                       block = be->desc->caps.kv_q8_block;
    for (size_t li = 0; li < st->n_layers; li++) {
        const size_t hd   = st->layers[li].head_dim;
        const size_t row  = int4 ? st->n_kv_heads * hd / 2 : st->n_kv_heads * hd;
        const size_t data = st->max_seq_len * row;
        const size_t span = (data + PAGE - 1) / PAGE * PAGE;
        const size_t skip = (row < PAGE ? row : PAGE) / 2 / 64 * 64;
        uint8_t     *k    = sess->k_cache_q8[li] ? vt->buffer_map(sess->k_cache_q8[li]) : nullptr;
        uint8_t     *v    = sess->v_cache_q8[li] ? vt->buffer_map(sess->v_cache_q8[li]) : nullptr;
        snprintf(msg,
                 sizeof msg,
                 "%s %s layer %zu (%zu-byte rows): K and V caches",
                 backend,
                 int4 ? "INT4" : "INT8",
                 li,
                 row);
        fails += geist_expect(k != nullptr && v != nullptr && k != v, msg);
        if (k == nullptr || v == nullptr) {
            continue;
        }
        if (block) {
            const uint8_t *b =
                    sess->kv_q8_block[li] ? vt->buffer_map(sess->kv_q8_block[li]) : nullptr;
            snprintf(msg,
                     sizeof msg,
                     "%s %s layer %zu: K at the block's start, V %zu bytes on (K's %zu "
                     "bytes, then %zu into the page)",
                     backend,
                     int4 ? "INT4" : "INT8",
                     li,
                     span + skip,
                     data,
                     skip);
            fails += geist_expect(b == k && v == k + span + skip, msg);
            snprintf(msg,
                     sizeof msg,
                     "%s %s layer %zu: V starts %zu bytes further into its page than K",
                     backend,
                     int4 ? "INT4" : "INT8",
                     li,
                     skip);
            fails += geist_expect(((uintptr_t) v - (uintptr_t) k) % PAGE == skip, msg);
        } else {
            snprintf(msg,
                     sizeof msg,
                     "%s %s layer %zu: no block, K and V buffers of their own",
                     backend,
                     int4 ? "INT4" : "INT8",
                     li);
            fails += geist_expect(sess->kv_q8_block[li] == nullptr, msg);
        }
        snprintf(msg, sizeof msg, "%s %s layer %zu: zeroed", backend, int4 ? "INT4" : "INT8", li);
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
             mode == GEIST_KV_INT4 ? "INT4" : "INT8");
    return geist_expect(ok, msg);
}

int main(void) {
    static const char *const BACKENDS[] = {"cpu_x86", "cpu_scalar"};
    static const uint32_t    KV_HEADS[] = {2, 8};
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
                fails += check_placement(BACKENDS[b], be, m, GEIST_KV_INT8);
                fails += check_placement(BACKENDS[b], be, m, GEIST_KV_INT4);
                fails += check_run(BACKENDS[b], be, m, GEIST_KV_INT8);
                fails += check_run(BACKENDS[b], be, m, GEIST_KV_INT4);
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
    printf("PASS: INT8/INT4 KV caches placed as the backend asks, and sessions run on them\n");
    return GEIST_TEST_PASS;
}
