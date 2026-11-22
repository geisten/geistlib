/*
 * kv_shared_prefill.h - the cases of test_kv_shared_prefill_int (a Gemma 4
 * GGUF) and test_kv_shared_prefill_unit (a Gemma 4 built in memory): a
 * prefill that leaves the KV-shared layers out for its cache rows against
 * one that runs them for every row (#591). The two tests' headers say what
 * is compared; the includer defines _POSIX_C_SOURCE, GEIST_INTERNAL_ARCH_LAYER
 * and GEIST_INTERNAL_ENGINE_LAYER first.
 */
#ifndef GEIST_TESTS_KV_SHARED_PREFILL_H
#define GEIST_TESTS_KV_SHARED_PREFILL_H

#include "../src/archs/transformer/arch_state.h"
#include "../src/archs/transformer/forward.h"
#include "../src/engine/model.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N_MAX = 1200, CTX = 2048, N_DECODE = 6 };

static geist_token_t g_ids[N_MAX];

/* A fixed prompt: BOS, then ids spread over the low vocabulary. */
static void kvs_ids(void) {
    g_ids[0] = 2;
    for (size_t i = 1; i < N_MAX; i++) {
        g_ids[i] = (geist_token_t) (3 + (i * 2654435761u) % 50000u);
    }
}

/* [0]: leaves cache rows out; [1]: the oracle, every layer for every row. */
struct pair {
    struct geist_model              *m[2];
    struct geist_session            *s[2];
    struct transformer_arch_session *a[2];
    size_t                           counted; /* cache rows the cases expect of s[0] so far */
    const char                      *what;    /* the session variant, for messages */
};

static int fail(const struct pair *p, const char *where, const char *msg) {
    fprintf(stderr, "FAIL [%s] %s: %s\n", p->what, where, msg);
    return 1;
}

/* Bytes one position takes in layer li's K and V buffers and in each of
 * its scale buffers, as snapshot.c lays them out. */
static void
row_bytes(const struct transformer_arch_session *a, size_t li, size_t *k, size_t *v, size_t *sc) {
    const size_t elems = a->model->n_kv_heads * a->model->layers[li].head_dim;
    if (a->kv_int8_enabled) {
        *k  = elems;
        *v  = a->kv_int4_packed_enabled ? elems / 2 : elems;
        *sc = a->model->n_kv_heads * sizeof(float);
    } else {
        *k  = elems * (a->kv_f16_enabled ? sizeof(uint16_t) : sizeof(float));
        *v  = *k;
        *sc = 0;
    }
}

static int buf_differs(const struct geist_backend_vtbl *v,
                       struct geist_buffer             *x,
                       struct geist_buffer             *y,
                       size_t                           bytes) {
    if (x == nullptr || y == nullptr) {
        return x != y;
    }
    const void *px = v->buffer_map(x), *py = v->buffer_map(y);
    const int   d = px == nullptr || py == nullptr || memcmp(px, py, bytes) != 0;
    if (px != nullptr) {
        v->buffer_unmap(x);
    }
    if (py != nullptr) {
        v->buffer_unmap(y);
    }
    return d;
}

/* Everything a later call can observe of the two sessions. */
static int same_state(struct pair *p, const char *where) {
    const struct transformer_arch_session *a = p->a[0], *b = p->a[1];
    const struct transformer_arch_state   *st = a->model;
    const struct geist_backend_vtbl       *v  = st->backend->desc->vtbl;
    if (geist_session_length(p->s[0]) != geist_session_length(p->s[1])) {
        return fail(p, where, "lengths differ");
    }
    size_t       n0 = 0, n1 = 0;
    const float *l0 = geist_session_peek_logits(&n0, p->s[0]);
    const float *l1 = geist_session_peek_logits(&n1, p->s[1]);
    if ((l0 == nullptr) != (l1 == nullptr) || n0 != n1) {
        return fail(p, where, "one session has logits pending, the other not");
    }
    if (l0 != nullptr && memcmp(l0, l1, n0 * sizeof(float)) != 0) {
        size_t i = 0;
        while (l0[i] == l1[i]) {
            i++;
        }
        fprintf(stderr,
                "  logit %zu: %.9g against the oracle's %.9g\n",
                i,
                (double) l0[i],
                (double) l1[i]);
        return fail(p, where, "logits differ");
    }
    if (a->kv_len != b->kv_len || a->logits_valid != b->logits_valid ||
        (a->logits_valid && a->next_token_pending != b->next_token_pending)) {
        return fail(p, where, "kv_len or the pending token differ");
    }
    if (a->kv_kivi_enabled && (a->kivi_drained_count != b->kivi_drained_count ||
                               a->kivi_residual_count != b->kivi_residual_count)) {
        return fail(p, where, "KIVI drain counters differ");
    }
    if (a->tok_hist != nullptr &&
        memcmp(a->tok_hist, b->tok_hist, a->kv_len * sizeof(geist_token_t)) != 0) {
        return fail(p, where, "token histories differ");
    }
    for (size_t li = 0; li < st->n_layers; li++) {
        size_t kb, vb, sc;
        row_bytes(a, li, &kb, &vb, &sc);
        const size_t n = a->kv_len;
        int          d = 0;
        if (a->kv_kivi_enabled) {
            /* The undrained rows of the FP32 ring; the 2-bit groups follow
             * from the same rows through the same drains. */
            const size_t ring = a->kivi_residual_count * a->model->n_kv_heads *
                                st->layers[li].head_dim * sizeof(float);
            d                 = buf_differs(v, a->k_residual[li], b->k_residual[li], ring) ||
                                buf_differs(v, a->v_residual[li], b->v_residual[li], ring);
        } else if (a->kv_int8_enabled) {
            d = buf_differs(v, a->k_cache_q8[li], b->k_cache_q8[li], n * kb) ||
                buf_differs(v, a->v_cache_q8[li], b->v_cache_q8[li], n * vb) ||
                buf_differs(v, a->k_cache_scale[li], b->k_cache_scale[li], n * sc) ||
                buf_differs(v, a->v_cache_scale[li], b->v_cache_scale[li], n * sc);
        } else {
            d = buf_differs(v, a->k_cache[li], b->k_cache[li], n * kb) ||
                buf_differs(v, a->v_cache[li], b->v_cache[li], n * vb);
        }
        if (d) {
            fprintf(stderr, "  layer %zu\n", li);
            return fail(p, where, "a layer's K/V rows differ");
        }
    }
    return 0;
}

/* The cache rows the plan gives a prefill of n rows at the session's
 * current length, walked as the text prefill walks it. */
static size_t plan_cache_rows(const struct transformer_arch_session *a, size_t n) {
    size_t cache = 0, pos = a->kv_len;
    for (size_t off = 0, rows = 0; off < n; off += rows, pos += rows) {
        bool c = false;
        rows   = transformer_prefill_chunk(a, pos, n - off);
        rows   = transformer_prefill_pass_rows(a, rows, off + rows == n, &c);
        cache += c ? rows : 0;
    }
    return cache;
}

static int count_ok(struct pair *p, const char *where) {
    if (p->a[0]->n_cache_rows != p->counted) {
        fprintf(stderr, "  %zu cache rows, the plan has %zu\n", p->a[0]->n_cache_rows, p->counted);
        return fail(p, where, "the cache rows are not the plan's");
    }
    return p->a[1]->n_cache_rows != 0 ? fail(p, where, "the oracle left rows out") : 0;
}

static int reset(struct pair *p) {
    return geist_session_reset(p->s[0]) != GEIST_OK || geist_session_reset(p->s[1]) != GEIST_OK;
}

static int prefill(struct pair *p, size_t n, const geist_token_t *ids, const char *where) {
    p->counted += plan_cache_rows(p->a[0], n);
    for (int i = 0; i < 2; i++) {
        if (geist_session_prefill_tokens(p->s[i], n, ids) != GEIST_OK) {
            fprintf(stderr, "  %s\n", geist_session_errmsg(p->s[i]));
            return fail(p, where, "prefill failed");
        }
    }
    return same_state(p, where) || count_ok(p, where);
}

static int decode(struct pair *p, size_t n, geist_token_t *out, const char *where) {
    for (size_t t = 0; t < n; t++) {
        geist_token_t tok[2] = {-1, -2};
        for (int i = 0; i < 2; i++) {
            if (geist_session_decode_step(p->s[i], &tok[i]) != GEIST_OK) {
                return fail(p, where, "decode_step failed");
            }
        }
        if (tok[0] != tok[1]) {
            return fail(p, where, "decoded tokens differ");
        }
        if (out != nullptr) {
            out[t] = tok[0];
        }
        if (same_state(p, where)) {
            return 1;
        }
    }
    return count_ok(p, where);
}

/* One prefill call of n tokens, then decoding. */
static int case_prefill_decode(struct pair *p, size_t n) {
    char where[64];
    snprintf(where, sizeof where, "prefill %zu + decode", n);
    return reset(p) || prefill(p, n, g_ids, where) || decode(p, N_DECODE, nullptr, where);
}

/* The same pass twice on a session: n tokens of one chunk, a reset, the
 * same n tokens. The second pass has the first one's rows at the first
 * one's position, and layer 0 once took that for its own input already
 * normalised by a layer before it (the tail-norm fusion's prenorm_layer 0
 * means none): the logits of the repeat were not the first pass's. */
static int case_repeat(struct pair *p, size_t n) {
    char where[64];
    snprintf(where, sizeof where, "prefill %zu, reset, prefill %zu", n, n);
    size_t nv = 0;
    if (reset(p) || prefill(p, n, g_ids, where)) {
        return 1;
    }
    const float *l     = geist_session_peek_logits(&nv, p->s[0]);
    float       *first = l != nullptr ? malloc(nv * sizeof *first) : nullptr;
    if (first == nullptr) {
        return fail(p, where, "no logits");
    }
    memcpy(first, l, nv * sizeof *first);
    int fails = reset(p) || prefill(p, n, g_ids, where);
    l         = fails ? nullptr : geist_session_peek_logits(&nv, p->s[0]);
    fails     = l == nullptr || memcmp(first, l, nv * sizeof *first) != 0;
    free(first);
    return fails ? fail(p, where, "the repeat has other logits") : 0;
}

/* A step taken again: decode a token at a position, truncate back to it,
 * feed another token there. Against the same tokens on an emptied session. */
static int case_step_again(struct pair *p, size_t n) {
    const char   *where = "decode, truncate, another token at that position";
    geist_token_t tok;
    size_t        nv = 0;
    if (reset(p) || prefill(p, n, g_ids, where)) {
        return 1;
    }
    for (int i = 0; i < 2; i++) {
        /* The second step runs the forward of the first one's token. */
        if (geist_session_decode_step(p->s[i], &tok) != GEIST_OK ||
            geist_session_decode_step(p->s[i], &tok) != GEIST_OK ||
            geist_session_truncate(p->s[i], n) != GEIST_OK) {
            return fail(p, where, "decode or truncate failed");
        }
    }
    if (prefill(p, 1, g_ids + n + 3, where)) {
        return 1;
    }
    const float *l     = geist_session_peek_logits(&nv, p->s[0]);
    float       *again = l != nullptr ? malloc(nv * sizeof *again) : nullptr;
    if (again == nullptr) {
        return fail(p, where, "no logits");
    }
    memcpy(again, l, nv * sizeof *again);
    int fails = reset(p) || prefill(p, n, g_ids, where) || prefill(p, 1, g_ids + n + 3, where);
    l         = fails ? nullptr : geist_session_peek_logits(&nv, p->s[0]);
    fails     = l == nullptr || memcmp(again, l, nv * sizeof *again) != 0;
    free(again);
    return fails ? fail(p, where, "the token fed after the truncate has other logits") : 0;
}

static int case_two_calls(struct pair *p, size_t n1, size_t n2) {
    const char *where = "two prefill calls";
    return reset(p) || prefill(p, n1, g_ids, where) || prefill(p, n2, g_ids + n1, where) ||
           decode(p, 4, nullptr, where);
}

static int case_truncate(struct pair *p, size_t n, size_t keep, size_t again) {
    const char *where = "prefill, truncate, prefill";
    if (reset(p) || prefill(p, n, g_ids, where)) {
        return 1;
    }
    for (int i = 0; i < 2; i++) {
        if (geist_session_truncate(p->s[i], keep) != GEIST_OK) {
            return fail(p, where, "truncate failed");
        }
    }
    /* Other tokens than the ones dropped: stale rows would show. */
    return same_state(p, where) || prefill(p, again, g_ids + n, where) ||
           decode(p, 4, nullptr, where);
}

static int case_snapshot(struct pair *p, size_t n) {
    const char   *where   = "snapshot, decode, restore, decode";
    size_t        size[2] = {0, 0};
    void         *img[2]  = {nullptr, nullptr};
    geist_token_t first[N_DECODE], second[N_DECODE];
    int           fails = reset(p) || prefill(p, n, g_ids, where);
    for (int i = 0; i < 2 && !fails; i++) {
        size_t wrote = 0;
        fails        = geist_session_snapshot_size(&size[i], p->s[i]) != GEIST_OK ||
                       (img[i] = malloc(size[i])) == nullptr ||
                       geist_session_snapshot(&wrote, size[i], img[i], p->s[i]) != GEIST_OK ||
                       wrote != size[i];
    }
    if (fails) {
        free(img[0]);
        free(img[1]);
        return fail(p, where, "prefill or snapshot failed");
    }
    fails = size[0] != size[1] || decode(p, N_DECODE, first, where);
    for (int i = 0; i < 2 && !fails; i++) {
        fails = geist_session_restore(size[i], img[i], p->s[i]) != GEIST_OK;
    }
    fails = fails || same_state(p, where) || decode(p, N_DECODE, second, where) ||
            memcmp(first, second, sizeof first) != 0;
    free(img[0]);
    free(img[1]);
    return fails ? fail(p, where, "the restored session decodes otherwise") : 0;
}

static int case_pin_prefix(struct pair *p, size_t n_prefix, size_t n_turn) {
    const char   *where = "pinned prefix, turn, reset, turn";
    geist_token_t first[4], second[4];
    if (reset(p)) {
        return 1;
    }
    p->counted += plan_cache_rows(p->a[0], n_prefix); /* at length 0, as pin_prefix prefills */
    for (int i = 0; i < 2; i++) {
        if (geist_session_pin_prefix(p->s[i], n_prefix, g_ids) != GEIST_OK) {
            return fail(p, where, "pin_prefix failed");
        }
    }
    int fails = same_state(p, where) || count_ok(p, where) ||
                prefill(p, n_turn, g_ids + n_prefix, where) || decode(p, 4, first, where) ||
                reset(p) || same_state(p, where) || prefill(p, n_turn, g_ids + n_prefix, where) ||
                decode(p, 4, second, where) || memcmp(first, second, sizeof first) != 0;
    for (int i = 0; i < 2; i++) {
        fails = geist_session_pin_prefix(p->s[i], 0, nullptr) != GEIST_OK || fails;
    }
    return fails ? fail(p, where, "the turn after the reset differs") : 0;
}

/* verify_forward returns a token for each of its k rows: every row's output
 * is read, so none may be a cache row. */
static int case_verify(struct pair *p, size_t n, size_t k) {
    const char   *where = "verify pass";
    geist_token_t out[2][16];
    if (k > 16 || reset(p) || prefill(p, n, g_ids, where)) {
        return 1;
    }
    for (int i = 0; i < 2; i++) {
        if (transformer_verify_forward(p->a[i], k, g_ids + n, out[i]) != GEIST_OK) {
            return fail(p, where, "verify_forward failed");
        }
    }
    if (memcmp(out[0], out[1], k * sizeof(geist_token_t)) != 0) {
        return fail(p, where, "the rows' tokens differ");
    }
    return same_state(p, where) || count_ok(p, where);
}

/* The audio and image path: rows of d_model floats, chunks of m_max. */
static int case_soft_tokens(struct pair *p, size_t n) {
    const char  *where = "soft-token prefill";
    const size_t d     = p->a[0]->model->d_model;
    float       *soft  = malloc(n * d * sizeof *soft);
    if (soft == nullptr || reset(p) || prefill(p, 5, g_ids, where)) {
        free(soft);
        return 1;
    }
    uint32_t x = 12345u;
    for (size_t i = 0; i < n * d; i++) {
        x       = x * 1664525u + 1013904223u;
        soft[i] = ((float) (x >> 8) / (float) (1u << 24) - 0.5f) * 2.0f;
    }
    int          fails = 0;
    const size_t m     = p->a[0]->m_max;
    for (size_t off = 0, rows = 0; off < n; off += rows) {
        bool c = false;
        rows   = n - off > m ? m : n - off;
        rows   = transformer_prefill_pass_rows(p->a[0], rows, off + rows == n, &c);
        p->counted += c ? rows : 0;
    }
    for (int i = 0; i < 2 && !fails; i++) {
        fails = transformer_prefill_audio_batch(p->a[i], n, soft) != GEIST_OK;
    }
    free(soft);
    if (fails) {
        return fail(p, where, "the soft-token prefill failed");
    }
    return same_state(p, where) || count_ok(p, where) || prefill(p, 20, g_ids + 5, where) ||
           decode(p, 4, nullptr, where);
}

/* A decision readout of the last row's candidates. */
static int case_decision(struct pair *p, size_t n) {
    const char         *where   = "decision readout";
    const geist_token_t cand[6] = {5, 77, 1024, 4097, 30000, 49999};
    float               out[2][6];
    if (!transformer_decision_supported(geist_model_internal_arch_meta(p->m[0]))) {
        printf("  [%s] no decision readout on this model and backend\n", p->what);
        return 0;
    }
    if (reset(p)) {
        return 1;
    }
    p->counted += plan_cache_rows(p->a[0], n);
    for (int i = 0; i < 2; i++) {
        struct geist_arch_readout *r    = nullptr;
        size_t                     rows = 0, bytes = 0;
        uint64_t                   ns = 0;
        enum geist_status          s =
                transformer_decision_create(geist_session_internal_arch_session(p->s[i]), 6, &r);
        if (s == GEIST_OK) {
            s = transformer_decision_prefill(&rows, &bytes, &ns, r, n, 6, g_ids, cand, out[i]);
        }
        transformer_decision_destroy(r);
        if (s != GEIST_OK) {
            return fail(p, where, "the readout failed");
        }
    }
    if (memcmp(out[0], out[1], sizeof out[0]) != 0) {
        return fail(p, where, "the candidates' logits differ");
    }
    return same_state(p, where) || count_ok(p, where);
}

static bool open_pair(struct pair          *p,
                      struct geist_backend *be,
                      enum geist_kv_mode    kv,
                      size_t                m_max,
                      const char           *what) {
    const struct geist_session_opts o = {
            .max_seq_len = CTX, .top_p = 1.0f, .kv_mode = kv, .m_max = m_max};
    p->what    = what;
    p->counted = 0;
    for (int i = 0; i < 2; i++) {
        p->s[i] = nullptr;
        if (geist_session_create(p->m[i], be, &o, &p->s[i]) != GEIST_OK) {
            if (i == 1) {
                geist_session_destroy(p->s[0]);
            }
            return false;
        }
        p->a[i] = tf_sess(geist_session_internal_arch_session(p->s[i]));
    }
    return true;
}

static void close_pair(struct pair *p) {
    geist_session_destroy(p->s[0]);
    geist_session_destroy(p->s[1]);
}

/* The shortest last chunk a GPU backend splits; on a CPU backend, which
 * splits none, a short last chunk's rows. */
static size_t kvs_split_rows(const struct geist_backend *be) {
    return strncmp(geist_backend_name(be), "cpu_", 4) == 0
                   ? 10
                   : TRANSFORMER_PREFILL_TAIL_ALIGN + TRANSFORMER_PREFILL_TAIL_MIN;
}

/* The model, from a file or from memory. */
struct kvs_source {
    const char *path;
    const void *data;
    size_t      size;
};

static enum geist_status kvs_load(const struct kvs_source         *src,
                                  struct geist_backend            *be,
                                  const struct geist_session_opts *o,
                                  struct geist_model             **out) {
    return src->path != nullptr
                   ? geist_model_load_with_opts(src->path, be, o, out)
                   : geist_model_load_from_memory_with_opts(src->data, src->size, be, o, out);
}

/* Every case on `backend`; the failures. A backend not in the build runs
 * nothing and fails nothing (*ran stays). */
static int kvs_run_backend(const struct kvs_source *src, const char *backend, size_t *ran) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(backend, nullptr, nullptr, &be) != GEIST_OK) {
        printf("backend %s: not in this build\n", backend);
        return 0;
    }
    *ran += 1;
    struct pair                     p    = {0};
    const struct geist_session_opts load = {.max_seq_len = CTX, .top_p = 1.0f};
    /* The flag is read when the model loads: the oracle first, with it. */
    setenv("GEIST_KV_SHARED_FULL", "1", 1);
    enum geist_status s = kvs_load(src, be, &load, &p.m[1]);
    unsetenv("GEIST_KV_SHARED_FULL");
    if (s == GEIST_OK) {
        s = kvs_load(src, be, &load, &p.m[0]);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "model_load on %s: %s\n", backend, geist_last_create_error());
        geist_model_destroy(p.m[1]);
        geist_backend_destroy(be);
        return 1;
    }
    const struct transformer_arch_state *st = tf_state(geist_model_internal_arch_meta(p.m[0]));
    if (st->n_cache_layers == st->n_layers) {
        geist_model_destroy(p.m[0]);
        geist_model_destroy(p.m[1]);
        geist_backend_destroy(be);
        GEIST_SKIP("the model has no KV-shared layers at its end");
    }
    int fails = 0;
    /* A layer after n_cache_layers must own nothing a row could leave in. */
    if (!open_pair(&p, be, GEIST_KV_AUTO, 0, "default")) {
        fprintf(stderr, "session_create on %s failed\n", backend);
        fails = 1;
    } else {
        const struct transformer_arch_session *a = p.a[0];
        for (size_t li = st->n_cache_layers; li < st->n_layers; li++) {
            fails += geist_expect(a->k_cache[li] == nullptr && a->v_cache[li] == nullptr &&
                                          a->k_cache_q8[li] == nullptr &&
                                          a->k_residual[li] == nullptr &&
                                          (a->dn_S == nullptr || a->dn_S[li] == nullptr),
                                  "a KV-shared layer owns no cache and no state");
        }
        /* The shortest last chunk a GPU backend splits, and the rows it
         * then runs through every layer; a CPU backend splits none
         * (prefill_tail_rows, arch_ops.c), and sp only sizes its prompts. */
        const bool   cpu = strncmp(geist_backend_name(be), "cpu_", 4) == 0;
        const size_t m   = a->m_max;
        const size_t sp  = kvs_split_rows(be);
        const size_t t   = TRANSFORMER_PREFILL_TAIL_MIN;
        printf("backend %s: %zu of %zu layers write a cache, chunk %zu, a last chunk %s\n",
               geist_backend_name(be),
               st->n_cache_layers,
               st->n_layers,
               m,
               cpu       ? "is not split"
               : sp <= m ? "splits from 192 rows"
                         : "is too short to split");
        /* Around every boundary of the plan: a chunk too short to split,
         * the shortest that splits, a whole chunk, one row more (the 8-row
         * last chunk of #554), a chunk and a last one that splits or not. */
        const size_t lens[] = {1,
                               2,
                               9,
                               sp - 1,
                               sp,
                               sp + 1,
                               sp + 7,
                               m - 1,
                               m,
                               m + 1,
                               m + 2,
                               m + 9,
                               m + sp - 1,
                               m + sp,
                               m + sp + 9,
                               2 * m,
                               2 * m + 1,
                               2 * m + sp,
                               3 * m + 37};
        for (size_t i = 0; i < sizeof lens / sizeof lens[0] && fails == 0; i++) {
            if (lens[i] >= 1 && lens[i] <= N_MAX) {
                fails += case_prefill_decode(&p, lens[i]);
            }
        }
        /* The plan left rows out at all, and how many for one known case:
         * of a chunk and sp rows the first chunk on a CPU backend, and all
         * but the second chunk's tail where that splits. */
        fails += geist_expect(p.counted > 0, "cache rows were planned");
        if (fails == 0 && reset(&p) == 0 && sp <= m) {
            const size_t before = p.a[0]->n_cache_rows;
            fails += prefill(&p, m + sp, g_ids, "count");
            fails += geist_expect(p.a[0]->n_cache_rows - before == (cpu ? m : m + sp - t),
                                  "a chunk and a last chunk: the rows left out");
        }
        fails = fails || case_repeat(&p, 1) || case_repeat(&p, 5) || case_repeat(&p, m);
        fails = fails || case_repeat(&p, m + sp) || case_step_again(&p, 9);
        fails = fails || case_two_calls(&p, m + sp + 11, sp + 5);
        fails = fails || case_truncate(&p, m + sp + 20, m + sp + 1, sp + 3);
        fails = fails || case_snapshot(&p, m + sp + 13);
        fails = fails || case_pin_prefix(&p, m + sp + 9, sp + 7);
        fails = fails || case_verify(&p, sp + 14, 8);
        fails = fails || case_soft_tokens(&p, m + sp + 5);
        fails = fails || case_soft_tokens(&p, m + 1);
        fails = fails || case_decision(&p, m + sp + 9);
        close_pair(&p);
    }
    /* A session chunk of 16: many chunks, and the 8-row last chunk of #554. */
    if (fails == 0 && open_pair(&p, be, GEIST_KV_AUTO, 16, "m_max 16")) {
        const size_t lens[] = {17, 26, 32, 33, 100, 16 * 9 + 10};
        for (size_t i = 0; i < sizeof lens / sizeof lens[0] && fails == 0; i++) {
            fails += case_prefill_decode(&p, lens[i]);
        }
        fails = fails || case_snapshot(&p, 75);
        close_pair(&p);
    }
    /* Each KV representation; a backend that refuses one is told so. */
    const struct {
        enum geist_kv_mode kv;
        const char        *name;
    } modes[] = {{GEIST_KV_FP32, "kv fp32"},
                 {GEIST_KV_F16, "kv f16"},
                 {GEIST_KV_INT8, "kv int8"},
                 {GEIST_KV_INT4, "kv int4"},
                 {GEIST_KV_KIVI, "kv kivi"}};
    for (size_t i = 0; i < sizeof modes / sizeof modes[0] && fails == 0; i++) {
        if (!open_pair(&p, be, modes[i].kv, 0, modes[i].name)) {
            printf("  [%s] not on this backend\n", modes[i].name);
            continue;
        }
        const size_t m  = p.a[0]->m_max;
        const size_t sp = kvs_split_rows(be);
        /* A last chunk that splits (KIVI: that would), after one chunk and
         * after two, which crosses a KIVI drain. */
        fails = case_prefill_decode(&p, m + sp + 13) || case_prefill_decode(&p, 2 * m + sp + 21) ||
                case_truncate(&p, m + sp + 30, m + sp + 25, 19);
        if (modes[i].kv != GEIST_KV_KIVI) { /* KIVI has no snapshot */
            fails = fails || case_snapshot(&p, m + sp + 7);
        }
        close_pair(&p);
    }
    printf("backend %s: %s\n", geist_backend_name(be), fails ? "FAIL" : "ok");
    geist_model_destroy(p.m[0]);
    geist_model_destroy(p.m[1]);
    geist_backend_destroy(be);
    return fails;
}

#endif /* GEIST_TESTS_KV_SHARED_PREFILL_H */
