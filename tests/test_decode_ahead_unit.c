/*
 * test_decode_ahead_unit - a session that commits decode steps without
 * waiting for them (#590, struct transformer_ahead) against one that waits
 * for every step (GEIST_DECODE_AHEAD=0 at model load), on models built in
 * memory, on Metal: the same calls on both, and after each call the same
 * status, token, length and snapshot image (the cache rows, the DeltaNet
 * state, the pending logits and token, the token history, taken without
 * running the owed forward), and at the marked points the same logits.
 *
 * Calls: greedy decoding with nothing in between (the forwards overlap on
 * the device), with a snapshot or a logits read after every step, a stop
 * after each of the first steps (reset and another prompt), another token
 * fed after a step, several tokens prefilled after a step, truncate,
 * snapshot and restore, checkpoint and back,
 * a pinned prefix, cancel, speculative decoding (drafts, a verify pass and
 * its rewind), a context that fills up, a batched step after
 * single ones, two sessions of one model taking turns, a session destroyed
 * with a forward in flight; sampling and repetition penalties (which keep
 * the waited path); every KV representation. The sampler of a session is
 * fixed at its creation, so there is no change of it between steps to test.
 *
 * Models: a Qwen3.5 hybrid (DeltaNet and attention layers) and a llama,
 * which decode ahead; a Gemma 4 (per-layer embeddings, KV-shared layers)
 * and a Qwen3-MoE, which do not and must behave as before.
 */
#define _POSIX_C_SOURCE 200809L
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "../src/archs/transformer/arch_state.h"
#include "../src/engine/model.h"
#include "model_fixtures.h"
#include "test_helpers.h"

#if defined(GEIST_BACKEND_METAL) && defined(__APPLE__)
#include "../src/backends/metal/metal_internal.h"
/* Sequences committed and not waited for, the forward of a session that
 * decodes ahead among them. */
static unsigned in_flight(const struct transformer_arch_session *a) {
    const struct metal_state *st = a->model->backend->state;
    return atomic_load(&st->async_live);
}
#else
static unsigned in_flight(const struct transformer_arch_session *a) {
    return a->ahead.inflight ? 1u : 0u;
}
#endif

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CTX = 96, N_PROMPT = 9, N_STEPS = 12 };

/* [0]: decodes ahead where it can; [1]: the oracle, every step waited for. */
struct pair {
    struct geist_model              *m[2];
    struct geist_session            *s[2];
    struct transformer_arch_session *a[2];
    const char                      *what;
    bool                             ahead; /* s[0] is expected to decode ahead */
    geist_token_t                    vocab;
};

static geist_token_t g_ids[CTX];

static int fail(const struct pair *p, const char *where, const char *msg) {
    fprintf(stderr, "FAIL [%s] %s: %s\n", p->what, where, msg);
    return 1;
}

/* Status, length and snapshot image, none of which runs the owed forward. */
static int same_image(struct pair *p, const char *where) {
    if (geist_session_length(p->s[0]) != geist_session_length(p->s[1])) {
        return fail(p, where, "lengths differ");
    }
    if (p->a[1]->ahead.inflight || p->a[1]->ahead.ids != nullptr) {
        return fail(p, where, "the oracle decoded ahead");
    }
    size_t            size[2] = {0, 0}, wrote[2] = {0, 0};
    void             *img[2] = {nullptr, nullptr};
    enum geist_status st[2];
    for (int i = 0; i < 2; i++) {
        st[i] = geist_session_snapshot_size(&size[i], p->s[i]);
        if (st[i] == GEIST_OK) {
            img[i] = malloc(size[i] + 1);
            st[i]  = img[i] != nullptr ? geist_session_snapshot(&wrote[i], size[i], img[i], p->s[i])
                                       : GEIST_E_OOM;
        }
    }
    int d = st[0] != st[1] || size[0] != size[1] || wrote[0] != wrote[1];
    /* Bytes 8..15 name the model instance (snap_header.model_id), and the
     * two sessions are of two loads of the model. */
    if (!d && st[0] == GEIST_OK && size[0] >= 16) {
        memset((uint8_t *) img[0] + 8, 0, 8);
        memset((uint8_t *) img[1] + 8, 0, 8);
    }
    if (!d && st[0] == GEIST_OK && memcmp(img[0], img[1], size[0]) != 0) {
        size_t i = 0;
        while (((const uint8_t *) img[0])[i] == ((const uint8_t *) img[1])[i]) {
            i++;
        }
        fprintf(stderr, "  byte %zu of %zu\n", i, size[0]);
        d = 1;
    }
    free(img[0]);
    free(img[1]);
    return d ? fail(p, where, "snapshot images differ") : 0;
}

/* The image, then the logits (which run the owed forward), then the image
 * of what that leaves. */
static int same_state(struct pair *p, const char *where) {
    if (same_image(p, where)) {
        return 1;
    }
    size_t       n0 = 0, n1 = 0;
    const float *l0 = geist_session_peek_logits(&n0, p->s[0]);
    const float *l1 = geist_session_peek_logits(&n1, p->s[1]);
    if ((l0 == nullptr) != (l1 == nullptr) || n0 != n1) {
        return fail(p, where, "one session has logits pending, the other not");
    }
    if (l0 != nullptr && memcmp(l0, l1, n0 * sizeof(float)) != 0) {
        return fail(p, where, "logits differ");
    }
    return same_image(p, where);
}

static int reset(struct pair *p) {
    return geist_session_reset(p->s[0]) != GEIST_OK || geist_session_reset(p->s[1]) != GEIST_OK;
}

static int prefill(struct pair *p, size_t n, const geist_token_t *ids, const char *where) {
    enum geist_status st[2];
    for (int i = 0; i < 2; i++) {
        st[i] = geist_session_prefill_tokens(p->s[i], n, ids);
    }
    return st[0] != st[1] ? fail(p, where, "prefill statuses differ") : same_image(p, where);
}

enum check { TOKENS, IMAGE, STATE };

/* n decode steps; *out the tokens (nullptr: not kept); a status other than
 * OK must be the same on both and ends the run. */
static int decode(struct pair *p, size_t n, enum check c, geist_token_t *out, const char *where) {
    for (size_t t = 0; t < n; t++) {
        geist_token_t     tok[2] = {-1, -2};
        enum geist_status st[2];
        st[0] = geist_session_decode_step(p->s[0], &tok[0]);
        /* Not waited for when the step returns: taking the forward before
         * it over must not have waited for this one (the oracle's step
         * below does, like any read of the device). */
        const unsigned live = in_flight(p->a[0]);
        st[1]               = geist_session_decode_step(p->s[1], &tok[1]);
        if (st[0] != st[1]) {
            return fail(p, where, "decode_step statuses differ");
        }
        if (st[0] == GEIST_OK && p->ahead && live == 0) {
            return fail(p, where, "the forward in flight was waited for before the step returned");
        }
        if (st[0] != GEIST_OK) {
            return same_image(p, where);
        }
        if (tok[0] != tok[1]) {
            fprintf(stderr, "  step %zu: %d against the oracle's %d\n", t, tok[0], tok[1]);
            return fail(p, where, "decoded tokens differ");
        }
        if (out != nullptr) {
            out[t] = tok[0];
        }
        /* The session that can has the forward of this token in flight. */
        if (p->a[0]->ahead.inflight != p->ahead) {
            return fail(p, where, p->ahead ? "no forward in flight" : "a forward in flight");
        }
        if ((c == IMAGE && same_image(p, where)) || (c == STATE && same_state(p, where))) {
            return 1;
        }
    }
    return 0;
}

static int start(struct pair *p, size_t n, const char *where) {
    return reset(p) || prefill(p, n, g_ids, where);
}

static int case_plain(struct pair *p) {
    return start(p, N_PROMPT, "plain") || decode(p, N_STEPS, TOKENS, nullptr, "plain") ||
           same_state(p, "plain, end") || start(p, N_PROMPT, "image") ||
           decode(p, N_STEPS, IMAGE, nullptr, "image per step") || start(p, N_PROMPT, "state") ||
           decode(p, N_STEPS, STATE, nullptr, "logits per step");
}

/* Stop after k steps: the token of the last one is not continued. */
static int case_stop(struct pair *p) {
    for (size_t k = 1; k <= 3; k++) {
        if (start(p, N_PROMPT, "stop") || decode(p, k, TOKENS, nullptr, "stop") ||
            start(p, N_PROMPT + 3, "stop, next prompt") ||
            decode(p, 5, TOKENS, nullptr, "stop, next prompt") || same_state(p, "stop, end")) {
            return 1;
        }
    }
    /* A reset right after a step, and a step with nothing pending. */
    return start(p, N_PROMPT, "reset") || decode(p, 2, TOKENS, nullptr, "reset") || reset(p) ||
           same_image(p, "reset after a step") ||
           decode(p, 1, TOKENS, nullptr, "step with nothing pending");
}

/* Another token than the next step would take, and several, after a step. */
static int case_feed(struct pair *p) {
    for (size_t k = 1; k <= 2; k++) {
        if (start(p, N_PROMPT, "feed") || decode(p, k, TOKENS, nullptr, "feed") ||
            prefill(p, 1, g_ids + 40 + k, "one token after a step") ||
            decode(p, 4, TOKENS, nullptr, "one token after a step") ||
            prefill(p, 7, g_ids + 50, "tokens after a step") ||
            decode(p, 4, IMAGE, nullptr, "tokens after a step") || same_state(p, "feed, end")) {
            return 1;
        }
    }
    return 0;
}

static int both(struct pair *p,
                enum geist_status (*op)(struct geist_session *, size_t),
                size_t      n,
                const char *where) {
    const enum geist_status s0 = op(p->s[0], n), s1 = op(p->s[1], n);
    return s0 != s1 ? fail(p, where, "statuses differ") : same_image(p, where);
}

static enum geist_status op_checkpoint(struct geist_session *s, size_t n) {
    (void) n;
    return geist_session_checkpoint(s);
}

static enum geist_status op_pin(struct geist_session *s, size_t n) {
    return geist_session_pin_prefix(s, n, g_ids);
}

/* Truncate after steps (refused alike on DeltaNet layers), and to a
 * checkpoint taken between steps. */
static int case_truncate(struct pair *p) {
    if (start(p, N_PROMPT, "truncate") || decode(p, 3, TOKENS, nullptr, "truncate") ||
        both(p, geist_session_truncate, N_PROMPT + 1, "truncate after steps") ||
        prefill(p, 2, g_ids + 30, "prefill after truncate") ||
        decode(p, 4, TOKENS, nullptr, "after truncate") || same_state(p, "truncate, end")) {
        return 1;
    }
    if (start(p, N_PROMPT, "checkpoint") || decode(p, 2, TOKENS, nullptr, "checkpoint") ||
        both(p, op_checkpoint, 0, "checkpoint after steps")) {
        return 1;
    }
    const size_t at = geist_session_length(p->s[0]);
    return decode(p, 3, TOKENS, nullptr, "past the checkpoint") ||
           both(p, geist_session_truncate, at, "back to the checkpoint") ||
           prefill(p, 1, g_ids + 33, "prefill at the checkpoint") ||
           decode(p, 4, TOKENS, nullptr, "from the checkpoint") || same_state(p, "checkpoint, end");
}

/* A snapshot with a forward in flight, restored later. */
static int case_snapshot(struct pair *p) {
    const char   *where   = "snapshot, decode, restore, decode";
    size_t        size[2] = {0, 0}, wrote = 0;
    void         *img[2] = {nullptr, nullptr};
    geist_token_t first[6], again[6];
    int           fails = start(p, N_PROMPT, where) || decode(p, 3, TOKENS, nullptr, where);
    if (!fails && geist_session_snapshot_size(&size[0], p->s[0]) != GEIST_OK) {
        return same_image(p, where); /* no snapshots of this session (KIVI): alike on both */
    }
    for (int i = 0; i < 2 && !fails; i++) {
        fails = geist_session_snapshot_size(&size[i], p->s[i]) != GEIST_OK ||
                (img[i] = malloc(size[i])) == nullptr ||
                geist_session_snapshot(&wrote, size[i], img[i], p->s[i]) != GEIST_OK;
    }
    fails = fails || size[0] != size[1] || decode(p, 6, TOKENS, first, where);
    for (int i = 0; i < 2 && !fails; i++) {
        fails = geist_session_restore(size[i], img[i], p->s[i]) != GEIST_OK;
    }
    fails = fails || same_image(p, where) || decode(p, 6, TOKENS, again, where) ||
            memcmp(first, again, sizeof first) != 0 || same_state(p, where);
    free(img[0]);
    free(img[1]);
    return fails ? fail(p, where, "the restored session decodes otherwise") : 0;
}

static int case_pin(struct pair *p) {
    return reset(p) || both(p, op_pin, 5, "pin a prefix") ||
           prefill(p, 4, g_ids + 5, "turn on a prefix") ||
           decode(p, 4, TOKENS, nullptr, "turn on a prefix") || reset(p) ||
           same_image(p, "reset to the prefix") || prefill(p, 3, g_ids + 20, "second turn") ||
           decode(p, 4, TOKENS, nullptr, "second turn") || same_state(p, "prefix, end");
}

/* A cancel between steps stops the next step on both, and both go on. */
static int case_cancel(struct pair *p) {
    if (start(p, N_PROMPT, "cancel") || decode(p, 3, TOKENS, nullptr, "cancel")) {
        return 1;
    }
    geist_session_cancel(p->s[0]);
    geist_session_cancel(p->s[1]);
    return decode(p, 1, TOKENS, nullptr, "cancelled step") ||
           decode(p, 4, TOKENS, nullptr, "after the cancel") || same_state(p, "cancel, end");
}

/* Speculative decoding (n-gram drafts, a verify pass, a rewind) after
 * single steps, and single steps after it. */
static int case_spec(struct pair *p) {
    const char   *where = "speculative decoding after steps";
    geist_token_t hist[2][CTX];
    if (start(p, N_PROMPT, where)) {
        return 1;
    }
    memcpy(hist[0], g_ids, N_PROMPT * sizeof g_ids[0]);
    if (decode(p, 3, TOKENS, hist[0] + N_PROMPT, where)) {
        return 1;
    }
    memcpy(hist[1], hist[0], sizeof hist[0]);
    size_t n = N_PROMPT + 3;
    for (int round = 0; round < 4; round++) {
        size_t            got[2] = {0, 0};
        enum geist_status st[2];
        for (int i = 0; i < 2; i++) {
            st[i] = geist_session_decode_speculative(
                    p->s[i], 4, n, hist[i], 5, hist[i] + n, &got[i]);
        }
        if (st[0] != st[1] || got[0] != got[1] ||
            memcmp(hist[0] + n, hist[1] + n, got[0] * sizeof hist[0][0]) != 0) {
            return fail(p, where, "statuses or tokens differ");
        }
        if (st[0] != GEIST_OK) {
            break; /* refused alike */
        }
        n += got[0];
        if (same_image(p, where)) {
            return 1;
        }
    }
    return decode(p, 3, TOKENS, nullptr, "steps after speculative decoding") ||
           same_state(p, "speculative, end");
}

/* Decode into the end of the context: the same status at the same step. */
static int case_full(struct pair *p) {
    return start(p, CTX - 5, "full context") || decode(p, 8, IMAGE, nullptr, "full context") ||
           same_state(p, "full context, end");
}

/* Single steps, then a batched one (run where the model batches): the
 * owed forwards run in the batch. Sessions p and q of the same models. */
static int case_batch(struct pair *p, struct pair *q) {
    const char *where = "batched step after single steps";
    if (start(p, N_PROMPT, where) || start(q, N_PROMPT + 2, where) ||
        decode(p, 2, TOKENS, nullptr, where) || decode(q, 3, TOKENS, nullptr, where)) {
        return 1;
    }
    for (int round = 0; round < 3; round++) {
        geist_token_t     tok[2][2];
        enum geist_status st[2];
        for (int i = 0; i < 2; i++) {
            struct geist_session *const two[2] = {p->s[i], q->s[i]};
            st[i]                              = geist_session_decode_batch(2, two, tok[i]);
        }
        if (st[0] != st[1] || memcmp(tok[0], tok[1], sizeof tok[0]) != 0) {
            return fail(p, where, "batched statuses or tokens differ");
        }
        if (same_image(p, where) || same_image(q, where)) {
            return 1;
        }
    }
    return decode(p, 3, TOKENS, nullptr, "single steps after a batch") ||
           same_state(p, "batch, end") || same_state(q, "batch, end");
}

/* Two sessions of one model taking turns: both have a forward in flight. */
static int case_turns(struct pair *p, struct pair *q) {
    const char *where = "two sessions taking turns";
    if (start(p, N_PROMPT, where) || start(q, N_PROMPT + 4, where)) {
        return 1;
    }
    for (int t = 0; t < 8; t++) {
        if (decode(p, 1, TOKENS, nullptr, where) || decode(q, 2, TOKENS, nullptr, where)) {
            return 1;
        }
    }
    return same_state(p, where) || same_state(q, where);
}

static bool open_pair(struct pair                     *p,
                      const struct pair               *models,
                      struct geist_backend            *be,
                      const struct geist_session_opts *o,
                      bool                             ahead,
                      const char                      *what) {
    *p       = *models;
    p->what  = what;
    p->ahead = ahead;
    for (int i = 0; i < 2; i++) {
        p->s[i] = nullptr;
        if (geist_session_create(p->m[i], be, o, &p->s[i]) != GEIST_OK) {
            geist_session_destroy(p->s[0]);
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

/* Every case on one model. `ahead`: its greedy dense-cache sessions decode
 * ahead. */
static int
run_model(struct geist_backend *be, const struct tf_buf *g, const char *name, bool ahead) {
    struct pair                     models = {.what = name};
    const struct geist_session_opts load   = {.max_seq_len = CTX, .top_p = 1.0f};
    /* The flag is read when the model loads: the oracle first, with it off. */
    setenv("GEIST_DECODE_AHEAD", "0", 1);
    enum geist_status s =
            geist_model_load_from_memory_with_opts(g->b, g->n, be, &load, &models.m[1]);
    unsetenv("GEIST_DECODE_AHEAD");
    if (s == GEIST_OK) {
        s = geist_model_load_from_memory_with_opts(g->b, g->n, be, &load, &models.m[0]);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "%s: model_load: %s\n", name, geist_last_create_error());
        geist_model_destroy(models.m[1]);
        return 1;
    }
    int  fails = 0;
    char what[96];
    const struct {
        const char               *name;
        struct geist_session_opts o;
        bool                      ahead;
    } variants[] = {
            {"default", {.max_seq_len = CTX, .top_p = 1.0f}, ahead},
            {"FP32 KV", {.max_seq_len = CTX, .top_p = 1.0f, .kv_mode = GEIST_KV_FP32}, ahead},
            {"F16 KV", {.max_seq_len = CTX, .top_p = 1.0f, .kv_mode = GEIST_KV_F16}, ahead},
            {"INT8 KV", {.max_seq_len = CTX, .top_p = 1.0f, .kv_mode = GEIST_KV_INT8}, false},
            {"INT4 KV", {.max_seq_len = CTX, .top_p = 1.0f, .kv_mode = GEIST_KV_INT4}, false},
            {"KIVI KV", {.max_seq_len = CTX, .top_p = 1.0f, .kv_mode = GEIST_KV_KIVI}, false},
            {"sampling",
             {.max_seq_len = CTX, .top_p = 0.9f, .temperature = 0.8f, .random_seed = 7},
             false},
            {"top-k", {.max_seq_len = CTX, .top_p = 1.0f, .temperature = 1.0f, .top_k = 5}, false},
            {"penalties", {.max_seq_len = CTX, .top_p = 1.0f, .repeat_penalty = 1.3f}, false},
    };
    for (size_t vi = 0; vi < sizeof variants / sizeof variants[0]; vi++) {
        struct pair p, q;
        /* A model that never decodes ahead: its default session and a
         * sampling one show that it is left alone. */
        if (!ahead && vi != 0 && vi != 6) {
            continue;
        }
        snprintf(what, sizeof what, "%s, %s", name, variants[vi].name);
        if (!open_pair(&p, &models, be, &variants[vi].o, variants[vi].ahead, what)) {
            printf("  %s: no such session on this backend\n", what);
            continue;
        }
        int f = case_plain(&p);
        if (vi == 0) {
            f = f || case_stop(&p) || case_feed(&p) || case_truncate(&p) || case_snapshot(&p) ||
                case_pin(&p) || case_cancel(&p) || case_spec(&p) || case_full(&p);
            if (!f && open_pair(&q, &models, be, &variants[vi].o, variants[vi].ahead, what)) {
                f = case_turns(&p, &q) || case_batch(&p, &q);
                /* q goes with a forward in flight, p after a reset. */
                f = f || start(&q, N_PROMPT, "destroy") ||
                    decode(&q, 2, TOKENS, nullptr, "destroy");
                close_pair(&q);
            }
        } else {
            f = f || case_snapshot(&p);
        }
        printf("  %s: %s%s\n", what, f ? "FAIL" : "ok", variants[vi].ahead ? ", ahead" : "");
        fails += f;
        close_pair(&p);
    }
    geist_model_destroy(models.m[0]);
    geist_model_destroy(models.m[1]);
    return fails;
}

int main(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("metal", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        GEIST_SKIP("no Metal backend (the only one that decodes ahead)");
    }
    for (size_t i = 0; i < CTX; i++) {
        g_ids[i] = (geist_token_t) (1 + (i * 37 + 11) % 200);
    }
    struct tf_vocab v      = tf_make_vocab("\xc4\xa0", false);
    struct tf_buf   hybrid = mf_qwen35_gguf(&(struct mf_qwen35) {.layers     = 4,
                                                                 .interval   = 4,
                                                                 .d_model    = 64,
                                                                 .heads      = 4,
                                                                 .kv_heads   = 2,
                                                                 .head_dim   = 16,
                                                                 .rope_dims  = 8,
                                                                 .ffn        = 128,
                                                                 .dn_k_heads = 2,
                                                                 .dn_v_heads = 4,
                                                                 .dn_head_k  = 16,
                                                                 .dn_head_v  = 16,
                                                                 .dn_conv    = 4,
                                                                 .seed       = 1,
                                                                 .tok        = &v});
    struct tf_buf   llama  = mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                                               .d_model  = 128,
                                                               .heads    = 4,
                                                               .kv_heads = 2,
                                                               .ffn      = 256,
                                                               .vocab    = 512,
                                                               .context  = 256,
                                                               .seed     = 14});
    struct tf_buf   untied = mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                                               .d_model  = 128,
                                                               .heads    = 4,
                                                               .kv_heads = 2,
                                                               .ffn      = 256,
                                                               .vocab    = 512,
                                                               .context  = 256,
                                                               .seed     = 15,
                                                               .untied   = true});
    struct tf_buf   moe    = mf_qwen3moe_gguf(&(struct mf_qwen3moe) {.layers    = 2,
                                                                     .d_model   = 64,
                                                                     .heads     = 4,
                                                                     .kv_heads  = 2,
                                                                     .head_dim  = 16,
                                                                     .experts   = 8,
                                                                     .used      = 2,
                                                                     .expert_ff = 32,
                                                                     .vocab     = 256,
                                                                     .seed      = 5});
    struct tf_buf   gemma  = mf_gemma4_gguf(&(struct mf_gemma4) {.layers  = 7,
                                                                 .shared  = 3,
                                                                 .period  = 3,
                                                                 .d_model = 32,
                                                                 .heads   = 2,
                                                                 .hd_full = 16,
                                                                 .hd_swa  = 8,
                                                                 .ffn     = 64,
                                                                 .hpl     = 8,
                                                                 .window  = 24,
                                                                 .context = CTX,
                                                                 .seed    = 11});
    int             fails  = 0;
    fails += run_model(be, &hybrid, "qwen35 hybrid", true);
    fails += run_model(be, &llama, "llama", true);
    fails += run_model(be, &untied, "llama, untied head", false);
    fails += run_model(be, &moe, "qwen3moe", false);
    fails += run_model(be, &gemma, "gemma4", false);
    free(hybrid.b);
    free(llama.b);
    free(untied.b);
    free(moe.b);
    free(gemma.b);
    geist_backend_destroy(be);
    if (fails != 0) {
        fprintf(stderr, "%d model variant(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    puts("PASS: a session that decodes ahead is the one that waits, call for call");
    return GEIST_TEST_PASS;
}
