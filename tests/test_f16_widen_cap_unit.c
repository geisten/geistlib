/*
 * test_f16_widen_cap_unit — a half-precision projection that no kernel can
 * run is refused at load, not at the first prefill (#564).
 *
 * Metal has no F16/BF16 linear: its resolver refuses them, and the loader
 * widens a small one (<= 4M elements) to F32 instead. A larger one used to
 * load anyway and then fail the first prefill with "resolver installed no
 * kernel". The test wraps cpu_scalar in a resolver that refuses F16/BF16
 * the same way, and loads an in-memory llama whose attn_q is F16:
 *   - below the cap the projection is widened, and a prefill runs;
 *   - above the cap the load fails with GEIST_E_UNSUPPORTED, and the message
 *     names the tensor.
 */
#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>
#include <geist_weight.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

static enum geist_status (*base_resolve)(struct geist_backend *, struct geist_weight *);

/* Metal's answer for half-precision weights: no kernel. */
[[nodiscard]] static enum geist_status no_half_resolve(struct geist_backend *be,
                                                       struct geist_weight  *w) {
    if (w->dtype == GEIST_DTYPE_F16 || w->dtype == GEIST_DTYPE_BF16) {
        return GEIST_E_UNSUPPORTED;
    }
    return base_resolve(be, w);
}

/* A llama of `heads` 64-wide heads with an F16 attn_q of (64 heads)^2. */
static struct tf_buf f16_llama(uint32_t heads) {
    return mf_llama_gguf(&(struct mf_llama) {.layers   = 1,
                                             .d_model  = heads * 64,
                                             .heads    = heads,
                                             .kv_heads = 1,
                                             .ffn      = 128,
                                             .vocab    = 32,
                                             .context  = 64,
                                             .seed     = 7,
                                             .f16_q    = true});
}

static int check_small(struct geist_backend *be) {
    /* 512 x 512 = 256K elements: widened. */
    struct tf_buf       g  = f16_llama(8);
    struct geist_model *m  = nullptr;
    enum geist_status   ls = geist_model_load_from_memory(g.b, g.n, be, &m);
    int fails = geist_expect(ls == GEIST_OK, "F16 attn_q below the cap loads (widened to F32)");
    if (ls == GEIST_OK) {
        const struct geist_session_opts o    = {.top_p = 1.0f};
        struct geist_session           *s    = nullptr;
        const geist_token_t             t[4] = {1, 5, 9, 3};
        bool                            ok   = geist_session_create(m, be, &o, &s) == GEIST_OK &&
                                               geist_session_prefill_tokens(s, 4, t) == GEIST_OK;
        size_t                          n    = 0;
        const float                    *l    = ok ? geist_session_peek_logits(&n, s) : nullptr;
        ok                                   = ok && l != nullptr && n == 32;
        for (size_t i = 0; ok && i < n; i++) {
            ok = isfinite(l[i]);
        }
        fails += geist_expect(ok, "widened F16 attn_q: prefill gives finite logits");
        geist_session_destroy(s);
        geist_model_destroy(m);
    }
    free(g.b);
    return fails;
}

static int check_large(struct geist_backend *be) {
    /* 2112 x 2112 = 4.46M elements: past the widen cap. */
    struct tf_buf       g  = f16_llama(33);
    struct geist_model *m  = nullptr;
    enum geist_status   ls = geist_model_load_from_memory(g.b, g.n, be, &m);
    int fails = geist_expect(ls == GEIST_E_UNSUPPORTED,
                             "F16 attn_q above the cap is refused at load (GEIST_E_UNSUPPORTED)");
    const char *why = geist_last_create_error();
    fails += geist_expect(why != nullptr && strstr(why, "blk.0.attn_q.weight") != nullptr &&
                                  strstr(why, "F16") != nullptr,
                          "the load error names the tensor and its dtype");
    if (ls == GEIST_OK) {
        geist_model_destroy(m);
    } else if (why != nullptr) {
        printf("  load error: %s\n", why);
    }
    free(g.b);
    return fails;
}

int main(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        printf("SKIP: cpu_scalar backend not in this build\n");
        return GEIST_TEST_SKIP;
    }
    const struct geist_backend_descriptor *orig = be->desc;
    struct geist_backend_vtbl              vtbl = *orig->vtbl;
    struct geist_backend_descriptor        desc = *orig;
    base_resolve                                = vtbl.resolve_weight;
    vtbl.resolve_weight                         = no_half_resolve;
    desc.vtbl                                   = &vtbl;
    be->desc                                    = &desc;

    int fails = check_small(be) + check_large(be);

    be->desc = orig;
    geist_backend_destroy(be);
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
