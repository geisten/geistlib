/*
 * test_backend_vulkan_fallback_unit — work that leaves the GPU is counted
 * per site, and GEIST_VK_STRICT=1 turns it into an error (#474 items 4, 9).
 *
 * The Vulkan backend declines a fused op (GEIST_E_UNSUPPORTED) when its
 * shader does not apply, and the arch then runs the op on the host; weights
 * without a GPU kernel run on a host row-dequant path. Checked here:
 *
 *   - a declined fused op (argmax, linear_t, linear_t_pair, embedding,
 *     kv_append_f16, attn_qgate_split) counts one fallback at its site and
 *     still returns GEIST_E_UNSUPPORTED; under strict mode it returns
 *     GEIST_E_BACKEND with an error naming the site;
 *   - a failed argmax dispatch is GEIST_E_BACKEND, not a decline (which
 *     would send the arch to a host scan of logits the GPU never wrote);
 *   - a host view of a mapped tensor and a host buffer copy are counted, and
 *     refused under strict mode;
 *   - a weight resolved onto the host path is summed at resolve, its linear
 *     counts at run time, and strict mode refuses it at resolve;
 *   - decoding the in-memory llama and Qwen3.5-hybrid fixtures leaves no
 *     fused op and no weight on the host, and runs with strict mode on.
 *
 * SKIPs when the Vulkan backend is not built or has no device.
 */
#include "test_helpers.h"
#include "model_fixtures.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(GEIST_BACKEND_VULKAN) && GEIST_BACKEND_VULKAN
#include "src/backends/vulkan/vk_internal.h"
#include "src/quant/quant.h"

static VkResult VKAPI_CALL fail_begin(VkCommandBuffer cmd, const VkCommandBufferBeginInfo *info) {
    (void) cmd;
    (void) info;
    return VK_ERROR_DEVICE_LOST;
}

enum { FEAT = 64 };

/* A [rows, FEAT] F32 view of a fresh buffer of `role` (nullptr buffer on
 * failure). */
static struct geist_tensor
tensor_2d(struct geist_backend *be, size_t rows, enum geist_buffer_role role) {
    struct geist_tensor t = {.dtype  = GEIST_DTYPE_F32,
                             .layout = GEIST_LAYOUT_DENSE,
                             .ndim   = 2,
                             .shape  = {(int64_t) rows, FEAT},
                             .stride = {FEAT, 1}};
    if (be->desc->vtbl->buffer_create(be, rows * FEAT * sizeof(float), role, 0, &t.buffer) !=
        GEIST_OK) {
        t.buffer = nullptr;
    }
    return t;
}

static uint64_t total(const struct vk_state *st) {
    uint64_t n = 0;
    for (size_t i = 0; i < VK_FB_COUNT; i++) {
        n += st->fallbacks[i];
    }
    return n;
}

/* `s` came from a declined op at `site`: one fallback counted there, and
 * the status is the one strict mode asks for. */
static int expect_declined(struct vk_state  *st,
                           enum geist_status s,
                           enum vk_fb        site,
                           uint64_t          before,
                           const char       *op) {
    char msg[160];
    snprintf(msg,
             sizeof msg,
             "%s%s: declined -> %s and one %s fallback (got %d, count %llu)",
             op,
             st->strict ? " (strict)" : "",
             st->strict ? "GEIST_E_BACKEND" : "GEIST_E_UNSUPPORTED",
             vk_fallback_name(site),
             (int) s,
             (unsigned long long) (st->fallbacks[site] - before));
    int fails = geist_expect(s == (st->strict ? GEIST_E_BACKEND : GEIST_E_UNSUPPORTED) &&
                                     st->fallbacks[site] == before + 1,
                             msg);
    if (st->strict) {
        const char *err = geist_backend_errmsg(st->backend);
        snprintf(msg, sizeof msg, "%s (strict): error names the site (%s)", op, err);
        fails += geist_expect(err != nullptr && strstr(err, vk_fallback_name(site)) != nullptr &&
                                      strstr(err, "GEIST_VK_STRICT") != nullptr,
                              msg);
    }
    return fails;
}

/* Every fused op the backend can decline, called so that it does. */
static int check_declines(struct geist_backend *be) {
    struct vk_state                  *st    = be->state;
    const struct geist_backend_fused *f     = geist_backend_fused_tbl(be);
    int                               fails = 0;

    /* argmax over an empty tensor: nothing to bind. */
    struct geist_tensor empty = {.dtype  = GEIST_DTYPE_F32,
                                 .layout = GEIST_LAYOUT_DENSE,
                                 .ndim   = 1,
                                 .shape  = {0},
                                 .stride = {1}};
    int32_t             idx   = -1;
    uint64_t            b     = st->fallbacks[VK_FB_ARGMAX];
    fails += expect_declined(st, f->argmax_f32(be, &empty, &idx), VK_FB_ARGMAX, b, "argmax_f32");

    /* linear_t / linear_t_pair with a weight never resolved to VRAM. */
    static float              wraw[FEAT * FEAT];
    const struct geist_weight w  = {.raw        = wraw,
                                    .raw_nbytes = sizeof wraw,
                                    .n_in       = FEAT,
                                    .n_out      = FEAT,
                                    .dtype      = GEIST_DTYPE_F32};
    struct geist_tensor       x  = tensor_2d(be, 1, GEIST_BUFFER_SCRATCH);
    struct geist_tensor       y0 = tensor_2d(be, 1, GEIST_BUFFER_SCRATCH);
    struct geist_tensor       y1 = tensor_2d(be, 1, GEIST_BUFFER_SCRATCH);
    if (x.buffer != nullptr && y0.buffer != nullptr && y1.buffer != nullptr) {
        b = st->fallbacks[VK_FB_LINEAR_T];
        fails += expect_declined(
                st, f->linear_t(be, &x, &w, nullptr, 1, &y0), VK_FB_LINEAR_T, b, "linear_t");
        b = st->fallbacks[VK_FB_LINEAR_T];
        fails += expect_declined(st,
                                 f->linear_t_pair(be, &x, &w, nullptr, &w, nullptr, 1, &y0, &y1),
                                 VK_FB_LINEAR_T,
                                 b,
                                 "linear_t_pair");

        /* Embedding: a dtype the gather shader has no code for. */
        struct geist_tensor table = x;
        table.dtype               = GEIST_DTYPE_Q3_K;
        b                         = st->fallbacks[VK_FB_EMBED];
        fails += expect_declined(st,
                                 f->embedding_lookup_scaled(be, &table, 0, 1.0f, &y0),
                                 VK_FB_EMBED,
                                 b,
                                 "embedding_lookup_scaled");

        /* qgate split: the joint row is not 2 * heads * head_dim wide. */
        b = st->fallbacks[VK_FB_QGATE];
        fails += expect_declined(
                st, f->attn_qgate_split(be, &x, 4, 16, &y0, &y1), VK_FB_QGATE, b, "qgate_split");

        /* kv_append: caches with no VkBuffer behind them (host aliases). */
        struct geist_tensor kv = x;
        kv.ndim                = 3;
        kv.shape[0]            = 1;
        kv.shape[1]            = 4;
        kv.shape[2]            = FEAT / 4;
        kv.stride[0]           = FEAT;
        kv.stride[1]           = FEAT / 4;
        kv.stride[2]           = 1;
        static uint16_t     host_cache[FEAT];
        struct geist_buffer alias = {
                .owner = st, .host_alias = host_cache, .bytes = sizeof host_cache};
        struct geist_tensor cache = y0;
        cache.buffer              = &alias;
        b                         = st->fallbacks[VK_FB_KV_APPEND];
        fails += expect_declined(st,
                                 f->kv_append_f16(be, &kv, &kv, 0, &cache, &cache),
                                 VK_FB_KV_APPEND,
                                 b,
                                 "kv_append_f16");
    } else {
        fails += geist_expect(false, "scratch buffers");
    }
    struct geist_buffer *bufs[] = {x.buffer, y0.buffer, y1.buffer};
    for (size_t i = 0; i < 3; i++) {
        if (bufs[i] != nullptr) {
            be->desc->vtbl->buffer_destroy(be, bufs[i]);
        }
    }
    return fails;
}

/* Host views and host buffer copies (vk_tensor_host, vk_buffer_copy). */
static int check_host_access(struct geist_backend *be) {
    struct vk_state    *st    = be->state;
    int                 fails = 0;
    struct geist_tensor a     = tensor_2d(be, 1, GEIST_BUFFER_SCRATCH);
    struct geist_tensor c     = tensor_2d(be, 1, GEIST_BUFFER_SCRATCH);
    if (a.buffer == nullptr || c.buffer == nullptr || !a.buffer->host_visible) {
        fails += geist_expect(a.buffer != nullptr && c.buffer != nullptr, "host-access buffers");
    } else {
        uint64_t b = st->fallbacks[VK_FB_HOST_VIEW];
        size_t   n = 0;
        void    *p = vk_tensor_host(&a, &n);
        fails += geist_expect(p != nullptr && n == FEAT && st->fallbacks[VK_FB_HOST_VIEW] == b + 1,
                              "a host view is counted");
        /* Two bookkeeping aliases (no VkBuffer): the copy goes through host
         * memory. */
        uint8_t             src[32], dst[32] = {0};
        struct geist_buffer hs = {.owner = st, .host_alias = src, .bytes = sizeof src};
        struct geist_buffer hd = {.owner = st, .host_alias = dst, .bytes = sizeof dst};
        for (size_t i = 0; i < sizeof src; i++) {
            src[i] = (uint8_t) i;
        }
        b = st->fallbacks[VK_FB_HOST_COPY];
        fails += geist_expect(vk_buffer_copy(&hd, 0, &hs, 0, 16) == GEIST_OK &&
                                      memcmp(dst, src, 16) == 0 &&
                                      st->fallbacks[VK_FB_HOST_COPY] == b + 1,
                              "a host buffer copy is counted");
        st->strict = true;
        memset(dst, 0, sizeof dst);
        fails += geist_expect(vk_buffer_copy(&hd, 0, &hs, 0, 16) == GEIST_E_BACKEND &&
                                      dst[1] == 0 && st->fallbacks[VK_FB_HOST_COPY] == b + 2,
                              "strict: a host buffer copy is refused");
        st->strict = false;
        st->strict = true;
        b          = st->fallbacks[VK_FB_HOST_VIEW];
        fails += geist_expect(vk_tensor_host(&a, &n) == nullptr &&
                                      st->fallbacks[VK_FB_HOST_VIEW] == b + 1,
                              "strict: a host view is refused");
        st->strict = false;
    }
    if (a.buffer != nullptr) {
        be->desc->vtbl->buffer_destroy(be, a.buffer);
    }
    if (c.buffer != nullptr) {
        be->desc->vtbl->buffer_destroy(be, c.buffer);
    }
    return fails;
}

/* A failed argmax dispatch is an error, not a decline. */
static int check_argmax_dispatch_failure(struct geist_backend *be) {
    struct vk_state                  *st     = be->state;
    const struct geist_backend_fused *f      = geist_backend_fused_tbl(be);
    struct geist_tensor               logits = tensor_2d(be, 1, GEIST_BUFFER_SCRATCH);
    int                               fails  = 0;
    if (logits.buffer == nullptr) {
        return geist_expect(false, "logits buffer");
    }
    float *p = be->desc->vtbl->buffer_map(logits.buffer);
    for (size_t i = 0; i < FEAT; i++) {
        p[i] = (float) (i == 37);
    }
    be->desc->vtbl->buffer_unmap(logits.buffer);
    int32_t idx = -1;
    fails += geist_expect(f->argmax_f32(be, &logits, &idx) == GEIST_OK && idx == 37,
                          "argmax on a working device");

    const uint64_t                 before = total(st);
    const PFN_vkBeginCommandBuffer real   = st->fn.BeginCommandBuffer;
    vk_seq_flush(st);
    st->fn.BeginCommandBuffer = fail_begin;
    const enum geist_status s = f->argmax_f32(be, &logits, &idx);
    st->fn.BeginCommandBuffer = real;
    (void) vk_seq_take_failure(st);
    char msg[128];
    snprintf(msg, sizeof msg, "failed argmax dispatch -> GEIST_E_BACKEND (got %d)", (int) s);
    fails += geist_expect(s == GEIST_E_BACKEND, msg);
    fails += geist_expect(total(st) == before, "a failed dispatch is not counted as a fallback");
    fails += geist_expect(f->argmax_f32(be, &logits, &idx) == GEIST_OK && idx == 37,
                          "argmax after recovery");
    be->desc->vtbl->buffer_destroy(be, logits.buffer);
    return fails;
}

/* A Q3_K weight (no GPU kernel) resolves onto the host path. */
static int check_host_weight(struct geist_backend *be) {
    struct vk_state                 *st    = be->state;
    const struct geist_backend_vtbl *vt    = be->desc->vtbl;
    int                              fails = 0;
    enum { N_OUT = 4 };
    static uint8_t      raw[N_OUT * Q3_K_BLOCK_BYTES]; /* zero blocks: d = 0 */
    struct geist_weight w = {.raw        = raw,
                             .raw_nbytes = sizeof raw,
                             .n_in       = (int32_t) Q3_K_BLOCK_ELEMS,
                             .n_out      = N_OUT,
                             .dtype      = GEIST_DTYPE_Q3_K};

    st->strict                 = true;
    const enum geist_status ss = vt->resolve_weight(be, &w);
    st->strict                 = false;
    fails += geist_expect(ss == GEIST_E_BACKEND && w.linear_m1 == nullptr && st->host_weights == 0,
                          "strict: a host-path weight is refused at resolve");
    const char *err = geist_backend_errmsg(be);
    fails += geist_expect(err != nullptr && strstr(err, "GEIST_VK_STRICT") != nullptr,
                          "strict: the resolve error names strict mode");

    fails += geist_expect(vt->resolve_weight(be, &w) == GEIST_OK && w.linear_m1 != nullptr &&
                                  w.linear_mN != nullptr,
                          "a Q3_K weight resolves onto the host path");
    fails += geist_expect(st->host_weights == 1 && st->host_weight_bytes == sizeof raw &&
                                  !st->host_weights_noted,
                          "the host weight is summed at resolve");
    float x[Q3_K_BLOCK_ELEMS], y[2 * N_OUT];
    for (size_t i = 0; i < Q3_K_BLOCK_ELEMS; i++) {
        x[i] = 1.0f;
    }
    const uint64_t b = st->fallbacks[VK_FB_HOST_LINEAR];
    w.linear_m1(x, &w, be, y);
    fails += geist_expect(st->fallbacks[VK_FB_HOST_LINEAR] == b + 1 && st->host_weights_noted,
                          "the host linear is counted and the summary noted");
    fails += geist_expect(y[0] == 0.0f && y[N_OUT - 1] == 0.0f, "zero blocks give zero output");
    float x2[2 * Q3_K_BLOCK_ELEMS];
    memcpy(x2, x, sizeof x);
    memcpy(x2 + Q3_K_BLOCK_ELEMS, x, sizeof x);
    w.linear_mN(2, x2, &w, be, y);
    fails += geist_expect(st->fallbacks[VK_FB_HOST_LINEAR] == b + 2, "linear_mN is counted too");
    return fails;
}

/* Decode a fixture model; returns failures. `strict` sets the env var the
 * backend reads at create. */
static int check_model(const char *name, const struct tf_buf *g, size_t vocab, bool strict) {
    if (strict) {
        setenv("GEIST_VK_STRICT", "1", 1);
    } else {
        unsetenv("GEIST_VK_STRICT");
    }
    struct geist_backend *be = nullptr;
    if (geist_backend_create("vulkan", nullptr, nullptr, &be) != GEIST_OK) {
        unsetenv("GEIST_VK_STRICT");
        return geist_expect(false, "vulkan backend for a model run");
    }
    struct vk_state *st    = be->state;
    int              fails = 0;
    char             msg[192];
    snprintf(msg, sizeof msg, "%s%s: strict flag read at create", name, strict ? " strict" : "");
    fails += geist_expect(st->strict == strict, msg);

    struct geist_model *m = nullptr;
    if (geist_model_load_from_memory(g->b, g->n, be, &m) != GEIST_OK) {
        snprintf(msg, sizeof msg, "%s: model load: %s", name, geist_last_create_error());
        fails += geist_expect(false, msg);
    } else {
        struct geist_session_opts  o = {.kv_mode = GEIST_KV_F16, .top_p = 1.0f, .max_seq_len = 64};
        struct geist_session      *s = nullptr;
        static const geist_token_t prompt[] = {1, 5, 9, 13, 17, 21, 25, 29};
        geist_token_t              tok      = -1;
        bool ok = geist_session_create(m, be, &o, &s) == GEIST_OK &&
                  geist_session_prefill_tokens(s, sizeof prompt / sizeof prompt[0], prompt) ==
                          GEIST_OK;
        for (int i = 0; ok && i < 6; i++) {
            ok = geist_session_decode_step(s, &tok) == GEIST_OK && tok >= 0 && (size_t) tok < vocab;
        }
        snprintf(msg,
                 sizeof msg,
                 "%s%s: prefill and decode (%s)",
                 name,
                 strict ? " strict" : "",
                 geist_backend_errmsg(be));
        fails += geist_expect(ok, msg);
        for (size_t i = 0; i < VK_FB_COUNT; i++) {
            snprintf(msg,
                     sizeof msg,
                     "%s%s: no %s fallback (%llu)",
                     name,
                     strict ? " strict" : "",
                     vk_fallback_name((enum vk_fb) i),
                     (unsigned long long) st->fallbacks[i]);
            fails += geist_expect(st->fallbacks[i] == 0, msg);
        }
        snprintf(msg, sizeof msg, "%s: no weight on the host path", name);
        fails += geist_expect(st->host_weights == 0 && st->fallbacks[VK_FB_HOST_LINEAR] == 0, msg);
        if (s != nullptr) {
            geist_session_destroy(s);
        }
        geist_model_destroy(m);
    }
    geist_backend_destroy(be);
    unsetenv("GEIST_VK_STRICT");
    return fails;
}

int main(void) {
    unsetenv("GEIST_VK_STRICT");
    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("vulkan", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "SKIP: vulkan backend unavailable (%d)\n", (int) s);
        return GEIST_TEST_SKIP;
    }
    struct vk_state *st    = be->state;
    int              fails = geist_expect(!st->strict && total(st) == 0, "a fresh backend");
    fails += check_declines(be);
    st->strict = true;
    fails += check_declines(be);
    st->strict = false;
    fails += check_host_access(be);
    fails += check_argmax_dispatch_failure(be);
    fails += check_host_weight(be);
    geist_backend_destroy(be);

    struct tf_vocab v     = tf_make_vocab("\xc4\xa0", false);
    struct tf_buf   llama = mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                                              .d_model  = 128,
                                                              .heads    = 4,
                                                              .kv_heads = 2,
                                                              .ffn      = 256,
                                                              .vocab    = 512,
                                                              .context  = 256,
                                                              .seed     = 14});
    struct tf_buf   qwen  = mf_qwen35_gguf(&(struct mf_qwen35) {.layers     = 4,
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
    for (int strict = 0; strict < 2; strict++) {
        fails += check_model("llama", &llama, 512, strict != 0);
        fails += check_model("qwen35", &qwen, v.n_tok, strict != 0);
    }
    free(llama.b);
    free(qwen.b);
    tf_free_vocab(&v);

    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("vulkan fallback accounting: pass\n");
    return GEIST_TEST_PASS;
}

#else
int main(void) {
    fprintf(stderr, "SKIP: vulkan backend not built\n");
    return GEIST_TEST_SKIP;
}
#endif
