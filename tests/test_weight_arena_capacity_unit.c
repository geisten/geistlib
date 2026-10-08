/*
 * test_weight_arena_capacity_unit — the β-mode weight arena holds what the
 * loader puts in it, on every backend (#561, #468).
 *
 * 1. compute_weight_arena_capacity on a two-tensor in-memory GGUF (a 32x64
 *    F16 matrix, a 16-element F32 vector): without caps.weights_device_copy
 *    the first chunk counts every tensor plus the F32 copy of the small
 *    half-precision matrix, which load_layer_proj adds whenever the
 *    backend's resolve_weight refuses it — with and without
 *    caps.weights_need_backend_arena (Metal leaves that cap false, refuses
 *    F16, and still runs the arena under GEIST_WEIGHT_MMAP=0, #561).
 *
 * 2. The fixture models (model_fixtures.h: llama tied and untied, llama
 *    with an F16 attn_q, Qwen3.5 hybrid) on cpu_scalar, against the
 *    mmap-alias default:
 *    - β mode (GEIST_WEIGHT_MMAP=0): one arena chunk holds the model, so
 *      the pre-scan covers everything the loader stores, and greedy
 *      decoding gives the same tokens;
 *    - β mode with the caps of a device-copy backend (Vulkan's): the
 *      loader's storage intent alone decides the arena — the matrices and
 *      the lookup tables alias the GGUF, the norms are in the arena — and
 *      the model decodes the same tokens.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "test_helpers.h"
#include "model_fixtures.h"

#include "gguf_reader.h"
#include "src/archs/transformer/arch_state.h"
#include "src/archs/transformer/weight_load.h"
#include "src/engine/model.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t put_u32(uint8_t *p, uint32_t v) {
    memcpy(p, &v, 4);
    return 4;
}

static size_t put_u64(uint8_t *p, uint64_t v) {
    memcpy(p, &v, 8);
    return 8;
}

static size_t put_str(uint8_t *p, const char *s) {
    const uint64_t n = strlen(s);
    size_t         o = put_u64(p, n);
    memcpy(p + o, s, n);
    return o + n;
}

static size_t put_tensor_info(uint8_t       *p,
                              const char    *name,
                              uint32_t       n_dims,
                              const uint64_t dims[static n_dims],
                              uint32_t       dtype,
                              uint64_t       offset) {
    size_t o = put_str(p, name);
    o += put_u32(p + o, n_dims);
    for (uint32_t i = 0; i < n_dims; i++) {
        o += put_u64(p + o, dims[i]);
    }
    o += put_u32(p + o, dtype);
    return o + put_u64(p + o, offset);
}

enum { F16_ELEMS = 64 * 32, F16_BYTES = F16_ELEMS * 2, F32_BYTES = 16 * 4 };

static size_t build_gguf(uint8_t *buf) {
    size_t o = 0;
    memcpy(buf + o, "GGUF", 4);
    o += 4;
    o += put_u32(buf + o, 3); /* version */
    o += put_u64(buf + o, 2); /* n_tensors */
    o += put_u64(buf + o, 0); /* n_kv */
    const uint64_t mat[2] = {64, 32};
    const uint64_t vec[1] = {16};
    o += put_tensor_info(buf + o, "blk.0.attn_k.weight", 2, mat, GGUF_TYPE_F16, 0);
    o += put_tensor_info(buf + o, "blk.0.attn_norm.weight", 1, vec, GGUF_TYPE_F32, F16_BYTES);
    while (o % 32 != 0) {
        buf[o++] = 0;
    }
    memset(buf + o, 0, F16_BYTES + F32_BYTES);
    return o + F16_BYTES + F32_BYTES;
}

static size_t capacity(struct gguf_ctx *g, bool need_backend_arena, bool device_copy) {
    struct geist_backend_descriptor desc = {.name = "caps-only"};
    desc.caps.weights_need_backend_arena = need_backend_arena;
    desc.caps.weights_device_copy        = device_copy;
    struct geist_backend be              = {.desc = &desc};
    size_t               cap             = 0;
    if (compute_weight_arena_capacity(&be, g, &cap) != GEIST_OK) {
        return 0;
    }
    return cap;
}

static int check_prescan(void) {
    static uint8_t   buf[16384];
    const size_t     n   = build_gguf(buf);
    const char      *err = nullptr;
    struct gguf_ctx *g   = gguf_open_memory(buf, n, &err);
    if (g == nullptr) {
        fprintf(stderr, "gguf_open_memory: %s\n", err != nullptr ? err : "(null)");
        return 1;
    }
    /* The matrix as in the file and as F32; the vector; the 64 MB
     * headroom. Every size here is already a multiple of 64. */
    const size_t want = F16_BYTES + F16_ELEMS * sizeof(float) + 64 + (64u << 20);

    int fails = 0;
    fails += geist_expect(capacity(g, true, false) == want,
                          "arena counts the widen with the cap set");
    fails += geist_expect(capacity(g, false, false) == want,
                          "arena counts the widen without the cap (metal, #561)");
    fails += geist_expect(capacity(g, true, true) > 0,
                          "a device-copy backend starts the arena at one chunk");
    gguf_close(g);
    return fails;
}

/* ---- Fixture models ----------------------------------------------------- */

/* Prefill + 6 greedy steps; false on any failure. */
static bool decode(struct geist_model *m, struct geist_backend *be, geist_token_t out[6]) {
    struct geist_session_opts  o         = {.top_p = 1.0f, .max_seq_len = 64};
    struct geist_session      *s         = nullptr;
    static const geist_token_t prompt[3] = {7, 3, 11};
    bool                       ok        = geist_session_create(m, be, &o, &s) == GEIST_OK &&
                                           geist_session_prefill_tokens(s, 3, prompt) == GEIST_OK;
    for (int i = 0; ok && i < 6; i++) {
        ok = geist_session_decode_step(s, &out[i]) == GEIST_OK;
    }
    if (s != nullptr) {
        geist_session_destroy(s);
    }
    return ok;
}

enum load_mode { MODE_MMAP, MODE_ARENA, MODE_DEVICE_COPY };

/* Load `g` on cpu_scalar in `mode`, check the arena, decode into out. */
static int load_and_decode(const char          *name,
                           const struct tf_buf *g,
                           enum load_mode       mode,
                           geist_token_t        out[6]) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK) {
        return geist_expect(false, "cpu_scalar backend");
    }
    const struct geist_backend_descriptor *own  = be->desc;
    struct geist_backend_descriptor        copy = *own;
    if (mode == MODE_DEVICE_COPY) {
        /* Vulkan's load path on a CPU backend: the arena through the
         * backend, matrices left to resolve_weight. */
        copy.caps.weights_need_backend_arena = true;
        copy.caps.weights_device_copy        = true;
        be->desc                             = &copy;
    }
    if (mode == MODE_ARENA) {
        setenv("GEIST_WEIGHT_MMAP", "0", 1);
    }
    int                 fails = 0;
    char                msg[160];
    struct geist_model *m = nullptr;
    if (geist_model_load_from_memory(g->b, g->n, be, &m) != GEIST_OK) {
        snprintf(msg,
                 sizeof msg,
                 "%s: loads (mode %d): %s",
                 name,
                 (int) mode,
                 geist_backend_errmsg(be));
        fails += geist_expect(false, msg);
    }
    unsetenv("GEIST_WEIGHT_MMAP");
    if (m != nullptr) {
        const struct transformer_arch_state *st = m->text_decoder.arch_meta;
        const struct geist_backend_vtbl     *v  = be->desc->vtbl;
        /* Host address of a view's first byte. */
#define HOST_OF(view) ((const uint8_t *) v->buffer_map((view).buffer) + (view).offset)
        const uint8_t *lo = g->b, *hi = g->b + g->n;
#define IN_FILE(p) ((p) >= lo && (p) < hi)
        if (mode == MODE_ARENA) {
            snprintf(msg,
                     sizeof msg,
                     "%s: the pre-scan's chunk holds the model (%zu chunks)",
                     name,
                     st->n_weight_arena_chunks);
            fails += geist_expect(st->n_weight_arena_chunks == 1, msg);
            snprintf(msg, sizeof msg, "%s: β mode copies token_embd", name);
            fails += geist_expect(!IN_FILE(HOST_OF(st->embed_table)), msg);
        }
        if (mode == MODE_DEVICE_COPY) {
            const bool untied = st->output_table.buffer != st->embed_table.buffer;
            snprintf(msg, sizeof msg, "%s: device copy: arena open", name);
            fails += geist_expect(st->n_weight_arena_chunks >= 1, msg);
            snprintf(msg,
                     sizeof msg,
                     "%s: device copy: token_embd (%s) aliases the file",
                     name,
                     untied ? "lookup" : "lm_head");
            fails += geist_expect(IN_FILE(HOST_OF(st->embed_table)), msg);
            if (untied) {
                snprintf(msg, sizeof msg, "%s: device copy: output.weight aliases the file", name);
                fails += geist_expect(IN_FILE(HOST_OF(st->output_table)), msg);
            }
            snprintf(msg, sizeof msg, "%s: device copy: ffn_down aliases the file", name);
            fails += geist_expect(IN_FILE(HOST_OF(st->layers[0].down_proj)), msg);
            snprintf(msg, sizeof msg, "%s: device copy: output_norm is in the arena", name);
            fails += geist_expect(!IN_FILE(HOST_OF(st->output_norm)), msg);
        }
#undef IN_FILE
#undef HOST_OF
        if (!decode(m, be, out)) {
            snprintf(msg, sizeof msg, "%s: decodes (mode %d)", name, (int) mode);
            fails += geist_expect(false, msg);
        }
        geist_model_destroy(m);
    }
    be->desc = own;
    geist_backend_destroy(be);
    return fails;
}

static int check_fixture(const char *name, struct tf_buf g) {
    geist_token_t ref[6] = {0}, arena[6] = {0}, dev[6] = {0};
    int           fails = load_and_decode(name, &g, MODE_MMAP, ref);
    fails += load_and_decode(name, &g, MODE_ARENA, arena);
    fails += load_and_decode(name, &g, MODE_DEVICE_COPY, dev);
    char msg[160];
    snprintf(msg, sizeof msg, "%s: β mode decodes as mmap-alias", name);
    fails += geist_expect(memcmp(ref, arena, sizeof ref) == 0, msg);
    snprintf(msg, sizeof msg, "%s: device-copy loading decodes as mmap-alias", name);
    fails += geist_expect(memcmp(ref, dev, sizeof ref) == 0, msg);
    printf("  %s: tokens %d %d %d %d %d %d\n",
           name,
           ref[0],
           ref[1],
           ref[2],
           ref[3],
           ref[4],
           ref[5]);
    free(g.b);
    return fails;
}

int main(void) {
    int fails = check_prescan();

    const struct mf_llama llama  = {.layers   = 2,
                                    .d_model  = 128,
                                    .heads    = 4,
                                    .kv_heads = 2,
                                    .ffn      = 256,
                                    .vocab    = 64,
                                    .context  = 64,
                                    .seed     = 3};
    struct mf_llama       untied = llama, f16 = llama;
    untied.untied = true;
    f16.f16_q     = true;
    fails += check_fixture("llama", mf_llama_gguf(&llama));
    fails += check_fixture("llama-untied", mf_llama_gguf(&untied));
    fails += check_fixture("llama-f16-q", mf_llama_gguf(&f16));

    struct tf_vocab v = tf_make_vocab("\xc4\xa0", false);
    fails += check_fixture("qwen35",
                           mf_qwen35_gguf(&(struct mf_qwen35) {.layers     = 3,
                                                               .interval   = 3,
                                                               .d_model    = 128,
                                                               .heads      = 4,
                                                               .kv_heads   = 2,
                                                               .head_dim   = 32,
                                                               .rope_dims  = 16,
                                                               .ffn        = 256,
                                                               .dn_k_heads = 2,
                                                               .dn_v_heads = 4,
                                                               .dn_head_k  = 16,
                                                               .dn_head_v  = 16,
                                                               .dn_conv    = 4,
                                                               .seed       = 5,
                                                               .tok        = &v}));
    tf_free_vocab(&v);

    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("weight arena capacity: pass\n");
    return GEIST_TEST_PASS;
}
