/*
 * src/archs/transformer/rotation.c — prism.hadamard loader and the
 * forward-pass entry point. See rotation.h.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "rotation.h"

#include "checked.h"
#include "forward/internal.h"
#include "gguf_reader.h"
#include "heap.h"

#include <geist_backend.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Folded-weight kinds the qwen35 forward pass rotates the input of: the
 * GGUF name suffix, the mixer whose layers carry it (MIXER_ANY = every
 * layer), and the weight whose n_in is the rotated width. One row per
 * kind, so a new kind is one line rather than an enum plus two switches.
 * The rows must stay in step with the transformer_rotate() call sites in
 * forward/ — that pairing is what check_weight_names enforces. */
#define MIXER_ANY ((enum transformer_mixer_kind) - 1)

static const struct {
    const char                 *name;
    enum transformer_mixer_kind mixer;
    size_t                      w_off; /* into struct transformer_layer_weights */
} ROT_KINDS[] = {
        {"attn_qkv", GEIST_MIXER_DELTANET, offsetof(struct transformer_layer_weights, dn_qkv_w)},
        {"attn_gate", GEIST_MIXER_DELTANET, offsetof(struct transformer_layer_weights, dn_z_w)},
        {"ssm_out", GEIST_MIXER_DELTANET, offsetof(struct transformer_layer_weights, dn_out_w)},
        {"attn_q", GEIST_MIXER_ATTN, offsetof(struct transformer_layer_weights, q_proj_w)},
        {"attn_k", GEIST_MIXER_ATTN, offsetof(struct transformer_layer_weights, k_proj_w)},
        {"attn_v", GEIST_MIXER_ATTN, offsetof(struct transformer_layer_weights, v_proj_w)},
        {"attn_output", GEIST_MIXER_ATTN, offsetof(struct transformer_layer_weights, o_proj_w)},
        {"ffn_gate", MIXER_ANY, offsetof(struct transformer_layer_weights, gate_proj_w)},
        {"ffn_up", MIXER_ANY, offsetof(struct transformer_layer_weights, up_proj_w)},
        {"ffn_down", MIXER_ANY, offsetof(struct transformer_layer_weights, down_proj_w)},
};

enum { ROT_KIND_COUNT = (int) (sizeof ROT_KINDS / sizeof ROT_KINDS[0]) };

static bool kind_on_layer(size_t k, enum transformer_mixer_kind mixer) {
    return ROT_KINDS[k].mixer == MIXER_ANY || ROT_KINDS[k].mixer == mixer;
}

/* The weight a kind names, for its input width. */
static const struct geist_weight *kind_weight(const struct transformer_layer_weights *L, size_t k) {
    return (const struct geist_weight *) ((const char *) L + ROT_KINDS[k].w_off);
}

[[nodiscard]] static enum geist_status fail(struct transformer_arch_state *st,
                                            enum geist_status              code,
                                            const char                    *what,
                                            const char                    *detail) {
    geist_backend_set_error(st->backend,
                            code,
                            "prism.hadamard: %s%s%s",
                            what,
                            detail != nullptr ? ": " : "",
                            detail != nullptr ? detail : "");
    return code;
}

static bool meta_str_is(struct gguf_ctx *g, const char *key, const char *want) {
    size_t      len = 0;
    const char *v   = gguf_get_meta_string(g, key, &len);
    return v != nullptr && len == strlen(want) && memcmp(v, want, len) == 0;
}

/* Walk a GGUF string array. The reader validated every element's extent
 * when it opened the file, so the (u64 length, bytes) pairs can be read
 * back without re-checking the cursor against the payload end. */
struct str_iter {
    const uint8_t *p;
    uint64_t       left;
};

static bool str_next(struct str_iter *it, const char **s, size_t *len) {
    if (it->left == 0) {
        return false;
    }
    uint64_t n = 0;
    memcpy(&n, it->p, sizeof n);
    *s   = (const char *) (it->p + sizeof n);
    *len = (size_t) n;
    it->p += sizeof n + (size_t) n;
    it->left--;
    return true;
}

/* "blk.<i>.<kind>.weight" -> (i, kind); false on anything else. */
static bool parse_layer_name(const char *s, size_t len, size_t *layer, size_t *kind) {
    static const char pre[] = "blk.";
    static const char suf[] = ".weight";
    if (len <= sizeof pre - 1 + sizeof suf - 1 || memcmp(s, pre, sizeof pre - 1) != 0 ||
        memcmp(s + len - (sizeof suf - 1), suf, sizeof suf - 1) != 0) {
        return false;
    }
    size_t i = sizeof pre - 1, idx = 0, digits = 0;
    while (i < len && s[i] >= '0' && s[i] <= '9' && digits < 6) {
        idx = idx * 10 + (size_t) (s[i] - '0');
        i++;
        digits++;
    }
    if (digits == 0 || i >= len || s[i] != '.') {
        return false;
    }
    i++;
    /* The digit loop can walk into the suffix ("blk.123456.weight"), so
     * the remainder is not a length until it is known to be one. */
    if (i + (sizeof suf - 1) > len) {
        return false;
    }
    const size_t kind_len = len - (sizeof suf - 1) - i;
    for (size_t k = 0; k < (size_t) ROT_KIND_COUNT; k++) {
        if (strlen(ROT_KINDS[k].name) == kind_len &&
            memcmp(s + i, ROT_KINDS[k].name, kind_len) == 0) {
            *layer = idx;
            *kind  = k;
            return true;
        }
    }
    return false;
}

static bool is_name(const char *s, size_t len, const char *want) {
    return len == strlen(want) && memcmp(s, want, len) == 0;
}

/* Exactly the folded set the forward pass rotates: every kind of every
 * layer's mixer, all three FFN projections, and output.weight — no more,
 * no less, no duplicates. */
[[nodiscard]] static enum geist_status check_weight_names(struct transformer_arch_state *st,
                                                          struct gguf_ctx               *g) {
    uint32_t       vt = 0;
    uint64_t       n  = 0;
    const uint8_t *p  = nullptr;
    if (!gguf_get_meta_array_info(g, "prism.hadamard.weight_names", &vt, &n, &p) ||
        vt != GGUF_META_VT_STRING || n == 0) {
        return fail(st, GEIST_E_FORMAT, "weight_names missing or not a string array", nullptr);
    }
    size_t cells = 0;
    if (ckd_mul(&cells, st->n_layers, (size_t) ROT_KIND_COUNT)) {
        return fail(st, GEIST_E_FORMAT, "layer count overflows", nullptr);
    }
    if (ckd_add(&cells, cells, 1)) {
        return fail(st, GEIST_E_FORMAT, "layer count overflows", nullptr);
    }
    bool *seen = heap_calloc_aligned(cells, sizeof(bool), alignof(bool));
    if (seen == nullptr) {
        return fail(st, GEIST_E_OOM, "out of memory", nullptr);
    }
    bool             *seen_output = &seen[cells - 1];
    enum geist_status s           = GEIST_OK;
    struct str_iter   it          = {.p = p, .left = n};
    const char       *name        = nullptr;
    size_t            len         = 0;
    while (s == GEIST_OK && str_next(&it, &name, &len)) {
        size_t layer = 0;
        size_t kind  = (size_t) ROT_KIND_COUNT;
        bool  *slot  = nullptr;
        if (is_name(name, len, "output.weight")) {
            slot = seen_output;
        } else if (parse_layer_name(name, len, &layer, &kind) && layer < st->n_layers &&
                   kind_on_layer(kind, st->layers[layer].mixer)) {
            slot = &seen[layer * ROT_KIND_COUNT + kind];
        }
        if (slot == nullptr) {
            s = fail(st, GEIST_E_FORMAT, "weight is not on a rotated path", nullptr);
        } else if (*slot) {
            s = fail(st, GEIST_E_FORMAT, "duplicate weight name", nullptr);
        } else {
            *slot = true;
        }
    }
    for (size_t l = 0; s == GEIST_OK && l < st->n_layers; l++) {
        for (size_t k = 0; k < (size_t) ROT_KIND_COUNT; k++) {
            if (kind_on_layer(k, st->layers[l].mixer) && !seen[l * ROT_KIND_COUNT + k]) {
                s = fail(st, GEIST_E_FORMAT, "layer only partially rotated", ROT_KINDS[k].name);
                break;
            }
        }
    }
    if (s == GEIST_OK && !*seen_output) {
        s = fail(st, GEIST_E_FORMAT, "output.weight not rotated", nullptr);
    }
    void *vp = seen;
    safe_free(&vp);
    return s;
}

[[nodiscard]] static enum geist_status check_inverse_names(struct transformer_arch_state *st,
                                                           struct gguf_ctx               *g) {
    uint32_t       vt = 0;
    uint64_t       n  = 0;
    const uint8_t *p  = nullptr;
    if (!gguf_get_meta_array_info(g, "prism.hadamard.inverse_weight_names", &vt, &n, &p)) {
        st->rotation.embed_inverse = false;
        return GEIST_OK;
    }
    struct str_iter it   = {.p = p, .left = n};
    const char     *name = nullptr;
    size_t          len  = 0;
    if (vt != GGUF_META_VT_STRING || n != 1 || !str_next(&it, &name, &len) ||
        !is_name(name, len, "token_embd.weight")) {
        return fail(
                st, GEIST_E_FORMAT, "inverse_weight_names must be [token_embd.weight]", nullptr);
    }
    st->rotation.embed_inverse = true;
    return GEIST_OK;
}

/* sign_widths / sign_values -> one F32 backend buffer per width. */
[[nodiscard]] static enum geist_status load_signs(struct transformer_arch_state *st,
                                                  struct gguf_ctx               *g) {
    uint32_t       wvt = 0, vvt = 0;
    uint64_t       nw = 0, nv = 0;
    const uint8_t *wp = nullptr, *vp = nullptr;
    if (!gguf_get_meta_array_info(g, "prism.hadamard.sign_widths", &wvt, &nw, &wp) ||
        !gguf_get_meta_array_info(g, "prism.hadamard.sign_values", &vvt, &nv, &vp) ||
        wvt != GGUF_META_VT_I32 || vvt != GGUF_META_VT_I32 || nw == 0) {
        return fail(st, GEIST_E_FORMAT, "explicit sign_mode needs I32 sign_widths/values", nullptr);
    }
    if (nw > sizeof st->rotation.signs / sizeof st->rotation.signs[0]) {
        return fail(st, GEIST_E_UNSUPPORTED, "too many sign widths", nullptr);
    }
    const struct geist_backend_vtbl *vt  = st->backend->desc->vtbl;
    uint64_t                         off = 0;
    for (size_t i = 0; i < (size_t) nw; i++) {
        int32_t w = 0;
        memcpy(&w, wp + i * sizeof w, sizeof w);
        if (w <= 0 || (size_t) w % st->rotation.block != 0 || (uint64_t) w > nv - off) {
            return fail(st, GEIST_E_FORMAT, "bad sign width", nullptr);
        }
        for (size_t j = 0; j < i; j++) {
            if (st->rotation.signs[j].width == (size_t) w) {
                return fail(st, GEIST_E_FORMAT, "duplicate sign width", nullptr);
            }
        }
        float *host = heap_alloc_array_aligned(float, (size_t) w);
        if (host == nullptr) {
            return fail(st, GEIST_E_OOM, "out of memory", nullptr);
        }
        bool ok = true;
        for (size_t j = 0; j < (size_t) w; j++) {
            int32_t v = 0;
            memcpy(&v, vp + (off + j) * sizeof v, sizeof v);
            ok      = ok && (v == 1 || v == -1);
            host[j] = (float) v;
        }
        enum geist_status    s   = ok ? GEIST_OK : GEIST_E_FORMAT;
        struct geist_buffer *buf = nullptr;
        if (s == GEIST_OK) {
            s = vt->buffer_create(st->backend,
                                  (size_t) w * sizeof(float),
                                  GEIST_BUFFER_WEIGHT,
                                  GEIST_MEMORY_AUTO,
                                  &buf);
        }
        if (s == GEIST_OK) {
            s = vt->buffer_upload(buf, (size_t) w * sizeof(float), (const uint8_t *) host);
        }
        void *hp = host;
        safe_free(&hp);
        if (buf != nullptr) {
            st->rotation.signs[st->rotation.n_signs].width = (size_t) w;
            st->rotation.signs[st->rotation.n_signs].buf   = buf;
            st->rotation.n_signs++;
        }
        if (s != GEIST_OK) {
            return fail(
                    st, s, ok ? "sign buffer upload failed" : "sign values must be +/-1", nullptr);
        }
        off += (uint64_t) w;
    }
    if (off != nv) {
        return fail(st, GEIST_E_FORMAT, "sign_values length mismatch", nullptr);
    }
    return GEIST_OK;
}

static struct geist_buffer *signs_for(const struct transformer_arch_state *st, size_t width) {
    for (size_t i = 0; i < st->rotation.n_signs; i++) {
        if (st->rotation.signs[i].width == width) {
            return st->rotation.signs[i].buf;
        }
    }
    return nullptr;
}

/* Every rotated input width is a whole number of blocks and, in explicit
 * mode, has its sign vector. */
[[nodiscard]] static enum geist_status
check_width(struct transformer_arch_state *st, size_t width, bool explicit_signs) {
    if (width == 0 || width % st->rotation.block != 0) {
        return fail(st, GEIST_E_FORMAT, "block size does not divide an input width", nullptr);
    }
    if (explicit_signs && signs_for(st, width) == nullptr) {
        return fail(st, GEIST_E_FORMAT, "no sign vector for an input width", nullptr);
    }
    return GEIST_OK;
}

enum geist_status transformer_rotation_load(struct transformer_arch_state *st) {
    st->rotation             = (struct transformer_rotation) {0};
    struct gguf_ctx *g       = (struct gguf_ctx *) st->gguf;
    uint32_t         version = 0;
    if (g == nullptr || !gguf_get_meta_u32(g, "prism.hadamard.version", &version)) {
        return GEIST_OK;
    }
    if (version != 1) {
        return fail(st, GEIST_E_UNSUPPORTED, "unsupported version", nullptr);
    }
    if (!meta_str_is(g, "general.architecture", "qwen35")) {
        return fail(st, GEIST_E_UNSUPPORTED, "only wired for the qwen35 family", nullptr);
    }
    if (st->n_mtp_layers != 0) {
        return fail(st, GEIST_E_UNSUPPORTED, "MTP layers are not rotated", nullptr);
    }
    if (st->output_table.buffer == st->embed_table.buffer) {
        return fail(st, GEIST_E_UNSUPPORTED, "tied lm_head", nullptr);
    }
    if (geist_backend_fused_tbl(st->backend)->hadamard_rotate == nullptr) {
        return fail(
                st, GEIST_E_UNSUPPORTED, "backend has no hadamard_rotate", st->backend->desc->name);
    }
    if (!meta_str_is(g, "prism.hadamard.transform", "normalized-sylvester-walsh-hadamard")) {
        return fail(st, GEIST_E_UNSUPPORTED, "unsupported transform", nullptr);
    }
    if (!meta_str_is(g, "prism.hadamard.axis", "input-last-dimension")) {
        return fail(st, GEIST_E_UNSUPPORTED, "unsupported axis", nullptr);
    }
    const bool explicit_signs = meta_str_is(g, "prism.hadamard.sign_mode", "explicit");
    if (!explicit_signs && !meta_str_is(g, "prism.hadamard.sign_mode", "identity")) {
        return fail(st, GEIST_E_UNSUPPORTED, "unsupported sign_mode", nullptr);
    }
    uint32_t block = 0;
    if (!gguf_get_meta_u32(g, "prism.hadamard.block_size", &block) || block == 0 ||
        (block & (block - 1)) != 0) {
        return fail(st, GEIST_E_FORMAT, "block_size missing or not a power of two", nullptr);
    }
    st->rotation.block = block;
    /* Absent (or not a bool) = tiled order, as in the fork. */
    bool grouped = false;
    if (!gguf_get_meta_bool(g, "prism.hadamard.gdn_v_grouped", &grouped)) {
        grouped = false;
    }
    st->rotation.gdn_v_grouped = grouped;

    enum geist_status s = check_weight_names(st, g);
    if (s == GEIST_OK) {
        s = check_inverse_names(st, g);
    }
    if (s == GEIST_OK && explicit_signs) {
        s = load_signs(st, g);
    }
    /* Widths: the residual stream (every normed input, the lm_head input,
     * the embedding rows) and each layer's mixer-output and FFN widths. */
    if (s == GEIST_OK) {
        s = check_width(st, st->d_model, explicit_signs);
    }
    for (size_t l = 0; s == GEIST_OK && l < st->n_layers; l++) {
        const struct transformer_layer_weights *L = &st->layers[l];
        for (size_t k = 0; s == GEIST_OK && k < (size_t) ROT_KIND_COUNT; k++) {
            if (kind_on_layer(k, L->mixer)) {
                s = check_width(st, (size_t) kind_weight(L, k)->n_in, explicit_signs);
            }
        }
        if (s == GEIST_OK && grouped && L->mixer == GEIST_MIXER_DELTANET) {
            const size_t nk = st->config.dn_n_k_heads, nv = st->config.dn_n_v_heads;
            if (nk == 0 || nv % nk != 0 || (size_t) L->dn_out_w.n_in != nv * st->config.dn_head_v) {
                s = fail(st, GEIST_E_FORMAT, "bad grouped-value head geometry", nullptr);
            }
        }
    }
    if (s != GEIST_OK) {
        transformer_rotation_release(st);
        return s;
    }
    st->rotation.active = true;
    return GEIST_OK;
}

void transformer_rotation_release(struct transformer_arch_state *st) {
    for (size_t i = 0; i < st->rotation.n_signs; i++) {
        if (st->rotation.signs[i].buf != nullptr) {
            st->backend->desc->vtbl->buffer_destroy(st->backend, st->rotation.signs[i].buf);
        }
    }
    st->rotation = (struct transformer_rotation) {0};
}

enum geist_status transformer_rotate(const struct transformer_arch_state *st,
                                     size_t                               rows,
                                     size_t                               width,
                                     bool                                 grouped_v,
                                     bool                                 inverse,
                                     struct geist_buffer                 *x,
                                     struct geist_buffer                 *y) {
    if (!st->rotation.active || rows == 0) {
        return GEIST_OK;
    }
    struct geist_buffer             *sb   = signs_for(st, width);
    const struct geist_tensor        tx   = view_2d(x, (int64_t) rows, (int64_t) width);
    struct geist_tensor              ty   = view_2d(y, (int64_t) rows, (int64_t) width);
    const struct geist_tensor        ts   = view_1d(sb, (int64_t) width);
    const bool                       perm = grouped_v && st->rotation.gdn_v_grouped;
    const struct geist_hadamard_args args = {
            .x        = &tx,
            .signs    = sb != nullptr ? &ts : nullptr,
            .y        = &ty,
            .block    = st->rotation.block,
            .perm_hd  = perm ? st->config.dn_head_v : 0,
            .perm_nk  = perm ? st->config.dn_n_k_heads : 0,
            .perm_rep = perm ? st->config.dn_n_v_heads / st->config.dn_n_k_heads : 0,
            .inverse  = inverse,
    };
    return geist_backend_fused_tbl(st->backend)->hadamard_rotate(st->backend, &args);
}
