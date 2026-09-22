/*
 * test_rotation_load_unit — transformer_rotation_load, the gate between a
 * prism.hadamard GGUF and the rotated call sites of the qwen35 forward.
 *
 * A folded weight whose input is not rotated (or an unfolded one whose
 * input is) computes garbage without failing anything, so the loader must
 * accept exactly the folded set the forward rotates and refuse the rest.
 * Each case builds an in-memory GGUF carrying only the metadata and a
 * two-layer state (layer 0 DeltaNet, layer 1 attention) with just the
 * fields the loader reads:
 *
 *   accepted: the full description (explicit signs, grouped values,
 *             inverse embedding), identity signs, no prism keys at all;
 *   FORMAT:   unknown / misplaced / duplicate names, a partially rotated
 *             layer, unrotated output, bad block sizes, sign tables that
 *             are short, long, not +/-1 or miss a width, bad inverse names,
 *             bad grouped-value geometry;
 *   UNSUPPORTED: other versions / transforms / axes / sign modes, another
 *             family, MTP layers, a tied lm_head, a backend without
 *             fused->hadamard_rotate.
 *
 * Every refusal must also leave no sign buffers behind.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "test_helpers.h"

#include "../src/archs/transformer/rotation.h"
#include "gguf_reader.h"

#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- GGUF metadata builder ---------------------------------------------- */

struct buf {
    uint8_t *b;
    size_t   n, cap;
};

static void put(struct buf *o, const void *p, size_t n) {
    if (o->n + n > o->cap) {
        fprintf(stderr, "test buffer overflow\n");
        exit(GEIST_TEST_ERROR);
    }
    memcpy(o->b + o->n, p, n);
    o->n += n;
}
static void put_u32(struct buf *o, uint32_t v) {
    put(o, &v, 4);
}
static void put_u64(struct buf *o, uint64_t v) {
    put(o, &v, 8);
}
static void put_str(struct buf *o, const char *s) {
    put_u64(o, strlen(s));
    put(o, s, strlen(s));
}

enum { VT_U32 = 4, VT_I32 = 5, VT_BOOL = 7, VT_STRING = 8, VT_ARRAY = 9 };

/* One description of the prism.hadamard keys; nullptr / 0 / false mean
 * "omit the key". */
struct spec {
    const char    *arch;
    bool           has_version;
    uint32_t       version;
    const char    *transform, *axis, *sign_mode;
    uint32_t       block;
    const char   **names;
    size_t         n_names;
    const char   **inv;
    size_t         n_inv; /* 0 = key absent */
    const int32_t *widths;
    size_t         n_widths;
    const int32_t *vals;
    size_t         n_vals;
    bool           grouped;
};

static size_t n_meta(const struct spec *s) {
    return 1 + s->has_version + (s->transform != nullptr) + (s->axis != nullptr) +
           (s->sign_mode != nullptr) + (s->block != 0) + (s->names != nullptr) + (s->n_inv != 0) +
           (s->widths != nullptr) + (s->vals != nullptr) + s->grouped;
}

static void kv_str(struct buf *o, const char *k, const char *v) {
    put_str(o, k);
    put_u32(o, VT_STRING);
    put_str(o, v);
}
static void kv_strs(struct buf *o, const char *k, const char **v, size_t n) {
    put_str(o, k);
    put_u32(o, VT_ARRAY);
    put_u32(o, VT_STRING);
    put_u64(o, n);
    for (size_t i = 0; i < n; i++) {
        put_str(o, v[i]);
    }
}
static void kv_i32s(struct buf *o, const char *k, const int32_t *v, size_t n) {
    put_str(o, k);
    put_u32(o, VT_ARRAY);
    put_u32(o, VT_I32);
    put_u64(o, n);
    put(o, v, n * sizeof *v);
}

static struct buf build(const struct spec *s) {
    struct buf o = {.b = xmalloc(1 << 16), .cap = 1 << 16};
    put_u32(&o, 0x46554747u);
    put_u32(&o, 3);
    put_u64(&o, 1); /* the reader wants at least one tensor */
    put_u64(&o, n_meta(s));
    kv_str(&o, "general.architecture", s->arch);
    if (s->has_version) {
        put_str(&o, "prism.hadamard.version");
        put_u32(&o, VT_U32);
        put_u32(&o, s->version);
    }
    if (s->transform != nullptr)
        kv_str(&o, "prism.hadamard.transform", s->transform);
    if (s->axis != nullptr)
        kv_str(&o, "prism.hadamard.axis", s->axis);
    if (s->sign_mode != nullptr)
        kv_str(&o, "prism.hadamard.sign_mode", s->sign_mode);
    if (s->block != 0) {
        put_str(&o, "prism.hadamard.block_size");
        put_u32(&o, VT_U32);
        put_u32(&o, s->block);
    }
    if (s->names != nullptr)
        kv_strs(&o, "prism.hadamard.weight_names", s->names, s->n_names);
    if (s->n_inv != 0)
        kv_strs(&o, "prism.hadamard.inverse_weight_names", s->inv, s->n_inv);
    if (s->widths != nullptr)
        kv_i32s(&o, "prism.hadamard.sign_widths", s->widths, s->n_widths);
    if (s->vals != nullptr)
        kv_i32s(&o, "prism.hadamard.sign_values", s->vals, s->n_vals);
    if (s->grouped) {
        put_str(&o, "prism.hadamard.gdn_v_grouped");
        put_u32(&o, VT_BOOL);
        put(&o, "\1", 1);
    }
    put_str(&o, "w"); /* F32 [32] at offset 0 */
    put_u32(&o, 1);
    put_u64(&o, 32);
    put_u32(&o, GGUF_TYPE_F32);
    put_u64(&o, 0);
    while (o.n % 32 != 0) {
        put(&o, "", 1);
    }
    for (size_t i = 0; i < 128; i++) {
        put(&o, "", 1);
    }
    return o;
}

/* ---- The base description ---------------------------------------------- */

/* d_model 128, DeltaNet value width 128 (2 v-heads x 64, 1 k-head), attention
 * q_out 128, FFN 256; block 64. */
static const char *BASE_NAMES[] = {
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_out.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.1.attn_q.weight",
        "blk.1.attn_k.weight",
        "blk.1.attn_v.weight",
        "blk.1.attn_output.weight",
        "blk.1.ffn_gate.weight",
        "blk.1.ffn_up.weight",
        "blk.1.ffn_down.weight",
        "output.weight",
};
enum { N_BASE = sizeof BASE_NAMES / sizeof BASE_NAMES[0] };
static const char   *INV[]    = {"token_embd.weight"};
static const int32_t WIDTHS[] = {128, 256};
static int32_t       VALS[128 + 256];

static struct spec base(void) {
    for (size_t i = 0; i < sizeof VALS / sizeof VALS[0]; i++) {
        VALS[i] = (i * 7 + 3) % 5 < 2 ? -1 : 1;
    }
    return (struct spec) {
            .arch        = "qwen35",
            .has_version = true,
            .version     = 1,
            .transform   = "normalized-sylvester-walsh-hadamard",
            .axis        = "input-last-dimension",
            .sign_mode   = "explicit",
            .block       = 64,
            .names       = BASE_NAMES,
            .n_names     = N_BASE,
            .inv         = INV,
            .n_inv       = 1,
            .widths      = WIDTHS,
            .n_widths    = 2,
            .vals        = VALS,
            .n_vals      = 128 + 256,
            .grouped     = true,
    };
}

static struct geist_backend *g_be;

struct model {
    struct transformer_arch_state    st;
    struct transformer_layer_weights layers[2];
    struct buf                       file;
    int                              dummy_embed, dummy_out;
};

static void model_init(struct model *m, const struct spec *s) {
    *m                        = (struct model) {0};
    m->file                   = build(s);
    const char *err           = nullptr;
    m->st.gguf                = gguf_open_memory(m->file.b, m->file.n, &err);
    m->st.backend             = g_be;
    m->st.n_layers            = 2;
    m->st.layers              = m->layers;
    m->st.d_model             = 128;
    m->st.config.dn_n_k_heads = 1;
    m->st.config.dn_n_v_heads = 2;
    m->st.config.dn_head_v    = 64;
    m->st.embed_table.buffer  = (struct geist_buffer *) &m->dummy_embed;
    m->st.output_table.buffer = (struct geist_buffer *) &m->dummy_out;

    struct transformer_layer_weights *dn = &m->layers[0], *at = &m->layers[1];
    dn->mixer         = GEIST_MIXER_DELTANET;
    at->mixer         = GEIST_MIXER_ATTN;
    dn->dn_qkv_w.n_in = dn->dn_z_w.n_in = dn->dn_out_w.n_in = 128;
    at->q_proj_w.n_in = at->k_proj_w.n_in = at->v_proj_w.n_in = at->o_proj_w.n_in = 128;
    for (size_t l = 0; l < 2; l++) {
        m->layers[l].gate_proj_w.n_in = m->layers[l].up_proj_w.n_in = 128;
        m->layers[l].down_proj_w.n_in                               = 256;
    }
    if (m->st.gguf == nullptr) {
        fprintf(stderr, "test GGUF did not parse: %s\n", err != nullptr ? err : "?");
        exit(GEIST_TEST_ERROR);
    }
}

static void model_free(struct model *m) {
    transformer_rotation_release(&m->st);
    gguf_close(m->st.gguf);
    free(m->file.b);
}

static int g_fails = 0;

/* Load `s` into a fresh model (after `tweak`, if any) and check the status. */
static struct model *expect(const char        *what,
                            const struct spec *s,
                            enum geist_status  want,
                            void (*tweak)(struct model *)) {
    static struct model m;
    model_init(&m, s);
    if (tweak != nullptr) {
        tweak(&m);
    }
    const enum geist_status got = transformer_rotation_load(&m.st);
    const bool ok = got == want &&
                    (want == GEIST_OK || (!m.st.rotation.active && m.st.rotation.n_signs == 0));
    if (!ok) {
        fprintf(stderr,
                "FAIL: %s (status %d, want %d; %s)\n",
                what,
                (int) got,
                (int) want,
                geist_backend_errmsg(g_be));
        g_fails++;
    }
    return &m;
}

static void tw_tied(struct model *m) {
    m->st.output_table.buffer = m->st.embed_table.buffer;
}
static void tw_mtp(struct model *m) {
    m->st.n_mtp_layers = 1;
}
static void tw_bad_groups(struct model *m) {
    m->st.config.dn_n_v_heads = 3; /* 3 * 64 != ssm_out width 128 */
}
static struct geist_backend_descriptor g_no_slot_desc;
static void                            tw_no_slot(struct model *m) {
    (void) m;
    g_no_slot_desc                               = *g_be->desc;
    static const struct geist_backend_fused none = {0};
    g_no_slot_desc.fused                         = &none;
    g_be->desc                                   = &g_no_slot_desc;
}

int main(void) {
    GEIST_SKIP_IF(geist_backend_create("cpu_scalar", nullptr, nullptr, &g_be) != GEIST_OK,
                  "cpu_scalar backend not compiled in");

    /* ---- accepted ---- */
    struct spec   s = base();
    struct model *m = expect("full description", &s, GEIST_OK, nullptr);
    g_fails += geist_expect(m->st.rotation.active && m->st.rotation.block == 64 &&
                                    m->st.rotation.n_signs == 2 && m->st.rotation.embed_inverse &&
                                    m->st.rotation.gdn_v_grouped,
                            "full description: state filled");
    model_free(m);

    s           = base();
    s.sign_mode = "identity";
    s.widths    = nullptr;
    s.vals      = nullptr;
    m           = expect("identity signs", &s, GEIST_OK, nullptr);
    g_fails += geist_expect(m->st.rotation.active && m->st.rotation.n_signs == 0,
                            "identity signs: active, no sign buffers");
    model_free(m);

    s = (struct spec) {.arch = "qwen35"};
    m = expect("no prism keys", &s, GEIST_OK, nullptr);
    g_fails += geist_expect(!m->st.rotation.active, "no prism keys: inactive");
    model_free(m);

    /* ---- FORMAT: the folded set ---- */
    const char *names[N_BASE + 1];
    memcpy(names, BASE_NAMES, sizeof BASE_NAMES);

    s        = base();
    names[0] = "blk.0.ssm_alpha.weight";
    s.names  = names;
    model_free(expect("unfolded kind listed", &s, GEIST_E_FORMAT, nullptr));
    names[0] = BASE_NAMES[0];
    /* An attention kind listed ON TOP of the full DeltaNet set: nothing is
     * missing, so only the per-mixer check can refuse it. */
    names[N_BASE] = "blk.0.attn_q.weight";
    s.n_names     = N_BASE + 1;
    model_free(expect("kind on the wrong mixer", &s, GEIST_E_FORMAT, nullptr));
    s.n_names = N_BASE;
    names[0]  = "blk.2.attn_qkv.weight";
    model_free(expect("layer out of range", &s, GEIST_E_FORMAT, nullptr));
    names[0] = "blk.x.attn_qkv.weight";
    model_free(expect("malformed layer index", &s, GEIST_E_FORMAT, nullptr));
    names[0] = "blk.0.attn_gate.weight";
    model_free(expect("duplicate name", &s, GEIST_E_FORMAT, nullptr));
    names[0] = BASE_NAMES[0];

    s.n_names = N_BASE - 1; /* drop output.weight */
    model_free(expect("output not rotated", &s, GEIST_E_FORMAT, nullptr));
    memcpy(names, BASE_NAMES + 1, (N_BASE - 1) * sizeof names[0]); /* drop attn_qkv */
    model_free(expect("layer partially rotated", &s, GEIST_E_FORMAT, nullptr));
    memcpy(names, BASE_NAMES, sizeof BASE_NAMES);
    names[N_BASE] = "token_embd.weight";
    s.n_names     = N_BASE + 1;
    model_free(expect("embedding listed as folded", &s, GEIST_E_FORMAT, nullptr));

    s       = base();
    s.names = nullptr;
    model_free(expect("weight_names missing", &s, GEIST_E_FORMAT, nullptr));

    /* ---- FORMAT: block, signs, inverse, geometry ---- */
    s       = base();
    s.block = 48;
    model_free(expect("block not a power of two", &s, GEIST_E_FORMAT, nullptr));
    s.block = 512;
    model_free(expect("block does not divide the widths", &s, GEIST_E_FORMAT, nullptr));
    s.block = 0;
    model_free(expect("block missing", &s, GEIST_E_FORMAT, nullptr));

    s        = base();
    s.n_vals = 128 + 255;
    model_free(expect("sign_values short", &s, GEIST_E_FORMAT, nullptr));
    s.n_vals = 128 + 256;
    int32_t long_vals[128 + 256 + 1];
    memcpy(long_vals, VALS, sizeof VALS);
    long_vals[128 + 256] = 1;
    s.vals               = long_vals;
    s.n_vals             = 128 + 256 + 1;
    model_free(expect("sign_values long", &s, GEIST_E_FORMAT, nullptr));
    s        = base();
    VALS[17] = 2;
    model_free(expect("sign value not +/-1", &s, GEIST_E_FORMAT, nullptr));
    s          = base();
    s.n_widths = 1;
    s.n_vals   = 128;
    model_free(expect("no sign vector for the FFN width", &s, GEIST_E_FORMAT, nullptr));
    const int32_t dup_widths[] = {128, 128};
    s                          = base();
    s.widths                   = dup_widths;
    s.n_vals                   = 256;
    model_free(expect("duplicate sign width", &s, GEIST_E_FORMAT, nullptr));

    s                     = base();
    const char *bad_inv[] = {"output.weight"};
    s.inv                 = bad_inv;
    model_free(expect("inverse names wrong", &s, GEIST_E_FORMAT, nullptr));

    s = base();
    model_free(expect("bad grouped-value geometry", &s, GEIST_E_FORMAT, tw_bad_groups));

    /* ---- UNSUPPORTED ---- */
    s         = base();
    s.version = 2;
    model_free(expect("version 2", &s, GEIST_E_UNSUPPORTED, nullptr));
    s           = base();
    s.transform = "hadamard-v2";
    model_free(expect("other transform", &s, GEIST_E_UNSUPPORTED, nullptr));
    s      = base();
    s.axis = "output-first-dimension";
    model_free(expect("other axis", &s, GEIST_E_UNSUPPORTED, nullptr));
    s           = base();
    s.sign_mode = "random";
    model_free(expect("other sign mode", &s, GEIST_E_UNSUPPORTED, nullptr));
    s      = base();
    s.arch = "llama";
    model_free(expect("other family", &s, GEIST_E_UNSUPPORTED, nullptr));
    s = base();
    model_free(expect("MTP layers", &s, GEIST_E_UNSUPPORTED, tw_mtp));
    model_free(expect("tied lm_head", &s, GEIST_E_UNSUPPORTED, tw_tied));
    const struct geist_backend_descriptor *real = g_be->desc;
    model_free(expect("backend without the slot", &s, GEIST_E_UNSUPPORTED, tw_no_slot));
    g_be->desc = real;

    geist_backend_destroy(g_be);
    if (g_fails == 0) {
        printf("PASS test_rotation_load_unit\n");
    }
    return g_fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
