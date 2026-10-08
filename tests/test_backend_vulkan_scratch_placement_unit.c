/*
 * test_backend_vulkan_scratch_placement_unit — where a session's scratch
 * lives on Vulkan, and that a device-local pool changes nothing but that
 * (#488).
 *
 * The transformer arch slices one scratch pool into ~20 activation slots.
 * Under GEIST_VK_SCRATCH_DEVICE=0 it is host-visible: mapped, then sliced with
 * buffer_create_aliased(host pointer). Without resizable BAR a large pool
 * spills into system memory and every GPU op on it crosses the bus. By
 * default (or =1) a session whose host paths never map those
 * slots puts them in device-local memory, sliced with buffer_create_view
 * (offset, no host pointer); h_a, h_b and logits — which the host reads —
 * stay in a small host-visible pool. Checked here on the in-memory llama:
 *
 *   1. placement: opted out, every slot is host-visible (no device pool);
 *      otherwise every slot but h_a / h_b / logits is device-local and
 *      unmappable, and those three are mappable. The backend's scratch
 *      counters move with it.
 *   2. the all-ones slot (uploaded through a view, not mapped) holds ones;
 *   3. greedy decoding through the device-local pool yields the same
 *      tokens and the same logits as through the host-visible one, for the
 *      default chunk and for a small one (prefill in several chunks);
 *   4. a CPU fallback that would need a device-local slot fails loudly:
 *      the arch's host linear returns GEIST_E_BACKEND naming the reason and
 *      leaves the slot untouched; a Vulkan op whose GPU path does not apply
 *      returns an error and counts the refused host view (stat_host_denied)
 *      instead of reading the bytes, while the same op on mappable memory
 *      runs on the host.
 *
 * On lavapipe every memory type is device-local and host-visible, so the
 * spill itself cannot happen here; what is checked is the placement
 * decision and correctness. SKIPs (exit 77) when no Vulkan device is
 * present; exit 99 when a fixture cannot be set up.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "test_helpers.h"
#include "model_fixtures.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(GEIST_BACKEND_VULKAN) && GEIST_BACKEND_VULKAN
#include "src/archs/transformer/arch_state.h"
#include "src/archs/transformer/forward/internal.h"
#include "src/backends/vulkan/vk_internal.h"
#include "src/engine/model.h"

constexpr size_t VOCAB  = 512;
constexpr size_t PROMPT = 70;
constexpr size_t DECODE = 8;

/* One backend and the fixture model loaded on it. */
struct rig {
    struct geist_backend *be;
    struct geist_model   *m;
};

/* A Vulkan backend created with GEIST_VK_SCRATCH_DEVICE set or not (read
 * once, at create), with the fixture llama on it. 0 ok, 77 no device,
 * 99 setup failure. */
static int rig_open(bool device, const struct tf_buf *g, struct rig *r) {
    *r = (struct rig) {0};
    setenv("GEIST_VK_SCRATCH_DEVICE", device ? "1" : "0", 1);
    const enum geist_status s = geist_backend_create("vulkan", nullptr, nullptr, &r->be);
    unsetenv("GEIST_VK_SCRATCH_DEVICE");
    if (s != GEIST_OK || r->be == nullptr) {
        fprintf(stdout, "SKIP: vulkan backend unavailable (%d)\n", (int) s);
        return GEIST_TEST_SKIP;
    }
    if (geist_model_load_from_memory(g->b, g->n, r->be, &r->m) != GEIST_OK) {
        fprintf(stderr, "ERROR: model load: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }
    return 0;
}

static void rig_close(struct rig *r) {
    geist_model_destroy(r->m);
    geist_backend_destroy(r->be);
    *r = (struct rig) {0};
}

/* The slots that stay host-visible with a device-local pool. */
static bool host_slot(const struct transformer_arch_session *sess, const struct geist_buffer *b) {
    return b == sess->scratch_h_a || b == sess->scratch_h_b || b == sess->scratch_logits;
}

/* ---- 4: forced CPU fallbacks --------------------------------------------- */

/* The arch's host linear (what runs when the backend has no device kernel
 * for a weight) on two device-local slots: an error naming the reason, and
 * the output slot still as zeroed at creation. */
static int check_host_linear_refused(struct rig *r, struct transformer_arch_session *sess) {
    struct transformer_arch_state   *st = sess->model;
    const struct geist_backend_vtbl *vt = r->be->desc->vtbl;
    const struct geist_weight       *w  = &st->layers[0].q_proj_w;
    int                              fails =
            geist_expect(w->linear_m1 != nullptr, "host linear: the resolver installed a kernel");
    if (w->linear_m1 == nullptr) {
        return fails;
    }
    const enum geist_status s = linear_w_or_legacy(
            r->be, vt, sess->scratch_normed, sess->scratch_q, w, 1, nullptr, nullptr, nullptr);
    const char *e = geist_backend_errmsg(r->be);
    char        msg[256];
    snprintf(msg,
             sizeof msg,
             "host linear on device-local slots fails loudly (status %d, \"%s\")",
             (int) s,
             e != nullptr ? e : "");
    fails += geist_expect(s == GEIST_E_BACKEND && e != nullptr && strstr(e, "device-local"), msg);

    const size_t n = (size_t) w->n_out;
    float       *y = malloc(n * sizeof(float));
    if (y == nullptr) {
        return fails + 1;
    }
    bool zero = vt->buffer_download(n * sizeof(float), (uint8_t *) y, sess->scratch_q) == GEIST_OK;
    for (size_t i = 0; zero && i < n; i++) {
        zero = y[i] == 0.0f;
    }
    free(y);
    return fails + geist_expect(zero, "host linear: the output slot was not written");
}

/* An F32 [n] view of `b`. */
static struct geist_tensor vec(struct geist_buffer *b, size_t n) {
    return (struct geist_tensor) {.buffer = b,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 1,
                                  .shape  = {(int64_t) n},
                                  .stride = {1}};
}

/* prims->add with one operand that has no VkBuffer behind it (a host alias):
 * the GPU path does not apply and the op falls back to the host. With the
 * other operands device-local that fallback must be refused and counted;
 * with them host-visible it runs. */
static int check_op_fallback(struct rig *r, struct transformer_arch_session *sess) {
    struct vk_state                 *vs = r->be->state;
    const struct geist_backend_vtbl *vt = r->be->desc->vtbl;
    const size_t                     n  = sess->model->d_model;
    float                           *a  = malloc(n * sizeof(float));
    struct geist_buffer             *ab = nullptr;
    if (a == nullptr ||
        vt->buffer_create_aliased(r->be, a, n * sizeof(float), GEIST_BUFFER_SCRATCH, &ab) !=
                GEIST_OK) {
        free(a);
        return 1;
    }
    for (size_t i = 0; i < n; i++) {
        a[i] = 1.5f;
    }
    struct geist_tensor ta = vec(ab, n);

    struct geist_tensor     tb     = vec(sess->scratch_q, n);
    struct geist_tensor     ty     = vec(sess->scratch_attn, n);
    const uint64_t          denied = vs->stat_host_denied;
    const enum geist_status s1     = r->be->desc->prims->add(r->be, &ta, &tb, &ty);
    char                    msg[192];
    snprintf(msg,
             sizeof msg,
             "add falling back to the host on device-local slots is refused (status %d, "
             "refusals %llu -> %llu)",
             (int) s1,
             (unsigned long long) denied,
             (unsigned long long) vs->stat_host_denied);
    int fails = geist_expect(s1 != GEIST_OK && vs->stat_host_denied > denied, msg);

    /* Control: the same fallback over the host-visible slots runs. */
    tb                         = vec(sess->scratch_h_a, n);
    ty                         = vec(sess->scratch_h_b, n);
    const enum geist_status s2 = r->be->desc->prims->add(r->be, &ta, &tb, &ty);
    const float            *y  = vt->buffer_map(sess->scratch_h_b);
    bool                    ok = s2 == GEIST_OK && y != nullptr;
    for (size_t i = 0; ok && i < n; i++) {
        ok = y[i] == 1.5f;
    }
    if (y != nullptr) {
        vt->buffer_unmap(sess->scratch_h_b);
    }
    fails += geist_expect(ok, "the same add over host-visible slots runs on the host");
    vt->buffer_destroy(r->be, ab);
    free(a);
    return fails;
}

/* ---- 1 + 2: placement ---------------------------------------------------- */

static int check_placement(struct rig *r, bool device) {
    struct transformer_arch_state   *st   = geist_model_internal_arch_meta(r->m);
    struct vk_state                 *vs   = r->be->state;
    const struct geist_backend_vtbl *vt   = r->be->desc->vtbl;
    const uint64_t                   dev0 = vs->stat_scratch_n[VK_PLACEMENT_DEVICE];
    const struct geist_session_opts  o    = {.top_p = 1.0f};
    struct transformer_arch_session *sess = transformer_session_alloc(st, &o);
    const char                      *tag  = device ? "opted in" : "default";
    char                             msg[192];
    snprintf(msg, sizeof msg, "%s: session", tag);
    int fails = geist_expect(sess != nullptr, msg);
    if (sess == nullptr) {
        return fails;
    }
    snprintf(msg, sizeof msg, "%s: device-local pool %s", tag, device ? "present" : "absent");
    fails += geist_expect(sess->scratch_device == device &&
                                  (sess->scratch_dev_pool_buf != nullptr) == device,
                          msg);
    if (device) {
        snprintf(msg, sizeof msg, "%s: the pool is counted device-local", tag);
        fails += geist_expect(vs->stat_scratch_n[VK_PLACEMENT_DEVICE] > dev0 &&
                                      vk_buffer_placement(sess->scratch_dev_pool_buf) ==
                                              VK_PLACEMENT_DEVICE,
                              msg);
    }

    const struct {
        const char          *name;
        struct geist_buffer *b;
    } slots[] = {
            {"normed", sess->scratch_normed},
            {"q", sess->scratch_q},
            {"k", sess->scratch_k},
            {"v", sess->scratch_v},
            {"attn", sess->scratch_attn},
            {"o", sess->scratch_o},
            {"post_attn", sess->scratch_post_attn},
            {"h_post_attn", sess->scratch_h_post_attn},
            {"pre_ff", sess->scratch_pre_ff},
            {"gate", sess->scratch_gate},
            {"up", sess->scratch_up},
            {"ffn_out", sess->scratch_ffn_out},
            {"post_ff", sess->scratch_post_ff},
            {"h_post_ff", sess->scratch_h_post_ff},
            {"ones", sess->scratch_ones_headdim_max},
            {"h_a", sess->scratch_h_a},
            {"h_b", sess->scratch_h_b},
            {"logits", sess->scratch_logits},
    };
    for (size_t i = 0; i < sizeof slots / sizeof slots[0]; i++) {
        struct geist_buffer    *b      = slots[i].b;
        const bool              on_dev = device && !host_slot(sess, b);
        const enum vk_placement where  = vk_buffer_placement(b);
        void                   *p      = b != nullptr ? vt->buffer_map(b) : nullptr;
        if (p != nullptr) {
            vt->buffer_unmap(b);
        }
        snprintf(msg,
                 sizeof msg,
                 "%s: slot %s is %s (placement %d, mapped %d)",
                 tag,
                 slots[i].name,
                 on_dev ? "device-local and unmappable" : "host-visible and mappable",
                 (int) where,
                 p != nullptr);
        fails += geist_expect(b != nullptr &&
                                      (on_dev ? where == VK_PLACEMENT_DEVICE && p == nullptr
                                              : where != VK_PLACEMENT_DEVICE && p != nullptr),
                              msg);
    }

    /* 2: the all-ones slot reads back as ones, mapped or staged. */
    float ones[TRANSFORMER_HEAD_DIM_MAX];
    memset(ones, 0, sizeof ones);
    const bool got =
            vt->buffer_download(sizeof ones, (uint8_t *) ones, sess->scratch_ones_headdim_max) ==
            GEIST_OK;
    bool all = got;
    for (size_t i = 0; all && i < TRANSFORMER_HEAD_DIM_MAX; i++) {
        all = ones[i] == 1.0f;
    }
    snprintf(msg, sizeof msg, "%s: the all-ones slot holds ones", tag);
    fails += geist_expect(all, msg);

    if (device) {
        fails += check_host_linear_refused(r, sess);
        fails += check_op_fallback(r, sess);
    }
    transformer_session_free(st, sess);
    return fails;
}

/* ---- 3: decoding parity --------------------------------------------------- */

/* Prefill PROMPT tokens and greedily decode DECODE more on a fresh session
 * of `m_max` (0: the default); the tokens and the final logits. */
static enum geist_status decode_run(struct rig   *r,
                                    size_t        m_max,
                                    geist_token_t out[static DECODE],
                                    float         logits[static VOCAB]) {
    const struct geist_session_opts o = {.top_p = 1.0f, .m_max = m_max};
    struct geist_session           *s = nullptr;
    geist_token_t                   prompt[PROMPT];
    for (size_t i = 0; i < PROMPT; i++) {
        prompt[i] = (geist_token_t) (2 + (i * 97 + i / 7) % (VOCAB - 2));
    }
    enum geist_status st = geist_session_create(r->m, r->be, &o, &s);
    if (st == GEIST_OK) {
        st = geist_session_prefill_tokens(s, PROMPT, prompt);
    }
    for (size_t i = 0; st == GEIST_OK && i < DECODE; i++) {
        st = geist_session_decode_step(s, &out[i]);
    }
    if (st == GEIST_OK) {
        size_t       n = 0;
        const float *l = geist_session_peek_logits(&n, s);
        if (l == nullptr || n != VOCAB) {
            st = GEIST_E_BACKEND;
        } else {
            memcpy(logits, l, VOCAB * sizeof(float));
        }
    }
    geist_session_destroy(s);
    return st;
}

static int check_parity(struct rig *host, struct rig *dev, size_t m_max) {
    geist_token_t           th[DECODE], td[DECODE];
    static float            lh[VOCAB], ld[VOCAB];
    struct vk_state        *vs     = dev->be->state;
    const uint64_t          dev0   = vs->stat_scratch_n[VK_PLACEMENT_DEVICE];
    const uint64_t          denied = vs->stat_host_denied;
    const enum geist_status sh     = decode_run(host, m_max, th, lh);
    const enum geist_status sd     = decode_run(dev, m_max, td, ld);
    char                    msg[192];
    snprintf(msg,
             sizeof msg,
             "m_max %zu: both sessions prefill and decode (host-visible %d, device-local %d)",
             m_max,
             (int) sh,
             (int) sd);
    int fails = geist_expect(sh == GEIST_OK && sd == GEIST_OK, msg);
    if (fails != 0) {
        return fails;
    }
    snprintf(msg, sizeof msg, "m_max %zu: the decoding session had a device-local pool", m_max);
    fails += geist_expect(vs->stat_scratch_n[VK_PLACEMENT_DEVICE] > dev0, msg);
    snprintf(msg, sizeof msg, "m_max %zu: no host view of device memory was needed", m_max);
    fails += geist_expect(vs->stat_host_denied == denied, msg);
    snprintf(msg, sizeof msg, "m_max %zu: the same %zu greedy tokens", m_max, DECODE);
    fails += geist_expect(memcmp(th, td, sizeof th) == 0, msg);
    size_t diff = 0;
    for (size_t i = 0; i < VOCAB; i++) {
        diff += lh[i] != ld[i] || !isfinite(lh[i]);
    }
    snprintf(msg, sizeof msg, "m_max %zu: identical finite logits (%zu differ)", m_max, diff);
    return fails + geist_expect(diff == 0, msg);
}

int main(void) {
    struct tf_buf g    = mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                                           .d_model  = 256,
                                                           .heads    = 4,
                                                           .kv_heads = 2,
                                                           .ffn      = 512,
                                                           .vocab    = VOCAB,
                                                           .context  = 256,
                                                           .seed     = 23});
    struct rig    host = {0}, dev = {0};
    int           rc = rig_open(false, &g, &host);
    if (rc == 0) {
        rc = rig_open(true, &g, &dev);
    }
    if (rc != 0) {
        rig_close(&dev);
        rig_close(&host);
        free(g.b);
        return rc;
    }
    int fails = check_placement(&host, false);
    fails += check_placement(&dev, true);
    fails += check_parity(&host, &dev, 0);
    fails += check_parity(&host, &dev, 16);
    rig_close(&dev);
    rig_close(&host);
    free(g.b);
    if (fails != 0) {
        return GEIST_TEST_FAIL;
    }
    printf("PASS: scratch placed device-local unless opted out, decoding unchanged, CPU "
           "fallbacks on device-local scratch refused\n");
    return GEIST_TEST_PASS;
}

#else
int main(void) {
    printf("SKIP: built without the Vulkan backend\n");
    return GEIST_TEST_SKIP;
}
#endif
