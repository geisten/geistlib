/*
 * test_backend_vulkan_vram_budget_unit — device memory budget and the
 * weight spill (#466).
 *
 * The Vulkan path keeps the KV cache and the x ring in device memory;
 * weights go there too until one would leave less than a reserve, then to
 * host memory (read over the bus). Device-local allocations are checked
 * against the heap (lowered by GEIST_VK_VRAM_BUDGET) before they are made,
 * so what does not fit fails with a named error, not a bare driver status.
 * Checked here:
 *
 *   - GEIST_VK_VRAM_BUDGET parses bytes and K/M/G suffixes; a malformed or
 *     overflowing value is ignored;
 *   - a device-local buffer past the budget is GEIST_E_OOM with an error
 *     naming the MiB it needs, the MiB in use and the limit; freeing a
 *     buffer returns its bytes; host-visible buffers are not counted;
 *   - a model whose weights do not fit loads with them in host memory and
 *     decodes the same tokens as with them in VRAM; a KV cache that does
 *     not fit fails session creation with that error; with room the model
 *     loads without spilling and decodes.
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

constexpr size_t MIB = (size_t) 1 << 20;
/* Session rows: the KV cache is sized from the session's max_seq_len, and
 * 4096 rows of F16 KV (8 MiB here) are enough to tell the budgets apart. */
constexpr size_t CTX = 4096;

/* A Vulkan backend created under GEIST_VK_VRAM_BUDGET=v and
 * GEIST_VK_WEIGHT_RESERVE=reserve (nullptr: unset). */
static struct geist_backend *backend_reserve(const char *v, const char *reserve) {
    if (v != nullptr) {
        setenv("GEIST_VK_VRAM_BUDGET", v, 1);
    } else {
        unsetenv("GEIST_VK_VRAM_BUDGET");
    }
    if (reserve != nullptr) {
        setenv("GEIST_VK_WEIGHT_RESERVE", reserve, 1);
    } else {
        unsetenv("GEIST_VK_WEIGHT_RESERVE");
    }
    struct geist_backend *be = nullptr;
    if (geist_backend_create("vulkan", nullptr, nullptr, &be) != GEIST_OK) {
        be = nullptr;
    }
    unsetenv("GEIST_VK_VRAM_BUDGET");
    unsetenv("GEIST_VK_WEIGHT_RESERVE");
    return be;
}

static struct geist_backend *backend_with(const char *v) {
    return backend_reserve(v, nullptr);
}

static int check_parse(void) {
    static const struct {
        const char *v;
        size_t      bytes;
    } cases[] = {
            {"4096", 4096},
            {"64K", (size_t) 64 << 10},
            {"3m", 3 * MIB},
            {"2G", (size_t) 2 << 30},
            {"", 0},
            {"abc", 0},
            {"12X", 0},
            {"5MB", 0},
            {"-1", 0},
            {"99999999999999999999", 0},
            {"18446744073709551615K", 0},
    };
    int fails = 0;
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        struct geist_backend *be = backend_with(cases[i].v);
        if (be == nullptr) {
            return fails + geist_expect(false, "backend for the budget parse");
        }
        const size_t got = ((struct vk_state *) be->state)->vram_budget;
        char         msg[128];
        snprintf(msg,
                 sizeof msg,
                 "GEIST_VK_VRAM_BUDGET=\"%s\" -> %zu (got %zu)",
                 cases[i].v,
                 cases[i].bytes,
                 got);
        fails += geist_expect(got == cases[i].bytes, msg);
        geist_backend_destroy(be);
    }
    return fails;
}

static int check_buffers(void) {
    struct geist_backend *be = backend_with("4M");
    if (be == nullptr) {
        return geist_expect(false, "backend with a 4 MiB budget");
    }
    const struct geist_backend_vtbl *vt    = be->desc->vtbl;
    struct vk_state                 *st    = be->state;
    int                              fails = 0;
    struct geist_buffer             *a = nullptr, *b = nullptr, *h = nullptr;

    fails += geist_expect(
            vt->buffer_create(be, 3 * MIB, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE, &a) ==
                            GEIST_OK &&
                    st->vram_used >= 3 * MIB,
            "3 MiB of a 4 MiB budget");
    const enum geist_status s =
            vt->buffer_create(be, 3 * MIB, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE, &b);
    const char *err = geist_backend_errmsg(be);
    char        msg[320];
    snprintf(msg, sizeof msg, "3 MiB more is GEIST_E_OOM (%d): %s", (int) s, err);
    fails += geist_expect(s == GEIST_E_OOM && b == nullptr, msg);
    fails += geist_expect(err != nullptr && strstr(err, "out of device memory") != nullptr &&
                                  strstr(err, "needs 3 MiB") != nullptr &&
                                  strstr(err, "3 of 4 MiB") != nullptr &&
                                  strstr(err, "GEIST_VK_VRAM_BUDGET") != nullptr,
                          "the error names needed, in-use and limit");
    fails += geist_expect(st->vram_used >= 3 * MIB && st->vram_used < 4 * MIB,
                          "a refused buffer takes nothing");

    /* KV-cache role buffers are device-local by default. */
    fails += geist_expect(vt->buffer_create(be, 2 * MIB, GEIST_BUFFER_KV_CACHE, 0, &b) ==
                                          GEIST_E_OOM &&
                                  strstr(geist_backend_errmsg(be), "KV-cache") != nullptr,
                          "a KV-cache buffer past the budget names the KV cache");

    /* Host-visible memory is not device memory. */
    fails += geist_expect(vt->buffer_create(be, 8 * MIB, GEIST_BUFFER_SCRATCH, 0, &h) == GEIST_OK &&
                                  st->vram_used < 4 * MIB,
                          "a host-visible buffer past the budget is not counted");

    vt->buffer_destroy(be, a);
    fails += geist_expect(st->vram_used == 0, "freeing returns the bytes");
    fails += geist_expect(
            vt->buffer_create(be, 3 * MIB, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE, &b) ==
                    GEIST_OK,
            "3 MiB fit again after the free");
    if (b != nullptr) {
        vt->buffer_destroy(be, b);
    }
    if (h != nullptr) {
        vt->buffer_destroy(be, h);
    }
    fails += geist_expect(st->vram_used == 0, "nothing left in use");
    geist_backend_destroy(be);
    return fails;
}

/* `budget` nullptr: no limit beyond the heap. Returns failures; `expect`
 * is what the load + session + decode should end in. */
enum outcome { RUNS, SESSION_FAILS };

/* Device bytes after the load and after the session (for sizing budgets),
 * weights spilled to host memory, and the decoded tokens. */
struct usage {
    size_t        load, session, spilled;
    geist_token_t tok[4];
};

static int run_model(const struct tf_buf *g,
                     const char          *budget,
                     const char          *reserve,
                     size_t               ctx,
                     enum outcome         expect,
                     struct usage        *out) {
    struct geist_backend *be = backend_reserve(budget, reserve);
    if (be == nullptr) {
        return geist_expect(false, "backend for a model run");
    }
    int                 fails = 0;
    char                msg[384];
    struct geist_model *m  = nullptr;
    enum geist_status   ls = geist_model_load_from_memory(g->b, g->n, be, &m);
    if (ls != GEIST_OK) {
        snprintf(msg, sizeof msg, "budget %s: load (%s)", budget, geist_last_create_error());
        fails += geist_expect(false, msg);
    } else {
        struct vk_state          *st     = be->state;
        const size_t              loaded = st->vram_used;
        struct geist_session_opts o  = {.kv_mode = GEIST_KV_F16, .top_p = 1.0f, .max_seq_len = ctx};
        struct geist_session     *s  = nullptr;
        enum geist_status         ss = geist_session_create(m, be, &o, &s);
        if (expect == SESSION_FAILS) {
            const char *err = geist_backend_errmsg(be);
            snprintf(msg,
                     sizeof msg,
                     "budget %s ctx %zu: session creation fails on the KV cache (%d: %s)",
                     budget,
                     ctx,
                     (int) ss,
                     err);
            fails += geist_expect(ss == GEIST_E_OOM && s == nullptr && err != nullptr &&
                                          strstr(err, "out of device memory") != nullptr,
                                  msg);
        } else {
            static const geist_token_t prompt[] = {1, 5, 9, 13};
            geist_token_t              tok[4]   = {-1, -1, -1, -1};
            bool ok = ss == GEIST_OK && geist_session_prefill_tokens(s, 4, prompt) == GEIST_OK;
            for (int i = 0; ok && i < 4; i++) {
                ok = geist_session_decode_step(s, &tok[i]) == GEIST_OK;
            }
            snprintf(msg,
                     sizeof msg,
                     "budget %s ctx %zu: loads and decodes (%s)",
                     budget != nullptr ? budget : "none",
                     ctx,
                     geist_backend_errmsg(be));
            fails += geist_expect(ok, msg);
            if (out != nullptr) {
                *out = (struct usage) {
                        .load = loaded, .session = st->vram_used, .spilled = st->spilled_weights};
                memcpy(out->tok, tok, sizeof tok);
            }
        }
        if (s != nullptr) {
            geist_session_destroy(s);
        }
        /* Weight copies live in the backend's registry until it is
         * destroyed; the session's device memory goes with the session. */
        fails += geist_expect(st->vram_used == loaded, "session teardown returns its device bytes");
        geist_model_destroy(m);
    }
    geist_backend_destroy(be);
    return fails;
}

int main(void) {
    unsetenv("GEIST_VK_VRAM_BUDGET");
    struct geist_backend *probe = nullptr;
    enum geist_status     s     = geist_backend_create("vulkan", nullptr, nullptr, &probe);
    if (s != GEIST_OK) {
        fprintf(stderr, "SKIP: vulkan backend unavailable (%d)\n", (int) s);
        return GEIST_TEST_SKIP;
    }
    geist_backend_destroy(probe);

    int fails = check_parse();
    fails += check_buffers();

    /* d_model 512 x ffn 1024 F32: 2 MiB FFN matrices, above the 1 MiB
     * device-copy threshold, so the weights go to device memory at load. */
    struct tf_buf g   = mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                                          .d_model  = 512,
                                                          .heads    = 8,
                                                          .kv_heads = 4,
                                                          .ffn      = 1024,
                                                          .vocab    = 256,
                                                          .context  = 4096,
                                                          .seed     = 3});
    struct usage  use = {0};
    fails += run_model(&g, nullptr, nullptr, CTX, RUNS, &use);
    char msg[160];
    snprintf(msg,
             sizeof msg,
             "weights (%zu B) and the KV cache (%zu B) are in device memory",
             use.load,
             use.session - use.load);
    fails += geist_expect(
            use.load >= 4 * MIB && use.session - use.load >= 4 * MIB && use.spilled == 0, msg);
    /* A budget of exactly the load, default reserve (1/16 of it, more than
     * the weights): the weights spill to host memory, and a short session
     * (32 KiB of KV) decodes the same tokens as with them in VRAM. */
    char budget[32];
    snprintf(budget, sizeof budget, "%zuK", use.load >> 10);
    struct usage spill = {0};
    fails += run_model(&g, budget, nullptr, 16, RUNS, &spill);
    snprintf(msg, sizeof msg, "%zu weights spilled, same tokens as in VRAM", spill.spilled);
    fails +=
            geist_expect(spill.spilled > 0 && memcmp(spill.tok, use.tok, sizeof use.tok) == 0, msg);
    /* No reserve, so the weights fill the budget before they spill. Room
     * for the weights and half the KV cache: the load works, the session
     * does not. Room for both: it runs, nothing spilled. */
    snprintf(budget, sizeof budget, "%zuK", (use.load + (use.session - use.load) / 2) >> 10);
    fails += run_model(&g, budget, "0", CTX, SESSION_FAILS, nullptr);
    snprintf(budget, sizeof budget, "%zuK", (use.session + MIB) >> 10);
    struct usage room = {0};
    fails += run_model(&g, budget, "0", CTX, RUNS, &room);
    fails += geist_expect(room.spilled == 0, "with room nothing spills");
    free(g.b);

    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("vulkan device-memory budget: pass\n");
    return GEIST_TEST_PASS;
}

#else
int main(void) {
    fprintf(stderr, "SKIP: vulkan backend not built\n");
    return GEIST_TEST_SKIP;
}
#endif
