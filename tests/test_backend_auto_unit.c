/*
 * test_backend_auto_unit — what the backend name "auto" picks, and the CPU
 * fallback of a model that the GPU it picked refuses.
 *
 *   1. "auto", "" and nullptr pick the same backend: metal on Apple silicon
 *      when it is built in and finds a device, else the first registry
 *      entry (the preferred CPU backend). GEIST_BACKEND=<name> replaces it
 *      with that explicit name.
 *   2. Weight formats: a llama whose ffn_down is Q5_0, which metal has no
 *      kernel for. Explicit "metal" refuses it at load with
 *      GEIST_E_UNSUPPORTED, naming the tensor; "auto" loads it on the CPU
 *      backend, geist_model_backend_name says so, and a session created
 *      with the "auto" handle decodes. Both load paths (file, memory).
 *   3. Memory budget at load: with metal's working set squeezed, explicit
 *      "metal" refuses with GEIST_E_OOM, "auto" loads on the CPU.
 *   4. Memory budget at session create: a budget that holds the weights
 *      but not a session of the load's max_seq_len. Explicit "metal" loads
 *      and then refuses the session; "auto" loads on the CPU, where the
 *      session fits.
 *
 * The budget is forced small by writing the backend state directly, as in
 * test_metal_budget_unit. Parts 2-4 run only where "auto" picks metal.
 */
#include "test_helpers.h"
#include "model_fixtures.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef GEIST_BACKEND_METAL
#include "../src/backends/metal/metal_internal.h"
#endif

extern const struct geist_backend_descriptor *const geist_backend_registry[];

static int g_fail = 0;
#define check(ok, what) (g_fail |= geist_expect((ok), (what)))

static struct geist_backend *create(const char *name) {
    struct geist_backend *be = nullptr;
    return geist_backend_create(name, nullptr, nullptr, &be) == GEIST_OK ? be : nullptr;
}

static bool is(const char *a, const char *b) {
    return a != nullptr && b != nullptr && strcmp(a, b) == 0;
}

/* 1. The choice. Returns whether "auto" picked metal. */
static bool check_choice(void) {
    unsetenv("GEIST_BACKEND");
    struct geist_backend *metal = create("metal");
#if defined(__APPLE__) && defined(__aarch64__)
    const bool want_gpu = metal != nullptr;
#else
    const bool want_gpu = false;
#endif
    const char           *want = want_gpu ? "metal" : geist_backend_registry[0]->name;
    struct geist_backend *a    = create("auto");
    struct geist_backend *e    = create("");
    struct geist_backend *n    = create(nullptr);
    check(a != nullptr && is(geist_backend_name(a), want), "\"auto\" picks the preferred backend");
    check(e != nullptr && is(geist_backend_name(e), want), "\"\" is \"auto\"");
    check(n != nullptr && is(geist_backend_name(n), want), "nullptr is \"auto\"");
    check(a != nullptr && a->auto_cpu == want_gpu, "only a GPU picked by \"auto\" falls back");
    check(metal == nullptr || !metal->auto_cpu, "explicit \"metal\" never falls back");
    geist_backend_destroy(a);
    geist_backend_destroy(e);
    geist_backend_destroy(n);
    geist_backend_destroy(metal);

    /* GEIST_BACKEND names the backend explicitly. */
    const char *cpu = geist_backend_registry[0]->name;
    setenv("GEIST_BACKEND", cpu, 1);
    struct geist_backend *env = create("auto");
    check(env != nullptr && is(geist_backend_name(env), cpu) && !env->auto_cpu,
          "GEIST_BACKEND=<cpu> overrides \"auto\" and does not fall back");
    geist_backend_destroy(env);
    setenv("GEIST_BACKEND", "auto", 1);
    env = create("auto");
    check(env != nullptr && is(geist_backend_name(env), want), "GEIST_BACKEND=auto is \"auto\"");
    geist_backend_destroy(env);
    unsetenv("GEIST_BACKEND");
    return want_gpu;
}

static struct tf_buf llama(bool q5_0_down) {
    return mf_llama_gguf(&(struct mf_llama) {.layers    = 2,
                                             .d_model   = 128,
                                             .heads     = 4,
                                             .kv_heads  = 2,
                                             .ffn       = 256,
                                             .vocab     = 64,
                                             .context   = 256,
                                             .seed      = 7,
                                             .q5_0_down = q5_0_down});
}

/* A session on `m`, created with the caller's handle `be`, prefills and
 * decodes. */
static bool decodes(struct geist_model *m, struct geist_backend *be) {
    const struct geist_session_opts so     = {.max_seq_len = 256};
    struct geist_session           *s      = nullptr;
    const geist_token_t             ids[4] = {1, 2, 3, 4};
    geist_token_t                   t      = 0;
    const bool                      ok     = geist_session_create(m, be, &so, &s) == GEIST_OK &&
                                             geist_session_prefill_tokens(s, 4, ids) == GEIST_OK &&
                                             geist_session_decode_step(s, &t) == GEIST_OK;
    geist_session_destroy(s);
    return ok;
}

#ifdef GEIST_BACKEND_METAL
static bool load_opts(struct tf_buf                    g,
                      struct geist_backend            *be,
                      const struct geist_session_opts *so,
                      struct geist_model             **m) {
    return geist_model_load_from_memory_with_opts(g.b, g.n, be, so, m) == GEIST_OK;
}

static size_t allocated(struct geist_backend *be) {
    struct metal_state *st = be->state;
    return metal_msg_send_ulong0(st, st->device, "currentAllocatedSize");
}

static void set_budget(struct geist_backend *be, size_t budget) {
    ((struct metal_state *) be->state)->ws_budget = budget;
}
#endif

/* 2. A weight format metal has no kernel for. */
static void check_unsupported(const char *path) {
    struct tf_buf         g     = llama(true);
    struct geist_backend *metal = create("metal");
    struct geist_backend *a     = create("auto");
    const char           *cpu   = geist_backend_registry[0]->name;
    struct geist_model   *m     = nullptr;

    check(metal != nullptr &&
                  geist_model_load_from_memory(g.b, g.n, metal, &m) == GEIST_E_UNSUPPORTED &&
                  m == nullptr,
          "explicit metal refuses a Q5_0 matrix at load");
    check(strstr(geist_last_create_error(), "blk.0.ffn_down.weight") != nullptr,
          "the refusal names the tensor");

    check(a != nullptr && geist_model_load_from_memory(g.b, g.n, a, &m) == GEIST_OK,
          "auto loads the Q5_0 model (memory)");
    check(is(geist_model_backend_name(m), cpu), "... on the CPU backend");
    check(m != nullptr && decodes(m, a), "... and a session on the auto handle decodes");
    struct geist_backend *other = create(cpu);
    struct geist_session *s     = nullptr;
    check(m != nullptr && geist_session_create(m, other, nullptr, &s) == GEIST_E_INVALID_ARG,
          "another backend is still not the model's");
    geist_backend_destroy(other);
    geist_model_destroy(m);
    m = nullptr;

    check(geist_model_load(path, a, &m) == GEIST_OK && is(geist_model_backend_name(m), cpu),
          "auto loads the Q5_0 model on the CPU (file)");
    geist_model_destroy(m);
    m = nullptr;

    struct tf_buf f32 = llama(false);
    check(geist_model_load_from_memory(f32.b, f32.n, a, &m) == GEIST_OK &&
                  is(geist_model_backend_name(m), "metal") && decodes(m, a),
          "a model metal supports stays on metal");
    geist_model_destroy(m);
    free(f32.b);
    free(g.b);
    geist_backend_destroy(a);
    geist_backend_destroy(metal);
}

#ifdef GEIST_BACKEND_METAL
/* 3. and 4. The working-set budget. */
static void check_budget(void) {
    struct tf_buf                   g     = llama(false);
    struct geist_backend           *metal = create("metal");
    struct geist_backend           *a     = create("auto");
    const char                     *cpu   = geist_backend_registry[0]->name;
    const struct geist_session_opts so    = {.max_seq_len = 256};
    if (metal == nullptr || a == nullptr) {
        check(false, "metal and auto backends");
        free(g.b);
        return;
    }
    const size_t real = ((struct metal_state *) a->state)->ws_budget;

    /* 3. At load. */
    struct geist_model *m = nullptr;
    set_budget(metal, allocated(metal) + (64u << 10));
    check(geist_model_load_from_memory(g.b, g.n, metal, &m) == GEIST_E_OOM,
          "explicit metal over budget: GEIST_E_OOM at load");
    geist_model_destroy(m);
    m = nullptr;
    set_budget(a, allocated(a) + (64u << 10));
    check(geist_model_load_from_memory(g.b, g.n, a, &m) == GEIST_OK &&
                  is(geist_model_backend_name(m), cpu) && decodes(m, a),
          "auto over budget at load: runs on the CPU");
    geist_model_destroy(m);
    m = nullptr;

    /* 4. At session create: measure what the weights and a session take. */
    set_budget(a, real);
    const size_t base = allocated(a);
    check(load_opts(g, a, &so, &m), "load within the real budget");
    const size_t          loaded = allocated(a);
    struct geist_session *s      = nullptr;
    check(m != nullptr && is(geist_model_backend_name(m), "metal") &&
                  geist_session_create(m, a, &so, &s) == GEIST_OK,
          "a session within the real budget");
    const size_t with_session = allocated(a);
    geist_session_destroy(s);
    geist_model_destroy(m);
    m = nullptr;
    check(with_session > loaded + (16u << 10), "a session takes device memory");
    const size_t weights  = loaded - base;
    const size_t sess     = with_session - loaded;
    const size_t headroom = weights + sess / 2;

    set_budget(metal, allocated(metal) + headroom);
    check(load_opts(g, metal, &so, &m) && is(geist_model_backend_name(m), "metal") &&
                  geist_session_create(m, metal, &so, &s) != GEIST_OK && s == nullptr,
          "explicit metal: loads, then refuses the session");
    geist_model_destroy(m);
    m = nullptr;

    set_budget(a, allocated(a) + headroom);
    check(load_opts(g, a, &so, &m) && is(geist_model_backend_name(m), cpu) && decodes(m, a),
          "auto: a session that does not fit moves the model to the CPU");
    geist_model_destroy(m);

    free(g.b);
    geist_backend_destroy(a);
    geist_backend_destroy(metal);
}
#endif

int main(void) {
    const bool gpu = check_choice();
    if (gpu) {
        char          path[] = "/tmp/geist_backend_auto_XXXXXX";
        const int     fd     = mkstemp(path);
        struct tf_buf g      = llama(true);
        const bool    ok     = fd >= 0 && write(fd, g.b, g.n) == (ssize_t) g.n;
        if (fd >= 0) {
            close(fd);
        }
        free(g.b);
        if (!ok) {
            return GEIST_TEST_ERROR;
        }
        check_unsupported(path);
        unlink(path);
#ifdef GEIST_BACKEND_METAL
        check_budget();
#endif
    } else {
        printf("note: \"auto\" is not a GPU here; fallback parts skipped\n");
    }
    if (g_fail == 0 && gpu) {
        printf("PASS: \"auto\" picks metal; a refused model falls back to the CPU\n");
    } else if (g_fail == 0) {
        printf("PASS: \"auto\" picks %s\n", geist_backend_registry[0]->name);
    }
    return g_fail == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
