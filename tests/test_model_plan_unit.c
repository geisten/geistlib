/*
 * test_model_plan_unit — geist_model_plan (#625): what a load would cost,
 * from the GGUF header alone. On models built in memory (model_fixtures.h:
 * a llama and a Qwen3.5-style hybrid), for every backend in the build:
 *
 *   - kv_bytes_per_token equals geist_session_kv_bytes_per_token of a
 *     session created with the same opts on a model loaded with them, for
 *     every KV mode, and when GEIST_KV_INT8 decides an AUTO mode (one
 *     resolution, not two);
 *   - context_length equals geist_model_context_length of the loaded model;
 *     weight_bytes is the tensor bytes; plan from a file equals plan from
 *     memory;
 *   - where the backend reports its allocations: planning allocates nothing,
 *     and loading with a longer max_seq_len grows them by
 *     model_bytes_per_token per position (the RoPE tables; page rounding
 *     aside), so nothing else that scales with the window is missing;
 *   - refusals: nullptr arguments, a missing file, bytes that are no GGUF;
 *     *out is zero after a failure.
 */
#include "test_helpers.h"
#include "model_fixtures.h"

#include <geist.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *const BACKENDS[] = {"cpu_x86", "cpu_neon", "cpu_scalar", "vulkan", "metal"};

static const struct {
    enum geist_kv_mode kv;
    const char        *name;
} MODES[] = {
        {GEIST_KV_AUTO, "AUTO"},
        {GEIST_KV_FP32, "FP32"},
        {GEIST_KV_F16, "F16"},
        {GEIST_KV_INT8, "INT8"},
        {GEIST_KV_INT4, "INT4"},
        {GEIST_KV_KIVI, "KIVI"},
};

struct fixture {
    const char   *name;
    struct tf_buf g;
};

/* The session's report for opts on a model loaded with them; false if the
 * backend refuses the mode (the plan is then not compared). */
static bool session_bytes(const struct fixture            *f,
                          struct geist_backend            *be,
                          const struct geist_session_opts *o,
                          size_t                          *bytes,
                          size_t                          *ctx) {
    struct geist_model   *m = nullptr;
    struct geist_session *s = nullptr;
    bool ok = geist_model_load_from_memory_with_opts(f->g.b, f->g.n, be, o, &m) == GEIST_OK &&
              geist_session_create(m, be, o, &s) == GEIST_OK &&
              geist_session_kv_bytes_per_token(s, bytes) == GEIST_OK;
    *ctx    = m != nullptr ? geist_model_context_length(m) : 0;
    geist_session_destroy(s);
    geist_model_destroy(m);
    return ok;
}

static int compare_modes(const char *backend, const struct fixture *f, struct geist_backend *be) {
    char what[200];
    int  fails = 0;
    for (size_t i = 0; i < sizeof MODES / sizeof MODES[0]; i++) {
        const struct geist_session_opts o = {
                .kv_mode = MODES[i].kv, .top_p = 1.0f, .max_seq_len = 64};
        size_t want = 0, ctx = 0;
        if (!session_bytes(f, be, &o, &want, &ctx)) {
            continue;
        }
        struct geist_model_plan plan = {};
        const enum geist_status s    = geist_model_plan_from_memory(f->g.b, f->g.n, be, &o, &plan);
        snprintf(what,
                 sizeof what,
                 "%s %s KV %s: plan = session (%zu vs %zu), context %zu",
                 backend,
                 f->name,
                 MODES[i].name,
                 plan.kv_bytes_per_token,
                 want,
                 plan.context_length);
        fails +=
                geist_expect(s == GEIST_OK && plan.kv_bytes_per_token == want && want > 0 &&
                                     plan.context_length == ctx && plan.model_bytes_per_token > 0 &&
                                     plan.weight_bytes > 0 && plan.weight_bytes < f->g.n,
                             what);
    }
    /* The environment decides an AUTO mode for both, the same way. */
    setenv("GEIST_KV_INT8", "1", 1);
    const struct geist_session_opts o = {
            .kv_mode = GEIST_KV_AUTO, .top_p = 1.0f, .max_seq_len = 64};
    size_t                  want = 0, ctx = 0;
    struct geist_model_plan plan = {};
    const bool              have = session_bytes(f, be, &o, &want, &ctx);
    const enum geist_status s    = geist_model_plan_from_memory(f->g.b, f->g.n, be, &o, &plan);
    unsetenv("GEIST_KV_INT8");
    snprintf(what,
             sizeof what,
             "%s %s: GEIST_KV_INT8=1 resolves AUTO alike (%zu vs %zu)",
             backend,
             f->name,
             plan.kv_bytes_per_token,
             want);
    fails += geist_expect(!have || (s == GEIST_OK && plan.kv_bytes_per_token == want), what);
    return fails;
}

static bool allocated(struct geist_backend *be, uint64_t *bytes) {
    struct geist_backend_resources r = {};
    if (geist_backend_resources_snapshot(be, &r) != GEIST_OK) {
        return false;
    }
    *bytes = r.allocated_bytes;
    return true;
}

/* Where the backend counts its allocations. */
static int memory(const char *backend, const struct fixture *f, struct geist_backend *be) {
    uint64_t before = 0, after = 0;
    if (!allocated(be, &before)) {
        return 0;
    }
    char                    what[200];
    int                     fails = 0;
    struct geist_model_plan plan  = {};
    const bool ok = geist_model_plan_from_memory(f->g.b, f->g.n, be, nullptr, &plan) == GEIST_OK;
    snprintf(what,
             sizeof what,
             "%s %s: planning allocates nothing on the backend",
             backend,
             f->name);
    fails += geist_expect(ok && allocated(be, &after) && after == before, what);

    /* A model with a longer window costs model_bytes_per_token more per position. */
    uint64_t            size[2] = {0, 0};
    const size_t        len[2]  = {64, 576};
    struct geist_model *m       = nullptr;
    for (size_t i = 0; i < 2; i++) {
        const struct geist_session_opts o = {.top_p = 1.0f, .max_seq_len = len[i]};
        uint64_t                        a = 0, b = 0;
        if (!allocated(be, &a) ||
            geist_model_load_from_memory_with_opts(f->g.b, f->g.n, be, &o, &m) != GEIST_OK ||
            !allocated(be, &b)) {
            fails += geist_expect(false, "load for the window check");
            geist_model_destroy(m);
            return fails;
        }
        size[i] = b - a;
        geist_model_destroy(m);
        m = nullptr;
    }
    const uint64_t grew = size[1] - size[0],
                   want = (uint64_t) (len[1] - len[0]) * plan.model_bytes_per_token;
    const uint64_t slack =
            4 * (uint64_t) sysconf(_SC_PAGESIZE); /* four tables, each page-rounded */
    snprintf(what,
             sizeof what,
             "%s %s: a longer window grows by model_bytes_per_token (%llu vs %llu)",
             backend,
             f->name,
             (unsigned long long) grew,
             (unsigned long long) want);
    fails += geist_expect(grew + slack >= want && grew <= want + slack, what);
    return fails;
}

static int file_and_errors(const char *backend, const struct fixture *f, struct geist_backend *be) {
    char what[200];
    int  fails  = 0;
    char path[] = "/tmp/geist_plan_XXXXXX";
    int  fd     = mkstemp(path);
    bool wrote  = fd >= 0 && write(fd, f->g.b, f->g.n) == (ssize_t) f->g.n;
    if (fd >= 0) {
        close(fd);
    }
    struct geist_model_plan a = {}, b = {};
    snprintf(
            what, sizeof what, "%s %s: plan from a file equals plan from memory", backend, f->name);
    fails += geist_expect(wrote && geist_model_plan(path, be, nullptr, &a) == GEIST_OK &&
                                  geist_model_plan_from_memory(f->g.b, f->g.n, be, nullptr, &b) ==
                                          GEIST_OK &&
                                  !memcmp(&a, &b, sizeof a),
                          what);
    unlink(path);

    struct geist_model_plan z = {.weight_bytes = 7};
    snprintf(what, sizeof what, "%s: refusals leave *out zero", backend);
    fails += geist_expect(
            geist_model_plan("/no/such/model.gguf", be, nullptr, &z) == GEIST_E_IO &&
                    z.weight_bytes == 0 &&
                    geist_model_plan_from_memory("not a gguf", 10, be, nullptr, &z) ==
                            GEIST_E_FORMAT &&
                    geist_model_plan(nullptr, be, nullptr, &z) == GEIST_E_INVALID_ARG &&
                    geist_model_plan(path, nullptr, nullptr, &z) == GEIST_E_INVALID_ARG &&
                    geist_model_plan_from_memory(f->g.b, f->g.n, be, nullptr, nullptr) ==
                            GEIST_E_INVALID_ARG,
            what);
    return fails;
}

int main(void) {
    struct tf_vocab v       = tf_make_vocab("\xc4\xa0", false);
    struct fixture  fixes[] = {
            {"llama",
             mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                               .d_model  = 128,
                                               .heads    = 4,
                                               .kv_heads = 2,
                                               .ffn      = 256,
                                               .vocab    = 512,
                                               .context  = 8192,
                                               .seed     = 14})},
            {"qwen35",
             mf_qwen35_gguf(&(struct mf_qwen35) {.layers     = 4,
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
                                                 .tok        = &v})},
    };
    int fails = 0, ran = 0;
    for (size_t bi = 0; bi < sizeof BACKENDS / sizeof BACKENDS[0]; bi++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(BACKENDS[bi], nullptr, nullptr, &be) != GEIST_OK ||
            be == nullptr) {
            continue; /* not in this build */
        }
        ran++;
        for (size_t f = 0; f < sizeof fixes / sizeof fixes[0]; f++) {
            fails += compare_modes(BACKENDS[bi], &fixes[f], be);
            fails += memory(BACKENDS[bi], &fixes[f], be);
            fails += file_and_errors(BACKENDS[bi], &fixes[f], be);
        }
        geist_backend_destroy(be);
    }
    for (size_t f = 0; f < sizeof fixes / sizeof fixes[0]; f++) {
        free(fixes[f].g.b);
    }
    tf_free_vocab(&v);
    if (ran == 0) {
        GEIST_SKIP("no backend in this build");
    }
    if (fails) {
        fprintf(stderr, "test_model_plan_unit: %d failure(s)\n", fails);
        return 1;
    }
    printf("test_model_plan_unit: plan = session for every KV mode and the env, file = memory, "
           "window growth, refusals on %d backend(s) passed\n",
           ran);
    return 0;
}
