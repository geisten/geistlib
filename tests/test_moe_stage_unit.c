/*
 * test_moe_stage_unit - a mixture-of-experts prefill that copies cold
 * experts to the GPU (#239), on any build: the GPU is cpu_scalar under
 * another name with the fused ops a discrete GPU's MoE path needs written
 * on the host (linear_t, silu_mul, rows_gather, rows_weighted_sum, submit,
 * stage_reserve, stage_weights) and memory of its own.
 *
 * The model is mf_qwen3moe_gguf's (F32 weights, 2 layers of 8 experts, 2
 * per token), on that GPU with 8 of the 16 experts hot, a 21-token prompt
 * prefilled in chunks of 8. Against cpu_scalar alone (cosine >= 0.999; the
 * cold experts not staged run on the CPU backend, cpu_x86's W8A8 included):
 *   - default: every cold expert gets a host copy (GEIST_W_HOST) on the
 *     GPU, the load measures the threshold (whatever it picks);
 *   - GEIST_MOE_STAGE_M_MIN=1: the prefill chunks stage cold experts, their
 *     rows run on the GPU, every staged weight is a host copy, no call
 *     stages more than the reserved slots;
 *   - GEIST_MOE_STAGE_MAX=1: at most one expert (three weights) per step;
 *   - GEIST_MOE_STAGE=0 and GEIST_MOE_STAGE_HOST=0: no host copies, nothing
 *     staged, the experts' rows as before.
 * Which cold experts get the host copies, with experts of 1.5 MiB and 3 MiB
 * for copies (two), the resident set from GEIST_MOE_PROFILE, the profile
 * cache in a temporary directory: the fresh start takes the resident
 * ranking's next two; after one prefill the cache's prefill rows decide
 * (the most rows first), the resident set unchanged.
 * The real copies and their ordering on Vulkan: test_backend_vulkan_stage_unit.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "../src/archs/transformer/arch_state.h"
#include "../src/archs/transformer/moe_profile.h"
#include "../src/engine/model.h"
#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { VOCAB = 96, N_PROMPT = 21 };

static const struct mf_qwen3moe GEOM = {.layers    = 2,
                                        .d_model   = 64,
                                        .heads     = 4,
                                        .kv_heads  = 2,
                                        .head_dim  = 16,
                                        .experts   = 8,
                                        .used      = 2,
                                        .expert_ff = 32,
                                        .vocab     = VOCAB,
                                        .seed      = 11};

static const struct geist_session_opts OPTS = {.top_p = 1.0f, .m_max = 8, .max_seq_len = 256};

/* ---- The host "GPU" ---------------------------------------------------- */

static const struct geist_backend_descriptor *g_orig;
static struct geist_backend_descriptor        g_fake;
static struct geist_backend_fused             g_fused;
static struct geist_backend_vtbl              g_vtbl;
static size_t                                 g_slots, g_stage_calls, g_stage_max_n;
static bool                                   g_stage_bad;

static float *at(const struct geist_tensor *t) {
    uint8_t *p = g_orig->vtbl->buffer_map(t->buffer);
    return p != nullptr ? (float *) (p + t->offset) : nullptr;
}

static size_t row_stride(const struct geist_tensor *t, size_t cols) {
    return t->ndim >= 2 ? (size_t) t->stride[t->ndim - 2] : cols;
}

static enum geist_status fake_linear_t(struct geist_backend      *be,
                                       const struct geist_tensor *x,
                                       const struct geist_weight *w,
                                       const struct geist_tensor *t_w,
                                       size_t                     m,
                                       struct geist_tensor       *y) {
    (void) be;
    (void) t_w;
    if (w == nullptr || w->raw == nullptr || w->dtype != GEIST_DTYPE_F32)
        return GEIST_E_UNSUPPORTED;
    const size_t n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    const float *xp = at(x), *wp = w->raw;
    float       *yp = at(y);
    if (xp == nullptr || yp == nullptr)
        return GEIST_E_BACKEND;
    const size_t xs = row_stride(x, n_in), ys = row_stride(y, n_out);
    for (size_t r = 0; r < m; r++)
        for (size_t o = 0; o < n_out; o++) {
            double acc = 0.0;
            for (size_t i = 0; i < n_in; i++)
                acc += (double) xp[r * xs + i] * (double) wp[o * n_in + i];
            yp[r * ys + o] = (float) acc;
        }
    return GEIST_OK;
}

static enum geist_status fake_silu_mul(struct geist_backend      *be,
                                       const struct geist_tensor *x,
                                       const struct geist_tensor *z,
                                       struct geist_tensor       *y) {
    (void) be;
    const size_t n  = (size_t) (x->ndim >= 2 ? x->shape[0] * x->shape[1] : x->shape[0]);
    const float *xp = at(x), *zp = at(z);
    float       *yp = at(y);
    if (xp == nullptr || zp == nullptr || yp == nullptr)
        return GEIST_E_BACKEND;
    for (size_t i = 0; i < n; i++)
        yp[i] = xp[i] / (1.0f + expf(-xp[i])) * zp[i];
    return GEIST_OK;
}

static enum geist_status fake_rows_gather(struct geist_backend      *be,
                                          size_t                     n,
                                          size_t                     d,
                                          const uint32_t             src[static n],
                                          const struct geist_tensor *x,
                                          struct geist_tensor       *out) {
    (void) be;
    const float *xp = at(x);
    float       *op = at(out);
    if (xp == nullptr || op == nullptr)
        return GEIST_E_BACKEND;
    for (size_t p = 0; p < n; p++)
        memcpy(op + p * d, xp + (size_t) src[p] * d, d * sizeof(float));
    return GEIST_OK;
}

static enum geist_status fake_rows_weighted_sum(struct geist_backend      *be,
                                                size_t                     m,
                                                size_t                     d,
                                                size_t                     n,
                                                const uint32_t             start[static m + 1],
                                                const uint32_t             rows[static n],
                                                const float                w[static n],
                                                const float                add[static m * d],
                                                const struct geist_tensor *y,
                                                struct geist_tensor       *out) {
    (void) be;
    const float *yp = at(y);
    float       *op = at(out);
    if (yp == nullptr || op == nullptr)
        return GEIST_E_BACKEND;
    for (size_t t = 0; t < m; t++)
        for (size_t c = 0; c < d; c++) {
            float acc = add[t * d + c];
            for (uint32_t j = start[t]; j < start[t + 1]; j++)
                acc += w[j] * yp[(size_t) rows[j] * d + c];
            op[t * d + c] = acc;
        }
    return GEIST_OK;
}

/* cpu_scalar has no buffer_copy: the arch's GPU path copies x and rows. */
static enum geist_status fake_buffer_copy(struct geist_buffer       *dst,
                                          size_t                     dst_offset,
                                          const struct geist_buffer *src,
                                          size_t                     src_offset,
                                          size_t                     n_bytes) {
    uint8_t       *d  = g_orig->vtbl->buffer_map(dst);
    const uint8_t *sp = g_orig->vtbl->buffer_map((struct geist_buffer *) src);
    if (d == nullptr || sp == nullptr)
        return GEIST_E_BACKEND;
    memmove(d + dst_offset, sp + src_offset, n_bytes);
    return GEIST_OK;
}

static void fake_submit(struct geist_backend *be) {
    (void) be;
}

static bool fake_supported(struct geist_backend *be, const struct geist_fusion_query *q) {
    if (q->op == GEIST_FUSED_ROWS_WEIGHTED_SUM || q->op == GEIST_FUSED_ROWS_GATHER)
        return true;
    if (q->op == GEIST_FUSED_LINEAR_GROUPED || q->op == GEIST_FUSED_MOE_DEVICE)
        return false;
    return g_orig->fused->supported != nullptr && g_orig->fused->supported(be, q);
}

static enum geist_status fake_memory_info(const struct geist_backend  *be,
                                          struct geist_backend_memory *out) {
    (void) be;
    *out = (struct geist_backend_memory) {.total_bytes = (uint64_t) 1 << 30,
                                          .free_bytes  = (uint64_t) 1 << 30};
    return GEIST_OK;
}

static enum geist_status fake_stage_reserve(struct geist_backend            *be,
                                            size_t                           n_slots,
                                            size_t                           n_like,
                                            const struct geist_weight *const like[static n_like]) {
    (void) be;
    (void) like;
    g_slots = n_slots > g_slots ? n_slots : g_slots;
    return n_slots > 0 ? GEIST_OK : GEIST_E_INVALID_ARG;
}

static enum geist_status fake_stage_weights(struct geist_backend            *be,
                                            size_t                           n,
                                            const struct geist_weight *const w[static n]) {
    (void) be;
    if (n > g_slots)
        return GEIST_E_UNSUPPORTED;
    for (size_t i = 0; i < n; i++)
        g_stage_bad = g_stage_bad || w[i] == nullptr || (w[i]->flags & GEIST_W_HOST) == 0;
    g_stage_calls += n > 0;
    g_stage_max_n = n > g_stage_max_n ? n : g_stage_max_n;
    return GEIST_OK;
}

static void fake_install(struct geist_backend *be) {
    g_orig                     = be->desc;
    g_fake                     = *g_orig;
    g_fused                    = *g_orig->fused;
    g_fused.supported          = fake_supported;
    g_fused.linear_t           = fake_linear_t;
    g_fused.silu_mul           = fake_silu_mul;
    g_fused.rows_gather        = fake_rows_gather;
    g_fused.rows_weighted_sum  = fake_rows_weighted_sum;
    g_fused.submit             = fake_submit;
    g_fused.stage_reserve      = fake_stage_reserve;
    g_fused.stage_weights      = fake_stage_weights;
    g_fake.name                = "hostgpu";
    g_fake.fused               = &g_fused;
    g_vtbl                     = *g_orig->vtbl;
    g_vtbl.buffer_copy         = fake_buffer_copy;
    g_fake.vtbl                = &g_vtbl;
    g_fake.memory_info         = fake_memory_info;
    g_fake.caps.batched_submit = true;
    be->desc                   = &g_fake;
}

/* ---- The checks -------------------------------------------------------- */

static bool logits_after(struct geist_session *s,
                         size_t                n,
                         const geist_token_t   ids[static n],
                         size_t                step,
                         float                 out[static VOCAB]) {
    if (geist_session_reset(s) != GEIST_OK)
        return false;
    for (size_t i = 0; i < n; i += step) {
        const size_t k = n - i < step ? n - i : step;
        if (geist_session_prefill_tokens(s, k, ids + i) != GEIST_OK)
            return false;
    }
    size_t       nl = 0;
    const float *lg = geist_session_peek_logits(&nl, s);
    if (lg == nullptr || nl != VOCAB)
        return false;
    memcpy(out, lg, VOCAB * sizeof(float));
    return true;
}

static double cosine(const float a[static VOCAB], const float b[static VOCAB]) {
    double ab = 0.0, aa = 0.0, bb = 0.0;
    for (size_t i = 0; i < VOCAB; i++) {
        ab += (double) a[i] * b[i];
        aa += (double) a[i] * a[i];
        bb += (double) b[i] * b[i];
    }
    return ab / sqrt(aa * bb > 0.0 ? aa * bb : 1.0);
}

struct outcome {
    size_t   n_stage_w, stage_m_min;
    uint64_t rows_staged, rows_gpu;
    double   cos;
};

/* Loads the fixture on the host GPU under the current environment and
 * prefills the prompt in chunks; false when anything fails. */
static bool run(const struct tf_buf *g, const float ref[static VOCAB], struct outcome *o) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK)
        return false;
    fake_install(be);
    g_slots = g_stage_calls = g_stage_max_n = 0;
    g_stage_bad                             = false;
    struct geist_model   *m                 = nullptr;
    struct geist_session *s                 = nullptr;
    geist_token_t         ids[N_PROMPT];
    float                 lg[VOCAB];
    for (size_t i = 0; i < N_PROMPT; i++)
        ids[i] = (geist_token_t) ((i * 37 + 5) % VOCAB);
    bool ok = geist_model_load_from_memory(g->b, g->n, be, &m) == GEIST_OK &&
              geist_session_create(m, be, &OPTS, &s) == GEIST_OK &&
              logits_after(s, N_PROMPT, ids, OPTS.m_max, lg);
    if (!ok)
        fprintf(stderr, "%s\n", geist_backend_errmsg(be));
    if (ok) {
        const struct transformer_arch_state *st = tf_state(geist_model_internal_arch_meta(m));
        const struct transformer_moe_gpu    *mg = st->moe_gpu;
        ok                                      = mg != nullptr && st->moe_cpu != nullptr;
        if (ok)
            *o = (struct outcome) {.n_stage_w   = mg->n_stage_w,
                                   .stage_m_min = mg->stage_m_min,
                                   .rows_staged = mg->rows_staged,
                                   .rows_gpu    = mg->rows_gpu,
                                   .cos         = cosine(lg, ref)};
    }
    geist_session_destroy(s);
    geist_model_destroy(m);
    be->desc = g_orig; /* teardown through the real table */
    geist_backend_destroy(be);
    return ok;
}

/* ---- Host copies chosen by the profile's prefill rows ------------------ */

/* Experts of 1.5 MiB (F32 256 x 512, three of them): GEIST_MOE_STAGE_HOST
 * = 3 MiB holds two host copies. */
static const struct mf_qwen3moe GEOM_BIG = {.layers    = 2,
                                            .d_model   = 256,
                                            .heads     = 4,
                                            .kv_heads  = 2,
                                            .head_dim  = 64,
                                            .experts   = 8,
                                            .used      = 2,
                                            .expert_ff = 512,
                                            .vocab     = VOCAB,
                                            .seed      = 11};

enum { N_BIG = 2 * 8 };

/* Loads `g` on the host GPU, notes which experts are resident (hot) and
 * which have a host copy (host), the prefill rows the profile had at load
 * at threshold 1 (rows), then prefills the prompt in chunks of 8. */
static bool placed(const struct tf_buf *g,
                   bool                 hot[static N_BIG],
                   bool                 host[static N_BIG],
                   uint64_t             rows[static N_BIG]) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK)
        return false;
    fake_install(be);
    struct geist_model   *m = nullptr;
    struct geist_session *s = nullptr;
    geist_token_t         ids[N_PROMPT];
    float                 lg[VOCAB];
    for (size_t i = 0; i < N_PROMPT; i++)
        ids[i] = (geist_token_t) ((i * 37 + 5) % VOCAB);
    bool ok = geist_model_load_from_memory(g->b, g->n, be, &m) == GEIST_OK;
    if (ok) {
        const struct transformer_arch_state *st = tf_state(geist_model_internal_arch_meta(m));
        const struct transformer_moe_gpu    *mg = st->moe_gpu;
        ok = mg != nullptr && mg->stage_w != nullptr && moe_slots(&st->config) == 8;
        for (size_t i = 0; ok && i < N_BIG; i++) {
            hot[i]  = mg->twins[i * 3].raw != nullptr;
            host[i] = mg->stage_w[i * 3].raw != nullptr;
        }
        if (ok && !moe_profile_prefill_rows(st->moe_prof, 1, N_BIG, rows))
            memset(rows, 0, N_BIG * sizeof rows[0]);
    }
    ok = ok && geist_session_create(m, be, &OPTS, &s) == GEIST_OK &&
         logits_after(s, N_PROMPT, ids, OPTS.m_max, lg);
    if (!ok)
        fprintf(stderr, "%s\n", geist_backend_errmsg(be));
    geist_session_destroy(s);
    geist_model_destroy(m); /* writes the profile cache */
    be->desc = g_orig;
    geist_backend_destroy(be);
    return ok;
}

/* Expert i before expert j for a host copy: more prefill rows, then the
 * resident ranking's count (check_prefill_choice's profile), then the lower
 * e * 2 + l. */
static bool before(const uint64_t rows[static N_BIG], uint32_t i, uint32_t j) {
    const uint64_t ci = i / 8 == 0 ? 8 - i % 8 : i % 8 + 1, cj = j / 8 == 0 ? 8 - j % 8 : j % 8 + 1;
    if (rows[i] != rows[j])
        return rows[i] > rows[j];
    if (ci != cj)
        return ci > cj;
    return (i % 8) * 2 + i / 8 < (j % 8) * 2 + j / 8;
}

/* The fresh start picks the host copies in the resident ranking's order;
 * after a prefill, the profile cache's prefill rows pick them, and the
 * resident experts stay the same. */
static int check_prefill_choice(void) {
    int         fails = 0;
    const char *tmp   = getenv("TMPDIR");
    char        dir[256], prof[300];
    snprintf(dir, sizeof dir, "%s/geist_moe_stage_XXXXXX", tmp != nullptr ? tmp : "/tmp");
    if (mkdtemp(dir) == nullptr)
        return geist_expect(false, "a temporary cache directory");
    /* The resident ranking (GEIST_MOE_PROFILE wins over the cache): layer
     * 0's experts 0-3, layer 1's 4-7 hot; the cold ones by count, then
     * e * 2 + l: (l1,e3) and (l0,e4) at 4, (l1,e2) and (l0,e5) at 3, ... */
    snprintf(prof, sizeof prof, "%s/usage.txt", dir);
    FILE *f = fopen(prof, "w");
    if (f != nullptr) {
        fputs("USAGE fixture profile corpus 9 0 8,7,6,5,4,3,2,1\n"
              "USAGE fixture profile corpus 9 1 1,2,3,4,5,6,7,8\n",
              f);
        fclose(f);
    }
    setenv("GEIST_MOE_PROFILE", prof, 1);
    setenv("GEIST_MOE_PROFILE_CACHE", dir, 1);
    setenv("GEIST_MOE_STAGE_HOST", "3", 1);
    setenv("GEIST_MOE_STAGE_M_MIN", "1", 1);
    struct tf_buf g = mf_qwen3moe_gguf(&GEOM_BIG);
    bool          hot0[N_BIG], host0[N_BIG], hot1[N_BIG], host1[N_BIG];
    uint64_t      rows0[N_BIG], rows1[N_BIG];
    const bool    ok0 = placed(&g, hot0, host0, rows0);
    const bool    ok1 = ok0 && placed(&g, hot1, host1, rows1);
    fails += geist_expect(ok0 && ok1, "the big fixture loads and prefills twice");
    if (ok0 && ok1) {
        bool want_hot[N_BIG] = {0}, fresh = true, zero = true;
        for (size_t e = 0; e < 4; e++)
            want_hot[e] = want_hot[8 + 4 + e] = true;
        for (size_t i = 0; i < N_BIG; i++) {
            fresh = fresh && host0[i] == (i == 8 + 3 || i == 4);
            zero  = zero && rows0[i] == 0;
        }
        fails += geist_expect(memcmp(hot0, want_hot, sizeof hot0) == 0 &&
                                      memcmp(hot1, want_hot, sizeof hot1) == 0,
                              "the resident experts follow GEIST_MOE_PROFILE, both loads");
        fails += geist_expect(zero && fresh,
                              "fresh start: no prefill rows, host copies in the resident order");
        /* The two cold experts with the most rows (equal bytes), ties by
         * count, then e * 2 + l. */
        uint32_t want[2] = {UINT32_MAX, UINT32_MAX};
        uint64_t total   = 0;
        for (size_t k = 0; k < 2; k++)
            for (uint32_t i = 0; i < N_BIG; i++)
                if (!hot1[i] && i != want[0] &&
                    (want[k] == UINT32_MAX || before(rows1, i, want[k])))
                    want[k] = i;
        bool picked = true;
        for (size_t i = 0; i < N_BIG; i++) {
            total += rows1[i];
            picked = picked && host1[i] == (i == want[0] || i == want[1]);
        }
        printf("prefill-aware host copies: (l%u,e%u) %llu rows, (l%u,e%u) %llu rows; fresh "
               "start (l1,e3), (l0,e4)\n",
               want[0] / 8,
               want[0] % 8,
               (unsigned long long) rows1[want[0]],
               want[1] / 8,
               want[1] % 8,
               (unsigned long long) rows1[want[1]]);
        fails += geist_expect(total == 2ull * N_PROMPT * 2,
                              "the cache holds every prefill row of the first run");
        fails += geist_expect(picked, "after a prefill: the most prefill rows get the host copies");
    }
    unsetenv("GEIST_MOE_STAGE_M_MIN");
    unsetenv("GEIST_MOE_STAGE_HOST");
    unsetenv("GEIST_MOE_PROFILE");
    setenv("GEIST_MOE_PROFILE_CACHE", "0", 1);
    remove(prof);
    DIR *dp = opendir(dir); /* the cache file is the one entry left */
    for (struct dirent *de = dp != nullptr ? readdir(dp) : nullptr; de != nullptr;
         de                = readdir(dp)) {
        char path[600];
        if (de->d_name[0] != '.' &&
            (size_t) snprintf(path, sizeof path, "%s/%s", dir, de->d_name) < sizeof path)
            remove(path);
    }
    if (dp != nullptr)
        closedir(dp);
    rmdir(dir);
    free(g.b);
    return fails;
}

int main(void) {
    setenv("GEIST_MOE_PROFILE_CACHE", "0", 1);
    setenv("GEIST_MOE_HOT", "8", 1);
    struct tf_buf         g   = mf_qwen3moe_gguf(&GEOM);
    struct geist_backend *cpu = nullptr;
    struct geist_model   *m   = nullptr;
    struct geist_session *s   = nullptr;
    float                 ref[VOCAB];
    int                   fails = 0;
    geist_token_t         ids[N_PROMPT];
    for (size_t i = 0; i < N_PROMPT; i++)
        ids[i] = (geist_token_t) ((i * 37 + 5) % VOCAB);
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &cpu) != GEIST_OK ||
        geist_model_load_from_memory(g.b, g.n, cpu, &m) != GEIST_OK ||
        geist_session_create(m, cpu, &OPTS, &s) != GEIST_OK ||
        !logits_after(s, N_PROMPT, ids, OPTS.m_max, ref)) {
        printf("FAIL: the reference on cpu_scalar\n");
        return GEIST_TEST_ERROR;
    }
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(cpu);

    const size_t   cold = GEOM.layers * GEOM.experts - 8;
    struct outcome o    = {0};

    fails += geist_expect(run(&g, ref, &o), "default: the model runs on the host GPU");
    printf("default: %zu host copies, threshold %zu, %llu rows staged, cosine %.6f\n",
           o.n_stage_w,
           o.stage_m_min,
           (unsigned long long) o.rows_staged,
           o.cos);
    fails += geist_expect(o.n_stage_w == cold && g_slots > 0,
                          "default: every cold expert has a host copy, slots reserved");
    fails += geist_expect(o.cos >= 0.999, "default: the logits match cpu_scalar's");

    setenv("GEIST_MOE_STAGE_M_MIN", "1", 1);
    fails += geist_expect(run(&g, ref, &o), "threshold 1: runs");
    printf("threshold 1: %llu rows staged, %llu on the GPU, %zu stage calls, at most %zu "
           "weights, cosine %.6f\n",
           (unsigned long long) o.rows_staged,
           (unsigned long long) o.rows_gpu,
           g_stage_calls,
           g_stage_max_n,
           o.cos);
    fails += geist_expect(o.stage_m_min == 1 && o.rows_staged > 0 && g_stage_calls > 0,
                          "threshold 1: the prefill chunks stage cold experts");
    fails += geist_expect(!g_stage_bad && g_stage_max_n <= g_slots,
                          "threshold 1: only host copies staged, within the slots");
    fails += geist_expect(o.cos >= 0.999, "threshold 1: the logits match cpu_scalar's");

    setenv("GEIST_MOE_STAGE_MAX", "1", 1);
    fails += geist_expect(run(&g, ref, &o), "one per step: runs");
    fails += geist_expect(o.rows_staged > 0 && g_stage_max_n == 3 && g_slots == 3,
                          "one per step: one expert's three weights per stage call");
    fails += geist_expect(o.cos >= 0.999, "one per step: the logits match cpu_scalar's");
    unsetenv("GEIST_MOE_STAGE_MAX");

    setenv("GEIST_MOE_STAGE_HOST", "0", 1);
    fails += geist_expect(run(&g, ref, &o), "no host memory: runs");
    fails += geist_expect(o.n_stage_w == 0 && o.rows_staged == 0 && g_stage_calls == 0,
                          "no host memory for copies: nothing staged");
    unsetenv("GEIST_MOE_STAGE_HOST");

    setenv("GEIST_MOE_STAGE", "0", 1);
    fails += geist_expect(run(&g, ref, &o), "staging off: runs");
    fails += geist_expect(o.n_stage_w == 0 && o.rows_staged == 0 && g_slots == 0 && o.cos >= 0.999,
                          "GEIST_MOE_STAGE=0: no host copies, no slots, the same logits");
    unsetenv("GEIST_MOE_STAGE");
    unsetenv("GEIST_MOE_STAGE_M_MIN");

    fails += check_prefill_choice();
    free(g.b);
    if (fails == 0)
        printf("PASS: cold experts staged for prefill on a host GPU (#239)\n");
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
