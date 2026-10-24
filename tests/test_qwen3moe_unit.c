/*
 * test_qwen3moe_unit - the Qwen3-MoE family on an in-memory model (#48).
 *
 * mf_qwen3moe_gguf builds a two-layer Qwen3-MoE with F32 weights: 8
 * experts, 2 used per token. On every CPU backend in the build:
 *   - a 21-token prompt prefilled in m_max = 8 chunks (seq = m_max, then a
 *     tail) and the same prompt one token at a time (seq = 1, the decode
 *     shapes) give the same next-token logits, within 1e-4 of their scale;
 *   - a session reset and the same prefill reproduce the logits bit for bit;
 *   - every backend's logits match cpu_scalar's within that backend's
 *     tolerance for F32 weights: cpu_neon runs them exactly (relative L2
 *     <= 1e-3); cpu_x86 quantizes them to W8A8 and keeps an INT8 KV cache by
 *     design, so its bar is test_f32q_unit's, cosine >= 0.999 (the router,
 *     marked GEIST_W_EXACT, stays F32 there too);
 *   - fault injection: layer 0's expert down projections lose their kernels;
 *     the prefill returns GEIST_E_UNSUPPORTED and leaves no logits, and the
 *     session gives the original logits again once they are back.
 * Hot experts on a GPU backend (#78 phase 3), for each GPU backend in the
 * build: the model on cpu_scalar with the GPU as split backend and
 * GEIST_MOE_HOT = half, then all of the 16 experts, the prompt fed one
 * token at a time (hot experts run at decode only); the experts are placed,
 * rows run on the GPU, and the logits match cpu_scalar's (relative L2
 * <= 1e-3, F32 weights on both).
 * The model on a GPU backend itself (#78 phase 3): attention and router run
 * there, the experts on the state's own CPU backend, 0, 8 and 16 of them
 * with a copy on the GPU; prefill in chunks and token by token; the logits
 * match cpu_scalar's within the CPU experts' tolerance (cosine >= 0.999:
 * cpu_x86 runs F32 experts as W8A8).
 * Load refusals: a missing expert tensor (named in the error), a router
 * gating other than softmax, a shared expert, more experts used than exist.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "../src/archs/transformer/arch_state.h"
#include "../src/archs/transformer/moe_gpu.h"
#include "../src/base/heap.h"
#include "../src/engine/model.h"
#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* Next-token logits after `n` prompt ids fed `step` at a time. */
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

static double rel_l2(const float a[static VOCAB], const float b[static VOCAB]) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < VOCAB; i++) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return sqrt(num / (den > 0.0 ? den : 1.0));
}

/* Remove (or restore) the kernels of layer 0's expert down projections. */
static void break_layer0_down(struct geist_session *s, struct geist_weight *saved, bool restore) {
    struct transformer_arch_session  *a = geist_session_internal_arch_session(s);
    struct transformer_layer_weights *L = &a->model->layers[0];
    for (size_t e = 0; e < L->moe_n_experts; e++) {
        if (restore) {
            L->moe_down_w[e] = saved[e];
        } else {
            saved[e]                        = L->moe_down_w[e];
            L->moe_down_w[e].linear_m1      = nullptr;
            L->moe_down_w[e].linear_mN      = nullptr;
            L->moe_down_w[e].linear_pair_m1 = nullptr;
            L->moe_down_w[e].linear_pair_mN = nullptr;
        }
    }
}

/* One backend: prefill shapes, reset reuse, fault injection. Writes its
 * prefill logits to `ref`. Returns failures, or -1 when not built. */
static int check_backend(const char *name, float ref[static VOCAB]) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(name, nullptr, nullptr, &be) != GEIST_OK) {
        printf("skip %s: not in this build\n", name);
        return -1;
    }
    struct tf_buf         g     = mf_qwen3moe_gguf(&GEOM);
    struct geist_model   *m     = nullptr;
    struct geist_session *s     = nullptr;
    int                   fails = 0;
    geist_token_t         ids[N_PROMPT];
    for (size_t i = 0; i < N_PROMPT; i++)
        ids[i] = (geist_token_t) ((i * 37 + 5) % VOCAB);

    if (geist_model_load_from_memory(g.b, g.n, be, &m) != GEIST_OK ||
        geist_session_create(m, be, &OPTS, &s) != GEIST_OK) {
        fails += geist_expect(false, "the qwen3moe fixture loads");
        goto out;
    }
    float chunked[VOCAB], single[VOCAB], again[VOCAB];
    if (!logits_after(s, N_PROMPT, ids, N_PROMPT, chunked) ||
        !logits_after(s, N_PROMPT, ids, 1, single) ||
        !logits_after(s, N_PROMPT, ids, N_PROMPT, again)) {
        fails += geist_expect(false, "prefill runs");
        goto out;
    }
    double scale = 0.0, diff = 0.0;
    for (size_t i = 0; i < VOCAB; i++) {
        scale = fmax(scale, fabs(chunked[i]));
        diff  = fmax(diff, fabs((double) chunked[i] - single[i]));
    }
    printf("%s: seq=m_max chunks vs seq=1: max|d| %.3g of scale %.3g\n", name, diff, scale);
    fails += geist_expect(diff <= 1e-4 * scale, "seq = m_max and seq = 1 agree");
    fails += geist_expect(memcmp(chunked, again, sizeof chunked) == 0,
                          "a reset session reproduces its logits bit for bit");
    memcpy(ref, chunked, sizeof chunked);

    /* KV cache: the family asks for F16 under AUTO; a session that asks
     * for FP32 gets FP32, and the two agree to F16 rounding. */
    {
        struct transformer_arch_session *a = geist_session_internal_arch_session(s);
        fails += geist_expect(a->kv_f16_enabled && !a->kv_int8_enabled,
                              "qwen3moe gets an F16 KV cache under AUTO");
        struct geist_session     *s32 = nullptr;
        struct geist_session_opts o32 = OPTS;
        o32.kv_mode                   = GEIST_KV_FP32;
        float f32[VOCAB];
        if (geist_session_create(m, be, &o32, &s32) == GEIST_OK &&
            logits_after(s32, N_PROMPT, ids, N_PROMPT, f32)) {
            struct transformer_arch_session *a32 = geist_session_internal_arch_session(s32);
            const double                     d   = rel_l2(chunked, f32);
            const double                     c   = cosine(chunked, f32);
            printf("%s: F16 vs FP32 KV logits, relative L2 %.3g, cosine %.6f\n", name, d, c);
            fails += geist_expect(!a32->kv_f16_enabled, "kv_mode FP32 is honoured");
            /* The kernels are exact on F16 input (test_attention_f32_unit);
             * this is the model end to end, where a routing near-tie can
             * turn the rounding into a swapped expert: the cpu_x86 bar. */
            fails += geist_expect(c >= 0.999, "F16 KV logits match FP32 KV");
        } else {
            fails += geist_expect(false, "an FP32-KV session runs");
        }
        geist_session_destroy(s32);
    }

    struct geist_weight saved[16];
    break_layer0_down(s, saved, false);
    size_t                  nl = 0;
    const enum geist_status st = geist_session_reset(s) == GEIST_OK
                                         ? geist_session_prefill_tokens(s, N_PROMPT, ids)
                                         : GEIST_E_INTERNAL;
    fails += geist_expect(st == GEIST_E_UNSUPPORTED, "a missing expert kernel fails the prefill");
    fails += geist_expect(geist_session_peek_logits(&nl, s) == nullptr,
                          "a failed prefill leaves no logits");
    break_layer0_down(s, saved, true);
    float healed[VOCAB];
    fails += geist_expect(logits_after(s, N_PROMPT, ids, N_PROMPT, healed) &&
                                  memcmp(healed, chunked, sizeof healed) == 0,
                          "the session recovers once the kernels are back");
out:
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    free(g.b);
    return fails;
}

/* A model the family must refuse; the error must contain `needle`. */
static int check_refused(const char *what, const struct mf_qwen3moe *c, const char *needle) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK)
        return geist_expect(false, "cpu_scalar backend");
    struct tf_buf       g  = mf_qwen3moe_gguf(c);
    struct geist_model *m  = nullptr;
    enum geist_status   s  = geist_model_load_from_memory(g.b, g.n, be, &m);
    const char         *em = geist_last_create_error();
    const bool ok = s != GEIST_OK && (needle == nullptr || (em != nullptr && strstr(em, needle)));
    if (!ok)
        fprintf(stderr, "  %s: status %d, error '%s'\n", what, (int) s, em ? em : "");
    geist_model_destroy(m);
    geist_backend_destroy(be);
    free(g.b);
    return geist_expect(ok, what);
}

/* The fixture on cpu_scalar, `hot` experts on GPU backend `gpu`; its logits
 * against `ref`. Returns failures, or -1 when `gpu` is not built. */
static int check_hot(const char *gpu, size_t hot, const float ref[static VOCAB]) {
    struct geist_backend *cpu = nullptr, *gbe = nullptr;
    if (geist_backend_create(gpu, nullptr, nullptr, &gbe) != GEIST_OK) {
        printf("skip hot experts on %s: not in this build\n", gpu);
        return -1;
    }
    int fails = geist_expect(geist_backend_create("cpu_scalar", nullptr, nullptr, &cpu) == GEIST_OK,
                             "cpu_scalar");
    char n[16];
    snprintf(n, sizeof n, "%zu", hot);
    setenv("GEIST_MOE_HOT", n, 1);
    struct tf_buf                   g  = mf_qwen3moe_gguf(&GEOM);
    const struct geist_session_opts lo = {.split_backend = gbe, .split_layers = 1};
    struct geist_model             *m  = nullptr;
    struct geist_session           *s  = nullptr;
    geist_token_t                   ids[N_PROMPT];
    for (size_t i = 0; i < N_PROMPT; i++)
        ids[i] = (geist_token_t) ((i * 37 + 5) % VOCAB);
    float out[VOCAB];
    if (fails > 0 || geist_model_load_from_memory_with_opts(g.b, g.n, cpu, &lo, &m) != GEIST_OK ||
        geist_session_create(m, cpu, &OPTS, &s) != GEIST_OK ||
        !logits_after(s, N_PROMPT, ids, 1, out)) {
        fprintf(stderr, "%s / %s\n", geist_backend_errmsg(cpu), geist_backend_errmsg(gbe));
        fails += geist_expect(false, "the fixture runs with hot experts");
    } else {
        const struct transformer_arch_state *st = geist_model_internal_arch_meta(m);
        const struct transformer_moe_gpu    *mg = st->moe_gpu;
        const double                         d  = rel_l2(out, ref);
        printf("%s, %zu hot experts: %zu placed, %llu rows on the GPU, %llu on the CPU, relative "
               "L2 %.3g\n",
               gpu,
               hot,
               mg != nullptr ? mg->n_hot : 0,
               mg != nullptr ? (unsigned long long) mg->rows_gpu : 0ULL,
               mg != nullptr ? (unsigned long long) mg->rows_cpu : 0ULL,
               d);
        fails += geist_expect(mg != nullptr && mg->n_hot == hot, "the experts are placed");
        fails += geist_expect(mg != nullptr && mg->rows_gpu > 0, "rows run on the GPU");
        fails += geist_expect(hot == 16 || (mg != nullptr && mg->rows_cpu > 0),
                              "the cold experts' rows run on the CPU");
        fails += geist_expect(d <= 1e-3, "the logits match cpu_scalar's");
    }
    unsetenv("GEIST_MOE_HOT");
    geist_session_destroy(s);
    geist_model_destroy(m);
    free(g.b);
    geist_backend_destroy(cpu);
    geist_backend_destroy(gbe);
    return fails;
}

/* The fixture loaded on GPU backend `gpu`, `hot` experts with a copy there;
 * its logits (chunked and token by token) against `ref`. Returns failures,
 * or -1 when `gpu` is not built. */
static int check_on_gpu(const char *gpu, size_t hot, const float ref[static VOCAB]) {
    struct geist_backend *gbe = nullptr;
    if (geist_backend_create(gpu, nullptr, nullptr, &gbe) != GEIST_OK) {
        printf("skip the model on %s: not in this build\n", gpu);
        return -1;
    }
    char n[16];
    snprintf(n, sizeof n, "%zu", hot);
    setenv("GEIST_MOE_HOT", n, 1);
    struct tf_buf         g     = mf_qwen3moe_gguf(&GEOM);
    struct geist_model   *m     = nullptr;
    struct geist_session *s     = nullptr;
    int                   fails = 0;
    geist_token_t         ids[N_PROMPT];
    for (size_t i = 0; i < N_PROMPT; i++)
        ids[i] = (geist_token_t) ((i * 37 + 5) % VOCAB);
    float chunked[VOCAB], single[VOCAB];
    if (geist_model_load_from_memory(g.b, g.n, gbe, &m) != GEIST_OK ||
        geist_session_create(m, gbe, &OPTS, &s) != GEIST_OK ||
        !logits_after(s, N_PROMPT, ids, N_PROMPT, chunked) ||
        !logits_after(s, N_PROMPT, ids, 1, single)) {
        fprintf(stderr, "%s\n", geist_backend_errmsg(gbe));
        fails += geist_expect(false, "the fixture runs on the GPU backend");
    } else {
        const struct transformer_arch_state *st = geist_model_internal_arch_meta(m);
        const struct transformer_moe_gpu    *mg = st->moe_gpu;
        const double                         c1 = cosine(chunked, ref), c2 = cosine(single, ref);
        printf("%s main, %zu hot: experts on %s, %zu placed, %llu rows on the GPU, cosine %.6f "
               "(chunked) %.6f (token by token)\n",
               gpu,
               hot,
               st->moe_cpu != nullptr ? geist_backend_name(st->moe_cpu) : "-",
               mg != nullptr ? mg->n_hot : 0,
               mg != nullptr ? (unsigned long long) mg->rows_gpu : 0ULL,
               c1,
               c2);
        fails += geist_expect(st->moe_cpu != nullptr, "the experts have a CPU home");
        fails += geist_expect(hot == 0 || (mg != nullptr && mg->n_hot == hot && mg->rows_gpu > 0),
                              "the hot experts are placed and run");
        fails += geist_expect(c1 >= 0.999 && c2 >= 0.999, "the logits match cpu_scalar's");

        /* #164: only the experts without a GPU copy are resolved on the
         * CPU; a session that cannot get its GPU buffers is refused. */
        struct transformer_arch_session *a       = geist_session_internal_arch_session(s);
        bool                             cold_ok = true, hot_bare = true;
        for (size_t l = 0; l < GEOM.layers; l++) {
            for (size_t e = 0; e < GEOM.experts; e++) {
                const struct geist_weight *w = &st->layers[l].moe_gate_w[e];
                if (moe_gpu_hot(a, l, e))
                    hot_bare = hot_bare && w->linear_m1 == nullptr;
                else
                    cold_ok = cold_ok && w->linear_m1 != nullptr;
            }
        }
        fails += geist_expect(cold_ok, "the experts without a GPU copy run on the CPU");
        fails += geist_expect(hot_bare, "the experts with a GPU copy have no CPU copy");
        if (hot > 0) {
            moe_gpu_session_free(a);
            heap_fail_allocations(true);
            const enum geist_status fs = moe_gpu_session_alloc(a);
            heap_fail_allocations(false);
            fails += geist_expect(fs != GEIST_OK,
                                  "no GPU buffers for a session: refused, the hot experts have no "
                                  "CPU copy");
            moe_gpu_session_free(a);
            float again[VOCAB];
            fails += geist_expect(moe_gpu_session_alloc(a) == GEIST_OK &&
                                          logits_after(s, N_PROMPT, ids, N_PROMPT, again) &&
                                          cosine(again, ref) >= 0.999,
                                  "with its buffers back the session runs again");
        }
        /* #201: with every expert on a GPU that routes, decode never leaves it. */
        if (mg != nullptr && mg->any_device)
            fails += geist_expect(hot == 16 && mg->n_device > 0,
                                  "the token-by-token run routed on the device");
    }
    unsetenv("GEIST_MOE_HOT");
    geist_session_destroy(s);
    geist_model_destroy(m);
    free(g.b);
    geist_backend_destroy(gbe);
    return fails;
}

int main(void) {
    int       fails = 0;
    float     scalar[VOCAB], other[VOCAB];
    const int sc = check_backend("cpu_scalar", scalar);
    fails += sc > 0 ? sc : 0;
    const char *const fast[] = {"cpu_neon", "cpu_x86"};
    for (size_t b = 0; b < 2; b++) {
        const int r = check_backend(fast[b], other);
        if (r < 0)
            continue;
        fails += r;
        if (sc == 0) {
            const double d = rel_l2(other, scalar), c = cosine(other, scalar);
            printf("%s vs cpu_scalar: relative L2 %.3g, cosine %.6f\n", fast[b], d, c);
            const bool exact_f32 = strcmp(fast[b], "cpu_neon") == 0;
            fails += geist_expect(exact_f32 ? d <= 1e-3 : c >= 0.999,
                                  "a fast backend matches cpu_scalar within its F32 tolerance");
        }
    }

    if (sc == 0) {
        const char *const gpus[] = {"vulkan", "metal"};
        for (size_t b = 0; b < 2; b++) {
            for (size_t hot = 8; hot <= 16; hot += 8) {
                const int r = check_hot(gpus[b], hot, scalar);
                fails += r > 0 ? r : 0;
                if (r < 0)
                    break;
            }
            for (size_t hot = 0; hot <= 16; hot += 8) {
                const int r = check_on_gpu(gpus[b], hot, scalar);
                fails += r > 0 ? r : 0;
                if (r < 0)
                    break;
            }
        }
    }

    struct mf_qwen3moe c = GEOM;
    c.drop               = "blk.1.ffn_up_exps.weight";
    fails += check_refused("a missing expert tensor is refused by name", &c, "ffn_up_exps");
    c        = GEOM;
    c.gating = 2;
    fails += check_refused("sigmoid gating is refused", &c, "expert_gating_func");
    c               = GEOM;
    c.shared_expert = true;
    fails += check_refused("a shared expert is refused", &c, "shared expert");
    c      = GEOM;
    c.used = 9;
    fails += check_refused("more experts used than exist is refused", &c, nullptr);

    if (fails == 0)
        printf("PASS: qwen3moe prefill shapes, reset reuse, fault injection, load refusals\n");
    return fails == 0 ? 0 : 1;
}
