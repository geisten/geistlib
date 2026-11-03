/*
 * test_llama_rope_freqs_unit - Llama 3.x RoPE frequency factors (#379).
 *
 * Llama 3.1-3.3 GGUFs carry rope_freqs.weight: one divisor per rotary pair
 * (rope_scaling type "llama3"), which llama.cpp passes to every rope as
 * ggml's freq_factors. Pair i then turns at theta^(-2i/d) / factor[i].
 *   - rope_compute_ff_at against the definition in double, with Llama
 *     3.2's own factors (computed as transformers does) out to position
 *     20000, past the 8192 the model was trained on;
 *   - the llama family reads the tensor: an in-memory llama with factors
 *     1, 1.5, 2, 2.5 holds them, gives other logits than without, and all
 *     ones give the logits of a file without the tensor;
 *   - every backend in the build matches cpu_scalar with the factors
 *     (cpu_neon and Metal by relative L2 1e-3, others by cosine 0.999);
 *   - a factor tensor that is not one per pair is refused.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "../src/archs/transformer/arch_state.h"
#include "../src/engine/model.h"
#include "gemma4_kernels.h"
#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { VOCAB = 96, N_PROMPT = 21 };

static const struct mf_llama GEOM = {.layers   = 2,
                                     .d_model  = 64,
                                     .heads    = 8,
                                     .kv_heads = 4,
                                     .ffn      = 96,
                                     .vocab    = VOCAB,
                                     .context  = 256,
                                     .seed     = 7};

static const struct geist_session_opts OPTS = {.top_p = 1.0f, .m_max = 8, .max_seq_len = 256};

/* ---- the table against the definition ---------------------------------- */

static int check_table(void) {
    /* Llama 3.2 3B: head_dim 128, theta 500000, rope_scaling factor 32,
     * low/high_freq_factor 1/4, original context 8192 (transformers'
     * _compute_llama3_parameters, which the converter writes as 1/scale). */
    enum { HD = 128, HALF = HD / 2 };
    const double theta = 500000.0, factor = 32.0, lo = 1.0, hi = 4.0, orig = 8192.0;
    float        ff[HALF];
    for (size_t i = 0; i < HALF; i++) {
        const double freq = pow(theta, -(double) (2 * i) / HD);
        const double wl   = 6.283185307179586 / freq;
        double       f    = 1.0;
        if (wl > orig / lo) {
            f = factor;
        } else if (wl >= orig / hi) {
            const double smooth = (orig / wl - lo) / (hi - lo);
            f                   = 1.0 / ((1.0 - smooth) / factor + smooth);
        }
        ff[i] = (float) f;
    }
    static const size_t POS[] = {0, 1, 7, 511, 4096, 8191, 8192, 12345, 19999};
    float               cos_t[HD], sin_t[HD], plain_c[HD], plain_s[HD];
    double              worst = 0.0, moved = 0.0;
    int                 fails = 0;
    for (size_t k = 0; k < sizeof POS / sizeof POS[0]; k++) {
        rope_compute_ff_at(POS[k], 1, HD, (float) theta, ff, cos_t, sin_t);
        rope_compute_at(POS[k], 1, HD, HD, false, (float) theta, plain_c, plain_s);
        for (size_t i = 0; i < HALF; i++) {
            const double angle = (double) POS[k] * pow(theta, -(double) (2 * i) / HD) / ff[i];
            /* float: ggml's angle is the position multiplied down i times,
             * each step one rounding of the running product. */
            const double tol = 1e-6 + 2.0 * (double) (i + 2) * FLT_EPSILON * angle;
            const double dc = fabs(cos_t[i] - cos(angle)), ds = fabs(sin_t[i] - sin(angle));
            const bool   ok = dc <= tol && ds <= tol && cos_t[i] == cos_t[i + HALF] &&
                              sin_t[i] == sin_t[i + HALF];
            if (!ok && fails < 5) {
                fprintf(stderr,
                        "FAIL: pos %zu pair %zu: cos %.9g want %.9g, sin %.9g want %.9g\n",
                        POS[k],
                        i,
                        cos_t[i],
                        cos(angle),
                        sin_t[i],
                        sin(angle));
            }
            fails += !ok;
            worst = fmax(worst, fmax(dc, ds) / tol);
            moved = fmax(moved, fabs((double) plain_c[i] - cos_t[i]));
        }
    }
    printf("rope_compute_ff_at vs double: worst error %.3g of its bound; the factors move cos "
           "by up to %.3g\n",
           worst,
           moved);
    fails += geist_expect(moved > 0.5, "Llama 3.2's factors change the table");
    return fails;
}

/* ---- the llama family -------------------------------------------------- */

static bool logits_of(struct geist_session *s, float out[static VOCAB]) {
    geist_token_t ids[N_PROMPT];
    for (size_t i = 0; i < N_PROMPT; i++)
        ids[i] = (geist_token_t) ((i * 37 + 5) % VOCAB);
    if (geist_session_reset(s) != GEIST_OK ||
        geist_session_prefill_tokens(s, N_PROMPT, ids) != GEIST_OK)
        return false;
    size_t       nl = 0;
    const float *lg = geist_session_peek_logits(&nl, s);
    if (lg == nullptr || nl != VOCAB)
        return false;
    memcpy(out, lg, VOCAB * sizeof(float));
    return true;
}

static double rel_l2(const float a[static VOCAB], const float b[static VOCAB]) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < VOCAB; i++) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return sqrt(num / (den > 0.0 ? den : 1.0));
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

/* The fixture with n_ff factors (0: none) on backend `name`: its logits in
 * out. Returns 0, failures, or -1 when the backend is not built. */
static int run(const char *name, uint32_t n_ff, bool ones, float out[static VOCAB]) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(name, nullptr, nullptr, &be) != GEIST_OK)
        return -1;
    struct mf_llama c           = GEOM;
    c.rope_freqs                = n_ff;
    c.rope_freqs_ones           = ones;
    struct tf_buf         g     = mf_llama_gguf(&c);
    struct geist_model   *m     = nullptr;
    struct geist_session *s     = nullptr;
    int                   fails = 0;
    if (geist_model_load_from_memory(g.b, g.n, be, &m) != GEIST_OK ||
        geist_session_create(m, be, &OPTS, &s) != GEIST_OK || !logits_of(s, out)) {
        fails += geist_expect(false, "the llama fixture runs");
    } else if (n_ff != 0) {
        const struct transformer_arch_session *a  = geist_session_internal_arch_session(s);
        const struct transformer_arch_state   *st = a->model;
        bool                                   ok = st->n_rope_freq_factors == n_ff;
        for (size_t i = 0; ok && i < n_ff; i++)
            ok = st->rope_freq_factors[i] == (ones ? 1.0f : 1.0f + 0.5f * (float) i);
        fails += geist_expect(ok, "the model holds the file's factors");
    }
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    free(g.b);
    return fails;
}

static int check_family(void) {
    const uint32_t pairs = GEOM.d_model / GEOM.heads / 2;
    float          none[VOCAB], ones[VOCAB], step[VOCAB], other[VOCAB];
    int            fails = 0;
    if (run("cpu_scalar", 0, false, none) != 0 || run("cpu_scalar", pairs, true, ones) != 0 ||
        run("cpu_scalar", pairs, false, step) != 0)
        return geist_expect(false, "cpu_scalar runs the fixture with and without factors");
    const double d_ones = rel_l2(ones, none), d_step = rel_l2(step, none);
    printf("cpu_scalar: factors 1 vs none: relative L2 %.3g; factors 1, 1.5, .. vs none: %.3g\n",
           d_ones,
           d_step);
    fails += geist_expect(d_ones <= 1e-5, "factors of 1 leave the logits as without the tensor");
    fails += geist_expect(d_step >= 1e-2, "the factors reach the forward pass");

    const char *const more[] = {"cpu_neon", "cpu_x86", "metal", "vulkan"};
    for (size_t b = 0; b < sizeof more / sizeof more[0]; b++) {
        const int r = run(more[b], pairs, false, other);
        if (r < 0) {
            printf("skip %s: not in this build\n", more[b]);
            continue;
        }
        fails += r;
        if (r != 0)
            continue;
        const double d = rel_l2(other, step), cs = cosine(other, step);
        printf("%s vs cpu_scalar with factors: relative L2 %.3g, cosine %.6f\n", more[b], d, cs);
        const bool exact = strcmp(more[b], "cpu_neon") == 0 || strcmp(more[b], "metal") == 0;
        fails += geist_expect(exact ? d <= 1e-3 : cs >= 0.999,
                              "a backend applies the factors as cpu_scalar does");
    }
    return fails;
}

static int check_refuses_wrong_count(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK)
        return geist_expect(false, "cpu_scalar backend");
    struct mf_llama c     = GEOM;
    c.rope_freqs          = GEOM.d_model / GEOM.heads; /* one per dim, not per pair */
    struct tf_buf       g = mf_llama_gguf(&c);
    struct geist_model *m = nullptr;
    const int fails       = geist_expect(geist_model_load_from_memory(g.b, g.n, be, &m) != GEIST_OK,
                                         "factors that are not one per pair are refused");
    geist_model_destroy(m);
    geist_backend_destroy(be);
    free(g.b);
    return fails;
}

int main(void) {
    int fails = check_table();
    fails += check_family();
    fails += check_refuses_wrong_count();
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
