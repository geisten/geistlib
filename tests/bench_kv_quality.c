/*
 * bench_kv_quality — KV-quant quality sweep via top-1 accuracy AND KL
 * divergence vs an FP32 reference.
 *
 * Tokenizes a fixed paragraph, then for each KV mode teacher-forces the
 * sequence one token at a time through the public session API, reading the
 * full next-token distribution at every position with peek_logits. Two
 * quality signals per mode:
 *
 *   top-1 acc : fraction of positions where the mode's argmax matches the
 *               actual next token. Coarse — saturates, sub-1% noise.
 *   mean KL   : mean KL(P_fp32 || P_mode) over all positions, in nats.
 *               Sensitive — resolves quality differences top-1 cannot.
 *
 * Modes: FP32 / INT8 / INT8+ROT / INT4 / INT4+ROT / INT2 / INT2+ROT / KIVI.
 * INT4/INT2 are the issue-#61 low-bit quality-sims (GEIST_KV_INT4 /
 * GEIST_KV_QBITS) that reuse the INT8 storage path with an N-bit grid. The
 * KV mode is resolved per session from GEIST_KV_* env vars, so the model
 * loads once and each mode just creates a session with those vars set. An
 * FP32 reference session runs in lockstep to supply P_fp32 for the KL.
 *
 * SKIPs if no GGUF or tokenizer is reachable.
 */
#define GEIST_INTERNAL_ENGINE_LAYER
#define GEIST_INTERNAL_ARCH_LAYER

/* setenv/unsetenv need a POSIX feature macro on glibc (Pi5); no-op on macOS.
 * Must precede any system header. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "test_helpers.h"

#include "src/engine/gguf_tokenizer.h"
#include "src/engine/model.h"
#include "src/engine/sp_bpe_tokenizer.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ~400 tokens so the KIVI drained 2-bit path is actually exercised
 * (KIVI_K_GROUP_SIZE=128: a drain triggers around token 128 and again
 * around 256, giving us ≥256 tokens of *drained 2-bit* attention reads
 * mixed with the residual). */
static const char *DEFAULT_TEXT =
        "Raspberry Pi is a series of small single-board computers developed "
        "in the United Kingdom by the Raspberry Pi Foundation. The original "
        "model became more popular than anticipated, selling outside its "
        "target market for uses such as robotics. It is widely used in many "
        "areas, such as for weather monitoring, because of its low cost, "
        "modularity, and open design. After the release of the second board "
        "type, the Raspberry Pi Foundation set up a new entity, Raspberry "
        "Pi Trading, to handle commercial operations while the original "
        "Foundation focused on educational outreach. The boards run a "
        "Debian-based operating system called Raspberry Pi OS as the "
        "official supported distribution. The Pi 5 was released in 2023 "
        "with a Broadcom BCM2712 system-on-chip featuring a quad-core "
        "Arm Cortex-A76 processor clocked at 2.4 gigahertz, paired with "
        "an 800 megahertz VideoCore VII graphics processor. The board "
        "includes options for 4 or 8 gigabytes of LPDDR4X memory, two "
        "micro-HDMI ports supporting dual 4K displays, two USB 3.0 ports, "
        "two USB 2.0 ports, gigabit Ethernet, and Wi-Fi 5 plus Bluetooth "
        "5.0 wireless connectivity. A new dedicated power button replaces "
        "the previous practice of pulling the power cable to shut down. "
        "An internal real-time clock with battery backup keeps the system "
        "time accurate across reboots and unpowered intervals. The board "
        "also exposes a PCI Express interface via a flat ribbon connector "
        "for high-speed peripherals such as NVMe storage.";

/* One KV mode: which GEIST_KV_* vars to set before creating its session.
 * A nullptr field means unset that var. */
struct kv_mode_cfg {
    const char *label;
    const char *int8;  /* GEIST_KV_INT8 */
    const char *kivi;  /* GEIST_KV_KIVI */
    const char *rot;   /* GEIST_KV_ROT  */
    const char *int4;  /* GEIST_KV_INT4 (4-bit quality-sim on the INT8 path) */
    const char *qbits; /* GEIST_KV_QBITS (N-bit quality-sim; overrides int4) */
};

static const struct kv_mode_cfg MODES[] = {
        {"FP32", "0", nullptr, nullptr, nullptr, nullptr},
        {"INT8", "1", nullptr, "0", nullptr, nullptr},
        {"INT8+ROT", "1", nullptr, "1", nullptr, nullptr},
        {"INT4", nullptr, nullptr, "0", "1", nullptr},
        {"INT4+ROT", nullptr, nullptr, "1", "1", nullptr},
        {"INT2", nullptr, nullptr, "0", nullptr, "2"},
        {"INT2+ROT", nullptr, nullptr, "1", nullptr, "2"},
        {"KIVI", nullptr, "1", nullptr, nullptr, nullptr},
};

static size_t mode_index(const char *label) {
    for (size_t i = 0; i < sizeof(MODES) / sizeof(MODES[0]); i++)
        if (strcmp(MODES[i].label, label) == 0)
            return i;
    return 0; /* labels are compile-time constants — unreachable */
}

static void set_or_unset(const char *name, const char *val) {
    if (val != nullptr)
        setenv(name, val, 1);
    else
        unsetenv(name);
}

static void apply_mode_env(const struct kv_mode_cfg *m) {
    set_or_unset("GEIST_KV_INT8", m->int8);
    set_or_unset("GEIST_KV_KIVI", m->kivi);
    set_or_unset("GEIST_KV_ROT", m->rot);
    set_or_unset("GEIST_KV_INT4", m->int4);
    set_or_unset("GEIST_KV_QBITS", m->qbits);
}

static uint32_t argmax_f32(const float *x, size_t n) {
    uint32_t best = 0;
    for (size_t i = 1; i < n; i++)
        if (x[i] > x[best])
            best = (uint32_t) i;
    return best;
}

/* KL(P_ref || P_mode) in nats over the shared finite support. Some models
 * (e.g. BitNet) leave non-finite garbage in unused vocab slots; a slot that
 * is non-finite in EITHER distribution is dropped from both before
 * normalizing, so the two softmaxes are over the same token set. Streaming
 * log-sum-exp — no full-vocab buffers. */
static double kl_div(const float *lr, const float *lm, size_t n) {
    bool   have = false;
    double maxr = 0.0, maxm = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(lr[i]) || !isfinite(lm[i]))
            continue;
        if (!have || lr[i] > maxr)
            maxr = lr[i];
        if (!have || lm[i] > maxm)
            maxm = lm[i];
        have = true;
    }
    if (!have)
        return 0.0;
    double sr = 0.0, sm = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(lr[i]) || !isfinite(lm[i]))
            continue;
        sr += exp((double) lr[i] - maxr);
        sm += exp((double) lm[i] - maxm);
    }
    const double logZr = maxr + log(sr);
    const double logZm = maxm + log(sm);
    double       kl    = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(lr[i]) || !isfinite(lm[i]))
            continue;
        const double lp_r = (double) lr[i] - logZr;
        kl += exp(lp_r) * (lp_r - ((double) lm[i] - logZm));
    }
    return kl < 0.0 ? 0.0 : kl; /* clamp fp noise on identical dists */
}

/* Teacher-force `ids` through both sessions in lockstep; fill top-1 acc, mean
 * KL(ref||mode), and mean KL split at the sequence midpoint (issue #69: does
 * quality degrade with context depth?). Returns false on a session error. */
static bool score_mode(struct geist_session *ref,
                       struct geist_session *mode,
                       const uint32_t       *ids,
                       size_t                n_ids,
                       double               *acc_out,
                       double               *kl_out,
                       double               *kl_early_out,
                       double               *kl_late_out) {
    if (geist_session_reset(ref) != GEIST_OK || geist_session_reset(mode) != GEIST_OK)
        return false;
    const geist_token_t t0 = (geist_token_t) ids[0];
    if (geist_session_prefill_tokens(ref, 1, &t0) != GEIST_OK ||
        geist_session_prefill_tokens(mode, 1, &t0) != GEIST_OK)
        return false;

    const size_t mid      = n_ids / 2; /* positions < mid = early, >= mid = late */
    double       kl_early = 0.0, kl_late = 0.0;
    size_t       n_early = 0, n_late = 0;
    size_t       correct = 0;
    for (size_t i = 1; i < n_ids; i++) {
        size_t       nr = 0, nm = 0;
        const float *lr = geist_session_peek_logits(ref, &nr);
        const float *lm = geist_session_peek_logits(mode, &nm);
        if (lr == nullptr || lm == nullptr || nr != nm || nr == 0)
            return false;
        const double kl = kl_div(lr, lm, nr);
        if (i < mid) {
            kl_early += kl;
            n_early++;
        } else {
            kl_late += kl;
            n_late++;
        }
        if (argmax_f32(lm, nm) == ids[i])
            correct++;
        const geist_token_t ti = (geist_token_t) ids[i];
        if (geist_session_prefill_tokens(ref, 1, &ti) != GEIST_OK ||
            geist_session_prefill_tokens(mode, 1, &ti) != GEIST_OK)
            return false;
    }
    const size_t n_eval = n_early + n_late;
    *acc_out            = n_eval > 0 ? (double) correct / (double) n_eval : 0.0;
    *kl_out             = n_eval > 0 ? (kl_early + kl_late) / (double) n_eval : 0.0;
    *kl_early_out       = n_early > 0 ? kl_early / (double) n_early : 0.0;
    *kl_late_out        = n_late > 0 ? kl_late / (double) n_late : 0.0;
    return true;
}

/* Tokenize via the sp_bpe tokenizer, else the GGUF-embedded one. */
static bool tokenize(struct geist_model *model,
                     const char         *text,
                     uint32_t          **ids_out,
                     size_t             *n_out,
                     bool               *had_tokenizer) {
    *had_tokenizer               = false;
    struct sp_bpe_tokenizer *tok = geist_model_internal_tokenizer(model);
    if (tok != nullptr) {
        *had_tokenizer = true;
        return sp_bpe_tokenizer_encode(tok, text, ids_out, n_out);
    }
    struct gguf_tokenizer *gtok = geist_model_internal_gguf_tokenizer(model);
    if (gtok == nullptr)
        return false;
    *had_tokenizer   = true;
    const size_t cap = strlen(text) + 16;
    int32_t     *tmp = (int32_t *) malloc(cap * sizeof(int32_t));
    bool         ok  = false;
    if (tmp != nullptr && gguf_tokenizer_encode(gtok, text, tmp, cap, n_out)) {
        *ids_out = (uint32_t *) malloc(*n_out * sizeof(uint32_t));
        if (*ids_out != nullptr) {
            for (size_t i = 0; i < *n_out; i++)
                (*ids_out)[i] = (uint32_t) tmp[i];
            ok = true;
        }
    }
    free(tmp);
    return ok;
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : geist_test_find_gguf();
    GEIST_SKIP_IF(model_path == nullptr, "no GGUF model found — pass path or set GEIST_GGUF_PATH");
    const char *text = (argc > 2) ? argv[2] : DEFAULT_TEXT;

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK)
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    struct geist_model *model = nullptr;
    s                         = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr, "model_load(%s): %s\n", model_path, geist_status_to_string(s));
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    uint32_t  *ids           = nullptr;
    size_t     n_ids         = 0;
    bool       had_tokenizer = false;
    const bool enc_ok        = tokenize(model, text, &ids, &n_ids, &had_tokenizer);
    if (!had_tokenizer) {
        free(ids);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        GEIST_SKIP_IF(true, "model carries no usable tokenizer");
    }
    if (!enc_ok || n_ids < 2) {
        fprintf(stderr, "tokenizer encode failed or text too short (n_ids=%zu)\n", n_ids);
        free(ids);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    /* FP32 reference session, created once with FP32 env, reused for every
     * mode's KL. Greedy opts (all-zero) → AUTO mode, env resolves the rest. */
    struct geist_session_opts opts = {0};
    apply_mode_env(&MODES[mode_index("FP32")]);
    struct geist_session *ref = nullptr;
    if (geist_session_create(model, be, &opts, &ref) != GEIST_OK) {
        fprintf(stderr, "ref session create failed\n");
        free(ids);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    printf("model:    %s\n", model_path);
    printf("backend:  %s\n", geist_backend_name(be));
    printf("n_tokens: %zu (n_eval=%zu)\n\n", n_ids, n_ids - 1);

    const size_t n_modes = sizeof(MODES) / sizeof(MODES[0]);
    double       acc[sizeof(MODES) / sizeof(MODES[0])];
    double       kl[sizeof(MODES) / sizeof(MODES[0])];
    printf("%-10s  %-7s  %-9s  %-9s  %s\n",
           "kv_mode",
           "top-1",
           "mean KL",
           "KL early",
           "KL late  (fp32||·, nats; #69 depth check)");
    printf("%-10s  %-7s  %-9s  %-9s  %s\n", "-------", "-----", "-------", "--------", "-------");
    for (size_t m = 0; m < n_modes; m++) {
        apply_mode_env(&MODES[m]);
        struct geist_session *sess     = nullptr;
        double                kl_early = 0.0, kl_late = 0.0;
        if (geist_session_create(model, be, &opts, &sess) != GEIST_OK ||
            !score_mode(ref, sess, ids, n_ids, &acc[m], &kl[m], &kl_early, &kl_late)) {
            fprintf(stderr, "mode %s failed\n", MODES[m].label);
            geist_session_destroy(sess);
            geist_session_destroy(ref);
            free(ids);
            geist_model_destroy(model);
            geist_backend_destroy(be);
            return GEIST_TEST_FAIL;
        }
        geist_session_destroy(sess);
        printf("%-10s  %.4f   %.5f    %.5f    %.5f\n",
               MODES[m].label,
               acc[m],
               kl[m],
               kl_early,
               kl_late);
    }
    geist_session_destroy(ref);

    /* Issue #61 headlines: rotation's effect where it matters (low bit),
     * measured by KL reduction (lower KL = closer to FP32). */
    const double kl_i4   = kl[mode_index("INT4")];
    const double kl_i4r  = kl[mode_index("INT4+ROT")];
    const double kl_i2   = kl[mode_index("INT2")];
    const double kl_i2r  = kl[mode_index("INT2+ROT")];
    const double kl_kivi = kl[mode_index("KIVI")];
    printf("\n");
    printf("4-bit: rotation cuts KL %.5f → %.5f  (%+.0f%%)\n",
           kl_i4,
           kl_i4r,
           kl_i4 > 0.0 ? 100.0 * (kl_i4r - kl_i4) / kl_i4 : 0.0);
    printf("2-bit: rotation cuts KL %.5f → %.5f  (%+.0f%%)   vs KIVI(asym) %.5f\n",
           kl_i2,
           kl_i2r,
           kl_i2 > 0.0 ? 100.0 * (kl_i2r - kl_i2) / kl_i2 : 0.0,
           kl_kivi);

    free(ids);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return GEIST_TEST_PASS;
}
