/*
 * tests/spec_tie.h — is a speculative-vs-sequential divergence a near-tie?
 *
 * The verify pass scores K draft rows in one M>1 forward; decode_step runs
 * M=1 forwards. The two take different kernels and round differently
 * (by up to about a logit on quantized paths; see SPEC_TIE_LOGITS), so on a
 * prompt where the model's top two tokens are nearly equal the greedy picks
 * can split. That is no speculative bug. These helpers replay the reference
 * up to the first divergence and report the logit margin there.
 */
#ifndef GEIST_TESTS_SPEC_TIE_H
#define GEIST_TESTS_SPEC_TIE_H

#include <geist.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stddef.h>

/* |logit(a) - logit(b)| of the sequential reference before its step i
 * (prompt prefilled, then i decode_steps), or NAN if anything fails. */
static inline float spec_ref_margin(struct geist_model              *model,
                                    struct geist_backend            *be,
                                    const struct geist_session_opts *opts,
                                    size_t                           prompt_n,
                                    const geist_token_t             *prompt,
                                    size_t                           i,
                                    geist_token_t                    a,
                                    geist_token_t                    b) {
    struct geist_session *s = nullptr;
    float                 m = NAN;
    if (geist_session_create(model, be, opts, &s) != GEIST_OK) {
        return m;
    }
    bool ok = geist_session_prefill_tokens(s, prompt_n, prompt) == GEIST_OK;
    for (size_t j = 0; ok && j < i; j++) {
        geist_token_t t;
        ok = geist_session_decode_step(s, &t) == GEIST_OK;
    }
    size_t       n = 0;
    const float *l = ok ? geist_session_peek_logits(&n, s) : nullptr;
    if (l != nullptr && a >= 0 && b >= 0 && (size_t) a < n && (size_t) b < n) {
        m = fabsf(l[a] - l[b]);
    }
    geist_session_destroy(s);
    return m;
}

/* Below this logit margin the greedy pick is decided by numerics, not by
 * the speculative logic. Measured splits (#32), all with cpu_scalar exact
 * where the logic is concerned:
 *   0.038  Llama 3.2 3B, cpu_neon, f32 KV (verify vs decode_step)
 *   0.157  Qwen3 0.6B, cpu_neon, INT4 KV
 *   0.37   Llama 3.2 3B, cpu_scalar, INT4 KV + rotation (batched vs single
 *          attention over the quantized cache)
 *   0.505  the same on cpu_neon
 *   1.06   Llama 3.2 3B, cpu_neon, re-prefill vs verify: the M>1 and M=1
 *          int8 kernels quantize activations differently, and the 128k-vocab
 *          output projection adds that up; passes on cpu_scalar.
 * Run with GEIST_BACKEND=cpu_scalar to check the logic with no headroom. */
#define SPEC_TIE_LOGITS 1.5f

/* 0 when spec equals ref over n tokens, or when their first difference
 * is a near-tie of the reference's logits (reported, and the streams are
 * not compared past it: they legitimately diverge there); else 1. */
static inline int spec_compare(struct geist_model              *model,
                               struct geist_backend            *be,
                               const struct geist_session_opts *opts,
                               size_t                           prompt_n,
                               const geist_token_t             *prompt,
                               size_t                           n,
                               const geist_token_t             *ref,
                               const geist_token_t             *spec,
                               const char                      *label) {
    for (size_t i = 0; i < n; i++) {
        if (spec[i] == ref[i]) {
            continue;
        }
        const float m = spec_ref_margin(model, be, opts, prompt_n, prompt, i, ref[i], spec[i]);
        if (m < SPEC_TIE_LOGITS) {
            fprintf(stderr,
                    "[%s] note @ %zu: ref=%d spec=%d is a near-tie (margin %.3g logits); "
                    "not compared further\n",
                    label,
                    i,
                    ref[i],
                    spec[i],
                    (double) m);
            return 0;
        }
        fprintf(stderr,
                "[%s] FAIL @ %zu: ref=%d spec=%d, margin %.3g logits\n",
                label,
                i,
                ref[i],
                spec[i],
                (double) m);
        return 1;
    }
    return 0;
}

#endif /* GEIST_TESTS_SPEC_TIE_H */
