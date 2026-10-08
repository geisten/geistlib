/*
 * src/engine/sampler_penalties.c — repetition control (#695): llama.cpp's
 * `penalties` and `dry` samplers, ported operation for operation.
 *
 * Layer: ENGINE. Pure host arithmetic on a copy of the logits.
 *
 * Built with -fno-fast-math -ffp-contract=off (mk/common.mk): the target-wide
 * -ffast-math would turn `logit / repeat_penalty` into a multiply by the
 * reciprocal and fuse the penalty's multiply-add, and llama.cpp does
 * neither — tests/test_sampler_penalties_unit.c compares its floats exactly.
 * It would also fold the isfinite() checks on the caller's options to true.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "sampler.h"

#include "heap.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
/* llama.cpp's defaults (common/common.h) for the fields geist.h maps 0 to. */
constexpr size_t PEN_DEFAULT_LAST_N      = 64;
constexpr float  DRY_DEFAULT_BASE        = 1.75f;
constexpr size_t DRY_DEFAULT_ALLOWED_LEN = 2;
constexpr size_t DRY_DEFAULT_LAST_N      = 64;

enum geist_status geist_sampler_penalty_params_from_opts(struct geist_sampler_penalty_params *out,
                                                         const struct geist_session_opts *opts) {
    *out = (struct geist_sampler_penalty_params) {.repeat = 1.0f};
    if (opts == nullptr) {
        return GEIST_OK;
    }
    const float rep  = opts->repeat_penalty;
    const float base = opts->dry_base;
    if (!isfinite(rep) || rep < 0.0f || !isfinite(opts->frequency_penalty) ||
        !isfinite(opts->presence_penalty) || !isfinite(opts->dry_multiplier) || !isfinite(base) ||
        (base != 0.0f && base < 1.0f) || opts->repeat_last_n < 0 || opts->dry_allowed_length < 0 ||
        opts->dry_penalty_last_n < 0 || (rep != 0.0f && !isfinite(1.0f / rep))) {
        return GEIST_E_INVALID_ARG;
    }
    *out = (struct geist_sampler_penalty_params) {
            .last_n  = opts->repeat_last_n > 0 ? (size_t) opts->repeat_last_n : PEN_DEFAULT_LAST_N,
            .repeat  = rep != 0.0f ? rep : 1.0f,
            .freq    = opts->frequency_penalty,
            .present = opts->presence_penalty,
            .dry_multiplier     = opts->dry_multiplier,
            .dry_base           = base != 0.0f ? base : DRY_DEFAULT_BASE,
            .dry_allowed_length = opts->dry_allowed_length > 0 ? (size_t) opts->dry_allowed_length
                                                               : DRY_DEFAULT_ALLOWED_LEN,
            .dry_last_n         = opts->dry_penalty_last_n > 0 ? (size_t) opts->dry_penalty_last_n
                                                               : DRY_DEFAULT_LAST_N,
    };
    return GEIST_OK;
}

bool geist_sampler_penalties_on(const struct geist_sampler_penalty_params *p) {
    return p->last_n != 0 && !(p->repeat == 1.0f && p->freq == 0.0f && p->present == 0.0f);
}

bool geist_sampler_dry_on(const struct geist_sampler_penalty_params *p) {
    return p->dry_multiplier != 0.0f && p->dry_base >= 1.0f && p->dry_last_n != 0;
}

bool geist_sampler_penalties_active(const struct geist_sampler_penalties *pen) {
    return pen->logits != nullptr;
}

void geist_sampler_penalties_destroy(struct geist_sampler_penalties *pen) {
    if (pen == nullptr) {
        return;
    }
    safe_free((void **) &pen->breakers);
    safe_free((void **) &pen->tails);
    safe_free((void **) &pen->logits);
    safe_free((void **) &pen->counts);
    safe_free((void **) &pen->dry_max);
    safe_free((void **) &pen->window);
    safe_free((void **) &pen->repeat_count);
    safe_free((void **) &pen->touched);
    *pen = (struct geist_sampler_penalties) {0};
}

enum geist_status geist_sampler_penalties_init(struct geist_sampler_penalties            *pen,
                                               const struct geist_sampler_penalty_params *p,
                                               size_t                                     n_vocab,
                                               size_t                                     max_ctx) {
    geist_sampler_penalties_destroy(pen);
    pen->p            = *p;
    const bool pen_on = geist_sampler_penalties_on(p);
    const bool dry_on = geist_sampler_dry_on(p);
    if (!pen_on && !dry_on) {
        return GEIST_OK;
    }
    if (n_vocab == 0 || n_vocab > (size_t) INT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }
    size_t cap = 0;
    if (pen_on) {
        cap = p->last_n;
    }
    if (dry_on && p->dry_last_n > cap) {
        cap = p->dry_last_n;
    }
    /* No window holds more tokens than the context. */
    cap               = cap < max_ctx ? cap : max_ctx;
    cap               = cap > 0 ? cap : 1;
    pen->n_vocab      = n_vocab;
    pen->cap          = cap;
    pen->logits       = heap_alloc_array_aligned(float, n_vocab);
    pen->counts       = heap_calloc_array_aligned(uint32_t, n_vocab);
    pen->dry_max      = dry_on ? heap_calloc_array_aligned(uint32_t, n_vocab) : nullptr;
    pen->window       = heap_alloc_array_aligned(geist_token_t, cap);
    pen->repeat_count = heap_alloc_array_aligned(uint32_t, cap);
    pen->touched      = heap_alloc_array_aligned(geist_token_t, cap);
    if (pen->logits == nullptr || pen->counts == nullptr || (dry_on && pen->dry_max == nullptr) ||
        pen->window == nullptr || pen->repeat_count == nullptr || pen->touched == nullptr) {
        geist_sampler_penalties_destroy(pen);
        return GEIST_E_OOM;
    }
    return GEIST_OK;
}

static int breaker_cmp(const void *a, const void *b) {
    const struct geist_dry_breaker *x = a;
    const struct geist_dry_breaker *y = b;
    if (x->head != y->head) {
        return x->head < y->head ? -1 : 1;
    }
    return x->tail_off < y->tail_off ? -1 : (x->tail_off > y->tail_off ? 1 : 0);
}

enum geist_status geist_sampler_penalties_set_breakers(struct geist_sampler_penalties *pen,
                                                       size_t                          n_words,
                                                       const geist_token_t            *packed) {
    if (pen == nullptr || (n_words > 0 && packed == nullptr) || n_words > UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }
    /* Pass 1: validate and count. */
    size_t n_seq = 0;
    size_t n_ids = 0;
    for (size_t i = 0; i < n_words;) {
        const geist_token_t len = packed[i];
        if (len < 1 || (size_t) len > n_words - i - 1) {
            return GEIST_E_INVALID_ARG;
        }
        n_seq++;
        n_ids += (size_t) len - 1;
        i += 1 + (size_t) len;
    }
    struct geist_dry_breaker *b =
            n_seq > 0 ? heap_alloc_array_aligned(struct geist_dry_breaker, n_seq) : nullptr;
    geist_token_t *t = n_ids > 0 ? heap_alloc_array_aligned(geist_token_t, n_ids) : nullptr;
    if ((n_seq > 0 && b == nullptr) || (n_ids > 0 && t == nullptr)) {
        safe_free((void **) &b);
        safe_free((void **) &t);
        return GEIST_E_OOM;
    }
    /* Pass 2: fill. */
    size_t s = 0;
    size_t o = 0;
    for (size_t i = 0; i < n_words;) {
        const size_t len = (size_t) packed[i];
        b[s]             = (struct geist_dry_breaker) {
                .head = packed[i + 1], .tail_off = (uint32_t) o, .tail_len = (uint32_t) (len - 1)};
        if (len > 1) {
            memcpy(t + o, packed + i + 2, (len - 1) * sizeof *t);
        }
        o += len - 1;
        s++;
        i += 1 + len;
    }
    if (n_seq > 1) {
        qsort(b, n_seq, sizeof *b, breaker_cmp);
    }
    safe_free((void **) &pen->breakers);
    safe_free((void **) &pen->tails);
    pen->breakers   = b;
    pen->n_breakers = n_seq;
    pen->tails      = t;
    return GEIST_OK;
}

/* The breakers whose head is `tok`: [*first, return). */
static size_t
breakers_for(const struct geist_sampler_penalties *pen, geist_token_t tok, size_t *first) {
    size_t lo = 0;
    size_t hi = pen->n_breakers;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (pen->breakers[mid].head < tok) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    *first   = lo;
    size_t e = lo;
    while (e < pen->n_breakers && pen->breakers[e].head == tok) {
        e++;
    }
    return e;
}

/* llama_sampler_penalties_apply over the last `n` tokens `w`. */
static void
penalties_apply(struct geist_sampler_penalties *pen, size_t n, const geist_token_t *w, float *out) {
    const struct geist_sampler_penalty_params *p = &pen->p;
    for (size_t i = 0; i < n; i++) {
        pen->counts[w[i]]++;
    }
    /* Each distinct token once; its count is cleared as it is used, which
     * also leaves `counts` zero for the next call. */
    for (size_t i = 0; i < n; i++) {
        const geist_token_t tok   = w[i];
        const uint32_t      count = pen->counts[tok];
        if (count == 0) {
            continue;
        }
        pen->counts[tok] = 0;
        float l          = out[tok];
        if (l <= 0) {
            l *= p->repeat;
        } else {
            l /= p->repeat;
        }
        l -= (float) count * p->freq + (float) (count > 0) * p->present;
        out[tok] = l;
    }
}

/* llama_sampler_dry_apply over the last `n` tokens `w` (oldest first;
 * llama.cpp's last_tokens.rat(i) is w[n - 1 - i]). */
static void
dry_apply(struct geist_sampler_penalties *pen, size_t n, const geist_token_t *w, float *out) {
    const struct geist_sampler_penalty_params *p       = &pen->p;
    const size_t                               allowed = p->dry_allowed_length;
#define RAT(i) w[n - 1 - (i)]
    if (n <= allowed) {
        return;
    }

    /* Step 1: the most recent restart sequence limits the repeat length. */
    size_t rep_limit = n;
    for (size_t i = 0; i < n; i++) {
        size_t       first;
        const size_t end = breakers_for(pen, RAT(i), &first);
        if (first == end) {
            continue;
        }
        bool   found   = false;
        size_t longest = 0;
        for (size_t b = first; b < end; b++) {
            const size_t seq_len = pen->breakers[b].tail_len;
            if ((!found || seq_len > longest) && seq_len <= i) {
                const geist_token_t *tail  = pen->tails + pen->breakers[b].tail_off;
                bool                 match = true;
                for (size_t off = 0; off < seq_len; off++) {
                    if (tail[off] != RAT(i - off - 1)) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    found   = true;
                    longest = seq_len;
                }
            }
        }
        if (found) {
            rep_limit = i - longest;
            break;
        }
    }
    if (rep_limit < allowed) {
        return;
    }

    /* Step 2: reverse Z-algorithm — repeat_count[last - k] is the length of
     * the longest suffix of the window that also ends k tokens earlier. */
    uint32_t *rc = pen->repeat_count;
    memset(rc, 0, n * sizeof *rc);
    {
        const size_t last = n - 1;
        size_t       rt   = 0;
        size_t       lt   = 0;
        for (size_t k = 1; k < n; k++) {
            if (k > rt) {
                size_t m = 0;
                while (m + k < n && RAT(m) == RAT(m + k)) {
                    m++;
                }
                rc[last - k] = (uint32_t) (m < rep_limit ? m : rep_limit);
                if (m > 0) {
                    lt = k;
                    rt = k + m - 1;
                }
            } else {
                const size_t pr             = k - lt;
                const size_t right_part_len = rt - k + 1;
                if (rc[last - pr] < right_part_len) {
                    const size_t m = rc[last - pr];
                    rc[last - k]   = (uint32_t) (m < rep_limit ? m : rep_limit);
                } else {
                    size_t i = rt + 1;
                    while (i < n && RAT(i) == RAT(i - k)) {
                        i++;
                    }
                    const size_t m = i - k;
                    rc[last - k]   = (uint32_t) (m < rep_limit ? m : rep_limit);
                    lt             = k;
                    rt             = i - 1;
                }
            }
        }
    }

    /* Step 3: the token after each repeat would extend it; keep the longest
     * repeat per such token (stored + 1, so 0 means none). */
    size_t n_touched = 0;
    for (size_t i = 0; i + 1 < n; i++) {
        const size_t repeat_len = rc[i];
        if (repeat_len >= allowed) {
            const geist_token_t tok = RAT(n - 2 - i);
            if (pen->dry_max[tok] == 0) {
                pen->touched[n_touched++] = tok;
            }
            if (pen->dry_max[tok] < repeat_len + 1) {
                pen->dry_max[tok] = (uint32_t) (repeat_len + 1);
            }
        }
    }
#undef RAT

    /* Step 4: the penalty, skipped for single-token breakers. The clamp
     * keeps pow() finite; llama.cpp computes it in float, the power in
     * double (std::pow(float, int) promotes). */
    constexpr float FLOAT_MAX_LOG = 88.7228391f;
    int             max_exponent  = 0;
    if (p->dry_base > 1.000001f) {
        max_exponent = (int) (FLOAT_MAX_LOG / logf(p->dry_base));
    }
    for (size_t j = 0; j < n_touched; j++) {
        const geist_token_t tok    = pen->touched[j];
        const size_t        maxlen = pen->dry_max[tok] - 1;
        pen->dry_max[tok]          = 0;
        size_t       first;
        const size_t end    = breakers_for(pen, tok, &first);
        bool         single = false;
        for (size_t b = first; b < end; b++) {
            if (pen->breakers[b].tail_len == 0) {
                single = true;
                break;
            }
        }
        if (single) {
            continue;
        }
        size_t repeat_exp = maxlen - allowed;
        if (max_exponent > 0 && repeat_exp > (size_t) max_exponent) {
            repeat_exp = (size_t) max_exponent;
        }
        const float penalty = (float) ((double) p->dry_multiplier *
                                       pow((double) p->dry_base, (double) repeat_exp));
        out[tok] -= penalty;
    }
}

const float *geist_sampler_penalties_apply(struct geist_sampler_penalties *pen,
                                           size_t                          n_hist,
                                           const geist_token_t            *hist,
                                           const float                    *logits) {
    if (pen->logits == nullptr) {
        return logits;
    }
    /* The window: the last `cap` text tokens, oldest first, at the end of
     * pen->window. */
    size_t got = 0;
    for (size_t i = n_hist; i > 0 && got < pen->cap; i--) {
        const geist_token_t tok = hist[i - 1];
        if (tok >= 0 && (size_t) tok < pen->n_vocab) {
            pen->window[pen->cap - 1 - got] = tok;
            got++;
        }
    }
    const geist_token_t *w = pen->window + (pen->cap - got);

    float *out = pen->logits;
    memcpy(out, logits, pen->n_vocab * sizeof *out);
    if (geist_sampler_penalties_on(&pen->p)) {
        const size_t n = got < pen->p.last_n ? got : pen->p.last_n;
        penalties_apply(pen, n, w + (got - n), out);
    }
    if (geist_sampler_dry_on(&pen->p)) {
        const size_t n = got < pen->p.dry_last_n ? got : pen->p.dry_last_n;
        dry_apply(pen, n, w + (got - n), out);
    }
    return out;
}
