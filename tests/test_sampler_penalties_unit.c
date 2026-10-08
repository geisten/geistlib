/*
 * test_sampler_penalties_unit — the repetition penalties (#695) against
 * llama.cpp.
 *
 * The expected logits come from llama.cpp itself (the pinned checkout,
 * 2d8d612e4): a harness that feeds the same history through
 * llama_sampler_accept and runs llama_sampler_init_penalties /
 * llama_sampler_init_dry_testing on the same eight logits, printed with %a.
 * Every comparison is exact.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "test_helpers.h"

#include "src/engine/sampler.h"

#include <geist.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const float L8[8] = {2.0f, -1.5f, 0.5f, 3.0f, -0.25f, 1.25f, 0.0f, -2.0f};

static int fails;

static void check(bool cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        fails++;
    }
}

/* Penalties and DRY on L8 with history `hist`; compares all eight logits,
 * twice (the scratch must be clean after a call). */
static void run_case(const char                                *name,
                     const struct geist_sampler_penalty_params *p,
                     size_t                                     n_hist,
                     const geist_token_t                       *hist,
                     size_t                                     n_words,
                     const geist_token_t                       *breakers,
                     const float                                expect[8]) {
    struct geist_sampler_penalties pen = {0};
    if (geist_sampler_penalties_init(&pen, p, 8, 4096) != GEIST_OK) {
        fprintf(stderr, "FAIL: %s: init\n", name);
        fails++;
        return;
    }
    if (n_words > 0 && geist_sampler_penalties_set_breakers(&pen, n_words, breakers) != GEIST_OK) {
        fprintf(stderr, "FAIL: %s: breakers\n", name);
        fails++;
    }
    for (int pass = 0; pass < 2; pass++) {
        const float *got = geist_sampler_penalties_apply(&pen, n_hist, hist, L8);
        for (size_t i = 0; i < 8; i++) {
            if (got[i] != expect[i]) {
                fprintf(stderr,
                        "FAIL: %s pass %d: logit %zu = %a, llama.cpp %a\n",
                        name,
                        pass,
                        i,
                        (double) got[i],
                        (double) expect[i]);
                fails++;
            }
        }
    }
    geist_sampler_penalties_destroy(&pen);
}

#define N(a) (sizeof(a) / sizeof((a)[0]))

int main(void) {
    const struct geist_sampler_penalty_params off = {.repeat = 1.0f};

    /* P1: all three penalties, window 6 of 8 history tokens. */
    {
        struct geist_sampler_penalty_params p = off;
        p.last_n                              = 6;
        p.repeat                              = 1.3f;
        p.freq                                = 0.4f;
        p.present                             = 0.25f;
        const geist_token_t h[]               = {3, 1, 3, 5, 0, 3, 6, 1};
        const float         e[]               = {0x1.c6e47p-1f,
                                                 -0x1.4cccccp+1f,
                                                 0x1p-1f,
                                                 0x1.41f82p+0f,
                                                 -0x1p-2f,
                                                 0x1.3f03f4p-2f,
                                                 -0x1.4cccccp-1f,
                                                 -0x1p+1f};
        run_case("P1", &p, N(h), h, 0, nullptr, e);
    }
    /* P2: repeat_penalty only, window longer than the history. */
    {
        struct geist_sampler_penalty_params p = off;
        p.last_n                              = 64;
        p.repeat                              = 1.1f;
        const geist_token_t h[]               = {3, 1, 3, 5, 0, 3, 6, 1};
        const float         e[]               = {0x1.d1745cp+0f,
                                                 -0x1.a66668p+0f,
                                                 0x1p-1f,
                                                 0x1.5d1746p+1f,
                                                 -0x1p-2f,
                                                 0x1.22e8bap+0f,
                                                 0x0p+0f,
                                                 -0x1p+1f};
        run_case("P2", &p, N(h), h, 0, nullptr, e);
    }
    /* P3: negative (encouraging) frequency and presence penalties. */
    {
        struct geist_sampler_penalty_params p = off;
        p.last_n                              = 64;
        p.freq                                = -0.3f;
        p.present                             = -0.5f;
        const geist_token_t h[]               = {7, 7, 7, 2, 4, 4};
        const float         e[]               = {0x1p+1f,
                                                 -0x1.8p+0f,
                                                 0x1.4cccccp+0f,
                                                 0x1.8p+1f,
                                                 0x1.b33334p-1f,
                                                 0x1.4p+0f,
                                                 0x0p+0f,
                                                 -0x1.33333p-1f};
        run_case("P3", &p, N(h), h, 0, nullptr, e);
    }
    /* Soft tokens (negative ids) are not history: P3 with them interleaved. */
    {
        struct geist_sampler_penalty_params p = off;
        p.last_n                              = 64;
        p.freq                                = -0.3f;
        p.present                             = -0.5f;
        const geist_token_t h[]               = {-1, 7, 7, -1, -1, 7, 2, 4, -1, 4};
        const float         e[]               = {0x1p+1f,
                                                 -0x1.8p+0f,
                                                 0x1.4cccccp+0f,
                                                 0x1.8p+1f,
                                                 0x1.b33334p-1f,
                                                 0x1.4p+0f,
                                                 0x0p+0f,
                                                 -0x1.33333p-1f};
        run_case("P3 with soft tokens", &p, N(h), h, 0, nullptr, e);
    }

    const struct geist_sampler_penalty_params dry = {
            .repeat             = 1.0f,
            .dry_multiplier     = 0.8f,
            .dry_base           = 1.75f,
            .dry_allowed_length = 2,
            .dry_last_n         = 64,
    };
    /* D1: 0 1 2 3 4 0 1 2 3 — a 4 would extend a 4-token repeat. */
    {
        const geist_token_t h[] = {0, 1, 2, 3, 4, 0, 1, 2, 3};
        const float         e[] = {0x1p+1f,
                                   -0x1.8p+0f,
                                   0x1p-1f,
                                   0x1.8p+1f,
                                   -0x1.59999ap+1f,
                                   0x1.4p+0f,
                                   0x0p+0f,
                                   -0x1p+1f};
        run_case("D1", &dry, N(h), h, 0, nullptr, e);
    }
    /* D2: a single-token breaker (5) one token back stops the repeat. */
    {
        const geist_token_t h[]  = {0, 1, 2, 5, 3, 4, 0, 1, 2, 5, 3};
        const geist_token_t br[] = {1, 5};
        run_case("D2", &dry, N(h), h, N(br), br, L8);
    }
    /* D3: a two-token breaker {6, 7} limits the repeat to 3; the candidate
     * 6 is not a single-token breaker, so it is penalized. */
    {
        struct geist_sampler_penalty_params p = dry;
        p.dry_multiplier                      = 1.0f;
        p.dry_base                            = 2.0f;
        p.dry_allowed_length                  = 1;
        const geist_token_t h[]               = {1, 2, 6, 7, 3, 1, 2, 6, 7, 3, 1, 2};
        const geist_token_t br[]              = {2, 6, 7, 1, 4};
        const float         e[]               = {
                0x1p+1f, -0x1.8p+0f, 0x1p-1f, 0x1.8p+1f, -0x1p-2f, 0x1.4p+0f, -0x1p+2f, -0x1p+1f};
        run_case("D3", &p, N(h), h, N(br), br, e);
    }
    /* D4: a candidate that is a single-token breaker is never penalized. */
    {
        struct geist_sampler_penalty_params p = dry;
        p.dry_multiplier                      = 1.0f;
        p.dry_base                            = 2.0f;
        p.dry_allowed_length                  = 1;
        const geist_token_t h[]               = {1, 2, 4, 1, 2};
        const geist_token_t br[]              = {1, 4};
        run_case("D4", &p, N(h), h, N(br), br, L8);
    }
    /* D5: a DRY window shorter than the history, allowed length 0. */
    {
        struct geist_sampler_penalty_params p = dry;
        p.dry_multiplier                      = 0.5f;
        p.dry_base                            = 1.5f;
        p.dry_allowed_length                  = 0;
        p.dry_last_n                          = 5;
        const geist_token_t h[]               = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1};
        const float         e[]               = {
                0x1.8p+0f, -0x1p+1f, -0x1p-2f, 0x1.4p+1f, -0x1p-2f, 0x1.4p+0f, 0x0p+0f, -0x1p+1f};
        run_case("D5", &p, N(h), h, 0, nullptr, e);
    }
    /* D6: the exponent clamp — base 3 over a 400-token period-4 repeat. */
    {
        struct geist_sampler_penalty_params p = dry;
        p.dry_multiplier                      = 0.75f;
        p.dry_base                            = 3.0f;
        p.dry_last_n                          = 1000;
        geist_token_t h[400];
        for (size_t i = 0; i < 400; i++) {
            h[i] = (geist_token_t) (i % 4);
        }
        const float e[] = {-0x1.4d98d6p+126f,
                           -0x1.8p+0f,
                           0x1p-1f,
                           0x1.8p+1f,
                           -0x1p-2f,
                           0x1.4p+0f,
                           0x0p+0f,
                           -0x1p+1f};
        run_case("D6", &p, N(h), h, 0, nullptr, e);
    }
    /* C1: penalties and DRY together, in llama.cpp's chain order. */
    {
        struct geist_sampler_penalty_params p = dry;
        p.last_n                              = 4;
        p.repeat                              = 1.2f;
        p.freq                                = 0.1f;
        p.present                             = 0.2f;
        const geist_token_t h[]               = {0, 1, 2, 3, 4, 0, 1, 2, 3};
        const float         e[]               = {0x1.5ddddcp+0f,
                                                 -0x1.0ccccep+1f,
                                                 0x1.ddddd8p-4f,
                                                 0x1.19999ap+1f,
                                                 -0x1.59999ap+1f,
                                                 0x1.4p+0f,
                                                 0x0p+0f,
                                                 -0x1p+1f};
        run_case("C1", &p, N(h), h, 0, nullptr, e);
    }

    /* Greedy takes the argmax of the penalized copy: after a 3, repeat 2.0
     * turns 3.0 into 1.5 and token 0 (2.0) wins. */
    {
        struct geist_sampler_penalty_params p = off;
        p.last_n                              = 64;
        p.repeat                              = 2.0f;
        struct geist_sampler_penalties pen    = {0};
        check(geist_sampler_penalties_init(&pen, &p, 8, 64) == GEIST_OK, "init repeat 2.0");
        const geist_token_t h[] = {3};
        const float        *got = geist_sampler_penalties_apply(&pen, 1, h, L8);
        check(geist_sampler_argmax(8, L8) == 3, "argmax without penalty is 3");
        check(geist_sampler_argmax(8, got) == 0, "argmax with repeat 2.0 after a 3 is 0");
        geist_sampler_penalties_destroy(&pen);
    }

    /* Off: nothing allocated, the logits come back untouched. */
    {
        struct geist_sampler_penalties pen = {0};
        check(geist_sampler_penalties_init(&pen, &off, 8, 64) == GEIST_OK, "init off");
        check(!geist_sampler_penalties_active(&pen), "off is inactive");
        const geist_token_t h[] = {1, 2, 3};
        check(geist_sampler_penalties_apply(&pen, 3, h, L8) == L8, "off returns the logits");
        geist_sampler_penalties_destroy(&pen);
    }

    /* Opts: zero → everything off; the zero defaults; validation. */
    {
        struct geist_session_opts           o = {0};
        struct geist_sampler_penalty_params p;
        check(geist_sampler_penalty_params_from_opts(&p, &o) == GEIST_OK, "zero opts ok");
        check(!geist_sampler_penalties_on(&p) && !geist_sampler_dry_on(&p), "zero opts are off");
        check(p.last_n == 64 && p.repeat == 1.0f, "zero opts: window 64, repeat 1");
        check(p.dry_base == 1.75f && p.dry_allowed_length == 2 && p.dry_last_n == 64,
              "zero opts: llama.cpp's DRY defaults");
        check(geist_sampler_penalty_params_from_opts(&p, nullptr) == GEIST_OK &&
                      !geist_sampler_penalties_on(&p),
              "nullptr opts are off");

        o.repeat_penalty = 1.1f;
        check(geist_sampler_penalty_params_from_opts(&p, &o) == GEIST_OK &&
                      geist_sampler_penalties_on(&p) && p.repeat == 1.1f,
              "repeat_penalty 1.1 is on");
        o                = (struct geist_session_opts) {0};
        o.dry_multiplier = 0.8f;
        check(geist_sampler_penalty_params_from_opts(&p, &o) == GEIST_OK &&
                      geist_sampler_dry_on(&p),
              "dry_multiplier 0.8 is on");

        const struct {
            struct geist_session_opts o;
            const char               *what;
        } bad[] = {
                {{.repeat_penalty = -1.0f}, "negative repeat_penalty"},
                {{.repeat_penalty = NAN}, "NaN repeat_penalty"},
                {{.repeat_penalty = 1e-39f}, "repeat_penalty whose inverse overflows"},
                {{.frequency_penalty = INFINITY}, "infinite frequency_penalty"},
                {{.presence_penalty = NAN}, "NaN presence_penalty"},
                {{.repeat_last_n = -1}, "negative repeat_last_n"},
                {{.dry_base = 0.5f}, "dry_base below 1"},
                {{.dry_allowed_length = -2}, "negative dry_allowed_length"},
                {{.dry_penalty_last_n = -1}, "negative dry_penalty_last_n"},
        };
        for (size_t i = 0; i < N(bad); i++) {
            check(geist_sampler_penalty_params_from_opts(&p, &bad[i].o) == GEIST_E_INVALID_ARG,
                  bad[i].what);
        }
    }

    /* Malformed breaker packings are refused, the old table kept. */
    {
        struct geist_sampler_penalties pen = {0};
        check(geist_sampler_penalties_init(&pen, &dry, 8, 64) == GEIST_OK, "init dry");
        const geist_token_t ok[] = {1, 5, 2, 6, 7};
        check(geist_sampler_penalties_set_breakers(&pen, N(ok), ok) == GEIST_OK, "valid packing");
        const geist_token_t short_seq[] = {3, 6, 7};
        const geist_token_t zero_len[]  = {0};
        check(geist_sampler_penalties_set_breakers(&pen, N(short_seq), short_seq) ==
                      GEIST_E_INVALID_ARG,
              "length past the end");
        check(geist_sampler_penalties_set_breakers(&pen, N(zero_len), zero_len) ==
                      GEIST_E_INVALID_ARG,
              "zero-length sequence");
        check(pen.n_breakers == 2, "the valid table stays");
        geist_sampler_penalties_destroy(&pen);
    }

    if (fails == 0) {
        printf("test_sampler_penalties_unit: OK\n");
        return 0;
    }
    fprintf(stderr, "test_sampler_penalties_unit: %d failure(s)\n", fails);
    return 1;
}
