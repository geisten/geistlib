/*
 * test_sampler_unit — verifies argmax, temperature, top-k, top-p sampling
 * on synthetic FP32 logit arrays. Deterministic via xorshift RNG seed.
 *
 * Phase B-4b smoke test — the sampler module operates standalone on
 * caller-provided logit arrays; no LM* required.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "test_helpers.h"

#include "src/engine/sampler.h"

#include <geist.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(bool cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        return 1;
    }
    return 0;
}

int main(void) {
    int fails = 0;

    /* ---- 1. Argmax: deterministic, stable on ties ---- */
    {
        float         logits[] = {-1.0f, 2.0f, 0.5f, 3.0f, 3.0f, -0.5f};
        geist_token_t t        = geist_sampler_argmax(6, logits);
        fails += check(t == 3, "argmax picks first occurrence of max (idx 3)");
    }

    /* ---- 2. Temperature: highly peaked → near-argmax ---- */
    {
        struct geist_rng rng;
        geist_rng_seed(&rng, 42);
        float logits[] = {0.0f, 0.0f, 0.0f, 10.0f, 0.0f};
        int   hits_3   = 0;
        for (int i = 0; i < 1000; i++) {
            if (geist_sampler_temperature(5, logits, 0.1f /* low T */, &rng) == 3) {
                hits_3++;
            }
        }
        fails += check(hits_3 > 990, "low-T temperature sample heavily favors max");
    }

    /* ---- 3. Temperature=0 → argmax exactly ---- */
    {
        struct geist_rng rng;
        geist_rng_seed(&rng, 1);
        float         logits[] = {1.0f, 2.0f, 3.0f, 0.5f};
        geist_token_t t        = geist_sampler_temperature(4, logits, 0.0f, &rng);
        fails += check(t == 2, "temperature=0 returns argmax (idx 2)");
    }

    /* ---- 4. Uniform logits → samples spread across all indices ---- */
    {
        struct geist_rng rng;
        geist_rng_seed(&rng, 7);
        float logits[]  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        int   counts[5] = {0};
        for (int i = 0; i < 5000; i++) {
            geist_token_t t = geist_sampler_temperature(5, logits, 1.0f, &rng);
            if (t >= 0 && t < 5)
                counts[t]++;
        }
        /* Expected ~1000 each ± noise. Each bucket should be > 800 with high
         * confidence over 5000 draws of a uniform-5 distribution. */
        for (int i = 0; i < 5; i++) {
            if (counts[i] < 800 || counts[i] > 1200) {
                fprintf(stderr, "  uniform: bucket %d = %d (expected 800-1200)\n", i, counts[i]);
                fails++;
            }
        }
        if (fails == 0) {
            printf("uniform sample distribution: [%d %d %d %d %d]\n",
                   counts[0],
                   counts[1],
                   counts[2],
                   counts[3],
                   counts[4]);
        }
    }

    /* ---- 5. Top-K: only top-2 indices may be returned ---- */
    {
        struct geist_rng rng;
        geist_rng_seed(&rng, 11);
        /* Top 2 logits are at indices 4 (=5.0) and 2 (=4.0). */
        float                          logits[] = {0.0f, 1.0f, 4.0f, 2.0f, 5.0f, 1.5f, 0.0f, 0.5f};
        bool                           seen[8]  = {false};
        struct geist_sampler_workspace ws       = {0};
        fails += check(geist_sampler_workspace_init(&ws, 8) == GEIST_OK, "top-k ws init");
        for (int i = 0; i < 1000; i++) {
            geist_token_t t = geist_sampler_top_k_ws(&ws, logits, 2, 1.0f, &rng);
            fails += check(t >= 0 && t < 8, "top-k token in range");
            seen[t] = true;
        }
        geist_sampler_workspace_destroy(&ws);
        fails += check(seen[2] && seen[4], "top-k=2 samples both top indices (2, 4)");
        int other = 0;
        for (int i = 0; i < 8; i++)
            if (i != 2 && i != 4 && seen[i])
                other++;
        fails += check(other == 0, "top-k=2 never samples outside the top-2");
    }

    /* ---- 6. Top-P: nucleus must cover ≥ p of mass ---- */
    {
        struct geist_rng rng;
        geist_rng_seed(&rng, 13);
        /* Make most mass concentrate at index 0 — top_p=0.5 should pretty
         * much always pick index 0. */
        float                          logits[] = {10.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        int                            hits_0   = 0;
        struct geist_sampler_workspace ws       = {0};
        fails += check(geist_sampler_workspace_init(&ws, 5) == GEIST_OK, "top-p ws init");
        for (int i = 0; i < 1000; i++) {
            if (geist_sampler_top_p_ws(&ws, logits, 0.5f, 1.0f, &rng) == 0) {
                hits_0++;
            }
        }
        geist_sampler_workspace_destroy(&ws);
        fails += check(hits_0 > 990, "top-p=0.5 with peaked logits picks index 0 nearly always");
    }

    /* ---- 7. Workspace reuse: O(1) allocations across multiple calls ---- */
    {
        struct geist_sampler_workspace ws = {0};
        enum geist_status              s  = geist_sampler_workspace_init(&ws, 8);
        fails += check(s == GEIST_OK, "workspace init OK");

        struct geist_rng rng;
        geist_rng_seed(&rng, 17);
        float logits[] = {0.1f, 0.2f, 0.3f, 5.0f, 0.1f, 0.0f, 0.0f, 0.0f};
        for (int i = 0; i < 100; i++) {
            geist_token_t t = geist_sampler_top_k_ws(&ws, logits, 3, 1.0f, &rng);
            (void) t;
        }
        geist_sampler_workspace_destroy(&ws);
        printf("workspace reuse OK over 100 calls\n");
    }

    /* ---- 8. RNG determinism ---- */
    {
        struct geist_rng a, b;
        geist_rng_seed(&a, 99);
        geist_rng_seed(&b, 99);
        for (int i = 0; i < 1000; i++) {
            uint64_t va = geist_rng_next_u64(&a);
            uint64_t vb = geist_rng_next_u64(&b);
            if (va != vb) {
                fprintf(stderr,
                        "RNG diverged at iter %d: %llu vs %llu\n",
                        i,
                        (unsigned long long) va,
                        (unsigned long long) vb);
                fails++;
                break;
            }
        }
        printf("RNG is deterministic per seed\n");
    }

    if (fails == 0) {
        printf("PASS: all 8 sampler checks\n");
        return GEIST_TEST_PASS;
    }
    fprintf(stderr, "FAILED: %d check(s)\n", fails);
    return GEIST_TEST_FAIL;
}
