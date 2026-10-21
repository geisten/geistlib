/*
 * test_neon_decode_triple_unit — cpu_neon's q/k/v decode in one parallel
 * region (linear_triple_m1, #16) gives each weight the bytes of its own
 * linear_m1.
 *
 * cpu_neon installs linear_triple_m1 on its Q4_K and Q6_K weights. For
 * every mix of the two dtypes over three weights of unequal heights (q
 * taller than k and v, as with grouped-query attention), and once with the
 * three weights in reverse order, the triple's three outputs must match
 * the single kernels' bit for bit. A Q4_K weight's linear_m1 quantizes
 * through its own cached path and a Q6_K's through the workspace; the
 * triple quantizes once for all three.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include "heap.h"
#include "quant.h"
#include "quant_fixtures.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

constexpr size_t N_IN = 512;

static size_t dtype_index(enum geist_dtype dt) {
    for (size_t d = 0; d < N_DTYPES; d++) {
        if (DTYPES[d].dt == dt) {
            return d;
        }
    }
    return N_DTYPES;
}

struct weight {
    struct geist_weight w;
    uint8_t            *raw;
    float              *y1, *y3; /* linear_m1's and the triple's */
};

/* 1: resolved; 0: not; -1: OOM. */
static int
weight_make(struct geist_backend *be, enum geist_dtype dt, size_t n_out, struct weight *t) {
    *t           = (struct weight) {0};
    size_t bytes = 0;
    t->raw       = make_weight(dtype_index(dt), N_IN, n_out, &bytes);
    t->y1        = malloc(n_out * sizeof *t->y1);
    t->y3        = malloc(n_out * sizeof *t->y3);
    if (t->raw == nullptr || t->y1 == nullptr || t->y3 == nullptr) {
        return -1;
    }
    t->w = (struct geist_weight) {.raw        = t->raw,
                                  .raw_nbytes = bytes,
                                  .n_in       = (int32_t) N_IN,
                                  .n_out      = (int32_t) n_out,
                                  .dtype      = (uint16_t) dt};
    return be->desc->vtbl->resolve_weight(be, &t->w) == GEIST_OK ? 1 : 0;
}

static void weight_free(struct weight *t) {
    if ((t->w.flags & GEIST_W_AUX_HEAP_OWNED) != 0) {
        void *aux = (void *) t->w.aux_fp32;
        safe_free(&aux);
    }
    free(t->raw);
    free(t->y1);
    free(t->y3);
    *t = (struct weight) {0};
}

static int check_mix(struct geist_backend  *be,
                     const enum geist_dtype dt[static 3],
                     const size_t           n_out[static 3],
                     const float            x[static N_IN]) {
    struct weight t[3];
    int           made = 1;
    for (size_t i = 0; i < 3; i++) {
        const int r = weight_make(be, dt[i], n_out[i], &t[i]);
        made        = made < r ? made : r;
    }
    char what[160];
    snprintf(what,
             sizeof what,
             "%s %zu + %s %zu + %s %zu rows",
             DTYPES[dtype_index(dt[0])].name,
             n_out[0],
             DTYPES[dtype_index(dt[1])].name,
             n_out[1],
             DTYPES[dtype_index(dt[2])].name,
             n_out[2]);
    int fails = geist_expect(made == 1, "the three weights resolve");
    if (made == 1) {
        char msg[224];
        snprintf(msg, sizeof msg, "%s: all three carry the same linear_triple_m1", what);
        const bool has = t[0].w.linear_triple_m1 != nullptr &&
                         t[0].w.linear_triple_m1 == t[1].w.linear_triple_m1 &&
                         t[0].w.linear_triple_m1 == t[2].w.linear_triple_m1;
        fails += geist_expect(has, msg);
        if (has) {
            for (size_t i = 0; i < 3; i++) {
                t[i].w.linear_m1(x, &t[i].w, be, t[i].y1);
            }
            t[0].w.linear_triple_m1(x, &t[0].w, &t[1].w, &t[2].w, be, t[0].y3, t[1].y3, t[2].y3);
            for (size_t i = 0; i < 3; i++) {
                snprintf(msg, sizeof msg, "%s: output %zu as its linear_m1's", what, i);
                fails += geist_expect(memcmp(t[i].y1, t[i].y3, n_out[i] * sizeof(float)) == 0, msg);
            }
        }
    }
    for (size_t i = 0; i < 3; i++) {
        weight_free(&t[i]);
    }
    return fails;
}

int main(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_neon", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        GEIST_SKIP("cpu_neon backend not built");
    }
    struct weight probe;
    const int     q4         = weight_make(be, GEIST_DTYPE_Q4_K, 8, &probe);
    const bool    has_triple = q4 == 1 && probe.w.linear_triple_m1 != nullptr;
    weight_free(&probe);
    if (!has_triple) {
        geist_backend_destroy(be);
        GEIST_SKIP("cpu_neon has no triple decode on this host (no dotprod)");
    }

    float *x = malloc(N_IN * sizeof *x);
    if (x == nullptr) {
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    for (size_t i = 0; i < N_IN; i++) {
        x[i] = next_f();
    }
    static const enum geist_dtype KQ[2]       = {GEIST_DTYPE_Q4_K, GEIST_DTYPE_Q6_K};
    static const size_t           HEIGHTS[3]  = {96, 32, 32};
    static const size_t           REVERSED[3] = {33, 7, 96};
    int                           fails       = 0;
    for (size_t mix = 0; mix < 8; mix++) {
        const enum geist_dtype dt[3] = {KQ[mix & 1], KQ[(mix >> 1) & 1], KQ[(mix >> 2) & 1]};
        fails += check_mix(be, dt, HEIGHTS, x);
        fails += check_mix(be, dt, REVERSED, x);
    }
    free(x);
    geist_backend_destroy(be);
    if (fails != 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: the q/k/v decode triple matches each weight's linear_m1 bit for bit\n");
    return GEIST_TEST_PASS;
}
