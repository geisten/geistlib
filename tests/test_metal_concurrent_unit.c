/* The Metal command sequence encodes into a concurrent compute encoder and
 * puts a memory barrier before a dispatch only when it depends on one since
 * the last barrier (#590, metal_seq_order). Checked on a real GPU: where
 * the barriers land (counted in metal_state), that chains of dependent
 * dispatches compute what the serial encoder computes -- a missing barrier
 * lets the dispatches of a chain run at once and the sums come out short --
 * and that a kind of sequence which spares no barrier goes back to serial
 * encoders. */
#include "test_helpers.h"
#include <geist.h>
#include <geist_backend.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(GEIST_BACKEND_METAL) && defined(__APPLE__)
#include "../src/backends/metal/metal_internal.h"

static int g_fail = 0;
#define check(ok, what) (g_fail |= geist_expect((ok), (what)))

enum { N = 1 << 16, CHAIN = 96 };

static struct geist_tensor tensor_of(struct geist_buffer *buf) {
    return (struct geist_tensor) {.buffer = buf,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 1,
                                  .shape  = {N},
                                  .stride = {1}};
}

static bool fill(const struct geist_backend_vtbl *v, struct geist_buffer *buf, float value) {
    static float host[N];
    for (size_t i = 0; i < N; i++) {
        host[i] = value;
    }
    return v->buffer_upload(buf, sizeof host, (const uint8_t *) host) == GEIST_OK;
}

static bool all_equal(const struct geist_backend_vtbl *v, struct geist_buffer *buf, float want) {
    static float host[N];
    if (v->buffer_download(sizeof host, (uint8_t *) host, buf) != GEIST_OK) {
        return false;
    }
    for (size_t i = 0; i < N; i++) {
        if (host[i] != want) {
            printf("  element %zu: %g, want %g\n", i, (double) host[i], (double) want);
            return false;
        }
    }
    return true;
}
#endif

int main(void) {
#if !defined(GEIST_BACKEND_METAL) || !defined(__APPLE__)
    return GEIST_TEST_SKIP;
#else
    struct geist_backend *be = nullptr;
    if (geist_backend_create("metal", nullptr, nullptr, &be) != GEIST_OK) {
        puts("SKIP: no Metal device");
        return GEIST_TEST_SKIP;
    }
    struct metal_state                    *st = be->state;
    const struct geist_backend_vtbl       *v  = be->desc->vtbl;
    const struct geist_backend_primitives *p  = be->desc->prims;
    if (!st->seq_concurrent) {
        puts("SKIP: GEIST_METAL_CONCURRENT=0");
        geist_backend_destroy(be);
        return GEIST_TEST_SKIP;
    }
    struct geist_buffer *buf[5] = {nullptr};
    struct geist_tensor  t[5];
    for (size_t i = 0; i < 5; i++) {
        if (v->buffer_create(
                    be, N * sizeof(float), GEIST_BUFFER_SCRATCH, GEIST_MEMORY_AUTO, &buf[i]) !=
            GEIST_OK) {
            return GEIST_TEST_ERROR;
        }
        t[i] = tensor_of(buf[i]);
    }
    /* The order rules compare MTLBuffers; the cases below need five. */
    bool distinct = true;
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < i; j++) {
            distinct = distinct && buf[i]->buffer != buf[j]->buffer;
        }
    }
    check(distinct, "five scratch buffers are five MTLBuffers");

    /* A kernel's reads and writes are known from the end of the first
     * sequence that used it (until then it counts as writing everything):
     * one sequence with an add first. */
    int warm = v->parallel_region_begin(be, GEIST_REGION_DECODE_STEP);
    check(warm != 0 && p->add(be, &t[0], &t[1], &t[2]) == GEIST_OK, "warm-up add");
    v->parallel_region_end(be, warm);

    /* 1. Where the barriers land. a=1, b=2. */
    bool           ok  = fill(v, buf[0], 1.0f) && fill(v, buf[1], 2.0f) && fill(v, buf[2], 0.0f) &&
                         fill(v, buf[3], 0.0f) && fill(v, buf[4], 0.0f);
    int            tok = v->parallel_region_begin(be, GEIST_REGION_DECODE_STEP);
    const uint64_t before = st->seq_barriers;
    ok                    = ok && tok != 0;
    ok                    = ok && p->add(be, &t[0], &t[1], &t[2]) == GEIST_OK; /* c = a + b */
    check(st->seq_barriers == before, "the first dispatch of an encoder needs no barrier");
    ok = ok && p->add(be, &t[0], &t[1], &t[3]) == GEIST_OK; /* d = a + b: reads what c read */
    check(st->seq_barriers == before, "two dispatches that only share inputs overlap");
    ok = ok && p->add(be, &t[2], &t[3], &t[4]) == GEIST_OK; /* e = c + d: reads both outputs */
    check(st->seq_barriers == before + 1, "a dispatch that reads an output waits for it");
    ok = ok && p->add(be, &t[0], &t[1], &t[2]) == GEIST_OK; /* c = a + b: e's dispatch read c */
    check(st->seq_barriers == before + 2,
          "a dispatch that overwrites an input waits for the reader");
    ok = ok && p->add(be, &t[0], &t[1], &t[2]) == GEIST_OK; /* c again: written since the barrier */
    check(st->seq_barriers == before + 3, "two writers of one buffer do not overlap");
    v->parallel_region_end(be, tok);
    check(ok, "sequence of adds encodes");
    check(all_equal(v, buf[2], 3.0f) && all_equal(v, buf[3], 3.0f) && all_equal(v, buf[4], 6.0f),
          "c = d = a + b, e = c + d");

    /* 2. A chain in place: a += b, CHAIN times in one sequence. Every
     * dispatch reads and writes a; run at once they would all read the
     * same a. */
    ok  = fill(v, buf[0], 0.0f) && fill(v, buf[1], 1.0f);
    tok = v->parallel_region_begin(be, GEIST_REGION_DECODE_STEP);
    ok  = ok && tok != 0;
    for (int i = 0; i < CHAIN && ok; i++) {
        ok = p->add(be, &t[0], &t[1], &t[0]) == GEIST_OK;
    }
    v->parallel_region_end(be, tok);
    check(ok && all_equal(v, buf[0], (float) CHAIN), "in-place chain adds up");

    /* 3. A chain through two buffers, with a dispatch that depends on
     * nothing in between: c = a + b; a = c + b; d = b + b (independent). */
    ok  = fill(v, buf[0], 0.0f) && fill(v, buf[1], 1.0f);
    tok = v->parallel_region_begin(be, GEIST_REGION_PREFILL_BATCH);
    ok  = ok && tok != 0;
    for (int i = 0; i < CHAIN && ok; i++) {
        ok = p->add(be, &t[0], &t[1], &t[2]) == GEIST_OK &&
             p->add(be, &t[1], &t[1], &t[3]) == GEIST_OK &&
             p->add(be, &t[2], &t[1], &t[0]) == GEIST_OK;
    }
    v->parallel_region_end(be, tok);
    check(ok && all_equal(v, buf[0], (float) (2 * CHAIN)) && all_equal(v, buf[3], 2.0f),
          "two-buffer chain adds up across command-buffer rotations");

    /* 4. A kind of sequence that spares (almost) no barrier goes back to
     * serial encoders: decode steps that are one long chain. The prefill
     * kind, which ran few dispatches, is still on trial. */
    check(!st->seq_trial[1].decided, "the cases above stay under the trial length");
    ok = fill(v, buf[0], 0.0f) && fill(v, buf[1], 1.0f);
    for (int step = 0; step < 48 && ok; step++) {
        tok = v->parallel_region_begin(be, GEIST_REGION_DECODE_STEP);
        ok  = tok != 0;
        for (int i = 0; i < CHAIN && ok; i++) {
            ok = p->add(be, &t[0], &t[1], &t[0]) == GEIST_OK;
        }
        v->parallel_region_end(be, tok);
    }
    check(ok && all_equal(v, buf[0], (float) (48 * CHAIN)), "long chain adds up");
    check(st->seq_trial[1].decided && st->seq_trial[1].serial,
          "a chain-only kind of sequence goes back to serial encoders");
    check(!st->seq_trial[0].serial, "the other kind is unaffected");
    const uint64_t barriers = st->seq_barriers;
    ok                      = fill(v, buf[0], 0.0f);
    tok                     = v->parallel_region_begin(be, GEIST_REGION_DECODE_STEP);
    ok                      = ok && tok != 0;
    for (int i = 0; i < CHAIN && ok; i++) {
        ok = p->add(be, &t[0], &t[1], &t[0]) == GEIST_OK;
    }
    v->parallel_region_end(be, tok);
    check(ok && all_equal(v, buf[0], (float) CHAIN) && st->seq_barriers == barriers,
          "serial encoders after the trial: same sums, no barrier encoded");

    for (size_t i = 0; i < 5; i++) {
        v->buffer_destroy(be, buf[i]);
    }
    geist_backend_destroy(be);
    return g_fail ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
#endif
}
