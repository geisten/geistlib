/*
 * test_backend_vulkan_weight_index_unit — the weight registry's hash index
 * (#469, #474 item 8).
 *
 * resolve_weight uploads each weight to device memory and records host
 * pointer → buffer; every linear_t and embedding call looks the weight up
 * by its host pointer. The lookup used to scan the whole registry. Checked
 * here, with 1500 weights at consecutive 64-byte addresses (the densest
 * layout a model's mmap can produce, and growth of the index past several
 * doublings):
 *
 *   - every weight is found and maps to its own buffer, the same one a
 *     scan of the dense list finds; pointers between the weights are not
 *     found;
 *   - the index stays a power of two at most half full;
 *   - resolving a host pointer again replaces its buffer in place (the
 *     registry does not grow) and the new bytes are what the device holds;
 *   - the time of a model-sized lookup pattern (300 registered weights,
 *     each looked up 250 times) is printed for the hash index and for a
 *     scan, for the record (not asserted).
 *
 * SKIPs when the Vulkan backend is not built or has no device.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <string.h>
#include <time.h>

#if defined(GEIST_BACKEND_VULKAN) && GEIST_BACKEND_VULKAN
#include "src/backends/vulkan/vk_internal.h"

enum { N = 1500, SIDE = 4, STRIDE = SIDE * SIDE }; /* 4x4 F32: 64 bytes each */

static float host[N * STRIDE];

static struct geist_weight weight_at(size_t i) {
    return (struct geist_weight) {.raw        = &host[i * STRIDE],
                                  .raw_nbytes = STRIDE * sizeof(float),
                                  .n_in       = SIDE,
                                  .n_out      = SIDE,
                                  .dtype      = GEIST_DTYPE_F32};
}

static struct geist_buffer *scan(const struct vk_state *st, const void *p) {
    for (size_t i = 0; i < st->n_weights; i++) {
        if (st->weights[i].host == p) {
            return st->weights[i].gpu;
        }
    }
    return nullptr;
}

static double now_s(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

int main(void) {
    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("vulkan", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "SKIP: vulkan backend unavailable (%d)\n", (int) s);
        return GEIST_TEST_SKIP;
    }
    const struct geist_backend_vtbl *vt    = be->desc->vtbl;
    struct vk_state                 *st    = be->state;
    int                              fails = 0;
    for (size_t i = 0; i < N * STRIDE; i++) {
        host[i] = (float) i;
    }

    size_t resolved = 0;
    for (size_t i = 0; i < N; i++) {
        struct geist_weight w = weight_at(i);
        resolved += vt->resolve_weight(be, &w) == GEIST_OK;
    }
    char msg[160];
    snprintf(msg, sizeof msg, "%zu of %d weights resolve", resolved, N);
    fails += geist_expect(resolved == N && st->n_weights == N, msg);

    const size_t cap = st->cap_weight_index;
    snprintf(msg, sizeof msg, "index of %zu slots: a power of two, at most half full", cap);
    fails += geist_expect(cap >= 2 * N && (cap & (cap - 1)) == 0, msg);

    size_t found = 0, same = 0, distinct = 0, misses = 0;
    for (size_t i = 0; i < N; i++) {
        const void          *p = &host[i * STRIDE];
        struct geist_buffer *b = vk_weight_lookup(st, p);
        found += b != nullptr;
        same += b == scan(st, p);
        distinct += i == 0 || b != vk_weight_lookup(st, &host[(i - 1) * STRIDE]);
        const struct vk_weight_entry *e = vk_weight_entry_of(st, p);
        found -= e == nullptr || e->host != p || e->gpu != b;
        /* An address inside the weight, not its start, is not a weight. */
        misses += vk_weight_lookup(st, &host[i * STRIDE + 1]) == nullptr &&
                  vk_weight_lookup(st, &host[i * STRIDE + SIDE]) == nullptr;
    }
    fails += geist_expect(found == N, "every weight is found by its host pointer");
    fails += geist_expect(same == N, "the index agrees with a scan of the list");
    fails += geist_expect(distinct == N, "every weight has its own buffer");
    fails += geist_expect(misses == N, "pointers inside a weight are not found");
    fails += geist_expect(vk_weight_lookup(st, nullptr) == nullptr &&
                                  vk_weight_lookup(st, &host[N * STRIDE - 1] + 1) == nullptr,
                          "pointers outside every weight are not found");

    /* Re-resolve one weight with new bytes: replaced in place. */
    const size_t         k   = N / 2;
    struct geist_buffer *old = vk_weight_lookup(st, &host[k * STRIDE]);
    for (size_t j = 0; j < STRIDE; j++) {
        host[k * STRIDE + j] = -1.0f - (float) j;
    }
    struct geist_weight w = weight_at(k);
    fails += geist_expect(vt->resolve_weight(be, &w) == GEIST_OK, "re-resolve");
    struct geist_buffer *now = vk_weight_lookup(st, &host[k * STRIDE]);
    fails += geist_expect(st->n_weights == N && now != nullptr && now != old &&
                                  now == scan(st, &host[k * STRIDE]),
                          "a re-resolve replaces the buffer and does not grow the registry");
    float got[STRIDE] = {0};
    fails += geist_expect(vt->buffer_download(sizeof got, (uint8_t *) got, now) == GEIST_OK &&
                                  memcmp(got, &host[k * STRIDE], sizeof got) == 0,
                          "the device holds the new bytes");
    fails += geist_expect(vk_weight_lookup(st, &host[(k + 1) * STRIDE]) != nullptr &&
                                  vk_weight_lookup(st, &host[(k - 1) * STRIDE]) != nullptr,
                          "neighbours still found");

    geist_backend_destroy(be);

    /* For the record: a model-sized registry (300 weights), each weight
     * looked up 250 times, in the hash index and by a scan. */
    enum { MODEL = 300, CALLS = 250, ROUNDS = 20 };
    if (geist_backend_create("vulkan", nullptr, nullptr, &be) != GEIST_OK) {
        return GEIST_TEST_FAIL;
    }
    st = be->state;
    for (size_t i = 0; i < MODEL; i++) {
        struct geist_weight mw = weight_at(i);
        fails += geist_expect(vt->resolve_weight(be, &mw) == GEIST_OK, "model-sized registry");
    }
    size_t       sink = 0;
    const double t0   = now_s();
    for (int r = 0; r < ROUNDS; r++) {
        for (size_t c = 0; c < CALLS; c++) {
            for (size_t i = 0; i < MODEL; i++) {
                sink += vk_weight_lookup(st, &host[((i * 7 + c) % MODEL) * STRIDE]) != nullptr;
            }
        }
    }
    const double t1 = now_s();
    for (int r = 0; r < ROUNDS; r++) {
        for (size_t c = 0; c < CALLS; c++) {
            for (size_t i = 0; i < MODEL; i++) {
                sink += scan(st, &host[((i * 7 + c) % MODEL) * STRIDE]) != nullptr;
            }
        }
    }
    const double t2 = now_s();
    printf("  %d lookups over %d registered weights: index %.1f us, scan %.1f us (%zu)\n",
           MODEL * CALLS,
           MODEL,
           (t1 - t0) * 1e6 / ROUNDS,
           (t2 - t1) * 1e6 / ROUNDS,
           sink);

    geist_backend_destroy(be);
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("vulkan weight index: pass\n");
    return GEIST_TEST_PASS;
}

#else
int main(void) {
    fprintf(stderr, "SKIP: vulkan backend not built\n");
    return GEIST_TEST_SKIP;
}
#endif
