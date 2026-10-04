/*
 * test_backend_vulkan_narrowing_unit — a failed submit reaches buffer_map,
 * and sizes the shaders cannot index are refused, not truncated (#474,
 * items 1 and 6).
 *
 * 1. buffer_map after a failed submit: the batch was dropped, so the mapping
 *    still holds the bytes from before it. buffer_map has no status, so it
 *    must return nullptr with GEIST_E_BACKEND as the backend error instead
 *    of those stale bytes; a download reports the same failure.
 * 2. vk_ckd_u32 refuses values past UINT32_MAX and leaves its output alone.
 * 3. An op handed a position past UINT32_MAX (attention's q_offset and
 *    sliding_window: plain scalars, no large allocation needed) returns
 *    GEIST_E_INVALID_ARG instead of dispatching with the value wrapped.
 * 4. resolve_weight refuses a weight whose max_m batch would not fit the
 *    x ring (it used to become a per-call UNSUPPORTED and a silent host
 *    linear), and creates the ring at load for one that fits.
 *
 * SKIPs (exit 77) when the Vulkan backend is not built or has no device.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <stdio.h>
#include <string.h>

#if defined(GEIST_BACKEND_VULKAN) && GEIST_BACKEND_VULKAN
#include "src/backends/vulkan/vk_internal.h"

static VkResult VKAPI_CALL fail_submit(VkQueue             q,
                                       uint32_t            n,
                                       const VkSubmitInfo *submits,
                                       VkFence             fence) {
    (void) q;
    (void) n;
    (void) submits;
    (void) fence;
    return VK_ERROR_DEVICE_LOST;
}

enum { N = 256 };

/* A host-visible F32 buffer of N elements viewed with the given shape. */
static struct geist_tensor
host_tensor(struct geist_backend *be, size_t ndim, const int64_t shape[static ndim]) {
    struct geist_tensor t = {.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE};
    t.ndim                = (int) ndim;
    int64_t stride        = 1;
    for (size_t d = ndim; d-- > 0;) {
        t.shape[d]  = shape[d];
        t.stride[d] = stride;
        stride *= shape[d];
    }
    if (be->desc->vtbl->buffer_create(be, N * sizeof(float), GEIST_BUFFER_SCRATCH, 0, &t.buffer) !=
        GEIST_OK) {
        t.buffer = nullptr;
    }
    return t;
}

static bool fill(struct geist_backend *be, struct geist_tensor *t, float v) {
    float *p = be->desc->vtbl->buffer_map(t->buffer);
    if (p == nullptr) {
        return false;
    }
    for (size_t i = 0; i < N; i++) {
        p[i] = v;
    }
    be->desc->vtbl->buffer_unmap(t->buffer);
    return true;
}

/* Part 1: a dropped batch is reported by buffer_map and by download. */
static int check_failed_submit(struct geist_backend *be) {
    const struct geist_backend_primitives *p     = be->desc->prims;
    const struct geist_backend_vtbl       *vt    = be->desc->vtbl;
    struct vk_state                       *st    = be->state;
    const int64_t                          sh[1] = {N};
    struct geist_tensor                    x = host_tensor(be, 1, sh), z = host_tensor(be, 1, sh),
                                           y = host_tensor(be, 1, sh);
    int fails = geist_expect(x.buffer != nullptr && z.buffer != nullptr && y.buffer != nullptr,
                             "buffers");
    if (fails == 0) {
        fails += geist_expect(fill(be, &x, 0.5f) && fill(be, &z, 2.0f) && fill(be, &y, 7.0f),
                              "fill on a working device");
        /* Sanity: a working device delivers the add through buffer_map. */
        fails += geist_expect(p->add(be, &x, &z, &y) == GEIST_OK, "add on a working device");
        const float *yp = vt->buffer_map(y.buffer);
        fails += geist_expect(yp != nullptr && yp[0] == 2.5f && yp[N - 1] == 2.5f,
                              "buffer_map returns the add's result");
        vt->buffer_unmap(y.buffer);
        fails += geist_expect(fill(be, &y, 7.0f), "reset y");

        const PFN_vkQueueSubmit real = st->fn.QueueSubmit;
        st->fn.QueueSubmit           = fail_submit;
        fails += geist_expect(p->add(be, &x, &z, &y) == GEIST_OK, "add records into the batch");
        be->err_code      = GEIST_OK;
        const void *stale = vt->buffer_map(y.buffer); /* flushes: the submit fails */
        fails += geist_expect(stale == nullptr,
                              "buffer_map after a failed submit returns nullptr, not stale bytes");
        fails += geist_expect(be->err_code == GEIST_E_BACKEND,
                              "buffer_map after a failed submit sets GEIST_E_BACKEND");

        /* Same through download, which has a status. */
        fails += geist_expect(p->add(be, &x, &z, &y) == GEIST_OK, "second add records");
        float host[N];
        fails += geist_expect(vt->buffer_download(sizeof host, (uint8_t *) host, y.buffer) ==
                                      GEIST_E_BACKEND,
                              "download after a failed submit returns GEIST_E_BACKEND");
        st->fn.QueueSubmit = real;

        /* Reported once: the device works again and the mapping is live. */
        fails += geist_expect(p->add(be, &x, &z, &y) == GEIST_OK, "add after recovery");
        yp = vt->buffer_map(y.buffer);
        fails += geist_expect(yp != nullptr && yp[0] == 2.5f, "buffer_map after recovery");
        vt->buffer_unmap(y.buffer);
    }
    struct geist_buffer *bufs[] = {x.buffer, z.buffer, y.buffer};
    for (size_t i = 0; i < 3; i++) {
        if (bufs[i] != nullptr) {
            vt->buffer_destroy(be, bufs[i]);
        }
    }
    return fails;
}

/* Part 2: the helper itself. */
static int check_ckd_u32(void) {
    uint32_t out   = 42;
    int      fails = 0;
    fails += geist_expect(!vk_ckd_u32(0, &out) && out == 0, "0 fits");
    fails += geist_expect(!vk_ckd_u32(UINT32_MAX, &out) && out == UINT32_MAX, "UINT32_MAX fits");
    out = 42;
    fails += geist_expect(vk_ckd_u32((size_t) UINT32_MAX + 1u, &out) && out == 42,
                          "UINT32_MAX + 1 is refused and leaves *out alone");
    fails += geist_expect(vk_ckd_u32(SIZE_MAX, &out) && out == 42, "SIZE_MAX is refused");
    return fails;
}

/* Part 3: an op refuses a position the shader would see wrapped. */
static int check_attention_refuses(struct geist_backend *be) {
    const struct geist_backend_primitives *p  = be->desc->prims;
    const struct geist_backend_vtbl       *vt = be->desc->vtbl;
    /* q [1, 2, 64], k/v [4, 1, 64]: 128 / 256 floats, inside one buffer each. */
    const int64_t       qs[3] = {1, 2, 64}, ks[3] = {4, 1, 64};
    struct geist_tensor q = host_tensor(be, 3, qs), k = host_tensor(be, 3, ks),
                        v = host_tensor(be, 3, ks), o = host_tensor(be, 3, qs);
    int fails = geist_expect(q.buffer != nullptr && k.buffer != nullptr && v.buffer != nullptr &&
                                     o.buffer != nullptr,
                             "attention buffers");
    if (fails == 0) {
        fails += geist_expect(fill(be, &q, 0.1f) && fill(be, &k, 0.2f) && fill(be, &v, 0.3f) &&
                                      fill(be, &o, 0.0f),
                              "attention fill");
        fails += geist_expect(p->attention(be, &q, &k, &v, 3, 0, &o) == GEIST_OK,
                              "attention at q_offset 3");
        const size_t      wide = (size_t) UINT32_MAX + 4u; /* wraps to 3 */
        enum geist_status s    = p->attention(be, &q, &k, &v, wide, 0, &o);
        fails += geist_expect(s == GEIST_E_INVALID_ARG && be->err_code == GEIST_E_INVALID_ARG,
                              "attention refuses a q_offset past UINT32_MAX");
        s = p->attention(be, &q, &k, &v, 3, wide, &o);
        fails += geist_expect(s == GEIST_E_INVALID_ARG,
                              "attention refuses a sliding_window past UINT32_MAX");
        fails += geist_expect(vt->buffer_map(o.buffer) != nullptr, "no failure left pending");
        vt->buffer_unmap(o.buffer);
    }
    struct geist_buffer *bufs[] = {q.buffer, k.buffer, v.buffer, o.buffer};
    for (size_t i = 0; i < 4; i++) {
        if (bufs[i] != nullptr) {
            vt->buffer_destroy(be, bufs[i]);
        }
    }
    return fails;
}

/* Part 4: the x ring is sized against max_m x n_in at resolve. */
static int check_ring_at_resolve(struct geist_backend *be) {
    const struct geist_backend_vtbl *vt = be->desc->vtbl;
    struct vk_state                 *st = be->state;
    /* One F32 row: 512 x n_in x 4 B passes the 192 MB ring at 98304. */
    const size_t big_in = (size_t) VK_XRING_CAP / ((size_t) VK_MAX_M * sizeof(float)) + 256u;
    float       *raw    = xmalloc(big_in * sizeof(float));
    memset(raw, 0, big_in * sizeof(float));
    struct geist_weight w     = {.raw        = raw,
                                 .raw_nbytes = big_in * sizeof(float),
                                 .n_in       = (int32_t) big_in,
                                 .n_out      = 1,
                                 .dtype      = GEIST_DTYPE_F32};
    int                 fails = 0;
    fails += geist_expect(vt->resolve_weight(be, &w) == GEIST_E_INVALID_ARG &&
                                  w.linear_m1 == nullptr,
                          "resolve refuses an n_in whose max_m batch overflows the x ring");
    struct geist_weight ok = {.raw        = raw,
                              .raw_nbytes = 256 * sizeof(float),
                              .n_in       = 256,
                              .n_out      = 1,
                              .dtype      = GEIST_DTYPE_F32};
    fails += geist_expect(vt->resolve_weight(be, &ok) == GEIST_OK && ok.linear_m1 != nullptr,
                          "resolve accepts a weight that fits");
    fails += geist_expect(st->xring != nullptr && st->argmax_out != nullptr,
                          "the x ring and the argmax word exist after resolve");
    free(raw);
    return fails;
}

int main(void) {
    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("vulkan", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "SKIP: vulkan backend unavailable (%d)\n", (int) s);
        return GEIST_TEST_SKIP;
    }
    int fails = check_failed_submit(be);
    fails += check_ckd_u32();
    fails += check_attention_refuses(be);
    fails += check_ring_at_resolve(be);
    geist_backend_destroy(be);
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("vulkan narrowing and sticky flush: pass\n");
    return GEIST_TEST_PASS;
}

#else
int main(void) {
    fprintf(stderr, "SKIP: vulkan backend not built\n");
    return GEIST_TEST_SKIP;
}
#endif
