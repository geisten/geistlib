/*
 * test_backend_vulkan_dispatch_fail_unit — a failed GPU dispatch is an
 * error, not a silent host fallback (#474, item 2).
 *
 * The Vulkan ops try the GPU first and keep a host loop for geometries the
 * shaders do not cover. A command-buffer or descriptor failure (device
 * lost, out of memory) must not re-run the op on the host over mapped
 * memory and return GEIST_OK. This test makes vkBeginCommandBuffer fail and
 * calls each such op on host-visible buffers, where a host fallback would
 * succeed: every op must return GEIST_E_BACKEND.
 *
 * SKIPs (exit 0) when the Vulkan backend is not built or has no device.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <string.h>

#if defined(GEIST_BACKEND_VULKAN) && GEIST_BACKEND_VULKAN
#include "src/backends/vulkan/vk_internal.h"

static VkResult VKAPI_CALL fail_begin(VkCommandBuffer cmd, const VkCommandBufferBeginInfo *info) {
    (void) cmd;
    (void) info;
    return VK_ERROR_DEVICE_LOST;
}

enum { ROWS = 4, FEAT = 64, N = ROWS * FEAT };

/* A [ROWS, FEAT] F32 view of a host-visible buffer filled with `v`. */
static struct geist_tensor scratch_2d(struct geist_backend *be, float v) {
    const struct geist_backend_vtbl *vt = be->desc->vtbl;
    struct geist_tensor              t  = {.dtype  = GEIST_DTYPE_F32,
                                           .layout = GEIST_LAYOUT_DENSE,
                                           .ndim   = 2,
                                           .shape  = {ROWS, FEAT},
                                           .stride = {FEAT, 1}};
    if (vt->buffer_create(be, N * sizeof(float), GEIST_BUFFER_SCRATCH, 0, &t.buffer) != GEIST_OK) {
        t.buffer = nullptr;
        return t;
    }
    float *p = vt->buffer_map(t.buffer);
    for (size_t i = 0; i < N; i++) {
        p[i] = v;
    }
    vt->buffer_unmap(t.buffer);
    return t;
}

static int expect_backend_error(enum geist_status s, const char *op) {
    char msg[96];
    snprintf(msg, sizeof msg, "%s: failed dispatch returns GEIST_E_BACKEND (got %d)", op, (int) s);
    return geist_expect(s == GEIST_E_BACKEND, msg);
}

int main(void) {
    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("vulkan", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "SKIP: vulkan backend unavailable (%d)\n", (int) s);
        return GEIST_TEST_SKIP;
    }
    const struct geist_backend_primitives *p  = be->desc->prims;
    const struct geist_backend_vtbl       *vt = be->desc->vtbl;
    struct vk_state                       *st = be->state;

    struct geist_tensor x = scratch_2d(be, 0.5f);
    struct geist_tensor z = scratch_2d(be, 2.0f);
    struct geist_tensor y = scratch_2d(be, 0.0f);
    struct geist_tensor w = {.buffer = nullptr,
                             .dtype  = GEIST_DTYPE_F32,
                             .layout = GEIST_LAYOUT_DENSE,
                             .ndim   = 1,
                             .shape  = {FEAT},
                             .stride = {1}};
    if (vt->buffer_create(be, FEAT * sizeof(float), GEIST_BUFFER_SCRATCH, 0, &w.buffer) !=
        GEIST_OK) {
        w.buffer = nullptr;
    }
    int fails = geist_expect(x.buffer != nullptr && z.buffer != nullptr && y.buffer != nullptr &&
                                     w.buffer != nullptr,
                             "buffers");
    if (fails == 0) {
        /* Sanity: with a working device the same calls succeed. */
        fails += geist_expect(p->add(be, &x, &z, &y) == GEIST_OK, "add on a working device");

        const PFN_vkBeginCommandBuffer real = st->fn.BeginCommandBuffer;
        vk_seq_flush(st); /* the next dispatch opens a new command buffer */
        st->fn.BeginCommandBuffer = fail_begin;
        fails += expect_backend_error(p->add(be, &x, &z, &y), "add");
        fails += expect_backend_error(p->mul(be, &x, &z, &y), "mul");
        fails += expect_backend_error(p->gelu_tanh(be, &x, &y), "gelu_tanh");
        fails += expect_backend_error(p->silu(be, &x, &y), "silu");
        fails += expect_backend_error(p->relu_squared(be, &x, &y), "relu_squared");
        fails += expect_backend_error(p->scale_f32(be, &x, 0.5f, &y), "scale_f32");
        fails += expect_backend_error(p->rmsnorm(be, &x, &w, 1e-6f, &y), "rmsnorm");
        const struct geist_backend_fused *f = geist_backend_fused_tbl(be);
        fails += expect_backend_error(f->gelu_tanh_mul(be, &x, &z, &y), "gelu_tanh_mul");
        fails += expect_backend_error(f->silu_mul(be, &x, &z, &y), "silu_mul");
        fails += expect_backend_error(f->sigmoid_mul(be, &x, &z, &y), "sigmoid_mul");
        fails += expect_backend_error(f->rmsnorm_add(be, &z, &x, &w, 1e-6f, &y), "rmsnorm_add");
        st->fn.BeginCommandBuffer = real;

        /* The device works again once the failure is gone. */
        fails += geist_expect(p->add(be, &x, &z, &y) == GEIST_OK, "add after recovery");
    }
    struct geist_buffer *bufs[] = {x.buffer, z.buffer, y.buffer, w.buffer};
    for (size_t i = 0; i < 4; i++) {
        if (bufs[i] != nullptr) {
            vt->buffer_destroy(be, bufs[i]);
        }
    }
    geist_backend_destroy(be);
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("vulkan dispatch failure: pass\n");
    return GEIST_TEST_PASS;
}

#else
int main(void) {
    fprintf(stderr, "SKIP: vulkan backend not built\n");
    return GEIST_TEST_SKIP;
}
#endif
