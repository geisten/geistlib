/* The Metal backend drives Metal from plain C, where nothing drains an
 * autoreleased command buffer or encoder unless the backend pushes its own
 * pool (#527: +7.2 KiB of footprint per decode token without one). Each
 * iteration takes the three submission paths — a command sequence with work,
 * a standalone compute op, a standalone blit — and once warm, phys_footprint
 * must stay flat. Opt in because this initializes a real Metal runtime. */
#include "test_helpers.h"
#include <geist.h>
#include <geist_backend.h>
#include <stdio.h>
#include <stdlib.h>
#if defined(GEIST_BACKEND_METAL) && defined(__APPLE__)
#include <mach/mach.h>

static uint64_t footprint(void) {
    task_vm_info_data_t    info  = {0};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &info, &count) != KERN_SUCCESS)
        return 0;
    return info.phys_footprint;
}
#endif
int main(void) {
#if !defined(GEIST_BACKEND_METAL) || !defined(__APPLE__)
    return GEIST_TEST_SKIP;
#else
    if (!getenv("GEIST_TEST_METAL_AUTORELEASE"))
        return GEIST_TEST_SKIP;
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
    puts("SKIP: ASan quarantines freed memory, so footprint growth says nothing here");
    return GEIST_TEST_SKIP;
#endif
#endif
    struct geist_backend *be = nullptr;
    if (geist_backend_create("metal", nullptr, nullptr, &be) != GEIST_OK)
        return GEIST_TEST_ERROR;
    const struct geist_backend_vtbl       *v = be->desc->vtbl;
    const struct geist_backend_primitives *p = be->desc->prims;
    /* Warm-up: with pools in place the footprint still climbs a few hundred
     * KiB before it plateaus (driver and malloc caches, not a leak). */
    enum { N = 1024, WARM = 1024, ITERS = 2048 };
    struct geist_buffer *buf[3] = {nullptr};
    struct geist_tensor  t[3];
    for (size_t i = 0; i < 3; i++) {
        if (v->buffer_create(
                    be, N * sizeof(float), GEIST_BUFFER_SCRATCH, GEIST_MEMORY_AUTO, &buf[i]) !=
            GEIST_OK)
            return GEIST_TEST_ERROR;
        t[i] = (struct geist_tensor) {.buffer = buf[i],
                                      .dtype  = GEIST_DTYPE_F32,
                                      .layout = GEIST_LAYOUT_DENSE,
                                      .ndim   = 1,
                                      .shape  = {N},
                                      .stride = {1}};
    }
    bool     ok     = true;
    uint64_t before = 0;
    for (size_t i = 0; i < WARM + ITERS && ok; i++) {
        if (i == WARM)
            before = footprint();
        const int tok = v->parallel_region_begin(be, GEIST_REGION_DECODE_STEP);
        ok            = tok != 0 && p->add(be, &t[0], &t[1], &t[2]) == GEIST_OK;
        v->parallel_region_end(be, tok);
        ok = ok && p->add(be, &t[0], &t[1], &t[2]) == GEIST_OK;
        ok = ok && v->buffer_copy(buf[0], 0, buf[2], 0, N * sizeof(float)) == GEIST_OK;
    }
    const uint64_t after = footprint();
    /* The #527 bound per decode token, applied per iteration. Without the
     * pools an iteration grew ~6 KiB, about 2 KiB per submission path. */
    const double per_iter = ((double) after - (double) before) / ITERS / 1024.0;
    printf("footprint %.1f -> %.1f MiB, %+.3f KiB per iteration over %d\n",
           (double) before / 1048576.0,
           (double) after / 1048576.0,
           per_iter,
           ITERS);
    ok = ok && before != 0 && per_iter <= 0.5;
    for (size_t i = 0; i < 3; i++)
        v->buffer_destroy(be, buf[i]);
    geist_backend_destroy(be);
    return ok ? 0 : GEIST_TEST_FAIL;
#endif
}
