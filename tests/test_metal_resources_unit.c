/* Provider-oracle test in the SAME backend/device, not another process/device.
 * Opt in because this initializes a real Metal runtime. Outputs are allocation
 * observations; deliberately no claim of unique physical pages or exact release. */
#include <geist_util.h>
#include "test_helpers.h"
#include <geist.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef GEIST_BACKEND_METAL
#include "../src/backends/metal/metal_internal.h"
#include <sys/mman.h>
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>

static bool sample(struct geist_backend *be, const char *phase) {
    struct metal_state            *st = be->state;
    struct geist_backend_resources value;
    if (geist_backend_resources_snapshot(be, &value) != GEIST_OK ||
        value.source != GEIST_RESOURCE_METAL_DEVICE)
        return false;
    unsigned long provider = metal_msg_send_ulong0(st, st->device, "currentAllocatedSize");
    printf("{\"phase\":\"%s\",\"allocated_bytes\":%llu,\"provider_bytes\":%lu,\"unified\":%s}\n",
           phase,
           (unsigned long long) value.allocated_bytes,
           provider,
           value.unified_memory ? "true" : "false");
    return value.allocated_bytes == provider;
}
struct observer {
    struct geist_backend *be;
    atomic_bool           done, failed;
};
static void *observe(void *arg) {
    struct observer *o = arg;
    while (!atomic_load(&o->done)) {
        struct geist_backend_resources value;
        if (geist_backend_resources_snapshot(o->be, &value) != GEIST_OK)
            atomic_store(&o->failed, true);
    }
    return nullptr;
}
#endif
int main(void) {
#ifndef GEIST_BACKEND_METAL
    return GEIST_TEST_SKIP;
#else
    if (!getenv("GEIST_TEST_METAL_RESOURCES"))
        return GEIST_TEST_SKIP;
    extern const struct geist_backend_descriptor geist_backend_metal;
    struct geist_backend                         partial = {.desc = &geist_backend_metal};
    struct geist_backend_resources               absent;
    if (geist_backend_resources_snapshot(&partial, &absent) != GEIST_E_INVALID_STATE ||
        absent.source)
        return GEIST_TEST_ERROR;
    struct geist_backend *be = nullptr;
    if (geist_backend_create("metal", nullptr, nullptr, &be) != GEIST_OK)
        return GEIST_TEST_ERROR;
    struct metal_state *st   = be->state;
    bool                ok   = sample(be, "initial");
    size_t              size = 16 * 1024 * 1024;
    void               *shared =
            metal_msg_send_id_size_uint(st, st->device, "newBufferWithLength:options:", size, 0);
    void *private = metal_msg_send_id_size_uint(
            st, st->device, "newBufferWithLength:options:", size, 2ul << 4);
    ok           = shared && private && sample(be, "shared-private") && ok;
    size_t page  = (size_t) sysconf(_SC_PAGESIZE);
    void  *pages = mmap(nullptr, page * 4, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (pages == MAP_FAILED)
        return GEIST_TEST_ERROR;
    void *aliases[3] = {
            metal_msg_send_id_ptr_size_uint_ptr(
                    st,
                    st->device,
                    "newBufferWithBytesNoCopy:length:options:deallocator:",
                    pages,
                    page * 4,
                    0,
                    nullptr),
            metal_msg_send_id_ptr_size_uint_ptr(
                    st,
                    st->device,
                    "newBufferWithBytesNoCopy:length:options:deallocator:",
                    pages,
                    page * 4,
                    0,
                    nullptr),
            metal_msg_send_id_ptr_size_uint_ptr(
                    st,
                    st->device,
                    "newBufferWithBytesNoCopy:length:options:deallocator:",
                    (char *) pages + page,
                    page * 2,
                    0,
                    nullptr),
    };
    ok         = aliases[0] && aliases[1] && aliases[2] && sample(be, "overlapping-aliases") && ok;
    void *desc = metal_msg_send_id0(st, metal_objc_get_class(st, "MTLHeapDescriptor"), "new");
    metal_msg_send_void_ulong(st, desc, "setSize:", size * 2);
    metal_msg_send_void_ulong(st, desc, "setStorageMode:", 2);
    void *heap = metal_msg_send_id_id(st, st->device, "newHeapWithDescriptor:", desc);
    void *heap_buffer =
            metal_msg_send_id_size_uint(st, heap, "newBufferWithLength:options:", size, 2ul << 4);
    ok = heap && heap_buffer && sample(be, "heap") && ok;
    /* Exhausting a fixed-size heap fails without risking host-wide OOM. */
    void *failed = metal_msg_send_id_size_uint(
            st, heap, "newBufferWithLength:options:", size * 4, 2ul << 4);
    ok                       = !failed && sample(be, "allocation-failed") && ok;
    struct observer observer = {.be = be};
    atomic_init(&observer.done, false);
    atomic_init(&observer.failed, false);
    pthread_t thread;
    if (pthread_create(&thread, nullptr, observe, &observer))
        return GEIST_TEST_ERROR;
    for (unsigned i = 0; i < 50; i++) {
        void *buffer = metal_msg_send_id_size_uint(
                st, st->device, "newBufferWithLength:options:", 65536, 0);
        if (!buffer)
            ok = false;
        metal_msg_send_void0(st, buffer, "release");
    }
    atomic_store(&observer.done, true);
    pthread_join(thread, nullptr);
    ok = !atomic_load(&observer.failed) && ok;
    for (unsigned i = 0; i < 3; i++)
        metal_msg_send_void0(st, aliases[i], "release");
    munmap(pages, page * 4);
    metal_msg_send_void0(st, shared, "release");
    metal_msg_send_void0(st, private, "release");
    metal_msg_send_void0(st, heap_buffer, "release");
    metal_msg_send_void0(st, heap, "release");
    metal_msg_send_void0(st, desc, "release");
    ok = sample(be, "released") && ok;
    geist_backend_destroy(be);
    be = nullptr;
    struct geist_backend_resources value;
    ok = geist_backend_resources_snapshot(be, &value) == GEIST_E_INVALID_ARG && ok;
    return ok ? 0 : 1;
#endif
}
