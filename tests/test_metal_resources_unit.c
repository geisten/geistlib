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
/* #555: a slice of a file mapping, private or shared, is wrapped in place
 * (NoCopy over its pages, base_off at the bytes); anonymous memory is
 * copied. */
static bool aliasing(struct geist_backend *be) {
    struct metal_state *st     = be->state;
    const size_t        page   = (size_t) sysconf(_SC_PAGESIZE);
    char                path[] = "/tmp/geist_metal_alias_XXXXXX";
    const int           fd     = mkstemp(path);
    if (fd < 0 || ftruncate(fd, (off_t) (page * 4)) != 0)
        return false;
    uint8_t *maps[3] = {
            mmap(nullptr, page * 4, PROT_READ, MAP_PRIVATE, fd, 0),
            mmap(nullptr, page * 4, PROT_READ, MAP_SHARED, fd, 0),
            mmap(nullptr, page * 4, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0),
    };
    close(fd);
    unlink(path);
    bool ok = true;
    for (unsigned i = 0; i < 3; i++) {
        struct geist_buffer *buf = nullptr;
        ok = ok && maps[i] != MAP_FAILED &&
             metal_buffer_create_aliased(be, maps[i] + page + 64, 256, GEIST_BUFFER_WEIGHT, &buf) ==
                     GEIST_OK;
        const bool wrapped = ok &&
                             metal_msg_send_id0(st, buf->buffer, "contents") == maps[i] + page &&
                             buf->base_off == 64;
        ok                 = ok && wrapped == (i < 2);
        metal_buffer_destroy(be, buf);
        if (maps[i] != MAP_FAILED)
            munmap(maps[i], page * 4);
    }
    printf("{\"phase\":\"aliasing\",\"ok\":%s}\n", ok ? "true" : "false");
    return ok;
}
/* #530: a buffer joins the residency set when a dispatch first binds it and
 * leaves with its handle; a buffer nothing binds never joins. No set exists
 * before macOS 15 or with GEIST_METAL_KEEP_ALIVE_S=0. */
static bool residency(struct geist_backend *be) {
    struct metal_state *st = be->state;
    if (st->residency_set == nullptr) {
        printf("{\"phase\":\"residency\",\"set\":false}\n");
        return true;
    }
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    struct geist_buffer             *a = nullptr, *b = nullptr, *c = nullptr;
    if (v->buffer_create(be, 64, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_MAPPED, &a) != GEIST_OK ||
        v->buffer_create(be, 64, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_MAPPED, &b) != GEIST_OK ||
        v->buffer_create(be, 64, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_MAPPED, &c) != GEIST_OK ||
        metal_ensure_attention_pipeline(be) != GEIST_OK)
        return false;
    const unsigned long n0  = metal_msg_send_ulong0(st, st->residency_set, "allocationCount");
    const int           tok = v->parallel_region_begin(be, GEIST_REGION_DECODE_STEP);
    bool ok = tok != 0 && v->buffer_copy(b, 0, a, 0, 64) == GEIST_OK; /* binds a and b */
    v->parallel_region_end(be, tok);
    ok = ok && metal_msg_send_ulong0(st, st->residency_set, "allocationCount") == n0 + 2;
    v->buffer_destroy(be, a);
    ok = ok && metal_msg_send_ulong0(st, st->residency_set, "allocationCount") == n0 + 1;
    v->buffer_destroy(be, b);
    v->buffer_destroy(be, c);
    ok = ok && metal_msg_send_ulong0(st, st->residency_set, "allocationCount") == n0;
    ok = ok && st->res_keep_s == 180; /* the default keep-alive */
    setenv("GEIST_METAL_KEEP_ALIVE_S", "0", 1);
    struct geist_backend *off = nullptr;
    ok = ok && geist_backend_create("metal", nullptr, nullptr, &off) == GEIST_OK &&
         ((struct metal_state *) off->state)->residency_set == nullptr;
    geist_backend_destroy(off);
    unsetenv("GEIST_METAL_KEEP_ALIVE_S");
    printf("{\"phase\":\"residency\",\"set\":true,\"ok\":%s}\n", ok ? "true" : "false");
    return ok;
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
    struct metal_state *st = be->state;
    bool                ok = sample(be, "initial");
    ok                     = residency(be) && ok;
    size_t size            = 16 * 1024 * 1024;
    void  *shared =
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
    ok = aliasing(be) && ok;
    ok = sample(be, "released") && ok;
    geist_backend_destroy(be);
    be = nullptr;
    struct geist_backend_resources value;
    ok = geist_backend_resources_snapshot(be, &value) == GEIST_E_INVALID_ARG && ok;
    return ok ? 0 : 1;
#endif
}
