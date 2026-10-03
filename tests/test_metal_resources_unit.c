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
#include "model_fixtures.h"
#include <geist_util.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
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
 * (NoCopy over its pages, base_off at the bytes); so is read-only anonymous
 * memory (#577). Writable anonymous memory is copied. */
static bool aliasing(struct geist_backend *be) {
    struct metal_state *st     = be->state;
    const size_t        page   = (size_t) sysconf(_SC_PAGESIZE);
    char                path[] = "/tmp/geist_metal_alias_XXXXXX";
    const int           fd     = mkstemp(path);
    if (fd < 0 || ftruncate(fd, (off_t) (page * 4)) != 0)
        return false;
    uint8_t *maps[4] = {
            mmap(nullptr, page * 4, PROT_READ, MAP_PRIVATE, fd, 0),
            mmap(nullptr, page * 4, PROT_READ, MAP_SHARED, fd, 0),
            mmap(nullptr, page * 4, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0),
            mmap(nullptr, page * 4, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0),
    };
    const bool expect_wrapped[4] = {true, true, false, true};
    close(fd);
    unlink(path);
    bool ok = maps[3] != MAP_FAILED && mprotect(maps[3], page * 4, PROT_READ) == 0;
    for (unsigned i = 0; i < 4; i++) {
        struct geist_buffer *buf = nullptr;
        ok = ok && maps[i] != MAP_FAILED &&
             metal_buffer_create_aliased(be, maps[i] + page + 64, 256, GEIST_BUFFER_WEIGHT, &buf) ==
                     GEIST_OK;
        const bool wrapped = ok &&
                             metal_msg_send_id0(st, buf->buffer, "contents") == maps[i] + page &&
                             buf->base_off == 64;
        ok = ok && wrapped == expect_wrapped[i];
        metal_buffer_destroy(be, buf);
        if (maps[i] != MAP_FAILED)
            munmap(maps[i], page * 4);
    }
    printf("{\"phase\":\"aliasing\",\"ok\":%s}\n", ok ? "true" : "false");
    return ok;
}
/* #577: the kernel splits a large anonymous mapping into several VM entries
 * (128 MiB each on current macOS). A read-only range that crosses such a seam
 * is still wrapped in place, not copied; Gemma 4 E4B's 1.9 GB per-layer
 * embedding table is one such tensor. */
static bool aliasing_across_entries(struct geist_backend *be) {
    struct metal_state *st    = be->state;
    const size_t        page  = (size_t) sysconf(_SC_PAGESIZE);
    const size_t        bytes = (size_t) 384 << 20;
    uint8_t *map = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (map == MAP_FAILED)
        return false;
    /* Find the first entry boundary inside the mapping. */
    mach_vm_address_t                     addr  = (mach_vm_address_t) map;
    mach_vm_size_t                        size  = 0;
    natural_t                             depth = 0;
    vm_region_submap_short_info_data_64_t info  = {0};
    mach_msg_type_number_t                count = VM_REGION_SUBMAP_SHORT_INFO_COUNT_64;
    bool            ok   = mprotect(map, bytes, PROT_READ) == 0 &&
                           mach_vm_region_recurse(mach_task_self(),
                                                  &addr,
                                                  &size,
                                                  &depth,
                                                  (vm_region_recurse_info_t) &info,
                                                  &count) == KERN_SUCCESS;
    const uintptr_t seam = (uintptr_t) (addr + size);
    const bool split = ok && seam > (uintptr_t) map + page && seam < (uintptr_t) map + bytes - page;
    /* Straddle the seam when there is one; otherwise the whole mapping is one
     * entry and the range still has to wrap. */
    uint8_t             *p   = split ? (uint8_t *) seam - page - 64 : map + page + 64;
    struct geist_buffer *buf = nullptr;
    ok = ok && metal_buffer_create_aliased(be, p, page * 3, GEIST_BUFFER_WEIGHT, &buf) == GEIST_OK;
    ok = ok &&
         metal_msg_send_id0(st, buf->buffer, "contents") ==
                 (void *) ((uintptr_t) p & ~(page - 1)) &&
         buf->base_off == ((uintptr_t) p & (page - 1));
    metal_buffer_destroy(be, buf);
    munmap(map, bytes);
    printf("{\"phase\":\"aliasing-across-entries\",\"split\":%s,\"ok\":%s}\n",
           split ? "true" : "false",
           ok ? "true" : "false");
    return ok;
}
/* #528: a range inside a live Metal buffer aliases as a view of it — its
 * MTLBuffer, no new allocation — and the GPU honours the view's offset,
 * also when copying between two overlapping views of one buffer. In an open
 * batch a view is referenced by its own binds, not by a sibling's. */
static bool views(struct geist_backend *be) {
    struct metal_state              *st  = be->state;
    const struct geist_backend_vtbl *v   = be->desc->vtbl;
    const size_t                     off = 3 * 16384 + 64;
    struct geist_buffer *pool = nullptr, *src = nullptr, *a = nullptr, *b = nullptr, *c = nullptr;
    if (v->buffer_create(be, 1u << 20, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_MAPPED, &pool) !=
                GEIST_OK ||
        v->buffer_create(be, 64, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_MAPPED, &src) != GEIST_OK)
        return false;
    uint8_t *pm = v->buffer_map(pool);
    for (size_t i = 0; i < (1u << 20); i++)
        pm[i] = (uint8_t) (i * 7u);
    memset(v->buffer_map(src), 0xc3, 64);
    const unsigned long before = metal_msg_send_ulong0(st, st->device, "currentAllocatedSize");
    bool ok = v->buffer_create_aliased(be, pm + off, 4096, GEIST_BUFFER_SCRATCH, &a) == GEIST_OK &&
              v->buffer_create_aliased(be, pm + off + 256, 4096, GEIST_BUFFER_SCRATCH, &b) ==
                      GEIST_OK &&
              v->buffer_create_aliased(be, pm + off + 8192, 4096, GEIST_BUFFER_SCRATCH, &c) ==
                      GEIST_OK;
    ok      = ok && a->buffer == pool->buffer && b->buffer == pool->buffer &&
              v->buffer_map(a) == pm + off &&
              metal_msg_send_ulong0(st, st->device, "currentAllocatedSize") == before;
    /* A GPU write through the view lands at its offset and nowhere else. */
    ok = ok && v->buffer_copy(a, 16, src, 0, 64) == GEIST_OK &&
         pm[off + 15] == (uint8_t) ((off + 15) * 7u) && pm[off + 16] == 0xc3 &&
         pm[off + 79] == 0xc3 && pm[off + 80] == (uint8_t) ((off + 80) * 7u);
    /* Overlapping views of one MTLBuffer copy like memmove. */
    uint8_t want[768];
    memcpy(want, pm + off, sizeof want);
    memmove(want + 256, want, 512);
    ok            = ok && v->buffer_copy(b, 0, a, 0, 512) == GEIST_OK &&
                    memcmp(pm + off, want, sizeof want) == 0;
    ok            = ok && metal_ensure_attention_pipeline(be) == GEIST_OK;
    const int tok = ok ? v->parallel_region_begin(be, GEIST_REGION_DECODE_STEP) : 0;
    ok            = ok && tok != 0 && v->buffer_copy(a, 0, src, 0, 64) == GEIST_OK &&
                    metal_seq_references(st, a->buffer, a->base_off, a->bytes) &&
                    !metal_seq_references(st, c->buffer, c->base_off, c->bytes);
    v->parallel_region_end(be, tok);
    /* A view keeps its bytes alive past the parent's handle. */
    v->buffer_destroy(be, pool);
    ok = ok && ((const uint8_t *) v->buffer_map(a))[16] == 0xc3;
    v->buffer_destroy(be, a);
    v->buffer_destroy(be, b);
    v->buffer_destroy(be, c);
    v->buffer_destroy(be, src);
    printf("{\"phase\":\"views\",\"ok\":%s}\n", ok ? "true" : "false");
    return ok;
}
/* #541: a model loaded from a file runs on Metal without its weight pages
 * becoming private copies — on a MAP_PRIVATE mapping, wiring a NoCopy
 * buffer copy-on-writes every page it makes resident. */
static bool weights_not_copied(void) {
    char          path[]  = "/tmp/geist_metal_map_XXXXXX";
    const int     fd      = mkstemp(path);
    struct tf_buf g       = mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                                              .d_model  = 128,
                                                              .heads    = 4,
                                                              .kv_heads = 2,
                                                              .ffn      = 256,
                                                              .vocab    = 64,
                                                              .context  = 64,
                                                              .seed     = 7});
    const bool    written = fd >= 0 && write(fd, g.b, g.n) == (ssize_t) g.n;
    if (fd >= 0)
        close(fd);
    free(g.b);
    struct geist_backend *be     = nullptr;
    struct geist_model   *m      = nullptr;
    struct geist_session *s      = nullptr;
    const geist_token_t   ids[4] = {1, 2, 3, 4};
    geist_token_t         t      = 0;
    bool   ok    = written && geist_backend_create("metal", nullptr, nullptr, &be) == GEIST_OK &&
                   geist_model_load(path, be, &m) == GEIST_OK &&
                   geist_session_create(m, be, nullptr, &s) == GEIST_OK &&
                   geist_session_prefill_tokens(s, 4, ids) == GEIST_OK &&
                   geist_session_decode_step(s, &t) == GEIST_OK;
    size_t pages = 0, copied = 0;
    if (ok) {
        const struct metal_state *st = be->state;
        const size_t              pg = (size_t) getpagesize();
        for (size_t i = 0; i < st->buf_reg_count; i++) {
            const struct geist_buffer *w = st->buf_reg[i].buf;
            if (w->role != GEIST_BUFFER_WEIGHT || !(w->memory_flags & GEIST_MEMORY_ALIASED))
                continue;
            const uintptr_t lo = (uintptr_t) w->mapped & ~(pg - 1);
            const uintptr_t hi = ((uintptr_t) w->mapped + w->bytes + pg - 1) & ~(pg - 1);
            char            vec[256];
            for (uintptr_t a = lo; a < hi; a += pg * sizeof vec) {
                const size_t len = hi - a < pg * sizeof vec ? hi - a : pg * sizeof vec;
                if (mincore((void *) a, len, vec) != 0)
                    return false;
                for (size_t k = 0; k < len / pg; k++) {
                    pages += 1;
                    copied += (vec[k] & (MINCORE_COPIED | MINCORE_ANONYMOUS)) != 0;
                }
            }
        }
    }
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    unlink(path);
    ok = ok && pages > 0 && copied == 0;
    printf("{\"phase\":\"weights-not-copied\",\"pages\":%zu,\"copied\":%zu,\"ok\":%s}\n",
           pages,
           copied,
           ok ? "true" : "false");
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
    ok                     = views(be) && ok;
    ok                     = weights_not_copied() && ok;
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
    ok = aliasing_across_entries(be) && ok;
    ok = sample(be, "released") && ok;
    geist_backend_destroy(be);
    be = nullptr;
    struct geist_backend_resources value;
    ok = geist_backend_resources_snapshot(be, &value) == GEIST_E_INVALID_ARG && ok;
    return ok ? 0 : 1;
#endif
}
