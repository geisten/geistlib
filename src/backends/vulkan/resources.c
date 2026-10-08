/*
 * src/backends/vulkan/resources.c — buffers, staging, the weight registry, and tensor accessors.
 *
 * Layer: BACKEND (vulkan).
 */
#include "vk_internal.h"

#include "checked.h"
#include "tensor_view.h" /* geist_tensor_elems: checked element count */

/* ====================================================================== */
/* Buffers                                                                 */
/* ====================================================================== */

[[nodiscard]] static uint32_t
vk_find_mem_type(const struct vk_state *st, uint32_t type_bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < st->mem_props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (st->mem_props.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return UINT32_MAX;
}

/* The device-memory limit for a device-local allocation from `mem_type`:
 * its heap, lowered by GEIST_VK_VRAM_BUDGET. */
static size_t vk_vram_limit(const struct vk_state *st, uint32_t mem_type) {
    const uint32_t heap = st->mem_props.memoryTypes[mem_type].heapIndex;
    const size_t   size = (size_t) st->mem_props.memoryHeaps[heap].size;
    return st->vram_budget != 0 && st->vram_budget < size ? st->vram_budget : size;
}

[[nodiscard]] static bool vk_vram_fits(const struct vk_state *st, uint32_t mem_type, size_t n) {
    const size_t limit = vk_vram_limit(st, mem_type);
    return st->vram_used <= limit && n <= limit - st->vram_used;
}

/* The driver's view of heap `heap` (VK_EXT_memory_budget): `budget` is what
 * this process may allocate from it, which already leaves out what other
 * processes hold, and `usage` is what this process holds. false without the
 * extension. */
[[nodiscard]] static bool
vk_heap_budget(const struct vk_state *st, uint32_t heap, size_t *budget, size_t *usage) {
    if (!st->has_mem_budget || heap >= VK_MAX_MEMORY_HEAPS) {
        return false;
    }
    VkPhysicalDeviceMemoryBudgetPropertiesEXT b = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 p = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &b};
    st->fn.GetPhysicalDeviceMemoryProperties2(st->phys, &p);
    if (b.heapBudget[heap] == 0) {
        return false; /* a driver that does not fill the struct */
    }
    *budget = (size_t) b.heapBudget[heap];
    *usage  = (size_t) b.heapUsage[heap];
    return true;
}

bool vk_weight_fits_vram(const struct vk_state *st, size_t n) {
    const uint32_t type = vk_find_mem_type(st, UINT32_MAX, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX) {
        return false;
    }
    const size_t limit   = vk_vram_limit(st, type);
    const size_t reserve = st->weight_reserve == SIZE_MAX ? limit / 16u : st->weight_reserve;
    size_t       free    = limit > st->vram_used ? limit - st->vram_used : 0;
    size_t       budget = 0, usage = 0;
    /* what other processes hold is not ours to fill (VK_EXT_memory_budget) */
    if (vk_heap_budget(st, st->mem_props.memoryTypes[type].heapIndex, &budget, &usage)) {
        const size_t avail = budget > usage ? budget - usage : 0;
        free               = avail < free ? avail : free;
    }
    return free > reserve && n <= free - reserve;
}

/* Bytes of heap `heap` in use on the whole device, by every process, as the
 * driver reports it: the heap minus what this process can still allocate. */
[[nodiscard]] static bool vk_heap_in_use(const struct vk_state *st, uint32_t heap, size_t *out) {
    size_t budget = 0, usage = 0;
    if (!vk_heap_budget(st, heap, &budget, &usage)) {
        return false;
    }
    const size_t size  = (size_t) st->mem_props.memoryHeaps[heap].size;
    const size_t avail = budget > usage ? budget - usage : 0;
    *out               = size > avail ? size - avail : 0;
    return true;
}

/* The error of a device-local allocation that does not fit: what it needs,
 * what is in use and the limit, so a model larger than the device fails
 * with numbers instead of a bare driver status. Weights spill to host
 * memory before they get here (vk_resolve_weight); KV and scratch do not.
 * With VK_EXT_memory_budget it also names what the whole device holds: when
 * another model or process has the memory, this backend's own count alone
 * makes the failure look impossible. */
static void vk_vram_exhausted(struct geist_backend  *be,
                              uint32_t               mem_type,
                              size_t                 n,
                              enum geist_buffer_role role) {
    const struct vk_state *st     = be->state;
    const uint32_t         heap   = st->mem_props.memoryTypes[mem_type].heapIndex;
    const char            *what   = role == GEIST_BUFFER_KV_CACHE ? "KV-cache" : "device";
    const char            *capped = st->vram_budget != 0 ? " (GEIST_VK_VRAM_BUDGET)" : "";
    size_t                 device = 0;
    if (vk_heap_in_use(st, heap, &device)) {
        geist_backend_set_error(be,
                                GEIST_E_OOM,
                                "vulkan: out of device memory: a %s buffer needs %zu MiB, %zu of "
                                "%zu MiB are in use by this backend%s; the device reports %zu of "
                                "%zu MiB in use (other models or processes included)",
                                what,
                                (n + (1u << 20) - 1) >> 20,
                                st->vram_used >> 20,
                                vk_vram_limit(st, mem_type) >> 20,
                                capped,
                                device >> 20,
                                (size_t) st->mem_props.memoryHeaps[heap].size >> 20);
        return;
    }
    geist_backend_set_error(be,
                            GEIST_E_OOM,
                            "vulkan: out of device memory: a %s buffer needs %zu MiB, %zu of "
                            "%zu MiB are in use%s (the model does not fit)",
                            what,
                            (n + (1u << 20) - 1) >> 20,
                            st->vram_used >> 20,
                            vk_vram_limit(st, mem_type) >> 20,
                            capped);
}

/* The heap device-local buffers (weights, KV cache) are placed in, its size
 * lowered by GEIST_VK_VRAM_BUDGET, and what is left of it: the driver's
 * budget minus this process's usage when VK_EXT_memory_budget is there (every
 * process counted), else the limit minus this backend's own allocations. */
enum geist_status vk_memory_info(const struct geist_backend *be, struct geist_backend_memory *out) {
    const struct vk_state *st = be->state;
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return GEIST_E_INVALID_STATE;
    }
    const uint32_t type = vk_find_mem_type(st, UINT32_MAX, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX) {
        return GEIST_E_UNSUPPORTED;
    }
    const size_t limit  = vk_vram_limit(st, type);
    size_t       free   = limit > st->vram_used ? limit - st->vram_used : 0;
    size_t       budget = 0, usage = 0;
    const bool   wide =
            vk_heap_budget(st, st->mem_props.memoryTypes[type].heapIndex, &budget, &usage);
    if (wide) {
        const size_t avail = budget > usage ? budget - usage : 0;
        free               = avail < free ? avail : free;
    }
    *out = (struct geist_backend_memory) {.total_bytes    = limit,
                                          .free_bytes     = free,
                                          .device_wide    = wide,
                                          .unified_memory = st->unified_memory};
    return GEIST_OK;
}

[[nodiscard]] enum geist_status vk_buffer_create(struct geist_backend  *be,
                                                 size_t                 bytes,
                                                 enum geist_buffer_role role,
                                                 unsigned int           memory_flags,
                                                 struct geist_buffer  **out) {
    struct vk_state *st = be->state;
    if (bytes == 0 || out == nullptr) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan: bad buffer_create args");
        return GEIST_E_INVALID_ARG;
    }
    /* Everything the engine creates must satisfy the arch layer's
     * buffer_map contract (it maps WEIGHT-role cos/sin tables for CPU rope,
     * scratch pools, logits...), so default is host-visible. DEVICE_LOCAL
     * VRAM on explicit GEIST_MEMORY_DEVICE (resolve_weight copies, x ring)
     * and for KV_CACHE-role buffers — the arch touches dense KV only via
     * buffer_copy and v->attention, and decode attention re-reads the whole
     * cache every token (host-resident KV = hundreds of MB/token of PCIe). */
    const bool host_req     = (memory_flags & (GEIST_MEMORY_HOST | GEIST_MEMORY_HOST_VISIBLE |
                                               GEIST_MEMORY_MAPPED)) != 0;
    const bool device_local = !host_req && ((memory_flags & GEIST_MEMORY_DEVICE) != 0 ||
                                            role == GEIST_BUFFER_KV_CACHE);

    struct geist_buffer *buf = geist_backend_alloc(be, sizeof(*buf), alignof(struct geist_buffer));
    if (buf == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "vulkan: buffer handle alloc failed");
        return GEIST_E_OOM;
    }
    *buf = (struct geist_buffer) {.owner        = st,
                                  .bytes        = bytes,
                                  .role         = role,
                                  .memory_flags = memory_flags,
                                  .host_visible = !device_local};

    VkBufferCreateInfo binfo = {.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .size        = bytes,
                                .usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    if (st->fn.CreateBuffer(st->device, &binfo, nullptr, &buf->buf) != VK_SUCCESS) {
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: vkCreateBuffer(%zu) failed", bytes);
        return GEIST_E_BACKEND;
    }
    VkMemoryRequirements req;
    st->fn.GetBufferMemoryRequirements(st->device, buf->buf, &req);
    const VkMemoryPropertyFlags want = device_local ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                                    : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t                    mem_type = UINT32_MAX;
    if (!device_local && bytes <= (200u << 20) && role != GEIST_BUFFER_STAGING &&
        role != GEIST_BUFFER_IO && (memory_flags & GEIST_MEMORY_HOST) == 0) {
        /* BAR helps buffers the GPU reads hot and the host rarely touches
         * (scratch pool, rope tables). Staging/IO stay in system RAM: the
         * host READS those, and CPU reads from BAR are uncached PCIe. */
        /* Small host-visible buffers (scratch pool, staging, tables) go
         * into the BAR window when available: DEVICE_LOCAL + HOST_VISIBLE
         * means GPU ops touch activations at VRAM speed while the arch's
         * buffer_map contract still holds. The 256 MB heap is precious —
         * big allocations (weight arena) stay in system RAM. */
        mem_type = vk_find_mem_type(st,
                                    req.memoryTypeBits,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    if (mem_type == UINT32_MAX) {
        mem_type = vk_find_mem_type(st, req.memoryTypeBits, want);
    }
    if (mem_type == UINT32_MAX && device_local) {
        /* VRAM exhausted or odd heap layout — host-visible still works. */
        mem_type          = vk_find_mem_type(st,
                                             req.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        buf->host_visible = mem_type != UINT32_MAX;
    }
    if (mem_type == UINT32_MAX) {
        st->fn.DestroyBuffer(st->device, buf->buf, nullptr);
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: no memory type for buffer");
        return GEIST_E_BACKEND;
    }
    VkMemoryAllocateInfo minfo = {.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                  .allocationSize  = req.size,
                                  .memoryTypeIndex = mem_type};
    const bool           vram = device_local && (st->mem_props.memoryTypes[mem_type].propertyFlags &
                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    if (vram && !vk_vram_fits(st, mem_type, (size_t) req.size)) {
        st->fn.DestroyBuffer(st->device, buf->buf, nullptr);
        geist_backend_free(be, buf);
        vk_vram_exhausted(be, mem_type, (size_t) req.size, role);
        return GEIST_E_OOM;
    }
    VkResult r = st->fn.AllocateMemory(st->device, &minfo, nullptr, &buf->mem);
    if (r != VK_SUCCESS && !device_local) {
        /* BAR heap exhausted (it is only 256 MB) — fall back to plain
         * host-visible system memory. */
        const uint32_t fb = vk_find_mem_type(st,
                                             req.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (fb != UINT32_MAX && fb != mem_type) {
            minfo.memoryTypeIndex = fb;
            mem_type              = fb;
            r                     = st->fn.AllocateMemory(st->device, &minfo, nullptr, &buf->mem);
        }
    }
    if (r == VK_SUCCESS) {
        r = st->fn.BindBufferMemory(st->device, buf->buf, buf->mem, 0);
    }
    if (r == VK_SUCCESS && buf->host_visible) {
        r = st->fn.MapMemory(st->device, buf->mem, 0, VK_WHOLE_SIZE, 0, &buf->mapped);
    }
    if (r != VK_SUCCESS) {
        if (buf->mem != VK_NULL_HANDLE) {
            st->fn.FreeMemory(st->device, buf->mem, nullptr);
        }
        st->fn.DestroyBuffer(st->device, buf->buf, nullptr);
        geist_backend_free(be, buf);
        if (vram && r == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
            vk_vram_exhausted(be, mem_type, (size_t) req.size, role);
        } else {
            geist_backend_set_error(
                    be, GEIST_E_OOM, "vulkan: allocating %zu bytes failed (%d)", bytes, (int) r);
        }
        return GEIST_E_OOM;
    }
    if (vram) {
        buf->vram_bytes = (size_t) minfo.allocationSize;
        st->vram_used += buf->vram_bytes;
    }
    buf->device_mem = (st->mem_props.memoryTypes[mem_type].propertyFlags &
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    if (buf->host_visible && buf->device_mem) {
        buf->bar_bytes = (size_t) minfo.allocationSize;
        st->bar_used += buf->bar_bytes;
    }
    const bool verbose = getenv("GEIST_VK_VERBOSE") != nullptr;
    if (verbose && buf->host_visible) {
        fprintf(stderr,
                "  buffer %zu KiB role=%d bar=%d\n",
                bytes >> 10,
                (int) role,
                buf->device_mem);
    }
    if (role == GEIST_BUFFER_SCRATCH) {
        const enum vk_placement where = vk_buffer_placement(buf);
        st->stat_scratch_n[where]++;
        st->stat_scratch_bytes[where] += bytes;
        if (verbose && where == VK_PLACEMENT_HOST) {
            fprintf(stderr,
                    "geist vulkan: scratch buffer of %zu KiB is in host memory, outside "
                    "device-local memory; GPU ops on it read over the bus (#488)\n",
                    bytes >> 10);
        }
    }
    /* Register mapped buffers for the alias-containment lookup (arch pools
     * hand out slices of these; buffer_create_aliased resolves them back
     * to (VkBuffer, offset) so GPU ops can bind pool slices directly). */
    if (buf->mapped != nullptr) {
        if (st->n_hostbufs == st->cap_hostbufs) {
            const size_t          cap = st->cap_hostbufs == 0 ? 32 : st->cap_hostbufs * 2;
            struct geist_buffer **nb =
                    geist_backend_alloc(be, cap * sizeof(*nb), alignof(struct geist_buffer *));
            if (nb != nullptr) {
                if (st->n_hostbufs > 0) { /* hostbufs is nullptr before the first grow */
                    memcpy(nb, st->hostbufs, st->n_hostbufs * sizeof(*nb));
                }
                geist_backend_free(be, st->hostbufs);
                st->hostbufs     = nb;
                st->cap_hostbufs = cap;
            }
        }
        if (st->n_hostbufs < st->cap_hostbufs) {
            st->hostbufs[st->n_hostbufs++] = buf;
        }
    }
    *out = buf;
    return GEIST_OK;
}

/* The vtable's buffer_create. One policy on top of vk_buffer_create: the
 * arch asks for its scratch pool device-local (SCRATCH role, DEVICE flag, no
 * host flag) when its host paths never map the pool (#488). Honoured unless
 * GEIST_VK_SCRATCH_DEVICE=0; then the request is served
 * host-visible and the arch keeps its mapped pool. Internal callers (the x
 * ring, weight copies) call vk_buffer_create and are not affected. */
[[nodiscard]] enum geist_status vk_buffer_create_api(struct geist_backend  *be,
                                                     size_t                 bytes,
                                                     enum geist_buffer_role role,
                                                     unsigned int           memory_flags,
                                                     struct geist_buffer  **out) {
    const struct vk_state *st = be->state;
    const unsigned int     host_req =
            GEIST_MEMORY_HOST | GEIST_MEMORY_HOST_VISIBLE | GEIST_MEMORY_MAPPED;
    if (role == GEIST_BUFFER_SCRATCH && (memory_flags & GEIST_MEMORY_DEVICE) != 0 &&
        (memory_flags & host_req) == 0 && !st->scratch_device) {
        memory_flags &= ~(unsigned int) GEIST_MEMORY_DEVICE;
    }
    return vk_buffer_create(be, bytes, role, memory_flags, out);
}

/* A slice of a buffer_create buffer by offset (vtbl buffer_create_view): the
 * same borrowed handle buffer_create_aliased hands out for a mapped parent,
 * found by (parent, offset) instead of by host address, so it works for a
 * device-local parent as well. */
[[nodiscard]] enum geist_status vk_buffer_create_view(struct geist_backend  *be,
                                                      struct geist_buffer   *parent,
                                                      size_t                 offset,
                                                      size_t                 n_bytes,
                                                      enum geist_buffer_role role,
                                                      struct geist_buffer  **out) {
    struct vk_state *st = be->state;
    if (parent == nullptr || out == nullptr || parent->owner != st ||
        parent->buf == VK_NULL_HANDLE || n_bytes == 0 || offset > parent->bytes ||
        n_bytes > parent->bytes - offset) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan: bad buffer view args");
        return GEIST_E_INVALID_ARG;
    }
    struct geist_buffer *buf = geist_backend_alloc(be, sizeof(*buf), alignof(struct geist_buffer));
    if (buf == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "vulkan: view handle alloc failed");
        return GEIST_E_OOM;
    }
    uint8_t *host = parent->host_alias != nullptr ? parent->host_alias : parent->mapped;
    *buf          = (struct geist_buffer) {.owner        = st,
                                           .buf          = parent->buf,
                                           .host_alias   = host != nullptr ? host + offset : nullptr,
                                           .bytes        = n_bytes,
                                           .base_off     = parent->base_off + offset,
                                           .role         = role,
                                           .memory_flags = GEIST_MEMORY_ALIASED,
                                           .host_visible = parent->host_visible,
                                           .device_mem   = parent->device_mem,
                                           .borrowed     = true,
                                           .view         = true};
    *out          = buf;
    return GEIST_OK;
}

enum vk_placement vk_buffer_placement(const struct geist_buffer *buf) {
    if (buf == nullptr || buf->buf == VK_NULL_HANDLE) {
        return VK_PLACEMENT_NONE;
    }
    if (!buf->device_mem) {
        return VK_PLACEMENT_HOST;
    }
    return buf->host_visible ? VK_PLACEMENT_BAR : VK_PLACEMENT_DEVICE;
}

[[nodiscard]] enum geist_status vk_buffer_create_aliased(struct geist_backend  *be,
                                                         void                  *host_ptr,
                                                         size_t                 n_bytes,
                                                         enum geist_buffer_role role,
                                                         struct geist_buffer  **out) {
    /* A handle on a host region (scratch-pool slice, weight arena, GGUF
     * mmap) that owns no Vulkan resources. Device copies of aliased weights
     * are made at resolve_weight time, keyed by this pointer. buffer_map
     * returns the pointer unchanged, so the arch's CPU paths keep working. */
    struct vk_state *st = be->state;
    if (host_ptr == nullptr || out == nullptr) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan: bad aliased-buffer args");
        return GEIST_E_INVALID_ARG;
    }
    struct geist_buffer *buf = geist_backend_alloc(be, sizeof(*buf), alignof(struct geist_buffer));
    if (buf == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "vulkan: aliased handle alloc failed");
        return GEIST_E_OOM;
    }
    *buf = (struct geist_buffer) {.owner        = st,
                                  .host_alias   = host_ptr,
                                  .bytes        = n_bytes,
                                  .role         = role,
                                  .memory_flags = GEIST_MEMORY_ALIASED,
                                  .host_visible = true};
    /* If the region lives inside one of our mapped buffers (arch scratch
     * pool / weight arena), borrow its VkBuffer so GPU ops can
     * bind this slice. Pointers outside any known buffer (GGUF mmap) stay
     * pure bookkeeping — ops on them fall back to the CPU path. */
    for (size_t i = 0; i < st->n_hostbufs; ++i) {
        struct geist_buffer *p  = st->hostbufs[i];
        const uint8_t       *lo = p->mapped, *ptr = host_ptr;
        if (ptr >= lo && ptr + n_bytes <= lo + p->bytes) {
            buf->buf        = p->buf;
            buf->base_off   = (size_t) (ptr - lo);
            buf->borrowed   = true;
            buf->device_mem = p->device_mem;
            break;
        }
    }
    *out = buf;
    return GEIST_OK;
}

void vk_buffer_destroy(struct geist_backend *be, struct geist_buffer *buf) {
    if (buf == nullptr) {
        return;
    }
    struct vk_state *st = buf->owner;
    vk_seq_flush(st); /* the open batch may reference this buffer */
    if (buf->buf != VK_NULL_HANDLE && !buf->borrowed) {
        /* drop cached descriptor sets that reference this buffer — the
         * driver may recycle the handle value for a future buffer */
        for (uint32_t i = 0; i < VK_DSET_CACHE; ++i) {
            if (st->dset_cache[i].key != 0) {
                st->dset_cache[i].key = UINT64_MAX; /* tombstone: never matches */
            }
        }
        for (size_t i = 0; i < st->n_hostbufs; ++i) {
            if (st->hostbufs[i] == buf) {
                st->hostbufs[i] = st->hostbufs[--st->n_hostbufs];
                break;
            }
        }
        if (buf->mapped != nullptr) {
            st->fn.UnmapMemory(st->device, buf->mem);
        }
        st->fn.DestroyBuffer(st->device, buf->buf, nullptr);
        st->fn.FreeMemory(st->device, buf->mem, nullptr);
        st->bar_used -= buf->bar_bytes;
        st->vram_used -= buf->vram_bytes;
    }
    geist_backend_free(be, buf);
}

/* Scratch that the GPU reads hot and the host maps lives in the BAR window
 * (host-visible + device-local). Without resizable BAR that window is 256 MB
 * and everything past it falls back to system memory read over PCIe, so the
 * arch sizes its default prefill chunk to the room that is left. A heap of
 * 1 GiB or more is resizable BAR or unified memory: no limit worth planning
 * for. `reserve` leaves room for other users of the window (the desktop). */
size_t vk_fast_host_bytes(struct geist_backend *be) {
    const struct vk_state *st   = be->state;
    size_t                 heap = 0;
    for (uint32_t i = 0; i < st->mem_props.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags f    = st->mem_props.memoryTypes[i].propertyFlags;
        const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((f & want) == want) {
            const size_t h =
                    (size_t) st->mem_props.memoryHeaps[st->mem_props.memoryTypes[i].heapIndex].size;
            heap = h > heap ? h : heap;
        }
    }
    constexpr size_t reserve = 48u << 20;
    if (heap == 0 || heap >= (size_t) (1u << 30)) {
        return SIZE_MAX;
    }
    return heap > st->bar_used + reserve ? heap - st->bar_used - reserve : 0;
}

/* Copies `size` bytes between buf (at its base_off + off) and `staging`
 * (at 0) on the transfer command buffer and waits for it. */
[[nodiscard]] static enum geist_status vk_xfer_wait(struct vk_state     *st,
                                                    struct geist_buffer *buf,
                                                    VkBuffer             staging,
                                                    size_t               off,
                                                    size_t               size,
                                                    bool                 upload) {
    struct geist_backend    *be     = st->backend;
    VkCommandBufferBeginInfo begin  = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                       .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    const VkBufferCopy       region = {.srcOffset = upload ? 0 : buf->base_off + off,
                                       .dstOffset = upload ? buf->base_off + off : 0,
                                       .size      = size};
    if (st->fn.BeginCommandBuffer(st->xfer_cmd, &begin) != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: begin transfer cmd failed");
        return GEIST_E_BACKEND;
    }
    if (upload) {
        st->fn.CmdCopyBuffer(st->xfer_cmd, staging, buf->buf, 1, &region);
    } else {
        st->fn.CmdCopyBuffer(st->xfer_cmd, buf->buf, staging, 1, &region);
    }
    VkSubmitInfo submit = {.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                           .commandBufferCount = 1,
                           .pCommandBuffers    = &st->xfer_cmd};
    if (st->fn.EndCommandBuffer(st->xfer_cmd) != VK_SUCCESS ||
        st->fn.QueueSubmit(st->queue, 1, &submit, st->xfer_fence) != VK_SUCCESS ||
        st->fn.WaitForFences(st->device, 1, &st->xfer_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: transfer submit/wait failed");
        return GEIST_E_BACKEND;
    }
    (void) st->fn.ResetFences(st->device, 1, &st->xfer_fence);
    return GEIST_OK;
}

/* A host-visible, mapped staging buffer of `size` bytes. */
[[nodiscard]] static enum geist_status vk_staging_create(struct vk_state *st,
                                                         size_t           size,
                                                         VkBuffer        *out_buf,
                                                         VkDeviceMemory  *out_mem,
                                                         void           **out_map) {
    VkBufferCreateInfo binfo = {.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .size        = size,
                                .usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    *out_buf                 = VK_NULL_HANDLE;
    *out_mem                 = VK_NULL_HANDLE;
    *out_map                 = nullptr;
    if (st->fn.CreateBuffer(st->device, &binfo, nullptr, out_buf) != VK_SUCCESS) {
        geist_backend_set_error(st->backend, GEIST_E_BACKEND, "vulkan: staging buffer failed");
        return GEIST_E_BACKEND;
    }
    VkMemoryRequirements req;
    st->fn.GetBufferMemoryRequirements(st->device, *out_buf, &req);
    const uint32_t       mem_type = vk_find_mem_type(st,
                                                     req.memoryTypeBits,
                                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo minfo    = {.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                     .allocationSize  = req.size,
                                     .memoryTypeIndex = mem_type};
    if (mem_type == UINT32_MAX ||
        st->fn.AllocateMemory(st->device, &minfo, nullptr, out_mem) != VK_SUCCESS ||
        st->fn.BindBufferMemory(st->device, *out_buf, *out_mem, 0) != VK_SUCCESS ||
        st->fn.MapMemory(st->device, *out_mem, 0, VK_WHOLE_SIZE, 0, out_map) != VK_SUCCESS) {
        if (*out_mem != VK_NULL_HANDLE) {
            st->fn.FreeMemory(st->device, *out_mem, nullptr);
        }
        st->fn.DestroyBuffer(st->device, *out_buf, nullptr);
        *out_buf = VK_NULL_HANDLE;
        *out_mem = VK_NULL_HANDLE;
        *out_map = nullptr;
        geist_backend_set_error(
                st->backend, GEIST_E_OOM, "vulkan: staging alloc/map failed (%zu B)", size);
        return GEIST_E_OOM;
    }
    return GEIST_OK;
}

/* ponytail: 64 MiB, synchronous chunks; double-buffer (memcpy of chunk i+1
 * while chunk i transfers) if loads still show up as the bottleneck. */
constexpr size_t VK_UP_STAGE_BYTES = (size_t) 64 << 20;

/* Upload through the persistent staging buffer, VK_UP_STAGE_BYTES at a time. */
[[nodiscard]] static enum geist_status
vk_staged_upload(struct geist_buffer *buf, size_t n_bytes, const uint8_t *src) {
    struct vk_state *st = buf->owner;
    if (st->up_buf == VK_NULL_HANDLE) {
        void                   *map = nullptr;
        const enum geist_status s =
                vk_staging_create(st, VK_UP_STAGE_BYTES, &st->up_buf, &st->up_mem, &map);
        if (s != GEIST_OK) {
            return s;
        }
        st->up_map = map;
    }
    for (size_t off = 0; off < n_bytes; off += VK_UP_STAGE_BYTES) {
        const size_t k = n_bytes - off < VK_UP_STAGE_BYTES ? n_bytes - off : VK_UP_STAGE_BYTES;
        memcpy(st->up_map, src + off, k);
        const enum geist_status s = vk_xfer_wait(st, buf, st->up_buf, off, k, true);
        if (s != GEIST_OK) {
            return s;
        }
    }
    return GEIST_OK;
}

/* One blocking download through a staging buffer of its own size (reads are
 * rare and small: logits, test checks). */
[[nodiscard]] static enum geist_status
vk_staged_download(struct geist_buffer *buf, size_t n_bytes, uint8_t *dst) {
    struct vk_state  *st      = buf->owner;
    VkBuffer          staging = VK_NULL_HANDLE;
    VkDeviceMemory    mem     = VK_NULL_HANDLE;
    void             *map     = nullptr;
    enum geist_status s       = vk_staging_create(st, n_bytes, &staging, &mem, &map);
    if (s != GEIST_OK) {
        return s;
    }
    s = vk_xfer_wait(st, buf, staging, 0, n_bytes, false);
    if (s == GEIST_OK) {
        memcpy(dst, map, n_bytes);
    }
    st->fn.UnmapMemory(st->device, mem);
    st->fn.FreeMemory(st->device, mem, nullptr);
    st->fn.DestroyBuffer(st->device, staging, nullptr);
    return s;
}

[[nodiscard]] enum geist_status
vk_buffer_upload(struct geist_buffer *buf, size_t n_bytes, const uint8_t src[static n_bytes]) {
    if (buf == nullptr || n_bytes > buf->bytes) {
        return GEIST_E_INVALID_ARG;
    }
    vk_seq_flush(buf->owner);
    if (buf->host_alias != nullptr) {
        memcpy(buf->host_alias, src, n_bytes);
        return GEIST_OK;
    }
    if (buf->mapped != nullptr) {
        memcpy(buf->mapped, src, n_bytes);
        return GEIST_OK;
    }
    return vk_staged_upload(buf, n_bytes, src);
}

[[nodiscard]] enum geist_status
vk_buffer_download(size_t n_bytes, uint8_t dst[static n_bytes], const struct geist_buffer *buf) {
    if (buf == nullptr || n_bytes > buf->bytes) {
        return GEIST_E_INVALID_ARG;
    }
    vk_seq_flush(buf->owner);
    if (vk_seq_take_failure(buf->owner) != GEIST_OK) {
        return GEIST_E_BACKEND;
    }
    if (buf->host_alias != nullptr) {
        memcpy(dst, buf->host_alias, n_bytes);
        return GEIST_OK;
    }
    if (buf->mapped != nullptr) {
        memcpy(dst, buf->mapped, n_bytes);
        return GEIST_OK;
    }
    return vk_staged_download((struct geist_buffer *) buf, n_bytes, dst);
}

void *vk_buffer_map(struct geist_buffer *buf) {
    if (buf == nullptr) {
        return nullptr;
    }
    vk_seq_flush(buf->owner); /* host is about to read/write — drain the batch */
    void *p = buf->host_alias != nullptr ? buf->host_alias : buf->mapped;
    if (p == nullptr) {
        return nullptr; /* device-local — caller falls back (download reports a failure) */
    }
    /* A dropped batch leaves the mapping holding whatever was there before:
     * hand out no pointer rather than stale results. buffer_map has no status,
     * so the failure is the backend error and the caller's nullptr check. */
    if (vk_seq_take_failure(buf->owner) != GEIST_OK) {
        return nullptr;
    }
    return p;
}

void vk_buffer_unmap(struct geist_buffer *buf) {
    (void) buf; /* persistent coherent mappings — nothing to flush */
}

/* ====================================================================== */
/* Staging and the weight registry                                         */
/*                                                                         */
/* The host-pointer linear kernels get x, y and w->raw as host pointers:   */
/* weights were copied to VRAM at resolve time, x/y round-trip through     */
/* persistent host-visible staging (one submit + fence wait per call).     */
/* The hot path is linear_t on device-resident activations (ops.c).        */
/* ====================================================================== */

[[nodiscard]] enum geist_status vk_stage_reserve_role(struct geist_backend  *be,
                                                      struct geist_buffer  **slot,
                                                      size_t                 bytes,
                                                      enum geist_buffer_role role) {
    if (*slot != nullptr && (*slot)->bytes >= bytes) {
        return GEIST_OK;
    }
    if (*slot != nullptr) {
        vk_buffer_destroy(be, *slot);
        *slot = nullptr;
    }
    size_t cap = 1u << 20; /* 1 MiB floor, then powers of two */
    while (cap < bytes) {
        cap *= 2;
    }
    return vk_buffer_create(be, cap, role, GEIST_MEMORY_AUTO, slot);
}

[[nodiscard]] enum geist_status
vk_stage_reserve(struct geist_backend *be, struct geist_buffer **slot, size_t bytes) {
    return vk_stage_reserve_role(be, slot, bytes, GEIST_BUFFER_STAGING);
}

/* Registry copy of the weight a tensor views (nullptr: not resolved, or the
 * tensor is not host-aliased). */
struct geist_buffer *vk_weight_of(struct vk_state *st, const struct geist_tensor *t) {
    if (t == nullptr || t->buffer == nullptr || t->buffer->host_alias == nullptr) {
        return nullptr;
    }
    return vk_weight_lookup(st, (const uint8_t *) t->buffer->host_alias + t->offset);
}

/* Home slot of `host` in a table of `cap` (a power of two) slots:
 * Fibonacci hashing of the pointer bits above the 16-byte alignment. */
static size_t vk_weight_slot(const void *host, size_t cap) {
    const uint64_t h = ((uint64_t) (uintptr_t) host >> 4) * 0x9E3779B97F4A7C15ull;
    return (size_t) (h >> 32) & (cap - 1);
}

struct vk_weight_entry *vk_weight_entry_of(const struct vk_state *st, const void *host) {
    if (st->cap_weight_index == 0) {
        return nullptr;
    }
    const size_t mask = st->cap_weight_index - 1;
    for (size_t i = vk_weight_slot(host, st->cap_weight_index);; i = (i + 1) & mask) {
        const uint32_t e = st->weight_index[i];
        if (e == 0) {
            return nullptr; /* the table is at most half full: always ends */
        }
        if (st->weights[e - 1].host == host) {
            return &st->weights[e - 1];
        }
    }
}

struct geist_buffer *vk_weight_lookup(const struct vk_state *st, const void *host) {
    const struct vk_weight_entry *e = vk_weight_entry_of(st, host);
    return e != nullptr ? e->gpu : nullptr;
}

static void vk_weight_index_put(uint32_t *table, size_t cap, const void *host, size_t idx) {
    size_t i = vk_weight_slot(host, cap);
    while (table[i] != 0) {
        i = (i + 1) & (cap - 1);
    }
    table[i] = (uint32_t) idx + 1;
}

[[nodiscard]] enum geist_status vk_weight_index_add(struct geist_backend *be, size_t idx) {
    struct vk_state *st = be->state;
    if (idx >= UINT32_MAX) {
        return GEIST_E_OOM;
    }
    size_t need = 0;
    if (ckd_mul(&need, st->n_weights, (size_t) 2)) {
        return GEIST_E_OOM;
    }
    if (need > st->cap_weight_index) {
        size_t cap = st->cap_weight_index == 0 ? 128 : st->cap_weight_index;
        while (cap < need) {
            if (ckd_mul(&cap, cap, (size_t) 2)) {
                return GEIST_E_OOM;
            }
        }
        size_t bytes = 0;
        if (ckd_mul(&bytes, cap, sizeof(uint32_t))) {
            return GEIST_E_OOM;
        }
        uint32_t *table = geist_backend_alloc(be, bytes, alignof(uint32_t));
        if (table == nullptr) {
            return GEIST_E_OOM;
        }
        memset(table, 0, bytes);
        for (size_t i = 0; i < st->n_weights; ++i) {
            if (i != idx) {
                vk_weight_index_put(table, cap, st->weights[i].host, i);
            }
        }
        geist_backend_free(be, st->weight_index);
        st->weight_index     = table;
        st->cap_weight_index = cap;
    }
    vk_weight_index_put(st->weight_index, st->cap_weight_index, st->weights[idx].host, idx);
    return GEIST_OK;
}

/* Access-range helpers: byte spans inside the bound VkBuffer. */
struct vk_access vk_acc(uint64_t lo_bytes, uint64_t n_bytes, bool write) {
    return (struct vk_access) {.lo = lo_bytes, .hi = lo_bytes + n_bytes, .write = write};
}

struct vk_access vk_acc_all(bool write) {
    return (struct vk_access) {.lo = 0, .hi = UINT64_MAX, .write = write};
}

/* Byte span of an F32 DENSE tensor inside its VkBuffer (slab-stride aware). */
struct vk_access vk_acc_tensor(const struct geist_tensor *t, bool write) {
    const uint64_t lo = t->buffer->base_off + t->offset;
    uint64_t       span;
    size_t         rows, cols, stride;
    if (vk_t_geom(t, &rows, &cols, &stride)) {
        span = ((uint64_t) (rows - 1) * stride + cols) * 4u;
    } else {
        span = (uint64_t) vk_t_n(t) * 4u;
    }
    return vk_acc(lo, span, write);
}

/* GPU view of a tensor: VkBuffer + f32 element offset. False when the
 * tensor's buffer has no VkBuffer behind it (e.g. GGUF-mmap aliases), or
 * when the offset does not fit the shaders' uint32 element index. */
bool vk_tensor_gpu(const struct geist_tensor *t, VkDescriptorBufferInfo *out, uint32_t *elem_off) {
    if (t == nullptr || t->buffer == nullptr || t->buffer->buf == VK_NULL_HANDLE) {
        return false;
    }
    size_t byte_off;
    if (ckd_add(&byte_off, t->buffer->base_off, t->offset) || byte_off % 4 != 0 ||
        vk_ckd_u32(byte_off / 4, elem_off)) {
        return false;
    }
    *out = (VkDescriptorBufferInfo) {.buffer = t->buffer->buf, .range = VK_WHOLE_SIZE};
    return true;
}

/* Same, but offsets in f16 elements (for F16 KV-cache views). */
bool vk_tensor_gpu_f16(const struct geist_tensor *t,
                       VkDescriptorBufferInfo    *out,
                       uint32_t                  *elem_off) {
    if (t == nullptr || t->buffer == nullptr || t->buffer->buf == VK_NULL_HANDLE) {
        return false;
    }
    size_t byte_off;
    if (ckd_add(&byte_off, t->buffer->base_off, t->offset) || byte_off % 2 != 0 ||
        vk_ckd_u32(byte_off / 2, elem_off)) {
        return false;
    }
    *out = (VkDescriptorBufferInfo) {.buffer = t->buffer->buf, .range = VK_WHOLE_SIZE};
    return true;
}

/* Element count of a DENSE tensor of `dtype`, 0 on any mismatch (metadata
 * only; checked product, see geist_tensor_elems). */
static size_t vk_dense_n(const struct geist_tensor *t, enum geist_dtype dtype) {
    size_t n = 0;
    if (t == nullptr || t->dtype != dtype || t->layout != GEIST_LAYOUT_DENSE ||
        t->buffer == nullptr || geist_tensor_elems(t, &n)) {
        return 0;
    }
    return n;
}

/* Element count of an F16 DENSE tensor (metadata only). */
size_t vk_t_n16(const struct geist_tensor *t) {
    return vk_dense_n(t, GEIST_DTYPE_F16);
}

struct vk_access vk_acc_tensor16(const struct geist_tensor *t, bool write) {
    return vk_acc(t->buffer->base_off + t->offset, vk_t_n16(t) * 2u, write);
}

/* ====================================================================== */
/* Tensor accessors, host views and copies                                  */
/* ====================================================================== */

/* Element count of an F32 DENSE tensor, 0 on any mismatch. Metadata only. */
size_t vk_t_n(const struct geist_tensor *t) {
    return vk_dense_n(t, GEIST_DTYPE_F32);
}

void *vk_tensor_host(const struct geist_tensor *t, size_t *out_n) {
    const size_t n = vk_t_n(t);
    if (n == 0) {
        return nullptr;
    }
    uint8_t *base = t->buffer->host_alias != nullptr ? t->buffer->host_alias : t->buffer->mapped;
    if (base == nullptr) {
        /* Device-local (#488): a CPU fallback cannot touch it, and the op
         * fails rather than read the bytes over the bus. Counted so a test
         * can tell this refusal from any other bad input. */
        if (t->buffer->buf != VK_NULL_HANDLE) {
            t->buffer->owner->stat_host_denied++;
            if (getenv("GEIST_VK_VERBOSE") != nullptr) {
                fprintf(stderr,
                        "geist vulkan: a CPU fallback needs a device-local buffer (%zu KiB, "
                        "role %d); failing the op\n",
                        t->buffer->bytes >> 10,
                        (int) t->buffer->role);
            }
        }
        return nullptr;
    }
    if (vk_fallback(t->buffer->owner, VK_FB_HOST_VIEW) != GEIST_E_UNSUPPORTED) {
        return nullptr; /* GEIST_VK_STRICT: the error names the site */
    }
    vk_seq_flush(t->buffer->owner); /* host access — drain pending GPU work */
    if (vk_seq_take_failure(t->buffer->owner) != GEIST_OK) {
        return nullptr;
    }
    if (out_n != nullptr) {
        *out_n = n;
    }
    return base + t->offset;
}

/* Row geometry: rows × cols with row stride in elements. 2D views may
 * carry a slab stride (stride[0] > cols — the PLE per-layer-input slab);
 * other ranks are treated as one contiguous run. */
bool vk_t_geom(const struct geist_tensor *t, size_t *rows, size_t *cols, size_t *stride) {
    const size_t n = vk_t_n(t);
    if (n == 0) {
        return false;
    }
    if (t->ndim == 2 && t->stride[1] == 1 && t->stride[0] > t->shape[1]) {
        *rows   = (size_t) t->shape[0];
        *cols   = (size_t) t->shape[1];
        *stride = (size_t) t->stride[0];
        return true;
    }
    *rows   = 1;
    *cols   = n;
    *stride = n;
    return true;
}

[[nodiscard]] enum geist_status vk_buffer_copy(struct geist_buffer       *dst,
                                               size_t                     dst_offset,
                                               const struct geist_buffer *src,
                                               size_t                     src_offset,
                                               size_t                     n_bytes) {
    if (dst == nullptr || src == nullptr || n_bytes == 0) {
        return GEIST_E_INVALID_ARG;
    }
    /* A range copied onto itself is a no-op (the layer loop seeds
     * scratch_h_a from itself), and vkCmdCopyBuffer forbids overlapping
     * source and destination regions. */
    if (dst == src && dst_offset == src_offset) {
        return GEIST_OK;
    }
    struct vk_state *st = dst->owner;
    if (dst->buf != VK_NULL_HANDLE && src->buf != VK_NULL_HANDLE) {
        /* On-device copy appended to the sequence — keeps KV appends from
         * breaking the per-token batch (kv_store.c uses this path). */
        enum geist_status s = vk_seq_open_cmd(st);
        if (s != GEIST_OK) {
            return s;
        }
        const VkDescriptorBufferInfo cinf[2] = {{.buffer = src->buf}, {.buffer = dst->buf}};
        const struct vk_access       cacc[2] = {vk_acc(src->base_off + src_offset, n_bytes, false),
                                                vk_acc(dst->base_off + dst_offset, n_bytes, true)};
        vk_seq_hazard(st, cinf, cacc, 2);
        const VkBufferCopy region = {.srcOffset = src->base_off + src_offset,
                                     .dstOffset = dst->base_off + dst_offset,
                                     .size      = n_bytes};
        st->fn.CmdCopyBuffer(st->seq_cmd, src->buf, dst->buf, 1, &region);
        st->seq_dispatches++;
        st->seq_in_cmd++;
        vk_prof_stamp(st, VK_PIPE_COUNT);
        return GEIST_OK;
    }
    /* Host fallback. */
    uint8_t       *d  = dst->host_alias != nullptr ? dst->host_alias : dst->mapped;
    const uint8_t *sp = src->host_alias != nullptr ? src->host_alias : src->mapped;
    if (d == nullptr || sp == nullptr || dst_offset + n_bytes > dst->bytes ||
        src_offset + n_bytes > src->bytes) {
        return GEIST_E_UNSUPPORTED;
    }
    const enum geist_status fs = vk_fallback(st, VK_FB_HOST_COPY);
    if (fs != GEIST_E_UNSUPPORTED) {
        return fs;
    }
    vk_seq_flush(st);
    if (vk_seq_take_failure(st) != GEIST_OK) {
        return GEIST_E_BACKEND; /* the source holds no results of the dropped batch */
    }
    memcpy(d + dst_offset, sp + src_offset, n_bytes);
    return GEIST_OK;
}

/* Stage x into the device-local ring (one in-batch CmdCopyBuffer). Returns
 * the ring element offset; false = not stageable (caller falls back). */
[[nodiscard]] bool vk_xring_stage(struct geist_backend      *be,
                                  const struct geist_tensor *t_x,
                                  size_t                     m,
                                  size_t                     n_in,
                                  uint32_t                  *out_elem_off) {
    struct vk_state       *st = be->state;
    VkDescriptorBufferInfo src_bi;
    uint32_t               src_elem;
    if (!vk_tensor_gpu(t_x, &src_bi, &src_elem)) {
        return false;
    }
    size_t bytes;
    if (ckd_mul(&bytes, m, n_in) || ckd_mul(&bytes, bytes, sizeof(float))) {
        return false;
    }
    if (st->xring == nullptr &&
        vk_buffer_create(be, VK_XRING_CAP, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE, &st->xring) !=
                GEIST_OK) {
        return false;
    }
    if (bytes > st->xring->bytes) {
        return false; /* resolve_weight refuses an n_in whose max_m batch would not fit */
    }
    if (st->xring_used + bytes > st->xring->bytes) {
        vk_seq_flush(st); /* drains the batch and resets the ring */
    }
    uint32_t elem_off; /* < VK_XRING_CAP / 4; checked all the same */
    if (vk_ckd_u32(st->xring_used / sizeof(float), &elem_off) || vk_seq_open_cmd(st) != GEIST_OK) {
        return false;
    }
    {
        const VkDescriptorBufferInfo cinf[2] = {{.buffer = t_x->buffer->buf},
                                                {.buffer = st->xring->buf}};
        const struct vk_access       cacc[2] = {vk_acc_tensor(t_x, false),
                                                vk_acc(st->xring_used, bytes, true)};
        vk_seq_hazard(st, cinf, cacc, 2);
    }
    const size_t x_stride = t_x->ndim >= 2 ? (size_t) t_x->stride[t_x->ndim - 2] : n_in;
    const size_t src_byte = t_x->buffer->base_off + t_x->offset;
    if (m == 1 || x_stride == n_in) {
        const VkBufferCopy r = {.srcOffset = src_byte,
                                .dstOffset = st->xring_used,
                                .size      = m == 1 ? n_in * sizeof(float) : bytes};
        st->fn.CmdCopyBuffer(st->seq_cmd, t_x->buffer->buf, st->xring->buf, 1, &r);
    } else {
        VkBufferCopy regions[64];
        for (size_t r0 = 0; r0 < m; r0 += 64) {
            const uint32_t nr = (uint32_t) (m - r0 > 64 ? 64 : m - r0);
            for (uint32_t i = 0; i < nr; ++i) {
                regions[i] = (VkBufferCopy) {
                        .srcOffset = src_byte + (r0 + i) * x_stride * sizeof(float),
                        .dstOffset = st->xring_used + (r0 + i) * n_in * sizeof(float),
                        .size      = n_in * sizeof(float)};
            }
            st->fn.CmdCopyBuffer(st->seq_cmd, t_x->buffer->buf, st->xring->buf, nr, regions);
        }
    }
    st->seq_dispatches++;
    st->seq_in_cmd++;
    vk_prof_stamp(st, VK_PIPE_COUNT);
    *out_elem_off  = elem_off;
    st->xring_used = (st->xring_used + bytes + 63) & ~(size_t) 63;
    return true;
}
