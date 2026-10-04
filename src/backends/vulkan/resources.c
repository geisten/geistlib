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
        role != GEIST_BUFFER_IO) {
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
    VkResult             r     = st->fn.AllocateMemory(st->device, &minfo, nullptr, &buf->mem);
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
        geist_backend_set_error(
                be, GEIST_E_OOM, "vulkan: allocating %zu bytes failed (%d)", bytes, (int) r);
        return GEIST_E_OOM;
    }
    buf->device_mem = (st->mem_props.memoryTypes[mem_type].propertyFlags &
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    if (buf->host_visible && buf->device_mem) {
        buf->bar_bytes = (size_t) minfo.allocationSize;
        st->bar_used += buf->bar_bytes;
    }
    if (getenv("GEIST_VK_VERBOSE") != nullptr && buf->host_visible) {
        fprintf(stderr,
                "  buffer %zu KiB role=%d bar=%d\n",
                bytes >> 10,
                (int) role,
                buf->device_mem);
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
                memcpy(nb, st->hostbufs, st->n_hostbufs * sizeof(*nb));
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

[[nodiscard]] enum geist_status vk_buffer_create_aliased(struct geist_backend  *be,
                                                         void                  *host_ptr,
                                                         size_t                 n_bytes,
                                                         enum geist_buffer_role role,
                                                         struct geist_buffer  **out) {
    /* Bookkeeping handle only: records the host region (scratch-pool slice,
     * weight arena, GGUF mmap). No Vulkan resources — device copies of
     * aliased weights are made at resolve_weight time (Phase 2), keyed by
     * this pointer. buffer_map returns the pointer unchanged, so the arch
     * layer's CPU-side paths keep working. */
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
     * pool / weight arena since P3), borrow its VkBuffer so GPU ops can
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

/* One blocking staging round-trip. Direction: upload (src != nullptr) or
 * download (dst != nullptr). Each call allocates a fresh staging buffer, so
 * this is suitable for load-time transfers but not a transfer hot path. */
[[nodiscard]] static enum geist_status
vk_staged_copy(struct geist_buffer *buf, size_t n_bytes, const uint8_t *src, uint8_t *dst) {
    struct vk_state      *st = buf->owner;
    struct geist_backend *be = st->backend;

    VkBufferCreateInfo binfo       = {.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                      .size        = n_bytes,
                                      .usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer           staging     = VK_NULL_HANDLE;
    VkDeviceMemory     staging_mem = VK_NULL_HANDLE;
    void              *mapped      = nullptr;
    enum geist_status  status      = GEIST_E_BACKEND;

    if (st->fn.CreateBuffer(st->device, &binfo, nullptr, &staging) != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: staging buffer failed");
        return GEIST_E_BACKEND;
    }
    VkMemoryRequirements req;
    st->fn.GetBufferMemoryRequirements(st->device, staging, &req);
    uint32_t             mem_type = vk_find_mem_type(st,
                                                     req.memoryTypeBits,
                                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo minfo    = {.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                     .allocationSize  = req.size,
                                     .memoryTypeIndex = mem_type};
    if (mem_type == UINT32_MAX ||
        st->fn.AllocateMemory(st->device, &minfo, nullptr, &staging_mem) != VK_SUCCESS ||
        st->fn.BindBufferMemory(st->device, staging, staging_mem, 0) != VK_SUCCESS ||
        st->fn.MapMemory(st->device, staging_mem, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
        geist_backend_set_error(
                be, GEIST_E_OOM, "vulkan: staging alloc/map failed (%zu B)", n_bytes);
        status = GEIST_E_OOM;
        goto out;
    }
    if (src != nullptr) {
        memcpy(mapped, src, n_bytes);
    }

    VkCommandBufferBeginInfo begin  = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                       .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VkBufferCopy             region = {.size = n_bytes};
    if (st->fn.BeginCommandBuffer(st->xfer_cmd, &begin) != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: begin transfer cmd failed");
        goto out;
    }
    if (src != nullptr) {
        st->fn.CmdCopyBuffer(st->xfer_cmd, staging, buf->buf, 1, &region);
    } else {
        st->fn.CmdCopyBuffer(st->xfer_cmd, buf->buf, staging, 1, &region);
    }
    if (st->fn.EndCommandBuffer(st->xfer_cmd) != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: end transfer cmd failed");
        goto out;
    }
    VkSubmitInfo submit = {.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                           .commandBufferCount = 1,
                           .pCommandBuffers    = &st->xfer_cmd};
    if (st->fn.QueueSubmit(st->queue, 1, &submit, st->xfer_fence) != VK_SUCCESS ||
        st->fn.WaitForFences(st->device, 1, &st->xfer_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: transfer submit/wait failed");
        goto out;
    }
    (void) st->fn.ResetFences(st->device, 1, &st->xfer_fence);
    if (dst != nullptr) {
        memcpy(dst, mapped, n_bytes);
    }
    status = GEIST_OK;

out:
    if (mapped != nullptr) {
        st->fn.UnmapMemory(st->device, staging_mem);
    }
    if (staging_mem != VK_NULL_HANDLE) {
        st->fn.FreeMemory(st->device, staging_mem, nullptr);
    }
    st->fn.DestroyBuffer(st->device, staging, nullptr);
    return status;
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
    return vk_staged_copy(buf, n_bytes, src, nullptr);
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
    return vk_staged_copy((struct geist_buffer *) buf, n_bytes, nullptr, dst);
}

void *vk_buffer_map(struct geist_buffer *buf) {
    if (buf == nullptr) {
        return nullptr;
    }
    vk_seq_flush(buf->owner); /* host is about to read/write — drain the batch */
    if (buf->host_alias != nullptr) {
        return buf->host_alias;
    }
    return buf->mapped; /* nullptr for device-local — caller must fall back */
}

void vk_buffer_unmap(struct geist_buffer *buf) {
    (void) buf; /* persistent coherent mappings — nothing to flush */
}

/* ====================================================================== */
/* Linear dispatch (Phase 2: synchronous per-call round-trip)              */
/*                                                                         */
/* The main contract hands the resolved kernels host pointers (x, y) and   */
/* a host w->raw. Weights were copied to VRAM at resolve time; x/y round-  */
/* trip through persistent host-visible staging. One submit + fence wait   */
/* per linear — correct first. Phase 3 moves the hot loop onto linear_t    */
/* with device-resident activations and batched submits.                   */
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
 * tensor's buffer has no VkBuffer behind it (e.g. GGUF-mmap aliases). */
bool vk_tensor_gpu(const struct geist_tensor *t, VkDescriptorBufferInfo *out, uint32_t *elem_off) {
    if (t == nullptr || t->buffer == nullptr || t->buffer->buf == VK_NULL_HANDLE) {
        return false;
    }
    const size_t byte_off = t->buffer->base_off + t->offset;
    if (byte_off % 4 != 0) {
        return false;
    }
    *out      = (VkDescriptorBufferInfo) {.buffer = t->buffer->buf, .range = VK_WHOLE_SIZE};
    *elem_off = (uint32_t) (byte_off / 4);
    return true;
}

/* Same, but offsets in f16 elements (for F16 KV-cache views). */
bool vk_tensor_gpu_f16(const struct geist_tensor *t,
                       VkDescriptorBufferInfo    *out,
                       uint32_t                  *elem_off) {
    if (t == nullptr || t->buffer == nullptr || t->buffer->buf == VK_NULL_HANDLE) {
        return false;
    }
    const size_t byte_off = t->buffer->base_off + t->offset;
    if (byte_off % 2 != 0) {
        return false;
    }
    *out      = (VkDescriptorBufferInfo) {.buffer = t->buffer->buf, .range = VK_WHOLE_SIZE};
    *elem_off = (uint32_t) (byte_off / 2);
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
        return nullptr; /* device-local — CPU ops can't touch it */
    }
    t->buffer->owner->stat_cpu_falls++;
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
    vk_seq_flush(st);
    uint8_t       *d  = dst->host_alias != nullptr ? dst->host_alias : dst->mapped;
    const uint8_t *sp = src->host_alias != nullptr ? src->host_alias : src->mapped;
    if (d == nullptr || sp == nullptr || dst_offset + n_bytes > dst->bytes ||
        src_offset + n_bytes > src->bytes) {
        return GEIST_E_UNSUPPORTED;
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
    const size_t bytes = m * n_in * sizeof(float);
    if (st->xring == nullptr &&
        vk_buffer_create(be, VK_XRING_CAP, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE, &st->xring) !=
                GEIST_OK) {
        return false;
    }
    if (bytes > st->xring->bytes) {
        return false;
    }
    if (st->xring_used + bytes > st->xring->bytes) {
        vk_seq_flush(st); /* drains the batch and resets the ring */
    }
    if (vk_seq_open_cmd(st) != GEIST_OK) {
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
    *out_elem_off  = (uint32_t) (st->xring_used / sizeof(float));
    st->xring_used = (st->xring_used + bytes + 63) & ~(size_t) 63;
    return true;
}
