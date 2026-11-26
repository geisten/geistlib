/*
 * src/backends/vulkan/decknet.c — the integer part of a decknet stream's
 * pass on the GPU (#533): <geist_decknet_gpu.h> for Vulkan.
 *
 * Layer: BACKEND (vulkan). Compiled only with DECKNET=1 (mk/common.mk), so a
 * default library has none of it; src/engine/decknet.c is its one caller.
 *
 * The two kernels of src/backends/metal/decknet.c as compute shaders
 * (shaders/decknet_sample.comp, shaders/decknet_sums.comp), in the same
 * unsigned 32-bit arithmetic that wraps, on the same tables: the results are
 * the bits of src/archs/decknet/ on the CPU.
 *
 * A frame is two command buffers, submitted one behind the other with a
 * fence each before anything is waited for: the first has what the CPU
 * needs first (stage 0), and the CPU computes on it while the second runs.
 *
 * Where the buffers live is the device's kind (enum decknet_gpu_memory). On
 * a GPU that shares the host's memory (the Raspberry Pi 5's V3D, an
 * integrated GPU, a software device) every buffer is host memory and stays
 * mapped: nothing is uploaded or read back, the caller writes the frame and
 * reads the results in place. A discrete GPU would read and write such
 * memory over the bus, access by access, so there the buffers the shaders
 * work on are device-local, and a frame's command buffers carry the copies:
 * the frame from a staging buffer in, and out again only what the CPU reads
 * (the plan's spans of the pixel memory after each stage, and the sums).
 * GEIST_VK_DECKNET_MEMORY=host|device says it instead of the device's kind.
 *
 * The buffers, the pipelines and the command buffers are this state's own
 * and no op of a model reads them, so a frame is no part of the backend's
 * command sequence. Like everything on this backend, a frame assumes that no
 * other thread uses the backend meanwhile.
 */
#include <geist_decknet_gpu.h>

#include "vk_internal.h"

#include "cleanup.h"

#include "shaders/decknet_sample_spv.h"
#include "shaders/decknet_sums_spv.h"

/* A frame's bytes are read out of words (decknet_sample.comp). */
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);

struct decknet_sample_params {
    uint32_t stride, format, height, first, n;
};
struct decknet_sum_params {
    uint32_t n_px, n_crops, crop_w, crop_h, n_parts, out_base;
};

constexpr size_t   DECKNET_GPU_KERNELS = 2;
constexpr size_t   DECKNET_GPU_STAGES  = 2;
constexpr uint32_t DECKNET_GPU_LOCAL   = 64;    /* local_size_x of both shaders */
constexpr uint32_t DECKNET_GPU_ROW     = 32768; /* groups in x; every device takes 65535 */
/* Storage buffers the shaders bind; their layouts are the backend's. */
constexpr uint32_t DECKNET_SAMPLE_BINDINGS = 4;
constexpr uint32_t DECKNET_SUMS_BINDINGS   = 5;

/* Where the buffers of a frame's shaders live. */
enum decknet_gpu_memory : uint32_t {
    DECKNET_GPU_HOST   = 0, /* host memory, mapped: the device works on it in place */
    DECKNET_GPU_DEVICE = 1, /* device-local, with staging buffers on the host's side */
};
static_assert(sizeof(enum decknet_gpu_memory) == sizeof(uint32_t));

struct geist_decknet_gpu {
    struct geist_backend     *be;
    VkPipeline                sample, sums;
    VkDescriptorPool          sets;
    VkDescriptorSet           set_sample, set_sums[DECKNET_GPU_KERNELS];
    VkCommandPool             pool;
    VkCommandBuffer           cmd[DECKNET_GPU_STAGES];
    VkFence                   fence[DECKNET_GPU_STAGES];
    size_t                    n_samples, n_first;
    struct geist_buffer      *b_samples, *b_dst, *b_frame, *b_pixels, *b_sums;
    struct geist_buffer      *b_px[DECKNET_GPU_KERNELS], *b_crop[DECKNET_GPU_KERNELS];
    struct geist_buffer      *b_taps[DECKNET_GPU_KERNELS];
    struct decknet_sum_params k[DECKNET_GPU_KERNELS];
    bool                      submitted;
    /* DECKNET_GPU_DEVICE: the host's side of the sample table, the frame,
     * the pixels and the sums (nullptr otherwise, and the b_ ones are
     * mapped), and what a stage copies back of the pixels. */
    enum decknet_gpu_memory memory;
    struct geist_buffer    *h_samples, *h_frame, *h_pixels, *h_sums;
    size_t                  frame_bytes;
    uint32_t                n_back[DECKNET_GPU_STAGES];
    VkBufferCopy           *back[DECKNET_GPU_STAGES];
};

static VkPipelineLayout layout_of(const struct vk_state *st, uint32_t n_bindings) {
    return st->seq_playouts[n_bindings - 2];
}

static void decknet_gpu_destroy(struct geist_decknet_gpu *gpu) {
    if (gpu == nullptr) {
        return;
    }
    struct geist_backend *be = gpu->be;
    struct vk_state      *st = be->state;
    if (gpu->submitted) {
        /* A submission nobody waited for: its buffers go below. */
        (void) st->fn.WaitForFences(
                st->device, (uint32_t) DECKNET_GPU_STAGES, gpu->fence, VK_TRUE, UINT64_MAX);
    }
    for (size_t i = 0; i < DECKNET_GPU_STAGES; i++) {
        if (gpu->fence[i] != VK_NULL_HANDLE) {
            st->fn.DestroyFence(st->device, gpu->fence[i], nullptr);
        }
    }
    if (gpu->pool != VK_NULL_HANDLE) {
        st->fn.DestroyCommandPool(st->device, gpu->pool, nullptr); /* and its command buffers */
    }
    if (gpu->sets != VK_NULL_HANDLE) {
        st->fn.DestroyDescriptorPool(st->device, gpu->sets, nullptr); /* and its sets */
    }
    if (gpu->sample != VK_NULL_HANDLE) {
        st->fn.DestroyPipeline(st->device, gpu->sample, nullptr);
    }
    if (gpu->sums != VK_NULL_HANDLE) {
        st->fn.DestroyPipeline(st->device, gpu->sums, nullptr);
    }
    struct geist_buffer *bufs[] = {gpu->b_samples,
                                   gpu->b_dst,
                                   gpu->b_frame,
                                   gpu->b_pixels,
                                   gpu->b_sums,
                                   gpu->b_px[0],
                                   gpu->b_px[1],
                                   gpu->b_crop[0],
                                   gpu->b_crop[1],
                                   gpu->b_taps[0],
                                   gpu->b_taps[1],
                                   gpu->h_samples,
                                   gpu->h_frame,
                                   gpu->h_pixels,
                                   gpu->h_sums};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) {
        if (bufs[i] != nullptr) {
            vk_buffer_destroy(be, bufs[i]);
        }
    }
    for (size_t i = 0; i < DECKNET_GPU_STAGES; i++) {
        heap_freep(&gpu->back[i]);
    }
    heap_freep(&gpu);
}

/* A mapped buffer of n words in host memory, zeroed. */
[[nodiscard]] static enum geist_status
words(struct geist_backend *be, size_t n, struct geist_buffer **out) {
    size_t bytes = 0;
    if (n == 0 || ckd_mul(&bytes, n, sizeof(uint32_t))) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "vulkan decknet: a table of %zu words", n);
        return GEIST_E_UNSUPPORTED;
    }
    const enum geist_status s =
            vk_buffer_create(be, bytes, GEIST_BUFFER_STAGING, GEIST_MEMORY_HOST_VISIBLE, out);
    if (s == GEIST_OK) {
        memset((*out)->mapped, 0, bytes);
    }
    return s;
}

/* `regions` of `src` to `dst`, waited for: the copies of create and of a new
 * sample table, on the first stage's command buffer and fence while no frame
 * is under way. */
[[nodiscard]] static enum geist_status copy_wait(struct geist_decknet_gpu *gpu,
                                                 uint32_t                  n,
                                                 const VkBufferCopy        regions[static n],
                                                 VkBuffer                  src,
                                                 VkBuffer                  dst) {
    struct geist_backend          *be    = gpu->be;
    struct vk_state               *st    = be->state;
    const VkCommandBufferBeginInfo binfo = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    const VkSubmitInfo             sinfo = {.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                            .commandBufferCount = 1,
                                            .pCommandBuffers    = &gpu->cmd[0]};
    bool ok = !gpu->submitted && st->fn.ResetFences(st->device, 1, &gpu->fence[0]) == VK_SUCCESS &&
              st->fn.BeginCommandBuffer(gpu->cmd[0], &binfo) == VK_SUCCESS;
    if (ok) {
        st->fn.CmdCopyBuffer(gpu->cmd[0], src, dst, n, regions);
        ok = st->fn.EndCommandBuffer(gpu->cmd[0]) == VK_SUCCESS &&
             st->fn.QueueSubmit(st->queue, 1, &sinfo, gpu->fence[0]) == VK_SUCCESS &&
             st->fn.WaitForFences(st->device, 1, &gpu->fence[0], VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    }
    if (!ok) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan decknet: a table was not copied");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

/* A buffer of n words for the shaders with the words at `from` in it, or
 * zeros (nullptr). In host memory it is mapped and written; a device-local
 * one is filled through the host's side of the sample table, a piece at a
 * time, which build() has made by then and clears afterwards. */
[[nodiscard]] static enum geist_status
table(struct geist_decknet_gpu *gpu, size_t n, const uint32_t *from, struct geist_buffer **out) {
    struct geist_backend *be = gpu->be;
    if (gpu->memory == DECKNET_GPU_HOST) {
        const enum geist_status s = words(be, n, out);
        if (s == GEIST_OK && from != nullptr) {
            memcpy((*out)->mapped, from, n * sizeof from[0]);
        }
        return s;
    }
    size_t bytes = 0;
    if (n == 0 || ckd_mul(&bytes, n, sizeof(uint32_t))) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "vulkan decknet: a table of %zu words", n);
        return GEIST_E_UNSUPPORTED;
    }
    enum geist_status s =
            vk_buffer_create(be, bytes, GEIST_BUFFER_STAGING, GEIST_MEMORY_DEVICE, out);
    const struct geist_buffer *stage = gpu->h_samples;
    for (size_t off = 0; s == GEIST_OK && off < bytes; off += stage->bytes) {
        const size_t k = bytes - off < stage->bytes ? bytes - off : stage->bytes;
        if (from != nullptr) {
            memcpy(stage->mapped, (const uint8_t *) from + off, k);
        } else {
            memset(stage->mapped, 0, k);
        }
        const VkBufferCopy region = {.dstOffset = off, .size = k};
        s                         = copy_wait(gpu, 1, &region, stage->buf, (*out)->buf);
    }
    return s;
}

/* The buffers `bufs` as the bindings 0 .. n - 1 of `set`. */
static void bind(const struct vk_state *st,
                 uint32_t               n,
                 struct geist_buffer   *bufs[static n],
                 VkDescriptorSet        set) {
    VkDescriptorBufferInfo info[DECKNET_SUMS_BINDINGS];
    VkWriteDescriptorSet   write[DECKNET_SUMS_BINDINGS];
    for (uint32_t i = 0; i < n; i++) {
        info[i]  = (VkDescriptorBufferInfo) {.buffer = bufs[i]->buf, .range = VK_WHOLE_SIZE};
        write[i] = (VkWriteDescriptorSet) {.sType      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                           .dstSet     = set,
                                           .dstBinding = i,
                                           .descriptorCount = 1,
                                           .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                           .pBufferInfo     = &info[i]};
    }
    st->fn.UpdateDescriptorSets(st->device, n, write, 0, nullptr);
}

/* The command buffers and the fences of a frame; the copies of build() use
 * the first of each. */
[[nodiscard]] static enum geist_status build_commands(struct geist_decknet_gpu *gpu) {
    struct geist_backend         *be    = gpu->be;
    struct vk_state              *st    = be->state;
    const VkCommandPoolCreateInfo pinfo = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                           .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                           .queueFamilyIndex = st->queue_family};
    bool ok = st->fn.CreateCommandPool(st->device, &pinfo, nullptr, &gpu->pool) == VK_SUCCESS;
    const VkCommandBufferAllocateInfo ainfo = {
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool        = gpu->pool,
            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = (uint32_t) DECKNET_GPU_STAGES};
    ok = ok && st->fn.AllocateCommandBuffers(st->device, &ainfo, gpu->cmd) == VK_SUCCESS;
    const VkFenceCreateInfo finfo = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (size_t i = 0; i < DECKNET_GPU_STAGES; i++) {
        ok = ok && st->fn.CreateFence(st->device, &finfo, nullptr, &gpu->fence[i]) == VK_SUCCESS;
    }
    if (!ok) {
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "vulkan decknet: no command buffers or fences");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

/* The descriptor sets: one for the samples, one per kernel. */
[[nodiscard]] static enum geist_status build_sets(struct geist_decknet_gpu *gpu) {
    struct geist_backend            *be    = gpu->be;
    struct vk_state                 *st    = be->state;
    const VkDescriptorPoolSize       size  = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              .descriptorCount =
                                                      DECKNET_SAMPLE_BINDINGS +
                                                      DECKNET_GPU_KERNELS * DECKNET_SUMS_BINDINGS};
    const VkDescriptorPoolCreateInfo dinfo = {.sType =
                                                      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                              .maxSets       = 1 + DECKNET_GPU_KERNELS,
                                              .poolSizeCount = 1,
                                              .pPoolSizes    = &size};
    const VkDescriptorSetLayout      layouts[1 + DECKNET_GPU_KERNELS] = {
            st->seq_dlayouts[DECKNET_SAMPLE_BINDINGS - 2],
            st->seq_dlayouts[DECKNET_SUMS_BINDINGS - 2],
            st->seq_dlayouts[DECKNET_SUMS_BINDINGS - 2]};
    VkDescriptorSet sets[1 + DECKNET_GPU_KERNELS];
    bool ok = st->fn.CreateDescriptorPool(st->device, &dinfo, nullptr, &gpu->sets) == VK_SUCCESS;
    const VkDescriptorSetAllocateInfo sinfo = {
            .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool     = gpu->sets,
            .descriptorSetCount = 1 + DECKNET_GPU_KERNELS,
            .pSetLayouts        = layouts};
    ok = ok && st->fn.AllocateDescriptorSets(st->device, &sinfo, sets) == VK_SUCCESS;
    if (!ok) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan decknet: no descriptor sets");
        return GEIST_E_BACKEND;
    }
    gpu->set_sample = sets[0];
    bind(st,
         DECKNET_SAMPLE_BINDINGS,
         (struct geist_buffer *[]) {gpu->b_frame, gpu->b_samples, gpu->b_dst, gpu->b_pixels},
         gpu->set_sample);
    for (size_t i = 0; i < DECKNET_GPU_KERNELS; i++) {
        gpu->set_sums[i] = sets[1 + i];
        bind(st,
             DECKNET_SUMS_BINDINGS,
             (struct geist_buffer *[]) {
                     gpu->b_pixels, gpu->b_px[i], gpu->b_crop[i], gpu->b_taps[i], gpu->b_sums},
             gpu->set_sums[i]);
    }
    return GEIST_OK;
}

/* What a stage copies back of the pixel memory: the plan's spans as bytes. */
[[nodiscard]] static enum geist_status back_regions(struct geist_decknet_gpu            *gpu,
                                                    const struct geist_decknet_gpu_plan *plan,
                                                    size_t                               stage) {
    struct geist_backend *be = gpu->be;
    const size_t          n  = plan->n_read[stage];
    if (n == 0) {
        return GEIST_OK;
    }
    if (n > UINT32_MAX) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "vulkan decknet: %zu spans to read", n);
        return GEIST_E_UNSUPPORTED;
    }
    gpu->back[stage] = heap_alloc_array_aligned(VkBufferCopy, n);
    if (gpu->back[stage] == nullptr) {
        return GEIST_E_OOM;
    }
    constexpr VkDeviceSize pixel = 4 * sizeof(uint32_t);
    for (size_t i = 0; i < n; i++) {
        const struct geist_decknet_gpu_span span = plan->read[stage][i];
        if (span.first > plan->n_pixels || span.n > plan->n_pixels - span.first) {
            geist_backend_set_error(
                    be, GEIST_E_UNSUPPORTED, "vulkan decknet: a span outside the pixels");
            return GEIST_E_UNSUPPORTED;
        }
        if (span.n != 0) { /* a copy of no bytes is not one */
            gpu->back[stage][gpu->n_back[stage]++] =
                    (VkBufferCopy) {.srcOffset = span.first * pixel,
                                    .dstOffset = span.first * pixel,
                                    .size      = span.n * pixel};
        }
    }
    return GEIST_OK;
}

/* Where this device's buffers go; GEIST_VK_DECKNET_MEMORY overrides. */
static enum decknet_gpu_memory memory_of(const struct vk_state *st) {
    const char *env = getenv("GEIST_VK_DECKNET_MEMORY");
    if (env != nullptr && strcmp(env, "host") == 0) {
        return DECKNET_GPU_HOST;
    }
    if (env != nullptr && strcmp(env, "device") == 0) {
        return DECKNET_GPU_DEVICE;
    }
    return st->unified_memory ? DECKNET_GPU_HOST : DECKNET_GPU_DEVICE;
}

[[nodiscard]] static enum geist_status build(struct geist_decknet_gpu            *gpu,
                                             const struct geist_decknet_gpu_plan *plan,
                                             struct geist_decknet_gpu_memory     *mem) {
    struct geist_backend *be = gpu->be;
    struct vk_state      *st = be->state;
    if (layout_of(st, DECKNET_SUMS_BINDINGS) == VK_NULL_HANDLE) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "vulkan decknet: %s binds fewer than %u storage buffers",
                                st->device_name,
                                DECKNET_SUMS_BINDINGS);
        return GEIST_E_UNSUPPORTED;
    }
    gpu->memory       = memory_of(st);
    const bool staged = gpu->memory == DECKNET_GPU_DEVICE;
    /* The counts the shaders index with, in 32 bits; the plan says they fit,
     * and nothing is built on a plan that lies. */
    size_t n_sums = 0, pixel_words = 0, sample_words = 0;
    bool   fits         = plan->n_samples != 0 && plan->n_first <= plan->n_samples &&
                          plan->n_samples <= UINT32_MAX && plan->n_pixels <= UINT32_MAX &&
                          plan->frame_bytes <= UINT32_MAX - 3 && plan->crop_w <= UINT32_MAX &&
                          plan->crop_h <= UINT32_MAX && !ckd_mul(&pixel_words, plan->n_pixels, 4) &&
                          !ckd_mul(&sample_words, plan->n_samples, 4);
    enum geist_status s = vk_make_pipeline(be,
                                           st,
                                           decknet_sample_spv,
                                           sizeof decknet_sample_spv,
                                           layout_of(st, DECKNET_SAMPLE_BINDINGS),
                                           false,
                                           &gpu->sample);
    if (s == GEIST_OK) {
        s = vk_make_pipeline(be,
                             st,
                             decknet_sums_spv,
                             sizeof decknet_sums_spv,
                             layout_of(st, DECKNET_SUMS_BINDINGS),
                             false,
                             &gpu->sums);
    }
    if (s == GEIST_OK) {
        s = build_commands(gpu);
    }
    if (s == GEIST_OK && fits && staged) {
        s = words(be, sample_words, &gpu->h_samples); /* and what table() fills through */
    }
    for (size_t i = 0; fits && i < DECKNET_GPU_KERNELS; i++) {
        const struct geist_decknet_gpu_kernel *k = &plan->kernel[i];
        size_t                                 n = 0, n_taps = 0;
        fits = !ckd_mul(&n, k->n_px, k->n_crops) && n != 0 && !ckd_mul(&n, n, k->n_parts) &&
               n != 0 && n <= UINT32_MAX - n_sums && k->n_px <= UINT32_MAX &&
               k->n_crops <= UINT32_MAX && !ckd_mul(&n_taps, k->n_taps, 2) &&
               !ckd_add(&n_taps, n_taps, k->n_parts) && n_taps <= UINT32_MAX;
        if (!fits) {
            break;
        }
        gpu->k[i] = (struct decknet_sum_params) {.n_px     = (uint32_t) k->n_px,
                                                 .n_crops  = (uint32_t) k->n_crops,
                                                 .crop_w   = (uint32_t) plan->crop_w,
                                                 .crop_h   = (uint32_t) plan->crop_h,
                                                 .n_parts  = (uint32_t) k->n_parts,
                                                 .out_base = (uint32_t) n_sums};
        n_sums += n;
        if (s == GEIST_OK) {
            s = table(gpu, k->n_px, k->px, &gpu->b_px[i]);
        }
        if (s == GEIST_OK) {
            s = table(gpu, k->n_crops, k->crop, &gpu->b_crop[i]);
        }
        if (s == GEIST_OK) {
            s = table(gpu, n_taps, (const uint32_t *) k->taps, &gpu->b_taps[i]);
        }
    }
    if (!fits || ckd_mul(&n_sums, n_sums, 4)) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "vulkan decknet: the plan does not fit 32 bits");
        return GEIST_E_UNSUPPORTED;
    }
    const size_t frame_words = (plan->frame_bytes + 3) / 4;
    if (s == GEIST_OK) {
        s = table(gpu, sample_words, nullptr, &gpu->b_samples);
    }
    if (s == GEIST_OK) {
        s = table(gpu, plan->n_samples, plan->dst, &gpu->b_dst);
    }
    if (s == GEIST_OK) {
        s = table(gpu, frame_words, nullptr, &gpu->b_frame);
    }
    if (s == GEIST_OK) {
        s = table(gpu, pixel_words, nullptr, &gpu->b_pixels);
    }
    if (s == GEIST_OK) {
        s = table(gpu, n_sums, nullptr, &gpu->b_sums);
    }
    if (s == GEIST_OK && staged) {
        memset(gpu->h_samples->mapped, 0, gpu->h_samples->bytes); /* table() went through it */
        s = words(be, frame_words, &gpu->h_frame);
    }
    if (s == GEIST_OK && staged) {
        s = words(be, pixel_words, &gpu->h_pixels);
    }
    if (s == GEIST_OK && staged) {
        s = words(be, n_sums, &gpu->h_sums);
    }
    for (size_t stage = 0; s == GEIST_OK && staged && stage < DECKNET_GPU_STAGES; stage++) {
        s = back_regions(gpu, plan, stage);
    }
    if (s == GEIST_OK) {
        s = build_sets(gpu);
    }
    if (s != GEIST_OK) {
        return s;
    }
    gpu->n_samples   = plan->n_samples;
    gpu->n_first     = plan->n_first;
    gpu->frame_bytes = frame_words * sizeof(uint32_t);
    *mem             = (struct geist_decknet_gpu_memory) {
            .samples = (staged ? gpu->h_samples : gpu->b_samples)->mapped,
            .frame   = (staged ? gpu->h_frame : gpu->b_frame)->mapped,
            .pixels  = (staged ? gpu->h_pixels : gpu->b_pixels)->mapped,
            .sums    = (staged ? gpu->h_sums : gpu->b_sums)->mapped};
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status decknet_gpu_create(struct geist_backend                *be,
                                                          const struct geist_decknet_gpu_plan *plan,
                                                          struct geist_decknet_gpu           **out,
                                                          struct geist_decknet_gpu_memory *mem) {
    *out = nullptr;
    *mem = (struct geist_decknet_gpu_memory) {0};
    if (be->state == nullptr) {
        return GEIST_E_INVALID_STATE;
    }
    struct geist_decknet_gpu *gpu = heap_calloc_array_aligned(struct geist_decknet_gpu, 1);
    if (gpu == nullptr) {
        return GEIST_E_OOM;
    }
    gpu->be                   = be;
    const enum geist_status s = build(gpu, plan, mem);
    if (s != GEIST_OK) {
        decknet_gpu_destroy(gpu);
        *mem = (struct geist_decknet_gpu_memory) {0};
        return s;
    }
    *out = gpu;
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status decknet_gpu_samples_written(struct geist_decknet_gpu *gpu) {
    if (gpu->h_samples == nullptr) {
        return GEIST_OK; /* mapped memory: the device reads what the host wrote */
    }
    const VkBufferCopy all = {.size = gpu->h_samples->bytes};
    return copy_wait(gpu, 1, &all, gpu->h_samples->buf, gpu->b_samples->buf);
}

/* A barrier between what was written at `from` and what reads it at `to`. */
static void barrier(const struct vk_state *st,
                    VkCommandBuffer        cmd,
                    VkPipelineStageFlags   from,
                    VkAccessFlags          written,
                    VkPipelineStageFlags   to,
                    VkAccessFlags          read) {
    const VkMemoryBarrier b = {.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                               .srcAccessMask = written,
                               .dstAccessMask = read};
    st->fn.CmdPipelineBarrier(cmd, from, to, 0, 1, &b, 0, nullptr, 0, nullptr);
}

/* Device-local buffers: behind a stage's shaders, what the CPU reads of the
 * pixels they wrote goes to the host's side. */
static void encode_back(const struct geist_decknet_gpu *gpu, size_t stage) {
    const struct vk_state *st = gpu->be->state;
    if (gpu->h_pixels == nullptr) {
        return;
    }
    barrier(st,
            gpu->cmd[stage],
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_READ_BIT);
    if (gpu->n_back[stage] != 0) {
        st->fn.CmdCopyBuffer(gpu->cmd[stage],
                             gpu->b_pixels->buf,
                             gpu->h_pixels->buf,
                             gpu->n_back[stage],
                             gpu->back[stage]);
    }
}

/* n invocations of the bound pipeline, in rows of at most DECKNET_GPU_ROW
 * groups; the shaders return at once in those behind the last one. */
static void dispatch(const struct vk_state *st, VkCommandBuffer cmd, size_t n) {
    const uint32_t groups = (uint32_t) ((n + DECKNET_GPU_LOCAL - 1) / DECKNET_GPU_LOCAL);
    const uint32_t rows   = (groups + DECKNET_GPU_ROW - 1) / DECKNET_GPU_ROW;
    st->fn.CmdDispatch(cmd, (groups + rows - 1) / rows, rows, 1);
}

/* The samples [first, first + n) into `cmd`. */
static void encode_samples(const struct geist_decknet_gpu     *gpu,
                           VkCommandBuffer                     cmd,
                           const struct decknet_sample_params *frame,
                           size_t                              first,
                           size_t                              n) {
    const struct vk_state *st = gpu->be->state;
    if (n == 0) {
        return;
    }
    struct decknet_sample_params p      = *frame;
    const VkPipelineLayout       layout = layout_of(st, DECKNET_SAMPLE_BINDINGS);
    p.first                             = (uint32_t) first;
    p.n                                 = (uint32_t) n;
    st->fn.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gpu->sample);
    st->fn.CmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &gpu->set_sample, 0, nullptr);
    st->fn.CmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof p, &p);
    dispatch(st, cmd, n);
}

[[nodiscard]] static enum geist_status decknet_gpu_submit(struct geist_decknet_gpu *gpu,
                                                          enum geist_decknet_format format,
                                                          size_t                    stride,
                                                          size_t                    height) {
    struct geist_backend *be = gpu->be;
    struct vk_state      *st = be->state;
    if (gpu->submitted || stride > UINT32_MAX || height > UINT32_MAX) {
        return GEIST_E_INVALID_STATE;
    }
    const struct decknet_sample_params frame = {
            .stride = (uint32_t) stride, .format = (uint32_t) format, .height = (uint32_t) height};
    const VkCommandBufferBeginInfo binfo = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    bool                           ok =
            st->fn.ResetFences(st->device, (uint32_t) DECKNET_GPU_STAGES, gpu->fence) == VK_SUCCESS;
    for (size_t stage = 0; stage < DECKNET_GPU_STAGES; stage++) {
        /* Begin resets a command buffer of this pool. */
        ok = ok && st->fn.BeginCommandBuffer(gpu->cmd[stage], &binfo) == VK_SUCCESS;
    }
    if (ok) {
        if (gpu->h_frame != nullptr) {
            /* The frame's bytes from the staging buffer to the device. */
            const size_t rows =
                    format == GEIST_DECKNET_FORMAT_NV12 ? height + (height + 1) / 2 : height;
            const size_t       n   = (stride * rows + 3) / 4 * 4;
            const VkBufferCopy rgn = {.size = n < gpu->frame_bytes ? n : gpu->frame_bytes};
            st->fn.CmdCopyBuffer(gpu->cmd[0], gpu->h_frame->buf, gpu->b_frame->buf, 1, &rgn);
            barrier(st,
                    gpu->cmd[0],
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_READ_BIT);
        }
        encode_samples(gpu, gpu->cmd[0], &frame, 0, gpu->n_first);
        encode_back(gpu, 0);
        encode_samples(gpu, gpu->cmd[1], &frame, gpu->n_first, gpu->n_samples - gpu->n_first);
        /* The sums read what the samples of both stages wrote. */
        const VkPipelineLayout layout = layout_of(st, DECKNET_SUMS_BINDINGS);
        barrier(st,
                gpu->cmd[1],
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT);
        st->fn.CmdBindPipeline(gpu->cmd[1], VK_PIPELINE_BIND_POINT_COMPUTE, gpu->sums);
        for (size_t i = 0; i < DECKNET_GPU_KERNELS; i++) {
            st->fn.CmdBindDescriptorSets(gpu->cmd[1],
                                         VK_PIPELINE_BIND_POINT_COMPUTE,
                                         layout,
                                         0,
                                         1,
                                         &gpu->set_sums[i],
                                         0,
                                         nullptr);
            st->fn.CmdPushConstants(gpu->cmd[1],
                                    layout,
                                    VK_SHADER_STAGE_COMPUTE_BIT,
                                    0,
                                    sizeof gpu->k[i],
                                    &gpu->k[i]);
            dispatch(st, gpu->cmd[1], (size_t) gpu->k[i].n_px * gpu->k[i].n_crops);
        }
        encode_back(gpu, 1);
        if (gpu->h_sums != nullptr) {
            const VkBufferCopy all = {.size = gpu->h_sums->bytes};
            st->fn.CmdCopyBuffer(gpu->cmd[1], gpu->b_sums->buf, gpu->h_sums->buf, 1, &all);
        }
    }
    for (size_t stage = 0; stage < DECKNET_GPU_STAGES; stage++) {
        ok = ok && st->fn.EndCommandBuffer(gpu->cmd[stage]) == VK_SUCCESS;
    }
    /* One submission per stage: a fence is a submission's. */
    size_t n_submitted = 0;
    for (; ok && n_submitted < DECKNET_GPU_STAGES; n_submitted++) {
        const VkSubmitInfo sinfo = {.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                    .commandBufferCount = 1,
                                    .pCommandBuffers    = &gpu->cmd[n_submitted]};
        if (st->fn.QueueSubmit(st->queue, 1, &sinfo, gpu->fence[n_submitted]) != VK_SUCCESS) {
            break;
        }
    }
    if (n_submitted < DECKNET_GPU_STAGES) {
        /* What did go to the queue ends before the caller sees the error. */
        if (n_submitted != 0) {
            (void) st->fn.WaitForFences(
                    st->device, (uint32_t) n_submitted, gpu->fence, VK_TRUE, UINT64_MAX);
        }
        for (size_t stage = n_submitted; stage < DECKNET_GPU_STAGES; stage++) {
            (void) st->fn.ResetCommandBuffer(gpu->cmd[stage], 0); /* out of whatever state */
        }
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan decknet: a frame was not submitted");
        return GEIST_E_BACKEND;
    }
    gpu->submitted = true;
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status decknet_gpu_wait(struct geist_decknet_gpu *gpu,
                                                        size_t                    stage) {
    struct geist_backend *be = gpu->be;
    struct vk_state      *st = be->state;
    if (!gpu->submitted || stage >= DECKNET_GPU_STAGES) {
        return GEIST_E_INVALID_STATE;
    }
    const VkResult r = st->fn.WaitForFences(st->device, 1, &gpu->fence[stage], VK_TRUE, UINT64_MAX);
    if (stage == DECKNET_GPU_STAGES - 1) {
        gpu->submitted = false;
    }
    if (r != VK_SUCCESS) {
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "vulkan decknet: stage %zu failed (%d)", stage, (int) r);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

const struct geist_decknet_gpu_ops geist_decknet_gpu_vulkan = {
        .create          = decknet_gpu_create,
        .destroy         = decknet_gpu_destroy,
        .samples_written = decknet_gpu_samples_written,
        .submit          = decknet_gpu_submit,
        .wait            = decknet_gpu_wait,
};
