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
 * Every buffer is host-visible and stays mapped, so nothing is uploaded or
 * read back: the caller writes the frame and reads the results in place. On
 * a GPU that shares the host's memory (the Raspberry Pi 5's V3D) that is the
 * memory the shaders work on; a discrete GPU reads and writes it over the
 * bus, which is correct and not what one would measure there.
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
                                   gpu->b_taps[1]};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) {
        if (bufs[i] != nullptr) {
            vk_buffer_destroy(be, bufs[i]);
        }
    }
    heap_freep(&gpu);
}

/* A mapped buffer of n words, zeroed. */
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

/* The same with the n words at `from` in it. */
[[nodiscard]] static enum geist_status
table(struct geist_backend *be, size_t n, const uint32_t *from, struct geist_buffer **out) {
    const enum geist_status s = words(be, n, out);
    if (s == GEIST_OK) {
        memcpy((*out)->mapped, from, n * sizeof from[0]);
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

/* The descriptor sets (one for the samples, one per kernel), the command
 * buffers and the fences of a frame. */
[[nodiscard]] static enum geist_status build_frame(struct geist_decknet_gpu *gpu) {
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
    const VkCommandPoolCreateInfo pinfo = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                           .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                           .queueFamilyIndex = st->queue_family};
    ok = ok && st->fn.CreateCommandPool(st->device, &pinfo, nullptr, &gpu->pool) == VK_SUCCESS;
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
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "vulkan decknet: no descriptor sets, command buffers or fences");
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
    /* The counts the shaders index with, in 32 bits; the plan says they fit,
     * and nothing is built on a plan that lies. */
    size_t n_sums = 0, pixel_words = 0, sample_words = 0;
    bool   fits = plan->n_samples != 0 && plan->n_first <= plan->n_samples &&
                  plan->n_samples <= UINT32_MAX && plan->n_pixels <= UINT32_MAX &&
                  plan->frame_bytes <= UINT32_MAX - 3 && plan->crop_w <= UINT32_MAX &&
                  plan->crop_h <= UINT32_MAX && !ckd_mul(&pixel_words, plan->n_pixels, 4) &&
                  !ckd_mul(&sample_words, plan->n_samples, 4);
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
            s = table(be, k->n_px, k->px, &gpu->b_px[i]);
        }
        if (s == GEIST_OK) {
            s = table(be, k->n_crops, k->crop, &gpu->b_crop[i]);
        }
        if (s == GEIST_OK) {
            s = table(be, n_taps, (const uint32_t *) k->taps, &gpu->b_taps[i]);
        }
    }
    if (!fits || ckd_mul(&n_sums, n_sums, 4)) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "vulkan decknet: the plan does not fit 32 bits");
        return GEIST_E_UNSUPPORTED;
    }
    if (s == GEIST_OK) {
        s = words(be, sample_words, &gpu->b_samples);
    }
    if (s == GEIST_OK) {
        s = table(be, plan->n_samples, plan->dst, &gpu->b_dst);
    }
    if (s == GEIST_OK) {
        s = words(be, (plan->frame_bytes + 3) / 4, &gpu->b_frame);
    }
    if (s == GEIST_OK) {
        s = words(be, pixel_words, &gpu->b_pixels);
    }
    if (s == GEIST_OK) {
        s = words(be, n_sums, &gpu->b_sums);
    }
    if (s == GEIST_OK) {
        s = build_frame(gpu);
    }
    if (s != GEIST_OK) {
        return s;
    }
    gpu->n_samples = plan->n_samples;
    gpu->n_first   = plan->n_first;
    *mem           = (struct geist_decknet_gpu_memory) {.samples = gpu->b_samples->mapped,
                                                        .frame   = gpu->b_frame->mapped,
                                                        .pixels  = gpu->b_pixels->mapped,
                                                        .sums    = gpu->b_sums->mapped};
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
    (void) gpu; /* mapped memory: the device reads what the host wrote */
    return GEIST_OK;
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
        encode_samples(gpu, gpu->cmd[0], &frame, 0, gpu->n_first);
        encode_samples(gpu, gpu->cmd[1], &frame, gpu->n_first, gpu->n_samples - gpu->n_first);
        /* The sums read what the samples of both stages wrote. */
        const VkMemoryBarrier  written = {.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
        const VkPipelineLayout layout  = layout_of(st, DECKNET_SUMS_BINDINGS);
        st->fn.CmdPipelineBarrier(gpu->cmd[1],
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0,
                                  1,
                                  &written,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
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
