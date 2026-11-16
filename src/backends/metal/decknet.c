/*
 * src/backends/metal/decknet.c — the integer part of a decknet stream's pass
 * on the GPU (#533): <geist_decknet_gpu.h> for Metal.
 *
 * Layer: BACKEND (metal). Compiled only with DECKNET=1 (mk/common.mk), so a
 * default library has none of it; src/engine/decknet.c is its one caller.
 *
 * Two kernels, both in unsigned 32-bit arithmetic that wraps, which is what
 * makes their results the bits of src/archs/decknet/ on the CPU:
 *
 *   decknet_sample   one thread per sample: the 2 x 2 camera pixels (of an
 *                    NV12 or YUYV frame converted here, per pixel), blended
 *                    with the table's weights, into the pixel memory;
 *   decknet_sums     one thread per pixel a kernel is evaluated at: per part
 *                    of the kernel the sum of the crop pixels of its taps.
 *
 * A frame is two command buffers on the backend's queue, committed one
 * behind the other before anything is waited for: the first has what the
 * CPU needs first (stage 0), and the CPU computes on it while the second
 * runs. Every buffer is shared memory (Apple GPUs have no other), so
 * nothing is uploaded or read back: the caller writes the frame and reads
 * the results in place.
 *
 * The buffers are this state's own and no op of a model reads them, so a
 * frame is no part of the backend's command sequence. They are bound the
 * backend's way (metal_msg_send_set_buffer), which also puts them into its
 * residency set. Like everything on this backend, a frame assumes that no
 * other thread uses the backend meanwhile.
 */
#include <geist_decknet_gpu.h>

#include "metal_internal.h"

#include "checked.h"

static const char metal_decknet_source[] =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct Sample { uint w0, w1, w2, xy; };\n"
        "struct SampleParams { uint stride, format, height, first, n; };\n"
        /* BT.601 limited range, the formula of docs/DECKNET.md. */
        "static inline uint3 yuv(int y, int u, int v) {\n"
        "    int c = 298 * (y - 16) + 128;\n"
        "    int3 rgb = int3(c + 409 * (v - 128),\n"
        "                    c - 100 * (u - 128) - 208 * (v - 128),\n"
        "                    c + 516 * (u - 128)) >> 8;\n"
        "    return uint3(clamp(rgb, 0, 255));\n"
        "}\n"
        /* Formats: 0 RGB8, 1 NV12, 2 YUYV (enum geist_decknet_format). */
        "static inline uint3 pixel(device const uchar *f, SampleParams p, uint x, uint y) {\n"
        "    if (p.format == 0u) {\n"
        "        uint o = y * p.stride + x * 3u;\n"
        "        return uint3(f[o], f[o + 1u], f[o + 2u]);\n"
        "    }\n"
        "    if (p.format == 1u) {\n"
        "        uint o = (p.height + y / 2u) * p.stride + (x & ~1u);\n"
        "        return yuv(f[y * p.stride + x], f[o], f[o + 1u]);\n"
        "    }\n"
        "    uint o = y * p.stride + (x & ~1u) * 2u;\n"
        "    return yuv(f[y * p.stride + x * 2u], f[o + 1u], f[o + 3u]);\n"
        "}\n"
        "kernel void decknet_sample(device const uchar *frame [[buffer(0)]],\n"
        "                           device const Sample *tab [[buffer(1)]],\n"
        "                           device const uint *dst [[buffer(2)]],\n"
        "                           device uint4 *pixels [[buffer(3)]],\n"
        "                           constant SampleParams &p [[buffer(4)]],\n"
        "                           uint gid [[thread_position_in_grid]]) {\n"
        "    if (gid >= p.n) { return; }\n"
        "    uint i = p.first + gid;\n"
        "    Sample t = tab[i];\n"
        "    uint x = t.xy & 0xFFFFu, y = t.xy >> 16u;\n"
        "    uint3 b00 = pixel(frame, p, x, y);\n"
        "    uint3 sum = (b00 << 24u) + 128u;\n"
        "    sum += (pixel(frame, p, x + 1u, y) - b00) * t.w0;\n"
        "    sum += (pixel(frame, p, x, y + 1u) - b00) * t.w1;\n"
        "    sum += (pixel(frame, p, x + 1u, y + 1u) - b00) * t.w2;\n"
        "    pixels[dst[i]] = uint4(sum >> 8u, 0u);\n"
        "}\n"
        "struct SumParams { uint n_px, n_crops, crop_w, crop_h, n_parts, out_base; };\n"
        "kernel void decknet_sums(device const uint4 *pixels [[buffer(0)]],\n"
        "                         device const uint *px [[buffer(1)]],\n"
        "                         device const uint *crop [[buffer(2)]],\n"
        "                         device const int *taps [[buffer(3)]],\n"
        "                         device uint4 *sums [[buffer(4)]],\n"
        "                         constant SumParams &k [[buffer(5)]],\n"
        "                         uint gid [[thread_position_in_grid]]) {\n"
        "    if (gid >= k.n_px * k.n_crops) { return; }\n"
        "    uint at = px[gid % k.n_px];\n"
        "    uint base = crop[gid / k.n_px] * k.crop_w * k.crop_h;\n"
        "    int x = int(at % k.crop_w), y = int(at / k.crop_w);\n"
        "    uint t = k.n_parts;\n"
        "    for (uint part = 0u; part < k.n_parts; part++) {\n"
        "        uint4 acc = uint4(0u);\n"
        "        uint n = uint(taps[part]);\n"
        "        for (uint j = 0u; j < n; j++, t += 2u) {\n"
        "            uint yy = uint(clamp(y + taps[t], 0, int(k.crop_h) - 1));\n"
        "            uint xx = uint(clamp(x + taps[t + 1u], 0, int(k.crop_w) - 1));\n"
        "            acc += pixels[base + yy * k.crop_w + xx];\n"
        "        }\n"
        "        sums[k.out_base + gid * k.n_parts + part] = acc;\n"
        "    }\n"
        "}\n";

struct decknet_sample_params {
    uint32_t stride, format, height, first, n;
};
struct decknet_sum_params {
    uint32_t n_px, n_crops, crop_w, crop_h, n_parts, out_base;
};

constexpr size_t DECKNET_GPU_KERNELS = 2;
constexpr size_t DECKNET_GPU_STAGES  = 2;

struct geist_decknet_gpu {
    struct geist_backend     *be;
    struct metal_library_h   *library;
    struct metal_function_h  *sample_fn, *sums_fn;
    struct metal_pipeline_h  *sample, *sums;
    size_t                    sample_threads, sums_threads; /* per threadgroup */
    size_t                    n_samples, n_first;
    struct geist_buffer      *b_samples, *b_dst, *b_frame, *b_pixels, *b_sums;
    struct geist_buffer      *b_px[DECKNET_GPU_KERNELS], *b_crop[DECKNET_GPU_KERNELS];
    struct geist_buffer      *b_taps[DECKNET_GPU_KERNELS];
    struct decknet_sum_params k[DECKNET_GPU_KERNELS];
    /* A submission: its two command buffers, autoreleased into `pool`. */
    struct metal_cmdbuf_h *cmd[DECKNET_GPU_STAGES];
    struct metal_pool_h   *pool;
    bool                   submitted;
};

static void
bind(struct metal_state *st, struct metal_encoder_h *enc, struct geist_buffer *buf, size_t index) {
    metal_msg_send_set_buffer(st, enc, buf->buffer, 0, index);
}

/* n threads of the bound pipeline in groups of `per_group`, which is the
 * pipeline's maxTotalThreadsPerThreadgroup: Metal drops a dispatch with
 * more threads in a group than that, without an error. The kernels return
 * at once in the threads behind the last one. */
static void
dispatch(struct metal_state *st, struct metal_encoder_h *enc, size_t n, size_t per_group) {
    metal_msg_send_dispatch(st,
                            enc,
                            (struct metal_size) {(n + per_group - 1) / per_group, 1, 1},
                            (struct metal_size) {per_group, 1, 1});
}

static void decknet_gpu_destroy(struct geist_decknet_gpu *gpu) {
    if (gpu == nullptr) {
        return;
    }
    struct geist_backend *be = gpu->be;
    struct metal_state   *st = be->state;
    if (gpu->submitted) {
        /* A submission nobody waited for: its buffers go below. */
        metal_msg_send_void0(st, gpu->cmd[DECKNET_GPU_STAGES - 1], "waitUntilCompleted");
        metal_pool_pop(st, gpu->pool);
    }
    struct geist_buffer **bufs[] = {&gpu->b_samples,
                                    &gpu->b_dst,
                                    &gpu->b_frame,
                                    &gpu->b_pixels,
                                    &gpu->b_sums,
                                    &gpu->b_px[0],
                                    &gpu->b_px[1],
                                    &gpu->b_crop[0],
                                    &gpu->b_crop[1],
                                    &gpu->b_taps[0],
                                    &gpu->b_taps[1]};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) {
        if (*bufs[i] != nullptr) {
            metal_buffer_destroy_internal(be, *bufs[i]);
        }
    }
    metal_msg_send_void0(st, gpu->sample, "release");
    metal_msg_send_void0(st, gpu->sums, "release");
    metal_msg_send_void0(st, gpu->sample_fn, "release");
    metal_msg_send_void0(st, gpu->sums_fn, "release");
    metal_msg_send_void0(st, gpu->library, "release");
    heap_freep(&gpu);
}

/* A shared buffer of n words, zeroed. */
[[nodiscard]] static enum geist_status
words(struct geist_backend *be, size_t n, struct geist_buffer **out) {
    size_t bytes = 0;
    if (n == 0 || ckd_mul(&bytes, n, sizeof(uint32_t))) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "metal decknet: a table of %zu words", n);
        return GEIST_E_UNSUPPORTED;
    }
    return metal_new_buffer(be, bytes, GEIST_BUFFER_STAGING, GEIST_MEMORY_HOST_VISIBLE, true, out);
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

/* One pipeline of the library, and how many threads a group of it may have. */
[[nodiscard]] static enum geist_status pipeline(struct geist_backend     *be,
                                                struct metal_library_h   *library,
                                                const char               *name,
                                                struct metal_function_h **fn,
                                                struct metal_pipeline_h **pipe,
                                                size_t                   *threads) {
    struct metal_state     *st = be->state;
    const enum geist_status s  = metal_create_named_pipeline(
            be, library, metal_objc_get_class(st, "NSString"), name, fn, pipe);
    if (s != GEIST_OK) {
        return s;
    }
    *threads = metal_msg_send_ulong0(st, *pipe, "maxTotalThreadsPerThreadgroup");
    if (*threads == 0) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal decknet: %s has no threads", name);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status build(struct geist_decknet_gpu            *gpu,
                                             const struct geist_decknet_gpu_plan *plan,
                                             struct geist_decknet_gpu_memory     *mem) {
    struct geist_backend *be        = gpu->be;
    struct metal_state   *st        = be->state;
    void                 *ns_string = metal_objc_get_class(st, "NSString");
    void                 *source    = ns_string != nullptr
                                              ? metal_msg_send_id_cstr(
                                                        st, ns_string, "stringWithUTF8String:", metal_decknet_source)
                                              : nullptr;
    void                 *err       = nullptr;
    gpu->library = source != nullptr
                           ? metal_msg_send_id_id_id_err(st,
                                                         st->device,
                                                         "newLibraryWithSource:options:error:",
                                                         source,
                                                         nullptr,
                                                         &err)
                           : nullptr;
    if (gpu->library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal decknet: shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    enum geist_status s = pipeline(be,
                                   gpu->library,
                                   "decknet_sample",
                                   &gpu->sample_fn,
                                   &gpu->sample,
                                   &gpu->sample_threads);
    if (s == GEIST_OK) {
        s = pipeline(
                be, gpu->library, "decknet_sums", &gpu->sums_fn, &gpu->sums, &gpu->sums_threads);
    }
    /* The counts the kernels index with, in 32 bits; the plan says they fit,
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
                be, GEIST_E_UNSUPPORTED, "metal decknet: the plan does not fit 32 bits");
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
    struct metal_state       *st  = be->state;
    struct geist_decknet_gpu *gpu = heap_calloc_array_aligned(struct geist_decknet_gpu, 1);
    if (gpu == nullptr) {
        return GEIST_E_OOM;
    }
    gpu->be = be;
    /* The source string, the function names and whatever the compiler
     * leaves behind are autoreleased. */
    struct metal_pool_h    *pool = metal_pool_push(st);
    const enum geist_status s    = build(gpu, plan, mem);
    metal_pool_pop(st, pool);
    if (s != GEIST_OK) {
        decknet_gpu_destroy(gpu);
        *mem = (struct geist_decknet_gpu_memory) {0};
        return s;
    }
    *out = gpu;
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status decknet_gpu_samples_written(struct geist_decknet_gpu *gpu) {
    (void) gpu; /* shared memory: the device reads what the host wrote */
    return GEIST_OK;
}

/* The samples [first, first + n) into `enc`. */
static void encode_samples(struct geist_decknet_gpu           *gpu,
                           struct metal_encoder_h             *enc,
                           const struct decknet_sample_params *frame,
                           size_t                              first,
                           size_t                              n) {
    struct metal_state *st = gpu->be->state;
    if (n == 0) {
        return;
    }
    struct decknet_sample_params p = *frame;
    p.first                        = (uint32_t) first;
    p.n                            = (uint32_t) n;
    (void) metal_msg_send_id_id(st, enc, "setComputePipelineState:", gpu->sample);
    bind(st, enc, gpu->b_frame, 0);
    bind(st, enc, gpu->b_samples, 1);
    bind(st, enc, gpu->b_dst, 2);
    bind(st, enc, gpu->b_pixels, 3);
    metal_msg_send_set_bytes(st, enc, &p, sizeof p, 4);
    dispatch(st, enc, n, gpu->sample_threads);
}

[[nodiscard]] static enum geist_status decknet_gpu_submit(struct geist_decknet_gpu *gpu,
                                                          enum geist_decknet_format format,
                                                          size_t                    stride,
                                                          size_t                    height) {
    struct geist_backend *be = gpu->be;
    struct metal_state   *st = be->state;
    if (gpu->submitted || stride > UINT32_MAX || height > UINT32_MAX) {
        return GEIST_E_INVALID_STATE;
    }
    const struct decknet_sample_params frame = {
            .stride = (uint32_t) stride, .format = (uint32_t) format, .height = (uint32_t) height};
    gpu->pool                                       = metal_pool_push(st);
    struct metal_encoder_h *enc[DECKNET_GPU_STAGES] = {nullptr, nullptr};
    for (size_t stage = 0; stage < DECKNET_GPU_STAGES; stage++) {
        /* Unretained: the buffers are this state's, and outlive the wait. */
        gpu->cmd[stage] =
                metal_msg_send_id0(st, st->command_queue, "commandBufferWithUnretainedReferences");
        enc[stage] = gpu->cmd[stage] != nullptr
                             ? metal_msg_send_id0(st, gpu->cmd[stage], "computeCommandEncoder")
                             : nullptr;
        if (enc[stage] == nullptr) {
            /* Nothing was committed: an encoder that is open ends, and the
             * command buffers go with the pool. */
            metal_msg_send_void0(st, enc[0], "endEncoding");
            metal_pool_pop(st, gpu->pool);
            geist_backend_set_error(be, GEIST_E_BACKEND, "metal decknet: no command buffer");
            return GEIST_E_BACKEND;
        }
    }
    encode_samples(gpu, enc[0], &frame, 0, gpu->n_first);
    encode_samples(gpu, enc[1], &frame, gpu->n_first, gpu->n_samples - gpu->n_first);
    (void) metal_msg_send_id_id(st, enc[1], "setComputePipelineState:", gpu->sums);
    bind(st, enc[1], gpu->b_pixels, 0);
    bind(st, enc[1], gpu->b_sums, 4);
    for (size_t i = 0; i < DECKNET_GPU_KERNELS; i++) {
        bind(st, enc[1], gpu->b_px[i], 1);
        bind(st, enc[1], gpu->b_crop[i], 2);
        bind(st, enc[1], gpu->b_taps[i], 3);
        metal_msg_send_set_bytes(st, enc[1], &gpu->k[i], sizeof gpu->k[i], 5);
        dispatch(st, enc[1], (size_t) gpu->k[i].n_px * gpu->k[i].n_crops, gpu->sums_threads);
    }
    for (size_t stage = 0; stage < DECKNET_GPU_STAGES; stage++) {
        metal_msg_send_void0(st, enc[stage], "endEncoding");
        metal_msg_send_void0(st, gpu->cmd[stage], "commit");
    }
    gpu->submitted = true;
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status decknet_gpu_wait(struct geist_decknet_gpu *gpu,
                                                        size_t                    stage) {
    struct geist_backend *be = gpu->be;
    struct metal_state   *st = be->state;
    if (!gpu->submitted || stage >= DECKNET_GPU_STAGES) {
        return GEIST_E_INVALID_STATE;
    }
    metal_msg_send_void0(st, gpu->cmd[stage], "waitUntilCompleted");
    enum geist_status s   = GEIST_OK;
    void             *err = metal_msg_send_id0(st, gpu->cmd[stage], "error");
    if (err != nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal decknet: stage %zu failed%s%s",
                                stage,
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        s = GEIST_E_BACKEND;
    }
    if (stage == DECKNET_GPU_STAGES - 1) {
        metal_pool_pop(st, gpu->pool);
        gpu->submitted = false;
    }
    return s;
}

const struct geist_decknet_gpu_ops geist_decknet_gpu_metal = {
        .create          = decknet_gpu_create,
        .destroy         = decknet_gpu_destroy,
        .samples_written = decknet_gpu_samples_written,
        .submit          = decknet_gpu_submit,
        .wait            = decknet_gpu_wait,
};
