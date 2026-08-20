/*
 * src/backends/vulkan/ops.c — op implementations, resolver, probe, and the descriptor.
 *
 * Layer: BACKEND (vulkan). Split from the former monolithic backend.c;
 * pure moves, no behavior change.
 */
#include "vk_internal.h"

[[nodiscard]] static enum geist_status vk_dispatch_linear(struct geist_backend *be,
                                                          enum vk_pipe          pipe,
                                                          struct geist_buffer  *wbuf,
                                                          const float          *x,
                                                          float                *y,
                                                          size_t                m,
                                                          size_t                n_in,
                                                          size_t                n_out) {
    struct vk_state *st = be->state;
    vk_seq_flush(st); /* host x/y round-trip — must not interleave with a batch */
    /* x_stage: the GPU reads it hot (GEMM B tiles), the host only writes —
     * SCRATCH role makes it BAR-eligible. y_stage stays in system RAM
     * (the host reads results back; CPU reads from BAR are uncached). */
    enum geist_status s =
            vk_stage_reserve_role(be, &st->x_stage, m * n_in * sizeof(float), GEIST_BUFFER_SCRATCH);
    if (s == GEIST_OK) {
        s = vk_stage_reserve(be, &st->y_stage, m * n_out * sizeof(float));
    }
    if (s != GEIST_OK) {
        return s;
    }
    memcpy(st->x_stage->mapped, x, m * n_in * sizeof(float));

    const VkDescriptorBufferInfo binfo[3] = {
            {.buffer = st->x_stage->buf, .range = VK_WHOLE_SIZE},
            {.buffer = wbuf->buf, .range = VK_WHOLE_SIZE},
            {.buffer = st->y_stage->buf, .range = VK_WHOLE_SIZE},
    };
    const struct vk_push push = {.n_in           = (uint32_t) n_in,
                                 .n_out          = (uint32_t) n_out,
                                 .blocks_per_row = (uint32_t) (n_in / 256),
                                 .rows           = (uint32_t) m,
                                 .x_stride       = (uint32_t) n_in,
                                 .y_stride       = (uint32_t) n_out};
    enum vk_pipe         eff  = pipe;
    uint32_t             gx   = vk_linear_gx(pipe, (uint32_t) n_out);
    uint32_t             gy   = vk_linear_gy(pipe, (uint32_t) m);
    vk_linear_cm_route(st, &eff, (uint32_t) m, (uint32_t) n_out, &gx, &gy);
    s = vk_seq_dispatch(be, eff, binfo, &push, sizeof(push), gx, gy, 1);
    if (s != GEIST_OK) {
        return s;
    }
    vk_seq_flush(st);
    memcpy(y, st->y_stage->mapped, m * n_out * sizeof(float));
    return GEIST_OK;
}

/* Resolver-installed kernels. The signature has no error path — failures
 * report to stderr and zero y so a defect is loud in the parity gate
 * rather than silent garbage (same policy as the Metal backend). */
static void vk_linear_run(const float               *x,
                          const struct geist_weight *w,
                          size_t                     m,
                          struct geist_backend      *be,
                          float                     *y,
                          enum vk_pipe               matvec_pipe,
                          enum vk_pipe               matmul_pipe) {
    struct vk_state     *st   = be->state;
    struct geist_buffer *wbuf = vk_weight_lookup(st, w->raw);
    const size_t         n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    if (m > 1 && st->subgroup_size != 32u) {
        /* The register-tiled GEMM shaders hard-assume 32-lane subgroups;
         * on e.g. lavapipe (8 lanes) they compute garbage — the lavapipe
         * CI leg caught exactly that on its first run. The matvec kernels
         * are subgroup-size-agnostic, so loop them: correct everywhere,
         * and the software tier is a correctness gate, not a benchmark. */
        for (size_t r = 0; r < m; r++) {
            vk_linear_run(x + r * n_in, w, 1, be, y + r * n_out, matvec_pipe, matmul_pipe);
        }
        return;
    }
    if (wbuf == nullptr ||
        vk_dispatch_linear(be, m == 1 ? matvec_pipe : matmul_pipe, wbuf, x, y, m, n_in, n_out) !=
                GEIST_OK) {
        fprintf(stderr,
                "geist vulkan: linear dispatch failed (%s) — zeroing output\n",
                geist_backend_errmsg(be));
        memset(y, 0, m * n_out * sizeof(float));
    }
}

static void
vk_w_q4k_m1(const float *x, const struct geist_weight *w, struct geist_backend *be, float *y) {
    vk_linear_run(x, w, 1, be, y, VK_PIPE_MATVEC_Q4K, VK_PIPE_MATMUL_Q4K);
}

static void vk_w_q4k_mN(const float               *x,
                        const struct geist_weight *w,
                        size_t                     m,
                        struct geist_backend      *be,
                        float                     *y) {
    vk_linear_run(x, w, m, be, y, VK_PIPE_MATVEC_Q4K, VK_PIPE_MATMUL_Q4K);
}

static void
vk_w_q6k_m1(const float *x, const struct geist_weight *w, struct geist_backend *be, float *y) {
    vk_linear_run(x, w, 1, be, y, VK_PIPE_MATVEC_Q6K, VK_PIPE_MATMUL_Q6K);
}

static void vk_w_q6k_mN(const float               *x,
                        const struct geist_weight *w,
                        size_t                     m,
                        struct geist_backend      *be,
                        float                     *y) {
    vk_linear_run(x, w, m, be, y, VK_PIPE_MATVEC_Q6K, VK_PIPE_MATMUL_Q6K);
}

static void
vk_w_f32_m1(const float *x, const struct geist_weight *w, struct geist_backend *be, float *y) {
    vk_linear_run(x, w, 1, be, y, VK_PIPE_MATVEC_F32, VK_PIPE_MATMUL_F32);
}

static void vk_w_f32_mN(const float               *x,
                        const struct geist_weight *w,
                        size_t                     m,
                        struct geist_backend      *be,
                        float                     *y) {
    vk_linear_run(x, w, m, be, y, VK_PIPE_MATVEC_F32, VK_PIPE_MATMUL_F32);
}

/* ---- CPU fallback for dtypes without a GPU kernel yet (F16/BF16/...) ----
 * Row-dequant + naive dot, following cpu_scalar_w_quant_*; keeps model
 * loading alive for mixed-dtype GGUFs until those dtypes get shaders. */

static bool vk_dequant_row(const struct geist_weight *w, size_t j, float *row) {
    const uint8_t *base = (const uint8_t *) w->raw;
    const size_t   n_in = (size_t) w->n_in;
    switch ((enum geist_dtype) w->dtype) {
    case GEIST_DTYPE_F16: {
        const uint8_t *r = base + j * n_in * 2;
        for (size_t i = 0; i < n_in; i++) {
            const uint16_t h = (uint16_t) r[2 * i] | ((uint16_t) r[2 * i + 1] << 8);
            row[i]           = fp16_to_fp32(h);
        }
        return true;
    }
    case GEIST_DTYPE_BF16: {
        const uint8_t *r = base + j * n_in * 2;
        for (size_t i = 0; i < n_in; i++) {
            const uint32_t b = (uint32_t) ((uint16_t) r[2 * i] | ((uint16_t) r[2 * i + 1] << 8))
                               << 16;
            memcpy(&row[i], &b, sizeof b);
        }
        return true;
    }
    case GEIST_DTYPE_Q3_K:
        dequant_q3_K_row(base + j * n_in / Q3_K_BLOCK_ELEMS * Q3_K_BLOCK_BYTES, row, n_in);
        return true;
    case GEIST_DTYPE_Q5_K:
        dequant_q5_K_row(base + j * n_in / Q5_K_BLOCK_ELEMS * Q5_K_BLOCK_BYTES, row, n_in);
        return true;
    case GEIST_DTYPE_Q8_0:
        dequant_q8_0_row(base + j * n_in / Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES, row, n_in);
        return true;
    default:
        return false;
    }
}

static void vk_w_cpu_mN(const float               *x,
                        const struct geist_weight *w,
                        size_t                     m,
                        struct geist_backend      *be,
                        float                     *y) {
    (void) be;
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    float       *row   = heap_alloc_aligned(n_in * sizeof(float), OPTIMAL_ALIGNMENT);
    if (row == nullptr) {
        return;
    }
    for (size_t j = 0; j < n_out; j++) {
        if (!vk_dequant_row(w, j, row)) {
            for (size_t i = 0; i < m; i++) {
                y[i * n_out + j] = 0;
            }
            continue;
        }
        for (size_t i = 0; i < m; i++) {
            double acc = 0.0;
            for (size_t k = 0; k < n_in; k++) {
                acc += (double) x[i * n_in + k] * (double) row[k];
            }
            y[i * n_out + j] = (float) acc;
        }
    }
    safe_free((void **) &row);
}

static void
vk_w_cpu_m1(const float *x, const struct geist_weight *w, struct geist_backend *be, float *y) {
    vk_w_cpu_mN(x, w, 1, be, y);
}

/* ---- resolve_weight: upload GPU-supported dtypes to VRAM, register,     */
/*      install kernels; CPU fallback for the rest.                        */

/* GPU copies of Q6_K are repacked to 216-byte blocks (210 + 6 pad) so
 * every field sits 4-byte aligned and the kernels use word loads. */
enum { VK_Q6K_GPU_BLOCK = 216 };

[[nodiscard]] static size_t vk_weight_bytes(const struct geist_weight *w) {
    const size_t n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    switch ((enum geist_dtype) w->dtype) {
    case GEIST_DTYPE_Q4_K:
        return n_out * (n_in / Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES;
    case GEIST_DTYPE_Q6_K:
        return n_out * (n_in / Q6_K_BLOCK_ELEMS) * VK_Q6K_GPU_BLOCK;
    case GEIST_DTYPE_F32:
        return n_out * n_in * sizeof(float);
    default:
        return 0;
    }
}

[[nodiscard]] static enum geist_status vk_resolve_weight(struct geist_backend *be,
                                                         struct geist_weight  *w) {
    struct vk_state *st = be->state;
    if (w == nullptr || w->raw == nullptr || w->n_in <= 0 || w->n_out <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    geist_kernel_linear_m1_fn m1;
    geist_kernel_linear_mN_fn mN;
    switch ((enum geist_dtype) w->dtype) {
    case GEIST_DTYPE_Q4_K:
        if (w->n_in % 256 != 0) {
            return GEIST_E_UNSUPPORTED;
        }
        m1 = vk_w_q4k_m1;
        mN = vk_w_q4k_mN;
        break;
    case GEIST_DTYPE_Q6_K:
        if (w->n_in % 256 != 0) {
            return GEIST_E_UNSUPPORTED;
        }
        m1 = vk_w_q6k_m1;
        mN = vk_w_q6k_mN;
        break;
    case GEIST_DTYPE_F32:
        m1 = vk_w_f32_m1;
        mN = vk_w_f32_mN;
        break;
    case GEIST_DTYPE_F16:
    case GEIST_DTYPE_BF16:
    case GEIST_DTYPE_Q3_K:
    case GEIST_DTYPE_Q5_K:
    case GEIST_DTYPE_Q8_0:
        w->linear_m1 = vk_w_cpu_m1;
        w->linear_mN = vk_w_cpu_mN;
        return GEIST_OK;
    default:
        return GEIST_E_UNSUPPORTED;
    }

    /* Upload to VRAM and register. An existing entry for the same host
     * pointer is REPLACED, not reused: the same address can carry new bytes
     * after a model reload (or a freed+remalloc'd test blob) — the latest
     * resolve is authoritative. Tied weights resolving twice re-upload the
     * same bytes once more at load time; harmless. */
    struct vk_weight_entry *slot = nullptr;
    for (size_t i = 0; i < st->n_weights; ++i) {
        if (st->weights[i].host == w->raw) {
            slot = &st->weights[i];
            break;
        }
    }
    if (slot == nullptr) {
        if (st->n_weights == st->cap_weights) {
            const size_t            cap = st->cap_weights == 0 ? 64 : st->cap_weights * 2;
            struct vk_weight_entry *nw =
                    geist_backend_alloc(be, cap * sizeof(*nw), alignof(struct vk_weight_entry));
            if (nw == nullptr) {
                return GEIST_E_OOM;
            }
            memcpy(nw, st->weights, st->n_weights * sizeof(*nw));
            geist_backend_free(be, st->weights);
            st->weights     = nw;
            st->cap_weights = cap;
        }
        slot  = &st->weights[st->n_weights++];
        *slot = (struct vk_weight_entry) {0};
    }
    const size_t         bytes = vk_weight_bytes(w);
    struct geist_buffer *gpu   = nullptr;
    enum geist_status    s =
            vk_buffer_create(be, bytes, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE, &gpu);
    if (s == GEIST_OK && w->dtype == GEIST_DTYPE_Q6_K) {
        const size_t n_blocks = bytes / VK_Q6K_GPU_BLOCK;
        uint8_t     *packed   = heap_alloc_aligned(bytes, 64);
        if (packed == nullptr) {
            s = GEIST_E_OOM;
        } else {
            const uint8_t *srcb = (const uint8_t *) w->raw;
            for (size_t i = 0; i < n_blocks; ++i) {
                memcpy(packed + i * VK_Q6K_GPU_BLOCK,
                       srcb + i * Q6_K_BLOCK_BYTES,
                       Q6_K_BLOCK_BYTES);
                memset(packed + i * VK_Q6K_GPU_BLOCK + Q6_K_BLOCK_BYTES,
                       0,
                       VK_Q6K_GPU_BLOCK - Q6_K_BLOCK_BYTES);
            }
            s = vk_buffer_upload(gpu, bytes, packed);
            safe_free((void **) &packed);
        }
    } else if (s == GEIST_OK) {
        s = vk_buffer_upload(gpu, bytes, (const uint8_t *) w->raw);
    }
    if (s != GEIST_OK) {
        if (gpu != nullptr) {
            vk_buffer_destroy(be, gpu);
        }
        if (slot->gpu == nullptr) {
            st->n_weights--; /* fresh slot never got a buffer — roll back */
        }
        return s;
    }
    if (slot->gpu != nullptr) {
        vk_buffer_destroy(be, slot->gpu);
    }
    *slot        = (struct vk_weight_entry) {.host = w->raw, .gpu = gpu};
    w->linear_m1 = m1;
    w->linear_mN = mN;
    return GEIST_OK;
}

static uint32_t vk_groups(size_t n) {
    return (uint32_t) ((n + 255) / 256);
}

/* Dispatch geometry of the linear pipes. matvec q4k/q6k: 8 rows per
 * workgroup. matmul_q4k: 4 output rows x 16 batch rows per workgroup. */
uint32_t vk_linear_gx(enum vk_pipe pipe, uint32_t n_out) {
    if (pipe == VK_PIPE_MATVEC_Q4K) {
        return (n_out + 7u) / 8u; /* 2 warps x 4 rows per workgroup */
    }
    if (pipe == VK_PIPE_MATVEC_Q6K) {
        return (n_out + 7u) / 8u;
    }
    if (pipe == VK_PIPE_MATMUL_Q4K) {
        return (n_out + 7u) / 8u;
    }
    if (pipe == VK_PIPE_MATMUL_Q6K || pipe == VK_PIPE_MATMUL_F32) {
        return (n_out + 3u) / 4u;
    }
    return n_out;
}

uint32_t vk_linear_gy(enum vk_pipe pipe, uint32_t m) {
    if (pipe == VK_PIPE_MATMUL_Q4K) {
        return (m + 31u) / 32u;
    }
    return (pipe == VK_PIPE_MATMUL_Q6K || pipe == VK_PIPE_MATMUL_F32) ? (m + 15u) / 16u : m;
}

/* Reroute conforming quant GEMMs onto the tensor-core pipelines. */
void vk_linear_cm_route(struct vk_state *st,
                        enum vk_pipe    *pipe,
                        uint32_t         m,
                        uint32_t         n_out,
                        uint32_t        *gx,
                        uint32_t        *gy) {
    enum vk_pipe cm;
    if (*pipe == VK_PIPE_MATMUL_Q4K) {
        cm = VK_PIPE_MM_Q4K_CM;
    } else if (*pipe == VK_PIPE_MATMUL_Q6K) {
        cm = VK_PIPE_MM_Q6K_CM;
    } else {
        return;
    }
    if ((m & 15u) != 0 || n_out % 64u != 0 || st->pipes[cm] == VK_NULL_HANDLE) {
        return;
    }
    /* small n_out starves the SMs on the 64-row tile — use the 32x32 one
     * (workgroup count is the wall clock at ~1 workgroup/SM) */
    if (cm == VK_PIPE_MM_Q4K_CM && n_out < 4096u &&
        st->pipes[VK_PIPE_MM_Q4K_CM32] != VK_NULL_HANDLE) {
        *pipe = VK_PIPE_MM_Q4K_CM32;
        *gx   = n_out / 32u;
        *gy   = (m + 31u) / 32u;
        return;
    }
    *pipe = cm;
    *gx   = n_out / 64u;
    *gy   = (m + 63u) / 64u;
}

/* GPU-first attempt for the 3-buffer elementwise family (add/mul/gelu_mul):
 * push {n, a_off, b_off, y_off, cols, a_stride, b_stride, y_stride};
 * cols == 0 → all-contiguous fast path in the shader. */
static bool vk_try_ew3(struct geist_backend      *be,
                       enum vk_pipe               pipe,
                       const struct geist_tensor *a,
                       const struct geist_tensor *b,
                       const struct geist_tensor *y) {
    const size_t n = vk_t_n(a);
    if (!VK_OPS(be, 2u) || n == 0 || n != vk_t_n(b) || n != vk_t_n(y)) {
        return false;
    }
    size_t ra, ca, sa, rb, cb, sb, ry, cy, sy;
    if (!vk_t_geom(a, &ra, &ca, &sa) || !vk_t_geom(b, &rb, &cb, &sb) ||
        !vk_t_geom(y, &ry, &cy, &sy)) {
        return false;
    }
    uint32_t cols = 0;
    if (sa != ca || sb != cb || sy != cy) {
        /* mixed contiguous/strided operands: unify on the strided cols */
        cols = (uint32_t) (sa != ca ? ca : (sb != cb ? cb : cy));
        if ((sa != ca && ca != cols) || (sb != cb && cb != cols) || (sy != cy && cy != cols) ||
            n % cols != 0) {
            return false;
        }
        if (sa == ca) {
            sa = cols;
        }
        if (sb == cb) {
            sb = cols;
        }
        if (sy == cy) {
            sy = cols;
        }
    }
    VkDescriptorBufferInfo bi[3];
    uint32_t               off[3];
    if (!vk_tensor_gpu(a, &bi[0], &off[0]) || !vk_tensor_gpu(b, &bi[1], &off[1]) ||
        !vk_tensor_gpu(y, &bi[2], &off[2])) {
        return false;
    }
    const uint32_t         push[8] = {(uint32_t) n,
                                      off[0],
                                      off[1],
                                      off[2],
                                      cols,
                                      (uint32_t) sa,
                                      (uint32_t) sb,
                                      (uint32_t) sy};
    const struct vk_access acc[3]  = {
            vk_acc_tensor(a, false), vk_acc_tensor(b, false), vk_acc_tensor(y, true)};
    return vk_seq_dispatch_acc(be, pipe, bi, acc, push, sizeof(push), vk_groups(n), 1, 1) ==
           GEIST_OK;
}

/* Strided-aware CPU fallback core for the elementwise trio. op: 0=add,
 * 1=mul, 2=gelu_tanh_mul. */
[[nodiscard]] static enum geist_status vk_ew3_cpu(struct geist_backend      *be,
                                                  int                        op,
                                                  const struct geist_tensor *a,
                                                  const struct geist_tensor *b,
                                                  struct geist_tensor       *y,
                                                  const char                *name) {
    size_t       na = 0, nb = 0, ny = 0;
    const float *ap = vk_tensor_host(a, &na);
    const float *bp = vk_tensor_host(b, &nb);
    float       *yp = vk_tensor_host(y, &ny);
    if (ap == nullptr || bp == nullptr || yp == nullptr || na != nb || na != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan %s: bad inputs", name);
        return GEIST_E_INVALID_ARG;
    }
    size_t ra = 1, ca = na, sa = na, rb = 1, cb = na, sb = na, ry = 1, cy = na, sy = na;
    (void) vk_t_geom(a, &ra, &ca, &sa);
    (void) vk_t_geom(b, &rb, &cb, &sb);
    (void) vk_t_geom(y, &ry, &cy, &sy);
    const size_t cols = sa != ca ? ca : (sb != cb ? cb : cy);
    const size_t rows = (sa != ca || sb != cb || sy != cy) ? na / cols : 1;
    const size_t cc   = rows == 1 ? na : cols;
    if (sa == ca) {
        sa = cc;
    }
    if (sb == cb) {
        sb = cc;
    }
    if (sy == cy) {
        sy = cc;
    }
    for (size_t r = 0; r < rows; r++) {
        const float *arow = ap + r * sa;
        const float *brow = bp + r * sb;
        float       *yrow = yp + r * sy;
        for (size_t i = 0; i < cc; i++) {
            switch (op) {
            case 0:
                yrow[i] = arow[i] + brow[i];
                break;
            case 1:
                yrow[i] = arow[i] * brow[i];
                break;
            default: {
                const float v = arow[i];
                const float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
                yrow[i]       = (0.5f * v * (1.0f + tanhf(u))) * brow[i];
            }
            }
        }
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_add(struct geist_backend      *be,
                                              const struct geist_tensor *a,
                                              const struct geist_tensor *b,
                                              struct geist_tensor       *y) {
    if (vk_try_ew3(be, VK_PIPE_ADD, a, b, y)) {
        return GEIST_OK;
    }
    return vk_ew3_cpu(be, 0, a, b, y, "add");
}

[[nodiscard]] static enum geist_status vk_mul(struct geist_backend      *be,
                                              const struct geist_tensor *a,
                                              const struct geist_tensor *b,
                                              struct geist_tensor       *y) {
    if (vk_try_ew3(be, VK_PIPE_MUL, a, b, y)) {
        return GEIST_OK;
    }
    return vk_ew3_cpu(be, 1, a, b, y, "mul");
}

static constexpr float VK_GELU_K0 = 0.7978845608028654f;

/* sqrt(2/pi) */
static constexpr float VK_GELU_K1 = 0.044715f;

[[nodiscard]] static enum geist_status
vk_gelu_tanh(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y) {
    {
        const size_t           n = vk_t_n(x);
        VkDescriptorBufferInfo bi[2];
        uint32_t               off[2];
        if (VK_OPS(be, 2u) && n != 0 && n == vk_t_n(y) && vk_tensor_gpu(x, &bi[0], &off[0]) &&
            vk_tensor_gpu(y, &bi[1], &off[1])) {
            const uint32_t         push[4] = {(uint32_t) n, off[0], off[1], 0};
            const struct vk_access acc[2]  = {vk_acc_tensor(x, false), vk_acc_tensor(y, true)};
            if (vk_seq_dispatch_acc(
                        be, VK_PIPE_GELU, bi, acc, push, sizeof(push), vk_groups(n), 1, 1) ==
                GEIST_OK) {
                return GEIST_OK;
            }
        }
    }
    size_t       nx = 0, ny = 0;
    const float *xp = vk_tensor_host(x, &nx);
    float       *yp = vk_tensor_host(y, &ny);
    if (xp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan gelu_tanh: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    for (size_t i = 0; i < nx; i++) {
        const float v = xp[i];
        const float u = VK_GELU_K0 * (v + VK_GELU_K1 * v * v * v);
        yp[i]         = 0.5f * v * (1.0f + tanhf(u));
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_gelu_tanh_mul(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        const struct geist_tensor *z,
                                                        struct geist_tensor       *y) {
    if (vk_try_ew3(be, VK_PIPE_GELU_MUL, x, z, y)) {
        return GEIST_OK;
    }
    return vk_ew3_cpu(be, 2, x, z, y, "gelu_tanh_mul");
}

[[nodiscard]] static enum geist_status vk_gelu_tanh_mul_scaled(struct geist_backend      *be,
                                                               const struct geist_tensor *x,
                                                               const struct geist_tensor *z,
                                                               const float               *scale,
                                                               struct geist_tensor       *y) {
    size_t       nx = 0, nz = 0, ny = 0;
    const float *xp = vk_tensor_host(x, &nx);
    const float *zp = vk_tensor_host(z, &nz);
    float       *yp = vk_tensor_host(y, &ny);
    if (xp == nullptr || zp == nullptr || yp == nullptr || scale == nullptr || nx != nz ||
        nx != ny || y->ndim < 1) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan gelu_tanh_mul_scaled: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    const size_t feat = (size_t) y->shape[y->ndim - 1];
    if (feat == 0 || nx % feat != 0) {
        geist_backend_set_error(
                be, GEIST_E_INVALID_ARG, "vulkan gelu_tanh_mul_scaled: feature mismatch");
        return GEIST_E_INVALID_ARG;
    }
    const size_t rows = nx / feat;
    for (size_t r = 0; r < rows; r++) {
        const size_t base = r * feat;
        for (size_t j = 0; j < feat; j++) {
            const size_t i = base + j;
            const float  v = xp[i];
            const float  u = VK_GELU_K0 * (v + VK_GELU_K1 * v * v * v);
            yp[i]          = (0.5f * v * (1.0f + tanhf(u))) * zp[i] * scale[j];
        }
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
vk_relu_squared(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y) {
    size_t       nx = 0, ny = 0;
    const float *xp = vk_tensor_host(x, &nx);
    float       *yp = vk_tensor_host(y, &ny);
    if (xp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan relu_squared: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    for (size_t i = 0; i < nx; i++) {
        const float v = xp[i] > 0.0f ? xp[i] : 0.0f;
        yp[i]         = v * v;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
vk_silu(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y) {
    size_t       nx = 0, ny = 0;
    const float *xp = vk_tensor_host(x, &nx);
    float       *yp = vk_tensor_host(y, &ny);
    if (xp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan silu: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    for (size_t i = 0; i < nx; i++) {
        /* Overflow-safe sigmoid form (see cpu_scalar_silu, #275). */
        const float v = xp[i];
        const float e = expf(-fabsf(v));
        yp[i]         = (v >= 0.0f) ? v / (1.0f + e) : (v * e) / (1.0f + e);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_rmsnorm(struct geist_backend      *be,
                                                  const struct geist_tensor *x,
                                                  const struct geist_tensor *w,
                                                  float                      eps,
                                                  struct geist_tensor       *y) {
    {
        const size_t           n    = vk_t_n(x);
        const size_t           feat = n != 0 ? (size_t) x->shape[x->ndim - 1] : 0;
        VkDescriptorBufferInfo bi[3];
        uint32_t               off[3];
        if (VK_OPS(be, 4u) && feat != 0 && n % feat == 0 && vk_t_n(w) == feat && vk_t_n(y) == n &&
            vk_tensor_gpu(x, &bi[0], &off[0]) && vk_tensor_gpu(w, &bi[1], &off[1]) &&
            vk_tensor_gpu(y, &bi[2], &off[2])) {
            const struct {
                uint32_t rows, feat, x, w, y;
                float    eps;
            } push = {(uint32_t) (n / feat), (uint32_t) feat, off[0], off[1], off[2], eps};
            const struct vk_access acc[3] = {
                    vk_acc_tensor(x, false), vk_acc_tensor(w, false), vk_acc_tensor(y, true)};
            if (vk_seq_dispatch_acc(be,
                                    VK_PIPE_RMSNORM,
                                    bi,
                                    acc,
                                    &push,
                                    sizeof(push),
                                    (uint32_t) (n / feat),
                                    1,
                                    1) == GEIST_OK) {
                return GEIST_OK;
            }
        }
    }
    size_t       nx = 0, nw = 0, ny = 0;
    const float *xp = vk_tensor_host(x, &nx);
    const float *wp = vk_tensor_host(w, &nw);
    float       *yp = vk_tensor_host(y, &ny);
    if (xp == nullptr || wp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan rmsnorm: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    const size_t feat = (size_t) x->shape[x->ndim - 1];
    if (feat == 0 || nw != feat || nx % feat != 0) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan rmsnorm: feature mismatch");
        return GEIST_E_INVALID_ARG;
    }
    const size_t n_rows = nx / feat;
    for (size_t r = 0; r < n_rows; r++) {
        const float *row_x = xp + r * feat;
        float       *row_y = yp + r * feat;
        double       sumsq = 0.0;
        for (size_t i = 0; i < feat; i++) {
            sumsq += (double) row_x[i] * (double) row_x[i];
        }
        const float inv = (float) (1.0 / sqrt(sumsq / (double) feat + (double) eps));
        for (size_t i = 0; i < feat; i++) {
            row_y[i] = row_x[i] * inv * wp[i];
        }
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_rope_apply(struct geist_backend      *be,
                                                     struct geist_tensor       *x,
                                                     const struct geist_tensor *cos,
                                                     const struct geist_tensor *sin) {
    {
        VkDescriptorBufferInfo bi[3];
        uint32_t               off[3];
        if (VK_OPS(be, 8u) && x != nullptr && x->ndim == 3 && vk_t_n(x) != 0 && vk_t_n(cos) != 0 &&
            vk_t_n(sin) != 0 && vk_tensor_gpu(x, &bi[0], &off[0]) &&
            vk_tensor_gpu(cos, &bi[1], &off[1]) && vk_tensor_gpu(sin, &bi[2], &off[2])) {
            const size_t   seq     = (size_t) x->shape[0];
            const size_t   heads   = (size_t) x->shape[1];
            const size_t   hd      = (size_t) x->shape[2];
            const size_t   pairs   = seq * heads * hd / 2;
            const uint32_t push[6] = {
                    (uint32_t) pairs, (uint32_t) heads, (uint32_t) hd, off[0], off[1], off[2]};
            const struct vk_access acc[3] = {
                    vk_acc_tensor(x, true), vk_acc_tensor(cos, false), vk_acc_tensor(sin, false)};
            if (vk_seq_dispatch_acc(
                        be, VK_PIPE_ROPE, bi, acc, push, sizeof(push), vk_groups(pairs), 1, 1) ==
                GEIST_OK) {
                return GEIST_OK;
            }
        }
    }
    size_t       nx = 0, nc = 0, ns = 0;
    float       *xp   = vk_tensor_host(x, &nx);
    const float *cosp = vk_tensor_host(cos, &nc);
    const float *sinp = vk_tensor_host(sin, &ns);
    if (xp == nullptr || cosp == nullptr || sinp == nullptr || x->ndim != 3) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan rope_apply: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    rope_apply(xp, cosp, sinp, (size_t) x->shape[0], (size_t) x->shape[1], (size_t) x->shape[2]);
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_embedding_lookup(struct geist_backend      *be,
                                                           const struct geist_tensor *embed_table,
                                                           geist_token_t              token_id,
                                                           struct geist_tensor       *out) {
    size_t       n_table = 0, n_out = 0;
    const float *tablep = vk_tensor_host(embed_table, &n_table);
    float       *outp   = vk_tensor_host(out, &n_out);
    if (tablep == nullptr || outp == nullptr || embed_table->ndim != 2) {
        return GEIST_E_INVALID_ARG;
    }
    const int64_t vocab_size = embed_table->shape[0];
    const int64_t d_model    = embed_table->shape[1];
    if (token_id < 0 || (int64_t) token_id >= vocab_size || n_out != (size_t) d_model) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "vulkan embedding_lookup: token %d out of range",
                                (int) token_id);
        return GEIST_E_INVALID_ARG;
    }
    memcpy(outp, tablep + (size_t) token_id * (size_t) d_model, (size_t) d_model * sizeof(float));
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_attention(struct geist_backend      *be,
                                                    const struct geist_tensor *q,
                                                    const struct geist_tensor *k,
                                                    const struct geist_tensor *v,
                                                    size_t                     q_offset,
                                                    size_t                     sliding_window,
                                                    struct geist_tensor       *out) {
    /* Flash-decoding: n_q == 1 with f16 KV and enough context to make the
     * 8-workgroup direct kernel starve the GPU. Partials go into the
     * device x-ring; a combine pass reduces per head. */
    struct vk_state *stt = be->state;
    if (VK_OPS(be, 16u) && q != nullptr && k != nullptr && v != nullptr && out != nullptr &&
        q->ndim == 3 && q->shape[0] == 1 && k->dtype == GEIST_DTYPE_F16 &&
        (size_t) k->shape[0] > 192 && q->shape[2] <= 512 && vk_t_n(q) != 0 && vk_t_n16(k) != 0 &&
        stt->pipes[VK_PIPE_ATTN_PART_F16] != VK_NULL_HANDLE) {
        const uint32_t         qh         = (uint32_t) q->shape[1];
        const uint32_t         hd         = (uint32_t) q->shape[2];
        const uint32_t         n_kv       = (uint32_t) k->shape[0];
        const uint32_t         kvh        = (uint32_t) k->shape[1];
        const uint32_t         n_chunks   = (n_kv + 127u) / 128u;
        const size_t           part_bytes = (size_t) qh * n_chunks * (hd + 2u) * 4u;
        VkDescriptorBufferInfo bq, bk, bv, bo;
        uint32_t               qo, ko, vo, oo;
        if (stt->xring == nullptr &&
            vk_buffer_create(
                    be, VK_XRING_CAP, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE, &stt->xring) !=
                    GEIST_OK) {
            goto attn_generic;
        }
        if (stt->xring_used + part_bytes > stt->xring->bytes) {
            vk_seq_flush(stt);
        }
        if (part_bytes > stt->xring->bytes || !vk_tensor_gpu(q, &bq, &qo) ||
            !vk_tensor_gpu_f16(k, &bk, &ko) || !vk_tensor_gpu_f16(v, &bv, &vo) ||
            !vk_tensor_gpu(out, &bo, &oo) ||
            ((qo | ko | vo) & 3u) != 0u /* 4-wide K/V/Q streams */) {
            goto attn_generic;
        }
        const uint32_t         po        = (uint32_t) (stt->xring_used / 4u);
        const uint32_t         push1[11] = {n_kv,
                                            qh,
                                            kvh,
                                            hd,
                                            (uint32_t) q_offset,
                                            (uint32_t) sliding_window,
                                            qo,
                                            ko,
                                            vo,
                                            po,
                                            n_chunks};
        VkDescriptorBufferInfo bi1[4]    = {
                bq, bk, bv, {.buffer = stt->xring->buf, .range = VK_WHOLE_SIZE}};
        const struct vk_access acc1[4] = {vk_acc_tensor(q, false),
                                          vk_acc_tensor16(k, false),
                                          vk_acc_tensor16(v, false),
                                          vk_acc(stt->xring_used, part_bytes, true)};
        if (vk_seq_dispatch_acc(
                    be, VK_PIPE_ATTN_PART_F16, bi1, acc1, push1, sizeof(push1), n_chunks, qh, 1) !=
            GEIST_OK) {
            goto attn_generic;
        }
        const uint32_t         push2[5] = {qh, hd, n_chunks, po, oo};
        VkDescriptorBufferInfo bi2[2]   = {{.buffer = stt->xring->buf, .range = VK_WHOLE_SIZE}, bo};
        const struct vk_access acc2[2]  = {vk_acc(stt->xring_used, part_bytes, false),
                                           vk_acc_tensor(out, true)};
        stt->xring_used                 = (stt->xring_used + part_bytes + 63u) & ~(size_t) 63u;
        if (vk_seq_dispatch_acc(be, VK_PIPE_ATTN_COMB, bi2, acc2, push2, sizeof(push2), qh, 1, 1) ==
            GEIST_OK) {
            return GEIST_OK;
        }
    }
attn_generic:;
    {
        VkDescriptorBufferInfo bi[4];
        uint32_t               off[4];
        const bool             kv16 = k != nullptr && k->dtype == GEIST_DTYPE_F16;
        if (VK_OPS(be, 16u) && q != nullptr && k != nullptr && q->ndim == 3 && k->ndim == 3 &&
            vk_t_n(q) != 0 &&
            (kv16 ? (vk_t_n16(k) != 0 && vk_t_n16(v) != 0 &&
                     vk_tensor_gpu_f16(k, &bi[1], &off[1]) && vk_tensor_gpu_f16(v, &bi[2], &off[2]))
                  : (vk_t_n(k) != 0 && vk_t_n(v) != 0 && vk_tensor_gpu(k, &bi[1], &off[1]) &&
                     vk_tensor_gpu(v, &bi[2], &off[2]))) &&
            vk_t_n(out) != 0 && vk_tensor_gpu(q, &bi[0], &off[0]) &&
            vk_tensor_gpu(out, &bi[3], &off[3]) &&
            /* f16 kernel streams q/k/v as 4-wide vectors */
            (!kv16 || ((off[0] | off[1] | off[2]) & 3u) == 0u)) {
            const uint32_t         n_q      = (uint32_t) q->shape[0];
            const uint32_t         qh       = (uint32_t) q->shape[1];
            const uint32_t         hd       = (uint32_t) q->shape[2];
            const uint32_t         n_kv     = (uint32_t) k->shape[0];
            const uint32_t         kvh      = (uint32_t) k->shape[1];
            const uint32_t         push[11] = {n_q,
                                               n_kv,
                                               qh,
                                               kvh,
                                               hd,
                                               (uint32_t) q_offset,
                                               (uint32_t) sliding_window,
                                               off[0],
                                               off[1],
                                               off[2],
                                               off[3]};
            const struct vk_access acc[4]   = {
                    vk_acc_tensor(q, false),
                    kv16 ? vk_acc_tensor16(k, false) : vk_acc_tensor(k, false),
                    kv16 ? vk_acc_tensor16(v, false) : vk_acc_tensor(v, false),
                    vk_acc_tensor(out, true)};
            if (vk_seq_dispatch_acc(be,
                                    kv16 ? VK_PIPE_ATTENTION_F16 : VK_PIPE_ATTENTION,
                                    bi,
                                    acc,
                                    push,
                                    sizeof(push),
                                    n_q,
                                    qh,
                                    1) == GEIST_OK) {
                return GEIST_OK;
            }
        }
    }
    size_t       nq = 0, nk = 0, nv = 0, no = 0;
    const float *qp = vk_tensor_host(q, &nq);
    const float *kp = vk_tensor_host(k, &nk);
    const float *vp = vk_tensor_host(v, &nv);
    float       *op = vk_tensor_host(out, &no);
    if (qp == nullptr || kp == nullptr || vp == nullptr || op == nullptr || q->ndim != 3 ||
        k->ndim != 3 || v->ndim != 3) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan attention: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    attention_mqa_causal_kv(qp,
                            kp,
                            vp,
                            (size_t) q->shape[0],
                            (size_t) k->shape[0],
                            q_offset,
                            (size_t) q->shape[1],
                            (size_t) k->shape[1],
                            (size_t) q->shape[2],
                            sliding_window,
                            op);
    return GEIST_OK;
}

/* ---- New batched-submit ops (Phase 3) --------------------------------- */

[[nodiscard]] static enum geist_status vk_rmsnorm_add(struct geist_backend      *be,
                                                      const struct geist_tensor *res,
                                                      const struct geist_tensor *x,
                                                      const struct geist_tensor *w,
                                                      float                      eps,
                                                      struct geist_tensor       *y) {
    const size_t n    = vk_t_n(x);
    const size_t feat = n != 0 ? (size_t) x->shape[x->ndim - 1] : 0;
    {
        VkDescriptorBufferInfo bi[4];
        uint32_t               off[4];
        if (VK_OPS(be, 4u) && feat != 0 && n % feat == 0 && vk_t_n(w) == feat && vk_t_n(res) == n &&
            vk_t_n(y) == n && vk_tensor_gpu(x, &bi[0], &off[0]) &&
            vk_tensor_gpu(w, &bi[1], &off[1]) && vk_tensor_gpu(res, &bi[2], &off[2]) &&
            vk_tensor_gpu(y, &bi[3], &off[3])) {
            const struct {
                uint32_t rows, feat, x, w, r, y;
                float    eps;
            } push = {(uint32_t) (n / feat), (uint32_t) feat, off[0], off[1], off[2], off[3], eps};
            const struct vk_access acc[4] = {vk_acc_tensor(x, false),
                                             vk_acc_tensor(w, false),
                                             vk_acc_tensor(res, false),
                                             vk_acc_tensor(y, true)};
            if (vk_seq_dispatch_acc(be,
                                    VK_PIPE_RMSNORM_ADD,
                                    bi,
                                    acc,
                                    &push,
                                    sizeof(push),
                                    (uint32_t) (n / feat),
                                    1,
                                    1) == GEIST_OK) {
                return GEIST_OK;
            }
        }
    }
    /* CPU fallback: y = res + rmsnorm(x) * w */
    size_t       nx = 0, nw = 0, nr = 0, ny = 0;
    const float *xp = vk_tensor_host(x, &nx);
    const float *wp = vk_tensor_host(w, &nw);
    const float *rp = vk_tensor_host(res, &nr);
    float       *yp = vk_tensor_host(y, &ny);
    if (xp == nullptr || wp == nullptr || rp == nullptr || yp == nullptr || nx != ny || nr != nx ||
        feat == 0 || nw != feat || nx % feat != 0) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan rmsnorm_add: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    for (size_t r = 0; r < nx / feat; r++) {
        const float *row_x = xp + r * feat;
        const float *row_r = rp + r * feat;
        float       *row_y = yp + r * feat;
        double       sumsq = 0.0;
        for (size_t i = 0; i < feat; i++) {
            sumsq += (double) row_x[i] * (double) row_x[i];
        }
        const float inv = (float) (1.0 / sqrt(sumsq / (double) feat + (double) eps));
        for (size_t i = 0; i < feat; i++) {
            row_y[i] = row_r[i] + row_x[i] * inv * wp[i];
        }
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_scale_f32(struct geist_backend      *be,
                                                    const struct geist_tensor *x,
                                                    float                      scale,
                                                    struct geist_tensor       *y) {
    const size_t n = vk_t_n(x);
    {
        VkDescriptorBufferInfo bi[2];
        uint32_t               off[2];
        if (n != 0 && n == vk_t_n(y) && vk_tensor_gpu(x, &bi[0], &off[0]) &&
            vk_tensor_gpu(y, &bi[1], &off[1])) {
            const struct {
                uint32_t n, x, y;
                float    scale;
            } push                        = {(uint32_t) n, off[0], off[1], scale};
            const struct vk_access acc[2] = {vk_acc_tensor(x, false), vk_acc_tensor(y, true)};
            if (vk_seq_dispatch_acc(
                        be, VK_PIPE_SCALE, bi, acc, &push, sizeof(push), vk_groups(n), 1, 1) ==
                GEIST_OK) {
                return GEIST_OK;
            }
        }
    }
    size_t       nx = 0, ny = 0;
    const float *xp = vk_tensor_host(x, &nx);
    float       *yp = vk_tensor_host(y, &ny);
    if (xp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan scale_f32: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    for (size_t i = 0; i < nx; i++) {
        yp[i] = xp[i] * scale;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
vk_argmax_f32(struct geist_backend *be, const struct geist_tensor *logits, int32_t *out_index) {
    struct vk_state *st = be->state;
    if (!VK_OPS(be, 128u)) {
        return GEIST_E_UNSUPPORTED;
    }
    const size_t           n = vk_t_n(logits);
    VkDescriptorBufferInfo bi[2];
    uint32_t               off[2];
    if (n == 0 || out_index == nullptr || !vk_tensor_gpu(logits, &bi[0], &off[0])) {
        return GEIST_E_UNSUPPORTED; /* arch scans on the host */
    }
    if (st->argmax_out == nullptr) {
        if (vk_buffer_create(be, 16, GEIST_BUFFER_STAGING, GEIST_MEMORY_AUTO, &st->argmax_out) !=
            GEIST_OK) {
            return GEIST_E_UNSUPPORTED;
        }
    }
    bi[1] = (VkDescriptorBufferInfo) {.buffer = st->argmax_out->buf, .range = VK_WHOLE_SIZE};
    const uint32_t         push[3] = {(uint32_t) n, off[0], 0};
    const struct vk_access acc[2]  = {vk_acc_tensor(logits, false), vk_acc_all(true)};
    enum geist_status      s =
            vk_seq_dispatch_acc(be, VK_PIPE_ARGMAX, bi, acc, push, sizeof(push), 1, 1, 1);
    if (s != GEIST_OK) {
        return GEIST_E_UNSUPPORTED;
    }
    vk_seq_flush(st); /* the one intended sync point per decoded token */
    *out_index = ((const int32_t *) st->argmax_out->mapped)[0];
    return GEIST_OK;
}

/* Tensor-path linear: x staged into the VRAM ring, weight from the VRAM
 * registry, y written to its host-visible home — no host round-trip, no
 * flush. THE hot path since Phase 3. */
[[nodiscard]] static enum geist_status vk_linear_t(struct geist_backend      *be,
                                                   const struct geist_tensor *t_x,
                                                   const struct geist_weight *w,
                                                   const struct geist_tensor *t_w,
                                                   size_t                     m,
                                                   struct geist_tensor       *t_y) {
    (void) t_w;
    struct vk_state *st = be->state;
    if (!VK_OPS(be, 1u)) {
        return GEIST_E_UNSUPPORTED;
    }
    enum vk_pipe mv, mm;
    switch ((enum geist_dtype) w->dtype) {
    case GEIST_DTYPE_Q4_K:
        mv = VK_PIPE_MATVEC_Q4K;
        mm = VK_PIPE_MATMUL_Q4K;
        break;
    case GEIST_DTYPE_Q6_K:
        mv = VK_PIPE_MATVEC_Q6K;
        mm = VK_PIPE_MATMUL_Q6K;
        break;
    case GEIST_DTYPE_F32:
        mv = VK_PIPE_MATVEC_F32;
        mm = VK_PIPE_MATMUL_F32;
        break;
    default:
        return GEIST_E_UNSUPPORTED;
    }
    struct geist_buffer   *wbuf = vk_weight_lookup(st, w->raw);
    VkDescriptorBufferInfo bi[3];
    uint32_t               xo, yo;
    if (wbuf == nullptr || m == 0 || vk_t_n(t_x) == 0 || vk_t_n(t_y) == 0 ||
        !vk_tensor_gpu(t_y, &bi[2], &yo)) {
        return GEIST_E_UNSUPPORTED;
    }
    const uint32_t n_in  = (uint32_t) w->n_in;
    const uint32_t n_out = (uint32_t) w->n_out;
    uint32_t       x_stride;
    if (t_x->buffer->device_mem) {
        /* BAR-resident activations: bind in place, no staging copy. */
        if (!vk_tensor_gpu(t_x, &bi[0], &xo)) {
            return GEIST_E_UNSUPPORTED;
        }
        x_stride = t_x->ndim >= 2 ? (uint32_t) t_x->stride[t_x->ndim - 2] : n_in;
    } else {
        if (!vk_xring_stage(be, t_x, m, n_in, &xo)) {
            return GEIST_E_UNSUPPORTED;
        }
        bi[0]    = (VkDescriptorBufferInfo) {.buffer = st->xring->buf, .range = VK_WHOLE_SIZE};
        x_stride = n_in; /* ring copy is contiguous */
    }
    bi[1] = (VkDescriptorBufferInfo) {.buffer = wbuf->buf, .range = VK_WHOLE_SIZE};
    const uint32_t       y_stride = t_y->ndim >= 2 ? (uint32_t) t_y->stride[t_y->ndim - 2] : n_out;
    const struct vk_push push     = {.n_in           = n_in,
                                     .n_out          = n_out,
                                     .blocks_per_row = n_in / 256,
                                     .rows           = (uint32_t) m,
                                     .x_offset       = xo,
                                     .y_offset       = yo,
                                     .x_stride       = x_stride,
                                     .y_stride       = y_stride};
    enum vk_pipe         lpipe    = m == 1 ? mv : mm;
    uint32_t             gx       = vk_linear_gx(lpipe, n_out);
    uint32_t             gy       = vk_linear_gy(lpipe, (uint32_t) m);
    /* Tensor-core path for conforming GEMMs (shaders assume w_offset == 0,
     * which holds for all registry uploads). */
    if (m > 1) {
        vk_linear_cm_route(st, &lpipe, (uint32_t) m, n_out, &gx, &gy);
    }
    const struct vk_access acc[3] = {
            t_x->buffer->device_mem ? vk_acc_tensor(t_x, false)
                                    : vk_acc((uint64_t) xo * 4u, (uint64_t) m * n_in * 4u, false),
            vk_acc_all(false),
            vk_acc_tensor(t_y, true)};
    return vk_seq_dispatch_acc(be, lpipe, bi, acc, &push, sizeof(push), gx, gy, 1);
}

[[nodiscard]] static enum geist_status vk_linear_t_pair(struct geist_backend      *be,
                                                        const struct geist_tensor *t_x,
                                                        const struct geist_weight *w0,
                                                        const struct geist_tensor *t_w0,
                                                        const struct geist_weight *w1,
                                                        const struct geist_tensor *t_w1,
                                                        size_t                     m,
                                                        struct geist_tensor       *t_y0,
                                                        struct geist_tensor       *t_y1) {
    /* Two appended dispatches; a fused two-weight kernel is a Phase-3c
     * candidate once the profiler ranks it. Check both up front so the
     * fallback never sees a half-done pair. */
    struct vk_state *st = be->state;
    if (vk_weight_lookup(st, w0->raw) == nullptr || vk_weight_lookup(st, w1->raw) == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    enum geist_status s = vk_linear_t(be, t_x, w0, t_w0, m, t_y0);
    if (s == GEIST_OK) {
        s = vk_linear_t(be, t_x, w1, t_w1, m, t_y1);
    }
    return s;
}

[[nodiscard]] static enum geist_status
vk_embedding_lookup_scaled(struct geist_backend      *be,
                           const struct geist_tensor *embed_table,
                           geist_token_t              token_id,
                           float                      scale,
                           struct geist_tensor       *out) {
    struct vk_state *st = be->state;
    if (!VK_OPS(be, 64u)) {
        return GEIST_E_UNSUPPORTED;
    }
    if (embed_table == nullptr || out == nullptr || embed_table->ndim != 2 ||
        embed_table->buffer == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    const int64_t vocab = embed_table->shape[0];
    const int64_t d     = embed_table->shape[1];
    if (token_id < 0 || (int64_t) token_id >= vocab) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "vulkan embed_scaled: token %d out of range",
                                (int) token_id);
        return GEIST_E_INVALID_ARG;
    }
    uint32_t dtype_code;
    switch ((enum geist_dtype) embed_table->dtype) {
    case GEIST_DTYPE_F32:
        dtype_code = 0;
        break;
    case GEIST_DTYPE_F16:
        dtype_code = 1;
        break;
    case GEIST_DTYPE_BF16:
        dtype_code = 2;
        break;
    case GEIST_DTYPE_Q4_K:
        dtype_code = 8;
        break;
    case GEIST_DTYPE_Q5_K:
        dtype_code = 9;
        break;
    case GEIST_DTYPE_Q6_K:
        dtype_code = 10;
        break;
    default:
        return GEIST_E_UNSUPPORTED;
    }
    /* Table bytes: prefer the resolve-time VRAM copy (embed tables go
     * through resolve_weight); fall back to a bindable host region. */
    const uint8_t *host =
            embed_table->buffer->host_alias != nullptr
                    ? (const uint8_t *) embed_table->buffer->host_alias + embed_table->offset
                    : nullptr;
    struct geist_buffer   *wbuf = host != nullptr ? vk_weight_lookup(st, host) : nullptr;
    VkDescriptorBufferInfo bi[2];
    uint32_t               w_elem_off = 0;
    if (wbuf != nullptr) {
        bi[0] = (VkDescriptorBufferInfo) {.buffer = wbuf->buf, .range = VK_WHOLE_SIZE};
    } else {
        if (dtype_code == 10) {
            /* Q6_K GPU copies are repacked to 216-byte blocks; the arena
             * holds canonical 210-byte blocks the shader can't read. */
            return GEIST_E_UNSUPPORTED;
        }
        struct geist_tensor bytes_view = *embed_table;
        bytes_view.dtype               = GEIST_DTYPE_F32; /* only for the offset calc */
        if (embed_table->buffer->buf == VK_NULL_HANDLE ||
            (embed_table->buffer->base_off + embed_table->offset) % 4 != 0) {
            return GEIST_E_UNSUPPORTED;
        }
        bi[0]      = (VkDescriptorBufferInfo) {.buffer = embed_table->buffer->buf,
                                               .range  = VK_WHOLE_SIZE};
        w_elem_off = (uint32_t) (embed_table->buffer->base_off + embed_table->offset);
        (void) bytes_view;
    }
    uint32_t yo;
    if (vk_t_n(out) != (size_t) d || !vk_tensor_gpu(out, &bi[1], &yo)) {
        return GEIST_E_UNSUPPORTED;
    }
    uint32_t bpr;
    switch (dtype_code) {
    case 8:
        bpr = (uint32_t) (d / 256);
        break;
    case 9:
        bpr = (uint32_t) (d / 256);
        break;
    case 10:
        bpr = (uint32_t) (d / 256);
        break;
    default:
        bpr = 0;
        break;
    }
    const struct {
        uint32_t n_in, token, dtype, bpr, w_byte, y;
        float    scale;
    } push = {(uint32_t) d, (uint32_t) token_id, dtype_code, bpr, w_elem_off, yo, scale};
    const struct vk_access acc[2] = {vk_acc_all(false), vk_acc_tensor(out, true)};
    return vk_seq_dispatch_acc(
            be, VK_PIPE_EMBED, bi, acc, &push, sizeof(push), vk_groups((size_t) d), 1, 1);
}

/* Fused decode FFN front (m == 1, both weights Q4_K): one dispatch for
 * gelu(x.gate^T) * (x.up^T) — replaces two matvecs + gelu_mul. */
[[nodiscard]] static enum geist_status vk_ffn_gate_up(struct geist_backend      *be,
                                                      const struct geist_tensor *t_x,
                                                      const struct geist_tensor *gate_w,
                                                      const struct geist_tensor *up_w,
                                                      struct geist_tensor       *y) {
    struct vk_state *st = be->state;
    if (!VK_OPS(be, 1u) || gate_w == nullptr || up_w == nullptr || t_x == nullptr ||
        gate_w->dtype != GEIST_DTYPE_Q4_K || up_w->dtype != GEIST_DTYPE_Q4_K || gate_w->ndim != 2 ||
        t_x->shape[0] != 1) {
        return GEIST_E_UNSUPPORTED;
    }
    const uint32_t n_out = (uint32_t) gate_w->shape[0];
    const uint32_t n_in  = (uint32_t) gate_w->shape[1];
    if (n_in % 256u != 0u || up_w->shape[0] != gate_w->shape[0] ||
        up_w->shape[1] != gate_w->shape[1]) {
        return GEIST_E_UNSUPPORTED;
    }
    const uint8_t *g_host = gate_w->buffer != nullptr && gate_w->buffer->host_alias != nullptr
                                    ? (const uint8_t *) gate_w->buffer->host_alias + gate_w->offset
                                    : nullptr;
    const uint8_t *u_host = up_w->buffer != nullptr && up_w->buffer->host_alias != nullptr
                                    ? (const uint8_t *) up_w->buffer->host_alias + up_w->offset
                                    : nullptr;
    struct geist_buffer   *gbuf = g_host != nullptr ? vk_weight_lookup(st, g_host) : nullptr;
    struct geist_buffer   *ubuf = u_host != nullptr ? vk_weight_lookup(st, u_host) : nullptr;
    VkDescriptorBufferInfo bi[4];
    uint32_t               xo, yo;
    if (gbuf == nullptr || ubuf == nullptr || vk_t_n(y) < n_out || !vk_tensor_gpu(y, &bi[3], &yo)) {
        return GEIST_E_UNSUPPORTED;
    }
    if (t_x->buffer != nullptr && t_x->buffer->device_mem) {
        if (!vk_tensor_gpu(t_x, &bi[0], &xo)) {
            return GEIST_E_UNSUPPORTED;
        }
    } else {
        if (!vk_xring_stage(be, t_x, 1, n_in, &xo)) {
            return GEIST_E_UNSUPPORTED;
        }
        bi[0] = (VkDescriptorBufferInfo) {.buffer = st->xring->buf, .range = VK_WHOLE_SIZE};
    }
    bi[1] = (VkDescriptorBufferInfo) {.buffer = gbuf->buf, .range = VK_WHOLE_SIZE};
    bi[2] = (VkDescriptorBufferInfo) {.buffer = ubuf->buf, .range = VK_WHOLE_SIZE};
    const struct vk_push   push   = {.n_in           = n_in,
                                     .n_out          = n_out,
                                     .blocks_per_row = n_in / 256u,
                                     .rows           = 1,
                                     .x_offset       = xo,
                                     .y_offset       = yo,
                                     .x_stride       = n_in,
                                     .y_stride       = n_out};
    const struct vk_access acc[4] = {vk_acc_tensor(t_x, false),
                                     vk_acc_all(false),
                                     vk_acc_all(false),
                                     vk_acc_tensor(y, true)};
    return vk_seq_dispatch_acc(
            be, VK_PIPE_FFN_GATE_UP, bi, acc, &push, sizeof(push), (n_out + 3u) / 4u, 1, 1);
}

/* ffn_gate_up with the pre-FFN rmsnorm folded into the kernel's x loads
 * (each 32-thread workgroup recomputes the row's inverse RMS — ~1 us of
 * L2-hot reads vs a 9 us serial norm dispatch). Decode only, Q4_K. */
[[nodiscard]] static enum geist_status vk_ffn_norm_gate_up(struct geist_backend      *be,
                                                           const struct geist_tensor *t_x,
                                                           const struct geist_tensor *norm_w,
                                                           float                      eps,
                                                           const struct geist_tensor *gate_w,
                                                           const struct geist_tensor *up_w,
                                                           struct geist_tensor       *y) {
    struct vk_state *st = be->state;
    if (!VK_OPS(be, 1u) || gate_w == nullptr || up_w == nullptr || t_x == nullptr ||
        norm_w == nullptr || gate_w->dtype != GEIST_DTYPE_Q4_K || up_w->dtype != GEIST_DTYPE_Q4_K ||
        gate_w->ndim != 2 || t_x->shape[0] != 1) {
        return GEIST_E_UNSUPPORTED;
    }
    const uint32_t n_out = (uint32_t) gate_w->shape[0];
    const uint32_t n_in  = (uint32_t) gate_w->shape[1];
    if (n_in % 256u != 0u || n_out % 8u != 0u || up_w->shape[0] != gate_w->shape[0] ||
        up_w->shape[1] != gate_w->shape[1] || vk_t_n(norm_w) != n_in) {
        return GEIST_E_UNSUPPORTED;
    }
    const uint8_t *g_host = gate_w->buffer != nullptr && gate_w->buffer->host_alias != nullptr
                                    ? (const uint8_t *) gate_w->buffer->host_alias + gate_w->offset
                                    : nullptr;
    const uint8_t *u_host = up_w->buffer != nullptr && up_w->buffer->host_alias != nullptr
                                    ? (const uint8_t *) up_w->buffer->host_alias + up_w->offset
                                    : nullptr;
    struct geist_buffer   *gbuf = g_host != nullptr ? vk_weight_lookup(st, g_host) : nullptr;
    struct geist_buffer   *ubuf = u_host != nullptr ? vk_weight_lookup(st, u_host) : nullptr;
    VkDescriptorBufferInfo bi[5];
    uint32_t               xo, nwo, yo;
    if (gbuf == nullptr || ubuf == nullptr || vk_t_n(y) < n_out || !vk_tensor_gpu(y, &bi[4], &yo) ||
        !vk_tensor_gpu(t_x, &bi[0], &xo) || !vk_tensor_gpu(norm_w, &bi[3], &nwo) ||
        (xo & 3u) != 0u || (nwo & 3u) != 0u) {
        return GEIST_E_UNSUPPORTED;
    }
    bi[1] = (VkDescriptorBufferInfo) {.buffer = gbuf->buf, .range = VK_WHOLE_SIZE};
    bi[2] = (VkDescriptorBufferInfo) {.buffer = ubuf->buf, .range = VK_WHOLE_SIZE};
    const struct {
        uint32_t n_in, n_out, blocks_per_row, x_offset, nw_offset, y_offset;
        float    eps;
    } push                        = {n_in, n_out, n_in / 256u, xo, nwo, yo, eps};
    const struct vk_access acc[5] = {vk_acc_tensor(t_x, false),
                                     vk_acc_all(false),
                                     vk_acc_all(false),
                                     vk_acc_tensor(norm_w, false),
                                     vk_acc_tensor(y, true)};
    return vk_seq_dispatch_acc(
            be, VK_PIPE_FFN_NORM_GU, bi, acc, &push, sizeof(push), n_out / 8u, 1, 1);
}

/* Gemma-3n PLE block in THREE dispatches (replaces gate matvec +
 * gelu_mul + proj matvec + rmsnorm_add): the gate GEMV gets the gelu*ple
 * epilogue folded in; the proj tail keeps the multi-workgroup matvec +
 * rmsnorm_add pair. A single-workgroup proj+norm fusion was measured at
 * 68 us vs 19 us for the pair — one SM streaming the 1.5 MB proj weight
 * is a bandwidth wall, so the norm's full-vector reduction stays a
 * separate dispatch. Decode only (rows == 1), F32 weights. */
[[nodiscard]] static enum geist_status vk_ple_block(struct geist_backend      *be,
                                                    const struct geist_tensor *x,
                                                    const struct geist_tensor *gate_w,
                                                    const struct geist_tensor *ple_in,
                                                    const struct geist_tensor *proj_w,
                                                    const struct geist_tensor *res,
                                                    const struct geist_tensor *norm_w,
                                                    float                      eps,
                                                    struct geist_tensor       *gate_scratch,
                                                    struct geist_tensor       *proj_scratch,
                                                    struct geist_tensor       *y) {
    struct vk_state *st = be->state;
    if (!VK_OPS(be, 1u) || x == nullptr || gate_w == nullptr || proj_w == nullptr || x->ndim != 2 ||
        x->shape[0] != 1 || gate_w->dtype != GEIST_DTYPE_F32 || proj_w->dtype != GEIST_DTYPE_F32 ||
        gate_w->ndim != 2 || proj_w->ndim != 2) {
        return GEIST_E_UNSUPPORTED;
    }
    const uint32_t d_in = (uint32_t) gate_w->shape[1];
    const uint32_t hpl  = (uint32_t) gate_w->shape[0];
    const uint32_t feat = (uint32_t) proj_w->shape[0];
    if ((uint32_t) x->shape[1] != d_in || (uint32_t) proj_w->shape[1] != hpl ||
        vk_t_n(norm_w) != feat || vk_t_n(res) < feat || vk_t_n(y) < feat ||
        vk_t_n(gate_scratch) < hpl || vk_t_n(proj_scratch) < feat) {
        return GEIST_E_UNSUPPORTED;
    }
    const uint8_t *g_host = gate_w->buffer != nullptr && gate_w->buffer->host_alias != nullptr
                                    ? (const uint8_t *) gate_w->buffer->host_alias + gate_w->offset
                                    : nullptr;
    const uint8_t *p_host = proj_w->buffer != nullptr && proj_w->buffer->host_alias != nullptr
                                    ? (const uint8_t *) proj_w->buffer->host_alias + proj_w->offset
                                    : nullptr;
    struct geist_buffer   *gwbuf = g_host != nullptr ? vk_weight_lookup(st, g_host) : nullptr;
    struct geist_buffer   *pwbuf = p_host != nullptr ? vk_weight_lookup(st, p_host) : nullptr;
    VkDescriptorBufferInfo b_x, b_ple, b_gs, b_ps;
    uint32_t               xo, po, gso, pso;
    if (gwbuf == nullptr || pwbuf == nullptr || !vk_tensor_gpu(x, &b_x, &xo) ||
        !vk_tensor_gpu(ple_in, &b_ple, &po) || !vk_tensor_gpu(gate_scratch, &b_gs, &gso) ||
        !vk_tensor_gpu(proj_scratch, &b_ps, &pso)) {
        return GEIST_E_UNSUPPORTED;
    }
    {
        const VkDescriptorBufferInfo bi[4] = {
                b_x, {.buffer = gwbuf->buf, .range = VK_WHOLE_SIZE}, b_ple, b_gs};
        const struct {
            uint32_t n_in, x_offset, p_offset, y_offset;
        } push                        = {d_in, xo, po, gso};
        const struct vk_access acc[4] = {vk_acc_tensor(x, false),
                                         vk_acc_all(false),
                                         vk_acc_tensor(ple_in, false),
                                         vk_acc_tensor(gate_scratch, true)};
        enum geist_status      s =
                vk_seq_dispatch_acc(be, VK_PIPE_PLE_GATE, bi, acc, &push, sizeof(push), hpl, 1, 1);
        if (s != GEIST_OK) {
            return s;
        }
    }
    {
        const VkDescriptorBufferInfo bi[3] = {
                b_gs, {.buffer = pwbuf->buf, .range = VK_WHOLE_SIZE}, b_ps};
        const struct vk_push   push   = {.n_in     = hpl,
                                         .n_out    = feat,
                                         .rows     = 1,
                                         .x_offset = gso,
                                         .y_offset = pso,
                                         .x_stride = hpl,
                                         .y_stride = feat};
        const struct vk_access acc[3] = {vk_acc_tensor(gate_scratch, false),
                                         vk_acc_all(false),
                                         vk_acc_tensor(proj_scratch, true)};
        enum geist_status      s      = vk_seq_dispatch_acc(
                be, VK_PIPE_MATVEC_F32, bi, acc, &push, sizeof(push), feat, 1, 1);
        if (s != GEIST_OK) {
            return s;
        }
    }
    return vk_rmsnorm_add(be, res, proj_scratch, norm_w, eps, y);
}

/* Fused q/k/v prep: per-head norms + rope + F32 cache append in ONE
 * dispatch (which-axis on WorkGroupID.z). Falls back (UNSUPPORTED) when
 * the tensors don't share the expected pool/arena buffers. */
[[nodiscard]] static enum geist_status vk_attn_qkv_prep(struct geist_backend      *be,
                                                        struct geist_tensor       *q,
                                                        struct geist_tensor       *k,
                                                        struct geist_tensor       *v,
                                                        const struct geist_tensor *q_norm_w,
                                                        const struct geist_tensor *k_norm_w,
                                                        const struct geist_tensor *v_norm_w,
                                                        const struct geist_tensor *cos,
                                                        const struct geist_tensor *sin,
                                                        float                      eps,
                                                        size_t                     q_position,
                                                        struct geist_tensor       *k_cache,
                                                        struct geist_tensor       *v_cache) {
    if (!VK_OPS(be, 4u) || q == nullptr || q->ndim != 3 || q_norm_w == nullptr || cos == nullptr ||
        sin == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    const bool has_kv = k != nullptr;
    const bool kv16   = has_kv && k_cache != nullptr && k_cache->dtype == GEIST_DTYPE_F16;
    if (has_kv &&
        (v == nullptr || k_norm_w == nullptr || v_norm_w == nullptr || k_cache == nullptr ||
         v_cache == nullptr ||
         (!kv16 && (k_cache->dtype != GEIST_DTYPE_F32 || v_cache->dtype != GEIST_DTYPE_F32)))) {
        return GEIST_E_UNSUPPORTED;
    }
    const uint32_t seq = (uint32_t) q->shape[0];
    const uint32_t qh  = (uint32_t) q->shape[1];
    const uint32_t hd  = (uint32_t) q->shape[2];
    if (hd > 512u || vk_t_n(q) == 0) {
        return GEIST_E_UNSUPPORTED;
    }
    VkDescriptorBufferInfo bi[6];
    uint32_t               qo, ko = 0, vo = 0, qwo, kwo = 0, vwo = 0, co, so, kco = 0, vco = 0;
    VkDescriptorBufferInfo tmp;
    if (!vk_tensor_gpu(q, &bi[0], &qo) || !vk_tensor_gpu(q_norm_w, &bi[1], &qwo) ||
        !vk_tensor_gpu(cos, &bi[2], &co) || !vk_tensor_gpu(sin, &bi[3], &so)) {
        return GEIST_E_UNSUPPORTED;
    }
    uint32_t kvh = 1;
    if (has_kv) {
        VkDescriptorBufferInfo b_k, b_v, b_kw, b_vw;
        const bool             caches_ok = kv16 ? (vk_tensor_gpu_f16(k_cache, &bi[4], &kco) &&
                                                   vk_tensor_gpu_f16(v_cache, &bi[5], &vco))
                                                : (vk_tensor_gpu(k_cache, &bi[4], &kco) &&
                                                   vk_tensor_gpu(v_cache, &bi[5], &vco));
        if (vk_t_n(k) == 0 || vk_t_n(v) == 0 || !vk_tensor_gpu(k, &b_k, &ko) ||
            !vk_tensor_gpu(v, &b_v, &vo) || !vk_tensor_gpu(k_norm_w, &b_kw, &kwo) ||
            !vk_tensor_gpu(v_norm_w, &b_vw, &vwo) || !caches_ok) {
            return GEIST_E_UNSUPPORTED;
        }
        /* q/k/v + ones share the pool buffer; q/k gammas share the arena */
        if (b_k.buffer != bi[0].buffer || b_v.buffer != bi[0].buffer ||
            b_vw.buffer != bi[0].buffer || b_kw.buffer != bi[1].buffer) {
            return GEIST_E_UNSUPPORTED;
        }
        kvh = (uint32_t) k->shape[1];
    } else {
        bi[4] = bi[0]; /* unused bindings — anything valid */
        bi[5] = bi[0];
    }
    (void) tmp;
    const struct {
        uint32_t seq, qh, kvh, hd, q_position, has_kv;
        uint32_t qo, ko, vo, qwo, kwo, vwo, co, so, kco, vco;
        float    eps;
    } push              = {seq,
                           qh,
                           kvh,
                           hd,
                           (uint32_t) q_position,
                           has_kv ? 1u : 0u,
                           qo,
                           ko,
                           vo,
                           qwo,
                           kwo,
                           vwo,
                           co,
                           so,
                           kco,
                           vco,
                           eps};
    struct vk_access a0 = vk_acc_tensor(q, true);
    if (has_kv) {
        const struct vk_access ak = vk_acc_tensor(k, true);
        const struct vk_access av = vk_acc_tensor(v, true);
        a0.lo                     = a0.lo < ak.lo ? a0.lo : ak.lo;
        a0.lo                     = a0.lo < av.lo ? a0.lo : av.lo;
        a0.hi                     = a0.hi > ak.hi ? a0.hi : ak.hi;
        a0.hi                     = a0.hi > av.hi ? a0.hi : av.hi;
    }
    const struct vk_access acc[6] = {
            a0,
            vk_acc_all(false),
            vk_acc_tensor(cos, false),
            vk_acc_tensor(sin, false),
            has_kv ? (kv16 ? vk_acc_tensor16(k_cache, true) : vk_acc_tensor(k_cache, true))
                   : vk_acc(0, 0, false),
            has_kv ? (kv16 ? vk_acc_tensor16(v_cache, true) : vk_acc_tensor(v_cache, true))
                   : vk_acc(0, 0, false),
    };
    return vk_seq_dispatch_acc(be,
                               kv16 ? VK_PIPE_QKV_PREP_F16 : VK_PIPE_QKV_PREP,
                               bi,
                               acc,
                               &push,
                               sizeof(push),
                               seq,
                               qh,
                               3);
}

/* F32 -> F16 converting KV append (enables the F16 cache: GEIST_KV_AUTO
 * upgrades when this slot exists). */
[[nodiscard]] static enum geist_status vk_kv_append_f16(struct geist_backend      *be,
                                                        const struct geist_tensor *k_src,
                                                        const struct geist_tensor *v_src,
                                                        size_t                     q_position,
                                                        struct geist_tensor       *k_cache,
                                                        struct geist_tensor       *v_cache) {
    if (k_src == nullptr || v_src == nullptr || k_cache == nullptr || v_cache == nullptr ||
        k_src->ndim != 3) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t           seq    = (size_t) k_src->shape[0];
    const size_t           kv_row = (size_t) (k_src->shape[1] * k_src->shape[2]);
    const size_t           n      = seq * kv_row;
    VkDescriptorBufferInfo bi[4];
    uint32_t               kso, vso, kdo, vdo;
    if (n == 0 || vk_t_n(k_src) == 0 || vk_t_n(v_src) == 0 || !vk_tensor_gpu(k_src, &bi[0], &kso) ||
        !vk_tensor_gpu(v_src, &bi[1], &vso) || !vk_tensor_gpu_f16(k_cache, &bi[2], &kdo) ||
        !vk_tensor_gpu_f16(v_cache, &bi[3], &vdo)) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "vulkan kv_append_f16: bad inputs");
        return GEIST_E_UNSUPPORTED;
    }
    const uint32_t         push[5] = {(uint32_t) n,
                                      kso,
                                      vso,
                                      kdo + (uint32_t) (q_position * kv_row),
                                      vdo + (uint32_t) (q_position * kv_row)};
    const struct vk_access acc[4]  = {
            vk_acc_tensor(k_src, false),
            vk_acc_tensor(v_src, false),
            vk_acc(k_cache->buffer->base_off + k_cache->offset + q_position * kv_row * 2,
                   n * 2,
                   true),
            vk_acc(v_cache->buffer->base_off + v_cache->offset + q_position * kv_row * 2,
                   n * 2,
                   true)};
    return vk_seq_dispatch_acc(
            be, VK_PIPE_KV_APPEND_F16, bi, acc, push, sizeof(push), vk_groups(n), 1, 1);
}

/* ====================================================================== */
/* Descriptor                                                              */
/* ====================================================================== */

static const struct geist_backend_vtbl vk_vtbl = {
        .create                = vk_create,
        .destroy               = vk_destroy,
        .buffer_create         = vk_buffer_create,
        .buffer_destroy        = vk_buffer_destroy,
        .buffer_create_aliased = vk_buffer_create_aliased,
        .buffer_upload         = vk_buffer_upload,
        .buffer_download       = vk_buffer_download,
        .buffer_map            = vk_buffer_map,
        .buffer_unmap          = vk_buffer_unmap,
#ifndef VK_NO_COPY
        .buffer_copy = vk_buffer_copy,
#endif
        .resolve_weight = vk_resolve_weight,
};

/* Probe pairing for the fused table below. Mirrors the entry checks of
 * vk_ffn_gate_up / vk_ffn_norm_gate_up (decode-only matvec kernels,
 * Q4_K weights, GPU-resident via vk_weight_lookup). */
static bool vk_fused_supported(struct geist_backend *be, const struct geist_fusion_query *q) {
    if (q == nullptr || be == nullptr || be->state == nullptr) {
        return false;
    }
    struct vk_state *st = be->state;
    switch (q->op) {
    case GEIST_FUSED_GELU_TANH_MUL:
    case GEIST_FUSED_GELU_TANH_MUL_SCALED:
        return VK_OPS(be, 1u);
    case GEIST_FUSED_FFN_GATE_UP:
    case GEIST_FUSED_FFN_NORM_GATE_UP: {
        if (!VK_OPS(be, 1u) || q->m != 1 || q->gate_w == nullptr || q->up_w == nullptr ||
            q->gate_w->dtype != GEIST_DTYPE_Q4_K || q->up_w->dtype != GEIST_DTYPE_Q4_K ||
            q->d_model % 256u != 0u ||
            (q->op == GEIST_FUSED_FFN_NORM_GATE_UP && q->inter % 8u != 0u) ||
            (size_t) q->gate_w->n_in != q->d_model || (size_t) q->up_w->n_in != q->d_model ||
            q->gate_w->n_out != q->up_w->n_out) {
            return false;
        }
        /* Residency: both weights must be registered GPU buffers. */
        return q->gate_w->raw != nullptr && q->up_w->raw != nullptr &&
               vk_weight_lookup(st, (const uint8_t *) q->gate_w->raw) != nullptr &&
               vk_weight_lookup(st, (const uint8_t *) q->up_w->raw) != nullptr;
    }
    case GEIST_FUSED_RMSNORM_ADD:
        return VK_OPS(be, 4u);
    case GEIST_FUSED_ARGMAX_F32:
        return VK_OPS(be, 128u);
    case GEIST_FUSED_ATTN_QKV_PREP:
        return VK_OPS(be, 4u) && q->head_dim > 0 && (q->head_dim % 2u) == 0u;
    case GEIST_FUSED_PLE_BLOCK:
        /* vk_ple_block is a decode-only (rows == 1) kernel over F32
         * gate/proj matrices. */
        return VK_OPS(be, 1u) && q->m == 1 && q->gate_w != nullptr && q->up_w != nullptr &&
               q->gate_w->dtype == GEIST_DTYPE_F32 && q->up_w->dtype == GEIST_DTYPE_F32;
    case GEIST_FUSED_EMBEDDING_LOOKUP_SCALED:
        /* Dtype set of vk_embedding_lookup_scaled's dtype_code switch. */
        if (!VK_OPS(be, 64u)) {
            return false;
        }
        switch ((enum geist_dtype) q->table_dtype) {
        case GEIST_DTYPE_F32:
        case GEIST_DTYPE_F16:
        case GEIST_DTYPE_BF16:
        case GEIST_DTYPE_Q4_K:
        case GEIST_DTYPE_Q5_K:
        case GEIST_DTYPE_Q6_K:
            return true;
        default:
            return false;
        }
    default:
        return false;
    }
}

static const struct geist_backend_primitives vk_prims = {
        .rmsnorm          = vk_rmsnorm,
        .add              = vk_add,
        .mul              = vk_mul,
        .gelu_tanh        = vk_gelu_tanh,
        .silu             = vk_silu,
        .relu_squared     = vk_relu_squared,
        .rope_apply       = vk_rope_apply,
        .embedding_lookup = vk_embedding_lookup,
        .attention        = vk_attention,
#ifndef VK_NO_SCALE
        .scale_f32 = vk_scale_f32,
#endif
};

static const struct geist_backend_fused vk_fused = {
        .supported            = vk_fused_supported,
        .gelu_tanh_mul        = vk_gelu_tanh_mul,
        .gelu_tanh_mul_scaled = vk_gelu_tanh_mul_scaled,
/* Phase 3: batched-submit paths — one flush per token (argmax). */
#ifndef VK_NO_LINEAR_T
        .linear_t      = vk_linear_t,
        .linear_t_pair = vk_linear_t_pair,
#endif
#ifndef VK_NO_RMSADD
        .rmsnorm_add = vk_rmsnorm_add,
#endif
#ifndef VK_NO_EMBED
        .embedding_lookup_scaled = vk_embedding_lookup_scaled,
#endif
#ifndef VK_NO_ARGMAX
        .argmax_f32 = vk_argmax_f32,
#endif
        .ffn_gate_up      = vk_ffn_gate_up,
        .ffn_norm_gate_up = vk_ffn_norm_gate_up,
        .ple_block        = vk_ple_block,
        .attn_qkv_prep    = vk_attn_qkv_prep,
        .kv_append_f16    = vk_kv_append_f16,
};

const struct geist_backend_descriptor geist_backend_vulkan = {
        .name  = "vulkan",
        .vtbl  = &vk_vtbl,
        .prims = &vk_prims,
        .fused = &vk_fused,
        .caps  = {.kv_f16_attention           = true,
                  .batched_submit             = true,
                  .weights_need_backend_arena = true,
                  .max_m                      = 512,
                  .preferred_kv_mode          = GEIST_KV_FP32},
};
