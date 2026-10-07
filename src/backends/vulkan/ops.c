/*
 * src/backends/vulkan/ops.c — op implementations, resolver, probe, and the descriptor.
 *
 * Layer: BACKEND (vulkan).
 */
#include "vk_internal.h"

#include "checked.h"
#include "hadamard.h" /* host fallback of hadamard_rotate */

/* Everything the backend knows per weight dtype, in one place (#465): the
 * file block, how the VRAM copy lays it out, the (matvec, matmul) pipeline
 * pair and the embedding gather's dtype code. A dtype not listed has no GPU
 * path at all; one with has_linear false is only gathered (embedding).
 *
 * Layouts of the VRAM copy:
 *   VERBATIM  the file blocks as they are;
 *   PAD216    Q6_K's 210-byte blocks padded to 216 so every field is 4-byte
 *             aligned (VK_Q6K_GPU_BLOCK);
 *   SOA       struct-of-arrays: every block's quant bytes back to back, then
 *             one f16 scale per block (padded to a whole word). Same bytes
 *             per block as the file, but each lane's quant load is
 *             contiguous and word-aligned. See mv_legacy.glsl. scale_first
 *             says where the scale sits in the file block (ggml's TQ2_0
 *             keeps it last);
 *   DENSE     not block-quantized (F32): n_in elements of elem_bytes.
 *
 * block_elems also sets `blocks_per_row` in the push block and the shape
 * rule the resolver enforces (n_in % block_elems == 0); the f32 kernels
 * ignore blocks_per_row, so DENSE keeps 256 there. */
enum vk_layout { VK_LAYOUT_VERBATIM, VK_LAYOUT_PAD216, VK_LAYOUT_SOA, VK_LAYOUT_DENSE };

struct vk_dtype {
    enum geist_dtype dtype;
    uint32_t         block_elems;
    uint32_t         block_bytes; /* file bytes per block; DENSE: bytes per element */
    enum vk_layout   layout;
    bool             scale_first;
    bool             has_linear;
    enum vk_pipe     mv, mm;
    int32_t          embed_code; /* embed_lookup_scaled.comp dtype, -1: none */
};

static const struct vk_dtype vk_dtypes[] = {
        {GEIST_DTYPE_F32,
         256,
         sizeof(float),
         VK_LAYOUT_DENSE,
         false,
         true,
         VK_PIPE_MATVEC_F32,
         VK_PIPE_MATMUL_F32,
         0},
        {GEIST_DTYPE_F16, 1, 2, VK_LAYOUT_DENSE, false, false, VK_PIPE_COUNT, VK_PIPE_COUNT, 1},
        {GEIST_DTYPE_BF16, 1, 2, VK_LAYOUT_DENSE, false, false, VK_PIPE_COUNT, VK_PIPE_COUNT, 2},
        {GEIST_DTYPE_Q4_0,
         Q4_0_BLOCK_ELEMS,
         Q4_0_BLOCK_BYTES,
         VK_LAYOUT_SOA,
         true,
         true,
         VK_PIPE_MATVEC_Q4_0,
         VK_PIPE_MATMUL_Q4_0,
         3},
        {GEIST_DTYPE_Q4_1,
         Q4_1_BLOCK_ELEMS,
         Q4_1_BLOCK_BYTES,
         VK_LAYOUT_VERBATIM,
         false,
         true,
         VK_PIPE_MATVEC_Q4_1,
         VK_PIPE_MATMUL_Q4_1,
         4},
        {GEIST_DTYPE_Q8_0,
         Q8_0_BLOCK_ELEMS,
         Q8_0_BLOCK_BYTES,
         VK_LAYOUT_SOA,
         true,
         true,
         VK_PIPE_MATVEC_Q8_0,
         VK_PIPE_MATMUL_Q8_0,
         5},
        {GEIST_DTYPE_Q4_K,
         Q4_K_BLOCK_ELEMS,
         Q4_K_BLOCK_BYTES,
         VK_LAYOUT_VERBATIM,
         false,
         true,
         VK_PIPE_MATVEC_Q4K,
         VK_PIPE_MATMUL_Q4K,
         8},
        {GEIST_DTYPE_Q5_K,
         Q5_K_BLOCK_ELEMS,
         Q5_K_BLOCK_BYTES,
         VK_LAYOUT_VERBATIM,
         false,
         true,
         VK_PIPE_MATVEC_Q5K,
         VK_PIPE_MATMUL_Q5K,
         9},
        {GEIST_DTYPE_Q6_K,
         Q6_K_BLOCK_ELEMS,
         Q6_K_BLOCK_BYTES,
         VK_LAYOUT_PAD216,
         false,
         true,
         VK_PIPE_MATVEC_Q6K,
         VK_PIPE_MATMUL_Q6K,
         10},
        {GEIST_DTYPE_PQ2_0,
         PQ2_0_BLOCK_ELEMS,
         PQ2_0_BLOCK_BYTES,
         VK_LAYOUT_SOA,
         true,
         true,
         VK_PIPE_MATVEC_PQ2_0,
         VK_PIPE_MATMUL_PQ2_0,
         11},
        {GEIST_DTYPE_TQ2_0,
         TQ2_0_BLOCK_ELEMS,
         TQ2_0_BLOCK_BYTES,
         VK_LAYOUT_SOA,
         false,
         true,
         VK_PIPE_MATVEC_TQ2_0,
         VK_PIPE_MATMUL_TQ2_0,
         -1},
};

/* The table row of `dt`, or nullptr when the backend has no GPU path. */
static const struct vk_dtype *vk_dtype_of(enum geist_dtype dt) {
    for (size_t i = 0; i < sizeof vk_dtypes / sizeof vk_dtypes[0]; i++) {
        if (vk_dtypes[i].dtype == dt) {
            return &vk_dtypes[i];
        }
    }
    return nullptr;
}

/* The table row of `dt` when it has GPU linear kernels, else nullptr. */
static const struct vk_dtype *vk_linear_dtype(enum geist_dtype dt) {
    const struct vk_dtype *d = vk_dtype_of(dt);
    return d != nullptr && d->has_linear ? d : nullptr;
}

[[nodiscard]] static enum geist_status vk_dispatch_linear(struct geist_backend *be,
                                                          enum vk_pipe          pipe,
                                                          struct geist_buffer  *wbuf,
                                                          const float          *x,
                                                          float                *y,
                                                          size_t                m,
                                                          size_t                n_in,
                                                          size_t                n_out,
                                                          size_t                blocks_per_row) {
    struct vk_state *st = be->state;
    size_t           x_bytes, y_bytes;
    uint32_t         n_in32, n_out32, bpr32, m32;
    if (ckd_mul(&x_bytes, m, n_in) || ckd_mul(&x_bytes, x_bytes, sizeof(float)) ||
        ckd_mul(&y_bytes, m, n_out) || ckd_mul(&y_bytes, y_bytes, sizeof(float)) ||
        vk_ckd_u32(n_in, &n_in32) || vk_ckd_u32(n_out, &n_out32) ||
        vk_ckd_u32(blocks_per_row, &bpr32) || vk_ckd_u32(m, &m32)) {
        return vk_too_wide(be, "linear");
    }
    vk_seq_flush(st); /* host x/y round-trip — must not interleave with a batch */
    /* x_stage: the GPU reads it hot (GEMM B tiles), the host only writes —
     * SCRATCH role makes it BAR-eligible. y_stage stays in system RAM
     * (the host reads results back; CPU reads from BAR are uncached). */
    enum geist_status s = vk_stage_reserve_role(be, &st->x_stage, x_bytes, GEIST_BUFFER_SCRATCH);
    if (s == GEIST_OK) {
        s = vk_stage_reserve(be, &st->y_stage, y_bytes);
    }
    if (s != GEIST_OK) {
        return s;
    }
    memcpy(st->x_stage->mapped, x, x_bytes);

    const VkDescriptorBufferInfo binfo[3] = {
            {.buffer = st->x_stage->buf, .range = VK_WHOLE_SIZE},
            {.buffer = wbuf->buf, .range = VK_WHOLE_SIZE},
            {.buffer = st->y_stage->buf, .range = VK_WHOLE_SIZE},
    };
    const struct vk_push push = {.n_in           = n_in32,
                                 .n_out          = n_out32,
                                 .blocks_per_row = bpr32,
                                 .rows           = m32,
                                 .x_stride       = n_in32,
                                 .y_stride       = n_out32};
    enum vk_pipe         eff  = pipe;
    uint32_t             gx   = vk_linear_gx(pipe, n_out32);
    uint32_t             gy   = vk_linear_gy(pipe, m32);
    vk_linear_cm_route(st, &eff, m32, n_out32, &gx, &gy);
    s = vk_seq_dispatch(be, eff, binfo, &push, sizeof(push), gx, gy, 1);
    if (s != GEIST_OK) {
        return s;
    }
    vk_seq_flush(st);
    if (st->seq_failed) {
        /* y_stage holds no result of this dispatch. The flag stays set: the
         * resolved kernels have no status, so the next readback reports it. */
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: a submitted batch failed");
        return GEIST_E_BACKEND;
    }
    memcpy(y, st->y_stage->mapped, y_bytes);
    return GEIST_OK;
}

/* Resolver-installed kernels. The signature has no error path — failures
 * report to stderr and zero y so a defect is loud in the parity gate
 * rather than silent garbage. */
static void vk_linear_run(size_t                     m,
                          const float               *x,
                          const struct geist_weight *w,
                          struct geist_backend      *be,
                          float                     *y) {
    struct vk_state     *st   = be->state;
    struct geist_buffer *wbuf = vk_weight_lookup(st, w->raw);
    const size_t         n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    /* Only dtypes vk_resolve_weight installed these kernels for get here. */
    const struct vk_dtype *ld = vk_linear_dtype((enum geist_dtype) w->dtype);
    if (ld != nullptr && m > 1 && !st->gemm_sg32) {
        /* The register-tiled GEMM shaders hard-assume 32-lane subgroups;
         * on a device that can neither run nor pin them at 32 (lavapipe:
         * 8 lanes only) they compute garbage. The matvec kernels are
         * subgroup-size-agnostic, so loop them: correct everywhere. */
        for (size_t r = 0; r < m; r++) {
            vk_linear_run(1, x + r * n_in, w, be, y + r * n_out);
        }
        return;
    }
    if (ld == nullptr || wbuf == nullptr ||
        vk_dispatch_linear(
                be, m == 1 ? ld->mv : ld->mm, wbuf, x, y, m, n_in, n_out, n_in / ld->block_elems) !=
                GEIST_OK) {
        fprintf(stderr,
                "geist vulkan: linear dispatch failed (%s) — zeroing output\n",
                geist_backend_errmsg(be));
        memset(y, 0, m * n_out * sizeof(float));
    }
}

/* The m == 1 entry of the host-pointer kernels (vk_linear_run is linear_mN). */
static void
vk_w_m1(const float *x, const struct geist_weight *w, struct geist_backend *be, float *y) {
    vk_linear_run(1, x, w, be, y);
}

/* ---- CPU fallback for dtypes without a GPU kernel (F16/BF16/...) -------
 * Row-dequant + naive dot, following cpu_scalar_w_quant_*, so mixed-dtype
 * GGUFs still load. */

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
        dequant_q3_K_row(n_in, base + j * n_in / Q3_K_BLOCK_ELEMS * Q3_K_BLOCK_BYTES, row);
        return true;
    case GEIST_DTYPE_Q5_K:
        dequant_q5_K_row(n_in, base + j * n_in / Q5_K_BLOCK_ELEMS * Q5_K_BLOCK_BYTES, row);
        return true;
    case GEIST_DTYPE_Q8_0:
        dequant_q8_0_row(n_in, base + j * n_in / Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES, row);
        return true;
    case GEIST_DTYPE_TQ2_0:
        dequant_tq2_0_row(n_in, base + j * n_in / TQ2_0_BLOCK_ELEMS * TQ2_0_BLOCK_BYTES, row);
        return true;
    case GEIST_DTYPE_PQ2_0:
        dequant_pq2_0_row(n_in, base + j * n_in / PQ2_0_BLOCK_ELEMS * PQ2_0_BLOCK_BYTES, row);
        return true;
    case GEIST_DTYPE_Q4_0:
        dequant_q4_0_row(n_in, base + j * n_in / Q4_0_BLOCK_ELEMS * Q4_0_BLOCK_BYTES, row);
        return true;
    case GEIST_DTYPE_Q4_1:
        dequant_q4_1_row(n_in, base + j * n_in / Q4_1_BLOCK_ELEMS * Q4_1_BLOCK_BYTES, row);
        return true;
    default:
        return false;
    }
}

static void vk_w_cpu_mN(size_t                     m,
                        const float               *x,
                        const struct geist_weight *w,
                        struct geist_backend      *be,
                        float                     *y) {
    struct vk_state *st    = be->state;
    const size_t     n_in  = (size_t) w->n_in;
    const size_t     n_out = (size_t) w->n_out;
    st->fallbacks[VK_FB_HOST_LINEAR]++;
    if (!st->host_weights_noted) {
        st->host_weights_noted = true;
        fprintf(stderr,
                "geist vulkan: %zu weight(s), %zu MiB, run on the host row-dequant path "
                "(no GPU kernel for their dtype or row length)\n",
                st->host_weights,
                st->host_weight_bytes >> 20);
    }
    /* Row scratch lives in the backend state (grown on demand, freed at
     * destroy): the resolved kernels are allocation-free in steady state, and
     * a failed grow zeroes y and says why instead of leaving it unwritten. */
    if (st->cpu_row_cap < n_in) {
        float *bigger = geist_backend_alloc(be, n_in * sizeof(float), OPTIMAL_ALIGNMENT);
        if (bigger == nullptr) {
            geist_backend_set_error(be, GEIST_E_OOM, "vulkan: row scratch alloc failed");
            memset(y, 0, m * n_out * sizeof(float));
            return;
        }
        geist_backend_free(be, st->cpu_row);
        st->cpu_row     = bigger;
        st->cpu_row_cap = n_in;
    }
    float *row = st->cpu_row;
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
}

static void
vk_w_cpu_m1(const float *x, const struct geist_weight *w, struct geist_backend *be, float *y) {
    vk_w_cpu_mN(1, x, w, be, y);
}

/* Install the host row-dequant kernels on `w`, or refuse them under
 * GEIST_VK_STRICT=1. Counted so the first host linear can say how much of
 * the model left the GPU (#474). */
[[nodiscard]] static enum geist_status vk_resolve_host(struct geist_backend *be,
                                                       struct geist_weight  *w) {
    struct vk_state *st = be->state;
    if (st->strict) {
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "vulkan: GEIST_VK_STRICT=1 and weight dtype %u (%dx%d) has no "
                                "GPU kernel",
                                (unsigned) w->dtype,
                                (int) w->n_out,
                                (int) w->n_in);
        return GEIST_E_BACKEND;
    }
    st->host_weights++;
    st->host_weight_bytes += w->raw_nbytes;
    w->linear_m1 = vk_w_cpu_m1;
    w->linear_mN = vk_w_cpu_mN;
    return GEIST_OK;
}

/* ---- resolve_weight: upload GPU-supported dtypes to VRAM, register,     */
/*      install kernels; CPU fallback for the rest.                        */

/* GPU copies of Q6_K are repacked to 216-byte blocks (210 + 6 pad) so
 * every field sits 4-byte aligned and the kernels use word loads. */
enum { VK_Q6K_GPU_BLOCK = 216 };

/* Bytes of an SOA copy of n_blocks blocks with qbytes quant bytes each:
 * the quants, then the f16 scales padded to a whole word. true on overflow. */
[[nodiscard]] static bool vk_soa_bytes(size_t n_blocks, size_t qbytes, size_t *out) {
    size_t q, sc;
    if (ckd_mul(&q, n_blocks, qbytes) || ckd_mul(&sc, n_blocks, (size_t) 2) ||
        geist_ckd_round_up_pow2(sc, 4, &sc) || ckd_add(out, q, sc)) {
        return true;
    }
    return false;
}

/* Bytes of the VRAM copy, or 0 on overflow / a dtype without a linear. */
[[nodiscard]] static size_t vk_weight_bytes(const struct geist_weight *w) {
    const struct vk_dtype *d = vk_dtype_of((enum geist_dtype) w->dtype);
    if (d == nullptr || !d->has_linear) {
        return 0;
    }
    const size_t n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    size_t       blocks = 0, bytes = 0;
    bool         ovf = false;
    switch (d->layout) {
    case VK_LAYOUT_DENSE:
        ovf = ckd_mul(&blocks, n_out, n_in) || ckd_mul(&bytes, blocks, (size_t) d->block_bytes);
        break;
    case VK_LAYOUT_VERBATIM:
        ovf = ckd_mul(&blocks, n_out, n_in / d->block_elems) ||
              ckd_mul(&bytes, blocks, (size_t) d->block_bytes);
        break;
    case VK_LAYOUT_PAD216:
        ovf = ckd_mul(&blocks, n_out, n_in / d->block_elems) ||
              ckd_mul(&bytes, blocks, (size_t) VK_Q6K_GPU_BLOCK);
        break;
    case VK_LAYOUT_SOA:
        ovf = ckd_mul(&blocks, n_out, n_in / d->block_elems) ||
              vk_soa_bytes(blocks, (size_t) d->block_bytes - 2, &bytes);
        break;
    }
    return ovf ? 0 : bytes;
}

/* Source layout -> GPU layout for the dtypes that are not uploaded verbatim.
 * `bytes` is vk_weight_bytes(w). Returns a heap block of `bytes` bytes
 * (caller frees), or nullptr when the dtype uploads as-is; `*failed` is set
 * when a repack was needed but the allocation failed. */
[[nodiscard]] static uint8_t *
vk_repack_weight(const struct geist_weight *w, size_t bytes, bool *failed) {
    const struct vk_dtype *d   = vk_dtype_of((enum geist_dtype) w->dtype);
    const uint8_t         *src = (const uint8_t *) w->raw;
    *failed                    = false;
    if (d == nullptr || (d->layout != VK_LAYOUT_PAD216 && d->layout != VK_LAYOUT_SOA)) {
        return nullptr;
    }
    const size_t bb = d->block_bytes;
    /* n_out * blocks_per_row, which vk_weight_bytes already proved fits. */
    const size_t n_blocks = (size_t) w->n_out * ((size_t) w->n_in / d->block_elems);
    uint8_t     *packed   = heap_alloc_aligned(bytes, 64);
    if (packed == nullptr) {
        *failed = true;
        return nullptr;
    }
    if (d->layout == VK_LAYOUT_PAD216) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < n_blocks; ++i) {
            memcpy(packed + i * VK_Q6K_GPU_BLOCK, src + i * bb, bb);
            memset(packed + i * VK_Q6K_GPU_BLOCK + bb, 0, VK_Q6K_GPU_BLOCK - bb);
        }
        return packed;
    }
    const size_t qbytes = bb - 2;
    const size_t qskip  = d->scale_first ? 2 : 0;
    const size_t scoff  = d->scale_first ? 0 : qbytes;
    uint8_t     *qs     = packed;
    uint8_t     *sc     = packed + n_blocks * qbytes;
    memset(sc + n_blocks * 2, 0, bytes - n_blocks * qbytes - n_blocks * 2); /* word pad */
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < n_blocks; ++i) {
        memcpy(qs + i * qbytes, src + i * bb + qskip, qbytes);
        memcpy(sc + i * 2, src + i * bb + scoff, 2);
    }
    return packed;
}

/* linear_t stages up to VK_MAX_M rows of x into the device x ring. Check that
 * against this weight's n_in at load, so a batch that cannot fit fails the load
 * instead of turning into a per-call UNSUPPORTED and a silent host linear, and
 * create the ring and the argmax word here, off the hot path (#474). */
[[nodiscard]] static enum geist_status vk_ring_reserve(struct geist_backend *be, size_t n_in) {
    struct vk_state *st = be->state;
    size_t           need;
    if (ckd_mul(&need, (size_t) VK_MAX_M, n_in) || ckd_mul(&need, need, sizeof(float)) ||
        need > VK_XRING_CAP) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "vulkan: n_in=%zu at max_m=%d does not fit the %u MB x ring",
                                n_in,
                                (int) VK_MAX_M,
                                (unsigned) (VK_XRING_CAP >> 20));
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = GEIST_OK;
    if (st->xring == nullptr) {
        s = vk_buffer_create(
                be, VK_XRING_CAP, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE, &st->xring);
    }
    if (s == GEIST_OK && st->argmax_out == nullptr) {
        s = vk_buffer_create(be, 16, GEIST_BUFFER_STAGING, GEIST_MEMORY_AUTO, &st->argmax_out);
    }
    return s;
}

[[nodiscard]] static enum geist_status vk_resolve_weight(struct geist_backend *be,
                                                         struct geist_weight  *w) {
    struct vk_state *st = be->state;
    if (w == nullptr || w->raw == nullptr || w->n_in <= 0 || w->n_out <= 0 || w->raw_nbytes == 0u) {
        return GEIST_E_INVALID_ARG;
    }
    /* vk_weight_bytes(w) below derives the upload size from the shape and
     * reads that many bytes out of `raw`, so a short source is read past
     * its end before it ever reaches the device. */
    if (!quant_weight_extent_ok(w)) {
        return GEIST_E_FORMAT;
    }
    const struct vk_dtype *ld = vk_linear_dtype((enum geist_dtype) w->dtype);
    if (ld == nullptr) {
        switch ((enum geist_dtype) w->dtype) {
        case GEIST_DTYPE_F16:
        case GEIST_DTYPE_BF16:
            /* No half-precision GPU kernel. A small matrix (Bonsai's BF16
             * ssm_alpha / ssm_beta, 48 x 5120) is refused so the arch widens
             * it to F32 at load (weight_load/layer_wiring.c, same rule as
             * metal) and it runs on the device; a large one keeps the host
             * row-dequant path below. Keep the cap in step with
             * widen_max_elems there. */
            if ((size_t) w->n_out * (size_t) w->n_in <= (4u << 20)) {
                return GEIST_E_UNSUPPORTED;
            }
            [[fallthrough]];
        case GEIST_DTYPE_Q3_K:
            return vk_resolve_host(be, w);
        default:
            geist_backend_set_error(be,
                                    GEIST_E_UNSUPPORTED,
                                    "vulkan: no kernel for weight dtype %u (%dx%d)",
                                    (unsigned) w->dtype,
                                    (int) w->n_out,
                                    (int) w->n_in);
            return GEIST_E_UNSUPPORTED;
        }
    }
    if (w->dtype != GEIST_DTYPE_F32 && (size_t) w->n_in % ld->block_elems != 0) {
        /* Row length is not a whole number of blocks: the GPU kernels index
         * by block. Every dtype but the two native k-quants has a CPU dequant
         * row (vk_dequant_row) and keeps working through it. */
        if (w->dtype != GEIST_DTYPE_Q4_K && w->dtype != GEIST_DTYPE_Q6_K) {
            return vk_resolve_host(be, w);
        }
        return GEIST_E_UNSUPPORTED;
    }
    const enum geist_status rs = vk_ring_reserve(be, (size_t) w->n_in);
    if (rs != GEIST_OK) {
        return rs;
    }
    /* Upload to VRAM and register. An existing entry for the same host
     * pointer is REPLACED, not reused: the same address can carry new bytes
     * after a model reload (or a freed+remalloc'd test blob) — the latest
     * resolve is authoritative. Tied weights resolving twice re-upload the
     * same bytes once more at load time; harmless. */
    struct vk_weight_entry *slot  = vk_weight_entry_of(st, w->raw);
    const bool              fresh = slot == nullptr;
    if (fresh) {
        if (st->n_weights == st->cap_weights) {
            const size_t            cap = st->cap_weights == 0 ? 64 : st->cap_weights * 2;
            struct vk_weight_entry *nw =
                    geist_backend_alloc(be, cap * sizeof(*nw), alignof(struct vk_weight_entry));
            if (nw == nullptr) {
                return GEIST_E_OOM;
            }
            if (st->n_weights > 0) { /* weights is nullptr before the first grow */
                memcpy(nw, st->weights, st->n_weights * sizeof(*nw));
            }
            geist_backend_free(be, st->weights);
            st->weights     = nw;
            st->cap_weights = cap;
        }
        slot  = &st->weights[st->n_weights++];
        *slot = (struct vk_weight_entry) {0};
    }
    const size_t bytes = vk_weight_bytes(w);
    if (bytes == 0) {
        geist_backend_set_error(be, GEIST_E_FORMAT, "vulkan: weight size overflows");
        return GEIST_E_FORMAT;
    }
    struct geist_buffer *gpu = nullptr;
    enum geist_status    s =
            vk_buffer_create(be, bytes, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE, &gpu);
    if (s == GEIST_OK) {
        bool     failed = false;
        uint8_t *packed = vk_repack_weight(w, bytes, &failed);
        if (failed) {
            s = GEIST_E_OOM;
        } else if (packed != nullptr) {
            s = vk_buffer_upload(gpu, bytes, packed);
            safe_free((void **) &packed);
        } else {
            s = vk_buffer_upload(gpu, bytes, (const uint8_t *) w->raw);
        }
    }
    if (s != GEIST_OK) {
        if (gpu != nullptr) {
            vk_buffer_destroy(be, gpu);
        }
        if (fresh) {
            st->n_weights--; /* fresh slot never got a buffer — roll back */
        }
        return s;
    }
    if (slot->gpu != nullptr) {
        vk_buffer_destroy(be, slot->gpu);
    }
    *slot = (struct vk_weight_entry) {.host = w->raw, .gpu = gpu};
    if (fresh) {
        s = vk_weight_index_add(be, (size_t) (slot - st->weights));
        if (s != GEIST_OK) {
            vk_buffer_destroy(be, gpu);
            st->n_weights--; /* unindexed: never reachable, roll back */
            geist_backend_set_error(be, GEIST_E_OOM, "vulkan: weight index alloc failed");
            return s;
        }
    }
    w->linear_m1 = vk_w_m1;
    w->linear_mN = vk_linear_run;
    return GEIST_OK;
}

/* 256-lane workgroups covering n (already narrowed through vk_ckd_u32). */
static uint32_t vk_groups(uint32_t n) {
    return n / 256u + (n % 256u != 0u ? 1u : 0u);
}

/* Dispatch geometry of the linear pipes (rows per workgroup in the shader
 * headers). Register-tiled quant GEMMs: 8 output rows x 32 batch rows;
 * matmul_q6k / matmul_f32: 4 x 16. */
uint32_t vk_linear_gx(enum vk_pipe pipe, uint32_t n_out) {
    switch (pipe) {
    case VK_PIPE_MATVEC_Q4K:
    case VK_PIPE_MATVEC_Q6K:
    case VK_PIPE_MATMUL_Q4K:
    case VK_PIPE_MATVEC_Q4_0:
    case VK_PIPE_MATVEC_Q4_1:
    case VK_PIPE_MATVEC_Q8_0:
    case VK_PIPE_MATVEC_Q5K:
    case VK_PIPE_MATVEC_TQ2_0:
    case VK_PIPE_MATMUL_TQ2_0:
    case VK_PIPE_MATMUL_PQ2_0:
    case VK_PIPE_MATMUL_Q4_0:
    case VK_PIPE_MATMUL_Q4_1:
    case VK_PIPE_MATMUL_Q8_0:
    case VK_PIPE_MATMUL_Q5K:
        return (n_out + 7u) / 8u; /* 8 rows per workgroup */
    case VK_PIPE_MATVEC_PQ2_0:
        return (n_out + 31u) / 32u; /* 32 rows (lanes) x 8 k-slices (warps) */
    case VK_PIPE_MATMUL_Q6K:
    case VK_PIPE_MATMUL_F32:
        return (n_out + 3u) / 4u;
    default:
        return n_out;
    }
}

uint32_t vk_linear_gy(enum vk_pipe pipe, uint32_t m) {
    switch (pipe) {
    case VK_PIPE_MATMUL_Q4K:
    case VK_PIPE_MATMUL_Q4_0:
    case VK_PIPE_MATMUL_Q4_1:
    case VK_PIPE_MATMUL_Q8_0:
    case VK_PIPE_MATMUL_Q5K:
    case VK_PIPE_MATMUL_TQ2_0:
    case VK_PIPE_MATMUL_PQ2_0:
        return (m + 31u) / 32u; /* 32 batch rows per workgroup */
    case VK_PIPE_MATMUL_Q6K:
    case VK_PIPE_MATMUL_F32:
        return (m + 15u) / 16u;
    default:
        return m;
    }
}

/* Reroute conforming quant GEMMs onto the tensor-core pipelines. */
void vk_linear_cm_route(struct vk_state *st,
                        enum vk_pipe    *pipe,
                        uint32_t         m,
                        uint32_t         n_out,
                        uint32_t        *gx,
                        uint32_t        *gy) {
    /* The tensor-core tiles keep the native subgroup size (only the tiled
     * GEMMs are pinned to 32), and they were written for 32 lanes. */
    if (st->subgroup_size != 32u) {
        return;
    }
    enum vk_pipe cm;
    if (*pipe == VK_PIPE_MATMUL_Q4K) {
        cm = VK_PIPE_MM_Q4K_CM;
    } else if (*pipe == VK_PIPE_MATMUL_Q6K) {
        cm = VK_PIPE_MM_Q6K_CM;
    } else if (*pipe == VK_PIPE_MATMUL_PQ2_0) {
        /* 128-token tile from a full tile of tokens up; the 128 x 64 tile below
         * it (a 128-wide tile would run half empty at the default chunk of 64) */
        cm = m < 128u ? VK_PIPE_MM_PQ2_0_CM64
                      : (st->pq2_f32_acc ? VK_PIPE_MM_PQ2_0_CM_F32 : VK_PIPE_MM_PQ2_0_CM);
    } else {
        return;
    }
    /* PQ2_0 tiles cover 128 weight rows and 128 (or 64) tokens, the k-quant
     * tiles 64 x 64 */
    const bool     pq2       = cm == VK_PIPE_MM_PQ2_0_CM || cm == VK_PIPE_MM_PQ2_0_CM_F32 ||
                               cm == VK_PIPE_MM_PQ2_0_CM64;
    const uint32_t tile_rows = pq2 ? 128u : 64u;
    const uint32_t tile_toks = cm == VK_PIPE_MM_PQ2_0_CM64 ? 64u : tile_rows;
    if ((m & 15u) != 0 || n_out % tile_rows != 0 || st->pipes[cm] == VK_NULL_HANDLE) {
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
    *gx   = n_out / tile_rows;
    *gy   = (m + tile_toks - 1u) / tile_toks;
}

/* GPU-first attempt for the 3-buffer elementwise family (add, mul, gelu_mul,
 * silu_mul, sigmoid_mul):
 * push {n, a_off, b_off, y_off, cols, a_stride, b_stride, y_stride};
 * cols == 0 → all-contiguous fast path in the shader.
 * GEIST_E_UNSUPPORTED: the shader does not apply (take the host path);
 * any other error is a failed dispatch the caller returns (#474). */
[[nodiscard]] static enum geist_status vk_try_ew3(struct geist_backend      *be,
                                                  enum vk_pipe               pipe,
                                                  const struct geist_tensor *a,
                                                  const struct geist_tensor *b,
                                                  const struct geist_tensor *y) {
    const size_t n = vk_t_n(a);
    if (n == 0 || n != vk_t_n(b) || n != vk_t_n(y)) {
        return GEIST_E_UNSUPPORTED;
    }
    size_t ra, ca, sa, rb, cb, sb, ry, cy, sy;
    if (!vk_t_geom(a, &ra, &ca, &sa) || !vk_t_geom(b, &rb, &cb, &sb) ||
        !vk_t_geom(y, &ry, &cy, &sy)) {
        return GEIST_E_UNSUPPORTED;
    }
    size_t cols = 0;
    if (sa != ca || sb != cb || sy != cy) {
        /* mixed contiguous/strided operands: unify on the strided cols */
        cols = sa != ca ? ca : (sb != cb ? cb : cy);
        if ((sa != ca && ca != cols) || (sb != cb && cb != cols) || (sy != cy && cy != cols) ||
            n % cols != 0) {
            return GEIST_E_UNSUPPORTED;
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
        return GEIST_E_UNSUPPORTED;
    }
    uint32_t push[8] = {0, off[0], off[1], off[2]};
    if (vk_ckd_u32(n, &push[0]) || vk_ckd_u32(cols, &push[4]) || vk_ckd_u32(sa, &push[5]) ||
        vk_ckd_u32(sb, &push[6]) || vk_ckd_u32(sy, &push[7])) {
        return vk_too_wide(be, "elementwise");
    }
    const struct vk_access acc[3] = {
            vk_acc_tensor(a, false), vk_acc_tensor(b, false), vk_acc_tensor(y, true)};
    return vk_seq_dispatch_acc(be, pipe, bi, acc, push, sizeof(push), vk_groups(push[0]), 1, 1);
}

/* GPU-first attempt for the unary elementwise family (gelu_tanh, silu,
 * relu_squared): push {n, x_off, y_off, 0}; in place is fine. Results as
 * vk_try_ew3. */
[[nodiscard]] static enum geist_status vk_try_ew2(struct geist_backend      *be,
                                                  enum vk_pipe               pipe,
                                                  const struct geist_tensor *x,
                                                  const struct geist_tensor *y) {
    const size_t           n = vk_t_n(x);
    VkDescriptorBufferInfo bi[2];
    uint32_t               off[2];
    if (n == 0 || n != vk_t_n(y) || !vk_tensor_gpu(x, &bi[0], &off[0]) ||
        !vk_tensor_gpu(y, &bi[1], &off[1])) {
        return GEIST_E_UNSUPPORTED;
    }
    uint32_t push[4] = {0, off[0], off[1], 0};
    if (vk_ckd_u32(n, &push[0])) {
        return vk_too_wide(be, "elementwise");
    }
    const struct vk_access acc[2] = {vk_acc_tensor(x, false), vk_acc_tensor(y, true)};
    return vk_seq_dispatch_acc(be, pipe, bi, acc, push, sizeof(push), vk_groups(push[0]), 1, 1);
}

/* The 3-buffer elementwise ops (vk_try_ew3 above, vk_ew3_cpu below). */
enum vk_ew3_op { EW3_ADD, EW3_MUL, EW3_GELU_MUL, EW3_SILU_MUL, EW3_SIGMOID_MUL };

/* Strided-aware CPU fallback core for the 3-buffer elementwise ops. */
[[nodiscard]] static enum geist_status vk_ew3_cpu(struct geist_backend      *be,
                                                  enum vk_ew3_op             op,
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
        if (op == EW3_GELU_MUL) {
            gelu_tanh_mul_fp32(cc, arow, brow, yrow);
            continue;
        }
        for (size_t i = 0; i < cc; i++) {
            switch (op) {
            case EW3_ADD:
                yrow[i] = arow[i] + brow[i];
                break;
            case EW3_MUL:
                yrow[i] = arow[i] * brow[i];
                break;
            case EW3_SILU_MUL: {
                const float v = arow[i];
                const float e = expf(-fabsf(v));
                yrow[i]       = ((v >= 0.0f) ? v / (1.0f + e) : (v * e) / (1.0f + e)) * brow[i];
                break;
            }
            case EW3_SIGMOID_MUL:
                yrow[i] = arow[i] * (1.0f / (1.0f + expf(-brow[i])));
                break;
            case EW3_GELU_MUL: /* handled per row above */
                break;
            }
        }
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_add(struct geist_backend      *be,
                                              const struct geist_tensor *a,
                                              const struct geist_tensor *b,
                                              struct geist_tensor       *y) {
    const enum geist_status gs = vk_try_ew3(be, VK_PIPE_ADD, a, b, y);
    if (gs != GEIST_E_UNSUPPORTED) {
        return gs;
    }
    return vk_ew3_cpu(be, EW3_ADD, a, b, y, "add");
}

[[nodiscard]] static enum geist_status vk_mul(struct geist_backend      *be,
                                              const struct geist_tensor *a,
                                              const struct geist_tensor *b,
                                              struct geist_tensor       *y) {
    const enum geist_status gs = vk_try_ew3(be, VK_PIPE_MUL, a, b, y);
    if (gs != GEIST_E_UNSUPPORTED) {
        return gs;
    }
    return vk_ew3_cpu(be, EW3_MUL, a, b, y, "mul");
}

/* Host fallback of the unary elementwise ops: `fn` is the shared reference
 * kernel (gemma4_kernels.h) the CPU backends run. */
[[nodiscard]] static enum geist_status vk_ew2_cpu(struct geist_backend *be,
                                                  const char           *name,
                                                  void (*fn)(size_t n, const float *x, float *y),
                                                  const struct geist_tensor *x,
                                                  struct geist_tensor       *y) {
    size_t       nx = 0, ny = 0;
    const float *xp = vk_tensor_host(x, &nx);
    float       *yp = vk_tensor_host(y, &ny);
    if (xp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan %s: bad inputs", name);
        return GEIST_E_INVALID_ARG;
    }
    fn(nx, xp, yp);
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
vk_gelu_tanh(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y) {
    const enum geist_status gs = vk_try_ew2(be, VK_PIPE_GELU, x, y);
    if (gs != GEIST_E_UNSUPPORTED) {
        return gs;
    }
    return vk_ew2_cpu(be, "gelu_tanh", gelu_tanh_fp32, x, y);
}

[[nodiscard]] static enum geist_status vk_gelu_tanh_mul(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        const struct geist_tensor *z,
                                                        struct geist_tensor       *y) {
    const enum geist_status gs = vk_try_ew3(be, VK_PIPE_GELU_MUL, x, z, y);
    if (gs != GEIST_E_UNSUPPORTED) {
        return gs;
    }
    return vk_ew3_cpu(be, EW3_GELU_MUL, x, z, y, "gelu_tanh_mul");
}

[[nodiscard]] static enum geist_status
vk_relu_squared(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y) {
    const enum geist_status gs = vk_try_ew2(be, VK_PIPE_RELU2, x, y);
    if (gs != GEIST_E_UNSUPPORTED) {
        return gs;
    }
    return vk_ew2_cpu(be, "relu_squared", relu_squared_fp32, x, y);
}

[[nodiscard]] static enum geist_status
vk_silu(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y) {
    const enum geist_status gs = vk_try_ew2(be, VK_PIPE_SILU, x, y);
    if (gs != GEIST_E_UNSUPPORTED) {
        return gs;
    }
    return vk_ew2_cpu(be, "silu", silu_fp32_ooo, x, y);
}

[[nodiscard]] static enum geist_status vk_silu_mul(struct geist_backend      *be,
                                                   const struct geist_tensor *x,
                                                   const struct geist_tensor *z,
                                                   struct geist_tensor       *y) {
    const enum geist_status gs = vk_try_ew3(be, VK_PIPE_SILU_MUL, x, z, y);
    if (gs != GEIST_E_UNSUPPORTED) {
        return gs;
    }
    return vk_ew3_cpu(be, EW3_SILU_MUL, x, z, y, "silu_mul");
}

[[nodiscard]] static enum geist_status vk_sigmoid_mul(struct geist_backend      *be,
                                                      const struct geist_tensor *x,
                                                      const struct geist_tensor *gate,
                                                      struct geist_tensor       *y) {
    const enum geist_status gs = vk_try_ew3(be, VK_PIPE_SIGMOID_MUL, x, gate, y);
    if (gs != GEIST_E_UNSUPPORTED) {
        return gs;
    }
    return vk_ew3_cpu(be, EW3_SIGMOID_MUL, x, gate, y, "sigmoid_mul");
}

/* qwen35's joint [query | gate] projection split on the device. UNSUPPORTED
 * (nothing dispatched) leaves the architecture's mapped host gather. */
[[nodiscard]] static enum geist_status vk_attn_qgate_split(struct geist_backend      *be,
                                                           const struct geist_tensor *joint,
                                                           size_t                     heads,
                                                           size_t                     head_dim,
                                                           struct geist_tensor       *q,
                                                           struct geist_tensor       *gate) {
    if (joint == nullptr || q == nullptr || gate == nullptr || heads == 0 || head_dim == 0 ||
        joint->ndim != 2 || q->ndim != 2 || gate->ndim != 2) {
        return GEIST_E_UNSUPPORTED;
    }
    const size_t rows = (size_t) joint->shape[0];
    if (vk_t_n(joint) == 0 || (size_t) joint->shape[1] != 2 * heads * head_dim ||
        (size_t) q->shape[0] != rows || (size_t) q->shape[1] != heads * head_dim ||
        (size_t) gate->shape[0] != rows || (size_t) gate->shape[1] != heads * head_dim ||
        vk_t_n(q) == 0 || vk_t_n(gate) == 0 || joint->stride[1] != 1 || q->stride[1] != 1 ||
        gate->stride[1] != 1) {
        return GEIST_E_UNSUPPORTED;
    }
    VkDescriptorBufferInfo bi[3];
    uint32_t               oj, oq, og;
    if (!vk_tensor_gpu(joint, &bi[0], &oj) || !vk_tensor_gpu(q, &bi[1], &oq) ||
        !vk_tensor_gpu(gate, &bi[2], &og)) {
        return GEIST_E_UNSUPPORTED;
    }
    uint32_t n, push[9] = {0, 0, 0, oj, oq, og};
    if (vk_ckd_u32(rows * heads * head_dim, &n) || vk_ckd_u32(rows, &push[0]) ||
        vk_ckd_u32(heads, &push[1]) || vk_ckd_u32(head_dim, &push[2]) ||
        vk_ckd_u32((size_t) joint->stride[0], &push[6]) ||
        vk_ckd_u32((size_t) q->stride[0], &push[7]) ||
        vk_ckd_u32((size_t) gate->stride[0], &push[8])) {
        return vk_too_wide(be, "attn_qgate_split");
    }
    const struct vk_access acc[3] = {
            vk_acc_tensor(joint, false), vk_acc_tensor(q, true), vk_acc_tensor(gate, true)};
    return vk_seq_dispatch_acc(
            be, VK_PIPE_QGATE_SPLIT, bi, acc, push, sizeof(push), vk_groups(n), 1, 1);
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
        if (feat != 0 && n % feat == 0 && vk_t_n(w) == feat && vk_t_n(y) == n &&
            vk_tensor_gpu(x, &bi[0], &off[0]) && vk_tensor_gpu(w, &bi[1], &off[1]) &&
            vk_tensor_gpu(y, &bi[2], &off[2])) {
            struct {
                uint32_t rows, feat, x, w, y;
                float    eps;
            } push = {0, 0, off[0], off[1], off[2], eps};
            if (vk_ckd_u32(n / feat, &push.rows) || vk_ckd_u32(feat, &push.feat)) {
                return vk_too_wide(be, "rmsnorm");
            }
            const struct vk_access acc[3] = {
                    vk_acc_tensor(x, false), vk_acc_tensor(w, false), vk_acc_tensor(y, true)};
            return vk_seq_dispatch_acc(
                    be, VK_PIPE_RMSNORM, bi, acc, &push, sizeof(push), push.rows, 1, 1);
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
        /* cos/sin rows are `rot` wide: the rotated prefix of each head. */
        const size_t rot =
                cos != nullptr && cos->ndim >= 1 ? (size_t) cos->shape[cos->ndim - 1] : 0;
        if (x != nullptr && x->ndim == 3 && vk_t_n(x) != 0 && vk_t_n(cos) != 0 &&
            vk_t_n(sin) != 0 && rot != 0 && rot % 2 == 0 && rot <= (size_t) x->shape[2] &&
            vk_tensor_gpu(x, &bi[0], &off[0]) && vk_tensor_gpu(cos, &bi[1], &off[1]) &&
            vk_tensor_gpu(sin, &bi[2], &off[2])) {
            const size_t seq     = (size_t) x->shape[0];
            const size_t heads   = (size_t) x->shape[1];
            const size_t hd      = (size_t) x->shape[2];
            uint32_t     push[7] = {0, 0, 0, off[0], off[1], off[2]};
            if (vk_ckd_u32(seq * heads * (rot / 2), &push[0]) || vk_ckd_u32(heads, &push[1]) ||
                vk_ckd_u32(hd, &push[2]) || vk_ckd_u32(rot, &push[6])) {
                return vk_too_wide(be, "rope_apply");
            }
            const struct vk_access acc[3] = {
                    vk_acc_tensor(x, true), vk_acc_tensor(cos, false), vk_acc_tensor(sin, false)};
            return vk_seq_dispatch_acc(
                    be, VK_PIPE_ROPE, bi, acc, push, sizeof(push), vk_groups(push[0]), 1, 1);
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
    /* The cos/sin row width is the rotated width (see rope_apply). */
    rope_apply((size_t) x->shape[0],
               (size_t) x->shape[1],
               (size_t) x->shape[2],
               (size_t) cos->shape[cos->ndim - 1],
               xp,
               cosp,
               sinp);
    return GEIST_OK;
}

/* BitNet activation fake-quant on the device rows (see act_quant_i8_f32.comp).
 * x [rows, n] F32, in place. */
[[nodiscard]] static enum geist_status vk_bitnet_act_quant(struct geist_backend *be,
                                                           struct geist_tensor  *x) {
    /* The pipeline layouts start at two bindings: the second one aliases the
     * first and the shader never reads it. */
    VkDescriptorBufferInfo bi[2];
    uint32_t               off[1];
    /* Rows matter here (one absmax per row), so the geometry is read from the
     * shape: vk_t_geom folds a dense tensor into a single row. */
    if (x == nullptr || vk_t_n(x) == 0 || x->ndim != 2 || x->stride[1] != 1 ||
        x->stride[0] < x->shape[1] || !vk_tensor_gpu(x, &bi[0], &off[0])) {
        return GEIST_E_UNSUPPORTED;
    }
    const size_t rows = (size_t) x->shape[0], cols = (size_t) x->shape[1],
                 stride = (size_t) x->stride[0];
    bi[1]               = bi[0];
    uint32_t push[4]    = {0, off[0], 0, 0}, rows32;
    if (vk_ckd_u32(cols, &push[0]) || vk_ckd_u32(stride, &push[2]) || vk_ckd_u32(rows, &rows32)) {
        return vk_too_wide(be, "bitnet_act_quant");
    }
    const struct vk_access acc[2] = {vk_acc_tensor(x, true), vk_acc_tensor(x, true)};
    return vk_seq_dispatch_acc(be, VK_PIPE_ACT_QUANT, bi, acc, push, sizeof(push), rows32, 1, 1);
}

/* Largest Hadamard block the shader stages in shared memory. */
enum { VK_HADAMARD_MAX_BLOCK = 1024 };

/* fused->hadamard_rotate (Ternary-Bonsai's rotated weight basis). The op is not
 * optional — a model that needs it refuses to load without it — so a geometry
 * the shader does not cover (block > 1024, host-side tensors) runs the shared
 * host implementation on mapped memory instead of failing. */
[[nodiscard]] static enum geist_status vk_hadamard_rotate(struct geist_backend             *be,
                                                          const struct geist_hadamard_args *args) {
    if (args == nullptr || args->x == nullptr || args->y == nullptr || args->x->ndim != 2) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan hadamard_rotate: bad args");
        return GEIST_E_INVALID_ARG;
    }
    const size_t rows = (size_t) args->x->shape[0], width = (size_t) args->x->shape[1];
    const size_t block = args->block, rep = args->perm_rep;
    const bool   permute = rep > 1;
    size_t       pn      = 0;
    const bool   geometry_ok =
            block >= 2 && (block & (block - 1)) == 0 && width != 0 && width % block == 0 &&
            (!permute || (!args->inverse && !ckd_mul(&pn, args->perm_hd, args->perm_nk) &&
                          !ckd_mul(&pn, pn, rep) && pn == width && args->perm_hd != 0));
    VkDescriptorBufferInfo bi[3];
    uint32_t               off[3];
    const bool             has_signs = args->signs != nullptr;
    if (geometry_ok && block <= VK_HADAMARD_MAX_BLOCK && rows != 0 &&
        vk_t_n(args->x) == rows * width && vk_t_n(args->y) == rows * width &&
        vk_tensor_gpu(args->x, &bi[0], &off[0]) && vk_tensor_gpu(args->y, &bi[2], &off[2]) &&
        (!has_signs ||
         (vk_t_n(args->signs) == width && vk_tensor_gpu(args->signs, &bi[1], &off[1])))) {
        if (!has_signs) {
            bi[1]  = bi[0]; /* bound but never read (flags bit 1 clear) */
            off[1] = 0;
        }
        const float scale    = 1.0f / sqrtf((float) block);
        uint32_t    push[11] = {0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                (args->inverse ? 1u : 0u) | (has_signs ? 2u : 0u),
                                off[0],
                                off[1],
                                off[2],
                                0};
        uint32_t    groups;
        if (vk_ckd_u32(width, &push[0]) || vk_ckd_u32(block, &push[1]) ||
            vk_ckd_u32(width / block, &push[2]) || vk_ckd_u32(args->perm_hd, &push[3]) ||
            vk_ckd_u32(args->perm_nk, &push[4]) || vk_ckd_u32(rep, &push[5]) ||
            vk_ckd_u32(rows * (width / block), &groups)) {
            return vk_too_wide(be, "hadamard_rotate");
        }
        memcpy(&push[10], &scale, sizeof scale);
        const struct vk_access acc[3] = {vk_acc_tensor(args->x, false),
                                         has_signs ? vk_acc_tensor(args->signs, false)
                                                   : vk_acc_tensor(args->x, false),
                                         vk_acc_tensor(args->y, true)};
        return vk_seq_dispatch_acc(be, VK_PIPE_HADAMARD, bi, acc, push, sizeof(push), groups, 1, 1);
    }
    size_t                  nx = 0, ns = 0, ny = 0;
    const float            *xp = vk_tensor_host(args->x, &nx);
    const float            *sp = has_signs ? vk_tensor_host(args->signs, &ns) : nullptr;
    float                  *yp = vk_tensor_host(args->y, &ny);
    const enum geist_status s  = xp == nullptr || yp == nullptr || (has_signs && sp == nullptr)
                                         ? GEIST_E_INVALID_ARG
                                         : geist_hadamard_apply(args, nx, ns, ny, xp, sp, yp);
    if (s != GEIST_OK) {
        geist_backend_set_error(be, s, "vulkan hadamard_rotate: bad inputs");
    }
    return s;
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
    if (q != nullptr && k != nullptr && v != nullptr && out != nullptr && q->ndim == 3 &&
        q->shape[0] == 1 && k->dtype == GEIST_DTYPE_F16 && (size_t) k->shape[0] > 192 &&
        q->shape[2] <= 512 && vk_t_n(q) != 0 && vk_t_n16(k) != 0 &&
        stt->pipes[VK_PIPE_ATTN_PART_F16] != VK_NULL_HANDLE) {
        uint32_t qh, hd, n_kv, kvh, qpos, win;
        if (vk_ckd_u32((size_t) q->shape[1], &qh) || vk_ckd_u32((size_t) q->shape[2], &hd) ||
            vk_ckd_u32((size_t) k->shape[0], &n_kv) || vk_ckd_u32((size_t) k->shape[1], &kvh) ||
            vk_ckd_u32(q_offset, &qpos) || vk_ckd_u32(sliding_window, &win)) {
            return vk_too_wide(be, "attention");
        }
        const uint32_t         n_chunks   = n_kv / 128u + (n_kv % 128u != 0u ? 1u : 0u);
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
        uint32_t po; /* < VK_XRING_CAP / 4; checked all the same */
        if (vk_ckd_u32(stt->xring_used / 4u, &po)) {
            return vk_too_wide(be, "attention");
        }
        const uint32_t         push1[11] = {n_kv, qh, kvh, hd, qpos, win, qo, ko, vo, po, n_chunks};
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
        return vk_seq_dispatch_acc(
                be, VK_PIPE_ATTN_COMB, bi2, acc2, push2, sizeof(push2), qh, 1, 1);
    }
attn_generic:;
    {
        VkDescriptorBufferInfo bi[4];
        uint32_t               off[4];
        const bool             kv16 = k != nullptr && k->dtype == GEIST_DTYPE_F16;
        if (q != nullptr && k != nullptr && q->ndim == 3 && k->ndim == 3 && vk_t_n(q) != 0 &&
            (kv16 ? (vk_t_n16(k) != 0 && vk_t_n16(v) != 0 &&
                     vk_tensor_gpu_f16(k, &bi[1], &off[1]) && vk_tensor_gpu_f16(v, &bi[2], &off[2]))
                  : (vk_t_n(k) != 0 && vk_t_n(v) != 0 && vk_tensor_gpu(k, &bi[1], &off[1]) &&
                     vk_tensor_gpu(v, &bi[2], &off[2]))) &&
            vk_t_n(out) != 0 && vk_tensor_gpu(q, &bi[0], &off[0]) &&
            vk_tensor_gpu(out, &bi[3], &off[3]) &&
            /* f16 kernel streams q/k/v as 4-wide vectors */
            (!kv16 || ((off[0] | off[1] | off[2]) & 3u) == 0u)) {
            uint32_t n_q, qh, hd, n_kv, kvh, qpos, win;
            if (vk_ckd_u32((size_t) q->shape[0], &n_q) || vk_ckd_u32((size_t) q->shape[1], &qh) ||
                vk_ckd_u32((size_t) q->shape[2], &hd) || vk_ckd_u32((size_t) k->shape[0], &n_kv) ||
                vk_ckd_u32((size_t) k->shape[1], &kvh) || vk_ckd_u32(q_offset, &qpos) ||
                vk_ckd_u32(sliding_window, &win)) {
                return vk_too_wide(be, "attention");
            }
            const uint32_t push[11] = {
                    n_q, n_kv, qh, kvh, hd, qpos, win, off[0], off[1], off[2], off[3]};
            const struct vk_access acc[4] = {
                    vk_acc_tensor(q, false),
                    kv16 ? vk_acc_tensor16(k, false) : vk_acc_tensor(k, false),
                    kv16 ? vk_acc_tensor16(v, false) : vk_acc_tensor(v, false),
                    vk_acc_tensor(out, true)};
            /* Tensor-core kernel: prefill only (n_q > 1; decode has the
             * attn_part_f16/attn_comb path above), one variant per head_dim
             * (128: Qwen3; 256: qwen35/Bonsai, Gemma's local layers; 512:
             * Gemma 4's global layers, two column halves in z). Same push
             * layout and bindings as VK_PIPE_ATTENTION_F16, sliding window
             * included, just a 16-row dispatch. */
            const enum vk_pipe cm_pipe = hd == 128   ? VK_PIPE_ATTENTION_F16_HD128_CM
                                         : hd == 256 ? VK_PIPE_ATTENTION_F16_CM
                                         : hd == 512 ? VK_PIPE_ATTENTION_F16_HD512_CM
                                                     : VK_PIPE_COUNT;
            if (kv16 && n_q > 1 && cm_pipe != VK_PIPE_COUNT &&
                stt->pipes[cm_pipe] != VK_NULL_HANDLE) {
                return vk_seq_dispatch_acc(be,
                                           cm_pipe,
                                           bi,
                                           acc,
                                           push,
                                           sizeof(push),
                                           n_q / 16u + (n_q % 16u != 0u ? 1u : 0u),
                                           qh,
                                           hd == 512 ? 2u : 1u);
            }
            return vk_seq_dispatch_acc(be,
                                       kv16 ? VK_PIPE_ATTENTION_F16 : VK_PIPE_ATTENTION,
                                       bi,
                                       acc,
                                       push,
                                       sizeof(push),
                                       n_q,
                                       qh,
                                       1);
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
    attention_mqa_causal_kv((size_t) q->shape[0],
                            (size_t) k->shape[0],
                            q_offset,
                            (size_t) q->shape[1],
                            (size_t) k->shape[1],
                            (size_t) q->shape[2],
                            sliding_window,
                            qp,
                            kp,
                            vp,
                            op);
    return GEIST_OK;
}

/* ---- Batched-submit ops ----------------------------------------------- */

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
        if (feat != 0 && n % feat == 0 && vk_t_n(w) == feat && vk_t_n(res) == n && vk_t_n(y) == n &&
            vk_tensor_gpu(x, &bi[0], &off[0]) && vk_tensor_gpu(w, &bi[1], &off[1]) &&
            vk_tensor_gpu(res, &bi[2], &off[2]) && vk_tensor_gpu(y, &bi[3], &off[3])) {
            struct {
                uint32_t rows, feat, x, w, r, y;
                float    eps;
            } push = {0, 0, off[0], off[1], off[2], off[3], eps};
            if (vk_ckd_u32(n / feat, &push.rows) || vk_ckd_u32(feat, &push.feat)) {
                return vk_too_wide(be, "rmsnorm_add");
            }
            const struct vk_access acc[4] = {vk_acc_tensor(x, false),
                                             vk_acc_tensor(w, false),
                                             vk_acc_tensor(res, false),
                                             vk_acc_tensor(y, true)};
            return vk_seq_dispatch_acc(
                    be, VK_PIPE_RMSNORM_ADD, bi, acc, &push, sizeof(push), push.rows, 1, 1);
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
            struct {
                uint32_t n, x, y;
                float    scale;
            } push = {0, off[0], off[1], scale};
            if (vk_ckd_u32(n, &push.n)) {
                return vk_too_wide(be, "scale_f32");
            }
            const struct vk_access acc[2] = {vk_acc_tensor(x, false), vk_acc_tensor(y, true)};
            return vk_seq_dispatch_acc(
                    be, VK_PIPE_SCALE, bi, acc, &push, sizeof(push), vk_groups(push.n), 1, 1);
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
    struct vk_state       *st = be->state;
    const size_t           n  = vk_t_n(logits);
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
    uint32_t push[3] = {0, off[0], 0};
    if (vk_ckd_u32(n, &push[0])) {
        return vk_too_wide(be, "argmax");
    }
    const struct vk_access acc[2] = {vk_acc_tensor(logits, false), vk_acc_all(true)};
    enum geist_status      s =
            vk_seq_dispatch_acc(be, VK_PIPE_ARGMAX, bi, acc, push, sizeof(push), 1, 1, 1);
    if (s != GEIST_OK) {
        return s; /* a failed dispatch is an error, not a host scan */
    }
    vk_seq_flush(st); /* the one intended sync point per decoded token */
    if (vk_seq_take_failure(st) != GEIST_OK) {
        return GEIST_E_BACKEND; /* the staging word is not this token's argmax */
    }
    *out_index = ((const int32_t *) st->argmax_out->mapped)[0];
    return GEIST_OK;
}

/* Tensor-path linear: x staged into the VRAM ring, weight from the VRAM
 * registry, y written to its host-visible home — no host round-trip, no
 * flush. The hot path. */
[[nodiscard]] static enum geist_status vk_linear_t(struct geist_backend      *be,
                                                   const struct geist_tensor *t_x,
                                                   const struct geist_weight *w,
                                                   const struct geist_tensor *t_w,
                                                   size_t                     m,
                                                   struct geist_tensor       *t_y) {
    (void) t_w;
    struct vk_state       *st = be->state;
    const struct vk_dtype *ld = vk_linear_dtype((enum geist_dtype) w->dtype);
    if (ld == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    const enum vk_pipe     mv = ld->mv, mm = ld->mm;
    struct geist_buffer   *wbuf = vk_weight_lookup(st, w->raw);
    VkDescriptorBufferInfo bi[3];
    uint32_t               xo, yo;
    if (wbuf == nullptr || m == 0 || vk_t_n(t_x) == 0 || vk_t_n(t_y) == 0 ||
        !vk_tensor_gpu(t_y, &bi[2], &yo)) {
        return GEIST_E_UNSUPPORTED;
    }
    uint32_t n_in, n_out, m32, x_stride, y_stride;
    if (vk_ckd_u32((size_t) w->n_in, &n_in) || vk_ckd_u32((size_t) w->n_out, &n_out) ||
        vk_ckd_u32(m, &m32) ||
        vk_ckd_u32(t_y->ndim >= 2 ? (size_t) t_y->stride[t_y->ndim - 2] : n_out, &y_stride)) {
        return vk_too_wide(be, "linear_t");
    }
    if (t_x->buffer->device_mem) {
        /* BAR-resident activations: bind in place, no staging copy. */
        if (!vk_tensor_gpu(t_x, &bi[0], &xo)) {
            return GEIST_E_UNSUPPORTED;
        }
        if (vk_ckd_u32(t_x->ndim >= 2 ? (size_t) t_x->stride[t_x->ndim - 2] : n_in, &x_stride)) {
            return vk_too_wide(be, "linear_t");
        }
    } else {
        if (!vk_xring_stage(be, t_x, m, n_in, &xo)) {
            return GEIST_E_UNSUPPORTED;
        }
        bi[0]    = (VkDescriptorBufferInfo) {.buffer = st->xring->buf, .range = VK_WHOLE_SIZE};
        x_stride = n_in; /* ring copy is contiguous */
    }
    bi[1] = (VkDescriptorBufferInfo) {.buffer = wbuf->buf, .range = VK_WHOLE_SIZE};
    const struct vk_push push = {.n_in           = n_in,
                                 .n_out          = n_out,
                                 .blocks_per_row = n_in / ld->block_elems,
                                 .rows           = m32,
                                 .x_offset       = xo,
                                 .y_offset       = yo,
                                 .x_stride       = x_stride,
                                 .y_stride       = y_stride};
    if (m > 1 && !st->gemm_sg32) {
        /* The register-tiled GEMMs assume 32-lane subgroups (see
         * vk_linear_run). Elsewhere run the size-agnostic matvec once per
         * batch row: correct on any device. The last row's offsets bound
         * every row's. */
        uint32_t last;
        if (vk_ckd_u32((size_t) xo + (size_t) (m32 - 1u) * x_stride, &last) ||
            vk_ckd_u32((size_t) yo + (size_t) (m32 - 1u) * y_stride, &last)) {
            return vk_too_wide(be, "linear_t");
        }
        for (uint32_t r = 0; r < m32; ++r) {
            struct vk_push row             = push;
            row.rows                       = 1;
            row.x_offset                   = push.x_offset + r * push.x_stride;
            row.y_offset                   = push.y_offset + r * push.y_stride;
            const struct vk_access racc[3] = {
                    vk_acc((uint64_t) row.x_offset * 4u, (uint64_t) n_in * 4u, false),
                    vk_acc_all(false),
                    vk_acc((uint64_t) row.y_offset * 4u, (uint64_t) n_out * 4u, true)};
            const enum geist_status rs = vk_seq_dispatch_acc(
                    be, mv, bi, racc, &row, sizeof(row), vk_linear_gx(mv, n_out), 1, 1);
            if (rs != GEIST_OK) {
                return rs;
            }
        }
        return GEIST_OK;
    }
    enum vk_pipe lpipe = m == 1 ? mv : mm;
    uint32_t     gx    = vk_linear_gx(lpipe, n_out);
    uint32_t     gy    = vk_linear_gy(lpipe, m32);
    /* Tensor-core path for conforming GEMMs (shaders assume w_offset == 0,
     * which holds for all registry uploads). */
    if (m > 1) {
        vk_linear_cm_route(st, &lpipe, m32, n_out, &gx, &gy);
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
    /* Two appended dispatches. Check both weights up front so the fallback
     * never sees a half-done pair. */
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
    const struct vk_dtype *ed = vk_dtype_of((enum geist_dtype) embed_table->dtype);
    if (ed == nullptr || ed->embed_code < 0) {
        return GEIST_E_UNSUPPORTED;
    }
    const uint32_t dtype_code = (uint32_t) ed->embed_code;
    /* Table bytes: prefer the resolve-time VRAM copy (embed tables go
     * through resolve_weight); fall back to a bindable host region. */
    const uint8_t *host =
            embed_table->buffer->host_alias != nullptr
                    ? (const uint8_t *) embed_table->buffer->host_alias + embed_table->offset
                    : nullptr;
    struct geist_buffer *wbuf = host != nullptr ? vk_weight_lookup(st, host) : nullptr;
    if (wbuf == nullptr && host != nullptr && embed_table->buffer->bytes > embed_table->offset &&
        vk_linear_dtype((enum geist_dtype) embed_table->dtype) != nullptr) {
        /* An untied table (separate output.weight) is never resolved by the
         * arch layer: register it now, so the repacked dtypes can be read.
         * Only dtypes with a GPU copy: a host-path table is not a weight
         * that left the GPU, and strict mode must not refuse it here. */
        struct geist_weight ew = {.raw        = host,
                                  .raw_nbytes = embed_table->buffer->bytes - embed_table->offset,
                                  .n_in       = (int32_t) d,
                                  .n_out      = (int32_t) vocab,
                                  .dtype      = (uint16_t) embed_table->dtype};
        if (vk_resolve_weight(be, &ew) == GEIST_OK) {
            wbuf = vk_weight_lookup(st, host);
        }
    }
    VkDescriptorBufferInfo bi[2];
    uint32_t               w_elem_off = 0;
    if (wbuf != nullptr) {
        bi[0] = (VkDescriptorBufferInfo) {.buffer = wbuf->buf, .range = VK_WHOLE_SIZE};
    } else {
        if (ed->layout == VK_LAYOUT_PAD216 || ed->layout == VK_LAYOUT_SOA) {
            /* The shader reads the repacked GPU layout; the arena holds the
             * file layout. */
            return GEIST_E_UNSUPPORTED;
        }
        if (embed_table->buffer->buf == VK_NULL_HANDLE ||
            (embed_table->buffer->base_off + embed_table->offset) % 4 != 0) {
            return GEIST_E_UNSUPPORTED;
        }
        bi[0] = (VkDescriptorBufferInfo) {.buffer = embed_table->buffer->buf,
                                          .range  = VK_WHOLE_SIZE};
        if (vk_ckd_u32(embed_table->buffer->base_off + embed_table->offset, &w_elem_off)) {
            return vk_too_wide(be, "embedding_lookup_scaled");
        }
    }
    uint32_t yo;
    if (vk_t_n(out) != (size_t) d || !vk_tensor_gpu(out, &bi[1], &yo)) {
        return GEIST_E_UNSUPPORTED;
    }
    /* Blocks per row (0 for the non-block dtypes: F32/F16/BF16). */
    struct {
        uint32_t n_in, token, dtype, bpr, w_byte, y;
        float    scale;
        uint32_t n_rows;
    } push = {0, 0, dtype_code, 0, w_elem_off, yo, scale, 0};
    /* 0 <= token_id < vocab was checked above. */
    if (vk_ckd_u32((size_t) d, &push.n_in) || vk_ckd_u32((size_t) token_id, &push.token) ||
        vk_ckd_u32((size_t) vocab, &push.n_rows)) {
        return vk_too_wide(be, "embedding_lookup_scaled");
    }
    if (ed->layout != VK_LAYOUT_DENSE) {
        push.bpr = push.n_in / ed->block_elems;
    }
    const struct vk_access acc[2] = {vk_acc_all(false), vk_acc_tensor(out, true)};
    return vk_seq_dispatch_acc(
            be, VK_PIPE_EMBED, bi, acc, &push, sizeof(push), vk_groups(push.n_in), 1, 1);
}

/* Geometry the decode gate/up kernels run, shared by the probe and both
 * entries: Q4_K gate and up of one shape, whole 256-element blocks, sizes
 * that fit the push constants; the norm variant tiles 8 outputs per
 * workgroup. */
static bool vk_ffn_gate_up_geometry_ok(bool             with_norm,
                                       enum geist_dtype gate_dtype,
                                       enum geist_dtype up_dtype,
                                       int64_t          n_in,
                                       int64_t          n_out,
                                       int64_t          up_n_in,
                                       int64_t          up_n_out) {
    return gate_dtype == GEIST_DTYPE_Q4_K && up_dtype == GEIST_DTYPE_Q4_K && n_in > 0 &&
           n_out > 0 && n_in <= UINT32_MAX && n_out <= UINT32_MAX && n_in % 256 == 0 &&
           (!with_norm || n_out % 8 == 0) && up_n_in == n_in && up_n_out == n_out;
}

/* Fused decode FFN front (m == 1, both weights Q4_K): one dispatch for
 * gelu(x.gate^T) * (x.up^T) — replaces two matvecs + gelu_mul. */
[[nodiscard]] static enum geist_status vk_ffn_gate_up(struct geist_backend      *be,
                                                      const struct geist_tensor *t_x,
                                                      const struct geist_tensor *gate_w,
                                                      const struct geist_tensor *up_w,
                                                      struct geist_tensor       *y) {
    struct vk_state *st = be->state;
    if (gate_w == nullptr || up_w == nullptr || t_x == nullptr || gate_w->ndim != 2 ||
        up_w->ndim != 2 || t_x->shape[0] != 1 ||
        !vk_ffn_gate_up_geometry_ok(false,
                                    gate_w->dtype,
                                    up_w->dtype,
                                    gate_w->shape[1],
                                    gate_w->shape[0],
                                    up_w->shape[1],
                                    up_w->shape[0])) {
        return GEIST_E_UNSUPPORTED;
    }
    uint32_t n_out, n_in; /* in range per vk_ffn_gate_up_geometry_ok; checked all the same */
    if (vk_ckd_u32((size_t) gate_w->shape[0], &n_out) ||
        vk_ckd_u32((size_t) gate_w->shape[1], &n_in)) {
        return vk_too_wide(be, "ffn_gate_up");
    }
    struct geist_buffer   *gbuf = vk_weight_of(st, gate_w);
    struct geist_buffer   *ubuf = vk_weight_of(st, up_w);
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
    if (gate_w == nullptr || up_w == nullptr || t_x == nullptr || norm_w == nullptr ||
        gate_w->ndim != 2 || up_w->ndim != 2 || t_x->shape[0] != 1 ||
        !vk_ffn_gate_up_geometry_ok(true,
                                    gate_w->dtype,
                                    up_w->dtype,
                                    gate_w->shape[1],
                                    gate_w->shape[0],
                                    up_w->shape[1],
                                    up_w->shape[0])) {
        return GEIST_E_UNSUPPORTED;
    }
    uint32_t n_out, n_in; /* in range per vk_ffn_gate_up_geometry_ok; checked all the same */
    if (vk_ckd_u32((size_t) gate_w->shape[0], &n_out) ||
        vk_ckd_u32((size_t) gate_w->shape[1], &n_in)) {
        return vk_too_wide(be, "ffn_norm_gate_up");
    }
    if (vk_t_n(norm_w) != n_in) {
        return GEIST_E_UNSUPPORTED;
    }
    struct geist_buffer   *gbuf = vk_weight_of(st, gate_w);
    struct geist_buffer   *ubuf = vk_weight_of(st, up_w);
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
 * rmsnorm_add pair: a single-workgroup proj+norm fusion is bandwidth-bound
 * (one SM streams the 1.5 MB proj weight; 68 us vs 19 us). Decode only
 * (rows == 1), F32 weights. */
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
    if (x == nullptr || gate_w == nullptr || proj_w == nullptr || x->ndim != 2 ||
        x->shape[0] != 1 || gate_w->dtype != GEIST_DTYPE_F32 || proj_w->dtype != GEIST_DTYPE_F32 ||
        gate_w->ndim != 2 || proj_w->ndim != 2) {
        return GEIST_E_UNSUPPORTED;
    }
    uint32_t d_in, hpl, feat;
    if (vk_ckd_u32((size_t) gate_w->shape[1], &d_in) ||
        vk_ckd_u32((size_t) gate_w->shape[0], &hpl) ||
        vk_ckd_u32((size_t) proj_w->shape[0], &feat)) {
        return vk_too_wide(be, "ple_block");
    }
    if (x->shape[1] != gate_w->shape[1] || proj_w->shape[1] != gate_w->shape[0] ||
        vk_t_n(norm_w) != feat || vk_t_n(res) < feat || vk_t_n(y) < feat ||
        vk_t_n(gate_scratch) < hpl || vk_t_n(proj_scratch) < feat) {
        return GEIST_E_UNSUPPORTED;
    }
    struct geist_buffer   *gwbuf = vk_weight_of(st, gate_w);
    struct geist_buffer   *pwbuf = vk_weight_of(st, proj_w);
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

/* sh[512] in qkv_prep_f{16,32}.comp: one head row in shared memory. */
static constexpr uint32_t VK_QKV_PREP_MAX_HEAD_DIM = 512;

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
    if (q == nullptr || q->ndim != 3 || q_norm_w == nullptr || cos == nullptr || sin == nullptr) {
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
    uint32_t seq, qh, hd, qpos;
    if (vk_ckd_u32((size_t) q->shape[0], &seq) || vk_ckd_u32((size_t) q->shape[1], &qh) ||
        vk_ckd_u32((size_t) q->shape[2], &hd) || vk_ckd_u32(q_position, &qpos)) {
        return vk_too_wide(be, "attn_qkv_prep");
    }
    if (hd == 0u || (hd % 2u) != 0u || hd > VK_QKV_PREP_MAX_HEAD_DIM || vk_t_n(q) == 0) {
        return GEIST_E_UNSUPPORTED;
    }
    VkDescriptorBufferInfo bi[6];
    uint32_t               qo, ko = 0, vo = 0, qwo, kwo = 0, vwo = 0, co, so, kco = 0, vco = 0;
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
        if (vk_ckd_u32((size_t) k->shape[1], &kvh)) {
            return vk_too_wide(be, "attn_qkv_prep");
        }
    } else {
        bi[4] = bi[0]; /* unused bindings — anything valid */
        bi[5] = bi[0];
    }
    const struct {
        uint32_t seq, qh, kvh, hd, q_position, has_kv;
        uint32_t qo, ko, vo, qwo, kwo, vwo, co, so, kco, vco;
        float    eps;
    } push              = {seq,
                           qh,
                           kvh,
                           hd,
                           qpos,
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
    /* The cache row this append starts at, in f16 elements past the views. */
    size_t   pos_elems;
    uint32_t push[5] = {0, kso, vso, 0, 0};
    if (ckd_mul(&pos_elems, q_position, kv_row) || vk_ckd_u32(n, &push[0]) ||
        vk_ckd_u32((size_t) kdo + pos_elems, &push[3]) ||
        vk_ckd_u32((size_t) vdo + pos_elems, &push[4])) {
        return vk_too_wide(be, "kv_append_f16");
    }
    const struct vk_access acc[4] = {
            vk_acc_tensor(k_src, false),
            vk_acc_tensor(v_src, false),
            vk_acc(k_cache->buffer->base_off + k_cache->offset + q_position * kv_row * 2,
                   n * 2,
                   true),
            vk_acc(v_cache->buffer->base_off + v_cache->offset + q_position * kv_row * 2,
                   n * 2,
                   true)};
    return vk_seq_dispatch_acc(
            be, VK_PIPE_KV_APPEND_F16, bi, acc, push, sizeof(push), vk_groups(push[0]), 1, 1);
}

/* Gated-DeltaNet mixer on the device: causal conv + silu, then the delta-rule
 * recurrence with the gated per-head RMSNorm (layer_deltanet.c is the host
 * oracle). Two dispatches per call; both recurrent-state tensors advance
 * exactly once per input row. UNSUPPORTED (before anything was dispatched)
 * lets the architecture take its host path — which cannot see VRAM-resident
 * state, so the geometry limits below cover every published qwen35 variant. */
/* The geometry vk_deltanet_mix runs: the shaders cover d_k <= 256 (q/k
 * staging), d_v <= 128 (one thread per value column) and a conv of 2..8
 * taps, with every row index in uint32. The probe and the op both ask
 * this (#470). */
static bool
vk_deltanet_geometry_ok(size_t seq, size_t n_kh, size_t n_vh, size_t dk, size_t dv, size_t K) {
    if (seq == 0 || n_kh == 0 || n_vh == 0 || n_vh % n_kh != 0 || dk == 0 || dk > 256 || dv == 0 ||
        dv > 128 || K < 2 || K > 8) {
        return false;
    }
    size_t keyd = 0, vald = 0, convd = 0;
    return !ckd_mul(&keyd, n_kh, dk) && !ckd_mul(&vald, n_vh, dv) && !ckd_mul(&convd, keyd, 2) &&
           !ckd_add(&convd, convd, vald) && convd <= UINT32_MAX / 4u &&
           seq <= UINT32_MAX / (convd + 1u);
}

[[nodiscard]] static enum geist_status vk_deltanet_mix(struct geist_backend                 *be,
                                                       const struct geist_deltanet_mix_args *a) {
    if (a == nullptr || a->qkv == nullptr || a->z == nullptr || a->beta == nullptr ||
        a->alpha == nullptr || a->conv_w == nullptr || a->ssm_a == nullptr ||
        a->dt_bias == nullptr || a->norm_w == nullptr || a->conv_state == nullptr ||
        a->delta_state == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    const size_t seq = a->seq, n_kh = a->n_k_heads, n_vh = a->n_v_heads;
    const size_t dk = a->head_k, dv = a->head_v, K = a->conv_kernel;
    if (!vk_deltanet_geometry_ok(seq, n_kh, n_vh, dk, dv, K)) {
        return GEIST_E_UNSUPPORTED;
    }
    const size_t keyd  = n_kh * dk;
    const size_t vald  = n_vh * dv;
    const size_t convd = 2 * keyd + vald;
    /* Row-major contiguous views only — the shaders index rows by convd/vald. */
    if (vk_t_n(a->qkv) != seq * convd || vk_t_n(a->z) < seq * vald ||
        vk_t_n(a->beta) < seq * n_vh || vk_t_n(a->alpha) < seq * n_vh ||
        vk_t_n(a->conv_w) != convd * K || vk_t_n(a->ssm_a) != n_vh || vk_t_n(a->dt_bias) != n_vh ||
        vk_t_n(a->norm_w) != dv || vk_t_n(a->conv_state) != (K - 1) * convd ||
        vk_t_n(a->delta_state) != n_vh * dk * dv) {
        return GEIST_E_UNSUPPORTED;
    }
    VkDescriptorBufferInfo bqkv, bz, bb, ba, bw, bA, bdt, bnw, bcs, bs;
    uint32_t               oqkv, oz, ob, oa, ow, oA, odt, onw, ocs, os;
    if (!vk_tensor_gpu(a->qkv, &bqkv, &oqkv) || !vk_tensor_gpu(a->z, &bz, &oz) ||
        !vk_tensor_gpu(a->beta, &bb, &ob) || !vk_tensor_gpu(a->alpha, &ba, &oa) ||
        !vk_tensor_gpu(a->conv_w, &bw, &ow) || !vk_tensor_gpu(a->ssm_a, &bA, &oA) ||
        !vk_tensor_gpu(a->dt_bias, &bdt, &odt) || !vk_tensor_gpu(a->norm_w, &bnw, &onw) ||
        !vk_tensor_gpu(a->conv_state, &bcs, &ocs) || !vk_tensor_gpu(a->delta_state, &bs, &os)) {
        return GEIST_E_UNSUPPORTED;
    }
    /* In range per vk_deltanet_geometry_ok; checked all the same. */
    uint32_t seq32, n_kh32, n_vh32, dk32, dv32, keyd32, convd32, K32;
    if (vk_ckd_u32(seq, &seq32) || vk_ckd_u32(n_kh, &n_kh32) || vk_ckd_u32(n_vh, &n_vh32) ||
        vk_ckd_u32(dk, &dk32) || vk_ckd_u32(dv, &dv32) || vk_ckd_u32(keyd, &keyd32) ||
        vk_ckd_u32(convd, &convd32) || vk_ckd_u32(K, &K32)) {
        return vk_too_wide(be, "deltanet_mix");
    }
    {
        const VkDescriptorBufferInfo bi[3]   = {bqkv, bw, bcs};
        const uint32_t               push[6] = {seq32, convd32, K32, oqkv, ow, ocs};
        const struct vk_access       acc[3]  = {vk_acc_tensor(a->qkv, true),
                                                vk_acc_tensor(a->conv_w, false),
                                                vk_acc_tensor(a->conv_state, true)};
        const enum geist_status      s =
                vk_seq_dispatch_acc(be,
                                    VK_PIPE_DN_CONV,
                                    bi,
                                    acc,
                                    push,
                                    sizeof(push),
                                    convd32 / 128u + (convd32 % 128u != 0u ? 1u : 0u),
                                    1,
                                    1);
        if (s != GEIST_OK) {
            return s;
        }
    }
    const VkDescriptorBufferInfo bi[8] = {bqkv, bz, bb, ba, bA, bdt, bnw, bs};
    const struct {
        uint32_t seq, n_kh, n_vh, dk, dv, keyd, convd;
        uint32_t qkv_off, z_off, beta_off, alpha_off, a_off, dtb_off, nw_off, s_off;
        float    eps, qscale;
    } push                        = {seq32,
                                     n_kh32,
                                     n_vh32,
                                     dk32,
                                     dv32,
                                     keyd32,
                                     convd32,
                                     oqkv,
                                     oz,
                                     ob,
                                     oa,
                                     oA,
                                     odt,
                                     onw,
                                     os,
                                     a->eps,
                                     1.0f / sqrtf((float) dk)};
    const struct vk_access acc[8] = {vk_acc_tensor(a->qkv, false),
                                     vk_acc_tensor(a->z, true),
                                     vk_acc_tensor(a->beta, false),
                                     vk_acc_tensor(a->alpha, false),
                                     vk_acc_tensor(a->ssm_a, false),
                                     vk_acc_tensor(a->dt_bias, false),
                                     vk_acc_tensor(a->norm_w, false),
                                     vk_acc_tensor(a->delta_state, true)};
    return vk_seq_dispatch_acc(be, VK_PIPE_DN_DELTA, bi, acc, &push, sizeof(push), n_vh32, 1, 1);
}

/* ====================================================================== */
/* Descriptor                                                              */
/* ====================================================================== */

static const struct geist_backend_vtbl vk_vtbl = {
        .create                = vk_create,
        .destroy               = vk_destroy,
        .buffer_create         = vk_buffer_create_api,
        .buffer_destroy        = vk_buffer_destroy,
        .buffer_create_aliased = vk_buffer_create_aliased,
        .buffer_create_view    = vk_buffer_create_view,
        .buffer_upload         = vk_buffer_upload,
        .buffer_download       = vk_buffer_download,
        .buffer_map            = vk_buffer_map,
        .buffer_unmap          = vk_buffer_unmap,
        .buffer_copy           = vk_buffer_copy,
        .resolve_weight        = vk_resolve_weight,
        .fast_host_bytes       = vk_fast_host_bytes,
};

/* Probe pairing for the fused table below: a yes means the bound entry
 * runs on the GPU and succeeds. Geometry comes from the same predicate the
 * entry checks (vk_ffn_gate_up_geometry_ok, vk_deltanet_geometry_ok); a
 * host-only op answers no. */
static bool vk_fused_supported(struct geist_backend *be, const struct geist_fusion_query *q) {
    if (q == nullptr || be == nullptr || be->state == nullptr) {
        return false;
    }
    struct vk_state *st = be->state;
    switch (q->op) {
    case GEIST_FUSED_GELU_TANH_MUL:
        return true;
    case GEIST_FUSED_SILU_MUL:
    case GEIST_FUSED_BITNET_ACT_QUANT:
        return true;
    case GEIST_FUSED_FFN_GATE_UP:
    case GEIST_FUSED_FFN_NORM_GATE_UP: {
        if (q->m != 1 || q->gate_w == nullptr || q->up_w == nullptr ||
            (size_t) q->gate_w->n_in != q->d_model ||
            !vk_ffn_gate_up_geometry_ok(q->op == GEIST_FUSED_FFN_NORM_GATE_UP,
                                        (enum geist_dtype) q->gate_w->dtype,
                                        (enum geist_dtype) q->up_w->dtype,
                                        q->gate_w->n_in,
                                        q->gate_w->n_out,
                                        q->up_w->n_in,
                                        q->up_w->n_out)) {
            return false;
        }
        /* Residency: both weights must be registered GPU buffers. */
        return q->gate_w->raw != nullptr && q->up_w->raw != nullptr &&
               vk_weight_lookup(st, (const uint8_t *) q->gate_w->raw) != nullptr &&
               vk_weight_lookup(st, (const uint8_t *) q->up_w->raw) != nullptr;
    }
    case GEIST_FUSED_RMSNORM_ADD:
        return true;
    case GEIST_FUSED_ARGMAX_F32:
        return true;
    case GEIST_FUSED_DELTANET_MIX:
        return vk_deltanet_geometry_ok(q->m,
                                       q->dn_n_k_heads,
                                       q->dn_n_v_heads,
                                       q->dn_head_k,
                                       q->dn_head_v,
                                       q->dn_conv_kernel);
    case GEIST_FUSED_HADAMARD_ROTATE:
        /* The shader covers block <= VK_HADAMARD_MAX_BLOCK; every other
         * geometry the host transform accepts runs on mapped memory. */
        return geist_hadamard_query_ok(q);
    case GEIST_FUSED_ATTN_QKV_PREP:
        return q->head_dim > 0 && (q->head_dim % 2u) == 0u &&
               q->head_dim <= VK_QKV_PREP_MAX_HEAD_DIM;
    case GEIST_FUSED_PLE_BLOCK:
        /* vk_ple_block is a decode-only (rows == 1) kernel over F32
         * gate/proj matrices. */
        return q->m == 1 && q->gate_w != nullptr && q->up_w != nullptr &&
               q->gate_w->dtype == GEIST_DTYPE_F32 && q->up_w->dtype == GEIST_DTYPE_F32;
    case GEIST_FUSED_EMBEDDING_LOOKUP_SCALED: {
        const struct vk_dtype *ed = vk_dtype_of((enum geist_dtype) q->table_dtype);
        return ed != nullptr && ed->embed_code >= 0;
    }
    default:
        return false;
    }
}

/* ---- Fused-op entry points: a decline is a fallback (#474).            */
/* The arch takes the host path on GEIST_E_UNSUPPORTED; vk_fallback counts */
/* it per site and, under GEIST_VK_STRICT=1, turns it into an error.      */

static enum geist_status
vk_declined(struct geist_backend *be, enum geist_status s, enum vk_fb site) {
    return s == GEIST_E_UNSUPPORTED ? vk_fallback(be->state, site) : s;
}

[[nodiscard]] static enum geist_status vk_fb_linear_t(struct geist_backend      *be,
                                                      const struct geist_tensor *t_x,
                                                      const struct geist_weight *w,
                                                      const struct geist_tensor *t_w,
                                                      size_t                     m,
                                                      struct geist_tensor       *t_y) {
    return vk_declined(be, vk_linear_t(be, t_x, w, t_w, m, t_y), VK_FB_LINEAR_T);
}

[[nodiscard]] static enum geist_status vk_fb_linear_t_pair(struct geist_backend      *be,
                                                           const struct geist_tensor *t_x,
                                                           const struct geist_weight *w0,
                                                           const struct geist_tensor *t_w0,
                                                           const struct geist_weight *w1,
                                                           const struct geist_tensor *t_w1,
                                                           size_t                     m,
                                                           struct geist_tensor       *t_y0,
                                                           struct geist_tensor       *t_y1) {
    return vk_declined(
            be, vk_linear_t_pair(be, t_x, w0, t_w0, w1, t_w1, m, t_y0, t_y1), VK_FB_LINEAR_T);
}

[[nodiscard]] static enum geist_status
vk_fb_embedding_lookup_scaled(struct geist_backend      *be,
                              const struct geist_tensor *embed_table,
                              geist_token_t              token_id,
                              float                      scale,
                              struct geist_tensor       *out) {
    return vk_declined(
            be, vk_embedding_lookup_scaled(be, embed_table, token_id, scale, out), VK_FB_EMBED);
}

[[nodiscard]] static enum geist_status
vk_fb_argmax_f32(struct geist_backend *be, const struct geist_tensor *logits, int32_t *out_index) {
    return vk_declined(be, vk_argmax_f32(be, logits, out_index), VK_FB_ARGMAX);
}

[[nodiscard]] static enum geist_status vk_fb_kv_append_f16(struct geist_backend      *be,
                                                           const struct geist_tensor *k_src,
                                                           const struct geist_tensor *v_src,
                                                           size_t                     q_position,
                                                           struct geist_tensor       *k_cache,
                                                           struct geist_tensor       *v_cache) {
    return vk_declined(
            be, vk_kv_append_f16(be, k_src, v_src, q_position, k_cache, v_cache), VK_FB_KV_APPEND);
}

[[nodiscard]] static enum geist_status vk_fb_attn_qgate_split(struct geist_backend      *be,
                                                              const struct geist_tensor *joint,
                                                              size_t                     heads,
                                                              size_t                     head_dim,
                                                              struct geist_tensor       *q,
                                                              struct geist_tensor       *gate) {
    return vk_declined(be, vk_attn_qgate_split(be, joint, heads, head_dim, q, gate), VK_FB_QGATE);
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
        .scale_f32        = vk_scale_f32,
};

static const struct geist_backend_fused vk_fused = {
        .supported     = vk_fused_supported,
        .gelu_tanh_mul = vk_gelu_tanh_mul,
        /* Batched-submit paths: one flush per token (argmax). */
        .linear_t                = vk_fb_linear_t,
        .linear_t_pair           = vk_fb_linear_t_pair,
        .rmsnorm_add             = vk_rmsnorm_add,
        .embedding_lookup_scaled = vk_fb_embedding_lookup_scaled,
        .argmax_f32              = vk_fb_argmax_f32,
        .ffn_gate_up             = vk_ffn_gate_up,
        .ffn_norm_gate_up        = vk_ffn_norm_gate_up,
        .ple_block               = vk_ple_block,
        .attn_qkv_prep           = vk_attn_qkv_prep,
        .kv_append_f16           = vk_fb_kv_append_f16,
        .deltanet_mix            = vk_deltanet_mix,
        .attn_qgate_split        = vk_fb_attn_qgate_split,
        .sigmoid_mul             = vk_sigmoid_mul,
        .silu_mul                = vk_silu_mul,
        .bitnet_act_quant        = vk_bitnet_act_quant,
        .hadamard_rotate         = vk_hadamard_rotate,
};

const struct geist_backend_descriptor geist_backend_vulkan = {
        .name        = "vulkan",
        .memory_info = vk_memory_info,
        .vtbl        = &vk_vtbl,
        .prims       = &vk_prims,
        .fused       = &vk_fused,
        .caps        = {.kv_f16_attention           = true,
                        .batched_submit             = true,
                        .weights_need_backend_arena = true,
                        .weights_device_copy        = true,
                        .max_m                      = VK_MAX_M,
                        /* the DeltaNet mixer is sequential over tokens: its cost does not
                         * grow with the chunk, so GEIST_M_MAX above 64 is not capped for
                         * qwen35 hybrids. The default chunk stays 64: 128 makes the
                         * scratch pool spill out of a 256 MB BAR heap (#488) */
                 /* 128 where the scratch pool fits the BAR window (the PQ2_0
                  * tensor-core tile is 128 tokens wide); the arch lowers it to
                  * 64 when it does not (vtbl->fast_host_bytes) */
                 .preferred_m_max   = 128,
                 .dn_subchunk       = true,
                 .preferred_kv_mode = GEIST_KV_FP32},
};
