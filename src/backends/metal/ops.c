/*
 * src/backends/metal/ops.c — op implementations, encode helpers, probe, and the descriptor.
 */
#include <geist_util.h>
#include "metal_internal.h"

#include "checked.h"
#include "hadamard.h"

/* Submission of one op. Inside a command sequence this thread owns, the op
 * encodes into the sequence's encoder and leaves the submit to the
 * sequence; otherwise it runs on its own command buffer, committed and
 * waited for before the op returns. The caller holds
 * metal_standalone_pool for the standalone buffer's autoreleased objects.
 * `what` names the op in the backend error; nullptr sets none.
 * metal_op_begin/_end are always inlined, so the sequenced path costs what
 * the open-coded checks did; the standalone path waits for the GPU anyway. */

/* A standalone command buffer (*out_cmd) and its compute encoder; nullptr
 * when either fails. */
static inline void *
metal_standalone_begin(struct geist_backend *be, const char *what, void **out_cmd) {
    struct metal_state *st  = be->state;
    void               *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    *out_cmd  = cmd;
    if (enc == nullptr && what != nullptr) {
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                cmd == nullptr ? "metal %s: command buffer failed"
                                               : "metal %s: encoder failed",
                                what);
    }
    return enc;
}

/* Ends, commits and waits for a standalone command buffer. */
[[nodiscard]] static inline enum geist_status
metal_standalone_end(struct geist_backend *be, void *cmd, void *enc, const char *what) {
    struct metal_state *st = be->state;
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err == nullptr) {
        return GEIST_OK;
    }
    if (what != nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal %s: command failed%s%s",
                                what,
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
    }
    return GEIST_E_BACKEND;
}

/* The op's encoder: the sequence's (*out_cmd = nullptr) or a standalone
 * one; nullptr when there is none. */
[[gnu::always_inline]] static inline void *
metal_op_begin(struct geist_backend *be, const char *what, void **out_cmd) {
    struct metal_state *st = be->state;
    if (!metal_seq_mine(st)) {
        return metal_standalone_begin(be, what, out_cmd);
    }
    *out_cmd = nullptr;
    if (st->sequence_compute_encoder == nullptr) {
        if (what != nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal %s: command sequence has no encoder", what);
        }
        return nullptr;
    }
    return metal_sequence_encoder(st);
}

/* Finishes metal_op_begin: the sequence now has work, or the standalone
 * buffer runs. */
[[nodiscard, gnu::always_inline]] static inline enum geist_status
metal_op_end(struct geist_backend *be, void *cmd, void *enc, const char *what) {
    if (cmd == nullptr) {
        ((struct metal_state *) be->state)->sequence_has_work = true;
        return GEIST_OK;
    }
    return metal_standalone_end(be, cmd, enc, what);
}

/* Below this many rows a quantized linear reads the weights once per row
 * through the matvec kernel instead of one pass of a tiled GEMM: a bounded
 * GEMM tile costs about as much as ten matvec rows (#578). */
#define METAL_MATVEC_MAX_ROWS 9u
static void metal_encode_q4k_linear(struct metal_state            *st,
                                    void                          *enc,
                                    const struct geist_tensor     *x,
                                    const struct geist_tensor     *w,
                                    const struct geist_tensor     *y,
                                    const struct metal_q4k_params *params,
                                    bool                           m_tile8) {
    /* A few rows (a ragged tail, a short appended turn) go row by row
     * through the matvec kernel: the m8 tile kernel costs as much as a full
     * 64-row GEMM for any row count (#578). */
    if (m_tile8 && params->rows <= METAL_MATVEC_MAX_ROWS && params->n_out >= 4u) {
        for (uint32_t r = 0; r < params->rows; r++) {
            struct metal_q4k_params one = *params;
            one.rows                    = 1u;
            one.x_offset += r * params->x_row_stride;
            one.y_offset += r * params->y_row_stride;
            metal_encode_q4k_linear(st, enc, x, w, y, &one, false);
        }
        return;
    }
    const bool m_tile16 = m_tile8 && params->rows >= METAL_Q4K_M16_TILE;
    /* The bounded kernel clamps its row and column indices and copies a
     * partial tile out through threadgroup memory, so any rows >= 8 shape
     * takes it, as the other formats' GEMMs do (#578). The partial-tile
     * epilogue stores float4s, so y rows must start 16-byte aligned. */
    /* The fast kernel covers a ragged last row tile itself (#35): it
     * clamps its activation row and writes a partial tile through
     * threadgroup memory, so only n_out has to fill whole 64-wide tiles. */
    const bool full_tiles = (params->rows % 32u) == 0u && (params->n_out % 64u) == 0u;
    const bool fast_tiles = (params->n_out % 64u) == 0u;
    const bool m_tile_sg =
            m_tile8 && st->q4k_mm_sg_pipeline != nullptr && params->rows >= 8u &&
            params->n_out >= 64u &&
            (full_tiles || ((params->y_offset % 4u) == 0u && (params->y_row_stride % 4u) == 0u));
    /* interior fast variant: no bounds checks, vectorized activation
     * staging (needs full tiles, n_in%32 and 8-float-aligned x rows). */
    const bool m_tile_sg_fast = m_tile_sg && fast_tiles && st->q4k_mm_sg_fast_pipeline != nullptr &&
                                (params->n_in % 32u) == 0u && (params->x_offset % 8u) == 0u &&
                                (params->x_row_stride % 8u) == 0u;
    const bool m_tile8_active = m_tile8 && !m_tile16;
    const bool n_tile4        = !m_tile8 && params->rows == 1u;
    metal_msg_send_set_pipeline(st,
                                enc,
                                m_tile_sg_fast   ? st->q4k_mm_sg_fast_pipeline
                                : m_tile_sg      ? st->q4k_mm_sg_pipeline
                                : m_tile16       ? st->q4k_matmul_m16_pipeline
                                : m_tile8_active ? st->q4k_matmul_m8_pipeline
                                                 : st->q4k_n4_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, w->buffer->buffer, w->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    if (m_tile_sg) {
        metal_msg_send_set_threadgroup_memory(
                st, enc, m_tile_sg_fast && (params->rows % 32u) == 0u ? 6144u : 8192u, 0u);
    }

    const struct metal_size groups = {
            .width  = n_tile4     ? (params->n_out + 3u) / 4u
                      : m_tile_sg ? (params->rows + 31u) / 32u
                                  : params->n_out,
            .height = m_tile_sg  ? (params->n_out + 63u) / 64u
                      : m_tile16 ? (params->rows + METAL_Q4K_M16_TILE - 1u) / METAL_Q4K_M16_TILE
                      : m_tile8_active ? (params->rows + METAL_Q4K_M_TILE - 1u) / METAL_Q4K_M_TILE
                                       : params->rows,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = m_tile_sg ? 32u
                      : n_tile4 ? METAL_Q4K_N4_THREADS
                                : METAL_Q4K_THREADS_PER_ROW,
            .height = m_tile_sg ? 4u : 1u,
            .depth  = 1,
    };
    const enum metal_profile_stage profile_stage =
            n_tile4 ? METAL_PROFILE_DISPATCH_Q4K_LINEAR_N4 : METAL_PROFILE_DISPATCH_Q4K_LINEAR_BASE;
    metal_profile_add_dispatch(st, profile_stage, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

/* matmul_q6k_mm_sg_fast, the Q4_K fast kernel's structure (32-wide K
 * steps, every thread dequantizes) on Q6_K blocks (#17), for any row count:
 * a ragged last row tile is clamped and copied out through threadgroup
 * memory (#35). matmul_q6k_sg_fast steps K by 8 with a quarter of the
 * threads dequantizing. GEIST_METAL_Q6K_MM_FAST=0 keeps the old kernels.
 * False when the shape does not fit; nothing is encoded then. */
static bool metal_encode_q6k_mm32(struct metal_state            *st,
                                  void                          *enc,
                                  const struct geist_tensor     *x,
                                  const struct geist_tensor     *w,
                                  const struct geist_tensor     *y,
                                  const struct metal_q4k_params *params) {
    static int mm_fast_env = -1;
    if (mm_fast_env < 0) {
        const char *e = getenv("GEIST_METAL_Q6K_MM_FAST");
        mm_fast_env   = e == nullptr || e[0] != '0';
    }
    if (!mm_fast_env || st->q6k_mm32_pipeline == nullptr || (params->n_out % 64u) != 0u ||
        (params->n_in % 32u) != 0u || (params->x_offset % 8u) != 0u ||
        (params->x_row_stride % 8u) != 0u) {
        return false;
    }
    metal_msg_send_set_pipeline(st, enc, st->q6k_mm32_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, w->buffer->buffer, w->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    metal_msg_send_set_threadgroup_memory(st, enc, (params->rows % 32u) == 0u ? 6144u : 8192u, 0u);
    const struct metal_size groups = {
            .width = (params->rows + 31u) / 32u, .height = params->n_out / 64u, .depth = 1};
    const struct metal_size threads = {.width = 32u, .height = 4u, .depth = 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_Q6K_LINEAR_BASE, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
    return true;
}

static void metal_encode_q6k_linear(struct metal_state            *st,
                                    void                          *enc,
                                    const struct geist_tensor     *x,
                                    const struct geist_tensor     *w,
                                    const struct geist_tensor     *y,
                                    const struct metal_q4k_params *params,
                                    bool                           m_tile8) {
    if (m_tile8 && params->rows > METAL_MATVEC_MAX_ROWS &&
        metal_encode_q6k_mm32(st, enc, x, w, y, params)) {
        return;
    }
    /* As for Q4_K, whole 64-row tiles take the fast kernel and only a
     * ragged tail the bounded one (#578). */
    if (m_tile8 && params->rows > 64u && (params->rows % 64u) != 0u) {
        struct metal_q4k_params head = *params;
        struct metal_q4k_params tail = *params;
        head.rows                    = params->rows & ~63u;
        tail.rows                    = params->rows - head.rows;
        tail.x_offset += head.rows * params->x_row_stride;
        tail.y_offset += head.rows * params->y_row_stride;
        metal_encode_q6k_linear(st, enc, x, w, y, &head, true);
        metal_encode_q6k_linear(st, enc, x, w, y, &tail, true);
        return;
    }
    if (m_tile8 && params->rows > 1u && params->rows <= METAL_MATVEC_MAX_ROWS &&
        params->n_out >= 4u) {
        for (uint32_t r = 0; r < params->rows; r++) {
            struct metal_q4k_params one = *params;
            one.rows                    = 1u;
            one.x_offset += r * params->x_row_stride;
            one.y_offset += r * params->y_row_stride;
            metal_encode_q6k_linear(st, enc, x, w, y, &one, false);
        }
        return;
    }
    /* rows==1 always takes the matvec kernel — the 64-row GEMM tile is a
     * waste for a single row, and n4 (llama mul_mv structure) wins at any
     * n_out that fills at least one threadgroup. */
    const bool single_row = params->rows == 1u;
    const bool n_tile4    = single_row && params->n_out >= 4u;
    const bool m_tile_sg  = m_tile8 && !n_tile4 && st->q6k_matmul_sg_pipeline != nullptr;
    /* interior fast variant: no bounds checks, vectorized activation
     * staging, direct simdgroup_store output. */
    const bool m_tile_sg_fast = m_tile_sg && st->q6k_matmul_sg_fast_pipeline != nullptr &&
                                (params->rows % 64u) == 0u && (params->n_out % 32u) == 0u &&
                                (params->x_offset % 8u) == 0u && (params->x_row_stride % 8u) == 0u;
    metal_msg_send_set_pipeline(st,
                                enc,
                                m_tile_sg_fast ? st->q6k_matmul_sg_fast_pipeline
                                : m_tile_sg    ? st->q6k_matmul_sg_pipeline
                                : n_tile4      ? st->q6k_n4_pipeline
                                               : st->q6k_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, w->buffer->buffer, w->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);

    /* The q6k sg kernels are 128-thread / 4-simdgroup with a 32-output x
     * 64-batch-row tile (b0=tg.y*64, o0=tg.x*32). */
    const struct metal_size groups = {
            .width  = m_tile_sg ? (params->n_out + 31u) / 32u
                      : n_tile4 ? (params->n_out + 3u) / 4u
                                : params->n_out,
            .height = m_tile_sg ? (params->rows + 63u) / 64u : params->rows,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = m_tile_sg ? 32u
                      : n_tile4 ? METAL_Q4K_N4_THREADS
                                : METAL_Q4K_THREADS_PER_ROW,
            .height = m_tile_sg ? 4u : 1,
            .depth  = 1,
    };
    const enum metal_profile_stage profile_stage =
            n_tile4 ? METAL_PROFILE_DISPATCH_Q6K_LINEAR_N4 : METAL_PROFILE_DISPATCH_Q6K_LINEAR_BASE;
    metal_profile_add_dispatch(st, profile_stage, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

/* The pipelines and grid geometry for one quant dtype; a new format is
 * one case here. */
struct metal_quant_pipes {
    void    *n4;        /* decode GEMV (rows == 1) */
    void    *mm;        /* bounded 64x32 simdgroup GEMM */
    void    *mm_fast;   /* interior variant, nullable */
    void    *base;      /* naive single-row kernel, nullable */
    void    *m8;        /* naive 8-row kernel, nullable */
    uint32_t n4_tile;   /* output rows one n4 threadgroup covers */
    bool     gemm_only; /* no naive kernels: n4 for rows==1, GEMM above */
    bool     mm_fast64; /* mm_fast tiles 64 activation rows, not 32 */
};

static struct metal_quant_pipes
metal_quant_pipes_for(const struct metal_state *st, enum geist_dtype dtype, uint32_t n_out) {
    switch (dtype) {
    case GEIST_DTYPE_PQ2_0: {
        /* 8 rows per simdgroup halve the activation traffic (a thread
         * reads 128 bytes of x per iteration against R*8 bytes of
         * weights) but halve the threadgroup count too, so it pays only
         * from tuning.pq2_n8_min_n_out output rows up (on 27B shapes: the
         * FFN and lm_head GEMVs gain, the 1024-row attn kv ones lose). */
        const bool n8 = n_out >= st->tuning.pq2_n8_min_n_out && st->pq2_n8_pipeline != nullptr;
        return (struct metal_quant_pipes) {.n4 = n8 ? st->pq2_n8_pipeline : st->pq2_n4_pipeline,
                                           .mm = st->pq2_mm_pipeline,
                                           .mm_fast   = st->pq2_mm_fast_pipeline,
                                           .n4_tile   = n8 ? 16u : 8u,
                                           .gemm_only = true,
                                           .mm_fast64 = true};
    }
    case GEIST_DTYPE_Q4_0:
        return (struct metal_quant_pipes) {.n4      = st->q40_n4_pipeline,
                                           .mm      = st->q40_mm_pipeline,
                                           .mm_fast = st->q40_mm_fast_pipeline,
                                           .base    = st->q40_pipeline,
                                           .m8      = st->q40_m8_pipeline,
                                           .n4_tile = 8u};
    case GEIST_DTYPE_Q8_0:
        return (struct metal_quant_pipes) {.n4      = st->q80_n4_pipeline,
                                           .mm      = st->q80_mm_pipeline,
                                           .mm_fast = st->q80_mm_fast_pipeline,
                                           .base    = st->q80_pipeline,
                                           .m8      = st->q80_m8_pipeline,
                                           .n4_tile = 8u};
    case GEIST_DTYPE_Q4_1:
        return (struct metal_quant_pipes) {.n4      = st->q41_n4_pipeline,
                                           .mm      = st->q41_mm_pipeline,
                                           .mm_fast = st->q41_mm_fast_pipeline,
                                           .base    = st->q41_pipeline,
                                           .m8      = st->q41_m8_pipeline,
                                           .n4_tile = 4u};
    case GEIST_DTYPE_IQ4_NL:
        return (struct metal_quant_pipes) {.n4        = st->iq4nl_n4_pipeline,
                                           .mm        = st->iq4nl_mm_pipeline,
                                           .n4_tile   = 8u,
                                           .gemm_only = true};
    case GEIST_DTYPE_IQ4_XS:
        return (struct metal_quant_pipes) {.n4        = st->iq4xs_n4_pipeline,
                                           .mm        = st->iq4xs_mm_pipeline,
                                           .mm_fast   = st->iq4xs_mm_fast_pipeline,
                                           .n4_tile   = 4u,
                                           .gemm_only = true};
    case GEIST_DTYPE_Q3_K:
        return (struct metal_quant_pipes) {.n4        = st->q3k_n4_pipeline,
                                           .mm        = st->q3k_mm_pipeline,
                                           .n4_tile   = 4u,
                                           .gemm_only = true};
    case GEIST_DTYPE_I2_S:
        return (struct metal_quant_pipes) {.n4        = st->i2s_n4_pipeline,
                                           .mm        = st->i2s_mm_pipeline,
                                           .n4_tile   = 4u,
                                           .gemm_only = true};
    case GEIST_DTYPE_TQ2_0:
        return (struct metal_quant_pipes) {.n4        = st->tq2_n4_pipeline,
                                           .mm        = st->tq2_mm_pipeline,
                                           .n4_tile   = 4u,
                                           .gemm_only = true};
    case GEIST_DTYPE_IQ3_S:
        return (struct metal_quant_pipes) {.n4        = st->iq3s_n4_pipeline,
                                           .mm        = st->iq3s_mm_pipeline,
                                           .n4_tile   = 4u,
                                           .gemm_only = true};
    default: /* Q5_K */
        return (struct metal_quant_pipes) {.n4      = st->q5k_n4_pipeline,
                                           .mm      = st->q5k_mm_pipeline,
                                           .mm_fast = st->q5k_mm_fast_pipeline,
                                           .base    = st->q5k_pipeline,
                                           .m8      = st->q5k_m8_pipeline,
                                           .n4_tile = 4u};
    }
}

/* The fast PQ2_0 GEMM reads x as f16 (metal_qsg_mm_pq2_fast_source, #103)
 * from this grow-only private scratch of rows x n_in halves. It grows to the
 * widest call it sees -- on Bonsai 2 27B 320 x 17408 halves (11 MB) once the
 * first ffn_down has run -- so later chunks do not reallocate it (AGENT.md
 * §3); the old buffer stays alive while a command buffer still holds it. If
 * it cannot grow, the bounded GEMM takes the call (metal_pq2_xh_ready).
 * Serial-encoder ordering makes the reuse across GEMMs safe. */
static void metal_pq2_xh_reserve(struct geist_backend *be, size_t rows, size_t n_in) {
    struct metal_state *st = be->state;
    size_t              cap;
    if (ckd_mul(&cap, rows, n_in) || ckd_mul(&cap, cap, sizeof(uint16_t)) ||
        st->pq2_xh_capacity >= cap) {
        return;
    }
    metal_buffer_destroy_internal(be, st->pq2_xh_buffer);
    st->pq2_xh_buffer   = nullptr;
    st->pq2_xh_capacity = 0;
    if (metal_new_buffer(be, cap, GEIST_BUFFER_SCRATCH, 0, false, &st->pq2_xh_buffer) == GEIST_OK) {
        st->pq2_xh_capacity = cap;
    }
}

static bool metal_pq2_xh_ready(const struct metal_state      *st,
                               const struct metal_q4k_params *params) {
    return st->cvt_f16_pipeline != nullptr && st->pq2_xh_buffer != nullptr &&
           (size_t) params->rows * params->n_in * sizeof(uint16_t) <= st->pq2_xh_capacity;
}

static bool metal_pq2_m8_fits(const struct metal_state *st, const struct metal_q4k_params *params) {
    return params->rows >= 2u && params->rows <= 8u && st->pq2_m8_pipeline != nullptr &&
           st->cvt_f16_pipeline != nullptr && st->pq2_xh_buffer != nullptr &&
           (size_t) 8u * params->n_in * sizeof(uint16_t) <= st->pq2_xh_capacity &&
           (params->n_in % 128u) == 0u && (params->x_offset % 4u) == 0u &&
           (params->x_row_stride % 4u) == 0u;
}

/* x (rows x n_in f32 at x_offset, x_row_stride) -> pq2_xh_buffer as a dense
 * rows x n_in f16 copy. The fast path's alignment (n_in % 32, offset and
 * stride % 8) makes every float4 whole. */
static void metal_encode_pq2_x_f16(struct metal_state            *st,
                                   void                          *enc,
                                   const struct geist_tensor     *x,
                                   const struct metal_q4k_params *params) {
    const struct {
        uint32_t n4, xo4, xs4;
    } cp = {params->n_in / 4u, params->x_offset / 4u, params->x_row_stride / 4u};
    metal_msg_send_set_pipeline(st, enc, st->cvt_f16_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, st->pq2_xh_buffer->buffer, st->pq2_xh_buffer->base_off, 1);
    metal_msg_send_set_bytes(st, enc, &cp, sizeof cp, 2);
    const struct metal_size groups  = {(cp.n4 + 255u) / 256u, params->rows, 1};
    const struct metal_size threads = {256u, 1u, 1u};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_PQ2_X_F16, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_q40_q80_linear(struct metal_state            *st,
                                        void                          *enc,
                                        const struct geist_tensor     *x,
                                        const struct geist_tensor     *w,
                                        const struct geist_tensor     *y,
                                        const struct metal_q4k_params *params,
                                        enum geist_dtype               dtype) {
    /* PQ2_0 at 2..8 rows (speculative verify, #105): matmul_pq2_m8 reads
     * 8 f16 rows of x, so the scratch holds 8 (metal_pq2_m8_fits). */
    if (dtype == GEIST_DTYPE_PQ2_0 && metal_pq2_m8_fits(st, params)) {
        struct metal_q4k_params xh = *params;
        metal_encode_pq2_x_f16(st, enc, x, params);
        xh.x_offset     = 0u;
        xh.x_row_stride = params->n_in;
        metal_msg_send_set_pipeline(st, enc, st->pq2_m8_pipeline);
        metal_msg_send_set_buffer(
                st, enc, st->pq2_xh_buffer->buffer, st->pq2_xh_buffer->base_off, 0);
        metal_msg_send_set_buffer(st, enc, w->buffer->buffer, w->buffer->base_off, 1);
        metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 2);
        metal_msg_send_set_bytes(st, enc, &xh, sizeof xh, 3);
        metal_msg_send_set_threadgroup_memory(st, enc, 64u + 4u * 128u * 4u, 0u);
        const struct metal_size groups  = {(params->n_out + 15u) / 16u, 1u, 1u};
        const struct metal_size threads = {32u, 4u, 1u};
        metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_PQ2_M8, groups);
        metal_msg_send_dispatch(st, enc, groups, threads);
        return;
    }
    const struct metal_quant_pipes p = metal_quant_pipes_for(st, dtype, params->n_out);
    /* rows==1 -> simdgroup GEMV (llama mul_mv structure); rows>=8 -> 64x32
     * simdgroup GEMM (bounds-checked, arbitrary rows/n_out); the naive
     * kernels remain the fallback for tiny shapes. gemm_only formats have no naive
     * kernels, so their GEMM takes every rows>=2 shape. */
    const bool n_tile4 = params->rows == 1u && params->n_out >= 4u && p.n4 != nullptr;
    /* gemm_only: every shape the n4 GEMV does not take, including
     * rows == 1 with n_out < 4. The GEMM is bounds-checked, so a tiny
     * shape is slow, not wrong -- and there is no naive kernel below it
     * to fall back to. */
    const bool m_tile_sg =
            (p.gemm_only ? !n_tile4 : (params->rows >= 8u && params->n_out >= 64u)) &&
            p.mm != nullptr;
    /* interior fast variant: no bounds checks, vectorized activation
     * staging (needs full tiles, n_in%32 and 8-float-aligned x rows). */
    const bool m_tile_sg_fast = m_tile_sg && p.mm_fast != nullptr && (params->rows >= 8u) &&
                                (params->n_out % 64u) == 0u && (params->n_in % 32u) == 0u &&
                                (params->x_offset % 8u) == 0u &&
                                (params->x_row_stride % 8u) == 0u &&
                                (!p.mm_fast64 || metal_pq2_xh_ready(st, params));
    const bool tiled          = !n_tile4 && !m_tile_sg && params->rows >= 8u;
    /* The naive kernels, for the formats that have them. A gemm_only
     * format has none, but m_tile_sg above takes every shape its GEMV
     * does not, so base stays null only when a pipeline failed to
     * build -- and then there is nothing to encode. */
    void *const base = tiled ? p.m8 : p.base;
    void *const pipe = n_tile4 ? p.n4 : m_tile_sg_fast ? p.mm_fast : m_tile_sg ? p.mm : base;
    if (pipe == nullptr) {
        return;
    }
    /* PQ2_0's fast GEMM reads x as f16: one conversion into pq2_xh_buffer
     * first, then the GEMM's x is that dense copy. */
    const bool              fast64 = m_tile_sg_fast && p.mm_fast64;
    struct metal_q4k_params xh     = *params;
    if (fast64) {
        metal_encode_pq2_x_f16(st, enc, x, params);
        xh.x_offset     = 0u;
        xh.x_row_stride = params->n_in;
    }
    metal_msg_send_set_pipeline(st, enc, pipe);
    if (fast64) {
        metal_msg_send_set_buffer(
                st, enc, st->pq2_xh_buffer->buffer, st->pq2_xh_buffer->base_off, 0);
    } else {
        metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    }
    metal_msg_send_set_buffer(st, enc, w->buffer->buffer, w->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 2);
    metal_msg_send_set_bytes(st, enc, fast64 ? &xh : params, sizeof(*params), 3);
    const uint32_t tile_rows = fast64 ? 64u : 32u;
    if (m_tile_sg) {
        metal_msg_send_set_threadgroup_memory(
                st,
                enc,
                fast64 ? ((params->rows % 64u) == 0u ? 10240u : 16384u)
                : m_tile_sg_fast && (params->rows % 32u) == 0u ? 6144u
                                                               : 8192u,
                0u);
    }
    if (n_tile4 && dtype == GEIST_DTYPE_PQ2_0) {
        /* matvec_pq2_n{4,8}'s 256-entry half4 code table */
        metal_msg_send_set_threadgroup_memory(st, enc, 256u * 8u, 0u);
    }
    const struct metal_size groups = {
            .width  = n_tile4     ? (params->n_out + p.n4_tile - 1u) / p.n4_tile
                      : m_tile_sg ? (params->rows + tile_rows - 1u) / tile_rows
                                  : params->n_out,
            .height = n_tile4     ? params->rows
                      : m_tile_sg ? (params->n_out + 63u) / 64u
                      : tiled     ? (params->rows + 7u) / 8u
                                  : params->rows,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = n_tile4     ? METAL_Q4K_N4_THREADS
                      : m_tile_sg ? 32u
                                  : METAL_Q4K_THREADS_PER_ROW,
            .height = m_tile_sg ? 4u : 1u,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st,
                               n_tile4          ? METAL_PROFILE_DISPATCH_Q4K_LINEAR_N4
                               : m_tile_sg_fast ? METAL_PROFILE_DISPATCH_Q4K_LINEAR_MM_FAST
                                                : METAL_PROFILE_DISPATCH_Q4K_LINEAR_BASE,
                               groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

/* From this row width a single-row norm takes the width below instead of
 * the 256 threads of a prefill row. */
static constexpr uint32_t METAL_WIDE_ROWS_MIN_COLS = 1024u;

/* One row per threadgroup: at decode (rows == 1) that leaves the whole GPU
 * running a single threadgroup, latency-bound. The norm kernels keep up to
 * four elements per thread in registers, so a row of cols >= 1024 takes
 * cols/4 threads (rounded up to a simdgroup, at most 1024): every load in
 * flight at once, and as few simdgroups as that allows to meet at the
 * barrier and in the partial sum. A chain of 1536-wide decode norms on the
 * M1 Max: 8.0 us each at 1024 threads, 5.6 at 384; cols 1024/2048/3072 take
 * 256/512/768 threads, each the fastest of 256..1024 measured (#17).
 * Prefill already fills the machine with rows, so it keeps 256. The simd
 * kernels take their stride from threads_per_threadgroup. */
static uint32_t metal_rows_threads(uint32_t rows, uint32_t cols) {
    if (rows != 1u || cols < METAL_WIDE_ROWS_MIN_COLS) {
        return METAL_ELEM_THREADS;
    }
    const uint32_t quarter = (cols / 4u + 31u) / 32u * 32u;
    return quarter < 1024u ? quarter : 1024u;
}

static void metal_encode_rmsnorm_rows(struct metal_state             *st,
                                      void                           *enc,
                                      const struct geist_tensor      *x,
                                      const struct geist_tensor      *w,
                                      const struct geist_tensor      *y,
                                      const struct metal_rows_params *params) {

    metal_msg_send_set_pipeline(st, enc, st->rmsnorm_rows_simd_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, w->buffer->buffer, w->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    const struct metal_size groups = {
            .width  = params->rows,
            .height = 1,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = metal_rows_threads(params->rows, params->cols),
            .height = 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_RMSNORM_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_rmsnorm_add_rows(struct metal_state                  *st,
                                          void                                *enc,
                                          const struct geist_tensor           *res,
                                          const struct geist_tensor           *x,
                                          const struct geist_tensor           *w,
                                          const struct geist_tensor           *y,
                                          const struct metal_post_norm_params *params) {

    metal_msg_send_set_pipeline(st, enc, st->rmsnorm_add_rows_simd_pipeline);
    metal_msg_send_set_buffer(st, enc, res->buffer->buffer, res->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, w->buffer->buffer, w->buffer->base_off, 2);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 3);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 4);
    const struct metal_size groups = {
            .width  = params->rows,
            .height = 1,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = metal_rows_threads(params->rows, params->cols),
            .height = 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_RMSNORM_ADD_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_rmsnorm_add_norm_rows(struct metal_state                   *st,
                                               void                                 *enc,
                                               const struct geist_tensor            *res,
                                               const struct geist_tensor            *x,
                                               const struct geist_tensor            *w,
                                               const struct geist_tensor            *z_w,
                                               const struct geist_tensor            *y,
                                               const struct geist_tensor            *z,
                                               const struct metal_post_norm2_params *params) {
    /* Without w the kernel never reads buffer 2; x stands in. */
    const struct geist_tensor *wb = w != nullptr ? w : x;
    metal_msg_send_set_pipeline(st, enc, st->rmsnorm_add_norm_rows_simd_pipeline);
    metal_msg_send_set_buffer(st, enc, res->buffer->buffer, res->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, wb->buffer->buffer, wb->buffer->base_off, 2);
    metal_msg_send_set_buffer(st, enc, z_w->buffer->buffer, z_w->buffer->base_off, 3);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 4);
    metal_msg_send_set_buffer(st, enc, z->buffer->buffer, z->buffer->base_off, 5);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 6);
    const struct metal_size groups = {
            .width  = params->rows,
            .height = 1,
            .depth  = 1,
    };
    /* The thread count of rmsnorm_rows / rmsnorm_add_rows at this shape:
     * the same partial sums in the same order. */
    const struct metal_size threads = {
            .width  = metal_rows_threads(params->rows, params->cols),
            .height = 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ADD_NORM_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

/* a (op) b per element: add_rows, mul_rows, gelu_mul_rows and
 * silu_mul_rows share the bindings and the grid. */
static void metal_encode_binary_rows(struct metal_state                    *st,
                                     void                                  *enc,
                                     void                                  *pipeline,
                                     enum metal_profile_stage               stage,
                                     const struct geist_tensor             *a,
                                     const struct geist_tensor             *b,
                                     const struct geist_tensor             *y,
                                     const struct metal_binary_rows_params *params) {
    metal_msg_send_set_pipeline(st, enc, pipeline);
    metal_msg_send_set_buffer(st, enc, a->buffer->buffer, a->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, b->buffer->buffer, b->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    const struct metal_size groups = {
            .width  = (params->rows * params->cols + METAL_ELEM_THREADS - 1u) / METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, stage, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_scale_rows(struct metal_state                   *st,
                                    void                                 *enc,
                                    const struct geist_tensor            *x,
                                    const struct geist_tensor            *y,
                                    const struct metal_scale_rows_params *params) {
    metal_msg_send_set_pipeline(st, enc, st->scale_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 1);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 2);
    const struct metal_size groups = {
            .width  = (params->rows * params->cols + METAL_ELEM_THREADS - 1u) / METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_SCALE_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_qgate_split(struct metal_state              *st,
                                     void                            *enc,
                                     const struct geist_tensor       *joint,
                                     const struct geist_tensor       *q,
                                     const struct geist_tensor       *gate,
                                     const struct metal_qgate_params *params) {
    metal_msg_send_set_pipeline(st, enc, st->qgate_split_pipeline);
    metal_msg_send_set_buffer(st, enc, joint->buffer->buffer, joint->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, q->buffer->buffer, q->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, gate->buffer->buffer, gate->buffer->base_off, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    const size_t            total  = (size_t) params->rows * params->heads * params->head_dim;
    const struct metal_size groups = {(total + METAL_ELEM_THREADS - 1u) / METAL_ELEM_THREADS, 1, 1};
    const struct metal_size threads = {METAL_ELEM_THREADS, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_QGATE_SPLIT, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_sigmoid_mul(struct metal_state                    *st,
                                     void                                  *enc,
                                     const struct geist_tensor             *x,
                                     const struct geist_tensor             *gate,
                                     const struct geist_tensor             *y,
                                     const struct metal_binary_rows_params *params) {
    metal_msg_send_set_pipeline(st, enc, st->sigmoid_mul_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, gate->buffer->buffer, gate->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    const size_t            total  = (size_t) params->rows * params->cols;
    const struct metal_size groups = {(total + METAL_ELEM_THREADS - 1u) / METAL_ELEM_THREADS, 1, 1};
    const struct metal_size threads = {METAL_ELEM_THREADS, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_SIGMOID_MUL, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_gelu_rows(struct metal_state                   *st,
                                   void                                 *enc,
                                   const struct geist_tensor            *x,
                                   const struct geist_tensor            *y,
                                   const struct metal_scale_rows_params *params) {
    metal_msg_send_set_pipeline(st, enc, st->gelu_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 1);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 2);
    const struct metal_size groups = {
            .width  = (params->rows * params->cols + METAL_ELEM_THREADS - 1u) / METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_GELU_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

/* One-input elementwise rows (silu_rows, relu2_rows): same buffers and
 * params, the pipeline picks the function. */
static void metal_encode_unary_rows(struct metal_state                   *st,
                                    void                                 *enc,
                                    void                                 *pipeline,
                                    const struct geist_tensor            *x,
                                    const struct geist_tensor            *y,
                                    const struct metal_scale_rows_params *params) {
    metal_msg_send_set_pipeline(st, enc, pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 1);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 2);
    const struct metal_size groups = {
            .width  = (params->rows * params->cols + METAL_ELEM_THREADS - 1u) / METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    const struct metal_size threads = {METAL_ELEM_THREADS, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_GELU_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_embed_lookup_scaled(struct metal_state              *st,
                                             void                            *enc,
                                             const struct geist_tensor       *embed_table,
                                             const struct geist_tensor       *out,
                                             const struct metal_embed_params *params) {

    metal_msg_send_set_pipeline(st, enc, st->embed_lookup_scaled_pipeline);
    metal_msg_send_set_buffer(
            st, enc, embed_table->buffer->buffer, embed_table->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, out->buffer->buffer, out->buffer->base_off, 1);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 2);
    const struct metal_size groups = {
            .width  = (params->n + METAL_ELEM_THREADS - 1u) / METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_EMBED, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_rope_rows(struct metal_state             *st,
                                   void                           *enc,
                                   struct geist_tensor            *x,
                                   const struct geist_tensor      *cos,
                                   const struct geist_tensor      *sin,
                                   const struct metal_rope_params *params) {

    metal_msg_send_set_pipeline(st, enc, st->rope_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, cos->buffer->buffer, cos->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, sin->buffer->buffer, sin->buffer->base_off, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    /* One thread per rotated pair: the table row is as wide as the rotated
     * block, so half comes from the stride, not from head_dim (#432). */
    const size_t            half   = (size_t) params->rope_row_stride / 2u;
    const size_t            total  = (size_t) params->rows * (size_t) params->heads * half;
    const struct metal_size groups = {
            .width  = (total + METAL_ELEM_THREADS - 1u) / METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ROPE_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_attention_rows(struct metal_state                  *st,
                                        void                                *enc,
                                        const struct geist_tensor           *q,
                                        const struct geist_tensor           *k_cache,
                                        const struct geist_tensor           *v_cache,
                                        const struct geist_tensor           *y,
                                        const struct metal_attention_params *params) {

    void *pipeline = k_cache->dtype == GEIST_DTYPE_F16 ? st->attention_rows_f16_pipeline
                                                       : st->attention_rows_pipeline;
    metal_msg_send_set_pipeline(st, enc, pipeline);
    metal_msg_send_set_buffer(st, enc, q->buffer->buffer, q->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, k_cache->buffer->buffer, k_cache->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, v_cache->buffer->buffer, v_cache->buffer->base_off, 2);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 3);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 4);
    const struct metal_size groups = {
            .width  = params->rows,
            .height = params->q_heads,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    /* Books the SCALAR stage: this two-pass kernel is the O(kv) fallback
     * the head_dim-512 full-attention layers take (flash gate is <=256). */
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ATTENTION_QNORM_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_f32_matmul(struct metal_state            *st,
                                    void                          *enc,
                                    const struct geist_tensor     *x,
                                    const struct geist_tensor     *w,
                                    const struct geist_tensor     *y,
                                    const struct metal_f32_params *params) {
    /* F16/BF16 weights (#564) take the same two kernels with the weight
     * converted on load; the 64x32 GEMM stays F32-only. */
    void *base_pipe = st->f32_matmul_pipeline, *sg_pipe = st->f32_matmul_sg_pipeline;
    if (w->dtype == GEIST_DTYPE_F16) {
        base_pipe = st->f16w_matmul_pipeline;
        sg_pipe   = st->f16w_matmul_sg_pipeline;
    } else if (w->dtype == GEIST_DTYPE_BF16) {
        base_pipe = st->bf16w_matmul_pipeline;
        sg_pipe   = st->bf16w_matmul_sg_pipeline;
    }
    /* Multi-row (prefill): full-tile shapes take the 64x32 4-simdgroup GEMM
     * (mm_sg structure, f32 staging = bit-identical to the 8x8 kernel);
     * others the 8x8 simdgroup GEMM. Single-row keeps the reduction kernel. */
    if (params->rows == 1u && w->dtype == GEIST_DTYPE_F32 &&
        st->f32_matvec_rows_pipeline != nullptr &&
        ((params->n_in | params->x_offset | params->w_offset) % 4u) == 0u) {
        /* Decode: a simdgroup per output row (#196). */
        metal_msg_send_set_pipeline(st, enc, st->f32_matvec_rows_pipeline);
        metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
        metal_msg_send_set_buffer(st, enc, w->buffer->buffer, w->buffer->base_off, 1);
        metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 2);
        metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
        const struct metal_size rgroups  = {(params->n_out + 3u) / 4u, 1, 1};
        const struct metal_size rthreads = {128, 1, 1};
        metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_F32_MATMUL, rgroups);
        metal_msg_send_dispatch(st, enc, rgroups, rthreads);
        return;
    }
    const bool use_sg = params->rows > 1u && sg_pipe != nullptr;
    const bool use_mm = use_sg && w->dtype == GEIST_DTYPE_F32 &&
                        st->f32_matmul_mm_pipeline != nullptr && params->rows >= 8u &&
                        (params->n_out % 64u) == 0u && (params->n_in % 32u) == 0u &&
                        (params->x_offset % 8u) == 0u && (params->x_row_stride % 8u) == 0u &&
                        (params->w_offset % 4u) == 0u;
    metal_msg_send_set_pipeline(st,
                                enc,
                                use_mm   ? st->f32_matmul_mm_pipeline
                                : use_sg ? sg_pipe
                                         : base_pipe);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, w->buffer->buffer, w->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    if (use_mm) {
        /* sa 64x32 f32 + sb 32x32 f32 */
        metal_msg_send_set_threadgroup_memory(st, enc, 12288u, 0u);
    }
    const struct metal_size groups = {
            .width  = use_mm   ? (params->rows + 31u) / 32u
                      : use_sg ? (params->n_out + 7u) / 8u
                               : params->n_out,
            .height = use_mm   ? params->n_out / 64u
                      : use_sg ? (params->rows + 7u) / 8u
                               : params->rows,
            .depth  = 1,
    };
    const struct metal_size threads = {
            /* rows == 1 (decode): n_out threadgroups is all the parallelism
             * there is, so widen them; multi-row shapes take the sg/mm
             * kernels anyway. */
            .width  = use_sg ? 32u
                      : (params->rows == 1u && params->n_in >= METAL_WIDE_ROWS_MIN_COLS)
                              ? 1024u
                              : METAL_ELEM_THREADS,
            .height = use_mm ? 4u : 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_F32_MATMUL, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

/* A dense, contiguous F16 or BF16 [rows, cols] matrix inside its buffer;
 * the offset comes back in elements. */
static bool metal_tensor_is_half_matrix(const struct geist_tensor *t,
                                        size_t                    *out_rows,
                                        size_t                    *out_cols,
                                        size_t                    *out_offset_elems) {
    if (t == nullptr || t->buffer == nullptr ||
        (t->dtype != GEIST_DTYPE_F16 && t->dtype != GEIST_DTYPE_BF16) || t->ndim != 2 ||
        t->shape[0] <= 0 || t->shape[1] <= 0 || t->offset % sizeof(uint16_t) != 0) {
        return false;
    }
    /* Resolved weights arrive as BLOCK_QUANTIZED views without strides; a
     * DENSE view must be contiguous. */
    if (t->layout == GEIST_LAYOUT_DENSE && (t->stride[0] != t->shape[1] || t->stride[1] != 1)) {
        return false;
    }
    const size_t rows = (size_t) t->shape[0];
    const size_t cols = (size_t) t->shape[1];
    if (rows > SIZE_MAX / cols) {
        return false;
    }
    const size_t elems = rows * cols;
    if (t->offset > t->buffer->bytes || elems > (t->buffer->bytes - t->offset) / sizeof(uint16_t)) {
        return false;
    }
    *out_rows         = rows;
    *out_cols         = cols;
    *out_offset_elems = t->offset / sizeof(uint16_t);
    return true;
}

[[nodiscard]] static enum geist_status metal_f32_linear(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        const struct geist_tensor *w,
                                                        struct geist_tensor       *y,
                                                        bool                       matrix) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t rows         = 1;
    size_t n_in         = 0;
    size_t x_offset     = 0;
    size_t x_row_stride = 0;
    size_t y_rows       = 1;
    size_t y_cols       = 0;
    size_t y_offset     = 0;
    size_t y_row_stride = 0;
    size_t w_rows       = 0;
    size_t w_cols       = 0;
    size_t w_offset     = 0;
    size_t w_row_stride = 0;

    bool ok = false;
    if (matrix) {
        ok = metal_tensor_is_f32_matrix(x, &rows, &n_in, &x_offset, &x_row_stride) &&
             metal_tensor_is_f32_matrix(y, &y_rows, &y_cols, &y_offset, &y_row_stride);
    } else {
        ok           = metal_tensor_is_f32_vector(x, &n_in, &x_offset) &&
                       metal_tensor_is_f32_vector(y, &y_cols, &y_offset);
        x_row_stride = n_in;
        y_row_stride = y_cols;
    }
    if (w != nullptr && (w->dtype == GEIST_DTYPE_F16 || w->dtype == GEIST_DTYPE_BF16)) {
        /* Dense half-precision weights (#564): w_offset in elements, like
         * the F32 path's. */
        ok = ok && metal_tensor_is_half_matrix(w, &w_rows, &w_cols, &w_offset) && w_cols == n_in &&
             y_cols == w_rows && y_rows == rows;
    } else {
        ok = ok && metal_tensor_is_f32_matrix(w, &w_rows, &w_cols, &w_offset, &w_row_stride) &&
             w_row_stride == w_cols && w_cols == n_in && y_cols == w_rows && y_rows == rows;
    }
    if (!ok) {
        geist_backend_set_error(
                be,
                GEIST_E_UNSUPPORTED,
                matrix ? "metal F32 matmul: expected x F32 [rows,n], w F32 [out,n], y F32 "
                         "[rows,out]"
                       : "metal F32 matvec: expected x F32 [n], w F32 [out,n], y F32 [out]");
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || n_in > UINT32_MAX || w_rows > UINT32_MAX || x_offset > UINT32_MAX ||
        w_offset > UINT32_MAX || y_offset > UINT32_MAX || x_row_stride > UINT32_MAX ||
        y_row_stride > UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }
    if (x->buffer->owner != be->state || w->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    struct metal_state           *st     = be->state;
    const struct metal_f32_params params = {
            .n_in         = (uint32_t) n_in,
            .n_out        = (uint32_t) w_rows,
            .rows         = (uint32_t) rows,
            .x_offset     = (uint32_t) x_offset,
            .w_offset     = (uint32_t) w_offset,
            .y_offset     = (uint32_t) y_offset,
            .x_row_stride = (uint32_t) x_row_stride,
            .y_row_stride = (uint32_t) y_row_stride,
    };

    if (metal_seq_mine(st)) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal F32 linear: command sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_f32_matmul(st, metal_sequence_encoder(st), x, w, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);

    void *cmd = nullptr;
    void *enc = metal_standalone_begin(be, "F32 linear", &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_f32_matmul(st, enc, x, w, y, &params);
    return metal_standalone_end(be, cmd, enc, "F32 linear");
}

[[nodiscard]] static enum geist_status metal_matvec_f32_dense(struct geist_backend      *be,
                                                              const struct geist_tensor *x,
                                                              const struct geist_tensor *w,
                                                              struct geist_tensor       *y) {

    return metal_f32_linear(be, x, w, y, false);
}

[[nodiscard]] static enum geist_status metal_matmul_f32_dense(struct geist_backend      *be,
                                                              const struct geist_tensor *x,
                                                              const struct geist_tensor *w,
                                                              struct geist_tensor       *y) {

    return metal_f32_linear(be, x, w, y, true);
}

[[nodiscard]] static enum geist_status metal_rmsnorm(struct geist_backend      *be,
                                                     const struct geist_tensor *x,
                                                     const struct geist_tensor *w,
                                                     float                      eps,
                                                     struct geist_tensor       *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, x_off = 0, x_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    size_t w_n = 0, w_off = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &cols, &x_off, &x_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) ||
        !metal_tensor_is_f32_vector(w, &w_n, &w_off) || y_rows != rows || y_cols != cols ||
        w_n != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || x_off > UINT32_MAX || y_off > UINT32_MAX ||
        w_off > UINT32_MAX || x_stride > UINT32_MAX || y_stride > UINT32_MAX ||
        x->buffer->owner != be->state || w->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state            *st     = be->state;
    const struct metal_rows_params params = {
            .rows         = (uint32_t) rows,
            .cols         = (uint32_t) cols,
            .x_offset     = (uint32_t) x_off,
            .w_offset     = (uint32_t) w_off,
            .y_offset     = (uint32_t) y_off,
            .x_row_stride = (uint32_t) x_stride,
            .y_row_stride = (uint32_t) y_stride,
            .eps          = eps,
    };
    if (metal_seq_mine(st)) {
        metal_encode_rmsnorm_rows(st, metal_sequence_encoder(st), x, w, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);

    void *cmd = nullptr;
    void *enc = metal_standalone_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_rmsnorm_rows(st, enc, x, w, y, &params);
    return metal_standalone_end(be, cmd, enc, nullptr);
}

[[nodiscard]] static enum geist_status metal_rmsnorm_add(struct geist_backend      *be,
                                                         const struct geist_tensor *res,
                                                         const struct geist_tensor *x,
                                                         const struct geist_tensor *w,
                                                         float                      eps,
                                                         struct geist_tensor       *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, x_off = 0, x_stride = 0;
    size_t r_rows = 0, r_cols = 0, r_off = 0, r_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    size_t w_n = 0, w_off = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &cols, &x_off, &x_stride) ||
        !metal_tensor_is_f32_rows(res, &r_rows, &r_cols, &r_off, &r_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) ||
        !metal_tensor_is_f32_vector(w, &w_n, &w_off) || r_rows != rows || r_cols != cols ||
        y_rows != rows || y_cols != cols || w_n != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || x_off > UINT32_MAX || r_off > UINT32_MAX ||
        y_off > UINT32_MAX || w_off > UINT32_MAX || x_stride > UINT32_MAX ||
        r_stride > UINT32_MAX || y_stride > UINT32_MAX || x->buffer->owner != be->state ||
        res->buffer->owner != be->state || w->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state                 *st     = be->state;
    const struct metal_post_norm_params params = {
            .rows                = (uint32_t) rows,
            .cols                = (uint32_t) cols,
            .residual_offset     = (uint32_t) r_off,
            .x_offset            = (uint32_t) x_off,
            .w_offset            = (uint32_t) w_off,
            .y_offset            = (uint32_t) y_off,
            .residual_row_stride = (uint32_t) r_stride,
            .x_row_stride        = (uint32_t) x_stride,
            .y_row_stride        = (uint32_t) y_stride,
            .eps                 = eps,
    };
    if (metal_seq_mine(st)) {
        metal_encode_rmsnorm_add_rows(st, metal_sequence_encoder(st), res, x, w, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);

    void *cmd = nullptr;
    void *enc = metal_standalone_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_rmsnorm_add_rows(st, enc, res, x, w, y, &params);
    return metal_standalone_end(be, cmd, enc, nullptr);
}

/* Fused-table rmsnorm_add_norm (geist_backend.h): rmsnorm_add (or add),
 * scale and rmsnorm in one dispatch (#17). */
[[nodiscard]] static enum geist_status metal_rmsnorm_add_norm(struct geist_backend      *be,
                                                              float                      eps,
                                                              float                      scale,
                                                              const struct geist_tensor *res,
                                                              const struct geist_tensor *x,
                                                              const struct geist_tensor *w,
                                                              const struct geist_tensor *z_w,
                                                              struct geist_tensor       *y,
                                                              struct geist_tensor       *z) {

    if (be == nullptr || be->state == nullptr || res == nullptr || x == nullptr || z_w == nullptr ||
        y == nullptr || z == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, x_off = 0, x_stride = 0;
    size_t r_rows = 0, r_cols = 0, r_off = 0, r_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    size_t z_rows = 0, z_cols = 0, z_off = 0, z_stride = 0;
    size_t w_n = 0, w_off = 0, zw_n = 0, zw_off = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &cols, &x_off, &x_stride) ||
        !metal_tensor_is_f32_rows(res, &r_rows, &r_cols, &r_off, &r_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) ||
        !metal_tensor_is_f32_rows(z, &z_rows, &z_cols, &z_off, &z_stride) ||
        !metal_tensor_is_f32_vector(z_w, &zw_n, &zw_off) ||
        (w != nullptr && (!metal_tensor_is_f32_vector(w, &w_n, &w_off) || w_n != cols)) ||
        r_rows != rows || r_cols != cols || y_rows != rows || y_cols != cols || z_rows != rows ||
        z_cols != cols || zw_n != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || x_off > UINT32_MAX || r_off > UINT32_MAX ||
        y_off > UINT32_MAX || z_off > UINT32_MAX || w_off > UINT32_MAX || zw_off > UINT32_MAX ||
        x_stride > UINT32_MAX || r_stride > UINT32_MAX || y_stride > UINT32_MAX ||
        z_stride > UINT32_MAX || x->buffer->owner != be->state || res->buffer->owner != be->state ||
        (w != nullptr && w->buffer->owner != be->state) || z_w->buffer->owner != be->state ||
        y->buffer->owner != be->state || z->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state                  *st     = be->state;
    const struct metal_post_norm2_params params = {
            .rows                = (uint32_t) rows,
            .cols                = (uint32_t) cols,
            .residual_offset     = (uint32_t) r_off,
            .x_offset            = (uint32_t) x_off,
            .w_offset            = (uint32_t) w_off,
            .zw_offset           = (uint32_t) zw_off,
            .y_offset            = (uint32_t) y_off,
            .z_offset            = (uint32_t) z_off,
            .residual_row_stride = (uint32_t) r_stride,
            .x_row_stride        = (uint32_t) x_stride,
            .y_row_stride        = (uint32_t) y_stride,
            .z_row_stride        = (uint32_t) z_stride,
            .has_w               = w != nullptr ? 1u : 0u,
            .eps                 = eps,
            .scale               = scale,
    };
    if (metal_seq_mine(st)) {
        metal_encode_rmsnorm_add_norm_rows(
                st, metal_sequence_encoder(st), res, x, w, z_w, y, z, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);

    void *cmd = nullptr;
    void *enc = metal_standalone_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_rmsnorm_add_norm_rows(st, enc, res, x, w, z_w, y, z, &params);
    return metal_standalone_end(be, cmd, enc, nullptr);
}

/* The pipeline is passed as the address of its state field: it is read
 * only after metal_ensure_q4k_pipeline built it. */
[[nodiscard, gnu::always_inline]] static inline enum geist_status
metal_binary_rows(struct geist_backend      *be,
                  void *const               *pipeline,
                  enum metal_profile_stage   stage,
                  const struct geist_tensor *a,
                  const struct geist_tensor *b,
                  struct geist_tensor       *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, a_off = 0, a_stride = 0;
    size_t b_rows = 0, b_cols = 0, b_off = 0, b_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    if (!metal_tensor_is_f32_rows(a, &rows, &cols, &a_off, &a_stride) ||
        !metal_tensor_is_f32_rows(b, &b_rows, &b_cols, &b_off, &b_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) || b_rows != rows ||
        y_rows != rows || b_cols != cols || y_cols != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || a_off > UINT32_MAX || b_off > UINT32_MAX ||
        y_off > UINT32_MAX || a_stride > UINT32_MAX || b_stride > UINT32_MAX ||
        y_stride > UINT32_MAX || a->buffer->owner != be->state || b->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state                   *st     = be->state;
    const struct metal_binary_rows_params params = {
            .rows         = (uint32_t) rows,
            .cols         = (uint32_t) cols,
            .a_offset     = (uint32_t) a_off,
            .b_offset     = (uint32_t) b_off,
            .y_offset     = (uint32_t) y_off,
            .a_row_stride = (uint32_t) a_stride,
            .b_row_stride = (uint32_t) b_stride,
            .y_row_stride = (uint32_t) y_stride,
    };
    if (metal_seq_mine(st)) {
        metal_encode_binary_rows(
                st, metal_sequence_encoder(st), *pipeline, stage, a, b, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);

    void *cmd = nullptr;
    void *enc = metal_standalone_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_binary_rows(st, enc, *pipeline, stage, a, b, y, &params);
    return metal_standalone_end(be, cmd, enc, nullptr);
}

[[nodiscard]] static enum geist_status metal_add(struct geist_backend      *be,
                                                 const struct geist_tensor *a,
                                                 const struct geist_tensor *b,
                                                 struct geist_tensor       *y) {
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    return metal_binary_rows(be, &st->add_rows_pipeline, METAL_PROFILE_DISPATCH_ADD_ROWS, a, b, y);
}

[[nodiscard]] static enum geist_status metal_mul(struct geist_backend      *be,
                                                 const struct geist_tensor *a,
                                                 const struct geist_tensor *b,
                                                 struct geist_tensor       *y) {
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    return metal_binary_rows(be, &st->mul_rows_pipeline, METAL_PROFILE_DISPATCH_MUL_ROWS, a, b, y);
}

[[nodiscard]] static enum geist_status metal_scale_f32(struct geist_backend      *be,
                                                       const struct geist_tensor *x,
                                                       float                      scale,
                                                       struct geist_tensor       *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, x_off = 0, x_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &cols, &x_off, &x_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) || y_rows != rows ||
        y_cols != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || x_off > UINT32_MAX || y_off > UINT32_MAX ||
        x_stride > UINT32_MAX || y_stride > UINT32_MAX || x->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state                  *st     = be->state;
    const struct metal_scale_rows_params params = {
            .rows         = (uint32_t) rows,
            .cols         = (uint32_t) cols,
            .x_offset     = (uint32_t) x_off,
            .y_offset     = (uint32_t) y_off,
            .x_row_stride = (uint32_t) x_stride,
            .y_row_stride = (uint32_t) y_stride,
            .scale        = scale,
    };
    if (metal_seq_mine(st)) {
        metal_encode_scale_rows(st, metal_sequence_encoder(st), x, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);

    void *cmd = nullptr;
    void *enc = metal_standalone_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_scale_rows(st, enc, x, y, &params);
    return metal_standalone_end(be, cmd, enc, nullptr);
}

[[nodiscard]] static enum geist_status metal_attn_qgate_split(struct geist_backend      *be,
                                                              const struct geist_tensor *joint,
                                                              size_t                     heads,
                                                              size_t                     head_dim,
                                                              struct geist_tensor       *q,
                                                              struct geist_tensor       *gate) {
    if (be == nullptr || be->state == nullptr || heads == 0 || head_dim == 0) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, jo = 0, js = 0;
    size_t qr = 0, qc = 0, qo = 0, qs = 0;
    size_t gr = 0, gc = 0, go = 0, gs = 0;
    if (!metal_tensor_is_f32_matrix(joint, &rows, &cols, &jo, &js) ||
        !metal_tensor_is_f32_matrix(q, &qr, &qc, &qo, &qs) ||
        !metal_tensor_is_f32_matrix(gate, &gr, &gc, &go, &gs) || cols != heads * 2u * head_dim ||
        qr != rows || gr != rows || qc != heads * head_dim || gc != heads * head_dim) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || heads > UINT32_MAX || head_dim > UINT32_MAX || jo > UINT32_MAX ||
        qo > UINT32_MAX || go > UINT32_MAX || js > UINT32_MAX || qs > UINT32_MAX ||
        gs > UINT32_MAX || joint->buffer->owner != be->state || q->buffer->owner != be->state ||
        gate->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK)
        return s;
    struct metal_state             *st     = be->state;
    const struct metal_qgate_params params = {.rows             = (uint32_t) rows,
                                              .heads            = (uint32_t) heads,
                                              .head_dim         = (uint32_t) head_dim,
                                              .joint_offset     = (uint32_t) jo,
                                              .q_offset         = (uint32_t) qo,
                                              .gate_offset      = (uint32_t) go,
                                              .joint_row_stride = (uint32_t) js,
                                              .q_row_stride     = (uint32_t) qs,
                                              .gate_row_stride  = (uint32_t) gs};
    if (metal_seq_mine(st)) {
        metal_encode_qgate_split(st, metal_sequence_encoder(st), joint, q, gate, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);

    void *cmd = nullptr;
    void *enc = metal_standalone_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_qgate_split(st, enc, joint, q, gate, &params);
    return metal_standalone_end(be, cmd, enc, nullptr);
}

struct metal_hadamard_params {
    uint32_t width, block, hd, nk, rep, inv, sgn, xo, xs, yo, ys, so;
};

static void metal_encode_hadamard(struct metal_state                 *st,
                                  void                               *enc,
                                  const struct geist_hadamard_args   *a,
                                  const struct metal_hadamard_params *p,
                                  size_t                              rows) {
    const bool four = p->block >= 128u && p->block <= 4096u && st->hadamard4_pipeline != nullptr;
    metal_msg_send_set_pipeline(st, enc, four ? st->hadamard4_pipeline : st->hadamard_pipeline);
    metal_msg_send_set_buffer(st, enc, a->x->buffer->buffer, a->x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, a->y->buffer->buffer, a->y->buffer->base_off, 1);
    /* No signs: bind x again as a dummy; p->sgn == 0 keeps it unread. */
    const struct geist_tensor *sg = a->signs != nullptr ? a->signs : a->x;
    metal_msg_send_set_buffer(st, enc, sg->buffer->buffer, sg->buffer->base_off, 2);
    metal_msg_send_set_bytes(st, enc, p, sizeof(*p), 3);
    metal_msg_send_set_threadgroup_memory(st, enc, p->block * sizeof(float), 0u);
    const struct metal_size groups = {p->width / p->block, rows, 1};
    /* Blocks of 128..4096 (Bonsai's 1024) take hadamard_rows4: block/4
     * threads with four elements each, shuffles below stride 32 and
     * registers for the top two strides, 3 shared-memory stages instead of
     * 10 at 1024. Bit-identical to hadamard_rows and 3.3x faster on 320
     * rows of 5120 (0.254 -> 0.077 ms, M1 Max, #77). hadamard_rows keeps
     * one thread per element pair and stage, up to 1024, for the rest. */
    const struct metal_size threads = {
            four ? p->block / 4u : (p->block < 1024u ? p->block : 1024u), 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_HADAMARD, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

/* The geometry metal_hadamard_rotate runs: the shared contract
 * (hadamard.h) within the kernel's block range. Threadgroup memory holds
 * one block; 4096 floats is 16 KB, and setThreadgroupMemoryLength wants a
 * multiple of 16 bytes, so four floats is the floor. The probe and the
 * kernel both ask this, so they cannot drift apart. */
static bool metal_hadamard_geometry_ok(
        size_t width, size_t block, size_t perm_hd, size_t perm_nk, size_t perm_rep, bool inverse) {
    return block >= 4 && block <= 4096 && width <= UINT32_MAX &&
           geist_hadamard_geometry_ok(width, block, perm_hd, perm_nk, perm_rep, inverse);
}

/* fused->hadamard_rotate. Same contract as the CPU slot (hadamard.h). */
[[nodiscard]] static enum geist_status metal_hadamard_rotate(struct geist_backend             *be,
                                                             const struct geist_hadamard_args *a) {
    if (be == nullptr || be->state == nullptr || a == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, xo = 0, xs = 0, yr = 0, yc = 0, yo = 0, ys = 0;
    size_t sr = 0, sc = 0, so = 0, ss = 0;
    if (!metal_tensor_is_f32_rows(a->x, &rows, &cols, &xo, &xs) ||
        !metal_tensor_is_f32_rows(a->y, &yr, &yc, &yo, &ys) || yr != rows || yc != cols ||
        (a->signs != nullptr &&
         (!metal_tensor_is_f32_rows(a->signs, &sr, &sc, &so, &ss) || sr != 1 || sc != cols))) {
        return GEIST_E_INVALID_ARG;
    }
    const bool perm = a->perm_rep > 1;
    if (!metal_hadamard_geometry_ok(
                cols, a->block, a->perm_hd, a->perm_nk, a->perm_rep, a->inverse) ||
        (perm && a->x->buffer == a->y->buffer)) {
        return GEIST_E_INVALID_ARG;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || xo > UINT32_MAX || xs > UINT32_MAX ||
        yo > UINT32_MAX || ys > UINT32_MAX || so > UINT32_MAX || a->x->buffer->owner != be->state ||
        a->y->buffer->owner != be->state ||
        (a->signs != nullptr && a->signs->buffer->owner != be->state)) {
        return GEIST_E_INVALID_ARG;
    }
    if (rows == 0) {
        return GEIST_OK;
    }
    /* Same aliasing contract as the host path (hadamard.c): y may alias x
     * only exactly. A threadgroup stages a whole block before it writes,
     * so a shifted y races against the blocks its neighbours have not
     * read yet -- buffer identity alone does not catch that. */
    if (a->x->buffer->buffer == a->y->buffer->buffer) {
        const size_t xb   = a->x->buffer->base_off + xo * sizeof(float);
        const size_t yb   = a->y->buffer->base_off + yo * sizeof(float);
        const size_t xn   = ((rows - 1) * xs + cols) * sizeof(float);
        const size_t yn   = ((rows - 1) * ys + cols) * sizeof(float);
        const bool   same = xb == yb && xs == ys;
        if (!same && xb < yb + yn && yb < xb + xn) {
            return GEIST_E_INVALID_ARG;
        }
    }
    enum geist_status s = metal_ensure_hadamard_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state                *st = be->state;
    const struct metal_hadamard_params p  = {
            .width = (uint32_t) cols,
            .block = (uint32_t) a->block,
            .hd    = (uint32_t) (perm ? a->perm_hd : 0),
            .nk    = (uint32_t) (perm ? a->perm_nk : 0),
            .rep   = (uint32_t) (perm ? a->perm_rep : 0),
            .inv   = a->inverse ? 1u : 0u,
            .sgn   = a->signs != nullptr ? 1u : 0u,
            .xo    = (uint32_t) xo,
            .xs    = (uint32_t) xs,
            .yo    = (uint32_t) yo,
            .ys    = (uint32_t) ys,
            .so    = (uint32_t) so,
    };
    if (metal_seq_mine(st)) {
        metal_encode_hadamard(st, metal_sequence_encoder(st), a, &p, rows);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);

    void *cmd = nullptr;
    void *enc = metal_standalone_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_hadamard(st, enc, a, &p, rows);
    return metal_standalone_end(be, cmd, enc, nullptr);
}

[[nodiscard]] static enum geist_status metal_sigmoid_mul(struct geist_backend      *be,
                                                         const struct geist_tensor *x,
                                                         const struct geist_tensor *gate,
                                                         struct geist_tensor       *y) {
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, xo = 0, xs = 0;
    size_t gr = 0, gc = 0, go = 0, gs = 0;
    size_t yr = 0, yc = 0, yo = 0, ys = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &cols, &xo, &xs) ||
        !metal_tensor_is_f32_rows(gate, &gr, &gc, &go, &gs) ||
        !metal_tensor_is_f32_rows(y, &yr, &yc, &yo, &ys) || gr != rows || yr != rows ||
        gc != cols || yc != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || xo > UINT32_MAX || go > UINT32_MAX ||
        yo > UINT32_MAX || xs > UINT32_MAX || gs > UINT32_MAX || ys > UINT32_MAX ||
        x->buffer->owner != be->state || gate->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK)
        return s;
    struct metal_state                   *st     = be->state;
    const struct metal_binary_rows_params params = {.rows         = (uint32_t) rows,
                                                    .cols         = (uint32_t) cols,
                                                    .a_offset     = (uint32_t) xo,
                                                    .b_offset     = (uint32_t) go,
                                                    .y_offset     = (uint32_t) yo,
                                                    .a_row_stride = (uint32_t) xs,
                                                    .b_row_stride = (uint32_t) gs,
                                                    .y_row_stride = (uint32_t) ys};
    if (metal_seq_mine(st)) {
        metal_encode_sigmoid_mul(st, metal_sequence_encoder(st), x, gate, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);

    void *cmd = nullptr;
    void *enc = metal_standalone_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_sigmoid_mul(st, enc, x, gate, y, &params);
    return metal_standalone_end(be, cmd, enc, nullptr);
}

[[nodiscard]] static enum geist_status
metal_gelu_tanh(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, x_off = 0, x_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &cols, &x_off, &x_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) || y_rows != rows ||
        y_cols != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || x_off > UINT32_MAX || y_off > UINT32_MAX ||
        x_stride > UINT32_MAX || y_stride > UINT32_MAX || x->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state                  *st     = be->state;
    const struct metal_scale_rows_params params = {
            .rows         = (uint32_t) rows,
            .cols         = (uint32_t) cols,
            .x_offset     = (uint32_t) x_off,
            .y_offset     = (uint32_t) y_off,
            .x_row_stride = (uint32_t) x_stride,
            .y_row_stride = (uint32_t) y_stride,
            .scale        = 0.0f,
    };
    if (metal_seq_mine(st)) {
        metal_encode_gelu_rows(st, metal_sequence_encoder(st), x, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);

    void *cmd = nullptr;
    void *enc = metal_standalone_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_gelu_rows(st, enc, x, y, &params);
    return metal_standalone_end(be, cmd, enc, nullptr);
}

[[nodiscard]] static enum geist_status metal_unary_rows(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        struct geist_tensor       *y,
                                                        bool                       relu2) {
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, x_off = 0, x_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &cols, &x_off, &x_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) || y_rows != rows ||
        y_cols != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || x_off > UINT32_MAX || y_off > UINT32_MAX ||
        x_stride > UINT32_MAX || y_stride > UINT32_MAX || x->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state                  *st     = be->state;
    const struct metal_scale_rows_params params = {
            .rows         = (uint32_t) rows,
            .cols         = (uint32_t) cols,
            .x_offset     = (uint32_t) x_off,
            .y_offset     = (uint32_t) y_off,
            .x_row_stride = (uint32_t) x_stride,
            .y_row_stride = (uint32_t) y_stride,
            .scale        = 0.0f,
    };
    void *pipeline = relu2 ? st->relu2_rows_pipeline : st->silu_rows_pipeline;
    if (metal_seq_mine(st)) {
        metal_encode_unary_rows(st, metal_sequence_encoder(st), pipeline, x, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);

    void *cmd = nullptr;
    void *enc = metal_standalone_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_unary_rows(st, enc, pipeline, x, y, &params);
    return metal_standalone_end(be, cmd, enc, nullptr);
}

[[nodiscard]] static enum geist_status
metal_silu(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y) {
    return metal_unary_rows(be, x, y, false);
}

[[nodiscard]] static enum geist_status
metal_relu_squared(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y) {
    return metal_unary_rows(be, x, y, true);
}

[[nodiscard]] static enum geist_status metal_gelu_tanh_mul(struct geist_backend      *be,
                                                           const struct geist_tensor *x,
                                                           const struct geist_tensor *z,
                                                           struct geist_tensor       *y) {
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    return metal_binary_rows(
            be, &st->gelu_mul_rows_pipeline, METAL_PROFILE_DISPATCH_GELU_MUL_ROWS, x, z, y);
}

[[nodiscard]] static enum geist_status metal_silu_mul(struct geist_backend      *be,
                                                      const struct geist_tensor *x,
                                                      const struct geist_tensor *z,
                                                      struct geist_tensor       *y) {
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    return metal_binary_rows(
            be, &st->silu_mul_rows_pipeline, METAL_PROFILE_DISPATCH_GELU_MUL_ROWS, x, z, y);
}

/* Shared table-geometry check for the embed lookups: row_bytes /
 * blocks_per_row per dtype, UNSUPPORTED for anything else. */
[[nodiscard]] static enum geist_status
metal_embed_table_geometry(struct geist_backend      *be,
                           const struct geist_tensor *embed_table,
                           size_t                     d_model,
                           size_t                    *out_row_bytes,
                           size_t                    *out_blocks_per_row) {
    size_t row_bytes      = 0;
    size_t blocks_per_row = 0;
    if (embed_table->layout == GEIST_LAYOUT_DENSE && embed_table->dtype == GEIST_DTYPE_F32) {
        if (d_model > SIZE_MAX / sizeof(float)) {
            return GEIST_E_INVALID_ARG;
        }
        row_bytes = d_model * sizeof(float);
    } else if (embed_table->layout == GEIST_LAYOUT_DENSE &&
               (embed_table->dtype == GEIST_DTYPE_F16 || embed_table->dtype == GEIST_DTYPE_BF16)) {
        if (d_model > SIZE_MAX / sizeof(uint16_t)) {
            return GEIST_E_INVALID_ARG;
        }
        row_bytes = d_model * sizeof(uint16_t);
    } else if (embed_table->layout == GEIST_LAYOUT_BLOCK_QUANTIZED &&
               (embed_table->dtype == GEIST_DTYPE_Q4_0 || embed_table->dtype == GEIST_DTYPE_Q8_0 ||
                embed_table->dtype == GEIST_DTYPE_PQ2_0 || embed_table->dtype == GEIST_DTYPE_Q4_K ||
                embed_table->dtype == GEIST_DTYPE_Q5_K || embed_table->dtype == GEIST_DTYPE_Q6_K)) {
        /* embed_lookup_scaled dispatches on the raw enum value, so every
         * dtype its ternary chain names is pinned here: renumbering the
         * enum is a build error instead of wrong embedding rows. */
        static_assert(GEIST_DTYPE_F32 == 0, "embed shader hardcodes F32 as 0");
        static_assert(GEIST_DTYPE_F16 == 1, "embed shader hardcodes F16 as 1");
        static_assert(GEIST_DTYPE_BF16 == 2, "embed shader hardcodes BF16 as 2");
        static_assert(GEIST_DTYPE_Q4_0 == 5, "embed shader hardcodes Q4_0 as 5");
        static_assert(GEIST_DTYPE_Q8_0 == 7, "embed shader hardcodes Q8_0 as 7");
        static_assert(GEIST_DTYPE_Q4_K == 9, "embed shader hardcodes Q4_K as 9");
        static_assert(GEIST_DTYPE_Q5_K == 10, "embed shader hardcodes Q5_K as 10");
        static_assert(GEIST_DTYPE_PQ2_0 == 22, "embed shader hardcodes PQ2_0 as 22");
        const enum geist_dtype dtype = (enum geist_dtype) embed_table->dtype;
        if (d_model % metal_quant_block_elems(dtype) != 0 ||
            quant_raw_bytes(dtype, d_model, &row_bytes)) {
            return GEIST_E_INVALID_ARG;
        }
        blocks_per_row = d_model / metal_quant_block_elems(dtype);
    } else {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "metal embedding_lookup_scaled: unsupported table dtype/layout");
        return GEIST_E_UNSUPPORTED;
    }
    *out_row_bytes      = row_bytes;
    *out_blocks_per_row = blocks_per_row;
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
metal_embedding_lookup_scaled(struct geist_backend      *be,
                              const struct geist_tensor *embed_table,
                              geist_token_t              token_id,
                              float                      scale,
                              struct geist_tensor       *out) {

    if (be == nullptr || be->state == nullptr || embed_table == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t out_n      = 0;
    size_t out_offset = 0;
    if (!metal_tensor_is_f32_vector(out, &out_n, &out_offset) || embed_table->buffer == nullptr ||
        embed_table->ndim != 2 || embed_table->shape[0] <= 0 || embed_table->shape[1] <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t vocab   = (size_t) embed_table->shape[0];
    const size_t d_model = (size_t) embed_table->shape[1];
    if (token_id < 0 || (size_t) token_id >= vocab || out_n != d_model) {
        return GEIST_E_INVALID_ARG;
    }

    size_t row_bytes      = 0;
    size_t blocks_per_row = 0;
    {
        const enum geist_status gs =
                metal_embed_table_geometry(be, embed_table, d_model, &row_bytes, &blocks_per_row);
        if (gs != GEIST_OK) {
            return gs;
        }
    }

    if (row_bytes == 0 || vocab > SIZE_MAX / row_bytes ||
        embed_table->offset > embed_table->buffer->bytes ||
        vocab * row_bytes > embed_table->buffer->bytes - embed_table->offset ||
        out->offset > out->buffer->bytes ||
        d_model > (out->buffer->bytes - out->offset) / sizeof(float) || d_model > UINT32_MAX ||
        blocks_per_row > UINT32_MAX || embed_table->offset > UINT32_MAX ||
        out_offset > UINT32_MAX || (size_t) token_id > UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }
    if (embed_table->buffer->owner != be->state || out->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    struct metal_state             *st     = be->state;
    const struct metal_embed_params params = {
            .n              = (uint32_t) d_model,
            .dtype          = (uint32_t) embed_table->dtype,
            .blocks_per_row = (uint32_t) blocks_per_row,
            .w_byte_offset  = (uint32_t) embed_table->offset,
            .y_offset       = (uint32_t) out_offset,
            .token_id       = (uint32_t) token_id,
            .scale          = scale,
    };

    if (metal_seq_mine(st)) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be,
                    GEIST_E_BACKEND,
                    "metal embedding_lookup_scaled: command sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_embed_lookup_scaled(st, metal_sequence_encoder(st), embed_table, out, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);

    void *cmd = nullptr;
    void *enc = metal_standalone_begin(be, "embedding_lookup_scaled", &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_embed_lookup_scaled(st, enc, embed_table, out, &params);
    return metal_standalone_end(be, cmd, enc, "embedding_lookup_scaled");
}

/* Batched twin: one dispatch embeds a whole prefill chunk.
 * ids travel via setBytes (constant buffer), so the row count is bounded
 * by the 4 KB setBytes budget. */
enum { METAL_EMBED_ROWS_MAX = 1024 };

[[nodiscard]] static enum geist_status
metal_embedding_lookup_scaled_rows(struct geist_backend      *be,
                                   size_t                     n_rows,
                                   const struct geist_tensor *embed_table,
                                   const geist_token_t        ids[static n_rows],
                                   float                      scale,
                                   struct geist_tensor       *out) {
    if (be == nullptr || be->state == nullptr || embed_table == nullptr || ids == nullptr ||
        out == nullptr || n_rows == 0) {
        return GEIST_E_INVALID_ARG;
    }
    if (n_rows > METAL_EMBED_ROWS_MAX) {
        return GEIST_E_UNSUPPORTED; /* caller falls back to the per-token loop */
    }
    size_t rows = 0, cols = 0, o_off = 0, o_stride = 0;
    if (!metal_tensor_is_f32_matrix(out, &rows, &cols, &o_off, &o_stride) ||
        embed_table->buffer == nullptr || embed_table->ndim != 2 || embed_table->shape[0] <= 0 ||
        embed_table->shape[1] <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t vocab   = (size_t) embed_table->shape[0];
    const size_t d_model = (size_t) embed_table->shape[1];
    if (rows != n_rows || cols != d_model || o_stride != d_model) {
        return GEIST_E_INVALID_ARG;
    }
    size_t row_bytes = 0, blocks_per_row = 0;
    {
        const enum geist_status gs =
                metal_embed_table_geometry(be, embed_table, d_model, &row_bytes, &blocks_per_row);
        if (gs != GEIST_OK) {
            return gs;
        }
    }
    if (row_bytes == 0 || vocab > SIZE_MAX / row_bytes ||
        embed_table->offset > embed_table->buffer->bytes ||
        vocab * row_bytes > embed_table->buffer->bytes - embed_table->offset ||
        out->offset > out->buffer->bytes ||
        n_rows * d_model > (out->buffer->bytes - out->offset) / sizeof(float) ||
        d_model > UINT32_MAX || blocks_per_row > UINT32_MAX || embed_table->offset > UINT32_MAX ||
        o_off > UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }
    if (embed_table->buffer->owner != be->state || out->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    uint32_t ids_u32[METAL_EMBED_ROWS_MAX];
    for (size_t i = 0; i < n_rows; i++) {
        if (ids[i] < 0 || (size_t) ids[i] >= vocab) {
            return GEIST_E_INVALID_ARG;
        }
        ids_u32[i] = (uint32_t) ids[i];
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state             *st     = be->state;
    const struct metal_embed_params params = {
            .n              = (uint32_t) d_model,
            .dtype          = (uint32_t) embed_table->dtype,
            .blocks_per_row = (uint32_t) blocks_per_row,
            .w_byte_offset  = (uint32_t) embed_table->offset,
            .y_offset       = (uint32_t) o_off,
            .token_id       = (uint32_t) n_rows, /* row count in the batch kernel */
            .scale          = scale,
    };
    if (!metal_seq_mine(st)) {
        return GEIST_E_UNSUPPORTED; /* prefill runs sequenced; loop covers the rest */
    }
    void *enc = metal_sequence_encoder(st);
    if (enc == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "metal embedding_lookup_scaled_rows: no encoder");
        return GEIST_E_BACKEND;
    }
    metal_msg_send_set_pipeline(st, enc, st->embed_lookup_scaled_rows_pipeline);
    metal_msg_send_set_buffer(
            st, enc, embed_table->buffer->buffer, embed_table->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, out->buffer->buffer, out->buffer->base_off, 1);
    metal_msg_send_set_bytes(st, enc, &params, sizeof params, 2);
    metal_msg_send_set_bytes(st, enc, ids_u32, n_rows * sizeof(uint32_t), 3);
    const struct metal_size groups = {
            .width  = ((uint32_t) d_model + METAL_ELEM_THREADS - 1u) / METAL_ELEM_THREADS,
            .height = (uint32_t) n_rows,
            .depth  = 1,
    };
    const struct metal_size threads = {.width = METAL_ELEM_THREADS, .height = 1, .depth = 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_EMBED, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
    st->sequence_has_work = true;
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status metal_q40_q80_linear(struct geist_backend      *be,
                                                            const struct geist_tensor *x,
                                                            const struct geist_tensor *w,
                                                            struct geist_tensor       *y,
                                                            enum geist_dtype           dtype,
                                                            bool                       matrix) {
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 1, n_in = 0, x_offset = 0, x_row_stride = 0;
    size_t y_rows = 1, y_cols = 0, y_offset = 0, y_row_stride = 0;
    size_t n_out = 0, w_cols = 0, w_offset = 0;
    bool   ok;
    if (matrix) {
        ok = metal_tensor_is_f32_matrix(x, &rows, &n_in, &x_offset, &x_row_stride) &&
             metal_tensor_is_f32_matrix(y, &y_rows, &y_cols, &y_offset, &y_row_stride) &&
             y_rows == rows;
    } else {
        ok           = metal_tensor_is_f32_vector(x, &n_in, &x_offset) &&
                       metal_tensor_is_f32_vector(y, &y_cols, &y_offset);
        x_row_stride = n_in;
        y_row_stride = y_cols;
    }
    const bool w_ok = metal_tensor_is_q40_q80_matrix(w, dtype, &n_out, &w_cols, &w_offset);
    if (!ok || !w_ok || w_cols != n_in || y_cols != n_out) {
        geist_backend_set_error(
                be,
                GEIST_E_UNSUPPORTED,
                "metal quant linear shape: io=%d w=%d rows=%zu n=%zu/%zu out=%zu/%zu",
                ok,
                w_ok,
                rows,
                n_in,
                w_cols,
                y_cols,
                n_out);
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || n_in > UINT32_MAX || n_out > UINT32_MAX || x_offset > UINT32_MAX ||
        y_offset > UINT32_MAX || w_offset > UINT32_MAX || x_row_stride > UINT32_MAX ||
        y_row_stride > UINT32_MAX || x->buffer->owner != be->state ||
        w->buffer->owner != be->state || y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st = be->state;
    if (dtype == GEIST_DTYPE_PQ2_0 && rows >= 8u && n_out % 64u == 0u && n_in % 32u == 0u) {
        metal_pq2_xh_reserve(be, rows, n_in);
    } else if (dtype == GEIST_DTYPE_PQ2_0 && rows >= 2u && rows <= 8u) {
        metal_pq2_xh_reserve(be, 8u, n_in);
    }
    const struct metal_q4k_params params = {
            .n_in           = (uint32_t) n_in,
            .n_out          = (uint32_t) n_out,
            .rows           = (uint32_t) rows,
            .blocks_per_row = (uint32_t) (n_in / metal_quant_block_elems(dtype)),
            .x_offset       = (uint32_t) x_offset,
            .w_byte_offset  = (uint32_t) w_offset,
            .y_offset       = (uint32_t) y_offset,
            .x_row_stride   = (uint32_t) x_row_stride,
            .y_row_stride   = (uint32_t) y_row_stride,
    };
    if (metal_seq_mine(st)) {
        metal_encode_q40_q80_linear(st, metal_sequence_encoder(st), x, w, y, &params, dtype);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);

    void *cmd = nullptr;
    void *enc = metal_standalone_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_q40_q80_linear(st, enc, x, w, y, &params, dtype);
    return metal_standalone_end(be, cmd, enc, nullptr);
}

[[nodiscard]] static enum geist_status metal_matvec_q4k(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        const struct geist_tensor *w,
                                                        struct geist_tensor       *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t n_in     = 0;
    size_t x_offset = 0;
    size_t y_n      = 0;
    size_t y_offset = 0;
    size_t n_out    = 0;
    size_t w_cols   = 0;
    size_t w_offset = 0;
    if (!metal_tensor_is_f32_vector(x, &n_in, &x_offset) ||
        !metal_tensor_is_f32_vector(y, &y_n, &y_offset) ||
        !metal_tensor_is_q4k_matrix(w, &n_out, &w_cols, &w_offset) || w_cols != n_in ||
        y_n != n_out) {
        geist_backend_set_error(
                be,
                GEIST_E_UNSUPPORTED,
                "metal Q4_K matvec: expected x F32 [n], w Q4_K [out,n], y F32 [out]");
        return GEIST_E_UNSUPPORTED;
    }
    if (n_in > UINT32_MAX || n_out > UINT32_MAX || x_offset > UINT32_MAX || y_offset > UINT32_MAX ||
        w_offset > UINT32_MAX) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "metal Q4_K matvec: tensor too large");
        return GEIST_E_INVALID_ARG;
    }
    if (x->buffer->owner != be->state || w->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    struct metal_state           *st     = be->state;
    const struct metal_q4k_params params = {
            .n_in           = (uint32_t) n_in,
            .n_out          = (uint32_t) n_out,
            .rows           = 1,
            .blocks_per_row = (uint32_t) (n_in / METAL_Q4K_BLOCK_ELEMS),
            .x_offset       = (uint32_t) x_offset,
            .w_byte_offset  = (uint32_t) w_offset,
            .y_offset       = (uint32_t) y_offset,
            .x_row_stride   = (uint32_t) n_in,
            .y_row_stride   = (uint32_t) n_out,
    };

    if (metal_seq_mine(st)) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal Q4_K matvec: command sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_q4k_linear(st, metal_sequence_encoder(st), x, w, y, &params, false);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);

    void *cmd = nullptr;
    void *enc = metal_standalone_begin(be, "Q4_K matvec", &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_q4k_linear(st, enc, x, w, y, &params, false);
    return metal_standalone_end(be, cmd, enc, "Q4_K matvec");
}

[[nodiscard]] static enum geist_status metal_matmul_q4k(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        const struct geist_tensor *w,
                                                        struct geist_tensor       *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows         = 0;
    size_t n_in         = 0;
    size_t x_offset     = 0;
    size_t x_row_stride = 0;
    size_t y_rows       = 0;
    size_t y_cols       = 0;
    size_t y_offset     = 0;
    size_t y_row_stride = 0;
    size_t n_out        = 0;
    size_t w_cols       = 0;
    size_t w_offset     = 0;
    if (!metal_tensor_is_f32_matrix(x, &rows, &n_in, &x_offset, &x_row_stride) ||
        !metal_tensor_is_f32_matrix(y, &y_rows, &y_cols, &y_offset, &y_row_stride) ||
        !metal_tensor_is_q4k_matrix(w, &n_out, &w_cols, &w_offset) || y_rows != rows ||
        w_cols != n_in || y_cols != n_out) {
        geist_backend_set_error(
                be,
                GEIST_E_UNSUPPORTED,
                "metal Q4_K matmul: expected x F32 [rows,n], w Q4_K [out,n], y F32 [rows,out]");
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || n_in > UINT32_MAX || n_out > UINT32_MAX || x_offset > UINT32_MAX ||
        y_offset > UINT32_MAX || w_offset > UINT32_MAX || x_row_stride > UINT32_MAX ||
        y_row_stride > UINT32_MAX) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "metal Q4_K matmul: tensor too large");
        return GEIST_E_INVALID_ARG;
    }
    if (x->buffer->owner != be->state || w->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    struct metal_state           *st     = be->state;
    const struct metal_q4k_params params = {
            .n_in           = (uint32_t) n_in,
            .n_out          = (uint32_t) n_out,
            .rows           = (uint32_t) rows,
            .blocks_per_row = (uint32_t) (n_in / METAL_Q4K_BLOCK_ELEMS),
            .x_offset       = (uint32_t) x_offset,
            .w_byte_offset  = (uint32_t) w_offset,
            .y_offset       = (uint32_t) y_offset,
            .x_row_stride   = (uint32_t) x_row_stride,
            .y_row_stride   = (uint32_t) y_row_stride,
    };

    if (metal_seq_mine(st)) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal Q4_K matmul: command sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_q4k_linear(st, metal_sequence_encoder(st), x, w, y, &params, true);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);

    void *cmd = nullptr;
    void *enc = metal_standalone_begin(be, "Q4_K matmul", &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_q4k_linear(st, enc, x, w, y, &params, true);
    return metal_standalone_end(be, cmd, enc, "Q4_K matmul");
}

[[nodiscard]] static enum geist_status metal_q5k_linear(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        const struct geist_tensor *w,
                                                        struct geist_tensor       *y,
                                                        bool                       matrix) {
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 1, n_in = 0, x_offset = 0, x_row_stride = 0;
    size_t y_rows = 1, y_cols = 0, y_offset = 0, y_row_stride = 0;
    size_t n_out = 0, w_cols = 0, w_offset = 0;
    bool   ok;
    if (matrix) {
        ok = metal_tensor_is_f32_matrix(x, &rows, &n_in, &x_offset, &x_row_stride) &&
             metal_tensor_is_f32_matrix(y, &y_rows, &y_cols, &y_offset, &y_row_stride) &&
             y_rows == rows;
    } else {
        ok           = metal_tensor_is_f32_vector(x, &n_in, &x_offset) &&
                       metal_tensor_is_f32_vector(y, &y_cols, &y_offset);
        x_row_stride = n_in;
        y_row_stride = y_cols;
    }
    if (!ok || !metal_tensor_is_q5k_matrix(w, &n_out, &w_cols, &w_offset) || w_cols != n_in ||
        y_cols != n_out) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || n_in > UINT32_MAX || n_out > UINT32_MAX || x_offset > UINT32_MAX ||
        y_offset > UINT32_MAX || w_offset > UINT32_MAX || x_row_stride > UINT32_MAX ||
        y_row_stride > UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }
    if (x->buffer->owner != be->state || w->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state           *st     = be->state;
    const struct metal_q4k_params params = {
            .n_in           = (uint32_t) n_in,
            .n_out          = (uint32_t) n_out,
            .rows           = (uint32_t) rows,
            .blocks_per_row = (uint32_t) (n_in / METAL_Q5K_BLOCK_ELEMS),
            .x_offset       = (uint32_t) x_offset,
            .w_byte_offset  = (uint32_t) w_offset,
            .y_offset       = (uint32_t) y_offset,
            .x_row_stride   = (uint32_t) x_row_stride,
            .y_row_stride   = (uint32_t) y_row_stride,
    };
    if (metal_seq_mine(st)) {
        metal_encode_q40_q80_linear(
                st, metal_sequence_encoder(st), x, w, y, &params, GEIST_DTYPE_Q5_K);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);

    void *cmd = nullptr;
    void *enc = metal_standalone_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_q40_q80_linear(st, enc, x, w, y, &params, GEIST_DTYPE_Q5_K);
    return metal_standalone_end(be, cmd, enc, nullptr);
}

[[nodiscard]] static enum geist_status metal_matvec_q5k(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        const struct geist_tensor *w,
                                                        struct geist_tensor       *y) {
    return metal_q5k_linear(be, x, w, y, false);
}

[[nodiscard]] static enum geist_status metal_matmul_q5k(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        const struct geist_tensor *w,
                                                        struct geist_tensor       *y) {
    return metal_q5k_linear(be, x, w, y, true);
}

[[nodiscard]] static enum geist_status metal_q6k_linear(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        const struct geist_tensor *w,
                                                        struct geist_tensor       *y,
                                                        bool                       matrix) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows         = 1;
    size_t n_in         = 0;
    size_t x_offset     = 0;
    size_t x_row_stride = 0;
    size_t y_rows       = 1;
    size_t y_cols       = 0;
    size_t y_offset     = 0;
    size_t y_row_stride = 0;
    size_t n_out        = 0;
    size_t w_cols       = 0;
    size_t w_offset     = 0;
    bool   ok           = false;
    if (matrix) {
        ok = metal_tensor_is_f32_matrix(x, &rows, &n_in, &x_offset, &x_row_stride) &&
             metal_tensor_is_f32_matrix(y, &y_rows, &y_cols, &y_offset, &y_row_stride) &&
             y_rows == rows;
    } else {
        ok           = metal_tensor_is_f32_vector(x, &n_in, &x_offset) &&
                       metal_tensor_is_f32_vector(y, &y_cols, &y_offset);
        x_row_stride = n_in;
        y_row_stride = y_cols;
    }
    if (!ok || !metal_tensor_is_q6k_matrix(w, &n_out, &w_cols, &w_offset) || w_cols != n_in ||
        y_cols != n_out) {
        geist_backend_set_error(
                be,
                GEIST_E_UNSUPPORTED,
                matrix ? "metal Q6_K matmul: expected x F32 [rows,n], w Q6_K [out,n], y F32 "
                         "[rows,out]"
                       : "metal Q6_K matvec: expected x F32 [n], w Q6_K [out,n], y F32 [out]");
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || n_in > UINT32_MAX || n_out > UINT32_MAX || x_offset > UINT32_MAX ||
        y_offset > UINT32_MAX || w_offset > UINT32_MAX || x_row_stride > UINT32_MAX ||
        y_row_stride > UINT32_MAX) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "metal Q6_K linear: tensor too large");
        return GEIST_E_INVALID_ARG;
    }
    if (x->buffer->owner != be->state || w->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    struct metal_state           *st     = be->state;
    const struct metal_q4k_params params = {
            .n_in           = (uint32_t) n_in,
            .n_out          = (uint32_t) n_out,
            .rows           = (uint32_t) rows,
            .blocks_per_row = (uint32_t) (n_in / METAL_Q6K_BLOCK_ELEMS),
            .x_offset       = (uint32_t) x_offset,
            .w_byte_offset  = (uint32_t) w_offset,
            .y_offset       = (uint32_t) y_offset,
            .x_row_stride   = (uint32_t) x_row_stride,
            .y_row_stride   = (uint32_t) y_row_stride,
    };

    if (metal_seq_mine(st)) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal Q6_K linear: command sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_q6k_linear(st, metal_sequence_encoder(st), x, w, y, &params, matrix);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);

    void *cmd = nullptr;
    void *enc = metal_standalone_begin(be, "Q6_K linear", &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_q6k_linear(st, enc, x, w, y, &params, matrix);
    return metal_standalone_end(be, cmd, enc, "Q6_K linear");
}

[[nodiscard]] static enum geist_status metal_matvec_q6k(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        const struct geist_tensor *w,
                                                        struct geist_tensor       *y) {
    return metal_q6k_linear(be, x, w, y, false);
}

[[nodiscard]] static enum geist_status metal_matmul_q6k(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        const struct geist_tensor *w,
                                                        struct geist_tensor       *y) {
    return metal_q6k_linear(be, x, w, y, true);
}

[[nodiscard]] static enum geist_status metal_rope_apply(struct geist_backend      *be,
                                                        struct geist_tensor       *x,
                                                        const struct geist_tensor *cos,
                                                        const struct geist_tensor *sin) {

    if (be == nullptr || be->state == nullptr || x == nullptr || cos == nullptr || sin == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t rows = 0, heads = 0, head_dim = 0, x_offset = 0;
    size_t cos_rows = 0, cos_cols = 0, cos_offset = 0, cos_stride = 0;
    size_t sin_rows = 0, sin_cols = 0, sin_offset = 0, sin_stride = 0;
    if (!metal_tensor_is_f32_3d(x, &rows, &heads, &head_dim, &x_offset) ||
        !metal_tensor_is_f32_matrix(cos, &cos_rows, &cos_cols, &cos_offset, &cos_stride) ||
        !metal_tensor_is_f32_matrix(sin, &sin_rows, &sin_cols, &sin_offset, &sin_stride)) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "metal rope_apply: tensors must be F32 DENSE x[seq,heads,dim]");
        return GEIST_E_UNSUPPORTED;
    }
    /* A cos/sin row is as wide as the ROTATED block, which may be narrower
     * than the head (qwen35 rotates 64 of 256 — #432). */
    if (head_dim == 0 || (head_dim % 2u) != 0 || cos_rows != rows || sin_rows != rows ||
        cos_cols == 0 || (cos_cols % 2u) != 0 || cos_cols > head_dim || sin_cols != cos_cols ||
        cos_stride != cos_cols || sin_stride != cos_cols) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "metal rope_apply: shape mismatch");
        return GEIST_E_INVALID_ARG;
    }
    if (rows > UINT32_MAX || heads > UINT32_MAX || head_dim > UINT32_MAX || x_offset > UINT32_MAX ||
        cos_offset > UINT32_MAX || sin_offset > UINT32_MAX || rows > UINT32_MAX / heads ||
        rows * heads > UINT32_MAX / (cos_cols / 2u) || x->buffer->owner != be->state ||
        cos->buffer->owner != be->state || sin->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = metal_ensure_attention_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state            *st     = be->state;
    const struct metal_rope_params params = {
            .rows            = (uint32_t) rows,
            .heads           = (uint32_t) heads,
            .head_dim        = (uint32_t) head_dim,
            .x_offset        = (uint32_t) x_offset,
            .cos_offset      = (uint32_t) cos_offset,
            .sin_offset      = (uint32_t) sin_offset,
            .x_row_stride    = (uint32_t) (heads * head_dim),
            .rope_row_stride = (uint32_t) cos_stride,
            .rope_row_offset = 0,
    };
    if (metal_seq_mine(st)) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal rope_apply: sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_rope_rows(st, metal_sequence_encoder(st), x, cos, sin, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);

    void *cmd = nullptr;
    void *enc = metal_standalone_begin(be, "rope_apply", &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_rope_rows(st, enc, x, cos, sin, &params);
    return metal_standalone_end(be, cmd, enc, "rope_apply");
}

/* The prefill flash kernel for a head_dim, or nullptr when none is
 * instantiated (head_dim not a multiple of 32, above 256 not of 64, or
 * above 512) or the attention tables are not built. */
static void *metal_flash_pipeline(const struct metal_state *st, size_t head_dim) {
    if (head_dim == 0u || head_dim % 32u != 0u || head_dim / 32u > METAL_FLASH_SLOTS) {
        return nullptr;
    }
    return st->attention_flash_pipeline[head_dim / 32u - 1u];
}

/* The f32-KV flash (prefill) path below converts the live K/V rows into
 * this f16 staging on every call. #64 kept it: per call it cost 1-5 % of
 * the old flash kernel at a 320-row chunk, while a flash kernel reading the
 * F32 cache directly measured 11-14 % slower on Bonsai 2 27B (decode
 * reads the F32 cache directly). Against the 4-5x faster kernel of #91 the
 * F32-KV call is 0-12 % slower than the F16-KV one (Bonsai and Gemma 4 E2B
 * shapes, 1k-8k rows), the staging's share. It is sized for every row of the cache k
 * views (the session's context), so a growing KV length never reallocates
 * it mid-session (AGENT.md §3). A larger cache on this backend (another
 * session or model) grows it once more. */
[[nodiscard]] static enum geist_status metal_attn_kvf16_reserve(struct geist_backend      *be,
                                                                struct metal_state        *st,
                                                                const struct geist_tensor *k,
                                                                size_t                     k_off) {
    const size_t cap = (k->buffer->bytes / sizeof(float) - k_off) * sizeof(uint16_t);
    if (st->attn_kvf16_capacity >= cap) {
        return GEIST_OK;
    }
    metal_buffer_destroy_internal(be, st->attn_kf16_buffer);
    metal_buffer_destroy_internal(be, st->attn_vf16_buffer);
    st->attn_kf16_buffer    = nullptr;
    st->attn_vf16_buffer    = nullptr;
    st->attn_kvf16_capacity = 0;
    enum geist_status s =
            metal_new_buffer(be, cap, GEIST_BUFFER_SCRATCH, 0, true, &st->attn_kf16_buffer);
    if (s == GEIST_OK) {
        s = metal_new_buffer(be, cap, GEIST_BUFFER_SCRATCH, 0, true, &st->attn_vf16_buffer);
    }
    if (s != GEIST_OK) {
        metal_buffer_destroy_internal(be, st->attn_kf16_buffer);
        st->attn_kf16_buffer = nullptr;
        return s;
    }
    st->attn_kvf16_capacity = cap;
    return GEIST_OK;
}

/* rows>1 fast path: on an F32 cache, convert K/V to persistent f16
 * staging first (one kv_append_rows_f16 dispatch); then the prefill flash
 * kernel for the head_dim instead of the scalar kernel. Serial-encoder ordering makes the
 * staging reuse across layers safe. */
[[nodiscard]] static enum geist_status metal_attention_flash_kv(struct geist_backend      *be,
                                                                struct metal_state        *st,
                                                                const struct geist_tensor *q,
                                                                const struct geist_tensor *k,
                                                                const struct geist_tensor *value,
                                                                struct geist_tensor       *out,
                                                                size_t                     q_rows,
                                                                size_t                     k_rows,
                                                                size_t                     q_heads,
                                                                size_t                     k_heads,
                                                                size_t                     head_dim,
                                                                size_t                     q_offset,
                                                                size_t sliding_window,
                                                                size_t q_off,
                                                                size_t k_off,
                                                                size_t v_off,
                                                                size_t out_off,
                                                                bool   kv_native_f16) {

    const size_t kv_out = k_heads * head_dim;
    const size_t elems  = k_rows * kv_out;
    if (!kv_native_f16) {
        const enum geist_status rs = metal_attn_kvf16_reserve(be, st, k, k_off);
        if (rs != GEIST_OK) {
            return rs;
        }
    }
    struct {
        uint32_t elems, kv_out, k_offset, v_offset, k_cache_offset, v_cache_offset, q_position;
    } ap = {(uint32_t) elems, (uint32_t) kv_out, (uint32_t) k_off, (uint32_t) v_off, 0u, 0u, 0u};
    struct {
        uint32_t rows, kv_len, qh, kvh, hd, qpos, sw, qo, kco, vco, yo;
    } fp       = {(uint32_t) q_rows,
                  (uint32_t) k_rows,
                  (uint32_t) q_heads,
                  (uint32_t) k_heads,
                  (uint32_t) head_dim,
                  (uint32_t) q_offset,
                  (uint32_t) sliding_window,
                  (uint32_t) q_off,
                  kv_native_f16 ? (uint32_t) k_off : 0u,
                  kv_native_f16 ? (uint32_t) v_off : 0u,
                  (uint32_t) out_off};
    void *kf16 = kv_native_f16 ? k->buffer->buffer : st->attn_kf16_buffer->buffer;
    void *vf16 = kv_native_f16 ? value->buffer->buffer : st->attn_vf16_buffer->buffer;

    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);
    void                                              *cmd  = nullptr;
    void                                              *enc  = metal_op_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    if (!kv_native_f16) {
        metal_msg_send_set_pipeline(st, enc, st->kv_append_rows_f16_pipeline);
        metal_msg_send_set_buffer(st, enc, k->buffer->buffer, k->buffer->base_off, 0);
        metal_msg_send_set_buffer(st, enc, value->buffer->buffer, value->buffer->base_off, 1);
        metal_msg_send_set_buffer(
                st, enc, st->attn_kf16_buffer->buffer, st->attn_kf16_buffer->base_off, 2);
        metal_msg_send_set_buffer(
                st, enc, st->attn_vf16_buffer->buffer, st->attn_vf16_buffer->base_off, 3);
        metal_msg_send_set_bytes(st, enc, &ap, sizeof(ap), 4);
        const struct metal_size cgroups  = {(elems + 255u) / 256u, 1, 1};
        const struct metal_size cthreads = {256, 1, 1};
        metal_msg_send_dispatch(st, enc, cgroups, cthreads);
    }

    /* 16 query rows per threadgroup; 4 simdgroups up to head_dim 256, 8
     * above (the Gemma global layers, head_dim 512). */
    metal_msg_send_set_pipeline(st, enc, metal_flash_pipeline(st, head_dim));
    metal_msg_send_set_buffer(st, enc, q->buffer->buffer, q->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, kf16, 0, 1);
    metal_msg_send_set_buffer(st, enc, vf16, 0, 2);
    metal_msg_send_set_buffer(st, enc, out->buffer->buffer, out->buffer->base_off, 3);
    metal_msg_send_set_bytes(st, enc, &fp, sizeof(fp), 4);
    const struct metal_size fgroups  = {(q_rows + 15u) / 16u, q_heads, 1};
    const struct metal_size fthreads = {head_dim > 256u ? 256u : 128u, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ATTENTION_ROWS, fgroups);
    metal_msg_send_dispatch(st, enc, fgroups, fthreads);

    return metal_op_end(be, cmd, enc, "flash attention");
}

static void *metal_decv_pipeline(const struct metal_state *st, size_t head_dim, bool f16) {
    if (head_dim == 0u || head_dim % 32u != 0u || head_dim / 32u > METAL_FLASH_SLOTS) {
        return nullptr;
    }
    return f16 ? st->attention_decv_f16_pipeline[head_dim / 32u - 1u]
               : st->attention_decv_f32kv_pipeline[head_dim / 32u - 1u];
}

/* rows==1 (decode) fast path: the split-KV decode kernel — the scalar
 * kernel uses one threadgroup per head (8 tgs on a 32-core GPU) and its
 * cost grows linearly with kv_len. An F32 cache is read by the _f32kv
 * variant, which rounds each element to half as it loads it: the same
 * result as staging the cache to f16 first, without converting every live
 * row on every token (#64). */
/* The counters of the kernels whose last threadgroup finishes the op
 * (#196): one per decode-attention head, then METAL_COUNT_PLE_TAIL.
 * Host-visible, zeroed once; every dispatch leaves its counter at zero. */
[[nodiscard]] static enum geist_status metal_ensure_counts(struct geist_backend *be) {
    struct metal_state *st = be->state;
    if (st->attn_dec_count_buffer != nullptr) {
        return GEIST_OK;
    }
    const size_t      bytes = METAL_COUNT_SLOTS * sizeof(uint32_t);
    enum geist_status s     = metal_new_buffer(
            be, bytes, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE, true, &st->attn_dec_count_buffer);
    if (s == GEIST_OK) {
        memset(st->attn_dec_count_buffer->mapped, 0, bytes);
    }
    return s;
}

[[nodiscard]] static enum geist_status metal_attention_dec_kv(struct geist_backend      *be,
                                                              struct metal_state        *st,
                                                              const struct geist_tensor *q,
                                                              const struct geist_tensor *k,
                                                              const struct geist_tensor *value,
                                                              struct geist_tensor       *out,
                                                              size_t                     k_rows,
                                                              size_t                     q_heads,
                                                              size_t                     k_heads,
                                                              size_t                     head_dim,
                                                              size_t                     q_offset,
                                                              size_t sliding_window,
                                                              size_t q_off,
                                                              size_t k_off,
                                                              size_t v_off,
                                                              size_t out_off,
                                                              bool   kv_native_f16) {

    uint32_t window = (uint32_t) q_offset + 1u;
    if ((uint32_t) k_rows < window) {
        window = (uint32_t) k_rows;
    }
    if (sliding_window > 0u && (uint32_t) sliding_window < window) {
        window = (uint32_t) sliding_window;
    }
    /* The 32-key-block kernel where one is built for this head_dim and the
     * offsets keep its float4/half4 loads aligned: one split per block it
     * touches, up to 32. The per-key kernels otherwise, 16 splits at most. */
    void *block_pipeline = metal_decv_pipeline(st, head_dim, kv_native_f16);
    if ((q_off | k_off | v_off) % 4u != 0u) {
        block_pipeline = nullptr;
    }
    uint32_t nsplit = window / 32u;
    if (block_pipeline != nullptr) {
        const uint32_t hi = (uint32_t) q_offset < (uint32_t) k_rows ? (uint32_t) q_offset
                                                                    : (uint32_t) k_rows - 1u;
        nsplit            = hi / 32u - (hi + 1u - window) / 32u + 1u;
    }
    if (nsplit < 1u) {
        nsplit = 1u;
    }
    if (nsplit > (block_pipeline != nullptr ? 32u : 16u)) {
        nsplit = block_pipeline != nullptr ? 32u : 16u;
    }
    const size_t partial_bytes = q_heads * 32u * (head_dim + 2u) * sizeof(float);
    if (st->attn_dec_partials_buffer != nullptr && st->attn_dec_partials_capacity < partial_bytes) {
        metal_buffer_destroy_internal(be, st->attn_dec_partials_buffer);
        st->attn_dec_partials_buffer   = nullptr;
        st->attn_dec_partials_capacity = 0;
    }
    if (st->attn_dec_partials_buffer == nullptr) {
        if (metal_new_buffer(be,
                             partial_bytes,
                             GEIST_BUFFER_SCRATCH,
                             GEIST_MEMORY_DEVICE,
                             false,
                             &st->attn_dec_partials_buffer) != GEIST_OK) {
            return GEIST_E_BACKEND;
        }
        st->attn_dec_partials_capacity = partial_bytes;
    }
    /* The block kernel merges the splits itself (#196): one dispatch per
     * decode attention instead of two. */
    const bool fused_combine = block_pipeline != nullptr && q_heads <= METAL_ATTN_DEC_MAX_HEADS &&
                               !st->attn_dec_two_pass;
    if (fused_combine && metal_ensure_counts(be) != GEIST_OK) {
        return GEIST_E_BACKEND;
    }
    struct {
        uint32_t rows, kv_len, qh, kvh, hd, qpos, sw, qo, kco, vco, yo;
    } fp = {1u,
            (uint32_t) k_rows,
            (uint32_t) q_heads,
            (uint32_t) k_heads,
            (uint32_t) head_dim,
            (uint32_t) q_offset,
            (uint32_t) sliding_window,
            (uint32_t) q_off,
            (uint32_t) k_off,
            (uint32_t) v_off,
            (uint32_t) out_off};
    /* The F16 cache binds at offset 0, as before; the F32 one at its
     * base, as the staging kernel bound it. */
    const size_t k_base = kv_native_f16 ? 0u : k->buffer->base_off;
    const size_t v_base = kv_native_f16 ? 0u : value->buffer->base_off;

    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);
    void                                              *cmd  = nullptr;
    void                                              *enc  = metal_op_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    const struct metal_size threads256 = {256, 1, 1};

    /* head_dim <= 256: 8-chunk lane unroll; 256 < hd <= 512 (gemma-3n
     * full-attention layers): the 16-chunk dec512 variant. */
    const bool wide     = head_dim > 256u;
    void      *pipeline = block_pipeline != nullptr
                                  ? block_pipeline
                                  : (kv_native_f16 ? (wide ? st->attention_dec512_f16_pipeline
                                                           : st->attention_dec_f16_pipeline)
                                                   : (wide ? st->attention_dec512_f32kv_pipeline
                                                           : st->attention_dec_f32kv_pipeline));
    metal_msg_send_set_pipeline(st, enc, pipeline);
    metal_msg_send_set_buffer(st, enc, q->buffer->buffer, q->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, k->buffer->buffer, k_base, 1);
    metal_msg_send_set_buffer(st, enc, value->buffer->buffer, v_base, 2);
    metal_msg_send_set_buffer(st,
                              enc,
                              st->attn_dec_partials_buffer->buffer,
                              st->attn_dec_partials_buffer->base_off,
                              3);
    metal_msg_send_set_bytes(st, enc, &fp, sizeof(fp), 4);
    /* struct SP: the per-key kernels read only ns. */
    const uint32_t sp[3] = {nsplit, fused_combine ? 1u : 0u, (uint32_t) out_off};
    metal_msg_send_set_bytes(st, enc, sp, sizeof(sp), 5);
    if (fused_combine) {
        metal_msg_send_set_buffer(
                st, enc, st->attn_dec_count_buffer->buffer, st->attn_dec_count_buffer->base_off, 6);
        metal_msg_send_set_buffer(st, enc, out->buffer->buffer, out->buffer->base_off, 7);
    }
    const struct metal_size dgroups   = {nsplit, q_heads, 1};
    const struct metal_size threads32 = {32, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ATTENTION_ROWS, dgroups);
    metal_msg_send_dispatch(st, enc, dgroups, block_pipeline != nullptr ? threads32 : threads256);
    if (fused_combine) {
        return metal_op_end(be, cmd, enc, "decode attention");
    }

    const uint32_t cb[4] = {(uint32_t) q_heads, (uint32_t) head_dim, nsplit, (uint32_t) out_off};
    metal_msg_send_set_pipeline(st, enc, st->attention_dec_combine_pipeline);
    metal_msg_send_set_buffer(st,
                              enc,
                              st->attn_dec_partials_buffer->buffer,
                              st->attn_dec_partials_buffer->base_off,
                              0);
    metal_msg_send_set_buffer(st, enc, out->buffer->buffer, out->buffer->base_off, 1);
    metal_msg_send_set_bytes(st, enc, cb, sizeof(cb), 2);
    const struct metal_size ggroups   = {(head_dim + 63u) / 64u, q_heads, 1};
    const struct metal_size threads64 = {64, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ATTENTION_COMBINE, ggroups);
    metal_msg_send_dispatch(st, enc, ggroups, threads64);

    return metal_op_end(be, cmd, enc, "decode attention");
}

/* Fused two-weight matvec (vtbl slot linear_t_pair): same structure as
 * ffn_gate_up but with a raw two-output epilogue — one activation pass
 * feeds both weights (used for the k/v projections, which share shape).
 * rows==1 + Q4_K + equal shapes only. */
[[nodiscard]] static enum geist_status metal_linear_t_pair(struct geist_backend      *be,
                                                           const struct geist_tensor *x,
                                                           const struct geist_weight *w0,
                                                           const struct geist_tensor *t_w0,
                                                           const struct geist_weight *w1,
                                                           const struct geist_tensor *t_w1,
                                                           size_t                     m,
                                                           struct geist_tensor       *y0,
                                                           struct geist_tensor       *y1) {

    if (be == nullptr || be->state == nullptr || x == nullptr || w0 == nullptr || t_w0 == nullptr ||
        w1 == nullptr || t_w1 == nullptr || y0 == nullptr || y1 == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    if (m != 1u || w0->dtype != GEIST_DTYPE_Q4_K || w1->dtype != GEIST_DTYPE_Q4_K) {
        return GEIST_E_UNSUPPORTED;
    }
    size_t rows = 0, d_in = 0, x_off = 0, x_stride = 0;
    size_t a_out = 0, a_in = 0, a_woff = 0;
    size_t b_out = 0, b_in = 0, b_woff = 0;
    size_t y0_rows = 0, y0_cols = 0, y0_off = 0, y0_stride = 0;
    size_t y1_rows = 0, y1_cols = 0, y1_off = 0, y1_stride = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &d_in, &x_off, &x_stride) ||
        !metal_tensor_is_q4k_matrix(t_w0, &a_out, &a_in, &a_woff) ||
        !metal_tensor_is_q4k_matrix(t_w1, &b_out, &b_in, &b_woff) ||
        !metal_tensor_is_f32_rows(y0, &y0_rows, &y0_cols, &y0_off, &y0_stride) ||
        !metal_tensor_is_f32_rows(y1, &y1_rows, &y1_cols, &y1_off, &y1_stride) || rows != 1u ||
        a_in != d_in || b_in != d_in || b_out != a_out || (d_in % 256u) != 0u || y0_rows != rows ||
        y1_rows != rows || y0_cols != a_out || y1_cols != a_out || y0_stride != y1_stride) {
        return GEIST_E_UNSUPPORTED;
    }
    if (d_in > UINT32_MAX || a_out > UINT32_MAX || x_off > UINT32_MAX || a_woff > UINT32_MAX ||
        b_woff > UINT32_MAX || y0_off > UINT32_MAX || y1_off > UINT32_MAX ||
        x->buffer->owner != be->state || t_w0->buffer->owner != be->state ||
        t_w1->buffer->owner != be->state || y0->buffer->owner != be->state ||
        y1->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st = be->state;
    if (st->q4k_pair_n4_pipeline == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    const struct metal_q4k_gate_up_params params = {
            .n_in               = (uint32_t) d_in,
            .n_out              = (uint32_t) a_out,
            .rows               = (uint32_t) rows,
            .blocks_per_row     = (uint32_t) (d_in / 256u),
            .x_offset           = (uint32_t) x_off,
            .gate_w_byte_offset = (uint32_t) a_woff,
            .up_w_byte_offset   = (uint32_t) b_woff,
            .gate_y_offset      = (uint32_t) y0_off,
            .up_y_offset        = (uint32_t) y1_off,
            .x_row_stride       = (uint32_t) x_stride,
            .y_row_stride       = (uint32_t) y0_stride,
    };
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);
    void                                              *cmd  = nullptr;
    void                                              *enc  = metal_op_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_msg_send_set_pipeline(st, enc, st->q4k_pair_n4_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, t_w0->buffer->buffer, t_w0->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, t_w1->buffer->buffer, t_w1->buffer->base_off, 2);
    metal_msg_send_set_buffer(st, enc, y0->buffer->buffer, y0->buffer->base_off, 3);
    metal_msg_send_set_buffer(st, enc, y1->buffer->buffer, y1->buffer->base_off, 4);
    metal_msg_send_set_bytes(st, enc, &params, sizeof(params), 5);
    const struct metal_size groups  = {(uint32_t) ((a_out + 3u) / 4u), (uint32_t) rows, 1};
    const struct metal_size threads = {METAL_Q4K_N4_THREADS, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_Q4K_QK_BASE, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
    return metal_op_end(be, cmd, enc, nullptr);
}

/* A weight the fused decode matvecs take (#17): Q4_K, Q6_K or Q8_0, with
 * its shape, byte offset, blocks per row and the output rows one
 * threadgroup of its matvec kernel covers. False for anything else. */
struct metal_mv_weight {
    enum geist_dtype dtype;
    size_t           n_out;
    size_t           n_in;
    size_t           offset;
    uint32_t         blocks_per_row;
    uint32_t         tg_rows;
};

static bool metal_mv_weight_of(const struct geist_tensor *t, struct metal_mv_weight *out) {
    if (t == nullptr) {
        return false;
    }
    size_t block = 0;
    bool   ok    = false;
    switch ((enum geist_dtype) t->dtype) {
    case GEIST_DTYPE_Q4_K:
        ok           = metal_tensor_is_q4k_matrix(t, &out->n_out, &out->n_in, &out->offset);
        block        = METAL_Q4K_BLOCK_ELEMS;
        out->tg_rows = 4u;
        break;
    case GEIST_DTYPE_Q6_K:
        ok           = metal_tensor_is_q6k_matrix(t, &out->n_out, &out->n_in, &out->offset);
        block        = METAL_Q6K_BLOCK_ELEMS;
        out->tg_rows = 4u;
        break;
    case GEIST_DTYPE_Q8_0:
        ok = metal_tensor_is_q40_q80_matrix(
                t, GEIST_DTYPE_Q8_0, &out->n_out, &out->n_in, &out->offset);
        block        = 32u;
        out->tg_rows = 8u;
        break;
    default:
        return false;
    }
    if (!ok || out->n_in > UINT32_MAX || out->n_out > UINT32_MAX || out->offset > UINT32_MAX) {
        return false;
    }
    out->dtype          = (enum geist_dtype) t->dtype;
    out->blocks_per_row = (uint32_t) (out->n_in / block);
    return true;
}

/* Fused FFN gate+up matvec (vtbl slots ffn_gate_up and ffn_gate_up_silu):
 * llama mul_mv structure, both weights against one activation pass,
 * act(g)*u written directly, act gelu or silu. rows==1 (decode), Q4_K or
 * Q8_0; prefill keeps the GEMMs. */
[[nodiscard]] static enum geist_status metal_gate_up(struct geist_backend      *be,
                                                     const struct geist_tensor *x,
                                                     const struct geist_tensor *gate_w,
                                                     const struct geist_tensor *up_w,
                                                     struct geist_tensor       *y,
                                                     bool                       silu) {

    if (be == nullptr || be->state == nullptr || x == nullptr || gate_w == nullptr ||
        up_w == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t                 rows = 0, d_in = 0, x_off = 0, x_stride = 0;
    size_t                 y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    struct metal_mv_weight g = {0}, u = {0};
    if (!metal_tensor_is_f32_rows(x, &rows, &d_in, &x_off, &x_stride) ||
        !metal_mv_weight_of(gate_w, &g) || !metal_mv_weight_of(up_w, &u) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) ||
        rows != 1u || /* decode only: matvec-structured kernel */
        g.dtype != u.dtype || g.dtype == GEIST_DTYPE_Q6_K || g.n_in != d_in || u.n_in != d_in ||
        u.n_out != g.n_out || y_rows != rows || y_cols != g.n_out) {
        return GEIST_E_UNSUPPORTED;
    }
    if (d_in > UINT32_MAX || x_off > UINT32_MAX || y_off > UINT32_MAX ||
        x->buffer->owner != be->state || gate_w->buffer->owner != be->state ||
        up_w->buffer->owner != be->state || y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st  = be->state;
    const bool          q4k = g.dtype == GEIST_DTYPE_Q4_K;
    void *const         pipeline =
            q4k ? (silu ? st->q4k_gate_up_silu_n4_pipeline : st->q4k_gate_up_n4_pipeline)
                : (silu ? st->q80_gate_up_silu_n4_pipeline : st->q80_gate_up_n4_pipeline);
    if (pipeline == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    const struct metal_q4k_gate_up_params params = {
            .n_in               = (uint32_t) d_in,
            .n_out              = (uint32_t) g.n_out,
            .rows               = (uint32_t) rows,
            .blocks_per_row     = g.blocks_per_row,
            .x_offset           = (uint32_t) x_off,
            .gate_w_byte_offset = (uint32_t) g.offset,
            .up_w_byte_offset   = (uint32_t) u.offset,
            .gate_y_offset      = (uint32_t) y_off,
            .up_y_offset        = (uint32_t) y_off, /* unused by the epilogue kernel */
            .x_row_stride       = (uint32_t) x_stride,
            .y_row_stride       = (uint32_t) y_stride,
    };
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);
    void                                              *cmd  = nullptr;
    void                                              *enc  = metal_op_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_msg_send_set_pipeline(st, enc, pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, gate_w->buffer->buffer, gate_w->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, up_w->buffer->buffer, up_w->buffer->base_off, 2);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 3);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 4);
    metal_msg_send_set_bytes(st, enc, &params, sizeof(params), 5);
    const struct metal_size groups = {
            (uint32_t) ((g.n_out + g.tg_rows - 1u) / g.tg_rows), (uint32_t) rows, 1};
    const struct metal_size threads = {METAL_Q4K_N4_THREADS, 1, 1};
    metal_profile_add_dispatch(st,
                               q4k ? METAL_PROFILE_DISPATCH_Q4K_GATE_UP_N4
                                   : METAL_PROFILE_DISPATCH_Q80_GATE_UP,
                               groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
    return metal_op_end(be, cmd, enc, nullptr);
}

[[nodiscard]] static enum geist_status metal_ffn_gate_up(struct geist_backend      *be,
                                                         const struct geist_tensor *x,
                                                         const struct geist_tensor *gate_w,
                                                         const struct geist_tensor *up_w,
                                                         struct geist_tensor       *y) {
    return metal_gate_up(be, x, gate_w, up_w, y, false);
}

[[nodiscard]] static enum geist_status metal_ffn_gate_up_silu(struct geist_backend      *be,
                                                              const struct geist_tensor *x,
                                                              const struct geist_tensor *gate_w,
                                                              const struct geist_tensor *up_w,
                                                              struct geist_tensor       *y) {
    return metal_gate_up(be, x, gate_w, up_w, y, true);
}

/* The mv3 kernel for a q/k/v dtype triple; nullptr where none is built. */
static void *metal_mv3_pipeline(const struct metal_state *st,
                                enum geist_dtype          a,
                                enum geist_dtype          b,
                                enum geist_dtype          c) {
    if (a == GEIST_DTYPE_Q4_K && b == GEIST_DTYPE_Q4_K && c == GEIST_DTYPE_Q4_K) {
        return st->mv3_q4k_q4k_q4k_pipeline;
    }
    if (a == GEIST_DTYPE_Q4_K && b == GEIST_DTYPE_Q4_K && c == GEIST_DTYPE_Q6_K) {
        return st->mv3_q4k_q4k_q6k_pipeline;
    }
    if (a == GEIST_DTYPE_Q8_0 && b == GEIST_DTYPE_Q8_0 && c == GEIST_DTYPE_Q8_0) {
        return st->mv3_q80_q80_q80_pipeline;
    }
    return nullptr;
}

/* Fused three-weight decode linear (vtbl slot linear_t_triple, #17): q, k
 * and v of a decode step in one dispatch. The grid is the three matvecs'
 * threadgroups side by side and each runs its own weight's matvec body, so
 * the weights may differ in dtype and shape (Gemma 4 E2B: q, k Q4_K, v
 * Q6_K) and every output keeps the matvec kernel's bits. m == 1 and the
 * dtype triples metal_mv3_pipeline knows; GEIST_E_UNSUPPORTED otherwise,
 * before anything is encoded. */
[[nodiscard]] static enum geist_status metal_linear_t_triple(struct geist_backend      *be,
                                                             const struct geist_tensor *x,
                                                             const struct geist_weight *w0,
                                                             const struct geist_tensor *t_w0,
                                                             const struct geist_weight *w1,
                                                             const struct geist_tensor *t_w1,
                                                             const struct geist_weight *w2,
                                                             const struct geist_tensor *t_w2,
                                                             size_t                     m,
                                                             struct geist_tensor       *y0,
                                                             struct geist_tensor       *y1,
                                                             struct geist_tensor       *y2) {
    if (be == nullptr || be->state == nullptr || x == nullptr || w0 == nullptr || w1 == nullptr ||
        w2 == nullptr || y0 == nullptr || y1 == nullptr || y2 == nullptr || m != 1u) {
        return GEIST_E_UNSUPPORTED;
    }
    const struct geist_tensor *const tw[3] = {t_w0, t_w1, t_w2};
    const struct geist_weight *const ww[3] = {w0, w1, w2};
    struct geist_tensor *const       ty[3] = {y0, y1, y2};
    size_t                           rows = 0, d_in = 0, x_off = 0, x_stride = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &d_in, &x_off, &x_stride) || rows != 1u ||
        x_off > UINT32_MAX || x->buffer->owner != be->state) {
        return GEIST_E_UNSUPPORTED;
    }
    struct metal_mv_weight  mw[3];
    struct metal_q4k_params p[3];
    uint32_t                groups[3];
    for (size_t i = 0; i < 3; i++) {
        size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
        if (!metal_mv_weight_of(tw[i], &mw[i]) || (uint16_t) mw[i].dtype != ww[i]->dtype ||
            mw[i].n_in != d_in || mw[i].n_out < 4u ||
            !metal_tensor_is_f32_rows(ty[i], &y_rows, &y_cols, &y_off, &y_stride) || y_rows != 1u ||
            y_cols != mw[i].n_out || y_off > UINT32_MAX || tw[i]->buffer->owner != be->state ||
            ty[i]->buffer->owner != be->state) {
            return GEIST_E_UNSUPPORTED;
        }
        p[i] = (struct metal_q4k_params) {
                .n_in           = (uint32_t) d_in,
                .n_out          = (uint32_t) mw[i].n_out,
                .rows           = 1u,
                .blocks_per_row = mw[i].blocks_per_row,
                .x_offset       = (uint32_t) x_off,
                .w_byte_offset  = (uint32_t) mw[i].offset,
                .y_offset       = (uint32_t) y_off,
                .x_row_stride   = (uint32_t) d_in,
                .y_row_stride   = (uint32_t) mw[i].n_out,
        };
        groups[i] = (uint32_t) ((mw[i].n_out + mw[i].tg_rows - 1u) / mw[i].tg_rows);
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st       = be->state;
    void *const         pipeline = metal_mv3_pipeline(st, mw[0].dtype, mw[1].dtype, mw[2].dtype);
    if (pipeline == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    /* struct T3 in metal_mv3_head */
    const struct {
        uint32_t                g1, g2;
        struct metal_q4k_params p0, p1, p2;
    } t3 = {groups[0], groups[0] + groups[1], p[0], p[1], p[2]};
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);
    void                                              *cmd  = nullptr;
    void                                              *enc  = metal_op_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_msg_send_set_pipeline(st, enc, pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    for (size_t i = 0; i < 3; i++) {
        metal_msg_send_set_buffer(
                st, enc, tw[i]->buffer->buffer, tw[i]->buffer->base_off, 1u + (uint32_t) i);
        metal_msg_send_set_buffer(
                st, enc, ty[i]->buffer->buffer, ty[i]->buffer->base_off, 4u + (uint32_t) i);
    }
    metal_msg_send_set_bytes(st, enc, &t3, sizeof(t3), 7);
    const struct metal_size grid    = {t3.g2 + groups[2], 1, 1};
    const struct metal_size threads = {METAL_Q4K_N4_THREADS, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_QKV3, grid);
    metal_msg_send_dispatch(st, enc, grid, threads);
    return metal_op_end(be, cmd, enc, nullptr);
}

/* Fused gemma attention q/k/v prep (vtbl slot): q_norm_rope_rows for q,
 * kv_norm_append_rows{,_f16} for k+v (norm + RoPE + cache append in one
 * dispatch): two dispatches for up to six decomposed ops. */
[[nodiscard]] static enum geist_status metal_attn_qkv_prep(struct geist_backend      *be,
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

    if (be == nullptr || be->state == nullptr || q == nullptr || q_norm_w == nullptr ||
        cos == nullptr || sin == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    const bool has_kv = k != nullptr;
    if (has_kv && (v == nullptr || k_norm_w == nullptr || v_norm_w == nullptr ||
                   k_cache == nullptr || v_cache == nullptr)) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, q_heads = 0, hd = 0, q_off = 0;
    size_t c_rows = 0, c_cols = 0, c_off = 0, c_stride = 0;
    size_t s_rows = 0, s_cols = 0, s_off = 0, s_stride = 0;
    size_t qw_n = 0, qw_off = 0;
    if (!metal_tensor_is_f32_3d(q, &rows, &q_heads, &hd, &q_off) ||
        !metal_tensor_is_f32_matrix(cos, &c_rows, &c_cols, &c_off, &c_stride) ||
        !metal_tensor_is_f32_matrix(sin, &s_rows, &s_cols, &s_off, &s_stride) ||
        !metal_tensor_is_f32_vector(q_norm_w, &qw_n, &qw_off) || hd == 0 || (hd % 2u) != 0 ||
        qw_n != hd || c_rows != rows || s_rows != rows || c_cols != hd || s_cols != hd ||
        c_stride != hd || s_stride != hd) {
        return GEIST_E_UNSUPPORTED;
    }
    size_t k_rows = 0, kv_heads = 0, k_hd = 0, k_off = 0;
    size_t v_rows = 0, v_heads = 0, v_hd = 0, v_off = 0;
    size_t kc_rows = 0, kc_heads = 0, kc_hd = 0, kc_off = 0;
    size_t vc_rows = 0, vc_heads = 0, vc_hd = 0, vc_off = 0;
    size_t kw_n = 0, kw_off = 0, vw_n = 0, vw_off = 0;
    bool   cache_f16 = false;
    if (has_kv) {
        if (!metal_tensor_is_f32_3d(k, &k_rows, &kv_heads, &k_hd, &k_off) ||
            !metal_tensor_is_f32_3d(v, &v_rows, &v_heads, &v_hd, &v_off) ||
            !metal_tensor_is_f32_vector(k_norm_w, &kw_n, &kw_off) ||
            !metal_tensor_is_f32_vector(v_norm_w, &vw_n, &vw_off) || k_rows != rows ||
            v_rows != rows || v_heads != kv_heads || k_hd != hd || v_hd != hd || kw_n != hd ||
            vw_n != hd) {
            return GEIST_E_UNSUPPORTED;
        }
        const bool cache_f32 =
                metal_tensor_is_f32_3d(k_cache, &kc_rows, &kc_heads, &kc_hd, &kc_off) &&
                metal_tensor_is_f32_3d(v_cache, &vc_rows, &vc_heads, &vc_hd, &vc_off);
        cache_f16 = !cache_f32 &&
                    metal_tensor_is_f16_3d(k_cache, &kc_rows, &kc_heads, &kc_hd, &kc_off) &&
                    metal_tensor_is_f16_3d(v_cache, &vc_rows, &vc_heads, &vc_hd, &vc_off);
        if ((!cache_f32 && !cache_f16) || kc_heads != kv_heads || vc_heads != kv_heads ||
            kc_hd != hd || vc_hd != hd || kc_rows < q_position + rows ||
            vc_rows < q_position + rows) {
            return GEIST_E_UNSUPPORTED;
        }
    }
    if (rows > UINT32_MAX || q_heads > UINT32_MAX || hd > UINT32_MAX || q_off > UINT32_MAX ||
        c_off > UINT32_MAX || s_off > UINT32_MAX || qw_off > UINT32_MAX ||
        q_position > UINT32_MAX || q->buffer->owner != be->state ||
        cos->buffer->owner != be->state || sin->buffer->owner != be->state ||
        q_norm_w->buffer->owner != be->state ||
        (has_kv && (k_off > UINT32_MAX || v_off > UINT32_MAX || kc_off > UINT32_MAX ||
                    vc_off > UINT32_MAX || kw_off > UINT32_MAX || vw_off > UINT32_MAX ||
                    k->buffer->owner != be->state || v->buffer->owner != be->state ||
                    k_norm_w->buffer->owner != be->state || v_norm_w->buffer->owner != be->state ||
                    k_cache->buffer->owner != be->state || v_cache->buffer->owner != be->state))) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_attention_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st = be->state;
    void               *kv_pipeline =
            cache_f16 ? st->kv_norm_append_rows_f16_pipeline : st->kv_norm_append_rows_pipeline;
    if (st->q_norm_rope_rows_pipeline == nullptr || (has_kv && kv_pipeline == nullptr)) {
        return GEIST_E_UNSUPPORTED;
    }
    struct {
        uint32_t rows, heads, hd, xo, wo, co, so, xs, rs, ro;
        float    eps;
    } nr = {(uint32_t) rows,
            (uint32_t) q_heads,
            (uint32_t) hd,
            (uint32_t) q_off,
            (uint32_t) qw_off,
            (uint32_t) c_off,
            (uint32_t) s_off,
            (uint32_t) (q_heads * hd),
            (uint32_t) hd,
            0u,
            eps};
    struct {
        uint32_t rows, heads, hd, xo, wo, co, so, cao, xs, rs, ro, qp;
        float    eps;
    } kp = {(uint32_t) rows,
            (uint32_t) kv_heads,
            (uint32_t) hd,
            (uint32_t) k_off,
            (uint32_t) kw_off,
            (uint32_t) c_off,
            (uint32_t) s_off,
            (uint32_t) kc_off,
            (uint32_t) (kv_heads * hd),
            (uint32_t) hd,
            0u,
            (uint32_t) q_position,
            eps};
    struct {
        uint32_t rows, heads, hd, xo, wo, cao, xs, qp;
        float    eps;
    } vp = {(uint32_t) rows,
            (uint32_t) kv_heads,
            (uint32_t) hd,
            (uint32_t) v_off,
            (uint32_t) vw_off,
            (uint32_t) vc_off,
            (uint32_t) (kv_heads * hd),
            (uint32_t) q_position,
            eps};

    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);
    void                                              *cmd  = nullptr;
    void                                              *enc  = metal_op_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    const struct metal_size threads256 = {256, 1, 1};
    void *const             qkv_pipeline =
            cache_f16 ? st->qkv_norm_append_rows_f16_pipeline : st->qkv_norm_append_rows_pipeline;
    if (has_kv && qkv_pipeline != nullptr) {
        /* q and k/v in one dispatch (#196): the q heads, then the k/v heads */
        metal_msg_send_set_pipeline(st, enc, qkv_pipeline);
        metal_msg_send_set_buffer(st, enc, q->buffer->buffer, q->buffer->base_off, 0);
        metal_msg_send_set_buffer(st, enc, q_norm_w->buffer->buffer, q_norm_w->buffer->base_off, 1);
        metal_msg_send_set_buffer(st, enc, cos->buffer->buffer, cos->buffer->base_off, 2);
        metal_msg_send_set_buffer(st, enc, sin->buffer->buffer, sin->buffer->base_off, 3);
        metal_msg_send_set_bytes(st, enc, &nr, sizeof(nr), 4);
        metal_msg_send_set_buffer(st, enc, k->buffer->buffer, k->buffer->base_off, 5);
        metal_msg_send_set_buffer(st, enc, v->buffer->buffer, v->buffer->base_off, 6);
        metal_msg_send_set_buffer(st, enc, k_norm_w->buffer->buffer, k_norm_w->buffer->base_off, 7);
        metal_msg_send_set_buffer(st, enc, v_norm_w->buffer->buffer, v_norm_w->buffer->base_off, 8);
        metal_msg_send_set_buffer(st, enc, k_cache->buffer->buffer, k_cache->buffer->base_off, 9);
        metal_msg_send_set_buffer(st, enc, v_cache->buffer->buffer, v_cache->buffer->base_off, 10);
        metal_msg_send_set_bytes(st, enc, &kp, sizeof(kp), 11);
        metal_msg_send_set_bytes(st, enc, &vp, sizeof(vp), 12);
        const struct metal_size groups = {(uint32_t) rows, (uint32_t) (q_heads + kv_heads), 1};
        metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_QKV_NORM_APPEND, groups);
        metal_msg_send_dispatch(st, enc, groups, threads256);
        return metal_op_end(be, cmd, enc, nullptr);
    }
    metal_msg_send_set_pipeline(st, enc, st->q_norm_rope_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, q->buffer->buffer, q->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, q_norm_w->buffer->buffer, q_norm_w->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, cos->buffer->buffer, cos->buffer->base_off, 2);
    metal_msg_send_set_buffer(st, enc, sin->buffer->buffer, sin->buffer->base_off, 3);
    metal_msg_send_set_bytes(st, enc, &nr, sizeof(nr), 4);
    const struct metal_size qgroups = {(uint32_t) rows, (uint32_t) q_heads, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_Q_NORM_ROPE, qgroups);
    metal_msg_send_dispatch(st, enc, qgroups, threads256);

    if (has_kv) {
        metal_msg_send_set_pipeline(st, enc, kv_pipeline);
        metal_msg_send_set_buffer(st, enc, k->buffer->buffer, k->buffer->base_off, 0);
        metal_msg_send_set_buffer(st, enc, v->buffer->buffer, v->buffer->base_off, 1);
        metal_msg_send_set_buffer(st, enc, k_norm_w->buffer->buffer, k_norm_w->buffer->base_off, 2);
        metal_msg_send_set_buffer(st, enc, v_norm_w->buffer->buffer, v_norm_w->buffer->base_off, 3);
        metal_msg_send_set_buffer(st, enc, cos->buffer->buffer, cos->buffer->base_off, 4);
        metal_msg_send_set_buffer(st, enc, sin->buffer->buffer, sin->buffer->base_off, 5);
        metal_msg_send_set_buffer(st, enc, k_cache->buffer->buffer, k_cache->buffer->base_off, 6);
        metal_msg_send_set_buffer(st, enc, v_cache->buffer->buffer, v_cache->buffer->base_off, 7);
        metal_msg_send_set_bytes(st, enc, &kp, sizeof(kp), 8);
        metal_msg_send_set_bytes(st, enc, &vp, sizeof(vp), 9);
        const struct metal_size kvgroups = {(uint32_t) rows, (uint32_t) kv_heads, 1};
        metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_KV_NORM_APPEND, kvgroups);
        metal_msg_send_dispatch(st, enc, kvgroups, threads256);
    }

    return metal_op_end(be, cmd, enc, nullptr);
}

/* Fused Llama / Qwen3 attention prep (vtbl slot attn_rope_append, #17):
 * rope_append_rows{,_f16}, one dispatch for what were the q/k norms, the
 * two RoPEs, the q scale and the KV append. */
[[nodiscard]] static enum geist_status metal_attn_rope_append(struct geist_backend      *be,
                                                              struct geist_tensor       *q,
                                                              struct geist_tensor       *k,
                                                              const struct geist_tensor *v,
                                                              const struct geist_tensor *q_norm_w,
                                                              const struct geist_tensor *k_norm_w,
                                                              const struct geist_tensor *cos,
                                                              const struct geist_tensor *sin,
                                                              float                      eps,
                                                              float                      q_scale,
                                                              size_t                     q_position,
                                                              struct geist_tensor       *k_cache,
                                                              struct geist_tensor       *v_cache) {
    if (be == nullptr || be->state == nullptr || q == nullptr || k == nullptr || v == nullptr ||
        cos == nullptr || sin == nullptr || k_cache == nullptr || v_cache == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, q_heads = 0, hd = 0, q_off = 0;
    size_t k_rows = 0, kv_heads = 0, k_hd = 0, k_off = 0;
    size_t v_rows = 0, v_heads = 0, v_hd = 0, v_off = 0;
    size_t c_rows = 0, n_rot = 0, c_off = 0, c_stride = 0;
    size_t s_rows = 0, s_cols = 0, s_off = 0, s_stride = 0;
    size_t qw_n = 0, qw_off = 0, kw_n = 0, kw_off = 0;
    size_t kc_rows = 0, kc_heads = 0, kc_hd = 0, kc_off = 0;
    size_t vc_rows = 0, vc_heads = 0, vc_hd = 0, vc_off = 0;
    if (!metal_tensor_is_f32_3d(q, &rows, &q_heads, &hd, &q_off) ||
        !metal_tensor_is_f32_3d(k, &k_rows, &kv_heads, &k_hd, &k_off) ||
        !metal_tensor_is_f32_3d(v, &v_rows, &v_heads, &v_hd, &v_off) ||
        !metal_tensor_is_f32_matrix(cos, &c_rows, &n_rot, &c_off, &c_stride) ||
        !metal_tensor_is_f32_matrix(sin, &s_rows, &s_cols, &s_off, &s_stride) ||
        (q_norm_w != nullptr && !metal_tensor_is_f32_vector(q_norm_w, &qw_n, &qw_off)) ||
        (k_norm_w != nullptr && !metal_tensor_is_f32_vector(k_norm_w, &kw_n, &kw_off)) || hd == 0 ||
        hd > 512u || n_rot == 0 || (n_rot % 2u) != 0 || n_rot > hd || k_rows != rows ||
        v_rows != rows || v_heads != kv_heads || k_hd != hd || v_hd != hd || c_rows != rows ||
        s_rows != rows || s_cols != n_rot || c_stride != n_rot || s_stride != n_rot ||
        (q_norm_w != nullptr && qw_n != hd) || (k_norm_w != nullptr && kw_n != hd)) {
        return GEIST_E_UNSUPPORTED;
    }
    const bool cache_f32 = metal_tensor_is_f32_3d(k_cache, &kc_rows, &kc_heads, &kc_hd, &kc_off) &&
                           metal_tensor_is_f32_3d(v_cache, &vc_rows, &vc_heads, &vc_hd, &vc_off);
    const bool cache_f16 = !cache_f32 &&
                           metal_tensor_is_f16_3d(k_cache, &kc_rows, &kc_heads, &kc_hd, &kc_off) &&
                           metal_tensor_is_f16_3d(v_cache, &vc_rows, &vc_heads, &vc_hd, &vc_off);
    if ((!cache_f32 && !cache_f16) || kc_heads != kv_heads || vc_heads != kv_heads || kc_hd != hd ||
        vc_hd != hd || kc_rows < q_position + rows || vc_rows < q_position + rows) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || q_heads + kv_heads > UINT32_MAX || q_off > UINT32_MAX ||
        k_off > UINT32_MAX || v_off > UINT32_MAX || c_off > UINT32_MAX || s_off > UINT32_MAX ||
        qw_off > UINT32_MAX || kw_off > UINT32_MAX || kc_off > UINT32_MAX || vc_off > UINT32_MAX ||
        q_position > UINT32_MAX || q->buffer->owner != be->state || k->buffer->owner != be->state ||
        v->buffer->owner != be->state || cos->buffer->owner != be->state ||
        sin->buffer->owner != be->state || k_cache->buffer->owner != be->state ||
        v_cache->buffer->owner != be->state ||
        (q_norm_w != nullptr && q_norm_w->buffer->owner != be->state) ||
        (k_norm_w != nullptr && k_norm_w->buffer->owner != be->state)) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_attention_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st = be->state;
    void *const         pipeline =
            cache_f16 ? st->rope_append_rows_f16_pipeline : st->rope_append_rows_pipeline;
    if (pipeline == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    /* struct RA in metal_rope_append_source */
    const struct {
        uint32_t rows, qh, kvh, hd, nrot, qo, ko, vo, qwo, kwo, co, so, kco, vco, qp, qn, kn, qsc;
        float    eps, qs;
    } ra = {(uint32_t) rows,
            (uint32_t) q_heads,
            (uint32_t) kv_heads,
            (uint32_t) hd,
            (uint32_t) n_rot,
            (uint32_t) q_off,
            (uint32_t) k_off,
            (uint32_t) v_off,
            (uint32_t) qw_off,
            (uint32_t) kw_off,
            (uint32_t) c_off,
            (uint32_t) s_off,
            (uint32_t) kc_off,
            (uint32_t) vc_off,
            (uint32_t) q_position,
            q_norm_w != nullptr,
            k_norm_w != nullptr,
            q_scale != 1.0f,
            eps,
            q_scale};
    /* Unused weight slots bind q: the kernel reads them only under qn/kn. */
    const struct geist_tensor *qw = q_norm_w != nullptr ? q_norm_w : q;
    const struct geist_tensor *kw = k_norm_w != nullptr ? k_norm_w : q;

    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);
    void                                              *cmd  = nullptr;
    void                                              *enc  = metal_op_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_msg_send_set_pipeline(st, enc, pipeline);
    metal_msg_send_set_buffer(st, enc, q->buffer->buffer, q->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, k->buffer->buffer, k->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, v->buffer->buffer, v->buffer->base_off, 2);
    metal_msg_send_set_buffer(st, enc, qw->buffer->buffer, qw->buffer->base_off, 3);
    metal_msg_send_set_buffer(st, enc, kw->buffer->buffer, kw->buffer->base_off, 4);
    metal_msg_send_set_buffer(st, enc, cos->buffer->buffer, cos->buffer->base_off, 5);
    metal_msg_send_set_buffer(st, enc, sin->buffer->buffer, sin->buffer->base_off, 6);
    metal_msg_send_set_buffer(st, enc, k_cache->buffer->buffer, k_cache->buffer->base_off, 7);
    metal_msg_send_set_buffer(st, enc, v_cache->buffer->buffer, v_cache->buffer->base_off, 8);
    metal_msg_send_set_bytes(st, enc, &ra, sizeof(ra), 9);
    const struct metal_size groups  = {(uint32_t) rows, (uint32_t) (q_heads + kv_heads), 1};
    const struct metal_size threads = {256, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ROPE_APPEND, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
    return metal_op_end(be, cmd, enc, nullptr);
}

/* Fused gemma-3n PLE block (vtbl slot), decode fast path: two dispatches
 * (ple_gate_f32: gate GEMV + gelu*ple; ple_proj_norm_f32: proj GEMV +
 * rmsnorm + residual add) replace four decomposed ops. rows==1 and F32
 * weights only — the kernels are naive GEMVs; prefill keeps the mm_sg
 * GEMM path via the decomposed fallback. */
[[nodiscard]] static enum geist_status metal_ple_block(struct geist_backend      *be,
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

    if (be == nullptr || be->state == nullptr || x == nullptr || gate_w == nullptr ||
        ple_in == nullptr || proj_w == nullptr || res == nullptr || gate_scratch == nullptr ||
        proj_scratch == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, d_in = 0, x_off = 0, x_stride = 0;
    size_t g_rows = 0, g_cols = 0, g_off = 0, g_stride = 0;
    size_t p_rows = 0, p_cols = 0, p_off = 0, p_stride = 0;
    size_t r_rows = 0, r_cols = 0, r_off = 0, r_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    size_t ps_rows = 0, ps_cols = 0, ps_off = 0, ps_stride = 0;
    size_t hpl = 0, gw_in = 0, gw_off = 0, gw_stride = 0;
    size_t d_model = 0, pw_in = 0, pw_off = 0, pw_stride = 0;
    size_t nw_n = 0, nw_off = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &d_in, &x_off, &x_stride) ||
        !metal_tensor_is_f32_matrix(gate_w, &hpl, &gw_in, &gw_off, &gw_stride) ||
        !metal_tensor_is_f32_rows(ple_in, &p_rows, &p_cols, &p_off, &p_stride) ||
        !metal_tensor_is_f32_matrix(proj_w, &d_model, &pw_in, &pw_off, &pw_stride) ||
        gw_stride != gw_in || pw_stride != pw_in || /* kernels assume dense */
        !metal_tensor_is_f32_rows(res, &r_rows, &r_cols, &r_off, &r_stride) ||
        (norm_w != nullptr && !metal_tensor_is_f32_vector(norm_w, &nw_n, &nw_off)) ||
        !metal_tensor_is_f32_rows(gate_scratch, &g_rows, &g_cols, &g_off, &g_stride) ||
        !metal_tensor_is_f32_rows(proj_scratch, &ps_rows, &ps_cols, &ps_off, &ps_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) ||
        rows != 1u || /* decode only: naive GEMV kernels */
        gw_in != d_in || p_rows != rows || p_cols != hpl || g_rows != rows || g_cols != hpl ||
        pw_in != hpl || r_rows != rows || r_cols != d_model || ps_rows != rows ||
        ps_cols != d_model || y_rows != rows || y_cols != d_model ||
        (norm_w != nullptr && nw_n != d_model)) {
        return GEIST_E_UNSUPPORTED;
    }
    if (d_in > UINT32_MAX || hpl > UINT32_MAX || d_model > UINT32_MAX || x_off > UINT32_MAX ||
        gw_off > UINT32_MAX || p_off > UINT32_MAX || g_off > UINT32_MAX || pw_off > UINT32_MAX ||
        r_off > UINT32_MAX || nw_off > UINT32_MAX || y_off > UINT32_MAX ||
        x->buffer->owner != be->state || gate_w->buffer->owner != be->state ||
        ple_in->buffer->owner != be->state || proj_w->buffer->owner != be->state ||
        res->buffer->owner != be->state ||
        (norm_w != nullptr && norm_w->buffer->owner != be->state) ||
        gate_scratch->buffer->owner != be->state || y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st = be->state;
    if (st->f32_ple_gate_pipeline == nullptr || st->f32_ple_proj_norm_pipeline == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    struct {
        uint32_t ni, no, rows, xo, wo, po, yo, xs, ps, ys;
    } gp                                                    = {(uint32_t) d_in,
                                                               (uint32_t) hpl,
                                                               (uint32_t) rows,
                                                               (uint32_t) x_off,
                                                               (uint32_t) gw_off,
                                                               (uint32_t) p_off,
                                                               (uint32_t) g_off,
                                                               (uint32_t) x_stride,
                                                               (uint32_t) p_stride,
                                                               (uint32_t) g_stride};
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);
    void                                              *cmd  = nullptr;
    void                                              *enc  = metal_op_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    const struct metal_size threads256 = {256, 1, 1};
    /* A simdgroup per gate row (#196) where the rows are float4-aligned. */
    const bool gate_sg =
            st->f32_ple_gate_sg_pipeline != nullptr && ((d_in | x_off | gw_off) % 4u) == 0u;
    metal_msg_send_set_pipeline(
            st, enc, gate_sg ? st->f32_ple_gate_sg_pipeline : st->f32_ple_gate_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, gate_w->buffer->buffer, gate_w->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, ple_in->buffer->buffer, ple_in->buffer->base_off, 2);
    metal_msg_send_set_buffer(
            st, enc, gate_scratch->buffer->buffer, gate_scratch->buffer->base_off, 3);
    metal_msg_send_set_bytes(st, enc, &gp, sizeof(gp), 4);
    const struct metal_size ggroups = {
            gate_sg ? ((uint32_t) hpl + 3u) / 4u : (uint32_t) hpl, (uint32_t) rows, 1};
    const struct metal_size threads128 = {128, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_F32_PLE_GATE, ggroups);
    metal_msg_send_dispatch(st, enc, ggroups, gate_sg ? threads128 : threads256);

    s = metal_op_end(be, cmd, enc, nullptr);
    if (s != GEIST_OK) {
        return s;
    }

    /* Proj side: the fused ple_proj_norm_f32 kernel is a single-threadgroup
     * GEMV and far too slow; route through the fast f32 matvec + fused
     * rmsnorm_add instead. */
    struct geist_tensor gate_1d = *gate_scratch;
    if (gate_1d.ndim == 2 && gate_1d.shape[0] == 1) {
        gate_1d.ndim      = 1;
        gate_1d.shape[0]  = gate_1d.shape[1];
        gate_1d.stride[0] = 1;
        gate_1d.shape[1]  = 0;
        gate_1d.stride[1] = 0;
    }
    struct geist_tensor proj_1d = *proj_scratch;
    if (proj_1d.ndim == 2 && proj_1d.shape[0] == 1) {
        proj_1d.ndim      = 1;
        proj_1d.shape[0]  = proj_1d.shape[1];
        proj_1d.stride[0] = 1;
        proj_1d.shape[1]  = 0;
        proj_1d.stride[1] = 0;
    }
    s = metal_matvec_f32_dense(be, &gate_1d, proj_w, &proj_1d);
    if (s != GEIST_OK || norm_w == nullptr) { /* without norm_w the arch finishes */
        return s;
    }
    return metal_rmsnorm_add(be, res, proj_scratch, norm_w, eps, y);
}

/* ple_stage (#196), decode, two dispatches: ple_gate_prenorm forms h = res
 * + rmsnorm(x) * post_w per threadgroup and runs the gate on it;
 * ple_proj_tail projects and its last threadgroup runs the layer tail. */
[[nodiscard]] static enum geist_status metal_ple_stage(struct geist_backend      *be,
                                                       float                      eps,
                                                       float                      scale,
                                                       const struct geist_tensor *res,
                                                       const struct geist_tensor *x,
                                                       const struct geist_tensor *post_w,
                                                       const struct geist_tensor *gate_w,
                                                       const struct geist_tensor *ple_in,
                                                       const struct geist_tensor *proj_w,
                                                       const struct geist_tensor *tail_w,
                                                       const struct geist_tensor *next_w,
                                                       struct geist_tensor       *h,
                                                       struct geist_tensor       *gate_scratch,
                                                       struct geist_tensor       *proj_scratch,
                                                       struct geist_tensor       *y,
                                                       struct geist_tensor       *z) {
    if (be == nullptr || be->state == nullptr || res == nullptr || x == nullptr ||
        post_w == nullptr || gate_w == nullptr || ple_in == nullptr || proj_w == nullptr ||
        tail_w == nullptr || next_w == nullptr || h == nullptr || gate_scratch == nullptr ||
        proj_scratch == nullptr || y == nullptr || z == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    /* rows (1), d_model and offsets of the row tensors: x res h proj y z */
    const struct geist_tensor *rt[6] = {x, res, h, proj_scratch, y, z};
    size_t                     ro[6] = {0};
    size_t                     d     = 0;
    for (size_t i = 0; i < 6; i++) {
        size_t r = 0, c = 0, stride = 0;
        if (!metal_tensor_is_f32_rows(rt[i], &r, &c, &ro[i], &stride) || r != 1u ||
            (i > 0 && c != d) || ro[i] > UINT32_MAX || rt[i]->buffer->owner != be->state) {
            return GEIST_E_UNSUPPORTED;
        }
        d = c;
    }
    const struct geist_tensor *vt[3] = {post_w, tail_w, next_w};
    size_t                     vo[3] = {0};
    for (size_t i = 0; i < 3; i++) {
        size_t n = 0;
        if (!metal_tensor_is_f32_vector(vt[i], &n, &vo[i]) || n != d || vo[i] > UINT32_MAX ||
            vt[i]->buffer->owner != be->state) {
            return GEIST_E_UNSUPPORTED;
        }
    }
    size_t g_rows = 0, g_cols = 0, g_off = 0, g_stride = 0;
    size_t p_rows = 0, p_cols = 0, p_off = 0, p_stride = 0;
    size_t hpl = 0, gw_in = 0, gw_off = 0, gw_stride = 0;
    size_t pw_out = 0, pw_in = 0, pw_off = 0, pw_stride = 0;
    if (!metal_tensor_is_f32_matrix(gate_w, &hpl, &gw_in, &gw_off, &gw_stride) ||
        !metal_tensor_is_f32_rows(ple_in, &p_rows, &p_cols, &p_off, &p_stride) ||
        !metal_tensor_is_f32_matrix(proj_w, &pw_out, &pw_in, &pw_off, &pw_stride) ||
        !metal_tensor_is_f32_rows(gate_scratch, &g_rows, &g_cols, &g_off, &g_stride) ||
        p_rows != 1u || g_rows != 1u || gw_in != d || gw_stride != gw_in || pw_stride != pw_in ||
        pw_out != d || pw_in != hpl || p_cols != hpl || g_cols != hpl || d == 0 || hpl == 0 ||
        d > METAL_PLE_PRENORM_MAX_D || (d % 4u) != 0u || (hpl % 4u) != 0u || (gw_off % 4u) != 0u ||
        (pw_off % 4u) != 0u || (g_off % 4u) != 0u) {
        return GEIST_E_UNSUPPORTED;
    }
    if (hpl > UINT32_MAX || gw_off > UINT32_MAX || pw_off > UINT32_MAX || p_off > UINT32_MAX ||
        g_off > UINT32_MAX || gate_w->buffer->owner != be->state ||
        ple_in->buffer->owner != be->state || proj_w->buffer->owner != be->state ||
        gate_scratch->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st = be->state;
    if (st->f32_ple_gate_prenorm_pipeline == nullptr || st->f32_ple_proj_tail_pipeline == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    s = metal_ensure_counts(be);
    if (s != GEIST_OK) {
        return s;
    }
    const struct {
        uint32_t ni, no, ro, xo, pwo, wo, po, yo, ho;
        float    eps;
    } gp = {(uint32_t) d,
            (uint32_t) hpl,
            (uint32_t) ro[1],
            (uint32_t) ro[0],
            (uint32_t) vo[0],
            (uint32_t) gw_off,
            (uint32_t) p_off,
            (uint32_t) g_off,
            (uint32_t) ro[2],
            eps};
    const struct {
        uint32_t ni, no, xo, wo, po, ho, two, nwo, yo, zo;
        float    eps, scale;
    } tp                                                    = {(uint32_t) hpl,
                                                               (uint32_t) d,
                                                               (uint32_t) g_off,
                                                               (uint32_t) pw_off,
                                                               (uint32_t) ro[3],
                                                               (uint32_t) ro[2],
                                                               (uint32_t) vo[1],
                                                               (uint32_t) vo[2],
                                                               (uint32_t) ro[4],
                                                               (uint32_t) ro[5],
                                                               eps,
                                                               scale};
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);
    void                                              *cmd  = nullptr;
    void                                              *enc  = metal_op_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    const struct metal_size threads = {256, 1, 1};
    metal_msg_send_set_pipeline(st, enc, st->f32_ple_gate_prenorm_pipeline);
    metal_msg_send_set_buffer(st, enc, res->buffer->buffer, res->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, post_w->buffer->buffer, post_w->buffer->base_off, 2);
    metal_msg_send_set_buffer(st, enc, gate_w->buffer->buffer, gate_w->buffer->base_off, 3);
    metal_msg_send_set_buffer(st, enc, ple_in->buffer->buffer, ple_in->buffer->base_off, 4);
    metal_msg_send_set_buffer(
            st, enc, gate_scratch->buffer->buffer, gate_scratch->buffer->base_off, 5);
    metal_msg_send_set_buffer(st, enc, h->buffer->buffer, h->buffer->base_off, 6);
    metal_msg_send_set_bytes(st, enc, &gp, sizeof(gp), 7);
    metal_msg_send_set_threadgroup_memory(st, enc, (uint32_t) (d * sizeof(float)), 0u);
    const struct metal_size ggroups = {((uint32_t) hpl + 7u) / 8u, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_F32_PLE_GATE, ggroups);
    metal_msg_send_dispatch(st, enc, ggroups, threads);

    metal_msg_send_set_pipeline(st, enc, st->f32_ple_proj_tail_pipeline);
    metal_msg_send_set_buffer(
            st, enc, gate_scratch->buffer->buffer, gate_scratch->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, proj_w->buffer->buffer, proj_w->buffer->base_off, 1);
    metal_msg_send_set_buffer(
            st, enc, proj_scratch->buffer->buffer, proj_scratch->buffer->base_off, 2);
    metal_msg_send_set_buffer(st, enc, h->buffer->buffer, h->buffer->base_off, 3);
    metal_msg_send_set_buffer(st, enc, tail_w->buffer->buffer, tail_w->buffer->base_off, 4);
    metal_msg_send_set_buffer(st, enc, next_w->buffer->buffer, next_w->buffer->base_off, 5);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 6);
    metal_msg_send_set_buffer(st, enc, z->buffer->buffer, z->buffer->base_off, 7);
    metal_msg_send_set_buffer(st,
                              enc,
                              st->attn_dec_count_buffer->buffer,
                              st->attn_dec_count_buffer->base_off +
                                      METAL_COUNT_PLE_TAIL * sizeof(uint32_t),
                              8);
    metal_msg_send_set_bytes(st, enc, &tp, sizeof(tp), 9);
    const struct metal_size pgroups = {((uint32_t) d + 7u) / 8u, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_F32_MATMUL, pgroups);
    metal_msg_send_dispatch(st, enc, pgroups, threads);
    return metal_op_end(be, cmd, enc, nullptr);
}

/* Device greedy argmax (vtbl slot): one 256-thread threadgroup scans the
 * [1, n] logits row on the GPU; the host flush then reads a 4-byte token
 * instead of mapping the whole 1 MB logits row. Tie-break
 * = lowest index, matching geist_sampler_argmax. */
[[nodiscard]] static enum geist_status
metal_argmax_f32(struct geist_backend *be, const struct geist_tensor *logits, int32_t *out_index) {

    if (be == nullptr || be->state == nullptr || logits == nullptr || out_index == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, n = 0, x_off = 0, x_stride = 0;
    if (!metal_tensor_is_f32_rows(logits, &rows, &n, &x_off, &x_stride) || rows != 1u || n == 0 ||
        n > UINT32_MAX || x_off > UINT32_MAX || logits->buffer->owner != be->state) {
        return GEIST_E_UNSUPPORTED;
    }
    enum geist_status s = metal_ensure_argmax_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st = be->state;

    struct {
        uint32_t n, xo;
    } ap                                                    = {(uint32_t) n, (uint32_t) x_off};
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);
    void                                              *cmd  = nullptr;
    void                                              *enc  = metal_op_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_msg_send_set_pipeline(st, enc, st->argmax_pipeline);
    metal_msg_send_set_buffer(st, enc, logits->buffer->buffer, logits->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, st->argmax_result_buffer, 0, 1);
    metal_msg_send_set_bytes(st, enc, &ap, sizeof(ap), 2);
    const struct metal_size groups  = {1, 1, 1};
    const struct metal_size threads = {256, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ARGMAX, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
    if (cmd == nullptr) {
        st->sequence_has_work = true;
        /* The 4-byte result read is the token's only host sync point. */
        metal_flush_if_referenced(st, st->argmax_result_buffer, 0, SIZE_MAX);
    } else if (metal_standalone_end(be, cmd, enc, nullptr) != GEIST_OK) {
        return GEIST_E_BACKEND;
    }
    *out_index = (int32_t) ((const uint32_t *) st->argmax_result_mapped)[0];
    return GEIST_OK;
}

/* Fused f32→f16 KV append (vtbl slot): convert seq rows of scratch K/V and
 * store them at row q_position of the f16 caches — the kernel the f32-KV
 * flash paths use for their staging, aimed at the cache instead. */
[[nodiscard]] static enum geist_status metal_kv_append_f16(struct geist_backend      *be,
                                                           const struct geist_tensor *k_src,
                                                           const struct geist_tensor *v_src,
                                                           size_t                     q_position,
                                                           struct geist_tensor       *k_cache,
                                                           struct geist_tensor       *v_cache) {

    if (be == nullptr || be->state == nullptr || k_src == nullptr || v_src == nullptr ||
        k_cache == nullptr || v_cache == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t s_rows = 0, s_heads = 0, s_hd = 0, k_off = 0;
    size_t vs_rows = 0, vs_heads = 0, vs_hd = 0, v_off = 0;
    size_t kc_rows = 0, kc_heads = 0, kc_hd = 0, kc_off = 0;
    size_t vc_rows = 0, vc_heads = 0, vc_hd = 0, vc_off = 0;
    if (!metal_tensor_is_f32_3d(k_src, &s_rows, &s_heads, &s_hd, &k_off) ||
        !metal_tensor_is_f32_3d(v_src, &vs_rows, &vs_heads, &vs_hd, &v_off) ||
        !metal_tensor_is_f16_3d(k_cache, &kc_rows, &kc_heads, &kc_hd, &kc_off) ||
        !metal_tensor_is_f16_3d(v_cache, &vc_rows, &vc_heads, &vc_hd, &vc_off) ||
        vs_rows != s_rows || vs_heads != s_heads || vs_hd != s_hd || kc_heads != s_heads ||
        kc_hd != s_hd || vc_heads != s_heads || vc_hd != s_hd || kc_rows < q_position + s_rows ||
        vc_rows < q_position + s_rows) {
        return GEIST_E_UNSUPPORTED;
    }
    const size_t kv_out = s_heads * s_hd;
    const size_t elems  = s_rows * kv_out;
    if (elems == 0 || elems > UINT32_MAX || k_off > UINT32_MAX || v_off > UINT32_MAX ||
        kc_off > UINT32_MAX || vc_off > UINT32_MAX || q_position > UINT32_MAX ||
        k_src->buffer->owner != be->state || v_src->buffer->owner != be->state ||
        k_cache->buffer->owner != be->state || v_cache->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_attention_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st = be->state;
    if (st->kv_append_rows_f16_pipeline == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    struct {
        uint32_t elems, kv_out, k_offset, v_offset, k_cache_offset, v_cache_offset, q_position;
    } ap = {(uint32_t) elems,
            (uint32_t) kv_out,
            (uint32_t) k_off,
            (uint32_t) v_off,
            (uint32_t) kc_off,
            (uint32_t) vc_off,
            (uint32_t) q_position};

    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);
    void                                              *cmd  = nullptr;
    void                                              *enc  = metal_op_begin(be, nullptr, &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_msg_send_set_pipeline(st, enc, st->kv_append_rows_f16_pipeline);
    metal_msg_send_set_buffer(st, enc, k_src->buffer->buffer, k_src->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, v_src->buffer->buffer, v_src->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, k_cache->buffer->buffer, k_cache->buffer->base_off, 2);
    metal_msg_send_set_buffer(st, enc, v_cache->buffer->buffer, v_cache->buffer->base_off, 3);
    metal_msg_send_set_bytes(st, enc, &ap, sizeof(ap), 4);
    const struct metal_size groups  = {(elems + 255u) / 256u, 1, 1};
    const struct metal_size threads = {256, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_KV_APPEND_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
    return metal_op_end(be, cmd, enc, nullptr);
}

[[nodiscard]] static enum geist_status metal_attention(struct geist_backend      *be,
                                                       const struct geist_tensor *q,
                                                       const struct geist_tensor *k,
                                                       const struct geist_tensor *value,
                                                       size_t                     q_offset,
                                                       size_t                     sliding_window,
                                                       struct geist_tensor       *out) {

    if (be == nullptr || be->state == nullptr || q == nullptr || k == nullptr || value == nullptr ||
        out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t q_rows = 0, q_heads = 0, head_dim = 0, q_off = 0;
    size_t k_rows = 0, k_heads = 0, k_head_dim = 0, k_off = 0;
    size_t v_rows = 0, v_heads = 0, v_head_dim = 0, v_off = 0;
    size_t out_rows = 0, out_heads = 0, out_head_dim = 0, out_off = 0;
    if (!metal_tensor_is_f32_3d(q, &q_rows, &q_heads, &head_dim, &q_off) ||
        !metal_tensor_is_f32_3d(out, &out_rows, &out_heads, &out_head_dim, &out_off)) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "metal attention: q/out tensors must be F32 DENSE 3D");
        return GEIST_E_UNSUPPORTED;
    }
    const bool kv_is_f32 = metal_tensor_is_f32_3d(k, &k_rows, &k_heads, &k_head_dim, &k_off) &&
                           metal_tensor_is_f32_3d(value, &v_rows, &v_heads, &v_head_dim, &v_off);
    const bool kv_is_f16 = !kv_is_f32 &&
                           metal_tensor_is_f16_3d(k, &k_rows, &k_heads, &k_head_dim, &k_off) &&
                           metal_tensor_is_f16_3d(value, &v_rows, &v_heads, &v_head_dim, &v_off);
    if (!kv_is_f32 && !kv_is_f16) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "metal attention: K/V cache must be matching F32 or F16 DENSE 3D");
        return GEIST_E_UNSUPPORTED;
    }
    if (q_rows != out_rows || q_heads != out_heads || head_dim != out_head_dim ||
        k_rows != v_rows || k_heads != v_heads || head_dim != k_head_dim ||
        head_dim != v_head_dim || k_heads == 0 || q_heads % k_heads != 0) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "metal attention: shape mismatch");
        return GEIST_E_INVALID_ARG;
    }
    if (head_dim > 512u) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "metal attention: head_dim exceeds shader limit");
        return GEIST_E_UNSUPPORTED;
    }
    if (q_rows > UINT32_MAX || k_rows > UINT32_MAX || q_heads > UINT32_MAX ||
        k_heads > UINT32_MAX || head_dim > UINT32_MAX || q_offset > UINT32_MAX ||
        sliding_window > UINT32_MAX || q_off > UINT32_MAX || k_off > UINT32_MAX ||
        v_off > UINT32_MAX || out_off > UINT32_MAX || q_offset > SIZE_MAX - q_rows ||
        q_offset + q_rows > UINT32_MAX || q->buffer->owner != be->state ||
        k->buffer->owner != be->state || value->buffer->owner != be->state ||
        out->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = metal_ensure_attention_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st = be->state;
    if (q_rows == 1u &&
        (head_dim <= 256u || (head_dim <= 512u && st->attention_dec512_f16_pipeline != nullptr)) &&
        k_rows >= 32u && st->attention_dec_f16_pipeline != nullptr &&
        st->attention_dec_combine_pipeline != nullptr) {
        s = metal_attention_dec_kv(be,
                                   st,
                                   q,
                                   k,
                                   value,
                                   out,
                                   k_rows,
                                   q_heads,
                                   k_heads,
                                   head_dim,
                                   q_offset,
                                   sliding_window,
                                   q_off,
                                   k_off,
                                   v_off,
                                   out_off,
                                   kv_is_f16);
        if (s == GEIST_OK) {
            return GEIST_OK;
        }
        /* fall through to the scalar kernel on failure */
    }
    /* Any q_rows > 1: the flash kernels clamp the query rows of a ragged
     * last tile and write only the valid ones (#35). They read whole
     * 64-key blocks, so a cache of fewer rows takes the scalar kernel. */
    if (q_rows > 1u && k_rows >= 64u && metal_flash_pipeline(st, head_dim) != nullptr &&
        st->kv_append_rows_f16_pipeline != nullptr) {
        s = metal_attention_flash_kv(be,
                                     st,
                                     q,
                                     k,
                                     value,
                                     out,
                                     q_rows,
                                     k_rows,
                                     q_heads,
                                     k_heads,
                                     head_dim,
                                     q_offset,
                                     sliding_window,
                                     q_off,
                                     k_off,
                                     v_off,
                                     out_off,
                                     kv_is_f16);
        if (s == GEIST_OK) {
            return GEIST_OK;
        }
        /* fall through to the scalar kernel on failure */
    }
    const struct metal_attention_params params = {
            .rows           = (uint32_t) q_rows,
            .kv_len         = (uint32_t) k_rows,
            .q_heads        = (uint32_t) q_heads,
            .kv_heads       = (uint32_t) k_heads,
            .head_dim       = (uint32_t) head_dim,
            .q_position     = (uint32_t) q_offset,
            .sliding_window = (uint32_t) sliding_window,
            .q_offset       = (uint32_t) q_off,
            .k_cache_offset = (uint32_t) k_off,
            .v_cache_offset = (uint32_t) v_off,
            .y_offset       = (uint32_t) out_off,
    };
    if (metal_seq_mine(st)) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal attention: sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_attention_rows(st, metal_sequence_encoder(st), q, k, value, out, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);

    void *cmd = nullptr;
    void *enc = metal_standalone_begin(be, "attention", &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_attention_rows(st, enc, q, k, value, out, &params);
    return metal_standalone_end(be, cmd, enc, "attention");
}

static void metal_linear_mN(size_t                     m,
                            const float               *x,
                            const struct geist_weight *w,
                            struct geist_backend      *be,
                            float                     *y) {
    struct metal_state  *st    = be->state;
    const size_t         n_in  = (size_t) w->n_in;
    const size_t         n_out = (size_t) w->n_out;
    size_t               xo    = 0;
    size_t               wo    = 0;
    size_t               yo    = 0;
    struct geist_buffer *bx    = metal_buf_reg_find(st, x, &xo);
    struct geist_buffer *bw    = metal_buf_reg_find(st, w->raw, &wo);
    struct geist_buffer *by    = metal_buf_reg_find(st, y, &yo);
    if (bx == nullptr || bw == nullptr || by == nullptr) {
        /* One of the pointers is plain host memory (the engine passes heap
         * scratch for small helper projections, e.g. single-row views).
         * Compute on host — same math as cpu_scalar (dequant row, double
         * accumulator), so token parity with the CPU backends holds. The
         * inputs may be GPU-pending, so drain the batch first. */
        metal_batch_flush(st);
        float *row = heap_alloc_array_aligned(float, n_in);
        if (row == nullptr) {
            memset(y, 0, m * n_out * sizeof(float));
            return;
        }
        for (size_t j = 0; j < n_out; j++) {
            /* An undecodable dtype reads as a zero row. */
            (void) quant_dequant_row(
                    (enum geist_dtype) w->dtype, n_in * n_out, j * n_in, n_in, w->raw, row);
            for (size_t i = 0; i < m; i++) {
                double acc = 0.0;
                for (size_t k = 0; k < n_in; k++) {
                    acc += (double) x[i * n_in + k] * (double) row[k];
                }
                y[i * n_out + j] = (float) acc;
            }
        }
        safe_free((void **) &row);
        return;
    }
    struct geist_tensor tx = {
            .buffer = bx,
            .offset = xo,
            .dtype  = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim   = 2,
            .shape  = {(int64_t) m, (int64_t) n_in},
            .stride = {(int64_t) n_in, 1},
    };
    struct geist_tensor ty = {
            .buffer = by,
            .offset = yo,
            .dtype  = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim   = 2,
            .shape  = {(int64_t) m, (int64_t) n_out},
            .stride = {(int64_t) n_out, 1},
    };
    struct geist_tensor tw = {
            .buffer = bw,
            .offset = wo,
            .dtype  = (enum geist_dtype) w->dtype,
            .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
            .ndim   = 2,
            .shape  = {(int64_t) n_out, (int64_t) n_in},
            .stride = {0, 0},
    };
    enum geist_status s;
    switch ((enum geist_dtype) w->dtype) {
    case GEIST_DTYPE_Q4_0:
    case GEIST_DTYPE_Q4_1:
    case GEIST_DTYPE_Q8_0:
    case GEIST_DTYPE_IQ4_NL:
    case GEIST_DTYPE_IQ4_XS:
    case GEIST_DTYPE_Q3_K:
    case GEIST_DTYPE_IQ3_S:
    case GEIST_DTYPE_PQ2_0:
    case GEIST_DTYPE_TQ2_0:
    case GEIST_DTYPE_I2_S:
        s = metal_q40_q80_linear(be, &tx, &tw, &ty, (enum geist_dtype) w->dtype, true);
        break;
    case GEIST_DTYPE_Q4_K:
        s = metal_matmul_q4k(be, &tx, &tw, &ty);
        break;
    case GEIST_DTYPE_Q5_K:
        s = metal_matmul_q5k(be, &tx, &tw, &ty);
        break;
    case GEIST_DTYPE_Q6_K:
        s = metal_matmul_q6k(be, &tx, &tw, &ty);
        break;
    case GEIST_DTYPE_F16:
    case GEIST_DTYPE_BF16:
    case GEIST_DTYPE_F32:
        tw.layout    = GEIST_LAYOUT_DENSE;
        tw.stride[0] = (int64_t) n_in;
        tw.stride[1] = 1;
        s            = metal_matmul_f32_dense(be, &tx, &tw, &ty);
        break;
    default:
        s = GEIST_E_UNSUPPORTED;
        break;
    }
    if (s != GEIST_OK) {
        fprintf(stderr,
                "geist metal: linear dispatch failed "
                "(status=%d dtype=%u m=%zu %zux%zu xo=%zu wo=%zu yo=%zu): %s\n",
                (int) s,
                (unsigned) w->dtype,
                m,
                n_out,
                n_in,
                xo,
                wo,
                yo,
                geist_backend_errmsg(be));
        memset(y, 0, m * n_out * sizeof(float));
    }
}

static void
metal_linear_m1(const float *x, const struct geist_weight *w, struct geist_backend *be, float *y) {
    metal_linear_mN(1, x, w, be, y);
}

/* Tensor-based linear (optional vtbl slot): dispatch the GEMM from
 * the engine's existing tensor views — no host pointers, so the op encodes
 * onto the open batch when one is active. UNSUPPORTED falls back to the
 * resolved host-pointer kernels. */
[[nodiscard]] static enum geist_status metal_linear_t(struct geist_backend      *be,
                                                      const struct geist_tensor *x,
                                                      const struct geist_weight *w,
                                                      const struct geist_tensor *t_w,
                                                      size_t                     m,
                                                      struct geist_tensor       *y) {

    if (w == nullptr || t_w == nullptr || x == nullptr || y == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    /* m == 1 (decode) routes to the matvec ops — the GEMM tile kernels are
     * an order of magnitude slower for a single row than the llama-style
     * mul_mv kernels. The engine passes [1, n] 2D views; rebuild them 1D. */
    if (m == 1) {
        struct geist_tensor x1 = *x;
        struct geist_tensor y1 = *y;
        if (x1.ndim == 2 && x1.shape[0] == 1) {
            x1.ndim      = 1;
            x1.shape[0]  = x1.shape[1];
            x1.stride[0] = x1.stride[1];
            x1.shape[1]  = 0;
            x1.stride[1] = 0;
        }
        if (y1.ndim == 2 && y1.shape[0] == 1) {
            y1.ndim      = 1;
            y1.shape[0]  = y1.shape[1];
            y1.stride[0] = y1.stride[1];
            y1.shape[1]  = 0;
            y1.stride[1] = 0;
        }
        switch ((enum geist_dtype) w->dtype) {
        case GEIST_DTYPE_Q4_0:
        case GEIST_DTYPE_Q4_1:
        case GEIST_DTYPE_Q8_0:
        case GEIST_DTYPE_IQ4_NL:
        case GEIST_DTYPE_IQ4_XS:
        case GEIST_DTYPE_Q3_K:
        case GEIST_DTYPE_IQ3_S:
        case GEIST_DTYPE_PQ2_0:
        case GEIST_DTYPE_TQ2_0:
        case GEIST_DTYPE_I2_S:
            return metal_q40_q80_linear(be, &x1, t_w, &y1, (enum geist_dtype) w->dtype, false);
        case GEIST_DTYPE_Q4_K:
            return metal_matvec_q4k(be, &x1, t_w, &y1);
        case GEIST_DTYPE_Q5_K:
            return metal_matvec_q5k(be, &x1, t_w, &y1);
        case GEIST_DTYPE_Q6_K:
            return metal_matvec_q6k(be, &x1, t_w, &y1);
        case GEIST_DTYPE_F16:
        case GEIST_DTYPE_BF16:
        case GEIST_DTYPE_F32:
            return metal_matvec_f32_dense(be, &x1, t_w, &y1);
        default:
            return GEIST_E_UNSUPPORTED;
        }
    }
    switch ((enum geist_dtype) w->dtype) {
    case GEIST_DTYPE_Q4_0:
    case GEIST_DTYPE_Q4_1:
    case GEIST_DTYPE_Q8_0:
    case GEIST_DTYPE_IQ4_NL:
    case GEIST_DTYPE_IQ4_XS:
    case GEIST_DTYPE_Q3_K:
    case GEIST_DTYPE_IQ3_S:
    case GEIST_DTYPE_PQ2_0:
    case GEIST_DTYPE_TQ2_0:
    case GEIST_DTYPE_I2_S:
        return metal_q40_q80_linear(be, x, t_w, y, (enum geist_dtype) w->dtype, true);
    case GEIST_DTYPE_Q4_K:
        return metal_matmul_q4k(be, x, t_w, y);
    case GEIST_DTYPE_Q5_K:
        return metal_matmul_q5k(be, x, t_w, y);
    case GEIST_DTYPE_Q6_K:
        return metal_matmul_q6k(be, x, t_w, y);
    case GEIST_DTYPE_F16:
    case GEIST_DTYPE_BF16:
    case GEIST_DTYPE_F32:
        return metal_matmul_f32_dense(be, x, t_w, y);
    default:
        return GEIST_E_UNSUPPORTED;
    }
}

[[nodiscard]] static enum geist_status metal_selected_prepare(const struct geist_weight *w,
                                                              struct geist_backend      *be) {
    const enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK)
        return s;
    const struct metal_state *st    = be->state;
    const enum geist_dtype    dtype = (enum geist_dtype) w->dtype;
    if (dtype == GEIST_DTYPE_F32 || dtype == GEIST_DTYPE_F16 || dtype == GEIST_DTYPE_BF16)
        return GEIST_OK;
    if (dtype == GEIST_DTYPE_Q4_K)
        return st->q4k_n4_pipeline != nullptr ? GEIST_OK : GEIST_E_UNSUPPORTED;
    if (dtype == GEIST_DTYPE_Q6_K)
        return (w->linear_rows_tile > 1 ? st->q6k_n4_pipeline : st->q6k_pipeline) != nullptr
                       ? GEIST_OK
                       : GEIST_E_UNSUPPORTED;
    const struct metal_quant_pipes p = metal_quant_pipes_for(st, dtype, (uint32_t) w->n_out);
    return (w->linear_rows_tile > 1 ? p.n4 : p.base) != nullptr ? GEIST_OK : GEIST_E_UNSUPPORTED;
}

/* Encode the original dense kernel on aligned, bounded row tiles. In
 * particular PQ2's n8/n4 choice uses the ORIGINAL output shape. */
[[nodiscard]] static enum geist_status metal_selected_rows(size_t              n_tiles,
                                                           const geist_token_t ids[static n_tiles],
                                                           const struct geist_tensor *x,
                                                           const struct geist_weight *w,
                                                           const struct geist_tensor *t_w,
                                                           struct geist_tensor       *y,
                                                           struct geist_backend      *be) {
    if (be == nullptr || be->state == nullptr || w == nullptr || x == nullptr || y == nullptr ||
        t_w == nullptr || x->buffer == nullptr || y->buffer == nullptr || t_w->buffer == nullptr ||
        w->n_in <= 0 || w->n_out <= 0 || n_tiles == 0)
        return GEIST_E_INVALID_ARG;
    struct metal_state *st   = be->state;
    const size_t        tile = w->linear_rows_tile;
    size_t              nx, xo, yr, yc, y_off, ys;
    if (!metal_tensor_is_f32_vector(x, &nx, &xo) ||
        !metal_tensor_is_f32_matrix(y, &yr, &yc, &y_off, &ys) || nx != (size_t) w->n_in ||
        yr != n_tiles || yc != tile || ys != tile || xo > UINT32_MAX || t_w->ndim != 2 ||
        t_w->dtype != w->dtype || t_w->shape[0] != w->n_out || t_w->shape[1] != w->n_in)
        return GEIST_E_INVALID_ARG;
    size_t stride, out_n, out_bytes, x_bytes;
    if (tile == 0 || quant_raw_bytes((enum geist_dtype) w->dtype, (size_t) w->n_in, &stride) ||
        ckd_mul(&out_n, n_tiles, tile) || ckd_mul(&out_bytes, out_n, sizeof(float)) ||
        ckd_mul(&x_bytes, (size_t) w->n_in, sizeof(float)) || x->offset > x->buffer->bytes ||
        x_bytes > x->buffer->bytes - x->offset || y->offset > y->buffer->bytes ||
        out_bytes > y->buffer->bytes - y->offset || t_w->offset > t_w->buffer->bytes ||
        w->raw_nbytes > t_w->buffer->bytes - t_w->offset || x->offset % sizeof(float) != 0 ||
        y->offset % sizeof(float) != 0 || x->buffer->owner != st || y->buffer->owner != st ||
        t_w->buffer->owner != st || out_n > UINT32_MAX || w->n_in <= 0)
        return GEIST_E_INVALID_ARG;
    /* Validate every tile/offset before encoding any GPU work. */
    for (size_t i = 0; i < n_tiles; i++) {
        size_t wo, yo, bytes;
        if (ids[i] < 0 || (size_t) ids[i] >= (size_t) w->n_out || (size_t) ids[i] % tile != 0 ||
            ckd_mul(&wo, (size_t) ids[i], stride) || ckd_add(&wo, wo, t_w->offset) ||
            ckd_mul(&yo, i, tile) || ckd_mul(&yo, yo, sizeof(float)) ||
            ckd_add(&yo, yo, y->offset) || wo > UINT32_MAX || yo / sizeof(float) > UINT32_MAX ||
            ckd_mul(&bytes,
                    (size_t) w->n_out - (size_t) ids[i] < tile ? (size_t) w->n_out - (size_t) ids[i]
                                                               : tile,
                    stride) ||
            wo > t_w->buffer->bytes || bytes > t_w->buffer->bytes - wo)
            return GEIST_E_INVALID_ARG;
    }
    /* Fail closed before encoding: an encoder whose pipeline is missing
     * dispatches nothing and would leave the previous query's values in y. */
    const enum geist_status s = metal_selected_prepare(w, be);
    if (s != GEIST_OK)
        return s;
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);
    void                                              *cmd  = nullptr;
    void                                              *enc  = metal_op_begin(be, nullptr, &cmd);
    if (enc == nullptr)
        return GEIST_E_BACKEND;
    const enum geist_dtype         dtype = (enum geist_dtype) w->dtype;
    const struct metal_quant_pipes pipes = metal_quant_pipes_for(st, dtype, (uint32_t) w->n_out);
    for (size_t i = 0; i < n_tiles; i++) {
        const size_t n  = (size_t) w->n_out - (size_t) ids[i] < tile
                                  ? (size_t) w->n_out - (size_t) ids[i]
                                  : tile;
        const size_t wo = t_w->offset + (size_t) ids[i] * stride;
        const size_t yo = y->offset / sizeof(float) + i * tile;
        if (dtype == GEIST_DTYPE_F32 || dtype == GEIST_DTYPE_F16 || dtype == GEIST_DTYPE_BF16) {
            const struct metal_f32_params params = {
                    .n_in         = (uint32_t) w->n_in,
                    .n_out        = (uint32_t) n,
                    .rows         = 1,
                    .x_offset     = (uint32_t) (x->offset / sizeof(float)),
                    .w_offset     = (uint32_t) (wo / (dtype == GEIST_DTYPE_F32 ? sizeof(float)
                                                                               : sizeof(uint16_t))),
                    .y_offset     = (uint32_t) yo,
                    .x_row_stride = (uint32_t) w->n_in,
                    .y_row_stride = (uint32_t) n,
            };
            metal_encode_f32_matmul(st, enc, x, t_w, y, &params);
        } else {
            const struct metal_q4k_params params = {
                    .n_in  = (uint32_t) w->n_in,
                    .n_out = (uint32_t) n,
                    .rows  = 1,
                    .blocks_per_row =
                            (uint32_t) ((size_t) w->n_in / metal_quant_block_elems(dtype)),
                    .x_offset      = (uint32_t) (x->offset / sizeof(float)),
                    .w_byte_offset = (uint32_t) wo,
                    .y_offset      = (uint32_t) yo,
                    .x_row_stride  = (uint32_t) w->n_in,
                    .y_row_stride  = (uint32_t) n,
            };
            if (dtype == GEIST_DTYPE_Q4_K) {
                metal_encode_q4k_linear(st, enc, x, t_w, y, &params, false);
            } else if (dtype == GEIST_DTYPE_Q6_K && tile == 4) {
                /* The tail must retain n4 even when fewer than four valid
                 * rows remain. The shader bounds-checks individual lanes. */
                metal_msg_send_set_pipeline(st, enc, st->q6k_n4_pipeline);
                metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
                metal_msg_send_set_buffer(st, enc, t_w->buffer->buffer, t_w->buffer->base_off, 1);
                metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 2);
                metal_msg_send_set_bytes(st, enc, &params, sizeof params, 3);
                const struct metal_size groups  = {.width = 1, .height = 1, .depth = 1};
                const struct metal_size threads = {
                        .width = METAL_Q4K_N4_THREADS, .height = 1, .depth = 1};
                metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_Q4K_LINEAR_N4, groups);
                metal_msg_send_dispatch(st, enc, groups, threads);
            } else if (dtype == GEIST_DTYPE_Q6_K) {
                metal_encode_q6k_linear(st, enc, x, t_w, y, &params, false);
            } else if (tile > 1) {
                metal_msg_send_set_pipeline(st, enc, pipes.n4);
                metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
                metal_msg_send_set_buffer(st, enc, t_w->buffer->buffer, t_w->buffer->base_off, 1);
                metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 2);
                metal_msg_send_set_bytes(st, enc, &params, sizeof params, 3);
                if (dtype == GEIST_DTYPE_PQ2_0)
                    metal_msg_send_set_threadgroup_memory(st, enc, 256u * 8u, 0u);
                const struct metal_size groups  = {.width  = ((uint32_t) n + pipes.n4_tile - 1u) /
                                                             pipes.n4_tile,
                                                   .height = 1,
                                                   .depth  = 1};
                const struct metal_size threads = {
                        .width = METAL_Q4K_N4_THREADS, .height = 1, .depth = 1};
                metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_Q4K_LINEAR_N4, groups);
                metal_msg_send_dispatch(st, enc, groups, threads);
            } else {
                metal_encode_q40_q80_linear(st, enc, x, t_w, y, &params, dtype);
            }
        }
    }
    return metal_op_end(be, cmd, enc, nullptr);
}

[[nodiscard]] static enum geist_status metal_resolve_weight(struct geist_backend *be,
                                                            struct geist_weight  *w) {
    if (be == nullptr || be->state == nullptr || w == nullptr || w->raw == nullptr ||
        w->n_in <= 0 || w->n_out <= 0 || w->raw_nbytes == 0u) {
        return GEIST_E_INVALID_ARG;
    }
    /* The kernels installed below index `raw` by shape, so a source shorter
     * than the shape reads past its end. Same contract as the CPU
     * resolvers. */
    if (!quant_weight_extent_ok(w)) {
        return GEIST_E_FORMAT;
    }
    /* The apply window just closed; fold the blob into the crossovers
     * before the first kernel choice reads them. */
    metal_tuning_resolve(be, be->state);
    switch ((enum geist_dtype) w->dtype) {
    case GEIST_DTYPE_Q4_0:
    case GEIST_DTYPE_Q4_1:
    case GEIST_DTYPE_Q8_0:
    case GEIST_DTYPE_Q4_K:
    case GEIST_DTYPE_Q5_K:
    case GEIST_DTYPE_Q6_K:
    case GEIST_DTYPE_IQ4_NL:
    case GEIST_DTYPE_IQ4_XS:
    case GEIST_DTYPE_Q3_K:
    case GEIST_DTYPE_IQ3_S:
    case GEIST_DTYPE_PQ2_0:
    case GEIST_DTYPE_TQ2_0:
    case GEIST_DTYPE_I2_S:
    case GEIST_DTYPE_F16:
    case GEIST_DTYPE_BF16:
    case GEIST_DTYPE_F32:
        w->linear_m1                   = metal_linear_m1;
        w->linear_mN                   = metal_linear_mN;
        const struct metal_state *st   = be->state;
        size_t                    tile = 0;
        if (w->dtype == GEIST_DTYPE_F32 || w->dtype == GEIST_DTYPE_F16 ||
            w->dtype == GEIST_DTYPE_BF16) {
            tile = 1;
        } else if (w->dtype == GEIST_DTYPE_Q4_K) {
            tile = 4;
        } else if (w->dtype == GEIST_DTYPE_Q6_K) {
            tile = w->n_out >= 4 ? 4 : 1;
        } else {
            const struct metal_quant_pipes p =
                    metal_quant_pipes_for(st, (enum geist_dtype) w->dtype, (uint32_t) w->n_out);
            if (w->n_out >= 4) {
                tile = w->dtype == GEIST_DTYPE_PQ2_0 &&
                                       (uint32_t) w->n_out >= st->tuning.pq2_n8_min_n_out
                               ? 16
                               : p.n4_tile;
            } else if (!p.gemm_only) {
                tile = 1;
            }
        }
        if (tile != 0 && w->dtype != GEIST_DTYPE_I2_S) {
            w->linear_rows         = metal_selected_rows;
            w->linear_rows_tile    = tile;
            w->linear_rows_prepare = metal_selected_prepare;
        }
        return GEIST_OK;
    default:
        /* Callers fall back per weight (linear_m1 stays null). */
        return GEIST_E_UNSUPPORTED;
    }
}

[[nodiscard]] static enum geist_status
metal_deltanet_mix(struct geist_backend *be, const struct geist_deltanet_mix_args *args) {
    if (be == nullptr || be->state == nullptr || args == nullptr || args->qkv == nullptr ||
        args->z == nullptr || args->beta == nullptr || args->alpha == nullptr ||
        args->conv_w == nullptr || args->ssm_a == nullptr || args->dt_bias == nullptr ||
        args->norm_w == nullptr || args->conv_state == nullptr || args->delta_state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (args->seq == 0 || args->n_k_heads == 0 || args->n_v_heads == 0 ||
        args->n_v_heads % args->n_k_heads != 0 || args->head_k == 0 || args->head_k > 256 ||
        args->head_v == 0 || args->head_v > 256 || args->conv_kernel < 2) {
        return GEIST_E_UNSUPPORTED;
    }

    const size_t key_dim   = args->n_k_heads * args->head_k;
    const size_t value_dim = args->n_v_heads * args->head_v;
    const size_t conv_dim  = 2u * key_dim + value_dim;
    size_t       q_rows = 0, q_cols = 0, q_off = 0, q_stride = 0;
    size_t       z_rows = 0, z_cols = 0, z_off = 0, z_stride = 0;
    size_t       b_rows = 0, b_cols = 0, b_off = 0, b_stride = 0;
    size_t       a_rows = 0, a_cols = 0, a_off = 0, a_stride = 0;
    size_t       cw_rows = 0, cw_cols = 0, cw_off = 0, cw_stride = 0;
    size_t       cs_rows = 0, cs_cols = 0, cs_off = 0, cs_stride = 0;
    size_t       aw_n = 0, aw_off = 0, dt_n = 0, dt_off = 0, nw_n = 0, nw_off = 0;
    size_t       s0 = 0, s1 = 0, s2 = 0, s_off = 0;
    if (!metal_tensor_is_f32_matrix(args->qkv, &q_rows, &q_cols, &q_off, &q_stride) ||
        !metal_tensor_is_f32_matrix(args->z, &z_rows, &z_cols, &z_off, &z_stride) ||
        !metal_tensor_is_f32_matrix(args->beta, &b_rows, &b_cols, &b_off, &b_stride) ||
        !metal_tensor_is_f32_matrix(args->alpha, &a_rows, &a_cols, &a_off, &a_stride) ||
        !metal_tensor_is_f32_matrix(args->conv_w, &cw_rows, &cw_cols, &cw_off, &cw_stride) ||
        !metal_tensor_is_f32_matrix(args->conv_state, &cs_rows, &cs_cols, &cs_off, &cs_stride) ||
        !metal_tensor_is_f32_vector(args->ssm_a, &aw_n, &aw_off) ||
        !metal_tensor_is_f32_vector(args->dt_bias, &dt_n, &dt_off) ||
        !metal_tensor_is_f32_vector(args->norm_w, &nw_n, &nw_off) ||
        !metal_tensor_is_f32_3d(args->delta_state, &s0, &s1, &s2, &s_off) || q_rows != args->seq ||
        q_cols != conv_dim || q_stride != conv_dim || z_rows != args->seq || z_cols != value_dim ||
        z_stride != value_dim || b_rows != args->seq || b_cols != args->n_v_heads ||
        b_stride != args->n_v_heads || a_rows != args->seq || a_cols != args->n_v_heads ||
        a_stride != args->n_v_heads || cw_rows != conv_dim || cw_cols != args->conv_kernel ||
        cw_stride != args->conv_kernel || cs_rows != args->conv_kernel - 1u ||
        cs_cols != conv_dim || cs_stride != conv_dim || aw_n != args->n_v_heads ||
        dt_n != args->n_v_heads || nw_n != args->head_v || s0 != args->n_v_heads ||
        s1 != args->head_k || s2 != args->head_v) {
        return GEIST_E_UNSUPPORTED;
    }
    struct metal_state        *st    = be->state;
    const struct geist_tensor *all[] = {args->qkv,
                                        args->z,
                                        args->beta,
                                        args->alpha,
                                        args->conv_w,
                                        args->ssm_a,
                                        args->dt_bias,
                                        args->norm_w,
                                        args->conv_state,
                                        args->delta_state};
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) {
        if (all[i]->buffer == nullptr || all[i]->buffer->owner != st) {
            return GEIST_E_INVALID_ARG;
        }
    }
    const size_t offsets[] = {
            q_off, z_off, b_off, a_off, cw_off, aw_off, dt_off, nw_off, cs_off, s_off};
    for (size_t i = 0; i < sizeof offsets / sizeof offsets[0]; i++) {
        if (offsets[i] > UINT32_MAX) {
            return GEIST_E_INVALID_ARG;
        }
    }
    if (args->seq > UINT32_MAX || args->n_k_heads > UINT32_MAX || args->n_v_heads > UINT32_MAX ||
        args->head_k > UINT32_MAX || args->head_v > UINT32_MAX || args->conv_kernel > UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status status = metal_ensure_deltanet_pipeline(be);
    if (status != GEIST_OK) {
        return status;
    }
    const struct metal_deltanet_params params = {.seq                = (uint32_t) args->seq,
                                                 .n_k_heads          = (uint32_t) args->n_k_heads,
                                                 .n_v_heads          = (uint32_t) args->n_v_heads,
                                                 .head_k             = (uint32_t) args->head_k,
                                                 .head_v             = (uint32_t) args->head_v,
                                                 .conv_kernel        = (uint32_t) args->conv_kernel,
                                                 .qkv_offset         = (uint32_t) q_off,
                                                 .z_offset           = (uint32_t) z_off,
                                                 .beta_offset        = (uint32_t) b_off,
                                                 .alpha_offset       = (uint32_t) a_off,
                                                 .conv_w_offset      = (uint32_t) cw_off,
                                                 .ssm_a_offset       = (uint32_t) aw_off,
                                                 .dt_bias_offset     = (uint32_t) dt_off,
                                                 .norm_w_offset      = (uint32_t) nw_off,
                                                 .conv_state_offset  = (uint32_t) cs_off,
                                                 .delta_state_offset = (uint32_t) s_off,
                                                 .eps                = args->eps};

    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool = metal_standalone_pool(st);
    void                                              *cmd  = nullptr;
    void                                              *enc  = metal_op_begin(be, "DeltaNet", &cmd);
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    /* seq>1 runs the chunked delta rule on simdgroup matrices (dnf_*, #99;
     * the recipe of the CPU's dn_run_prefill_chunked): seven dispatches per
     * piece of up to DNF_CALL tokens, the chunk length fixed at 64 inside
     * (the O(C^2) chunk cost's optimum, #322) whatever m the surrounding
     * GEMMs run at. The serial per-token mixer stays for shapes outside
     * the tiling and as the fallback when scratch allocation fails; decode
     * has its own two kernels. State buffers thread through Metal's
     * hazard tracking; scratch is reused per piece. DNF_DVB is dnf_state's
     * value-column block (the shader's DNF_DVB). */
    enum { DNF_CALL = 512u, DNF_DVB = 16u };
    bool fast = args->seq > 1 && args->conv_kernel == 4u && args->head_k % 32u == 0 &&
                args->head_k <= 128u && args->head_v % 32u == 0 && args->head_v <= 256u &&
                st->dnf_out_pipeline != nullptr;
    if (fast) {
        const size_t lp = ((args->seq < DNF_CALL ? args->seq : DNF_CALL) + 63u) & ~(size_t) 63u;
        const size_t scr_bytes =
                (lp * conv_dim + 2u * lp * args->n_v_heads + (args->conv_kernel - 1u) * conv_dim +
                 args->n_v_heads * lp * (1u + args->head_k + 2u * args->head_v)) *
                sizeof(float);
        if (scr_bytes > (size_t) 1u << 31) { /* the kernels index it with uint */
            fast = false;
        } else if (st->dn_scratch_bytes < scr_bytes) {
            metal_residency_forget(st, st->dn_scratch);
            metal_msg_send_void0(st, st->dn_scratch, "release");
            st->dn_scratch       = metal_msg_send_id_size_uint(st,
                                                               st->device,
                                                               "newBufferWithLength:options:",
                                                               scr_bytes,
                                                               METAL_RESOURCE_STORAGE_MODE_PRIVATE);
            st->dn_scratch_bytes = st->dn_scratch != nullptr ? scr_bytes : 0;
            fast                 = st->dn_scratch != nullptr;
        }
    }
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) {
        metal_msg_send_set_buffer(st, enc, all[i]->buffer->buffer, all[i]->buffer->base_off, i);
    }
    if (fast) {
        const uint32_t          cd     = (uint32_t) conv_dim;
        const uint32_t          vd     = (uint32_t) value_dim;
        const uint32_t          nvh    = params.n_v_heads;
        const struct metal_size t128   = {128, 1, 1};
        const struct metal_size t256   = {256, 1, 1};
        const struct metal_size g_flat = {((params.conv_kernel - 1u) * cd + 255u) / 256u, 1, 1};
        const struct metal_size g_st   = {params.head_v / DNF_DVB, nvh, 1};
        metal_msg_send_set_buffer(st, enc, st->dn_scratch, 0, 10);
        for (uint32_t off = 0; off < params.seq; off += DNF_CALL) {
            struct metal_deltanet_params sub = params;
            sub.seq = (params.seq - off < DNF_CALL) ? params.seq - off : DNF_CALL;
            sub.qkv_offset += off * cd;
            sub.z_offset += off * vd;
            sub.beta_offset += off * nvh;
            sub.alpha_offset += off * nvh;
            const uint32_t          n_ch   = (sub.seq + 63u) / 64u;
            const struct metal_size g_tok  = {n_ch * 64u, 1, 1};
            const struct metal_size g_conv = {(cd + 255u) / 256u, n_ch * 4u, 1};
            const struct metal_size g_ch   = {n_ch, nvh, 1};
            metal_msg_send_set_bytes(st, enc, &sub, sizeof sub, 11);
            metal_msg_send_set_pipeline(st, enc, st->dnf_cst_copy_pipeline);
            metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DN_PREP, g_flat);
            metal_msg_send_dispatch(st, enc, g_flat, t256);
            metal_msg_send_set_pipeline(st, enc, st->dnf_conv_pipeline);
            metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DN_PREP, g_conv);
            metal_msg_send_dispatch(st, enc, g_conv, t256);
            metal_msg_send_set_pipeline(st, enc, st->dnf_norm_pipeline);
            metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DN_PREP, g_tok);
            metal_msg_send_dispatch(st, enc, g_tok, t256);
            metal_msg_send_set_pipeline(st, enc, st->dnf_roll_pipeline);
            metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DN_PREP, g_flat);
            metal_msg_send_dispatch(st, enc, g_flat, t256);
            metal_msg_send_set_pipeline(st, enc, st->dnf_chunk_pipeline);
            metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DN_CHUNK, g_ch);
            metal_msg_send_dispatch(st, enc, g_ch, t128);
            metal_msg_send_set_pipeline(st, enc, st->dnf_state_pipeline);
            metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DN_STATE, g_st);
            metal_msg_send_dispatch(st, enc, g_st, t128);
            metal_msg_send_set_pipeline(st, enc, st->dnf_out_pipeline);
            metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DN_OUT, g_ch);
            metal_msg_send_dispatch(st, enc, g_ch, t128);
        }
    } else if (args->seq == 1 && st->dn_dec_v_pipeline != nullptr && 4u * args->head_v <= 1024u) {
        /* Decode: q/k prep per k-head, then the state update spread over
         * v-heads x 4 row groups (dn_dec_v). */
        metal_msg_send_set_bytes(st, enc, &params, sizeof params, 10);
        metal_msg_send_set_pipeline(st, enc, st->dn_dec_qk_pipeline);
        const struct metal_size gk = {args->n_k_heads, 1, 1};
        const struct metal_size tk = {256, 1, 1};
        metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DELTANET_DECODE, gk);
        metal_msg_send_dispatch(st, enc, gk, tk);
        metal_msg_send_set_pipeline(st, enc, st->dn_dec_v_pipeline);
        const struct metal_size gv = {args->n_v_heads, 1, 1};
        const struct metal_size tv = {4u * (uint32_t) args->head_v, 1, 1};
        metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DELTANET_DECODE, gv);
        metal_msg_send_dispatch(st, enc, gv, tv);
    } else {
        metal_msg_send_set_pipeline(st, enc, st->deltanet_mix_pipeline);
        metal_msg_send_set_bytes(st, enc, &params, sizeof params, 10);
        const struct metal_size groups  = {args->n_k_heads, 1, 1};
        const struct metal_size threads = {256, 1, 1};
        metal_profile_add_dispatch(st,
                                   args->seq == 1 ? METAL_PROFILE_DISPATCH_DELTANET_DECODE
                                                  : METAL_PROFILE_DISPATCH_DELTANET_PREFILL,
                                   groups);
        metal_msg_send_dispatch(st, enc, groups, threads);
    }
    return metal_op_end(be, cmd, enc, "DeltaNet");
}

/* Backend vtbl. */
static const struct geist_backend_vtbl metal_vtbl = {
        .create                = metal_create,
        .destroy               = metal_destroy,
        .buffer_create         = metal_buffer_create,
        .buffer_destroy        = metal_buffer_destroy,
        .buffer_create_aliased = metal_buffer_create_aliased,
        .buffer_upload         = metal_buffer_upload,
        .buffer_download       = metal_buffer_download,
        .buffer_map            = metal_buffer_map,
        .buffer_unmap          = metal_buffer_unmap,
        .buffer_copy           = metal_buffer_copy,
        .resolve_weight        = metal_resolve_weight,
        .parallel_region_begin = metal_parallel_region_begin,
        .parallel_region_end   = metal_parallel_region_end,
};

/* Probe pairing for the fused table below. gate_up (GeGLU and SwiGLU)
 * mirrors metal_gate_up's entry checks (decode-only matvec kernel); the
 * elementwise GEGLU epilogue works for any F32 row block. */
static bool metal_fused_supported(struct geist_backend *be, const struct geist_fusion_query *q) {
    if (q == nullptr) {
        return false;
    }
    switch (q->op) {
    case GEIST_FUSED_GELU_TANH_MUL:
    case GEIST_FUSED_SILU_MUL:
        return true; /* F32 elementwise, any geometry, any m */
    case GEIST_FUSED_FFN_GATE_UP:
    case GEIST_FUSED_FFN_GATE_UP_SILU: {
        /* Q4_K or Q8_0, both weights the same: metal_gate_up's checks. */
        if (q->m != 1 || q->gate_w == nullptr || q->up_w == nullptr ||
            q->gate_w->dtype != q->up_w->dtype ||
            (q->gate_w->dtype != GEIST_DTYPE_Q4_K && q->gate_w->dtype != GEIST_DTYPE_Q8_0) ||
            q->d_model % (q->gate_w->dtype == GEIST_DTYPE_Q4_K ? 256u : 32u) != 0u ||
            (size_t) q->gate_w->n_in != q->d_model || (size_t) q->up_w->n_in != q->d_model ||
            q->gate_w->n_out != q->up_w->n_out) {
            return false;
        }
        if (metal_ensure_q4k_pipeline(be) != GEIST_OK) {
            return false;
        }
        const struct metal_state *st   = be->state;
        const bool                silu = q->op == GEIST_FUSED_FFN_GATE_UP_SILU;
        return (q->gate_w->dtype == GEIST_DTYPE_Q4_K
                        ? (silu ? st->q4k_gate_up_silu_n4_pipeline : st->q4k_gate_up_n4_pipeline)
                        : (silu ? st->q80_gate_up_silu_n4_pipeline
                                : st->q80_gate_up_n4_pipeline)) != nullptr;
    }
    case GEIST_FUSED_RMSNORM_ADD:
    case GEIST_FUSED_RMSNORM_ADD_NORM:
    case GEIST_FUSED_ARGMAX_F32:
        /* All-F32 tensor ops; the kernels accept any well-formed row
         * block the arch passes. Pipeline creation failure at run time
         * is a real device error, not capability negotiation. */
        return true;
    case GEIST_FUSED_HADAMARD_ROTATE:
        /* Geometry only: the pipeline is still built at the first call,
         * so a shader compile failure surfaces there (#495). */
        return q->m <= UINT32_MAX &&
               metal_hadamard_geometry_ok(
                       q->width, q->block, q->perm_hd, q->perm_nk, q->perm_rep, q->inverse);
    case GEIST_FUSED_ATTN_QKV_PREP:
        /* Half-split RoPE per-head norm kernel: any row count, head_dim
         * must be even (mirrors metal_attn_qkv_prep's hd % 2 check). */
        return q->head_dim > 0 && (q->head_dim % 2u) == 0u;
    case GEIST_FUSED_ATTN_ROPE_APPEND:
        /* rope_append_rows stages a head in 512 floats of threadgroup
         * memory (mirrors metal_attn_rope_append's checks). */
        return q->head_dim > 0 && q->head_dim <= 512u && (q->head_dim % 2u) == 0u;
    case GEIST_FUSED_PLE_BLOCK:
        /* F32 gate/proj matrices AND m==1 (mirrors metal_ple_block's
         * checks — its kernels are naive decode GEMVs, rows==1 only).
         * Without the m check the plan binds fuse_ple_block_mN and every
         * gemma4 prefill fails at layer 0. */
        return q->m == 1 && q->gate_w != nullptr && q->up_w != nullptr &&
               q->gate_w->dtype == GEIST_DTYPE_F32 && q->up_w->dtype == GEIST_DTYPE_F32;
    case GEIST_FUSED_PLE_STAGE:
        /* ple_block's decode case, h in threadgroup memory (#196). */
        return q->m == 1 && q->gate_w != nullptr && q->up_w != nullptr &&
               q->gate_w->dtype == GEIST_DTYPE_F32 && q->up_w->dtype == GEIST_DTYPE_F32 &&
               q->d_model > 0 && q->d_model <= METAL_PLE_PRENORM_MAX_D && (q->d_model % 4u) == 0u &&
               q->inter > 0 && (q->inter % 4u) == 0u;
    case GEIST_FUSED_EMBEDDING_LOOKUP_SCALED:
        /* Dtype set of metal_embedding_lookup_scaled's row decoders. */
        switch ((enum geist_dtype) q->table_dtype) {
        case GEIST_DTYPE_F32:
        case GEIST_DTYPE_F16:
        case GEIST_DTYPE_BF16:
        case GEIST_DTYPE_Q4_0:
        case GEIST_DTYPE_Q8_0:
        case GEIST_DTYPE_Q4_K:
        case GEIST_DTYPE_Q5_K:
        case GEIST_DTYPE_Q6_K:
        case GEIST_DTYPE_PQ2_0:
            return true;
        default:
            return false;
        }
    default:
        return false;
    }
}

static const struct geist_backend_primitives metal_prims = {
        .rmsnorm      = metal_rmsnorm,
        .add          = metal_add,
        .mul          = metal_mul,
        .gelu_tanh    = metal_gelu_tanh,
        .silu         = metal_silu,
        .relu_squared = metal_relu_squared,
        .rope_apply   = metal_rope_apply,
        .attention    = metal_attention,
        .scale_f32    = metal_scale_f32,
};

static const struct geist_backend_fused metal_fused = {
        .supported                    = metal_fused_supported,
        .gelu_tanh_mul                = metal_gelu_tanh_mul,
        .silu_mul                     = metal_silu_mul,
        .linear_t                     = metal_linear_t,
        .linear_t_pair                = metal_linear_t_pair,
        .linear_t_triple              = metal_linear_t_triple,
        .embedding_lookup_scaled      = metal_embedding_lookup_scaled,
        .embedding_lookup_scaled_rows = metal_embedding_lookup_scaled_rows,
        .kv_append_f16                = metal_kv_append_f16,
        .argmax_f32                   = metal_argmax_f32,
        .ffn_gate_up                  = metal_ffn_gate_up,
        .ffn_gate_up_silu             = metal_ffn_gate_up_silu,
        .attn_qkv_prep                = metal_attn_qkv_prep,
        .attn_rope_append             = metal_attn_rope_append,
        .ple_block                    = metal_ple_block,
        .ple_stage                    = metal_ple_stage,
        .rmsnorm_add                  = metal_rmsnorm_add,
        .deltanet_mix                 = metal_deltanet_mix,
        .attn_qgate_split             = metal_attn_qgate_split,
        .sigmoid_mul                  = metal_sigmoid_mul,
        .hadamard_rotate              = metal_hadamard_rotate,
        .rmsnorm_add_norm             = metal_rmsnorm_add_norm,
};

/* The immutable device belongs to this live backend. This getter intentionally
 * does not touch command queues, inference state, buffer registries or budgets. */
static enum geist_status metal_resources_snapshot(const struct geist_backend     *be,
                                                  struct geist_backend_resources *out) {
    struct metal_state *st = be->state;
    if (st == nullptr || st->device == nullptr)
        return GEIST_E_INVALID_STATE;
    *out = (struct geist_backend_resources) {
            .allocated_bytes = metal_msg_send_ulong0(st, st->device, "currentAllocatedSize"),
            .source          = GEIST_RESOURCE_METAL_DEVICE,
            .unified_memory  = metal_msg_send_bool0(st, st->device, "hasUnifiedMemory"),
    };
    return GEIST_OK;
}

/* The working set the admission check (metal_budget_admit) enforces and
 * what is left of it. currentAllocatedSize counts this device's buffers in
 * this process only, so the answer is not device-wide. */
static enum geist_status metal_memory_info(const struct geist_backend  *be,
                                           struct geist_backend_memory *out) {
    struct metal_state *st = be->state;
    if (st == nullptr || st->device == nullptr)
        return GEIST_E_INVALID_STATE;
    if (st->ws_budget == 0)
        return GEIST_E_UNSUPPORTED;
    const uint64_t used = metal_msg_send_ulong0(st, st->device, "currentAllocatedSize");
    *out                = (struct geist_backend_memory) {
            .total_bytes    = st->ws_budget,
            .free_bytes     = used < st->ws_budget ? st->ws_budget - used : 0,
            .device_wide    = false,
            .unified_memory = metal_msg_send_bool0(st, st->device, "hasUnifiedMemory"),
    };
    return GEIST_OK;
}

const struct geist_backend_descriptor geist_backend_metal = {
        .name               = "metal",
        .resources_snapshot = metal_resources_snapshot,
        .memory_info        = metal_memory_info,
        .vtbl               = &metal_vtbl,
        .prims              = &metal_prims,
        .fused              = &metal_fused,
        .tunables           = metal_tunables,
        .caps               = {.kv_f16_attention  = true,
                               .kv_f16_by_default = true,
                               .batched_submit    = true,
                               /* Binding pages a buffer in whole (#529). */
                               .lookup_tables_on_host = true,
                               /* metal_linear_t switches on the resolver's dtypes. */
                               .resolve_weight_complete = true,
                               /* deltanet_mix encodes 64-token sub-chunks internally
                                * (#322), so DN models keep preferred_m_max. */
                 .dn_subchunk = true,
                 /* 512 would double the m_max-scaled logits scratch
                  * (m_max x VOCAB floats — 256 MB at vocab 262k) for no
                  * prefill gain; DN models are capped at 64 in
                  * arch_state anyway (O(C^2) chunk cost). */
                 .preferred_m_max   = 256,
                 .max_m             = 512, /* batched-submit pipeline bound */
                 .preferred_kv_mode = GEIST_KV_FP32},
};
