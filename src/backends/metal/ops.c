/*
 * src/backends/metal/ops.c — op implementations, encode helpers, probe, and the descriptor.
 *
 * Layer: BACKEND (metal). Split from the former monolithic backend.c;
 * pure moves, no behavior change.
 */
#include "metal_internal.h"

static void metal_encode_q4k_linear(struct metal_state            *st,
                                    void                          *enc,
                                    const struct geist_tensor     *x,
                                    const struct geist_tensor     *w,
                                    const struct geist_tensor     *y,
                                    const struct metal_q4k_params *params,
                                    bool                           m_tile8) {
    const bool m_tile16    = m_tile8 && params->rows >= METAL_Q4K_M16_TILE;
    const bool m_tile16_n2 = m_tile16 && st->use_q4k_m16_n2 &&
                             st->q4k_matmul_m16_n2_pipeline != nullptr && params->n_out >= 2u;
    const bool m_tile_sg   = m_tile8 && st->use_q4k_mm_sg && st->q4k_mm_sg_pipeline != nullptr &&
                             params->rows >= 32u && params->n_out >= 64u &&
                             (params->rows % 32u) == 0u && (params->n_out % 64u) == 0u;
    /* interior fast variant: no bounds checks, vectorized activation
     * staging (needs n_in%32 and 8-float-aligned x rows). */
    const bool m_tile_sg_fast = m_tile_sg && st->q4k_mm_sg_fast_pipeline != nullptr &&
                                (params->n_in % 32u) == 0u && (params->x_offset % 8u) == 0u &&
                                (params->x_row_stride % 8u) == 0u;
    const bool m_tile8_active = m_tile8 && !m_tile16;
    const bool n_tile4        = st->use_q4k_n4 && !m_tile8 && params->rows == 1u;
    metal_msg_send_set_pipeline(st,
                                enc,
                                m_tile_sg_fast   ? st->q4k_mm_sg_fast_pipeline
                                : m_tile_sg      ? st->q4k_mm_sg_pipeline
                                : m_tile16_n2    ? st->q4k_matmul_m16_n2_pipeline
                                : m_tile16       ? st->q4k_matmul_m16_pipeline
                                : m_tile8_active ? st->q4k_matmul_m8_pipeline
                                : n_tile4        ? st->q4k_n4_pipeline
                                                 : st->q4k_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, w->buffer->buffer, w->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    if (m_tile_sg) {
        metal_msg_send_set_threadgroup_memory(st, enc, m_tile_sg_fast ? 6144u : 8192u, 0u);
    }

    const struct metal_size groups = {
            .width  = n_tile4       ? (params->n_out + 3u) / 4u
                      : m_tile_sg   ? (params->rows + 31u) / 32u
                      : m_tile16_n2 ? (params->n_out + 1u) / 2u
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

static void metal_encode_q6k_linear(struct metal_state            *st,
                                    void                          *enc,
                                    const struct geist_tensor     *x,
                                    const struct geist_tensor     *w,
                                    const struct geist_tensor     *y,
                                    const struct metal_q4k_params *params,
                                    bool                           m_tile8) {
    /* rows==1 always takes the matvec kernel — the 64-row GEMM tile is a
     * waste for a single row, and n4 (llama mul_mv structure) wins at any
     * n_out that fills at least one threadgroup. */
    const bool single_row = params->rows == 1u;
    const bool n_tile4    = st->use_q6k_n4 && single_row && params->n_out >= 4u;
    const bool m_tile_sg  = m_tile8 && !n_tile4 && st->q6k_matmul_sg_pipeline != nullptr;
    /* interior fast variant: no bounds checks, vectorized activation
     * staging, direct simdgroup_store output. */
    const bool m_tile_sg_fast = m_tile_sg && st->q6k_matmul_sg_fast_pipeline != nullptr &&
                                (params->rows % 64u) == 0u && (params->n_out % 32u) == 0u &&
                                (params->x_offset % 8u) == 0u && (params->x_row_stride % 8u) == 0u;
    const bool m_tile16 = m_tile8 && !n_tile4 && !m_tile_sg && params->rows >= METAL_Q4K_M16_TILE;
    const bool m_tile8_active = m_tile8 && !n_tile4 && !m_tile_sg && !m_tile16;
    metal_msg_send_set_pipeline(st,
                                enc,
                                m_tile_sg_fast   ? st->q6k_matmul_sg_fast_pipeline
                                : m_tile_sg      ? st->q6k_matmul_sg_pipeline
                                : m_tile16       ? st->q6k_matmul_m16_pipeline
                                : m_tile8_active ? st->q6k_matmul_m8_pipeline
                                : n_tile4        ? st->q6k_n4_pipeline
                                                 : st->q6k_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, w->buffer->buffer, w->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);

    /* The q6k sg kernels are 128-thread / 4-simdgroup with a 32-output x
     * 64-batch-row tile (b0=tg.y*64, o0=tg.x*32). The old (n_out+7)/8 x
     * 32-thread dispatch here only ever ran simdgroup 0 — a latent bug the
     * old engine never hit because its prefill routed q6k through the
     * fused blocks, not this vtbl op. */
    const struct metal_size groups = {
            .width  = m_tile_sg ? (params->n_out + 31u) / 32u
                      : n_tile4 ? (params->n_out + 3u) / 4u
                                : params->n_out,
            .height = m_tile_sg  ? (params->rows + 63u) / 64u
                      : m_tile16 ? (params->rows + METAL_Q4K_M16_TILE - 1u) / METAL_Q4K_M16_TILE
                      : m_tile8_active ? (params->rows + METAL_Q4K_M_TILE - 1u) / METAL_Q4K_M_TILE
                                       : params->rows,
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

static void metal_encode_q40_q80_linear(struct metal_state            *st,
                                        void                          *enc,
                                        const struct geist_tensor     *x,
                                        const struct geist_tensor     *w,
                                        const struct geist_tensor     *y,
                                        const struct metal_q4k_params *params,
                                        enum geist_dtype               dtype) {
    void *n4 = dtype == GEIST_DTYPE_Q4_0     ? st->q40_n4_pipeline
               : dtype == GEIST_DTYPE_Q8_0   ? st->q80_n4_pipeline
               : dtype == GEIST_DTYPE_Q4_1   ? st->q41_n4_pipeline
               : dtype == GEIST_DTYPE_IQ4_NL ? st->iq4nl_n4_pipeline
               : dtype == GEIST_DTYPE_IQ4_XS ? st->iq4xs_n4_pipeline
               : dtype == GEIST_DTYPE_Q3_K   ? st->q3k_n4_pipeline
               : dtype == GEIST_DTYPE_IQ3_S  ? st->iq3s_n4_pipeline
                                             : st->q5k_n4_pipeline;
    void *mm = dtype == GEIST_DTYPE_Q4_0     ? st->q40_mm_pipeline
               : dtype == GEIST_DTYPE_Q8_0   ? st->q80_mm_pipeline
               : dtype == GEIST_DTYPE_Q4_1   ? st->q41_mm_pipeline
               : dtype == GEIST_DTYPE_IQ4_NL ? st->iq4nl_mm_pipeline
               : dtype == GEIST_DTYPE_IQ4_XS ? st->iq4xs_mm_pipeline
               : dtype == GEIST_DTYPE_Q3_K   ? st->q3k_mm_pipeline
               : dtype == GEIST_DTYPE_IQ3_S  ? st->iq3s_mm_pipeline
                                             : st->q5k_mm_pipeline;
    /* IQ4/Q3_K/IQ3_S: no naive fallback kernels and no fast GEMM
     * instances — n4 for rows==1, the bounded simdgroup GEMM for every
     * rows>=2 shape. */
    const bool iq4 = dtype == GEIST_DTYPE_IQ4_NL || dtype == GEIST_DTYPE_IQ4_XS ||
                     dtype == GEIST_DTYPE_Q3_K || dtype == GEIST_DTYPE_IQ3_S;
    /* rows==1 → simdgroup GEMV (llama mul_mv structure); rows>=8 → 64x32
     * simdgroup GEMM (bounds-checked, arbitrary rows/n_out); the naive
     * kernels remain the fallback for tiny shapes and the two kill-switch
     * envs (reused from the q4k levers). */
    const bool n_tile4 =
            params->rows == 1u && params->n_out >= 4u && (st->use_q4k_n4 || iq4) && n4 != nullptr;
    void      *mm_fast = dtype == GEIST_DTYPE_Q4_0     ? st->q40_mm_fast_pipeline
                         : dtype == GEIST_DTYPE_Q8_0   ? st->q80_mm_fast_pipeline
                         : dtype == GEIST_DTYPE_Q4_1   ? st->q41_mm_fast_pipeline
                         : dtype == GEIST_DTYPE_IQ4_XS ? st->iq4xs_mm_fast_pipeline
                         : iq4                         ? nullptr
                                                       : st->q5k_mm_fast_pipeline;
    const bool m_tile_sg =
            (iq4 ? params->rows >= 2u
                 : (params->rows >= 8u && params->n_out >= 64u && st->use_q4k_mm_sg)) &&
            mm != nullptr;
    /* interior fast variant: no bounds checks, vectorized activation
     * staging (needs full tiles, n_in%32 and 8-float-aligned x rows). */
    const bool m_tile_sg_fast = m_tile_sg && mm_fast != nullptr && (params->rows % 32u) == 0u &&
                                (params->n_out % 64u) == 0u && (params->n_in % 32u) == 0u &&
                                (params->x_offset % 8u) == 0u && (params->x_row_stride % 8u) == 0u;
    const bool tiled          = !n_tile4 && !m_tile_sg && params->rows >= 8u;
    void *base = dtype == GEIST_DTYPE_Q4_0   ? (tiled ? st->q40_m8_pipeline : st->q40_pipeline)
                 : dtype == GEIST_DTYPE_Q8_0 ? (tiled ? st->q80_m8_pipeline : st->q80_pipeline)
                 : dtype == GEIST_DTYPE_Q4_1 ? (tiled ? st->q41_m8_pipeline : st->q41_pipeline)
                                             : (tiled ? st->q5k_m8_pipeline : st->q5k_pipeline);
    metal_msg_send_set_pipeline(st,
                                enc,
                                n_tile4          ? n4
                                : m_tile_sg_fast ? mm_fast
                                : m_tile_sg      ? mm
                                                 : base);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, w->buffer->buffer, w->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    if (m_tile_sg) {
        metal_msg_send_set_threadgroup_memory(st, enc, m_tile_sg_fast ? 6144u : 8192u, 0u);
    }
    /* q40/q80 n4 kernels run 4 rows per simdgroup (8 per threadgroup);
     * q41/q5k still run 2 (4 per threadgroup). */
    const uint32_t n4_tile =
            (dtype == GEIST_DTYPE_Q4_0 || dtype == GEIST_DTYPE_Q8_0 || dtype == GEIST_DTYPE_IQ4_NL)
                    ? 8u
                    : 4u;
    const struct metal_size groups = {
            .width  = n_tile4     ? (params->n_out + n4_tile - 1u) / n4_tile
                      : m_tile_sg ? (params->rows + 31u) / 32u
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

static void metal_encode_rmsnorm_rows(struct metal_state             *st,
                                      void                           *enc,
                                      const struct geist_tensor      *x,
                                      const struct geist_tensor      *w,
                                      const struct geist_tensor      *y,
                                      const struct metal_rows_params *params) {

    metal_msg_send_set_pipeline(st,
                                enc,
                                st->use_rmsnorm_simd ? st->rmsnorm_rows_simd_pipeline
                                                     : st->rmsnorm_rows_pipeline);
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
            .width  = METAL_ELEM_THREADS,
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

    metal_msg_send_set_pipeline(st,
                                enc,
                                st->use_rmsnorm_simd ? st->rmsnorm_add_rows_simd_pipeline
                                                     : st->rmsnorm_add_rows_pipeline);
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
            .width  = METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_RMSNORM_ADD_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_add_rows(struct metal_state                    *st,
                                  void                                  *enc,
                                  const struct geist_tensor             *a,
                                  const struct geist_tensor             *b,
                                  const struct geist_tensor             *y,
                                  const struct metal_binary_rows_params *params) {
    metal_msg_send_set_pipeline(st, enc, st->add_rows_pipeline);
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
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ADD_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_mul_rows(struct metal_state                    *st,
                                  void                                  *enc,
                                  const struct geist_tensor             *a,
                                  const struct geist_tensor             *b,
                                  const struct geist_tensor             *y,
                                  const struct metal_binary_rows_params *params) {
    metal_msg_send_set_pipeline(st, enc, st->mul_rows_pipeline);
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
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_MUL_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_gelu_mul_rows(struct metal_state                    *st,
                                       void                                  *enc,
                                       const struct geist_tensor             *a,
                                       const struct geist_tensor             *b,
                                       const struct geist_tensor             *y,
                                       const struct metal_binary_rows_params *params) {

    metal_msg_send_set_pipeline(st, enc, st->gelu_mul_rows_pipeline);
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
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_GELU_MUL_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_silu_mul_rows(struct metal_state                    *st,
                                       void                                  *enc,
                                       const struct geist_tensor             *a,
                                       const struct geist_tensor             *b,
                                       const struct geist_tensor             *y,
                                       const struct metal_binary_rows_params *params) {

    metal_msg_send_set_pipeline(st, enc, st->silu_mul_rows_pipeline);
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
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_GELU_MUL_ROWS, groups);
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

static void metal_encode_silu_rows(struct metal_state                   *st,
                                   void                                 *enc,
                                   const struct geist_tensor            *x,
                                   const struct geist_tensor            *y,
                                   const struct metal_scale_rows_params *params) {
    metal_msg_send_set_pipeline(st, enc, st->silu_rows_pipeline);
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
    const size_t            half   = (size_t) params->head_dim / 2u;
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
     * the head_dim-512 full-attention layers take (flash gate is <=256) —
     * it was invisible to the profiler/skips until 2026-07-04. */
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ATTENTION_QNORM_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_f32_matmul(struct metal_state            *st,
                                    void                          *enc,
                                    const struct geist_tensor     *x,
                                    const struct geist_tensor     *w,
                                    const struct geist_tensor     *y,
                                    const struct metal_f32_params *params) {
    /* Multi-row (prefill): full-tile shapes take the 64x32 4-simdgroup GEMM
     * (mm_sg structure, f32 staging = bit-identical to the 8x8 kernel);
     * others the 8x8 simdgroup GEMM. Single-row keeps the reduction kernel. */
    const bool use_sg = params->rows > 1u && st->f32_matmul_sg_pipeline != nullptr;
    const bool use_mm = use_sg && st->f32_matmul_mm_pipeline != nullptr &&
                        !metal_env_disabled("GEIST_METAL_F32_MM") && (params->rows % 32u) == 0u &&
                        (params->n_out % 64u) == 0u && (params->n_in % 32u) == 0u &&
                        (params->x_offset % 8u) == 0u && (params->x_row_stride % 8u) == 0u &&
                        (params->w_offset % 4u) == 0u;
    metal_msg_send_set_pipeline(st,
                                enc,
                                use_mm   ? st->f32_matmul_mm_pipeline
                                : use_sg ? st->f32_matmul_sg_pipeline
                                         : st->f32_matmul_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, w->buffer->buffer, w->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    if (use_mm) {
        /* sa 64x32 f32 + sb 32x32 f32 */
        metal_msg_send_set_threadgroup_memory(st, enc, 12288u, 0u);
    }
    const struct metal_size groups = {
            .width  = use_mm   ? params->rows / 32u
                      : use_sg ? (params->n_out + 7u) / 8u
                               : params->n_out,
            .height = use_mm   ? params->n_out / 64u
                      : use_sg ? (params->rows + 7u) / 8u
                               : params->rows,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = use_sg ? 32u : METAL_ELEM_THREADS,
            .height = use_mm ? 4u : 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_F32_MATMUL, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
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
    ok = ok && metal_tensor_is_f32_matrix(w, &w_rows, &w_cols, &w_offset, &w_row_stride) &&
         w_row_stride == w_cols && w_cols == n_in && y_cols == w_rows && y_rows == rows;
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

    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal F32 linear: command sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_f32_matmul(st, metal_sequence_encoder(st), x, w, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal F32 linear: command buffer failed");
        return GEIST_E_BACKEND;
    }
    void *enc = metal_msg_send_id0(st, cmd, "computeCommandEncoder");
    if (enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal F32 linear: encoder failed");
        return GEIST_E_BACKEND;
    }
    metal_encode_f32_matmul(st, enc, x, w, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");

    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal F32 linear: command failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
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
    if (st->sequence_active) {
        metal_encode_rmsnorm_rows(st, metal_sequence_encoder(st), x, w, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_rmsnorm_rows(st, enc, x, w, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
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
    if (st->sequence_active) {
        metal_encode_rmsnorm_add_rows(st, metal_sequence_encoder(st), res, x, w, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_rmsnorm_add_rows(st, enc, res, x, w, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status metal_add(struct geist_backend      *be,
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
    if (st->sequence_active) {
        metal_encode_add_rows(st, metal_sequence_encoder(st), a, b, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_add_rows(st, enc, a, b, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status metal_mul(struct geist_backend      *be,
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
    if (st->sequence_active) {
        metal_encode_mul_rows(st, metal_sequence_encoder(st), a, b, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_mul_rows(st, enc, a, b, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
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
    if (st->sequence_active) {
        metal_encode_scale_rows(st, metal_sequence_encoder(st), x, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_scale_rows(st, enc, x, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
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
    if (st->sequence_active) {
        metal_encode_qgate_split(st, metal_sequence_encoder(st), joint, q, gate, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (enc == nullptr)
        return GEIST_E_BACKEND;
    metal_encode_qgate_split(st, enc, joint, q, gate, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
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
    if (st->sequence_active) {
        metal_encode_sigmoid_mul(st, metal_sequence_encoder(st), x, gate, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (enc == nullptr)
        return GEIST_E_BACKEND;
    metal_encode_sigmoid_mul(st, enc, x, gate, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
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
    if (st->sequence_active) {
        metal_encode_gelu_rows(st, metal_sequence_encoder(st), x, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_gelu_rows(st, enc, x, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status
metal_silu(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y) {
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
    if (st->sequence_active) {
        metal_encode_silu_rows(st, metal_sequence_encoder(st), x, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_silu_rows(st, enc, x, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status metal_gelu_tanh_mul(struct geist_backend      *be,
                                                           const struct geist_tensor *x,
                                                           const struct geist_tensor *z,
                                                           struct geist_tensor       *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, x_off = 0, x_stride = 0;
    size_t z_rows = 0, z_cols = 0, z_off = 0, z_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &cols, &x_off, &x_stride) ||
        !metal_tensor_is_f32_rows(z, &z_rows, &z_cols, &z_off, &z_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) || z_rows != rows ||
        y_rows != rows || z_cols != cols || y_cols != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || x_off > UINT32_MAX || z_off > UINT32_MAX ||
        y_off > UINT32_MAX || x_stride > UINT32_MAX || z_stride > UINT32_MAX ||
        y_stride > UINT32_MAX || x->buffer->owner != be->state || z->buffer->owner != be->state ||
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
            .a_offset     = (uint32_t) x_off,
            .b_offset     = (uint32_t) z_off,
            .y_offset     = (uint32_t) y_off,
            .a_row_stride = (uint32_t) x_stride,
            .b_row_stride = (uint32_t) z_stride,
            .y_row_stride = (uint32_t) y_stride,
    };
    if (st->sequence_active) {
        metal_encode_gelu_mul_rows(st, metal_sequence_encoder(st), x, z, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_gelu_mul_rows(st, enc, x, z, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status metal_silu_mul(struct geist_backend      *be,
                                                      const struct geist_tensor *x,
                                                      const struct geist_tensor *z,
                                                      struct geist_tensor       *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, x_off = 0, x_stride = 0;
    size_t z_rows = 0, z_cols = 0, z_off = 0, z_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &cols, &x_off, &x_stride) ||
        !metal_tensor_is_f32_rows(z, &z_rows, &z_cols, &z_off, &z_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) || z_rows != rows ||
        y_rows != rows || z_cols != cols || y_cols != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || x_off > UINT32_MAX || z_off > UINT32_MAX ||
        y_off > UINT32_MAX || x_stride > UINT32_MAX || z_stride > UINT32_MAX ||
        y_stride > UINT32_MAX || x->buffer->owner != be->state || z->buffer->owner != be->state ||
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
            .a_offset     = (uint32_t) x_off,
            .b_offset     = (uint32_t) z_off,
            .y_offset     = (uint32_t) y_off,
            .a_row_stride = (uint32_t) x_stride,
            .b_row_stride = (uint32_t) z_stride,
            .y_row_stride = (uint32_t) y_stride,
    };
    if (st->sequence_active) {
        metal_encode_silu_mul_rows(st, metal_sequence_encoder(st), x, z, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_silu_mul_rows(st, enc, x, z, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
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
               (embed_table->dtype == GEIST_DTYPE_Q4_0 || embed_table->dtype == GEIST_DTYPE_Q8_0)) {
        if ((d_model % METAL_Q40_Q80_BLOCK_ELEMS) != 0) {
            return GEIST_E_INVALID_ARG;
        }
        blocks_per_row           = d_model / METAL_Q40_Q80_BLOCK_ELEMS;
        const size_t block_bytes = embed_table->dtype == GEIST_DTYPE_Q4_0 ? METAL_Q40_BLOCK_BYTES
                                                                          : METAL_Q80_BLOCK_BYTES;
        if (blocks_per_row > SIZE_MAX / block_bytes) {
            return GEIST_E_INVALID_ARG;
        }
        row_bytes = blocks_per_row * block_bytes;
    } else if (embed_table->layout == GEIST_LAYOUT_BLOCK_QUANTIZED &&
               embed_table->dtype == GEIST_DTYPE_Q4_K) {
        if ((d_model % METAL_Q4K_BLOCK_ELEMS) != 0) {
            return GEIST_E_INVALID_ARG;
        }
        blocks_per_row = d_model / METAL_Q4K_BLOCK_ELEMS;
        if (blocks_per_row > SIZE_MAX / METAL_Q4K_BLOCK_BYTES) {
            return GEIST_E_INVALID_ARG;
        }
        row_bytes = blocks_per_row * METAL_Q4K_BLOCK_BYTES;
    } else if (embed_table->layout == GEIST_LAYOUT_BLOCK_QUANTIZED &&
               embed_table->dtype == GEIST_DTYPE_Q5_K) {
        if ((d_model % METAL_Q5K_BLOCK_ELEMS) != 0) {
            return GEIST_E_INVALID_ARG;
        }
        blocks_per_row = d_model / METAL_Q5K_BLOCK_ELEMS;
        if (blocks_per_row > SIZE_MAX / METAL_Q5K_BLOCK_BYTES) {
            return GEIST_E_INVALID_ARG;
        }
        row_bytes = blocks_per_row * METAL_Q5K_BLOCK_BYTES;
    } else if (embed_table->layout == GEIST_LAYOUT_BLOCK_QUANTIZED &&
               embed_table->dtype == GEIST_DTYPE_Q6_K) {
        if ((d_model % METAL_Q6K_BLOCK_ELEMS) != 0) {
            return GEIST_E_INVALID_ARG;
        }
        blocks_per_row = d_model / METAL_Q6K_BLOCK_ELEMS;
        if (blocks_per_row > SIZE_MAX / METAL_Q6K_BLOCK_BYTES) {
            return GEIST_E_INVALID_ARG;
        }
        row_bytes = blocks_per_row * METAL_Q6K_BLOCK_BYTES;
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

    if (st->sequence_active) {
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

    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "metal embedding_lookup_scaled: command buffer failed");
        return GEIST_E_BACKEND;
    }
    void *enc = metal_msg_send_id0(st, cmd, "computeCommandEncoder");
    if (enc == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "metal embedding_lookup_scaled: encoder failed");
        return GEIST_E_BACKEND;
    }

    metal_encode_embed_lookup_scaled(st, enc, embed_table, out, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");

    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal embedding_lookup_scaled: command failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

/* Batched twin (#322 step 3): one dispatch embeds a whole prefill chunk.
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
    if (!st->sequence_active) {
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

[[nodiscard]] static enum geist_status
metal_embedding_lookup(struct geist_backend      *be,
                       const struct geist_tensor *embed_table,
                       geist_token_t              token_id,
                       struct geist_tensor       *out) {

    return metal_embedding_lookup_scaled(be, embed_table, token_id, 1.0f, out);
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
    struct metal_state           *st     = be->state;
    const struct metal_q4k_params params = {
            .n_in  = (uint32_t) n_in,
            .n_out = (uint32_t) n_out,
            .rows  = (uint32_t) rows,
            .blocks_per_row =
                    (uint32_t) (n_in / ((dtype == GEIST_DTYPE_IQ4_XS || dtype == GEIST_DTYPE_Q3_K ||
                                         dtype == GEIST_DTYPE_IQ3_S)
                                                ? METAL_IQ4XS_BLOCK_ELEMS
                                                : METAL_Q40_Q80_BLOCK_ELEMS)),
            .x_offset      = (uint32_t) x_offset,
            .w_byte_offset = (uint32_t) w_offset,
            .y_offset      = (uint32_t) y_offset,
            .x_row_stride  = (uint32_t) x_row_stride,
            .y_row_stride  = (uint32_t) y_row_stride,
    };
    if (st->sequence_active) {
        metal_encode_q40_q80_linear(st, metal_sequence_encoder(st), x, w, y, &params, dtype);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_q40_q80_linear(st, enc, x, w, y, &params, dtype);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
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

    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal Q4_K matvec: command sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_q4k_linear(st, metal_sequence_encoder(st), x, w, y, &params, false);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal Q4_K matvec: command buffer failed");
        return GEIST_E_BACKEND;
    }
    void *enc = metal_msg_send_id0(st, cmd, "computeCommandEncoder");
    if (enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal Q4_K matvec: encoder failed");
        return GEIST_E_BACKEND;
    }

    metal_encode_q4k_linear(st, enc, x, w, y, &params, false);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");

    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal Q4_K matvec: command failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
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

    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal Q4_K matmul: command sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_q4k_linear(st, metal_sequence_encoder(st), x, w, y, &params, true);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal Q4_K matmul: command buffer failed");
        return GEIST_E_BACKEND;
    }
    void *enc = metal_msg_send_id0(st, cmd, "computeCommandEncoder");
    if (enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal Q4_K matmul: encoder failed");
        return GEIST_E_BACKEND;
    }

    metal_encode_q4k_linear(st, enc, x, w, y, &params, true);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");

    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal Q4_K matmul: command failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
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
    if (st->sequence_active) {
        metal_encode_q40_q80_linear(
                st, metal_sequence_encoder(st), x, w, y, &params, GEIST_DTYPE_Q5_K);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_q40_q80_linear(st, enc, x, w, y, &params, GEIST_DTYPE_Q5_K);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
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

    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal Q6_K linear: command sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_q6k_linear(st, metal_sequence_encoder(st), x, w, y, &params, matrix);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal Q6_K linear: command buffer failed");
        return GEIST_E_BACKEND;
    }
    void *enc = metal_msg_send_id0(st, cmd, "computeCommandEncoder");
    if (enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal Q6_K linear: encoder failed");
        return GEIST_E_BACKEND;
    }

    metal_encode_q6k_linear(st, enc, x, w, y, &params, matrix);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");

    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal Q6_K linear: command failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
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

    {
        static _Atomic int dbg = -1;
        if (dbg < 0) {
            const char *e = getenv("GEIST_METAL_DEBUG_LINEAR");
            dbg           = (e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0) ? 1 : 0;
        }
        if (dbg && x->buffer != nullptr && x->buffer->mapped != nullptr && cos->buffer != nullptr &&
            cos->buffer->mapped != nullptr) {
            const float *xp = (const float *) ((const uint8_t *) x->buffer->mapped + x->offset);
            const float *cp = (const float *) ((const uint8_t *) cos->buffer->mapped + cos->offset);
            size_t       nx = 1, nc = 1;
            for (int i = 0; i < x->ndim; i++)
                nx *= (size_t) x->shape[i];
            for (int i = 0; i < cos->ndim; i++)
                nc *= (size_t) cos->shape[i];
            size_t nanx = 0, nanc = 0;
            for (size_t i = 0; i < nx; i++)
                if (isnan(xp[i]))
                    nanx++;
            for (size_t i = 0; i < nc; i++)
                if (isnan(cp[i]))
                    nanc++;
            fprintf(stderr,
                    "rope_in x[%lld,%lld,%lld] nanx=%zu cos[%lld,%lld] nanc=%zu xoff=%zu "
                    "coff=%zu\n",
                    (long long) x->shape[0],
                    (long long) x->shape[1],
                    (long long) x->shape[2],
                    nanx,
                    (long long) cos->shape[0],
                    (long long) cos->shape[1],
                    nanc,
                    x->offset,
                    cos->offset);
        }
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
    if (head_dim == 0 || (head_dim % 2u) != 0 || cos_rows != rows || sin_rows != rows ||
        cos_cols != head_dim || sin_cols != head_dim || cos_stride != head_dim ||
        sin_stride != head_dim) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "metal rope_apply: shape mismatch");
        return GEIST_E_INVALID_ARG;
    }
    if (rows > UINT32_MAX || heads > UINT32_MAX || head_dim > UINT32_MAX || x_offset > UINT32_MAX ||
        cos_offset > UINT32_MAX || sin_offset > UINT32_MAX || rows > UINT32_MAX / heads ||
        rows * heads > UINT32_MAX / (head_dim / 2u) || x->buffer->owner != be->state ||
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
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal rope_apply: sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_rope_rows(st, metal_sequence_encoder(st), x, cos, sin, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal rope_apply: command encoder failed");
        return GEIST_E_BACKEND;
    }
    metal_encode_rope_rows(st, enc, x, cos, sin, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal rope_apply: command failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

/* rows>1 f32-KV fast path: convert K/V to persistent f16 staging (one
 * kv_append_rows_f16 dispatch) and run the no-norm simdgroup flash kernel.
 * The scalar f32 kernel this replaces is the dominant prefill cost; the
 * conversion is ~1%% of the savings. Serial-encoder ordering makes the
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

    const size_t kv_out    = k_heads * head_dim;
    const size_t elems     = k_rows * kv_out;
    const size_t f16_bytes = elems * 2u;
    if (!kv_native_f16 && st->attn_kvf16_capacity < f16_bytes) {
        metal_buffer_destroy_internal(be, st->attn_kf16_buffer);
        metal_buffer_destroy_internal(be, st->attn_vf16_buffer);
        st->attn_kf16_buffer    = nullptr;
        st->attn_vf16_buffer    = nullptr;
        st->attn_kvf16_capacity = 0;
        const size_t      cap   = f16_bytes * 2u; /* headroom: no regrow per chunk */
        enum geist_status bs =
                metal_new_buffer(be, cap, GEIST_BUFFER_SCRATCH, 0, true, &st->attn_kf16_buffer);
        if (bs == GEIST_OK) {
            bs = metal_new_buffer(be, cap, GEIST_BUFFER_SCRATCH, 0, true, &st->attn_vf16_buffer);
        }
        if (bs != GEIST_OK) {
            metal_buffer_destroy_internal(be, st->attn_kf16_buffer);
            st->attn_kf16_buffer = nullptr;
            return bs;
        }
        st->attn_kvf16_capacity = cap;
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

    void *cmd = nullptr;
    void *enc = nullptr;
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            return GEIST_E_BACKEND;
        }
        enc = metal_sequence_encoder(st);
    } else {
        cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
        enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
        if (cmd == nullptr || enc == nullptr) {
            return GEIST_E_BACKEND;
        }
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

    /* head_dim <= 256: 4-simdgroup kernel (128 threads). 256 < hd <= 512
     * (gemma-3n full-attention layers): 8-simdgroup variant (256 threads),
     * one query row + one 64-column output slice per simdgroup. */
    const bool sg8 = head_dim > 256u;
    metal_msg_send_set_pipeline(st,
                                enc,
                                sg8 ? st->attention_flash_sg8_f16_pipeline
                                    : st->attention_flash_sg_f16_pipeline);
    metal_msg_send_set_buffer(st, enc, q->buffer->buffer, q->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, kf16, 0, 1);
    metal_msg_send_set_buffer(st, enc, vf16, 0, 2);
    metal_msg_send_set_buffer(st, enc, out->buffer->buffer, out->buffer->base_off, 3);
    metal_msg_send_set_bytes(st, enc, &fp, sizeof(fp), 4);
    const struct metal_size fgroups  = {q_rows / 8u, q_heads, 1};
    const struct metal_size fthreads = {sg8 ? 256u : 128u, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ATTENTION_ROWS, fgroups);
    metal_msg_send_dispatch(st, enc, fgroups, fthreads);

    if (st->sequence_active) {
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal flash attention: command failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

/* rows==1 (decode) f32-KV fast path: convert K/V to the f16 staging and run
 * the split-KV decode kernel — the scalar kernel uses one threadgroup per
 * head (8 tgs on a 32-core GPU) and its cost grows linearly with kv_len. */
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

    const size_t kv_out    = k_heads * head_dim;
    const size_t elems     = k_rows * kv_out;
    const size_t f16_bytes = elems * 2u;
    if (!kv_native_f16 && st->attn_kvf16_capacity < f16_bytes) {
        metal_buffer_destroy_internal(be, st->attn_kf16_buffer);
        metal_buffer_destroy_internal(be, st->attn_vf16_buffer);
        st->attn_kf16_buffer    = nullptr;
        st->attn_vf16_buffer    = nullptr;
        st->attn_kvf16_capacity = 0;
        const size_t      cap   = f16_bytes * 2u;
        enum geist_status bs =
                metal_new_buffer(be, cap, GEIST_BUFFER_SCRATCH, 0, true, &st->attn_kf16_buffer);
        if (bs == GEIST_OK) {
            bs = metal_new_buffer(be, cap, GEIST_BUFFER_SCRATCH, 0, true, &st->attn_vf16_buffer);
        }
        if (bs != GEIST_OK) {
            metal_buffer_destroy_internal(be, st->attn_kf16_buffer);
            st->attn_kf16_buffer = nullptr;
            return bs;
        }
        st->attn_kvf16_capacity = cap;
    }
    uint32_t window = (uint32_t) q_offset + 1u;
    if ((uint32_t) k_rows < window) {
        window = (uint32_t) k_rows;
    }
    if (sliding_window > 0u && (uint32_t) sliding_window < window) {
        window = (uint32_t) sliding_window;
    }
    uint32_t nsplit = window / 32u;
    if (nsplit < 1u) {
        nsplit = 1u;
    }
    if (nsplit > 16u) {
        nsplit = 16u;
    }
    const size_t partial_bytes = q_heads * 16u * (head_dim + 2u) * sizeof(float);
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
    struct {
        uint32_t elems, kv_out, k_offset, v_offset, k_cache_offset, v_cache_offset, q_position;
    } ap = {(uint32_t) elems, (uint32_t) kv_out, (uint32_t) k_off, (uint32_t) v_off, 0u, 0u, 0u};
    struct {
        uint32_t rows, kv_len, qh, kvh, hd, qpos, sw, qo, kco, vco, yo;
    } fp       = {1u,
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

    void *cmd = nullptr;
    void *enc = nullptr;
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            return GEIST_E_BACKEND;
        }
        enc = metal_sequence_encoder(st);
    } else {
        cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
        enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
        if (cmd == nullptr || enc == nullptr) {
            return GEIST_E_BACKEND;
        }
    }
    const struct metal_size threads256 = {256, 1, 1};
    if (!kv_native_f16) {
        metal_msg_send_set_pipeline(st, enc, st->kv_append_rows_f16_pipeline);
        metal_msg_send_set_buffer(st, enc, k->buffer->buffer, k->buffer->base_off, 0);
        metal_msg_send_set_buffer(st, enc, value->buffer->buffer, value->buffer->base_off, 1);
        metal_msg_send_set_buffer(
                st, enc, st->attn_kf16_buffer->buffer, st->attn_kf16_buffer->base_off, 2);
        metal_msg_send_set_buffer(
                st, enc, st->attn_vf16_buffer->buffer, st->attn_vf16_buffer->base_off, 3);
        metal_msg_send_set_bytes(st, enc, &ap, sizeof(ap), 4);
        const struct metal_size cgroups = {(elems + 255u) / 256u, 1, 1};
        metal_msg_send_dispatch(st, enc, cgroups, threads256);
    }

    /* head_dim <= 256: 8-chunk lane unroll; 256 < hd <= 512 (gemma-3n
     * full-attention layers): the 16-chunk dec512 variant. */
    metal_msg_send_set_pipeline(st,
                                enc,
                                head_dim > 256u ? st->attention_dec512_f16_pipeline
                                                : st->attention_dec_f16_pipeline);
    metal_msg_send_set_buffer(st, enc, q->buffer->buffer, q->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, kf16, 0, 1);
    metal_msg_send_set_buffer(st, enc, vf16, 0, 2);
    metal_msg_send_set_buffer(st,
                              enc,
                              st->attn_dec_partials_buffer->buffer,
                              st->attn_dec_partials_buffer->base_off,
                              3);
    metal_msg_send_set_bytes(st, enc, &fp, sizeof(fp), 4);
    metal_msg_send_set_bytes(st, enc, &nsplit, sizeof(nsplit), 5);
    const struct metal_size dgroups = {nsplit, q_heads, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ATTENTION_ROWS, dgroups);
    metal_msg_send_dispatch(st, enc, dgroups, threads256);

    const uint32_t cb[4] = {(uint32_t) q_heads, (uint32_t) head_dim, nsplit, (uint32_t) out_off};
    metal_msg_send_set_pipeline(st, enc, st->attention_dec_combine_pipeline);
    metal_msg_send_set_buffer(st,
                              enc,
                              st->attn_dec_partials_buffer->buffer,
                              st->attn_dec_partials_buffer->base_off,
                              0);
    metal_msg_send_set_buffer(st, enc, out->buffer->buffer, out->buffer->base_off, 1);
    metal_msg_send_set_bytes(st, enc, cb, sizeof(cb), 2);
    const struct metal_size ggroups = {1, q_heads, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ATTENTION_ROWS, ggroups);
    metal_msg_send_dispatch(st, enc, ggroups, threads256);

    if (st->sequence_active) {
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal decode attention: command failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
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
    void *cmd = nullptr;
    void *enc = nullptr;
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            return GEIST_E_BACKEND;
        }
        enc = metal_sequence_encoder(st);
    } else {
        cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
        enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
        if (cmd == nullptr || enc == nullptr) {
            return GEIST_E_BACKEND;
        }
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
    if (st->sequence_active) {
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
}

/* Fused FFN gate+up matvec with GeGLU epilogue (vtbl slot): the restored
 * wip gate_up_q4k_n4 kernel — llama mul_mv structure, 2 rows/simdgroup,
 * both weights against one activation pass, gelu(g)*u written directly.
 * rows==1 (decode) and Q4_K only; prefill keeps the mm_sg GEMMs. */
[[nodiscard]] static enum geist_status metal_ffn_gate_up(struct geist_backend      *be,
                                                         const struct geist_tensor *x,
                                                         const struct geist_tensor *gate_w,
                                                         const struct geist_tensor *up_w,
                                                         struct geist_tensor       *y) {

    if (be == nullptr || be->state == nullptr || x == nullptr || gate_w == nullptr ||
        up_w == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, d_in = 0, x_off = 0, x_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    size_t g_out = 0, g_in = 0, g_woff = 0;
    size_t u_out = 0, u_in = 0, u_woff = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &d_in, &x_off, &x_stride) ||
        !metal_tensor_is_q4k_matrix(gate_w, &g_out, &g_in, &g_woff) ||
        !metal_tensor_is_q4k_matrix(up_w, &u_out, &u_in, &u_woff) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) ||
        rows != 1u || /* decode only: matvec-structured kernel */
        g_in != d_in || u_in != d_in || u_out != g_out || (d_in % 256u) != 0u || y_rows != rows ||
        y_cols != g_out) {
        return GEIST_E_UNSUPPORTED;
    }
    if (d_in > UINT32_MAX || g_out > UINT32_MAX || x_off > UINT32_MAX || g_woff > UINT32_MAX ||
        u_woff > UINT32_MAX || y_off > UINT32_MAX || x->buffer->owner != be->state ||
        gate_w->buffer->owner != be->state || up_w->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st = be->state;
    if (st->q4k_gate_up_n4_pipeline == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    const struct metal_q4k_gate_up_params params = {
            .n_in               = (uint32_t) d_in,
            .n_out              = (uint32_t) g_out,
            .rows               = (uint32_t) rows,
            .blocks_per_row     = (uint32_t) (d_in / 256u),
            .x_offset           = (uint32_t) x_off,
            .gate_w_byte_offset = (uint32_t) g_woff,
            .up_w_byte_offset   = (uint32_t) u_woff,
            .gate_y_offset      = (uint32_t) y_off,
            .up_y_offset        = (uint32_t) y_off, /* unused by the epilogue kernel */
            .x_row_stride       = (uint32_t) x_stride,
            .y_row_stride       = (uint32_t) y_stride,
    };
    void *cmd = nullptr;
    void *enc = nullptr;
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            return GEIST_E_BACKEND;
        }
        enc = metal_sequence_encoder(st);
    } else {
        cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
        enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
        if (cmd == nullptr || enc == nullptr) {
            return GEIST_E_BACKEND;
        }
    }
    metal_msg_send_set_pipeline(st, enc, st->q4k_gate_up_n4_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, gate_w->buffer->buffer, gate_w->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, up_w->buffer->buffer, up_w->buffer->base_off, 2);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 3);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, y->buffer->base_off, 4);
    metal_msg_send_set_bytes(st, enc, &params, sizeof(params), 5);
    const struct metal_size groups  = {(uint32_t) ((g_out + 3u) / 4u), (uint32_t) rows, 1};
    const struct metal_size threads = {METAL_Q4K_N4_THREADS, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_Q4K_GATE_UP_N4, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
    if (st->sequence_active) {
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
}

/* Fused gemma attention q/k/v prep (vtbl slot): q_norm_rope_rows for q,
 * kv_norm_append_rows{,_f16} for k+v (norm + RoPE + cache append in one
 * dispatch). Replaces up to six decomposed ops with two dispatches. */
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

    void *cmd = nullptr;
    void *enc = nullptr;
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            return GEIST_E_BACKEND;
        }
        enc = metal_sequence_encoder(st);
    } else {
        cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
        enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
        if (cmd == nullptr || enc == nullptr) {
            return GEIST_E_BACKEND;
        }
    }
    const struct metal_size threads256 = {256, 1, 1};
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

    if (st->sequence_active) {
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
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
        ple_in == nullptr || proj_w == nullptr || res == nullptr || norm_w == nullptr ||
        gate_scratch == nullptr || proj_scratch == nullptr || y == nullptr) {
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
        !metal_tensor_is_f32_vector(norm_w, &nw_n, &nw_off) ||
        !metal_tensor_is_f32_rows(gate_scratch, &g_rows, &g_cols, &g_off, &g_stride) ||
        !metal_tensor_is_f32_rows(proj_scratch, &ps_rows, &ps_cols, &ps_off, &ps_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) ||
        rows != 1u || /* decode only: naive GEMV kernels */
        gw_in != d_in || p_rows != rows || p_cols != hpl || g_rows != rows || g_cols != hpl ||
        pw_in != hpl || r_rows != rows || r_cols != d_model || ps_rows != rows ||
        ps_cols != d_model || y_rows != rows || y_cols != d_model || nw_n != d_model) {
        return GEIST_E_UNSUPPORTED;
    }
    if (d_in > UINT32_MAX || hpl > UINT32_MAX || d_model > UINT32_MAX || x_off > UINT32_MAX ||
        gw_off > UINT32_MAX || p_off > UINT32_MAX || g_off > UINT32_MAX || pw_off > UINT32_MAX ||
        r_off > UINT32_MAX || nw_off > UINT32_MAX || y_off > UINT32_MAX ||
        x->buffer->owner != be->state || gate_w->buffer->owner != be->state ||
        ple_in->buffer->owner != be->state || proj_w->buffer->owner != be->state ||
        res->buffer->owner != be->state || norm_w->buffer->owner != be->state ||
        gate_scratch->buffer->owner != be->state || y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st = be->state;
    if (!st->use_ple_block || st->f32_ple_gate_pipeline == nullptr ||
        st->f32_ple_proj_norm_pipeline == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    struct {
        uint32_t ni, no, rows, xo, wo, po, yo, xs, ps, ys;
    } gp      = {(uint32_t) d_in,
                 (uint32_t) hpl,
                 (uint32_t) rows,
                 (uint32_t) x_off,
                 (uint32_t) gw_off,
                 (uint32_t) p_off,
                 (uint32_t) g_off,
                 (uint32_t) x_stride,
                 (uint32_t) p_stride,
                 (uint32_t) g_stride};
    void *cmd = nullptr;
    void *enc = nullptr;
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            return GEIST_E_BACKEND;
        }
        enc = metal_sequence_encoder(st);
    } else {
        cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
        enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
        if (cmd == nullptr || enc == nullptr) {
            return GEIST_E_BACKEND;
        }
    }
    const struct metal_size threads256 = {256, 1, 1};
    metal_msg_send_set_pipeline(st, enc, st->f32_ple_gate_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, x->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, gate_w->buffer->buffer, gate_w->buffer->base_off, 1);
    metal_msg_send_set_buffer(st, enc, ple_in->buffer->buffer, ple_in->buffer->base_off, 2);
    metal_msg_send_set_buffer(
            st, enc, gate_scratch->buffer->buffer, gate_scratch->buffer->base_off, 3);
    metal_msg_send_set_bytes(st, enc, &gp, sizeof(gp), 4);
    const struct metal_size ggroups = {(uint32_t) hpl, (uint32_t) rows, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_F32_PLE_GATE, ggroups);
    metal_msg_send_dispatch(st, enc, ggroups, threads256);

    if (st->sequence_active) {
        st->sequence_has_work = true;
    } else {
        metal_msg_send_void0(st, enc, "endEncoding");
        metal_msg_send_void0(st, cmd, "commit");
        metal_msg_send_void0(st, cmd, "waitUntilCompleted");
        if (metal_msg_send_id0(st, cmd, "error") != nullptr) {
            return GEIST_E_BACKEND;
        }
    }

    /* Proj side: the fused ple_proj_norm_f32 kernel is a single-threadgroup
     * GEMV — 400us/layer at d_model 2048 (measured 2026-07-04, decode
     * 18.7->32.9 ms/tok). Route through the fast f32 matvec + fused
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
    if (s != GEIST_OK) {
        return s;
    }
    return metal_rmsnorm_add(be, res, proj_scratch, norm_w, eps, y);
}

/* Device greedy argmax (vtbl slot): one 256-thread threadgroup scans the
 * [1, n] logits row on the GPU; the host flush then reads a 4-byte token
 * instead of mapping the whole 1 MB logits row (plan Phase A1). Tie-break
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
    } ap      = {(uint32_t) n, (uint32_t) x_off};
    void *cmd = nullptr;
    void *enc = nullptr;
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            return GEIST_E_BACKEND;
        }
        enc = metal_sequence_encoder(st);
    } else {
        cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
        enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
        if (cmd == nullptr || enc == nullptr) {
            return GEIST_E_BACKEND;
        }
    }
    metal_msg_send_set_pipeline(st, enc, st->argmax_pipeline);
    metal_msg_send_set_buffer(st, enc, logits->buffer->buffer, logits->buffer->base_off, 0);
    metal_msg_send_set_buffer(st, enc, st->argmax_result_buffer, 0, 1);
    metal_msg_send_set_bytes(st, enc, &ap, sizeof(ap), 2);
    const struct metal_size groups  = {1, 1, 1};
    const struct metal_size threads = {256, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ARGMAX, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
    if (st->sequence_active) {
        st->sequence_has_work = true;
        /* The 4-byte result read is the token's only host sync point. */
        metal_flush_if_referenced(st, st->argmax_result_buffer);
    } else {
        metal_msg_send_void0(st, enc, "endEncoding");
        metal_msg_send_void0(st, cmd, "commit");
        metal_msg_send_void0(st, cmd, "waitUntilCompleted");
        if (metal_msg_send_id0(st, cmd, "error") != nullptr) {
            return GEIST_E_BACKEND;
        }
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

    void *cmd = nullptr;
    void *enc = nullptr;
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            return GEIST_E_BACKEND;
        }
        enc = metal_sequence_encoder(st);
    } else {
        cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
        enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
        if (cmd == nullptr || enc == nullptr) {
            return GEIST_E_BACKEND;
        }
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
    if (st->sequence_active) {
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
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

    {
        static _Atomic int dbg = -1;
        if (dbg < 0) {
            const char *e = getenv("GEIST_METAL_DEBUG_LINEAR");
            dbg           = (e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0) ? 1 : 0;
        }
        if (dbg && q->buffer != nullptr && q->buffer->mapped != nullptr && k->buffer != nullptr &&
            k->buffer->mapped != nullptr) {
            const float *qp = (const float *) ((const uint8_t *) q->buffer->mapped + q->offset);
            const float *kp = (const float *) ((const uint8_t *) k->buffer->mapped + k->offset);
            size_t       nq = 1, nk = 1;
            for (int i = 0; i < q->ndim; i++)
                nq *= (size_t) q->shape[i];
            for (int i = 0; i < k->ndim; i++)
                nk *= (size_t) k->shape[i];
            float  aq = 0, ak = 0;
            size_t nanq = 0, nank = 0;
            for (size_t i = 0; i < nq; i++) {
                if (isnan(qp[i]))
                    nanq++;
                else if (fabsf(qp[i]) > aq)
                    aq = fabsf(qp[i]);
            }
            for (size_t i = 0; i < nk; i++) {
                if (isnan(kp[i]))
                    nank++;
                else if (fabsf(kp[i]) > ak)
                    ak = fabsf(kp[i]);
            }
            size_t firstnan = (size_t) -1;
            for (size_t i = 0; i < nq && firstnan == (size_t) -1; i++)
                if (isnan(qp[i]))
                    firstnan = i;
            fprintf(stderr,
                    "attn firstnan_q=%zd (row %zd) ",
                    (ssize_t) firstnan,
                    firstnan == (size_t) -1
                            ? (ssize_t) -1
                            : (ssize_t) (firstnan / ((size_t) q->shape[1] * (size_t) q->shape[2])));
            fprintf(stderr,
                    "attn q[%lld,%lld,%lld] |q|=%g nanq=%zu k[%lld,..] |k|=%g nank=%zu qoff=%zu "
                    "sw=%zu kdt=%d\n",
                    (long long) q->shape[0],
                    (long long) q->shape[1],
                    (long long) q->shape[2],
                    (double) aq,
                    nanq,
                    (long long) k->shape[0],
                    (double) ak,
                    nank,
                    q_offset,
                    sliding_window,
                    (int) k->dtype);
        }
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
        st->attention_dec_combine_pipeline != nullptr &&
        st->kv_append_rows_f16_pipeline != nullptr && !metal_env_disabled("GEIST_METAL_FLASH")) {
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
    if (q_rows > 1u && (q_rows % 8u) == 0u &&
        ((head_dim <= 256u && (head_dim % 32u) == 0u) ||
         (head_dim <= 512u && (head_dim % 64u) == 0u &&
          st->attention_flash_sg8_f16_pipeline != nullptr)) &&
        k_rows >= 32u && st->attention_flash_sg_f16_pipeline != nullptr &&
        st->kv_append_rows_f16_pipeline != nullptr && !metal_env_disabled("GEIST_METAL_FLASH")) {
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
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal attention: sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_attention_rows(st, metal_sequence_encoder(st), q, k, value, out, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal attention: command encoder failed");
        return GEIST_E_BACKEND;
    }
    metal_encode_attention_rows(st, enc, q, k, value, out, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal attention: command failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
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
        const uint8_t *base = (const uint8_t *) w->raw;
        for (size_t j = 0; j < n_out; j++) {
            switch ((enum geist_dtype) w->dtype) {
            case GEIST_DTYPE_F32:
                memcpy(row, base + j * n_in * sizeof(float), n_in * sizeof(float));
                break;
            case GEIST_DTYPE_Q4_K:
                dequant_q4_K_row(n_in, base + j * n_in / Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES, row);
                break;
            case GEIST_DTYPE_Q5_K:
                dequant_q5_K_row(n_in, base + j * n_in / Q5_K_BLOCK_ELEMS * Q5_K_BLOCK_BYTES, row);
                break;
            case GEIST_DTYPE_Q4_0:
                dequant_q4_0_row(n_in, base + j * n_in / Q4_0_BLOCK_ELEMS * Q4_0_BLOCK_BYTES, row);
                break;
            case GEIST_DTYPE_Q4_1:
                dequant_q4_1_row(n_in, base + j * n_in / Q4_1_BLOCK_ELEMS * Q4_1_BLOCK_BYTES, row);
                break;
            case GEIST_DTYPE_Q8_0:
                dequant_q8_0_row(n_in, base + j * n_in / Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES, row);
                break;
            case GEIST_DTYPE_Q6_K:
                dequant_q6_K_row(n_in, base + j * n_in / Q6_K_BLOCK_ELEMS * Q6_K_BLOCK_BYTES, row);
                break;
            default:
                memset(row, 0, n_in * sizeof(float));
                break;
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
        metal_linear_debug_stats(m * n_in, m * n_out, m, x, y, w);
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
    metal_linear_debug_stats(m * n_in, m * n_out, m, x, y, w);
}

static void
metal_linear_m1(const float *x, const struct geist_weight *w, struct geist_backend *be, float *y) {
    metal_linear_mN(1, x, w, be, y);
}

/* Tensor-based linear (main's optional vtbl slot): dispatch the GEMM from
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
            return metal_q40_q80_linear(be, &x1, t_w, &y1, (enum geist_dtype) w->dtype, false);
        case GEIST_DTYPE_Q4_K:
            return metal_matvec_q4k(be, &x1, t_w, &y1);
        case GEIST_DTYPE_Q5_K:
            return metal_matvec_q5k(be, &x1, t_w, &y1);
        case GEIST_DTYPE_Q6_K:
            return metal_matvec_q6k(be, &x1, t_w, &y1);
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
        return metal_q40_q80_linear(be, x, t_w, y, (enum geist_dtype) w->dtype, true);
    case GEIST_DTYPE_Q4_K:
        return metal_matmul_q4k(be, x, t_w, y);
    case GEIST_DTYPE_Q5_K:
        return metal_matmul_q5k(be, x, t_w, y);
    case GEIST_DTYPE_Q6_K:
        return metal_matmul_q6k(be, x, t_w, y);
    case GEIST_DTYPE_F32:
        return metal_matmul_f32_dense(be, x, t_w, y);
    default:
        return GEIST_E_UNSUPPORTED;
    }
}

[[nodiscard]] static enum geist_status metal_resolve_weight(struct geist_backend *be,
                                                            struct geist_weight  *w) {
    (void) be;
    if (w == nullptr || w->raw == nullptr || w->n_in <= 0 || w->n_out <= 0 || w->raw_nbytes == 0u) {
        return GEIST_E_INVALID_ARG;
    }
    /* The kernels installed below index `raw` by shape, so a source shorter
     * than the shape reads past its end. Same contract as the CPU
     * resolvers. */
    if (!quant_weight_extent_ok(w)) {
        return GEIST_E_FORMAT;
    }
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
    case GEIST_DTYPE_F32:
        w->linear_m1 = metal_linear_m1;
        w->linear_mN = metal_linear_mN;
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
    void                              *cmd    = nullptr;
    void                              *enc    = nullptr;
    if (st->sequence_active) {
        enc = metal_sequence_encoder(st);
    } else {
        cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
        enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    }
    if (enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal DeltaNet: encoder failed");
        return GEIST_E_BACKEND;
    }
    /* seq>1 runs the chunked-prefill kernel sequence (CPU
     * dn_run_prefill_chunked port); the serial per-token mixer stays for
     * decode and as the fallback when scratch allocation fails.
     *
     * #322: the recipe is encoded in DN_SUBCHUNK-token sub-chunks
     * regardless of the caller's m — the O(C²) chunk cost stays at its
     * measured optimum (64; the arch-level cap A/B'd 64/128/256 at
     * 623/430/335 tok/s) while the surrounding GEMMs run at the full
     * batch for occupancy. State buffers thread through Metal's hazard
     * tracking; scratch is reused per sub-chunk. */
    enum { DN_SUBCHUNK = 64u };
    bool chunked = args->seq > 1 && (args->head_v % 4u) == 0u && st->use_dn_chunk &&
                   st->dn_chunk_gate_pipeline != nullptr;
    if (chunked) {
        const size_t C    = DN_SUBCHUNK;
        const size_t hist = args->conv_kernel - 1u;
        const size_t wsf  = 3u * C + 4u * C * args->head_k + 3u * C * args->head_v + 2u * C * C;
        const size_t scr_bytes = (C * conv_dim + 2u * C * args->n_v_heads + hist * conv_dim +
                                  args->n_v_heads * wsf) *
                                 sizeof(float);
        if (scr_bytes > (size_t) 1u << 31) {
            chunked = false;
        } else if (st->dn_scratch_bytes < scr_bytes) {
            metal_msg_send_void0(st, st->dn_scratch, "release");
            st->dn_scratch       = metal_msg_send_id_size_uint(st,
                                                               st->device,
                                                               "newBufferWithLength:options:",
                                                               scr_bytes,
                                                               METAL_RESOURCE_STORAGE_MODE_PRIVATE);
            st->dn_scratch_bytes = st->dn_scratch != nullptr ? scr_bytes : 0;
            chunked              = st->dn_scratch != nullptr;
        }
    }
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) {
        metal_msg_send_set_buffer(st, enc, all[i]->buffer->buffer, all[i]->buffer->base_off, i);
    }
    if (chunked) {
        const uint32_t          cd     = (uint32_t) conv_dim;
        const uint32_t          vd     = (uint32_t) value_dim;
        const uint32_t          hist   = params.conv_kernel - 1u;
        const struct metal_size t256   = {256, 1, 1};
        const struct metal_size g_flat = {(hist * cd + 255u) / 256u, 1, 1};
        metal_msg_send_set_buffer(st, enc, st->dn_scratch, 0, 10);
        for (uint32_t off = 0; off < params.seq; off += DN_SUBCHUNK) {
            struct metal_deltanet_params sub = params;
            sub.seq = (params.seq - off < DN_SUBCHUNK) ? params.seq - off : DN_SUBCHUNK;
            sub.qkv_offset += off * cd;
            sub.z_offset += off * vd;
            sub.beta_offset += off * params.n_v_heads;
            sub.alpha_offset += off * params.n_v_heads;
            const uint32_t          C       = sub.seq;
            const struct metal_size g_tok   = {C, 1, 1};
            const struct metal_size g_norm  = {C, params.n_k_heads, 1};
            const struct metal_size g_heads = {params.n_v_heads, 1, 1};
            metal_msg_send_set_bytes(st, enc, &sub, sizeof sub, 11);
            metal_msg_send_set_pipeline(st, enc, st->dn_cst_copy_pipeline);
            metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DN_PREP, g_flat);
            metal_msg_send_dispatch(st, enc, g_flat, t256);
            metal_msg_send_set_pipeline(st, enc, st->dn_conv_prep_pipeline);
            metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DN_PREP, g_tok);
            metal_msg_send_dispatch(st, enc, g_tok, t256);
            metal_msg_send_set_pipeline(st, enc, st->dn_qk_norm_pipeline);
            metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DN_NORM, g_norm);
            metal_msg_send_dispatch(st, enc, g_norm, t256);
            metal_msg_send_set_pipeline(st, enc, st->dn_state_roll_pipeline);
            metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DN_PREP, g_flat);
            metal_msg_send_dispatch(st, enc, g_flat, t256);
            const struct metal_size g_cc = {
                    (((C + 3u) / 4u) * C + 255u) / 256u, params.n_v_heads, 1};
            const struct metal_size g_cdv = {
                    (C * (params.head_v / 4u) + 255u) / 256u, params.n_v_heads, 1};
            const struct metal_size g_kv = {
                    (params.head_k * (params.head_v / 4u) + 255u) / 256u, params.n_v_heads, 1};
            const struct metal_size g_tv = {C, params.n_v_heads, 1};
            metal_msg_send_set_pipeline(st, enc, st->dn_chunk_stage_pipeline);
            metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DN_STAGE, g_heads);
            metal_msg_send_dispatch(st, enc, g_heads, t256);
            metal_msg_send_set_pipeline(st, enc, st->dn_chunk_amat_pipeline);
            metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DN_WIDE, g_cc);
            metal_msg_send_dispatch(st, enc, g_cc, t256);
            metal_msg_send_set_pipeline(st, enc, st->dn_chunk_subst_pipeline);
            metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DN_SUBST, g_heads);
            metal_msg_send_dispatch(st, enc, g_heads, t256);
            metal_msg_send_set_pipeline(st, enc, st->dn_chunk_vnew1_pipeline);
            metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DN_WIDE, g_cdv);
            metal_msg_send_dispatch(st, enc, g_cdv, t256);
            metal_msg_send_set_pipeline(st, enc, st->dn_chunk_vnew2_pipeline);
            metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DN_WIDE, g_cdv);
            metal_msg_send_dispatch(st, enc, g_cdv, t256);
            metal_msg_send_set_pipeline(st, enc, st->dn_chunk_out_pipeline);
            metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DN_WIDE, g_cdv);
            metal_msg_send_dispatch(st, enc, g_cdv, t256);
            metal_msg_send_set_pipeline(st, enc, st->dn_chunk_supd_pipeline);
            metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DN_WIDE, g_kv);
            metal_msg_send_dispatch(st, enc, g_kv, t256);
            metal_msg_send_set_pipeline(st, enc, st->dn_chunk_gate_pipeline);
            metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_DN_WIDE, g_tv);
            metal_msg_send_dispatch(st, enc, g_tv, t256);
        }
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
    if (st->sequence_active) {
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    if (metal_msg_send_id0(st, cmd, "error") != nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal DeltaNet: command failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

/* main-contract vtbl. The old fine-grained GPU ops (command_sequence_*,
 * ffn_geglu_block, ple_block, attention_block, greedy_head, matmul_q4k)
 * are not part of main's contract; their impls remain in this file as
 * internal/dead code (behind metal_legacy_ops.h) pending the Stage-6
 * cleanup, but are not exposed here. */
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

/* Probe pairing for the fused table below. gate_up mirrors
 * metal_ffn_gate_up's entry checks (decode-only matvec kernel); the
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
        if (q->m != 1 || q->gate_w == nullptr || q->up_w == nullptr ||
            q->gate_w->dtype != GEIST_DTYPE_Q4_K || q->up_w->dtype != GEIST_DTYPE_Q4_K ||
            q->d_model % 256u != 0u || (size_t) q->gate_w->n_in != q->d_model ||
            (size_t) q->up_w->n_in != q->d_model || q->gate_w->n_out != q->up_w->n_out) {
            return false;
        }
        if (metal_ensure_q4k_pipeline(be) != GEIST_OK) {
            return false;
        }
        return ((struct metal_state *) be->state)->q4k_gate_up_n4_pipeline != nullptr;
    case GEIST_FUSED_RMSNORM_ADD:
    case GEIST_FUSED_ARGMAX_F32:
        /* All-F32 tensor ops; the kernels accept any well-formed row
         * block the arch passes. Pipeline creation failure at run time
         * is a real device error, not capability negotiation. */
        return true;
    case GEIST_FUSED_ATTN_QKV_PREP:
        /* Half-split RoPE per-head norm kernel: any row count, head_dim
         * must be even (mirrors metal_attn_qkv_prep's hd % 2 check). */
        return q->head_dim > 0 && (q->head_dim % 2u) == 0u;
    case GEIST_FUSED_PLE_BLOCK:
        /* F32 gate/proj matrices AND m==1 (mirrors metal_ple_block's
         * checks — its kernels are naive decode GEMVs, rows==1 only).
         * The missing m check made the plan bind fuse_ple_block_mN and
         * hard-fail every gemma4 Metal prefill at layer 0. */
        return q->m == 1 && q->gate_w != nullptr && q->up_w != nullptr &&
               q->gate_w->dtype == GEIST_DTYPE_F32 && q->up_w->dtype == GEIST_DTYPE_F32;
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
            return true;
        default:
            return false;
        }
    default:
        return false;
    }
}

static const struct geist_backend_primitives metal_prims = {
        .rmsnorm          = metal_rmsnorm,
        .add              = metal_add,
        .mul              = metal_mul,
        .gelu_tanh        = metal_gelu_tanh,
        .silu             = metal_silu,
        .relu_squared     = nullptr,
        .rope_apply       = metal_rope_apply,
        .embedding_lookup = metal_embedding_lookup,
        .attention        = metal_attention,
        .scale_f32        = metal_scale_f32,
};

static const struct geist_backend_fused metal_fused = {
        .supported                    = metal_fused_supported,
        .gelu_tanh_mul                = metal_gelu_tanh_mul,
        .silu_mul                     = metal_silu_mul,
        .linear_t                     = metal_linear_t,
        .linear_t_pair                = metal_linear_t_pair,
        .embedding_lookup_scaled      = metal_embedding_lookup_scaled,
        .embedding_lookup_scaled_rows = metal_embedding_lookup_scaled_rows,
        .kv_append_f16                = metal_kv_append_f16,
        .argmax_f32                   = metal_argmax_f32,
        .ffn_gate_up                  = metal_ffn_gate_up,
        .attn_qkv_prep                = metal_attn_qkv_prep,
        .ple_block                    = metal_ple_block,
        .rmsnorm_add                  = metal_rmsnorm_add,
        .deltanet_mix                 = metal_deltanet_mix,
        .attn_qgate_split             = metal_attn_qgate_split,
        .sigmoid_mul                  = metal_sigmoid_mul,
};

const struct geist_backend_descriptor geist_backend_metal = {
        .name  = "metal",
        .vtbl  = &metal_vtbl,
        .prims = &metal_prims,
        .fused = &metal_fused,
        .caps  = {.kv_f16_attention = true,
                  .batched_submit   = true,
                  /* deltanet_mix encodes 64-token sub-chunks internally
                   * (#322), so DN models keep preferred_m_max. */
                 .dn_subchunk = true,
                 /* 256 since the simdgroup GEMM work. The original
                  * 2026-08-27 A/B was void — pre-#312, m_max requests
                  * below the default were silently ignored, so both
                  * arms ran identical configs. The post-#312 re-check
                  * with real chunking keeps 256 (gemma4-e2b 972 tok/s
                  * pp512). 512 doubles the m_max-scaled logits scratch
                  * (m_max x VOCAB floats — 256 MB at vocab 262k); DN
                  * models are capped at 64 in arch_state anyway
                  * (O(C^2) chunk cost). */
                 .preferred_m_max   = 256,
                 .max_m             = 512, /* batched-submit pipeline bound */
                 .preferred_kv_mode = GEIST_KV_FP32},
};
