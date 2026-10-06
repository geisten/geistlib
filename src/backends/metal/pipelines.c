/*
 * src/backends/metal/pipelines.c — compute-pipeline compilation and caching.
 *
 * Layer: BACKEND (metal). Split from the former monolithic backend.c;
 * pure moves, no behavior change.
 */
#include "metal_internal.h"
#include "iq_grids.h"

[[nodiscard]] static enum geist_status metal_create_named_pipeline(struct geist_backend *be,
                                                                   void                 *library,
                                                                   void                 *ns_string,
                                                                   const char           *name,
                                                                   void **out_function,
                                                                   void **out_pipeline) {

    if (be == nullptr || be->state == nullptr || library == nullptr || ns_string == nullptr ||
        name == nullptr || out_function == nullptr || out_pipeline == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    void *fn_name          = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", name);
    if (fn_name == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "metal: failed to create %s shader name", name);
        return GEIST_E_BACKEND;
    }
    *out_function = metal_msg_send_id_id(st, library, "newFunctionWithName:", fn_name);
    if (*out_function == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: %s shader function missing", name);
        return GEIST_E_BACKEND;
    }

    void *err     = nullptr;
    *out_pipeline = metal_msg_send_id_id_err(
            st, st->device, "newComputePipelineStateWithFunction:error:", *out_function, &err);
    if (*out_pipeline == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: %s pipeline creation failed%s%s",
                                name,
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

/* A pipeline built from a library that is already in metal_state. The
 * fields are offsets into the state, so the same rows drive creation
 * here and the release in metal_release_pipeline_tables. */
struct metal_pipeline_row {
    size_t      library;
    const char *name;
    size_t      function;
    size_t      pipeline;
};
#define METAL_PIPE(LIB, NAME, FIELD)                 \
    {offsetof(struct metal_state, LIB##_library),    \
     NAME,                                           \
     offsetof(struct metal_state, FIELD##_function), \
     offsetof(struct metal_state, FIELD##_pipeline)}
#define METAL_N(rows) (sizeof(rows) / sizeof((rows)[0]))

static const struct metal_pipeline_row metal_deltanet_rows[] = {
        METAL_PIPE(deltanet, "deltanet_mix", deltanet_mix),
        METAL_PIPE(deltanet, "dn_dec_qk", dn_dec_qk),
        METAL_PIPE(deltanet, "dn_dec_v", dn_dec_v),
        METAL_PIPE(deltanet, "dn_cst_copy", dn_cst_copy),
        METAL_PIPE(deltanet, "dn_conv_prep", dn_conv_prep),
        METAL_PIPE(deltanet, "dn_qk_norm", dn_qk_norm),
        METAL_PIPE(deltanet, "dn_state_roll", dn_state_roll),
        METAL_PIPE(deltanet, "dn_chunk_stage", dn_chunk_stage),
        METAL_PIPE(deltanet, "dn_chunk_subst", dn_chunk_subst),
        METAL_PIPE(deltanet, "dn_chunk_amat", dn_chunk_amat),
        METAL_PIPE(deltanet, "dn_chunk_vnew1", dn_chunk_vnew1),
        METAL_PIPE(deltanet, "dn_chunk_vnew2", dn_chunk_vnew2),
        METAL_PIPE(deltanet, "dn_chunk_out", dn_chunk_out),
        METAL_PIPE(deltanet, "dn_chunk_supd", dn_chunk_supd),
        METAL_PIPE(deltanet, "dn_chunk_gate", dn_chunk_gate),
};

static const struct metal_pipeline_row metal_quant_rows[] = {
        METAL_PIPE(q4k, "matvec_q4k", q4k),
        METAL_PIPE(q40_q80, "linear_q40", q40),
        METAL_PIPE(q40_q80, "linear_q40_m8", q40_m8),
        METAL_PIPE(q40_q80, "linear_q80", q80),
        METAL_PIPE(q40_q80, "linear_q80_m8", q80_m8),
        METAL_PIPE(q5k, "linear_q5k", q5k),
        METAL_PIPE(q5k, "linear_q5k_m8", q5k_m8),
        METAL_PIPE(q41, "linear_q41", q41),
        METAL_PIPE(q41, "linear_q41_m8", q41_m8),
        METAL_PIPE(quant_sg, "matvec_q40_n4", q40_n4),
        METAL_PIPE(quant_sg, "matmul_q40_mm_sg", q40_mm),
        METAL_PIPE(quant_sg, "matvec_q80_n4", q80_n4),
        METAL_PIPE(quant_sg, "matmul_q80_mm_sg", q80_mm),
        METAL_PIPE(quant_sg, "matvec_q41_n4", q41_n4),
        METAL_PIPE(quant_sg, "matmul_q41_mm_sg", q41_mm),
        METAL_PIPE(quant_sg, "matvec_q5k_n4", q5k_n4),
        METAL_PIPE(quant_sg, "matmul_q5k_mm_sg", q5k_mm),
        METAL_PIPE(quant_sg, "matvec_pq2_n4", pq2_n4),
        METAL_PIPE(quant_sg, "matvec_pq2_n8", pq2_n8),
        METAL_PIPE(quant_sg, "matmul_pq2_mm_sg", pq2_mm),
        METAL_PIPE(quant_sg, "matmul_pq2_mm_sg_fast", pq2_mm_fast),
        METAL_PIPE(quant_sg, "matvec_iq4nl_n4", iq4nl_n4),
        METAL_PIPE(quant_sg, "matmul_iq4nl_mm_sg", iq4nl_mm),
        METAL_PIPE(quant_sg, "matvec_iq4xs_n4", iq4xs_n4),
        METAL_PIPE(quant_sg, "matmul_iq4xs_mm_sg", iq4xs_mm),
        METAL_PIPE(quant_sg, "matvec_q3k_n4", q3k_n4),
        METAL_PIPE(quant_sg, "matmul_q3k_mm_sg", q3k_mm),
        METAL_PIPE(quant_sg, "matvec_tq2_n4", tq2_n4),
        METAL_PIPE(quant_sg, "matmul_tq2_mm_sg", tq2_mm),
        METAL_PIPE(quant_sg, "matvec_i2s_n4", i2s_n4),
        METAL_PIPE(quant_sg, "matmul_i2s_mm_sg", i2s_mm),
        METAL_PIPE(quant_sg, "matvec_iq3s_n4", iq3s_n4),
        METAL_PIPE(quant_sg, "matmul_iq3s_mm_sg", iq3s_mm),
        METAL_PIPE(quant_sg, "matmul_iq4xs_mm_sg_fast", iq4xs_mm_fast),
        METAL_PIPE(quant_sg, "matmul_q40_mm_sg_fast", q40_mm_fast),
        METAL_PIPE(quant_sg, "matmul_q80_mm_sg_fast", q80_mm_fast),
        METAL_PIPE(quant_sg, "matmul_q41_mm_sg_fast", q41_mm_fast),
        METAL_PIPE(quant_sg, "matmul_q5k_mm_sg_fast", q5k_mm_fast),
        METAL_PIPE(q4k_n4, "matvec_q4k_n4", q4k_n4),
        METAL_PIPE(q4k_gate_up_n4, "gate_up_q4k_n4", q4k_gate_up_n4),
        METAL_PIPE(q4k_pair_n4, "pair_q4k_n4", q4k_pair_n4),
        METAL_PIPE(q4k, "matmul_q4k_m8", q4k_matmul_m8),
        METAL_PIPE(q4k_m16, "matmul_q4k_m16", q4k_matmul_m16),
        METAL_PIPE(q4k_m16_n2, "matmul_q4k_m16_n2", q4k_matmul_m16_n2),
        METAL_PIPE(q6k, "matvec_q6k", q6k),
        METAL_PIPE(q6k_mm_sg, "matmul_q6k_sg", q6k_matmul_sg),
        METAL_PIPE(q6k_mm_sg_fast, "matmul_q6k_sg_fast", q6k_matmul_sg_fast),
        METAL_PIPE(q6k_n4, "matvec_q6k_n4", q6k_n4),
        METAL_PIPE(q6k, "matmul_q6k_m8", q6k_matmul_m8),
        METAL_PIPE(q6k_m16, "matmul_q6k_m16", q6k_matmul_m16),
        METAL_PIPE(elem, "rmsnorm_rows", rmsnorm_rows),
        METAL_PIPE(elem_simd, "rmsnorm_rows_simd", rmsnorm_rows_simd),
        METAL_PIPE(elem, "gelu_rows", gelu_rows),
        METAL_PIPE(silu, "silu_rows", silu_rows),
        METAL_PIPE(silu, "relu2_rows", relu2_rows),
        METAL_PIPE(silu, "silu_mul_rows", silu_mul_rows),
        METAL_PIPE(elem, "mul_rows", mul_rows),
        METAL_PIPE(elem, "gelu_mul_rows", gelu_mul_rows),
        METAL_PIPE(elem, "add_rows", add_rows),
        METAL_PIPE(elem, "scale_rows", scale_rows),
        METAL_PIPE(qgate, "qgate_split", qgate_split),
        METAL_PIPE(qgate, "sigmoid_mul_rows", sigmoid_mul),
        METAL_PIPE(elem, "rmsnorm_add_rows", rmsnorm_add_rows),
        METAL_PIPE(elem_simd, "rmsnorm_add_rows_simd", rmsnorm_add_rows_simd),
        METAL_PIPE(embed, "embed_lookup_scaled", embed_lookup_scaled),
        METAL_PIPE(embed, "embed_lookup_scaled_rows", embed_lookup_scaled_rows),
        METAL_PIPE(f32, "matmul_f32", f32_matmul),
        METAL_PIPE(f32, "matmul_f32_sg", f32_matmul_sg),
        METAL_PIPE(f32, "matmul_f32_mm_sg", f32_matmul_mm),
        METAL_PIPE(f32, "matmul_f16w", f16w_matmul),
        METAL_PIPE(f32, "matmul_f16w_sg", f16w_matmul_sg),
        METAL_PIPE(f32, "matmul_bf16w", bf16w_matmul),
        METAL_PIPE(f32, "matmul_bf16w_sg", bf16w_matmul_sg),
        METAL_PIPE(f32, "ple_gate_f32", f32_ple_gate),
        METAL_PIPE(f32, "ple_proj_norm_f32", f32_ple_proj_norm),
};

/* Built only with GEIST_METAL_Q4K_MM_SG on (the default). */
static const struct metal_pipeline_row metal_q4k_mm_sg_rows[] = {
        METAL_PIPE(q4k_mm_sg, "matmul_q4k_mm_sg", q4k_mm_sg),
        METAL_PIPE(q4k_mm_sg_fast, "matmul_q4k_mm_sg_fast", q4k_mm_sg_fast),
};

static void **metal_state_slot(struct metal_state *st, size_t offset) {
    return (void **) ((char *) st + offset);
}

[[nodiscard]] static enum geist_status
metal_create_pipelines(struct geist_backend           *be,
                       void                           *ns_string,
                       size_t                          n,
                       const struct metal_pipeline_row rows[static n]) {
    struct metal_state *st = be->state;
    for (size_t i = 0; i < n; i++) {
        const enum geist_status s =
                metal_create_named_pipeline(be,
                                            *metal_state_slot(st, rows[i].library),
                                            ns_string,
                                            rows[i].name,
                                            metal_state_slot(st, rows[i].function),
                                            metal_state_slot(st, rows[i].pipeline));
        if (s != GEIST_OK) {
            return s;
        }
    }
    return GEIST_OK;
}

static void metal_release_rows(struct metal_state             *st,
                               size_t                          n,
                               const struct metal_pipeline_row rows[static n]) {
    for (size_t i = 0; i < n; i++) {
        metal_msg_send_void0(st, *metal_state_slot(st, rows[i].pipeline), "release");
        metal_msg_send_void0(st, *metal_state_slot(st, rows[i].function), "release");
    }
}

void metal_release_pipeline_tables(struct metal_state *st) {
    metal_release_rows(st, METAL_N(metal_deltanet_rows), metal_deltanet_rows);
    metal_release_rows(st, METAL_N(metal_quant_rows), metal_quant_rows);
    metal_release_rows(st, METAL_N(metal_q4k_mm_sg_rows), metal_q4k_mm_sg_rows);
}

[[nodiscard]] enum geist_status metal_ensure_deltanet_pipeline(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    if (st->deltanet_mix_pipeline != nullptr && st->dn_chunk_gate_pipeline != nullptr) {
        return GEIST_OK;
    }
    void *ns_string = metal_objc_get_class(st, "NSString");
    /* Serial mixer + chunked-prefill kernels compile as one MSL unit
     * (the chunk sources reuse struct P and silu1). */
    void *source = nullptr;
    if (ns_string != nullptr) {
        const char *const parts[] = {metal_deltanet_source,
                                     metal_dn_decode_source,
                                     metal_dn_chunk_prep_source,
                                     metal_dn_chunk_ws_source,
                                     metal_dn_chunk_wide_source,
                                     metal_dn_chunk_wide2_source};
        size_t            total   = 0;
        for (size_t i = 0; i < sizeof parts / sizeof parts[0]; i++) {
            total += strlen(parts[i]);
        }
        char *buf = malloc(total + 1u);
        if (buf != nullptr) {
            size_t off = 0;
            for (size_t i = 0; i < sizeof parts / sizeof parts[0]; i++) {
                const size_t len = strlen(parts[i]);
                memcpy(buf + off, parts[i], len);
                off += len;
            }
            buf[off] = '\0';
            source   = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", buf);
            free(buf);
        }
    }
    if (source == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: DeltaNet shader source failed");
        return GEIST_E_BACKEND;
    }
    void *err            = nullptr;
    st->deltanet_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
    if (st->deltanet_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: DeltaNet shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    return metal_create_pipelines(be, ns_string, METAL_N(metal_deltanet_rows), metal_deltanet_rows);
}

[[nodiscard]] enum geist_status metal_ensure_q4k_pipeline(struct geist_backend *be) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    if (st->q4k_pipeline != nullptr && st->q40_pipeline != nullptr &&
        st->q40_m8_pipeline != nullptr && st->q80_pipeline != nullptr &&
        st->q80_m8_pipeline != nullptr && st->q5k_pipeline != nullptr &&
        st->q5k_m8_pipeline != nullptr && st->q41_pipeline != nullptr &&
        st->q41_m8_pipeline != nullptr && st->q40_n4_pipeline != nullptr &&
        st->q40_mm_pipeline != nullptr && st->q80_n4_pipeline != nullptr &&
        st->q80_mm_pipeline != nullptr && st->q41_n4_pipeline != nullptr &&
        st->q41_mm_pipeline != nullptr && st->q5k_n4_pipeline != nullptr &&
        st->q5k_mm_pipeline != nullptr && st->iq4nl_mm_pipeline != nullptr &&
        st->pq2_n4_pipeline != nullptr && st->pq2_mm_pipeline != nullptr &&
        st->pq2_mm_fast_pipeline != nullptr && st->pq2_n8_pipeline != nullptr &&
        st->iq4xs_mm_pipeline != nullptr && st->q3k_mm_pipeline != nullptr &&
        st->tq2_n4_pipeline != nullptr && st->tq2_mm_pipeline != nullptr &&
        st->i2s_n4_pipeline != nullptr && st->i2s_mm_pipeline != nullptr &&
        st->iq3s_mm_pipeline != nullptr && st->q4k_n4_pipeline != nullptr &&
        st->q4k_matmul_m8_pipeline != nullptr && st->q4k_matmul_m16_pipeline != nullptr &&
        st->q4k_matmul_m16_n2_pipeline != nullptr &&
        (!st->use_q4k_mm_sg || st->q4k_mm_sg_pipeline != nullptr) && st->q6k_pipeline != nullptr &&
        st->q6k_n4_pipeline != nullptr && st->q6k_matmul_m8_pipeline != nullptr &&
        st->q6k_matmul_m16_pipeline != nullptr && st->rmsnorm_rows_pipeline != nullptr &&
        st->rmsnorm_rows_simd_pipeline != nullptr && st->gelu_rows_pipeline != nullptr &&
        st->silu_rows_pipeline != nullptr && st->relu2_rows_pipeline != nullptr &&
        st->mul_rows_pipeline != nullptr && st->gelu_mul_rows_pipeline != nullptr &&
        st->add_rows_pipeline != nullptr && st->scale_rows_pipeline != nullptr &&
        st->rmsnorm_add_rows_pipeline != nullptr && st->rmsnorm_add_rows_simd_pipeline != nullptr &&
        st->qgate_split_pipeline != nullptr && st->sigmoid_mul_pipeline != nullptr &&
        st->embed_lookup_scaled_pipeline != nullptr && st->f32_matmul_pipeline != nullptr &&
        st->f16w_matmul_pipeline != nullptr && st->f16w_matmul_sg_pipeline != nullptr &&
        st->bf16w_matmul_pipeline != nullptr && st->bf16w_matmul_sg_pipeline != nullptr &&
        st->f32_ple_gate_pipeline != nullptr && st->f32_ple_proj_norm_pipeline != nullptr) {
        return GEIST_OK;
    }

    void *ns_string = metal_objc_get_class(st, "NSString");
    if (ns_string == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: NSString class unavailable");
        return GEIST_E_BACKEND;
    }
    void *source = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q4k_source);
    void *q40_q80_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q40_q80_source);
    void *q5k_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q5k_source);
    void *q41_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q41_source);
    /* The simdgroup quant kernels ship as several sub-4095-char C literals
     * and compile as one MSL unit; concatenate them here. The IQ3_S grid
     * (quant/iq_grids.h) is emitted as an MSL constant array at build
     * time — 512 hex literals nobody wants to hand-maintain twice. */
    void *quant_sg_source = nullptr;
    {
        enum { GRID_STR_CAP = 512 * 12 + 64 };
        char *grid_src = malloc(GRID_STR_CAP);
        if (grid_src != nullptr) {
            size_t off = (size_t) snprintf(grid_src, GRID_STR_CAP, "constant uint iq3sg[512]={");
            for (size_t gi = 0; gi < 512; gi++) {
                off += (size_t) snprintf(grid_src + off,
                                         GRID_STR_CAP - off,
                                         "%s0x%08xu",
                                         gi == 0 ? "" : ",",
                                         iq3s_grid[gi]);
            }
            off += (size_t) snprintf(grid_src + off, GRID_STR_CAP - off, "};\n");
        }
        const char *const parts[] = {
                metal_qsg_common_source,      grid_src != nullptr ? grid_src : "",
                metal_qsg_common2_source,     metal_qsg_n4_q40_source,
                metal_qsg_n4_q80_source,      metal_qsg_n4_q41_source,
                metal_qsg_n4_q5k_source,      metal_qsg_mm_q40_source,
                metal_qsg_mm_q80_source,      metal_qsg_mm_q41_source,
                metal_qsg_mm_q5k_source,      metal_qsg_n4_iq4nl_source,
                metal_qsg_n4_iq4xs_source,    metal_qsg_mm_iq4nl_source,
                metal_qsg_mm_iq4xs_source,    metal_qsg_n4_q3k_source,
                metal_qsg_n4_iq3s_source,     metal_qsg_mm_q3k_source,
                metal_qsg_mm_iq3s_source,     metal_qsg_mm_q40_fast_source,
                metal_qsg_mm_q80_fast_source, metal_qsg_mm_q41_fast_source,
                metal_qsg_mm_q5k_fast_source, metal_qsg_mm_iq4xs_fast_source,
                metal_qsg_pq2_dq_source,      metal_qsg_pq2_source,
                metal_qsg_pq2_n8_source,      metal_qsg_mm_pq2_source,
                metal_qsg_mm_pq2_fast_source, metal_qsg_tq2_source,
                metal_qsg_tq2_n4_source,      metal_qsg_mm_tq2_source,
                metal_qsg_i2s_source,         metal_qsg_i2s_n4_source,
                metal_qsg_mm_i2s_source};
        const size_t n_parts = sizeof parts / sizeof parts[0];
        size_t       total   = 0;
        for (size_t i = 0; i < n_parts; i++) {
            total += strlen(parts[i]);
        }
        char *buf = malloc(total + 1u);
        if (buf != nullptr) {
            size_t off = 0;
            for (size_t i = 0; i < n_parts; i++) {
                const size_t len = strlen(parts[i]);
                memcpy(buf + off, parts[i], len);
                off += len;
            }
            buf[off]        = '\0';
            quant_sg_source = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", buf);
            free(buf);
        }
        free(grid_src);
    }
    void *q4k_n4_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q4k_n4_source);
    void *q4k_m16_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q4k_m16_source);
    void *q4k_m16_n2_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q4k_m16_n2_source);
    void *q4k_mm_sg_ns_source =
            st->use_q4k_mm_sg
                    ? metal_msg_send_id_cstr(
                              st, ns_string, "stringWithUTF8String:", metal_q4k_mm_sg_source)
                    : nullptr;
    void *q4k_mm_sg_fast_ns_source =
            st->use_q4k_mm_sg
                    ? metal_msg_send_id_cstr(
                              st, ns_string, "stringWithUTF8String:", metal_q4k_mm_sg_fast_source)
                    : nullptr;
    void *q4k_gate_up_n4_src = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:", metal_q4k_gate_up_n4_source);
    void *q4k_pair_n4_src = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:", metal_q4k_pair_n4_source);
    void *q6_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q6k_source);
    void *q6_mm_sg_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q6k_mm_sg_source);
    void *q6_mm_sg_fast_source = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:", metal_q6k_mm_sg_fast_source);
    void *q6_n4_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q6k_n4_source);
    void *q6_m16_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q6k_m16_source);
    void *elem_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_elem_source);
    void *silu_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_silu_source);
    void *qgate_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_qgate_source);
    void *elem_simd_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_elem_simd_source);
    void *embed_source = nullptr;
    {
        /* single + batched embed kernels share the dequant helpers —
         * concatenated at init (4095-char literal limit). */
        const size_t em_a   = strlen(metal_embed_source);
        const size_t em_b   = strlen(metal_embed_rows_source);
        char        *em_src = malloc(em_a + em_b + 1u);
        if (em_src == nullptr) {
            geist_backend_set_error(be, GEIST_E_OOM, "metal: embed shader source alloc failed");
            return GEIST_E_OOM;
        }
        memcpy(em_src, metal_embed_source, em_a);
        memcpy(em_src + em_a, metal_embed_rows_source, em_b + 1u);
        embed_source = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", em_src);
        free(em_src);
    }
    void *f32_source = nullptr;
    {
        /* literals concatenated at init (C99 4095-char literal limit) */
        const char *const f32_parts[] = {
                metal_f32_source, metal_f32_mm_source, metal_f16_w_source, metal_bf16_w_source};
        size_t f32_len = 1u;
        for (size_t i = 0; i < sizeof f32_parts / sizeof f32_parts[0]; i++) {
            f32_len += strlen(f32_parts[i]);
        }
        char *f32_src = malloc(f32_len);
        if (f32_src == nullptr) {
            geist_backend_set_error(be, GEIST_E_OOM, "metal: f32 shader source alloc failed");
            return GEIST_E_OOM;
        }
        f32_src[0] = '\0';
        for (size_t i = 0; i < sizeof f32_parts / sizeof f32_parts[0]; i++) {
            strcat(f32_src, f32_parts[i]);
        }
        f32_source = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", f32_src);
        free(f32_src);
    }
    if (source == nullptr || q40_q80_source == nullptr || quant_sg_source == nullptr ||
        q4k_m16_source == nullptr || q4k_m16_n2_source == nullptr || q4k_n4_source == nullptr ||
        (st->use_q4k_mm_sg && q4k_mm_sg_ns_source == nullptr) || q4k_gate_up_n4_src == nullptr ||
        q4k_pair_n4_src == nullptr || q6_source == nullptr || q6_n4_source == nullptr ||
        q6_m16_source == nullptr || elem_source == nullptr || silu_source == nullptr ||
        qgate_source == nullptr || elem_simd_source == nullptr || embed_source == nullptr ||
        f32_source == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: failed to create shader source");
        return GEIST_E_BACKEND;
    }

    void *err       = nullptr;
    st->q4k_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
    if (st->q4k_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q4_K shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                 = nullptr;
    st->q40_q80_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", q40_q80_source, nullptr, &err);
    if (st->q40_q80_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q4_0/Q8_0 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err             = nullptr;
    st->q5k_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", q5k_source, nullptr, &err);
    if (st->q5k_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q5_K shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err             = nullptr;
    st->q41_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", q41_source, nullptr, &err);
    if (st->q41_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q4_1 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                  = nullptr;
    st->quant_sg_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", quant_sg_source, nullptr, &err);
    if (st->quant_sg_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: simdgroup quant shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                = nullptr;
    st->q4k_n4_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", q4k_n4_source, nullptr, &err);
    if (st->q4k_n4_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q4_K n4 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                        = nullptr;
    st->q4k_gate_up_n4_library = metal_msg_send_id_id_id_err(st,
                                                             st->device,
                                                             "newLibraryWithSource:options:error:",
                                                             q4k_gate_up_n4_src,
                                                             nullptr,
                                                             &err);
    if (st->q4k_gate_up_n4_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q4_K gate/up n4 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                     = nullptr;
    st->q4k_pair_n4_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", q4k_pair_n4_src, nullptr, &err);
    if (st->q4k_pair_n4_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q4_K pair n4 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                 = nullptr;
    st->q4k_m16_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", q4k_m16_source, nullptr, &err);
    if (st->q4k_m16_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q4_K m16 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                    = nullptr;
    st->q4k_m16_n2_library = metal_msg_send_id_id_id_err(st,
                                                         st->device,
                                                         "newLibraryWithSource:options:error:",
                                                         q4k_m16_n2_source,
                                                         nullptr,
                                                         &err);
    if (st->q4k_m16_n2_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q4_K m16 n2 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    if (st->use_q4k_mm_sg) {
        err                   = nullptr;
        st->q4k_mm_sg_library = metal_msg_send_id_id_id_err(st,
                                                            st->device,
                                                            "newLibraryWithSource:options:error:",
                                                            q4k_mm_sg_ns_source,
                                                            nullptr,
                                                            &err);
        if (st->q4k_mm_sg_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: Q4_K simdgroup mm shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        err = nullptr;
        st->q4k_mm_sg_fast_library =
                metal_msg_send_id_id_id_err(st,
                                            st->device,
                                            "newLibraryWithSource:options:error:",
                                            q4k_mm_sg_fast_ns_source,
                                            nullptr,
                                            &err);
        if (st->q4k_mm_sg_fast_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: Q4_K simdgroup mm fast shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
    }
    err             = nullptr;
    st->q6k_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", q6_source, nullptr, &err);
    if (st->q6k_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q6_K shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                   = nullptr;
    st->q6k_mm_sg_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", q6_mm_sg_source, nullptr, &err);
    if (st->q6k_mm_sg_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q6_K mm_sg shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                        = nullptr;
    st->q6k_mm_sg_fast_library = metal_msg_send_id_id_id_err(st,
                                                             st->device,
                                                             "newLibraryWithSource:options:error:",
                                                             q6_mm_sg_fast_source,
                                                             nullptr,
                                                             &err);
    if (st->q6k_mm_sg_fast_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q6_K mm_sg fast shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                 = nullptr;
    st->q6k_m16_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", q6_m16_source, nullptr, &err);
    if (st->q6k_m16_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q6_K m16 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                = nullptr;
    st->q6k_n4_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", q6_n4_source, nullptr, &err);
    if (st->q6k_n4_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q6_K n4 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err              = nullptr;
    st->elem_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", elem_source, nullptr, &err);
    if (st->elem_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: elementwise shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err              = nullptr;
    st->silu_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", silu_source, nullptr, &err);
    if (st->silu_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: SiLU shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err               = nullptr;
    st->qgate_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", qgate_source, nullptr, &err);
    if (st->qgate_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Qwen gate shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                   = nullptr;
    st->elem_simd_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", elem_simd_source, nullptr, &err);
    if (st->elem_simd_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: SIMD elementwise shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err               = nullptr;
    st->embed_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", embed_source, nullptr, &err);
    if (st->embed_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: embedding shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err             = nullptr;
    st->f32_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", f32_source, nullptr, &err);
    if (st->f32_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: F32 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    enum geist_status s =
            metal_create_pipelines(be, ns_string, METAL_N(metal_quant_rows), metal_quant_rows);
    if (s == GEIST_OK && st->use_q4k_mm_sg) {
        s = metal_create_pipelines(
                be, ns_string, METAL_N(metal_q4k_mm_sg_rows), metal_q4k_mm_sg_rows);
    }
    return s;
}

[[nodiscard]] enum geist_status metal_ensure_attention_pipeline(struct geist_backend *be) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    if (st->q_norm_rope_rows_pipeline != nullptr &&
        st->k_norm_rope_append_rows_pipeline != nullptr &&
        st->k_norm_rope_append_rows_f16_pipeline != nullptr &&
        st->v_norm_append_rows_pipeline != nullptr &&
        st->v_norm_append_rows_f16_pipeline != nullptr &&
        st->kv_norm_append_rows_pipeline != nullptr &&
        st->kv_norm_append_rows_f16_pipeline != nullptr && st->rope_rows_pipeline != nullptr &&
        st->kv_append_rows_pipeline != nullptr && st->kv_append_rows_f16_pipeline != nullptr &&
        st->attention_rows_pipeline != nullptr && st->attention_rows_f16_pipeline != nullptr) {
        return GEIST_OK;
    }

    void *ns_string = metal_objc_get_class(st, "NSString");
    if (ns_string == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: NSString class unavailable");
        return GEIST_E_BACKEND;
    }
    void *source = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:", metal_q_norm_rope_source);
    if (source == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "metal: failed to create Q norm/RoPE shader source");
        return GEIST_E_BACKEND;
    }
    void *err               = nullptr;
    st->q_norm_rope_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
    if (st->q_norm_rope_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q norm/RoPE shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    enum geist_status s = metal_create_named_pipeline(be,
                                                      st->q_norm_rope_library,
                                                      ns_string,
                                                      "q_norm_rope_rows",
                                                      &st->q_norm_rope_rows_function,
                                                      &st->q_norm_rope_rows_pipeline);

    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
                st, ns_string, "stringWithUTF8String:", metal_k_norm_rope_append_source);
        if (source == nullptr) {
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: failed to create K norm/RoPE append shader source");
            return GEIST_E_BACKEND;
        }
        err                            = nullptr;
        st->k_norm_rope_append_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->k_norm_rope_append_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: K norm/RoPE append shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(be,
                                        st->k_norm_rope_append_library,
                                        ns_string,
                                        "k_norm_rope_append_rows",
                                        &st->k_norm_rope_append_rows_function,
                                        &st->k_norm_rope_append_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->k_norm_rope_append_library,
                                        ns_string,
                                        "k_norm_rope_append_rows_f16",
                                        &st->k_norm_rope_append_rows_f16_function,
                                        &st->k_norm_rope_append_rows_f16_pipeline);
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
                st, ns_string, "stringWithUTF8String:", metal_v_norm_append_source);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create V norm append shader source");
            return GEIST_E_BACKEND;
        }
        err                       = nullptr;
        st->v_norm_append_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->v_norm_append_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: V norm append shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(be,
                                        st->v_norm_append_library,
                                        ns_string,
                                        "v_norm_append_rows",
                                        &st->v_norm_append_rows_function,
                                        &st->v_norm_append_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->v_norm_append_library,
                                        ns_string,
                                        "v_norm_append_rows_f16",
                                        &st->v_norm_append_rows_f16_function,
                                        &st->v_norm_append_rows_f16_pipeline);
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
                st, ns_string, "stringWithUTF8String:", metal_kv_norm_append_source);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create K/V norm append shader source");
            return GEIST_E_BACKEND;
        }
        err                        = nullptr;
        st->kv_norm_append_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->kv_norm_append_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: K/V norm append shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(be,
                                        st->kv_norm_append_library,
                                        ns_string,
                                        "kv_norm_append_rows",
                                        &st->kv_norm_append_rows_function,
                                        &st->kv_norm_append_rows_pipeline);
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
                st, ns_string, "stringWithUTF8String:", metal_kv_norm_append_f16_source);
        if (source == nullptr) {
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: failed to create F16 K/V norm append shader source");
            return GEIST_E_BACKEND;
        }
        err                            = nullptr;
        st->kv_norm_append_f16_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->kv_norm_append_f16_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: F16 K/V norm append shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(be,
                                        st->kv_norm_append_f16_library,
                                        ns_string,
                                        "kv_norm_append_rows_f16",
                                        &st->kv_norm_append_rows_f16_function,
                                        &st->kv_norm_append_rows_f16_pipeline);
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_attn_source);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create attention shader source");
            return GEIST_E_BACKEND;
        }
        err              = nullptr;
        st->attn_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->attn_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: attention shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->attn_library,
                                        ns_string,
                                        "rope_rows",
                                        &st->rope_rows_function,
                                        &st->rope_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->attn_library,
                                        ns_string,
                                        "kv_append_rows",
                                        &st->kv_append_rows_function,
                                        &st->kv_append_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->attn_library,
                                        ns_string,
                                        "copy_u32",
                                        &st->copy_u32_function,
                                        &st->copy_u32_pipeline);
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
                st, ns_string, "stringWithUTF8String:", metal_attn_f16_source);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create F16 attention shader source");
            return GEIST_E_BACKEND;
        }
        err                  = nullptr;
        st->attn_f16_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->attn_f16_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: F16 attention shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->attn_f16_library,
                                        ns_string,
                                        "kv_append_rows_f16",
                                        &st->kv_append_rows_f16_function,
                                        &st->kv_append_rows_f16_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->attn_library,
                                        ns_string,
                                        "attention_rows",
                                        &st->attention_rows_function,
                                        &st->attention_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->attn_f16_library,
                                        ns_string,
                                        "attention_rows_f16",
                                        &st->attention_rows_f16_function,
                                        &st->attention_rows_f16_pipeline);
    }
    if (s == GEIST_OK) {
        const size_t dl_h    = strlen(metal_attn_qnorm_dec_f16_source);
        const size_t dl_b    = strlen(metal_attn_dec_f16_body);
        const size_t dl_p    = strlen(metal_attn_dec_f16_plain_head);
        char        *dec_src = malloc(dl_h + 2u * dl_b + dl_p + 1u);
        if (dec_src == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_OOM, "metal: decode attention shader source alloc failed");
            return GEIST_E_OOM;
        }
        memcpy(dec_src, metal_attn_qnorm_dec_f16_source, dl_h);
        memcpy(dec_src + dl_h, metal_attn_dec_f16_body, dl_b);
        memcpy(dec_src + dl_h + dl_b, metal_attn_dec_f16_plain_head, dl_p);
        memcpy(dec_src + dl_h + dl_b + dl_p, metal_attn_dec_f16_body, dl_b + 1u);
        source = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", dec_src);
        free(dec_src);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create decode attention shader source");
            return GEIST_E_BACKEND;
        }
        err                            = nullptr;
        st->attn_qnorm_dec_f16_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->attn_qnorm_dec_f16_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: decode attention shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(be,
                                        st->attn_qnorm_dec_f16_library,
                                        ns_string,
                                        "attention_qnorm_dec_f16",
                                        &st->attention_qnorm_dec_f16_function,
                                        &st->attention_qnorm_dec_f16_pipeline);
        if (s == GEIST_OK) {
            s = metal_create_named_pipeline(be,
                                            st->attn_qnorm_dec_f16_library,
                                            ns_string,
                                            "attention_dec_f16",
                                            &st->attention_dec_f16_function,
                                            &st->attention_dec_f16_pipeline);
        }
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
                st, ns_string, "stringWithUTF8String:", metal_attn_dec_combine_source);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create attention combine shader source");
            return GEIST_E_BACKEND;
        }
        err                          = nullptr;
        st->attn_dec_combine_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->attn_dec_combine_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: attention combine shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(be,
                                        st->attn_dec_combine_library,
                                        ns_string,
                                        "attention_dec_combine",
                                        &st->attention_dec_combine_function,
                                        &st->attention_dec_combine_pipeline);
    }
    if (s == GEIST_OK) {
        const size_t len_a     = strlen(metal_attn_flash_sg_f16_source_a);
        const size_t len_b     = strlen(metal_attn_flash_sg_f16_source_b);
        const size_t len_c     = strlen(metal_attn_flash_sg_f16_plain_head);
        char        *flash_src = malloc(len_a + 2u * len_b + len_c + 1u);
        if (flash_src == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_OOM, "metal: flash attention shader source alloc failed");
            return GEIST_E_OOM;
        }
        memcpy(flash_src, metal_attn_flash_sg_f16_source_a, len_a);
        memcpy(flash_src + len_a, metal_attn_flash_sg_f16_source_b, len_b);
        memcpy(flash_src + len_a + len_b, metal_attn_flash_sg_f16_plain_head, len_c);
        memcpy(flash_src + len_a + len_b + len_c, metal_attn_flash_sg_f16_source_b, len_b + 1u);
        source = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", flash_src);
        free(flash_src);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create flash attention shader source");
            return GEIST_E_BACKEND;
        }
        err                           = nullptr;
        st->attn_flash_sg_f16_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->attn_flash_sg_f16_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: flash attention shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(be,
                                        st->attn_flash_sg_f16_library,
                                        ns_string,
                                        "attention_qnorm_flash_sg_f16",
                                        &st->attention_qnorm_flash_sg_f16_function,
                                        &st->attention_qnorm_flash_sg_f16_pipeline);
        if (s == GEIST_OK) {
            s = metal_create_named_pipeline(be,
                                            st->attn_flash_sg_f16_library,
                                            ns_string,
                                            "attention_flash_sg_f16",
                                            &st->attention_flash_sg_f16_function,
                                            &st->attention_flash_sg_f16_pipeline);
        }
    }
    if (s == GEIST_OK) {
        const size_t l_a     = strlen(metal_attn_flash_sg8_f16_source_a);
        const size_t l_b     = strlen(metal_attn_flash_sg8_f16_source_b);
        char        *sg8_src = malloc(l_a + l_b + 1u);
        if (sg8_src == nullptr) {
            geist_backend_set_error(be, GEIST_E_OOM, "metal: sg8 flash shader source alloc failed");
            return GEIST_E_OOM;
        }
        memcpy(sg8_src, metal_attn_flash_sg8_f16_source_a, l_a);
        memcpy(sg8_src + l_a, metal_attn_flash_sg8_f16_source_b, l_b + 1u);
        source = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", sg8_src);
        free(sg8_src);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create sg8 flash shader source");
            return GEIST_E_BACKEND;
        }
        err                            = nullptr;
        st->attn_flash_sg8_f16_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->attn_flash_sg8_f16_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: sg8 flash shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(be,
                                        st->attn_flash_sg8_f16_library,
                                        ns_string,
                                        "attention_flash_sg8_f16",
                                        &st->attention_flash_sg8_f16_function,
                                        &st->attention_flash_sg8_f16_pipeline);
    }
    if (s == GEIST_OK) {
        const size_t d_a        = strlen(metal_attn_dec512_f16_source_a);
        const size_t d_b        = strlen(metal_attn_dec512_f16_source_b);
        char        *dec512_src = malloc(d_a + d_b + 1u);
        if (dec512_src == nullptr) {
            geist_backend_set_error(be, GEIST_E_OOM, "metal: dec512 shader source alloc failed");
            return GEIST_E_OOM;
        }
        memcpy(dec512_src, metal_attn_dec512_f16_source_a, d_a);
        memcpy(dec512_src + d_a, metal_attn_dec512_f16_source_b, d_b + 1u);
        source = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", dec512_src);
        free(dec512_src);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create dec512 shader source");
            return GEIST_E_BACKEND;
        }
        err                         = nullptr;
        st->attn_dec512_f16_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->attn_dec512_f16_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: dec512 shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(be,
                                        st->attn_dec512_f16_library,
                                        ns_string,
                                        "attention_dec512_f16",
                                        &st->attention_dec512_f16_function,
                                        &st->attention_dec512_f16_pipeline);
    }
    return s;
}

[[nodiscard]] enum geist_status metal_ensure_hadamard_pipeline(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    if (st->hadamard_pipeline != nullptr) {
        return GEIST_OK;
    }
    void *ns_string = metal_objc_get_class(st, "NSString");
    void *source    = ns_string != nullptr
                              ? metal_msg_send_id_cstr(
                                        st, ns_string, "stringWithUTF8String:", metal_hadamard_source)
                              : nullptr;
    if (source == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: hadamard shader source failed");
        return GEIST_E_BACKEND;
    }
    void *err            = nullptr;
    st->hadamard_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
    if (st->hadamard_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: hadamard shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    return metal_create_named_pipeline(be,
                                       st->hadamard_library,
                                       ns_string,
                                       "hadamard_rows",
                                       &st->hadamard_function,
                                       &st->hadamard_pipeline);
}

[[nodiscard]] enum geist_status metal_ensure_argmax_pipeline(struct geist_backend *be) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    if (st->argmax_pipeline != nullptr && st->argmax_batch_pipeline != nullptr &&
        st->argmax_result_buffer != nullptr && st->argmax_result_mapped != nullptr &&
        st->argmax_result_capacity >= 1u) {
        return GEIST_OK;
    }

    void *ns_string = metal_objc_get_class(st, "NSString");
    if (ns_string == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: NSString class unavailable");
        return GEIST_E_BACKEND;
    }
    if (st->argmax_pipeline == nullptr || st->argmax_batch_pipeline == nullptr) {
        void *source =
                metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_argmax_source);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create argmax shader source");
            return GEIST_E_BACKEND;
        }
        void *err          = nullptr;
        st->argmax_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->argmax_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: argmax shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        enum geist_status s = metal_create_named_pipeline(be,
                                                          st->argmax_library,
                                                          ns_string,
                                                          "argmax_f32",
                                                          &st->argmax_function,
                                                          &st->argmax_pipeline);
        if (s != GEIST_OK) {
            return s;
        }
        s = metal_create_named_pipeline(be,
                                        st->argmax_library,
                                        ns_string,
                                        "argmax_f32_batch",
                                        &st->argmax_batch_function,
                                        &st->argmax_batch_pipeline);
        if (s != GEIST_OK) {
            return s;
        }
    }
    if (st->argmax_result_buffer == nullptr) {
        st->argmax_result_buffer = metal_msg_send_id_size_uint(st,
                                                               st->device,
                                                               "newBufferWithLength:options:",
                                                               sizeof(uint32_t),
                                                               METAL_RESOURCE_STORAGE_MODE_SHARED);
        if (st->argmax_result_buffer == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal argmax: result buffer allocation failed");
            return GEIST_E_BACKEND;
        }
        st->argmax_result_mapped = metal_msg_send_id0(st, st->argmax_result_buffer, "contents");
        if (st->argmax_result_mapped == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal argmax: result buffer is not mappable");
            return GEIST_E_BACKEND;
        }
        st->argmax_result_capacity = 1u;
    }
    return GEIST_OK;
}
