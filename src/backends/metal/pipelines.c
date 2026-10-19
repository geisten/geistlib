/*
 * src/backends/metal/pipelines.c — compute-pipeline compilation and caching.
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
 * here and the release in metal_release_pipeline_tables. A table is
 * built in order and stops at its first failure, so its last row's
 * pipeline is set only once every row's is (metal_ensure_tables). */
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
/* A prefill flash kernel's row: head_dim HD in slot HD/32 - 1. */
#define METAL_FLASH_PIPE(HD)                                                \
    {offsetof(struct metal_state, attn_flash_library),                      \
     "attention_flash_d" #HD,                                               \
     offsetof(struct metal_state, attention_flash_function[(HD) / 32 - 1]), \
     offsetof(struct metal_state, attention_flash_pipeline[(HD) / 32 - 1])}
#define METAL_N(rows) (sizeof(rows) / sizeof((rows)[0]))

static const struct metal_pipeline_row metal_deltanet_rows[] = {
        METAL_PIPE(deltanet, "deltanet_mix", deltanet_mix),
        METAL_PIPE(deltanet, "dn_dec_qk", dn_dec_qk),
        METAL_PIPE(deltanet, "dn_dec_v", dn_dec_v),
        METAL_PIPE(deltanet, "dnf_cst_copy", dnf_cst_copy),
        METAL_PIPE(deltanet, "dnf_conv", dnf_conv),
        METAL_PIPE(deltanet, "dnf_norm", dnf_norm),
        METAL_PIPE(deltanet, "dnf_roll", dnf_roll),
        METAL_PIPE(deltanet, "dnf_chunk", dnf_chunk),
        METAL_PIPE(deltanet, "dnf_state", dnf_state),
        METAL_PIPE(deltanet, "dnf_out", dnf_out),
};

static const struct metal_pipeline_row metal_quant_rows[] = {
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
        METAL_PIPE(quant_sg, "cvt_f32_f16_rows", cvt_f16),
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
        METAL_PIPE(q6k, "matvec_q6k", q6k),
        METAL_PIPE(q6k_mm_sg, "matmul_q6k_sg", q6k_matmul_sg),
        METAL_PIPE(q6k_mm_sg_fast, "matmul_q6k_sg_fast", q6k_matmul_sg_fast),
        METAL_PIPE(q6k_n4, "matvec_q6k_n4", q6k_n4),
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
        METAL_PIPE(q4k_mm_sg, "matmul_q4k_mm_sg", q4k_mm_sg),
        METAL_PIPE(q4k_mm_sg_fast, "matmul_q4k_mm_sg_fast", q4k_mm_sg_fast),
        METAL_PIPE(q6k_mm32, "matmul_q6k_mm_sg_fast", q6k_mm32),
};

static const struct metal_pipeline_row metal_attention_rows[] = {
        METAL_PIPE(q_norm_rope, "q_norm_rope_rows", q_norm_rope_rows),
        METAL_PIPE(k_norm_rope_append, "k_norm_rope_append_rows", k_norm_rope_append_rows),
        METAL_PIPE(k_norm_rope_append, "k_norm_rope_append_rows_f16", k_norm_rope_append_rows_f16),
        METAL_PIPE(v_norm_append, "v_norm_append_rows", v_norm_append_rows),
        METAL_PIPE(v_norm_append, "v_norm_append_rows_f16", v_norm_append_rows_f16),
        METAL_PIPE(kv_norm_append, "kv_norm_append_rows", kv_norm_append_rows),
        METAL_PIPE(kv_norm_append_f16, "kv_norm_append_rows_f16", kv_norm_append_rows_f16),
        METAL_PIPE(attn, "rope_rows", rope_rows),
        METAL_PIPE(attn, "kv_append_rows", kv_append_rows),
        METAL_PIPE(attn, "copy_u32", copy_u32),
        METAL_PIPE(attn_f16, "kv_append_rows_f16", kv_append_rows_f16),
        METAL_PIPE(attn, "attention_rows", attention_rows),
        METAL_PIPE(attn_f16, "attention_rows_f16", attention_rows_f16),
        METAL_PIPE(attn_qnorm_dec_f16, "attention_qnorm_dec_f16", attention_qnorm_dec_f16),
        METAL_PIPE(attn_qnorm_dec_f16, "attention_dec_f16", attention_dec_f16),
        METAL_PIPE(attn_dec_combine, "attention_dec_combine", attention_dec_combine),
        METAL_PIPE(attn_dec512_f16, "attention_dec512_f16", attention_dec512_f16),
        METAL_PIPE(attn_dec_f32kv, "attention_dec_f32kv", attention_dec_f32kv),
        METAL_FLASH_PIPE(32),
        METAL_FLASH_PIPE(64),
        METAL_FLASH_PIPE(96),
        METAL_FLASH_PIPE(128),
        METAL_FLASH_PIPE(160),
        METAL_FLASH_PIPE(192),
        METAL_FLASH_PIPE(224),
        METAL_FLASH_PIPE(256),
        METAL_FLASH_PIPE(320),
        METAL_FLASH_PIPE(384),
        METAL_FLASH_PIPE(448),
        METAL_FLASH_PIPE(512),
        METAL_PIPE(attn_dec_f32kv, "attention_dec512_f32kv", attention_dec512_f32kv),
};

static const struct metal_pipeline_row metal_hadamard_rows[] = {
        METAL_PIPE(hadamard, "hadamard_rows", hadamard),
        METAL_PIPE(hadamard, "hadamard_rows4", hadamard4),
};

static const struct metal_pipeline_row metal_argmax_rows[] = {
        METAL_PIPE(argmax, "argmax_f32", argmax),
        METAL_PIPE(argmax, "argmax_f32_batch", argmax_batch),
};

/* A library compiled from one MSL unit: the shader literals concatenated
 * in order. The sources are split into literals of at most 4095 chars
 * (the -Wpedantic limit), so a unit is often several parts. A nullptr
 * part stands for the IQ3_S grid, see metal_iq3s_grid_source. */
struct metal_library_row {
    size_t             library;
    const char        *label; /* names the library in errors */
    size_t             n_parts;
    const char *const *parts;
};
#define METAL_LIB(LIB, LABEL, ...)                                                  \
    {                                                                               \
        offsetof(struct metal_state, LIB##_library), LABEL,                         \
                sizeof((const char *const[]) {__VA_ARGS__}) / sizeof(const char *), \
                (const char *const[]) {                                             \
            __VA_ARGS__                                                             \
        }                                                                           \
    }

static const struct metal_library_row metal_deltanet_libs[] = {
        /* Serial mixer, decode and chunked-prefill kernels compile as one
         * unit (the later sources reuse struct P and silu1). */
        METAL_LIB(deltanet,
                  "DeltaNet",
                  metal_deltanet_source,
                  metal_dn_decode_source,
                  metal_dnf_prep_source,
                  metal_dnf_chunk_source,
                  metal_dnf_state_source,
                  metal_dnf_out_source),
};

static const struct metal_library_row metal_quant_libs[] = {
        METAL_LIB(q4k, "Q4_K", metal_q4k_source),
        METAL_LIB(q40_q80, "Q4_0/Q8_0", metal_q40_q80_source),
        METAL_LIB(q5k, "Q5_K", metal_q5k_source),
        METAL_LIB(q41, "Q4_1", metal_q41_source),
        /* The simdgroup quant kernels compile as one unit. */
        METAL_LIB(quant_sg,
                  "simdgroup quant",
                  metal_qsg_common_source,
                  nullptr,
                  metal_qsg_common2_source,
                  metal_qsg_n4_q40_source,
                  metal_qsg_n4_q80_source,
                  metal_qsg_n4_q41_source,
                  metal_qsg_n4_q5k_source,
                  metal_qsg_mm_q40_source,
                  metal_qsg_mm_q80_source,
                  metal_qsg_mm_q41_source,
                  metal_qsg_mm_q5k_source,
                  metal_qsg_n4_iq4nl_source,
                  metal_qsg_n4_iq4xs_source,
                  metal_qsg_mm_iq4nl_source,
                  metal_qsg_mm_iq4xs_source,
                  metal_qsg_n4_q3k_source,
                  metal_qsg_n4_iq3s_source,
                  metal_qsg_mm_q3k_source,
                  metal_qsg_mm_iq3s_source,
                  metal_qsg_mm_q40_fast_source,
                  metal_qsg_mm_q80_fast_source,
                  metal_qsg_mm_q41_fast_source,
                  metal_qsg_mm_q5k_fast_source,
                  metal_qsg_mm_iq4xs_fast_source,
                  metal_qsg_pq2_dq_source,
                  metal_qsg_pq2_source,
                  metal_qsg_pq2_n8_source,
                  metal_qsg_mm_pq2_source,
                  metal_qsg_mm_pq2_fast_source,
                  metal_qsg_cvt_f16_source,
                  metal_qsg_tq2_source,
                  metal_qsg_tq2_n4_source,
                  metal_qsg_mm_tq2_source,
                  metal_qsg_i2s_source,
                  metal_qsg_i2s_n4_source,
                  metal_qsg_mm_i2s_source),
        METAL_LIB(q4k_n4, "Q4_K n4", metal_q4k_n4_source),
        METAL_LIB(q4k_gate_up_n4, "Q4_K gate/up n4", metal_q4k_gate_up_n4_source),
        METAL_LIB(q4k_pair_n4, "Q4_K pair n4", metal_q4k_pair_n4_source),
        METAL_LIB(q4k_m16, "Q4_K m16", metal_q4k_m16_source),
        METAL_LIB(q4k_mm_sg, "Q4_K simdgroup mm", metal_q4k_mm_sg_source),
        METAL_LIB(q4k_mm_sg_fast, "Q4_K simdgroup mm fast", metal_q4k_mm_sg_fast_source),
        METAL_LIB(q6k_mm32, "Q6_K simdgroup mm fast", metal_q6k_mm32_source),
        METAL_LIB(q6k, "Q6_K", metal_q6k_source),
        METAL_LIB(q6k_mm_sg, "Q6_K mm_sg", metal_q6k_mm_sg_source),
        METAL_LIB(q6k_mm_sg_fast, "Q6_K mm_sg fast", metal_q6k_mm_sg_fast_source),
        METAL_LIB(q6k_n4, "Q6_K n4", metal_q6k_n4_source),
        METAL_LIB(elem, "elementwise", metal_elem_source),
        METAL_LIB(silu, "SiLU", metal_silu_source),
        METAL_LIB(qgate, "Qwen gate", metal_qgate_source),
        METAL_LIB(elem_simd, "SIMD elementwise", metal_elem_simd_source),
        /* single + batched embed kernels share the dequant helpers */
        METAL_LIB(embed, "embedding", metal_embed_source, metal_embed_rows_source),
        METAL_LIB(f32,
                  "F32",
                  metal_f32_source,
                  metal_f32_ple_source,
                  metal_f32_mm_source,
                  metal_f16_w_source,
                  metal_bf16_w_source),
};

static const struct metal_library_row metal_attention_libs[] = {
        METAL_LIB(q_norm_rope, "Q norm/RoPE", metal_q_norm_rope_source),
        METAL_LIB(k_norm_rope_append, "K norm/RoPE append", metal_k_norm_rope_append_source),
        METAL_LIB(v_norm_append, "V norm append", metal_v_norm_append_source),
        METAL_LIB(kv_norm_append, "K/V norm append", metal_kv_norm_append_source),
        METAL_LIB(kv_norm_append_f16, "F16 K/V norm append", metal_kv_norm_append_f16_source),
        METAL_LIB(attn, "attention", metal_attn_source),
        METAL_LIB(attn_f16, "F16 attention", metal_attn_f16_source),
        /* The q-norm and plain decode kernels share one body. */
        METAL_LIB(attn_qnorm_dec_f16,
                  "decode attention",
                  metal_attn_qnorm_dec_f16_source,
                  metal_attn_kv_f16_defs,
                  metal_attn_dec_f16_body,
                  metal_attn_dec_f16_plain_head,
                  metal_attn_dec_f16_body),
        METAL_LIB(attn_dec_combine, "attention combine", metal_attn_dec_combine_source),
        METAL_LIB(attn_dec512_f16,
                  "dec512",
                  metal_attn_kv_header,
                  metal_attn_kv_f16_defs,
                  metal_attn_dec512_f16_source_a,
                  metal_attn_dec512_f16_source_b),
        /* The prefill flash kernels, one template instantiated per head_dim. */
        METAL_LIB(
                attn_flash, "prefill flash", metal_attn_flash_source_a, metal_attn_flash_source_b),
        /* The plain decode kernels again, reading an F32 cache (#64). */
        METAL_LIB(attn_dec_f32kv,
                  "F32-KV decode attention",
                  metal_attn_kv_header,
                  metal_attn_kv_f32_defs,
                  metal_attn_dec_f16_plain_head,
                  metal_attn_dec_f16_body,
                  metal_attn_dec512_f16_source_a,
                  metal_attn_dec512_f16_source_b),
};

static const struct metal_library_row metal_hadamard_libs[] = {
        METAL_LIB(hadamard, "hadamard", metal_hadamard_source),
};

static const struct metal_library_row metal_argmax_libs[] = {
        METAL_LIB(argmax, "argmax", metal_argmax_source),
};

static void **metal_state_slot(struct metal_state *st, size_t offset) {
    return (void **) ((char *) st + offset);
}

/* The IQ3_S grid as an MSL constant array, emitted from quant/iq_grids.h
 * so the 512 entries live in one place; nullptr on allocation failure. */
static char *metal_iq3s_grid_source(void) {
    enum { GRID_STR_CAP = 512 * 12 + 64 };
    char *grid = malloc(GRID_STR_CAP);
    if (grid == nullptr) {
        return nullptr;
    }
    size_t off = (size_t) snprintf(grid, GRID_STR_CAP, "constant uint iq3sg[512]={");
    for (size_t i = 0; i < 512; i++) {
        off += (size_t) snprintf(
                grid + off, GRID_STR_CAP - off, "%s0x%08xu", i == 0 ? "" : ",", iq3s_grid[i]);
    }
    (void) snprintf(grid + off, GRID_STR_CAP - off, "};\n");
    return grid;
}

/* Concatenates the row's parts and compiles them into its library field. */
[[nodiscard]] static enum geist_status metal_compile_library(struct geist_backend *be,
                                                             void                 *ns_string,
                                                             const struct metal_library_row *row) {
    struct metal_state *st   = be->state;
    char               *grid = nullptr;
    for (size_t i = 0; i < row->n_parts && grid == nullptr; i++) {
        if (row->parts[i] == nullptr) {
            grid = metal_iq3s_grid_source();
            if (grid == nullptr) {
                geist_backend_set_error(
                        be, GEIST_E_OOM, "metal: %s shader source alloc failed", row->label);
                return GEIST_E_OOM;
            }
        }
    }
    size_t total = 1u;
    for (size_t i = 0; i < row->n_parts; i++) {
        total += strlen(row->parts[i] != nullptr ? row->parts[i] : grid);
    }
    char *buf = malloc(total);
    if (buf == nullptr) {
        free(grid);
        geist_backend_set_error(
                be, GEIST_E_OOM, "metal: %s shader source alloc failed", row->label);
        return GEIST_E_OOM;
    }
    size_t off = 0;
    for (size_t i = 0; i < row->n_parts; i++) {
        const char  *part = row->parts[i] != nullptr ? row->parts[i] : grid;
        const size_t len  = strlen(part);
        memcpy(buf + off, part, len);
        off += len;
    }
    buf[off]     = '\0';
    void *source = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", buf);
    free(buf);
    free(grid);
    if (source == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "metal: failed to create %s shader source", row->label);
        return GEIST_E_BACKEND;
    }
    void  *err     = nullptr;
    void **library = metal_state_slot(st, row->library);
    *library       = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
    if (*library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: %s shader compile failed%s%s",
                                row->label,
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
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

/* Builds a table's libraries and pipelines unless its last row shows
 * they are built (see struct metal_pipeline_row). */
[[nodiscard]] static enum geist_status
metal_ensure_tables(struct geist_backend           *be,
                    size_t                          n_libs,
                    const struct metal_library_row  libs[static n_libs],
                    size_t                          n_rows,
                    const struct metal_pipeline_row rows[static n_rows]) {
    struct metal_state *st = be->state;
    if (*metal_state_slot(st, rows[n_rows - 1].pipeline) != nullptr) {
        return GEIST_OK;
    }
    void *ns_string = metal_objc_get_class(st, "NSString");
    if (ns_string == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: NSString class unavailable");
        return GEIST_E_BACKEND;
    }
    for (size_t i = 0; i < n_libs; i++) {
        const enum geist_status s = metal_compile_library(be, ns_string, &libs[i]);
        if (s != GEIST_OK) {
            return s;
        }
    }
    return metal_create_pipelines(be, ns_string, n_rows, rows);
}

static void metal_release_rows(struct metal_state             *st,
                               size_t                          n,
                               const struct metal_pipeline_row rows[static n]) {
    for (size_t i = 0; i < n; i++) {
        metal_msg_send_void0(st, *metal_state_slot(st, rows[i].pipeline), "release");
        metal_msg_send_void0(st, *metal_state_slot(st, rows[i].function), "release");
    }
}

static void metal_release_libs(struct metal_state            *st,
                               size_t                         n,
                               const struct metal_library_row libs[static n]) {
    for (size_t i = 0; i < n; i++) {
        metal_msg_send_void0(st, *metal_state_slot(st, libs[i].library), "release");
    }
}

void metal_release_pipeline_tables(struct metal_state *st) {
    metal_release_rows(st, METAL_N(metal_deltanet_rows), metal_deltanet_rows);
    metal_release_rows(st, METAL_N(metal_quant_rows), metal_quant_rows);
    metal_release_rows(st, METAL_N(metal_attention_rows), metal_attention_rows);
    metal_release_rows(st, METAL_N(metal_hadamard_rows), metal_hadamard_rows);
    metal_release_rows(st, METAL_N(metal_argmax_rows), metal_argmax_rows);
    metal_release_libs(st, METAL_N(metal_deltanet_libs), metal_deltanet_libs);
    metal_release_libs(st, METAL_N(metal_quant_libs), metal_quant_libs);
    metal_release_libs(st, METAL_N(metal_attention_libs), metal_attention_libs);
    metal_release_libs(st, METAL_N(metal_hadamard_libs), metal_hadamard_libs);
    metal_release_libs(st, METAL_N(metal_argmax_libs), metal_argmax_libs);
}

#define METAL_TABLES(libs, rows) METAL_N(libs), libs, METAL_N(rows), rows

[[nodiscard]] enum geist_status metal_ensure_deltanet_pipeline(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    return metal_ensure_tables(be, METAL_TABLES(metal_deltanet_libs, metal_deltanet_rows));
}

[[nodiscard]] enum geist_status metal_ensure_q4k_pipeline(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    return metal_ensure_tables(be, METAL_TABLES(metal_quant_libs, metal_quant_rows));
}

[[nodiscard]] enum geist_status metal_ensure_attention_pipeline(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    return metal_ensure_tables(be, METAL_TABLES(metal_attention_libs, metal_attention_rows));
}

[[nodiscard]] enum geist_status metal_ensure_hadamard_pipeline(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    return metal_ensure_tables(be, METAL_TABLES(metal_hadamard_libs, metal_hadamard_rows));
}

[[nodiscard]] enum geist_status metal_ensure_argmax_pipeline(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state     *st = be->state;
    const enum geist_status s =
            metal_ensure_tables(be, METAL_TABLES(metal_argmax_libs, metal_argmax_rows));
    if (s != GEIST_OK) {
        return s;
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
