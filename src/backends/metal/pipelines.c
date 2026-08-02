/*
 * src/backends/metal/pipelines.c — compute-pipeline compilation and caching.
 *
 * Layer: BACKEND (metal). Split from the former monolithic backend.c;
 * pure moves, no behavior change.
 */
#include "metal_internal.h"

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

[[nodiscard]] enum geist_status metal_ensure_q4k_pipeline(struct geist_backend *be) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    if (st->q4k_pipeline != nullptr && st->q4k_n4_pipeline != nullptr &&
        st->q4k_matmul_m8_pipeline != nullptr && st->q4k_matmul_m16_pipeline != nullptr &&
        st->q4k_matmul_m16_n2_pipeline != nullptr &&
        (!st->use_q4k_mm_sg || st->q4k_mm_sg_pipeline != nullptr) && st->q6k_pipeline != nullptr &&
        st->q6k_n4_pipeline != nullptr && st->q6k_matmul_m8_pipeline != nullptr &&
        st->q6k_matmul_m16_pipeline != nullptr && st->rmsnorm_rows_pipeline != nullptr &&
        st->rmsnorm_rows_simd_pipeline != nullptr && st->gelu_rows_pipeline != nullptr &&
        st->mul_rows_pipeline != nullptr && st->gelu_mul_rows_pipeline != nullptr &&
        st->add_rows_pipeline != nullptr && st->scale_rows_pipeline != nullptr &&
        st->rmsnorm_add_rows_pipeline != nullptr && st->rmsnorm_add_rows_simd_pipeline != nullptr &&
        st->embed_lookup_scaled_pipeline != nullptr && st->f32_matmul_pipeline != nullptr &&
        st->f32_ple_gate_pipeline != nullptr && st->f32_ple_proj_norm_pipeline != nullptr) {
        return GEIST_OK;
    }

    void *ns_string = metal_objc_get_class(st, "NSString");
    if (ns_string == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: NSString class unavailable");
        return GEIST_E_BACKEND;
    }
    void *source = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q4k_source);
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
    void *elem_simd_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_elem_simd_source);
    void *embed_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_embed_source);
    void *f32_source = nullptr;
    {
        /* two literals concatenated at init (C99 4095-char literal limit) */
        const size_t f32_len_a = strlen(metal_f32_source);
        const size_t f32_len_b = strlen(metal_f32_mm_source);
        char        *f32_src   = malloc(f32_len_a + f32_len_b + 1u);
        if (f32_src == nullptr) {
            geist_backend_set_error(be, GEIST_E_OOM, "metal: f32 shader source alloc failed");
            return GEIST_E_OOM;
        }
        memcpy(f32_src, metal_f32_source, f32_len_a);
        memcpy(f32_src + f32_len_a, metal_f32_mm_source, f32_len_b + 1u);
        f32_source = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", f32_src);
        free(f32_src);
    }
    if (source == nullptr || q4k_n4_source == nullptr || q4k_m16_source == nullptr ||
        q4k_m16_n2_source == nullptr || (st->use_q4k_mm_sg && q4k_mm_sg_ns_source == nullptr) ||
        q4k_gate_up_n4_src == nullptr || q4k_pair_n4_src == nullptr || q6_source == nullptr ||
        q6_n4_source == nullptr || q6_m16_source == nullptr || elem_source == nullptr ||
        elem_simd_source == nullptr || embed_source == nullptr || f32_source == nullptr) {
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
    enum geist_status s = metal_create_named_pipeline(
            be, st->q4k_library, ns_string, "matvec_q4k", &st->q4k_function, &st->q4k_pipeline);
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q4k_n4_library,
                                        ns_string,
                                        "matvec_q4k_n4",
                                        &st->q4k_n4_function,
                                        &st->q4k_n4_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q4k_gate_up_n4_library,
                                        ns_string,
                                        "gate_up_q4k_n4",
                                        &st->q4k_gate_up_n4_function,
                                        &st->q4k_gate_up_n4_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q4k_pair_n4_library,
                                        ns_string,
                                        "pair_q4k_n4",
                                        &st->q4k_pair_n4_function,
                                        &st->q4k_pair_n4_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q4k_library,
                                        ns_string,
                                        "matmul_q4k_m8",
                                        &st->q4k_matmul_m8_function,
                                        &st->q4k_matmul_m8_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q4k_m16_library,
                                        ns_string,
                                        "matmul_q4k_m16",
                                        &st->q4k_matmul_m16_function,
                                        &st->q4k_matmul_m16_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q4k_m16_n2_library,
                                        ns_string,
                                        "matmul_q4k_m16_n2",
                                        &st->q4k_matmul_m16_n2_function,
                                        &st->q4k_matmul_m16_n2_pipeline);
    }
    if (s == GEIST_OK && st->use_q4k_mm_sg) {
        s = metal_create_named_pipeline(be,
                                        st->q4k_mm_sg_library,
                                        ns_string,
                                        "matmul_q4k_mm_sg",
                                        &st->q4k_mm_sg_function,
                                        &st->q4k_mm_sg_pipeline);
    }
    if (s == GEIST_OK && st->use_q4k_mm_sg) {
        s = metal_create_named_pipeline(be,
                                        st->q4k_mm_sg_fast_library,
                                        ns_string,
                                        "matmul_q4k_mm_sg_fast",
                                        &st->q4k_mm_sg_fast_function,
                                        &st->q4k_mm_sg_fast_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
                be, st->q6k_library, ns_string, "matvec_q6k", &st->q6k_function, &st->q6k_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q6k_mm_sg_library,
                                        ns_string,
                                        "matmul_q6k_sg",
                                        &st->q6k_matmul_sg_function,
                                        &st->q6k_matmul_sg_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q6k_mm_sg_fast_library,
                                        ns_string,
                                        "matmul_q6k_sg_fast",
                                        &st->q6k_matmul_sg_fast_function,
                                        &st->q6k_matmul_sg_fast_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q6k_n4_library,
                                        ns_string,
                                        "matvec_q6k_n4",
                                        &st->q6k_n4_function,
                                        &st->q6k_n4_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q6k_library,
                                        ns_string,
                                        "matmul_q6k_m8",
                                        &st->q6k_matmul_m8_function,
                                        &st->q6k_matmul_m8_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q6k_m16_library,
                                        ns_string,
                                        "matmul_q6k_m16",
                                        &st->q6k_matmul_m16_function,
                                        &st->q6k_matmul_m16_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->elem_library,
                                        ns_string,
                                        "rmsnorm_rows",
                                        &st->rmsnorm_rows_function,
                                        &st->rmsnorm_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->elem_simd_library,
                                        ns_string,
                                        "rmsnorm_rows_simd",
                                        &st->rmsnorm_rows_simd_function,
                                        &st->rmsnorm_rows_simd_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->elem_library,
                                        ns_string,
                                        "gelu_rows",
                                        &st->gelu_rows_function,
                                        &st->gelu_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->elem_library,
                                        ns_string,
                                        "mul_rows",
                                        &st->mul_rows_function,
                                        &st->mul_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->elem_library,
                                        ns_string,
                                        "gelu_mul_rows",
                                        &st->gelu_mul_rows_function,
                                        &st->gelu_mul_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->elem_library,
                                        ns_string,
                                        "add_rows",
                                        &st->add_rows_function,
                                        &st->add_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->elem_library,
                                        ns_string,
                                        "scale_rows",
                                        &st->scale_rows_function,
                                        &st->scale_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->elem_library,
                                        ns_string,
                                        "rmsnorm_add_rows",
                                        &st->rmsnorm_add_rows_function,
                                        &st->rmsnorm_add_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->elem_simd_library,
                                        ns_string,
                                        "rmsnorm_add_rows_simd",
                                        &st->rmsnorm_add_rows_simd_function,
                                        &st->rmsnorm_add_rows_simd_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->embed_library,
                                        ns_string,
                                        "embed_lookup_scaled",
                                        &st->embed_lookup_scaled_function,
                                        &st->embed_lookup_scaled_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->f32_library,
                                        ns_string,
                                        "matmul_f32",
                                        &st->f32_matmul_function,
                                        &st->f32_matmul_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->f32_library,
                                        ns_string,
                                        "matmul_f32_sg",
                                        &st->f32_matmul_sg_function,
                                        &st->f32_matmul_sg_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->f32_library,
                                        ns_string,
                                        "matmul_f32_mm_sg",
                                        &st->f32_matmul_mm_function,
                                        &st->f32_matmul_mm_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->f32_library,
                                        ns_string,
                                        "ple_gate_f32",
                                        &st->f32_ple_gate_function,
                                        &st->f32_ple_gate_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->f32_library,
                                        ns_string,
                                        "ple_proj_norm_f32",
                                        &st->f32_ple_proj_norm_function,
                                        &st->f32_ple_proj_norm_pipeline);
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
