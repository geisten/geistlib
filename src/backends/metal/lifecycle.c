/*
 * src/backends/metal/lifecycle.c — device/queue lifecycle and runtime loading.
 *
 * Layer: BACKEND (metal). Split from the former monolithic backend.c;
 * pure moves, no behavior change.
 */
#include "metal_internal.h"

static void metal_destroy_state(struct geist_backend *be, struct metal_state *st) {
    if (be == nullptr || st == nullptr) {
        return;
    }
    metal_profile_print_summary(st);
    free(st->buf_reg);
    st->buf_reg       = nullptr;
    st->buf_reg_count = 0;
    st->buf_reg_cap   = 0;
    metal_buffer_destroy_internal(be, st->attn_dec_partials_buffer);
    st->attn_dec_partials_buffer = nullptr;
    metal_buffer_destroy_internal(be, st->attn_kf16_buffer);
    metal_buffer_destroy_internal(be, st->attn_vf16_buffer);
    st->attn_kf16_buffer           = nullptr;
    st->attn_vf16_buffer           = nullptr;
    st->attn_kvf16_capacity        = 0;
    st->attn_dec_partials_capacity = 0;
    if (st->objc_msgSend != nullptr && st->sel_registerName != nullptr) {
        if (st->sequence_active) {
            metal_msg_send_void0(st, st->sequence_compute_encoder, "endEncoding");
        }
        metal_release_sequence_objects(st);
        metal_msg_send_void0(st, st->argmax_result_buffer, "release");
        metal_msg_send_void0(st, st->dn_scratch, "release");
        metal_msg_send_void0(st, st->argmax_batch_pipeline, "release");
        metal_msg_send_void0(st, st->argmax_batch_function, "release");
        metal_msg_send_void0(st, st->argmax_pipeline, "release");
        metal_msg_send_void0(st, st->argmax_function, "release");
        metal_msg_send_void0(st, st->deltanet_mix_pipeline, "release");
        metal_msg_send_void0(st, st->deltanet_mix_function, "release");
        metal_msg_send_void0(st, st->dn_cst_copy_pipeline, "release");
        metal_msg_send_void0(st, st->dn_cst_copy_function, "release");
        metal_msg_send_void0(st, st->dn_conv_prep_pipeline, "release");
        metal_msg_send_void0(st, st->dn_conv_prep_function, "release");
        metal_msg_send_void0(st, st->dn_qk_norm_pipeline, "release");
        metal_msg_send_void0(st, st->dn_qk_norm_function, "release");
        metal_msg_send_void0(st, st->dn_state_roll_pipeline, "release");
        metal_msg_send_void0(st, st->dn_state_roll_function, "release");
        metal_msg_send_void0(st, st->dn_chunk_stage_pipeline, "release");
        metal_msg_send_void0(st, st->dn_chunk_stage_function, "release");
        metal_msg_send_void0(st, st->dn_chunk_subst_pipeline, "release");
        metal_msg_send_void0(st, st->dn_chunk_subst_function, "release");
        metal_msg_send_void0(st, st->dn_chunk_amat_pipeline, "release");
        metal_msg_send_void0(st, st->dn_chunk_amat_function, "release");
        metal_msg_send_void0(st, st->dn_chunk_vnew1_pipeline, "release");
        metal_msg_send_void0(st, st->dn_chunk_vnew1_function, "release");
        metal_msg_send_void0(st, st->dn_chunk_vnew2_pipeline, "release");
        metal_msg_send_void0(st, st->dn_chunk_vnew2_function, "release");
        metal_msg_send_void0(st, st->dn_chunk_out_pipeline, "release");
        metal_msg_send_void0(st, st->dn_chunk_out_function, "release");
        metal_msg_send_void0(st, st->dn_chunk_supd_pipeline, "release");
        metal_msg_send_void0(st, st->dn_chunk_supd_function, "release");
        metal_msg_send_void0(st, st->dn_chunk_gate_pipeline, "release");
        metal_msg_send_void0(st, st->dn_chunk_gate_function, "release");
        metal_msg_send_void0(st, st->deltanet_library, "release");
        metal_msg_send_void0(st, st->qgate_split_pipeline, "release");
        metal_msg_send_void0(st, st->qgate_split_function, "release");
        metal_msg_send_void0(st, st->sigmoid_mul_pipeline, "release");
        metal_msg_send_void0(st, st->sigmoid_mul_function, "release");
        metal_msg_send_void0(st, st->qgate_library, "release");
        metal_msg_send_void0(st, st->rmsnorm_add_rows_simd_pipeline, "release");
        metal_msg_send_void0(st, st->rmsnorm_add_rows_simd_function, "release");
        metal_msg_send_void0(st, st->rmsnorm_add_rows_pipeline, "release");
        metal_msg_send_void0(st, st->rmsnorm_add_rows_function, "release");
        metal_msg_send_void0(st, st->embed_lookup_scaled_pipeline, "release");
        metal_msg_send_void0(st, st->embed_lookup_scaled_function, "release");
        metal_msg_send_void0(st, st->embed_lookup_scaled_rows_pipeline, "release");
        metal_msg_send_void0(st, st->embed_lookup_scaled_rows_function, "release");
        metal_msg_send_void0(st, st->f32_matmul_pipeline, "release");
        metal_msg_send_void0(st, st->f32_matmul_function, "release");
    }
    if (st->f32_matmul_sg_pipeline != nullptr) {
        metal_msg_send_void0(st, st->f32_matmul_sg_pipeline, "release");
        metal_msg_send_void0(st, st->f32_matmul_sg_function, "release");
        metal_msg_send_void0(st, st->f32_matmul_mm_pipeline, "release");
        metal_msg_send_void0(st, st->f32_matmul_mm_function, "release");
        metal_msg_send_void0(st, st->f32_ple_gate_pipeline, "release");
        metal_msg_send_void0(st, st->f32_ple_gate_function, "release");
        metal_msg_send_void0(st, st->f32_ple_proj_norm_pipeline, "release");
        metal_msg_send_void0(st, st->f32_ple_proj_norm_function, "release");
        metal_msg_send_void0(st, st->attention_rows_pipeline, "release");
        metal_msg_send_void0(st, st->attention_rows_function, "release");
        metal_msg_send_void0(st, st->attention_rows_f16_pipeline, "release");
        metal_msg_send_void0(st, st->attention_rows_f16_function, "release");
        metal_msg_send_void0(st, st->attention_qnorm_dec_f16_pipeline, "release");
        metal_msg_send_void0(st, st->attention_qnorm_dec_f16_function, "release");
        metal_msg_send_void0(st, st->attention_dec_f16_pipeline, "release");
        metal_msg_send_void0(st, st->attention_dec_f16_function, "release");
        metal_msg_send_void0(st, st->attention_dec_combine_pipeline, "release");
        metal_msg_send_void0(st, st->attention_dec_combine_function, "release");
        metal_msg_send_void0(st, st->attention_dec512_f16_pipeline, "release");
        metal_msg_send_void0(st, st->attention_dec512_f16_function, "release");
        metal_msg_send_void0(st, st->attn_dec512_f16_library, "release");
        metal_msg_send_void0(st, st->attention_flash_sg8_f16_pipeline, "release");
        metal_msg_send_void0(st, st->attention_flash_sg8_f16_function, "release");
        metal_msg_send_void0(st, st->attn_flash_sg8_f16_library, "release");
        metal_msg_send_void0(st, st->attention_qnorm_flash_sg_f16_pipeline, "release");
        metal_msg_send_void0(st, st->attention_qnorm_flash_sg_f16_function, "release");
        metal_msg_send_void0(st, st->attention_flash_sg_f16_pipeline, "release");
        metal_msg_send_void0(st, st->attention_flash_sg_f16_function, "release");
        metal_msg_send_void0(st, st->kv_append_rows_pipeline, "release");
        metal_msg_send_void0(st, st->kv_append_rows_function, "release");
        metal_msg_send_void0(st, st->copy_u32_pipeline, "release");
        metal_msg_send_void0(st, st->copy_u32_function, "release");
        metal_msg_send_void0(st, st->kv_append_rows_f16_pipeline, "release");
        metal_msg_send_void0(st, st->kv_append_rows_f16_function, "release");
        metal_msg_send_void0(st, st->rope_rows_pipeline, "release");
        metal_msg_send_void0(st, st->rope_rows_function, "release");
        metal_msg_send_void0(st, st->v_norm_append_rows_pipeline, "release");
        metal_msg_send_void0(st, st->v_norm_append_rows_function, "release");
        metal_msg_send_void0(st, st->v_norm_append_rows_f16_pipeline, "release");
        metal_msg_send_void0(st, st->v_norm_append_rows_f16_function, "release");
        metal_msg_send_void0(st, st->kv_norm_append_rows_pipeline, "release");
        metal_msg_send_void0(st, st->kv_norm_append_rows_function, "release");
        metal_msg_send_void0(st, st->kv_norm_append_rows_f16_pipeline, "release");
        metal_msg_send_void0(st, st->kv_norm_append_rows_f16_function, "release");
        metal_msg_send_void0(st, st->k_norm_rope_append_rows_pipeline, "release");
        metal_msg_send_void0(st, st->k_norm_rope_append_rows_function, "release");
        metal_msg_send_void0(st, st->k_norm_rope_append_rows_f16_pipeline, "release");
        metal_msg_send_void0(st, st->k_norm_rope_append_rows_f16_function, "release");
        metal_msg_send_void0(st, st->q_norm_rope_rows_pipeline, "release");
        metal_msg_send_void0(st, st->q_norm_rope_rows_function, "release");
        metal_msg_send_void0(st, st->add_rows_pipeline, "release");
        metal_msg_send_void0(st, st->add_rows_function, "release");
        metal_msg_send_void0(st, st->scale_rows_pipeline, "release");
        metal_msg_send_void0(st, st->scale_rows_function, "release");
        metal_msg_send_void0(st, st->gelu_mul_rows_pipeline, "release");
        metal_msg_send_void0(st, st->gelu_mul_rows_function, "release");
        metal_msg_send_void0(st, st->mul_rows_pipeline, "release");
        metal_msg_send_void0(st, st->mul_rows_function, "release");
        metal_msg_send_void0(st, st->gelu_rows_pipeline, "release");
        metal_msg_send_void0(st, st->gelu_rows_function, "release");
        metal_msg_send_void0(st, st->silu_rows_pipeline, "release");
        metal_msg_send_void0(st, st->silu_rows_function, "release");
        metal_msg_send_void0(st, st->silu_mul_rows_pipeline, "release");
        metal_msg_send_void0(st, st->silu_mul_rows_function, "release");
        metal_msg_send_void0(st, st->rmsnorm_rows_simd_pipeline, "release");
        metal_msg_send_void0(st, st->rmsnorm_rows_simd_function, "release");
        metal_msg_send_void0(st, st->rmsnorm_rows_pipeline, "release");
        metal_msg_send_void0(st, st->rmsnorm_rows_function, "release");
        metal_msg_send_void0(st, st->q6k_matmul_m8_pipeline, "release");
        metal_msg_send_void0(st, st->q6k_matmul_sg_pipeline, "release");
        metal_msg_send_void0(st, st->q6k_matmul_sg_fast_pipeline, "release");
        metal_msg_send_void0(st, st->q6k_matmul_sg_fast_function, "release");
        metal_msg_send_void0(st, st->q6k_matmul_sg_function, "release");
        metal_msg_send_void0(st, st->q6k_matmul_m8_function, "release");
        metal_msg_send_void0(st, st->q6k_matmul_m16_pipeline, "release");
        metal_msg_send_void0(st, st->q6k_matmul_m16_function, "release");
        metal_msg_send_void0(st, st->q6k_n4_pipeline, "release");
        metal_msg_send_void0(st, st->q6k_n4_function, "release");
        metal_msg_send_void0(st, st->q4k_qk_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_qk_function, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_function, "release");
        metal_msg_send_void0(st, st->q6k_pipeline, "release");
        metal_msg_send_void0(st, st->q6k_function, "release");
        metal_msg_send_void0(st, st->q4k_matmul_m8_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_matmul_m8_function, "release");
        metal_msg_send_void0(st, st->q4k_matmul_m16_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_matmul_m16_function, "release");
        metal_msg_send_void0(st, st->q4k_matmul_m16_n2_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_matmul_m16_n2_function, "release");
        metal_msg_send_void0(st, st->q4k_mm_sg_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_mm_sg_function, "release");
        metal_msg_send_void0(st, st->q4k_mm_sg_fast_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_mm_sg_fast_function, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_n4_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_n4_function, "release");
        metal_msg_send_void0(st, st->q4k_pair_n4_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_pair_n4_function, "release");
        metal_msg_send_void0(st, st->q4k_n4_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_n4_function, "release");
        metal_msg_send_void0(st, st->q4k_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_function, "release");
        metal_msg_send_void0(st, st->q40_pipeline, "release");
        metal_msg_send_void0(st, st->q40_function, "release");
        metal_msg_send_void0(st, st->q40_m8_pipeline, "release");
        metal_msg_send_void0(st, st->q40_m8_function, "release");
        metal_msg_send_void0(st, st->q80_pipeline, "release");
        metal_msg_send_void0(st, st->q80_function, "release");
        metal_msg_send_void0(st, st->q80_m8_pipeline, "release");
        metal_msg_send_void0(st, st->q80_m8_function, "release");
        metal_msg_send_void0(st, st->q41_pipeline, "release");
        metal_msg_send_void0(st, st->q41_function, "release");
        metal_msg_send_void0(st, st->q41_m8_pipeline, "release");
        metal_msg_send_void0(st, st->q41_m8_function, "release");
        metal_msg_send_void0(st, st->q5k_pipeline, "release");
        metal_msg_send_void0(st, st->q5k_function, "release");
        metal_msg_send_void0(st, st->q5k_m8_pipeline, "release");
        metal_msg_send_void0(st, st->q5k_m8_function, "release");
        metal_msg_send_void0(st, st->q40_n4_pipeline, "release");
        metal_msg_send_void0(st, st->q40_n4_function, "release");
        metal_msg_send_void0(st, st->q40_mm_pipeline, "release");
        metal_msg_send_void0(st, st->q40_mm_function, "release");
        metal_msg_send_void0(st, st->q80_n4_pipeline, "release");
        metal_msg_send_void0(st, st->q80_n4_function, "release");
        metal_msg_send_void0(st, st->q80_mm_pipeline, "release");
        metal_msg_send_void0(st, st->q80_mm_function, "release");
        metal_msg_send_void0(st, st->q41_n4_pipeline, "release");
        metal_msg_send_void0(st, st->q41_n4_function, "release");
        metal_msg_send_void0(st, st->q41_mm_pipeline, "release");
        metal_msg_send_void0(st, st->q41_mm_function, "release");
        metal_msg_send_void0(st, st->q5k_n4_pipeline, "release");
        metal_msg_send_void0(st, st->q5k_n4_function, "release");
        metal_msg_send_void0(st, st->q5k_mm_pipeline, "release");
        metal_msg_send_void0(st, st->q5k_mm_function, "release");
        metal_msg_send_void0(st, st->q40_mm_fast_pipeline, "release");
        metal_msg_send_void0(st, st->q40_mm_fast_function, "release");
        metal_msg_send_void0(st, st->q80_mm_fast_pipeline, "release");
        metal_msg_send_void0(st, st->q80_mm_fast_function, "release");
        metal_msg_send_void0(st, st->q41_mm_fast_pipeline, "release");
        metal_msg_send_void0(st, st->q41_mm_fast_function, "release");
        metal_msg_send_void0(st, st->q5k_mm_fast_pipeline, "release");
        metal_msg_send_void0(st, st->q5k_mm_fast_function, "release");
        metal_msg_send_void0(st, st->iq4nl_n4_pipeline, "release");
        metal_msg_send_void0(st, st->iq4nl_n4_function, "release");
        metal_msg_send_void0(st, st->iq4nl_mm_pipeline, "release");
        metal_msg_send_void0(st, st->iq4nl_mm_function, "release");
        metal_msg_send_void0(st, st->iq4xs_n4_pipeline, "release");
        metal_msg_send_void0(st, st->iq4xs_n4_function, "release");
        metal_msg_send_void0(st, st->iq4xs_mm_pipeline, "release");
        metal_msg_send_void0(st, st->iq4xs_mm_function, "release");
        metal_msg_send_void0(st, st->q3k_n4_pipeline, "release");
        metal_msg_send_void0(st, st->q3k_n4_function, "release");
        metal_msg_send_void0(st, st->q3k_mm_pipeline, "release");
        metal_msg_send_void0(st, st->q3k_mm_function, "release");
        metal_msg_send_void0(st, st->iq3s_n4_pipeline, "release");
        metal_msg_send_void0(st, st->iq3s_n4_function, "release");
        metal_msg_send_void0(st, st->iq3s_mm_pipeline, "release");
        metal_msg_send_void0(st, st->iq3s_mm_function, "release");
        metal_msg_send_void0(st, st->iq4xs_mm_fast_pipeline, "release");
        metal_msg_send_void0(st, st->iq4xs_mm_fast_function, "release");
        metal_msg_send_void0(st, st->attn_library, "release");
        metal_msg_send_void0(st, st->attn_qnorm_dec_f16_library, "release");
        metal_msg_send_void0(st, st->attn_dec_combine_library, "release");
        metal_msg_send_void0(st, st->attn_flash_sg_f16_library, "release");
        metal_msg_send_void0(st, st->kv_norm_append_library, "release");
        metal_msg_send_void0(st, st->kv_norm_append_f16_library, "release");
        metal_msg_send_void0(st, st->v_norm_append_library, "release");
        metal_msg_send_void0(st, st->k_norm_rope_append_library, "release");
        metal_msg_send_void0(st, st->q_norm_rope_library, "release");
        metal_msg_send_void0(st, st->elem_simd_library, "release");
        metal_msg_send_void0(st, st->elem_library, "release");
        metal_msg_send_void0(st, st->embed_library, "release");
        metal_msg_send_void0(st, st->argmax_library, "release");
        metal_msg_send_void0(st, st->f32_library, "release");
        metal_msg_send_void0(st, st->q4k_qk_library, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_n4_library, "release");
        metal_msg_send_void0(st, st->q4k_pair_n4_library, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_library, "release");
        metal_msg_send_void0(st, st->q4k_n4_library, "release");
        metal_msg_send_void0(st, st->q6k_mm_sg_library, "release");
        metal_msg_send_void0(st, st->q6k_mm_sg_fast_library, "release");
        metal_msg_send_void0(st, st->q6k_m16_library, "release");
        metal_msg_send_void0(st, st->q4k_m16_library, "release");
        metal_msg_send_void0(st, st->q4k_m16_n2_library, "release");
        metal_msg_send_void0(st, st->q4k_mm_sg_library, "release");
        metal_msg_send_void0(st, st->q4k_mm_sg_fast_library, "release");
        metal_msg_send_void0(st, st->q6k_n4_library, "release");
        metal_msg_send_void0(st, st->q6k_library, "release");
        metal_msg_send_void0(st, st->q4k_library, "release");
        metal_msg_send_void0(st, st->q40_q80_library, "release");
        metal_msg_send_void0(st, st->q5k_library, "release");
        metal_msg_send_void0(st, st->q41_library, "release");
        metal_msg_send_void0(st, st->quant_sg_library, "release");
        metal_msg_send_void0(st, st->silu_library, "release");
        metal_msg_send_void0(st, st->attn_f16_library, "release");
        metal_msg_send_void0(st, st->command_queue, "release");
        metal_msg_send_void0(st, st->device, "release");
    }
    st->q4k_pipeline                          = nullptr;
    st->q4k_function                          = nullptr;
    st->q40_pipeline                          = nullptr;
    st->q40_function                          = nullptr;
    st->q40_m8_pipeline                       = nullptr;
    st->q40_m8_function                       = nullptr;
    st->q80_pipeline                          = nullptr;
    st->q80_function                          = nullptr;
    st->q80_m8_pipeline                       = nullptr;
    st->q80_m8_function                       = nullptr;
    st->q41_pipeline                          = nullptr;
    st->q41_function                          = nullptr;
    st->q41_m8_pipeline                       = nullptr;
    st->q41_m8_function                       = nullptr;
    st->q5k_pipeline                          = nullptr;
    st->q5k_function                          = nullptr;
    st->q5k_m8_pipeline                       = nullptr;
    st->q5k_m8_function                       = nullptr;
    st->q40_n4_pipeline                       = nullptr;
    st->q40_n4_function                       = nullptr;
    st->q40_mm_pipeline                       = nullptr;
    st->q40_mm_function                       = nullptr;
    st->q80_n4_pipeline                       = nullptr;
    st->q80_n4_function                       = nullptr;
    st->q80_mm_pipeline                       = nullptr;
    st->q80_mm_function                       = nullptr;
    st->q41_n4_pipeline                       = nullptr;
    st->q41_n4_function                       = nullptr;
    st->q41_mm_pipeline                       = nullptr;
    st->q41_mm_function                       = nullptr;
    st->q5k_n4_pipeline                       = nullptr;
    st->q5k_n4_function                       = nullptr;
    st->q5k_mm_pipeline                       = nullptr;
    st->q5k_mm_function                       = nullptr;
    st->q40_mm_fast_pipeline                  = nullptr;
    st->q40_mm_fast_function                  = nullptr;
    st->q80_mm_fast_pipeline                  = nullptr;
    st->q80_mm_fast_function                  = nullptr;
    st->q41_mm_fast_pipeline                  = nullptr;
    st->q41_mm_fast_function                  = nullptr;
    st->q5k_mm_fast_pipeline                  = nullptr;
    st->q5k_mm_fast_function                  = nullptr;
    st->iq4nl_n4_pipeline                     = nullptr;
    st->iq4nl_n4_function                     = nullptr;
    st->iq4nl_mm_pipeline                     = nullptr;
    st->iq4nl_mm_function                     = nullptr;
    st->iq4xs_n4_pipeline                     = nullptr;
    st->iq4xs_n4_function                     = nullptr;
    st->iq4xs_mm_pipeline                     = nullptr;
    st->iq4xs_mm_function                     = nullptr;
    st->q3k_n4_pipeline                       = nullptr;
    st->q3k_n4_function                       = nullptr;
    st->q3k_mm_pipeline                       = nullptr;
    st->q3k_mm_function                       = nullptr;
    st->iq3s_n4_pipeline                      = nullptr;
    st->iq3s_n4_function                      = nullptr;
    st->iq3s_mm_pipeline                      = nullptr;
    st->iq3s_mm_function                      = nullptr;
    st->iq4xs_mm_fast_pipeline                = nullptr;
    st->iq4xs_mm_fast_function                = nullptr;
    st->q40_q80_library                       = nullptr;
    st->q5k_library                           = nullptr;
    st->q41_library                           = nullptr;
    st->quant_sg_library                      = nullptr;
    st->silu_library                          = nullptr;
    st->deltanet_library                      = nullptr;
    st->dn_cst_copy_pipeline                  = nullptr;
    st->dn_cst_copy_function                  = nullptr;
    st->dn_conv_prep_pipeline                 = nullptr;
    st->dn_conv_prep_function                 = nullptr;
    st->dn_qk_norm_pipeline                   = nullptr;
    st->dn_qk_norm_function                   = nullptr;
    st->dn_state_roll_pipeline                = nullptr;
    st->dn_state_roll_function                = nullptr;
    st->dn_chunk_stage_pipeline               = nullptr;
    st->dn_chunk_stage_function               = nullptr;
    st->dn_chunk_subst_pipeline               = nullptr;
    st->dn_chunk_subst_function               = nullptr;
    st->dn_chunk_amat_pipeline                = nullptr;
    st->dn_chunk_amat_function                = nullptr;
    st->dn_chunk_vnew1_pipeline               = nullptr;
    st->dn_chunk_vnew1_function               = nullptr;
    st->dn_chunk_vnew2_pipeline               = nullptr;
    st->dn_chunk_vnew2_function               = nullptr;
    st->dn_chunk_out_pipeline                 = nullptr;
    st->dn_chunk_out_function                 = nullptr;
    st->dn_chunk_supd_pipeline                = nullptr;
    st->dn_chunk_supd_function                = nullptr;
    st->dn_chunk_gate_pipeline                = nullptr;
    st->dn_chunk_gate_function                = nullptr;
    st->qgate_library                         = nullptr;
    st->q4k_n4_pipeline                       = nullptr;
    st->q4k_n4_function                       = nullptr;
    st->q4k_matmul_m8_pipeline                = nullptr;
    st->q4k_matmul_m8_function                = nullptr;
    st->q4k_matmul_m16_pipeline               = nullptr;
    st->q4k_matmul_m16_function               = nullptr;
    st->q4k_matmul_m16_n2_pipeline            = nullptr;
    st->q4k_matmul_m16_n2_function            = nullptr;
    st->q4k_mm_sg_pipeline                    = nullptr;
    st->q4k_mm_sg_function                    = nullptr;
    st->q4k_mm_sg_fast_pipeline               = nullptr;
    st->q4k_mm_sg_fast_function               = nullptr;
    st->q4k_gate_up_n4_pipeline               = nullptr;
    st->q4k_gate_up_n4_function               = nullptr;
    st->q4k_pair_n4_pipeline                  = nullptr;
    st->q4k_pair_n4_function                  = nullptr;
    st->q4k_m16_library                       = nullptr;
    st->q4k_mm_sg_library                     = nullptr;
    st->q4k_mm_sg_fast_library                = nullptr;
    st->q6k_pipeline                          = nullptr;
    st->q6k_function                          = nullptr;
    st->q6k_n4_pipeline                       = nullptr;
    st->q6k_n4_function                       = nullptr;
    st->q6k_matmul_m8_pipeline                = nullptr;
    st->q6k_matmul_sg_function                = nullptr;
    st->q6k_matmul_sg_pipeline                = nullptr;
    st->q6k_matmul_sg_fast_pipeline           = nullptr;
    st->q6k_matmul_sg_fast_function           = nullptr;
    st->q6k_matmul_m8_function                = nullptr;
    st->q6k_matmul_m16_pipeline               = nullptr;
    st->q6k_matmul_m16_function               = nullptr;
    st->q6k_mm_sg_library                     = nullptr;
    st->q6k_mm_sg_fast_library                = nullptr;
    st->q6k_m16_library                       = nullptr;
    st->q4k_qk_pipeline                       = nullptr;
    st->q4k_qk_function                       = nullptr;
    st->q4k_gate_up_pipeline                  = nullptr;
    st->q4k_gate_up_function                  = nullptr;
    st->rmsnorm_rows_pipeline                 = nullptr;
    st->rmsnorm_rows_function                 = nullptr;
    st->rmsnorm_rows_simd_pipeline            = nullptr;
    st->rmsnorm_rows_simd_function            = nullptr;
    st->gelu_rows_pipeline                    = nullptr;
    st->gelu_rows_function                    = nullptr;
    st->silu_rows_pipeline                    = nullptr;
    st->silu_rows_function                    = nullptr;
    st->silu_mul_rows_pipeline                = nullptr;
    st->silu_mul_rows_function                = nullptr;
    st->deltanet_mix_pipeline                 = nullptr;
    st->deltanet_mix_function                 = nullptr;
    st->qgate_split_pipeline                  = nullptr;
    st->qgate_split_function                  = nullptr;
    st->sigmoid_mul_pipeline                  = nullptr;
    st->sigmoid_mul_function                  = nullptr;
    st->mul_rows_pipeline                     = nullptr;
    st->mul_rows_function                     = nullptr;
    st->gelu_mul_rows_pipeline                = nullptr;
    st->gelu_mul_rows_function                = nullptr;
    st->add_rows_pipeline                     = nullptr;
    st->add_rows_function                     = nullptr;
    st->scale_rows_pipeline                   = nullptr;
    st->scale_rows_function                   = nullptr;
    st->rmsnorm_add_rows_pipeline             = nullptr;
    st->rmsnorm_add_rows_function             = nullptr;
    st->rmsnorm_add_rows_simd_pipeline        = nullptr;
    st->rmsnorm_add_rows_simd_function        = nullptr;
    st->embed_lookup_scaled_pipeline          = nullptr;
    st->embed_lookup_scaled_function          = nullptr;
    st->embed_lookup_scaled_rows_pipeline     = nullptr;
    st->embed_lookup_scaled_rows_function     = nullptr;
    st->f32_matmul_pipeline                   = nullptr;
    st->f32_matmul_function                   = nullptr;
    st->f32_matmul_sg_pipeline                = nullptr;
    st->f32_matmul_sg_function                = nullptr;
    st->f32_matmul_mm_pipeline                = nullptr;
    st->f32_matmul_mm_function                = nullptr;
    st->f32_ple_gate_pipeline                 = nullptr;
    st->f32_ple_gate_function                 = nullptr;
    st->f32_ple_proj_norm_pipeline            = nullptr;
    st->f32_ple_proj_norm_function            = nullptr;
    st->argmax_pipeline                       = nullptr;
    st->argmax_function                       = nullptr;
    st->argmax_batch_pipeline                 = nullptr;
    st->argmax_batch_function                 = nullptr;
    st->argmax_result_buffer                  = nullptr;
    st->dn_scratch                            = nullptr;
    st->dn_scratch_bytes                      = 0;
    st->argmax_result_mapped                  = nullptr;
    st->argmax_result_capacity                = 0;
    st->rope_rows_pipeline                    = nullptr;
    st->rope_rows_function                    = nullptr;
    st->kv_append_rows_pipeline               = nullptr;
    st->kv_append_rows_function               = nullptr;
    st->copy_u32_pipeline                     = nullptr;
    st->copy_u32_function                     = nullptr;
    st->kv_append_rows_f16_pipeline           = nullptr;
    st->kv_append_rows_f16_function           = nullptr;
    st->q_norm_rope_rows_pipeline             = nullptr;
    st->q_norm_rope_rows_function             = nullptr;
    st->k_norm_rope_append_rows_pipeline      = nullptr;
    st->k_norm_rope_append_rows_function      = nullptr;
    st->k_norm_rope_append_rows_f16_pipeline  = nullptr;
    st->k_norm_rope_append_rows_f16_function  = nullptr;
    st->v_norm_append_rows_pipeline           = nullptr;
    st->v_norm_append_rows_function           = nullptr;
    st->v_norm_append_rows_f16_pipeline       = nullptr;
    st->v_norm_append_rows_f16_function       = nullptr;
    st->kv_norm_append_rows_pipeline          = nullptr;
    st->kv_norm_append_rows_function          = nullptr;
    st->kv_norm_append_rows_f16_pipeline      = nullptr;
    st->kv_norm_append_rows_f16_function      = nullptr;
    st->attention_rows_pipeline               = nullptr;
    st->attention_rows_function               = nullptr;
    st->attention_rows_f16_pipeline           = nullptr;
    st->attention_rows_f16_function           = nullptr;
    st->attention_qnorm_dec_f16_pipeline      = nullptr;
    st->attention_qnorm_dec_f16_function      = nullptr;
    st->attention_dec_combine_pipeline        = nullptr;
    st->attention_dec_combine_function        = nullptr;
    st->attention_dec512_f16_pipeline         = nullptr;
    st->attention_dec512_f16_function         = nullptr;
    st->attn_dec512_f16_library               = nullptr;
    st->attention_flash_sg8_f16_pipeline      = nullptr;
    st->attention_flash_sg8_f16_function      = nullptr;
    st->attn_flash_sg8_f16_library            = nullptr;
    st->attention_qnorm_flash_sg_f16_pipeline = nullptr;
    st->attention_qnorm_flash_sg_f16_function = nullptr;
    st->attn_library                          = nullptr;
    st->attn_f16_library                      = nullptr;
    st->attn_qnorm_dec_f16_library            = nullptr;
    st->attn_dec_combine_library              = nullptr;
    st->attn_flash_sg_f16_library             = nullptr;
    st->kv_norm_append_library                = nullptr;
    st->kv_norm_append_f16_library            = nullptr;
    st->q_norm_rope_library                   = nullptr;
    st->k_norm_rope_append_library            = nullptr;
    st->v_norm_append_library                 = nullptr;
    st->elem_library                          = nullptr;
    st->elem_simd_library                     = nullptr;
    st->embed_library                         = nullptr;
    st->f32_library                           = nullptr;
    st->q4k_gate_up_n4_library                = nullptr;
    st->q4k_pair_n4_library                   = nullptr;
    st->q4k_gate_up_library                   = nullptr;
    st->q4k_n4_library                        = nullptr;
    st->q4k_m16_n2_library                    = nullptr;
    st->q6k_n4_library                        = nullptr;
    st->q6k_library                           = nullptr;
    st->q4k_library                           = nullptr;
    st->q4k_qk_library                        = nullptr;
    st->command_queue                         = nullptr;
    st->device                                = nullptr;
    if (st->metal_handle != nullptr) {
        dlclose(st->metal_handle);
        st->metal_handle = nullptr;
    }
    if (st->objc_handle != nullptr) {
        dlclose(st->objc_handle);
        st->objc_handle = nullptr;
    }
    geist_backend_free(be, st);
}

void metal_destroy(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return;
    }
    metal_destroy_state(be, be->state);
    be->state = nullptr;
}

[[nodiscard]] static enum geist_status metal_load_runtime(struct geist_backend *be,
                                                          struct metal_state   *st) {

#if defined(__APPLE__)
    static const char *const metal_paths[] = {
            "/System/Library/Frameworks/Metal.framework/Metal",
            "Metal.framework/Metal",
            nullptr,
    };
    static const char *const objc_paths[] = {
            "/usr/lib/libobjc.A.dylib",
            "libobjc.A.dylib",
            nullptr,
    };

    for (size_t i = 0; metal_paths[i] != nullptr; i++) {
        st->metal_handle = dlopen(metal_paths[i], RTLD_NOW | RTLD_LOCAL);
        if (st->metal_handle != nullptr) {
            break;
        }
    }
    if (st->metal_handle == nullptr) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "metal: Metal framework is unavailable");
        return GEIST_E_UNSUPPORTED;
    }

    for (size_t i = 0; objc_paths[i] != nullptr; i++) {
        st->objc_handle = dlopen(objc_paths[i], RTLD_NOW | RTLD_LOCAL);
        if (st->objc_handle != nullptr) {
            break;
        }
    }
    if (st->objc_handle == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "metal: Objective-C runtime is unavailable");
        return GEIST_E_UNSUPPORTED;
    }

    st->MTLCreateSystemDefaultDevice =
            metal_dlsym(st->metal_handle, "MTLCreateSystemDefaultDevice");
    st->objc_msgSend     = metal_dlsym(st->objc_handle, "objc_msgSend");
    st->sel_registerName = metal_dlsym(st->objc_handle, "sel_registerName");
    st->objc_getClass    = metal_dlsym(st->objc_handle, "objc_getClass");
    if (st->MTLCreateSystemDefaultDevice == nullptr || st->objc_msgSend == nullptr ||
        st->sel_registerName == nullptr || st->objc_getClass == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: runtime symbols are incomplete");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
#else
    (void) st;
    geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED, "metal: backend is only available on Apple platforms");
    return GEIST_E_UNSUPPORTED;
#endif
}

[[nodiscard]] enum geist_status metal_create(struct geist_backend            *be,
                                             const struct geist_backend_opts *opts) {

    (void) opts;
    struct metal_state *st = geist_backend_alloc(be, sizeof(*st), alignof(struct metal_state));
    if (st == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_OOM, "metal: failed to allocate %zu-byte state", sizeof(*st));
        return GEIST_E_OOM;
    }
    *st                    = (struct metal_state) {0};
    st->backend            = be;
    const char *ple_block  = getenv("GEIST_METAL_PLE_BLOCK");
    st->use_ple_block      = ple_block == nullptr || strcmp(ple_block, "0") != 0;
    const char *q4k_n4     = getenv("GEIST_METAL_Q4K_N4");
    st->use_q4k_n4         = q4k_n4 == nullptr || strcmp(q4k_n4, "0") != 0;
    const char *q4k_m16_n2 = getenv("GEIST_METAL_Q4K_M16_N2");
    st->use_q4k_m16_n2     = q4k_m16_n2 != nullptr && strcmp(q4k_m16_n2, "1") == 0;
    /* Simdgroup-matmul Q4_K GEMM (llama.cpp mul_mm-derived). Default ON: only
     * runs for full tiles (dispatch guard requires rows%32==0 && n_out%64==0),
     * so the partial-tile path never executes and non-conforming shapes fall
     * back to the m16 kernel. Numerical parity for both paths is covered by
     * tests/test_backend_metal_q4k_matmul_parity.c. Set
     * GEIST_METAL_Q4K_MM_SG=0 to disable. */
    const char *q4k_mm_sg    = getenv("GEIST_METAL_Q4K_MM_SG");
    st->use_q4k_mm_sg        = q4k_mm_sg == nullptr || strcmp(q4k_mm_sg, "0") != 0;
    const char *rmsnorm_simd = getenv("GEIST_METAL_RMSNORM_SIMD");
    st->use_rmsnorm_simd     = rmsnorm_simd == nullptr || strcmp(rmsnorm_simd, "0") != 0;
    const char *q6k_n4       = getenv("GEIST_METAL_Q6K_N4");
    st->use_q6k_n4           = q6k_n4 == nullptr || strcmp(q6k_n4, "0") != 0;
    /* Off by default: the plain-layout n4 kernel (llama mul_mv structure)
     * outruns the packed nt4 path and needs no load-time repack. */
    /* Command-buffer pipelining (llama n_cb-style): rotate every N
     * dispatches, default 192 (~3 buffers per decode token — llama's
     * measured optimum on M-series is 2-3 buffers per graph).
     * GEIST_METAL_PIPELINE=0 disables, =N sets the rotation period. */
    const char *pipeline_env = getenv("GEIST_METAL_PIPELINE");
    st->seq_rotate_every     = pipeline_env != nullptr ? (uint32_t) atoi(pipeline_env) : 192u;
    st->profile_enabled      = metal_env_enabled("GEIST_METAL_PROFILE");
    /* Chunked DeltaNet prefill (CPU dn_run_prefill_chunked port). Default
     * ON for seq>1; GEIST_METAL_DN_CHUNK=0 falls back to the serial
     * per-token mixer kernel. */
    const char *dn_chunk = getenv("GEIST_METAL_DN_CHUNK");
    st->use_dn_chunk     = dn_chunk == nullptr || strcmp(dn_chunk, "0") != 0;

    enum geist_status s = metal_load_runtime(be, st);
    if (s != GEIST_OK) {
        metal_destroy_state(be, st);
        return s;
    }

    st->device = metal_create_default_device(st);
    if (st->device == nullptr) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "metal: no default Metal device");
        metal_destroy_state(be, st);
        return GEIST_E_UNSUPPORTED;
    }
    metal_msg_send_void0(st, st->device, "retain");

    st->command_queue = metal_msg_send_id0(st, st->device, "newCommandQueue");
    if (st->command_queue == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: failed to create command queue");
        metal_destroy_state(be, st);
        return GEIST_E_BACKEND;
    }

    void       *name = metal_msg_send_id0(st, st->device, "name");
    const char *utf8 = name != nullptr ? metal_msg_send_cstr0(st, name, "UTF8String") : nullptr;
    snprintf(st->device_name,
             sizeof(st->device_name),
             "%s",
             utf8 != nullptr && utf8[0] != '\0' ? utf8 : "Apple Metal GPU");

    be->state = st;
    return GEIST_OK;
}

/* Batched-submit region hooks. main brackets each prefill batch and each
 * decode step with these; we open one command buffer per region and encode
 * every op onto it. Host access to GPU-referenced buffers flushes early
 * (see metal_flush_if_referenced). The engine treats the token as opaque;
 * flushes rotate st->sequence_token, so region_end closes the CURRENT
 * sequence, not the original token. */
int metal_parallel_region_begin(struct geist_backend *be, enum geist_parallel_region region) {
    if (be == nullptr || be->state == nullptr) {
        return 0;
    }
    struct metal_state *st = be->state;
    if (st->sequence_active) {
        return 0; /* nested region: leave the outer batch in charge */
    }
    const enum geist_command_sequence_kind kind =
            region == GEIST_REGION_PREFILL_BATCH ? GEIST_COMMAND_SEQUENCE_PREFILL_TEXT
                                                 : GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP;
    int tok = 0;
    if (metal_command_sequence_begin(be, kind, &tok) != GEIST_OK) {
        return 0;
    }
    metal_seq_ref_clear(st);
    return tok;
}

void metal_parallel_region_end(struct geist_backend *be, int token) {
    if (be == nullptr || be->state == nullptr || token == 0) {
        return;
    }
    struct metal_state *st = be->state;
    if (!st->sequence_active) {
        return;
    }
    (void) metal_command_sequence_end(be, st->sequence_token, true);
    metal_seq_ref_clear(st);
}
