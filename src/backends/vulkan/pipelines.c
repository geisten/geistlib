/*
 * src/backends/vulkan/pipelines.c — compute-pipeline creation.
 *
 * Layer: BACKEND (vulkan). Split from the former monolithic backend.c;
 * pure moves, no behavior change.
 */
#include "vk_internal.h"

/* ====================================================================== */
/* Compute pipelines                                                       */
/* ====================================================================== */

[[nodiscard]] static enum geist_status vk_make_pipeline(struct geist_backend *be,
                                                        struct vk_state      *st,
                                                        const uint32_t       *code,
                                                        size_t                code_bytes,
                                                        VkPipelineLayout      layout,
                                                        VkPipeline           *out) {
    VkShaderModuleCreateInfo minfo = {.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = code_bytes,
                                      .pCode    = code};
    VkShaderModule           mod   = VK_NULL_HANDLE;
    if (st->fn.CreateShaderModule(st->device, &minfo, nullptr, &mod) != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: shader module creation failed");
        return GEIST_E_BACKEND;
    }
    VkComputePipelineCreateInfo pinfo = {
            .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage  = {.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                       .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
                       .module = mod,
                       .pName  = "main"},
            .layout = layout};
    VkResult r = st->fn.CreateComputePipelines(st->device, VK_NULL_HANDLE, 1, &pinfo, nullptr, out);
    st->fn.DestroyShaderModule(st->device, mod, nullptr);
    if (r != VK_SUCCESS) {
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "vulkan: compute pipeline failed (%d)", (int) r);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] enum geist_status vk_create_pipelines(struct geist_backend *be, struct vk_state *st) {
    /* Set/pipeline layouts for 2, 3 and 4 storage-buffer bindings; every
     * shader declares a push block within the shared 128-byte range. */
    for (uint32_t n = 2; n <= 6; ++n) {
        VkDescriptorSetLayoutBinding bindings[6];
        for (uint32_t i = 0; i < n; ++i) {
            bindings[i] = (VkDescriptorSetLayoutBinding) {
                    .binding         = i,
                    .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT};
        }
        VkDescriptorSetLayoutCreateInfo linfo = {
                .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = n,
                .pBindings    = bindings};
        if (st->fn.CreateDescriptorSetLayout(
                    st->device, &linfo, nullptr, &st->seq_dlayouts[n - 2]) != VK_SUCCESS) {
            geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: descriptor layout failed");
            return GEIST_E_BACKEND;
        }
        VkPushConstantRange        push   = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                             .size       = VK_PUSH_RANGE};
        VkPipelineLayoutCreateInfo plinfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                             .setLayoutCount         = 1,
                                             .pSetLayouts            = &st->seq_dlayouts[n - 2],
                                             .pushConstantRangeCount = 1,
                                             .pPushConstantRanges    = &push};
        if (st->fn.CreatePipelineLayout(st->device, &plinfo, nullptr, &st->seq_playouts[n - 2]) !=
            VK_SUCCESS) {
            geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: pipeline layout failed");
            return GEIST_E_BACKEND;
        }
    }
    VkDescriptorPoolSize       psize  = {.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                         .descriptorCount = VK_SEQ_MAX_SETS * 6};
    VkDescriptorPoolCreateInfo dpinfo = {.sType   = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                         .maxSets = VK_SEQ_MAX_SETS,
                                         .poolSizeCount = 1,
                                         .pPoolSizes    = &psize};
    if (st->fn.CreateDescriptorPool(st->device, &dpinfo, nullptr, &st->seq_pool) != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: descriptor pool failed");
        return GEIST_E_BACKEND;
    }
    VkDescriptorPoolSize       csize  = {.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                         .descriptorCount = VK_DSET_CACHE * 6};
    VkDescriptorPoolCreateInfo dcinfo = {.sType   = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                         .maxSets = VK_DSET_CACHE,
                                         .poolSizeCount = 1,
                                         .pPoolSizes    = &csize};
    if (st->fn.CreateDescriptorPool(st->device, &dcinfo, nullptr, &st->dset_cache_pool) !=
        VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: dset cache pool failed");
        return GEIST_E_BACKEND;
    }
    VkCommandBufferAllocateInfo cainfo = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                          .commandPool        = st->cmd_pool,
                                          .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                          .commandBufferCount = VK_SEQ_CMDBUFS};
    if (st->fn.AllocateCommandBuffers(st->device, &cainfo, st->seq_cmds) != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: seq command buffers failed");
        return GEIST_E_BACKEND;
    }
    VkFenceCreateInfo finfo = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (st->fn.CreateFence(st->device, &finfo, nullptr, &st->seq_fence) != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: seq fence failed");
        return GEIST_E_BACKEND;
    }
    if (st->profile_enabled) {
        VkQueryPoolCreateInfo qinfo = {.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                       .queryType  = VK_QUERY_TYPE_TIMESTAMP,
                                       .queryCount = VK_SEQ_MAX_DISPATCH + 8};
        if (st->fn.CreateQueryPool(st->device, &qinfo, nullptr, &st->ts_pool) != VK_SUCCESS) {
            st->profile_enabled = false; /* profiling is best-effort */
        }
    }

    static const struct {
        const uint32_t *code;
        size_t          bytes;
    } blobs[VK_PIPE_COUNT] = {
            [VK_PIPE_MATVEC_Q4K]    = {matvec_q4k_spv, sizeof(matvec_q4k_spv)},
            [VK_PIPE_MATMUL_Q4K]    = {matmul_q4k_spv, sizeof(matmul_q4k_spv)},
            [VK_PIPE_MATVEC_Q6K]    = {matvec_q6k_spv, sizeof(matvec_q6k_spv)},
            [VK_PIPE_MATMUL_Q6K]    = {matmul_q6k_spv, sizeof(matmul_q6k_spv)},
            [VK_PIPE_MATVEC_F32]    = {matvec_f32_spv, sizeof(matvec_f32_spv)},
            [VK_PIPE_MATMUL_F32]    = {matmul_f32_spv, sizeof(matmul_f32_spv)},
            [VK_PIPE_ADD]           = {add_f32_spv, sizeof(add_f32_spv)},
            [VK_PIPE_MUL]           = {mul_f32_spv, sizeof(mul_f32_spv)},
            [VK_PIPE_GELU]          = {gelu_tanh_f32_spv, sizeof(gelu_tanh_f32_spv)},
            [VK_PIPE_GELU_MUL]      = {gelu_tanh_mul_f32_spv, sizeof(gelu_tanh_mul_f32_spv)},
            [VK_PIPE_SCALE]         = {scale_f32_spv, sizeof(scale_f32_spv)},
            [VK_PIPE_RMSNORM]       = {rmsnorm_f32_spv, sizeof(rmsnorm_f32_spv)},
            [VK_PIPE_RMSNORM_ADD]   = {rmsnorm_add_f32_spv, sizeof(rmsnorm_add_f32_spv)},
            [VK_PIPE_ROPE]          = {rope_f32_spv, sizeof(rope_f32_spv)},
            [VK_PIPE_ATTENTION]     = {attention_f32_spv, sizeof(attention_f32_spv)},
            [VK_PIPE_ARGMAX]        = {argmax_f32_spv, sizeof(argmax_f32_spv)},
            [VK_PIPE_EMBED]         = {embed_lookup_scaled_spv, sizeof(embed_lookup_scaled_spv)},
            [VK_PIPE_FFN_GATE_UP]   = {ffn_gate_up_gelu_q4k_spv, sizeof(ffn_gate_up_gelu_q4k_spv)},
            [VK_PIPE_QKV_PREP]      = {qkv_prep_f32_spv, sizeof(qkv_prep_f32_spv)},
            [VK_PIPE_MM_Q4K_CM]     = {matmul_q4k_cm_spv, sizeof(matmul_q4k_cm_spv)},
            [VK_PIPE_MM_Q6K_CM]     = {matmul_q6k_cm_spv, sizeof(matmul_q6k_cm_spv)},
            [VK_PIPE_ATTENTION_F16] = {attention_f16_spv, sizeof(attention_f16_spv)},
            [VK_PIPE_QKV_PREP_F16]  = {qkv_prep_f16_spv, sizeof(qkv_prep_f16_spv)},
            [VK_PIPE_KV_APPEND_F16] = {kv_append_f16_spv, sizeof(kv_append_f16_spv)},
            [VK_PIPE_ATTN_PART_F16] = {attn_part_f16_spv, sizeof(attn_part_f16_spv)},
            [VK_PIPE_ATTN_COMB]     = {attn_comb_spv, sizeof(attn_comb_spv)},
            [VK_PIPE_MM_Q4K_CM32]   = {matmul_q4k_cm32_spv, sizeof(matmul_q4k_cm32_spv)},
            [VK_PIPE_PLE_GATE]      = {ple_gate_f32_spv, sizeof(ple_gate_f32_spv)},
            [VK_PIPE_FFN_NORM_GU]   = {ffn_norm_gate_up_q4k_spv, sizeof(ffn_norm_gate_up_q4k_spv)},
    };
    for (int i = 0; i < VK_PIPE_COUNT; ++i) {
        if ((i == VK_PIPE_MM_Q4K_CM || i == VK_PIPE_MM_Q6K_CM || i == VK_PIPE_MM_Q4K_CM32) &&
            !st->has_coopmat) {
            continue; /* stays VK_NULL_HANDLE; linear_t falls back */
        }
        enum geist_status s = vk_make_pipeline(be,
                                               st,
                                               blobs[i].code,
                                               blobs[i].bytes,
                                               st->seq_playouts[vk_pipe_nbind[i] - 2],
                                               &st->pipes[i]);
        if (s != GEIST_OK) {
            if (i == VK_PIPE_MM_Q4K_CM || i == VK_PIPE_MM_Q6K_CM || i == VK_PIPE_MM_Q4K_CM32) {
                fprintf(stderr,
                        "geist vulkan: coopmat pipeline unavailable — using the "
                        "register-tiled GEMM\n");
                st->pipes[i] = VK_NULL_HANDLE;
                continue;
            }
            return s;
        }
    }
    return GEIST_OK;
}
