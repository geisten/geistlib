/*
 * src/backends/vulkan/lifecycle.c — instance/device lifecycle and loader plumbing.
 *
 * Layer: BACKEND (vulkan). Split from the former monolithic backend.c;
 * pure moves, no behavior change.
 */
#include "vk_internal.h"

/* ====================================================================== */
/* Create / destroy                                                        */
/* ====================================================================== */

static PFN_vkVoidFunction vk_iproc(struct vk_state *st, const char *name) {
    return st->fn.GetInstanceProcAddr(st->instance, name);
}

[[nodiscard]] static enum geist_status vk_load_runtime(struct geist_backend *be,
                                                       struct vk_state      *st) {
    static const char *const sonames[] = {
            "libvulkan.so.1", "libvulkan.so", "libvulkan.1.dylib", nullptr};
    for (size_t i = 0; sonames[i] != nullptr && st->lib == nullptr; ++i) {
        st->lib = dlopen(sonames[i], RTLD_NOW | RTLD_LOCAL);
    }
    if (st->lib == nullptr) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "vulkan: no libvulkan (%s)", dlerror());
        return GEIST_E_UNSUPPORTED;
    }
    union { /* object→function pointer cast, -Wpedantic-clean (as in metal) */
        void                     *obj;
        PFN_vkGetInstanceProcAddr fn;
    } gipa;
    gipa.obj                   = dlsym(st->lib, "vkGetInstanceProcAddr");
    st->fn.GetInstanceProcAddr = gipa.fn;
    if (st->fn.GetInstanceProcAddr == nullptr) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "vulkan: no vkGetInstanceProcAddr");
        return GEIST_E_UNSUPPORTED;
    }
    st->fn.CreateInstance =
            (PFN_vkCreateInstance) st->fn.GetInstanceProcAddr(nullptr, "vkCreateInstance");
    if (st->fn.CreateInstance == nullptr) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "vulkan: no vkCreateInstance");
        return GEIST_E_UNSUPPORTED;
    }
    return GEIST_OK;
}

/* Resolve one proc or fail create() loudly — every entry point in vk_fns is
 * mandatory core Vulkan, so a miss means a broken loader, not a feature gap. */
#define VK_LOAD_I(st, name)                                                           \
    do {                                                                              \
        (st)->fn.name = (PFN_vk##name) vk_iproc((st), "vk" #name);                    \
        if ((st)->fn.name == nullptr) {                                               \
            geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: missing vk" #name); \
            return GEIST_E_BACKEND;                                                   \
        }                                                                             \
    } while (0)

[[nodiscard]] static enum geist_status vk_load_instance_fns(struct geist_backend *be,
                                                            struct vk_state      *st) {
    VK_LOAD_I(st, DestroyInstance);
    VK_LOAD_I(st, EnumeratePhysicalDevices);
    VK_LOAD_I(st, GetPhysicalDeviceProperties2);
    VK_LOAD_I(st, GetPhysicalDeviceQueueFamilyProperties);
    VK_LOAD_I(st, GetPhysicalDeviceMemoryProperties);
    VK_LOAD_I(st, GetPhysicalDeviceFeatures2);
    VK_LOAD_I(st, EnumerateDeviceExtensionProperties);
    VK_LOAD_I(st, CreateDevice);
    VK_LOAD_I(st, GetDeviceProcAddr);
    /* Device-level entry points fetched through the instance proc addr work
     * on every ICD (they go through the loader trampoline). Good enough for
     * the scaffold; switch to GetDeviceProcAddr if dispatch overhead ever
     * shows in a profile. */
    VK_LOAD_I(st, DestroyDevice);
    VK_LOAD_I(st, GetDeviceQueue);
    VK_LOAD_I(st, CreateBuffer);
    VK_LOAD_I(st, DestroyBuffer);
    VK_LOAD_I(st, GetBufferMemoryRequirements);
    VK_LOAD_I(st, AllocateMemory);
    VK_LOAD_I(st, FreeMemory);
    VK_LOAD_I(st, BindBufferMemory);
    VK_LOAD_I(st, MapMemory);
    VK_LOAD_I(st, UnmapMemory);
    VK_LOAD_I(st, CreateCommandPool);
    VK_LOAD_I(st, DestroyCommandPool);
    VK_LOAD_I(st, AllocateCommandBuffers);
    VK_LOAD_I(st, BeginCommandBuffer);
    VK_LOAD_I(st, EndCommandBuffer);
    VK_LOAD_I(st, ResetCommandBuffer);
    VK_LOAD_I(st, CmdCopyBuffer);
    VK_LOAD_I(st, QueueSubmit);
    VK_LOAD_I(st, QueueWaitIdle);
    VK_LOAD_I(st, CreateFence);
    VK_LOAD_I(st, DestroyFence);
    VK_LOAD_I(st, ResetFences);
    VK_LOAD_I(st, WaitForFences);
    VK_LOAD_I(st, CreateShaderModule);
    VK_LOAD_I(st, DestroyShaderModule);
    VK_LOAD_I(st, CreateDescriptorSetLayout);
    VK_LOAD_I(st, DestroyDescriptorSetLayout);
    VK_LOAD_I(st, CreatePipelineLayout);
    VK_LOAD_I(st, DestroyPipelineLayout);
    VK_LOAD_I(st, CreateComputePipelines);
    VK_LOAD_I(st, DestroyPipeline);
    VK_LOAD_I(st, CreateDescriptorPool);
    VK_LOAD_I(st, DestroyDescriptorPool);
    VK_LOAD_I(st, AllocateDescriptorSets);
    VK_LOAD_I(st, UpdateDescriptorSets);
    VK_LOAD_I(st, CmdBindPipeline);
    VK_LOAD_I(st, CmdBindDescriptorSets);
    VK_LOAD_I(st, CmdPushConstants);
    VK_LOAD_I(st, CmdDispatch);
    VK_LOAD_I(st, CmdPipelineBarrier);
    VK_LOAD_I(st, ResetDescriptorPool);
    VK_LOAD_I(st, CreateQueryPool);
    VK_LOAD_I(st, DestroyQueryPool);
    VK_LOAD_I(st, CmdResetQueryPool);
    VK_LOAD_I(st, CmdWriteTimestamp);
    VK_LOAD_I(st, GetQueryPoolResults);
    return GEIST_OK;
}

static void vk_destroy_state(struct geist_backend *be, struct vk_state *st) {
    if (st == nullptr) {
        return;
    }
    if (st->device != VK_NULL_HANDLE) {
        (void) st->fn.QueueWaitIdle(st->queue);
        for (size_t i = 0; i < st->n_weights; ++i) {
            vk_buffer_destroy(be, st->weights[i].gpu);
        }
        geist_backend_free(be, st->weights);
        if (st->x_stage != nullptr) {
            vk_buffer_destroy(be, st->x_stage);
        }
        if (st->y_stage != nullptr) {
            vk_buffer_destroy(be, st->y_stage);
        }
        if (st->argmax_out != nullptr) {
            vk_buffer_destroy(be, st->argmax_out);
        }
        if (st->xring != nullptr) {
            vk_buffer_destroy(be, st->xring);
        }
        geist_backend_free(be, st->hostbufs);
        for (int i = 0; i < VK_PIPE_COUNT; ++i) {
            if (st->pipes[i] != VK_NULL_HANDLE) {
                st->fn.DestroyPipeline(st->device, st->pipes[i], nullptr);
            }
        }
        if (st->seq_pool != VK_NULL_HANDLE) {
            st->fn.DestroyDescriptorPool(st->device, st->seq_pool, nullptr);
        }
        if (st->dset_cache_pool != VK_NULL_HANDLE) {
            st->fn.DestroyDescriptorPool(st->device, st->dset_cache_pool, nullptr);
        }
        for (int i = 0; i < 5; ++i) {
            if (st->seq_playouts[i] != VK_NULL_HANDLE) {
                st->fn.DestroyPipelineLayout(st->device, st->seq_playouts[i], nullptr);
            }
            if (st->seq_dlayouts[i] != VK_NULL_HANDLE) {
                st->fn.DestroyDescriptorSetLayout(st->device, st->seq_dlayouts[i], nullptr);
            }
        }
        if (st->ts_pool != VK_NULL_HANDLE) {
            st->fn.DestroyQueryPool(st->device, st->ts_pool, nullptr);
        }
        if (st->seq_fence != VK_NULL_HANDLE) {
            st->fn.DestroyFence(st->device, st->seq_fence, nullptr);
        }
        if (st->xfer_fence != VK_NULL_HANDLE) {
            st->fn.DestroyFence(st->device, st->xfer_fence, nullptr);
        }
        if (st->cmd_pool != VK_NULL_HANDLE) {
            st->fn.DestroyCommandPool(st->device, st->cmd_pool, nullptr);
        }
        st->fn.DestroyDevice(st->device, nullptr);
    }
    if (st->instance != VK_NULL_HANDLE) {
        st->fn.DestroyInstance(st->instance, nullptr);
    }
    if (st->lib != nullptr) {
        dlclose(st->lib);
    }
    geist_backend_free(be, st);
}

[[nodiscard]] static enum geist_status vk_pick_device(struct geist_backend *be,
                                                      struct vk_state      *st) {
    uint32_t count = 0;
    if (st->fn.EnumeratePhysicalDevices(st->instance, &count, nullptr) != VK_SUCCESS ||
        count == 0) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "vulkan: no physical devices");
        return GEIST_E_UNSUPPORTED;
    }
    VkPhysicalDevice devs[16];
    if (count > 16) {
        count = 16;
    }
    (void) st->fn.EnumeratePhysicalDevices(st->instance, &count, devs);

    const char *env    = getenv("GEIST_VK_DEVICE");
    int         wanted = env != nullptr ? atoi(env) : -1;
    int         pick   = -1;
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceProperties2 props = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        st->fn.GetPhysicalDeviceProperties2(devs[i], &props);
        if (wanted >= 0 ? (int) i == wanted
                        : (pick < 0 &&
                           props.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) {
            pick = (int) i;
            snprintf(st->device_name, sizeof(st->device_name), "%s", props.properties.deviceName);
        }
    }
    if (pick < 0 && wanted < 0) {
        /* No discrete GPU — take device 0 (integrated GPU or llvmpipe is
         * still a working Vulkan device; useful for CI smoke). */
        pick                              = 0;
        VkPhysicalDeviceProperties2 props = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        st->fn.GetPhysicalDeviceProperties2(devs[0], &props);
        snprintf(st->device_name, sizeof(st->device_name), "%s", props.properties.deviceName);
    }
    if (pick < 0) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "vulkan: GEIST_VK_DEVICE=%d not found", wanted);
        return GEIST_E_UNSUPPORTED;
    }
    st->phys = devs[pick];
    st->fn.GetPhysicalDeviceMemoryProperties(st->phys, &st->mem_props);
    VkPhysicalDeviceSubgroupProperties sgp = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 pprops = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                          .pNext = &sgp};
    st->fn.GetPhysicalDeviceProperties2(st->phys, &pprops);
    st->ts_period_ns  = pprops.properties.limits.timestampPeriod;
    st->subgroup_size = sgp.subgroupSize;
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_create_device(struct geist_backend *be,
                                                        struct vk_state      *st) {
    /* Queue: first family with compute (graphics+compute queues are fine —
     * we submit compute + transfer only). */
    uint32_t nfam = 0;
    st->fn.GetPhysicalDeviceQueueFamilyProperties(st->phys, &nfam, nullptr);
    VkQueueFamilyProperties fams[32];
    if (nfam > 32) {
        nfam = 32;
    }
    st->fn.GetPhysicalDeviceQueueFamilyProperties(st->phys, &nfam, fams);
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < nfam; ++i) {
        if (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            family = i;
            break;
        }
    }
    if (family == UINT32_MAX) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "vulkan: no compute queue on %s", st->device_name);
        return GEIST_E_UNSUPPORTED;
    }
    st->queue_family = family;

    /* Probe the features the Phase-2 kernels want (fp16, int8 dot, coopmat),
     * then request exactly the supported subset. Base robustBufferAccess
     * stays off — it costs bandwidth and cpu_scalar is the safety net. */
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_have = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceVulkan13Features have13 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .pNext = &coop_have};
    VkPhysicalDeviceVulkan12Features have12 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &have13};
    VkPhysicalDeviceVulkan11Features have11 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &have12};
    VkPhysicalDeviceFeatures2 have2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                       .pNext = &have11};
    st->fn.GetPhysicalDeviceFeatures2(st->phys, &have2);

    bool coop_ext = false;
    {
        uint32_t next = 0;
        (void) st->fn.EnumerateDeviceExtensionProperties(st->phys, nullptr, &next, nullptr);
        VkExtensionProperties *exts = nullptr;
        if (next > 0) {
            exts = geist_backend_alloc(be, next * sizeof(*exts), alignof(VkExtensionProperties));
        }
        if (exts != nullptr) {
            (void) st->fn.EnumerateDeviceExtensionProperties(st->phys, nullptr, &next, exts);
            for (uint32_t i = 0; i < next; ++i) {
                if (strcmp(exts[i].extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
                    coop_ext = true;
                }
            }
            geist_backend_free(be, exts);
        }
    }

    st->has_fp16 = have12.shaderFloat16 && have11.storageBuffer16BitAccess;
    st->has_int8_dot =
            have13.shaderIntegerDotProduct && have12.shaderInt8 && have12.storageBuffer8BitAccess;
    st->has_coopmat = coop_ext && coop_have.cooperativeMatrix && st->has_fp16;

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_want = {
            .sType             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
            .cooperativeMatrix = st->has_coopmat};
    VkPhysicalDeviceVulkan13Features want13 = {
            .sType                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .pNext                   = st->has_coopmat ? &coop_want : nullptr,
            .shaderIntegerDotProduct = have13.shaderIntegerDotProduct,
            .synchronization2        = have13.synchronization2,
            .maintenance4            = have13.maintenance4};
    VkPhysicalDeviceVulkan12Features want12 = {
            .sType                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .pNext                   = &want13,
            .shaderFloat16           = have12.shaderFloat16,
            .shaderInt8              = have12.shaderInt8,
            .storageBuffer8BitAccess = have12.storageBuffer8BitAccess,
            .uniformAndStorageBuffer8BitAccess = have12.uniformAndStorageBuffer8BitAccess,
            .vulkanMemoryModel                 = have12.vulkanMemoryModel,
            .vulkanMemoryModelDeviceScope      = have12.vulkanMemoryModelDeviceScope,
            .bufferDeviceAddress               = have12.bufferDeviceAddress};
    VkPhysicalDeviceVulkan11Features want11 = {
            .sType                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            .pNext                    = &want12,
            .storageBuffer16BitAccess = have11.storageBuffer16BitAccess,
            .uniformAndStorageBuffer16BitAccess = have11.uniformAndStorageBuffer16BitAccess};
    VkPhysicalDeviceFeatures2 want2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                       .pNext = &want11};
    want2.features.shaderInt16      = have2.features.shaderInt16;

    const char *ext_names[1];
    uint32_t    n_ext = 0;
    if (st->has_coopmat) {
        ext_names[n_ext++] = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
    }

    const float             prio  = 1.0f;
    VkDeviceQueueCreateInfo qinfo = {.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                     .queueFamilyIndex = family,
                                     .queueCount       = 1,
                                     .pQueuePriorities = &prio};
    VkDeviceCreateInfo      dinfo = {.sType                 = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                     .pNext                 = &want2,
                                     .queueCreateInfoCount  = 1,
                                     .pQueueCreateInfos     = &qinfo,
                                     .enabledExtensionCount = n_ext,
                                     .ppEnabledExtensionNames = ext_names};
    VkResult                r     = st->fn.CreateDevice(st->phys, &dinfo, nullptr, &st->device);
    if (r != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: vkCreateDevice failed (%d)", (int) r);
        return GEIST_E_BACKEND;
    }
    st->fn.GetDeviceQueue(st->device, family, 0, &st->queue);

    VkCommandPoolCreateInfo pinfo = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                     .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                     .queueFamilyIndex = family};
    if (st->fn.CreateCommandPool(st->device, &pinfo, nullptr, &st->cmd_pool) != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: command pool creation failed");
        return GEIST_E_BACKEND;
    }
    VkCommandBufferAllocateInfo ainfo = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                         .commandPool        = st->cmd_pool,
                                         .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                         .commandBufferCount = 1};
    if (st->fn.AllocateCommandBuffers(st->device, &ainfo, &st->xfer_cmd) != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: command buffer alloc failed");
        return GEIST_E_BACKEND;
    }
    VkFenceCreateInfo finfo = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (st->fn.CreateFence(st->device, &finfo, nullptr, &st->xfer_fence) != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: fence creation failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] enum geist_status vk_create(struct geist_backend            *be,
                                          const struct geist_backend_opts *opts) {
    (void) opts;
    struct vk_state *st = geist_backend_alloc(be, sizeof(*st), alignof(struct vk_state));
    if (st == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "vulkan: failed to allocate state");
        return GEIST_E_OOM;
    }
    *st                 = (struct vk_state) {0};
    st->backend         = be;
    st->profile_enabled = getenv("GEIST_VK_PROFILE") != nullptr;
    const char *ops_env = getenv("GEIST_VK_GPU_OPS");
    st->gpu_ops = ops_env != nullptr ? (uint32_t) strtoul(ops_env, nullptr, 0) : 0xffffffffu;

    enum geist_status s = vk_load_runtime(be, st);
    if (s != GEIST_OK) {
        vk_destroy_state(be, st);
        return s;
    }

    VkApplicationInfo    app   = {.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                  .pApplicationName = "geist",
                                  .apiVersion       = VK_API_VERSION_1_3};
    VkInstanceCreateInfo iinfo = {.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                  .pApplicationInfo = &app};
    if (st->fn.CreateInstance(&iinfo, nullptr, &st->instance) != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "vulkan: vkCreateInstance failed");
        vk_destroy_state(be, st);
        return GEIST_E_UNSUPPORTED;
    }
    s = vk_load_instance_fns(be, st);
    if (s == GEIST_OK) {
        s = vk_pick_device(be, st);
    }
    if (s == GEIST_OK) {
        s = vk_create_device(be, st);
    }
    if (s == GEIST_OK) {
        s = vk_create_pipelines(be, st);
    }
    if (s != GEIST_OK) {
        vk_destroy_state(be, st);
        return s;
    }

    if (getenv("GEIST_VK_VERBOSE") != nullptr) {
        fprintf(stderr,
                "geist vulkan: %s (fp16 %d, int8-dot %d, coopmat %d)\n",
                st->device_name,
                st->has_fp16,
                st->has_int8_dot,
                st->has_coopmat);
    }
    be->state = st;
    return GEIST_OK;
}

void vk_destroy(struct geist_backend *be) {
    if (be->state != nullptr) {
        struct vk_state *st = be->state;
        if (getenv("GEIST_VK_VERBOSE") != nullptr || st->profile_enabled) {
            fprintf(stderr,
                    "geist vulkan stats: dispatches %llu, flushes %llu, host-op accesses "
                    "%llu, barriers %llu (elided %llu)\n",
                    (unsigned long long) st->stat_dispatches,
                    (unsigned long long) st->stat_flushes,
                    (unsigned long long) st->stat_cpu_falls,
                    (unsigned long long) st->stat_barriers,
                    (unsigned long long) st->stat_barriers_elided);
            fprintf(stderr,
                    "geist vulkan dset cache: %llu hits, %llu misses; submit+wait %.1f ms\n",
                    (unsigned long long) st->stat_dset_hits,
                    (unsigned long long) st->stat_dset_miss,
                    (double) st->stat_wait_ns / 1e6);
        }
        if (st->profile_enabled) {
            static const char *const names[VK_PIPE_COUNT + 1] = {
                    "matvec_q4k", "matmul_q4k", "matvec_q6k",  "matmul_q6k",  "matvec_f32",
                    "matmul_f32", "add",        "mul",         "gelu",        "gelu_mul",
                    "scale",      "rmsnorm",    "rmsnorm_add", "rope",        "attention",
                    "argmax",     "embed",      "ffn_gate_up", "qkv_prep",    "mm_q4k_cm",
                    "mm_q6k_cm",  "attn_f16",   "qkv_f16",     "kv_app_f16",  "attn_part",
                    "attn_comb",  "mm_cm32",    "ple_gate",    "ffn_norm_gu", "copy"};
            fprintf(stderr, "geist vulkan gpu profile:\n");
            for (int i = 0; i <= VK_PIPE_COUNT; ++i) {
                if (st->prof_calls[i] > 0) {
                    fprintf(stderr,
                            "  %-12s %8.1f ms  %8llu calls  %6.1f us/call\n",
                            names[i],
                            (double) st->prof_ns[i] / 1e6,
                            (unsigned long long) st->prof_calls[i],
                            (double) st->prof_ns[i] / 1e3 / (double) st->prof_calls[i]);
                }
            }
        }
        vk_destroy_state(be, be->state);
        be->state = nullptr;
    }
}
