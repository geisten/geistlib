/*
 * src/backends/vulkan/backend.c — Vulkan compute backend (discrete GPU).
 *
 * Phase 1 (scaffold): instance/device bring-up, buffer ops, feature probes.
 * Structured after src/backends/metal/backend.c, adapted for discrete-GPU
 * memory: no unified memory, so
 *
 *   - WEIGHT buffers created via buffer_create are DEVICE_LOCAL VRAM,
 *     filled through a staging copy in buffer_upload;
 *   - ACTIVATION / KV / SCRATCH / STAGING / IO buffers are HOST_VISIBLE|
 *     HOST_COHERENT so the arch layer's buffer_map contract holds;
 *   - buffer_create_aliased returns a bookkeeping handle around the host
 *     pointer (arch scratch pool, weight arena / GGUF mmap). It owns no
 *     Vulkan resources; the GPU-side copy of aliased weights is made once
 *     at resolve_weight time (Phase 2), keyed by the host pointer.
 *
 * libvulkan is loaded at runtime via dlopen — no link-time dependency,
 * matching the Metal backend's dlopen(Metal.framework) approach. Build
 * needs only the Vulkan headers (VK_NO_PROTOTYPES).
 *
 * Device policy: prefer the first DISCRETE_GPU, override with
 * GEIST_VK_DEVICE=<index>. Missing device or loader fails create() with
 * GEIST_E_UNSUPPORTED so "auto" and tests can fall back / skip.
 */
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <geist.h>
#include <geist_backend.h>
#include <geist_types.h>
#include <geist_weight.h>

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ====================================================================== */
/* Runtime loader                                                          */
/* ====================================================================== */

struct vk_fns {
    PFN_vkGetInstanceProcAddr                    GetInstanceProcAddr;
    /* global */
    PFN_vkCreateInstance                         CreateInstance;
    /* instance */
    PFN_vkDestroyInstance                        DestroyInstance;
    PFN_vkEnumeratePhysicalDevices               EnumeratePhysicalDevices;
    PFN_vkGetPhysicalDeviceProperties2           GetPhysicalDeviceProperties2;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties      GetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceFeatures2             GetPhysicalDeviceFeatures2;
    PFN_vkEnumerateDeviceExtensionProperties     EnumerateDeviceExtensionProperties;
    PFN_vkCreateDevice                           CreateDevice;
    PFN_vkGetDeviceProcAddr                      GetDeviceProcAddr;
    /* device */
    PFN_vkDestroyDevice          DestroyDevice;
    PFN_vkGetDeviceQueue         GetDeviceQueue;
    PFN_vkCreateBuffer           CreateBuffer;
    PFN_vkDestroyBuffer          DestroyBuffer;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
    PFN_vkAllocateMemory         AllocateMemory;
    PFN_vkFreeMemory             FreeMemory;
    PFN_vkBindBufferMemory       BindBufferMemory;
    PFN_vkMapMemory              MapMemory;
    PFN_vkUnmapMemory            UnmapMemory;
    PFN_vkCreateCommandPool      CreateCommandPool;
    PFN_vkDestroyCommandPool     DestroyCommandPool;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
    PFN_vkBeginCommandBuffer     BeginCommandBuffer;
    PFN_vkEndCommandBuffer       EndCommandBuffer;
    PFN_vkResetCommandBuffer     ResetCommandBuffer;
    PFN_vkCmdCopyBuffer          CmdCopyBuffer;
    PFN_vkQueueSubmit            QueueSubmit;
    PFN_vkQueueWaitIdle          QueueWaitIdle;
    PFN_vkCreateFence            CreateFence;
    PFN_vkDestroyFence           DestroyFence;
    PFN_vkResetFences            ResetFences;
    PFN_vkWaitForFences          WaitForFences;
};

struct vk_state {
    struct geist_backend *backend;
    void                 *lib; /* dlopen handle, may be nullptr after create */
    struct vk_fns         fn;

    VkInstance       instance;
    VkPhysicalDevice phys;
    VkDevice         device;
    VkQueue          queue;
    uint32_t         queue_family;

    VkPhysicalDeviceMemoryProperties mem_props;
    VkCommandPool                    cmd_pool;
    VkCommandBuffer                  xfer_cmd;
    VkFence                          xfer_fence;

    char device_name[256];

    /* Feature probes for the Phase-2 kernels. */
    bool has_fp16;        /* shaderFloat16 + 16-bit storage */
    bool has_int8_dot;    /* shaderIntegerDotProduct + 8-bit storage */
    bool has_coopmat;     /* VK_KHR_cooperative_matrix */
};

struct geist_buffer {
    struct vk_state       *owner;
    VkBuffer               buf;   /* VK_NULL_HANDLE for aliased handles */
    VkDeviceMemory         mem;
    void                  *mapped;     /* persistent map, host-visible only */
    void                  *host_alias; /* aliased mode: external host bytes */
    size_t                 bytes;
    enum geist_buffer_role role;
    unsigned int           memory_flags;
    bool                   host_visible;
};

/* ====================================================================== */
/* Create / destroy                                                        */
/* ====================================================================== */

static PFN_vkVoidFunction vk_iproc(struct vk_state *st, const char *name) {
    return st->fn.GetInstanceProcAddr(st->instance, name);
}

[[nodiscard]] static enum geist_status vk_load_runtime(struct geist_backend *be,
                                                       struct vk_state      *st) {
    static const char *const sonames[] = {"libvulkan.so.1", "libvulkan.so",
                                          "libvulkan.1.dylib", nullptr};
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
    st->fn.CreateInstance = (PFN_vkCreateInstance) st->fn.GetInstanceProcAddr(
            nullptr, "vkCreateInstance");
    if (st->fn.CreateInstance == nullptr) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "vulkan: no vkCreateInstance");
        return GEIST_E_UNSUPPORTED;
    }
    return GEIST_OK;
}

/* Resolve one proc or fail create() loudly — every entry point in vk_fns is
 * mandatory core Vulkan, so a miss means a broken loader, not a feature gap. */
#define VK_LOAD_I(st, name)                                                              \
    do {                                                                                 \
        (st)->fn.name = (PFN_vk##name) vk_iproc((st), "vk" #name);                       \
        if ((st)->fn.name == nullptr) {                                                  \
            geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: missing vk" #name);    \
            return GEIST_E_BACKEND;                                                      \
        }                                                                                \
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
    return GEIST_OK;
}

static void vk_destroy_state(struct geist_backend *be, struct vk_state *st) {
    if (st == nullptr) {
        return;
    }
    if (st->device != VK_NULL_HANDLE) {
        (void) st->fn.QueueWaitIdle(st->queue);
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
                        : (pick < 0 && props.properties.deviceType ==
                                               VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) {
            pick = (int) i;
            snprintf(st->device_name, sizeof(st->device_name), "%s",
                     props.properties.deviceName);
        }
    }
    if (pick < 0 && wanted < 0) {
        /* No discrete GPU — take device 0 (integrated GPU or llvmpipe is
         * still a working Vulkan device; useful for CI smoke). */
        pick = 0;
        VkPhysicalDeviceProperties2 props = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        st->fn.GetPhysicalDeviceProperties2(devs[0], &props);
        snprintf(st->device_name, sizeof(st->device_name), "%s", props.properties.deviceName);
    }
    if (pick < 0) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "vulkan: GEIST_VK_DEVICE=%d not found",
                                wanted);
        return GEIST_E_UNSUPPORTED;
    }
    st->phys = devs[pick];
    st->fn.GetPhysicalDeviceMemoryProperties(st->phys, &st->mem_props);
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
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "vulkan: no compute queue on %s",
                                st->device_name);
        return GEIST_E_UNSUPPORTED;
    }
    st->queue_family = family;

    /* Probe the features the Phase-2 kernels want (fp16, int8 dot, coopmat),
     * then request exactly the supported subset. Base robustBufferAccess
     * stays off — it costs bandwidth and cpu_scalar is the safety net. */
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_have = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceVulkan13Features have13 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .pNext = &coop_have};
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
                if (strcmp(exts[i].extensionName,
                           VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
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
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
            .cooperativeMatrix = st->has_coopmat};
    VkPhysicalDeviceVulkan13Features want13 = {
            .sType                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .pNext                    = st->has_coopmat ? &coop_want : nullptr,
            .shaderIntegerDotProduct  = have13.shaderIntegerDotProduct,
            .synchronization2         = have13.synchronization2,
            .maintenance4             = have13.maintenance4};
    VkPhysicalDeviceVulkan12Features want12 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .pNext = &want13,
            .shaderFloat16 = have12.shaderFloat16,
            .shaderInt8    = have12.shaderInt8,
            .storageBuffer8BitAccess           = have12.storageBuffer8BitAccess,
            .uniformAndStorageBuffer8BitAccess = have12.uniformAndStorageBuffer8BitAccess,
            .vulkanMemoryModel            = have12.vulkanMemoryModel,
            .vulkanMemoryModelDeviceScope = have12.vulkanMemoryModelDeviceScope,
            .bufferDeviceAddress          = have12.bufferDeviceAddress};
    VkPhysicalDeviceVulkan11Features want11 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            .pNext = &want12,
            .storageBuffer16BitAccess           = have11.storageBuffer16BitAccess,
            .uniformAndStorageBuffer16BitAccess = have11.uniformAndStorageBuffer16BitAccess};
    VkPhysicalDeviceFeatures2 want2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                       .pNext = &want11};
    want2.features.shaderInt16 = have2.features.shaderInt16;

    const char *ext_names[1];
    uint32_t    n_ext = 0;
    if (st->has_coopmat) {
        ext_names[n_ext++] = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
    }

    const float              prio  = 1.0f;
    VkDeviceQueueCreateInfo  qinfo = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                      .queueFamilyIndex = family,
                                      .queueCount       = 1,
                                      .pQueuePriorities = &prio};
    VkDeviceCreateInfo       dinfo = {.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                      .pNext                = &want2,
                                      .queueCreateInfoCount = 1,
                                      .pQueueCreateInfos    = &qinfo,
                                      .enabledExtensionCount   = n_ext,
                                      .ppEnabledExtensionNames = ext_names};
    VkResult r = st->fn.CreateDevice(st->phys, &dinfo, nullptr, &st->device);
    if (r != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: vkCreateDevice failed (%d)",
                                (int) r);
        return GEIST_E_BACKEND;
    }
    st->fn.GetDeviceQueue(st->device, family, 0, &st->queue);

    VkCommandPoolCreateInfo pinfo = {
            .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = family};
    if (st->fn.CreateCommandPool(st->device, &pinfo, nullptr, &st->cmd_pool) != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: command pool creation failed");
        return GEIST_E_BACKEND;
    }
    VkCommandBufferAllocateInfo ainfo = {
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
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

[[nodiscard]] static enum geist_status vk_create(struct geist_backend            *be,
                                                 const struct geist_backend_opts *opts) {
    (void) opts;
    struct vk_state *st = geist_backend_alloc(be, sizeof(*st), alignof(struct vk_state));
    if (st == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "vulkan: failed to allocate state");
        return GEIST_E_OOM;
    }
    *st         = (struct vk_state) {0};
    st->backend = be;

    enum geist_status s = vk_load_runtime(be, st);
    if (s != GEIST_OK) {
        vk_destroy_state(be, st);
        return s;
    }

    VkApplicationInfo app = {.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .pApplicationName = "geist",
                             .apiVersion       = VK_API_VERSION_1_3};
    VkInstanceCreateInfo iinfo = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
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
    if (s != GEIST_OK) {
        vk_destroy_state(be, st);
        return s;
    }

    if (getenv("GEIST_VK_VERBOSE") != nullptr) {
        fprintf(stderr, "geist vulkan: %s (fp16 %d, int8-dot %d, coopmat %d)\n",
                st->device_name, st->has_fp16, st->has_int8_dot, st->has_coopmat);
    }
    be->state = st;
    return GEIST_OK;
}

static void vk_destroy(struct geist_backend *be) {
    if (be->state != nullptr) {
        vk_destroy_state(be, be->state);
        be->state = nullptr;
    }
}

/* ====================================================================== */
/* Buffers                                                                 */
/* ====================================================================== */

[[nodiscard]] static uint32_t vk_find_mem_type(const struct vk_state *st,
                                               uint32_t               type_bits,
                                               VkMemoryPropertyFlags  want) {
    for (uint32_t i = 0; i < st->mem_props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (st->mem_props.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return UINT32_MAX;
}

[[nodiscard]] static enum geist_status vk_buffer_create(struct geist_backend  *be,
                                                        size_t                 bytes,
                                                        enum geist_buffer_role role,
                                                        unsigned int           memory_flags,
                                                        struct geist_buffer  **out) {
    struct vk_state *st = be->state;
    if (bytes == 0 || out == nullptr) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan: bad buffer_create args");
        return GEIST_E_INVALID_ARG;
    }
    /* WEIGHT defaults to VRAM; everything else must satisfy the arch
     * layer's buffer_map contract, so it lives host-visible. Explicit
     * memory_flags override the role default. */
    bool device_local = role == GEIST_BUFFER_WEIGHT;
    if (memory_flags & (GEIST_MEMORY_HOST | GEIST_MEMORY_HOST_VISIBLE | GEIST_MEMORY_MAPPED)) {
        device_local = false;
    } else if (memory_flags & GEIST_MEMORY_DEVICE) {
        device_local = true;
    }

    struct geist_buffer *buf =
            geist_backend_alloc(be, sizeof(*buf), alignof(struct geist_buffer));
    if (buf == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "vulkan: buffer handle alloc failed");
        return GEIST_E_OOM;
    }
    *buf = (struct geist_buffer) {.owner        = st,
                                  .bytes        = bytes,
                                  .role         = role,
                                  .memory_flags = memory_flags,
                                  .host_visible = !device_local};

    VkBufferCreateInfo binfo = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .size  = bytes,
                                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    if (st->fn.CreateBuffer(st->device, &binfo, nullptr, &buf->buf) != VK_SUCCESS) {
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: vkCreateBuffer(%zu) failed",
                                bytes);
        return GEIST_E_BACKEND;
    }
    VkMemoryRequirements req;
    st->fn.GetBufferMemoryRequirements(st->device, buf->buf, &req);
    const VkMemoryPropertyFlags want =
            device_local ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                         : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t mem_type = vk_find_mem_type(st, req.memoryTypeBits, want);
    if (mem_type == UINT32_MAX && device_local) {
        /* VRAM exhausted or odd heap layout — host-visible still works. */
        mem_type          = vk_find_mem_type(st, req.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        buf->host_visible = mem_type != UINT32_MAX;
    }
    if (mem_type == UINT32_MAX) {
        st->fn.DestroyBuffer(st->device, buf->buf, nullptr);
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: no memory type for buffer");
        return GEIST_E_BACKEND;
    }
    VkMemoryAllocateInfo minfo = {.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                  .allocationSize  = req.size,
                                  .memoryTypeIndex = mem_type};
    VkResult             r     = st->fn.AllocateMemory(st->device, &minfo, nullptr, &buf->mem);
    if (r == VK_SUCCESS) {
        r = st->fn.BindBufferMemory(st->device, buf->buf, buf->mem, 0);
    }
    if (r == VK_SUCCESS && buf->host_visible) {
        r = st->fn.MapMemory(st->device, buf->mem, 0, VK_WHOLE_SIZE, 0, &buf->mapped);
    }
    if (r != VK_SUCCESS) {
        if (buf->mem != VK_NULL_HANDLE) {
            st->fn.FreeMemory(st->device, buf->mem, nullptr);
        }
        st->fn.DestroyBuffer(st->device, buf->buf, nullptr);
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_OOM, "vulkan: allocating %zu bytes failed (%d)",
                                bytes, (int) r);
        return GEIST_E_OOM;
    }
    *out = buf;
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_buffer_create_aliased(struct geist_backend  *be,
                                                                void                  *host_ptr,
                                                                size_t                 n_bytes,
                                                                enum geist_buffer_role role,
                                                                struct geist_buffer  **out) {
    /* Bookkeeping handle only: records the host region (scratch-pool slice,
     * weight arena, GGUF mmap). No Vulkan resources — device copies of
     * aliased weights are made at resolve_weight time (Phase 2), keyed by
     * this pointer. buffer_map returns the pointer unchanged, so the arch
     * layer's CPU-side paths keep working. */
    struct vk_state *st = be->state;
    if (host_ptr == nullptr || out == nullptr) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan: bad aliased-buffer args");
        return GEIST_E_INVALID_ARG;
    }
    struct geist_buffer *buf =
            geist_backend_alloc(be, sizeof(*buf), alignof(struct geist_buffer));
    if (buf == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "vulkan: aliased handle alloc failed");
        return GEIST_E_OOM;
    }
    *buf = (struct geist_buffer) {.owner        = st,
                                  .host_alias   = host_ptr,
                                  .bytes        = n_bytes,
                                  .role         = role,
                                  .memory_flags = GEIST_MEMORY_ALIASED,
                                  .host_visible = true};
    *out = buf;
    return GEIST_OK;
}

static void vk_buffer_destroy(struct geist_backend *be, struct geist_buffer *buf) {
    if (buf == nullptr) {
        return;
    }
    struct vk_state *st = buf->owner;
    if (buf->buf != VK_NULL_HANDLE) {
        if (buf->mapped != nullptr) {
            st->fn.UnmapMemory(st->device, buf->mem);
        }
        st->fn.DestroyBuffer(st->device, buf->buf, nullptr);
        st->fn.FreeMemory(st->device, buf->mem, nullptr);
    }
    geist_backend_free(be, buf);
}

/* One blocking staging round-trip. Direction: upload (src != nullptr) or
 * download (dst != nullptr). ponytail: allocates a fresh staging buffer per
 * call — fine for load-time weight uploads; a persistent ring lands with the
 * Phase-2/3 hot path if transfers ever show up in a profile. */
[[nodiscard]] static enum geist_status vk_staged_copy(struct geist_buffer *buf,
                                                      size_t               n_bytes,
                                                      const uint8_t       *src,
                                                      uint8_t             *dst) {
    struct vk_state      *st = buf->owner;
    struct geist_backend *be = st->backend;

    VkBufferCreateInfo binfo = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .size  = n_bytes,
                                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer       staging     = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    void          *mapped      = nullptr;
    enum geist_status status   = GEIST_E_BACKEND;

    if (st->fn.CreateBuffer(st->device, &binfo, nullptr, &staging) != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: staging buffer failed");
        return GEIST_E_BACKEND;
    }
    VkMemoryRequirements req;
    st->fn.GetBufferMemoryRequirements(st->device, staging, &req);
    uint32_t mem_type = vk_find_mem_type(st, req.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo minfo = {.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                  .allocationSize  = req.size,
                                  .memoryTypeIndex = mem_type};
    if (mem_type == UINT32_MAX ||
        st->fn.AllocateMemory(st->device, &minfo, nullptr, &staging_mem) != VK_SUCCESS ||
        st->fn.BindBufferMemory(st->device, staging, staging_mem, 0) != VK_SUCCESS ||
        st->fn.MapMemory(st->device, staging_mem, 0, VK_WHOLE_SIZE, 0, &mapped) !=
                VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_OOM, "vulkan: staging alloc/map failed (%zu B)",
                                n_bytes);
        status = GEIST_E_OOM;
        goto out;
    }
    if (src != nullptr) {
        memcpy(mapped, src, n_bytes);
    }

    VkCommandBufferBeginInfo begin = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VkBufferCopy region = {.size = n_bytes};
    if (st->fn.BeginCommandBuffer(st->xfer_cmd, &begin) != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: begin transfer cmd failed");
        goto out;
    }
    if (src != nullptr) {
        st->fn.CmdCopyBuffer(st->xfer_cmd, staging, buf->buf, 1, &region);
    } else {
        st->fn.CmdCopyBuffer(st->xfer_cmd, buf->buf, staging, 1, &region);
    }
    if (st->fn.EndCommandBuffer(st->xfer_cmd) != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: end transfer cmd failed");
        goto out;
    }
    VkSubmitInfo submit = {.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                           .commandBufferCount = 1,
                           .pCommandBuffers    = &st->xfer_cmd};
    if (st->fn.QueueSubmit(st->queue, 1, &submit, st->xfer_fence) != VK_SUCCESS ||
        st->fn.WaitForFences(st->device, 1, &st->xfer_fence, VK_TRUE, UINT64_MAX) !=
                VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: transfer submit/wait failed");
        goto out;
    }
    (void) st->fn.ResetFences(st->device, 1, &st->xfer_fence);
    if (dst != nullptr) {
        memcpy(dst, mapped, n_bytes);
    }
    status = GEIST_OK;

out:
    if (mapped != nullptr) {
        st->fn.UnmapMemory(st->device, staging_mem);
    }
    if (staging_mem != VK_NULL_HANDLE) {
        st->fn.FreeMemory(st->device, staging_mem, nullptr);
    }
    st->fn.DestroyBuffer(st->device, staging, nullptr);
    return status;
}

[[nodiscard]] static enum geist_status
vk_buffer_upload(struct geist_buffer *buf, size_t n_bytes, const uint8_t *src) {
    if (buf == nullptr || n_bytes > buf->bytes) {
        return GEIST_E_INVALID_ARG;
    }
    if (buf->host_alias != nullptr) {
        memcpy(buf->host_alias, src, n_bytes);
        return GEIST_OK;
    }
    if (buf->mapped != nullptr) {
        memcpy(buf->mapped, src, n_bytes);
        return GEIST_OK;
    }
    return vk_staged_copy(buf, n_bytes, src, nullptr);
}

[[nodiscard]] static enum geist_status
vk_buffer_download(size_t n_bytes, uint8_t *dst, const struct geist_buffer *buf) {
    if (buf == nullptr || n_bytes > buf->bytes) {
        return GEIST_E_INVALID_ARG;
    }
    if (buf->host_alias != nullptr) {
        memcpy(dst, buf->host_alias, n_bytes);
        return GEIST_OK;
    }
    if (buf->mapped != nullptr) {
        memcpy(dst, buf->mapped, n_bytes);
        return GEIST_OK;
    }
    return vk_staged_copy((struct geist_buffer *) buf, n_bytes, nullptr, dst);
}

static void *vk_buffer_map(struct geist_buffer *buf) {
    if (buf == nullptr) {
        return nullptr;
    }
    if (buf->host_alias != nullptr) {
        return buf->host_alias;
    }
    return buf->mapped; /* nullptr for device-local — caller must fall back */
}

static void vk_buffer_unmap(struct geist_buffer *buf) {
    (void) buf; /* persistent coherent mappings — nothing to flush */
}

/* ====================================================================== */
/* Capability                                                              */
/* ====================================================================== */

static enum geist_support vk_supports_op(struct geist_backend                *be,
                                         const struct geist_op_support_query *query) {
    (void) be;
    (void) query;
    /* Phase 1: no compute ops yet. Phase 2 answers GEIST_OP_LINEAR for
     * Q4_K/Q6_K like the Metal backend. */
    return GEIST_SUPPORT_NONE;
}

/* ====================================================================== */
/* Descriptor                                                              */
/* ====================================================================== */

static const struct geist_backend_vtbl vk_vtbl = {
        .create                = vk_create,
        .destroy               = vk_destroy,
        .supports_op           = vk_supports_op,
        .buffer_create         = vk_buffer_create,
        .buffer_destroy        = vk_buffer_destroy,
        .buffer_create_aliased = vk_buffer_create_aliased,
        .buffer_upload         = vk_buffer_upload,
        .buffer_download       = vk_buffer_download,
        .buffer_map            = vk_buffer_map,
        .buffer_unmap          = vk_buffer_unmap,
        /* resolve_weight + level-2/3 ops land in Phase 2. */
};

const struct geist_backend_descriptor geist_backend_vulkan = {
        .name = "vulkan",
        .vtbl = &vk_vtbl,
};
