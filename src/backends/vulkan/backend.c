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

#include "gemma4_kernels.h" /* shared reference rope/attention kernels */
#include "heap.h"
#include "quant.h" /* CPU dequant helpers for the non-GPU dtype fallback */

#include <dlfcn.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Committed SPIR-V blobs — regenerate with `make vulkan-shaders`. */
#include "shaders/add_f32_spv.h"
#include "shaders/argmax_f32_spv.h"
#include "shaders/attention_f32_spv.h"
#include "shaders/embed_lookup_scaled_spv.h"
#include "shaders/ffn_gate_up_gelu_q4k_spv.h"
#include "shaders/attention_f16_spv.h"
#include "shaders/attn_comb_spv.h"
#include "shaders/attn_part_f16_spv.h"
#include "shaders/kv_append_f16_spv.h"
#include "shaders/matmul_q4k_cm32_spv.h"
#include "shaders/matmul_q4k_cm_spv.h"
#include "shaders/matmul_q6k_cm_spv.h"
#include "shaders/qkv_prep_f16_spv.h"
#include "shaders/qkv_prep_f32_spv.h"
#include "shaders/gelu_tanh_f32_spv.h"
#include "shaders/gelu_tanh_mul_f32_spv.h"
#include "shaders/matmul_f32_spv.h"
#include "shaders/matmul_q4k_spv.h"
#include "shaders/matmul_q6k_spv.h"
#include "shaders/matvec_f32_spv.h"
#include "shaders/ffn_norm_gate_up_q4k_spv.h"
#include "shaders/ple_gate_f32_spv.h"
#include "shaders/matvec_q4k_spv.h"
#include "shaders/matvec_q6k_spv.h"
#include "shaders/mul_f32_spv.h"
#include "shaders/rmsnorm_add_f32_spv.h"
#include "shaders/rmsnorm_f32_spv.h"
#include "shaders/rope_f32_spv.h"
#include "shaders/scale_f32_spv.h"

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
    /* compute pipeline machinery */
    PFN_vkCreateShaderModule        CreateShaderModule;
    PFN_vkDestroyShaderModule       DestroyShaderModule;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout;
    PFN_vkCreatePipelineLayout      CreatePipelineLayout;
    PFN_vkDestroyPipelineLayout     DestroyPipelineLayout;
    PFN_vkCreateComputePipelines    CreateComputePipelines;
    PFN_vkDestroyPipeline           DestroyPipeline;
    PFN_vkCreateDescriptorPool      CreateDescriptorPool;
    PFN_vkDestroyDescriptorPool     DestroyDescriptorPool;
    PFN_vkAllocateDescriptorSets    AllocateDescriptorSets;
    PFN_vkUpdateDescriptorSets      UpdateDescriptorSets;
    PFN_vkCmdBindPipeline           CmdBindPipeline;
    PFN_vkCmdBindDescriptorSets     CmdBindDescriptorSets;
    PFN_vkCmdPushConstants          CmdPushConstants;
    PFN_vkCmdDispatch               CmdDispatch;
    PFN_vkCmdPipelineBarrier        CmdPipelineBarrier;
    PFN_vkResetDescriptorPool       ResetDescriptorPool;
    PFN_vkCreateQueryPool           CreateQueryPool;
    PFN_vkDestroyQueryPool          DestroyQueryPool;
    PFN_vkCmdResetQueryPool         CmdResetQueryPool;
    PFN_vkCmdWriteTimestamp         CmdWriteTimestamp;
    PFN_vkGetQueryPoolResults       GetQueryPoolResults;
};

/* One compute pipeline per (op, dtype) pair; all share a single
 * 3-storage-buffer descriptor layout and the unified 9-u32 push block. */
enum vk_pipe {
    VK_PIPE_MATVEC_Q4K,
    VK_PIPE_MATMUL_Q4K,
    VK_PIPE_MATVEC_Q6K,
    VK_PIPE_MATMUL_Q6K,
    VK_PIPE_MATVEC_F32,
    VK_PIPE_MATMUL_F32,
    VK_PIPE_ADD,
    VK_PIPE_MUL,
    VK_PIPE_GELU,
    VK_PIPE_GELU_MUL,
    VK_PIPE_SCALE,
    VK_PIPE_RMSNORM,
    VK_PIPE_RMSNORM_ADD,
    VK_PIPE_ROPE,
    VK_PIPE_ATTENTION,
    VK_PIPE_ARGMAX,
    VK_PIPE_EMBED,
    VK_PIPE_FFN_GATE_UP,
    VK_PIPE_QKV_PREP,
    VK_PIPE_MM_Q4K_CM, /* tensor-core GEMMs; created only with coopmat */
    VK_PIPE_MM_Q6K_CM,
    VK_PIPE_ATTENTION_F16,
    VK_PIPE_QKV_PREP_F16,
    VK_PIPE_KV_APPEND_F16,
    VK_PIPE_ATTN_PART_F16,
    VK_PIPE_ATTN_COMB,
    VK_PIPE_MM_Q4K_CM32, /* small-n_out tensor-core tile */
    VK_PIPE_PLE_GATE,    /* fused PLE gate: gelu(x.gate_w) * ple_in */
    VK_PIPE_FFN_NORM_GU, /* ffn_gate_up with the pre-FFN rmsnorm folded in */
    VK_PIPE_COUNT,
};

struct vk_push {
    uint32_t n_in, n_out, blocks_per_row, rows;
    uint32_t x_offset, w_offset, y_offset, x_stride, y_stride;
};

/* resolve_weight-time VRAM copy of one aliased weight, keyed by the exact
 * host pointer the engine put into w->raw. */
struct vk_weight_entry {
    const void          *host;
    struct geist_buffer *gpu;
};

/* Per-binding access range for hazard tracking (byte offsets within the
 * bound VkBuffer). Ops that can't describe a binding precisely use
 * lo=0, hi=UINT64_MAX (whole buffer). */
struct vk_access {
    uint64_t lo;
    uint64_t hi;
    bool     write;
};

/* One tracked range since the last barrier. */
struct vk_dirty {
    VkBuffer buf;
    uint64_t lo;
    uint64_t hi;
    bool     write;
};

enum {
    VK_XRING_CAP    = 192u << 20, /* a full prefill chunk stages ~124 MB */
    VK_DIRTY_CAP    = 96,
    VK_DSET_CACHE   = 4096,
    VK_SEQ_CMDBUFS  = 64, /* rolling submission ring */
    VK_SEQ_ROTATE   = 64, /* dispatches per submit — keeps the GPU fed */
};

/* Cached descriptor set: decode re-binds the same (pipeline-layout,
 * buffers) tuple every token — building sets once kills the dominant
 * CPU cost of the encode loop (alloc + update per dispatch). */
struct vk_dset_entry {
    uint64_t        key; /* hash of nbind + buffer handles; 0 = empty */
    VkDescriptorSet set;
};

enum {
    VK_SEQ_MAX_SETS      = 4096, /* descriptor sets per flush window */
    VK_SEQ_MAX_DISPATCH  = 4000, /* rotate the sequence before pool runs dry */
    VK_PUSH_RANGE        = 128,  /* one push range covers every shader block */
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

    /* GEIST_VK_GPU_OPS bitmask (debug bisect): 1=linear_t 2=elementwise
     * 4=rmsnorm 8=rope 16=attention 32=copy 64=embed 128=argmax.
     * Default: all on. */
    uint32_t gpu_ops;

    /* GEIST_VK_VERBOSE stats. */
    uint64_t stat_flushes;
    uint64_t stat_dispatches;
    uint64_t stat_cpu_falls;

    /* GEIST_VK_PROFILE=1: GPU timestamps per dispatch, attributed by
     * pipeline (copies land in the extra slot). Execution is serialized by
     * the per-dispatch barriers, so consecutive deltas are exact. */
    bool         profile_enabled;
    float        ts_period_ns;
    VkQueryPool  ts_pool;
    uint32_t     ts_count;
    uint8_t      ts_pipe[VK_SEQ_MAX_DISPATCH + 8];
    uint64_t     prof_ns[VK_PIPE_COUNT + 1];
    uint64_t     prof_calls[VK_PIPE_COUNT + 1];

    /* Compute pipelines (Phase 2). */
    VkDescriptorSetLayout dset_layout;
    VkPipelineLayout      pipe_layout;
    VkDescriptorPool      dset_pool;
    VkDescriptorSet       dset;
    VkPipeline            pipes[VK_PIPE_COUNT];

    /* Weight registry: host pointer → VRAM buffer, filled by resolve_weight.
     * Linear search — a model has a few hundred weights; the lookup is one
     * pointer compare per entry once per linear call. */
    struct vk_weight_entry *weights;
    size_t                  n_weights;
    size_t                  cap_weights;

    /* Persistent host-visible activation staging (x up / y down) for the
     * synchronous host-pointer linear kernels (parity tests, CPU-dtype
     * fallbacks). The hot path uses linear_t + the sequence below. */
    struct geist_buffer *x_stage;
    struct geist_buffer *y_stage;

    /* Device-local x ring: linear_t copies activations into VRAM before
     * each matvec/matmul so the n_out (× m) workgroups hit L2/VRAM instead
     * of re-reading x from host memory over PCIe — the difference between
     * ~130 MB and ~8 KB of bus traffic per FFN matvec. Bump-allocated,
     * reset at every flush (the copies belong to the in-flight batch). */
    struct geist_buffer *xring;
    size_t               xring_used;

    /* ---- Sequence (Phase 3): ONE open command buffer per token/chunk. ----
     * GPU ops append dispatches (global memory barrier between each);
     * flush = submit + fence-wait, triggered by any host data access
     * (buffer_map / CPU-op fallback / argmax readback). Descriptor sets
     * are allocated per dispatch from seq_pool and bulk-freed at flush. */
    VkCommandBuffer       seq_cmds[VK_SEQ_CMDBUFS];
    VkCommandBuffer       seq_cmd;      /* currently recording */
    uint32_t              seq_cmd_idx;  /* next ring slot */
    uint32_t              seq_in_cmd;   /* dispatches in the open cmd buffer */
    VkFence               seq_fence;
    bool                  seq_open;
    uint32_t              seq_dispatches;
    VkDescriptorPool      seq_pool;
    VkDescriptorSetLayout seq_dlayouts[5]; /* index = binding count - 2 (2..6) */
    VkPipelineLayout      seq_playouts[5];

    /* Host-visible buffers created via buffer_create — containment lookup
     * so buffer_create_aliased can hand out GPU-bindable borrowed views
     * (the arch scratch pool / weight arena are such buffers since P3). */
    struct geist_buffer **hostbufs;
    size_t                n_hostbufs;
    size_t                cap_hostbufs;

    struct geist_buffer *argmax_out; /* 4-byte host-visible argmax result */

    VkDescriptorPool     dset_cache_pool; /* never reset; cache lives here */
    struct vk_dset_entry dset_cache[VK_DSET_CACHE];
    uint64_t             stat_dset_hits;
    uint64_t             stat_dset_miss;

    /* Hazard tracking: read/write ranges recorded since the last barrier.
     * A new dispatch inserts a barrier only when it conflicts (RAW / WAR /
     * WAW); independent dispatches overlap on the GPU. */
    struct vk_dirty dirty[VK_DIRTY_CAP];
    uint32_t        n_dirty;

    uint64_t stat_barriers;
    uint64_t stat_barriers_elided;
    uint64_t stat_wait_ns;
};

/* binding count per pipeline (descriptor set layout selector) */
static const uint32_t vk_pipe_nbind[VK_PIPE_COUNT] = {
        [VK_PIPE_MATVEC_Q4K] = 3, [VK_PIPE_MATMUL_Q4K] = 3, [VK_PIPE_MATVEC_Q6K] = 3,
        [VK_PIPE_MATMUL_Q6K] = 3, [VK_PIPE_MATVEC_F32] = 3, [VK_PIPE_MATMUL_F32] = 3,
        [VK_PIPE_ADD] = 3,        [VK_PIPE_MUL] = 3,        [VK_PIPE_GELU] = 2,
        [VK_PIPE_GELU_MUL] = 3,   [VK_PIPE_SCALE] = 2,      [VK_PIPE_RMSNORM] = 3,
        [VK_PIPE_RMSNORM_ADD] = 4, [VK_PIPE_ROPE] = 3,      [VK_PIPE_ATTENTION] = 4,
        [VK_PIPE_ARGMAX] = 2,     [VK_PIPE_EMBED] = 2,      [VK_PIPE_FFN_GATE_UP] = 4,
        [VK_PIPE_QKV_PREP] = 6, [VK_PIPE_MM_Q4K_CM] = 3, [VK_PIPE_MM_Q6K_CM] = 3,
        [VK_PIPE_ATTENTION_F16] = 4, [VK_PIPE_QKV_PREP_F16] = 6,
        [VK_PIPE_KV_APPEND_F16] = 4, [VK_PIPE_ATTN_PART_F16] = 4,
        [VK_PIPE_ATTN_COMB] = 2,     [VK_PIPE_MM_Q4K_CM32] = 3,
        [VK_PIPE_PLE_GATE] = 4,      [VK_PIPE_FFN_NORM_GU] = 5,
};

struct geist_buffer {
    struct vk_state       *owner;
    VkBuffer               buf;   /* VK_NULL_HANDLE for pure bookkeeping aliases */
    VkDeviceMemory         mem;
    void                  *mapped;     /* persistent map, host-visible only */
    void                  *host_alias; /* aliased mode: external host bytes */
    size_t                 bytes;
    size_t                 base_off; /* byte offset of logical start inside buf */
    enum geist_buffer_role role;
    unsigned int           memory_flags;
    bool                   host_visible;
    bool                   device_mem; /* memory type has DEVICE_LOCAL */
    bool                   borrowed;   /* buf/mem owned by a parent buffer */
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

static void vk_buffer_destroy(struct geist_backend *be, struct geist_buffer *buf);

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
    VkPhysicalDeviceProperties2 pprops = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    st->fn.GetPhysicalDeviceProperties2(st->phys, &pprops);
    st->ts_period_ns = pprops.properties.limits.timestampPeriod;
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

/* ====================================================================== */
/* Compute pipelines                                                       */
/* ====================================================================== */

[[nodiscard]] static enum geist_status vk_make_pipeline(struct geist_backend *be,
                                                        struct vk_state      *st,
                                                        const uint32_t       *code,
                                                        size_t                code_bytes,
                                                        VkPipelineLayout      layout,
                                                        VkPipeline           *out) {
    VkShaderModuleCreateInfo minfo = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = code_bytes,
                                      .pCode    = code};
    VkShaderModule           mod   = VK_NULL_HANDLE;
    if (st->fn.CreateShaderModule(st->device, &minfo, nullptr, &mod) != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: shader module creation failed");
        return GEIST_E_BACKEND;
    }
    VkComputePipelineCreateInfo pinfo = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                      .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
                      .module = mod,
                      .pName  = "main"},
            .layout = layout};
    VkResult r = st->fn.CreateComputePipelines(st->device, VK_NULL_HANDLE, 1, &pinfo, nullptr,
                                               out);
    st->fn.DestroyShaderModule(st->device, mod, nullptr);
    if (r != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: compute pipeline failed (%d)",
                                (int) r);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_create_pipelines(struct geist_backend *be,
                                                           struct vk_state      *st) {
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
        if (st->fn.CreateDescriptorSetLayout(st->device, &linfo, nullptr,
                                             &st->seq_dlayouts[n - 2]) != VK_SUCCESS) {
            geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: descriptor layout failed");
            return GEIST_E_BACKEND;
        }
        VkPushConstantRange        push   = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                             .size       = VK_PUSH_RANGE};
        VkPipelineLayoutCreateInfo plinfo = {
                .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount         = 1,
                .pSetLayouts            = &st->seq_dlayouts[n - 2],
                .pushConstantRangeCount = 1,
                .pPushConstantRanges    = &push};
        if (st->fn.CreatePipelineLayout(st->device, &plinfo, nullptr,
                                        &st->seq_playouts[n - 2]) != VK_SUCCESS) {
            geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: pipeline layout failed");
            return GEIST_E_BACKEND;
        }
    }
    VkDescriptorPoolSize       psize  = {.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                         .descriptorCount = VK_SEQ_MAX_SETS * 6};
    VkDescriptorPoolCreateInfo dpinfo = {
            .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets       = VK_SEQ_MAX_SETS,
            .poolSizeCount = 1,
            .pPoolSizes    = &psize};
    if (st->fn.CreateDescriptorPool(st->device, &dpinfo, nullptr, &st->seq_pool) !=
        VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: descriptor pool failed");
        return GEIST_E_BACKEND;
    }
    VkDescriptorPoolSize       csize  = {.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                         .descriptorCount = VK_DSET_CACHE * 6};
    VkDescriptorPoolCreateInfo dcinfo = {
            .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets       = VK_DSET_CACHE,
            .poolSizeCount = 1,
            .pPoolSizes    = &csize};
    if (st->fn.CreateDescriptorPool(st->device, &dcinfo, nullptr, &st->dset_cache_pool) !=
        VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "vulkan: dset cache pool failed");
        return GEIST_E_BACKEND;
    }
    VkCommandBufferAllocateInfo cainfo = {
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
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
        VkQueryPoolCreateInfo qinfo = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
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
            [VK_PIPE_MATVEC_Q4K]  = {matvec_q4k_spv, sizeof(matvec_q4k_spv)},
            [VK_PIPE_MATMUL_Q4K]  = {matmul_q4k_spv, sizeof(matmul_q4k_spv)},
            [VK_PIPE_MATVEC_Q6K]  = {matvec_q6k_spv, sizeof(matvec_q6k_spv)},
            [VK_PIPE_MATMUL_Q6K]  = {matmul_q6k_spv, sizeof(matmul_q6k_spv)},
            [VK_PIPE_MATVEC_F32]  = {matvec_f32_spv, sizeof(matvec_f32_spv)},
            [VK_PIPE_MATMUL_F32]  = {matmul_f32_spv, sizeof(matmul_f32_spv)},
            [VK_PIPE_ADD]         = {add_f32_spv, sizeof(add_f32_spv)},
            [VK_PIPE_MUL]         = {mul_f32_spv, sizeof(mul_f32_spv)},
            [VK_PIPE_GELU]        = {gelu_tanh_f32_spv, sizeof(gelu_tanh_f32_spv)},
            [VK_PIPE_GELU_MUL]    = {gelu_tanh_mul_f32_spv, sizeof(gelu_tanh_mul_f32_spv)},
            [VK_PIPE_SCALE]       = {scale_f32_spv, sizeof(scale_f32_spv)},
            [VK_PIPE_RMSNORM]     = {rmsnorm_f32_spv, sizeof(rmsnorm_f32_spv)},
            [VK_PIPE_RMSNORM_ADD] = {rmsnorm_add_f32_spv, sizeof(rmsnorm_add_f32_spv)},
            [VK_PIPE_ROPE]        = {rope_f32_spv, sizeof(rope_f32_spv)},
            [VK_PIPE_ATTENTION]   = {attention_f32_spv, sizeof(attention_f32_spv)},
            [VK_PIPE_ARGMAX]      = {argmax_f32_spv, sizeof(argmax_f32_spv)},
            [VK_PIPE_EMBED]       = {embed_lookup_scaled_spv, sizeof(embed_lookup_scaled_spv)},
            [VK_PIPE_FFN_GATE_UP] = {ffn_gate_up_gelu_q4k_spv,
                                     sizeof(ffn_gate_up_gelu_q4k_spv)},
            [VK_PIPE_QKV_PREP]    = {qkv_prep_f32_spv, sizeof(qkv_prep_f32_spv)},
            [VK_PIPE_MM_Q4K_CM]   = {matmul_q4k_cm_spv, sizeof(matmul_q4k_cm_spv)},
            [VK_PIPE_MM_Q6K_CM]   = {matmul_q6k_cm_spv, sizeof(matmul_q6k_cm_spv)},
            [VK_PIPE_ATTENTION_F16] = {attention_f16_spv, sizeof(attention_f16_spv)},
            [VK_PIPE_QKV_PREP_F16]  = {qkv_prep_f16_spv, sizeof(qkv_prep_f16_spv)},
            [VK_PIPE_KV_APPEND_F16] = {kv_append_f16_spv, sizeof(kv_append_f16_spv)},
            [VK_PIPE_ATTN_PART_F16] = {attn_part_f16_spv, sizeof(attn_part_f16_spv)},
            [VK_PIPE_ATTN_COMB]     = {attn_comb_spv, sizeof(attn_comb_spv)},
            [VK_PIPE_MM_Q4K_CM32]   = {matmul_q4k_cm32_spv, sizeof(matmul_q4k_cm32_spv)},
            [VK_PIPE_PLE_GATE]      = {ple_gate_f32_spv, sizeof(ple_gate_f32_spv)},
            [VK_PIPE_FFN_NORM_GU]   = {ffn_norm_gate_up_q4k_spv,
                                       sizeof(ffn_norm_gate_up_q4k_spv)},
    };
    for (int i = 0; i < VK_PIPE_COUNT; ++i) {
        if ((i == VK_PIPE_MM_Q4K_CM || i == VK_PIPE_MM_Q6K_CM ||
             i == VK_PIPE_MM_Q4K_CM32) && !st->has_coopmat) {
            continue; /* stays VK_NULL_HANDLE; linear_t falls back */
        }
        enum geist_status s = vk_make_pipeline(be, st, blobs[i].code, blobs[i].bytes,
                                               st->seq_playouts[vk_pipe_nbind[i] - 2],
                                               &st->pipes[i]);
        if (s != GEIST_OK) {
            if (i == VK_PIPE_MM_Q4K_CM || i == VK_PIPE_MM_Q6K_CM ||
                i == VK_PIPE_MM_Q4K_CM32) {
                fprintf(stderr, "geist vulkan: coopmat pipeline unavailable — using the "
                                "register-tiled GEMM\n");
                st->pipes[i] = VK_NULL_HANDLE;
                continue;
            }
            return s;
        }
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
    st->profile_enabled = getenv("GEIST_VK_PROFILE") != nullptr;
    const char *ops_env = getenv("GEIST_VK_GPU_OPS");
    st->gpu_ops         = ops_env != nullptr ? (uint32_t) strtoul(ops_env, nullptr, 0)
                                             : 0xffffffffu;

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
    if (s == GEIST_OK) {
        s = vk_create_pipelines(be, st);
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
                    "matvec_q4k", "matmul_q4k", "matvec_q6k", "matmul_q6k", "matvec_f32",
                    "matmul_f32", "add",        "mul",        "gelu",       "gelu_mul",
                    "scale",      "rmsnorm",    "rmsnorm_add", "rope",      "attention",
                    "argmax",     "embed",      "ffn_gate_up", "qkv_prep", "mm_q4k_cm", "mm_q6k_cm",
                    "attn_f16",   "qkv_f16",    "kv_app_f16", "attn_part", "attn_comb", "mm_cm32",
                    "ple_gate",   "ffn_norm_gu", "copy"};
            fprintf(stderr, "geist vulkan gpu profile:\n");
            for (int i = 0; i <= VK_PIPE_COUNT; ++i) {
                if (st->prof_calls[i] > 0) {
                    fprintf(stderr, "  %-12s %8.1f ms  %8llu calls  %6.1f us/call\n",
                            names[i], (double) st->prof_ns[i] / 1e6,
                            (unsigned long long) st->prof_calls[i],
                            (double) st->prof_ns[i] / 1e3 / (double) st->prof_calls[i]);
                }
            }
        }
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
    /* Everything the engine creates must satisfy the arch layer's
     * buffer_map contract (it maps WEIGHT-role cos/sin tables for CPU rope,
     * scratch pools, logits...), so default is host-visible. DEVICE_LOCAL
     * VRAM on explicit GEIST_MEMORY_DEVICE (resolve_weight copies, x ring)
     * and for KV_CACHE-role buffers — the arch touches dense KV only via
     * buffer_copy and v->attention, and decode attention re-reads the whole
     * cache every token (host-resident KV = hundreds of MB/token of PCIe). */
    const bool host_req = (memory_flags & (GEIST_MEMORY_HOST | GEIST_MEMORY_HOST_VISIBLE |
                                           GEIST_MEMORY_MAPPED)) != 0;
    const bool device_local =
            !host_req &&
            ((memory_flags & GEIST_MEMORY_DEVICE) != 0 || role == GEIST_BUFFER_KV_CACHE);

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
    uint32_t mem_type = UINT32_MAX;
    if (!device_local && bytes <= (200u << 20) && role != GEIST_BUFFER_STAGING &&
        role != GEIST_BUFFER_IO) {
        /* BAR helps buffers the GPU reads hot and the host rarely touches
         * (scratch pool, rope tables). Staging/IO stay in system RAM: the
         * host READS those, and CPU reads from BAR are uncached PCIe. */
        /* Small host-visible buffers (scratch pool, staging, tables) go
         * into the BAR window when available: DEVICE_LOCAL + HOST_VISIBLE
         * means GPU ops touch activations at VRAM speed while the arch's
         * buffer_map contract still holds. The 256 MB heap is precious —
         * big allocations (weight arena) stay in system RAM. */
        mem_type = vk_find_mem_type(st, req.memoryTypeBits,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    if (mem_type == UINT32_MAX) {
        mem_type = vk_find_mem_type(st, req.memoryTypeBits, want);
    }
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
    if (r != VK_SUCCESS && !device_local) {
        /* BAR heap exhausted (it is only 256 MB) — fall back to plain
         * host-visible system memory. */
        const uint32_t fb = vk_find_mem_type(st, req.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (fb != UINT32_MAX && fb != mem_type) {
            minfo.memoryTypeIndex = fb;
            mem_type              = fb;
            r = st->fn.AllocateMemory(st->device, &minfo, nullptr, &buf->mem);
        }
    }
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
    buf->device_mem =
            (st->mem_props.memoryTypes[mem_type].propertyFlags &
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    if (getenv("GEIST_VK_VERBOSE") != nullptr && buf->host_visible) {
        fprintf(stderr, "  buffer %zu KiB role=%d bar=%d\n", bytes >> 10, (int) role,
                buf->device_mem);
    }
    /* Register mapped buffers for the alias-containment lookup (arch pools
     * hand out slices of these; buffer_create_aliased resolves them back
     * to (VkBuffer, offset) so GPU ops can bind pool slices directly). */
    if (buf->mapped != nullptr) {
        if (st->n_hostbufs == st->cap_hostbufs) {
            const size_t          cap = st->cap_hostbufs == 0 ? 32 : st->cap_hostbufs * 2;
            struct geist_buffer **nb  = geist_backend_alloc(be, cap * sizeof(*nb),
                                                            alignof(struct geist_buffer *));
            if (nb != nullptr) {
                memcpy(nb, st->hostbufs, st->n_hostbufs * sizeof(*nb));
                geist_backend_free(be, st->hostbufs);
                st->hostbufs     = nb;
                st->cap_hostbufs = cap;
            }
        }
        if (st->n_hostbufs < st->cap_hostbufs) {
            st->hostbufs[st->n_hostbufs++] = buf;
        }
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
    /* If the region lives inside one of our mapped buffers (arch scratch
     * pool / weight arena since P3), borrow its VkBuffer so GPU ops can
     * bind this slice. Pointers outside any known buffer (GGUF mmap) stay
     * pure bookkeeping — ops on them fall back to the CPU path. */
    for (size_t i = 0; i < st->n_hostbufs; ++i) {
        struct geist_buffer *p = st->hostbufs[i];
        const uint8_t       *lo = p->mapped, *ptr = host_ptr;
        if (ptr >= lo && ptr + n_bytes <= lo + p->bytes) {
            buf->buf        = p->buf;
            buf->base_off   = (size_t) (ptr - lo);
            buf->borrowed   = true;
            buf->device_mem = p->device_mem;
            break;
        }
    }
    *out = buf;
    return GEIST_OK;
}

static void vk_seq_flush(struct vk_state *st);

static void vk_buffer_destroy(struct geist_backend *be, struct geist_buffer *buf) {
    if (buf == nullptr) {
        return;
    }
    struct vk_state *st = buf->owner;
    vk_seq_flush(st); /* the open batch may reference this buffer */
    if (buf->buf != VK_NULL_HANDLE && !buf->borrowed) {
        /* drop cached descriptor sets that reference this buffer — the
         * driver may recycle the handle value for a future buffer */
        for (uint32_t i = 0; i < VK_DSET_CACHE; ++i) {
            if (st->dset_cache[i].key != 0) {
                st->dset_cache[i].key = UINT64_MAX; /* tombstone: never matches */
            }
        }
        for (size_t i = 0; i < st->n_hostbufs; ++i) {
            if (st->hostbufs[i] == buf) {
                st->hostbufs[i] = st->hostbufs[--st->n_hostbufs];
                break;
            }
        }
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
    vk_seq_flush(buf->owner);
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
    vk_seq_flush(buf->owner);
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
    vk_seq_flush(buf->owner); /* host is about to read/write — drain the batch */
    if (buf->host_alias != nullptr) {
        return buf->host_alias;
    }
    return buf->mapped; /* nullptr for device-local — caller must fall back */
}

static void vk_buffer_unmap(struct geist_buffer *buf) {
    (void) buf; /* persistent coherent mappings — nothing to flush */
}

/* ====================================================================== */
/* Linear dispatch (Phase 2: synchronous per-call round-trip)              */
/*                                                                         */
/* The main contract hands the resolved kernels host pointers (x, y) and   */
/* a host w->raw. Weights were copied to VRAM at resolve time; x/y round-  */
/* trip through persistent host-visible staging. One submit + fence wait   */
/* per linear — correct first. Phase 3 moves the hot loop onto linear_t    */
/* with device-resident activations and batched submits.                   */
/* ====================================================================== */

static uint32_t vk_linear_gx(enum vk_pipe pipe, uint32_t n_out);
static uint32_t vk_linear_gy(enum vk_pipe pipe, uint32_t m);
static size_t   vk_t_n(const struct geist_tensor *t);
static bool     vk_t_geom(const struct geist_tensor *t, size_t *rows, size_t *cols,
                          size_t *stride);
static void     vk_linear_cm_route(struct vk_state *st, enum vk_pipe *pipe, uint32_t m,
                                   uint32_t n_out, uint32_t *gx, uint32_t *gy);

[[nodiscard]] static enum geist_status vk_stage_reserve_role(struct geist_backend  *be,
                                                             struct geist_buffer  **slot,
                                                             size_t                 bytes,
                                                             enum geist_buffer_role role) {
    if (*slot != nullptr && (*slot)->bytes >= bytes) {
        return GEIST_OK;
    }
    if (*slot != nullptr) {
        vk_buffer_destroy(be, *slot);
        *slot = nullptr;
    }
    size_t cap = 1u << 20; /* 1 MiB floor, then powers of two */
    while (cap < bytes) {
        cap *= 2;
    }
    return vk_buffer_create(be, cap, role, GEIST_MEMORY_AUTO, slot);
}

[[nodiscard]] static enum geist_status vk_stage_reserve(struct geist_backend *be,
                                                        struct geist_buffer **slot,
                                                        size_t                bytes) {
    return vk_stage_reserve_role(be, slot, bytes, GEIST_BUFFER_STAGING);
}

static struct geist_buffer *vk_weight_lookup(struct vk_state *st, const void *host) {
    for (size_t i = 0; i < st->n_weights; ++i) {
        if (st->weights[i].host == host) {
            return st->weights[i].gpu;
        }
    }
    return nullptr;
}

/* ---- Sequence core ---------------------------------------------------- */

static void vk_seq_flush(struct vk_state *st) {
    if (st == nullptr || !st->seq_open) {
        return;
    }
    st->stat_flushes++;
    st->seq_open       = false;
    st->seq_dispatches = 0;
    st->n_dirty        = 0;
    bool ok            = st->fn.EndCommandBuffer(st->seq_cmd) == VK_SUCCESS;
    if (ok) {
        VkSubmitInfo submit = {.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                               .commandBufferCount = 1,
                               .pCommandBuffers    = &st->seq_cmd};
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ok = st->fn.QueueSubmit(st->queue, 1, &submit, st->seq_fence) == VK_SUCCESS &&
             st->fn.WaitForFences(st->device, 1, &st->seq_fence, VK_TRUE, UINT64_MAX) ==
                     VK_SUCCESS;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        st->stat_wait_ns += (uint64_t) (t1.tv_sec - t0.tv_sec) * 1000000000ull +
                            (uint64_t) (t1.tv_nsec - t0.tv_nsec);
        (void) st->fn.ResetFences(st->device, 1, &st->seq_fence);
    }
    if (!ok) {
        /* Loud failure: outputs of the dropped batch are undefined and the
         * parity/token gates will catch it — same policy as the resolved
         * kernels. */
        fprintf(stderr, "geist vulkan: sequence flush failed — batch dropped\n");
        geist_backend_set_error(st->backend, GEIST_E_BACKEND, "vulkan: sequence flush failed");
    }
    if (ok && st->profile_enabled && st->ts_count > 1) {
        uint64_t ts[VK_SEQ_MAX_DISPATCH + 8];
        if (st->fn.GetQueryPoolResults(st->device, st->ts_pool, 0, st->ts_count,
                                       sizeof(uint64_t) * st->ts_count, ts, sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) ==
            VK_SUCCESS) {
            for (uint32_t i = 1; i < st->ts_count; ++i) {
                const uint64_t d = ts[i] - ts[i - 1];
                st->prof_ns[st->ts_pipe[i]] +=
                        (uint64_t) ((double) d * (double) st->ts_period_ns);
                st->prof_calls[st->ts_pipe[i]]++;
            }
        }
    }
    st->ts_count = 0;
    (void) st->fn.ResetDescriptorPool(st->device, st->seq_pool, 0);
    st->xring_used = 0;
}

[[nodiscard]] static enum geist_status vk_seq_open_cmd(struct vk_state *st) {
    if (st->seq_open) {
        return GEIST_OK;
    }
    st->seq_cmd_idx = 0;
    st->seq_in_cmd  = 0;
    st->seq_cmd     = st->seq_cmds[0];
    VkCommandBufferBeginInfo begin = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if (st->fn.BeginCommandBuffer(st->seq_cmd, &begin) != VK_SUCCESS) {
        geist_backend_set_error(st->backend, GEIST_E_BACKEND, "vulkan: seq begin failed");
        return GEIST_E_BACKEND;
    }
    st->seq_open = true;
    if (st->profile_enabled) {
        st->fn.CmdResetQueryPool(st->seq_cmd, st->ts_pool, 0, VK_SEQ_MAX_DISPATCH + 8);
        st->fn.CmdWriteTimestamp(st->seq_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, st->ts_pool,
                                 0);
        st->ts_count = 1;
    }
    return GEIST_OK;
}

/* Rolling submission: close the open command buffer, hand it to the GPU
 * WITHOUT waiting, and keep recording in the next ring slot. Keeps the
 * GPU fed (and boosted) while the CPU encodes the rest of the token. */
static void vk_seq_roll(struct vk_state *st) {
    if (!st->seq_open || st->seq_in_cmd == 0 || st->seq_cmd_idx + 1 >= VK_SEQ_CMDBUFS) {
        return;
    }
    if (st->fn.EndCommandBuffer(st->seq_cmd) != VK_SUCCESS) {
        return; /* keep recording — flush will fail loudly if truly broken */
    }
    VkSubmitInfo submit = {.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                           .commandBufferCount = 1,
                           .pCommandBuffers    = &st->seq_cmd};
    (void) st->fn.QueueSubmit(st->queue, 1, &submit, VK_NULL_HANDLE);
    st->seq_cmd_idx++;
    st->seq_cmd    = st->seq_cmds[st->seq_cmd_idx];
    st->seq_in_cmd = 0;
    VkCommandBufferBeginInfo begin = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    (void) st->fn.BeginCommandBuffer(st->seq_cmd, &begin);
}

/* Record a per-dispatch timestamp attributed to `slot` (pipe index, or
 * VK_PIPE_COUNT for transfer copies). */
static void vk_prof_stamp(struct vk_state *st, uint32_t slot) {
    if (!st->profile_enabled || st->ts_count >= VK_SEQ_MAX_DISPATCH + 8) {
        return;
    }
    st->fn.CmdWriteTimestamp(st->seq_cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, st->ts_pool,
                             st->ts_count);
    st->ts_pipe[st->ts_count] = (uint8_t) slot;
    st->ts_count++;
}

/* Execution barrier between dependent dispatches/copies in the batch. One
 * global memory barrier — coarse but correct on a single compute queue.
 * ponytail: per-buffer barriers if the profiler ever blames this. */
static void vk_seq_barrier(struct vk_state *st) {
    st->n_dirty = 0;
    st->stat_barriers++;
    static int no_bar = -1;
    if (no_bar < 0) {
        no_bar = getenv("GEIST_VK_NO_BARRIER") != nullptr; /* perf probe: WRONG results */
    }
    if (no_bar > 0) {
        return;
    }
    const VkMemoryBarrier mb = {
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                             VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
    st->fn.CmdPipelineBarrier(st->seq_cmd,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 1, &mb, 0, nullptr, 0, nullptr);
}

/* Barrier iff the new accesses conflict with anything recorded since the
 * last barrier, then record them. acc == nullptr → conservative (always
 * barrier when the batch is non-empty, like the pre-tracking behavior). */
static void vk_seq_hazard(struct vk_state              *st,
                          const VkDescriptorBufferInfo *infos,
                          const struct vk_access       *acc,
                          uint32_t                      n) {
    if (st->seq_dispatches == 0) {
        st->n_dirty = 0;
    }
    bool conflict = acc == nullptr && st->seq_dispatches > 0;
    if (acc != nullptr) {
        for (uint32_t i = 0; i < n && !conflict; ++i) {
            for (uint32_t d = 0; d < st->n_dirty; ++d) {
                const struct vk_dirty *e = &st->dirty[d];
                if (e->buf != infos[i].buffer || (!e->write && !acc[i].write)) {
                    continue; /* different buffer or read-read */
                }
                if (acc[i].lo < e->hi && e->lo < acc[i].hi) {
                    conflict = true;
                    break;
                }
            }
        }
        if (!conflict && st->n_dirty + n > VK_DIRTY_CAP) {
            conflict = true; /* table full — degrade to the old behavior */
        }
    }
    if (conflict && st->seq_dispatches > 0) {
        vk_seq_barrier(st);
    } else if (st->seq_dispatches > 0) {
        st->stat_barriers_elided++;
    }
    if (acc != nullptr) {
        for (uint32_t i = 0; i < n && st->n_dirty < VK_DIRTY_CAP; ++i) {
            st->dirty[st->n_dirty++] = (struct vk_dirty) {
                    .buf = infos[i].buffer, .lo = acc[i].lo, .hi = acc[i].hi,
                    .write = acc[i].write};
        }
    } else {
        st->n_dirty = 0; /* unknown writes — next dispatch must barrier */
    }
}

/* Append one compute dispatch to the open sequence. infos[] length must
 * equal the pipeline's binding count. */
[[nodiscard]] static enum geist_status vk_seq_dispatch_acc(struct geist_backend  *be,
                                                           enum vk_pipe           pipe,
                                                           const VkDescriptorBufferInfo *infos,
                                                           const struct vk_access *acc,
                                                           const void             *push,
                                                           uint32_t                push_bytes,
                                                           uint32_t                gx,
                                                           uint32_t                gy,
                                                           uint32_t                gz) {
    struct vk_state *st = be->state;
    if (st->seq_dispatches >= VK_SEQ_MAX_DISPATCH) {
        vk_seq_flush(st);
    }
    enum geist_status s = vk_seq_open_cmd(st);
    if (s != GEIST_OK) {
        return s;
    }
    const uint32_t nbind = vk_pipe_nbind[pipe];
    /* descriptor-set cache: same (nbind, buffers) tuple recurs every token */
    uint64_t h = 1469598103934665603ull ^ nbind;
    for (uint32_t i = 0; i < nbind; ++i) {
        h = (h ^ (uint64_t) infos[i].buffer) * 1099511628211ull;
    }
    if (h == 0) {
        h = 1;
    }
    VkDescriptorSet set    = VK_NULL_HANDLE;
    uint32_t        islot  = UINT32_MAX; /* first reusable slot (empty/tombstone) */
    bool            iempty = false;
    uint32_t        slot   = (uint32_t) (h & (VK_DSET_CACHE - 1));
    for (uint32_t probe = 0; probe < 16; ++probe, slot = (slot + 1) & (VK_DSET_CACHE - 1)) {
        const uint64_t k = st->dset_cache[slot].key;
        if (k == h) {
            set = st->dset_cache[slot].set;
            st->stat_dset_hits++;
            break;
        }
        if (islot == UINT32_MAX && (k == 0 || k == UINT64_MAX)) {
            islot  = slot;
            iempty = k == 0;
        }
        if (k == 0) {
            break;
        }
    }
    if (set == VK_NULL_HANDLE) {
        st->stat_dset_miss++;
        slot = islot != UINT32_MAX ? islot : slot;
        /* tombstoned slots reuse their old set object; empty slots get a
         * fresh one from the cache pool */
        const bool cacheable = islot != UINT32_MAX;
        if (cacheable && !iempty && st->dset_cache[slot].set != VK_NULL_HANDLE) {
            set = st->dset_cache[slot].set;
            st->dset_cache[slot].key = h;
        }
        if (set == VK_NULL_HANDLE) {
            VkDescriptorSetAllocateInfo ainfo2 = {
                    .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    .descriptorPool     = cacheable ? st->dset_cache_pool : st->seq_pool,
                    .descriptorSetCount = 1,
                    .pSetLayouts        = &st->seq_dlayouts[nbind - 2]};
            if (st->fn.AllocateDescriptorSets(st->device, &ainfo2, &set) != VK_SUCCESS) {
                /* cache pool dry — transient set from the per-flush pool */
                ainfo2.descriptorPool = st->seq_pool;
                if (st->fn.AllocateDescriptorSets(st->device, &ainfo2, &set) != VK_SUCCESS) {
                    vk_seq_flush(st);
                    s = vk_seq_open_cmd(st);
                    if (s != GEIST_OK || st->fn.AllocateDescriptorSets(st->device, &ainfo2,
                                                                       &set) != VK_SUCCESS) {
                        geist_backend_set_error(be, GEIST_E_BACKEND,
                                                "vulkan: descriptor alloc failed");
                        return GEIST_E_BACKEND;
                    }
                }
            } else if (cacheable) {
                st->dset_cache[slot] = (struct vk_dset_entry) {.key = h, .set = set};
            }
        }
        VkWriteDescriptorSet writes[6];
        for (uint32_t i = 0; i < nbind; ++i) {
            writes[i] = (VkWriteDescriptorSet) {
                    .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet          = set,
                    .dstBinding      = i,
                    .descriptorCount = 1,
                    .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo     = &infos[i]};
        }
        st->fn.UpdateDescriptorSets(st->device, nbind, writes, 0, nullptr);
    }
    vk_seq_hazard(st, infos, acc, nbind);
    st->fn.CmdBindPipeline(st->seq_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, st->pipes[pipe]);
    st->fn.CmdBindDescriptorSets(st->seq_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 st->seq_playouts[nbind - 2], 0, 1, &set, 0, nullptr);
    st->fn.CmdPushConstants(st->seq_cmd, st->seq_playouts[nbind - 2],
                            VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
    st->fn.CmdDispatch(st->seq_cmd, gx, gy, gz);
    st->seq_dispatches++;
    st->seq_in_cmd++;
    st->stat_dispatches++;
    vk_prof_stamp(st, (uint32_t) pipe);
    if (st->seq_in_cmd >= VK_SEQ_ROTATE) {
        vk_seq_roll(st);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_seq_dispatch(struct geist_backend         *be,
                                                       enum vk_pipe                  pipe,
                                                       const VkDescriptorBufferInfo *infos,
                                                       const void                   *push,
                                                       uint32_t                      push_bytes,
                                                       uint32_t                      gx,
                                                       uint32_t                      gy,
                                                       uint32_t                      gz) {
    return vk_seq_dispatch_acc(be, pipe, infos, nullptr, push, push_bytes, gx, gy, gz);
}

/* Access-range helpers: byte spans inside the bound VkBuffer. */
static struct vk_access vk_acc(uint64_t lo_bytes, uint64_t n_bytes, bool write) {
    return (struct vk_access) {.lo = lo_bytes, .hi = lo_bytes + n_bytes, .write = write};
}

static struct vk_access vk_acc_all(bool write) {
    return (struct vk_access) {.lo = 0, .hi = UINT64_MAX, .write = write};
}

/* Byte span of an F32 DENSE tensor inside its VkBuffer (slab-stride aware). */
static struct vk_access vk_acc_tensor(const struct geist_tensor *t, bool write) {
    const uint64_t lo = t->buffer->base_off + t->offset;
    uint64_t       span;
    size_t         rows, cols, stride;
    if (vk_t_geom(t, &rows, &cols, &stride)) {
        span = ((uint64_t) (rows - 1) * stride + cols) * 4u;
    } else {
        span = (uint64_t) vk_t_n(t) * 4u;
    }
    return vk_acc(lo, span, write);
}

/* GPU view of a tensor: VkBuffer + f32 element offset. False when the
 * tensor's buffer has no VkBuffer behind it (e.g. GGUF-mmap aliases). */
static bool vk_tensor_gpu(const struct geist_tensor *t, VkDescriptorBufferInfo *out,
                          uint32_t *elem_off) {
    if (t == nullptr || t->buffer == nullptr || t->buffer->buf == VK_NULL_HANDLE) {
        return false;
    }
    const size_t byte_off = t->buffer->base_off + t->offset;
    if (byte_off % 4 != 0) {
        return false;
    }
    *out      = (VkDescriptorBufferInfo) {.buffer = t->buffer->buf, .range = VK_WHOLE_SIZE};
    *elem_off = (uint32_t) (byte_off / 4);
    return true;
}

/* Same, but offsets in f16 elements (for F16 KV-cache views). */
static bool vk_tensor_gpu_f16(const struct geist_tensor *t, VkDescriptorBufferInfo *out,
                              uint32_t *elem_off) {
    if (t == nullptr || t->buffer == nullptr || t->buffer->buf == VK_NULL_HANDLE) {
        return false;
    }
    const size_t byte_off = t->buffer->base_off + t->offset;
    if (byte_off % 2 != 0) {
        return false;
    }
    *out      = (VkDescriptorBufferInfo) {.buffer = t->buffer->buf, .range = VK_WHOLE_SIZE};
    *elem_off = (uint32_t) (byte_off / 2);
    return true;
}

/* Element count of an F16 DENSE tensor (metadata only). */
static size_t vk_t_n16(const struct geist_tensor *t) {
    if (t == nullptr || t->dtype != GEIST_DTYPE_F16 || t->layout != GEIST_LAYOUT_DENSE ||
        t->buffer == nullptr || t->ndim < 1) {
        return 0;
    }
    size_t n = 1;
    for (int d = 0; d < t->ndim; d++) {
        if (t->shape[d] <= 0) {
            return 0;
        }
        n *= (size_t) t->shape[d];
    }
    return n;
}

static struct vk_access vk_acc_tensor16(const struct geist_tensor *t, bool write) {
    return vk_acc(t->buffer->base_off + t->offset, vk_t_n16(t) * 2u, write);
}

[[nodiscard]] static enum geist_status vk_dispatch_linear(struct geist_backend *be,
                                                          enum vk_pipe          pipe,
                                                          struct geist_buffer  *wbuf,
                                                          const float          *x,
                                                          float                *y,
                                                          size_t                m,
                                                          size_t                n_in,
                                                          size_t                n_out) {
    struct vk_state *st = be->state;
    vk_seq_flush(st); /* host x/y round-trip — must not interleave with a batch */
    /* x_stage: the GPU reads it hot (GEMM B tiles), the host only writes —
     * SCRATCH role makes it BAR-eligible. y_stage stays in system RAM
     * (the host reads results back; CPU reads from BAR are uncached). */
    enum geist_status s =
            vk_stage_reserve_role(be, &st->x_stage, m * n_in * sizeof(float),
                                  GEIST_BUFFER_SCRATCH);
    if (s == GEIST_OK) {
        s = vk_stage_reserve(be, &st->y_stage, m * n_out * sizeof(float));
    }
    if (s != GEIST_OK) {
        return s;
    }
    memcpy(st->x_stage->mapped, x, m * n_in * sizeof(float));

    const VkDescriptorBufferInfo binfo[3] = {
            {.buffer = st->x_stage->buf, .range = VK_WHOLE_SIZE},
            {.buffer = wbuf->buf, .range = VK_WHOLE_SIZE},
            {.buffer = st->y_stage->buf, .range = VK_WHOLE_SIZE},
    };
    const struct vk_push push = {.n_in           = (uint32_t) n_in,
                                 .n_out          = (uint32_t) n_out,
                                 .blocks_per_row = (uint32_t) (n_in / 256),
                                 .rows           = (uint32_t) m,
                                 .x_stride       = (uint32_t) n_in,
                                 .y_stride       = (uint32_t) n_out};
    enum vk_pipe eff = pipe;
    uint32_t     gx  = vk_linear_gx(pipe, (uint32_t) n_out);
    uint32_t     gy  = vk_linear_gy(pipe, (uint32_t) m);
    vk_linear_cm_route(st, &eff, (uint32_t) m, (uint32_t) n_out, &gx, &gy);
    s = vk_seq_dispatch(be, eff, binfo, &push, sizeof(push), gx, gy, 1);
    if (s != GEIST_OK) {
        return s;
    }
    vk_seq_flush(st);
    memcpy(y, st->y_stage->mapped, m * n_out * sizeof(float));
    return GEIST_OK;
}

/* Resolver-installed kernels. The signature has no error path — failures
 * report to stderr and zero y so a defect is loud in the parity gate
 * rather than silent garbage (same policy as the Metal backend). */
static void vk_linear_run(const float               *x,
                          const struct geist_weight *w,
                          size_t                     m,
                          struct geist_backend      *be,
                          float                     *y,
                          enum vk_pipe               matvec_pipe,
                          enum vk_pipe               matmul_pipe) {
    struct vk_state     *st   = be->state;
    struct geist_buffer *wbuf = vk_weight_lookup(st, w->raw);
    const size_t         n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    if (wbuf == nullptr ||
        vk_dispatch_linear(be, m == 1 ? matvec_pipe : matmul_pipe, wbuf, x, y, m, n_in,
                           n_out) != GEIST_OK) {
        fprintf(stderr, "geist vulkan: linear dispatch failed (%s) — zeroing output\n",
                geist_backend_errmsg(be));
        memset(y, 0, m * n_out * sizeof(float));
    }
}

static void vk_w_q4k_m1(const float *x, const struct geist_weight *w,
                        struct geist_backend *be, float *y) {
    vk_linear_run(x, w, 1, be, y, VK_PIPE_MATVEC_Q4K, VK_PIPE_MATMUL_Q4K);
}
static void vk_w_q4k_mN(const float *x, const struct geist_weight *w, size_t m,
                        struct geist_backend *be, float *y) {
    vk_linear_run(x, w, m, be, y, VK_PIPE_MATVEC_Q4K, VK_PIPE_MATMUL_Q4K);
}
static void vk_w_q6k_m1(const float *x, const struct geist_weight *w,
                        struct geist_backend *be, float *y) {
    vk_linear_run(x, w, 1, be, y, VK_PIPE_MATVEC_Q6K, VK_PIPE_MATMUL_Q6K);
}
static void vk_w_q6k_mN(const float *x, const struct geist_weight *w, size_t m,
                        struct geist_backend *be, float *y) {
    vk_linear_run(x, w, m, be, y, VK_PIPE_MATVEC_Q6K, VK_PIPE_MATMUL_Q6K);
}
static void vk_w_f32_m1(const float *x, const struct geist_weight *w,
                        struct geist_backend *be, float *y) {
    vk_linear_run(x, w, 1, be, y, VK_PIPE_MATVEC_F32, VK_PIPE_MATMUL_F32);
}
static void vk_w_f32_mN(const float *x, const struct geist_weight *w, size_t m,
                        struct geist_backend *be, float *y) {
    vk_linear_run(x, w, m, be, y, VK_PIPE_MATVEC_F32, VK_PIPE_MATMUL_F32);
}

/* ---- CPU fallback for dtypes without a GPU kernel yet (F16/BF16/...) ----
 * Row-dequant + naive dot, following cpu_scalar_w_quant_*; keeps model
 * loading alive for mixed-dtype GGUFs until those dtypes get shaders. */

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
        dequant_q3_K_row(base + j * n_in / Q3_K_BLOCK_ELEMS * Q3_K_BLOCK_BYTES, row, n_in);
        return true;
    case GEIST_DTYPE_Q5_K:
        dequant_q5_K_row(base + j * n_in / Q5_K_BLOCK_ELEMS * Q5_K_BLOCK_BYTES, row, n_in);
        return true;
    case GEIST_DTYPE_Q8_0:
        dequant_q8_0_row(base + j * n_in / Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES, row, n_in);
        return true;
    default:
        return false;
    }
}

static void vk_w_cpu_mN(const float *x, const struct geist_weight *w, size_t m,
                        struct geist_backend *be, float *y) {
    (void) be;
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    float       *row   = heap_alloc_aligned(n_in * sizeof(float), OPTIMAL_ALIGNMENT);
    if (row == nullptr) {
        return;
    }
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
    safe_free((void **) &row);
}

static void vk_w_cpu_m1(const float *x, const struct geist_weight *w,
                        struct geist_backend *be, float *y) {
    vk_w_cpu_mN(x, w, 1, be, y);
}

/* ---- resolve_weight: upload GPU-supported dtypes to VRAM, register,     */
/*      install kernels; CPU fallback for the rest.                        */

/* GPU copies of Q6_K are repacked to 216-byte blocks (210 + 6 pad) so
 * every field sits 4-byte aligned and the kernels use word loads. */
enum { VK_Q6K_GPU_BLOCK = 216 };

[[nodiscard]] static size_t vk_weight_bytes(const struct geist_weight *w) {
    const size_t n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    switch ((enum geist_dtype) w->dtype) {
    case GEIST_DTYPE_Q4_K: return n_out * (n_in / Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES;
    case GEIST_DTYPE_Q6_K: return n_out * (n_in / Q6_K_BLOCK_ELEMS) * VK_Q6K_GPU_BLOCK;
    case GEIST_DTYPE_F32: return n_out * n_in * sizeof(float);
    default: return 0;
    }
}

[[nodiscard]] static enum geist_status vk_resolve_weight(struct geist_backend *be,
                                                         struct geist_weight  *w) {
    struct vk_state *st = be->state;
    if (w == nullptr || w->raw == nullptr || w->n_in <= 0 || w->n_out <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    geist_kernel_linear_m1_fn m1;
    geist_kernel_linear_mN_fn mN;
    switch ((enum geist_dtype) w->dtype) {
    case GEIST_DTYPE_Q4_K:
        if (w->n_in % 256 != 0) {
            return GEIST_E_UNSUPPORTED;
        }
        m1 = vk_w_q4k_m1;
        mN = vk_w_q4k_mN;
        break;
    case GEIST_DTYPE_Q6_K:
        if (w->n_in % 256 != 0) {
            return GEIST_E_UNSUPPORTED;
        }
        m1 = vk_w_q6k_m1;
        mN = vk_w_q6k_mN;
        break;
    case GEIST_DTYPE_F32:
        m1 = vk_w_f32_m1;
        mN = vk_w_f32_mN;
        break;
    case GEIST_DTYPE_F16:
    case GEIST_DTYPE_BF16:
    case GEIST_DTYPE_Q3_K:
    case GEIST_DTYPE_Q5_K:
    case GEIST_DTYPE_Q8_0:
        w->linear_m1 = vk_w_cpu_m1;
        w->linear_mN = vk_w_cpu_mN;
        return GEIST_OK;
    default:
        return GEIST_E_UNSUPPORTED;
    }

    /* Upload to VRAM and register. An existing entry for the same host
     * pointer is REPLACED, not reused: the same address can carry new bytes
     * after a model reload (or a freed+remalloc'd test blob) — the latest
     * resolve is authoritative. Tied weights resolving twice re-upload the
     * same bytes once more at load time; harmless. */
    struct vk_weight_entry *slot = nullptr;
    for (size_t i = 0; i < st->n_weights; ++i) {
        if (st->weights[i].host == w->raw) {
            slot = &st->weights[i];
            break;
        }
    }
    if (slot == nullptr) {
        if (st->n_weights == st->cap_weights) {
            const size_t            cap = st->cap_weights == 0 ? 64 : st->cap_weights * 2;
            struct vk_weight_entry *nw =
                    geist_backend_alloc(be, cap * sizeof(*nw), alignof(struct vk_weight_entry));
            if (nw == nullptr) {
                return GEIST_E_OOM;
            }
            memcpy(nw, st->weights, st->n_weights * sizeof(*nw));
            geist_backend_free(be, st->weights);
            st->weights     = nw;
            st->cap_weights = cap;
        }
        slot  = &st->weights[st->n_weights++];
        *slot = (struct vk_weight_entry) {0};
    }
    const size_t         bytes = vk_weight_bytes(w);
    struct geist_buffer *gpu   = nullptr;
    enum geist_status    s =
            vk_buffer_create(be, bytes, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE, &gpu);
    if (s == GEIST_OK && w->dtype == GEIST_DTYPE_Q6_K) {
        const size_t n_blocks = bytes / VK_Q6K_GPU_BLOCK;
        uint8_t     *packed   = heap_alloc_aligned(bytes, 64);
        if (packed == nullptr) {
            s = GEIST_E_OOM;
        } else {
            const uint8_t *srcb = (const uint8_t *) w->raw;
            for (size_t i = 0; i < n_blocks; ++i) {
                memcpy(packed + i * VK_Q6K_GPU_BLOCK, srcb + i * Q6_K_BLOCK_BYTES,
                       Q6_K_BLOCK_BYTES);
                memset(packed + i * VK_Q6K_GPU_BLOCK + Q6_K_BLOCK_BYTES, 0,
                       VK_Q6K_GPU_BLOCK - Q6_K_BLOCK_BYTES);
            }
            s = vk_buffer_upload(gpu, bytes, packed);
            safe_free((void **) &packed);
        }
    } else if (s == GEIST_OK) {
        s = vk_buffer_upload(gpu, bytes, (const uint8_t *) w->raw);
    }
    if (s != GEIST_OK) {
        if (gpu != nullptr) {
            vk_buffer_destroy(be, gpu);
        }
        if (slot->gpu == nullptr) {
            st->n_weights--; /* fresh slot never got a buffer — roll back */
        }
        return s;
    }
    if (slot->gpu != nullptr) {
        vk_buffer_destroy(be, slot->gpu);
    }
    *slot = (struct vk_weight_entry) {.host = w->raw, .gpu = gpu};
    w->linear_m1 = m1;
    w->linear_mN = mN;
    return GEIST_OK;
}

/* ====================================================================== */
/* Level-2 ops — CPU loops over host-visible buffers (Phase 2)             */
/*                                                                         */
/* All activation/scratch buffers this backend creates are host-visible    */
/* (or aliased host regions), so the reference-op bodies from cpu_scalar   */
/* apply unchanged; only the pointer unwrap differs. The heavy lifting     */
/* (linears = the weight reads) already runs on the GPU; these small       */
/* F32 ops move to shaders in Phase 3 where fusion makes them pay.         */
/* ====================================================================== */

/* Element count of an F32 DENSE tensor, 0 on any mismatch. Metadata only. */
static size_t vk_t_n(const struct geist_tensor *t) {
    if (t == nullptr || t->dtype != GEIST_DTYPE_F32 || t->layout != GEIST_LAYOUT_DENSE ||
        t->buffer == nullptr || t->ndim < 1) {
        return 0;
    }
    size_t n = 1;
    for (int d = 0; d < t->ndim; d++) {
        if (t->shape[d] <= 0) {
            return 0;
        }
        n *= (size_t) t->shape[d];
    }
    return n;
}

static void *vk_tensor_host(const struct geist_tensor *t, size_t *out_n) {
    const size_t n = vk_t_n(t);
    if (n == 0) {
        return nullptr;
    }
    uint8_t *base = t->buffer->host_alias != nullptr ? t->buffer->host_alias
                                                     : t->buffer->mapped;
    if (base == nullptr) {
        return nullptr; /* device-local — CPU ops can't touch it */
    }
    t->buffer->owner->stat_cpu_falls++;
    vk_seq_flush(t->buffer->owner); /* host access — drain pending GPU work */
    if (out_n != nullptr) {
        *out_n = n;
    }
    return base + t->offset;
}

static uint32_t vk_groups(size_t n) {
    return (uint32_t) ((n + 255) / 256);
}

/* Dispatch geometry of the linear pipes. matvec q4k/q6k: 8 rows per
 * workgroup. matmul_q4k: 4 output rows x 16 batch rows per workgroup. */
static uint32_t vk_linear_gx(enum vk_pipe pipe, uint32_t n_out) {
    if (pipe == VK_PIPE_MATVEC_Q4K) {
        return (n_out + 7u) / 8u; /* 2 warps x 4 rows per workgroup */
    }
    if (pipe == VK_PIPE_MATVEC_Q6K) {
        return (n_out + 7u) / 8u;
    }
    if (pipe == VK_PIPE_MATMUL_Q4K) {
        return (n_out + 7u) / 8u;
    }
    if (pipe == VK_PIPE_MATMUL_Q6K || pipe == VK_PIPE_MATMUL_F32) {
        return (n_out + 3u) / 4u;
    }
    return n_out;
}

static uint32_t vk_linear_gy(enum vk_pipe pipe, uint32_t m) {
    if (pipe == VK_PIPE_MATMUL_Q4K) {
        return (m + 31u) / 32u;
    }
    return (pipe == VK_PIPE_MATMUL_Q6K || pipe == VK_PIPE_MATMUL_F32) ? (m + 15u) / 16u : m;
}

/* Reroute conforming quant GEMMs onto the tensor-core pipelines. */
static void vk_linear_cm_route(struct vk_state *st, enum vk_pipe *pipe, uint32_t m,
                               uint32_t n_out, uint32_t *gx, uint32_t *gy) {
    enum vk_pipe cm;
    if (*pipe == VK_PIPE_MATMUL_Q4K) {
        cm = VK_PIPE_MM_Q4K_CM;
    } else if (*pipe == VK_PIPE_MATMUL_Q6K) {
        cm = VK_PIPE_MM_Q6K_CM;
    } else {
        return;
    }
    if ((m & 15u) != 0 || n_out % 64u != 0 || st->pipes[cm] == VK_NULL_HANDLE) {
        return;
    }
    /* small n_out starves the SMs on the 64-row tile — use the 32-row one */
    if (cm == VK_PIPE_MM_Q4K_CM && n_out < 4096u &&
        st->pipes[VK_PIPE_MM_Q4K_CM32] != VK_NULL_HANDLE) {
        *pipe = VK_PIPE_MM_Q4K_CM32;
        *gx   = n_out / 32u;
        *gy   = (m + 63u) / 64u;
        return;
    }
    *pipe = cm;
    *gx   = n_out / 64u;
    *gy   = (m + 63u) / 64u;
}

#define VK_OPS(be, bit) ((((struct vk_state *) (be)->state)->gpu_ops & (bit)) != 0)

/* Row geometry: rows × cols with row stride in elements. 2D views may
 * carry a slab stride (stride[0] > cols — the PLE per-layer-input slab);
 * other ranks are treated as one contiguous run. */
static bool vk_t_geom(const struct geist_tensor *t, size_t *rows, size_t *cols,
                      size_t *stride) {
    const size_t n = vk_t_n(t);
    if (n == 0) {
        return false;
    }
    if (t->ndim == 2 && t->stride[1] == 1 && t->stride[0] > t->shape[1]) {
        *rows   = (size_t) t->shape[0];
        *cols   = (size_t) t->shape[1];
        *stride = (size_t) t->stride[0];
        return true;
    }
    *rows   = 1;
    *cols   = n;
    *stride = n;
    return true;
}

/* GPU-first attempt for the 3-buffer elementwise family (add/mul/gelu_mul):
 * push {n, a_off, b_off, y_off, cols, a_stride, b_stride, y_stride};
 * cols == 0 → all-contiguous fast path in the shader. */
static bool vk_try_ew3(struct geist_backend *be, enum vk_pipe pipe,
                       const struct geist_tensor *a, const struct geist_tensor *b,
                       const struct geist_tensor *y) {
    const size_t n = vk_t_n(a);
    if (!VK_OPS(be, 2u) || n == 0 || n != vk_t_n(b) || n != vk_t_n(y)) {
        return false;
    }
    size_t ra, ca, sa, rb, cb, sb, ry, cy, sy;
    if (!vk_t_geom(a, &ra, &ca, &sa) || !vk_t_geom(b, &rb, &cb, &sb) ||
        !vk_t_geom(y, &ry, &cy, &sy)) {
        return false;
    }
    uint32_t cols = 0;
    if (sa != ca || sb != cb || sy != cy) {
        /* mixed contiguous/strided operands: unify on the strided cols */
        cols = (uint32_t) (sa != ca ? ca : (sb != cb ? cb : cy));
        if ((sa != ca && ca != cols) || (sb != cb && cb != cols) || (sy != cy && cy != cols) ||
            n % cols != 0) {
            return false;
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
        return false;
    }
    const uint32_t push[8] = {(uint32_t) n, off[0],        off[1],        off[2],
                              cols,         (uint32_t) sa, (uint32_t) sb, (uint32_t) sy};
    const struct vk_access acc[3] = {vk_acc_tensor(a, false), vk_acc_tensor(b, false),
                                     vk_acc_tensor(y, true)};
    return vk_seq_dispatch_acc(be, pipe, bi, acc, push, sizeof(push), vk_groups(n), 1, 1) ==
           GEIST_OK;
}

/* Strided-aware CPU fallback core for the elementwise trio. op: 0=add,
 * 1=mul, 2=gelu_tanh_mul. */
[[nodiscard]] static enum geist_status vk_ew3_cpu(struct geist_backend *be, int op,
                                                  const struct geist_tensor *a,
                                                  const struct geist_tensor *b,
                                                  struct geist_tensor *y, const char *name) {
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
        for (size_t i = 0; i < cc; i++) {
            switch (op) {
            case 0: yrow[i] = arow[i] + brow[i]; break;
            case 1: yrow[i] = arow[i] * brow[i]; break;
            default: {
                const float v = arow[i];
                const float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
                yrow[i]       = (0.5f * v * (1.0f + tanhf(u))) * brow[i];
            }
            }
        }
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_add(struct geist_backend *be,
                                              const struct geist_tensor *a,
                                              const struct geist_tensor *b,
                                              struct geist_tensor *y) {
    if (vk_try_ew3(be, VK_PIPE_ADD, a, b, y)) {
        return GEIST_OK;
    }
    return vk_ew3_cpu(be, 0, a, b, y, "add");
}

[[nodiscard]] static enum geist_status vk_mul(struct geist_backend *be,
                                              const struct geist_tensor *a,
                                              const struct geist_tensor *b,
                                              struct geist_tensor *y) {
    if (vk_try_ew3(be, VK_PIPE_MUL, a, b, y)) {
        return GEIST_OK;
    }
    return vk_ew3_cpu(be, 1, a, b, y, "mul");
}

static constexpr float VK_GELU_K0 = 0.7978845608028654f; /* sqrt(2/pi) */
static constexpr float VK_GELU_K1 = 0.044715f;

[[nodiscard]] static enum geist_status vk_gelu_tanh(struct geist_backend *be,
                                                    const struct geist_tensor *x,
                                                    struct geist_tensor *y) {
    {
        const size_t           n = vk_t_n(x);
        VkDescriptorBufferInfo bi[2];
        uint32_t               off[2];
        if (VK_OPS(be, 2u) && n != 0 && n == vk_t_n(y) && vk_tensor_gpu(x, &bi[0], &off[0]) &&
            vk_tensor_gpu(y, &bi[1], &off[1])) {
            const uint32_t         push[4] = {(uint32_t) n, off[0], off[1], 0};
            const struct vk_access acc[2]  = {vk_acc_tensor(x, false), vk_acc_tensor(y, true)};
            if (vk_seq_dispatch_acc(be, VK_PIPE_GELU, bi, acc, push, sizeof(push),
                                    vk_groups(n), 1, 1) == GEIST_OK) {
                return GEIST_OK;
            }
        }
    }
    size_t       nx = 0, ny = 0;
    const float *xp = vk_tensor_host(x, &nx);
    float       *yp = vk_tensor_host(y, &ny);
    if (xp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan gelu_tanh: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    for (size_t i = 0; i < nx; i++) {
        const float v = xp[i];
        const float u = VK_GELU_K0 * (v + VK_GELU_K1 * v * v * v);
        yp[i]         = 0.5f * v * (1.0f + tanhf(u));
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_gelu_tanh_mul(struct geist_backend *be,
                                                        const struct geist_tensor *x,
                                                        const struct geist_tensor *z,
                                                        struct geist_tensor *y) {
    if (vk_try_ew3(be, VK_PIPE_GELU_MUL, x, z, y)) {
        return GEIST_OK;
    }
    return vk_ew3_cpu(be, 2, x, z, y, "gelu_tanh_mul");
}

[[nodiscard]] static enum geist_status vk_gelu_tanh_mul_scaled(struct geist_backend *be,
                                                               const struct geist_tensor *x,
                                                               const struct geist_tensor *z,
                                                               const float *scale,
                                                               struct geist_tensor *y) {
    size_t       nx = 0, nz = 0, ny = 0;
    const float *xp = vk_tensor_host(x, &nx);
    const float *zp = vk_tensor_host(z, &nz);
    float       *yp = vk_tensor_host(y, &ny);
    if (xp == nullptr || zp == nullptr || yp == nullptr || scale == nullptr || nx != nz ||
        nx != ny || y->ndim < 1) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan gelu_tanh_mul_scaled: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    const size_t feat = (size_t) y->shape[y->ndim - 1];
    if (feat == 0 || nx % feat != 0) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan gelu_tanh_mul_scaled: feature mismatch");
        return GEIST_E_INVALID_ARG;
    }
    const size_t rows = nx / feat;
    for (size_t r = 0; r < rows; r++) {
        const size_t base = r * feat;
        for (size_t j = 0; j < feat; j++) {
            const size_t i = base + j;
            const float  v = xp[i];
            const float  u = VK_GELU_K0 * (v + VK_GELU_K1 * v * v * v);
            yp[i]          = (0.5f * v * (1.0f + tanhf(u))) * zp[i] * scale[j];
        }
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_relu_squared(struct geist_backend *be,
                                                       const struct geist_tensor *x,
                                                       struct geist_tensor *y) {
    size_t       nx = 0, ny = 0;
    const float *xp = vk_tensor_host(x, &nx);
    float       *yp = vk_tensor_host(y, &ny);
    if (xp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan relu_squared: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    for (size_t i = 0; i < nx; i++) {
        const float v = xp[i] > 0.0f ? xp[i] : 0.0f;
        yp[i]         = v * v;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_silu(struct geist_backend *be,
                                               const struct geist_tensor *x,
                                               struct geist_tensor *y) {
    size_t       nx = 0, ny = 0;
    const float *xp = vk_tensor_host(x, &nx);
    float       *yp = vk_tensor_host(y, &ny);
    if (xp == nullptr || yp == nullptr || nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "vulkan silu: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    for (size_t i = 0; i < nx; i++) {
        const float v = xp[i];
        yp[i]         = v / (1.0f + expf(-v));
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_rmsnorm(struct geist_backend *be,
                                                  const struct geist_tensor *x,
                                                  const struct geist_tensor *w,
                                                  float eps,
                                                  struct geist_tensor *y) {
    {
        const size_t n    = vk_t_n(x);
        const size_t feat = n != 0 ? (size_t) x->shape[x->ndim - 1] : 0;
        VkDescriptorBufferInfo bi[3];
        uint32_t               off[3];
        if (VK_OPS(be, 4u) && feat != 0 && n % feat == 0 && vk_t_n(w) == feat &&
            vk_t_n(y) == n && vk_tensor_gpu(x, &bi[0], &off[0]) &&
            vk_tensor_gpu(w, &bi[1], &off[1]) && vk_tensor_gpu(y, &bi[2], &off[2])) {
            const struct {
                uint32_t rows, feat, x, w, y;
                float    eps;
            } push = {(uint32_t) (n / feat), (uint32_t) feat, off[0], off[1], off[2], eps};
            const struct vk_access acc[3] = {vk_acc_tensor(x, false), vk_acc_tensor(w, false),
                                             vk_acc_tensor(y, true)};
            if (vk_seq_dispatch_acc(be, VK_PIPE_RMSNORM, bi, acc, &push, sizeof(push),
                                    (uint32_t) (n / feat), 1, 1) == GEIST_OK) {
                return GEIST_OK;
            }
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

[[nodiscard]] static enum geist_status vk_rope_apply(struct geist_backend *be,
                                                     struct geist_tensor *x,
                                                     const struct geist_tensor *cos,
                                                     const struct geist_tensor *sin) {
    {
        VkDescriptorBufferInfo bi[3];
        uint32_t               off[3];
        if (VK_OPS(be, 8u) && x != nullptr && x->ndim == 3 && vk_t_n(x) != 0 && vk_t_n(cos) != 0 &&
            vk_t_n(sin) != 0 && vk_tensor_gpu(x, &bi[0], &off[0]) &&
            vk_tensor_gpu(cos, &bi[1], &off[1]) && vk_tensor_gpu(sin, &bi[2], &off[2])) {
            const size_t   seq   = (size_t) x->shape[0];
            const size_t   heads = (size_t) x->shape[1];
            const size_t   hd    = (size_t) x->shape[2];
            const size_t   pairs = seq * heads * hd / 2;
            const uint32_t push[6] = {(uint32_t) pairs, (uint32_t) heads, (uint32_t) hd,
                                      off[0], off[1], off[2]};
            const struct vk_access acc[3] = {vk_acc_tensor(x, true),
                                             vk_acc_tensor(cos, false),
                                             vk_acc_tensor(sin, false)};
            if (vk_seq_dispatch_acc(be, VK_PIPE_ROPE, bi, acc, push, sizeof(push),
                                    vk_groups(pairs), 1, 1) == GEIST_OK) {
                return GEIST_OK;
            }
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
    rope_apply(xp, cosp, sinp, (size_t) x->shape[0], (size_t) x->shape[1],
               (size_t) x->shape[2]);
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_embedding_lookup(struct geist_backend *be,
                                                           const struct geist_tensor *embed_table,
                                                           geist_token_t token_id,
                                                           struct geist_tensor *out) {
    size_t       n_table = 0, n_out = 0;
    const float *tablep = vk_tensor_host(embed_table, &n_table);
    float       *outp   = vk_tensor_host(out, &n_out);
    if (tablep == nullptr || outp == nullptr || embed_table->ndim != 2) {
        return GEIST_E_INVALID_ARG;
    }
    const int64_t vocab_size = embed_table->shape[0];
    const int64_t d_model    = embed_table->shape[1];
    if (token_id < 0 || (int64_t) token_id >= vocab_size || n_out != (size_t) d_model) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan embedding_lookup: token %d out of range",
                                (int) token_id);
        return GEIST_E_INVALID_ARG;
    }
    memcpy(outp, tablep + (size_t) token_id * (size_t) d_model,
           (size_t) d_model * sizeof(float));
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_attention(struct geist_backend *be,
                                                    const struct geist_tensor *q,
                                                    const struct geist_tensor *k,
                                                    const struct geist_tensor *v,
                                                    size_t q_offset,
                                                    size_t sliding_window,
                                                    struct geist_tensor *out) {
    /* Flash-decoding: n_q == 1 with f16 KV and enough context to make the
     * 8-workgroup direct kernel starve the GPU. Partials go into the
     * device x-ring; a combine pass reduces per head. */
    struct vk_state *stt = be->state;
    if (VK_OPS(be, 16u) && q != nullptr && k != nullptr && v != nullptr && out != nullptr &&
        q->ndim == 3 && q->shape[0] == 1 && k->dtype == GEIST_DTYPE_F16 &&
        (size_t) k->shape[0] > 192 && q->shape[2] <= 512 && vk_t_n(q) != 0 &&
        vk_t_n16(k) != 0 && stt->pipes[VK_PIPE_ATTN_PART_F16] != VK_NULL_HANDLE) {
        const uint32_t qh   = (uint32_t) q->shape[1];
        const uint32_t hd   = (uint32_t) q->shape[2];
        const uint32_t n_kv = (uint32_t) k->shape[0];
        const uint32_t kvh  = (uint32_t) k->shape[1];
        const uint32_t n_chunks = (n_kv + 127u) / 128u;
        const size_t   part_bytes = (size_t) qh * n_chunks * (hd + 2u) * 4u;
        VkDescriptorBufferInfo bq, bk, bv, bo;
        uint32_t               qo, ko, vo, oo;
        if (stt->xring == nullptr &&
            vk_buffer_create(be, VK_XRING_CAP, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE,
                             &stt->xring) != GEIST_OK) {
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
        const uint32_t po = (uint32_t) (stt->xring_used / 4u);
        const uint32_t push1[11] = {n_kv, qh, kvh, hd, (uint32_t) q_offset,
                                    (uint32_t) sliding_window, qo, ko, vo, po, n_chunks};
        VkDescriptorBufferInfo bi1[4] = {bq, bk, bv,
                                         {.buffer = stt->xring->buf, .range = VK_WHOLE_SIZE}};
        const struct vk_access acc1[4] = {vk_acc_tensor(q, false), vk_acc_tensor16(k, false),
                                          vk_acc_tensor16(v, false),
                                          vk_acc(stt->xring_used, part_bytes, true)};
        if (vk_seq_dispatch_acc(be, VK_PIPE_ATTN_PART_F16, bi1, acc1, push1, sizeof(push1),
                                n_chunks, qh, 1) != GEIST_OK) {
            goto attn_generic;
        }
        const uint32_t push2[5] = {qh, hd, n_chunks, po, oo};
        VkDescriptorBufferInfo bi2[2] = {{.buffer = stt->xring->buf, .range = VK_WHOLE_SIZE},
                                         bo};
        const struct vk_access acc2[2] = {vk_acc(stt->xring_used, part_bytes, false),
                                          vk_acc_tensor(out, true)};
        stt->xring_used = (stt->xring_used + part_bytes + 63u) & ~(size_t) 63u;
        if (vk_seq_dispatch_acc(be, VK_PIPE_ATTN_COMB, bi2, acc2, push2, sizeof(push2), qh, 1,
                                1) == GEIST_OK) {
            return GEIST_OK;
        }
    }
attn_generic:;
    {
        VkDescriptorBufferInfo bi[4];
        uint32_t               off[4];
        const bool kv16 = k != nullptr && k->dtype == GEIST_DTYPE_F16;
        if (VK_OPS(be, 16u) && q != nullptr && k != nullptr && q->ndim == 3 && k->ndim == 3 && vk_t_n(q) != 0 &&
            (kv16 ? (vk_t_n16(k) != 0 && vk_t_n16(v) != 0 &&
                     vk_tensor_gpu_f16(k, &bi[1], &off[1]) &&
                     vk_tensor_gpu_f16(v, &bi[2], &off[2]))
                  : (vk_t_n(k) != 0 && vk_t_n(v) != 0 && vk_tensor_gpu(k, &bi[1], &off[1]) &&
                     vk_tensor_gpu(v, &bi[2], &off[2]))) &&
            vk_t_n(out) != 0 &&
            vk_tensor_gpu(q, &bi[0], &off[0]) && vk_tensor_gpu(out, &bi[3], &off[3]) &&
            /* f16 kernel streams q/k/v as 4-wide vectors */
            (!kv16 || ((off[0] | off[1] | off[2]) & 3u) == 0u)) {
            const uint32_t n_q  = (uint32_t) q->shape[0];
            const uint32_t qh   = (uint32_t) q->shape[1];
            const uint32_t hd   = (uint32_t) q->shape[2];
            const uint32_t n_kv = (uint32_t) k->shape[0];
            const uint32_t kvh  = (uint32_t) k->shape[1];
            const uint32_t push[11] = {n_q,
                                       n_kv,
                                       qh,
                                       kvh,
                                       hd,
                                       (uint32_t) q_offset,
                                       (uint32_t) sliding_window,
                                       off[0],
                                       off[1],
                                       off[2],
                                       off[3]};
            const struct vk_access acc[4] = {
                    vk_acc_tensor(q, false),
                    kv16 ? vk_acc_tensor16(k, false) : vk_acc_tensor(k, false),
                    kv16 ? vk_acc_tensor16(v, false) : vk_acc_tensor(v, false),
                    vk_acc_tensor(out, true)};
            if (vk_seq_dispatch_acc(be, kv16 ? VK_PIPE_ATTENTION_F16 : VK_PIPE_ATTENTION, bi,
                                    acc, push, sizeof(push), n_q, qh, 1) == GEIST_OK) {
                return GEIST_OK;
            }
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
    attention_mqa_causal_kv(qp, kp, vp, (size_t) q->shape[0], (size_t) k->shape[0], q_offset,
                            (size_t) q->shape[1], (size_t) k->shape[1],
                            (size_t) q->shape[2], sliding_window, op);
    return GEIST_OK;
}

/* ---- New batched-submit ops (Phase 3) --------------------------------- */

[[nodiscard]] static enum geist_status vk_rmsnorm_add(struct geist_backend *be,
                                                      const struct geist_tensor *res,
                                                      const struct geist_tensor *x,
                                                      const struct geist_tensor *w,
                                                      float eps,
                                                      struct geist_tensor *y) {
    const size_t n    = vk_t_n(x);
    const size_t feat = n != 0 ? (size_t) x->shape[x->ndim - 1] : 0;
    {
        VkDescriptorBufferInfo bi[4];
        uint32_t               off[4];
        if (VK_OPS(be, 4u) && feat != 0 && n % feat == 0 && vk_t_n(w) == feat &&
            vk_t_n(res) == n && vk_t_n(y) == n && vk_tensor_gpu(x, &bi[0], &off[0]) &&
            vk_tensor_gpu(w, &bi[1], &off[1]) && vk_tensor_gpu(res, &bi[2], &off[2]) &&
            vk_tensor_gpu(y, &bi[3], &off[3])) {
            const struct {
                uint32_t rows, feat, x, w, r, y;
                float    eps;
            } push = {(uint32_t) (n / feat), (uint32_t) feat, off[0], off[1], off[2],
                      off[3],                eps};
            const struct vk_access acc[4] = {vk_acc_tensor(x, false), vk_acc_tensor(w, false),
                                             vk_acc_tensor(res, false),
                                             vk_acc_tensor(y, true)};
            if (vk_seq_dispatch_acc(be, VK_PIPE_RMSNORM_ADD, bi, acc, &push, sizeof(push),
                                    (uint32_t) (n / feat), 1, 1) == GEIST_OK) {
                return GEIST_OK;
            }
        }
    }
    /* CPU fallback: y = res + rmsnorm(x) * w */
    size_t       nx = 0, nw = 0, nr = 0, ny = 0;
    const float *xp = vk_tensor_host(x, &nx);
    const float *wp = vk_tensor_host(w, &nw);
    const float *rp = vk_tensor_host(res, &nr);
    float       *yp = vk_tensor_host(y, &ny);
    if (xp == nullptr || wp == nullptr || rp == nullptr || yp == nullptr || nx != ny ||
        nr != nx || feat == 0 || nw != feat || nx % feat != 0) {
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

[[nodiscard]] static enum geist_status vk_scale_f32(struct geist_backend *be,
                                                    const struct geist_tensor *x,
                                                    float scale,
                                                    struct geist_tensor *y) {
    const size_t n = vk_t_n(x);
    {
        VkDescriptorBufferInfo bi[2];
        uint32_t               off[2];
        if (n != 0 && n == vk_t_n(y) && vk_tensor_gpu(x, &bi[0], &off[0]) &&
            vk_tensor_gpu(y, &bi[1], &off[1])) {
            const struct {
                uint32_t n, x, y;
                float    scale;
            } push = {(uint32_t) n, off[0], off[1], scale};
            const struct vk_access acc[2] = {vk_acc_tensor(x, false), vk_acc_tensor(y, true)};
            if (vk_seq_dispatch_acc(be, VK_PIPE_SCALE, bi, acc, &push, sizeof(push),
                                    vk_groups(n), 1, 1) == GEIST_OK) {
                return GEIST_OK;
            }
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

[[nodiscard]] static enum geist_status vk_argmax_f32(struct geist_backend *be,
                                                     const struct geist_tensor *logits,
                                                     int32_t *out_index) {
    struct vk_state *st = be->state;
    if (!VK_OPS(be, 128u)) {
        return GEIST_E_UNSUPPORTED;
    }
    const size_t     n  = vk_t_n(logits);
    VkDescriptorBufferInfo bi[2];
    uint32_t               off[2];
    if (n == 0 || out_index == nullptr || !vk_tensor_gpu(logits, &bi[0], &off[0])) {
        return GEIST_E_UNSUPPORTED; /* arch scans on the host */
    }
    if (st->argmax_out == nullptr) {
        if (vk_buffer_create(be, 16, GEIST_BUFFER_STAGING, GEIST_MEMORY_AUTO,
                             &st->argmax_out) != GEIST_OK) {
            return GEIST_E_UNSUPPORTED;
        }
    }
    bi[1] = (VkDescriptorBufferInfo) {.buffer = st->argmax_out->buf, .range = VK_WHOLE_SIZE};
    const uint32_t         push[3] = {(uint32_t) n, off[0], 0};
    const struct vk_access acc[2]  = {vk_acc_tensor(logits, false), vk_acc_all(true)};
    enum geist_status      s =
            vk_seq_dispatch_acc(be, VK_PIPE_ARGMAX, bi, acc, push, sizeof(push), 1, 1, 1);
    if (s != GEIST_OK) {
        return GEIST_E_UNSUPPORTED;
    }
    vk_seq_flush(st); /* the one intended sync point per decoded token */
    *out_index = ((const int32_t *) st->argmax_out->mapped)[0];
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vk_buffer_copy(struct geist_buffer *dst,
                                                      size_t dst_offset,
                                                      const struct geist_buffer *src,
                                                      size_t src_offset,
                                                      size_t n_bytes) {
    if (dst == nullptr || src == nullptr || n_bytes == 0) {
        return GEIST_E_INVALID_ARG;
    }
    struct vk_state *st = dst->owner;
    if ((st->gpu_ops & 32u) != 0 && dst->buf != VK_NULL_HANDLE && src->buf != VK_NULL_HANDLE) {
        /* On-device copy appended to the sequence — keeps KV appends from
         * breaking the per-token batch (kv_store.c uses this path). */
        enum geist_status s = vk_seq_open_cmd(st);
        if (s != GEIST_OK) {
            return s;
        }
        const VkDescriptorBufferInfo cinf[2] = {{.buffer = src->buf}, {.buffer = dst->buf}};
        const struct vk_access       cacc[2] = {
                vk_acc(src->base_off + src_offset, n_bytes, false),
                vk_acc(dst->base_off + dst_offset, n_bytes, true)};
        vk_seq_hazard(st, cinf, cacc, 2);
        const VkBufferCopy region = {.srcOffset = src->base_off + src_offset,
                                     .dstOffset = dst->base_off + dst_offset,
                                     .size      = n_bytes};
        st->fn.CmdCopyBuffer(st->seq_cmd, src->buf, dst->buf, 1, &region);
        st->seq_dispatches++;
        st->seq_in_cmd++;
        vk_prof_stamp(st, VK_PIPE_COUNT);
        return GEIST_OK;
    }
    /* Host fallback. */
    vk_seq_flush(st);
    uint8_t       *d = dst->host_alias != nullptr ? dst->host_alias : dst->mapped;
    const uint8_t *sp = src->host_alias != nullptr ? src->host_alias : src->mapped;
    if (d == nullptr || sp == nullptr || dst_offset + n_bytes > dst->bytes ||
        src_offset + n_bytes > src->bytes) {
        return GEIST_E_UNSUPPORTED;
    }
    memcpy(d + dst_offset, sp + src_offset, n_bytes);
    return GEIST_OK;
}


/* Stage x into the device-local ring (one in-batch CmdCopyBuffer). Returns
 * the ring element offset; false = not stageable (caller falls back). */
[[nodiscard]] static bool vk_xring_stage(struct geist_backend      *be,
                                         const struct geist_tensor *t_x,
                                         size_t                     m,
                                         size_t                     n_in,
                                         uint32_t                  *out_elem_off) {
    struct vk_state       *st = be->state;
    VkDescriptorBufferInfo src_bi;
    uint32_t               src_elem;
    if (!vk_tensor_gpu(t_x, &src_bi, &src_elem)) {
        return false;
    }
    const size_t bytes = m * n_in * sizeof(float);
    if (st->xring == nullptr &&
        vk_buffer_create(be, VK_XRING_CAP, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE,
                         &st->xring) != GEIST_OK) {
        return false;
    }
    if (bytes > st->xring->bytes) {
        return false;
    }
    if (st->xring_used + bytes > st->xring->bytes) {
        vk_seq_flush(st); /* drains the batch and resets the ring */
    }
    if (vk_seq_open_cmd(st) != GEIST_OK) {
        return false;
    }
    {
        const VkDescriptorBufferInfo cinf[2] = {{.buffer = t_x->buffer->buf},
                                                {.buffer = st->xring->buf}};
        const struct vk_access       cacc[2] = {vk_acc_tensor(t_x, false),
                                                vk_acc(st->xring_used, bytes, true)};
        vk_seq_hazard(st, cinf, cacc, 2);
    }
    const size_t x_stride = t_x->ndim >= 2 ? (size_t) t_x->stride[t_x->ndim - 2] : n_in;
    const size_t src_byte = t_x->buffer->base_off + t_x->offset;
    if (m == 1 || x_stride == n_in) {
        const VkBufferCopy r = {.srcOffset = src_byte,
                                .dstOffset = st->xring_used,
                                .size      = m == 1 ? n_in * sizeof(float) : bytes};
        st->fn.CmdCopyBuffer(st->seq_cmd, t_x->buffer->buf, st->xring->buf, 1, &r);
    } else {
        VkBufferCopy regions[64];
        for (size_t r0 = 0; r0 < m; r0 += 64) {
            const uint32_t nr = (uint32_t) (m - r0 > 64 ? 64 : m - r0);
            for (uint32_t i = 0; i < nr; ++i) {
                regions[i] = (VkBufferCopy) {
                        .srcOffset = src_byte + (r0 + i) * x_stride * sizeof(float),
                        .dstOffset = st->xring_used + (r0 + i) * n_in * sizeof(float),
                        .size      = n_in * sizeof(float)};
            }
            st->fn.CmdCopyBuffer(st->seq_cmd, t_x->buffer->buf, st->xring->buf, nr, regions);
        }
    }
    st->seq_dispatches++;
    st->seq_in_cmd++;
    vk_prof_stamp(st, VK_PIPE_COUNT);
    *out_elem_off = (uint32_t) (st->xring_used / sizeof(float));
    st->xring_used = (st->xring_used + bytes + 63) & ~(size_t) 63;
    return true;
}

/* Tensor-path linear: x staged into the VRAM ring, weight from the VRAM
 * registry, y written to its host-visible home — no host round-trip, no
 * flush. THE hot path since Phase 3. */
[[nodiscard]] static enum geist_status vk_linear_t(struct geist_backend      *be,
                                                   const struct geist_tensor *t_x,
                                                   const struct geist_weight *w,
                                                   const struct geist_tensor *t_w,
                                                   size_t                     m,
                                                   struct geist_tensor       *t_y) {
    (void) t_w;
    struct vk_state *st = be->state;
    if (!VK_OPS(be, 1u)) {
        return GEIST_E_UNSUPPORTED;
    }
    enum vk_pipe     mv, mm;
    switch ((enum geist_dtype) w->dtype) {
    case GEIST_DTYPE_Q4_K: mv = VK_PIPE_MATVEC_Q4K; mm = VK_PIPE_MATMUL_Q4K; break;
    case GEIST_DTYPE_Q6_K: mv = VK_PIPE_MATVEC_Q6K; mm = VK_PIPE_MATMUL_Q6K; break;
    case GEIST_DTYPE_F32: mv = VK_PIPE_MATVEC_F32; mm = VK_PIPE_MATMUL_F32; break;
    default: return GEIST_E_UNSUPPORTED;
    }
    struct geist_buffer *wbuf = vk_weight_lookup(st, w->raw);
    VkDescriptorBufferInfo bi[3];
    uint32_t               xo, yo;
    if (wbuf == nullptr || m == 0 || vk_t_n(t_x) == 0 || vk_t_n(t_y) == 0 ||
        !vk_tensor_gpu(t_y, &bi[2], &yo)) {
        return GEIST_E_UNSUPPORTED;
    }
    const uint32_t n_in  = (uint32_t) w->n_in;
    const uint32_t n_out = (uint32_t) w->n_out;
    uint32_t       x_stride;
    if (t_x->buffer->device_mem) {
        /* BAR-resident activations: bind in place, no staging copy. */
        if (!vk_tensor_gpu(t_x, &bi[0], &xo)) {
            return GEIST_E_UNSUPPORTED;
        }
        x_stride = t_x->ndim >= 2 ? (uint32_t) t_x->stride[t_x->ndim - 2] : n_in;
    } else {
        if (!vk_xring_stage(be, t_x, m, n_in, &xo)) {
            return GEIST_E_UNSUPPORTED;
        }
        bi[0]    = (VkDescriptorBufferInfo) {.buffer = st->xring->buf,
                                             .range  = VK_WHOLE_SIZE};
        x_stride = n_in; /* ring copy is contiguous */
    }
    bi[1] = (VkDescriptorBufferInfo) {.buffer = wbuf->buf, .range = VK_WHOLE_SIZE};
    const uint32_t y_stride = t_y->ndim >= 2 ? (uint32_t) t_y->stride[t_y->ndim - 2]
                                             : n_out;
    const struct vk_push push = {.n_in           = n_in,
                                 .n_out          = n_out,
                                 .blocks_per_row = n_in / 256,
                                 .rows           = (uint32_t) m,
                                 .x_offset       = xo,
                                 .y_offset       = yo,
                                 .x_stride       = x_stride,
                                 .y_stride       = y_stride};
    enum vk_pipe lpipe = m == 1 ? mv : mm;
    uint32_t     gx    = vk_linear_gx(lpipe, n_out);
    uint32_t     gy    = vk_linear_gy(lpipe, (uint32_t) m);
    /* Tensor-core path for conforming GEMMs (shaders assume w_offset == 0,
     * which holds for all registry uploads). */
    if (m > 1) {
        vk_linear_cm_route(st, &lpipe, (uint32_t) m, n_out, &gx, &gy);
    }
    const struct vk_access acc[3] = {
            t_x->buffer->device_mem ? vk_acc_tensor(t_x, false)
                                    : vk_acc((uint64_t) xo * 4u, (uint64_t) m * n_in * 4u,
                                             false),
            vk_acc_all(false), vk_acc_tensor(t_y, true)};
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
    /* Two appended dispatches; a fused two-weight kernel is a Phase-3c
     * candidate once the profiler ranks it. Check both up front so the
     * fallback never sees a half-done pair. */
    struct vk_state *st = be->state;
    if (vk_weight_lookup(st, w0->raw) == nullptr ||
        vk_weight_lookup(st, w1->raw) == nullptr) {
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
    if (!VK_OPS(be, 64u)) {
        return GEIST_E_UNSUPPORTED;
    }
    if (embed_table == nullptr || out == nullptr || embed_table->ndim != 2 ||
        embed_table->buffer == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    const int64_t vocab = embed_table->shape[0];
    const int64_t d     = embed_table->shape[1];
    if (token_id < 0 || (int64_t) token_id >= vocab) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan embed_scaled: token %d out of range", (int) token_id);
        return GEIST_E_INVALID_ARG;
    }
    uint32_t dtype_code;
    switch ((enum geist_dtype) embed_table->dtype) {
    case GEIST_DTYPE_F32: dtype_code = 0; break;
    case GEIST_DTYPE_F16: dtype_code = 1; break;
    case GEIST_DTYPE_BF16: dtype_code = 2; break;
    case GEIST_DTYPE_Q4_K: dtype_code = 8; break;
    case GEIST_DTYPE_Q5_K: dtype_code = 9; break;
    case GEIST_DTYPE_Q6_K: dtype_code = 10; break;
    default: return GEIST_E_UNSUPPORTED;
    }
    /* Table bytes: prefer the resolve-time VRAM copy (embed tables go
     * through resolve_weight); fall back to a bindable host region. */
    const uint8_t *host = embed_table->buffer->host_alias != nullptr
                                  ? (const uint8_t *) embed_table->buffer->host_alias +
                                            embed_table->offset
                                  : nullptr;
    struct geist_buffer   *wbuf = host != nullptr ? vk_weight_lookup(st, host) : nullptr;
    VkDescriptorBufferInfo bi[2];
    uint32_t               w_elem_off = 0;
    if (wbuf != nullptr) {
        bi[0] = (VkDescriptorBufferInfo) {.buffer = wbuf->buf, .range = VK_WHOLE_SIZE};
    } else {
        if (dtype_code == 10) {
            /* Q6_K GPU copies are repacked to 216-byte blocks; the arena
             * holds canonical 210-byte blocks the shader can't read. */
            return GEIST_E_UNSUPPORTED;
        }
        struct geist_tensor bytes_view = *embed_table;
        bytes_view.dtype               = GEIST_DTYPE_F32; /* only for the offset calc */
        if (embed_table->buffer->buf == VK_NULL_HANDLE ||
            (embed_table->buffer->base_off + embed_table->offset) % 4 != 0) {
            return GEIST_E_UNSUPPORTED;
        }
        bi[0] = (VkDescriptorBufferInfo) {.buffer = embed_table->buffer->buf,
                                          .range  = VK_WHOLE_SIZE};
        w_elem_off = (uint32_t) (embed_table->buffer->base_off + embed_table->offset);
        (void) bytes_view;
    }
    uint32_t yo;
    if (vk_t_n(out) != (size_t) d || !vk_tensor_gpu(out, &bi[1], &yo)) {
        return GEIST_E_UNSUPPORTED;
    }
    uint32_t bpr;
    switch (dtype_code) {
    case 8: bpr = (uint32_t) (d / 256); break;
    case 9: bpr = (uint32_t) (d / 256); break;
    case 10: bpr = (uint32_t) (d / 256); break;
    default: bpr = 0; break;
    }
    const struct {
        uint32_t n_in, token, dtype, bpr, w_byte, y;
        float    scale;
    } push = {(uint32_t) d, (uint32_t) token_id, dtype_code, bpr, w_elem_off, yo, scale};
    const struct vk_access acc[2] = {vk_acc_all(false), vk_acc_tensor(out, true)};
    return vk_seq_dispatch_acc(be, VK_PIPE_EMBED, bi, acc, &push, sizeof(push),
                               vk_groups((size_t) d), 1, 1);
}

/* Fused decode FFN front (m == 1, both weights Q4_K): one dispatch for
 * gelu(x.gate^T) * (x.up^T) — replaces two matvecs + gelu_mul. */
[[nodiscard]] static enum geist_status vk_ffn_gate_up(struct geist_backend      *be,
                                                      const struct geist_tensor *t_x,
                                                      const struct geist_tensor *gate_w,
                                                      const struct geist_tensor *up_w,
                                                      struct geist_tensor       *y) {
    struct vk_state *st = be->state;
    if (!VK_OPS(be, 1u) || gate_w == nullptr || up_w == nullptr || t_x == nullptr ||
        gate_w->dtype != GEIST_DTYPE_Q4_K || up_w->dtype != GEIST_DTYPE_Q4_K ||
        gate_w->ndim != 2 || t_x->shape[0] != 1) {
        return GEIST_E_UNSUPPORTED;
    }
    const uint32_t n_out = (uint32_t) gate_w->shape[0];
    const uint32_t n_in  = (uint32_t) gate_w->shape[1];
    if (n_in % 256u != 0u || up_w->shape[0] != gate_w->shape[0] ||
        up_w->shape[1] != gate_w->shape[1]) {
        return GEIST_E_UNSUPPORTED;
    }
    const uint8_t *g_host = gate_w->buffer != nullptr && gate_w->buffer->host_alias != nullptr
                                    ? (const uint8_t *) gate_w->buffer->host_alias +
                                              gate_w->offset
                                    : nullptr;
    const uint8_t *u_host = up_w->buffer != nullptr && up_w->buffer->host_alias != nullptr
                                    ? (const uint8_t *) up_w->buffer->host_alias + up_w->offset
                                    : nullptr;
    struct geist_buffer *gbuf = g_host != nullptr ? vk_weight_lookup(st, g_host) : nullptr;
    struct geist_buffer *ubuf = u_host != nullptr ? vk_weight_lookup(st, u_host) : nullptr;
    VkDescriptorBufferInfo bi[4];
    uint32_t               xo, yo;
    if (gbuf == nullptr || ubuf == nullptr || vk_t_n(y) < n_out ||
        !vk_tensor_gpu(y, &bi[3], &yo)) {
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
    const struct vk_push push = {.n_in           = n_in,
                                 .n_out          = n_out,
                                 .blocks_per_row = n_in / 256u,
                                 .rows           = 1,
                                 .x_offset       = xo,
                                 .y_offset       = yo,
                                 .x_stride       = n_in,
                                 .y_stride       = n_out};
    const struct vk_access acc[4] = {vk_acc_tensor(t_x, false), vk_acc_all(false),
                                     vk_acc_all(false), vk_acc_tensor(y, true)};
    return vk_seq_dispatch_acc(be, VK_PIPE_FFN_GATE_UP, bi, acc, &push, sizeof(push),
                               (n_out + 3u) / 4u, 1, 1);
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
    if (!VK_OPS(be, 1u) || gate_w == nullptr || up_w == nullptr || t_x == nullptr ||
        norm_w == nullptr || gate_w->dtype != GEIST_DTYPE_Q4_K ||
        up_w->dtype != GEIST_DTYPE_Q4_K || gate_w->ndim != 2 || t_x->shape[0] != 1) {
        return GEIST_E_UNSUPPORTED;
    }
    const uint32_t n_out = (uint32_t) gate_w->shape[0];
    const uint32_t n_in  = (uint32_t) gate_w->shape[1];
    if (n_in % 256u != 0u || n_out % 8u != 0u || up_w->shape[0] != gate_w->shape[0] ||
        up_w->shape[1] != gate_w->shape[1] || vk_t_n(norm_w) != n_in) {
        return GEIST_E_UNSUPPORTED;
    }
    const uint8_t *g_host = gate_w->buffer != nullptr && gate_w->buffer->host_alias != nullptr
                                    ? (const uint8_t *) gate_w->buffer->host_alias +
                                              gate_w->offset
                                    : nullptr;
    const uint8_t *u_host = up_w->buffer != nullptr && up_w->buffer->host_alias != nullptr
                                    ? (const uint8_t *) up_w->buffer->host_alias + up_w->offset
                                    : nullptr;
    struct geist_buffer *gbuf = g_host != nullptr ? vk_weight_lookup(st, g_host) : nullptr;
    struct geist_buffer *ubuf = u_host != nullptr ? vk_weight_lookup(st, u_host) : nullptr;
    VkDescriptorBufferInfo bi[5];
    uint32_t               xo, nwo, yo;
    if (gbuf == nullptr || ubuf == nullptr || vk_t_n(y) < n_out ||
        !vk_tensor_gpu(y, &bi[4], &yo) || !vk_tensor_gpu(t_x, &bi[0], &xo) ||
        !vk_tensor_gpu(norm_w, &bi[3], &nwo) || (xo & 3u) != 0u || (nwo & 3u) != 0u) {
        return GEIST_E_UNSUPPORTED;
    }
    bi[1] = (VkDescriptorBufferInfo) {.buffer = gbuf->buf, .range = VK_WHOLE_SIZE};
    bi[2] = (VkDescriptorBufferInfo) {.buffer = ubuf->buf, .range = VK_WHOLE_SIZE};
    const struct {
        uint32_t n_in, n_out, blocks_per_row, x_offset, nw_offset, y_offset;
        float    eps;
    } push = {n_in, n_out, n_in / 256u, xo, nwo, yo, eps};
    const struct vk_access acc[5] = {vk_acc_tensor(t_x, false), vk_acc_all(false),
                                     vk_acc_all(false), vk_acc_tensor(norm_w, false),
                                     vk_acc_tensor(y, true)};
    return vk_seq_dispatch_acc(be, VK_PIPE_FFN_NORM_GU, bi, acc, &push, sizeof(push),
                               n_out / 8u, 1, 1);
}

/* Gemma-3n PLE block in THREE dispatches (replaces gate matvec +
 * gelu_mul + proj matvec + rmsnorm_add): the gate GEMV gets the gelu*ple
 * epilogue folded in; the proj tail keeps the multi-workgroup matvec +
 * rmsnorm_add pair. A single-workgroup proj+norm fusion was measured at
 * 68 us vs 19 us for the pair — one SM streaming the 1.5 MB proj weight
 * is a bandwidth wall, so the norm's full-vector reduction stays a
 * separate dispatch. Decode only (rows == 1), F32 weights. */
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
    if (!VK_OPS(be, 1u) || x == nullptr || gate_w == nullptr || proj_w == nullptr ||
        x->ndim != 2 || x->shape[0] != 1 || gate_w->dtype != GEIST_DTYPE_F32 ||
        proj_w->dtype != GEIST_DTYPE_F32 || gate_w->ndim != 2 || proj_w->ndim != 2) {
        return GEIST_E_UNSUPPORTED;
    }
    const uint32_t d_in = (uint32_t) gate_w->shape[1];
    const uint32_t hpl  = (uint32_t) gate_w->shape[0];
    const uint32_t feat = (uint32_t) proj_w->shape[0];
    if ((uint32_t) x->shape[1] != d_in || (uint32_t) proj_w->shape[1] != hpl ||
        vk_t_n(norm_w) != feat || vk_t_n(res) < feat || vk_t_n(y) < feat ||
        vk_t_n(gate_scratch) < hpl || vk_t_n(proj_scratch) < feat) {
        return GEIST_E_UNSUPPORTED;
    }
    const uint8_t *g_host = gate_w->buffer != nullptr && gate_w->buffer->host_alias != nullptr
                                    ? (const uint8_t *) gate_w->buffer->host_alias +
                                              gate_w->offset
                                    : nullptr;
    const uint8_t *p_host = proj_w->buffer != nullptr && proj_w->buffer->host_alias != nullptr
                                    ? (const uint8_t *) proj_w->buffer->host_alias +
                                              proj_w->offset
                                    : nullptr;
    struct geist_buffer *gwbuf = g_host != nullptr ? vk_weight_lookup(st, g_host) : nullptr;
    struct geist_buffer *pwbuf = p_host != nullptr ? vk_weight_lookup(st, p_host) : nullptr;
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
        } push = {d_in, xo, po, gso};
        const struct vk_access acc[4] = {vk_acc_tensor(x, false), vk_acc_all(false),
                                         vk_acc_tensor(ple_in, false),
                                         vk_acc_tensor(gate_scratch, true)};
        enum geist_status s = vk_seq_dispatch_acc(be, VK_PIPE_PLE_GATE, bi, acc, &push,
                                                  sizeof(push), hpl, 1, 1);
        if (s != GEIST_OK) {
            return s;
        }
    }
    {
        const VkDescriptorBufferInfo bi[3] = {
                b_gs, {.buffer = pwbuf->buf, .range = VK_WHOLE_SIZE}, b_ps};
        const struct vk_push push = {.n_in     = hpl,
                                     .n_out    = feat,
                                     .rows     = 1,
                                     .x_offset = gso,
                                     .y_offset = pso,
                                     .x_stride = hpl,
                                     .y_stride = feat};
        const struct vk_access acc[3] = {vk_acc_tensor(gate_scratch, false), vk_acc_all(false),
                                         vk_acc_tensor(proj_scratch, true)};
        enum geist_status s = vk_seq_dispatch_acc(be, VK_PIPE_MATVEC_F32, bi, acc, &push,
                                                  sizeof(push), feat, 1, 1);
        if (s != GEIST_OK) {
            return s;
        }
    }
    return vk_rmsnorm_add(be, res, proj_scratch, norm_w, eps, y);
}

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
    if (!VK_OPS(be, 4u) || q == nullptr || q->ndim != 3 || q_norm_w == nullptr ||
        cos == nullptr || sin == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    const bool has_kv = k != nullptr;
    const bool kv16   = has_kv && k_cache != nullptr && k_cache->dtype == GEIST_DTYPE_F16;
    if (has_kv && (v == nullptr || k_norm_w == nullptr || v_norm_w == nullptr ||
                   k_cache == nullptr || v_cache == nullptr ||
                   (!kv16 && (k_cache->dtype != GEIST_DTYPE_F32 ||
                              v_cache->dtype != GEIST_DTYPE_F32)))) {
        return GEIST_E_UNSUPPORTED;
    }
    const uint32_t seq = (uint32_t) q->shape[0];
    const uint32_t qh  = (uint32_t) q->shape[1];
    const uint32_t hd  = (uint32_t) q->shape[2];
    if (hd > 512u || vk_t_n(q) == 0) {
        return GEIST_E_UNSUPPORTED;
    }
    VkDescriptorBufferInfo bi[6];
    uint32_t qo, ko = 0, vo = 0, qwo, kwo = 0, vwo = 0, co, so, kco = 0, vco = 0;
    VkDescriptorBufferInfo tmp;
    if (!vk_tensor_gpu(q, &bi[0], &qo) || !vk_tensor_gpu(q_norm_w, &bi[1], &qwo) ||
        !vk_tensor_gpu(cos, &bi[2], &co) || !vk_tensor_gpu(sin, &bi[3], &so)) {
        return GEIST_E_UNSUPPORTED;
    }
    uint32_t kvh = 1;
    if (has_kv) {
        VkDescriptorBufferInfo b_k, b_v, b_kw, b_vw;
        const bool caches_ok =
                kv16 ? (vk_tensor_gpu_f16(k_cache, &bi[4], &kco) &&
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
        kvh = (uint32_t) k->shape[1];
    } else {
        bi[4] = bi[0]; /* unused bindings — anything valid */
        bi[5] = bi[0];
    }
    (void) tmp;
    const struct {
        uint32_t seq, qh, kvh, hd, q_position, has_kv;
        uint32_t qo, ko, vo, qwo, kwo, vwo, co, so, kco, vco;
        float    eps;
    } push = {seq, qh,  kvh, hd,  (uint32_t) q_position,
              has_kv ? 1u : 0u, qo,  ko,  vo,  qwo, kwo, vwo, co, so, kco, vco, eps};
    struct vk_access a0 = vk_acc_tensor(q, true);
    if (has_kv) {
        const struct vk_access ak = vk_acc_tensor(k, true);
        const struct vk_access av = vk_acc_tensor(v, true);
        a0.lo = a0.lo < ak.lo ? a0.lo : ak.lo;
        a0.lo = a0.lo < av.lo ? a0.lo : av.lo;
        a0.hi = a0.hi > ak.hi ? a0.hi : ak.hi;
        a0.hi = a0.hi > av.hi ? a0.hi : av.hi;
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
    return vk_seq_dispatch_acc(be, kv16 ? VK_PIPE_QKV_PREP_F16 : VK_PIPE_QKV_PREP, bi, acc,
                               &push, sizeof(push), seq, qh, 3);
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
    const size_t seq    = (size_t) k_src->shape[0];
    const size_t kv_row = (size_t) (k_src->shape[1] * k_src->shape[2]);
    const size_t n      = seq * kv_row;
    VkDescriptorBufferInfo bi[4];
    uint32_t               kso, vso, kdo, vdo;
    if (n == 0 || vk_t_n(k_src) == 0 || vk_t_n(v_src) == 0 ||
        !vk_tensor_gpu(k_src, &bi[0], &kso) || !vk_tensor_gpu(v_src, &bi[1], &vso) ||
        !vk_tensor_gpu_f16(k_cache, &bi[2], &kdo) ||
        !vk_tensor_gpu_f16(v_cache, &bi[3], &vdo)) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "vulkan kv_append_f16: bad inputs");
        return GEIST_E_UNSUPPORTED;
    }
    const uint32_t push[5] = {(uint32_t) n, kso, vso,
                              kdo + (uint32_t) (q_position * kv_row),
                              vdo + (uint32_t) (q_position * kv_row)};
    const struct vk_access acc[4] = {
            vk_acc_tensor(k_src, false), vk_acc_tensor(v_src, false),
            vk_acc(k_cache->buffer->base_off + k_cache->offset + q_position * kv_row * 2,
                   n * 2, true),
            vk_acc(v_cache->buffer->base_off + v_cache->offset + q_position * kv_row * 2,
                   n * 2, true)};
    return vk_seq_dispatch_acc(be, VK_PIPE_KV_APPEND_F16, bi, acc, push, sizeof(push),
                               vk_groups(n), 1, 1);
}

/* ====================================================================== */
/* Capability                                                              */
/* ====================================================================== */

static enum geist_support vk_supports_op(struct geist_backend                *be,
                                         const struct geist_op_support_query *query) {
    (void) be;
    if (query == nullptr || query->op != GEIST_OP_LINEAR || query->input_count < 2) {
        return GEIST_SUPPORT_NONE;
    }
    const struct geist_tensor_format *x_fmt = &query->inputs[0];
    const struct geist_tensor_format *w_fmt = &query->inputs[1];
    if (x_fmt->dtype == GEIST_DTYPE_F32 && x_fmt->layout == GEIST_LAYOUT_DENSE &&
        (w_fmt->dtype == GEIST_DTYPE_Q4_K || w_fmt->dtype == GEIST_DTYPE_Q6_K) &&
        w_fmt->layout == GEIST_LAYOUT_BLOCK_QUANTIZED) {
        return GEIST_SUPPORT_NATIVE;
    }
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
        .resolve_weight        = vk_resolve_weight,
        .rmsnorm               = vk_rmsnorm,
        .add                   = vk_add,
        .mul                   = vk_mul,
        .gelu_tanh             = vk_gelu_tanh,
        .gelu_tanh_mul         = vk_gelu_tanh_mul,
        .gelu_tanh_mul_scaled  = vk_gelu_tanh_mul_scaled,
        .relu_squared          = vk_relu_squared,
        .silu                  = vk_silu,
        .rope_apply            = vk_rope_apply,
        .embedding_lookup      = vk_embedding_lookup,
        .attention             = vk_attention,
        /* Phase 3: batched-submit paths — one flush per token (argmax). */
        #ifndef VK_NO_LINEAR_T
        .linear_t                = vk_linear_t,
        .linear_t_pair           = vk_linear_t_pair,
#endif
#ifndef VK_NO_COPY
        .buffer_copy             = vk_buffer_copy,
#endif
#ifndef VK_NO_SCALE
        .scale_f32               = vk_scale_f32,
#endif
#ifndef VK_NO_RMSADD
        .rmsnorm_add             = vk_rmsnorm_add,
#endif
#ifndef VK_NO_EMBED
        .embedding_lookup_scaled = vk_embedding_lookup_scaled,
#endif
#ifndef VK_NO_ARGMAX
        .argmax_f32              = vk_argmax_f32,
#endif
        .ffn_gate_up             = vk_ffn_gate_up,
        .ffn_norm_gate_up        = vk_ffn_norm_gate_up,
        .ple_block               = vk_ple_block,
        .attn_qkv_prep           = vk_attn_qkv_prep,
        .kv_append_f16           = vk_kv_append_f16,
};

const struct geist_backend_descriptor geist_backend_vulkan = {
        .name = "vulkan",
        .vtbl = &vk_vtbl,
};
