/*
 * src/backends/vulkan/vk_internal.h — shared state, types and cross-module
 * prototypes of the Vulkan backend (lifecycle.c, resources.c, pipelines.c,
 * sequence.c, ops.c).
 *
 * Layer: BACKEND (vulkan, internal).
 */
#ifndef GEIST_INTERNAL_VK_INTERNAL_H
#define GEIST_INTERNAL_VK_INTERNAL_H
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <geist.h>
#include <geist_backend.h>
#include <geist_types.h>
#include <geist_weight.h>
#include <geist_util.h> /* struct geist_backend_memory */

#include "checked.h"        /* ckd_* size arithmetic (AGENT.md §3) */
#include "gemma4_kernels.h" /* shared reference rope/attention kernels */
#include "heap.h"
#include "quant.h"             /* CPU dequant helpers for the non-GPU dtype fallback */
#include "shaders/vk_limits.h" /* constants shared with the shaders (#474) */

#include <dlfcn.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

/* Vulkan entry points, resolved at runtime from the dlopen'd loader. */
struct vk_fns {
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
    /* global */
    PFN_vkCreateInstance CreateInstance;
    /* instance */
    PFN_vkDestroyInstance                        DestroyInstance;
    PFN_vkEnumeratePhysicalDevices               EnumeratePhysicalDevices;
    PFN_vkGetPhysicalDeviceProperties2           GetPhysicalDeviceProperties2;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties      GetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties2     GetPhysicalDeviceMemoryProperties2;
    PFN_vkGetPhysicalDeviceFeatures2             GetPhysicalDeviceFeatures2;
    PFN_vkEnumerateDeviceExtensionProperties     EnumerateDeviceExtensionProperties;
    PFN_vkCreateDevice                           CreateDevice;
    PFN_vkGetDeviceProcAddr                      GetDeviceProcAddr;
    /* device */
    PFN_vkDestroyDevice               DestroyDevice;
    PFN_vkGetDeviceQueue              GetDeviceQueue;
    PFN_vkCreateBuffer                CreateBuffer;
    PFN_vkDestroyBuffer               DestroyBuffer;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
    PFN_vkAllocateMemory              AllocateMemory;
    PFN_vkFreeMemory                  FreeMemory;
    PFN_vkBindBufferMemory            BindBufferMemory;
    PFN_vkMapMemory                   MapMemory;
    PFN_vkUnmapMemory                 UnmapMemory;
    PFN_vkCreateCommandPool           CreateCommandPool;
    PFN_vkDestroyCommandPool          DestroyCommandPool;
    PFN_vkAllocateCommandBuffers      AllocateCommandBuffers;
    PFN_vkBeginCommandBuffer          BeginCommandBuffer;
    PFN_vkEndCommandBuffer            EndCommandBuffer;
    PFN_vkResetCommandBuffer          ResetCommandBuffer;
    PFN_vkCmdCopyBuffer               CmdCopyBuffer;
    PFN_vkQueueSubmit                 QueueSubmit;
    PFN_vkQueueWaitIdle               QueueWaitIdle;
    PFN_vkCreateFence                 CreateFence;
    PFN_vkDestroyFence                DestroyFence;
    PFN_vkResetFences                 ResetFences;
    PFN_vkWaitForFences               WaitForFences;
    /* compute pipeline machinery */
    PFN_vkCreateShaderModule         CreateShaderModule;
    PFN_vkDestroyShaderModule        DestroyShaderModule;
    PFN_vkCreateDescriptorSetLayout  CreateDescriptorSetLayout;
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout;
    PFN_vkCreatePipelineLayout       CreatePipelineLayout;
    PFN_vkDestroyPipelineLayout      DestroyPipelineLayout;
    PFN_vkCreateComputePipelines     CreateComputePipelines;
    PFN_vkDestroyPipeline            DestroyPipeline;
    PFN_vkCreateDescriptorPool       CreateDescriptorPool;
    PFN_vkDestroyDescriptorPool      DestroyDescriptorPool;
    PFN_vkAllocateDescriptorSets     AllocateDescriptorSets;
    PFN_vkUpdateDescriptorSets       UpdateDescriptorSets;
    PFN_vkCmdBindPipeline            CmdBindPipeline;
    PFN_vkCmdBindDescriptorSets      CmdBindDescriptorSets;
    PFN_vkCmdPushConstants           CmdPushConstants;
    PFN_vkCmdDispatch                CmdDispatch;
    PFN_vkCmdPipelineBarrier         CmdPipelineBarrier;
    PFN_vkResetDescriptorPool        ResetDescriptorPool;
    PFN_vkCreateQueryPool            CreateQueryPool;
    PFN_vkDestroyQueryPool           DestroyQueryPool;
    PFN_vkCmdResetQueryPool          CmdResetQueryPool;
    PFN_vkCmdWriteTimestamp          CmdWriteTimestamp;
    PFN_vkGetQueryPoolResults        GetQueryPoolResults;
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
    VK_PIPE_MM_Q4K_CM32,     /* small-n_out tensor-core tile */
    VK_PIPE_MM_PQ2_0_CM,     /* PQ2_0 tensor-core GEMM, f16 acc folded into f32 */
    VK_PIPE_MM_PQ2_0_CM_F32, /* the same with f32 accumulation throughout (GEIST_VK_PQ2_F32_ACC) */
    VK_PIPE_MM_PQ2_0_CM64,   /* 128 x 64 tile (f32 acc) for batches under 128 tokens */
    VK_PIPE_PLE_GATE,        /* fused PLE gate: gelu(x.gate_w) * ple_in */
    VK_PIPE_FFN_NORM_GU,     /* ffn_gate_up with the pre-FFN rmsnorm folded in */
    VK_PIPE_DN_CONV,         /* gated-DeltaNet causal conv + silu (deltanet_mix stage 1) */
    VK_PIPE_DN_DELTA,        /* gated-DeltaNet recurrence + gated rmsnorm (stage 2) */
    VK_PIPE_MATVEC_Q4_0,
    VK_PIPE_MATMUL_Q4_0,
    VK_PIPE_MATVEC_Q4_1,
    VK_PIPE_MATMUL_Q4_1,
    VK_PIPE_MATVEC_Q8_0,
    VK_PIPE_MATMUL_Q8_0,
    VK_PIPE_MATVEC_Q5K,
    VK_PIPE_MATMUL_Q5K,
    VK_PIPE_MATVEC_TQ2_0,
    VK_PIPE_MATMUL_TQ2_0,
    VK_PIPE_MATVEC_PQ2_0,
    VK_PIPE_MATMUL_PQ2_0,
    VK_PIPE_SILU,             /* y = silu(x) */
    VK_PIPE_HADAMARD,         /* blockwise orthonormal WHT of rows (prism.hadamard) */
    VK_PIPE_RELU2,            /* y = relu(x)^2 (BitNet FFN) */
    VK_PIPE_ACT_QUANT,        /* BitNet int8 absmax activation round trip, in place */
    VK_PIPE_SILU_MUL,         /* y = silu(a) * b (SwiGLU epilogue) */
    VK_PIPE_SIGMOID_MUL,      /* y = a * sigmoid(gate) (qwen35 attention gate) */
    VK_PIPE_QGATE_SPLIT,      /* [query | gate] per-head split (qwen35) */
    VK_PIPE_ATTENTION_F16_CM, /* tensor-core causal attention, no sliding window, head_dim==256 */
    VK_PIPE_ATTENTION_F16_HD128_CM, /* the same, head_dim 128 */
    VK_PIPE_ATTENTION_F16_HD512_CM, /* the same, head_dim 512 (two column halves) */
    VK_PIPE_MM_Q8_0_CM,             /* Q8_0 tensor-core GEMM, 64 x 64 tile */
    VK_PIPE_MM_Q4_0_CM,             /* Q4_0 tensor-core GEMM, 64 x 64 tile */
    VK_PIPE_MM_Q5K_CM,              /* Q5_K tensor-core GEMM, 64 x 64 tile */
    VK_PIPE_MM_Q4_1_CM,             /* Q4_1 tensor-core GEMM, 64 x 64 tile */
    VK_PIPE_MM_TQ2_0_CM,            /* TQ2_0 tensor-core GEMM, 64 x 64 tile */
    VK_PIPE_MM_TQ2_0_CM128,         /* TQ2_0 tensor-core GEMM, 128 x 128 tile */
    VK_PIPE_COUNT,
};

/* Pipelines that exist only with VK_KHR_cooperative_matrix; without it they
 * stay VK_NULL_HANDLE and vk_linear_cm_route keeps the register-tiled GEMM. */
static inline bool vk_pipe_needs_coopmat(int pipe) {
    return pipe == VK_PIPE_MM_Q4K_CM || pipe == VK_PIPE_MM_Q6K_CM || pipe == VK_PIPE_MM_Q4K_CM32 ||
           pipe == VK_PIPE_MM_PQ2_0_CM || pipe == VK_PIPE_MM_PQ2_0_CM_F32 ||
           pipe == VK_PIPE_MM_PQ2_0_CM64 || pipe == VK_PIPE_ATTENTION_F16_CM ||
           pipe == VK_PIPE_MM_Q8_0_CM || pipe == VK_PIPE_MM_Q4_0_CM || pipe == VK_PIPE_MM_Q5K_CM ||
           pipe == VK_PIPE_MM_Q4_1_CM || pipe == VK_PIPE_MM_TQ2_0_CM ||
           pipe == VK_PIPE_MM_TQ2_0_CM128 || pipe == VK_PIPE_ATTENTION_F16_HD128_CM ||
           pipe == VK_PIPE_ATTENTION_F16_HD512_CM;
}

/* The register-tiled GEMMs: one output row per 32-lane subgroup
 * (mm_legacy.glsl and its siblings). The tensor-core variants are not on
 * this list; they keep the native subgroup size. */
static inline bool vk_pipe_is_tiled_gemm(int pipe) {
    return pipe == VK_PIPE_MATMUL_Q4K || pipe == VK_PIPE_MATMUL_Q6K || pipe == VK_PIPE_MATMUL_F32 ||
           pipe == VK_PIPE_MATMUL_Q4_0 || pipe == VK_PIPE_MATMUL_Q4_1 ||
           pipe == VK_PIPE_MATMUL_Q8_0 || pipe == VK_PIPE_MATMUL_Q5K ||
           pipe == VK_PIPE_MATMUL_TQ2_0 || pipe == VK_PIPE_MATMUL_PQ2_0;
}

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
    VK_XRING_CAP   = 192u << 20, /* a full prefill chunk stages ~124 MB */
    VK_MAX_M       = 512,        /* caps.max_m; resolve_weight checks it x n_in fits the ring */
    VK_DIRTY_CAP   = 96,
    VK_DSET_CACHE  = 4096,
    VK_SEQ_CMDBUFS = 64, /* rolling submission ring */
    VK_SEQ_ROTATE  = 64, /* dispatches per submit — keeps the GPU fed */
};

/* Cached descriptor set: decode re-binds the same (pipeline-layout,
 * buffers) tuple every token — building sets once kills the dominant
 * CPU cost of the encode loop (alloc + update per dispatch). */
struct vk_dset_entry {
    uint64_t        key; /* hash of nbind + buffer handles; 0 = empty */
    VkDescriptorSet set;
    uint32_t        nbind; /* bindings of the layout `set` was allocated with */
};

enum {
    VK_MAX_BINDINGS     = 8,    /* storage buffers per dispatch (deltanet_mix needs 8) */
    VK_SEQ_MAX_SETS     = 4096, /* descriptor sets per flush window */
    VK_SEQ_MAX_DISPATCH = 4000, /* rotate the sequence before pool runs dry */
    VK_PUSH_RANGE       = 128,  /* one push range covers every shader block */
};

/* Where work can leave the GPU (vk_fallback). */
enum vk_fb {
    VK_FB_HOST_VIEW,   /* a host loop over a mapped tensor (vk_tensor_host) */
    VK_FB_HOST_LINEAR, /* a linear on the host row-dequant path (vk_w_cpu_mN) */
    VK_FB_HOST_COPY,   /* vk_buffer_copy through mapped memory */
    VK_FB_LINEAR_T,    /* linear_t / linear_t_pair declined: arch host linear */
    VK_FB_ARGMAX,      /* argmax declined: arch scans on the host */
    VK_FB_EMBED,       /* embedding lookup declined: arch gathers on the host */
    VK_FB_KV_APPEND,   /* kv_append_f16 declined */
    VK_FB_QGATE,       /* attn_qgate_split declined */
    VK_FB_COUNT
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
    size_t bar_used; /* live host-visible + device-local bytes (the BAR window) */
    /* Device memory taken by device-local requests (weight copies, KV
     * cache, x ring), checked against vram_budget before each allocation
     * so an oversized model fails with needed vs. available bytes instead
     * of a bare driver error (#466). vram_budget is GEIST_VK_VRAM_BUDGET
     * (bytes, K/M/G suffix) or 0: the heap size of the memory type. */
    size_t vram_used;
    size_t vram_budget;
    /* VK_EXT_memory_budget is enabled: vk_heap_budget reports the driver's
     * budget and usage, which see every allocation on the device (#665). */
    bool has_mem_budget;
    /* An integrated GPU: its device-local heap is system RAM. */
    bool            unified_memory;
    VkCommandPool   cmd_pool;
    VkCommandBuffer xfer_cmd;
    VkFence         xfer_fence;
    /* Persistent host-visible staging buffer for uploads (#469): created on
     * the first staged upload, VK_UP_STAGE_BYTES, reused for every chunk —
     * a fresh full-size staging buffer per weight cost page faults on
     * hundreds of MB per tensor. */
    VkBuffer       up_buf;
    VkDeviceMemory up_mem;
    uint8_t       *up_map;

    char device_name[256];

    /* From VkPhysicalDeviceSubgroupProperties. The register-tiled GEMM
     * shaders assume 32 lanes (2080-Ti-first). */
    uint32_t subgroup_size;
    /* The device can pin a compute pipeline to full 32-lane subgroups
     * (subgroupSizeControl + computeFullSubgroups, 32 in its range). */
    bool sg32_pinnable;
    /* The tiled GEMMs run on 32-lane subgroups: pinned at pipeline
     * creation, or the native size is 32. When false the mN dispatch
     * loops the (size-agnostic) matvec kernels instead (#471). */
    bool gemm_sg32;

    /* Device feature probes. */
    bool has_fp16;     /* shaderFloat16 + 16-bit storage */
    bool has_int8_dot; /* shaderIntegerDotProduct + 8-bit storage */
    bool has_coopmat;  /* VK_KHR_cooperative_matrix */
    bool pq2_f32_acc;  /* GEIST_VK_PQ2_F32_ACC: exact f32-accumulate PQ2_0 tensor-core GEMM */
    /* GEIST_VK_SCRATCH_DEVICE=1 (#488): a SCRATCH-role buffer_create that asks
     * for GEIST_MEMORY_DEVICE gets device-local, unmappable memory. Off by
     * default: such a request is served host-visible and the arch keeps its
     * mapped pool. See vk_buffer_create_api. */
    bool scratch_device;

    /* Set when a sequence flush failed (submit / wait / end); the next host
     * readback (argmax, download, host view) reports it as GEIST_E_BACKEND and
     * clears it — see vk_seq_take_failure. */
    bool seq_failed;

    /* Cached copies of x and y for the host linear (vk_w_cpu_mN), in floats. */
    float *cpu_row;
    size_t cpu_row_cap;

    /* GEIST_VK_VERBOSE stats. */
    uint64_t stat_flushes;
    uint64_t stat_dispatches;
    /* Host views refused because the buffer is device-local (vk_tensor_host):
     * a CPU fallback that would have needed the bytes, reported as an error
     * instead of a read over PCIe. */
    uint64_t stat_host_denied;
    /* Where SCRATCH-role memory landed (#488), per buffer_create: device-local
     * and unmappable, the BAR window (device-local + host-visible), or host
     * memory the GPU reads over the bus. Indexed by enum vk_placement;
     * cumulative over the backend's life (destroys do not decrement). One
     * buffer's placement: vk_buffer_placement. */
    uint64_t stat_scratch_n[3];
    uint64_t stat_scratch_bytes[3];

    /* Work that left the GPU (#474), counted per site by
     * vk_fallback: a coverage gap otherwise shows up only as a slowdown.
     * GEIST_VK_STRICT=1 (strict) turns every such fallback into an error,
     * and refuses host-path weights at resolve. Weights resolved onto the
     * host row-dequant path are summed at resolve and reported once, when
     * the first of them runs. */
    uint64_t fallbacks[VK_FB_COUNT];
    bool     strict;
    bool     host_weights_noted;
    size_t   host_weights;
    size_t   host_weight_bytes;

    /* GEIST_VK_PROFILE=1: GPU timestamps per dispatch, attributed by
     * pipeline (copies land in the extra slot). Execution is serialized by
     * the per-dispatch barriers, so consecutive deltas are exact. */
    bool        profile_enabled;
    float       ts_period_ns;
    VkQueryPool ts_pool;
    uint32_t    ts_count;
    uint8_t     ts_pipe[VK_SEQ_MAX_DISPATCH + 8];
    uint64_t    prof_ns[VK_PIPE_COUNT + 1];
    uint64_t    prof_calls[VK_PIPE_COUNT + 1];

    /* Compute pipelines (layouts: seq_dlayouts / seq_playouts). */
    VkPipeline pipes[VK_PIPE_COUNT];

    /* Weight registry: host pointer → VRAM buffer, filled by resolve_weight.
     * `weights` is the dense list (teardown); `weight_index` an open-
     * addressed table of entry index + 1 (0 = empty), a power of two at
     * most half full, so the lookup on every linear_t / embedding call is
     * one or two probes instead of a scan of a few hundred entries (#469).
     * Entries are never removed, only replaced in place. */
    struct vk_weight_entry *weights;
    size_t                  n_weights;
    size_t                  cap_weights;
    uint32_t               *weight_index;
    size_t                  cap_weight_index;

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

    /* Sequence: one open recording per token/chunk, rolled over the
     * seq_cmds ring every VK_SEQ_ROTATE dispatches. GPU ops append
     * dispatches (barrier only on a hazard, see vk_seq_hazard); flush =
     * submit + fence-wait, triggered by any host data access (buffer_map /
     * CPU-op fallback / argmax readback). Descriptor sets come from
     * dset_cache; seq_pool holds the uncached ones and is reset at flush. */
    VkCommandBuffer       seq_cmds[VK_SEQ_CMDBUFS];
    VkCommandBuffer       seq_cmd;     /* currently recording */
    uint32_t              seq_cmd_idx; /* next ring slot */
    uint32_t              seq_in_cmd;  /* dispatches in the open cmd buffer */
    VkFence               seq_fence;
    bool                  seq_open;
    uint32_t              seq_dispatches;
    VkDescriptorPool      seq_pool;
    VkDescriptorSetLayout seq_dlayouts[VK_MAX_BINDINGS - 1]; /* index = binding count - 2 */
    VkPipelineLayout      seq_playouts[VK_MAX_BINDINGS - 1];

    /* Host-visible buffers created via buffer_create — containment lookup
     * so buffer_create_aliased can hand out GPU-bindable borrowed views
     * (e.g. of the arch scratch pool / weight arena). */
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
        [VK_PIPE_MATVEC_Q4K]             = 3,
        [VK_PIPE_MATMUL_Q4K]             = 3,
        [VK_PIPE_MATVEC_Q6K]             = 3,
        [VK_PIPE_MATMUL_Q6K]             = 3,
        [VK_PIPE_MATVEC_F32]             = 3,
        [VK_PIPE_MATMUL_F32]             = 3,
        [VK_PIPE_ADD]                    = 3,
        [VK_PIPE_MUL]                    = 3,
        [VK_PIPE_GELU]                   = 2,
        [VK_PIPE_GELU_MUL]               = 3,
        [VK_PIPE_SCALE]                  = 2,
        [VK_PIPE_RMSNORM]                = 3,
        [VK_PIPE_RMSNORM_ADD]            = 4,
        [VK_PIPE_ROPE]                   = 3,
        [VK_PIPE_ATTENTION]              = 4,
        [VK_PIPE_ARGMAX]                 = 2,
        [VK_PIPE_EMBED]                  = 2,
        [VK_PIPE_FFN_GATE_UP]            = 4,
        [VK_PIPE_QKV_PREP]               = 6,
        [VK_PIPE_MM_Q4K_CM]              = 3,
        [VK_PIPE_MM_Q6K_CM]              = 3,
        [VK_PIPE_ATTENTION_F16]          = 4,
        [VK_PIPE_QKV_PREP_F16]           = 6,
        [VK_PIPE_KV_APPEND_F16]          = 4,
        [VK_PIPE_ATTN_PART_F16]          = 4,
        [VK_PIPE_ATTN_COMB]              = 2,
        [VK_PIPE_MM_Q4K_CM32]            = 3,
        [VK_PIPE_MM_PQ2_0_CM]            = 3,
        [VK_PIPE_MM_PQ2_0_CM_F32]        = 3,
        [VK_PIPE_MM_PQ2_0_CM64]          = 3,
        [VK_PIPE_PLE_GATE]               = 4,
        [VK_PIPE_FFN_NORM_GU]            = 5,
        [VK_PIPE_DN_CONV]                = 3,
        [VK_PIPE_DN_DELTA]               = 8,
        [VK_PIPE_MATVEC_Q4_0]            = 3,
        [VK_PIPE_MATMUL_Q4_0]            = 3,
        [VK_PIPE_MATVEC_Q4_1]            = 3,
        [VK_PIPE_MATMUL_Q4_1]            = 3,
        [VK_PIPE_MATVEC_Q8_0]            = 3,
        [VK_PIPE_MATMUL_Q8_0]            = 3,
        [VK_PIPE_MATVEC_Q5K]             = 3,
        [VK_PIPE_MATMUL_Q5K]             = 3,
        [VK_PIPE_MATVEC_TQ2_0]           = 3,
        [VK_PIPE_MATMUL_TQ2_0]           = 3,
        [VK_PIPE_MATVEC_PQ2_0]           = 3,
        [VK_PIPE_MATMUL_PQ2_0]           = 3,
        [VK_PIPE_SILU]                   = 2,
        [VK_PIPE_RELU2]                  = 2,
        [VK_PIPE_HADAMARD]               = 3,
        [VK_PIPE_ACT_QUANT]              = 2,
        [VK_PIPE_SILU_MUL]               = 3,
        [VK_PIPE_SIGMOID_MUL]            = 3,
        [VK_PIPE_QGATE_SPLIT]            = 3,
        [VK_PIPE_ATTENTION_F16_CM]       = 4,
        [VK_PIPE_ATTENTION_F16_HD128_CM] = 4,
        [VK_PIPE_ATTENTION_F16_HD512_CM] = 4,
        [VK_PIPE_MM_Q8_0_CM]             = 3,
        [VK_PIPE_MM_Q4_0_CM]             = 3,
        [VK_PIPE_MM_Q5K_CM]              = 3,
        [VK_PIPE_MM_Q4_1_CM]             = 3,
        [VK_PIPE_MM_TQ2_0_CM]            = 3,
        [VK_PIPE_MM_TQ2_0_CM128]         = 3,
};

struct geist_buffer {
    struct vk_state       *owner;
    VkBuffer               buf; /* VK_NULL_HANDLE for pure bookkeeping aliases */
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
    bool                   view;       /* buffer_create_view slice of a parent */
    size_t bar_bytes;  /* counted in vk_state.bar_used (host-visible + device-local) */
    size_t vram_bytes; /* counted in vk_state.vram_used */
};

/* ---- Cross-module prototypes ------------------------------------------ */
[[nodiscard]] enum geist_status vk_create(struct geist_backend            *be,
                                          const struct geist_backend_opts *opts);

void vk_destroy(struct geist_backend *be);

[[nodiscard]] enum geist_status vk_buffer_create(struct geist_backend  *be,
                                                 size_t                 bytes,
                                                 enum geist_buffer_role role,
                                                 unsigned int           memory_flags,
                                                 struct geist_buffer  **out);

[[nodiscard]] enum geist_status vk_buffer_create_aliased(struct geist_backend  *be,
                                                         void                  *host_ptr,
                                                         size_t                 n_bytes,
                                                         enum geist_buffer_role role,
                                                         struct geist_buffer  **out);

[[nodiscard]] enum geist_status vk_buffer_create_api(struct geist_backend  *be,
                                                     size_t                 bytes,
                                                     enum geist_buffer_role role,
                                                     unsigned int           memory_flags,
                                                     struct geist_buffer  **out);

[[nodiscard]] enum geist_status vk_buffer_create_view(struct geist_backend  *be,
                                                      struct geist_buffer   *parent,
                                                      size_t                 offset,
                                                      size_t                 n_bytes,
                                                      enum geist_buffer_role role,
                                                      struct geist_buffer  **out);

/* Where a buffer's bytes live (#488). NONE: a bookkeeping alias with no
 * VkBuffer behind it (GGUF mmap). */
enum vk_placement {
    VK_PLACEMENT_DEVICE = 0, /* device-local, not host-visible */
    VK_PLACEMENT_BAR    = 1, /* device-local and host-visible */
    VK_PLACEMENT_HOST   = 2, /* host memory, read by the GPU over the bus */
    VK_PLACEMENT_NONE   = 3,
};

[[nodiscard]] enum vk_placement vk_buffer_placement(const struct geist_buffer *buf);

void vk_buffer_destroy(struct geist_backend *be, struct geist_buffer *buf);

[[nodiscard]] enum geist_status
vk_buffer_upload(struct geist_buffer *buf, size_t n_bytes, const uint8_t src[static n_bytes]);

[[nodiscard]] enum geist_status
vk_buffer_download(size_t n_bytes, uint8_t dst[static n_bytes], const struct geist_buffer *buf);

void *vk_buffer_map(struct geist_buffer *buf);

void vk_buffer_unmap(struct geist_buffer *buf);

[[nodiscard]] enum geist_status vk_stage_reserve_role(struct geist_backend  *be,
                                                      struct geist_buffer  **slot,
                                                      size_t                 bytes,
                                                      enum geist_buffer_role role);

[[nodiscard]] enum geist_status
vk_stage_reserve(struct geist_backend *be, struct geist_buffer **slot, size_t bytes);

struct geist_buffer *vk_weight_lookup(const struct vk_state *st, const void *host);
/* The registry entry for `host`, or nullptr. */
struct vk_weight_entry *vk_weight_entry_of(const struct vk_state *st, const void *host);
/* Index weights[idx] (just appended) by its host pointer; GEIST_E_OOM when
 * the table cannot grow. */
[[nodiscard]] enum geist_status vk_weight_index_add(struct geist_backend *be, size_t idx);
struct geist_buffer            *vk_weight_of(struct vk_state *st, const struct geist_tensor *t);
size_t                          vk_fast_host_bytes(struct geist_backend *be);
/* geist_backend_descriptor::memory_info: the device-local heap's size and
 * what is left of it (see geist_backend_memory). */
[[nodiscard]] enum geist_status vk_memory_info(const struct geist_backend  *be,
                                               struct geist_backend_memory *out);

struct vk_access vk_acc(uint64_t lo_bytes, uint64_t n_bytes, bool write);

struct vk_access vk_acc_all(bool write);

struct vk_access vk_acc_tensor(const struct geist_tensor *t, bool write);

bool vk_tensor_gpu(const struct geist_tensor *t, VkDescriptorBufferInfo *out, uint32_t *elem_off);

bool vk_tensor_gpu_f16(const struct geist_tensor *t,
                       VkDescriptorBufferInfo    *out,
                       uint32_t                  *elem_off);

size_t vk_t_n16(const struct geist_tensor *t);

struct vk_access vk_acc_tensor16(const struct geist_tensor *t, bool write);

size_t vk_t_n(const struct geist_tensor *t);

void *vk_tensor_host(const struct geist_tensor *t, size_t *out_n);

bool vk_t_geom(const struct geist_tensor *t, size_t *rows, size_t *cols, size_t *stride);

[[nodiscard]] enum geist_status vk_buffer_copy(struct geist_buffer       *dst,
                                               size_t                     dst_offset,
                                               const struct geist_buffer *src,
                                               size_t                     src_offset,
                                               size_t                     n_bytes);

[[nodiscard]] bool vk_xring_stage(struct geist_backend      *be,
                                  const struct geist_tensor *t_x,
                                  size_t                     m,
                                  size_t                     n_in,
                                  uint32_t                  *out_elem_off);

[[nodiscard]] enum geist_status vk_create_pipelines(struct geist_backend *be, struct vk_state *st);

void                            vk_seq_flush(struct vk_state *st);
[[nodiscard]] enum geist_status vk_seq_take_failure(struct vk_state *st);

/* Record that the work at `site` leaves the GPU. Returns the status the
 * caller hands on: GEIST_E_UNSUPPORTED (take the documented fallback), or
 * under GEIST_VK_STRICT=1 GEIST_E_BACKEND with an error naming the site. */
[[nodiscard]] enum geist_status vk_fallback(struct vk_state *st, enum vk_fb site);
const char                     *vk_fallback_name(enum vk_fb site);

/* Checked size -> uint32_t narrowing for push constants, dispatch sizes and
 * element offsets: the shaders index in uint32, so a value that does not fit
 * fails the op instead of wrapping (AGENT.md §3/§5, #474). checked.h's
 * convention: true when `v` does not fit; *out is written only when it does. */
[[nodiscard]] static inline bool vk_ckd_u32(size_t v, uint32_t *out) {
    if (v > UINT32_MAX) {
        return true;
    }
    *out = (uint32_t) v;
    return false;
}

/* What an op returns when vk_ckd_u32 refused one of its values:
 * GEIST_E_INVALID_ARG, with an error naming the op. */
[[nodiscard]] enum geist_status vk_too_wide(struct geist_backend *be, const char *op);

[[nodiscard]] enum geist_status vk_seq_open_cmd(struct vk_state *st);

void vk_prof_stamp(struct vk_state *st, uint32_t slot);

void vk_seq_hazard(struct vk_state              *st,
                   const VkDescriptorBufferInfo *infos,
                   const struct vk_access       *acc,
                   uint32_t                      n);

[[nodiscard]] enum geist_status vk_seq_dispatch_acc(struct geist_backend         *be,
                                                    enum vk_pipe                  pipe,
                                                    const VkDescriptorBufferInfo *infos,
                                                    const struct vk_access       *acc,
                                                    const void                   *push,
                                                    uint32_t                      push_bytes,
                                                    uint32_t                      gx,
                                                    uint32_t                      gy,
                                                    uint32_t                      gz);

[[nodiscard]] enum geist_status vk_seq_dispatch(struct geist_backend         *be,
                                                enum vk_pipe                  pipe,
                                                const VkDescriptorBufferInfo *infos,
                                                const void                   *push,
                                                uint32_t                      push_bytes,
                                                uint32_t                      gx,
                                                uint32_t                      gy,
                                                uint32_t                      gz);

uint32_t vk_linear_gx(enum vk_pipe pipe, uint32_t n_out);

uint32_t vk_linear_gy(enum vk_pipe pipe, uint32_t m);

void vk_linear_cm_route(struct vk_state *st,
                        enum vk_pipe    *pipe,
                        uint32_t         m,
                        uint32_t         n_out,
                        uint32_t        *gx,
                        uint32_t        *gy);

[[nodiscard]] enum geist_status vk_gemm_dispatch(struct geist_backend         *be,
                                                 enum vk_pipe                  pipe,
                                                 const VkDescriptorBufferInfo *infos,
                                                 const struct vk_access       *acc,
                                                 const struct vk_push         *push);

#endif /* GEIST_INTERNAL_VK_INTERNAL_H */
