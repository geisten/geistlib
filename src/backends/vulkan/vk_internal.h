/*
 * src/backends/vulkan/vk_internal.h — shared state, types, macros, and
 * cross-module prototypes of the Vulkan backend.
 *
 * Layer: BACKEND (vulkan, internal). Split by responsibility into
 * lifecycle.c, resources.c, pipelines.c, sequence.c, ops.c — the same
 * template as the metal backend (profiling folded into sequence.c: one
 * function does not earn a file).
 */
#ifndef GEIST_INTERNAL_VK_INTERNAL_H
#define GEIST_INTERNAL_VK_INTERNAL_H
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
#include <stdatomic.h>
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
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
    /* global */
    PFN_vkCreateInstance CreateInstance;
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
    VK_XRING_CAP   = 192u << 20, /* a full prefill chunk stages ~124 MB */
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
};

enum {
    VK_SEQ_MAX_SETS     = 4096, /* descriptor sets per flush window */
    VK_SEQ_MAX_DISPATCH = 4000, /* rotate the sequence before pool runs dry */
    VK_PUSH_RANGE       = 128,  /* one push range covers every shader block */
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

    /* From VkPhysicalDeviceSubgroupProperties. The register-tiled GEMM
     * shaders assume 32 lanes (2080-Ti-first); on any other size the mN
     * dispatch loops the (size-agnostic) matvec kernels instead. */
    uint32_t subgroup_size;

    /* Feature probes for the Phase-2 kernels. */
    bool has_fp16;     /* shaderFloat16 + 16-bit storage */
    bool has_int8_dot; /* shaderIntegerDotProduct + 8-bit storage */
    bool has_coopmat;  /* VK_KHR_cooperative_matrix */

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
    bool        profile_enabled;
    float       ts_period_ns;
    VkQueryPool ts_pool;
    uint32_t    ts_count;
    uint8_t     ts_pipe[VK_SEQ_MAX_DISPATCH + 8];
    uint64_t    prof_ns[VK_PIPE_COUNT + 1];
    uint64_t    prof_calls[VK_PIPE_COUNT + 1];

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
    VkCommandBuffer       seq_cmd;     /* currently recording */
    uint32_t              seq_cmd_idx; /* next ring slot */
    uint32_t              seq_in_cmd;  /* dispatches in the open cmd buffer */
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
        [VK_PIPE_MATVEC_Q4K] = 3,    [VK_PIPE_MATMUL_Q4K] = 3,   [VK_PIPE_MATVEC_Q6K] = 3,
        [VK_PIPE_MATMUL_Q6K] = 3,    [VK_PIPE_MATVEC_F32] = 3,   [VK_PIPE_MATMUL_F32] = 3,
        [VK_PIPE_ADD] = 3,           [VK_PIPE_MUL] = 3,          [VK_PIPE_GELU] = 2,
        [VK_PIPE_GELU_MUL] = 3,      [VK_PIPE_SCALE] = 2,        [VK_PIPE_RMSNORM] = 3,
        [VK_PIPE_RMSNORM_ADD] = 4,   [VK_PIPE_ROPE] = 3,         [VK_PIPE_ATTENTION] = 4,
        [VK_PIPE_ARGMAX] = 2,        [VK_PIPE_EMBED] = 2,        [VK_PIPE_FFN_GATE_UP] = 4,
        [VK_PIPE_QKV_PREP] = 6,      [VK_PIPE_MM_Q4K_CM] = 3,    [VK_PIPE_MM_Q6K_CM] = 3,
        [VK_PIPE_ATTENTION_F16] = 4, [VK_PIPE_QKV_PREP_F16] = 6, [VK_PIPE_KV_APPEND_F16] = 4,
        [VK_PIPE_ATTN_PART_F16] = 4, [VK_PIPE_ATTN_COMB] = 2,    [VK_PIPE_MM_Q4K_CM32] = 3,
        [VK_PIPE_PLE_GATE] = 4,      [VK_PIPE_FFN_NORM_GU] = 5,
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
};
/* Guard: ops require a live device + pipeline set (see lifecycle). */
#define VK_OPS(be, bit) ((((struct vk_state *) (be)->state)->gpu_ops & (bit)) != 0)

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

void vk_buffer_destroy(struct geist_backend *be, struct geist_buffer *buf);

[[nodiscard]] enum geist_status
vk_buffer_upload(struct geist_buffer *buf, size_t n_bytes, const uint8_t *src);

[[nodiscard]] enum geist_status
vk_buffer_download(size_t n_bytes, uint8_t *dst, const struct geist_buffer *buf);

void *vk_buffer_map(struct geist_buffer *buf);

void vk_buffer_unmap(struct geist_buffer *buf);

[[nodiscard]] enum geist_status vk_stage_reserve_role(struct geist_backend  *be,
                                                      struct geist_buffer  **slot,
                                                      size_t                 bytes,
                                                      enum geist_buffer_role role);

[[nodiscard]] enum geist_status
vk_stage_reserve(struct geist_backend *be, struct geist_buffer **slot, size_t bytes);

struct geist_buffer *vk_weight_lookup(struct vk_state *st, const void *host);

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

void vk_seq_flush(struct vk_state *st);

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

#endif /* GEIST_INTERNAL_VK_INTERNAL_H */
