/*
 * geist_backend.h — extension API for backend authors.
 *
 * Include this in addition to <geist.h> when implementing a new backend
 * (cpu_neon, cpu_scalar, rknn-npu, etc.). Defines the vtable shape, the
 * descriptor each backend exports, and the engine-side registration
 * mechanism.
 *
 * @stability EXPERIMENTAL — vtable layout may evolve until 1.0.
 */
#ifndef GEIST_BACKEND_H
#define GEIST_BACKEND_H

#include <geist.h>
#include <geist_types.h>  /* tensor / op / dtype types the vtable speaks in */
#include <geist_weight.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ====================================================================== */
/* Backend Vtable                                                          */
/* ====================================================================== */

/* Execution phase the arch layer is about to enter. Backends may tune their
 * parallelism regime per phase (see parallel_region_begin). Prefill is
 * compute-bound and scales with cores; decode (m=1 GEMV) is memory-bound. */
enum geist_parallel_region {
    GEIST_REGION_PREFILL_BATCH,
    GEIST_REGION_DECODE_STEP,
};

/* Backends fill in this struct (statically) and reference it from their
 * descriptor. Engine dispatches op-calls through here. */
struct geist_backend_vtbl {
    /* ---- Lifecycle ---- */

    /* Optional create-time hook. Backend allocates per-instance state and
     * stashes it into geist_backend->state. Returns GEIST_OK on success.
     * If non-OK, engine reclaims geist_backend memory and propagates. */
    enum geist_status (*create)(struct geist_backend           *be,
                                              const struct geist_backend_opts *opts);

    /* Required. Tear down per-instance state and any held buffers. */
    void (*destroy)(struct geist_backend *be);

    /* ---- Capability ---- */

    /* Pre-flight check: can this backend execute this op signature? */
    enum geist_support (*supports_op)(struct geist_backend                 *be,
                                      const struct geist_op_support_query *query);

    /* ---- Buffer ops ---- */

    /* Allocate a buffer of the given size and role. Backend may pick
     * device-local vs host-coherent based on memory_flags. */
    enum geist_status (*buffer_create)(struct geist_backend      *be,
                                                     size_t                     bytes,
                                                     enum geist_buffer_role     role,
                                                     unsigned int               memory_flags,
                                                     struct geist_buffer      **out);

    void (*buffer_destroy)(struct geist_backend *be, struct geist_buffer *buf);

    /* Create a buffer that aliases an external host-resident region (e.g.
     * an mmap-backed weight tensor whose lifetime is bound to the GGUF
     * reader). The backend wraps host_ptr in a geist_buffer with the
     * GEIST_MEMORY_ALIASED bit set; buffer_destroy releases only the
     * buffer-handle struct and never frees host_ptr. CPU backends return
     * host_ptr unchanged from buffer_map; GPU backends MAY return nullptr
     * (caller must fall back). nullptr means the backend doesn't support
     * aliasing — caller must use buffer_create + buffer_upload. */
    enum geist_status (*buffer_create_aliased)(struct geist_backend  *be,
                                               void                  *host_ptr,
                                               size_t                 n_bytes,
                                               enum geist_buffer_role role,
                                               struct geist_buffer  **out);

    /* Copy host bytes into the buffer. Caller-provided source array. */
    enum geist_status (*buffer_upload)(struct geist_buffer *buf,
                                                     size_t               n_bytes,
                                                     const uint8_t        src[static n_bytes]);

    /* Copy buffer contents back to host. Caller-provided destination. */
    enum geist_status (*buffer_download)(size_t                     n_bytes,
                                                       uint8_t                    dst[static n_bytes],
                                                       const struct geist_buffer *buf);

    /* CPU shortcut: returns a host pointer that aliases the buffer.
     * Returns nullptr if the backend cannot produce a host alias for this
     * buffer (e.g. device-only GPU memory). For CPU backends this is the
     * fast path; production code should call sparingly on GPU backends. */
    void *(*buffer_map)(struct geist_buffer *buf);

    /* Counterpart to buffer_map; no-op on CPU, sync on GPU. */
    void (*buffer_unmap)(struct geist_buffer *buf);

    /* ---- Load-time weight resolver (P1.1, refactor v2) ----
     *
     * Inspect a weight tensor's dtype + shape and write direct function
     * pointers into `w->linear_m1` and `w->linear_mN`. Runs once per
     * weight at model load. Subsequent forward calls go through the
     * resolved pointers without per-call dispatch. Optionally allocate
     * `w->aux_fp32` via heap.h for pre-folded data (AWQ etc.).
     *
     * nullable: backends that don't yet implement the new flow (or that
     * fundamentally can't pre-resolve, e.g. a future fully-dynamic GPU
     * backend) leave this slot null. Callers fall back to the legacy
     * per-op vtable path. */
    enum geist_status (*resolve_weight)(struct geist_backend *be,
                                        struct geist_weight  *w);

    /* ---- Primitive Ops (Level 2 per Q17) ---- */
    /* Each op takes geist_tensor inputs/outputs whose .buffer was created
     * via this same backend. Return GEIST_OK on success; on error, set
     * the backend error slot via geist_backend_set_error_*. */

    /* (P2.e) The legacy `linear` op vtable slot was dropped after the
     * resolver path (resolve_weight + geist_weight::linear_m1/_mN) covered
     * every production dtype. All callers go through linear_w_or_legacy
     * in src/archs/transformer/forward.c, which dispatches solely on the
     * pre-resolved kernel pointers. Adding a new linear path means adding
     * a resolver case, not a vtable entry. */

    /* y = x * w * rsqrt(mean(x^2) + eps). w broadcasts across feature dim.
     * All tensors are F32 DENSE. x and y can be the same tensor (in-place). */
    enum geist_status (*rmsnorm)(struct geist_backend      *be,
                                               const struct geist_tensor *x,
                                               const struct geist_tensor *w,
                                               float                      eps,
                                               struct geist_tensor       *y);

    /* y = a + b. All F32 DENSE, same shape. y can alias a or b. */
    enum geist_status (*add)(struct geist_backend      *be,
                                           const struct geist_tensor *a,
                                           const struct geist_tensor *b,
                                           struct geist_tensor       *y);

    /* y = a * b (element-wise). All F32 DENSE, same shape. */
    enum geist_status (*mul)(struct geist_backend      *be,
                                           const struct geist_tensor *a,
                                           const struct geist_tensor *b,
                                           struct geist_tensor       *y);

    /* y = gelu_tanh(x). F32 DENSE, x and y can be the same tensor. */
    enum geist_status (*gelu_tanh)(struct geist_backend      *be,
                                                 const struct geist_tensor *x,
                                                 struct geist_tensor       *y);

    /* y = gelu_tanh(x) * z. F32 DENSE. Optional FFN fast path for GEGLU;
     * callers fall back to gelu_tanh + mul when nullptr. */
    enum geist_status (*gelu_tanh_mul)(struct geist_backend      *be,
                                       const struct geist_tensor *x,
                                       const struct geist_tensor *z,
                                       struct geist_tensor       *y);

    /* y[t,j] = gelu_tanh(x[t,j]) * z[t,j] * scale[j].
     * Optional GEGLU+AWQ fusion for transformer FFNs. scale is per-channel
     * across the last dimension. nullptr means callers use gelu_tanh_mul and
     * a separate scale pass. */
    enum geist_status (*gelu_tanh_mul_scaled)(struct geist_backend      *be,
                                              const struct geist_tensor *x,
                                              const struct geist_tensor *z,
                                              const float               *scale,
                                              struct geist_tensor       *y);

    /* y = max(x, 0)^2. F32 DENSE, x and y can be the same tensor.
     * Squared ReLU is BitNet b1.58 2B-4T's FFN activation; combining
     * the threshold + the square in one pass halves memory traffic
     * vs. relu(x) followed by mul(y, y). May be nullptr on backends
     * that don't implement it; callers must check. */
    enum geist_status (*relu_squared)(struct geist_backend      *be,
                                                    const struct geist_tensor *x,
                                                    struct geist_tensor       *y);

    /* y = silu(x) = x / (1 + exp(-x)). F32 DENSE, x and y can be the
     * same tensor. SiLU is Llama 2/3 + BitNet b1.58 3B's SwiGLU
     * activation. */
    enum geist_status (*silu)(struct geist_backend      *be,
                                            const struct geist_tensor *x,
                                            struct geist_tensor       *y);

    /* Rotary position embeddings, applied in place.
     *   x   shape [seq_len, n_heads, head_dim]   (F32 DENSE)
     *   cos shape [seq_len, head_dim]             (F32 DENSE)
     *   sin shape [seq_len, head_dim]             (F32 DENSE)
     * All shapes derived from tensor metadata. Rotates the first
     * n_rotated_dims columns of each head; n_rotated_dims is encoded as
     * cos->shape[-1] (typically == head_dim for full rotation). */
    enum geist_status (*rope_apply)(struct geist_backend      *be,
                                                  struct geist_tensor       *x,
                                                  const struct geist_tensor *cos,
                                                  const struct geist_tensor *sin);

    /* Embedding lookup: out = embed_table[token_id, :].
     *   embed_table shape [vocab_size, d_model]
     *   out         shape [d_model] (1D) or [1, d_model] (2D)
     * Returns GEIST_E_INVALID_ARG if token_id is out of range. */
    enum geist_status (*embedding_lookup)(struct geist_backend      *be,
                                                        const struct geist_tensor *embed_table,
                                                        geist_token_t              token_id,
                                                        struct geist_tensor       *out);

    /* Scaled dot-product attention with MQA broadcast and causal+window mask.
     *   q   shape [n_q,  n_q_heads,  head_dim]   (F32 DENSE)
     *   k   shape [n_kv, n_kv_heads, head_dim]   (F32 DENSE)
     *   v   shape [n_kv, n_kv_heads, head_dim]   (F32 DENSE)
     *   out shape [n_q,  n_q_heads,  head_dim]
     *
     *   q_offset       — position of q[0] in the absolute sequence;
     *                    causal mask permits q[t] → k[s] iff s <= q_offset + t.
     *   sliding_window — 0 = unbounded causal; >0 = additionally
     *                    s > q_offset + t - sliding_window. */
    enum geist_status (*attention)(struct geist_backend      *be,
                                                 const struct geist_tensor *q,
                                                 const struct geist_tensor *k,
                                                 const struct geist_tensor *v,
                                                 size_t                     q_offset,
                                                 size_t                     sliding_window,
                                                 struct geist_tensor       *out);

    /* Additional ops added in subsequent commits:
     *   silu_gate, ssm_step, ssm_scan, conv1d
     */

    /* ---- Optional Level-3 fast paths ---- */

    /* Experimental text-FFN fast path for Gemma-style GEGLU:
     *   y = down(gelu_tanh(gate(x)) * up(x) * optional_down_scale)
     * Backends may return GEIST_E_UNSUPPORTED when dtype/shape/layout do not
     * match their fused kernel. The caller then falls back to decomposed ops. */
    enum geist_status (*ffn_geglu_q4q6_mN)(struct geist_backend      *be,
                                           const float               *x,
                                           size_t                     m,
                                           size_t                     d_model,
                                           size_t                     inter,
                                           const struct geist_weight *gate,
                                           const struct geist_weight *up,
                                           const struct geist_weight *down,
                                           const float               *down_scale,
                                           float                     *y);

    /* Fused transformer block — backends with batched-submit (GPU) can
     * implement this to amortize dispatch overhead. nullptr means engine
     * decomposes into level-2 ops. */
    enum geist_status (*transformer_block)(struct geist_backend      *be,
                                           const struct geist_tensor *x,
                                           const void                *layer_weights,
                                           struct geist_tensor       *y);

    /* ---- Optional parallelism-regime hooks ----
     *
     * Let the arch layer ask the backend to enter a thread regime tuned for
     * an execution phase, keeping host-threading details (OpenMP, thread
     * pools) out of arch code. parallel_region_begin returns an opaque token
     * that MUST be passed back to parallel_region_end to restore the prior
     * regime; the token is 0 when nothing was changed. Backends that don't
     * manage host parallelism (e.g. GPU) leave both slots null — the arch
     * layer then runs at the ambient setting. Both null or both set. */
    int  (*parallel_region_begin)(struct geist_backend       *be,
                                  enum geist_parallel_region  region);
    void (*parallel_region_end)(struct geist_backend *be, int token);

    /* Optional tensor-based linear for batched-submit (GPU) backends. The
     * engine passes the x/weight/y views it already builds alongside the
     * resolved weight, letting the backend encode the GEMM asynchronously
     * instead of receiving host pointers (which force a pipeline flush per
     * call). Return GEIST_E_UNSUPPORTED to fall back to the resolved
     * linear_m1/linear_mN host-pointer kernels. nullptr = resolved kernels
     * only. */
    enum geist_status (*linear_t)(struct geist_backend      *be,
                                  const struct geist_tensor *x,
                                  const struct geist_weight *w,
                                  const struct geist_tensor *t_w,
                                  size_t                     m,
                                  struct geist_tensor       *y);

    /* Optional device-side buffer copy. Lets the arch layer move data
     * between two buffers of the same backend without mapping host
     * pointers, so batched-submit (GPU) backends keep the copy on-device
     * and avoid a pipeline flush. Both buffers must belong to this
     * backend. nullptr = arch falls back to buffer_map + memcpy. */
    enum geist_status (*buffer_copy)(struct geist_buffer       *dst,
                                     size_t                     dst_offset,
                                     const struct geist_buffer *src,
                                     size_t                     src_offset,
                                     size_t                     n_bytes);

    /* Optional y = x * scale (scalar). F32 DENSE; y may alias x. Keeps the
     * per-layer output scaling on-device for batched-submit backends.
     * nullptr = arch scales through a mapped host pointer. */
    enum geist_status (*scale_f32)(struct geist_backend      *be,
                                   const struct geist_tensor *x,
                                   float                      scale,
                                   struct geist_tensor       *y);

    /* Optional fused out = embed_table[token_id, :] * scale. Same contract
     * as embedding_lookup plus a scalar multiply; batched-submit backends
     * keep the per-token embed/PLE-table lookups (and their scaling)
     * on-device instead of dequantizing through a mapped host pointer.
     * nullptr = arch dequantizes on the host. */
    enum geist_status (*embedding_lookup_scaled)(
        struct geist_backend      *be,
        const struct geist_tensor *embed_table,
        geist_token_t              token_id,
        float                      scale,
        struct geist_tensor       *out);

    /* Optional fused f32→f16 KV-cache append: convert k_src/v_src (F32
     * DENSE [seq, kv_heads, head_dim]) and store them at row q_position
     * of the F16 caches (F16 DENSE 3D views onto the cache buffers).
     * Presence of this slot signals that the backend's attention accepts
     * F16 K/V — GEIST_KV_AUTO resolves to an F16 cache when it is set
     * (env GEIST_KV_F16=0 forces FP32). nullptr = FP32 KV cache. */
    enum geist_status (*kv_append_f16)(struct geist_backend      *be,
                                       const struct geist_tensor *k_src,
                                       const struct geist_tensor *v_src,
                                       size_t                     q_position,
                                       struct geist_tensor       *k_cache,
                                       struct geist_tensor       *v_cache);

    /* Optional fused two-weight linear: y0 = x·w0^T, y1 = x·w1^T with one
     * pass over the activations. w0/w1 must share dtype and shape (used
     * for the k/v projections). Backends may support only a subset (e.g.
     * seq==1, Q4_K) — GEIST_E_UNSUPPORTED falls back to two linear_t
     * calls. nullptr = always separate. */
    enum geist_status (*linear_t_pair)(struct geist_backend      *be,
                                       const struct geist_tensor *x,
                                       const struct geist_weight *w0,
                                       const struct geist_tensor *t_w0,
                                       const struct geist_weight *w1,
                                       const struct geist_tensor *t_w1,
                                       size_t                     m,
                                       struct geist_tensor       *y0,
                                       struct geist_tensor       *y1);

    /* Optional fused FFN gate+up matvec with GeGLU epilogue:
     *   y = gelu_tanh(x · gate_w^T) * (x · up_w^T)
     * One kernel reads x once for both weights and applies the activation
     * in the epilogue — replaces two linears + gelu_mul. x [rows, d_in],
     * gate_w/up_w resolved weight tensors [inter, d_in] (same dtype and
     * shape), y [rows, inter]. Backends may support only a subset (e.g.
     * rows==1, Q4_K) — GEIST_E_UNSUPPORTED falls back to the decomposed
     * ops. nullptr = always decomposed. */
    enum geist_status (*ffn_gate_up)(struct geist_backend      *be,
                                     const struct geist_tensor *x,
                                     const struct geist_tensor *gate_w,
                                     const struct geist_tensor *up_w,
                                     struct geist_tensor       *y);

    /* Optional fused gemma attention q/k/v prep:
     *   q: per-head rmsnorm(q)*q_norm_w, then RoPE — in place.
     *   k (when non-null): per-head rmsnorm*k_norm_w + RoPE, written back
     *      AND appended at row q_position of k_cache.
     *   v (when non-null): per-head rmsnorm*v_norm_w, written back AND
     *      appended to v_cache.
     * q [seq, n_q_heads, hd], k/v [seq, n_kv_heads, hd], norm weights
     * [hd], cos/sin [seq, hd] views already positioned at q_position,
     * caches F32 or F16 DENSE 3D views. Half-split (non-interleaved)
     * RoPE only. Replaces up to six decomposed ops (2 norms + 2 ropes +
     * append) with two dispatches. GEIST_E_UNSUPPORTED = arch falls back
     * to the decomposed ops. nullptr = always decomposed. */
    enum geist_status (*attn_qkv_prep)(struct geist_backend      *be,
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
                                       struct geist_tensor       *v_cache);

    /* Optional fused gemma-3n PLE block:
     *   gate = gelu_tanh(x · gate_w^T) * ple_in
     *   y    = res + rmsnorm(gate · proj_w^T) * norm_w
     * x [rows, d_in], gate_w [hpl, d_in], ple_in [rows, hpl] (row stride
     * may exceed hpl — slab views), proj_w [d_model, hpl], res/y
     * [rows, d_model], norm_w [d_model]; weights are resolved tensors.
     * gate_scratch [rows, hpl] and proj_scratch [rows, d_model] hold the
     * intermediates. Backends may support only a subset (e.g. rows==1,
     * F32 weights) — anything else returns GEIST_E_UNSUPPORTED and the
     * arch runs the decomposed ops. nullptr = always decomposed. */
    enum geist_status (*ple_block)(struct geist_backend      *be,
                                   const struct geist_tensor *x,
                                   const struct geist_tensor *gate_w,
                                   const struct geist_tensor *ple_in,
                                   const struct geist_tensor *proj_w,
                                   const struct geist_tensor *res,
                                   const struct geist_tensor *norm_w,
                                   float                      eps,
                                   struct geist_tensor       *gate_scratch,
                                   struct geist_tensor       *proj_scratch,
                                   struct geist_tensor       *y);

    /* Optional fused y = res + rmsnorm(x) * w — the post-norm residual
     * step. Same contract as rmsnorm followed by add; all F32 DENSE with
     * matching shapes, y may alias res or x. nullptr = arch issues
     * separate rmsnorm + add ops. */
    enum geist_status (*rmsnorm_add)(struct geist_backend      *be,
                                     const struct geist_tensor *res,
                                     const struct geist_tensor *x,
                                     const struct geist_tensor *w,
                                     float                      eps,
                                     struct geist_tensor       *y);
};

/* ====================================================================== */
/* Backend Descriptor                                                      */
/* ====================================================================== */

/* Each backend exports one of these as a `const` extern. The engine's
 * registry array points at descriptors of compiled-in backends. */
struct geist_backend_descriptor {
    const char *name;

    /* Vtable function pointers. */
    const struct geist_backend_vtbl *vtbl;

    /* Capability matrix — pointer to array of n_caps queries this backend
     * supports natively (or via emulation). May be nullptr if the backend
     * answers all capability queries dynamically via vtbl->supports_op. */
    const struct geist_op_support_query *caps;
    size_t                               n_caps;
};

/* ====================================================================== */
/* Engine-Side Internals Visible to Backends                               */
/* ====================================================================== */

/* The full struct geist_backend definition. Backends need read access to
 * .alloc (for routing internal allocations through the user-provided
 * allocator) and to the error slot (for setting detailed messages). */
struct geist_backend {
    const struct geist_backend_descriptor *desc;
    struct geist_allocator                 alloc;

    /* Backend-private state, set during create(). */
    void *state;

    /* Error slot — set via geist_backend_set_error*. */
    enum geist_status err_code;
    char              err_msg[512];
};

/* Helpers backends call to record an error. */
void geist_backend_set_error(struct geist_backend *be, enum geist_status code,
                             const char *fmt, ...);

/* Allocator convenience: route a backend allocation through be->alloc. */
[[nodiscard]] void *geist_backend_alloc(struct geist_backend *be, size_t bytes,
                                        size_t alignment);
void                geist_backend_free(struct geist_backend *be, void *ptr);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GEIST_BACKEND_H */
