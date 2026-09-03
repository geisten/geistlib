/*
 * geist_backend.h — extension API for backend authors.
 *
 * Include this in addition to <geist.h> when implementing a new backend
 * (cpu_neon, cpu_scalar, rknn-npu, etc.). Defines the three op tables a
 * backend exports, the descriptor tying them together, and the engine-side
 * registration mechanism.
 *
 * The surface is deliberately split into three tables plus a caps struct:
 *
 *   geist_backend_vtbl        — CORE: lifecycle, buffers, the load-time
 *                               weight resolver, thread regimes. Frozen.
 *   geist_backend_primitives  — the decomposed reference ops. A backend
 *                               implementing core + primitives runs every
 *                               model, just unfused.
 *   geist_backend_fused       — nullable fast paths. Optimizations only;
 *                               each has a decomposed twin in primitives.
 *   geist_backend_caps        — explicit capability bools. Never inferred
 *                               from slot presence or backend name.
 *
 * Adding backend functionality — where does it go?
 *   1. New dtype/layout for a linear path?
 *      → resolver case in resolve_weight. Never a new slot.
 *   2. New primitive (a new arch needs it for its DECOMPOSED path)?
 *      → geist_backend_primitives. Requires a cpu_scalar reference impl
 *        + a parity test. Expected to be rare.
 *   3. New fusion / fast path?
 *      → geist_backend_fused. Non-negotiable: (a) the decomposed
 *        primitive path already produces the same result, (b) plan-level
 *        binding in the arch (no `fused->x != nullptr` at call sites once
 *        probe-and-bind lands), (c) a fused-vs-decomposed parity test.
 *   4. New capability signal?
 *      → a bool in geist_backend_caps, with its consumer named in the
 *        comment. Never name-sniffing, never slot-presence inference.
 *   5. The core vtbl is FROZEN. Additions need a reason no other bucket
 *      can absorb.
 *
 * @stability EXPERIMENTAL — table layout may evolve until 1.0.
 */
#ifndef GEIST_BACKEND_H
#define GEIST_BACKEND_H

#include <geist.h>

#include <pthread.h>
#include <geist_types.h> /* tensor / dtype types the tables speak in */
#include <geist_weight.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ====================================================================== */
/* Core vtable — lifecycle, memory, resolver, thread regimes.              */
/* ====================================================================== */

/* Execution phase the arch layer is about to enter. Backends may tune their
 * parallelism regime per phase (see parallel_region_begin). Prefill is
 * compute-bound and scales with cores; decode (m=1 GEMV) is memory-bound. */
enum geist_parallel_region {
    GEIST_REGION_PREFILL_BATCH,
    GEIST_REGION_DECODE_STEP,
};

/* Backends fill this in (statically) and reference it from their
 * descriptor. Engine dispatches through here. */
struct geist_backend_vtbl {
    /* ---- Lifecycle ---- */

    /* Optional create-time hook. Backend allocates per-instance state and
     * stashes it into geist_backend->state. Returns GEIST_OK on success.
     * If non-OK, engine reclaims geist_backend memory and propagates. */
    enum geist_status (*create)(struct geist_backend *be, const struct geist_backend_opts *opts);

    /* Required. Tear down per-instance state and any held buffers. */
    void (*destroy)(struct geist_backend *be);

    /* ---- Buffer ops ---- */

    /* Allocate a buffer of the given size and role. Backend may pick
     * device-local vs host-coherent based on memory_flags. */
    enum geist_status (*buffer_create)(struct geist_backend  *be,
                                       size_t                 bytes,
                                       enum geist_buffer_role role,
                                       unsigned int           memory_flags,
                                       struct geist_buffer  **out);

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
     * per-op path. */
    enum geist_status (*resolve_weight)(struct geist_backend *be, struct geist_weight *w);

    /* ---- Optional parallelism-regime hooks ----
     *
     * Let the arch layer ask the backend to enter a thread regime tuned for
     * an execution phase, keeping host-threading details (OpenMP, thread
     * pools) out of arch code. parallel_region_begin returns an opaque token
     * that MUST be passed back to parallel_region_end to restore the prior
     * regime; the token is 0 when nothing was changed. Backends that don't
     * manage host parallelism (e.g. GPU) leave both slots null — the arch
     * layer then runs at the ambient setting. Both null or both set;
     * caps.manages_host_threads mirrors their presence. */
    int (*parallel_region_begin)(struct geist_backend *be, enum geist_parallel_region region);
    void (*parallel_region_end)(struct geist_backend *be, int token);
};

/* ====================================================================== */
/* Primitive ops — the decomposed reference path.                          */
/* ====================================================================== */

/* Each op takes geist_tensor inputs/outputs whose .buffer was created via
 * this same backend. Return GEIST_OK on success; on error, set the backend
 * error slot via geist_backend_set_error.
 *
 * These are what the arch's fallback path is written against: a backend
 * implementing core + primitives runs every model, just unfused. Individual
 * slots may still be nullptr where an arch-side host fallback exists (the
 * comment on each slot says which).
 *
 * (The legacy `linear` primitive was dropped after the resolver path
 * (resolve_weight + geist_weight::linear_m1/_mN) covered every production
 * dtype. Adding a new linear path means adding a resolver case, not a
 * slot.) */
struct geist_backend_primitives {
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

    /* y = silu(x) = x / (1 + exp(-x)). F32 DENSE, x and y can be the
     * same tensor. SiLU is Llama 2/3 + BitNet b1.58 3B's SwiGLU
     * activation. */
    enum geist_status (*silu)(struct geist_backend      *be,
                              const struct geist_tensor *x,
                              struct geist_tensor       *y);

    /* y = max(x, 0)^2. F32 DENSE, x and y can be the same tensor.
     * Squared ReLU is BitNet b1.58 2B-4T's FFN activation; combining
     * the threshold + the square in one pass halves memory traffic
     * vs. relu(x) followed by mul(y, y). May be nullptr on backends
     * that don't implement it; callers must check. */
    enum geist_status (*relu_squared)(struct geist_backend      *be,
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

    /* y = x * scale (scalar). F32 DENSE; y may alias x. Keeps the
     * per-layer output scaling on-device for batched-submit backends.
     * nullptr = arch scales through a mapped host pointer. */
    enum geist_status (*scale_f32)(struct geist_backend      *be,
                                   const struct geist_tensor *x,
                                   float                      scale,
                                   struct geist_tensor       *y);
};

/* ====================================================================== */
/* Fused fast paths — all optional, all with decomposed twins.             */
/* ====================================================================== */

/* Fused-op identifiers for load-time probing (see `supported` below).
 * Only ops whose call sites are plan-bound need an id; the rest join as
 * their call sites migrate (policy rule 3). */
enum geist_fused_op {
    GEIST_FUSED_GELU_TANH_MUL,
    GEIST_FUSED_SILU_MUL,
    GEIST_FUSED_GELU_TANH_MUL_SCALED,
    GEIST_FUSED_FFN_GEGLU_Q4Q6_MN,
    GEIST_FUSED_FFN_GATE_UP,
    GEIST_FUSED_FFN_NORM_GATE_UP,
    GEIST_FUSED_RMSNORM_ADD,
    GEIST_FUSED_ATTN_QKV_PREP,
    GEIST_FUSED_PLE_BLOCK,
    GEIST_FUSED_EMBEDDING_LOOKUP_SCALED,
    GEIST_FUSED_ARGMAX_F32,
};

/* Load-time capability probe for one fused op at one layer's geometry.
 * `m` carries the regime: 1 = decode; >1 = prefill, and a `true` answer
 * promises the op succeeds for ANY row count in [1, m]. The weight
 * pointers are the layer's RESOLVED weights (post resolve_weight, post
 * upload), so backends can check residency and layout, not just dtype;
 * fields an op doesn't use are nullptr/0. */
struct geist_fusion_query {
    enum geist_fused_op op;
    size_t              m;
    size_t              d_model;
    size_t              inter;
    size_t              head_dim;    /* attn_qkv_prep */
    size_t              n_q_heads;   /* attn_qkv_prep */
    size_t              n_kv_heads;  /* attn_qkv_prep */
    uint16_t            table_dtype; /* embedding_lookup_scaled: geist_dtype
                                      * of the (tensor-typed) lookup table */
    const struct geist_weight *gate_w;
    const struct geist_weight *up_w;
    const struct geist_weight *down_w;
};

/* Complete Gated-DeltaNet mixer after its four input projections. All
 * tensors are F32 DENSE backend views. qkv is [seq, conv_dim], z is
 * [seq, n_v_heads * head_v] and is overwritten with the mixer output,
 * beta/alpha are [seq, n_v_heads]. conv_state and delta_state are
 * persistent, mutable session state. A backend fusion must advance both
 * state tensors exactly once for every input row before returning OK. */
struct geist_deltanet_mix_args {
    struct geist_tensor       *qkv;
    struct geist_tensor       *z;
    const struct geist_tensor *beta;
    const struct geist_tensor *alpha;
    const struct geist_tensor *conv_w;
    const struct geist_tensor *ssm_a;
    const struct geist_tensor *dt_bias;
    const struct geist_tensor *norm_w;
    struct geist_tensor       *conv_state;
    struct geist_tensor       *delta_state;
    size_t                     seq;
    size_t                     n_k_heads;
    size_t                     n_v_heads;
    size_t                     head_k;
    size_t                     head_v;
    size_t                     conv_kernel;
    float                      eps;
};

/* Every slot here is an OPTIMIZATION: the arch must be able to produce the
 * same result from core + primitives. nullptr = always decomposed; a
 * non-null slot may still return GEIST_E_UNSUPPORTED for geometries its
 * kernel doesn't cover, and the caller falls back.
 *
 * Probe-and-bind: call sites migrate from per-call negotiation to
 * consulting `supported` once at plan-build time (the FFN front is
 * converted; remaining stages migrate as touched). The contract is
 * strict: if `supported` returns true for a query, the op MUST return
 * GEIST_OK (or a real error like OOM — never GEIST_E_UNSUPPORTED) for
 * every call matching that query. Probe and kernel live side by side in
 * the backend; test_fused_probe_agreement_unit checks the pairing. */
struct geist_backend_fused {
    /* Load-time probe backing the plan-bound call sites. nullptr = the
     * backend answers no to every probe (its fusions are then only
     * reachable through the remaining per-call negotiation sites). */
    bool (*supported)(struct geist_backend *be, const struct geist_fusion_query *q);

    /* y = gelu_tanh(x) * z. F32 DENSE. FFN fast path for GEGLU;
     * callers fall back to gelu_tanh + mul when nullptr. */
    enum geist_status (*gelu_tanh_mul)(struct geist_backend      *be,
                                       const struct geist_tensor *x,
                                       const struct geist_tensor *z,
                                       struct geist_tensor       *y);

    /* Fused y = silu(x) * z — the SwiGLU epilogue twin of
     * gelu_tanh_mul. Must match prims->silu + prims->mul bit-exactly
     * (same formula, one pass). nullptr = callers run the pair. */
    enum geist_status (*silu_mul)(struct geist_backend      *be,
                                  const struct geist_tensor *x,
                                  const struct geist_tensor *z,
                                  struct geist_tensor       *y);

    /* y[t,j] = gelu_tanh(x[t,j]) * z[t,j] * scale[j].
     * GEGLU+AWQ fusion for transformer FFNs. scale is per-channel
     * across the last dimension. nullptr means callers use gelu_tanh_mul
     * and a separate scale pass. */
    enum geist_status (*gelu_tanh_mul_scaled)(struct geist_backend      *be,
                                              const struct geist_tensor *x,
                                              const struct geist_tensor *z,
                                              const float               *scale,
                                              struct geist_tensor       *y);

    /* Text-FFN fast path for Gemma-style GEGLU:
     *   y = down(gelu_tanh(gate(x)) * up(x) * optional_down_scale)
     * Backends may return GEIST_E_UNSUPPORTED when dtype/shape/layout do
     * not match their fused kernel. The caller then falls back to
     * decomposed ops. */
    enum geist_status (*ffn_geglu_q4q6_mN)(struct geist_backend      *be,
                                           size_t                     m,
                                           size_t                     d_model,
                                           size_t                     inter,
                                           const float                x[static m * d_model],
                                           const struct geist_weight *gate,
                                           const struct geist_weight *up,
                                           const struct geist_weight *down,
                                           const float               *down_scale, /* nullable */
                                           float                      y[static m * d_model]);

    /* Tensor-based linear for batched-submit (GPU) backends. The engine
     * passes the x/weight/y views it already builds alongside the
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

    /* Fused two-weight linear: y0 = x·w0^T, y1 = x·w1^T with one pass
     * over the activations. w0/w1 must share dtype and shape (used for
     * the k/v projections). Backends may support only a subset (e.g.
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

    /* Fused out = embed_table[token_id, :] * scale. Same contract as
     * primitives->embedding_lookup plus a scalar multiply; batched-submit
     * backends keep the per-token embed/PLE-table lookups (and their
     * scaling) on-device instead of dequantizing through a mapped host
     * pointer. nullptr = arch dequantizes on the host. */
    enum geist_status (*embedding_lookup_scaled)(struct geist_backend      *be,
                                                 const struct geist_tensor *embed_table,
                                                 geist_token_t              token_id,
                                                 float                      scale,
                                                 struct geist_tensor       *out);

    /* Batched twin of embedding_lookup_scaled: one dispatch embeds
     * n_rows chunk tokens into out [n_rows, d_model]. Decomposed twin:
     * n_rows calls of embedding_lookup_scaled (the arch falls back to
     * that loop on nullptr or non-OK). Consumer: the prefill chunk
     * loop, which otherwise pays one tiny dispatch per token. */
    enum geist_status (*embedding_lookup_scaled_rows)(struct geist_backend      *be,
                                                      size_t                     n_rows,
                                                      const struct geist_tensor *embed_table,
                                                      const geist_token_t        ids[static n_rows],
                                                      float                      scale,
                                                      struct geist_tensor       *out);

    /* Fused f32→f16 KV-cache append: convert k_src/v_src (F32 DENSE
     * [seq, kv_heads, head_dim]) and store them at row q_position of the
     * F16 caches (F16 DENSE 3D views onto the cache buffers). Whether the
     * backend's attention accepts F16 K/V is signalled by
     * caps.kv_f16_attention, NOT by this slot's presence. nullptr = FP32
     * KV cache only. */
    enum geist_status (*kv_append_f16)(struct geist_backend      *be,
                                       const struct geist_tensor *k_src,
                                       const struct geist_tensor *v_src,
                                       size_t                     q_position,
                                       struct geist_tensor       *k_cache,
                                       struct geist_tensor       *v_cache);

    /* Device greedy argmax over a [1, n] F32 logits row. The backend
     * flushes its pending pipeline for a 4-byte index read instead of the
     * arch mapping the whole logits row. Tie-break = lowest index
     * (matches geist_sampler_argmax). GEIST_E_UNSUPPORTED = arch scans on
     * the host. nullptr = always host. */
    enum geist_status (*argmax_f32)(struct geist_backend      *be,
                                    const struct geist_tensor *logits,
                                    int32_t                   *out_index);

    /* Fused FFN gate+up matvec with GeGLU epilogue:
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

    /* ffn_gate_up with the pre-FFN rmsnorm folded in:
     *   xn = rmsnorm(x) * norm_w;  y = gelu_tanh(xn·gate_w^T) * (xn·up_w^T)
     * Each workgroup recomputes the row's inverse RMS from x — cheaper
     * than a separate norm dispatch on the serial decode chain. Same
     * contract as ffn_gate_up otherwise; GEIST_E_UNSUPPORTED falls back
     * to the decomposed rmsnorm + FFN front. nullptr = always decomposed. */
    enum geist_status (*ffn_norm_gate_up)(struct geist_backend      *be,
                                          const struct geist_tensor *x,
                                          const struct geist_tensor *norm_w,
                                          float                      eps,
                                          const struct geist_tensor *gate_w,
                                          const struct geist_tensor *up_w,
                                          struct geist_tensor       *y);

    /* Fused gemma attention q/k/v prep:
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

    /* Fused gemma-3n PLE block:
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

    /* Fused y = res + rmsnorm(x) * w — the post-norm residual step. Same
     * contract as rmsnorm followed by add; all F32 DENSE with matching
     * shapes, y may alias res or x. nullptr = arch issues separate
     * rmsnorm + add ops. */
    enum geist_status (*rmsnorm_add)(struct geist_backend      *be,
                                     const struct geist_tensor *res,
                                     const struct geist_tensor *x,
                                     const struct geist_tensor *w,
                                     float                      eps,
                                     struct geist_tensor       *y);

    /* Gated-DeltaNet's causal depthwise convolution, q/k normalization,
     * gated delta-rule recurrence, per-head RMSNorm and SiLU output gate.
     * See geist_deltanet_mix_args. This is a stateful fusion: OK means z
     * contains the mixer output and both recurrent state tensors have
     * advanced; GEIST_E_UNSUPPORTED guarantees neither state was changed
     * and lets the architecture run its host reference implementation. */
    enum geist_status (*deltanet_mix)(struct geist_backend                 *be,
                                      const struct geist_deltanet_mix_args *args);

    /* Split qwen35's joint per-head [query | output-gate] projection into
     * dense query and gate rows without a host gather. joint is
     * [rows, heads * 2 * head_dim], q/gate are [rows, heads * head_dim]. */
    enum geist_status (*attn_qgate_split)(struct geist_backend      *be,
                                          const struct geist_tensor *joint,
                                          size_t                     heads,
                                          size_t                     head_dim,
                                          struct geist_tensor       *q,
                                          struct geist_tensor       *gate);

    /* y = x * sigmoid(gate), elementwise over matching F32 DENSE rows.
     * y may alias x; nullptr leaves the architecture's mapped fallback. */
    enum geist_status (*sigmoid_mul)(struct geist_backend      *be,
                                     const struct geist_tensor *x,
                                     const struct geist_tensor *gate,
                                     struct geist_tensor       *y);
};

/* ====================================================================== */
/* Capability bits — explicit, never inferred.                             */
/* ====================================================================== */

/* Plain bools, embedded by value in the descriptor (zero-init = none).
 * Each names its consumer; a bit nobody reads gets deleted. */
struct geist_backend_caps {
    /* attention() reads F16 K/V views. Consumer: KV-mode resolution —
     * GEIST_KV_AUTO may pick an F16 cache (env GEIST_KV_F16=0 forces
     * FP32). Requires fused->kv_append_f16 to be usable. */
    bool kv_f16_attention;

    /* Batched-submit pipeline (GPU): host round-trips stall it.
     * Consumers: spec head disables itself; weight loading prefers the
     * backend-arena mode over mmap-alias; KV-mode resolution defaults to
     * FP32; session m_max cap is raised (no CPU quant-kernel stack
     * limit). */
    bool batched_submit;

    /* parallel_region_begin/end are implemented and meaningful.
     * Consumer: documentation/asserts only — callers still null-check
     * the hooks themselves. */
    bool manages_host_threads;

    /* Weights must live inside backend buffers (buffer_create + upload);
     * host-mmap'd GGUF pages cannot be bound by the device. Consumer:
     * weight loading disables the mmap-alias default (backend-arena
     * mode). Unified-memory GPUs (metal) leave this false. */
    bool weights_need_backend_arena;

    /* Bumped by the backend whenever kernel performance character
     * changes enough to invalidate measured calibrations without any
     * tunable being renamed (the "faster kernel, same knob" case).
     * Feeds the opaque calibration key. */
    uint32_t calibration_generation;

    /* deltanet_mix() sub-chunks long sequences internally (the O(C²)
     * chunk recipe runs at its own optimal granularity regardless of
     * the caller's m). Consumer: state_create skips the DN m_max cap,
     * so the surrounding GEMMs keep their occupancy-friendly batch. */
    bool dn_subchunk;

    /* Preferred prefill batch size (m_max) measured for this backend;
     * 0 = use the arch default. Consumer: state_create's m_max default.
     * (metal: 128 — mm_sg GEMM fast paths want rows%64==0 and fewer,
     * larger chunks.) */
    size_t preferred_m_max;

    /* Largest batch (rows) the backend's kernels accept per call;
     * 0 = uncapped. Every backend with per-call row limits MUST set
     * this (CPU quant kernels size stack arrays from it). Consumers:
     * session_alloc's m_max validation, the GEIST_M_MAX env clamp,
     * exec_plan's prefill-regime probes. */
    size_t max_m;

    /* KV-cache dtype default under GEIST_KV_AUTO (opts and GEIST_KV_*
     * env overrides win; see resolve_kv_mode). GEIST_KV_AUTO (= 0 =
     * zero-init) means no preference — the arch falls back to FP32.
     * Platform-specific compilands (cpu_neon/cpu_scalar) set this per
     * their own build target; the arch layer never sniffs platforms. */
    enum geist_kv_mode preferred_kv_mode;
};

/* ====================================================================== */
/* Backend Descriptor                                                      */
/* ====================================================================== */

/* ====================================================================== */
/* Calibration — measured per-machine tuning (EXPERIMENTAL)                */
/* ====================================================================== */

/* One measurable tuning knob a backend exposes to the calibration
 * driver. `name` doubles as the blob line key and matches the knob's
 * GEIST_* env override (documented per backend). `measure` runs the
 * backend-specific A/B or sweep for THIS machine and writes the best
 * value; budget_ns is a target, not a deadline. The driver owns
 * everything around it (repeats, variance notes, serialization). */
enum geist_tunable_kind {
    GEIST_TUNABLE_BOOL, /* value 0 / 1 */
    GEIST_TUNABLE_SIZE, /* positive size-class value */
};

struct geist_tunable {
    const char             *name;
    enum geist_tunable_kind kind;
    enum geist_status (*measure)(struct geist_backend *be,
                                 uint64_t              budget_ns,
                                 int64_t              *out_value);
};

/* Each backend exports one of these as a `const` extern. The engine's
 * registry array points at descriptors of compiled-in backends. */
struct geist_backend_descriptor {
    const char *name;

    /* Core table. Required. */
    const struct geist_backend_vtbl *vtbl;

    /* Primitive ops. Required (individual slots may be nullptr where a
     * documented host fallback exists). */
    const struct geist_backend_primitives *prims;

    /* Fused fast paths. nullptr = backend has no fusions at all. */
    const struct geist_backend_fused *fused;

    /* Measurable tuning knobs for geist_backend_calibrate(). nullptr =
     * backend has no calibrated tunables (seeds + env only). */
    const struct geist_tunable *(*tunables)(size_t *out_count);

    /* Explicit capability bits (by value; zero-init = none). */
    struct geist_backend_caps caps;
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

    /* Error slot — DIAGNOSTIC DETAIL ONLY. Control flow runs through
     * enum geist_status returns; code that reads geist_backend_errcode
     * to decide WHETHER something failed is a bug. Set via
     * geist_backend_set_error. Writes are guarded
     * by err_mu so concurrent sessions can't interleave garbage into the
     * message; reads (geist_backend_errmsg) return a pointer into the
     * slot and are last-writer-wins across sessions — read it after a
     * failing call on your own session.
     * ponytail: one shared slot; per-session error slots if concurrent
     * error attribution ever matters. */
    enum geist_status err_code;
    char              err_msg[512];
    pthread_mutex_t   err_mu;

    /* Applied calibration values (EXPERIMENTAL). Written only by
     * geist_backend_apply_calibration before the first resolve_weight;
     * `locked` is set by the weight-load path and freezes the store.
     * Backends consult it via geist_calibration_lookup when building
     * their kernel policy. */
    struct geist_calibration_state {
        bool   locked;
        size_t n_values;
        struct geist_calibration_value {
            char    name[48];
            int64_t value;
        } values[32];
    } calibration;
};

/* ====================================================================== */
/* Calibration API (EXPERIMENTAL)                                          */
/* ====================================================================== */

/* Measures all tunables of the backend and serializes the result into a
 * portable UTF-8 text blob owned by the caller. budget_ns is a target
 * budget, not a hard deadline. If buf is nullptr or buf_size is
 * insufficient, *required_size receives the needed size (including the
 * terminating NUL) and the call returns GEIST_E_INVALID_ARG without
 * measuring. The library performs no persistence — consumers decide
 * whether and where the blob is cached (key it via
 * geist_backend_calibration_key). @stability EXPERIMENTAL */
enum geist_status geist_backend_calibrate(struct geist_backend *be,
                                          uint64_t              budget_ns,
                                          char                 *buf,
                                          size_t                buf_size,
                                          size_t               *required_size);

/* Applies a caller-supplied calibration blob. Accepted only if the
 * blob's key exactly matches this backend's current execution
 * environment (otherwise GEIST_E_STALE_CALIBRATION) and only before the
 * first weight resolve (otherwise GEIST_E_INVALID_STATE). Atomic: on
 * any validation error the backend is unchanged. Effective precedence
 * per tunable: env override ?? calibration ?? built-in seed.
 * @stability EXPERIMENTAL */
enum geist_status geist_backend_apply_calibration(struct geist_backend *be,
                                                  const char           *blob,
                                                  size_t                blob_size);

/* Writes the opaque calibration key for this backend's execution
 * environment (same required_size contract as calibrate). The key's
 * contents are implementation-defined; consumers may use it verbatim as
 * a cache-lookup key. @stability EXPERIMENTAL */
enum geist_status geist_backend_calibration_key(const struct geist_backend *be,
                                                char                       *buf,
                                                size_t                      buf_size,
                                                size_t                     *required_size);

/* Backend-side helper: the applied calibration value for `name`, or
 * false when none was applied (use the seed). */
bool geist_calibration_lookup(const struct geist_backend *be, const char *name, int64_t *out);

/* The fused table, or a shared all-null table when the backend has none —
 * lets callers null-check slots without null-checking the table. */
extern const struct geist_backend_fused geist_backend_no_fused;
static inline const struct geist_backend_fused *
geist_backend_fused_tbl(const struct geist_backend *be) {
    return be->desc->fused != nullptr ? be->desc->fused : &geist_backend_no_fused;
}

/* Helpers backends call to record an error. */
void geist_backend_set_error(struct geist_backend *be,
                             enum geist_status     code,
                             const char           *fmt,
                             ...);

/* Allocator convenience: route a backend allocation through be->alloc. */
[[nodiscard]] void *geist_backend_alloc(struct geist_backend *be, size_t bytes, size_t alignment);
void                geist_backend_free(struct geist_backend *be, void *ptr);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GEIST_BACKEND_H */
