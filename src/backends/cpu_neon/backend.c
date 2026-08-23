/*
 * src/backends/cpu_neon/backend.c — ARM NEON-optimized backend.
 *
 * Layer: BACKEND.
 *
 * B-3 lite (this commit): walking-skeleton mirroring cpu_scalar's shape,
 *                         with linear() routing F32 DENSE through cblas_sgemm
 *                         (Accelerate on Mac, OpenBLAS on Pi 5). Quantized
 *                         kernels (Q3_K, Q4_K, Q8_0) wrap the existing
 *                         gguf_quant.c NEON paths in subsequent sub-commits.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"

#include <geist.h>
#include <geist_backend.h>

#include "quant.h"
#include "heap.h"

#include <stdarg.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- Lifecycle ---------- */

static void cpu_neon_omp_pool_init(void);

[[nodiscard]] static enum geist_status cpu_neon_create(struct geist_backend            *be,
                                                       const struct geist_backend_opts *opts) {
    (void) opts;
    struct cpu_neon_state *st =
            geist_backend_alloc(be, sizeof(*st), alignof(struct cpu_neon_state));
    if (st == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "cpu_neon: state alloc failed");
        return GEIST_E_OOM;
    }
    *st               = (struct cpu_neon_state) {0};
    st->ws_generation = cpu_neon_ws_next_generation();
    geist_hw_probe_fill(&st->hw);
    st->policy = cpu_neon_kernel_policy_default(&st->hw);

    be->state = st;
    cpu_neon_omp_pool_init();
    return GEIST_OK;
}

static void cpu_neon_destroy(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return;
    }
    struct cpu_neon_state *st = (struct cpu_neon_state *) be->state;
    /* Backend-owned scratch: freed directly via the workspace. No OMP
     * barrier needed because the storage lives on `st`, not in TLS. */
    cpu_neon_ws_destroy_all(st);
    geist_backend_free(be, be->state);
    be->state = nullptr;
}

/* ---------- Buffer ops (mirror cpu_scalar) ---------- */

[[nodiscard]] static enum geist_status cpu_neon_buffer_create(struct geist_backend  *be,
                                                              size_t                 bytes,
                                                              enum geist_buffer_role role,
                                                              unsigned int           memory_flags,
                                                              struct geist_buffer  **out) {
    if (out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (bytes == 0) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_neon: zero-byte buffer");
        return GEIST_E_INVALID_ARG;
    }

    struct geist_buffer *buf = geist_backend_alloc(be, sizeof(*buf), alignof(struct geist_buffer));
    if (buf == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "cpu_neon: buffer handle alloc");
        return GEIST_E_OOM;
    }
    void *host = heap_alloc_aligned(bytes, OPTIMAL_ALIGNMENT);
    if (host == nullptr) {
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_OOM, "cpu_neon: %zu-byte host alloc", bytes);
        return GEIST_E_OOM;
    }
    *buf = (struct geist_buffer) {
            .host         = host,
            .bytes        = bytes,
            .role         = role,
            .memory_flags = memory_flags,
    };
    *out = buf;
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status cpu_neon_buffer_create_aliased(struct geist_backend *be,
                                                                      void  *host_ptr,
                                                                      size_t n_bytes,
                                                                      enum geist_buffer_role role,
                                                                      struct geist_buffer  **out) {

    if (out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (host_ptr == nullptr || n_bytes == 0) {
        geist_backend_set_error(
                be, GEIST_E_INVALID_ARG, "cpu_neon: aliased buffer needs host_ptr + bytes");
        return GEIST_E_INVALID_ARG;
    }
    struct geist_buffer *buf = geist_backend_alloc(be, sizeof(*buf), alignof(struct geist_buffer));
    if (buf == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "cpu_neon: buffer handle alloc");
        return GEIST_E_OOM;
    }
    *buf = (struct geist_buffer) {
            .host         = host_ptr,
            .bytes        = n_bytes,
            .role         = role,
            .memory_flags = GEIST_MEMORY_ALIASED,
    };
    *out = buf;
    return GEIST_OK;
}

static void cpu_neon_buffer_destroy(struct geist_backend *be, struct geist_buffer *buf) {
    if (buf == nullptr) {
        return;
    }
    /* Aliased buffers (P0.3): host_ptr is owned externally — typically an
     * mmap'd region the gguf_reader retains. Don't free, just discard
     * the metadata header. */
    if ((buf->memory_flags & GEIST_MEMORY_ALIASED) == 0 && buf->host != nullptr) {
        safe_free(&buf->host);
    }
    geist_backend_free(be, buf);
}

[[nodiscard]] static enum geist_status cpu_neon_buffer_upload(struct geist_buffer *buf,
                                                              size_t               n_bytes,
                                                              const uint8_t src[static n_bytes]) {
    if (buf == nullptr || src == nullptr || n_bytes > buf->bytes) {
        return GEIST_E_INVALID_ARG;
    }
    memcpy(buf->host, src, n_bytes);
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status cpu_neon_buffer_download(size_t  n_bytes,
                                                                uint8_t dst[static n_bytes],
                                                                const struct geist_buffer *buf) {
    if (buf == nullptr || dst == nullptr || n_bytes > buf->bytes) {
        return GEIST_E_INVALID_ARG;
    }
    memcpy(dst, buf->host, n_bytes);
    return GEIST_OK;
}

static void *cpu_neon_buffer_map(struct geist_buffer *buf) {
    return buf != nullptr ? buf->host : nullptr;
}

static void cpu_neon_buffer_unmap(struct geist_buffer *buf) {
    (void) buf;
}

/* ---------- Parallelism-regime hooks (OpenMP thread management) ----------
 *
 * cpu_neon's matmul kernels parallelize via OpenMP `parallel for`, so the
 * global OMP_NUM_THREADS governs throughput — and the count that suits one
 * phase hurts another. The arch layer calls these around each phase; we map
 * the phase to omp_set_num_threads and restore afterwards. GPU/other backends
 * leave the vtable slots null and the arch layer runs at the ambient setting. */
#if defined(_OPENMP)
#include <omp.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
/* Performance ("P") core count on Apple Silicon. Both prefill and decode
 * regress badly when the schedule includes the slow efficiency ("E") cores:
 * a static OMP partition waits on the E-core chunks (M1 Max, Gemma 4 Q4_K_M
 * pp512: 10 cores → 91 tps vs 8 P-cores → 145; tg128: likewise). num_procs
 * counts all 10, so default to the P-core count instead. Returns 0 if the
 * sysctl is unavailable. */
static int apple_perf_cores(void) {
    int    v   = 0;
    size_t len = sizeof v;
    if (sysctlbyname("hw.perflevel0.physicalcpu", &v, &len, nullptr, 0) == 0 && v > 0) {
        return v;
    }
    return 0;
}
#else
static int apple_perf_cores(void) {
    return 0;
}
#endif

/* One-time OMP pool sizing at backend create. The region hooks below
 * cap the ACTIVE thread count per phase, but with an ambient pool of
 * all cores (OMP default: num_procs) the scheduler is free to place
 * the capped subset on efficiency cores — measured on M-series
 * (qwen3.5-4B decode): ambient 10-thread pool 114 ms/tok vs an
 * 8-thread pool 47 ms/tok, SAME active cap. Size the pool itself to
 * the performance-core count once, before any parallel region exists.
 * An explicit OMP_NUM_THREADS wins — never override the user. */
static void cpu_neon_omp_pool_init(void) {
    static _Atomic int done = 0;
    int                exp  = 0;
    if (!atomic_compare_exchange_strong(&done, &exp, 1))
        return;
    /* Decode fires ~200 tiny parallel regions per token; with the
     * default passive wait policy the workers sleep between them and
     * the wake latency dominates (measured qwen3.5-4B decode:
     * 163 ms/tok passive vs 46 ms active, same thread count). The
     * policy is runtime-init-only, but backend create runs before the
     * first parallel region, so setenv still takes effect. setenv with
     * overwrite=0 — an explicit user policy always wins. */
    setenv("OMP_WAIT_POLICY", "active", 0);
    if (getenv("OMP_NUM_THREADS") != nullptr)
        return;
    const int pc = apple_perf_cores();
    if (pc > 0)
        omp_set_num_threads(pc);
}

/* Target OMP thread count for `region`, cached on first use. 0 = "leave the
 * ambient OMP_NUM_THREADS alone". Env overrides force a count (>0) or disable
 * the adjustment (0): GEIST_PREFILL_THREADS, GEIST_DECODE_THREADS. */
static int cpu_neon_region_thread_count(enum geist_parallel_region region) {
    if (region == GEIST_REGION_PREFILL_BATCH) {
        /* Prefill is COMPUTE-bound (matmul) and scales ~linearly with cores, so
         * use them all — except on Apple, where the slow efficiency cores stall
         * a static OMP partition, so use the performance-core count instead
         * (M1 Max pp512: 91 → 145 tps once E-cores drop). On the homogeneous
         * Pi 5 all 4 A76 cores help (clean pp256 4t 30 vs 3t 24); measure on a
         * QUIESCED box — a stray background process eating a core inverts this
         * (4 OMP threads then oversubscribe). */
        static _Atomic int n = -1;
        if (n < 0) {
            const char *env = getenv("GEIST_PREFILL_THREADS");
            if (env != nullptr && env[0] != '\0') {
                const int v = atoi(env);
                n           = (v > 0) ? v : 0;
            } else {
                const int pc = apple_perf_cores();
                n            = (pc > 0) ? pc : omp_get_num_procs();
            }
        }
        return n;
    }
    /* GEIST_REGION_DECODE_STEP: decode (m=1 GEMV) is dominated by the 262K-wide
     * lm_head plus ~210 small matmuls. It scales with P-cores but regresses when
     * E-cores join the static schedule. Pi 5 (shared LPDDR): 3 threads beat 4.
     * Apple: P-core count (M1 Max tg128: ambient/E-core-polluted → ~10 tps,
     * 8 P-cores → ~31 tps). Other targets keep the ambient count. */
    static _Atomic int n = -1;
    if (n < 0) {
        const char *env = getenv("GEIST_DECODE_THREADS");
        if (env != nullptr && env[0] != '\0') {
            const int v = atoi(env);
            n           = (v > 0) ? v : 0;
        } else {
#if defined(GEIST_TARGET_PI5)
            n = 3;
#else
            /* Leave one P-core for the OMP master / OS: decode fires ~210 tiny
             * matmuls per token, and saturating all P-cores makes the schedule
             * contend with coordination work. M1 Max tg128: 8 P-cores → ~25 tps
             * (noisy), 7 → ~30 tps (stable). Mirrors Pi 5's 3-of-4. */
            const int pc = apple_perf_cores();
            n            = (pc > 1) ? pc - 1 : 0;
#endif
        }
    }
    return n;
}

static int cpu_neon_parallel_region_begin(struct geist_backend      *be,
                                          enum geist_parallel_region region) {
    (void) be;
    const int target = cpu_neon_region_thread_count(region);
    if (target <= 0)
        return 0;
    const int prev = omp_get_max_threads();
    /* Prefill bumps in either direction; decode only caps DOWN — never adds
     * threads to a memory-bound GEMV. */
    const bool apply = (region == GEIST_REGION_DECODE_STEP) ? (target < prev) : (target != prev);
    if (!apply)
        return 0;
    omp_set_num_threads(target);
    return prev; /* >0: restore to this in _end */
}

static void cpu_neon_parallel_region_end(struct geist_backend *be, int token) {
    (void) be;
    if (token > 0)
        omp_set_num_threads(token);
}
#else  /* !_OPENMP — no host threading to manage. */
static void cpu_neon_omp_pool_init(void) {
}

static int cpu_neon_parallel_region_begin(struct geist_backend      *be,
                                          enum geist_parallel_region region) {
    (void) be;
    (void) region;
    return 0;
}
static void cpu_neon_parallel_region_end(struct geist_backend *be, int token) {
    (void) be;
    (void) token;
}
#endif /* _OPENMP */

/* ---------- Vtable + Descriptor ---------- */

static const struct geist_backend_vtbl cpu_neon_vtbl = {
        .create                = cpu_neon_create,
        .destroy               = cpu_neon_destroy,
        .buffer_create         = cpu_neon_buffer_create,
        .buffer_destroy        = cpu_neon_buffer_destroy,
        .buffer_create_aliased = cpu_neon_buffer_create_aliased,
        .buffer_upload         = cpu_neon_buffer_upload,
        .buffer_download       = cpu_neon_buffer_download,
        .buffer_map            = cpu_neon_buffer_map,
        .buffer_unmap          = cpu_neon_buffer_unmap,
        .resolve_weight        = cpu_neon_resolve_weight,
        .parallel_region_begin = cpu_neon_parallel_region_begin,
        .parallel_region_end   = cpu_neon_parallel_region_end,
};

/* Probe pairing for the fused table below. The GEGLU tile kernel's
 * conditions mirror cpu_neon_ffn_geglu_q4q6_mN's entry checks — keep
 * the two in lockstep (test_fused_probe_agreement_unit). */
static bool cpu_neon_fused_supported(struct geist_backend *be, const struct geist_fusion_query *q) {
    (void) be;
    if (q == nullptr) {
        return false;
    }
    switch (q->op) {
    case GEIST_FUSED_GELU_TANH_MUL:
    case GEIST_FUSED_GELU_TANH_MUL_SCALED:
        return true; /* F32 elementwise, any geometry, any m */
    case GEIST_FUSED_FFN_GEGLU_Q4Q6_MN:
        return q->m >= 1 && q->m <= GEIST_QUANT_M_CAP && q->d_model > 0 && q->inter > 0 &&
               q->d_model % Q4_K_BLOCK_ELEMS == 0 && q->inter % Q6_K_BLOCK_ELEMS == 0 &&
               q->gate_w != nullptr && q->up_w != nullptr && q->down_w != nullptr &&
               q->gate_w->dtype == GEIST_DTYPE_Q4_K && q->up_w->dtype == GEIST_DTYPE_Q4_K &&
               q->down_w->dtype == GEIST_DTYPE_Q6_K && (size_t) q->gate_w->n_in == q->d_model &&
               (size_t) q->up_w->n_in == q->d_model && (size_t) q->down_w->n_in == q->inter &&
               (size_t) q->gate_w->n_out == q->inter && (size_t) q->up_w->n_out == q->inter &&
               (size_t) q->down_w->n_out == q->d_model;
    default:
        return false;
    }
}

static const struct geist_backend_primitives cpu_neon_prims = {
        .rmsnorm          = cpu_neon_rmsnorm,
        .add              = cpu_neon_add,
        .mul              = cpu_neon_mul,
        .gelu_tanh        = cpu_neon_gelu_tanh,
        .silu             = cpu_neon_silu,
        .relu_squared     = cpu_neon_relu_squared,
        .rope_apply       = cpu_neon_rope_apply,
        .embedding_lookup = cpu_neon_embedding_lookup,
        .attention        = cpu_neon_attention,
};

static const struct geist_backend_fused cpu_neon_fused = {
        .supported            = cpu_neon_fused_supported,
        .gelu_tanh_mul        = cpu_neon_gelu_tanh_mul,
        .gelu_tanh_mul_scaled = cpu_neon_gelu_tanh_mul_scaled,
        .ffn_geglu_q4q6_mN    = cpu_neon_ffn_geglu_q4q6_mN,
};

const struct geist_backend_descriptor geist_backend_cpu_neon = {
        .name  = "cpu_neon",
        .vtbl  = &cpu_neon_vtbl,
        .prims = &cpu_neon_prims,
        .fused = &cpu_neon_fused,
        .caps =
                {
                        .manages_host_threads = true,
                        .max_m                = GEIST_QUANT_M_CAP,
#if defined(__APPLE__)
                        /* Unified memory + Accelerate: FP32 KV wins. */
                        .preferred_kv_mode = GEIST_KV_FP32,
#else
                        /* Pi-class LPDDR: INT8 KV halves the cache traffic. */
                        .preferred_kv_mode = GEIST_KV_INT8,
#endif
                },
};
