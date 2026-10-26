/*
 * src/backends/cpu_neon/backend.c — ARM NEON-optimized backend.
 *
 * Layer: BACKEND.
 *
 * Descriptor, lifecycle and the OpenMP region hooks; buffers are the
 * shared host buffers (common/host_buffer.c). The linear
 * kernels are bound per weight at load time (weight_resolve.c).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"

#include <geist.h>
#include <geist_backend.h>

#include "hadamard.h"
#include "heap.h"
#include "hw_probe.h"
#include "omp_idle.h"
#include "par.h"
#include "env.h"
#include "quant.h"

#include <limits.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    /* Scratch lives on `st`, not in TLS, so no OMP barrier is needed. */
    cpu_neon_ws_destroy_all(st);
    cpu_neon_dump_kernel_hits();
    geist_backend_free(be, be->state);
    be->state = nullptr;
}

/* Parallel-region hooks. The kernels run on geist_par_for (par.h), and the
 * thread count that suits one phase hurts another: the arch layer calls
 * these around each phase, which set geist_par_set_max_threads for it (the
 * OpenMP team with OpenMP, the pool's count without) and restore the
 * previous count afterwards. */
#if defined(_OPENMP)
#include <omp.h>
#else
#include <unistd.h>
#endif

/* One-time thread setup at backend create, through the environment only:
 * libomp reads it at the first OpenMP API call, and a model's blocktime
 * (prefill_tuning.c) comes later, at model load. An explicit setting in
 * the environment wins.
 * - OMP_WAIT_POLICY=active: decode runs ~200 tiny regions per token, and
 *   a passive worker's wake-up dominates (qwen3.5-4B: 163 vs 46 ms/tok).
 * - KMP_BLOCKTIME: bounds that spin once the work stops (omp_idle.h).
 * - OMP_NUM_THREADS: the pool is sized to the P-cores, so the per-phase
 *   caps never land on efficiency cores (qwen3.5-4B: 114 vs 47 ms/tok); a
 *   static partition over E-cores waits on their chunks (M1 Max pp512: 91
 *   tps on 10 cores vs 145 on 8 P-cores).
 *   geist_par_for's own pool (no OpenMP) reads it too. */
static void cpu_neon_omp_pool_init(void) {
    static _Atomic int done = 0;
    int                exp  = 0;
    if (!atomic_compare_exchange_strong(&done, &exp, 1))
        return;
    setenv("OMP_WAIT_POLICY", "active", 0);
    geist_omp_idle_default();
    const int pc = geist_hw_perf_cores();
    if (pc > 0) {
        char n[16];
        snprintf(n, sizeof n, "%d", pc);
        setenv("OMP_NUM_THREADS", n, 0);
    }
}

/* The processors this process may use. */
static int online_cpus(void) {
#if defined(_OPENMP)
    return omp_get_num_procs();
#else
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 && n <= INT_MAX ? (int) n : 1;
#endif
}

/* Target thread count for `region`, cached on first use. 0 = "leave the
 * ambient OMP_NUM_THREADS alone". Env overrides force a count (>0) or disable
 * the adjustment (0): GEIST_PREFILL_THREADS, GEIST_DECODE_THREADS. */
static int cpu_neon_region_thread_count(enum geist_parallel_region region) {
    if (region == GEIST_REGION_PREFILL_BATCH) {
        /* Prefill is compute-bound and scales with cores: all of them, or
         * the P-cores on Apple. On Pi 5 all 4 cores help, but only on a
         * quiesced box; a background process eating a core inverts it. */
        static _Atomic int n = -1;
        if (n < 0) {
            const int env = geist_env_count("GEIST_PREFILL_THREADS");
            if (env >= 0) {
                n = env;
            } else {
                const int pc = geist_hw_perf_cores();
                n            = (pc > 0) ? pc : online_cpus();
            }
        }
        return n;
    }
    /* GEIST_REGION_DECODE_STEP: memory-bound GEMVs. Pi 5 (shared LPDDR):
     * 3 threads beat 4. Apple: P-cores minus one. Other targets keep the
     * ambient count. */
    static _Atomic int n = -1;
    if (n < 0) {
        const int env = geist_env_count("GEIST_DECODE_THREADS");
        if (env >= 0) {
            n = env;
        } else {
#if defined(GEIST_TARGET_PI5)
            n = 3;
#else
            /* One P-core left for the OMP master and the OS once the
             * P-cores saturate memory: ~210 tiny matmuls per token contend
             * with coordination work otherwise (M1 Max tg128: 8 P-cores
             * ~25 tps noisy, 7 ~30 tps stable). Below that the GEMV is
             * compute-bound per core (~17 GB/s each, APPLE.md) and every
             * core counts: a 3-core M1 VM decodes 11.0 tps on 2 threads,
             * 16.6 on 3 (#456). The spare starts at 6 P-cores: 8 on the
             * measured M1 Max, 4 on the base chips. */
            constexpr int spare_from = 6;
            const int     pc         = geist_hw_perf_cores();
            n                        = (pc >= spare_from) ? pc - 1 : pc;
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
    const size_t max  = geist_par_max_threads();
    const int    prev = max > (size_t) INT_MAX ? INT_MAX : (int) max;
    /* Prefill bumps in either direction; decode only caps DOWN — never adds
     * threads to a memory-bound GEMV. */
    const bool apply = (region == GEIST_REGION_DECODE_STEP) ? (target < prev) : (target != prev);
    if (!apply)
        return 0;
    geist_par_set_max_threads((size_t) target);
    return prev; /* >0: restore to this in _end */
}

static void cpu_neon_parallel_region_end(struct geist_backend *be, int token) {
    (void) be;
    if (token > 0)
        geist_par_set_max_threads((size_t) token);
}

static const struct geist_backend_vtbl cpu_neon_vtbl = {
        .create                = cpu_neon_create,
        .destroy               = cpu_neon_destroy,
        .buffer_create         = geist_host_buffer_create,
        .buffer_destroy        = geist_host_buffer_destroy,
        .buffer_create_aliased = geist_host_buffer_create_aliased,
        .buffer_upload         = geist_host_buffer_upload,
        .buffer_download       = geist_host_buffer_download,
        .buffer_map            = geist_host_buffer_map,
        .buffer_unmap          = geist_host_buffer_unmap,
        .resolve_weight        = cpu_neon_resolve_weight,
        .parallel_region_begin = cpu_neon_parallel_region_begin,
        .parallel_region_end   = cpu_neon_parallel_region_end,
};

/* Probe pairing for the fused table below (test_fused_probe_agreement_unit). */
static bool cpu_neon_fused_supported(struct geist_backend *be, const struct geist_fusion_query *q) {
    (void) be;
    if (q == nullptr) {
        return false;
    }
    switch (q->op) {
    case GEIST_FUSED_GELU_TANH_MUL:
    case GEIST_FUSED_GELU_TANH_MUL_SCALED:
    case GEIST_FUSED_SILU_MUL:
        return true; /* F32 elementwise, any geometry, any m */
    case GEIST_FUSED_HADAMARD_ROTATE:
        return geist_hadamard_query_ok(q); /* the shared host transform */
    case GEIST_FUSED_RMSNORM_ADD:
    case GEIST_FUSED_RMSNORM_ADD_NORM:
        return cpu_neon_res_norm_supported(q->d_model); /* F32 rows, any m */
    case GEIST_FUSED_ATTN_QKV_PREP:
    case GEIST_FUSED_ATTN_ROPE_APPEND:
        return q->head_dim > 0 && q->head_dim % 2 == 0 && q->n_q_heads > 0; /* any m */
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    case GEIST_FUSED_ATTN_KV_INT8:
        return cpu_neon_attention_kv_int8_supported(q);
    case GEIST_FUSED_ATTN_KV_INT4:
        return cpu_neon_attention_kv_int4_supported(q);
#endif
    default:
        return false;
    }
}

static const struct geist_backend_primitives cpu_neon_prims = {
        .rmsnorm      = cpu_neon_rmsnorm,
        .add          = cpu_neon_add,
        .mul          = cpu_neon_mul,
        .gelu_tanh    = cpu_neon_gelu_tanh,
        .silu         = cpu_neon_silu,
        .relu_squared = cpu_neon_relu_squared,
        .rope_apply   = cpu_neon_rope_apply,
        .attention    = cpu_neon_attention,
};

static const struct geist_backend_fused cpu_neon_fused = {
        .supported            = cpu_neon_fused_supported,
        .gelu_tanh_mul        = cpu_neon_gelu_tanh_mul,
        .gelu_tanh_mul_scaled = cpu_neon_gelu_tanh_mul_scaled,
        .silu_mul             = cpu_neon_silu_mul,
        .hadamard_rotate      = cpu_neon_hadamard_rotate,
        .rmsnorm_add          = cpu_neon_rmsnorm_add,
        .rmsnorm_add_norm     = cpu_neon_rmsnorm_add_norm,
        .attn_qkv_prep        = cpu_neon_attn_qkv_prep,
        .attn_rope_append     = cpu_neon_attn_rope_append,
        .kv_append_f16        = cpu_neon_kv_append_f16,
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        .attention_kv_int8 = cpu_neon_attention_kv_int8,
        .attention_kv_int4 = cpu_neon_attention_kv_int4,
#endif
};

const struct geist_backend_descriptor geist_backend_cpu_neon = {
        .name     = "cpu_neon",
        .vtbl     = &cpu_neon_vtbl,
        .prims    = &cpu_neon_prims,
        .fused    = &cpu_neon_fused,
        .tunables = cpu_neon_tunables,
        .caps =
                {
                        .manages_host_threads = true,
                        /* bump when kernel perf character changes without
                         * a tunable rename (see #318) */
                        .calibration_generation = 1,
                        .max_m                  = GEIST_QUANT_M_CAP,
                        /* host DeltaNet sub-chunks (layer_deltanet.c) */
                        .dn_subchunk      = true,
                        .kv_f16_attention = true, /* on request (#48) */
#if defined(__APPLE__)
                        /* Unified memory + Accelerate: FP32 KV wins. */
                        .preferred_kv_mode = GEIST_KV_FP32,
#else
                        /* Pi-class LPDDR: INT8 KV halves the cache traffic. */
                        .preferred_kv_mode = GEIST_KV_INT8,
#endif
                },
};
