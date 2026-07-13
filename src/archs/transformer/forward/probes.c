/*
 * src/archs/transformer/forward/probes.c - optional forward diagnostics.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

/* GEIST_DUMP_ACT_SPARSITY=1: count exact-zero elements in the FFN
 * down_proj input per layer and print a table on exit. Disabled by
 * default and kept out of layer.c's hot-path orchestration.
 *
 * Lifetime model:
 *   - The enable flag is resolved once into transformer_runtime_flags
 *     at state creation. atexit registration is bound by pthread_once
 *     so two-thread first calls cannot register the flush twice.
 *   - Per-layer counters are 64-bit and updated under a small mutex so
 *     interleaved updates from concurrent forwards do not tear or lose
 *     increments. Cost is paid only when the probe is enabled.
 *   - Layer indices ≥ ACT_SPARSITY_MAX_LAYERS no longer fail silently:
 *     the first overflow logs an explicit warning to stderr so users
 *     running > 64-layer models know data is being dropped.
 */
#define ACT_SPARSITY_MAX_LAYERS 64

static pthread_once_t  g_act_sparsity_once            = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_act_sparsity_mutex           = PTHREAD_MUTEX_INITIALIZER;
static bool            g_act_sparsity_overflow_warned = false;
static uint64_t        g_act_sparsity_zeros[ACT_SPARSITY_MAX_LAYERS];
static uint64_t        g_act_sparsity_total[ACT_SPARSITY_MAX_LAYERS];
/* All-zero 64-element blocks — the unit a block-skipping ternary GEMV
 * could drop (one x4 weight cache line per block; #102 Phase 3 gate). */
static uint64_t g_act_sparsity_zblk[ACT_SPARSITY_MAX_LAYERS];
static uint64_t g_act_sparsity_tblk[ACT_SPARSITY_MAX_LAYERS];
static size_t   g_act_sparsity_n_layers;

static void act_sparsity_flush(void) {
    if (g_act_sparsity_n_layers == 0) {
        return;
    }
    uint64_t z_sum = 0, t_sum = 0, zb_sum = 0, tb_sum = 0;
    fprintf(stderr, "\n=== FFN down_proj input sparsity (GEIST_DUMP_ACT_SPARSITY) ===\n");
    fprintf(stderr, "  layer  zeros            total           sparsity  zero-64blk\n");
    for (size_t i = 0; i < g_act_sparsity_n_layers; i++) {
        uint64_t z  = g_act_sparsity_zeros[i];
        uint64_t t  = g_act_sparsity_total[i];
        uint64_t zb = g_act_sparsity_zblk[i];
        uint64_t tb = g_act_sparsity_tblk[i];
        if (t == 0) {
            continue;
        }
        z_sum += z;
        t_sum += t;
        zb_sum += zb;
        tb_sum += tb;
        fprintf(stderr,
                "  %5zu  %15llu  %15llu  %7.2f%%  %7.2f%%\n",
                i,
                (unsigned long long) z,
                (unsigned long long) t,
                100.0 * (double) z / (double) t,
                tb ? 100.0 * (double) zb / (double) tb : 0.0);
    }
    if (t_sum > 0) {
        fprintf(stderr, "  -----  ---------------  ---------------  -------   -------\n");
        fprintf(stderr,
                "  TOTAL  %15llu  %15llu  %7.2f%%  %7.2f%%\n",
                (unsigned long long) z_sum,
                (unsigned long long) t_sum,
                100.0 * (double) z_sum / (double) t_sum,
                tb_sum ? 100.0 * (double) zb_sum / (double) tb_sum : 0.0);
    }
}

static void act_sparsity_register_once(void) {
    atexit(act_sparsity_flush);
}

void transformer_probe_ffn_sparsity(const struct geist_backend_vtbl *v,
                                    bool                             enabled,
                                    int                              layer_idx,
                                    struct geist_buffer             *buf,
                                    size_t                           n_elems) {

    if (!enabled || v == nullptr || buf == nullptr) {
        return;
    }
    pthread_once(&g_act_sparsity_once, act_sparsity_register_once);
    if (layer_idx < 0) {
        return;
    }
    if (layer_idx >= ACT_SPARSITY_MAX_LAYERS) {
        /* Warn once if a model exceeds the static cap so users do not
         * mistake silent truncation for genuine late-layer sparsity. */
        pthread_mutex_lock(&g_act_sparsity_mutex);
        if (!g_act_sparsity_overflow_warned) {
            fprintf(stderr,
                    "geist: GEIST_DUMP_ACT_SPARSITY: layer %d >= "
                    "ACT_SPARSITY_MAX_LAYERS (%d); higher layers will "
                    "be dropped from the report.\n",
                    layer_idx,
                    ACT_SPARSITY_MAX_LAYERS);
            g_act_sparsity_overflow_warned = true;
        }
        pthread_mutex_unlock(&g_act_sparsity_mutex);
        return;
    }
    const float *data = (const float *) v->buffer_map(buf);
    if (data == nullptr) {
        return;
    }

    /* Count outside the mutex so the per-token cost is just the (cheap)
     * tight loop. The mutex only serializes the global accumulator
     * update. */
    uint64_t z = 0, zb = 0;
    for (size_t b0 = 0; b0 < n_elems; b0 += 64) {
        const size_t end   = (b0 + 64 <= n_elems) ? b0 + 64 : n_elems;
        uint64_t     zhere = 0;
        for (size_t i = b0; i < end; i++) {
            if (data[i] == 0.0f) {
                zhere++;
            }
        }
        z += zhere;
        if (zhere == end - b0) {
            zb++;
        }
    }
    const uint64_t tb = (n_elems + 63) / 64;
    v->buffer_unmap(buf);

    pthread_mutex_lock(&g_act_sparsity_mutex);
    if ((size_t) (layer_idx + 1) > g_act_sparsity_n_layers) {
        g_act_sparsity_n_layers = (size_t) (layer_idx + 1);
    }
    g_act_sparsity_zeros[layer_idx] += z;
    g_act_sparsity_total[layer_idx] += (uint64_t) n_elems;
    g_act_sparsity_zblk[layer_idx] += zb;
    g_act_sparsity_tblk[layer_idx] += tb;
    pthread_mutex_unlock(&g_act_sparsity_mutex);
}
