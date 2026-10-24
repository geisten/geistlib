/*
 * src/archs/transformer/forward/moe.c - the mixture-of-experts FFN block.
 *
 * Layer: ARCHITECTURE (transformer forward). Contract: blocks.h.
 *
 * Per token a router scores every expert; the moe_n_used best each run a
 * SwiGLU FFN and their outputs are summed, weighted by the router's
 * probabilities (#48):
 *
 *   p      = softmax(router(x))                 over all n_experts
 *   S      = the n_used largest p (ties: lower expert id first)
 *   w_e    = p_e, or p_e / sum_{S} p when moe_norm_topk
 *   h_out  = h_post_attn + sum_{e in S} w_e * down_e(silu(gate_e(x)) * up_e(x)),
 *   x      = rmsnorm(h_post_attn)
 *
 * A shared expert (moe_n_shared = 1, qwen35moe) is one more selection per
 * token, slot moe_n_experts, weighted sigmoid(x . moe_shexp_gate).
 *
 * The routing runs on the host. The rows routed to one expert are gathered
 * into one [m, d_model] block, so the expert's projections run as one
 * m-row linear (the prefill kernels) instead of m single-row ones.
 *
 * Which experts are resident is a loader matter: the arithmetic here is
 * the same for every placement.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "blocks.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>

#include "../moe_gpu.h"
#include "../moe_adapt.h"
#include "../moe_profile.h"
#include "heap.h"
#include "par.h"
#include "profile.h"

/* Finite by the bits: the build's -ffast-math lets the compiler assume no
 * NaN or Inf, so isfinite() may fold to true on exactly the input this
 * check exists for. */
static bool finite_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    return (u & 0x7f800000u) != 0x7f800000u;
}

bool moe_route_topk(
        size_t n, size_t k, bool norm, const float logits[static n], uint32_t *ids, float *w) {
    if (k == 0 || k > n) {
        return false;
    }
    float mx = logits[0];
    for (size_t i = 0; i < n; i++) {
        if (!finite_bits(logits[i])) {
            return false;
        }
        if (logits[i] > mx) {
            mx = logits[i];
        }
    }
    /* Softmax is monotonic, so the k largest probabilities are the k
     * largest logits. Ranked by (logit descending, id ascending): the next
     * pick is the best candidate strictly after the previous one in that
     * order, which keeps ties deterministic without a taken-set. */
    float    last    = INFINITY;
    uint32_t last_id = 0;
    for (size_t j = 0; j < k; j++) {
        size_t best = n;
        for (size_t i = 0; i < n; i++) {
            const bool after = logits[i] < last || (logits[i] == last && i > last_id);
            if (j > 0 && !after) {
                continue;
            }
            if (best == n || logits[i] > logits[best]) {
                best = i;
            }
        }
        ids[j]  = (uint32_t) best;
        last    = logits[best];
        last_id = (uint32_t) best;
    }
    double denom = 0.0;
    for (size_t i = 0; i < n; i++) {
        denom += exp((double) (logits[i] - mx));
    }
    double sel = 0.0;
    for (size_t j = 0; j < k; j++) {
        w[j] = (float) (exp((double) (logits[ids[j]] - mx)) / denom);
        sel += w[j];
    }
    if (norm) {
        for (size_t j = 0; j < k; j++) {
            w[j] = (float) (w[j] / sel);
        }
    }
    return true;
}

/* Bucket each row's selections by expert (counting sort, rows in order):
 * expert e's selections are rows[start[e] .. start[e + 1]), each a flat
 * index row * k + slot into ids / wts. */
static void moe_bucket(size_t         n_exp,
                       size_t         n_sel,
                       const uint32_t ids[static n_sel],
                       uint32_t       rows[static n_sel],
                       size_t         start[static n_exp + 1]) {
    memset(start, 0, (n_exp + 1) * sizeof start[0]);
    for (size_t i = 0; i < n_sel; i++) {
        start[ids[i] + 1]++;
    }
    for (size_t e = 0; e < n_exp; e++) {
        start[e + 1] += start[e];
    }
    /* start[e] doubles as the fill cursor; shift it back afterwards. */
    for (size_t i = 0; i < n_sel; i++) {
        rows[start[ids[i]]++] = (uint32_t) i;
    }
    for (size_t e = n_exp; e > 0; e--) {
        start[e] = start[e - 1];
    }
    start[0] = 0;
}

/* GEIST_MOE_TRACE=<path>: append one record per routed row — int32 layer,
 * int32 position, the k chosen expert ids (uint32, ascending) and the
 * router-logit margin between the k-th choice and the best one left out
 * (float). A diagnostic for comparing routing across backends and engines
 * (#48). The variable is read once per process; without it a layer pays
 * one branch. */
static char           g_moe_trace_path[4096];
static pthread_once_t g_moe_trace_once = PTHREAD_ONCE_INIT;

static void moe_trace_init(void) {
    const char *path = getenv("GEIST_MOE_TRACE");
    /* A path that does not fit is not opened rather than truncated. */
    if (path != nullptr && strlen(path) < sizeof g_moe_trace_path)
        memcpy(g_moe_trace_path, path, strlen(path) + 1);
}

static void moe_trace(int             layer,
                      size_t          pos0,
                      size_t          rows,
                      size_t          n_exp,
                      size_t          k,
                      size_t          stride,
                      const float    *lg,
                      const uint32_t *ids) {
    (void) pthread_once(&g_moe_trace_once, moe_trace_init);
    if (g_moe_trace_path[0] == '\0')
        return;
    FILE *f = fopen(g_moe_trace_path, "ab");
    if (f == nullptr)
        return;
    for (size_t r = 0; r < rows; r++) {
        const float    *l  = lg + r * n_exp;
        const uint32_t *id = ids + r * stride;
        uint32_t        sorted[GEIST_MOE_MAX_EXPERTS];
        float           kth = l[id[k - 1]], out = -INFINITY;
        memcpy(sorted, id, k * sizeof sorted[0]);
        for (size_t a = 1; a < k; a++)
            for (size_t b = a; b > 0 && sorted[b - 1] > sorted[b]; b--) {
                const uint32_t t = sorted[b];
                sorted[b]        = sorted[b - 1];
                sorted[b - 1]    = t;
            }
        for (size_t i = 0, j = 0; i < n_exp; i++) {
            if (j < k && sorted[j] == i)
                j++;
            else if (l[i] > out)
                out = l[i];
        }
        const int32_t hdr[2] = {layer, (int32_t) (pos0 + r)};
        const float   margin = kth - out;
        fwrite(hdr, sizeof hdr, 1, f);
        fwrite(sorted, sizeof sorted[0], k, f);
        fwrite(&margin, sizeof margin, 1, f);
    }
    fclose(f);
}

/* The experts of one layer, shared by the workers of one parallel region. */
struct moe_job {
    const struct transformer_layer_weights *L;
    struct transformer_arch_session        *sess;
    struct geist_backend                   *be;
    size_t                                  d, inter, n_exp;
    uint32_t                                active[GEIST_MOE_MAX_EXPERTS]; /* on the CPU */
    size_t                                  n_active;
    uint32_t                                hot[GEIST_MOE_MAX_EXPERTS]; /* on the GPU (#78) */
    size_t                                  n_hot;
    size_t                                  layer;
    pthread_t                               caller; /* the only thread that runs the GPU's batch */
    _Atomic bool                            gpu_taken;
    _Atomic size_t                          next;
    _Atomic bool                            failed;
    enum geist_status                       gpu_status;
};

static void moe_linear(size_t                     m,
                       const float               *x,
                       const struct geist_weight *w,
                       struct geist_backend      *be,
                       float                     *y,
                       bool                      *ok) {
    if (m == 1 && w->linear_m1 != nullptr)
        w->linear_m1(x, w, be, y);
    else if (m > 1 && w->linear_mN != nullptr)
        w->linear_mN(m, x, w, be, y);
    else
        *ok = false;
}

/* Expert e over its m gathered rows, all on the calling thread (the
 * kernels' own geist_par_for runs inline inside a parallel region):
 * y = down(silu(gate(x)) * up(x)). Rows start at `row0` of the bucketed
 * host buffers. */
static bool moe_expert(const struct moe_job *j, size_t e) {
    struct transformer_arch_session *sess = j->sess;
    const size_t                     row0 = sess->moe_row_start[e];
    const size_t                     m    = sess->moe_row_start[e + 1] - row0;
    const float                     *x    = sess->moe_xg + row0 * j->d;
    float                           *g    = sess->moe_g + row0 * j->inter;
    float                           *u    = sess->moe_u + row0 * j->inter;
    float                           *y    = sess->moe_y + row0 * j->d;
    const struct geist_weight       *wg = &j->L->moe_gate_w[e], *wu = &j->L->moe_up_w[e];
    bool                             ok = true;

    /* gate and up read the same x: one pair kernel where the backend has it. */
    const bool pair =
            wg->n_in == wu->n_in &&
            (m == 1 ? wg->linear_pair_m1 != nullptr && wg->linear_pair_m1 == wu->linear_pair_m1
                    : wg->linear_pair_mN != nullptr && wg->linear_pair_mN == wu->linear_pair_mN);
    if (pair && m == 1)
        wg->linear_pair_m1(x, wg, wu, j->be, g, u);
    else if (pair)
        wg->linear_pair_mN(m, x, wg, wu, j->be, g, u);
    else {
        moe_linear(m, x, wg, j->be, g, &ok);
        moe_linear(m, x, wu, j->be, u, &ok);
    }
    if (!ok)
        return false;
    for (size_t i = 0; i < m * j->inter; i++)
        g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
    moe_linear(m, g, &j->L->moe_down_w[e], j->be, y, &ok);
    return ok;
}

/* The GPU's batch, once, on the thread that called blk_moe_ffn: a backend
 * may tie its open command sequence to that thread (Metal's seq_lock). */
static void moe_gpu_task(struct moe_job *j) {
    if (j->n_hot == 0 || atomic_exchange_explicit(&j->gpu_taken, true, memory_order_relaxed))
        return;
    j->gpu_status = moe_gpu_run(j->sess, j->layer, j->n_hot, j->hot);
    if (j->gpu_status != GEIST_OK)
        atomic_store_explicit(&j->failed, true, memory_order_relaxed);
}

/* A worker: takes the next CPU expert until none is left, so experts with
 * many rows and experts with one even out across the threads. The calling
 * thread first runs the GPU's batch and waits for it while the others run
 * the CPU's experts. */
static void moe_worker(void *ctx, size_t begin, size_t end) {
    struct moe_job *j = ctx;
    (void) begin;
    (void) end;
    if (pthread_equal(pthread_self(), j->caller))
        moe_gpu_task(j);
    for (size_t a;
         (a = atomic_fetch_add_explicit(&j->next, 1, memory_order_relaxed)) < j->n_active;) {
        if (!moe_expert(j, j->active[a]))
            atomic_store_explicit(&j->failed, true, memory_order_relaxed);
    }
}

enum moe_profile_stage {
    MOE_PROFILE_NORM = 0,
    MOE_PROFILE_ROUTER,
    MOE_PROFILE_ROUTE,
    MOE_PROFILE_GATHER,
    MOE_PROFILE_EXPERTS,
    MOE_PROFILE_SUM,
    MOE_PROFILE_COUNT,
};

static _Atomic uint64_t  g_moe_profile_ns[MOE_PROFILE_COUNT];
static _Atomic uint64_t  g_moe_profile_calls[MOE_PROFILE_COUNT];
static const char *const g_moe_profile_names[MOE_PROFILE_COUNT] = {
        "norm",
        "router",
        "route",
        "gather",
        "experts",
        "sum",
};
static struct transformer_forward_profile g_moe_profile = {
        .title       = "transformer moe ffn",
        .stage_names = g_moe_profile_names,
        .stage_count = MOE_PROFILE_COUNT,
        .ns          = g_moe_profile_ns,
        .calls       = g_moe_profile_calls,
};

/* Close stage `stage` and start the next one at now (GEIST_PROFILE_FORWARD). */
static inline void moe_profile_mark(size_t stage, uint64_t *t0) {
    if (*t0 != 0) {
        transformer_profile_add(&g_moe_profile, stage, *t0);
        *t0 = transformer_profile_now_ns();
    }
}

enum geist_status blk_moe_ffn(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_state    *st    = ctx->st;
    struct transformer_arch_session  *sess  = ctx->sess;
    struct transformer_layer_weights *L     = ctx->L;
    const struct geist_backend_vtbl  *v     = ctx->v;
    const size_t                      d     = st->d_model;
    const size_t                      n_exp = st->config.moe_n_experts;
    const size_t                      n_rt  = st->config.moe_n_used;
    const size_t                      k     = moe_k(&st->config); /* selections per token */
    const size_t                      slots = moe_slots(&st->config);
    const size_t                      seq   = ctx->seq;
    const size_t                      n_sel = seq * k;
    struct geist_buffer              *x     = sess->scratch_pre_ff;

    /* #166: the adaptive cache's step (a planned swap; probe or not). */
    struct moe_adapt *adapt = seq == 1 && st->moe_gpu != nullptr ? st->moe_gpu->adapt : nullptr;
    if (adapt != nullptr && ctx->layer_idx == 0) {
        const enum geist_status as = moe_adapt_step(st, &sess->moe_adapt_probe);
        if (as != GEIST_OK)
            fprintf(stderr,
                    "geist: moe: adaptive swap failed: %s\n",
                    geist_backend_errmsg(ctx->be));
    }
    uint64_t t0 = transformer_profile_enabled(&g_moe_profile) ? transformer_profile_now_ns() : 0;
    enum geist_status s = blk_rmsnorm(ctx, d, sess->scratch_h_post_attn, &L->ffn_norm, x);
    moe_profile_mark(MOE_PROFILE_NORM, &t0);
    if (s == GEIST_OK) {
        struct geist_tensor t_x = view_2d(x, ctx->SEQ, (int64_t) d);
        struct geist_tensor t_l = view_2d(sess->moe_logits, ctx->SEQ, (int64_t) n_exp);
        s                       = linear_w_or_legacy(
                ctx->be, v, x, sess->moe_logits, &L->moe_router_w, seq, &t_x, &L->moe_router, &t_l);
    }
    if (s != GEIST_OK) {
        return s;
    }
    /* #201: a decode step routes and runs its experts on the device. */
    if (moe_gpu_device(sess, (size_t) ctx->layer_idx, seq)) {
        s = moe_gpu_device_ffn(
                sess, (size_t) ctx->layer_idx, seq, x, sess->scratch_h_post_attn, ctx->h_out_buf);
        moe_profile_mark(MOE_PROFILE_EXPERTS, &t0);
        return s;
    }
    /* On a GPU x rides along with the router's batch: one wait for both. */
    if (sess->moe_x_io != nullptr) {
        s = v->buffer_copy(sess->moe_x_io, 0, x, 0, seq * d * sizeof(float));
        if (s != GEIST_OK)
            return s;
    }
    moe_profile_mark(MOE_PROFILE_ROUTER, &t0);

    /* Route on the host. */
    const float *lg = (const float *) v->buffer_map(sess->moe_logits);
    if (lg == nullptr) {
        return GEIST_E_BACKEND;
    }
    bool routed = true;
    for (size_t r = 0; r < seq && routed; r++) {
        routed = moe_route_topk(n_exp,
                                n_rt,
                                st->config.moe_norm_topk,
                                lg + r * n_exp,
                                sess->moe_ids + r * k,
                                sess->moe_wts + r * k);
    }
    for (size_t r = 0; r < seq && k > n_rt; r++)
        sess->moe_ids[r * k + n_rt] = (uint32_t) n_exp; /* the shared expert */
    if (routed) {
        moe_trace(ctx->layer_idx, ctx->q_position, seq, n_exp, n_rt, k, lg, sess->moe_ids);
        moe_profile_count(st->moe_prof, (size_t) ctx->layer_idx, seq, k, n_rt, sess->moe_ids);
    }
    v->buffer_unmap(sess->moe_logits);
    if (!routed) {
        geist_backend_set_error(ctx->be,
                                GEIST_E_BACKEND,
                                "moe: layer %d router logits are not finite",
                                ctx->layer_idx);
        return GEIST_E_BACKEND;
    }
    moe_bucket(slots, n_sel, sess->moe_ids, sess->moe_rows, sess->moe_row_start);
    moe_profile_mark(MOE_PROFILE_ROUTE, &t0);

    /* Gather each expert's rows into its slice of the bucketed buffer. A
     * GPU's device-local x is copied down first. */
    struct geist_buffer *xb = sess->moe_x_io != nullptr ? sess->moe_x_io : x;
    const float         *xs = (const float *) v->buffer_map(xb);
    if (xs == nullptr) {
        if (v->buffer_download(seq * d * sizeof(float), (uint8_t *) sess->moe_x_host, x) !=
            GEIST_OK)
            return GEIST_E_BACKEND;
        xs = sess->moe_x_host;
    }
    for (size_t i = 0; i < n_sel; i++) {
        memcpy(sess->moe_xg + i * d, xs + (sess->moe_rows[i] / k) * d, d * sizeof(float));
    }
    /* The shared expert's weight: sigmoid of x against its gate vector. */
    for (size_t r = 0; r < seq && k > n_rt; r++) {
        double dot = 0.0;
        for (size_t c = 0; c < d; c++)
            dot += (double) xs[r * d + c] * (double) L->moe_shexp_gate[c];
        sess->moe_wts[r * k + n_rt] = (float) (1.0 / (1.0 + exp(-dot)));
    }
    if (xs != sess->moe_x_host)
        v->buffer_unmap(xb);
    moe_profile_mark(MOE_PROFILE_GATHER, &t0);

    /* Every active expert in one parallel region. */
    struct moe_job  job = {0};
    struct moe_job *j   = &job;
    j->L                = L;
    j->sess             = sess;
    j->be               = st->moe_cpu != nullptr ? st->moe_cpu : ctx->be; /* the CPU experts' */
    j->d                = d;
    j->inter            = ctx->inter;
    j->n_exp            = slots;
    j->n_active         = 0;
    j->n_hot            = 0;
    j->layer            = (size_t) ctx->layer_idx;
    j->gpu_status       = GEIST_OK;
    size_t rows_gpu     = 0;
    for (size_t e = 0; e < slots; e++) {
        const size_t m = sess->moe_row_start[e + 1] - sess->moe_row_start[e];
        if (m == 0)
            continue;
        /* ponytail: with the model on the CPU, hot experts at decode only.
         * A prefill step's experts have many rows each; run one by one on
         * the GPU they took longer than the CPU for all of them (run 1669:
         * 139 vs 185 tok/s). A grouped GEMM would bring prefill in. With
         * the model on the GPU (moe_cpu set) the GPU is busy anyway. */
        if ((seq == 1 || st->moe_cpu != nullptr) && moe_gpu_hot(sess, j->layer, e)) {
            j->hot[j->n_hot++] = (uint32_t) e;
            rows_gpu += m;
        } else
            j->active[j->n_active++] = (uint32_t) e;
    }
    if (st->moe_gpu != nullptr)
        atomic_fetch_add_explicit(&st->moe_gpu->rows_cpu, n_sel - rows_gpu, memory_order_relaxed);
    /* One wait per layer (#161): a decode step with the model on the GPU
     * starts its hot experts now and does not wait for them; the CPU runs
     * the cold ones meanwhile and the GPU adds the hot outputs onto their
     * sum. GEIST_E_UNSUPPORTED (a weight not resident): the batch below. */
    bool launched = false;
    if (j->n_hot > 0 && moe_gpu_launchable(sess, j->layer, j->n_hot, j->hot)) {
        s = moe_gpu_launch(sess, j->layer, seq, j->n_hot, j->hot);
        if (s != GEIST_OK && s != GEIST_E_UNSUPPORTED) {
            geist_backend_set_error(ctx->be,
                                    s,
                                    "moe: layer %d: the GPU experts failed: %s",
                                    ctx->layer_idx,
                                    geist_backend_errmsg(st->moe_gpu->be));
            return s;
        }
        launched = s == GEIST_OK;
        s        = GEIST_OK;
    }
    /* #166: on a probe step the host waits for the hot batch now: the GPU
     * branch's time. */
    uint64_t gpu_ns = 0, cpu_t0 = 0;
    if (adapt != nullptr && launched) {
        if (sess->moe_adapt_probe && sess->moe_gpu_y_io != nullptr) {
            const struct geist_backend_vtbl *gv = st->moe_gpu->be->desc->vtbl;
            const uint64_t                   g0 = transformer_profile_now_ns();
            if (gv->buffer_map(sess->moe_gpu_y_io) != nullptr)
                gv->buffer_unmap(sess->moe_gpu_y_io);
            gpu_ns = transformer_profile_now_ns() - g0;
        }
        cpu_t0 = transformer_profile_now_ns();
    }
    j->caller = pthread_self();
    atomic_store_explicit(&j->gpu_taken, launched, memory_order_relaxed);
    atomic_store_explicit(&j->next, 0, memory_order_relaxed);
    atomic_store_explicit(&j->failed, false, memory_order_relaxed);
    const size_t tasks   = (j->n_hot > 0 && !launched) + j->n_active;
    const size_t threads = geist_par_max_threads();
    geist_par_for(tasks < threads ? tasks : threads, moe_worker, j);
    moe_gpu_task(j); /* a thread library that gave the caller no range */
    if (adapt != nullptr && launched)
        moe_adapt_record(adapt,
                         j->layer,
                         n_rt,
                         sess->moe_ids,
                         n_sel - rows_gpu,
                         transformer_profile_now_ns() - cpu_t0,
                         gpu_ns);
    moe_profile_mark(MOE_PROFILE_EXPERTS, &t0);
    if (j->gpu_status != GEIST_OK) {
        geist_backend_set_error(ctx->be,
                                j->gpu_status,
                                "moe: layer %d: the GPU experts failed: %s",
                                ctx->layer_idx,
                                geist_backend_errmsg(st->moe_gpu->be));
        return j->gpu_status;
    }
    if (atomic_load_explicit(&j->failed, memory_order_relaxed)) {
        geist_backend_set_error(ctx->be,
                                GEIST_E_UNSUPPORTED,
                                "moe: layer %d: an expert projection has no kernel",
                                ctx->layer_idx);
        return GEIST_E_UNSUPPORTED;
    }

    if (launched) {
        /* The cold experts' weighted sum per token on the host, in expert
         * order; the GPU adds the hot outputs, weighted, onto it without
         * the host waiting for them. */
        float *cold                          = sess->moe_x_host;
        bool   is_hot[GEIST_MOE_MAX_EXPERTS] = {false};
        for (size_t h = 0; h < j->n_hot; h++)
            is_hot[j->hot[h]] = true;
        memset(cold, 0, seq * d * sizeof(float));
        for (size_t i = 0; i < n_sel; i++) {
            const uint32_t sel = sess->moe_rows[i];
            if (is_hot[sess->moe_ids[sel]])
                continue;
            const float  wt  = sess->moe_wts[sel];
            float       *dst = cold + (sel / k) * d;
            const float *src = sess->moe_y + i * d;
            for (size_t c = 0; c < d; c++)
                dst[c] += wt * src[c];
        }
        s = moe_gpu_finish(sess, seq, j->n_hot, j->hot, cold, sess->moe_acc);
        if (s == GEIST_OK)
            s = blk_add(ctx, d, sess->scratch_h_post_attn, sess->moe_acc, ctx->h_out_buf);
        moe_profile_mark(MOE_PROFILE_SUM, &t0);
        return s;
    }

    /* Weighted sum, in expert order so the result does not depend on which
     * thread ran which expert. */
    float *acc = (float *) v->buffer_map(sess->moe_acc);
    if (acc == nullptr) {
        return GEIST_E_BACKEND;
    }
    memset(acc, 0, seq * d * sizeof(float));
    for (size_t i = 0; i < n_sel; i++) {
        const uint32_t sel = sess->moe_rows[i];
        const float    wt  = sess->moe_wts[sel];
        float         *dst = acc + (sel / k) * d;
        const float   *src = sess->moe_y + i * d;
        for (size_t c = 0; c < d; c++) {
            dst[c] += wt * src[c];
        }
    }
    v->buffer_unmap(sess->moe_acc);
    s = blk_add(ctx, d, sess->scratch_h_post_attn, sess->moe_acc, ctx->h_out_buf);
    moe_profile_mark(MOE_PROFILE_SUM, &t0);
    return s;
}

/* ---- Session state ---------------------------------------------------- */

/* Router logits, the gathered rows of one expert and the weighted sum, plus
 * the host-side routing (blk_moe_ffn). */
enum geist_status moe_session_alloc(struct transformer_arch_session *sess) {
    const struct transformer_arch_state *st    = sess->model;
    struct geist_backend                *be    = st->backend;
    const size_t                         n_exp = st->config.moe_n_experts;
    const size_t                         n_sel = sess->m_max * moe_k(&st->config);
    /* The host reads the logits and writes the sum: on a GPU they live in
     * cached host memory the device writes and reads in place. */
    const bool        io = be->desc->caps.batched_submit;
    enum geist_status s  = io ? be->desc->vtbl->buffer_create(be,
                                                              sess->m_max * n_exp * sizeof(float),
                                                              GEIST_BUFFER_IO,
                                                              GEIST_MEMORY_AUTO,
                                                              &sess->moe_logits)
                              : transformer_scratch_alloc(
                                        be, sess->m_max * n_exp * sizeof(float), &sess->moe_logits);
    if (s == GEIST_OK)
        s = io ? be->desc->vtbl->buffer_create(be,
                                               sess->m_max * st->d_model * sizeof(float),
                                               GEIST_BUFFER_IO,
                                               GEIST_MEMORY_AUTO,
                                               &sess->moe_acc)
               : transformer_scratch_alloc(
                         be, sess->m_max * st->d_model * sizeof(float), &sess->moe_acc);
    if (s == GEIST_OK && io)
        s = be->desc->vtbl->buffer_create(be,
                                          sess->m_max * st->d_model * sizeof(float),
                                          GEIST_BUFFER_IO,
                                          GEIST_MEMORY_AUTO,
                                          &sess->moe_x_io);
    if (s != GEIST_OK)
        return s;
    size_t inter = 0;
    for (size_t li = 0; li < st->n_layers; li++)
        inter = st->layers[li].intermediate > inter ? st->layers[li].intermediate : inter;
    sess->moe_ids       = heap_alloc_array_aligned(uint32_t, n_sel);
    sess->moe_wts       = heap_alloc_array_aligned(float, n_sel);
    sess->moe_rows      = heap_alloc_array_aligned(uint32_t, n_sel);
    sess->moe_row_start = heap_alloc_array_aligned(size_t, moe_slots(&st->config) + 1);
    sess->moe_xg        = heap_alloc_array_aligned(float, n_sel * st->d_model);
    sess->moe_y         = heap_alloc_array_aligned(float, n_sel * st->d_model);
    sess->moe_g         = heap_alloc_array_aligned(float, n_sel *inter);
    sess->moe_u         = heap_alloc_array_aligned(float, n_sel *inter);
    sess->moe_x_host    = heap_alloc_array_aligned(float, sess->m_max * st->d_model);
    if (sess->moe_ids == nullptr || sess->moe_wts == nullptr || sess->moe_rows == nullptr ||
        sess->moe_row_start == nullptr || sess->moe_xg == nullptr || sess->moe_y == nullptr ||
        sess->moe_g == nullptr || sess->moe_u == nullptr || sess->moe_x_host == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "transformer: MoE routing state alloc failed");
        return GEIST_E_OOM;
    }
    return moe_gpu_session_alloc(sess);
}

void moe_session_free(struct transformer_arch_session *sess) {
    moe_gpu_session_free(sess);
    void *host[] = {sess->moe_x_host,
                    sess->moe_xg,
                    sess->moe_y,
                    sess->moe_g,
                    sess->moe_u,
                    sess->moe_ids,
                    sess->moe_wts,
                    sess->moe_rows,
                    sess->moe_row_start};
    for (size_t i = 0; i < sizeof host / sizeof host[0]; i++)
        safe_free(&host[i]);
    sess->moe_xg = sess->moe_y = sess->moe_g = sess->moe_u = sess->moe_x_host = nullptr;
    sess->moe_wts                                                             = nullptr;
    sess->moe_ids = sess->moe_rows = nullptr;
    sess->moe_row_start            = nullptr;
    transformer_buffer_release(sess->model->backend, &sess->moe_logits);
    transformer_buffer_release(sess->model->backend, &sess->moe_acc);
    transformer_buffer_release(sess->model->backend, &sess->moe_x_io);
}
