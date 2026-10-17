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
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>

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
 * (#48); without the variable it costs one getenv per layer. */
static void moe_trace(int             layer,
                      size_t          pos0,
                      size_t          rows,
                      size_t          n_exp,
                      size_t          k,
                      const float    *lg,
                      const uint32_t *ids) {
    const char *path = getenv("GEIST_MOE_TRACE");
    if (path == nullptr || path[0] == '\0')
        return;
    FILE *f = fopen(path, "ab");
    if (f == nullptr)
        return;
    for (size_t r = 0; r < rows; r++) {
        const float    *l  = lg + r * n_exp;
        const uint32_t *id = ids + r * k;
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
    uint32_t                                active[GEIST_MOE_MAX_EXPERTS];
    size_t                                  n_active;
    _Atomic size_t                          next;
    _Atomic bool                            failed;
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

/* A worker: takes the next active expert until none is left, so experts
 * with many rows and experts with one even out across the threads. */
static void moe_worker(void *ctx, size_t begin, size_t end) {
    struct moe_job *j = ctx;
    (void) begin;
    (void) end;
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
    const size_t                      k     = st->config.moe_n_used;
    const size_t                      seq   = ctx->seq;
    const size_t                      n_sel = seq * k;
    struct geist_buffer              *x     = sess->scratch_pre_ff;

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
    moe_profile_mark(MOE_PROFILE_ROUTER, &t0);

    /* Route on the host. */
    const float *lg = (const float *) v->buffer_map(sess->moe_logits);
    if (lg == nullptr) {
        return GEIST_E_BACKEND;
    }
    bool routed = true;
    for (size_t r = 0; r < seq && routed; r++) {
        routed = moe_route_topk(n_exp,
                                k,
                                st->config.moe_norm_topk,
                                lg + r * n_exp,
                                sess->moe_ids + r * k,
                                sess->moe_wts + r * k);
    }
    if (routed)
        moe_trace(ctx->layer_idx, ctx->q_position, seq, n_exp, k, lg, sess->moe_ids);
    v->buffer_unmap(sess->moe_logits);
    if (!routed) {
        geist_backend_set_error(ctx->be,
                                GEIST_E_BACKEND,
                                "moe: layer %d router logits are not finite",
                                ctx->layer_idx);
        return GEIST_E_BACKEND;
    }
    moe_bucket(n_exp, n_sel, sess->moe_ids, sess->moe_rows, sess->moe_row_start);
    moe_profile_mark(MOE_PROFILE_ROUTE, &t0);

    /* Gather each expert's rows into its slice of the bucketed buffer. */
    const float *xs = (const float *) v->buffer_map(x);
    if (xs == nullptr) {
        return GEIST_E_BACKEND;
    }
    for (size_t i = 0; i < n_sel; i++) {
        memcpy(sess->moe_xg + i * d, xs + (sess->moe_rows[i] / k) * d, d * sizeof(float));
    }
    v->buffer_unmap(x);
    moe_profile_mark(MOE_PROFILE_GATHER, &t0);

    /* Every active expert in one parallel region. */
    struct moe_job  job = {0};
    struct moe_job *j   = &job;
    j->L                = L;
    j->sess             = sess;
    j->be               = ctx->be;
    j->d                = d;
    j->inter            = ctx->inter;
    j->n_exp            = n_exp;
    j->n_active         = 0;
    for (size_t e = 0; e < n_exp; e++) {
        if (sess->moe_row_start[e + 1] > sess->moe_row_start[e])
            j->active[j->n_active++] = (uint32_t) e;
    }
    atomic_store_explicit(&j->next, 0, memory_order_relaxed);
    atomic_store_explicit(&j->failed, false, memory_order_relaxed);
    const size_t threads = geist_par_max_threads();
    geist_par_for(j->n_active < threads ? j->n_active : threads, moe_worker, j);
    moe_profile_mark(MOE_PROFILE_EXPERTS, &t0);
    if (atomic_load_explicit(&j->failed, memory_order_relaxed)) {
        geist_backend_set_error(ctx->be,
                                GEIST_E_UNSUPPORTED,
                                "moe: layer %d: an expert projection has no kernel",
                                ctx->layer_idx);
        return GEIST_E_UNSUPPORTED;
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
