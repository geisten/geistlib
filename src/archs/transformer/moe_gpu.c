/*
 * src/archs/transformer/moe_gpu.c - hot MoE experts on a GPU backend (#78
 * phase 3). Contract: moe_gpu.h; design: docs/OFFLOAD.md.
 *
 * Load: every expert stays on the CPU backend. The split backend gets a
 * copy of the experts the router picks most often (GEIST_MOE_PROFILE), in
 * that order until its free memory, less a reserve for one session's
 * buffers and the backend's own, is used. Decode reads each expert a token
 * picks once, so the bytes the CPU no longer reads per token are the
 * selection share the copies cover.
 *
 * Run: blk_moe_ffn hands the layer's hot experts to moe_gpu_run as one task
 * of the parallel region the CPU experts run in. It packs their rows into
 * one block, copies it to VRAM, records gate, up, silu_mul and down per
 * expert, waits, and puts the outputs back where the CPU path writes them.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "moe_gpu.h"
#include "moe_profile.h"
#include "scratch_plan.h"

#include "forward/internal.h"
#include "env.h"
#include "gguf_reader.h"
#include "checked.h"
#include "clock.h"
#include "heap.h"
#include "par.h"
#include "parse.h"

#include <geist_backend.h>
#include <geist_util.h>

#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

bool moe_gpu_is_moe(const struct gguf_ctx *gguf) {
    size_t      len  = 0;
    const char *arch = gguf_get_meta_string(gguf, "general.architecture", &len);
    char        key[96];
    uint32_t    n = 0;
    return arch != nullptr && len + sizeof ".expert_count" <= sizeof key &&
           snprintf(key, sizeof key, "%.*s.expert_count", (int) len, arch) > 0 &&
           gguf_get_meta_u32(gguf, key, &n) && n > 0;
}

static const struct geist_weight *expert_proj(const struct transformer_layer_weights *L, size_t p) {
    return p == 0 ? L->moe_gate_w : p == 1 ? L->moe_up_w : L->moe_down_w;
}

/* Selection counts per (layer, expert) from the "profile" USAGE lines of
 * `path`: "USAGE <model> <set> <file> <tokens> <layer> <c0,c1,...>". */
static void profile_read(const char *path, size_t n_layers, size_t n_exp, uint64_t *count) {
    FILE *f = fopen(path, "r");
    if (f == nullptr) {
        fprintf(stderr, "geist: moe: GEIST_MOE_PROFILE '%s' cannot be read\n", path);
        return;
    }
    char *line = heap_alloc_array_aligned(char, 1 << 16);
    while (line != nullptr && fgets(line, 1 << 16, f) != nullptr) {
        char         set[32];
        unsigned int layer = 0;
        int          at    = 0;
        if (sscanf(line, "USAGE %*s %31s %*s %*s %u %n", set, &layer, &at) != 2 ||
            strcmp(set, "profile") != 0 || layer >= n_layers) {
            continue;
        }
        const char *c = line + at;
        for (size_t e = 0; e < n_exp && *c >= '0' && *c <= '9'; e++) {
            char *end = nullptr;
            count[layer * n_exp + e] += strtoull(c, &end, 10);
            c = *end == ',' ? end + 1 : end;
        }
    }
    void *p = line;
    safe_free(&p);
    fclose(f);
}

struct moe_rank {
    uint64_t count;
    uint32_t idx; /* e * n_layers + layer: ties go to the same share of every layer */
};

static int rank_cmp(const void *a, const void *b) {
    const struct moe_rank *x = a, *y = b;
    if (x->count != y->count) {
        return x->count > y->count ? -1 : 1;
    }
    return x->idx < y->idx ? -1 : x->idx > y->idx;
}

static int stage_cmp(const void *a, const void *b) {
    const struct moe_stage_cand *x = a, *y = b;
    /* rows / bytes against y's, without the division. */
    const double xb = (double) x->bytes, yb = (double) y->bytes;
    if ((double) x->rows * yb != (double) y->rows * xb)
        return (double) x->rows * yb > (double) y->rows * xb ? -1 : 1;
    if ((double) x->rows_lo * yb != (double) y->rows_lo * xb)
        return (double) x->rows_lo * yb > (double) y->rows_lo * xb ? -1 : 1;
    if (x->count != y->count)
        return x->count > y->count ? -1 : 1;
    return x->idx < y->idx ? -1 : x->idx > y->idx;
}

void moe_stage_order(size_t n, struct moe_stage_cand c[static n]) {
    qsort(c, n, sizeof *c, stage_cmp);
}

/* One more session's device buffers, for the budget below: its KV cache at
 * max_seq_len, the scratch pool, the DeltaNet state with its transaction
 * snapshot, and the hot experts' routing buffers (moe_gpu_session_alloc);
 * plus a twentieth and 16 MiB for what this leaves out (small per-layer
 * buffers, allocator rounding). On the RTX 2080 Ti at 2048 positions this
 * gives 419 / 447 MiB where a session took 348 / 386-412 MiB
 * (bench_session_vram, Qwen3.6-35B-A3B / Qwen3-30B-A3B). */
static size_t moe_session_bytes(const struct transformer_arch_state *st) {
    const size_t m = st->moe_session_m_max > 0 ? st->moe_session_m_max : st->m_max;
    struct transformer_scratch_plan sp;
    transformer_scratch_plan_build(st, m, &sp);
    size_t dn = 0, q_out_max = 0;
    for (size_t l = 0; l < st->n_layers; l++) {
        if (st->layers[l].mixer == GEIST_MIXER_DELTANET)
            dn++;
        if (st->layers[l].q_out > q_out_max)
            q_out_max = st->layers[l].q_out;
    }
    const size_t key_dim   = st->config.dn_n_k_heads * st->config.dn_head_k;
    const size_t value_dim = st->config.dn_n_v_heads * st->config.dn_head_v;
    const size_t conv_n    = st->config.dn_conv_kernel > 0
                                     ? (st->config.dn_conv_kernel - 1) * (2 * key_dim + value_dim)
                                     : 0;
    const size_t s_n       = st->config.dn_n_v_heads * st->config.dn_head_k * st->config.dn_head_v;
    /* the Qwen3.5 family's m-row slabs (families/qwen35.c): q+gate, gate,
     * and with DeltaNet layers its qkv, z, b and a rows */
    const size_t slab_row = st->config.has_attn_output_gate
                                    ? 3 * q_out_max + (dn > 0 ? 2 * key_dim + 2 * value_dim +
                                                                        2 * st->config.dn_n_v_heads
                                                              : 0)
                                    : 0;
    /* the routing buffers (moe_gpu_session_alloc): moe_gpu_x, _g, _u with
     * the padding rows (moe_pad) */
    const size_t n_sel    = m * moe_k(&st->config) + 15 * moe_slots(&st->config);
    const size_t moe_bufs = n_sel * (st->d_model + 2 * moe_inter(st)) * sizeof(float);
    size_t       b = 0, t = 0;
    if (ckd_mul(&t, dn, 2 * (conv_n + s_n) * sizeof(float)) || ckd_add(&b, t, sp.pool_bytes) ||
        ckd_mul(&t, m, slab_row * sizeof(float)) || ckd_add(&b, b, t) ||
        ckd_add(&b, b, st->moe_session_kv_bytes) || ckd_add(&b, b, moe_bufs) ||
        ckd_add(&b, b, b / 20 + ((size_t) 16 << 20)))
        return SIZE_MAX;
    return b;
}

/* Experts that fit `gbe` after a reserve: a sixteenth of the device and
 * 512 MiB (the planner's, plan_split_layers) plus one session's buffers.
 * Each further session the caller declared (st->moe_sessions,
 * geist_session_opts.batch_sessions) reserves moe_session_bytes more.
 * On unified memory only for the model on that GPU (#168): its experts'
 * buffers alias the mapping, so every expert that fits runs there and none
 * waits on the CPU. A CPU model with such a GPU beside it keeps them all. */
static size_t moe_fit(const struct transformer_arch_state *st,
                      struct geist_backend                *gbe,
                      size_t                               expert_bytes,
                      size_t                               stage_max) {
    struct geist_backend_memory mem;
    if (geist_backend_memory_info(gbe, &mem) != GEIST_OK || expert_bytes == 0 ||
        (mem.unified_memory && gbe != st->backend)) {
        return 0;
    }
    /* one session's moe_gpu_x, _g, _u, with the padding rows (moe_pad) */
    const size_t n_sel   = st->m_max * moe_k(&st->config) + 15 * moe_slots(&st->config);
    const size_t session = n_sel * (st->d_model + 2 * moe_inter(st)) * sizeof(float);
    size_t       reserve = (size_t) (mem.total_bytes / 16) + ((size_t) 512 << 20) + session;
    /* #239: the staging slots, three per staged expert, each as big as the
     * biggest expert projection of any layer, a sixteenth over its file
     * bytes for the backend's layout (Q6_K's padded blocks). Up to half of
     * the 512 MiB headroom above holds them (40-61 MiB at the default 16
     * experts per step); only what exceeds that comes out of the hot
     * experts: taking all of it from them cost Qwen3.6-35B-A3B 1.6 %
     * decode on the RTX 2080 Ti (23 fewer hot experts). */
    size_t slot = 0, stage = 0;
    for (size_t l = 0; stage_max > 0 && l < st->n_layers; l++)
        for (size_t p = 0; p < 3; p++) {
            const size_t b = expert_proj(&st->layers[l], p)[0].raw_nbytes;
            slot           = b > slot ? b : slot;
        }
    const size_t headroom = (size_t) 256 << 20;
    if (stage_max > 0 && (ckd_mul(&stage, 3 * stage_max, slot + slot / 16) ||
                          (stage > headroom && ckd_add(&reserve, reserve, stage - headroom))))
        return 0;
    size_t more = 0;
    if (st->moe_sessions > 1 && (ckd_mul(&more, st->moe_sessions - 1, moe_session_bytes(st)) ||
                                 ckd_add(&reserve, reserve, more)))
        return 0;
    return mem.free_bytes > reserve ? ((size_t) mem.free_bytes - reserve) / expert_bytes : 0;
}

/* An expert's packed rows on the device: m, or m rounded up to 16 when it
 * has more than one, so its GEMMs take the tensor-core tiles (those need
 * rows % 16 == 0; one row is a matvec). The padding rows repeat a real
 * token and are never summed. */
static size_t moe_pad(size_t m) {
    return m > 1 ? (m + 15) & ~(size_t) 15 : m;
}

/* #239: the most cold experts a prefill step of the model on `gbe` copies
 * there (0: no staging): the backend stages, holds the model itself, has
 * its own memory (or GEIST_MOE_STAGE_M_MIN asks anyway: tests on lavapipe),
 * and GEIST_MOE_STAGE is not 0. GEIST_MOE_STAGE_MAX sets
 * the count (default 16: a step of Qwen3.6-35B-A3B stages 14 on average
 * at 32 rows, and 32 slots were no faster but cost decode hot experts). */
static size_t moe_stage_wanted(const struct transformer_arch_state *st, struct geist_backend *gbe) {
    const struct geist_backend_fused *gf = gbe->desc->fused;
    struct geist_backend_memory       mem;
    if (gbe != st->backend || gf == nullptr || gf->stage_weights == nullptr ||
        gf->stage_reserve == nullptr || gf->rows_gather == nullptr ||
        gf->rows_weighted_sum == nullptr || gf->submit == nullptr ||
        geist_backend_memory_info(gbe, &mem) != GEIST_OK ||
        !geist_env_flag("GEIST_MOE_STAGE", true))
        return 0;
    /* One memory has nothing to copy; a test may still ask with a threshold. */
    if (mem.unified_memory && getenv("GEIST_MOE_STAGE_M_MIN") == nullptr)
        return 0;
    long n = 16;
    if (geist_parse_long(getenv("GEIST_MOE_STAGE_MAX"), &n) && (n < 0 || n > 64))
        n = n < 0 ? 0 : 64;
    return (size_t) n;
}

/* Expert e's gate, up and down on the device: its resident copy, else its
 * host twin for staging (#239), else nullptr. */
static const struct geist_weight *
expert_twins(const struct transformer_moe_gpu *mg, size_t slots, size_t layer, size_t e) {
    const struct geist_weight *tw = &mg->twins[(layer * slots + e) * 3];
    if (tw->raw != nullptr)
        return tw;
    tw = mg->stage_w != nullptr ? &mg->stage_w[(layer * slots + e) * 3] : nullptr;
    return tw != nullptr && tw->raw != nullptr ? tw : nullptr;
}

/* Projection p (gate, up, down) of expert e in layer `layer`: the CPU's
 * view, on the buffer aliased over the same bytes on the device. */
static struct geist_tensor expert_view(const struct transformer_arch_state *st,
                                       const struct transformer_moe_gpu    *mg,
                                       size_t                               layer,
                                       size_t                               e,
                                       size_t                               p) {
    const struct transformer_layer_weights *L = &st->layers[layer];
    struct geist_tensor t = (p == 0 ? L->moe_gate : p == 1 ? L->moe_up : L->moe_down)[e];
    t.buffer              = mg->bufs[layer * 6 + (e >= st->config.moe_n_experts ? 3 : 0) + p];
    return t;
}

/* The gate, up and down twins of experts hot[0, n_hot) of layer `layer`,
 * as the grouped linears take them: w[p][i]. */
static void hot_twins(const struct transformer_arch_state *st,
                      size_t                               layer,
                      size_t                               n_hot,
                      const uint32_t                       hot[static n_hot],
                      const struct geist_weight           *w[3][GEIST_MOE_MAX_EXPERTS]) {
    const size_t slots = moe_slots(&st->config);
    for (size_t i = 0; i < n_hot; i++)
        for (size_t p = 0; p < 3; p++)
            w[p][i] = &st->moe_gpu->twins[(layer * slots + hot[i]) * 3 + p];
}

/* Experts hot[0, n_hot) of layer `layer` over their rows of moe_gpu_x (in
 * hot[]'s order from row at0, each padded to moe_pad when `pad`), the
 * outputs into the same rows of y; a staged expert (#239) reads its slot. Phase by phase over all
 * of them, not expert by expert: the experts of a phase touch disjoint rows, so the backend's
 * hazard tracking puts a barrier only between the phases (gate and up, then silu_mul, then down),
 * and the small decode GEMVs run side by side. */
static enum geist_status record_phases(struct transformer_arch_session *sess,
                                       size_t                           layer,
                                       size_t                           n_hot,
                                       bool                             pad,
                                       size_t                           at0,
                                       const uint32_t                   hot[static n_hot],
                                       struct geist_buffer             *y) {
    const struct transformer_arch_state *st  = sess->model;
    const struct transformer_moe_gpu    *mg  = st->moe_gpu;
    struct geist_backend                *gbe = mg->be;
    const struct geist_backend_fused    *gf  = gbe->desc->fused;
    const size_t                         d = st->d_model, slots = moe_slots(&st->config);
    const size_t                         inter = st->layers[layer].intermediate;
    const size_t                        *rs    = sess->moe_row_start;
    enum geist_status                    s     = GEIST_OK;
    for (size_t phase = 0; phase < 3 && s == GEIST_OK; phase++) {
        size_t at = at0;
        for (size_t i = 0; i < n_hot && s == GEIST_OK; i++) {
            const size_t e = hot[i], m0 = rs[e + 1] - rs[e], m = pad ? moe_pad(m0) : m0;
            const struct geist_weight *tw = expert_twins(mg, slots, layer, e);
            struct geist_tensor        x =
                    view_2d_at(sess->moe_gpu_x, at * d * 4, (int64_t) m, (int64_t) d);
            struct geist_tensor g =
                    view_2d_at(sess->moe_gpu_g, at * inter * 4, (int64_t) m, (int64_t) inter);
            struct geist_tensor u =
                    view_2d_at(sess->moe_gpu_u, at * inter * 4, (int64_t) m, (int64_t) inter);
            struct geist_tensor yv = view_2d_at(y, at * d * 4, (int64_t) m, (int64_t) d);
            if (phase == 0) {
                struct geist_tensor wg = expert_view(st, mg, layer, e, 0);
                struct geist_tensor wu = expert_view(st, mg, layer, e, 1);
                s                      = gf->linear_t(gbe, &x, &tw[0], &wg, m, &g);
                if (s == GEIST_OK)
                    s = gf->linear_t(gbe, &x, &tw[1], &wu, m, &u);
            } else if (phase == 1) {
                s = gf->silu_mul(gbe, &g, &u, &g);
            } else {
                struct geist_tensor wd = expert_view(st, mg, layer, e, 2);
                s                      = gf->linear_t(gbe, &g, &tw[2], &wd, m, &yv);
            }
            at += m;
        }
    }
    return s;
}

void moe_gpu_destroy(struct transformer_moe_gpu *mg) {
    if (mg == nullptr) {
        return;
    }
    if (geist_env_flag("GEIST_VERBOSE", false) && mg->n_hot > 0) {
        fprintf(stderr,
                "geist: moe: %llu expert rows ran on %s, %llu on the CPU\n",
                (unsigned long long) atomic_load(&mg->rows_gpu),
                geist_backend_name(mg->be),
                (unsigned long long) atomic_load(&mg->rows_cpu));
        fprintf(stderr,
                "geist: moe: %llu GPU batches\n",
                (unsigned long long) atomic_load(&mg->batches));
        if (mg->stage_max > 0)
            fprintf(stderr,
                    "geist: moe: %llu cold experts staged on %s for %llu rows (#239)\n",
                    (unsigned long long) atomic_load(&mg->n_staged),
                    geist_backend_name(mg->be),
                    (unsigned long long) atomic_load(&mg->rows_staged));
        fprintf(stderr,
                "geist: moe: %llu layer steps routed on the device, %llu with the experts split\n",
                (unsigned long long) atomic_load(&mg->n_device),
                (unsigned long long) atomic_load(&mg->n_routed));
    }
    const struct geist_backend_vtbl *gv = mg->be->desc->vtbl;
    for (size_t i = 0; mg->bufs != nullptr && i < mg->n_bufs; i++) {
        if (mg->bufs[i] != nullptr)
            gv->buffer_destroy(mg->be, mg->bufs[i]);
    }
    void *b = mg->bufs, *t = mg->twins, *g = mg->grouped, *dv = mg->device, *m = mg;
    void *stw = mg->stage_w;
    safe_free(&stw);
    void *sw = mg->sel_w;
    safe_free(&sw);
    safe_free(&b);
    safe_free(&t);
    safe_free(&g);
    safe_free(&dv);
    safe_free(&m);
}

/* #239: the staging slots for stage_max experts, then a host twin
 * (GEIST_W_HOST) of each routed expert without a copy on the device, in
 * moe_stage_order, in 4 GiB or a quarter of the host's memory if less
 * (GEIST_MOE_STAGE_HOST=<MiB> sets the bytes). The prefill rows come from
 * the run-time profile (#165) at the threshold GEIST_MOE_STAGE_M_MIN sets,
 * else the one the profile kept, else 32; without any, rank order. Any
 * failure leaves staging off; the experts stay on the CPU. */
static void moe_stage_attach(const struct transformer_arch_state *st,
                             struct transformer_moe_gpu          *mg,
                             size_t                               stage_max,
                             const struct moe_rank               *rank) {
    struct geist_backend             *gbe = mg->be;
    const struct geist_backend_vtbl  *gv  = gbe->desc->vtbl;
    const struct geist_backend_fused *gf  = gbe->desc->fused;
    const size_t                      L = st->n_layers, n_exp = st->config.moe_n_experts;
    const size_t                      slots = moe_slots(&st->config);
    const struct geist_weight       **like =
            heap_calloc_array_aligned(const struct geist_weight *, 3 * L);
    for (size_t l = 0; like != nullptr && l < L; l++)
        for (size_t p = 0; p < 3; p++)
            like[l * 3 + p] = &expert_proj(&st->layers[l], p)[0];
    enum geist_status s =
            like != nullptr ? gf->stage_reserve(gbe, 3 * stage_max, 3 * L, like) : GEIST_E_OOM;
    void *lk = (void *) like;
    safe_free(&lk);
    long   mib  = -1;
    size_t used = 0;
#if defined(_SC_PHYS_PAGES)
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
#else
    const long pages = 0, page = 0; /* unknown: GEIST_MOE_STAGE_HOST only */
#endif
    /* 4 GiB, or a quarter of the host's memory if less. The profile's
     * hottest cold experts come first: on the RTX 2080 Ti box 4 GiB kept
     * 91 % of the prefill gain of all cold experts (9.1 GB) for
     * Qwen3-30B-A3B and 57 % of it (11.7 GB) for Qwen3.6-35B-A3B. */
    size_t budget = pages > 0 && page > 0 ? (size_t) pages / 4 * (size_t) page : 0;
    budget        = budget < ((size_t) 4 << 30) ? budget : (size_t) 4 << 30;
    if (geist_parse_long(getenv("GEIST_MOE_STAGE_HOST"), &mib) && mib >= 0)
        budget = (size_t) mib << 20;
    /* The cold routed experts, by the prefill rows a staging threshold
     * takes: the profile's most rows per pinned byte first (#239). Rows at
     * a quarter of the threshold break the ties, which are most of the
     * cold experts after one request: Qwen3.6-35B-A3B with a profile of
     * docs/ARCHITECTURE.md, 4 GiB of copies held 66 % of the rows a
     * prefill of docs/OFFLOAD.md stages by the threshold's rows and the
     * decode counts, 89 % with these (51 % by the decode counts alone). */
    long   th_env = -1;
    size_t th     = geist_parse_long(getenv("GEIST_MOE_STAGE_M_MIN"), &th_env) && th_env > 0
                            ? (size_t) th_env
                            : moe_profile_stage_m_min(st->moe_prof);
    th            = th > 0 ? th : 32;
    uint64_t              *prow = heap_calloc_array_aligned(uint64_t, 2 * L * n_exp);
    struct moe_stage_cand *cand = heap_calloc_array_aligned(struct moe_stage_cand, L * slots);
    const bool by_rows = prow != nullptr &&
                         moe_profile_prefill_rows(st->moe_prof, th, L * n_exp, prow) &&
                         moe_profile_prefill_rows(
                                 st->moe_prof, th >= 4 ? th / 4 : 1, L * n_exp, prow + L * n_exp);
    size_t     n_cand  = 0;
    for (size_t k = 0; cand != nullptr && prow != nullptr && k < L * slots; k++) {
        const size_t l = rank[k].idx % L, e = rank[k].idx / L;
        if (e >= n_exp || mg->twins[(l * slots + e) * 3].raw != nullptr)
            continue;
        size_t b = 0;
        for (size_t p = 0; p < 3; p++)
            b += expert_proj(&st->layers[l], p)[e].raw_nbytes;
        cand[n_cand++] = (struct moe_stage_cand) {.rows    = prow[l * n_exp + e],
                                                  .rows_lo = prow[(L + l) * n_exp + e],
                                                  .bytes   = b,
                                                  .count   = rank[k].count,
                                                  .idx     = rank[k].idx};
    }
    if (cand == nullptr || prow == nullptr)
        s = GEIST_E_OOM;
    else
        moe_stage_order(n_cand, cand);
    void *pr = prow;
    safe_free(&pr);
    if (s == GEIST_OK)
        mg->stage_w = heap_calloc_array_aligned(struct geist_weight, L * slots * 3);
    for (size_t k = 0; s == GEIST_OK && mg->stage_w != nullptr && k < n_cand; k++) {
        const size_t         l = cand[k].idx % L, e = cand[k].idx / L;
        struct geist_weight *tw = &mg->stage_w[(l * slots + e) * 3];
        const size_t         b  = (size_t) cand[k].bytes;
        if (b > budget - used)
            break;
        enum geist_status rs = GEIST_OK;
        for (size_t p = 0; p < 3 && rs == GEIST_OK; p++) {
            const struct geist_weight *w = &expert_proj(&st->layers[l], p)[e];
            tw[p]                        = (struct geist_weight) {.raw        = w->raw,
                                                                  .raw_nbytes = w->raw_nbytes,
                                                                  .n_in       = w->n_in,
                                                                  .n_out      = w->n_out,
                                                                  .dtype      = w->dtype,
                                                                  .flags      = GEIST_W_HOST};
            gbe->calibration.locked      = true;
            rs                           = gv->resolve_weight(gbe, &tw[p]);
        }
        if (rs != GEIST_OK) {
            memset(tw, 0, 3 * sizeof *tw);
            break; /* host memory is short: the rest stay CPU-only */
        }
        /* The backend holds its own copy; the CPU's repack (moe_resolve_cold)
         * faults the file pages back in from the page cache. */
        for (size_t p = 0; p < 3; p++)
            gguf_release_range(st->gguf, tw[p].raw, tw[p].raw_nbytes);
        used += b;
        mg->n_stage_w++;
    }
    void *cd = cand;
    safe_free(&cd);
    if (s == GEIST_OK && mg->stage_w != nullptr && mg->n_stage_w > 0) {
        mg->stage_max = stage_max;
    } else {
        void *sw = mg->stage_w;
        safe_free(&sw);
        mg->stage_w   = nullptr;
        mg->n_stage_w = 0;
    }
    if (geist_env_flag("GEIST_VERBOSE", false))
        fprintf(stderr,
                "geist: moe: staging %s: %zu cold experts held in host memory (%zu MiB) for "
                "%s, up to %zu per prefill step; chosen by %s\n",
                mg->stage_max > 0 ? "on" : "off",
                mg->n_stage_w,
                used >> 20,
                gbe->desc->name,
                stage_max,
                by_rows ? "the profile's prefill rows" : "the selection counts");
}

enum geist_status moe_gpu_attach(struct transformer_arch_state *st, struct geist_backend *gbe) {
    const size_t                      n_layers = st->n_layers, n_exp = st->config.moe_n_experts;
    const size_t                      slots   = moe_slots(&st->config);
    const struct geist_backend_vtbl  *gv      = gbe->desc->vtbl;
    const struct geist_backend_fused *gf      = gbe->desc->fused;
    const bool                        verbose = geist_env_flag("GEIST_VERBOSE", false);
    /* The GPU's expert kernels run silu(gate) * up without biases, weighted
     * by softmax routing: a router or expert FFN of another kind (GPT-OSS
     * #243, DeepSeek #242, Llama 4 #244) keeps every expert on the CPU. */
    const struct geist_arch_config         *cf = &st->config;
    const struct transformer_layer_weights *Lb = &st->layers[0];
    const bool                              plain =
            (cf->moe_gating == 0 || cf->moe_gating == GEIST_MOE_GATING_SOFTMAX) &&
            cf->moe_act == GEIST_MOE_ACT_SWIGLU && !cf->moe_weight_before && cf->moe_n_group <= 1 &&
            (cf->moe_w_scale == 0.0f || cf->moe_w_scale == 1.0f) && Lb->moe_router_b == nullptr &&
            Lb->moe_sel_b == nullptr && Lb->moe_gate_b == nullptr && Lb->moe_up_b == nullptr &&
            Lb->moe_down_b == nullptr;
    if (!plain || n_exp == 0 || st->layers[0].moe_gate_w == nullptr ||
        gv->resolve_weight == nullptr || gv->buffer_copy == nullptr || gf == nullptr ||
        gf->linear_t == nullptr || gf->silu_mul == nullptr) {
        if (verbose) {
            fprintf(stderr, "geist: moe: %s cannot run experts; all on the CPU\n", gbe->desc->name);
        }
        return GEIST_OK;
    }
    const struct transformer_layer_weights *L0 = &st->layers[0];
    const size_t eb        = L0->moe_gate_w[0].raw_nbytes + L0->moe_up_w[0].raw_nbytes +
                             L0->moe_down_w[0].raw_nbytes;
    const size_t stage_max = moe_stage_wanted(st, gbe);
    size_t       n         = moe_fit(st, gbe, eb, stage_max);
    /* GEIST_MOE_HOT=<n>: exactly n, past the budget (tests, measurements). */
    long force;
    if (geist_parse_long(getenv("GEIST_MOE_HOT"), &force) && force >= 0) {
        n = (size_t) force;
    }
    n = n < n_layers * slots ? n : n_layers * slots;

    struct transformer_moe_gpu *mg =
            heap_calloc_aligned(1, sizeof *mg, alignof(struct transformer_moe_gpu));
    struct moe_rank  *rank  = heap_calloc_array_aligned(struct moe_rank, n_layers * slots);
    uint64_t         *count = heap_calloc_array_aligned(uint64_t, n_layers *n_exp);
    enum geist_status s     = GEIST_OK;
    if (mg != nullptr) {
        mg->be      = gbe;
        mg->twins   = heap_calloc_array_aligned(struct geist_weight, n_layers * slots * 3);
        mg->bufs    = heap_calloc_array_aligned(struct geist_buffer *, n_layers * 6);
        mg->n_bufs  = mg->bufs != nullptr ? n_layers * 6 : 0;
        mg->grouped = heap_calloc_array_aligned(bool, n_layers);
    }
    if (mg == nullptr || rank == nullptr || count == nullptr || mg->twins == nullptr ||
        mg->bufs == nullptr || mg->grouped == nullptr) {
        geist_backend_set_error(st->backend, GEIST_E_OOM, "moe: no memory for the expert plan");
        s = GEIST_E_OOM;
    }
    /* GEIST_MOE_PROFILE wins; else the counts of earlier runs (#165). */
    const char *prof = getenv("GEIST_MOE_PROFILE");
    if (s == GEIST_OK && prof != nullptr && *prof != '\0') {
        profile_read(prof, n_layers, n_exp, count);
    } else if (s == GEIST_OK && moe_profile_loaded(st->moe_prof)) {
        moe_profile_counts(st->moe_prof, n_layers * n_exp, count, nullptr);
        prof = moe_profile_path(st->moe_prof);
    } else {
        prof = nullptr;
    }
    /* The shared expert runs for every token: first on the device. */
    for (size_t i = 0; s == GEIST_OK && i < n_layers * slots; i++) {
        const size_t l = i / slots, e = i % slots;
        rank[i] = (struct moe_rank) {e < n_exp ? count[l * n_exp + e] : UINT64_MAX,
                                     (uint32_t) (e * n_layers + l)};
    }
    if (s == GEIST_OK) {
        qsort(rank, n_layers * slots, sizeof *rank, rank_cmp);
    }
    /* The device sees each layer's expert tensors through one aliased
     * buffer: the copies are keyed by addresses inside it. A shared
     * expert is a tensor of its own: bufs[l * 6 + 3 + p]. */
    for (size_t l = 0; s == GEIST_OK && l < n_layers; l++) {
        for (size_t p = 0; p < 6 && s == GEIST_OK; p++) {
            const struct geist_weight *w = expert_proj(&st->layers[l], p % 3);
            if (p >= 3 && slots == n_exp)
                break;
            s = gv->buffer_create_aliased(gbe,
                                          (void *) w[p < 3 ? 0 : n_exp].raw,
                                          p < 3 ? n_exp * w[0].raw_nbytes : w[n_exp].raw_nbytes,
                                          GEIST_BUFFER_WEIGHT,
                                          &mg->bufs[l * 6 + p]);
        }
    }
    uint64_t covered = 0, total = 0;
    for (size_t i = 0; i < n_layers * n_exp; i++) {
        total += count[i];
    }
    for (size_t k = 0; s == GEIST_OK && k < n; k++) {
        const size_t         l = rank[k].idx % n_layers, e = rank[k].idx / n_layers;
        struct geist_weight *tw = &mg->twins[(l * slots + e) * 3];
        enum geist_status    rs = GEIST_OK;
        for (size_t p = 0; p < 3 && rs == GEIST_OK; p++) {
            const struct geist_weight *w = &expert_proj(&st->layers[l], p)[e];
            tw[p]                        = (struct geist_weight) {.raw        = w->raw,
                                                                  .raw_nbytes = w->raw_nbytes,
                                                                  .n_in       = w->n_in,
                                                                  .n_out      = w->n_out,
                                                                  .dtype      = w->dtype};
            gbe->calibration.locked      = true;
            rs                           = gv->resolve_weight(gbe, &tw[p]);
        }
        if (rs != GEIST_OK) {
            memset(tw, 0, 3 * sizeof *tw);
            /* A shared expert (ranked first) the device cannot take, e.g.
             * its dtype: it stays on the CPU, the routed ones still go. */
            if (e >= n_exp)
                continue;
            break; /* the device is full after all: the rest stay */
        }
        mg->n_hot++;
        covered += e < n_exp ? rank[k].count : 0;
        if (k == 0 && gf->submit != nullptr && gf->rows_weighted_sum != nullptr &&
            gf->rows_gather != nullptr && gf->supported != nullptr) {
            const struct geist_fusion_query q1 = {.op = GEIST_FUSED_ROWS_WEIGHTED_SUM,
                                                  .m  = st->m_max};
            const struct geist_fusion_query q2 = {.op = GEIST_FUSED_ROWS_GATHER,
                                                  .m  = st->m_max * moe_k(&st->config)};
            mg->one_wait = gbe == st->backend && gf->supported(gbe, &q1) && gf->supported(gbe, &q2);
        }
        if (!mg->grouped[l] && e < n_exp && gf->linear_t_grouped != nullptr &&
            gf->supported != nullptr) {
            const struct geist_fusion_query q = {.op     = GEIST_FUSED_LINEAR_GROUPED,
                                                 .m      = 1,
                                                 .gate_w = &tw[0],
                                                 .up_w   = &tw[1],
                                                 .down_w = &tw[2]};
            mg->grouped[l]                    = gf->supported(gbe, &q);
        }
    }
    /* Staging runs beside the resident experts, in their session buffers. */
    if (s == GEIST_OK && stage_max > 0 && mg->n_hot > 0)
        moe_stage_attach(st, mg, stage_max, rank);
    void *r = rank, *c = count;
    safe_free(&r);
    safe_free(&c);
    if (s != GEIST_OK) {
        moe_gpu_destroy(mg);
        return s;
    }
    if (verbose && st->moe_sessions > 1) {
        fprintf(stderr,
                "geist: moe: budget leaves room for %zu sessions (%zu MiB each beyond the first, "
                "m_max %zu)\n",
                st->moe_sessions,
                moe_session_bytes(st) >> 20,
                st->moe_session_m_max);
    }
    if (verbose) {
        fprintf(stderr,
                "geist: moe: %zu of %zu experts (%zu MiB) on %s; profile %s, cover %.1f %%\n",
                mg->n_hot,
                n_layers * slots,
                (mg->n_hot * eb) >> 20,
                gbe->desc->name,
                prof != nullptr ? prof : "none",
                total > 0 ? 100.0 * (double) covered / (double) total
                          : 100.0 * (double) mg->n_hot / (double) (n_layers * slots));
    }
    /* #201: a layer whose every routed expert is here routes and runs
     * them on the device at decode (GEIST_MOE_DEVICE=0: never). */
    if (gbe == st->backend && slots == n_exp && gf->moe_route != nullptr &&
        gf->linear_t_ids != nullptr && gf->moe_weighted_sum != nullptr &&
        gf->supported != nullptr && geist_env_flag("GEIST_MOE_DEVICE", true)) {
        mg->device = heap_calloc_array_aligned(bool, n_layers);
        for (size_t l = 0; mg->device != nullptr && l < n_layers; l++) {
            bool all = true;
            for (size_t e = 0; e < n_exp && all; e++)
                all = mg->twins[(l * slots + e) * 3].raw != nullptr;
            const struct geist_weight      *tw = &mg->twins[l * slots * 3];
            const struct geist_fusion_query q  = {.op     = GEIST_FUSED_MOE_DEVICE,
                                                  .m      = 1,
                                                  .gate_w = &tw[0],
                                                  .up_w   = &tw[1],
                                                  .down_w = &tw[2]};
            mg->device[l]                      = all && gf->supported(gbe, &q);
            mg->any_device                     = mg->any_device || mg->device[l];
        }
        /* #245: a batched decode's rows too, when every device layer takes
         * them (GEIST_DECODE_BATCH_MOE_DEVICE=0: the host routes them). */
        const size_t rows =
                st->m_max < TRANSFORMER_DECODE_BATCH_CAP ? st->m_max : TRANSFORMER_DECODE_BATCH_CAP;
        mg->device_rows = 1;
        bool rows_ok =
                mg->any_device && rows > 1 && geist_env_flag("GEIST_DECODE_BATCH_MOE_DEVICE", true);
        for (size_t l = 0; rows_ok && l < n_layers; l++) {
            const struct geist_weight      *tw = &mg->twins[l * slots * 3];
            const struct geist_fusion_query q  = {.op     = GEIST_FUSED_MOE_DEVICE,
                                                  .m      = rows,
                                                  .gate_w = &tw[0],
                                                  .up_w   = &tw[1],
                                                  .down_w = &tw[2]};
            rows_ok                            = !mg->device[l] || gf->supported(gbe, &q);
        }
        if (rows_ok)
            mg->device_rows = rows;
        if (verbose && mg->any_device)
            fprintf(stderr,
                    "geist: moe: decode routes on %s (#201), batched rows up to %zu\n",
                    gbe->desc->name,
                    mg->device_rows);
    }
    /* #162: with the experts split, a decode step routes on the device and
     * starts its hot experts without the host (GEIST_MOE_GPU_ROUTE=0: the
     * host routes). */
    mg->routed = mg->one_wait && !mg->any_device && gf->moe_route != nullptr &&
                 gf->linear_t_sel != nullptr && gf->submit_mark != nullptr &&
                 gf->map_marked != nullptr && slots - n_exp <= 1 &&
                 geist_env_flag("GEIST_MOE_GPU_ROUTE", true);
    if (mg->routed) {
        mg->sel_w = heap_calloc_array_aligned(const struct geist_weight *, n_layers * 3 * n_exp);
        for (size_t l = 0; mg->sel_w != nullptr && l < n_layers; l++)
            for (size_t e = 0; e < n_exp; e++)
                for (size_t p = 0; p < 3; p++) {
                    const struct geist_weight *tw      = &mg->twins[(l * slots + e) * 3 + p];
                    mg->sel_w[(l * 3 + p) * n_exp + e] = tw->raw != nullptr ? tw : nullptr;
                }
        mg->routed = mg->sel_w != nullptr;
    }
    if (verbose && mg->routed)
        fprintf(stderr,
                "geist: moe: decode routes on %s, cold experts on the CPU (#162)\n",
                gbe->desc->name);
    st->moe_gpu = mg;
    return GEIST_OK;
}

void moe_gpu_session_free(struct transformer_arch_session *sess) {
    const struct transformer_moe_gpu *mg  = sess->model->moe_gpu;
    void                             *h[] = {sess->moe_ids_hot,
                                             sess->moe_csr_start,
                                             sess->moe_csr_fill,
                                             sess->moe_csr_rows,
                                             sess->moe_csr_w};
    for (size_t i = 0; i < sizeof h / sizeof h[0]; i++)
        safe_free(&h[i]);
    sess->moe_ids_hot = sess->moe_csr_start = sess->moe_csr_fill = sess->moe_csr_rows = nullptr;
    sess->moe_csr_w                                                                   = nullptr;
    struct geist_buffer **b[] = {&sess->moe_gpu_x_io,
                                 &sess->moe_gpu_y_io,
                                 &sess->moe_gpu_x,
                                 &sess->moe_gpu_g,
                                 &sess->moe_gpu_u,
                                 &sess->moe_dev_ids,
                                 &sess->moe_dev_w};
    for (size_t i = 0; i < sizeof b / sizeof b[0]; i++) {
        if (*b[i] != nullptr) {
            mg->be->desc->vtbl->buffer_destroy(mg->be, *b[i]);
            *b[i] = nullptr;
        }
    }
}

enum geist_status moe_gpu_session_alloc(struct transformer_arch_session *sess) {
    const struct transformer_arch_state *st = sess->model;
    const struct transformer_moe_gpu    *mg = st->moe_gpu;
    if (mg == nullptr || mg->n_hot == 0) {
        return GEIST_OK;
    }
    const struct geist_backend_vtbl *gv    = mg->be->desc->vtbl;
    const size_t                     n_sel = sess->m_max * moe_k(&st->config);
    const size_t                     n_pad = n_sel + 15 * moe_slots(&st->config); /* moe_pad */
    const size_t                     xb    = n_sel * st->d_model * sizeof(float);
    const size_t                     gb    = n_pad * moe_inter(st) * sizeof(float);
    const size_t                     xpb   = n_pad * st->d_model * sizeof(float);
    enum geist_status                s =
            gv->buffer_create(mg->be, xb, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &sess->moe_gpu_x_io);
    if (s == GEIST_OK)
        s = gv->buffer_create(mg->be, xb, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &sess->moe_gpu_y_io);
    if (s == GEIST_OK)
        s = gv->buffer_create(
                mg->be, xpb, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE, &sess->moe_gpu_x);
    if (s == GEIST_OK)
        s = gv->buffer_create(
                mg->be, gb, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE, &sess->moe_gpu_g);
    if (s == GEIST_OK)
        s = gv->buffer_create(
                mg->be, gb, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE, &sess->moe_gpu_u);
    if (s == GEIST_OK && (mg->any_device || mg->routed)) {
        s = gv->buffer_create(mg->be,
                              (n_sel + 1) * sizeof(uint32_t),
                              GEIST_BUFFER_IO,
                              GEIST_MEMORY_AUTO,
                              &sess->moe_dev_ids);
        if (s == GEIST_OK)
            s = gv->buffer_create(mg->be,
                                  n_sel * sizeof(float),
                                  GEIST_BUFFER_SCRATCH,
                                  GEIST_MEMORY_DEVICE,
                                  &sess->moe_dev_w);
        uint32_t *idp = s == GEIST_OK ? gv->buffer_map(sess->moe_dev_ids) : nullptr;
        if (idp != nullptr) {
            memset(idp, 0, (n_sel + 1) * sizeof(uint32_t)); /* the flag word too */
            gv->buffer_unmap(sess->moe_dev_ids);
        } else if (s == GEIST_OK) {
            s = GEIST_E_BACKEND;
        }
    }
    if (s == GEIST_OK) {
        sess->moe_ids_hot   = heap_alloc_array_aligned(uint32_t, n_pad);
        sess->moe_csr_start = heap_alloc_array_aligned(uint32_t, sess->m_max + 1);
        sess->moe_csr_fill  = heap_alloc_array_aligned(uint32_t, sess->m_max);
        sess->moe_csr_rows  = heap_alloc_array_aligned(uint32_t, n_sel);
        sess->moe_csr_w     = heap_alloc_array_aligned(float, n_sel);
        if (sess->moe_ids_hot == nullptr || sess->moe_csr_start == nullptr ||
            sess->moe_csr_fill == nullptr || sess->moe_csr_rows == nullptr ||
            sess->moe_csr_w == nullptr)
            s = GEIST_E_OOM;
    }
    if (s != GEIST_OK && mg->no_cpu_copy) {
        /* Its hot experts exist only on the GPU (#164). */
        geist_backend_set_error(st->backend,
                                s,
                                "moe: no room for this session's buffers on %s (%s); its %zu "
                                "experts there have no CPU copy",
                                mg->be->desc->name,
                                geist_backend_errmsg(mg->be),
                                mg->n_hot);
        moe_gpu_session_free(sess);
        return s;
    }
    if (s != GEIST_OK) {
        fprintf(stderr,
                "geist: moe: no room for this session's buffers on %s (%s); its experts run on "
                "the CPU\n",
                mg->be->desc->name,
                geist_backend_errmsg(mg->be));
        moe_gpu_session_free(sess);
    }
    return GEIST_OK;
}

/* Resolves expert e of one projection array on `cpu`, or, when the GPU
 * holds a copy, only drops its file pages from this process (#164). */
static enum geist_status resolve_one(const struct transformer_arch_state *st,
                                     struct geist_backend                *cpu,
                                     struct geist_weight                 *w,
                                     bool                                 hot,
                                     bool                                 release,
                                     int                                  layer) {
    if (hot) {
        if (release)
            gguf_release_range(st->gguf, w->raw, w->raw_nbytes);
        return GEIST_OK;
    }
    if (cpu->desc->vtbl->resolve_weight == nullptr)
        return GEIST_OK;
    cpu->calibration.locked   = true;
    const enum geist_status s = cpu->desc->vtbl->resolve_weight(cpu, w);
    if (s == GEIST_E_UNSUPPORTED) {
        geist_backend_set_error(st->backend,
                                s,
                                "transformer: %s has no linear for layer %d's experts (dtype %u)",
                                cpu->desc->name,
                                layer,
                                (unsigned) w->dtype);
    } else if (s == GEIST_OK && (w->flags & GEIST_W_RAW_COLD) != 0) {
        /* A host repack the kernels read instead of the source (#729). */
        gguf_release_range(st->gguf, w->raw, w->raw_nbytes);
    }
    return s;
}

enum geist_status moe_resolve_cold(struct transformer_arch_state *st) {
    struct transformer_moe_gpu *mg    = st->moe_gpu;
    const size_t                slots = moe_slots(&st->config);
    enum geist_status           s     = GEIST_OK;
    const size_t                n     = st->n_layers + st->n_mtp_layers;
    /* A discrete GPU has copied its experts; on unified memory its buffers
     * alias the mapping itself, whose pages must stay. */
    struct geist_backend_memory mem;
    const bool release = mg != nullptr && geist_backend_memory_info(mg->be, &mem) == GEIST_OK &&
                         !mem.unified_memory;
    for (size_t i = 0; i < n && s == GEIST_OK; i++) {
        struct transformer_layer_weights *L =
                i < st->n_layers ? &st->layers[i] : &st->mtp_layers[i - st->n_layers].block;
        if (L->moe_gate_w == nullptr)
            continue;
        for (size_t e = 0; e < slots && s == GEIST_OK; e++) {
            /* An MTP block's experts lie past the table the copies cover. */
            const bool           hot = mg != nullptr && i < st->n_layers &&
                                       mg->twins[(i * slots + e) * 3].raw != nullptr;
            struct geist_weight *w[] = {&L->moe_gate_w[e], &L->moe_up_w[e], &L->moe_down_w[e]};
            for (size_t p = 0; p < 3 && s == GEIST_OK; p++)
                s = resolve_one(st, st->moe_cpu, w[p], hot, release, L->layer_idx);
        }
    }
    if (s == GEIST_OK && mg != nullptr)
        mg->no_cpu_copy = mg->n_hot > 0;
    return s;
}

bool moe_gpu_hot(const struct transformer_arch_session *sess, size_t layer, size_t e) {
    const struct transformer_moe_gpu *mg = sess->model->moe_gpu;
    /* An MTP block's layer index lies past the stack the copies cover. */
    return mg != nullptr && sess->moe_gpu_x != nullptr && layer < sess->model->n_layers &&
           mg->twins[(layer * moe_slots(&sess->model->config) + e) * 3].raw != nullptr;
}

enum geist_status moe_gpu_run(struct transformer_arch_session *sess,
                              size_t                           layer,
                              size_t                           n_hot,
                              const uint32_t                   hot[static n_hot]) {
    const struct transformer_arch_state *st    = sess->model;
    struct transformer_moe_gpu          *mg    = st->moe_gpu;
    struct geist_backend                *gbe   = mg->be;
    const struct geist_backend_vtbl     *gv    = gbe->desc->vtbl;
    const struct geist_backend_fused    *gf    = gbe->desc->fused;
    const size_t                         d     = st->d_model;
    const size_t                         inter = st->layers[layer].intermediate;
    const size_t                        *rs    = sess->moe_row_start;

    /* Decode (one row per expert) on a backend with the grouped linear:
     * one dispatch per projection for all of them, one silu_mul. */
    bool one_row =
            mg->grouped[layer] && gf->linear_t_grouped != nullptr && n_hot <= GEIST_MOE_MAX_EXPERTS;
    for (size_t i = 0; i < n_hot && one_row; i++)
        one_row = rs[hot[i] + 1] - rs[hot[i]] == 1;
    size_t            rows = 0;
    enum geist_status s    = GEIST_OK;
    if (one_row && gbe == st->backend && sess->moe_x_io != nullptr) {
        /* The model on this GPU, one token: its x is already in VRAM. The
         * experts' rows are device copies of it, no host round trip. */
        for (size_t i = 0; i < n_hot && s == GEIST_OK; i++)
            s = gv->buffer_copy(sess->moe_gpu_x,
                                i * d * sizeof(float),
                                sess->scratch_pre_ff,
                                0,
                                d * sizeof(float));
        rows = n_hot;
    } else {
        float *xio = gv->buffer_map(sess->moe_gpu_x_io);
        if (xio == nullptr) {
            return GEIST_E_BACKEND;
        }
        for (size_t i = 0; i < n_hot; i++) {
            const size_t m = rs[hot[i] + 1] - rs[hot[i]];
            memcpy(xio + rows * d, sess->moe_xg + rs[hot[i]] * d, m * d * sizeof(float));
            rows += m;
        }
        gv->buffer_unmap(sess->moe_gpu_x_io);
        s = gv->buffer_copy(sess->moe_gpu_x, 0, sess->moe_gpu_x_io, 0, rows * d * sizeof(float));
    }
    bool grouped_done = false;
    if (s == GEIST_OK && one_row) {
        const struct geist_weight *w[3][GEIST_MOE_MAX_EXPERTS];
        hot_twins(st, layer, n_hot, hot, w);
        struct geist_tensor x  = view_2d(sess->moe_gpu_x, (int64_t) n_hot, (int64_t) d);
        struct geist_tensor g  = view_2d(sess->moe_gpu_g, (int64_t) n_hot, (int64_t) inter);
        struct geist_tensor u  = view_2d(sess->moe_gpu_u, (int64_t) n_hot, (int64_t) inter);
        struct geist_tensor y  = view_2d(sess->moe_gpu_y_io, (int64_t) n_hot, (int64_t) d);
        enum geist_status   gs = gf->linear_t_grouped(gbe, n_hot, w[0], &x, &g);
        if (gs == GEIST_OK)
            gs = gf->linear_t_grouped(gbe, n_hot, w[1], &x, &u);
        if (gs == GEIST_OK)
            gs = gf->silu_mul(gbe, &g, &u, &g);
        if (gs == GEIST_OK)
            gs = gf->linear_t_grouped(gbe, n_hot, w[2], &g, &y);
        /* UNSUPPORTED is refused before anything is recorded (a weight not
         * resident): the experts then run one by one below. */
        grouped_done = gs == GEIST_OK;
        if (gs != GEIST_OK && gs != GEIST_E_UNSUPPORTED)
            s = gs;
    }
    if (s == GEIST_OK && !grouped_done)
        s = record_phases(sess, layer, n_hot, false, 0, hot, sess->moe_gpu_y_io);
    const float *yio = s == GEIST_OK ? gv->buffer_map(sess->moe_gpu_y_io) : nullptr; /* waits */
    if (yio == nullptr) {
        return s != GEIST_OK ? s : GEIST_E_BACKEND;
    }
    rows = 0;
    for (size_t i = 0; i < n_hot; i++) {
        const size_t m = rs[hot[i] + 1] - rs[hot[i]];
        memcpy(sess->moe_y + rs[hot[i]] * d, yio + rows * d, m * d * sizeof(float));
        rows += m;
    }
    gv->buffer_unmap(sess->moe_gpu_y_io);
    atomic_fetch_add_explicit(&mg->rows_gpu, rows, memory_order_relaxed);
    atomic_fetch_add_explicit(&mg->batches, 1, memory_order_relaxed);
    return GEIST_OK;
}

/* moe_gpu_launch's per-row mode (#245): the rows of a batched decode, each
 * selection one matvec of its own, so a row's result does not depend on
 * which other rows chose its expert. Needs grouped linears and at most
 * GEIST_MOE_MAX_EXPERTS routed rows. */
static bool moe_gpu_per_row(const struct transformer_moe_gpu *mg,
                            size_t                            layer,
                            size_t                            seq,
                            size_t                            k,
                            bool                              decode_rows) {
    return decode_rows && mg->grouped[layer] && seq * k <= GEIST_MOE_MAX_EXPERTS;
}

enum geist_status moe_gpu_launch(struct transformer_arch_session *sess,
                                 size_t                           layer,
                                 size_t                           seq,
                                 bool                             decode_rows,
                                 size_t                           n_res,
                                 size_t                           n_hot,
                                 const uint32_t                   hot[static n_hot]) {
    const struct transformer_arch_state *st    = sess->model;
    struct transformer_moe_gpu          *mg    = st->moe_gpu;
    struct geist_backend                *gbe   = mg->be;
    const struct geist_backend_fused    *gf    = gbe->desc->fused;
    const size_t                         d     = st->d_model;
    const size_t                         k     = moe_k(&st->config);
    const size_t                         inter = st->layers[layer].intermediate;
    const size_t                        *rs    = sess->moe_row_start;
    const bool                           per   = moe_gpu_per_row(mg, layer, seq, k, decode_rows);
    /* The hot experts' rows, in their order, gathered from x on the device:
     * packed row p reads token src[p]. sess->moe_rows is bucketed by expert;
     * its entries are row * k + slot. Per row: no padding. */
    uint32_t *src  = (uint32_t *) sess->moe_ids_hot;
    size_t    rows = 0, real = 0;
    bool      one = mg->grouped[layer];
    for (size_t i = 0; i < n_hot; i++) {
        const size_t m = rs[hot[i] + 1] - rs[hot[i]];
        one            = one && m == 1;
        real += m;
        for (size_t r = rs[hot[i]]; r < rs[hot[i] + 1]; r++)
            src[rows++] = sess->moe_rows[r] / (uint32_t) k;
        for (size_t r = m; !per && r < moe_pad(m); r++, rows++)
            src[rows] = src[rows - 1];
    }
    struct geist_tensor xin = view_2d(sess->scratch_pre_ff, (int64_t) seq, (int64_t) d);
    struct geist_tensor x   = view_2d(sess->moe_gpu_x, (int64_t) rows, (int64_t) d);
    enum geist_status   s   = gf->rows_gather(gbe, rows, d, src, &xin, &x);
    if (one || per) {
        /* A decode step: one row per expert, one grouped matvec each over
         * the routed ones; a shared expert (last in hot[], its own dtype)
         * after them as plain matvecs, over its m_sh rows. Per row (#245):
         * every routed row its own entry of the grouped matvecs. */
        const bool                 sh   = hot[n_hot - 1] >= st->config.moe_n_experts;
        const size_t               m_sh = sh ? rs[hot[n_hot - 1] + 1] - rs[hot[n_hot - 1]] : 0;
        const size_t               n_rt = rows - m_sh; /* routed packed rows */
        struct geist_tensor        g    = view_2d(sess->moe_gpu_g, (int64_t) rows, (int64_t) inter);
        struct geist_tensor        u    = view_2d(sess->moe_gpu_u, (int64_t) rows, (int64_t) inter);
        struct geist_tensor        xr   = view_2d(sess->moe_gpu_x, (int64_t) n_rt, (int64_t) d);
        struct geist_tensor        gr   = view_2d(sess->moe_gpu_g, (int64_t) n_rt, (int64_t) inter);
        struct geist_tensor        ur   = view_2d(sess->moe_gpu_u, (int64_t) n_rt, (int64_t) inter);
        const struct geist_weight *w[3][GEIST_MOE_MAX_EXPERTS];
        uint32_t                   ex[GEIST_MOE_MAX_EXPERTS]; /* packed row -> expert */
        size_t                     at = 0;
        for (size_t i = 0; i < n_hot; i++)
            for (size_t r = rs[hot[i]]; r < rs[hot[i] + 1]; r++)
                ex[at++] = hot[i];
        hot_twins(st, layer, rows, ex, w);
        if (s == GEIST_OK && n_rt > 0)
            s = gf->linear_t_grouped(gbe, n_rt, w[0], &xr, &gr);
        if (s == GEIST_OK && n_rt > 0)
            s = gf->linear_t_grouped(gbe, n_rt, w[1], &xr, &ur);
        struct geist_tensor xs, gs, us, t[3];
        if (m_sh > 0) {
            xs = view_2d_at(sess->moe_gpu_x, n_rt * d * 4, (int64_t) m_sh, (int64_t) d);
            gs = view_2d_at(sess->moe_gpu_g, n_rt * inter * 4, (int64_t) m_sh, (int64_t) inter);
            us = view_2d_at(sess->moe_gpu_u, n_rt * inter * 4, (int64_t) m_sh, (int64_t) inter);
            for (size_t p = 0; p < 3; p++)
                t[p] = expert_view(st, mg, layer, hot[n_hot - 1], p);
            if (s == GEIST_OK)
                s = gf->linear_t(gbe, &xs, w[0][n_rt], &t[0], m_sh, &gs);
            if (s == GEIST_OK)
                s = gf->linear_t(gbe, &xs, w[1][n_rt], &t[1], m_sh, &us);
        }
        if (s == GEIST_OK)
            s = gf->silu_mul(gbe, &g, &u, &g);
        /* ponytail: the outputs reuse x's rows (read by gate and up, which
         * the backend's hazard tracking orders before this write). */
        if (s == GEIST_OK && n_rt > 0)
            s = gf->linear_t_grouped(gbe, n_rt, w[2], &gr, &xr);
        if (s == GEIST_OK && m_sh > 0)
            s = gf->linear_t(gbe, &gs, w[2][n_rt], &t[2], m_sh, &xs);
    } else if (s == GEIST_OK) {
        /* A prefill chunk: each expert's rows as one GEMM, the outputs
         * over x. The resident experts first: the staged ones (#239) wait
         * for their copies, which run meanwhile. */
        size_t at = 0;
        for (size_t i = 0; i < n_res; i++)
            at += moe_pad(rs[hot[i] + 1] - rs[hot[i]]);
        s = record_phases(sess, layer, n_res, true, 0, hot, sess->moe_gpu_x);
        if (s == GEIST_OK && n_hot > n_res)
            s = record_phases(sess, layer, n_hot - n_res, true, at, hot + n_res, sess->moe_gpu_x);
    }
    if (s == GEIST_OK) {
        gf->submit(gbe);
        atomic_fetch_add_explicit(&mg->rows_gpu, real, memory_order_relaxed);
        atomic_fetch_add_explicit(&mg->batches, 1, memory_order_relaxed);
    }
    return s;
}

enum geist_status moe_gpu_finish(struct transformer_arch_session *sess,
                                 size_t                           seq,
                                 size_t                           layer,
                                 bool                             decode_rows,
                                 size_t                           n_hot,
                                 const uint32_t                   hot[static n_hot],
                                 const float          cold[static seq * sess->model->d_model],
                                 struct geist_buffer *out) {
    const struct transformer_arch_state *st = sess->model;
    const struct transformer_moe_gpu    *mg = st->moe_gpu;
    const size_t                         d = st->d_model, k = moe_k(&st->config);
    const size_t                        *rs  = sess->moe_row_start;
    const bool                           per = moe_gpu_per_row(mg, layer, seq, k, decode_rows);
    /* Per token, its hot rows (packed index) and weights: counted, then
     * placed (CSR). Packed rows follow hot[]'s order, as moe_gpu_launch. */
    uint32_t *start = sess->moe_csr_start, *rows = sess->moe_csr_rows;
    float    *w = sess->moe_csr_w;
    memset(start, 0, (seq + 1) * sizeof *start);
    for (size_t i = 0; i < n_hot; i++)
        for (size_t r = rs[hot[i]]; r < rs[hot[i] + 1]; r++)
            start[sess->moe_rows[r] / k + 1]++;
    for (size_t t = 0; t < seq; t++)
        start[t + 1] += start[t];
    uint32_t *fill = sess->moe_csr_fill;
    memcpy(fill, start, seq * sizeof *fill);
    size_t p = 0;
    for (size_t i = 0; i < n_hot; i++) {
        const size_t m = rs[hot[i] + 1] - rs[hot[i]];
        for (size_t r = rs[hot[i]]; r < rs[hot[i] + 1]; r++, p++) {
            const uint32_t sel = sess->moe_rows[r], t = sel / (uint32_t) k;
            rows[fill[t]] = (uint32_t) p;
            w[fill[t]++]  = sess->moe_wts[sel];
        }
        p += per ? 0 : moe_pad(m) - m; /* the padding rows, as moe_gpu_launch packs them */
    }
    struct geist_tensor y = view_2d(sess->moe_gpu_x, (int64_t) p, (int64_t) d);
    struct geist_tensor o = view_2d(out, (int64_t) seq, (int64_t) d);
    return mg->be->desc->fused->rows_weighted_sum(mg->be, seq, d, p, start, rows, w, cold, &y, &o);
}

bool moe_gpu_routed(const struct transformer_arch_session *sess, size_t layer, size_t seq) {
    const struct transformer_moe_gpu *mg = sess->model->moe_gpu;
    return seq == 1 && mg != nullptr && mg->routed && sess->moe_dev_ids != nullptr &&
           sess->moe_x_io != nullptr && layer < sess->model->n_layers && mg->grouped[layer];
}

uint64_t moe_gpu_route_launch(struct transformer_arch_session *sess,
                              size_t                           layer,
                              enum geist_status               *status) {
    const struct transformer_arch_state *st    = sess->model;
    struct transformer_moe_gpu          *mg    = st->moe_gpu;
    struct geist_backend                *gbe   = mg->be;
    const struct geist_backend_fused    *gf    = gbe->desc->fused;
    const size_t                         n_exp = st->config.moe_n_experts;
    const size_t                         n_rt  = st->config.moe_n_used;
    const size_t                         k     = moe_k(&st->config);
    const size_t                         slots = moe_slots(&st->config);
    const size_t                         d     = st->d_model;
    const size_t                         inter = st->layers[layer].intermediate;
    /* The work so far (the router) starts: the host waits for it alone
     * (map_marked) and routes too, while the device routes -- the same
     * ranking, moe_route_topk's -- into ids[0, n_rt) (the flag word after
     * the session's m_max * k) and runs the hot experts below. */
    const uint64_t      mark = gf->submit_mark(gbe);
    struct geist_tensor t_lg = view_2d(sess->moe_logits, 1, (int64_t) n_exp);
    struct geist_tensor t_w  = view_1d(sess->moe_dev_w, (int64_t) n_rt);
    enum geist_status   s    = gf->moe_route(gbe,
                                             1,
                                             n_exp,
                                             n_rt,
                                             st->config.moe_norm_topk,
                                             &t_lg,
                                             sess->moe_dev_ids,
                                             sess->m_max * k * sizeof(uint32_t),
                                             &t_w);
    /* Slot j of the routed ones multiplies expert ids[j] where it has a copy
     * here; the shared expert, when it has one, is slot n_rt. */
    const struct geist_weight *const *w[3] = {mg->sel_w + (layer * 3 + 0) * n_exp,
                                              mg->sel_w + (layer * 3 + 1) * n_exp,
                                              mg->sel_w + (layer * 3 + 2) * n_exp};
    const struct geist_weight        *sh =
            slots > n_exp ? &mg->twins[(layer * slots + n_exp) * 3] : nullptr;
    const bool          shared = sh != nullptr && sh->raw != nullptr;
    struct geist_tensor t_x    = view_2d(sess->scratch_pre_ff, 1, (int64_t) d);
    struct geist_tensor g      = view_2d(sess->moe_gpu_g, (int64_t) k, (int64_t) inter);
    struct geist_tensor u      = view_2d(sess->moe_gpu_u, (int64_t) k, (int64_t) inter);
    struct geist_tensor gr     = view_2d(sess->moe_gpu_g, (int64_t) n_rt, (int64_t) inter);
    struct geist_tensor ur     = view_2d(sess->moe_gpu_u, (int64_t) n_rt, (int64_t) inter);
    struct geist_tensor yr     = view_2d(sess->moe_gpu_x, (int64_t) n_rt, (int64_t) d);
    if (s == GEIST_OK)
        s = gf->linear_t_sel(gbe, n_exp, w[0], n_rt, false, sess->moe_dev_ids, &t_x, &gr);
    if (s == GEIST_OK)
        s = gf->linear_t_sel(gbe, n_exp, w[1], n_rt, false, sess->moe_dev_ids, &t_x, &ur);
    struct geist_tensor gs, us, ys, t[3];
    if (shared) {
        gs = view_2d_at(sess->moe_gpu_g, n_rt * inter * 4, 1, (int64_t) inter);
        us = view_2d_at(sess->moe_gpu_u, n_rt * inter * 4, 1, (int64_t) inter);
        ys = view_2d_at(sess->moe_gpu_x, n_rt * d * 4, 1, (int64_t) d);
        for (size_t p = 0; p < 3; p++)
            t[p] = expert_view(st, mg, layer, n_exp, p);
        if (s == GEIST_OK)
            s = gf->linear_t(gbe, &t_x, &sh[0], &t[0], 1, &gs);
        if (s == GEIST_OK)
            s = gf->linear_t(gbe, &t_x, &sh[1], &t[1], 1, &us);
    }
    if (s == GEIST_OK)
        s = gf->silu_mul(gbe, &g, &u, &g);
    if (s == GEIST_OK)
        s = gf->linear_t_sel(gbe, n_exp, w[2], n_rt, true, sess->moe_dev_ids, &gr, &yr);
    if (s == GEIST_OK && shared)
        s = gf->linear_t(gbe, &gs, &sh[2], &t[2], 1, &ys);
    if (s == GEIST_OK) {
        gf->submit(gbe);
        atomic_fetch_add_explicit(&mg->n_routed, 1, memory_order_relaxed);
    }
    *status = s;
    return mark;
}

enum geist_status moe_gpu_route_finish(struct transformer_arch_session *sess,
                                       size_t                           layer,
                                       const float          cold[static sess->model->d_model],
                                       struct geist_buffer *out) {
    const struct transformer_arch_state *st    = sess->model;
    struct transformer_moe_gpu          *mg    = st->moe_gpu;
    const size_t                         d     = st->d_model;
    const size_t                         k     = moe_k(&st->config);
    const size_t                         slots = moe_slots(&st->config);
    /* The one token's hot slots: their y rows (the slot) and weights. */
    uint32_t *start = sess->moe_csr_start, *rows = sess->moe_csr_rows;
    float    *w = sess->moe_csr_w;
    uint32_t  n = 0;
    for (size_t j = 0; j < k; j++) {
        if (mg->twins[(layer * slots + sess->moe_ids[j]) * 3].raw == nullptr)
            continue;
        rows[n] = (uint32_t) j;
        w[n++]  = sess->moe_wts[j];
    }
    start[0] = 0;
    start[1] = n;
    atomic_fetch_add_explicit(&mg->rows_gpu, n, memory_order_relaxed);
    struct geist_tensor y = view_2d(sess->moe_gpu_x, (int64_t) k, (int64_t) d);
    struct geist_tensor o = view_2d(out, 1, (int64_t) d);
    return mg->be->desc->fused->rows_weighted_sum(mg->be, 1, d, n, start, rows, w, cold, &y, &o);
}

bool moe_gpu_device(const struct transformer_arch_session *sess,
                    size_t                                 layer,
                    size_t                                 seq,
                    bool                                   decode_rows) {
    const struct transformer_moe_gpu *mg = sess->model->moe_gpu;
    return mg != nullptr && (seq == 1 || (decode_rows && seq <= mg->device_rows)) &&
           mg->device != nullptr && sess->moe_dev_ids != nullptr && layer < sess->model->n_layers &&
           mg->device[layer];
}

enum geist_status moe_gpu_device_ffn(struct transformer_arch_session *sess,
                                     size_t                           layer,
                                     size_t                           seq,
                                     struct geist_buffer             *x,
                                     struct geist_buffer             *add,
                                     struct geist_buffer             *out) {
    const struct transformer_arch_state    *st    = sess->model;
    const struct transformer_moe_gpu       *mg    = st->moe_gpu;
    struct geist_backend                   *gbe   = mg->be;
    const struct geist_backend_fused       *gf    = gbe->desc->fused;
    const struct transformer_layer_weights *L     = &st->layers[layer];
    const size_t                            n     = st->config.moe_n_experts;
    const size_t                            k     = st->config.moe_n_used;
    const size_t                            d     = st->d_model;
    const size_t                            inter = L->intermediate;
    const size_t                            nk    = seq * k;
    const struct geist_weight              *tw    = &mg->twins[layer * n * 3];
    struct geist_tensor t_lg = view_2d(sess->moe_logits, (int64_t) seq, (int64_t) n);
    struct geist_tensor t_w  = view_1d(sess->moe_dev_w, (int64_t) nk);
    struct geist_tensor t_x  = view_2d(x, (int64_t) seq, (int64_t) d);
    struct geist_tensor t_g  = view_2d(sess->moe_gpu_g, (int64_t) nk, (int64_t) inter);
    struct geist_tensor t_u  = view_2d(sess->moe_gpu_u, (int64_t) nk, (int64_t) inter);
    struct geist_tensor t_y  = view_2d(sess->moe_gpu_x, (int64_t) nk, (int64_t) d);
    struct geist_tensor t_a  = view_2d(add, (int64_t) seq, (int64_t) d);
    struct geist_tensor t_o  = view_2d(out, (int64_t) seq, (int64_t) d);
    /* The experts of a projection back to back: expert 0's view, all rows,
     * on the buffer aliased over the layer's expert tensor. */
    struct geist_tensor ex[3] = {L->moe_gate[0], L->moe_up[0], L->moe_down[0]};
    for (size_t p = 0; p < 3; p++) {
        ex[p].shape[0] *= (int64_t) n;
        ex[p].buffer = mg->bufs[layer * 6 + p];
    }
    /* The flag word follows the session's m_max * k ids. */
    enum geist_status s = gf->moe_route(gbe,
                                        seq,
                                        n,
                                        k,
                                        st->config.moe_norm_topk,
                                        &t_lg,
                                        sess->moe_dev_ids,
                                        sess->m_max * k * sizeof(uint32_t),
                                        &t_w);
    if (s == GEIST_OK)
        s = gf->linear_t_ids(gbe, n, seq, k, false, &tw[0], &ex[0], sess->moe_dev_ids, &t_x, &t_g);
    if (s == GEIST_OK)
        s = gf->linear_t_ids(gbe, n, seq, k, false, &tw[1], &ex[1], sess->moe_dev_ids, &t_x, &t_u);
    if (s == GEIST_OK)
        s = gf->silu_mul(gbe, &t_g, &t_u, &t_g);
    if (s == GEIST_OK)
        s = gf->linear_t_ids(gbe, n, seq, k, true, &tw[2], &ex[2], sess->moe_dev_ids, &t_g, &t_y);
    if (s == GEIST_OK)
        s = gf->moe_weighted_sum(gbe, seq, k, d, &t_w, &t_y, &t_a, &t_o);
    if (s == GEIST_OK)
        atomic_fetch_add_explicit(&st->moe_gpu->n_device, 1, memory_order_relaxed);
    if (s != GEIST_OK)
        geist_backend_set_error(st->backend,
                                s,
                                "moe: layer %zu: the device route failed: %s",
                                layer,
                                geist_backend_errmsg(gbe));
    return s;
}

enum geist_status moe_gpu_device_check(struct transformer_arch_session *sess) {
    if (sess->moe_dev_ids == nullptr)
        return GEIST_OK;
    const struct geist_backend_vtbl *gv = sess->model->moe_gpu->be->desc->vtbl;
    const size_t                     at = sess->m_max * sess->model->config.moe_n_used;
    uint32_t                        *p  = gv->buffer_map(sess->moe_dev_ids);
    if (p == nullptr)
        return GEIST_E_BACKEND;
    const bool bad = p[at] != 0;
    p[at]          = 0;
    gv->buffer_unmap(sess->moe_dev_ids);
    if (bad) {
        geist_backend_set_error(sess->model->backend,
                                GEIST_E_BACKEND,
                                "moe: router logits are not finite (routed on the device)");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

/* ---- Cold experts staged for a prefill step (#239) --------------------- */

bool moe_gpu_stageable(const struct transformer_arch_session *sess,
                       size_t                                 layer,
                       size_t                                 e,
                       size_t                                 m) {
    const struct transformer_moe_gpu *mg = sess->model->moe_gpu;
    return mg != nullptr && mg->stage_m_min > 0 && m >= mg->stage_m_min &&
           sess->moe_gpu_x != nullptr && sess->moe_x_io != nullptr &&
           layer < sess->model->n_layers &&
           mg->stage_w[(layer * moe_slots(&sess->model->config) + e) * 3].raw != nullptr;
}

enum geist_status moe_gpu_stage(struct transformer_arch_session *sess,
                                size_t                           layer,
                                size_t                           n,
                                const uint32_t                   e[static n]) {
    const struct transformer_arch_state *st    = sess->model;
    struct transformer_moe_gpu          *mg    = st->moe_gpu;
    const size_t                         slots = moe_slots(&st->config);
    const struct geist_weight           *w[3 * 64];
    if (n == 0 || n > mg->stage_max || n > 64)
        return GEIST_E_UNSUPPORTED;
    for (size_t i = 0; i < n; i++)
        for (size_t p = 0; p < 3; p++)
            w[i * 3 + p] = &mg->stage_w[(layer * slots + e[i]) * 3 + p];
    const enum geist_status s = mg->be->desc->fused->stage_weights(mg->be, 3 * n, w);
    if (s == GEIST_OK) {
        size_t rows = 0;
        for (size_t i = 0; i < n; i++)
            rows += sess->moe_row_start[e[i] + 1] - sess->moe_row_start[e[i]];
        atomic_fetch_add_explicit(&mg->n_staged, n, memory_order_relaxed);
        atomic_fetch_add_explicit(&mg->rows_staged, rows, memory_order_relaxed);
    }
    return s;
}

/* The calibration's CPU part: every thread runs one expert over its own m
 * rows at once, as the cold experts of a prefill step run, each thread
 * another of the layer's cold experts where there are enough. */
struct stage_cpu_sonde {
    const struct transformer_arch_state *st;
    struct geist_backend                *be;
    const uint32_t                      *pair; /* layer * slots + expert, n_pair of them */
    size_t                               n_pair, round, m, d, inter;
    float                               *x, *g, *u, *y;
    uint64_t                            *dt; /* [rounds * 64]: each task's own time, ns */
    atomic_size_t                        next;
    _Atomic bool                         failed;
};

static void stage_cpu_task(void *ctx, size_t begin, size_t end) {
    (void) begin;
    (void) end;
    struct stage_cpu_sonde                 *c     = ctx;
    const size_t                            t     = geist_par_slot(&c->next);
    const size_t                            slots = moe_slots(&c->st->config);
    const size_t                            k     = (c->round * 64 + t) % c->n_pair;
    const struct transformer_layer_weights *L     = &c->st->layers[c->pair[k] / slots];
    const size_t                            e     = c->pair[k] % slots;
    float                     *g = c->g + t * c->m * c->inter, *u = c->u + t * c->m * c->inter;
    const float               *x    = c->x + t * c->m * c->d;
    const struct geist_weight *w[3] = {&L->moe_gate_w[e], &L->moe_up_w[e], &L->moe_down_w[e]};
    if (w[0]->linear_mN == nullptr || w[1]->linear_mN == nullptr || w[2]->linear_mN == nullptr) {
        atomic_store_explicit(&c->failed, true, memory_order_relaxed);
        return;
    }
    const uint64_t t0 = geist_now_ns();
    w[0]->linear_mN(c->m, x, w[0], c->be, g);
    w[1]->linear_mN(c->m, x, w[1], c->be, u);
    for (size_t i = 0; i < c->m * c->inter; i++)
        g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
    w[2]->linear_mN(c->m, g, w[2], c->be, c->y + t * c->m * c->d);
    c->dt[c->round * 64 + t] = geist_now_ns() - t0;
}

static int cmp_u64(const void *a, const void *b) {
    const uint64_t x = *(const uint64_t *) a, y = *(const uint64_t *) b;
    return x < y ? -1 : x > y;
}

static uint64_t median5(uint64_t v[static 5]) {
    qsort(v, 5, sizeof v[0], cmp_u64);
    return v[2];
}

/* Records `reps` times gate, up, silu_mul and down of expert e of layer
 * `layer` (its device weights tw) over m rows of the sonde's buffers,
 * staging copies of the n_cold experts w before each when n_cold > 0, then
 * waits once: the wall time in ns, 0 on failure. Back to back, as a
 * prefill step's experts run: no wait between them, the GPU clocked up. */
static uint64_t stage_gpu_run(const struct transformer_arch_state *st,
                              size_t                               layer,
                              size_t                               e,
                              const struct geist_weight           *tw,
                              size_t                               m,
                              size_t                               reps,
                              size_t                               n_cold,
                              const struct geist_weight *const    *w,
                              struct geist_buffer                 *buf[static 4]) {
    const struct transformer_moe_gpu *mg  = st->moe_gpu;
    struct geist_backend             *gbe = mg->be;
    const struct geist_backend_fused *gf  = gbe->desc->fused;
    const size_t                      d = st->d_model, inter = st->layers[layer].intermediate;
    struct geist_tensor               x     = view_2d(buf[0], (int64_t) m, (int64_t) d);
    struct geist_tensor               g     = view_2d(buf[1], (int64_t) m, (int64_t) inter);
    struct geist_tensor               u     = view_2d(buf[2], (int64_t) m, (int64_t) inter);
    struct geist_tensor               y     = view_2d(buf[3], (int64_t) m, (int64_t) d);
    struct geist_tensor               w0    = expert_view(st, mg, layer, e, 0);
    struct geist_tensor               w1    = expert_view(st, mg, layer, e, 1);
    struct geist_tensor               w2    = expert_view(st, mg, layer, e, 2);
    float                             probe = 0.0f;
    enum geist_status                 s     = GEIST_OK;
    const uint64_t                    t0    = geist_now_ns();
    for (size_t r = 0; r < reps && s == GEIST_OK; r++) {
        if (n_cold > 0)
            s = gf->stage_weights(gbe, 3 * n_cold, w);
        if (s == GEIST_OK)
            s = gf->linear_t(gbe, &x, &tw[0], &w0, m, &g);
        if (s == GEIST_OK)
            s = gf->linear_t(gbe, &x, &tw[1], &w1, m, &u);
        if (s == GEIST_OK)
            s = gf->silu_mul(gbe, &g, &u, &g);
        if (s == GEIST_OK)
            s = gf->linear_t(gbe, &g, &tw[2], &w2, m, &y);
    }
    if (s == GEIST_OK) /* a download waits for the batch */
        s = gbe->desc->vtbl->buffer_download(sizeof probe, (uint8_t *) &probe, buf[3]);
    const uint64_t t1 = geist_now_ns();
    return s == GEIST_OK && t1 > t0 ? t1 - t0 : 0;
}

/* m_min = B / (bw * t_cpu) per expert: B its bytes, bw the staging copy
 * rate, t_cpu its time per row on the CPU (all threads, as a step's cold
 * experts run). In a prefill step the CPU's cold experts are the critical
 * path (#239: without them a step is 1.43x faster) while the GPU waits for
 * them. An expert moved over saves that path its CPU time and costs it at
 * most its copy on the one copy engine; its GEMMs (t_gpu per row and a
 * fixed part, both reported) fill the GPU's wait. The copy overlaps the
 * resident experts' GEMMs as well, so the break-even is at most this m:
 * the power of two at or below it is taken, which also keeps the
 * measurement's noise from moving it. SIZE_MAX: never. */
size_t moe_stage_m_min(double bytes, double bw, double t_cpu) {
    if (!(bw > 0.0) || !(t_cpu > 0.0) || !(bytes > 0.0))
        return SIZE_MAX;
    const double m = bytes / bw / t_cpu;
    if (!(m < 1e6))
        return SIZE_MAX;
    size_t p = 1;
    while ((double) (p * 2) <= m)
        p <<= 1;
    return p;
}

void moe_gpu_stage_calibrate(struct transformer_arch_state *st) {
    struct transformer_moe_gpu *mg = st->moe_gpu;
    if (mg == nullptr || mg->stage_max == 0)
        return;
    const bool verbose = geist_env_flag("GEIST_VERBOSE", false);
    long       force   = -1;
    if (geist_parse_long(getenv("GEIST_MOE_STAGE_M_MIN"), &force) && force >= 0) {
        mg->stage_m_min = (size_t) force;
        if (verbose)
            fprintf(stderr,
                    "geist: moe: staging threshold %zu rows (GEIST_MOE_STAGE_M_MIN)\n",
                    mg->stage_m_min);
        return;
    }
    /* A layer with a resident and a staged routed expert, the most staged. */
    const size_t slots = moe_slots(&st->config), n_exp = st->config.moe_n_experts;
    size_t       layer = SIZE_MAX, e_hot = 0, e_cold[64], n_cold = 0;
    for (size_t l = 0; l < st->n_layers && layer == SIZE_MAX; l++) {
        size_t h = SIZE_MAX, n = 0;
        for (size_t e = 0; e < n_exp; e++) {
            if (mg->twins[(l * slots + e) * 3].raw != nullptr && h == SIZE_MAX)
                h = e;
            else if (mg->stage_w[(l * slots + e) * 3].raw != nullptr && n < mg->stage_max)
                e_cold[n++] = e;
        }
        if (h != SIZE_MAX && n > 0) {
            layer  = l;
            e_hot  = h;
            n_cold = n;
        }
    }
    struct geist_backend            *gbe   = mg->be;
    const struct geist_backend_vtbl *gv    = gbe->desc->vtbl;
    const size_t                     d     = st->d_model;
    const size_t                     inter = layer != SIZE_MAX ? st->layers[layer].intermediate : 0;
    constexpr int                    M_LO  = 16;
    constexpr int                    M_HI  = 256;
    constexpr int                    M_CPU = 32;
    struct geist_buffer             *buf[4] = {nullptr};
    const size_t      bytes[4] = {M_HI * d * 4, M_HI * inter * 4, M_HI * inter * 4, M_HI * d * 4};
    enum geist_status s        = layer != SIZE_MAX ? GEIST_OK : GEIST_E_UNSUPPORTED;
    for (size_t i = 0; i < 4 && s == GEIST_OK; i++)
        s = gv->buffer_create(gbe, bytes[i], GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE, &buf[i]);
    float *zero = s == GEIST_OK ? heap_calloc_array_aligned(float, M_HI *d) : nullptr;
    if (s == GEIST_OK)
        s = zero != nullptr ? gv->buffer_upload(buf[0], M_HI * d * 4, (const uint8_t *) zero)
                            : GEIST_E_OOM;
    /* GPU: a resident expert at two row counts, R times back to back;
     * then R rounds of n_cold staged experts' copies, each read by one
     * GEMM of its first expert at M_LO rows. After a warm-up that clocks
     * the card and its link up. */
    constexpr int              R = 16;
    const struct geist_weight *th =
            s == GEIST_OK ? &mg->twins[(layer * slots + e_hot) * 3] : nullptr;
    const struct geist_weight *w[3 * 64];
    double                     staged_bytes = 0.0;
    for (size_t i = 0; s == GEIST_OK && i < n_cold; i++)
        for (size_t p = 0; p < 3; p++) {
            w[i * 3 + p] = &mg->stage_w[(layer * slots + e_cold[i]) * 3 + p];
            staged_bytes += (double) w[i * 3 + p]->raw_nbytes;
        }
    uint64_t lo[5], hi[5], cp[5];
    for (size_t r = 0; s == GEIST_OK && r < 3; r++) /* warm-up */
        if (stage_gpu_run(st, layer, e_hot, th, M_HI, 4 * R, 0, w, buf) == 0 ||
            stage_gpu_run(st, layer, e_cold[0], w[0], M_LO, R / 4, n_cold, w, buf) == 0)
            s = GEIST_E_BACKEND;
    for (size_t r = 0; s == GEIST_OK && r < 5; r++) {
        lo[r] = stage_gpu_run(st, layer, e_hot, th, M_LO, R, 0, w, buf);
        hi[r] = stage_gpu_run(st, layer, e_hot, th, M_HI, R, 0, w, buf);
        cp[r] = stage_gpu_run(st, layer, e_cold[0], w[0], M_LO, R, n_cold, w, buf);
        if (lo[r] == 0 || hi[r] == 0 || cp[r] == 0)
            s = GEIST_E_BACKEND;
    }
    if (s == GEIST_OK) { /* nothing stays staged */
        const struct geist_weight *none[1] = {nullptr};
        s                                  = gbe->desc->fused->stage_weights(gbe, 0, none);
    }
    for (size_t i = 0; i < 4; i++)
        if (buf[i] != nullptr)
            gv->buffer_destroy(gbe, buf[i]);
    void *z = zero;
    safe_free(&z);
    /* CPU: every thread one cold expert's CPU weights over M_CPU rows, each
     * round other experts from all layers, so that they come from memory
     * as a step's cold experts do, not from the cache. */
    const size_t           P    = geist_par_max_threads() < 64 ? geist_par_max_threads() : 64;
    uint32_t              *pair = heap_alloc_array_aligned(uint32_t, 6 * 64);
    struct stage_cpu_sonde c    = {
            .st = st, .be = st->moe_cpu, .pair = pair, .m = M_CPU, .d = d, .inter = inter};
    uint64_t cpu = UINT64_MAX;
    for (size_t i = 0; pair != nullptr && i < st->n_layers * slots && c.n_pair < 6 * 64; i++) {
        const size_t l = i % st->n_layers, e = i / st->n_layers;
        if (e < n_exp && mg->twins[(l * slots + e) * 3].raw == nullptr &&
            st->layers[l].intermediate == inter)
            pair[c.n_pair++] = (uint32_t) (l * slots + e);
    }
    if (s == GEIST_OK) {
        c.x = heap_calloc_array_aligned(float, P * M_CPU * d);
        c.y = heap_alloc_array_aligned(float, P * M_CPU * d);
        c.g = heap_alloc_array_aligned(float, P * M_CPU * inter);
        c.u = heap_alloc_array_aligned(float, P * M_CPU * inter);
        if (pair == nullptr || c.n_pair == 0 || c.x == nullptr || c.y == nullptr ||
            c.g == nullptr || c.u == nullptr)
            s = GEIST_E_OOM;
    }
    /* Each task times itself and the median of all counts: a round's wall
     * time is its slowest thread's, and one thread the host preempts made
     * whole rounds 2-3x slower right after a load. */
    c.dt = heap_calloc_array_aligned(uint64_t, 6 * 64);
    if (c.dt == nullptr)
        s = GEIST_E_OOM;
    for (size_t r = 0; s == GEIST_OK && r < 6; r++) {
        atomic_store_explicit(&c.next, 0, memory_order_relaxed);
        c.round = r;
        geist_par_for(P, stage_cpu_task, &c);
        if (atomic_load_explicit(&c.failed, memory_order_relaxed))
            s = GEIST_E_UNSUPPORTED;
    }
    if (s == GEIST_OK) {
        uint64_t *v = c.dt;
        size_t    n = 0;
        for (size_t r = 1; r < 6; r++) /* round 0 warms up */
            for (size_t t = 0; t < P; t++)
                v[n++] = c.dt[r * 64 + t];
        qsort(v, n, sizeof v[0], cmp_u64);
        cpu = v[n / 2];
    }
    void *cd = c.dt;
    safe_free(&cd);
    void *pp = pair;
    safe_free(&pp);
    void *cx = c.x, *cy = c.y, *cg = c.g, *cu = c.u;
    safe_free(&cx);
    safe_free(&cy);
    safe_free(&cg);
    safe_free(&cu);
    if (s != GEIST_OK) {
        mg->stage_m_min = 0;
        if (verbose)
            fprintf(stderr, "geist: moe: staging off, its calibration failed (%d)\n", (int) s);
        return;
    }
    const double t_lo = (double) median5(lo) * 1e-9 / R, t_hi = (double) median5(hi) * 1e-9 / R;
    const double t_gpu = t_hi > t_lo ? (t_hi - t_lo) / (M_HI - M_LO) : 0.0;
    const double t_fix = t_lo - t_gpu * M_LO > 0.0 ? t_lo - t_gpu * M_LO : 0.0;
    const double t_cp  = (double) median5(cp) * 1e-9 / R - t_lo;
    const double bw    = t_cp > 0.0 ? staged_bytes / t_cp : 0.0;
    /* P threads each run one expert's M_CPU rows in `cpu` ns: per row, all
     * threads together. */
    const double t_cpu = (double) cpu * 1e-9 / (double) (P * M_CPU);
    const double eb    = staged_bytes / (double) n_cold;
    const size_t m     = t_cpu > t_gpu ? moe_stage_m_min(eb, bw, t_cpu) : SIZE_MAX;
    mg->stage_m_min    = m <= st->m_max ? m : 0;
    /* Kept with the profile: the next load ranks its host copies by it. */
    moe_profile_set_stage_m_min(st->moe_prof, mg->stage_m_min);
    if (verbose)
        fprintf(stderr,
                "geist: moe: staging cost model: copy %.2f GB/s (%.2f MB per expert), GPU "
                "%.2f us per row + %.1f us, CPU %.2f us per row (%zu threads) -> threshold %zu "
                "rows%s\n",
                bw * 1e-9,
                eb * 1e-6,
                t_gpu * 1e6,
                t_fix * 1e6,
                t_cpu * 1e6,
                P,
                m == SIZE_MAX ? 0 : m,
                mg->stage_m_min == 0 ? " (above the chunk: off)" : "");
}
