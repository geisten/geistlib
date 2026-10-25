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

#include "forward/internal.h"
#include "gguf_reader.h"
#include "heap.h"

#include <geist_backend.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* Experts that fit `gbe` after a reserve: a sixteenth of the device and
 * 512 MiB (the planner's, plan_split_layers) plus one session's buffers.
 * On unified memory only for the model on that GPU (#168): its experts'
 * buffers alias the mapping, so every expert that fits runs there and none
 * waits on the CPU. A CPU model with such a GPU beside it keeps them all. */
static size_t
moe_fit(const struct transformer_arch_state *st, struct geist_backend *gbe, size_t expert_bytes) {
    struct geist_backend_memory mem;
    if (geist_backend_memory_info(gbe, &mem) != GEIST_OK || expert_bytes == 0 ||
        (mem.unified_memory && gbe != st->backend)) {
        return 0;
    }
    /* one session's moe_gpu_x, _g, _u, with the padding rows (moe_pad) */
    const size_t n_sel   = st->m_max * moe_k(&st->config) + 15 * moe_slots(&st->config);
    const size_t session = n_sel * (st->d_model + 2 * moe_inter(st)) * sizeof(float);
    const size_t reserve = (size_t) (mem.total_bytes / 16) + ((size_t) 512 << 20) + session;
    return mem.free_bytes > reserve ? ((size_t) mem.free_bytes - reserve) / expert_bytes : 0;
}

/* An expert's packed rows on the device: m, or m rounded up to 16 when it
 * has more than one, so its GEMMs take the tensor-core tiles (those need
 * rows % 16 == 0; one row is a matvec). The padding rows repeat a real
 * token and are never summed. */
static size_t moe_pad(size_t m) {
    return m > 1 ? (m + 15) & ~(size_t) 15 : m;
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
 * hot[]'s order, each padded to moe_pad when `pad`), the outputs into the
 * same rows of y. Phase by phase over all of them, not expert by expert:
 * the experts of a phase touch disjoint rows, so the backend's hazard
 * tracking puts a barrier only between the phases (gate and up, then
 * silu_mul, then down), and the small decode GEMVs run side by side. */
static enum geist_status record_phases(struct transformer_arch_session *sess,
                                       size_t                           layer,
                                       size_t                           n_hot,
                                       bool                             pad,
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
        size_t at = 0;
        for (size_t i = 0; i < n_hot && s == GEIST_OK; i++) {
            const size_t e = hot[i], m0 = rs[e + 1] - rs[e], m = pad ? moe_pad(m0) : m0;
            const struct geist_weight *tw = &mg->twins[(layer * slots + e) * 3];
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
    if (getenv("GEIST_VERBOSE") != nullptr && mg->n_hot > 0) {
        fprintf(stderr,
                "geist: moe: %llu expert rows ran on %s, %llu on the CPU\n",
                (unsigned long long) atomic_load(&mg->rows_gpu),
                geist_backend_name(mg->be),
                (unsigned long long) atomic_load(&mg->rows_cpu));
        fprintf(stderr,
                "geist: moe: %llu GPU batches\n",
                (unsigned long long) atomic_load(&mg->batches));
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
    void *sw = (void *) mg->sel_w;
    safe_free(&sw);
    safe_free(&b);
    safe_free(&t);
    safe_free(&g);
    safe_free(&dv);
    safe_free(&m);
}

enum geist_status moe_gpu_attach(struct transformer_arch_state *st, struct geist_backend *gbe) {
    const size_t                      n_layers = st->n_layers, n_exp = st->config.moe_n_experts;
    const size_t                      slots   = moe_slots(&st->config);
    const struct geist_backend_vtbl  *gv      = gbe->desc->vtbl;
    const struct geist_backend_fused *gf      = gbe->desc->fused;
    const bool                        verbose = getenv("GEIST_VERBOSE") != nullptr;
    if (n_exp == 0 || st->layers[0].moe_gate_w == nullptr || gv->resolve_weight == nullptr ||
        gv->buffer_copy == nullptr || gf == nullptr || gf->linear_t == nullptr ||
        gf->silu_mul == nullptr) {
        if (verbose) {
            fprintf(stderr, "geist: moe: %s cannot run experts; all on the CPU\n", gbe->desc->name);
        }
        return GEIST_OK;
    }
    const struct transformer_layer_weights *L0 = &st->layers[0];
    const size_t eb = L0->moe_gate_w[0].raw_nbytes + L0->moe_up_w[0].raw_nbytes +
                      L0->moe_down_w[0].raw_nbytes;
    size_t       n  = moe_fit(st, gbe, eb);
    /* GEIST_MOE_HOT=<n>: exactly n, past the budget (tests, measurements). */
    const char *force = getenv("GEIST_MOE_HOT");
    if (force != nullptr && *force != '\0') {
        n = (size_t) strtoull(force, nullptr, 10);
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
    void *r = rank, *c = count;
    safe_free(&r);
    safe_free(&c);
    if (s != GEIST_OK) {
        moe_gpu_destroy(mg);
        return s;
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
    const char *dev_env = getenv("GEIST_MOE_DEVICE");
    if (gbe == st->backend && slots == n_exp && gf->moe_route != nullptr &&
        gf->linear_t_ids != nullptr && gf->moe_weighted_sum != nullptr &&
        gf->supported != nullptr && !(dev_env != nullptr && strcmp(dev_env, "0") == 0)) {
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
        if (verbose && mg->any_device)
            fprintf(stderr, "geist: moe: decode routes on %s (#201)\n", gbe->desc->name);
    }
    /* #162: with the experts split, a decode step routes on the device and
     * starts its hot experts without the host (GEIST_MOE_GPU_ROUTE=0: the
     * host routes). */
    const char *rt_env = getenv("GEIST_MOE_GPU_ROUTE");
    mg->routed         = mg->one_wait && !mg->any_device && gf->moe_route != nullptr &&
                         gf->linear_t_sel != nullptr && gf->submit_mark != nullptr &&
                         gf->map_marked != nullptr && slots - n_exp <= 1 &&
                         !(rt_env != nullptr && strcmp(rt_env, "0") == 0);
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
        s = record_phases(sess, layer, n_hot, false, hot, sess->moe_gpu_y_io);
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

enum geist_status moe_gpu_launch(struct transformer_arch_session *sess,
                                 size_t                           layer,
                                 size_t                           seq,
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
    /* The hot experts' rows, in their order, gathered from x on the device:
     * packed row p reads token src[p]. sess->moe_rows is bucketed by expert;
     * its entries are row * k + slot. */
    uint32_t *src  = (uint32_t *) sess->moe_ids_hot;
    size_t    rows = 0, real = 0;
    bool      one = mg->grouped[layer];
    for (size_t i = 0; i < n_hot; i++) {
        const size_t m = rs[hot[i] + 1] - rs[hot[i]];
        one            = one && m == 1;
        real += m;
        for (size_t r = rs[hot[i]]; r < rs[hot[i] + 1]; r++)
            src[rows++] = sess->moe_rows[r] / (uint32_t) k;
        for (size_t r = m; r < moe_pad(m); r++, rows++)
            src[rows] = src[rows - 1];
    }
    struct geist_tensor xin = view_2d(sess->scratch_pre_ff, (int64_t) seq, (int64_t) d);
    struct geist_tensor x   = view_2d(sess->moe_gpu_x, (int64_t) rows, (int64_t) d);
    enum geist_status   s   = gf->rows_gather(gbe, rows, d, src, &xin, &x);
    if (one) {
        /* A decode step: one row per expert, one grouped matvec each over
         * the routed ones; a shared expert (last in hot[], its own dtype)
         * after them as plain matvecs. */
        const size_t               n_rt = n_hot - (hot[n_hot - 1] >= st->config.moe_n_experts);
        struct geist_tensor        g    = view_2d(sess->moe_gpu_g, (int64_t) rows, (int64_t) inter);
        struct geist_tensor        u    = view_2d(sess->moe_gpu_u, (int64_t) rows, (int64_t) inter);
        struct geist_tensor        xr   = view_2d(sess->moe_gpu_x, (int64_t) n_rt, (int64_t) d);
        struct geist_tensor        gr   = view_2d(sess->moe_gpu_g, (int64_t) n_rt, (int64_t) inter);
        struct geist_tensor        ur   = view_2d(sess->moe_gpu_u, (int64_t) n_rt, (int64_t) inter);
        const struct geist_weight *w[3][GEIST_MOE_MAX_EXPERTS];
        hot_twins(st, layer, n_hot, hot, w);
        if (s == GEIST_OK && n_rt > 0)
            s = gf->linear_t_grouped(gbe, n_rt, w[0], &xr, &gr);
        if (s == GEIST_OK && n_rt > 0)
            s = gf->linear_t_grouped(gbe, n_rt, w[1], &xr, &ur);
        struct geist_tensor xs, gs, us, t[3];
        if (n_rt < n_hot) {
            xs = view_2d_at(sess->moe_gpu_x, n_rt * d * 4, 1, (int64_t) d);
            gs = view_2d_at(sess->moe_gpu_g, n_rt * inter * 4, 1, (int64_t) inter);
            us = view_2d_at(sess->moe_gpu_u, n_rt * inter * 4, 1, (int64_t) inter);
            for (size_t p = 0; p < 3; p++)
                t[p] = expert_view(st, mg, layer, hot[n_rt], p);
            if (s == GEIST_OK)
                s = gf->linear_t(gbe, &xs, w[0][n_rt], &t[0], 1, &gs);
            if (s == GEIST_OK)
                s = gf->linear_t(gbe, &xs, w[1][n_rt], &t[1], 1, &us);
        }
        if (s == GEIST_OK)
            s = gf->silu_mul(gbe, &g, &u, &g);
        /* ponytail: the outputs reuse x's rows (read by gate and up, which
         * the backend's hazard tracking orders before this write). */
        if (s == GEIST_OK && n_rt > 0)
            s = gf->linear_t_grouped(gbe, n_rt, w[2], &gr, &xr);
        if (s == GEIST_OK && n_rt < n_hot)
            s = gf->linear_t(gbe, &gs, w[2][n_rt], &t[2], 1, &xs);
    } else if (s == GEIST_OK) {
        /* A prefill chunk: each expert's rows as one GEMM, the outputs
         * over x. */
        s = record_phases(sess, layer, n_hot, true, hot, sess->moe_gpu_x);
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
                                 size_t                           n_hot,
                                 const uint32_t                   hot[static n_hot],
                                 const float          cold[static seq * sess->model->d_model],
                                 struct geist_buffer *out) {
    const struct transformer_arch_state *st = sess->model;
    const struct transformer_moe_gpu    *mg = st->moe_gpu;
    const size_t                         d = st->d_model, k = moe_k(&st->config);
    const size_t                        *rs = sess->moe_row_start;
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
        p += moe_pad(m) - m; /* the padding rows, as moe_gpu_launch packs them */
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
                                             0,
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
        s = gf->linear_t_sel(gbe, n_exp, w[0], n_rt, false, sess->moe_dev_ids, 0, &t_x, &gr);
    if (s == GEIST_OK)
        s = gf->linear_t_sel(gbe, n_exp, w[1], n_rt, false, sess->moe_dev_ids, 0, &t_x, &ur);
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
        s = gf->linear_t_sel(gbe, n_exp, w[2], n_rt, true, sess->moe_dev_ids, 0, &gr, &yr);
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

bool moe_gpu_device(const struct transformer_arch_session *sess, size_t layer, size_t seq) {
    const struct transformer_moe_gpu *mg = sess->model->moe_gpu;
    return seq == 1 && mg != nullptr && mg->device != nullptr && sess->moe_dev_ids != nullptr &&
           layer < sess->model->n_layers && mg->device[layer];
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
                                        0,
                                        sess->m_max * k * sizeof(uint32_t),
                                        &t_w);
    if (s == GEIST_OK)
        s = gf->linear_t_ids(
                gbe, n, seq, k, false, &tw[0], &ex[0], sess->moe_dev_ids, 0, &t_x, &t_g);
    if (s == GEIST_OK)
        s = gf->linear_t_ids(
                gbe, n, seq, k, false, &tw[1], &ex[1], sess->moe_dev_ids, 0, &t_x, &t_u);
    if (s == GEIST_OK)
        s = gf->silu_mul(gbe, &t_g, &t_u, &t_g);
    if (s == GEIST_OK)
        s = gf->linear_t_ids(
                gbe, n, seq, k, true, &tw[2], &ex[2], sess->moe_dev_ids, 0, &t_g, &t_y);
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
