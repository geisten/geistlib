/*
 * src/archs/transformer/moe_adapt.c - the adaptive expert cache (#166).
 * Contract and policy: moe_adapt.h.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "moe_adapt.h"

#include "arch_state.h"
#include "heap.h"
#include "moe_gpu.h"
#include "forward/profile.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct moe_swap {
    uint32_t layer, promote, demote;
    double   gain_ns; /* the layer's predicted saving */
};

struct moe_adapt {
    size_t           n_layers, n_exp, slots;
    size_t           window, max_swaps;
    double           margin;  /* the CPU branch must exceed the GPU's by this share */
    uint32_t        *count;   /* [n_layers * n_exp] this window's selections */
    uint64_t        *moved;   /* [n_layers * n_exp] the window an expert last moved in, + 1 */
    double          *cpu_ns;  /* [n_layers] CPU branch, summed over the window */
    double          *gpu_ns;  /* [n_layers] GPU branch, summed over probe steps */
    uint32_t        *steps;   /* [n_layers] decode steps recorded */
    uint32_t        *probes;  /* [n_layers] probe steps recorded */
    uint64_t        *sel_cpu; /* [n_layers] selections that ran on the CPU */
    uint64_t        *sel_gpu; /* [n_layers] and on the GPU, on probe steps */
    size_t           tok;     /* decode steps in this window */
    bool             probing; /* the current step probes */
    struct moe_swap *plan;    /* [max_swaps] */
    size_t           n_plan, next;
    uint64_t         windows, swaps, swap_fail, swap_ns;
};

static size_t env_size(const char *name, size_t fallback) {
    const char *e = getenv(name);
    return e != nullptr && *e != '\0' ? (size_t) strtoull(e, nullptr, 10) : fallback;
}

struct moe_adapt *moe_adapt_create(const struct transformer_arch_state *st) {
    const char *on = getenv("GEIST_MOE_ADAPT");
    if (on == nullptr || strcmp(on, "1") != 0 || st->moe_gpu == nullptr || st->moe_gpu->n_hot == 0)
        return nullptr;
    struct moe_adapt *a = heap_calloc_aligned(1, sizeof *a, alignof(struct moe_adapt));
    if (a == nullptr)
        return nullptr;
    a->n_layers    = st->n_layers;
    a->n_exp       = st->config.moe_n_experts;
    a->slots       = moe_slots(&st->config);
    a->window      = env_size("GEIST_MOE_ADAPT_WINDOW", 64);
    a->max_swaps   = env_size("GEIST_MOE_ADAPT_SWAPS", 16);
    a->margin      = 0.1;
    a->window      = a->window < 4 ? 4 : a->window;
    const size_t n = a->n_layers;
    a->count       = heap_calloc_array_aligned(uint32_t, n * a->n_exp);
    a->moved       = heap_calloc_array_aligned(uint64_t, n * a->n_exp);
    a->cpu_ns      = heap_calloc_array_aligned(double, n);
    a->gpu_ns      = heap_calloc_array_aligned(double, n);
    a->steps       = heap_calloc_array_aligned(uint32_t, n);
    a->probes      = heap_calloc_array_aligned(uint32_t, n);
    a->sel_cpu     = heap_calloc_array_aligned(uint64_t, n);
    a->sel_gpu     = heap_calloc_array_aligned(uint64_t, n);
    a->plan        = heap_calloc_array_aligned(struct moe_swap, a->max_swaps + 1);
    if (a->count == nullptr || a->moved == nullptr || a->cpu_ns == nullptr ||
        a->gpu_ns == nullptr || a->steps == nullptr || a->probes == nullptr ||
        a->sel_cpu == nullptr || a->sel_gpu == nullptr || a->plan == nullptr) {
        moe_adapt_destroy(a);
        return nullptr;
    }
    return a;
}

void moe_adapt_destroy(struct moe_adapt *a) {
    if (a == nullptr)
        return;
    if (getenv("GEIST_VERBOSE") != nullptr)
        fprintf(stderr,
                "geist: moe: adaptive cache: %llu windows, %llu swaps (%.2f ms each), %llu "
                "failed\n",
                (unsigned long long) a->windows,
                (unsigned long long) a->swaps,
                a->swaps > 0 ? (double) a->swap_ns / 1e6 / (double) a->swaps : 0.0,
                (unsigned long long) a->swap_fail);
    void *p[] = {a->count,
                 a->moved,
                 a->cpu_ns,
                 a->gpu_ns,
                 a->steps,
                 a->probes,
                 a->sel_cpu,
                 a->sel_gpu,
                 a->plan,
                 a};
    for (size_t i = 0; i < sizeof p / sizeof p[0]; i++)
        safe_free(&p[i]);
}

void moe_adapt_record(struct moe_adapt *a,
                      size_t            layer,
                      size_t            n,
                      const uint32_t    ids[static n],
                      size_t            n_cold,
                      uint64_t          cpu_ns,
                      uint64_t          gpu_ns) {
    if (a == nullptr || layer >= a->n_layers)
        return;
    uint32_t *c = a->count + layer * a->n_exp;
    for (size_t i = 0; i < n; i++)
        if (ids[i] < a->n_exp)
            c[ids[i]]++;
    a->steps[layer]++;
    a->cpu_ns[layer] += (double) cpu_ns;
    a->sel_cpu[layer] += n_cold;
    if (a->probing && gpu_ns > 0) {
        a->probes[layer]++;
        a->gpu_ns[layer] += (double) gpu_ns;
        a->sel_gpu[layer] += n - n_cold;
    }
}

static int swap_cmp(const void *x, const void *y) {
    const struct moe_swap *a = x, *b = y;
    return a->gain_ns > b->gain_ns ? -1 : a->gain_ns < b->gain_ns;
}

struct ranked {
    uint32_t e, n;
};

static int by_n_desc(const void *x, const void *y) {
    const struct ranked *a = x, *b = y;
    return a->n != b->n ? (a->n > b->n ? -1 : 1) : (a->e < b->e ? -1 : a->e > b->e);
}

/* The next window's swaps (moe_adapt.h): per layer up to the balance
 * point, the layers with the largest predicted saving first. */
static void plan(struct moe_adapt *a, const struct transformer_moe_gpu *mg) {
    enum { PER_LAYER = 4 };
    struct moe_swap cand[256 * PER_LAYER];
    size_t          n_cand = 0;
    struct ranked   cold[GEIST_MOE_MAX_EXPERTS], hot[GEIST_MOE_MAX_EXPERTS];
    for (size_t l = 0; l < a->n_layers && n_cand + PER_LAYER <= 256 * PER_LAYER; l++) {
        if (a->steps[l] == 0 || a->probes[l] == 0 || a->sel_gpu[l] == 0 || a->sel_cpu[l] == 0)
            continue;
        const double steps = a->steps[l];
        double       cpu   = a->cpu_ns[l] / steps;
        double       gpu   = a->gpu_ns[l] / a->probes[l];
        static int   dbg   = -1;
        if (dbg < 0)
            dbg = getenv("GEIST_MOE_ADAPT_DEBUG") != nullptr;
        if (dbg && a->windows < 2)
            fprintf(stderr,
                    "adapt w%llu l%zu steps %u cpu %.1f us gpu %.1f us (probes %u) sel cpu %llu "
                    "gpu %llu\n",
                    (unsigned long long) a->windows,
                    l,
                    a->steps[l],
                    cpu / 1e3,
                    gpu / 1e3,
                    a->probes[l],
                    (unsigned long long) a->sel_cpu[l],
                    (unsigned long long) a->sel_gpu[l]);
        if (cpu <= gpu * (1.0 + a->margin))
            continue; /* the GPU branch is (about) the longer one: no change */
        const double ce = a->cpu_ns[l] / (double) a->sel_cpu[l]; /* per selection */
        const double ge = a->gpu_ns[l] / (double) a->sel_gpu[l];
        size_t       nc = 0, nh = 0;
        for (size_t e = 0; e < a->n_exp; e++) {
            /* An expert moved in the last two windows stays where it is. */
            const uint64_t mv = a->moved[l * a->n_exp + e];
            if (mv != 0 && a->windows + 1 - mv < 2)
                continue;
            const struct ranked r = {(uint32_t) e, a->count[l * a->n_exp + e]};
            if (mg->twins[(l * a->slots + e) * 3].raw != nullptr)
                hot[nh++] = r;
            else
                cold[nc++] = r;
        }
        qsort(cold, nc, sizeof cold[0], by_n_desc);
        qsort(hot, nh, sizeof hot[0], by_n_desc); /* the least selected at the end */
        for (size_t i = 0; i < PER_LAYER && i < nc && i < nh; i++) {
            const struct ranked c = cold[i], v = hot[nh - 1 - i];
            if (c.n < 2 * v.n + 2)
                break;                                       /* hysteresis: a clear winner only */
            const double dr  = (double) (c.n - v.n) / steps; /* selections per step moved */
            const double nc_ = cpu - ce * dr, ng_ = gpu + ge * dr;
            const double old = cpu > gpu ? cpu : gpu, now = nc_ > ng_ ? nc_ : ng_;
            if (now >= old)
                break; /* past the balance point */
            cand[n_cand++] = (struct moe_swap) {(uint32_t) l, c.e, v.e, old - now};
            cpu            = nc_;
            gpu            = ng_;
        }
    }
    qsort(cand, n_cand, sizeof cand[0], swap_cmp);
    a->n_plan = n_cand < a->max_swaps ? n_cand : a->max_swaps;
    memcpy(a->plan, cand, a->n_plan * sizeof cand[0]);
    a->next = 0;
}

enum geist_status moe_adapt_step(struct transformer_arch_state *st, bool *probe) {
    struct transformer_moe_gpu *mg = st->moe_gpu;
    struct moe_adapt           *a  = mg != nullptr ? mg->adapt : nullptr;
    *probe                         = false;
    if (a == nullptr)
        return GEIST_OK;
    enum geist_status s = GEIST_OK;
    /* One planned swap per step, only with one session on the model. */
    if (a->next < a->n_plan && atomic_load_explicit(&mg->n_sessions, memory_order_relaxed) == 1) {
        const struct moe_swap w = a->plan[a->next++];
        if (mg->twins[(w.layer * a->slots + w.promote) * 3].raw == nullptr &&
            mg->twins[(w.layer * a->slots + w.demote) * 3].raw != nullptr) {
            const uint64_t t0 = transformer_profile_now_ns();
            s                 = moe_gpu_swap(st, w.layer, w.promote, w.demote);
            a->swap_ns += transformer_profile_now_ns() - t0;
            if (s == GEIST_OK) {
                a->swaps++;
                a->moved[w.layer * a->n_exp + w.promote] = a->windows + 1;
                a->moved[w.layer * a->n_exp + w.demote]  = a->windows + 1;
            } else
                a->swap_fail++;
        }
    }
    if (a->tok == a->window) {
        plan(a, mg);
        a->windows++;
        a->tok = 0;
        memset(a->count, 0, a->n_layers * a->n_exp * sizeof a->count[0]);
        memset(a->cpu_ns, 0, a->n_layers * sizeof a->cpu_ns[0]);
        memset(a->gpu_ns, 0, a->n_layers * sizeof a->gpu_ns[0]);
        memset(a->steps, 0, a->n_layers * sizeof a->steps[0]);
        memset(a->probes, 0, a->n_layers * sizeof a->probes[0]);
        memset(a->sel_cpu, 0, a->n_layers * sizeof a->sel_cpu[0]);
        memset(a->sel_gpu, 0, a->n_layers * sizeof a->sel_gpu[0]);
    }
    /* Probe at the window's first and middle step, not on a swap step. */
    a->probing = (a->tok == 1 || a->tok == a->window / 2 + 1);
    a->tok++;
    *probe = a->probing;
    return s;
}
