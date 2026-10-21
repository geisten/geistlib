/*
 * src/backends/cpu_neon/kernels/decode_rows.h — decode GEMVs of several
 * weights in one parallel region (#16).
 *
 * Layer: BACKEND (cpu_neon, internal).
 *
 * A decode step runs a few hundred small GEMVs, each its own parallel
 * region; every region pays a fork, a join and a tail in which the threads
 * that finished early wait for the last. Projections that read the same
 * input (q, k and v) can share one region instead: each kernel prepares a
 * row job over the shared int8 activation, and cpu_neon_decode_parts
 * splits the jobs' joint row space across the threads, the first job's
 * rows first. A row is the same dot whichever thread computes it.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_NEON_DECODE_ROWS_H
#define GEIST_INTERNAL_BACKEND_CPU_NEON_DECODE_ROWS_H

#include "par.h"

#include <stddef.h>
#include <stdint.h>

/* One Q4_K W4A8 decode: rows of w_q4k against the int8 activation. */
struct q4k_decode_ctx {
    const struct block_q4_K_t *w;
    const int8_t              *x_q8;
    const int32_t             *sum32;
    const float               *x_scales; /* one per super-block */
    float                     *y;
    size_t                     n_blocks_per_row;
    size_t                     n_out;
};

/* Output rows [n0, n1) of a q4k_decode_ctx. */
void q4k_decode_rows(void *ctx, size_t n0, size_t n1);

/* One Q6_K W6A8 decode in the source layout. */
struct q6k_decode_ctx {
    const void    *w_q6k;
    const int8_t  *x_q8;
    const int16_t *bsums; /* 16 int16 per super-block */
    size_t         n_blocks_per_row;
    size_t         n_out;
    const float   *x_scales; /* one per super-block */
    float         *y;
};

/* Output rows [n0, n1) of a q6k_decode_ctx. */
void q6k_decode_rows(void *ctx, size_t n0, size_t n1);

/* The 16-element sums of x_q8 the Q6_K rows take, in the calling thread's
 * buffer (valid until its next call); nullptr when the buffer cannot
 * grow. */
[[nodiscard]] const int16_t *q6k_decode_bsums(size_t n_in, const int8_t x_q8[static n_in]);

/* One job of cpu_neon_decode_parts. */
struct cpu_neon_decode_part {
    geist_par_fn rows; /* rows [n0, n1) of ctx */
    void        *ctx;
    size_t       n_out;
};

struct cpu_neon_decode_job {
    const struct cpu_neon_decode_part *parts;
    size_t                             n_parts;
};

/* Joint rows [n0, n1): each part's share of them, the parts in order. */
static inline void cpu_neon_decode_job_rows(void *ctx, size_t n0, size_t n1) {
    const struct cpu_neon_decode_job *j   = ctx;
    size_t                            off = 0;
    for (size_t p = 0; p < j->n_parts && off < n1; p++) {
        const struct cpu_neon_decode_part *part = &j->parts[p];
        const size_t                       end  = off + part->n_out;
        if (n0 < end) {
            const size_t lo = n0 > off ? n0 - off : 0;
            const size_t hi = (n1 < end ? n1 : end) - off;
            part->rows(part->ctx, lo, hi);
        }
        off = end;
    }
}

/* Runs every part's rows, in one geist_par_for over their joint row space
 * (a static split, as the single decodes); returns when all are done. */
static inline void cpu_neon_decode_parts(size_t                            n_parts,
                                         const struct cpu_neon_decode_part parts[static n_parts]) {
    size_t n = 0;
    for (size_t p = 0; p < n_parts; p++) {
        n += parts[p].n_out;
    }
    struct cpu_neon_decode_job job = {.parts = parts, .n_parts = n_parts};
    geist_par_for(n, cpu_neon_decode_job_rows, &job);
}

#endif /* GEIST_INTERNAL_BACKEND_CPU_NEON_DECODE_ROWS_H */
