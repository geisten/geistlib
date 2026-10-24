/*
 * src/archs/transformer/moe_pool.h - helper threads for a decode step's
 * cold experts that nobody waits for unless they took work (#161).
 *
 * With the model on the GPU a decode step runs ~2-8 cold experts per layer
 * on the CPU while the GPU runs the hot ones. Run as a geist_par_for (an
 * OpenMP region), the step waits at the region's end for every thread of
 * the team, including one the OS descheduled before it took an expert: on
 * a host whose cores are busy (the desktop's CI runners) a layer's cold
 * region then took 6-7 ms instead of 0.2 and decode collapsed from ~60 to
 * 2.5 tok/s (runs 1733/1734).
 *
 * Here the tasks are claimed one by one from a counter tagged with the
 * job's generation; the caller claims too and waits only until every task
 * is done. A helper that wakes late finds the tasks taken and goes back to
 * waiting; it never touches a job it took no task of. Helpers spin for the
 * idle bound of omp_idle.h after their last task, then sleep. Their kernel
 * calls run on the helper itself (geist_par_set_inline).
 */
#ifndef GEIST_TRANSFORMER_MOE_POOL_H
#define GEIST_TRANSFORMER_MOE_POOL_H

#include <stddef.h>

struct moe_pool;

/* Task i of a job, i < n. */
typedef void (*moe_pool_fn)(void *ctx, size_t i);

/* A pool of n_helpers threads (0: nullptr). nullptr when none could be
 * created. */
[[nodiscard]] struct moe_pool *moe_pool_create(size_t n_helpers);

void moe_pool_destroy(struct moe_pool *p);

/* Runs fn(ctx, i) for every i < n on the calling thread and the helpers,
 * and returns true once all are done. false (nothing run) when another
 * thread is running a job on the pool: the caller runs its tasks another
 * way. */
[[nodiscard]] bool moe_pool_run(struct moe_pool *p, size_t n, moe_pool_fn fn, void *ctx);

#endif
