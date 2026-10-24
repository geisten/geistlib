/*
 * test_moe_pool_unit - helper threads for a decode step's cold experts
 * (#161, moe_pool.h).
 *
 *   - every task of a job runs exactly once, for job sizes 1..9 and many
 *     jobs in a row (helpers spinning between them);
 *   - the same with GEIST_IDLE_SPIN_MS=0: helpers sleep after every job and
 *     a job still completes (the caller does not wait for them to wake);
 *   - a helper stalled inside a task holds up only the job it took a task
 *     of: the caller returns once that task ends, and the next job runs;
 *   - a second caller while a job runs gets false and runs nothing;
 *   - a kernel call inside a task (geist_par_for) runs on the task's thread;
 *   - n = 0 runs nothing and returns true; no pool: false.
 */
#include "../src/archs/transformer/moe_pool.h"
#include "../src/base/par.h"
#include "test_helpers.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { MAX_TASKS = 16 };

struct job {
    _Atomic int     runs[MAX_TASKS];
    _Atomic int     inner_threads_bad;
    _Atomic bool    stall; /* task 0 sleeps while set */
    _Atomic bool    in_task0;
    pthread_t       owner[MAX_TASKS];
    pthread_mutex_t mu;
};

static void inner(void *ctx, size_t b, size_t e) {
    struct job *j = ctx;
    (void) b;
    (void) e;
    (void) j;
}

struct inner_ctx {
    pthread_t   self;
    _Atomic int bad;
};

static void inner_check(void *ctx, size_t b, size_t e) {
    struct inner_ctx *c = ctx;
    (void) b;
    (void) e;
    if (!pthread_equal(pthread_self(), c->self))
        atomic_fetch_add(&c->bad, 1);
}

static void task(void *ctx, size_t i) {
    struct job *j = ctx;
    atomic_fetch_add(&j->runs[i], 1);
    struct inner_ctx c = {.self = pthread_self()};
    geist_par_for(64, inner_check, &c);
    if (atomic_load(&c.bad) != 0)
        atomic_fetch_add(&j->inner_threads_bad, 1);
    geist_par_for(4, inner, j);
    if (i == 0) {
        atomic_store(&j->in_task0, true);
        while (atomic_load(&j->stall))
            usleep(100);
    }
}

static int run_jobs(struct moe_pool *p, int rounds) {
    int fails = 0;
    for (int r = 0; r < rounds; r++) {
        const size_t n = 1 + (size_t) (r % 9);
        struct job   j;
        memset(&j, 0, sizeof j);
        if (!moe_pool_run(p, n, task, &j)) {
            fails += geist_expect(false, "an idle pool takes the job");
            continue;
        }
        bool once = true;
        for (size_t i = 0; i < MAX_TASKS; i++)
            once = once && atomic_load(&j.runs[i]) == (i < n ? 1 : 0);
        if (!once || atomic_load(&j.inner_threads_bad) != 0) {
            fails += geist_expect(once, "every task runs exactly once");
            fails += geist_expect(atomic_load(&j.inner_threads_bad) == 0,
                                  "a task's geist_par_for stays on its thread");
            break;
        }
    }
    return fails;
}

struct second {
    struct moe_pool *p;
    bool             took;
};

static void *second_caller(void *arg) {
    struct second *s = arg;
    struct job     j;
    memset(&j, 0, sizeof j);
    s->took = moe_pool_run(s->p, 3, task, &j);
    return nullptr;
}

struct stalled {
    struct moe_pool *p;
    struct job       j;
    bool             ok;
};

static void *stalled_caller(void *arg) {
    struct stalled *s = arg;
    s->ok             = moe_pool_run(s->p, 4, task, &s->j);
    return nullptr;
}

int main(void) {
    int fails = 0;
    fails += geist_expect(!moe_pool_run(nullptr, 3, task, nullptr), "no pool: false");
    fails += geist_expect(moe_pool_create(0) == nullptr, "no helpers: no pool");

    struct moe_pool *p = moe_pool_create(4);
    fails += geist_expect(p != nullptr, "a pool of 4 helpers");
    if (p == nullptr)
        return 1;
    fails += geist_expect(moe_pool_run(p, 0, task, nullptr), "n = 0 runs nothing");
    fails += run_jobs(p, 20000);

    /* A job held up by a stalled task: another caller is turned away. */
    struct stalled st = {.p = p};
    atomic_store(&st.j.stall, true);
    pthread_t t1, t2;
    pthread_create(&t1, nullptr, stalled_caller, &st);
    while (!atomic_load(&st.j.in_task0))
        usleep(100);
    struct second sc = {.p = p};
    pthread_create(&t2, nullptr, second_caller, &sc);
    pthread_join(t2, nullptr);
    fails += geist_expect(!sc.took, "a second caller while a job runs gets false");
    atomic_store(&st.j.stall, false);
    pthread_join(t1, nullptr);
    fails += geist_expect(st.ok, "the stalled job completes once its task ends");
    fails += run_jobs(p, 100);
    moe_pool_destroy(p);

    /* Helpers that sleep after every job. */
    setenv("GEIST_IDLE_SPIN_MS", "0", 1);
    p = moe_pool_create(3);
    fails += geist_expect(p != nullptr, "a pool whose helpers sleep");
    if (p != nullptr) {
        for (int r = 0; r < 200 && fails == 0; r++) {
            fails += run_jobs(p, 5);
            if ((r & 15) == 0)
                usleep(200); /* let them fall asleep */
        }
        moe_pool_destroy(p);
    }
    if (fails == 0)
        printf("PASS: moe pool: tasks once each, no wait on idle helpers, busy pool refused\n");
    return fails == 0 ? 0 : 1;
}
