/*
 * src/base/omp_idle.h — how long an idle OpenMP worker spins (#651).
 *
 * Layer: ENGINE.
 *
 * cpu_neon asks libomp for OMP_WAIT_POLICY=active: decode runs ~200 tiny
 * parallel regions per token, and a sleeping worker's wake-up costs more
 * than the region. With "active", libomp's KMP_BLOCKTIME defaults to
 * infinite, so after the last region of a decode the workers kept 3-5
 * cores at 100 % for as long as the process lived (M1 Max, SmolLM2 360M
 * idle at a chat prompt: ~440 %). An explicit KMP_BLOCKTIME keeps the
 * waiting active and bounds it: GEIST_IDLE_SPIN_MS is far above any gap
 * inside or between decode steps and far below "waiting for a person".
 *
 * libomp reads its environment once, at the first OpenMP API call, so
 * everything here must happen before that: at backend create for the
 * default, at model load (before weight packing) for a model's value.
 * libgomp (gcc -fopenmp) reads it when the library loads, before main;
 * setenv in the process changes nothing there, and its default wait
 * policy is already bounded. An OMP_WAIT_POLICY=active from the
 * environment is not: libgomp then spins ~3e10 times. The docs pair it
 * with GOMP_SPINCOUNT=200000000, timed as the CPU the idle workers burn
 * after the last region: ~190 ms on a Pi 5 (Cortex-A76), 60 ms on a
 * GitHub ubuntu-24.04-arm runner (Neoverse-N2). No count fits both
 * within 2x, so the smaller one, which bounds the slower board.
 */
#ifndef GEIST_INTERNAL_OMP_IDLE_H
#define GEIST_INTERNAL_OMP_IDLE_H

constexpr int GEIST_IDLE_SPIN_MS = 200;

/* The idle bound in ms: the GEIST_IDLE_SPIN_MS environment variable if it
 * is a number >= 0 (read via `getenv_fn`, nullptr = getenv), else the
 * constant. */
int geist_omp_idle_spin_ms(const char *(*getenv_fn)(const char *) );

/* Sets KMP_BLOCKTIME to the idle bound unless the environment has one.
 * Call before the process's first OpenMP API call. */
void geist_omp_idle_default(void);

/* Sets KMP_BLOCKTIME to a model's `ms` (>= 0; < 0 does nothing). Replaces
 * the idle default, never an explicit KMP_BLOCKTIME, and only once: the
 * first model loaded decides. Call before that model's first parallel
 * region. */
void geist_omp_blocktime_apply(int ms);

#endif /* GEIST_INTERNAL_OMP_IDLE_H */
