#include "engine.h"

#include <fx.h>

/* SEEDED: two portability #ifs outside a wrapper header; a #if on a
 * project macro does not count, nor does __cplusplus. */
#ifdef __APPLE__
#define ENGINE_OS 1
#elif defined(__linux__)
#define ENGINE_OS 2
#else
#define ENGINE_OS 0
#endif

#if FX_POSIX
#define ENGINE_POSIX 1
#endif

#ifdef __cplusplus
#error "C only"
#endif

int fx_version(void) { return ENGINE_OS; }

int engine_step(int x) { return base_add(x, ENGINE_MAGIC); }
