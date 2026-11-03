/* A required primitive given as nullptr does not compile: it has no address. */
#include "backend_fixture.h"

#ifdef WIRING_OK
const struct geist_backend_primitives fixture_prims = FIXTURE_PRIMS(FIXTURE_PRIM_SLOTS);
#else
// expect: address of an rvalue|lvalue required
const struct geist_backend_primitives fixture_prims = GEIST_BACKEND_PRIMS(cpu_scalar_rmsnorm,
                                                                          cpu_scalar_add,
                                                                          cpu_scalar_mul,
                                                                          cpu_scalar_gelu_tanh,
                                                                          cpu_scalar_silu,
                                                                          cpu_scalar_rope_apply,
                                                                          nullptr);
#endif
