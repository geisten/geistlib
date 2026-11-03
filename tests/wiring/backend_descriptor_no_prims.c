/* A descriptor without its primitives does not compile. */
#include "backend_fixture.h"

static const struct geist_backend_vtbl       fixture_vtbl  = FIXTURE_VTBL(FIXTURE_VTBL_SLOTS);
static const struct geist_backend_primitives fixture_prims = FIXTURE_PRIMS(FIXTURE_PRIM_SLOTS);

#ifdef WIRING_OK
const struct geist_backend_descriptor fixture_backend =
        GEIST_BACKEND_DESCRIPTOR("fixture", fixture_vtbl, fixture_prims);
#else
// expect: too few arguments|requires [0-9]+ arguments|requires at least
const struct geist_backend_descriptor fixture_backend = GEIST_BACKEND_DESCRIPTOR("fixture", fixture_vtbl);
#endif
