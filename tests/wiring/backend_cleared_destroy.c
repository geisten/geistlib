/* A required slot cleared again among the optional ones does not compile:
 * the second initializer overrides the first. */
#include "backend_fixture.h"

#ifdef WIRING_OK
const struct geist_backend_vtbl fixture_vtbl =
        FIXTURE_VTBL(FIXTURE_VTBL_SLOTS, .resolve_weight = cpu_scalar_resolve_weight);
#else
// expect: initializer-overrides|override-init
const struct geist_backend_vtbl fixture_vtbl =
        FIXTURE_VTBL(FIXTURE_VTBL_SLOTS, .destroy = nullptr);
#endif
