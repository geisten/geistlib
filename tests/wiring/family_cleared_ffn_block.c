/* A required hook cleared again among the optional members does not compile:
 * the second initializer overrides the first. */
#include "family_fixture.h"

#ifdef WIRING_OK
TRANSFORMER_FAMILY(llama, fixture_populate, fixture_populate_layers, fixture_block, fixture_block,
                   .fused_residual_norms = true);
#else
// expect: initializer-overrides|override-init
TRANSFORMER_FAMILY(llama, fixture_populate, fixture_populate_layers, fixture_block, fixture_block,
                   .ffn_block = nullptr);
#endif
