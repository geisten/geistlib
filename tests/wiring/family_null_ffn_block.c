/* A required hook given as nullptr does not compile: it has no address. */
#include "family_fixture.h"

#ifdef WIRING_OK
TRANSFORMER_FAMILY(llama, fixture_populate, fixture_populate_layers, fixture_block, fixture_block);
#else
// expect: address of an rvalue|lvalue required
TRANSFORMER_FAMILY(llama, fixture_populate, fixture_populate_layers, fixture_block, nullptr);
#endif
