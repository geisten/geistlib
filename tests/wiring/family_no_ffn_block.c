/* A family without ffn_block does not compile: a missing macro argument. */
#include "family_fixture.h"

#ifdef WIRING_OK
TRANSFORMER_FAMILY(llama, fixture_populate, fixture_populate_layers, fixture_block, fixture_block);
#else
// expect: too few arguments|requires [0-9]+ arguments|requires at least
TRANSFORMER_FAMILY(llama, fixture_populate, fixture_populate_layers, fixture_block);
#endif
