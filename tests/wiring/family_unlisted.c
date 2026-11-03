/* A family table without a line in family_list.h does not compile: its name
 * is the list's. */
#include "family_fixture.h"

#ifdef WIRING_OK
TRANSFORMER_FAMILY(llama, fixture_populate, fixture_populate_layers, fixture_block, fixture_block);
#else
// expect: transformer_family_name_nosuch
TRANSFORMER_FAMILY(nosuch, fixture_populate, fixture_populate_layers, fixture_block, fixture_block);
#endif
