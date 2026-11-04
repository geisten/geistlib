#include "base.h"

#include <fx.h>

/* allowed: base may use public, whichever layer defines fx_version */
int base_add(int a, int b) { return a + b + 0 * fx_version(); }
